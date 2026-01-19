//! Bayesian inference for timing leak detection using 9D quantile model.
//!
//! This module computes the posterior probability of a timing leak using:
//! - 9D Gaussian model over quantile differences δ ∈ ℝ⁹
//! - Conjugate Gaussian prior with shaped covariance
//! - Closed-form posterior (no MCMC required)
//! - Monte Carlo integration for leak probability
//!
//! ## Model
//!
//! Likelihood: Δ | δ ~ N(δ, Σ_n)
//!
//! where:
//! - Δ is the observed quantile differences (9-vector)
//! - δ is the true per-decile timing differences (9-vector)
//! - Σ_n is the covariance matrix scaled for sample size
//!
//! ## Prior
//!
//! δ ~ N(0, Λ₀), where Λ₀ = σ²_prior × S
//! S = Σ_rate / tr(Σ_rate) (shaped to match empirical covariance structure)
//!
//! ## Posterior
//!
//! δ | Δ ~ N(δ_post, Λ_post)
//!
//! Computed via stable Cholesky solves (no explicit matrix inversion).
//!
//! ## Leak Probability
//!
//! P(leak | Δ) = P(max_k |δ_k| > θ_eff | Δ)
//!
//! Computed via Monte Carlo: draw samples from posterior, count exceedances.
//!
//! ## 2D Projection for Reporting
//!
//! The 9D posterior is projected to 2D (shift, tail) using GLS for interpretability.

extern crate alloc;

use alloc::vec::Vec;
use core::f64::consts::PI;

use nalgebra::Cholesky;
use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::constants::{B_TAIL, ONES};
use crate::math;
use crate::types::{Matrix2, Matrix9, Matrix9x2, Vector2, Vector9};

/// Number of Monte Carlo samples for leak probability estimation.
const N_MONTE_CARLO: usize = 1000;

/// Result from Bayesian analysis using 9D inference.
///
/// ## Mixture Posterior (v5.2)
///
/// When using a 2-component mixture prior, the posterior is also a mixture:
/// - Narrow component: N(δ_post_narrow, Λ_post_narrow) with weight narrow_weight_post
/// - Slab component: N(δ_post_slab, Λ_post_slab) with weight slab_weight_post
///
/// The `delta_post` and `lambda_post` fields are set to the weighted mixture mean
/// for backwards compatibility. For accurate inference, use the component-specific fields.
#[derive(Debug, Clone)]
pub struct BayesResult {
    /// Posterior probability of a significant leak: P(max_k |δ_k| > θ | Δ).
    pub leak_probability: f64,

    /// 9D posterior mean δ_post in nanoseconds.
    /// For mixture posteriors, this is the weighted mixture mean.
    pub delta_post: Vector9,

    /// 9D posterior covariance Λ_post.
    /// For mixture posteriors, this is set to the dominant component's covariance.
    pub lambda_post: Matrix9,

    /// Narrow component posterior mean (v5.2).
    pub delta_post_narrow: Vector9,

    /// Narrow component posterior covariance (v5.2).
    pub lambda_post_narrow: Matrix9,

    /// Slab component posterior mean (v5.2).
    pub delta_post_slab: Vector9,

    /// Slab component posterior covariance (v5.2).
    pub lambda_post_slab: Matrix9,

    /// Posterior weight for narrow component (v5.2).
    /// If this is 0, only slab component was used (single Gaussian case).
    pub narrow_weight_post: f64,

    /// Posterior weight for slab component: 1 - narrow_weight_post (v5.2).
    pub slab_weight_post: f64,

    /// 2D GLS projection β = (μ, τ) in nanoseconds.
    /// - β[0] = μ (uniform shift): All quantiles move equally
    /// - β[1] = τ (tail effect): Upper quantiles shift more than lower
    pub beta_proj: Vector2,

    /// 2D projection covariance.
    pub beta_proj_cov: Matrix2,

    /// Projection mismatch Q statistic: r'Σ_n⁻¹r where r = δ_post - Xβ_proj.
    /// High values indicate the 2D model doesn't capture the full pattern.
    pub projection_mismatch_q: f64,

    /// 95% credible interval for max effect magnitude: (2.5th, 97.5th) percentiles.
    pub effect_magnitude_ci: (f64, f64),

    /// Whether the computation encountered numerical issues.
    /// If true, the posterior is set to the prior (maximally uncertain).
    pub is_clamped: bool,

    /// Covariance matrix used for inference (after regularization).
    pub sigma_n: Matrix9,

    // ==================== v5.4 Gibbs sampler diagnostics ====================

    /// Posterior mean of latent scale λ (v5.4 Gibbs).
    /// Only populated when using `compute_bayes_gibbs()`.
    pub lambda_mean: f64,

    /// Posterior standard deviation of λ (v5.4 Gibbs).
    pub lambda_sd: f64,

    /// Coefficient of variation: λ_sd / λ_mean (v5.4 Gibbs).
    pub lambda_cv: f64,

    /// Effective sample size of λ chain (v5.4 Gibbs).
    pub lambda_ess: f64,

    /// True if mixing diagnostics pass: CV ≥ 0.1 AND ESS ≥ 20 (v5.4 Gibbs).
    pub lambda_mixing_ok: bool,
}

/// Compute Bayesian posterior for timing leak analysis using 9D model (single Gaussian prior).
///
/// **Note**: For v5.2 mixture prior, use `compute_bayes_factor_mixture` instead.
///
/// # Arguments
///
/// * `delta` - Observed quantile differences (9-vector)
/// * `sigma_n` - Covariance matrix scaled for inference sample size (Σ_rate / n)
/// * `lambda0` - Prior covariance (9×9, shaped: σ²_prior × S)
/// * `theta` - Minimum effect of concern (threshold for practical significance)
/// * `seed` - Random seed for Monte Carlo reproducibility
///
/// # Returns
///
/// `BayesResult` with 9D posterior, 2D projection, and leak probability.
pub fn compute_bayes_factor(
    delta: &Vector9,
    sigma_n: &Matrix9,
    lambda0: &Matrix9,
    theta: f64,
    seed: Option<u64>,
) -> BayesResult {
    // Regularize covariance for numerical stability
    let regularized = add_jitter(*sigma_n);

    // Compute single-component posterior
    let (delta_post, lambda_post, _log_ml) =
        match compute_component_posterior(delta, &regularized, lambda0) {
            Some(result) => result,
            None => {
                return neutral_result(&regularized, lambda0);
            }
        };

    // Compute 2D GLS projection
    let (beta_proj, beta_proj_cov, projection_mismatch_q) =
        compute_2d_projection(&delta_post, &lambda_post, &regularized);

    // Monte Carlo for leak probability and effect CI
    let (leak_probability, effect_magnitude_ci) = run_monte_carlo(
        &delta_post,
        &lambda_post,
        theta,
        seed.unwrap_or(crate::constants::DEFAULT_SEED),
    );

    BayesResult {
        leak_probability,
        delta_post,
        lambda_post,
        // For single Gaussian, set mixture fields to the same values
        delta_post_narrow: delta_post,
        lambda_post_narrow: lambda_post,
        delta_post_slab: delta_post,
        lambda_post_slab: lambda_post,
        narrow_weight_post: 1.0, // All weight on single component
        slab_weight_post: 0.0,
        beta_proj,
        beta_proj_cov,
        projection_mismatch_q,
        effect_magnitude_ci,
        is_clamped: false,
        sigma_n: regularized,
        // v5.4 Gibbs fields - not used in single Gaussian mode
        lambda_mean: 1.0,
        lambda_sd: 0.0,
        lambda_cv: 0.0,
        lambda_ess: 0.0,
        lambda_mixing_ok: true,
    }
}

/// Compute Bayesian posterior using 2-component mixture prior (v5.2).
///
/// Mixture prior: π(δ) = w·N(0, Λ₀,₁) + (1−w)·N(0, Λ₀,₂)
///
/// # Arguments
///
/// * `delta` - Observed quantile differences (9-vector)
/// * `sigma_n` - Covariance matrix scaled for inference sample size (Σ_rate / n)
/// * `lambda0_narrow` - Narrow component prior covariance (Λ₀,₁ = σ₁²R)
/// * `lambda0_slab` - Slab component prior covariance (Λ₀,₂ = σ₂²R)
/// * `prior_weight` - Weight w for narrow component (default 0.99)
/// * `theta` - Minimum effect of concern (threshold)
/// * `seed` - Random seed for Monte Carlo reproducibility
///
/// # Returns
///
/// `BayesResult` with mixture posterior.
pub fn compute_bayes_factor_mixture(
    delta: &Vector9,
    sigma_n: &Matrix9,
    lambda0_narrow: &Matrix9,
    lambda0_slab: &Matrix9,
    prior_weight: f64,
    theta: f64,
    seed: Option<u64>,
) -> BayesResult {
    let regularized = add_jitter(*sigma_n);
    let actual_seed = seed.unwrap_or(crate::constants::DEFAULT_SEED);

    // Compute per-component posteriors with log marginal likelihoods
    let narrow_result = compute_component_posterior(delta, &regularized, lambda0_narrow);
    let slab_result = compute_component_posterior(delta, &regularized, lambda0_slab);

    // Handle failures
    let (delta_post_1, lambda_post_1, log_ml_1) = match narrow_result {
        Some(r) => r,
        None => {
            return neutral_result_mixture(&regularized, lambda0_narrow, lambda0_slab);
        }
    };

    let (delta_post_2, lambda_post_2, log_ml_2) = match slab_result {
        Some(r) => r,
        None => {
            return neutral_result_mixture(&regularized, lambda0_narrow, lambda0_slab);
        }
    };

    // Posterior weights via log-sum-exp for numerical stability
    let log_w1 = math::ln(prior_weight.max(1e-12)) + log_ml_1;
    let log_w2 = math::ln((1.0 - prior_weight).max(1e-12)) + log_ml_2;
    let log_norm = log_sum_exp(log_w1, log_w2);

    let narrow_weight_post = math::exp(log_w1 - log_norm);
    let slab_weight_post = 1.0 - narrow_weight_post;

    #[cfg(feature = "std")]
    if std::env::var("TIMING_ORACLE_DEBUG").is_ok() {
        eprintln!("[DEBUG bayes] log_ml_narrow = {:.2}, log_ml_slab = {:.2}", log_ml_1, log_ml_2);
        eprintln!("[DEBUG bayes] log_w1 (narrow) = {:.2}, log_w2 (slab) = {:.2}", log_w1, log_w2);
        eprintln!("[DEBUG bayes] log_norm = {:.2}", log_norm);
    }

    // Monte Carlo from mixture posterior
    let (leak_probability, effect_magnitude_ci) = run_mixture_monte_carlo(
        &delta_post_1,
        &lambda_post_1,
        &delta_post_2,
        &lambda_post_2,
        narrow_weight_post,
        theta,
        actual_seed,
    );

    // Mixture mean for compatibility
    let delta_post = delta_post_1 * narrow_weight_post + delta_post_2 * slab_weight_post;

    // Use dominant component's covariance for compatibility
    let lambda_post = if narrow_weight_post >= 0.5 {
        lambda_post_1
    } else {
        lambda_post_2
    };

    // Compute 2D GLS projection from mixture mean
    let (beta_proj, beta_proj_cov, projection_mismatch_q) =
        compute_2d_projection(&delta_post, &lambda_post, &regularized);

    BayesResult {
        leak_probability,
        delta_post,
        lambda_post,
        delta_post_narrow: delta_post_1,
        lambda_post_narrow: lambda_post_1,
        delta_post_slab: delta_post_2,
        lambda_post_slab: lambda_post_2,
        narrow_weight_post,
        slab_weight_post,
        beta_proj,
        beta_proj_cov,
        projection_mismatch_q,
        effect_magnitude_ci,
        is_clamped: false,
        sigma_n: regularized,
        // v5.4 Gibbs fields - not used in mixture mode
        lambda_mean: 1.0,
        lambda_sd: 0.0,
        lambda_cv: 0.0,
        lambda_ess: 0.0,
        lambda_mixing_ok: true,
    }
}

/// Compute Bayesian posterior using Student's t prior with Gibbs sampling (v5.4).
///
/// This replaces the v5.2 mixture prior, using a Student's t prior (ν=4) that
/// is more robust to correlation-induced pathologies. The t-prior is represented
/// as a scale mixture of Gaussians and sampled via Gibbs.
///
/// # Arguments
///
/// * `delta` - Observed quantile differences (9-vector)
/// * `sigma_n` - Covariance matrix scaled for inference sample size (Σ_rate / n)
/// * `sigma_t` - Calibrated Student's t prior scale
/// * `l_r` - Cholesky factor of correlation matrix R
/// * `theta` - Minimum effect of concern (threshold)
/// * `seed` - Random seed for Gibbs sampling reproducibility
///
/// # Returns
///
/// `BayesResult` with posterior from Gibbs sampling, including lambda diagnostics.
pub fn compute_bayes_gibbs(
    delta: &Vector9,
    sigma_n: &Matrix9,
    sigma_t: f64,
    l_r: &Matrix9,
    theta: f64,
    seed: Option<u64>,
) -> BayesResult {
    use super::gibbs::run_gibbs_inference;

    let regularized = add_jitter(*sigma_n);
    let actual_seed = seed.unwrap_or(crate::constants::DEFAULT_SEED);

    // Run Gibbs sampler
    let gibbs_result = run_gibbs_inference(delta, &regularized, sigma_t, l_r, theta, actual_seed);

    BayesResult {
        leak_probability: gibbs_result.leak_probability,
        delta_post: gibbs_result.delta_post,
        lambda_post: gibbs_result.lambda_post,
        // For t-prior, no mixture components - set to same values
        delta_post_narrow: gibbs_result.delta_post,
        lambda_post_narrow: gibbs_result.lambda_post,
        delta_post_slab: gibbs_result.delta_post,
        lambda_post_slab: gibbs_result.lambda_post,
        narrow_weight_post: 1.0,
        slab_weight_post: 0.0,
        beta_proj: gibbs_result.beta_proj,
        beta_proj_cov: gibbs_result.beta_proj_cov,
        projection_mismatch_q: gibbs_result.projection_mismatch_q,
        effect_magnitude_ci: gibbs_result.effect_magnitude_ci,
        is_clamped: false,
        sigma_n: regularized,
        // v5.4 Gibbs diagnostics
        lambda_mean: gibbs_result.lambda_mean,
        lambda_sd: gibbs_result.lambda_sd,
        lambda_cv: gibbs_result.lambda_cv,
        lambda_ess: gibbs_result.lambda_ess,
        lambda_mixing_ok: gibbs_result.lambda_mixing_ok,
    }
}

/// Compute single-component Gaussian posterior.
///
/// Returns (δ_post, Λ_post, log_ml) where log_ml is the log marginal likelihood.
fn compute_component_posterior(
    delta: &Vector9,
    sigma_n: &Matrix9,
    lambda0: &Matrix9,
) -> Option<(Vector9, Matrix9, f64)> {
    // A = Σ_n + Λ₀ (SPD)
    let a = *sigma_n + *lambda0;
    let a_chol = Cholesky::new(a)?;

    // Solve A*x = Δ, then δ_post = Λ₀ * x
    let x = a_chol.solve(delta);
    let delta_post = lambda0 * x;

    // Λ_post = Λ₀ - Λ₀ * A⁻¹ * Λ₀ (via Cholesky solves)
    let mut a_inv_lambda0 = Matrix9::zeros();
    for j in 0..9 {
        let col = lambda0.column(j).into_owned();
        let y_col = a_chol.solve(&col);
        for i in 0..9 {
            a_inv_lambda0[(i, j)] = y_col[i];
        }
    }
    let mut lambda_post = lambda0 - lambda0 * a_inv_lambda0;

    // Ensure symmetry
    for i in 0..9 {
        for j in (i + 1)..9 {
            let avg = 0.5 * (lambda_post[(i, j)] + lambda_post[(j, i)]);
            lambda_post[(i, j)] = avg;
            lambda_post[(j, i)] = avg;
        }
    }

    // Log marginal likelihood: log N(Δ; 0, A)
    // = -0.5 * (9*log(2π) + log|A| + Δ'A⁻¹Δ)
    let log_det_a: f64 = 2.0 * a_chol.l().diagonal().iter().map(|&x| math::ln(x.abs().max(1e-300))).sum::<f64>();
    let quad_form = delta.dot(&a_chol.solve(delta));
    let log_ml = -0.5 * (9.0 * math::ln(2.0 * PI) + log_det_a + quad_form);

    #[cfg(feature = "std")]
    if std::env::var("TIMING_ORACLE_DEBUG_ML").is_ok() {
        eprintln!(
            "[DEBUG ML] log_det_a = {:.2}, quad_form = {:.2}, log_ml = {:.2}",
            log_det_a, quad_form, log_ml
        );
        eprintln!("[DEBUG ML] lambda0 diag = [{:.2e}, {:.2e}, ..., {:.2e}]",
            lambda0[(0, 0)], lambda0[(1, 1)], lambda0[(8, 8)]);
    }

    Some((delta_post, lambda_post, log_ml))
}

/// Log-sum-exp for numerical stability: log(exp(a) + exp(b)).
fn log_sum_exp(a: f64, b: f64) -> f64 {
    let max_val = a.max(b);
    if max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    max_val + math::ln(math::exp(a - max_val) + math::exp(b - max_val))
}

/// Monte Carlo from mixture posterior.
fn run_mixture_monte_carlo(
    delta_post_1: &Vector9,
    lambda_post_1: &Matrix9,
    delta_post_2: &Vector9,
    lambda_post_2: &Matrix9,
    narrow_weight: f64,
    theta: f64,
    seed: u64,
) -> (f64, (f64, f64)) {
    // Cholesky decompositions for sampling
    let chol_1 = match Cholesky::new(*lambda_post_1) {
        Some(c) => c,
        None => match Cholesky::new(add_jitter(*lambda_post_1)) {
            Some(c) => c,
            None => return (0.5, (0.0, 0.0)),
        },
    };

    let chol_2 = match Cholesky::new(*lambda_post_2) {
        Some(c) => c,
        None => match Cholesky::new(add_jitter(*lambda_post_2)) {
            Some(c) => c,
            None => return (0.5, (0.0, 0.0)),
        },
    };

    let l_1 = chol_1.l();
    let l_2 = chol_2.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut count = 0usize;
    let mut max_effects = Vec::with_capacity(N_MONTE_CARLO);

    for _ in 0..N_MONTE_CARLO {
        // Sample z ~ N(0, I_9)
        let mut z = Vector9::zeros();
        for i in 0..9 {
            z[i] = sample_standard_normal(&mut rng);
        }

        // Sample from mixture: with probability narrow_weight, use narrow component
        let delta_sample = if rng.random::<f64>() < narrow_weight {
            delta_post_1 + l_1 * z
        } else {
            delta_post_2 + l_2 * z
        };

        // Compute max_k |δ_k|
        let max_effect = delta_sample.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        max_effects.push(max_effect);

        if max_effect > theta {
            count += 1;
        }
    }

    // Compute 95% CI for max effect
    max_effects.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = math::round((N_MONTE_CARLO as f64) * 0.025) as usize;
    let hi_idx = math::round((N_MONTE_CARLO as f64) * 0.975) as usize;
    let ci = (
        max_effects[lo_idx.min(N_MONTE_CARLO - 1)],
        max_effects[hi_idx.min(N_MONTE_CARLO - 1)],
    );

    (count as f64 / N_MONTE_CARLO as f64, ci)
}

/// Compute 2D GLS projection of 9D posterior.
///
/// Projects δ_post onto the shift+tail basis X = [1 | b_tail] using
/// generalized least squares: β = (X'Σ_n⁻¹X)⁻¹ X'Σ_n⁻¹ δ_post
///
/// Returns (β_proj, β_proj_cov, Q_proj) where Q_proj is the projection mismatch statistic.
pub fn compute_2d_projection(
    delta_post: &Vector9,
    lambda_post: &Matrix9,
    sigma_n: &Matrix9,
) -> (Vector2, Matrix2, f64) {
    let design = build_design_matrix();

    // Cholesky of Σ_n for stable solves
    let sigma_n_chol = match Cholesky::new(*sigma_n) {
        Some(c) => c,
        None => {
            // Fallback: zero projection
            return (Vector2::zeros(), Matrix2::identity() * 1e6, 0.0);
        }
    };

    // Σ_n⁻¹ X via solve
    let mut sigma_n_inv_x = Matrix9x2::zeros();
    for j in 0..2 {
        let col = design.column(j).into_owned();
        let solved = sigma_n_chol.solve(&col);
        for i in 0..9 {
            sigma_n_inv_x[(i, j)] = solved[i];
        }
    }

    // X' Σ_n⁻¹ X (2×2)
    let xt_sigma_n_inv_x = design.transpose() * sigma_n_inv_x;

    let xt_chol = match Cholesky::new(xt_sigma_n_inv_x) {
        Some(c) => c,
        None => {
            return (Vector2::zeros(), Matrix2::identity() * 1e6, 0.0);
        }
    };

    // X' Σ_n⁻¹ δ_post
    let sigma_n_inv_delta = sigma_n_chol.solve(delta_post);
    let xt_sigma_n_inv_delta = design.transpose() * sigma_n_inv_delta;

    // β_proj = (X' Σ_n⁻¹ X)⁻¹ X' Σ_n⁻¹ δ_post
    let beta_proj = xt_chol.solve(&xt_sigma_n_inv_delta);

    // Projection covariance: Cov(β_proj | Δ) = A Λ_post A'
    // where A = (X' Σ_n⁻¹ X)⁻¹ X' Σ_n⁻¹
    //
    // Note: X' Σ_n⁻¹ = (Σ_n⁻¹ X)' = sigma_n_inv_x' (since Σ_n is symmetric)
    // We already computed sigma_n_inv_x = Σ_n⁻¹ X correctly via Cholesky solve.
    let a_matrix = xt_chol.inverse() * sigma_n_inv_x.transpose();
    let beta_proj_cov = a_matrix * lambda_post * a_matrix.transpose();

    // Projection mismatch: Q = r' Σ_n⁻¹ r where r = δ_post - X β_proj
    let delta_proj = design * beta_proj;
    let r_proj = delta_post - delta_proj;
    let sigma_n_inv_r = sigma_n_chol.solve(&r_proj);
    let q_proj = r_proj.dot(&sigma_n_inv_r);

    (beta_proj, beta_proj_cov, q_proj)
}

/// Monte Carlo integration for leak probability and effect CI.
///
/// Samples from the 9D posterior and computes:
/// - P(max_k |δ_k| > θ | Δ)
/// - 95% CI for max_k |δ_k|
fn run_monte_carlo(
    delta_post: &Vector9,
    lambda_post: &Matrix9,
    theta: f64,
    seed: u64,
) -> (f64, (f64, f64)) {
    // Cholesky decomposition of posterior covariance for sampling
    let chol = match Cholesky::new(*lambda_post) {
        Some(c) => c,
        None => {
            // Try with jitter
            let jittered = add_jitter(*lambda_post);
            match Cholesky::new(jittered) {
                Some(c) => c,
                None => return (0.5, (0.0, 0.0)),
            }
        }
    };
    let l = chol.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut count = 0usize;
    let mut max_effects = Vec::with_capacity(N_MONTE_CARLO);

    for _ in 0..N_MONTE_CARLO {
        // Sample z ~ N(0, I_9)
        let mut z = Vector9::zeros();
        for i in 0..9 {
            z[i] = sample_standard_normal(&mut rng);
        }

        // Transform to δ ~ N(δ_post, Λ_post)
        let delta_sample = delta_post + l * z;

        // Compute max_k |δ_k|
        let max_effect = delta_sample.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        max_effects.push(max_effect);

        if max_effect > theta {
            count += 1;
        }
    }

    // Compute 95% CI for max effect
    max_effects.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = math::round((N_MONTE_CARLO as f64) * 0.025) as usize;
    let hi_idx = math::round((N_MONTE_CARLO as f64) * 0.975) as usize;
    let ci = (
        max_effects[lo_idx.min(N_MONTE_CARLO - 1)],
        max_effects[hi_idx.min(N_MONTE_CARLO - 1)],
    );

    (count as f64 / N_MONTE_CARLO as f64, ci)
}

/// Sample from standard normal using Box-Muller transform.
fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    math::sqrt(-2.0 * math::ln(u1)) * math::cos(2.0 * PI * u2)
}

/// Result from max effect CI computation for Research mode.
#[derive(Debug, Clone)]
pub struct MaxEffectCI {
    /// Posterior mean of max_k |δ_k|.
    pub mean: f64,
    /// 95% credible interval for max_k |δ_k|: (2.5th, 97.5th percentile).
    pub ci: (f64, f64),
}

/// Compute 95% CI for max effect: max_k |δ_k|.
///
/// Used by Research mode for stopping conditions.
pub fn compute_max_effect_ci(
    delta_post: &Vector9,
    lambda_post: &Matrix9,
    seed: u64,
) -> MaxEffectCI {
    let chol = match Cholesky::new(*lambda_post) {
        Some(c) => c,
        None => {
            let jittered = add_jitter(*lambda_post);
            match Cholesky::new(jittered) {
                Some(c) => c,
                None => {
                    return MaxEffectCI {
                        mean: 0.0,
                        ci: (0.0, 0.0),
                    };
                }
            }
        }
    };
    let l = chol.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut max_effects = Vec::with_capacity(N_MONTE_CARLO);
    let mut sum = 0.0;

    for _ in 0..N_MONTE_CARLO {
        let mut z = Vector9::zeros();
        for i in 0..9 {
            z[i] = sample_standard_normal(&mut rng);
        }

        let delta_sample = delta_post + l * z;
        let max_effect = delta_sample.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        max_effects.push(max_effect);
        sum += max_effect;
    }

    let mean = sum / N_MONTE_CARLO as f64;

    max_effects.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = math::round((N_MONTE_CARLO as f64) * 0.025) as usize;
    let hi_idx = math::round((N_MONTE_CARLO as f64) * 0.975) as usize;
    let ci = (
        max_effects[lo_idx.min(N_MONTE_CARLO - 1)],
        max_effects[hi_idx.min(N_MONTE_CARLO - 1)],
    );

    MaxEffectCI { mean, ci }
}

/// Compute per-quantile exceedance probability P(|δ_k| > θ | Δ).
///
/// Returns a vector of 9 exceedance probabilities, one per decile.
pub fn compute_quantile_exceedances(
    delta_post: &Vector9,
    lambda_post: &Matrix9,
    theta: f64,
) -> [f64; 9] {
    let mut exceedances = [0.0; 9];
    for k in 0..9 {
        let mu = delta_post[k];
        let sigma = math::sqrt(lambda_post[(k, k)].max(1e-12));
        exceedances[k] = compute_single_quantile_exceedance(mu, sigma, theta);
    }
    exceedances
}

/// Compute P(|δ| > θ) for a single Gaussian marginal N(μ, σ²).
fn compute_single_quantile_exceedance(mu: f64, sigma: f64, theta: f64) -> f64 {
    if sigma < 1e-12 {
        // Degenerate case: point mass
        return if mu.abs() > theta { 1.0 } else { 0.0 };
    }
    let phi_upper = math::normal_cdf((theta - mu) / sigma);
    let phi_lower = math::normal_cdf((-theta - mu) / sigma);
    1.0 - (phi_upper - phi_lower)
}

/// Return a neutral result when computation fails (maximally uncertain).
fn neutral_result(sigma_n: &Matrix9, lambda0: &Matrix9) -> BayesResult {
    BayesResult {
        leak_probability: 0.5,
        delta_post: Vector9::zeros(),
        lambda_post: *lambda0,
        delta_post_narrow: Vector9::zeros(),
        lambda_post_narrow: *lambda0,
        delta_post_slab: Vector9::zeros(),
        lambda_post_slab: *lambda0,
        narrow_weight_post: 1.0,
        slab_weight_post: 0.0,
        beta_proj: Vector2::zeros(),
        beta_proj_cov: Matrix2::identity() * 1e6,
        projection_mismatch_q: 0.0,
        effect_magnitude_ci: (0.0, 0.0),
        is_clamped: true,
        sigma_n: *sigma_n,
        lambda_mean: 1.0,
        lambda_sd: 0.0,
        lambda_cv: 0.0,
        lambda_ess: 0.0,
        lambda_mixing_ok: true,
    }
}

/// Return a neutral result for mixture prior when computation fails.
fn neutral_result_mixture(sigma_n: &Matrix9, lambda0_narrow: &Matrix9, lambda0_slab: &Matrix9) -> BayesResult {
    BayesResult {
        leak_probability: 0.5,
        delta_post: Vector9::zeros(),
        lambda_post: *lambda0_narrow,
        delta_post_narrow: Vector9::zeros(),
        lambda_post_narrow: *lambda0_narrow,
        delta_post_slab: Vector9::zeros(),
        lambda_post_slab: *lambda0_slab,
        narrow_weight_post: 0.5,
        slab_weight_post: 0.5,
        beta_proj: Vector2::zeros(),
        beta_proj_cov: Matrix2::identity() * 1e6,
        projection_mismatch_q: 0.0,
        effect_magnitude_ci: (0.0, 0.0),
        is_clamped: true,
        sigma_n: *sigma_n,
        lambda_mean: 1.0,
        lambda_sd: 0.0,
        lambda_cv: 0.0,
        lambda_ess: 0.0,
        lambda_mixing_ok: true,
    }
}

/// Build design matrix for 2D projection (shift + tail).
///
/// X = [1 | b_tail] where:
/// - Column 0 (ones): Uniform shift - all quantiles move equally
/// - Column 1 (b_tail): Tail effect - upper quantiles shift more than lower
pub fn build_design_matrix() -> Matrix9x2 {
    let mut x = Matrix9x2::zeros();
    for i in 0..9 {
        x[(i, 0)] = ONES[i];
        x[(i, 1)] = B_TAIL[i];
    }
    x
}

/// Apply variance floor regularization for numerical stability.
///
/// Ensures minimum diagonal value of 1% of mean variance.
fn add_jitter(mut sigma: Matrix9) -> Matrix9 {
    let trace: f64 = (0..9).map(|i| sigma[(i, i)]).sum();
    let mean_var = trace / 9.0;

    let min_var = (0.01 * mean_var).max(1e-10);
    let jitter = 1e-10 + mean_var * 1e-8;

    for i in 0..9 {
        sigma[(i, i)] = sigma[(i, i)].max(min_var) + jitter;
    }
    sigma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_9d_posterior_with_zero_delta() {
        let delta = Vector9::zeros();
        let sigma_n = Matrix9::identity();
        let lambda0 = Matrix9::identity() * 100.0; // Wide prior

        let result = compute_bayes_factor(&delta, &sigma_n, &lambda0, 10.0, Some(42));

        // With zero observation, posterior should be pulled toward zero
        assert!(
            result.delta_post.norm() < 1.0,
            "Posterior should be near zero"
        );
        assert!(!result.is_clamped);
    }

    #[test]
    fn test_9d_posterior_with_large_delta() {
        let mut delta = Vector9::zeros();
        for i in 0..9 {
            delta[i] = 100.0; // Large uniform shift
        }
        let sigma_n = Matrix9::identity();
        let lambda0 = Matrix9::identity() * 100.0;

        let result = compute_bayes_factor(&delta, &sigma_n, &lambda0, 10.0, Some(42));

        // With large observation, leak probability should be high
        assert!(
            result.leak_probability > 0.9,
            "Large delta should give high leak probability, got {}",
            result.leak_probability
        );
    }

    #[test]
    fn test_monte_carlo_determinism() {
        let delta = Vector9::from_row_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
        let sigma_n = Matrix9::identity();
        let lambda0 = Matrix9::identity() * 100.0;

        let result1 = compute_bayes_factor(&delta, &sigma_n, &lambda0, 10.0, Some(42));
        let result2 = compute_bayes_factor(&delta, &sigma_n, &lambda0, 10.0, Some(42));

        assert_eq!(
            result1.leak_probability, result2.leak_probability,
            "Same seed should give same result"
        );
    }

    #[test]
    fn test_2d_projection_uniform_shift() {
        // Create a uniform shift pattern (all quantiles equal)
        let mut delta_post = Vector9::zeros();
        for i in 0..9 {
            delta_post[i] = 50.0;
        }
        let lambda_post = Matrix9::identity();
        let sigma_n = Matrix9::identity();

        let (beta_proj, _, q_proj) = compute_2d_projection(&delta_post, &lambda_post, &sigma_n);

        // Should project to pure shift with low mismatch
        assert!(
            (beta_proj[0] - 50.0).abs() < 1.0,
            "Shift should be ~50, got {}",
            beta_proj[0]
        );
        assert!(
            beta_proj[1].abs() < 5.0,
            "Tail should be ~0, got {}",
            beta_proj[1]
        );
        assert!(
            q_proj < 1.0,
            "Uniform shift should have low Q, got {}",
            q_proj
        );
    }

    #[test]
    fn test_quantile_exceedance_computation() {
        let mu = 100.0;
        let sigma = 10.0;
        let theta = 50.0;

        let exceedance = compute_single_quantile_exceedance(mu, sigma, theta);

        // μ = 100, θ = 50: almost certainly |δ| > θ
        assert!(
            exceedance > 0.99,
            "With μ=100, θ=50, exceedance should be ~1.0, got {}",
            exceedance
        );
    }

    #[test]
    fn test_quantile_exceedance_symmetric() {
        let sigma = 10.0;
        let theta = 50.0;

        let exc_pos = compute_single_quantile_exceedance(30.0, sigma, theta);
        let exc_neg = compute_single_quantile_exceedance(-30.0, sigma, theta);

        // Should be symmetric
        assert!(
            (exc_pos - exc_neg).abs() < 0.01,
            "Exceedance should be symmetric, got {} vs {}",
            exc_pos,
            exc_neg
        );
    }

    /// Regression test for spec v5.2: large effect + noisy likelihood regime with mixture prior.
    ///
    /// This test ensures that obvious timing leaks (effect ≫ θ) are detected
    /// even when measurement noise is high (SE ≫ θ). The mixture prior (v5.2)
    /// allows the posterior to escape to the slab component when data strongly
    /// supports large effects.
    ///
    /// Test values based on SILENT web app dataset:
    /// - θ_eff = 100 ns
    /// - True effect ≈ 100× threshold (10,000-18,000 ns)
    /// - Data SEs ≈ 25-80× threshold (2,500-8,000 ns)
    ///
    /// With the v5.2 mixture prior, P(leak) should be > 0.99.
    #[test]
    fn test_large_effect_noisy_likelihood_regression() {
        use crate::adaptive::{
            calibrate_narrow_scale, compute_prior_cov_9d, compute_slab_scale, MIXTURE_PRIOR_WEIGHT,
        };

        // Fixed values based on SILENT web app dataset
        let delta = Vector9::from_row_slice(&[
            10366.0, 13156.0, 13296.0, 12800.0, 11741.0, 12936.0, 13215.0, 11804.0, 18715.0,
        ]);

        // Variance = SE² (diagonal covariance)
        let ses = [2731.0, 2796.0, 2612.0, 2555.0, 2734.0, 3125.0, 3953.0, 5662.0, 8105.0];
        let mut sigma_n = Matrix9::zeros();
        for i in 0..9 {
            sigma_n[(i, i)] = ses[i] * ses[i];
        }

        // Sigma_rate = Sigma_n * n (n = 10,000 samples)
        let n = 10_000usize;
        let sigma_rate = sigma_n * (n as f64);

        let theta_eff = 100.0;
        let discrete_mode = false;
        let seed = 0xDEADBEEF_u64;

        // v5.2 mixture prior: compute slab first, then calibrate narrow
        let sigma_slab = compute_slab_scale(&sigma_rate, theta_eff, n);
        let sigma_narrow = calibrate_narrow_scale(
            &sigma_rate,
            theta_eff,
            n,
            sigma_slab,
            MIXTURE_PRIOR_WEIGHT,
            discrete_mode,
            seed,
        );
        let prior_cov_narrow = compute_prior_cov_9d(&sigma_rate, sigma_narrow, discrete_mode);
        let prior_cov_slab = compute_prior_cov_9d(&sigma_rate, sigma_slab, discrete_mode);

        // Debug output
        eprintln!("sigma_narrow = {}, sigma_slab = {}", sigma_narrow, sigma_slab);
        eprintln!(
            "Narrow prior diagonal: [{:.2e}, {:.2e}, ..., {:.2e}]",
            prior_cov_narrow[(0, 0)],
            prior_cov_narrow[(1, 1)],
            prior_cov_narrow[(8, 8)]
        );
        eprintln!(
            "Slab prior diagonal: [{:.2e}, {:.2e}, ..., {:.2e}]",
            prior_cov_slab[(0, 0)],
            prior_cov_slab[(1, 1)],
            prior_cov_slab[(8, 8)]
        );
        eprintln!(
            "Sigma_n diagonal: [{:.2e}, {:.2e}, ..., {:.2e}]",
            sigma_n[(0, 0)],
            sigma_n[(1, 1)],
            sigma_n[(8, 8)]
        );

        // Compute Bayes factor with mixture prior
        let result = compute_bayes_factor_mixture(
            &delta,
            &sigma_n,
            &prior_cov_narrow,
            &prior_cov_slab,
            MIXTURE_PRIOR_WEIGHT,
            theta_eff,
            Some(seed),
        );

        eprintln!("P(leak) = {}", result.leak_probability);
        eprintln!(
            "narrow_weight_post = {:.4}, slab_weight_post = {:.4}",
            result.narrow_weight_post, result.slab_weight_post
        );
        eprintln!(
            "delta_post (mixture) = [{:.1}, {:.1}, {:.1}]",
            result.delta_post[0], result.delta_post[1], result.delta_post[2]);

        // With effect ≈ 100× threshold, P(leak) MUST be > 0.99
        assert!(
            result.leak_probability > 0.99,
            "Large effect ({:.0}ns mean) with θ={:.0}ns should give P(leak) > 0.99, got {:.4}. \
             Slab weight = {:.4}. This regression indicates the v5.2 mixture prior may have broken.",
            delta.iter().sum::<f64>() / 9.0,
            theta_eff,
            result.leak_probability,
            result.slab_weight_post
        );
    }
}
