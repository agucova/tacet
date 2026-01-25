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

/// Result from Bayesian analysis using 9D inference with Gibbs sampling (v5.4+).
///
/// The posterior is computed using a Student's t prior (ν=4) via Gibbs sampling,
/// which is robust to correlation-induced pathologies.
#[derive(Debug, Clone)]
pub struct BayesResult {
    /// Posterior probability of a significant leak: P(max_k |δ_k| > θ | Δ).
    pub leak_probability: f64,

    /// 9D posterior mean δ_post in nanoseconds.
    pub delta_post: Vector9,

    /// 9D posterior covariance Λ_post.
    pub lambda_post: Matrix9,

    /// 2D GLS projection β = (μ, τ) in nanoseconds.
    /// - β[0] = μ (uniform shift): All quantiles move equally
    /// - β[1] = τ (tail effect): Upper quantiles shift more than lower
    pub beta_proj: Vector2,

    /// 2D projection covariance (empirical, from draws).
    pub beta_proj_cov: Matrix2,

    /// Retained β draws for dominance-based classification (spec §3.4.6).
    /// Each draw is β^(s) = A·δ^(s) where A is the GLS projection matrix.
    pub beta_draws: Vec<Vector2>,

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

    // ==================== v5.6 Gibbs sampler kappa diagnostics ====================
    /// v5.6: Posterior mean of likelihood precision κ.
    pub kappa_mean: f64,

    /// v5.6: Posterior standard deviation of κ.
    pub kappa_sd: f64,

    /// v5.6: Coefficient of variation: κ_sd / κ_mean.
    pub kappa_cv: f64,

    /// v5.6: Effective sample size of κ chain.
    pub kappa_ess: f64,

    /// v5.6: Whether κ mixing diagnostics pass: CV ≥ 0.1 AND ESS ≥ 20.
    pub kappa_mixing_ok: bool,
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
        beta_proj: gibbs_result.beta_proj,
        beta_proj_cov: gibbs_result.beta_proj_cov,
        beta_draws: gibbs_result.beta_draws,
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
        // v5.6 kappa diagnostics
        kappa_mean: gibbs_result.kappa_mean,
        kappa_sd: gibbs_result.kappa_sd,
        kappa_cv: gibbs_result.kappa_cv,
        kappa_ess: gibbs_result.kappa_ess,
        kappa_mixing_ok: gibbs_result.kappa_mixing_ok,
    }
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

    #[test]
    fn test_gibbs_determinism() {
        use crate::adaptive::calibrate_t_prior_scale;

        let delta = Vector9::from_row_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
        let sigma_n = Matrix9::identity() * 100.0;
        let sigma_rate = sigma_n * 1000.0;
        let theta = 10.0;

        let (sigma_t, l_r) = calibrate_t_prior_scale(&sigma_rate, theta, 1000, false, 42);

        let result1 = compute_bayes_gibbs(&delta, &sigma_n, sigma_t, &l_r, theta, Some(42));
        let result2 = compute_bayes_gibbs(&delta, &sigma_n, sigma_t, &l_r, theta, Some(42));

        assert_eq!(
            result1.leak_probability, result2.leak_probability,
            "Same seed should give same result"
        );
    }
}
