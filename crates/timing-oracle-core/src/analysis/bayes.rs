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
}

/// Compute Bayesian posterior for timing leak analysis using 9D model.
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

    // Compute posterior using stable Cholesky path
    // A = Σ_n + Λ₀ (SPD)
    let a = regularized + *lambda0;
    let a_chol = match Cholesky::new(a) {
        Some(c) => c,
        None => {
            return neutral_result(&regularized, lambda0);
        }
    };

    // Solve A*x = Δ, then δ_post = Λ₀ * x
    let x = a_chol.solve(delta);
    let delta_post = lambda0 * x;

    // Λ_post = Λ₀ - Λ₀ * A⁻¹ * Λ₀ (via Cholesky solves)
    // Compute Y = A⁻¹ * Λ₀ by solving A*Y = Λ₀
    let mut a_inv_lambda0 = Matrix9::zeros();
    for j in 0..9 {
        let col = lambda0.column(j).into_owned();
        let y_col = a_chol.solve(&col);
        for i in 0..9 {
            a_inv_lambda0[(i, j)] = y_col[i];
        }
    }
    // Λ_post = Λ₀ - Λ₀ * Y
    let mut lambda_post = lambda0 - lambda0 * a_inv_lambda0;

    // Ensure symmetry (numerical errors can break it slightly)
    for i in 0..9 {
        for j in (i + 1)..9 {
            let avg = 0.5 * (lambda_post[(i, j)] + lambda_post[(j, i)]);
            lambda_post[(i, j)] = avg;
            lambda_post[(j, i)] = avg;
        }
    }

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
        beta_proj,
        beta_proj_cov,
        projection_mismatch_q,
        effect_magnitude_ci,
        is_clamped: false,
        sigma_n: regularized,
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
        beta_proj: Vector2::zeros(),
        beta_proj_cov: Matrix2::identity() * 1e6,
        projection_mismatch_q: 0.0,
        effect_magnitude_ci: (0.0, 0.0),
        is_clamped: true,
        sigma_n: *sigma_n,
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

    /// Regression test for spec v5.1: large effect + noisy likelihood regime.
    ///
    /// This test ensures that obvious timing leaks (effect ≫ θ) are detected
    /// even when measurement noise is high (SE ≫ θ). The prior shape change
    /// from trace-normalized S to correlation matrix R was motivated by this
    /// failure mode.
    ///
    /// Test values based on SILENT web app dataset:
    /// - θ_eff = 100 ns
    /// - True effect ≈ 100× threshold (10,000-18,000 ns)
    /// - Data SEs ≈ 25-80× threshold (2,500-8,000 ns)
    ///
    /// With the v5.1 correlation-shaped prior, P(leak) should be > 0.99.
    #[test]
    fn test_large_effect_noisy_likelihood_regression() {
        use crate::adaptive::{calibrate_prior_scale, compute_prior_cov_9d};

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

        // Compute prior using v5.1 correlation-shaped method
        let sigma_prior = calibrate_prior_scale(&sigma_rate, theta_eff, n, discrete_mode, seed);
        let prior_cov = compute_prior_cov_9d(&sigma_rate, sigma_prior, discrete_mode);

        // Debug output
        eprintln!("sigma_prior = {}", sigma_prior);
        eprintln!("Prior diagonal: [{:.2e}, {:.2e}, ..., {:.2e}]",
            prior_cov[(0, 0)], prior_cov[(1, 1)], prior_cov[(8, 8)]);
        eprintln!("Sigma_n diagonal: [{:.2e}, {:.2e}, ..., {:.2e}]",
            sigma_n[(0, 0)], sigma_n[(1, 1)], sigma_n[(8, 8)]);
        eprintln!("Precision ratio Σ_n/Λ₀: [{:.4}, {:.4}, ..., {:.4}]",
            sigma_n[(0, 0)] / prior_cov[(0, 0)],
            sigma_n[(1, 1)] / prior_cov[(1, 1)],
            sigma_n[(8, 8)] / prior_cov[(8, 8)]);

        // Compute Bayes factor
        let result = compute_bayes_factor(&delta, &sigma_n, &prior_cov, theta_eff, Some(seed));

        eprintln!("P(leak) = {}", result.leak_probability);
        eprintln!("delta_post[0..3] = [{:.1}, {:.1}, {:.1}]",
            result.delta_post[0], result.delta_post[1], result.delta_post[2]);

        // With effect ≈ 100× threshold, P(leak) MUST be > 0.99
        assert!(
            result.leak_probability > 0.99,
            "Large effect ({:.0}ns mean) with θ={:.0}ns should give P(leak) > 0.99, got {:.4}. \
             This regression indicates the prior shape fix (v5.1) may have broken.",
            delta.iter().sum::<f64>() / 9.0,
            theta_eff,
            result.leak_probability
        );
    }
}
