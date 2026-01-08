//! Layer 2: Bayesian inference for timing leak detection (spec §2.5).
//!
//! This module computes the posterior probability of a timing leak using:
//! - Sample splitting (calibration/inference handled by the caller)
//! - Conjugate Gaussian model with closed-form posterior
//! - Monte Carlo integration for leak probability
//!
//! ## Model
//!
//! Δ = Xβ + ε,  ε ~ N(0, Σ₀)
//!
//! where:
//! - Δ is the observed quantile differences (9-vector)
//! - X is the design matrix [1 | b_tail] (9×2)
//! - β = (μ, τ)ᵀ is the effect vector (shift and tail components)
//! - Σ₀ is the null covariance from bootstrap estimation
//!
//! ## Prior
//!
//! β ~ N(0, Λ₀),  Λ₀ = diag(σ_μ², σ_τ²)
//!
//! ## Posterior
//!
//! β | Δ ~ N(β_post, Λ_post)
//!
//! where:
//! - Λ_post = (XᵀΣ₀⁻¹X + Λ₀⁻¹)⁻¹
//! - β_post = Λ_post Xᵀ Σ₀⁻¹ Δ
//!
//! ## Leak Probability
//!
//! P(significant leak | Δ) = P(max_k |(Xβ)_k| > θ | Δ)
//!
//! Computed via Monte Carlo: draw samples from posterior, count exceedances.

use nalgebra::Cholesky;
use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::constants::{B_TAIL, ONES};
use crate::types::{Matrix2, Matrix9, Matrix9x2, Vector2, Vector9};

/// Number of Monte Carlo samples for leak probability estimation.
const N_MONTE_CARLO: usize = 1000;

/// Result from Bayesian analysis.
#[derive(Debug, Clone)]
pub struct BayesResult {
    /// Posterior probability of a significant leak: P(max_k |(Xβ)_k| > θ | Δ)
    /// This is THE leak probability from the Bayesian layer (spec §2.5).
    pub posterior_probability: f64,
    /// 95% credible interval for total effect magnitude: (2.5th, 97.5th) percentiles of ||β||₂
    pub effect_magnitude_ci: (f64, f64),
    /// Posterior mean of β = (μ, τ)
    pub beta_post: Vector2,
    /// Posterior covariance of β
    pub lambda_post: Matrix2,
    /// Whether the computation encountered numerical issues
    pub is_clamped: bool,
    /// Null covariance used for inference.
    pub sigma0: Matrix9,
}

/// Compute the posterior probability of a timing leak via Monte Carlo integration.
///
/// This implements spec §2.5:
/// 1. Compute posterior distribution β | Δ ~ N(β_post, Λ_post)
/// 2. Draw N = 1000 samples from the posterior
/// 3. For each sample, compute max_k |(Xβ)_k| and check if > θ
/// 4. For each sample, compute ||β||₂
/// 5. Return the exceedance probability (under H1) and the CI of magnitudes
///
/// # Arguments
///
/// * `observed_diff` - Observed quantile differences (baseline - sample)
/// * `sigma0` - Covariance of delta under H0 (from calibration)
/// * `prior_sigmas` - Prior standard deviations (shift, tail) in ns
/// * `theta` - Minimum effect of concern (threshold for practical significance)
/// * `seed` - Random seed for reproducibility
pub fn compute_bayes_factor(
    observed_diff: &Vector9,
    sigma0: &Matrix9,
    prior_sigmas: (f64, f64),
    theta: f64,
    seed: Option<u64>,
) -> BayesResult {
    compute_posterior_monte_carlo(observed_diff, sigma0, prior_sigmas, theta, seed)
}

/// Compute posterior probability via Monte Carlo integration (spec §2.5).
pub fn compute_posterior_monte_carlo(
    observed_diff: &Vector9,
    sigma0: &Matrix9,
    prior_sigmas: (f64, f64),
    theta: f64,
    seed: Option<u64>,
) -> BayesResult {
    let design = build_design_matrix();
    let lambda0 = prior_covariance(prior_sigmas);

    // Compute posterior parameters (spec §2.5)
    // Λ_post = (XᵀΣ₀⁻¹X + Λ₀⁻¹)⁻¹
    // β_post = Λ_post Xᵀ Σ₀⁻¹ Δ

    // Always regularize covariance for consistency with effect decomposition (spec §2.6)
    // This ensures CI and probability use the same posterior distribution
    let regularized = add_jitter(*sigma0);
    let sigma0_chol = match Cholesky::new(regularized) {
        Some(c) => c,
        None => {
            // Fallback: return neutral result
            return neutral_result(sigma0, &lambda0, &design);
        }
    };

    // Σ₀⁻¹ = L⁻ᵀ L⁻¹
    let sigma0_inv = sigma0_chol.inverse();

    // XᵀΣ₀⁻¹X (2×2 matrix)
    let xt_sigma0_inv_x = design.transpose() * sigma0_inv * design;

    // Λ₀⁻¹
    let lambda0_inv = Matrix2::from_diagonal(&Vector2::new(
        1.0 / lambda0[(0, 0)].max(1e-12),
        1.0 / lambda0[(1, 1)].max(1e-12),
    ));

    // Λ_post = (XᵀΣ₀⁻¹X + Λ₀⁻¹)⁻¹
    let precision_post = xt_sigma0_inv_x + lambda0_inv;
    let lambda_post_chol = match Cholesky::new(precision_post) {
        Some(c) => c,
        None => {
            return neutral_result(sigma0, &lambda0, &design);
        }
    };
    let lambda_post = lambda_post_chol.inverse();

    // β_post = Λ_post Xᵀ Σ₀⁻¹ Δ
    let beta_post = lambda_post * design.transpose() * sigma0_inv * observed_diff;

    // Monte Carlo integration for leak probability and effect CI
    let (posterior_probability, effect_magnitude_ci) = run_monte_carlo(
        &design,
        &beta_post,
        &lambda_post,
        theta,
        seed.unwrap_or(42),
    );

    BayesResult {
        posterior_probability,
        effect_magnitude_ci,
        beta_post,
        lambda_post,
        is_clamped: false,
        sigma0: *sigma0,
    }
}

/// Monte Carlo integration for leak probability and effect CI (spec §2.5).
fn run_monte_carlo(
    design: &Matrix9x2,
    beta_post: &Vector2,
    lambda_post: &Matrix2,
    theta: f64,
    seed: u64,
) -> (f64, (f64, f64)) {
    // Cholesky decomposition of posterior covariance for sampling
    let chol = match Cholesky::new(*lambda_post) {
        Some(c) => c,
        None => {
            // If Cholesky fails, return 0.5 (uncertain)
            return (0.5, (0.0, 0.0));
        }
    };
    let l = chol.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut count = 0usize;
    let mut magnitudes = Vec::with_capacity(N_MONTE_CARLO);

    for _ in 0..N_MONTE_CARLO {
        // Sample z ~ N(0, I_2)
        let z = Vector2::new(
            sample_standard_normal(&mut rng),
            sample_standard_normal(&mut rng),
        );

        // Transform to β ~ N(β_post, Λ_post)
        let beta = beta_post + l * z;

        // 1. Exceedance check under H1: pred = X @ β, check max_k |pred[k]| > θ
        let pred = design * beta;
        let max_effect = pred.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if max_effect > theta {
            count += 1;
        }

        // 2. Effect magnitude: ||β||₂
        let magnitude = (beta[0].powi(2) + beta[1].powi(2)).sqrt();
        magnitudes.push(magnitude);
    }

    magnitudes.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = ((N_MONTE_CARLO as f64) * 0.025).round() as usize;
    let hi_idx = ((N_MONTE_CARLO as f64) * 0.975).round() as usize;
    let ci = (
        magnitudes[lo_idx.min(N_MONTE_CARLO - 1)],
        magnitudes[hi_idx.min(N_MONTE_CARLO - 1)],
    );

    (count as f64 / N_MONTE_CARLO as f64, ci)
}

/// Sample from standard normal using Box-Muller transform.
fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    // Box-Muller transform
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();

    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Return a neutral result when computation fails.
fn neutral_result(sigma0: &Matrix9, lambda0: &Matrix2, _design: &Matrix9x2) -> BayesResult {
    BayesResult {
        posterior_probability: 0.5,
        effect_magnitude_ci: (0.0, 0.0),
        beta_post: Vector2::zeros(),
        lambda_post: *lambda0,
        is_clamped: true,
        sigma0: *sigma0,
    }
}


pub(crate) fn build_design_matrix() -> Matrix9x2 {
    let mut x = Matrix9x2::zeros();
    for i in 0..9 {
        x[(i, 0)] = ONES[i];
        x[(i, 1)] = B_TAIL[i];
    }
    x
}

fn prior_covariance(prior_sigmas: (f64, f64)) -> Matrix2 {
    let (sigma_mu, sigma_tau) = prior_sigmas;
    let mut lambda0 = Matrix2::zeros();
    lambda0[(0, 0)] = sigma_mu.max(1e-12).powi(2);
    lambda0[(1, 1)] = sigma_tau.max(1e-12).powi(2);
    lambda0
}

/// Compute log pdf of MVN(0, sigma) at point x.
///
/// Returns None if Cholesky decomposition fails even after jitter.
#[cfg(test)]
fn mvn_log_pdf_zero(x: &Vector9, sigma: &Matrix9) -> Option<f64> {
    use crate::constants::LOG_2PI;

    let chol = match Cholesky::new(*sigma) {
        Some(c) => c,
        None => {
            let regularized = add_jitter(*sigma);
            Cholesky::new(regularized)?
        }
    };

    // Solve L * z = x
    let z = chol.l().solve_lower_triangular(x).unwrap_or(*x);
    let mahal_sq = z.dot(&z);
    let log_det = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();

    Some(-0.5 * (9.0 * LOG_2PI + log_det + mahal_sq))
}

/// Add diagonal jitter for numerical stability (spec §2.6).
///
/// When some quantiles have zero variance (common in discrete mode with ties),
/// the covariance matrix becomes ill-conditioned. Even if Cholesky succeeds,
/// the inverse has huge values for near-zero variance elements, causing them
/// to dominate the Bayesian regression incorrectly.
///
/// We regularize by ensuring a minimum diagonal value of 1% of mean variance.
/// This bounds the condition number to ~100, preventing numerical instability.
///
/// Formula: σ²ᵢ = max(σ²ᵢ, 0.01 × mean_var) + ε
/// where ε = 10⁻¹⁰ + mean_var × 10⁻⁸
fn add_jitter(mut sigma: Matrix9) -> Matrix9 {
    let trace: f64 = (0..9).map(|i| sigma[(i, i)]).sum();
    let mean_var = trace / 9.0;

    // Use 1% of mean variance as floor, with absolute minimum of 1e-10
    // This bounds the max/min diagonal ratio to ~100, keeping condition number reasonable
    let min_var = (0.01 * mean_var).max(1e-10);

    // Also add small jitter proportional to scale for numerical stability
    let jitter = 1e-10 + mean_var * 1e-8;

    for i in 0..9 {
        // Ensure minimum variance, then add jitter
        sigma[(i, i)] = sigma[(i, i)].max(min_var) + jitter;
    }
    sigma
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::LOG_2PI;

    #[test]
    fn test_mvn_log_pdf_at_zero() {
        let mean = Vector9::zeros();
        let cov = Matrix9::identity();
        let log_pdf_at_mean = mvn_log_pdf_zero(&mean, &cov).expect("Cholesky should succeed");
        let expected = -0.5 * 9.0 * LOG_2PI;
        assert!((log_pdf_at_mean - expected).abs() < 0.001);
    }

    #[test]
    fn test_bayes_factor_cholesky_fallback() {
        // Test that when Cholesky would fail on a pathological matrix,
        // the computation falls back gracefully
        use crate::types::Vector9;

        let observed_diff = Vector9::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Create a pathological covariance with huge condition number
        let mut pathological = Matrix9::identity();
        for i in 0..9 {
            pathological[(i, i)] = if i == 0 { 1e20 } else { 1e-20 };
        }

        let result = compute_bayes_factor(&observed_diff, &pathological, (10.0, 10.0), 10.0, Some(42));

        // Result should be valid (probability between 0 and 1)
        assert!(result.posterior_probability >= 0.0 && result.posterior_probability <= 1.0);
    }

    #[test]
    fn test_monte_carlo_no_effect() {
        // With zero observed difference, leak probability should be low
        let observed_diff = Vector9::zeros();
        let sigma0 = Matrix9::identity();

        let result = compute_posterior_monte_carlo(
            &observed_diff,
            &sigma0,
            (10.0, 10.0),
            10.0, // threshold
            Some(42),
        );

        // With no effect and identity covariance, most samples should be below θ
        assert!(result.posterior_probability < 0.5,
            "With zero effect, leak probability should be low, got {}", result.posterior_probability);
    }

    #[test]
    fn test_monte_carlo_large_effect() {
        // With large observed difference, leak probability should be high
        let observed_diff = Vector9::from_row_slice(&[100.0; 9]);
        let sigma0 = Matrix9::identity();

        let result = compute_posterior_monte_carlo(
            &observed_diff,
            &sigma0,
            (10.0, 10.0),
            10.0, // threshold
            Some(42),
        );

        // With 100ns effect and 10ns threshold, almost all samples should exceed
        assert!(result.posterior_probability > 0.9,
            "With large effect, leak probability should be high, got {}", result.posterior_probability);
    }

    #[test]
    fn test_monte_carlo_determinism() {
        // Same seed should give same result
        let observed_diff = Vector9::from_row_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
        let sigma0 = Matrix9::identity();

        let result1 = compute_posterior_monte_carlo(
            &observed_diff,
            &sigma0,
            (10.0, 10.0),
            10.0,
            Some(42),
        );

        let result2 = compute_posterior_monte_carlo(
            &observed_diff,
            &sigma0,
            (10.0, 10.0),
            10.0,
            Some(42),
        );

        assert_eq!(result1.posterior_probability, result2.posterior_probability,
            "Same seed should give same result");
    }

    #[test]
    fn test_zero_delta_gives_low_probability() {
        // With zero observed difference, the Monte Carlo probability should be low
        let observed_diff = Vector9::zeros();
        let sigma0 = Matrix9::identity();
        let result = compute_bayes_factor(&observed_diff, &sigma0, (10.0, 10.0), 10.0, Some(42));

        // With zero effect, we shouldn't be confident about a leak
        assert!(
            result.posterior_probability < 0.5,
            "Zero-delta should give low leak probability, got {}",
            result.posterior_probability
        );
    }

    #[test]
    fn test_exceedance_monotonic_in_theta() {
        let observed_diff = Vector9::zeros();
        let sigma0 = Matrix9::identity();

        let result_small = compute_bayes_factor(&observed_diff, &sigma0, (10.0, 10.0), 1.0, Some(42));
        let result_large = compute_bayes_factor(&observed_diff, &sigma0, (10.0, 10.0), 1.0e6, Some(42));

        assert!(
            result_small.posterior_probability > result_large.posterior_probability,
            "Exceedance probability should decrease as theta grows"
        );
    }
}
