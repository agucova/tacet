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

use crate::constants::{B_TAIL, LOG_2PI, ONES};
use crate::types::{Matrix2, Matrix9, Matrix9x2, Vector2, Vector9};

/// Number of Monte Carlo samples for leak probability estimation.
const N_MONTE_CARLO: usize = 1000;

/// Result from Bayesian analysis.
#[derive(Debug, Clone)]
pub struct BayesResult {
    /// Posterior probability of a significant leak: P(max_k |(Xβ)_k| > θ | Δ)
    pub posterior_probability: f64,
    /// Posterior mean of β = (μ, τ)
    pub beta_post: Vector2,
    /// Posterior covariance of β
    pub lambda_post: Matrix2,
    /// Whether the computation encountered numerical issues
    pub is_clamped: bool,
    /// Null covariance used for inference.
    pub sigma0: Matrix9,
    /// Log Bayes factor (for backwards compatibility, approximate)
    pub log_bayes_factor: f64,
    /// Leak covariance used for inference (for backwards compatibility)
    pub sigma1: Matrix9,
}

/// Compute the posterior probability of a timing leak via Monte Carlo integration.
///
/// This implements spec §2.5:
/// 1. Compute posterior distribution β | Δ ~ N(β_post, Λ_post)
/// 2. Draw N = 1000 samples from the posterior
/// 3. For each sample, compute max_k |(Xβ)_k| and check if > θ
/// 4. Return the proportion as leak_probability
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

    // Try to compute Σ₀⁻¹ via Cholesky
    let sigma0_chol = match Cholesky::new(*sigma0) {
        Some(c) => c,
        None => {
            let regularized = add_jitter(*sigma0);
            match Cholesky::new(regularized) {
                Some(c) => c,
                None => {
                    // Fallback: return neutral result
                    return neutral_result(sigma0, &lambda0, &design);
                }
            }
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

    // Monte Carlo integration for leak probability
    // Draw N samples from N(β_post, Λ_post) and count max_k |(Xβ)_k| > θ
    let posterior_probability = monte_carlo_leak_probability(
        &design,
        &beta_post,
        &lambda_post,
        theta,
        seed.unwrap_or(42),
    );

    // Compute approximate log Bayes factor for backwards compatibility
    // (This is not the same as the spec's approach but provides a similar signal)
    let sigma1 = sigma0 + design * lambda0 * design.transpose();
    let log_bf = match (
        mvn_log_pdf_zero(observed_diff, &sigma1),
        mvn_log_pdf_zero(observed_diff, sigma0),
    ) {
        (Some(log_pdf1), Some(log_pdf0)) => log_pdf1 - log_pdf0,
        _ => 0.0,
    };

    BayesResult {
        posterior_probability,
        beta_post,
        lambda_post,
        is_clamped: false,
        sigma0: *sigma0,
        log_bayes_factor: log_bf,
        sigma1,
    }
}

/// Monte Carlo integration for leak probability (spec §2.5).
///
/// Draw N = 1000 samples β ~ N(β_post, Λ_post)
/// For each sample: pred = X @ β, max_effect = max_k |pred[k]|
/// leak_probability = count(max_effect > θ) / N
fn monte_carlo_leak_probability(
    design: &Matrix9x2,
    beta_post: &Vector2,
    lambda_post: &Matrix2,
    theta: f64,
    seed: u64,
) -> f64 {
    // Cholesky decomposition of posterior covariance for sampling
    let chol = match Cholesky::new(*lambda_post) {
        Some(c) => c,
        None => {
            // If Cholesky fails, return 0.5 (uncertain)
            return 0.5;
        }
    };
    let l = chol.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut count = 0usize;

    for _ in 0..N_MONTE_CARLO {
        // Sample z ~ N(0, I_2)
        let z = Vector2::new(
            sample_standard_normal(&mut rng),
            sample_standard_normal(&mut rng),
        );

        // Transform to β ~ N(β_post, Λ_post)
        let beta = beta_post + l * z;

        // Compute predicted differences: pred = X @ β
        let pred = design * beta;

        // Compute max effect: max_k |pred[k]|
        let max_effect = pred.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

        // Count exceedances
        if max_effect > theta {
            count += 1;
        }
    }

    count as f64 / N_MONTE_CARLO as f64
}

/// Sample from standard normal using Box-Muller transform.
fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    // Box-Muller transform
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();

    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Return a neutral result when computation fails.
fn neutral_result(sigma0: &Matrix9, lambda0: &Matrix2, design: &Matrix9x2) -> BayesResult {
    let sigma1 = sigma0 + design * (*lambda0) * design.transpose();
    BayesResult {
        posterior_probability: 0.5,
        beta_post: Vector2::zeros(),
        lambda_post: *lambda0,
        is_clamped: true,
        sigma0: *sigma0,
        log_bayes_factor: 0.0,
        sigma1,
    }
}

/// Convert log Bayes factor to posterior probability (kept for backwards compatibility).
///
/// P(H1|data) = BF * prior_odds / (1 + BF * prior_odds)
///
/// Returns (probability, is_clamped) where is_clamped indicates if the result
/// hit numerical stability limits.
pub fn compute_posterior_probability(log_bf: f64, prior_no_leak: f64) -> (f64, bool) {
    let prior_no_leak = prior_no_leak.clamp(1e-12, 1.0 - 1e-12);
    let prior_odds = (1.0 - prior_no_leak) / prior_no_leak;

    let log_posterior_odds = log_bf + prior_odds.ln();

    if log_posterior_odds > 700.0 {
        (0.9999, true)
    } else if log_posterior_odds < -700.0 {
        (0.0001, true)
    } else {
        (1.0 / (1.0 + (-log_posterior_odds).exp()), false)
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
/// Per spec: caller should return log BF = 0 (neutral evidence) on failure.
fn mvn_log_pdf_zero(x: &Vector9, sigma: &Matrix9) -> Option<f64> {
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
/// Formula: ε = 10⁻¹⁰ + (tr(Σ)/9) × 10⁻⁸
///
/// The base jitter (10⁻¹⁰) handles near-zero variance cases.
/// The trace-scaled term adapts to the matrix's magnitude.
fn add_jitter(mut sigma: Matrix9) -> Matrix9 {
    let trace: f64 = (0..9).map(|i| sigma[(i, i)]).sum();
    let base_jitter = 1e-10;
    let adaptive_jitter = (trace / 9.0) * 1e-8;
    let jitter = base_jitter + adaptive_jitter;
    for i in 0..9 {
        sigma[(i, i)] += jitter;
    }
    sigma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posterior_probability_bounds() {
        let (prob_high, clamped_high) = compute_posterior_probability(100.0, 0.5);
        assert!(prob_high > 0.999);
        assert!(!clamped_high); // 100.0 is below the 700.0 threshold

        let (prob_low, clamped_low) = compute_posterior_probability(-100.0, 0.5);
        assert!(prob_low < 0.001);
        assert!(!clamped_low);

        let (prob_equal, clamped_equal) = compute_posterior_probability(0.0, 0.5);
        assert!((prob_equal - 0.5).abs() < 0.001);
        assert!(!clamped_equal);
    }

    #[test]
    fn test_posterior_probability_clamping() {
        // Test clamping at upper threshold
        let (prob_clamped_high, clamped_high) = compute_posterior_probability(800.0, 0.5);
        assert_eq!(prob_clamped_high, 0.9999);
        assert!(clamped_high);

        // Test clamping at lower threshold
        let (prob_clamped_low, clamped_low) = compute_posterior_probability(-800.0, 0.5);
        assert_eq!(prob_clamped_low, 0.0001);
        assert!(clamped_low);
    }

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

        // Result should be valid
        assert!(result.log_bayes_factor.is_finite());
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
}
