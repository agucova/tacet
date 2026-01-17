//! Layer 2: Bayesian inference for timing leak detection.
//!
//! See spec §2.5 (Bayesian Inference) for the full methodology.
//!
//! This module computes the posterior probability of a timing leak using:
//! - Conjugate Gaussian model with closed-form posterior (no MCMC required)
//! - Monte Carlo integration for leak probability
//!
//! ## Model (spec §2.5)
//!
//! Δ = Xβ + ε,  ε ~ N(0, Σ_n)
//!
//! where:
//! - Δ is the observed quantile differences (9-vector)
//! - X is the design matrix [1 | b_tail] (9×2) for effect decomposition
//! - β = (μ, τ)ᵀ is the effect vector (uniform shift and tail components)
//! - Σ_n is the covariance matrix (already scaled for inference sample size)
//!
//! ## Prior (spec §2.5)
//!
//! β ~ N(0, Λ₀),  Λ₀ = diag(σ_μ², σ_τ²)
//!
//! The prior scale is set proportional to θ: σ = 2θ. This ensures:
//! - P(|β| > θ | prior) ≈ 62%, representing reasonable uncertainty
//! - Avoids the miscalibration of max(2×MDE, θ) which gave ~99% when MDE >> θ
//!
//! ## Posterior (spec §2.5)
//!
//! β | Δ ~ N(β_post, Λ_post)
//!
//! where:
//! - Λ_post = (Xᵀ Σ_n⁻¹ X + Λ₀⁻¹)⁻¹
//! - β_post = Λ_post Xᵀ Σ_n⁻¹ Δ
//!
//! ## Leak Probability (spec §2.5)
//!
//! P(significant leak | Δ) = P(max_k |(Xβ)_k| > θ | Δ)
//!
//! Computed via Monte Carlo: draw 1000 samples from posterior, count exceedances.
//!
//! ## Covariance Scaling
//!
//! The caller is responsible for providing appropriately scaled covariance.
//! In the adaptive architecture, this is Σ_rate / n where:
//! - Σ_rate is the asymptotic rate matrix from covariance estimation
//! - n is the current inference sample size

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

/// Result from Bayesian analysis (spec §2.5).
///
/// Contains the posterior distribution parameters and derived quantities
/// needed for the adaptive sampling loop and effect estimation.
#[derive(Debug, Clone)]
pub struct BayesResult {
    /// Posterior probability of a significant leak: P(max_k |(Xβ)_k| > θ | Δ).
    ///
    /// This is the leak probability from the Bayesian layer (spec §2.5).
    /// Computed via Monte Carlo integration over the posterior distribution.
    /// Range: 0.0 to 1.0
    pub leak_probability: f64,

    /// 95% credible interval for total effect magnitude: (2.5th, 97.5th) percentiles of ||β||₂.
    ///
    /// Note: This CI may not contain the point estimate when β ≈ 0 because ||·|| is non-negative.
    pub effect_magnitude_ci: (f64, f64),

    /// Posterior mean of β = (μ, τ) in nanoseconds.
    ///
    /// - β[0] = μ (uniform shift): All quantiles move equally
    /// - β[1] = τ (tail effect): Upper quantiles shift more than lower
    pub beta_mean: Vector2,

    /// Posterior covariance of β.
    ///
    /// Λ_post = (Xᵀ Σ_n⁻¹ X + Λ₀⁻¹)⁻¹
    /// Used for effect significance testing and credible intervals.
    pub beta_cov: Matrix2,

    /// Whether the computation encountered numerical issues (e.g., Cholesky failure).
    ///
    /// If true, the posterior is set to the prior (maximally uncertain).
    pub is_clamped: bool,

    /// Covariance matrix used for inference (after regularization).
    ///
    /// This is the input Σ_n after applying variance floor regularization
    /// for numerical stability.
    pub sigma_n: Matrix9,

    /// Model fit Q statistic (spec §2.3.3, §2.6 Gate 8).
    ///
    /// Q = r' Σ_n^{-1} r where r = Δ - Xβ_post is the residual.
    /// Under the 2D model, Q ~ chi-squared(7).
    /// High Q indicates the 2D (shift + tail) model doesn't fit the data.
    /// Compare against q_thresh from calibration for model mismatch detection.
    pub model_fit_q: f64,
}

/// Compute Bayesian posterior for timing leak analysis (spec §2.5).
///
/// Uses conjugate Gaussian model: Δ = Xβ + ε, ε ~ N(0, Σ_n)
/// - Prior: β ~ N(0, Λ₀)
/// - Posterior: β|Δ ~ N(β_post, Λ_post)
///   - Λ_post = (Xᵀ Σ_n⁻¹ X + Λ₀⁻¹)⁻¹
///   - β_post = Λ_post Xᵀ Σ_n⁻¹ Δ
/// - Leak probability: P(max_k |Xβ|_k > θ | Δ) via Monte Carlo
///
/// # Arguments
///
/// * `delta` - Observed quantile differences (9-vector, baseline - sample)
/// * `sigma_n` - Covariance matrix already scaled for inference sample size.
///   In the adaptive architecture, this is Σ_rate / n.
/// * `prior_sigmas` - Prior standard deviations (σ_μ, σ_τ) in nanoseconds.
///   Typically set to 2θ to ensure calibrated priors.
/// * `theta` - Minimum effect of concern (threshold for practical significance)
/// * `seed` - Random seed for Monte Carlo reproducibility
///
/// # Returns
///
/// `BayesResult` containing:
/// - `leak_probability`: P(max_k |(Xβ)_k| > θ | Δ)
/// - `beta_mean`, `beta_cov`: Posterior distribution parameters
/// - `effect_magnitude_ci`: 95% CI for ||β||₂
pub fn compute_bayes_factor(
    delta: &Vector9,
    sigma_n: &Matrix9,
    prior_sigmas: (f64, f64),
    theta: f64,
    seed: Option<u64>,
) -> BayesResult {
    compute_posterior_monte_carlo(delta, sigma_n, prior_sigmas, theta, seed)
}

/// Compute posterior probability via Monte Carlo integration (spec §2.5).
///
/// Internal implementation that handles all the linear algebra for
/// conjugate Gaussian inference.
pub fn compute_posterior_monte_carlo(
    delta: &Vector9,
    sigma_n: &Matrix9,
    prior_sigmas: (f64, f64),
    theta: f64,
    seed: Option<u64>,
) -> BayesResult {
    let design = build_design_matrix();
    let lambda0 = prior_covariance(prior_sigmas);

    // Compute posterior parameters (spec §2.5)
    // Λ_post = (Xᵀ Σ_n⁻¹ X + Λ₀⁻¹)⁻¹
    // β_post = Λ_post Xᵀ Σ_n⁻¹ Δ

    // Always regularize covariance for numerical stability (spec §2.6)
    // This ensures CI and probability use the same posterior distribution
    let regularized = add_jitter(*sigma_n);
    let sigma_n_chol = match Cholesky::new(regularized) {
        Some(c) => c,
        None => {
            // Fallback: return neutral result (maximally uncertain)
            return neutral_result(sigma_n, &lambda0, &design);
        }
    };

    // Σ_n⁻¹ via Cholesky: L L^T = Σ_n, so Σ_n⁻¹ = L^{-T} L^{-1}
    let sigma_n_inv = sigma_n_chol.inverse();

    // Xᵀ Σ_n⁻¹ X (2×2 precision contribution from data)
    let xt_sigma_n_inv_x = design.transpose() * sigma_n_inv * design;

    // Λ₀⁻¹ (prior precision)
    let lambda0_inv = Matrix2::from_diagonal(&Vector2::new(
        1.0 / lambda0[(0, 0)].max(1e-12),
        1.0 / lambda0[(1, 1)].max(1e-12),
    ));

    // Posterior precision: Λ_post⁻¹ = Xᵀ Σ_n⁻¹ X + Λ₀⁻¹
    let precision_post = xt_sigma_n_inv_x + lambda0_inv;
    let lambda_post_chol = match Cholesky::new(precision_post) {
        Some(c) => c,
        None => {
            return neutral_result(sigma_n, &lambda0, &design);
        }
    };
    // Posterior covariance: Λ_post = (Xᵀ Σ_n⁻¹ X + Λ₀⁻¹)⁻¹
    let beta_cov = lambda_post_chol.inverse();

    // Posterior mean: β_post = Λ_post Xᵀ Σ_n⁻¹ Δ
    let beta_mean = beta_cov * design.transpose() * sigma_n_inv * delta;

    // Compute model fit Q statistic: Q = r' Σ_n^{-1} r
    // where r = Δ - Xβ_post is the residual
    let predicted = &design * &beta_mean;
    let residual = delta - predicted;
    let model_fit_q = residual.dot(&(sigma_n_inv * &residual));

    // Monte Carlo integration for leak probability and effect CI
    let (leak_probability, effect_magnitude_ci) =
        run_monte_carlo(&design, &beta_mean, &beta_cov, theta, seed.unwrap_or(crate::constants::DEFAULT_SEED));

    BayesResult {
        leak_probability,
        effect_magnitude_ci,
        beta_mean,
        beta_cov,
        is_clamped: false,
        sigma_n: regularized,
        model_fit_q,
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
        let magnitude = math::sqrt(math::sq(beta[0]) + math::sq(beta[1]));
        magnitudes.push(magnitude);
    }

    magnitudes.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = math::round((N_MONTE_CARLO as f64) * 0.025) as usize;
    let hi_idx = math::round((N_MONTE_CARLO as f64) * 0.975) as usize;
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

    math::sqrt(-2.0 * math::ln(u1)) * math::cos(2.0 * PI * u2)
}

/// Result from max effect CI computation for Research mode.
#[derive(Debug, Clone)]
pub struct MaxEffectCI {
    /// Posterior mean of max_k |(Xβ)_k|.
    pub mean: f64,
    /// 95% credible interval for max_k |(Xβ)_k|: (2.5th, 97.5th percentile).
    pub ci: (f64, f64),
}

/// Compute 95% CI for max effect: max_k |(Xβ)_k| (spec v4.1 research mode).
///
/// This is used by Research mode to determine stopping conditions:
/// - `CI.lower > 1.1 * theta_floor` → EffectDetected
/// - `CI.upper < 0.9 * theta_floor` → NoEffectDetected
///
/// # Arguments
///
/// * `beta_mean` - Posterior mean of β = (μ, τ)
/// * `beta_cov` - Posterior covariance of β
/// * `seed` - Random seed for Monte Carlo reproducibility
///
/// # Returns
///
/// `MaxEffectCI` with mean and 95% credible interval.
pub fn compute_max_effect_ci(
    beta_mean: &Vector2,
    beta_cov: &Matrix2,
    seed: u64,
) -> MaxEffectCI {
    let design = build_design_matrix();

    // Cholesky decomposition of posterior covariance for sampling
    let chol = match Cholesky::new(*beta_cov) {
        Some(c) => c,
        None => {
            // If Cholesky fails, return degenerate result
            return MaxEffectCI {
                mean: 0.0,
                ci: (0.0, 0.0),
            };
        }
    };
    let l = chol.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut max_effects = Vec::with_capacity(N_MONTE_CARLO);
    let mut sum = 0.0;

    for _ in 0..N_MONTE_CARLO {
        // Sample z ~ N(0, I_2)
        let z = Vector2::new(
            sample_standard_normal(&mut rng),
            sample_standard_normal(&mut rng),
        );

        // Transform to β ~ N(β_post, Λ_post)
        let beta = beta_mean + l * z;

        // Compute max_k |(Xβ)_k|
        let pred = &design * beta;
        let max_effect = pred.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
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

/// Return a neutral result when computation fails (maximally uncertain).
///
/// Sets leak_probability = 0.5 (no information) and posterior = prior.
/// Marks is_clamped = true to signal the numerical issue to callers.
fn neutral_result(sigma_n: &Matrix9, lambda0: &Matrix2, _design: &Matrix9x2) -> BayesResult {
    BayesResult {
        leak_probability: 0.5,
        effect_magnitude_ci: (0.0, 0.0),
        beta_mean: Vector2::zeros(),
        beta_cov: *lambda0,
        is_clamped: true,
        sigma_n: *sigma_n,
        // Model fit undefined when posterior computation fails
        model_fit_q: f64::NAN,
    }
}

/// Build design matrix for effect decomposition (spec §2.5).
///
/// X = [1 | b_tail] where:
/// - Column 0 (ones): Uniform shift - all quantiles move equally
/// - Column 1 (b_tail): Tail effect - upper quantiles shift more than lower
///   b_tail = [-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5]
///
/// The tail basis is centered (sums to zero) so μ and τ are orthogonal.
pub fn build_design_matrix() -> Matrix9x2 {
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
    lambda0[(0, 0)] = math::sq(sigma_mu.max(1e-12));
    lambda0[(1, 1)] = math::sq(sigma_tau.max(1e-12));
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
    let log_det = 2.0 * chol.l().diagonal().iter().map(|d| math::ln(*d)).sum::<f64>();

    Some(-0.5 * (9.0 * LOG_2PI + log_det + mahal_sq))
}

/// Apply variance floor regularization for numerical stability (spec §2.6).
///
/// When some quantiles have zero or near-zero variance (common in discrete mode
/// with ties), the covariance matrix becomes ill-conditioned. Even if Cholesky
/// succeeds, the inverse has huge values for near-zero variance elements,
/// causing them to dominate the Bayesian regression incorrectly.
///
/// We regularize by ensuring a minimum diagonal value of 1% of mean variance.
/// This bounds the condition number to ~100, preventing numerical instability.
///
/// Formula (spec §2.6):
///   σ²ᵢ ← max(σ²ᵢ, 0.01 × σ̄²) + ε
/// where σ̄² = tr(Σ)/9 and ε = 10⁻¹⁰ + σ̄² × 10⁻⁸
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

        let delta = Vector9::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Create a pathological covariance with huge condition number
        let mut pathological = Matrix9::identity();
        for i in 0..9 {
            pathological[(i, i)] = if i == 0 { 1e20 } else { 1e-20 };
        }

        let result = compute_bayes_factor(&delta, &pathological, (10.0, 10.0), 10.0, Some(42));

        // Result should be valid (probability between 0 and 1)
        assert!(result.leak_probability >= 0.0 && result.leak_probability <= 1.0);
    }

    #[test]
    fn test_monte_carlo_no_effect() {
        // With zero observed difference, leak probability should be low
        let delta = Vector9::zeros();
        let sigma_n = Matrix9::identity();

        let result = compute_posterior_monte_carlo(
            &delta,
            &sigma_n,
            (10.0, 10.0),
            10.0, // threshold
            Some(42),
        );

        // With no effect and identity covariance, most samples should be below θ
        assert!(
            result.leak_probability < 0.5,
            "With zero effect, leak probability should be low, got {}",
            result.leak_probability
        );
    }

    #[test]
    fn test_monte_carlo_large_effect() {
        // With large observed difference, leak probability should be high
        let delta = Vector9::from_row_slice(&[100.0; 9]);
        let sigma_n = Matrix9::identity();

        let result = compute_posterior_monte_carlo(
            &delta,
            &sigma_n,
            (10.0, 10.0),
            10.0, // threshold
            Some(42),
        );

        // With 100ns effect and 10ns threshold, almost all samples should exceed
        assert!(
            result.leak_probability > 0.9,
            "With large effect, leak probability should be high, got {}",
            result.leak_probability
        );
    }

    #[test]
    fn test_monte_carlo_determinism() {
        // Same seed should give same result
        let delta = Vector9::from_row_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
        let sigma_n = Matrix9::identity();

        let result1 = compute_posterior_monte_carlo(&delta, &sigma_n, (10.0, 10.0), 10.0, Some(42));

        let result2 = compute_posterior_monte_carlo(&delta, &sigma_n, (10.0, 10.0), 10.0, Some(42));

        assert_eq!(
            result1.leak_probability, result2.leak_probability,
            "Same seed should give same result"
        );
    }

    #[test]
    fn test_zero_delta_gives_low_probability() {
        // With zero observed difference, the Monte Carlo probability should be low
        let delta = Vector9::zeros();
        let sigma_n = Matrix9::identity();
        let result = compute_bayes_factor(&delta, &sigma_n, (10.0, 10.0), 10.0, Some(42));

        // With zero effect, we shouldn't be confident about a leak
        assert!(
            result.leak_probability < 0.5,
            "Zero-delta should give low leak probability, got {}",
            result.leak_probability
        );
    }

    #[test]
    fn test_exceedance_monotonic_in_theta() {
        let delta = Vector9::zeros();
        let sigma_n = Matrix9::identity();

        let result_small = compute_bayes_factor(&delta, &sigma_n, (10.0, 10.0), 1.0, Some(42));
        let result_large = compute_bayes_factor(&delta, &sigma_n, (10.0, 10.0), 1.0e6, Some(42));

        assert!(
            result_small.leak_probability > result_large.leak_probability,
            "Exceedance probability should decrease as theta grows"
        );
    }

    /// Compute prior exceedance probability P(max_k |(Xβ)_k| > θ | prior) via Monte Carlo.
    ///
    /// This is used to calibrate the prior scale σ so that the prior exceedance
    /// probability matches the spec's intended ~62%.
    fn compute_prior_exceedance_probability(sigma_prior: f64, theta: f64, n_samples: usize) -> f64 {
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let design = build_design_matrix();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12345);
        let mut count = 0usize;

        for _ in 0..n_samples {
            // Sample β ~ N(0, σ²I₂)
            let beta = Vector2::new(
                sample_standard_normal(&mut rng) * sigma_prior,
                sample_standard_normal(&mut rng) * sigma_prior,
            );

            // Compute pred = Xβ
            let pred = design * beta;

            // Check if max_k |pred_k| > θ
            let max_effect = pred.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
            if max_effect > theta {
                count += 1;
            }
        }

        count as f64 / n_samples as f64
    }

    #[test]
    fn test_prior_exceedance_with_current_scale() {
        // Current spec: σ = 2θ
        // The spec claims this gives P(|β| > θ) ≈ 62%, but the actual leak test
        // computes P(max_k |(Xβ)_k| > θ), which is higher due to the design matrix.
        let theta = 100.0;
        let sigma_current = 2.0 * theta; // Current: σ = 2θ

        let exceedance = compute_prior_exceedance_probability(sigma_current, theta, 100_000);

        // This should show the actual prior exceedance is much higher than 62%
        eprintln!(
            "Prior exceedance with σ = 2θ: {:.1}% (spec claims ~62%)",
            exceedance * 100.0
        );

        // The actual value should be around 85%, demonstrating the bug
        assert!(
            exceedance > 0.80,
            "Expected high exceedance with σ=2θ, got {:.1}%",
            exceedance * 100.0
        );
    }

    #[test]
    fn test_find_correct_prior_scale() {
        // Find the σ/θ ratio that gives ~62% prior exceedance probability
        let theta = 100.0;
        let target = 0.62;
        let tolerance = 0.02;

        // Binary search for the correct ratio
        let mut lo = 0.5;
        let mut hi = 2.5;

        for _ in 0..20 {
            let mid = (lo + hi) / 2.0;
            let sigma = mid * theta;
            let exceedance = compute_prior_exceedance_probability(sigma, theta, 50_000);

            if exceedance < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let optimal_ratio = (lo + hi) / 2.0;
        let final_exceedance =
            compute_prior_exceedance_probability(optimal_ratio * theta, theta, 100_000);

        eprintln!(
            "Optimal σ/θ ratio for 62% exceedance: {:.3} (actual: {:.1}%)",
            optimal_ratio,
            final_exceedance * 100.0
        );

        // Verify we found a good value
        assert!(
            (final_exceedance - target).abs() < tolerance,
            "Could not find σ giving ~62% exceedance. Best: σ={:.3}θ gives {:.1}%",
            optimal_ratio,
            final_exceedance * 100.0
        );

        // The correct ratio should be around 1.0-1.2 (not 2.0)
        assert!(
            optimal_ratio > 0.8 && optimal_ratio < 1.5,
            "Expected ratio between 0.8 and 1.5, got {:.3}",
            optimal_ratio
        );
    }
}
