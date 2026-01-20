//! Effect decomposition using Bayesian linear regression (spec §3.4.6).
//!
//! This module decomposes timing differences into interpretable components:
//!
//! - **Uniform shift (μ)**: All quantiles move equally (e.g., branch timing)
//! - **Tail effect (τ)**: Upper quantiles shift more than lower (e.g., cache misses)
//!
//! ## Design Matrix (spec §3.4.6)
//!
//! X = [1 | b_tail] where:
//! - Column 0: ones = [1, 1, ..., 1] - uniform shift affects all quantiles equally
//! - Column 1: b_tail = [-0.5, -0.375, ..., 0.5] - tail effect is antisymmetric
//!
//! The tail basis is centered (sums to zero) so μ and τ are orthogonal.
//!
//! ## Model
//!
//! Δ = Xβ + ε,  ε ~ N(0, Σ_n)
//!
//! With the same conjugate Gaussian model as the Bayesian layer (spec §3.4.6).
//!
//! ## Effect Pattern Classification (spec §3.4.6)
//!
//! An effect component is "significant" if |effect| > 2×SE:
//! - UniformShift: Only μ is significant
//! - TailEffect: Only τ is significant
//! - Mixed: Both are significant
//! - Indeterminate: Neither is significant (classify by relative magnitude)

use nalgebra::Cholesky;

use crate::math;
use crate::result::EffectPattern;
use crate::types::{Matrix2, Matrix9, Matrix9x2, Vector2, Vector9};

/// Result of effect decomposition (internal, detailed).
///
/// Contains the full posterior distribution for diagnostic purposes.
/// For the simplified public API, use `EffectEstimate`.
#[derive(Debug, Clone)]
pub struct EffectDecomposition {
    /// Posterior mean of (shift, tail) effects in nanoseconds.
    /// - β[0] = μ (uniform shift)
    /// - β[1] = τ (tail effect)
    pub posterior_mean: Vector2,

    /// Posterior covariance of (shift, tail) effects.
    /// Λ_post = (Xᵀ Σ_n⁻¹ X + Λ₀⁻¹)⁻¹
    pub posterior_cov: Matrix2,

    /// 95% credible interval for shift effect (μ).
    pub shift_ci: (f64, f64),

    /// 95% credible interval for tail effect (τ).
    pub tail_ci: (f64, f64),

    /// Classified effect pattern (spec §3.4.6).
    pub pattern: EffectPattern,
}

/// Simplified effect estimate for public API.
///
/// This is the format used by the adaptive architecture to report
/// effect sizes to callers.
#[derive(Debug, Clone)]
pub struct EffectEstimate {
    /// Uniform shift in nanoseconds (positive = baseline slower).
    ///
    /// All quantiles shift by approximately this amount.
    /// Example: A branch that adds constant overhead.
    pub shift_ns: f64,

    /// Tail effect in nanoseconds (positive = baseline has heavier upper tail).
    ///
    /// Upper quantiles shift more than lower quantiles.
    /// Example: Cache misses that occur probabilistically.
    pub tail_ns: f64,

    /// 95% credible interval for total effect magnitude in nanoseconds.
    ///
    /// Computed from ||β||₂ samples from the posterior distribution.
    /// Note: May not contain point estimate when β ≈ 0.
    pub credible_interval_ns: (f64, f64),

    /// Dominant effect pattern (spec §3.4.6).
    pub pattern: EffectPattern,
}

impl EffectDecomposition {
    /// Convert to simplified effect estimate.
    ///
    /// Uses the magnitude CI from the posterior samples rather than
    /// computing it from the marginal CIs (which would be incorrect
    /// for non-independent components).
    pub fn to_estimate(&self, magnitude_ci: (f64, f64)) -> EffectEstimate {
        EffectEstimate {
            shift_ns: self.posterior_mean[0],
            tail_ns: self.posterior_mean[1],
            credible_interval_ns: magnitude_ci,
            pattern: self.pattern,
        }
    }
}

/// Decompose timing differences into shift and tail effects (spec §3.4.6).
///
/// Uses Bayesian linear regression with the same model as the Bayesian layer:
/// - Design matrix X = [ones | b_tail] (9×2)
/// - Gaussian prior on β: N(0, Λ₀), Λ₀ = diag(σ_μ², σ_τ²)
/// - Likelihood: Δ | β ~ N(Xβ, Σ_n)
///
/// # Arguments
///
/// * `delta` - Observed quantile differences (9-vector, baseline - sample)
/// * `sigma_n` - Covariance matrix (already scaled for inference sample size)
/// * `prior_sigmas` - Prior standard deviations (σ_μ, σ_τ) in nanoseconds
///
/// # Returns
///
/// An `EffectDecomposition` with posterior estimates and credible intervals.
///
/// # Note
///
/// This function duplicates some computation with `compute_bayes_gibbs`.
/// In the adaptive architecture, prefer using `BayesResult.beta_mean` and
/// `BayesResult.beta_cov` directly, then calling `classify_pattern` for
/// the pattern classification.
pub fn decompose_effect(
    delta: &Vector9,
    sigma_n: &Matrix9,
    prior_sigmas: (f64, f64),
) -> EffectDecomposition {
    // Build design matrix X = [ones | b_tail]
    let design_matrix = crate::analysis::bayes::build_design_matrix();

    // Compute posterior using Bayesian linear regression
    let (posterior_mean, posterior_cov) =
        bayesian_linear_regression(&design_matrix, delta, sigma_n, prior_sigmas);

    // Compute 95% credible intervals (marginal, for each component)
    let shift_ci = compute_credible_interval(posterior_mean[0], posterior_cov[(0, 0)]);
    let tail_ci = compute_credible_interval(posterior_mean[1], posterior_cov[(1, 1)]);

    // Classify the effect pattern (spec §3.4.6)
    let pattern = classify_pattern(&posterior_mean, &posterior_cov);

    EffectDecomposition {
        posterior_mean,
        posterior_cov,
        shift_ci,
        tail_ci,
        pattern,
    }
}

/// Bayesian linear regression with Gaussian prior (spec §3.4.6).
///
/// Prior: β ~ N(0, Λ₀), Λ₀ = diag(σ_μ², σ_τ²)
/// Likelihood: Δ | β ~ N(Xβ, Σ_n)
///
/// Posterior: β | Δ ~ N(β_post, Λ_post)
/// where:
///   Λ_post = (Xᵀ Σ_n⁻¹ X + Λ₀⁻¹)⁻¹
///   β_post = Λ_post Xᵀ Σ_n⁻¹ Δ
fn bayesian_linear_regression(
    design: &Matrix9x2,
    delta: &Vector9,
    sigma_n: &Matrix9,
    prior_sigmas: (f64, f64),
) -> (Vector2, Matrix2) {
    // Apply variance floor regularization (spec §3.3.2)
    // This prevents near-zero variance quantiles from dominating the regression
    let regularized = regularize_covariance(sigma_n);

    let chol = match Cholesky::new(regularized) {
        Some(c) => c,
        None => {
            // Fallback: add more jitter if still failing
            let extra_reg = regularized + Matrix9::identity() * 1e-6;
            Cholesky::new(extra_reg).expect("Regularized covariance should be positive definite")
        }
    };

    // Σ_n⁻¹ via Cholesky
    let sigma_n_inv = chol.inverse();

    // Compute sufficient statistics
    let xt_sigma_n_inv = design.transpose() * sigma_n_inv;
    let xt_sigma_n_inv_x = xt_sigma_n_inv * design; // Xᵀ Σ_n⁻¹ X (data precision)
    let xt_sigma_n_inv_delta = xt_sigma_n_inv * delta; // Xᵀ Σ_n⁻¹ Δ (data contribution)

    // Prior precision: Λ₀⁻¹ = diag(1/σ_μ², 1/σ_τ²)
    let (sigma_mu, sigma_tau) = prior_sigmas;
    let mut prior_precision = Matrix2::zeros();
    prior_precision[(0, 0)] = 1.0 / math::sq(sigma_mu.max(1e-12));
    prior_precision[(1, 1)] = 1.0 / math::sq(sigma_tau.max(1e-12));

    // Posterior precision: Λ_post⁻¹ = Xᵀ Σ_n⁻¹ X + Λ₀⁻¹
    let posterior_precision = xt_sigma_n_inv_x + prior_precision;

    // Posterior covariance: Λ_post = (Xᵀ Σ_n⁻¹ X + Λ₀⁻¹)⁻¹
    let posterior_cov = match Cholesky::new(posterior_precision) {
        Some(c) => c.inverse(),
        None => Matrix2::identity() * 1e6, // Very wide prior if inversion fails
    };

    // Posterior mean: β_post = Λ_post Xᵀ Σ_n⁻¹ Δ
    let posterior_mean = posterior_cov * xt_sigma_n_inv_delta;

    (posterior_mean, posterior_cov)
}

/// Compute 95% credible interval assuming normal posterior.
///
/// For a Gaussian posterior with given mean and variance, the 95% CI is:
///   [mean - 1.96×σ, mean + 1.96×σ]
fn compute_credible_interval(mean: f64, variance: f64) -> (f64, f64) {
    let std = math::sqrt(variance);
    let z = 1.96; // 97.5th percentile of standard normal
    (mean - z * std, mean + z * std)
}

/// Classify the effect pattern based on posterior estimates (spec §3.4.6).
///
/// An effect component is "significant" if its magnitude exceeds twice
/// its posterior standard error: |effect| > 2×SE
///
/// Classification rules:
/// - UniformShift: |μ| > 2σ_μ, |τ| ≤ 2σ_τ
/// - TailEffect: |τ| > 2σ_τ, |μ| ≤ 2σ_μ
/// - Mixed: Both significant
/// - Indeterminate: Neither significant
///
/// # Arguments
///
/// * `beta_mean` - Posterior mean β = (μ, τ)
/// * `beta_cov` - Posterior covariance Λ_post
pub fn classify_pattern(beta_mean: &Vector2, beta_cov: &Matrix2) -> EffectPattern {
    let shift = beta_mean[0]; // μ
    let tail = beta_mean[1]; // τ

    // Posterior standard errors
    let shift_se = math::sqrt(beta_cov[(0, 0)]);
    let tail_se = math::sqrt(beta_cov[(1, 1)]);

    // Check if effects are "significant" (|effect| > 2×SE)
    let shift_significant = shift.abs() > 2.0 * shift_se;
    let tail_significant = tail.abs() > 2.0 * tail_se;

    match (shift_significant, tail_significant) {
        (true, false) => EffectPattern::UniformShift,
        (false, true) => EffectPattern::TailEffect,
        (true, true) => EffectPattern::Mixed,
        // When neither effect is statistically significant, return Indeterminate
        (false, false) => EffectPattern::Indeterminate,
    }
}

/// Apply variance floor regularization for numerical stability (spec §3.3.2).
///
/// When some quantiles have zero or near-zero variance (common in discrete mode
/// with ties), the covariance matrix becomes ill-conditioned. Even if Cholesky
/// succeeds, the inverse has huge values for near-zero variance elements,
/// causing them to dominate the Bayesian regression incorrectly.
///
/// We regularize by ensuring a minimum diagonal value of 1% of mean variance.
/// This bounds the condition number to ~100, preventing numerical instability.
///
/// Formula (spec §3.3.2):
///   σ²ᵢ ← max(σ²ᵢ, 0.01 × σ̄²) + ε
/// where σ̄² = tr(Σ)/9 and ε = 10⁻¹⁰ + σ̄² × 10⁻⁸
fn regularize_covariance(sigma: &Matrix9) -> Matrix9 {
    let trace: f64 = (0..9).map(|i| sigma[(i, i)]).sum();
    let mean_var = trace / 9.0;

    // Use 1% of mean variance as floor, with absolute minimum of 1e-10
    let min_var = (0.01 * mean_var).max(1e-10);

    // Also add small jitter proportional to scale for numerical stability
    let jitter = 1e-10 + mean_var * 1e-8;

    let mut regularized = *sigma;
    for i in 0..9 {
        regularized[(i, i)] = regularized[(i, i)].max(min_var) + jitter;
    }
    regularized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::bayes::build_design_matrix;
    use crate::constants::B_TAIL;

    #[test]
    fn test_design_matrix_structure() {
        let x = build_design_matrix();

        // First column should be all ones
        for i in 0..9 {
            assert!((x[(i, 0)] - 1.0).abs() < 1e-10);
        }

        // Second column should match B_TAIL
        for i in 0..9 {
            assert!((x[(i, 1)] - B_TAIL[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_credible_interval_symmetry() {
        let (low, high) = compute_credible_interval(0.0, 1.0);
        assert!(
            (low + high).abs() < 0.01,
            "CI should be symmetric around zero"
        );
    }

    #[test]
    fn test_classify_uniform_shift() {
        // Large shift, no tail effect
        let mean = Vector2::new(10.0, 0.1);
        let cov = Matrix2::identity();
        let pattern = classify_pattern(&mean, &cov);
        assert_eq!(pattern, EffectPattern::UniformShift);
    }

    #[test]
    fn test_classify_tail_effect() {
        // No shift, large tail effect
        let mean = Vector2::new(0.1, 10.0);
        let cov = Matrix2::identity();
        let pattern = classify_pattern(&mean, &cov);
        assert_eq!(pattern, EffectPattern::TailEffect);
    }

    #[test]
    fn test_classify_indeterminate() {
        // Neither shift nor tail significant (both small relative to SE)
        let mean = Vector2::new(0.5, 0.5);
        let cov = Matrix2::identity();
        let pattern = classify_pattern(&mean, &cov);
        assert_eq!(pattern, EffectPattern::Indeterminate);
    }
}
