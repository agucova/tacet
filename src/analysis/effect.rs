//! Effect decomposition: separate uniform shift from tail effects.
//!
//! This module implements Bayesian linear regression to decompose timing
//! differences into two components:
//!
//! - **Uniform shift**: Constant offset across all quantiles (e.g., branch timing)
//! - **Tail effect**: Deviation in upper quantiles (e.g., cache misses)
//!
//! The model is: delta = beta_0 * ones + beta_1 * b_tail + epsilon
//!
//! Where:
//! - delta is the 9-vector of observed quantile differences
//! - ones = [1, 1, ..., 1] captures uniform shift
//! - b_tail = [-0.5, -0.375, ..., 0.5] captures tail effects
//! - epsilon ~ MVN(0, Sigma) is noise

use nalgebra::Cholesky;
use rand_distr::{Distribution, StandardNormal};

use crate::constants::{B_TAIL, ONES};
use crate::result::EffectPattern;
use crate::types::{Matrix2, Matrix9, Matrix9x2, Vector2, Vector9};

/// Result of effect decomposition.
#[derive(Debug, Clone)]
pub struct EffectDecomposition {
    /// Posterior mean of (shift, tail) effects in nanoseconds.
    pub posterior_mean: Vector2,
    /// Posterior covariance of (shift, tail) effects.
    pub posterior_cov: Matrix2,
    /// 95% credible interval for shift effect.
    pub shift_ci: (f64, f64),
    /// 95% credible interval for tail effect.
    pub tail_ci: (f64, f64),
    /// Classified effect pattern.
    pub pattern: EffectPattern,
}

/// Decompose timing differences into shift and tail effects.
///
/// Uses Bayesian linear regression with:
/// - Design matrix X = [ones | b_tail] (9x2)
/// - Gaussian prior on beta: N(0, prior_var * I)
/// - Likelihood: delta | beta ~ N(X*beta, Sigma)
///
/// # Arguments
///
/// * `observed_diff` - Observed quantile differences (9-vector)
/// * `covariance` - Covariance matrix of differences (9x9)
/// * `prior_sigmas` - Prior standard deviations (shift, tail) in nanoseconds
///
/// # Returns
///
/// An `EffectDecomposition` with posterior estimates and credible intervals.
pub fn decompose_effect(
    observed_diff: &Vector9,
    covariance: &Matrix9,
    prior_sigmas: (f64, f64),
) -> EffectDecomposition {
    // Build design matrix X = [ones | b_tail]
    let design_matrix = crate::analysis::bayes::build_design_matrix();

    // Compute posterior using Bayesian linear regression
    let (posterior_mean, posterior_cov) =
        bayesian_linear_regression(&design_matrix, observed_diff, covariance, prior_sigmas);

    // Compute 95% credible intervals
    let shift_ci = compute_credible_interval(posterior_mean[0], posterior_cov[(0, 0)]);
    let tail_ci = compute_credible_interval(posterior_mean[1], posterior_cov[(1, 1)]);

    // Classify the effect pattern
    let pattern = classify_pattern(&posterior_mean, &posterior_cov);

    EffectDecomposition {
        posterior_mean,
        posterior_cov,
        shift_ci,
        tail_ci,
        pattern,
    }
}

/// Bayesian linear regression with Gaussian prior.
///
/// Prior: beta ~ N(0, prior_var * I)
/// Likelihood: y | beta ~ N(X*beta, Sigma)
///
/// Posterior: beta | y ~ N(mu_post, Sigma_post)
/// where:
///   Sigma_post^-1 = X^T Sigma^-1 X + prior_precision
///   mu_post = Sigma_post * X^T * Sigma^-1 * y
fn bayesian_linear_regression(
    design: &Matrix9x2,
    observed: &Vector9,
    covariance: &Matrix9,
    prior_sigmas: (f64, f64),
) -> (Vector2, Matrix2) {
    // Apply variance floor regularization (spec §2.6)
    // This prevents near-zero variance quantiles from dominating the regression
    let regularized = regularize_covariance(covariance);

    let chol = match Cholesky::new(regularized) {
        Some(c) => c,
        None => {
            // Fallback: add more jitter if still failing
            let extra_reg = regularized + Matrix9::identity() * 1e-6;
            Cholesky::new(extra_reg).expect("Regularized covariance should be positive definite")
        }
    };

    let sigma_inv = chol.inverse();
    let xt_sigma_inv = design.transpose() * sigma_inv;
    let xt_sigma_inv_x = xt_sigma_inv * design;
    let xt_sigma_inv_y = xt_sigma_inv * observed;

    let (sigma_mu, sigma_tau) = prior_sigmas;
    let mut prior_precision = Matrix2::zeros();
    prior_precision[(0, 0)] = 1.0 / sigma_mu.max(1e-12).powi(2);
    prior_precision[(1, 1)] = 1.0 / sigma_tau.max(1e-12).powi(2);

    let posterior_precision = xt_sigma_inv_x + prior_precision;
    let posterior_cov = match Cholesky::new(posterior_precision) {
        Some(c) => c.inverse(),
        None => Matrix2::identity() * 1e6,
    };

    let posterior_mean = posterior_cov * xt_sigma_inv_y;
    (posterior_mean, posterior_cov)
}

/// Compute 95% credible interval assuming normal posterior.
/// Compute 95% credible interval assuming normal posterior.
fn compute_credible_interval(mean: f64, variance: f64) -> (f64, f64) {
    // 95% CI: mean +/- 1.96 * std
    let std = variance.sqrt();
    let z = 1.96;
    (mean - z * std, mean + z * std)
}

/// Classify the effect pattern based on posterior estimates.
fn classify_pattern(posterior_mean: &Vector2, posterior_cov: &Matrix2) -> EffectPattern {
    let shift = posterior_mean[0];
    let tail = posterior_mean[1];

    // Standard errors
    let shift_se = posterior_cov[(0, 0)].sqrt();
    let tail_se = posterior_cov[(1, 1)].sqrt();

    // Check if effects are "significant" (|effect| > 2*SE)
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

/// Regularize covariance matrix for numerical stability (spec §2.6).
///
/// Ensures minimum diagonal value of 1% of mean variance, bounding
/// condition number to ~100. This prevents near-zero variance quantiles
/// from dominating the Bayesian regression.
fn regularize_covariance(covariance: &Matrix9) -> Matrix9 {
    let trace: f64 = (0..9).map(|i| covariance[(i, i)]).sum();
    let mean_var = trace / 9.0;

    // Use 1% of mean variance as floor, with absolute minimum of 1e-10
    let min_var = (0.01 * mean_var).max(1e-10);

    // Also add small jitter proportional to scale
    let jitter = 1e-10 + mean_var * 1e-8;

    let mut regularized = *covariance;
    for i in 0..9 {
        regularized[(i, i)] = regularized[(i, i)].max(min_var) + jitter;
    }
    regularized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::bayes::build_design_matrix;

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
        assert!((low + high).abs() < 0.01, "CI should be symmetric around zero");
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
