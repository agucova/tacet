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
    /// 95% credible interval for total effect magnitude.
    pub effect_magnitude_ci: (f64, f64),
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
    let design_matrix = build_design_matrix();

    // Compute posterior using Bayesian linear regression
    let (posterior_mean, posterior_cov) =
        bayesian_linear_regression(&design_matrix, observed_diff, covariance, prior_sigmas);

    // Compute 95% credible intervals
    let shift_ci = compute_credible_interval(posterior_mean[0], posterior_cov[(0, 0)]);
    let tail_ci = compute_credible_interval(posterior_mean[1], posterior_cov[(1, 1)]);

    // Classify the effect pattern
    let pattern = classify_pattern(&posterior_mean, &posterior_cov);

    // Compute effect magnitude CI from posterior samples
    // Reduced from 5000 to 500: for 2D Gaussian, 500 draws gives ~3-5% relative error
    // on 95% CI bounds, which is acceptable for this heuristic estimate
    let effect_magnitude_ci = sample_effect_magnitude_ci(&posterior_mean, &posterior_cov, 500);

    EffectDecomposition {
        posterior_mean,
        posterior_cov,
        shift_ci,
        tail_ci,
        effect_magnitude_ci,
        pattern,
    }
}

/// Build the 9x2 design matrix [ones | b_tail].
fn build_design_matrix() -> Matrix9x2 {
    let mut x = Matrix9x2::zeros();

    for i in 0..9 {
        x[(i, 0)] = ONES[i];
        x[(i, 1)] = B_TAIL[i];
    }

    x
}

/// Bayesian linear regression with Gaussian prior.
///
/// Prior: beta ~ N(0, prior_var * I)
/// Likelihood: y | beta ~ N(X*beta, Sigma)
///
/// Posterior: beta | y ~ N(mu_post, Sigma_post)
/// where:
///   Sigma_post^-1 = X^T Sigma^-1 X + prior_var^-1 * I
///   mu_post = Sigma_post * X^T * Sigma^-1 * y
fn bayesian_linear_regression(
    design: &Matrix9x2,
    observed: &Vector9,
    covariance: &Matrix9,
    prior_sigmas: (f64, f64),
) -> (Vector2, Matrix2) {
    // Compute Cholesky of covariance for efficient inversion
    let chol = match Cholesky::new(*covariance) {
        Some(c) => c,
        None => {
            // Regularize if not positive definite
            let regularized = covariance + Matrix9::identity() * 1e-10;
            Cholesky::new(regularized).expect("Regularized covariance should be positive definite")
        }
    };

    // Compute Sigma^-1 * X using forward/backward substitution
    // We need X^T * Sigma^-1 * X and X^T * Sigma^-1 * y
    let sigma_inv_x = solve_cholesky_matrix(&chol, design);
    let sigma_inv_y = chol.solve(observed);

    // X^T * Sigma^-1 * X
    let xt_sigma_inv_x = design.transpose() * sigma_inv_x;

    // X^T * Sigma^-1 * y
    let xt_sigma_inv_y = design.transpose() * sigma_inv_y;

    // Prior precision: diag(1/sigma^2)
    let (sigma_mu, sigma_tau) = prior_sigmas;
    let mut prior_precision = Matrix2::zeros();
    prior_precision[(0, 0)] = 1.0 / sigma_mu.max(1e-12).powi(2);
    prior_precision[(1, 1)] = 1.0 / sigma_tau.max(1e-12).powi(2);

    // Posterior precision: X^T Sigma^-1 X + prior precision
    let posterior_precision = xt_sigma_inv_x + prior_precision;

    // Posterior covariance: inverse of posterior precision
    let posterior_cov = match Cholesky::new(posterior_precision) {
        Some(c) => c.inverse(),
        None => {
            // Fallback to large variance if inversion fails
            Matrix2::identity() * 1e6
        }
    };

    // Posterior mean: Sigma_post * X^T * Sigma^-1 * y
    let posterior_mean = posterior_cov * xt_sigma_inv_y;

    (posterior_mean, posterior_cov)
}

/// Solve Sigma^-1 * M using Cholesky decomposition.
fn solve_cholesky_matrix(
    chol: &Cholesky<f64, nalgebra::Const<9>>,
    matrix: &Matrix9x2,
) -> Matrix9x2 {
    // Solve column by column
    let col0 = chol.solve(&matrix.column(0).into_owned());
    let col1 = chol.solve(&matrix.column(1).into_owned());

    let mut result = Matrix9x2::zeros();
    result.set_column(0, &col0);
    result.set_column(1, &col1);
    result
}

/// Compute 95% credible interval assuming normal posterior.
fn compute_credible_interval(mean: f64, variance: f64) -> (f64, f64) {
    // 95% CI: mean +/- 1.96 * std
    let std = variance.sqrt();
    let z = 1.96;
    (mean - z * std, mean + z * std)
}

/// Sample posterior to estimate CI of total effect magnitude.
fn sample_effect_magnitude_ci(
    posterior_mean: &Vector2,
    posterior_cov: &Matrix2,
    n_samples: usize,
) -> (f64, f64) {
    let chol = match Cholesky::new(*posterior_cov) {
        Some(c) => c,
        None => {
            let regularized = posterior_cov + Matrix2::identity() * 1e-12;
            Cholesky::new(regularized).expect("Regularized covariance should be positive definite")
        }
    };

    let mut rng = rand::rng();
    let mut magnitudes = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let z0: f64 = StandardNormal.sample(&mut rng);
        let z1: f64 = StandardNormal.sample(&mut rng);
        let z = Vector2::new(z0, z1);
        let sample = posterior_mean + chol.l() * z;
        let magnitude = (sample[0].powi(2) + sample[1].powi(2)).sqrt();
        magnitudes.push(magnitude);
    }

    magnitudes.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = ((n_samples as f64) * 0.025).round() as usize;
    let hi_idx = ((n_samples as f64) * 0.975).round() as usize;
    let lo = magnitudes[lo_idx.min(n_samples - 1)];
    let hi = magnitudes[hi_idx.min(n_samples - 1)];

    (lo, hi)
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
        (false, false) => {
            // Neither significant: classify by relative magnitude
            if shift.abs() >= tail.abs() {
                EffectPattern::UniformShift
            } else {
                EffectPattern::TailEffect
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
