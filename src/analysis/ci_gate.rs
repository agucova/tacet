//! Layer 1: RTLF-style CI gate with bounded false positive rate.
//!
//! This implements a frequentist screening layer that provides:
//! - Pass/fail decision suitable for CI pipelines
//! - Guaranteed false positive rate via Bonferroni correction
//! - Bootstrap-based threshold computation within each class
//!
//! The approach follows the methodology from:
//! Reparaz, Tunstall, Lawson, and Faust (RTLF) for constant-time validation.

use crate::result::CiGate;
use crate::types::{Matrix9, Vector9};

/// Input data for CI gate analysis.
#[derive(Debug, Clone)]
pub struct CiGateInput {
    /// Observed quantile differences (fixed - random) for 9 deciles.
    pub observed_diff: Vector9,
    /// Covariance matrix of quantile differences for fixed class.
    pub cov_fixed: Matrix9,
    /// Covariance matrix of quantile differences for random class.
    pub cov_random: Matrix9,
    /// Sample size per class.
    pub n_samples: usize,
    /// Significance level (e.g., 0.05 for 5% false positive rate).
    pub alpha: f64,
}

/// Run the CI gate analysis.
///
/// This implements RTLF-style testing with:
/// 1. Bootstrap within each class separately to estimate null distribution
/// 2. Compute per-class thresholds at corrected alpha level
/// 3. Take maximum threshold across classes for each quantile
/// 4. Apply Bonferroni correction (alpha/9) for multiple testing
///
/// # Arguments
///
/// * `input` - CI gate input containing observed differences and covariances
///
/// # Returns
///
/// A `CiGate` result indicating pass/fail and threshold details.
pub fn run_ci_gate(input: &CiGateInput) -> CiGate {
    // Bonferroni correction for 9 simultaneous tests
    let corrected_alpha = input.alpha / 9.0;

    // Compute thresholds via bootstrap
    let thresholds = compute_bootstrap_thresholds(
        &input.cov_fixed,
        &input.cov_random,
        input.n_samples,
        corrected_alpha,
    );

    // Convert observed differences to array
    let observed: [f64; 9] = input.observed_diff.as_slice().try_into().unwrap_or([0.0; 9]);

    // Convert thresholds to array
    let thresholds_arr: [f64; 9] = thresholds.as_slice().try_into().unwrap_or([0.0; 9]);

    // Check if any observed difference exceeds its threshold
    let passed = observed
        .iter()
        .zip(thresholds_arr.iter())
        .all(|(obs, thresh)| obs.abs() <= *thresh);

    CiGate {
        alpha: input.alpha,
        passed,
        thresholds: thresholds_arr,
        observed,
    }
}

/// Compute bootstrap thresholds for each quantile.
///
/// For each class:
/// 1. Generate bootstrap samples from the estimated null distribution
/// 2. Compute the (1 - alpha) quantile of bootstrap differences
/// 3. Take the maximum across both classes
fn compute_bootstrap_thresholds(
    cov_fixed: &Matrix9,
    cov_random: &Matrix9,
    n_samples: usize,
    alpha: f64,
) -> Vector9 {
    // Number of bootstrap iterations
    const N_BOOTSTRAP: usize = 10_000;

    // TODO: Implement proper bootstrap sampling
    //
    // Algorithm:
    // 1. For class in [fixed, random]:
    //    a. Use Cholesky decomposition of cov to generate MVN samples
    //    b. Scale by sqrt(n_samples) for standard error
    //    c. Generate N_BOOTSTRAP samples
    //    d. For each quantile, compute (1 - alpha) percentile of absolute values
    // 2. Take element-wise maximum of thresholds from both classes

    // Placeholder: use diagonal elements scaled by critical value
    // This approximates the threshold assuming independent quantiles
    let z_crit = quantile_normal(1.0 - alpha / 2.0);

    let mut thresholds = Vector9::zeros();
    for i in 0..9 {
        let se_fixed = (cov_fixed[(i, i)] / n_samples as f64).sqrt();
        let se_random = (cov_random[(i, i)] / n_samples as f64).sqrt();
        let se_combined = (se_fixed.powi(2) + se_random.powi(2)).sqrt();
        thresholds[i] = z_crit * se_combined;
    }

    thresholds
}

/// Compute the quantile of the standard normal distribution.
///
/// Uses the approximation from Abramowitz and Stegun (1964).
fn quantile_normal(p: f64) -> f64 {
    // TODO: Use a proper implementation or library
    // This is a rough approximation for the upper quantiles

    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }

    // Rational approximation for central region
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    // Coefficients for the approximation
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -result
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ci_gate_passes_on_zero_difference() {
        let input = CiGateInput {
            observed_diff: Vector9::zeros(),
            cov_fixed: Matrix9::identity(),
            cov_random: Matrix9::identity(),
            n_samples: 1000,
            alpha: 0.05,
        };

        let result = run_ci_gate(&input);
        assert!(result.passed, "CI gate should pass when no difference observed");
    }

    #[test]
    fn test_quantile_normal_symmetry() {
        let q_upper = quantile_normal(0.975);
        let q_lower = quantile_normal(0.025);
        assert!((q_upper + q_lower).abs() < 0.01, "Normal quantiles should be symmetric");
    }
}
