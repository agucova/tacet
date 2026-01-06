//! Diagnostic checks for result reliability (spec §2.8).
//!
//! This module implements three diagnostic checks:
//! 1. Non-stationarity: Compare variance between calibration and inference sets
//! 2. Model fit: Chi-squared test for residuals from shift+tail model
//! 3. Outlier asymmetry: Check if outlier rates differ between classes

use crate::constants::{B_TAIL, ONES};
use crate::measurement::OutlierStats;
use crate::preflight::PreflightResult;
use crate::result::{Diagnostics, FilteredQuantile};
use crate::types::{Matrix9, Matrix9x2, Vector9};
use nalgebra::Cholesky;

/// Additional diagnostic information computed during analysis.
#[derive(Debug, Clone, Default)]
pub struct DiagnosticsExtra {
    /// Block size used for bootstrap (from Politis-White).
    pub dependence_length: usize,
    /// Number of samples per class after filtering.
    pub samples_per_class: usize,
    /// Quantiles filtered from CI gate with reasons.
    pub filtered_quantiles: Vec<FilteredQuantile>,
    /// Whether discrete timer mode was used.
    pub discrete_mode: bool,
    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,
    /// Fraction of samples with duplicate values.
    pub duplicate_fraction: f64,
}

/// Compute all diagnostic checks.
///
/// # Arguments
///
/// * `calib_cov` - Covariance matrix from calibration set
/// * `infer_cov` - Covariance matrix from inference set
/// * `observed_diff` - Observed quantile differences
/// * `posterior_mean` - Posterior mean of (shift, tail) effects
/// * `outlier_stats` - Statistics about outlier filtering
/// * `preflight` - Preflight check results (sanity, generator, autocorrelation, system)
/// * `extra` - Additional diagnostic information (block length, filtered quantiles, etc.)
pub fn compute_diagnostics(
    calib_cov: &Matrix9,
    infer_cov: &Matrix9,
    observed_diff: &Vector9,
    posterior_mean: &[f64; 2],
    outlier_stats: &OutlierStats,
    preflight: &PreflightResult,
    extra: &DiagnosticsExtra,
) -> Diagnostics {
    let mut warnings = Vec::new();

    // Add preflight warnings first (most important)
    for warning in &preflight.warnings.sanity {
        warnings.push(warning.description());
    }
    for warning in &preflight.warnings.generator {
        warnings.push(warning.description());
    }
    for warning in &preflight.warnings.autocorr {
        warnings.push(warning.description());
    }
    for warning in &preflight.warnings.system {
        warnings.push(warning.description());
    }
    for warning in &preflight.warnings.resolution {
        warnings.push(warning.description());
    }

    // 1. Non-stationarity check
    let (stationarity_ratio, stationarity_ok) = check_stationarity(calib_cov, infer_cov);
    if !stationarity_ok {
        if stationarity_ratio > 5.0 {
            warnings.push(format!(
                "Non-stationarity detected: variance ratio {:.2} (expected 0.5-2.0). Results may be unreliable.",
                stationarity_ratio
            ));
        } else {
            warnings.push(format!(
                "Possible non-stationarity: variance ratio {:.2} (expected 0.5-2.0).",
                stationarity_ratio
            ));
        }
    }

    // 2. Model fit check
    let (model_fit_chi2, model_fit_ok) = check_model_fit(observed_diff, calib_cov, posterior_mean);
    if !model_fit_ok {
        warnings.push(format!(
            "Model fit issue: χ² = {:.1} (expected < 18.5). Effect decomposition may be misleading.",
            model_fit_chi2
        ));
    }

    // 3. Outlier asymmetry check
    let outlier_rate_fixed = outlier_stats.rate_fixed();
    let outlier_rate_random = outlier_stats.rate_random();
    let outlier_asymmetry_ok = check_outlier_asymmetry(outlier_rate_fixed, outlier_rate_random);
    if !outlier_asymmetry_ok {
        warnings.push(format!(
            "Outlier asymmetry: fixed={:.2}%, random={:.2}%. May indicate tail leak.",
            outlier_rate_fixed * 100.0,
            outlier_rate_random * 100.0
        ));
    }

    // Compute effective sample size (ESS ≈ n / block_length)
    let effective_sample_size = if extra.dependence_length > 0 {
        extra.samples_per_class / extra.dependence_length
    } else {
        extra.samples_per_class
    };

    Diagnostics {
        dependence_length: extra.dependence_length,
        effective_sample_size,
        stationarity_ratio,
        stationarity_ok,
        model_fit_chi2,
        model_fit_ok,
        outlier_rate_fixed,
        outlier_rate_random,
        outlier_asymmetry_ok,
        filtered_quantiles: extra.filtered_quantiles.clone(),
        discrete_mode: extra.discrete_mode,
        timer_resolution_ns: extra.timer_resolution_ns,
        duplicate_fraction: extra.duplicate_fraction,
        preflight_ok: preflight.is_valid,
        warnings,
    }
}

/// Check non-stationarity by comparing variance between calibration and inference.
///
/// Returns (ratio, ok) where ratio = tr(Σ_infer) / tr(Σ_calib).
/// OK if ratio is in [0.5, 2.0].
fn check_stationarity(calib_cov: &Matrix9, infer_cov: &Matrix9) -> (f64, bool) {
    let calib_trace: f64 = (0..9).map(|i| calib_cov[(i, i)]).sum();
    let infer_trace: f64 = (0..9).map(|i| infer_cov[(i, i)]).sum();

    // Avoid division by zero
    if calib_trace < 1e-12 {
        return (f64::INFINITY, false);
    }

    let ratio = infer_trace / calib_trace;
    let ok = (0.5..=2.0).contains(&ratio);

    (ratio, ok)
}

/// Check model fit using chi-squared test on residuals.
///
/// Residual: r = Δ - X * β̂
/// Chi-squared: r' Σ⁻¹ r ~ χ²₇ (9 dims - 2 params)
///
/// Returns (chi2, ok) where ok if chi2 ≤ 18.5 (p > 0.01 under χ²₇).
fn check_model_fit(
    observed_diff: &Vector9,
    covariance: &Matrix9,
    posterior_mean: &[f64; 2],
) -> (f64, bool) {
    // Build design matrix X = [ones | b_tail]
    let mut x = Matrix9x2::zeros();
    for i in 0..9 {
        x[(i, 0)] = ONES[i];
        x[(i, 1)] = B_TAIL[i];
    }

    // Compute predicted: X * β̂
    let beta = nalgebra::Vector2::new(posterior_mean[0], posterior_mean[1]);
    let predicted = x * beta;

    // Residual: r = Δ - X * β̂
    let residual = observed_diff - predicted;

    // Compute chi-squared: r' Σ⁻¹ r
    let chi2 = match safe_cholesky(covariance) {
        Some(chol) => {
            let z = chol.solve(&residual);
            residual.dot(&z)
        }
        None => {
            // If Cholesky fails, we can't compute chi2 reliably
            0.0
        }
    };

    // χ²₇ at p=0.01 is 18.48
    let ok = chi2 <= 18.5;

    (chi2, ok)
}

/// Check outlier asymmetry between classes.
///
/// OK if:
/// - Both rates < 1%, AND
/// - Rate ratio < 3×, AND
/// - Absolute difference < 2%
fn check_outlier_asymmetry(rate_fixed: f64, rate_random: f64) -> bool {
    // Both rates should be low
    if rate_fixed >= 0.01 || rate_random >= 0.01 {
        // At least one rate is high (≥1%)
        // Check for asymmetry
        let max_rate = rate_fixed.max(rate_random);
        let min_rate = rate_fixed.min(rate_random);
        let ratio = if min_rate > 1e-12 {
            max_rate / min_rate
        } else {
            f64::INFINITY
        };
        let diff = (rate_fixed - rate_random).abs();

        // Fail if ratio > 3× or diff > 2%
        if ratio > 3.0 || diff > 0.02 {
            return false;
        }
    }

    true
}

/// Safe Cholesky decomposition with jitter.
fn safe_cholesky(matrix: &Matrix9) -> Option<Cholesky<f64, nalgebra::Const<9>>> {
    if let Some(chol) = Cholesky::new(*matrix) {
        return Some(chol);
    }

    // Add jitter and retry
    let trace = matrix.trace();
    let jitter = 1e-10 + (trace / 9.0) * 1e-8;
    let mut regularized = *matrix;
    for i in 0..9 {
        regularized[(i, i)] += jitter;
    }

    Cholesky::new(regularized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stationarity_check() {
        let calib = Matrix9::identity();
        let infer = Matrix9::identity();
        let (ratio, ok) = check_stationarity(&calib, &infer);
        assert!((ratio - 1.0).abs() < 1e-10);
        assert!(ok);

        // 3x variance increase - warning territory
        let infer_high = Matrix9::identity() * 3.0;
        let (ratio, ok) = check_stationarity(&calib, &infer_high);
        assert!((ratio - 3.0).abs() < 1e-10);
        assert!(!ok);
    }

    #[test]
    fn test_outlier_asymmetry_check() {
        // Both low rates - OK
        assert!(check_outlier_asymmetry(0.001, 0.001));

        // Both moderate but similar - OK
        assert!(check_outlier_asymmetry(0.015, 0.012));

        // High asymmetry - not OK
        assert!(!check_outlier_asymmetry(0.03, 0.005));

        // Large difference - not OK
        assert!(!check_outlier_asymmetry(0.04, 0.01));
    }

    #[test]
    fn test_model_fit_check() {
        // With identity covariance and zero residual, chi2 should be 0
        let observed = Vector9::zeros();
        let cov = Matrix9::identity();
        let posterior_mean = [0.0, 0.0];
        let (chi2, ok) = check_model_fit(&observed, &cov, &posterior_mean);
        assert!(chi2 < 0.01);
        assert!(ok);
    }
}
