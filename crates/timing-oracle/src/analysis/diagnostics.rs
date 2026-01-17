//! Diagnostic checks for result reliability (spec §2.8).
//!
//! This module implements three diagnostic checks:
//! 1. Non-stationarity: Compare variance between calibration and inference sets
//! 2. Model fit: Chi-squared test for residuals from shift+tail model
//! 3. Outlier asymmetry: Check if outlier rates differ between classes

use crate::constants::{B_TAIL, ONES};
use crate::measurement::OutlierStats;
use crate::preflight::PreflightResult;
use crate::result::Diagnostics;
use crate::types::{Matrix9, Matrix9x2, Vector9};
use nalgebra::Cholesky;

/// Additional diagnostic information computed during analysis.
#[derive(Debug, Clone, Default)]
pub struct DiagnosticsExtra {
    /// Block size used for bootstrap (from Politis-White).
    pub dependence_length: usize,
    /// Number of samples per class after filtering.
    pub samples_per_class: usize,
    /// Whether discrete timer mode was used.
    pub discrete_mode: bool,
    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,
    /// Fraction of samples with duplicate values.
    pub duplicate_fraction: f64,
    /// Number of calibration samples used.
    pub calibration_samples: usize,
    /// Bootstrap-calibrated model mismatch threshold (99th percentile of Q*).
    pub q_thresh: f64,
}

/// Compute all diagnostic checks.
///
/// # Arguments
///
/// * `calib_cov` - Covariance matrix from calibration set
/// * `observed_diff` - Observed quantile differences
/// * `posterior_mean` - Posterior mean of (shift, tail) effects
/// * `outlier_stats` - Statistics about outlier filtering
/// * `preflight` - Preflight check results (sanity, generator, autocorrelation, system)
/// * `interleaved_samples` - Raw timing samples in measurement order
/// * `extra` - Additional diagnostic information (block length, filtered quantiles, etc.)
pub fn compute_diagnostics(
    calib_cov: &Matrix9,
    observed_diff: &Vector9,
    posterior_mean: &[f64; 2],
    outlier_stats: &OutlierStats,
    preflight: &PreflightResult,
    interleaved_samples: &[crate::types::TimingSample],
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

    // 1. Stationarity check (spec §3.2.1)
    let (stationarity_ratio, stationarity_ok) = check_stationarity_windowed(interleaved_samples);
    if !stationarity_ok {
        warnings.push(
            "Timing distribution appears to drift during measurement (stationarity suspect)."
                .to_string(),
        );
    }

    // 2. Model fit check (spec §4.1)
    let (model_fit_chi2, model_fit_ok) = check_model_fit(observed_diff, calib_cov, posterior_mean);
    if !model_fit_ok {
        warnings.push(format!(
            "Model fit issue: χ² = {:.1} (expected < 18.5). Effect decomposition may be misleading.",
            model_fit_chi2
        ));
    }

    // 3. Outlier asymmetry check (spec §3.3)
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

    if outlier_stats.outlier_fraction > 0.001 {
        warnings.push(format!(
            "High winsorization rate: {:.2}% of samples capped. Results may be less reliable.",
            outlier_stats.outlier_fraction * 100.0
        ));
    }

    // 4. Per-class dependence estimation (spec §3.2.2)
    let dependence_length = estimate_joint_dependence_length(interleaved_samples);
    let effective_sample_size = if dependence_length > 0 {
        extra.samples_per_class / dependence_length
    } else {
        extra.samples_per_class
    };

    Diagnostics {
        dependence_length,
        effective_sample_size,
        stationarity_ratio,
        stationarity_ok,
        model_fit_chi2,
        model_fit_ok,
        model_fit_threshold: if extra.q_thresh > 0.0 { extra.q_thresh } else { 18.48 },
        outlier_rate_baseline: outlier_rate_fixed,
        outlier_rate_sample: outlier_rate_random,
        outlier_asymmetry_ok,
        discrete_mode: extra.discrete_mode,
        timer_resolution_ns: extra.timer_resolution_ns,
        duplicate_fraction: extra.duplicate_fraction,
        preflight_ok: preflight.is_valid,
        calibration_samples: extra.calibration_samples,
        total_time_secs: 0.0, // Will be filled in by caller
        warnings,
        quality_issues: Vec::new(),
        preflight_warnings: {
            let mut pw = Vec::new();
            for w in &preflight.warnings.sanity {
                pw.push(w.to_warning_info());
            }
            for w in &preflight.warnings.generator {
                pw.push(w.to_warning_info());
            }
            for w in &preflight.warnings.autocorr {
                pw.push(w.to_warning_info());
            }
            for w in &preflight.warnings.system {
                pw.push(w.to_warning_info());
            }
            for w in &preflight.warnings.resolution {
                pw.push(w.to_warning_info());
            }
            pw
        },
        // Reproduction info - to be filled in by caller with config context
        seed: None,
        attacker_model: None,
        threshold_ns: 0.0,
        timer_name: String::new(),
        platform: String::new(),
    }
}

/// Check stationarity using windowed median/IQR (spec §3.2.1).
fn check_stationarity_windowed(samples: &[crate::types::TimingSample]) -> (f64, bool) {
    let n = samples.len();
    if n < 100 {
        return (1.0, true); // Too few samples for windowing
    }

    let w = 10; // 10 windows
    let window_size = n / w;
    let mut window_medians = Vec::with_capacity(w);
    let mut window_iqrs = Vec::with_capacity(w);
    let mut window_variances = Vec::with_capacity(w);

    for i in 0..w {
        let start = i * window_size;
        let end = if i == w - 1 { n } else { (i + 1) * window_size };
        let mut window_data: Vec<f64> = samples[start..end].iter().map(|s| s.time_ns).collect();

        window_data.sort_by(|a, b| a.total_cmp(b));
        let median = window_data[window_data.len() / 2];
        let q1 = window_data[window_data.len() / 4];
        let q3 = window_data[window_data.len() * 3 / 4];
        let iqr = q3 - q1;

        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
        let variance =
            window_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window_data.len() as f64;

        window_medians.push(median);
        window_iqrs.push(iqr);
        window_variances.push(variance);
    }

    let mut global_data: Vec<f64> = samples.iter().map(|s| s.time_ns).collect();
    global_data.sort_by(|a, b| a.total_cmp(b));
    let global_median = global_data[global_data.len() / 2];

    let max_median = window_medians
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_median = window_medians.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg_iqr = window_iqrs.iter().sum::<f64>() / w as f64;

    let drift_floor = 0.05 * global_median;
    let threshold = (2.0 * avg_iqr).max(drift_floor);
    let median_drift_ok = (max_median - min_median) <= threshold;

    // Check for monotonic variance drift (>50% change)
    let first_var = window_variances[0];
    let last_var = window_variances[w - 1];
    let var_drift_ratio = if first_var > 1e-12 {
        last_var / first_var
    } else {
        1.0
    };
    let var_drift_ok = (0.5..=1.5).contains(&var_drift_ratio);

    let ratio = if min_median > 1e-12 {
        max_median / min_median
    } else {
        1.0
    };
    (ratio, median_drift_ok && var_drift_ok)
}

/// Estimate dependence length using per-class ACF (spec §3.2.2).
fn estimate_joint_dependence_length(samples: &[crate::types::TimingSample]) -> usize {
    use crate::types::Class;

    let mut fixed: Vec<f64> = Vec::new();
    let mut random: Vec<f64> = Vec::new();
    for s in samples {
        match s.class {
            Class::Baseline => fixed.push(s.time_ns),
            Class::Sample => random.push(s.time_ns),
        }
    }

    if fixed.len() < 10 || random.len() < 10 {
        return 1;
    }

    let m_f = crate::statistics::estimate_dependence_length(&fixed, 100);
    let m_r = crate::statistics::estimate_dependence_length(&random, 100);

    m_f.max(m_r)
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
        use crate::types::{Class, TimingSample};
        let samples: Vec<TimingSample> = (0..200)
            .map(|i| TimingSample {
                time_ns: 100.0,
                class: if i % 2 == 0 {
                    Class::Baseline
                } else {
                    Class::Sample
                },
            })
            .collect();

        let (ratio, ok) = check_stationarity_windowed(&samples);
        assert!((ratio - 1.0).abs() < 1e-10);
        assert!(ok);

        // Strong drift
        let mut drifting_samples = samples.clone();
        for sample in drifting_samples.iter_mut().take(200).skip(100) {
            sample.time_ns = 200.0;
        }
        let (ratio, ok) = check_stationarity_windowed(&drifting_samples);
        assert!(ratio > 1.5);
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
