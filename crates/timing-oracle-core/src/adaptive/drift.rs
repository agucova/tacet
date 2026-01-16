//! Condition drift detection for quality gate 6 (spec Section 2.6).
//!
//! Detects when measurement conditions change between calibration and the
//! adaptive loop, which can invalidate the covariance estimate and cause
//! false positives or negatives.

use alloc::string::String;
use alloc::vec::Vec;

use crate::statistics::StatsSnapshot;

/// Snapshot of measurement statistics at a point in time.
///
/// Captures per-class statistics for comparison between calibration
/// and post-test phases.
#[derive(Debug, Clone)]
pub struct CalibrationSnapshot {
    /// Statistics for baseline class.
    pub baseline: StatsSnapshot,
    /// Statistics for sample class.
    pub sample: StatsSnapshot,
}

impl CalibrationSnapshot {
    /// Create a new calibration snapshot from per-class statistics.
    pub fn new(baseline: StatsSnapshot, sample: StatsSnapshot) -> Self {
        Self { baseline, sample }
    }
}

/// Detected drift between calibration and post-test statistics.
///
/// Contains per-class drift metrics that indicate how much measurement
/// conditions changed during the test.
#[derive(Debug, Clone, Copy)]
pub struct ConditionDrift {
    /// Variance ratio for baseline class: post_variance / cal_variance.
    /// Values far from 1.0 indicate variance changed significantly.
    pub variance_ratio_baseline: f64,

    /// Variance ratio for sample class.
    pub variance_ratio_sample: f64,

    /// Absolute change in lag-1 autocorrelation for baseline class.
    pub autocorr_change_baseline: f64,

    /// Absolute change in lag-1 autocorrelation for sample class.
    pub autocorr_change_sample: f64,

    /// Mean drift in standard deviations for baseline class:
    /// |post_mean - cal_mean| / cal_std_dev
    pub mean_drift_baseline: f64,

    /// Mean drift in standard deviations for sample class.
    pub mean_drift_sample: f64,
}

impl ConditionDrift {
    /// Compute drift between calibration and post-test snapshots.
    ///
    /// # Arguments
    ///
    /// * `cal` - Statistics snapshot from calibration phase
    /// * `post` - Statistics snapshot from full test run
    ///
    /// # Returns
    ///
    /// A `ConditionDrift` struct with per-class drift metrics.
    pub fn compute(cal: &CalibrationSnapshot, post: &CalibrationSnapshot) -> Self {
        Self {
            variance_ratio_baseline: compute_variance_ratio(
                cal.baseline.variance,
                post.baseline.variance,
            ),
            variance_ratio_sample: compute_variance_ratio(cal.sample.variance, post.sample.variance),
            autocorr_change_baseline: libm::fabs(
                post.baseline.autocorr_lag1 - cal.baseline.autocorr_lag1,
            ),
            autocorr_change_sample: libm::fabs(
                post.sample.autocorr_lag1 - cal.sample.autocorr_lag1,
            ),
            mean_drift_baseline: compute_mean_drift(
                cal.baseline.mean,
                post.baseline.mean,
                cal.baseline.variance,
            ),
            mean_drift_sample: compute_mean_drift(
                cal.sample.mean,
                post.sample.mean,
                cal.sample.variance,
            ),
        }
    }

    /// Check if drift exceeds thresholds.
    ///
    /// Returns `true` if any drift metric exceeds its threshold, indicating
    /// measurement conditions changed significantly.
    pub fn is_significant(&self, thresholds: &DriftThresholds) -> bool {
        // Variance drift check (either direction)
        if self.variance_ratio_baseline > thresholds.max_variance_ratio
            || self.variance_ratio_baseline < thresholds.min_variance_ratio
        {
            return true;
        }
        if self.variance_ratio_sample > thresholds.max_variance_ratio
            || self.variance_ratio_sample < thresholds.min_variance_ratio
        {
            return true;
        }

        // Autocorrelation change check
        if self.autocorr_change_baseline > thresholds.max_autocorr_change {
            return true;
        }
        if self.autocorr_change_sample > thresholds.max_autocorr_change {
            return true;
        }

        // Mean drift check
        if self.mean_drift_baseline > thresholds.max_mean_drift_sigmas {
            return true;
        }
        if self.mean_drift_sample > thresholds.max_mean_drift_sigmas {
            return true;
        }

        false
    }

    /// Get a human-readable description of the most significant drift.
    pub fn description(&self, thresholds: &DriftThresholds) -> String {
        let mut issues = Vec::new();

        if self.variance_ratio_baseline > thresholds.max_variance_ratio {
            issues.push(alloc::format!(
                "baseline variance increased {:.1}x",
                self.variance_ratio_baseline
            ));
        } else if self.variance_ratio_baseline < thresholds.min_variance_ratio {
            issues.push(alloc::format!(
                "baseline variance decreased to {:.1}x",
                self.variance_ratio_baseline
            ));
        }

        if self.variance_ratio_sample > thresholds.max_variance_ratio {
            issues.push(alloc::format!(
                "sample variance increased {:.1}x",
                self.variance_ratio_sample
            ));
        } else if self.variance_ratio_sample < thresholds.min_variance_ratio {
            issues.push(alloc::format!(
                "sample variance decreased to {:.1}x",
                self.variance_ratio_sample
            ));
        }

        if self.autocorr_change_baseline > thresholds.max_autocorr_change {
            issues.push(alloc::format!(
                "baseline autocorrelation changed by {:.2}",
                self.autocorr_change_baseline
            ));
        }

        if self.autocorr_change_sample > thresholds.max_autocorr_change {
            issues.push(alloc::format!(
                "sample autocorrelation changed by {:.2}",
                self.autocorr_change_sample
            ));
        }

        if self.mean_drift_baseline > thresholds.max_mean_drift_sigmas {
            issues.push(alloc::format!(
                "baseline mean drifted {:.1}\u{03C3}",
                self.mean_drift_baseline
            ));
        }

        if self.mean_drift_sample > thresholds.max_mean_drift_sigmas {
            issues.push(alloc::format!(
                "sample mean drifted {:.1}\u{03C3}",
                self.mean_drift_sample
            ));
        }

        if issues.is_empty() {
            String::from("no significant drift")
        } else {
            issues.join(", ")
        }
    }
}

/// Thresholds for condition drift detection.
///
/// These control when Gate 6 (ConditionsChanged) triggers.
#[derive(Debug, Clone, Copy)]
pub struct DriftThresholds {
    /// Maximum allowed variance ratio (post/cal). Default: 2.0
    pub max_variance_ratio: f64,

    /// Minimum allowed variance ratio (post/cal). Default: 0.5
    pub min_variance_ratio: f64,

    /// Maximum allowed change in lag-1 autocorrelation. Default: 0.3
    pub max_autocorr_change: f64,

    /// Maximum allowed mean drift in standard deviations. Default: 3.0
    pub max_mean_drift_sigmas: f64,
}

impl Default for DriftThresholds {
    fn default() -> Self {
        Self {
            max_variance_ratio: 2.0,
            min_variance_ratio: 0.5,
            max_autocorr_change: 0.3,
            max_mean_drift_sigmas: 3.0,
        }
    }
}

/// Compute variance ratio, handling edge cases.
fn compute_variance_ratio(cal_variance: f64, post_variance: f64) -> f64 {
    if cal_variance < 1e-15 {
        // If calibration variance was essentially zero, any change is infinite
        // But if post is also zero, call it 1.0 (no change)
        if post_variance < 1e-15 {
            1.0
        } else {
            f64::INFINITY
        }
    } else {
        post_variance / cal_variance
    }
}

/// Compute mean drift in standard deviations.
fn compute_mean_drift(cal_mean: f64, post_mean: f64, cal_variance: f64) -> f64 {
    let cal_std = libm::sqrt(cal_variance);
    if cal_std < 1e-15 {
        // If calibration had zero variance, any mean change is significant
        // But if means are equal, call it 0.0
        if libm::fabs(post_mean - cal_mean) < 1e-15 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        libm::fabs(post_mean - cal_mean) / cal_std
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(mean: f64, variance: f64, autocorr: f64) -> StatsSnapshot {
        StatsSnapshot {
            mean,
            variance,
            autocorr_lag1: autocorr,
            count: 1000,
        }
    }

    #[test]
    fn test_no_drift() {
        let cal = CalibrationSnapshot::new(
            make_snapshot(100.0, 25.0, 0.3),
            make_snapshot(100.0, 25.0, 0.3),
        );
        let post = CalibrationSnapshot::new(
            make_snapshot(100.0, 25.0, 0.3),
            make_snapshot(100.0, 25.0, 0.3),
        );

        let drift = ConditionDrift::compute(&cal, &post);
        let thresholds = DriftThresholds::default();

        assert!(!drift.is_significant(&thresholds));
        assert!(libm::fabs(drift.variance_ratio_baseline - 1.0) < 1e-10);
        assert!(libm::fabs(drift.variance_ratio_sample - 1.0) < 1e-10);
    }

    #[test]
    fn test_variance_increase_detected() {
        let cal = CalibrationSnapshot::new(
            make_snapshot(100.0, 25.0, 0.3),
            make_snapshot(100.0, 25.0, 0.3),
        );
        let post = CalibrationSnapshot::new(
            make_snapshot(100.0, 75.0, 0.3), // 3x variance increase
            make_snapshot(100.0, 25.0, 0.3),
        );

        let drift = ConditionDrift::compute(&cal, &post);
        let thresholds = DriftThresholds::default();

        assert!(drift.is_significant(&thresholds));
        assert!(libm::fabs(drift.variance_ratio_baseline - 3.0) < 1e-10);
    }

    #[test]
    fn test_variance_decrease_detected() {
        let cal = CalibrationSnapshot::new(
            make_snapshot(100.0, 100.0, 0.3),
            make_snapshot(100.0, 100.0, 0.3),
        );
        let post = CalibrationSnapshot::new(
            make_snapshot(100.0, 25.0, 0.3), // 4x variance decrease
            make_snapshot(100.0, 100.0, 0.3),
        );

        let drift = ConditionDrift::compute(&cal, &post);
        let thresholds = DriftThresholds::default();

        assert!(drift.is_significant(&thresholds));
        assert!(libm::fabs(drift.variance_ratio_baseline - 0.25) < 1e-10);
    }

    #[test]
    fn test_autocorr_change_detected() {
        let cal = CalibrationSnapshot::new(
            make_snapshot(100.0, 25.0, 0.1),
            make_snapshot(100.0, 25.0, 0.1),
        );
        let post = CalibrationSnapshot::new(
            make_snapshot(100.0, 25.0, 0.6), // 0.5 autocorr change
            make_snapshot(100.0, 25.0, 0.1),
        );

        let drift = ConditionDrift::compute(&cal, &post);
        let thresholds = DriftThresholds::default();

        assert!(drift.is_significant(&thresholds));
        assert!(libm::fabs(drift.autocorr_change_baseline - 0.5) < 1e-10);
    }

    #[test]
    fn test_mean_drift_detected() {
        let cal = CalibrationSnapshot::new(
            make_snapshot(100.0, 25.0, 0.3), // std = 5
            make_snapshot(100.0, 25.0, 0.3),
        );
        let post = CalibrationSnapshot::new(
            make_snapshot(120.0, 25.0, 0.3), // 20/5 = 4 sigma drift
            make_snapshot(100.0, 25.0, 0.3),
        );

        let drift = ConditionDrift::compute(&cal, &post);
        let thresholds = DriftThresholds::default();

        assert!(drift.is_significant(&thresholds));
        assert!(libm::fabs(drift.mean_drift_baseline - 4.0) < 1e-10);
    }

    #[test]
    fn test_small_drift_allowed() {
        let cal = CalibrationSnapshot::new(
            make_snapshot(100.0, 25.0, 0.3),
            make_snapshot(100.0, 25.0, 0.3),
        );
        let post = CalibrationSnapshot::new(
            make_snapshot(102.0, 30.0, 0.35), // Small changes within thresholds
            make_snapshot(98.0, 20.0, 0.25),
        );

        let drift = ConditionDrift::compute(&cal, &post);
        let thresholds = DriftThresholds::default();

        assert!(!drift.is_significant(&thresholds));
    }

    #[test]
    fn test_description() {
        let cal = CalibrationSnapshot::new(
            make_snapshot(100.0, 25.0, 0.1),
            make_snapshot(100.0, 25.0, 0.1),
        );
        let post = CalibrationSnapshot::new(
            make_snapshot(100.0, 75.0, 0.1), // 3x variance
            make_snapshot(100.0, 25.0, 0.1),
        );

        let drift = ConditionDrift::compute(&cal, &post);
        let thresholds = DriftThresholds::default();
        let desc = drift.description(&thresholds);

        assert!(desc.contains("baseline variance increased"));
    }

    #[test]
    fn test_custom_thresholds() {
        let cal = CalibrationSnapshot::new(
            make_snapshot(100.0, 25.0, 0.3),
            make_snapshot(100.0, 25.0, 0.3),
        );
        let post = CalibrationSnapshot::new(
            make_snapshot(100.0, 60.0, 0.3), // 2.4x variance
            make_snapshot(100.0, 25.0, 0.3),
        );

        let drift = ConditionDrift::compute(&cal, &post);

        // Default threshold (2.0) would trigger
        let default_thresholds = DriftThresholds::default();
        assert!(drift.is_significant(&default_thresholds));

        // Relaxed threshold (3.0) would not trigger
        let relaxed_thresholds = DriftThresholds {
            max_variance_ratio: 3.0,
            ..Default::default()
        };
        assert!(!drift.is_significant(&relaxed_thresholds));
    }
}
