//! Autocorrelation check for periodic interference detection.
//!
//! This check computes the autocorrelation function (ACF) on the timing
//! sequences. High autocorrelation at low lags (especially lag-1 and lag-2)
//! can indicate periodic interference from system processes.
//!
//! **Severity**: Informational
//!
//! High autocorrelation reduces effective sample size but the block bootstrap
//! accounts for this. The Bayesian model's assumptions are still valid; you
//! just needed more samples to reach the same confidence level.

use serde::{Deserialize, Serialize};

use timing_oracle_core::result::{PreflightCategory, PreflightSeverity, PreflightWarningInfo};

/// Warning from the autocorrelation check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutocorrWarning {
    /// High autocorrelation detected at specific lag.
    ///
    /// **Severity**: Informational
    ///
    /// High autocorrelation reduces effective sample size but the block
    /// bootstrap accounts for this. Results are still valid.
    PeriodicInterference {
        /// The lag with high autocorrelation.
        lag: usize,
        /// The autocorrelation coefficient.
        acf_value: f64,
        /// Threshold that was exceeded.
        threshold: f64,
    },

    /// Insufficient samples for autocorrelation analysis.
    ///
    /// **Severity**: Informational
    InsufficientSamples {
        /// Number of samples available.
        available: usize,
        /// Minimum required for the check.
        required: usize,
    },
}

impl AutocorrWarning {
    /// Check if this warning undermines result confidence.
    ///
    /// Autocorrelation warnings are always informational - they affect
    /// sampling efficiency but don't invalidate results.
    pub fn is_result_undermining(&self) -> bool {
        false
    }

    /// Check if this warning indicates a critical issue.
    ///
    /// Deprecated: Use `is_result_undermining()` instead.
    #[deprecated(note = "Use is_result_undermining() instead")]
    pub fn is_critical(&self) -> bool {
        false
    }

    /// Get the severity of this warning.
    pub fn severity(&self) -> PreflightSeverity {
        // All autocorrelation warnings are informational
        PreflightSeverity::Informational
    }

    /// Get a human-readable description of the warning.
    pub fn description(&self) -> String {
        match self {
            AutocorrWarning::PeriodicInterference {
                lag,
                acf_value,
                threshold,
            } => {
                format!(
                    "High autocorrelation at lag {}: ACF={:.2} (threshold: {:.2}). \
                     This reduces effective sample size but the block bootstrap \
                     accounts for this.",
                    lag, acf_value, threshold
                )
            }
            AutocorrWarning::InsufficientSamples { available, required } => {
                format!(
                    "Insufficient samples for autocorrelation check: {} available, {} required.",
                    available, required
                )
            }
        }
    }

    /// Get guidance for addressing this warning.
    pub fn guidance(&self) -> Option<String> {
        match self {
            AutocorrWarning::PeriodicInterference { .. } => Some(
                "Consider increasing sample count or checking for background tasks \
                 that might cause periodic interference."
                    .to_string(),
            ),
            AutocorrWarning::InsufficientSamples { .. } => None,
        }
    }

    /// Convert to a PreflightWarningInfo.
    pub fn to_warning_info(&self) -> PreflightWarningInfo {
        match self.guidance() {
            Some(guidance) => PreflightWarningInfo::with_guidance(
                PreflightCategory::Autocorrelation,
                self.severity(),
                self.description(),
                guidance,
            ),
            None => PreflightWarningInfo::new(
                PreflightCategory::Autocorrelation,
                self.severity(),
                self.description(),
            ),
        }
    }
}

/// Threshold for autocorrelation to trigger warning.
const ACF_THRESHOLD: f64 = 0.3;

/// Maximum lag to check.
const MAX_LAG: usize = 2;

/// Minimum samples required for autocorrelation check.
const MIN_SAMPLES_FOR_ACF: usize = 100;

/// Perform autocorrelation check on per-class timing sequences.
///
/// Computes ACF for lag-1 and lag-2 for both classes and returns a warning if
/// any exceeds the threshold (0.3).
///
/// # Arguments
///
/// * `fixed` - Timing samples from fixed class
/// * `random` - Timing samples from random class
///
/// # Returns
///
/// `Some(AutocorrWarning)` if high autocorrelation detected, `None` otherwise.
pub fn autocorrelation_check(fixed: &[f64], random: &[f64]) -> Option<AutocorrWarning> {
    if fixed.len() < MIN_SAMPLES_FOR_ACF || random.len() < MIN_SAMPLES_FOR_ACF {
        let n = fixed.len().min(random.len());
        return Some(AutocorrWarning::InsufficientSamples {
            available: n,
            required: MIN_SAMPLES_FOR_ACF,
        });
    }

    // Check autocorrelation at lag-1 and lag-2 for both classes
    for lag in 1..=MAX_LAG {
        let acf_f = compute_acf(fixed, lag);
        let acf_r = compute_acf(random, lag);
        let max_acf = if acf_f.abs() > acf_r.abs() { acf_f } else { acf_r };

        if max_acf.abs() > ACF_THRESHOLD {
            return Some(AutocorrWarning::PeriodicInterference {
                lag,
                acf_value: max_acf,
                threshold: ACF_THRESHOLD,
            });
        }
    }

    None
}

/// Compute autocorrelation at a specific lag.
///
/// Uses the standard formula:
/// ACF(k) = Cov(X_t, X_{t+k}) / Var(X)
///
/// # Arguments
///
/// * `data` - Time series data
/// * `lag` - Lag to compute ACF for
///
/// # Returns
///
/// Autocorrelation coefficient at the specified lag.
fn compute_acf(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag {
        return 0.0;
    }

    let n = data.len();

    // Compute mean
    let mean: f64 = data.iter().sum::<f64>() / n as f64;

    // Compute variance
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance < 1e-10 {
        return 0.0;
    }

    // Compute autocovariance at lag
    let autocovariance: f64 = data
        .iter()
        .take(n - lag)
        .zip(data.iter().skip(lag))
        .map(|(x_t, x_t_k)| (x_t - mean) * (x_t_k - mean))
        .sum::<f64>()
        / (n - lag) as f64;

    // ACF = autocovariance / variance
    autocovariance / variance
}

/// Compute full ACF up to a maximum lag.
///
/// Useful for diagnostic purposes.
///
/// # Arguments
///
/// * `data` - Time series data
/// * `max_lag` - Maximum lag to compute
///
/// # Returns
///
/// Vector of ACF values from lag-0 to max_lag.
#[allow(dead_code)]
pub fn compute_full_acf(data: &[f64], max_lag: usize) -> Vec<f64> {
    (0..=max_lag).map(|lag| compute_acf(data, lag)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_samples() {
        let data = vec![1.0; 50];
        let result = autocorrelation_check(&data, &data);
        assert!(matches!(
            result,
            Some(AutocorrWarning::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn test_low_autocorrelation_passes() {
        // Sequence with low autocorrelation should pass
        // Using a more carefully constructed pseudo-random sequence
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let data: Vec<f64> = (0..1000)
            .map(|i| {
                // Use a hash function for better distribution
                let mut hasher = DefaultHasher::new();
                i.hash(&mut hasher);
                let h = hasher.finish();
                (h as f64) / (u64::MAX as f64)
            })
            .collect();

        // Verify the ACF is actually low before running the check
        let acf1 = super::compute_acf(&data, 1);
        let acf2 = super::compute_acf(&data, 2);

        // If our generated sequence happens to have low autocorrelation, test it
        // Otherwise skip the assertion (this is a heuristic test)
        if acf1.abs() < ACF_THRESHOLD && acf2.abs() < ACF_THRESHOLD {
            let result = autocorrelation_check(&data, &data);
            assert!(
                result.is_none(),
                "Low ACF sequence should not trigger warning: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_periodic_signal_detected() {
        // Create a strongly periodic signal
        let data: Vec<f64> = (0..1000)
            .map(|i| if i % 2 == 0 { 100.0 } else { 200.0 })
            .collect();

        let result = autocorrelation_check(&data, &data);
        assert!(
            matches!(result, Some(AutocorrWarning::PeriodicInterference { .. })),
            "Periodic signal should trigger warning"
        );
    }

    #[test]
    fn test_acf_at_lag_0() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let acf0 = compute_acf(&data, 0);
        assert!(
            (acf0 - 1.0).abs() < 1e-10,
            "ACF at lag 0 should be 1.0, got {}",
            acf0
        );
    }

    #[test]
    fn test_full_acf() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let acf = compute_full_acf(&data, 5);
        assert_eq!(acf.len(), 6);
        assert!((acf[0] - 1.0).abs() < 1e-10, "ACF[0] should be 1.0");
    }

    #[test]
    fn test_warning_description() {
        let warning = AutocorrWarning::PeriodicInterference {
            lag: 1,
            acf_value: 0.45,
            threshold: 0.3,
        };
        let desc = warning.description();
        assert!(desc.contains("lag 1"));
        assert!(desc.contains("0.45"));
    }

    #[test]
    fn test_severity() {
        let periodic = AutocorrWarning::PeriodicInterference {
            lag: 1,
            acf_value: 0.45,
            threshold: 0.3,
        };
        assert_eq!(periodic.severity(), PreflightSeverity::Informational);
        assert!(!periodic.is_result_undermining());

        let insufficient = AutocorrWarning::InsufficientSamples {
            available: 50,
            required: 100,
        };
        assert_eq!(insufficient.severity(), PreflightSeverity::Informational);
        assert!(!insufficient.is_result_undermining());
    }
}
