//! Timer resolution check.
//!
//! This check detects when the timer resolution is too coarse relative to
//! the operation being measured, which leads to unreliable statistics.
//!
//! On ARM (aarch64), the virtual timer runs at ~24 MHz (~41ns resolution),
//! not at CPU frequency. For operations faster than the timer resolution,
//! most measurements will be 0 or 1 tick, making statistical analysis meaningless.
//!
//! **Severity**: Mixed
//!
//! - `InsufficientResolution`: ResultUndermining - measurements are too quantized
//! - `HighQuantization`: Informational - some quantization but still useful

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::result::{PreflightCategory, PreflightSeverity, PreflightWarningInfo};

/// Warning from the resolution check.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub enum ResolutionWarning {
    /// Timer resolution is too coarse for the operation being measured.
    ///
    /// **Severity**: ResultUndermining
    ///
    /// The statistical analysis will be unreliable because most measurements
    /// are quantized to the same few values.
    InsufficientResolution {
        /// Number of unique timing values observed.
        unique_values: usize,
        /// Total number of samples.
        total_samples: usize,
        /// Fraction of samples that were exactly zero.
        zero_fraction: f64,
        /// Estimated timer resolution in nanoseconds.
        timer_resolution_ns: f64,
    },

    /// Many samples have identical timing values.
    ///
    /// **Severity**: Informational
    ///
    /// This may indicate quantization effects from coarse timer resolution,
    /// but the statistical analysis is still valid.
    HighQuantization {
        /// Number of unique timing values observed.
        unique_values: usize,
        /// Total number of samples.
        total_samples: usize,
    },
}

impl ResolutionWarning {
    /// Check if this warning undermines result confidence.
    pub fn is_result_undermining(&self) -> bool {
        matches!(self, ResolutionWarning::InsufficientResolution { .. })
    }

    /// Get the severity of this warning.
    pub fn severity(&self) -> PreflightSeverity {
        match self {
            ResolutionWarning::InsufficientResolution { .. } => {
                PreflightSeverity::ResultUndermining
            }
            ResolutionWarning::HighQuantization { .. } => PreflightSeverity::Informational,
        }
    }

    /// Get a human-readable description of the warning.
    pub fn description(&self) -> String {
        match self {
            ResolutionWarning::InsufficientResolution {
                unique_values,
                total_samples,
                zero_fraction,
                timer_resolution_ns,
            } => {
                alloc::format!(
                    "Timer resolution (~{:.0}ns) is too coarse for this operation. \
                     Only {} unique values in {} samples ({:.0}% are zero).",
                    timer_resolution_ns,
                    unique_values,
                    total_samples,
                    zero_fraction * 100.0
                )
            }
            ResolutionWarning::HighQuantization {
                unique_values,
                total_samples,
            } => {
                alloc::format!(
                    "High quantization: only {} unique values in {} samples. \
                     Timer resolution may be affecting measurement quality.",
                    unique_values,
                    total_samples
                )
            }
        }
    }

    /// Get guidance for addressing this warning.
    pub fn guidance(&self) -> Option<String> {
        match self {
            ResolutionWarning::InsufficientResolution { .. } => Some(
                "Consider: (1) measuring multiple iterations per sample, \
                 (2) using a more complex operation, or \
                 (3) running with `sudo` to enable kperf (macOS) or perf_event (Linux) \
                 for ~1ns resolution."
                    .into(),
            ),
            ResolutionWarning::HighQuantization { .. } => Some(
                "Consider running with `sudo` to enable PMU-based timing \
                 for better resolution."
                    .into(),
            ),
        }
    }

    /// Convert to a PreflightWarningInfo.
    pub fn to_warning_info(&self) -> PreflightWarningInfo {
        match self.guidance() {
            Some(guidance) => PreflightWarningInfo::with_guidance(
                PreflightCategory::Resolution,
                self.severity(),
                self.description(),
                guidance,
            ),
            None => PreflightWarningInfo::new(
                PreflightCategory::Resolution,
                self.severity(),
                self.description(),
            ),
        }
    }
}

/// Minimum unique values expected per 1000 samples for reliable analysis.
const MIN_UNIQUE_PER_1000: usize = 20;

/// Fraction of zero values that triggers critical warning.
const CRITICAL_ZERO_FRACTION: f64 = 0.5;

/// Perform resolution check on timing samples.
///
/// Detects when timer resolution is too coarse by checking:
/// 1. How many unique timing values exist
/// 2. What fraction of samples are exactly zero
///
/// # Arguments
///
/// * `samples` - Timing samples in nanoseconds
/// * `timer_resolution_ns` - Estimated timer resolution (e.g., from cycles_per_ns)
///
/// # Returns
///
/// A warning if resolution issues are detected, None otherwise.
pub fn resolution_check(samples: &[f64], timer_resolution_ns: f64) -> Option<ResolutionWarning> {
    if samples.len() < 100 {
        return None; // Not enough samples to assess
    }

    // Count unique values (with small tolerance for floating point)
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let mut unique_count = 1;
    let mut last_value = sorted[0];
    for &val in &sorted[1..] {
        // Consider values different if they differ by more than 0.1ns
        if (val - last_value).abs() > 0.1 {
            unique_count += 1;
            last_value = val;
        }
    }

    // Count zeros
    let zero_count = samples.iter().filter(|&&x| x.abs() < 0.1).count();
    let zero_fraction = zero_count as f64 / samples.len() as f64;

    // Check for critical issue: very few unique values AND many zeros
    let expected_unique =
        (samples.len() as f64 / 1000.0 * MIN_UNIQUE_PER_1000 as f64).max(10.0) as usize;

    if unique_count < expected_unique && zero_fraction > CRITICAL_ZERO_FRACTION {
        return Some(ResolutionWarning::InsufficientResolution {
            unique_values: unique_count,
            total_samples: samples.len(),
            zero_fraction,
            timer_resolution_ns,
        });
    }

    // Check for high quantization (less severe)
    if unique_count < expected_unique / 2 {
        return Some(ResolutionWarning::HighQuantization {
            unique_values: unique_count,
            total_samples: samples.len(),
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_resolution() {
        // Simulated ARM-style data: mostly zeros with occasional 41ns ticks
        let mut samples = alloc::vec![0.0; 800];
        samples.extend(alloc::vec![41.0; 150]);
        samples.extend(alloc::vec![82.0; 50]);

        let result = resolution_check(&samples, 41.0);
        assert!(result.is_some(), "Should detect insufficient resolution");
        assert!(
            result.as_ref().unwrap().is_result_undermining(),
            "Should be result-undermining"
        );
        assert_eq!(
            result.as_ref().unwrap().severity(),
            PreflightSeverity::ResultUndermining
        );
    }

    #[test]
    fn test_severity() {
        let insufficient = ResolutionWarning::InsufficientResolution {
            unique_values: 3,
            total_samples: 1000,
            zero_fraction: 0.8,
            timer_resolution_ns: 41.0,
        };
        assert_eq!(
            insufficient.severity(),
            PreflightSeverity::ResultUndermining
        );
        assert!(insufficient.is_result_undermining());

        let high_quant = ResolutionWarning::HighQuantization {
            unique_values: 5,
            total_samples: 1000,
        };
        assert_eq!(high_quant.severity(), PreflightSeverity::Informational);
        assert!(!high_quant.is_result_undermining());
    }
}
