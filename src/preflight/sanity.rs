//! Fixed-vs-Fixed sanity check.
//!
//! This check splits the fixed samples in half and compares quantiles
//! between the two halves. If a large difference is detected between
//! identical input classes, it indicates a broken measurement harness
//! or environmental interference.
//!
//! The check uses a simple threshold approach: if max|Δ| > 5× median noise,
//! the harness is likely broken.

use serde::{Deserialize, Serialize};

use crate::statistics::compute_deciles;

/// Warning from the sanity check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SanityWarning {
    /// Fixed-vs-Fixed comparison detected a spurious "leak".
    ///
    /// This is a critical warning indicating the measurement harness
    /// is producing unreliable results.
    BrokenHarness {
        /// Leak probability from Fixed-vs-Fixed comparison.
        leak_probability: f64,
    },

    /// Insufficient samples to perform sanity check.
    InsufficientSamples {
        /// Number of samples available.
        available: usize,
        /// Minimum required for the check.
        required: usize,
    },
}

impl SanityWarning {
    /// Check if this warning indicates a critical issue.
    pub fn is_critical(&self) -> bool {
        matches!(self, SanityWarning::BrokenHarness { .. })
    }

    /// Get a human-readable description of the warning.
    pub fn description(&self) -> String {
        match self {
            SanityWarning::BrokenHarness { leak_probability } => {
                format!(
                    "CRITICAL: Fixed-vs-Fixed comparison detected spurious 'leak' \
                     (probability: {:.1}%). This indicates a broken measurement harness \
                     or significant environmental interference. Results are unreliable.",
                    leak_probability * 100.0
                )
            }
            SanityWarning::InsufficientSamples { available, required } => {
                format!(
                    "Insufficient samples for sanity check: {} available, {} required. \
                     Skipping Fixed-vs-Fixed validation.",
                    available, required
                )
            }
        }
    }
}

/// Minimum samples required to perform sanity check.
const MIN_SAMPLES_FOR_SANITY: usize = 1000;

/// Multiplier for noise threshold: if max|Δ| > NOISE_MULTIPLIER × median_noise, harness is broken.
const NOISE_MULTIPLIER: f64 = 5.0;

/// Perform Fixed-vs-Fixed sanity check.
///
/// Splits the fixed samples in half and compares quantiles between the halves.
/// If the max quantile difference exceeds 5× the median noise level, returns
/// a warning indicating a broken harness.
///
/// This is a simplified check that doesn't require CI gate or Bayesian inference.
/// The idea: identical inputs should produce similar timing distributions. If they
/// don't, the harness is broken.
///
/// # Arguments
///
/// * `fixed_samples` - All timing samples from the fixed input class
///
/// # Returns
///
/// `Some(SanityWarning)` if an issue is detected, `None` otherwise.
#[allow(unused_variables)] // timer_resolution_ns kept for API compatibility
pub fn sanity_check(fixed_samples: &[f64], timer_resolution_ns: f64) -> Option<SanityWarning> {
    // Check if we have enough samples
    if fixed_samples.len() < MIN_SAMPLES_FOR_SANITY {
        return Some(SanityWarning::InsufficientSamples {
            available: fixed_samples.len(),
            required: MIN_SAMPLES_FOR_SANITY,
        });
    }

    // Split samples in half
    let mid = fixed_samples.len() / 2;
    let first_half = &fixed_samples[..mid];
    let second_half = &fixed_samples[mid..];

    // Compute quantile differences
    let q_first = compute_deciles(first_half);
    let q_second = compute_deciles(second_half);

    // Max absolute quantile difference
    let max_diff = (0..9)
        .map(|i| (q_first[i] - q_second[i]).abs())
        .fold(0.0_f64, f64::max);

    // Estimate noise level from IQR of the combined samples
    // IQR is robust to outliers and gives a sense of typical variation
    let mut all_samples: Vec<f64> = fixed_samples.to_vec();
    all_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q25_idx = all_samples.len() / 4;
    let q75_idx = 3 * all_samples.len() / 4;
    let iqr = all_samples[q75_idx] - all_samples[q25_idx];

    // Noise threshold: expect quantile differences to be small relative to IQR
    // For identical distributions split in half, quantile differences should be
    // roughly O(IQR / sqrt(n)), so 5× that is a conservative threshold
    let n = fixed_samples.len() as f64;
    let expected_noise = iqr / n.sqrt();
    let threshold = NOISE_MULTIPLIER * expected_noise;

    if max_diff > threshold && threshold > 0.0 {
        // Convert to a "leak probability" for the warning message
        // This is a rough approximation based on how much we exceeded the threshold
        let leak_probability = (max_diff / threshold).min(1.0);
        Some(SanityWarning::BrokenHarness { leak_probability })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_samples() {
        let samples = vec![1.0; 100];
        let result = sanity_check(&samples, 1.0);
        assert!(matches!(
            result,
            Some(SanityWarning::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn test_identical_samples_pass() {
        // Create samples with small deterministic variation
        let samples: Vec<f64> = (0..2000).map(|i| 100.0 + (i % 10) as f64).collect();
        let result = sanity_check(&samples, 1.0);

        // Identical pattern in both halves should pass
        assert!(
            !matches!(result, Some(SanityWarning::BrokenHarness { .. })),
            "Identical samples should not trigger broken harness warning"
        );
    }

    #[test]
    fn test_warning_description() {
        let warning = SanityWarning::BrokenHarness {
            leak_probability: 0.95,
        };
        let desc = warning.description();
        assert!(desc.contains("CRITICAL"));
        assert!(desc.contains("95.0%"));
    }
}
