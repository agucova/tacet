//! Fixed-vs-Fixed sanity check.
//!
//! This check splits the fixed samples in half and runs the analysis
//! between the two halves. If a "leak" is detected between identical
//! input classes, it indicates a broken measurement harness or
//! environmental interference.

use serde::{Deserialize, Serialize};

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

/// Threshold for leak probability to trigger warning.
const LEAK_THRESHOLD: f64 = 0.5;

/// Perform Fixed-vs-Fixed sanity check.
///
/// Splits the fixed samples in half and runs analysis between the halves.
/// If a leak is detected, returns a warning indicating a broken harness.
///
/// # Arguments
///
/// * `fixed_samples` - All timing samples from the fixed input class
///
/// # Returns
///
/// `Some(SanityWarning)` if an issue is detected, `None` otherwise.
pub fn sanity_check(fixed_samples: &[f64]) -> Option<SanityWarning> {
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

    // TODO: Run the actual analysis between the two halves.
    // This requires access to the analysis module which computes
    // quantile differences and Bayesian model comparison.
    //
    // For now, we compute a simplified check using basic statistics.
    let leak_probability = compute_simplified_leak_check(first_half, second_half);

    if leak_probability > LEAK_THRESHOLD {
        Some(SanityWarning::BrokenHarness { leak_probability })
    } else {
        None
    }
}

/// Simplified leak check using basic statistics.
///
/// TODO: Replace with proper quantile-based analysis once the
/// analysis module interface is available.
fn compute_simplified_leak_check(first: &[f64], second: &[f64]) -> f64 {
    if first.is_empty() || second.is_empty() {
        return 0.0;
    }

    // Compute means
    let mean1: f64 = first.iter().sum::<f64>() / first.len() as f64;
    let mean2: f64 = second.iter().sum::<f64>() / second.len() as f64;

    // Compute pooled standard deviation
    let var1: f64 = first.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / first.len() as f64;
    let var2: f64 = second.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / second.len() as f64;

    let pooled_std = ((var1 + var2) / 2.0).sqrt();

    if pooled_std < 1e-10 {
        return 0.0;
    }

    // Compute effect size (Cohen's d)
    let effect_size = (mean1 - mean2).abs() / pooled_std;

    // Convert to approximate "leak probability" using sigmoid
    // Effect size > 0.2 starts indicating potential issues
    let sigmoid = 1.0 / (1.0 + (-10.0 * (effect_size - 0.2)).exp());

    sigmoid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_samples() {
        let samples = vec![1.0; 100];
        let result = sanity_check(&samples);
        assert!(matches!(
            result,
            Some(SanityWarning::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn test_identical_samples_pass() {
        // Identical samples should not trigger a warning
        let samples: Vec<f64> = (0..2000).map(|i| 100.0 + (i % 10) as f64).collect();
        let result = sanity_check(&samples);

        // Should either be None or not be a BrokenHarness warning
        match result {
            None => {}
            Some(SanityWarning::InsufficientSamples { .. }) => {}
            Some(SanityWarning::BrokenHarness { leak_probability }) => {
                assert!(
                    leak_probability < LEAK_THRESHOLD,
                    "Identical samples should not trigger broken harness warning"
                );
            }
        }
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
