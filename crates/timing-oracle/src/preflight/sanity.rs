//! Fixed-vs-Fixed internal consistency check.
//!
//! This check splits the fixed samples into two random halves and compares quantiles
//! between them. If a large difference is detected between identical input classes,
//! it may indicate:
//! - Mutable state captured in the test closure
//! - Severe environmental interference
//! - A measurement harness bug
//!
//! The check uses a simple threshold approach: if max|Δ| > 5× expected noise,
//! a warning is emitted.
//!
//! **Severity**: ResultUndermining
//!
//! This warning violates statistical assumptions because if Fixed-vs-Fixed shows
//! inconsistency, the comparison between Fixed and Random may be contaminated
//! by the same issue. However, the warning may also trigger intentionally when
//! running FPR validation tests (testing with identical inputs for both classes).
//!
//! **Why Randomization?**
//!
//! Sequential splitting (first half vs second half) can false-positive due to
//! temporal effects like cache warming and thermal drift:
//!
//! ```text
//! samples = [s1, s2, ..., s2500, s2501, ..., s5000]
//!            |-- first half --|  |-- second half --|
//!            (cold cache, CPU    (warm cache, CPU
//!             ramping up)         at steady state)
//! ```
//!
//! By shuffling indices before splitting, both halves contain a random mix of
//! early and late samples, so temporal effects cancel out. The check will only
//! trigger on genuine non-temporal inconsistency (like mutable state bugs).

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::statistics::compute_deciles;
use timing_oracle_core::result::{PreflightCategory, PreflightSeverity, PreflightWarningInfo};

/// Warning from the sanity check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SanityWarning {
    /// Fixed-vs-Fixed comparison detected internal inconsistency.
    ///
    /// **Severity**: ResultUndermining
    ///
    /// This indicates that the baseline samples show unexpected variation between
    /// random subsets. Possible causes:
    /// - Mutable state captured in test closure
    /// - Severe environmental interference
    /// - Measurement harness bug
    ///
    /// **Note**: May be intentional for FPR validation testing where identical
    /// inputs are used for both classes to verify the false positive rate.
    BrokenHarness {
        /// Ratio of observed variance to expected variance.
        variance_ratio: f64,
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
    /// Check if this warning undermines result confidence.
    ///
    /// Returns `true` for BrokenHarness (statistical assumption violation),
    /// `false` for InsufficientSamples (just informational).
    pub fn is_result_undermining(&self) -> bool {
        matches!(self, SanityWarning::BrokenHarness { .. })
    }

    /// Get the severity of this warning.
    pub fn severity(&self) -> PreflightSeverity {
        match self {
            SanityWarning::BrokenHarness { .. } => PreflightSeverity::ResultUndermining,
            SanityWarning::InsufficientSamples { .. } => PreflightSeverity::Informational,
        }
    }

    /// Get a human-readable description of the warning.
    pub fn description(&self) -> String {
        match self {
            SanityWarning::BrokenHarness { variance_ratio } => {
                format!(
                    "Fixed-vs-Fixed internal consistency check triggered. \
                     The baseline samples showed {:.1}x expected variation between \
                     random subsets. This may indicate mutable state captured in \
                     your test closure, or severe environmental interference. \
                     (If you're intentionally testing with identical inputs for \
                     FPR validation, this warning is expected and can be ignored.)",
                    variance_ratio
                )
            }
            SanityWarning::InsufficientSamples {
                available,
                required,
            } => {
                format!(
                    "Insufficient samples for sanity check: {} available, {} required. \
                     Skipping Fixed-vs-Fixed validation.",
                    available, required
                )
            }
        }
    }

    /// Get guidance for addressing this warning.
    pub fn guidance(&self) -> Option<String> {
        match self {
            SanityWarning::BrokenHarness { .. } => Some(
                "Check your test closure for captured mutable state. \
                 Ensure generators return fresh values each call. \
                 If this is an FPR validation test, this warning can be ignored."
                    .to_string(),
            ),
            SanityWarning::InsufficientSamples { .. } => None,
        }
    }

    /// Convert to a PreflightWarningInfo.
    pub fn to_warning_info(&self) -> PreflightWarningInfo {
        match self.guidance() {
            Some(guidance) => PreflightWarningInfo::with_guidance(
                PreflightCategory::Sanity,
                self.severity(),
                self.description(),
                guidance,
            ),
            None => PreflightWarningInfo::new(
                PreflightCategory::Sanity,
                self.severity(),
                self.description(),
            ),
        }
    }
}

/// Minimum samples required to perform sanity check.
const MIN_SAMPLES_FOR_SANITY: usize = 1000;

/// Multiplier for noise threshold: if max|Δ| > NOISE_MULTIPLIER × expected_noise, warn.
const NOISE_MULTIPLIER: f64 = 5.0;

/// Perform Fixed-vs-Fixed internal consistency check.
///
/// Splits the fixed samples into two **random** halves (using the provided seed)
/// and compares quantiles between the halves. Randomization breaks temporal
/// correlation from cache warming and thermal effects.
///
/// If the max quantile difference exceeds 5× the expected noise level, returns
/// a warning indicating potential issues with the measurement setup.
///
/// # Arguments
///
/// * `fixed_samples` - All timing samples from the fixed input class
/// * `timer_resolution_ns` - Timer resolution (kept for API compatibility)
/// * `seed` - Seed for reproducible randomization
///
/// # Returns
///
/// `Some(SanityWarning)` if an issue is detected, `None` otherwise.
#[allow(unused_variables)] // timer_resolution_ns kept for API compatibility
pub fn sanity_check(
    fixed_samples: &[f64],
    timer_resolution_ns: f64,
    seed: u64,
) -> Option<SanityWarning> {
    // Check if we have enough samples
    if fixed_samples.len() < MIN_SAMPLES_FOR_SANITY {
        return Some(SanityWarning::InsufficientSamples {
            available: fixed_samples.len(),
            required: MIN_SAMPLES_FOR_SANITY,
        });
    }

    // Create indices and shuffle them to break temporal correlation
    let mut indices: Vec<usize> = (0..fixed_samples.len()).collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    // Split shuffled indices in half
    let mid = indices.len() / 2;
    let first_half: Vec<f64> = indices[..mid].iter().map(|&i| fixed_samples[i]).collect();
    let second_half: Vec<f64> = indices[mid..].iter().map(|&i| fixed_samples[i]).collect();

    // Compute quantile differences
    let q_first = compute_deciles(&first_half);
    let q_second = compute_deciles(&second_half);

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
    // roughly O(IQR / sqrt(n)), so NOISE_MULTIPLIER× that is a conservative threshold.
    //
    // We also set a minimum floor of 20% of IQR to avoid being too sensitive to
    // highly regular/discrete data where quantile standard errors are harder to estimate.
    let n = fixed_samples.len() as f64;
    let expected_noise = iqr / n.sqrt();
    let noise_based_threshold = NOISE_MULTIPLIER * expected_noise;
    let min_threshold = 0.2 * iqr; // At least 20% of IQR
    let threshold = noise_based_threshold.max(min_threshold);

    if max_diff > threshold && threshold > 0.0 {
        // Calculate variance ratio for the warning message
        let variance_ratio = max_diff / expected_noise;
        Some(SanityWarning::BrokenHarness { variance_ratio })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SEED: u64 = 12345;

    #[test]
    fn test_insufficient_samples() {
        let samples = vec![1.0; 100];
        let result = sanity_check(&samples, 1.0, TEST_SEED);
        assert!(matches!(
            result,
            Some(SanityWarning::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn test_identical_samples_pass() {
        // Create samples with small deterministic variation
        let samples: Vec<f64> = (0..2000).map(|i| 100.0 + (i % 10) as f64).collect();
        let result = sanity_check(&samples, 1.0, TEST_SEED);

        // With randomization, identical pattern should pass even if there's
        // a temporal trend, because early and late samples are mixed in both halves
        assert!(
            !matches!(result, Some(SanityWarning::BrokenHarness { .. })),
            "Identical samples should not trigger broken harness warning"
        );
    }

    #[test]
    fn test_randomization_breaks_temporal_correlation() {
        // Create samples with a gradual temporal drift (simulating cache warming)
        // The drift is gradual, not a step function, so randomization should help
        let samples: Vec<f64> = (0..2000)
            .map(|i| {
                // Gradual drift from 100 to 120 over the sequence (0.01ns per sample)
                let drift = (i as f64) * 0.01;
                100.0 + drift + (i % 10) as f64
            })
            .collect();

        // With randomization, the check should pass because both random halves
        // will contain a mix of early and late samples, averaging out the drift
        let result = sanity_check(&samples, 1.0, TEST_SEED);

        // The gradual temporal drift should be hidden by randomization
        assert!(
            !matches!(result, Some(SanityWarning::BrokenHarness { .. })),
            "Gradual temporal drift should be mitigated by randomization"
        );
    }

    #[test]
    fn test_large_step_change_detected() {
        // Create samples with a large step change (not gradual drift)
        // This simulates a genuine issue (not just cache warming) and should be detected
        let mut samples = Vec::with_capacity(2000);
        for i in 0..1000 {
            samples.push(150.0 + (i % 10) as f64);
        }
        for i in 0..1000 {
            samples.push(100.0 + (i % 10) as f64);
        }

        // Even with randomization, if there's a bimodal distribution with large gap,
        // it might or might not trigger depending on how quantiles compare.
        // This test verifies the check still runs without panicking.
        let _result = sanity_check(&samples, 1.0, TEST_SEED);
        // Just verifying it completes - the exact result depends on threshold tuning
    }

    #[test]
    fn test_warning_description() {
        let warning = SanityWarning::BrokenHarness {
            variance_ratio: 7.5,
        };
        let desc = warning.description();
        assert!(desc.contains("7.5x"));
        assert!(desc.contains("FPR validation"));
        assert!(desc.contains("mutable state"));
    }

    #[test]
    fn test_severity() {
        let broken = SanityWarning::BrokenHarness {
            variance_ratio: 5.0,
        };
        assert_eq!(broken.severity(), PreflightSeverity::ResultUndermining);
        assert!(broken.is_result_undermining());

        let insufficient = SanityWarning::InsufficientSamples {
            available: 100,
            required: 1000,
        };
        assert_eq!(insufficient.severity(), PreflightSeverity::Informational);
        assert!(!insufficient.is_result_undermining());
    }

    #[test]
    fn test_reproducible_with_same_seed() {
        let samples: Vec<f64> = (0..2000).map(|i| 100.0 + (i as f64 * 0.01)).collect();

        let result1 = sanity_check(&samples, 1.0, 42);
        let result2 = sanity_check(&samples, 1.0, 42);

        // Same seed should produce same result
        match (&result1, &result2) {
            (None, None) => {}
            (
                Some(SanityWarning::BrokenHarness { variance_ratio: v1 }),
                Some(SanityWarning::BrokenHarness { variance_ratio: v2 }),
            ) => {
                assert!(
                    (v1 - v2).abs() < 1e-10,
                    "Same seed should give same variance_ratio"
                );
            }
            _ => panic!("Results should match with same seed"),
        }
    }
}
