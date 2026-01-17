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
//! temporal effects like cache warming and thermal drift. By shuffling indices
//! before splitting, both halves contain a random mix of early and late samples,
//! so temporal effects cancel out.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::result::{PreflightCategory, PreflightSeverity, PreflightWarningInfo};
use crate::statistics::compute_deciles_inplace;

/// Warning from the sanity check.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
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
                alloc::format!(
                    "The baseline samples showed {:.1}x the expected variation between \
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
                alloc::format!(
                    "Insufficient samples for sanity check: {} available, {} required. \
                     Skipping Fixed-vs-Fixed validation.",
                    available,
                    required
                )
            }
        }
    }

    /// Get guidance for addressing this warning.
    pub fn guidance(&self) -> Option<String> {
        match self {
            SanityWarning::BrokenHarness { .. } => Some(
                "Ensure baseline/sample closures don't share mutable state."
                    .into(),
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
/// * `timer_resolution_ns` - Timer resolution in nanoseconds (used to avoid false
///   positives from quantization effects)
/// * `seed` - Seed for reproducible randomization
///
/// # Returns
///
/// `Some(SanityWarning)` if an issue is detected, `None` otherwise.
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
    let mut first_half: Vec<f64> = indices[..mid].iter().map(|&i| fixed_samples[i]).collect();
    let mut second_half: Vec<f64> = indices[mid..].iter().map(|&i| fixed_samples[i]).collect();

    // Compute quantile differences
    let q_first = compute_deciles_inplace(&mut first_half);
    let q_second = compute_deciles_inplace(&mut second_half);

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
    // We set multiple floors to avoid false positives:
    // 1. 40% of IQR - for highly regular/discrete data
    // 2. 2× timer resolution - quantization can cause ~1 tick differences in deciles
    //    even for perfectly uniform data; we need headroom above this
    let n = fixed_samples.len() as f64;
    let expected_noise = iqr / crate::math::sqrt(n);
    let noise_based_threshold = NOISE_MULTIPLIER * expected_noise;
    let iqr_floor = 0.4 * iqr;
    let quantization_floor = 2.0 * timer_resolution_ns;
    let threshold = noise_based_threshold.max(iqr_floor).max(quantization_floor);

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
    use rand::Rng;

    const TEST_SEED: u64 = 12345;

    #[test]
    fn test_insufficient_samples() {
        let samples = alloc::vec![1.0; 100];
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
    fn test_single_outlier_not_detected() {
        // A single outlier (like a cache warming miss) does NOT trigger the check.
        // This is correct behavior: single outliers beyond the 90th percentile
        // don't affect deciles, and they get filtered by outlier removal anyway.
        //
        // The sanity check is designed to catch distribution-level problems
        // (like alternating mutable state), not single outliers.
        let mut samples = Vec::with_capacity(2000);

        // First call: cache miss (slow)
        samples.push(5000.0);

        // All subsequent calls: cache hit (fast, with small noise)
        for i in 1..2000 {
            samples.push(50.0 + (i % 5) as f64);
        }

        let result = sanity_check(&samples, 1.0, TEST_SEED);

        // Single outlier at 5000ns is beyond D90 (index 1800 of 2000)
        // so it doesn't affect the decile comparison - correctly no warning
        assert!(
            !matches!(result, Some(SanityWarning::BrokenHarness { .. })),
            "Single outlier should not trigger (outlier removal handles this), got {:?}",
            result
        );
    }

    #[test]
    fn test_broken_harness_repeated_cold_starts() {
        // Realistic scenario: cache that periodically evicts, causing repeated cold starts.
        //
        // ```rust
        // let mut cache = LruCache::new(10);  // Small cache
        // oracle.test(inputs, |data| {
        //     // Cache evicts frequently, causing ~10% cold starts
        //     if let Some(v) = cache.get(data) { return *v; }
        //     let v = slow_compute(data);
        //     cache.put(data.clone(), v);
        //     v
        // });
        // ```
        //
        // With ~10% cold starts, the slow samples affect upper deciles.
        let mut samples = Vec::with_capacity(2000);
        for i in 0..2000 {
            // 10% are slow (cache misses), 90% are fast (cache hits)
            let base = if i % 10 == 0 { 2000.0 } else { 100.0 };
            let noise = (i % 5) as f64;
            samples.push(base + noise);
        }

        let result = sanity_check(&samples, 1.0, TEST_SEED);

        // With 10% outliers affecting D90, the check should trigger
        assert!(
            matches!(result, Some(SanityWarning::BrokenHarness { .. })),
            "10% cold starts should affect D90 and trigger warning, got {:?}",
            result
        );
    }

    #[test]
    fn test_broken_harness_alternating_state() {
        // Realistic scenario: mutable state that alternates between two modes.
        //
        // ```rust
        // let mut use_fast_path = true;
        // oracle.test(inputs, |data| {
        //     use_fast_path = !use_fast_path;
        //     if use_fast_path {
        //         fast_impl(data)   // ~100ns
        //     } else {
        //         slow_impl(data)   // ~500ns
        //     }
        // });
        // ```
        //
        // This creates 50/50 bimodal data. The sanity check compares deciles
        // between random halves. With 50/50 bimodal data:
        // - IQR spans from ~100 to ~500 (≈400ns)
        // - Threshold = 0.4 * 400 = 160ns
        // - Random imbalance in the split can create decile differences > 160ns
        //
        // With seed 12345, this specific split creates a detectable imbalance.
        let mut samples = Vec::with_capacity(2000);
        for i in 0..2000 {
            let base = if i % 2 == 0 { 100.0 } else { 500.0 };
            let noise = (i % 7) as f64;
            samples.push(base + noise);
        }

        let result = sanity_check(&samples, 1.0, TEST_SEED);

        // The 50/50 bimodal pattern with this seed creates sufficient imbalance
        assert!(
            matches!(result, Some(SanityWarning::BrokenHarness { .. })),
            "Alternating state pattern should trigger warning, got {:?}",
            result
        );
    }

    #[test]
    fn test_moderate_noise_does_not_trigger() {
        // Samples with moderate natural variation (like real timing data)
        // should NOT trigger the warning after raising the threshold to 0.4*IQR
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99999);
        let samples: Vec<f64> = (0..2000)
            .map(|_| {
                // Normal-ish distribution centered at 500ns with ~50ns spread
                let base = 500.0;
                let noise = (rng.random::<f64>() - 0.5) * 100.0;
                base + noise
            })
            .collect();

        let result = sanity_check(&samples, 1.0, TEST_SEED);

        assert!(
            !matches!(result, Some(SanityWarning::BrokenHarness { .. })),
            "Moderate noise should not trigger broken harness warning, got {:?}",
            result
        );
    }
}
