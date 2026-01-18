//! Discrete mode detection utilities.
//!
//! This module provides functions to detect when discrete mode should be used,
//! based on the uniqueness ratio of timing measurements.

use hashbrown::HashSet;

/// Compute minimum uniqueness ratio across both classes.
///
/// Returns the minimum of (unique_baseline/n, unique_sample/n).
/// Values are quantized to 0.001ns buckets before counting.
///
/// Discrete mode should be triggered when this returns < 0.10 (spec §3.7).
///
/// # Arguments
///
/// * `baseline` - Baseline class timing values in nanoseconds
/// * `sample` - Sample class timing values in nanoseconds
///
/// # Returns
///
/// The minimum uniqueness ratio, in range [0.0, 1.0].
pub fn compute_min_uniqueness_ratio(baseline: &[f64], sample: &[f64]) -> f64 {
    fn count_unique(data: &[f64]) -> usize {
        // Quantize to 0.001ns to avoid floating-point comparison issues
        let unique: HashSet<i64> = data.iter().map(|&v| (v * 1000.0) as i64).collect();
        unique.len()
    }

    let n_baseline = baseline.len().max(1);
    let n_sample = sample.len().max(1);

    let ratio_baseline = count_unique(baseline) as f64 / n_baseline as f64;
    let ratio_sample = count_unique(sample) as f64 / n_sample as f64;

    ratio_baseline.min(ratio_sample)
}

/// Discrete mode threshold (spec §3.7).
///
/// When the minimum uniqueness ratio falls below this threshold,
/// implementations SHOULD use discrete mode (m-out-of-n bootstrap,
/// mid-distribution quantiles).
pub const DISCRETE_MODE_THRESHOLD: f64 = 0.10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_identical() {
        let baseline = vec![100.0; 1000];
        let sample = vec![100.0; 1000];
        let ratio = compute_min_uniqueness_ratio(&baseline, &sample);
        assert!(
            ratio < 0.01,
            "All identical values should give very low ratio"
        );
    }

    #[test]
    fn test_all_unique() {
        let baseline: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let sample: Vec<f64> = (1000..2000).map(|i| i as f64).collect();
        let ratio = compute_min_uniqueness_ratio(&baseline, &sample);
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "All unique values should give ratio ~1.0"
        );
    }

    #[test]
    fn test_boundary_10_percent() {
        // 10% unique values: 100 unique out of 1000
        let mut baseline = vec![0.0; 1000];
        for i in 0..100 {
            baseline[i * 10] = i as f64;
        }
        let sample: Vec<f64> = (0..1000).map(|i| i as f64).collect(); // all unique

        let ratio = compute_min_uniqueness_ratio(&baseline, &sample);
        // Should be exactly at or just above 0.10
        assert!(
            ratio >= 0.09 && ratio <= 0.11,
            "10% unique should be near threshold, got {}",
            ratio
        );
    }

    #[test]
    fn test_empty_slices() {
        let empty: Vec<f64> = vec![];
        let ratio = compute_min_uniqueness_ratio(&empty, &empty);
        // Should handle empty gracefully (returns 0.0 due to 0/1)
        assert!(ratio == 0.0);
    }

    #[test]
    fn test_quantization_precision() {
        // Values that differ by less than 0.001ns should be treated as same
        let baseline = vec![100.0, 100.0001, 100.0002];
        let sample = vec![200.0, 200.0001, 200.0002];
        let ratio = compute_min_uniqueness_ratio(&baseline, &sample);
        // All should be quantized to same bucket
        assert!(
            ratio < 0.5,
            "Sub-0.001ns differences should be treated as identical"
        );
    }
}
