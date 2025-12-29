//! Pooled symmetric outlier filtering.
//!
//! Implements outlier filtering that:
//! 1. Pools all samples from both classes
//! 2. Computes a threshold at a given percentile
//! 3. Applies the same threshold symmetrically to both classes
//!
//! This approach prevents class-specific filtering that could
//! artificially create or mask timing differences.

/// Statistics about outlier filtering.
#[derive(Debug, Clone, Copy)]
pub struct OutlierStats {
    /// Total samples before filtering.
    pub total_samples: usize,
    /// Samples remaining after filtering.
    pub retained_samples: usize,
    /// Number of outliers removed.
    pub outliers_removed: usize,
    /// Fraction of samples that were outliers (0.0 to 1.0).
    pub outlier_fraction: f64,
    /// The threshold value used for filtering.
    pub threshold: u64,
}

impl OutlierStats {
    /// Create stats for when no filtering was applied.
    pub fn no_filtering(total_samples: usize) -> Self {
        Self {
            total_samples,
            retained_samples: total_samples,
            outliers_removed: 0,
            outlier_fraction: 0.0,
            threshold: u64::MAX,
        }
    }
}

/// Filter outliers from both sample sets using pooled symmetric thresholding.
///
/// This function:
/// 1. Combines all samples from both classes
/// 2. Computes the threshold at the given percentile
/// 3. Removes samples above the threshold from both classes
///
/// # Arguments
///
/// * `fixed` - Samples from the Fixed class (in cycles)
/// * `random` - Samples from the Random class (in cycles)
/// * `percentile` - Percentile for threshold (e.g., 0.999 for 99.9th percentile)
///
/// # Returns
///
/// A tuple of (filtered_fixed, filtered_random, stats).
///
/// # Note
///
/// If `percentile >= 1.0`, no filtering is applied and all samples are returned.
pub fn filter_outliers(
    fixed: &[u64],
    random: &[u64],
    percentile: f64,
) -> (Vec<u64>, Vec<u64>, OutlierStats) {
    let total_samples = fixed.len() + random.len();

    // Skip filtering if percentile is 1.0 or higher
    if percentile >= 1.0 {
        return (
            fixed.to_vec(),
            random.to_vec(),
            OutlierStats::no_filtering(total_samples),
        );
    }

    // Pool all samples
    let mut pooled: Vec<u64> = Vec::with_capacity(total_samples);
    pooled.extend_from_slice(fixed);
    pooled.extend_from_slice(random);

    // Compute threshold at percentile
    let threshold = compute_percentile(&mut pooled, percentile);

    // Filter both classes with the same threshold
    let filtered_fixed: Vec<u64> = fixed.iter().copied().filter(|&x| x <= threshold).collect();
    let filtered_random: Vec<u64> = random.iter().copied().filter(|&x| x <= threshold).collect();

    let retained = filtered_fixed.len() + filtered_random.len();
    let removed = total_samples - retained;

    let stats = OutlierStats {
        total_samples,
        retained_samples: retained,
        outliers_removed: removed,
        outlier_fraction: removed as f64 / total_samples as f64,
        threshold,
    };

    (filtered_fixed, filtered_random, stats)
}

/// Compute the value at a given percentile.
///
/// Uses linear interpolation between adjacent values.
///
/// # Arguments
///
/// * `data` - Mutable slice (will be sorted in place)
/// * `percentile` - Value between 0.0 and 1.0
///
/// # Returns
///
/// The value at the given percentile.
fn compute_percentile(data: &mut [u64], percentile: f64) -> u64 {
    if data.is_empty() {
        return 0;
    }

    if data.len() == 1 {
        return data[0];
    }

    // Sort the data
    data.sort_unstable();

    // Clamp percentile to valid range
    let p = percentile.clamp(0.0, 1.0);

    // Compute index (using linear interpolation for fractional indices)
    let n = data.len();
    let idx = p * (n - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;

    if lower == upper {
        data[lower]
    } else {
        // Linear interpolation
        let frac = idx - lower as f64;
        let lower_val = data[lower] as f64;
        let upper_val = data[upper] as f64;
        (lower_val + frac * (upper_val - lower_val)).round() as u64
    }
}

/// Compute the threshold for outlier filtering without modifying the input.
///
/// This is a non-destructive version that clones the data first.
pub fn compute_outlier_threshold(fixed: &[u64], random: &[u64], percentile: f64) -> u64 {
    if percentile >= 1.0 {
        return u64::MAX;
    }

    let mut pooled: Vec<u64> = Vec::with_capacity(fixed.len() + random.len());
    pooled.extend_from_slice(fixed);
    pooled.extend_from_slice(random);

    compute_percentile(&mut pooled, percentile)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_percentile() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        assert_eq!(compute_percentile(&mut data.clone(), 0.0), 1);
        // 50th percentile with linear interpolation is between 5 and 6
        let median = compute_percentile(&mut data.clone(), 0.5);
        assert!(median == 5 || median == 6, "Median should be 5 or 6, got {}", median);
        assert_eq!(compute_percentile(&mut data.clone(), 1.0), 10);
    }

    #[test]
    fn test_filter_outliers_basic() {
        let fixed = vec![100, 110, 105, 1000, 108]; // 1000 is an outlier
        let random = vec![102, 107, 103, 109, 2000]; // 2000 is an outlier

        let (f, r, stats) = filter_outliers(&fixed, &random, 0.8);

        // With 10 samples, 80th percentile should filter out high values
        assert!(stats.outliers_removed > 0);
        assert_eq!(f.len() + r.len(), stats.retained_samples);
    }

    #[test]
    fn test_filter_outliers_no_filtering() {
        let fixed = vec![100, 110, 105];
        let random = vec![102, 107, 103];

        let (f, r, stats) = filter_outliers(&fixed, &random, 1.0);

        assert_eq!(f.len(), 3);
        assert_eq!(r.len(), 3);
        assert_eq!(stats.outliers_removed, 0);
        assert_eq!(stats.outlier_fraction, 0.0);
    }

    #[test]
    fn test_symmetric_filtering() {
        // Both classes should use the same threshold
        let fixed = vec![100, 200, 300];
        let random = vec![150, 250, 350];

        let threshold = compute_outlier_threshold(&fixed, &random, 0.5);

        // Threshold should be computed from pooled data
        // Pooled sorted: [100, 150, 200, 250, 300, 350]
        // 50th percentile of 6 elements is around index 2.5
        assert!(threshold >= 200 && threshold <= 250);
    }
}
