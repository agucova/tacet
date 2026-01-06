//! Quantile computation using Type 2 quantiles (inverse empirical CDF with averaging).
//!
//! This module implements Type 2 quantiles following Hyndman & Fan (1996),
//! which is more appropriate for bootstrap-based inference than interpolating estimators.
//!
//! **Type 2 formula** (for sorted sample x of size n at probability p):
//! ```text
//! h = n * p + 0.5
//! q = (x[floor(h)] + x[ceil(h)]) / 2
//! ```
//!
//! This uses the inverse of the empirical distribution function with averaging
//! at discontinuities. See spec §2.1 for theoretical justification.
//!
//! # Reference
//!
//! Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages."
//! The American Statistician 50(4):361–365.

use crate::constants::DECILES;
use crate::types::Vector9;

/// Compute a single quantile from a mutable slice using Type 2 quantiles.
///
/// Uses Type 2 quantile definition (Hyndman & Fan 1996):
/// ```text
/// h = n * p + 0.5
/// q = (x[floor(h)] + x[ceil(h)]) / 2
/// ```
///
/// Uses `select_nth_unstable()` for O(n) expected time complexity.
/// The slice is partially reordered as a side effect.
///
/// # Arguments
///
/// * `data` - Mutable slice of measurements (will be partially reordered)
/// * `p` - Quantile probability in [0, 1]
///
/// # Returns
///
/// The quantile value at probability `p`.
///
/// # Panics
///
/// Panics if `data` is empty or if `p` is outside [0, 1].
pub fn compute_quantile(data: &mut [f64], p: f64) -> f64 {
    assert!(!data.is_empty(), "Cannot compute quantile of empty slice");
    assert!(
        (0.0..=1.0).contains(&p),
        "Quantile probability must be in [0, 1]"
    );

    let n = data.len();

    // Handle edge cases
    if n == 1 {
        return data[0];
    }

    // Type 2 quantile: h = n * p + 0.5
    let h = n as f64 * p + 0.5;

    // Convert to 0-based indices with bounds checking
    let floor_idx = (h.floor() as usize).saturating_sub(1).min(n - 1);
    let ceil_idx = (h.ceil() as usize).saturating_sub(1).min(n - 1);

    if floor_idx == ceil_idx {
        // Single index case - just select that element
        let (_, &mut val, _) = data.select_nth_unstable_by(floor_idx, |a, b| a.total_cmp(b));
        return val;
    }

    // Need both elements - select the larger index first, then the smaller
    // This works because select_nth_unstable guarantees the nth element is in place
    // and all elements before it are <= the nth element
    let (_, &mut ceil_val, _) = data.select_nth_unstable_by(ceil_idx, |a, b| a.total_cmp(b));
    let (_, &mut floor_val, _) = data.select_nth_unstable_by(floor_idx, |a, b| a.total_cmp(b));

    // Average the two values for Type 2
    (floor_val + ceil_val) / 2.0
}

/// Compute all 9 deciles [0.1, 0.2, ..., 0.9] using Type 2 quantiles.
///
/// Uses Type 2 quantile definition (Hyndman & Fan 1996) with averaging
/// at discontinuities. Returns a Vector9 containing the quantile values
/// at each decile. The input slice is cloned and sorted once for efficiency.
///
/// # Arguments
///
/// * `data` - Slice of timing measurements
///
/// # Returns
///
/// A `Vector9` with decile values at positions 0-8 corresponding to
/// quantiles 0.1-0.9.
///
/// # Panics
///
/// Panics if `data` is empty.
pub fn compute_deciles(data: &[f64]) -> Vector9 {
    assert!(!data.is_empty(), "Cannot compute deciles of empty slice");

    // Sort once, then compute all quantiles from sorted data - O(n log n) total
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    compute_deciles_sorted(&sorted)
}

/// Compute all 9 deciles using Type 2 quantiles with unstable sort.
///
/// Uses unstable sort (O(n log n)) followed by Type 2 quantile computation.
/// This is semantically identical to `compute_deciles()` but uses unstable
/// sort which is faster when stability is not needed.
///
/// # Arguments
///
/// * `data` - Slice of timing measurements
///
/// # Returns
///
/// A `Vector9` with decile values at positions 0-8 corresponding to
/// quantiles 0.1-0.9.
///
/// # Panics
///
/// Panics if `data` is empty.
///
/// # Note
///
/// Despite the name, this function uses O(n log n) sorting, not O(n) selection.
/// The name is historical; true multi-quantile selection was not significantly
/// faster in practice due to overhead.
pub fn compute_deciles_fast(data: &[f64]) -> Vector9 {
    assert!(!data.is_empty(), "Cannot compute deciles of empty slice");

    // Use unstable sort which is faster than stable sort (don't need stability)
    let mut working = data.to_vec();
    working.sort_unstable_by(|a, b| a.total_cmp(b));

    compute_deciles_sorted(&working)
}

/// Compute deciles with a reusable buffer to avoid allocations.
///
/// This is an optimization for hot loops where `compute_deciles_fast` is called
/// repeatedly. By reusing the same buffer, we avoid repeated allocations.
///
/// # Arguments
///
/// * `data` - Input data slice
/// * `buffer` - Reusable working buffer (will be resized as needed)
///
/// # Returns
///
/// A `Vector9` with decile values.
pub fn compute_deciles_with_buffer(data: &[f64], buffer: &mut Vec<f64>) -> Vector9 {
    assert!(!data.is_empty(), "Cannot compute deciles of empty slice");

    // Reuse buffer, avoiding allocation if it's already large enough
    buffer.clear();
    buffer.extend_from_slice(data);
    buffer.sort_unstable_by(|a, b| a.total_cmp(b));

    compute_deciles_sorted(buffer)
}

/// Compute deciles by sorting a mutable slice in-place.
///
/// This is the most efficient version for hot loops where you already have
/// the data in a mutable buffer and don't need to preserve the unsorted order.
/// It sorts the buffer in-place (no allocations) and reads deciles directly.
///
/// # Arguments
///
/// * `data` - Mutable slice that will be sorted in-place
///
/// # Returns
///
/// A `Vector9` with decile values.
///
/// # Note
///
/// After this call, `data` will be sorted in ascending order.
pub fn compute_deciles_inplace(data: &mut [f64]) -> Vector9 {
    assert!(!data.is_empty(), "Cannot compute deciles of empty slice");

    // Sort in-place (no allocation)
    data.sort_unstable_by(|a, b| a.total_cmp(b));

    // Read deciles from sorted data
    compute_deciles_sorted(data)
}

/// Compute all 9 deciles from pre-sorted timing measurements using Type 2 quantiles.
///
/// This is an optimization when you need to compute deciles multiple times
/// on the same data - sort once and reuse.
///
/// Uses Type 2 quantile definition (Hyndman & Fan 1996):
/// ```text
/// h = n * p + 0.5
/// q = (x[floor(h)] + x[ceil(h)]) / 2
/// ```
///
/// Type 2 uses the inverse of the empirical distribution function with averaging
/// at discontinuities, which is more appropriate for bootstrap-based inference
/// than interpolating estimators like R-7.
///
/// # Arguments
///
/// * `sorted` - Slice of timing measurements that MUST be sorted in ascending order
///
/// # Returns
///
/// A `Vector9` with decile values at positions 0-8 corresponding to
/// quantiles 0.1-0.9.
///
/// # Panics
///
/// Panics if `sorted` is empty.
///
/// # Safety
///
/// The caller must ensure the data is sorted. No verification is performed.
pub fn compute_deciles_sorted(sorted: &[f64]) -> Vector9 {
    assert!(!sorted.is_empty(), "Cannot compute deciles of empty slice");

    let n = sorted.len();
    let mut result = Vector9::zeros();

    for (i, &p) in DECILES.iter().enumerate() {
        // Type 2 quantile: h = n * p + 0.5
        // Then average x[floor(h)] and x[ceil(h)]
        // Indices are 1-based in the formula, so we adjust for 0-based
        let h = n as f64 * p + 0.5;

        // Convert to 0-based indices with bounds checking
        let floor_idx = (h.floor() as usize).saturating_sub(1).min(n - 1);
        let ceil_idx = (h.ceil() as usize).saturating_sub(1).min(n - 1);

        // Average the two values (handles the case where they're the same)
        result[i] = (sorted[floor_idx] + sorted[ceil_idx]) / 2.0;
    }

    result
}

/// Compute mid-distribution quantiles for discrete data (spec §2.4).
///
/// Mid-distribution quantiles handle ties correctly by using:
/// ```text
/// F_mid(x) = F(x) - ½p(x)
/// q̂_mid = F⁻¹_mid(k)
/// ```
///
/// where F(x) is the empirical CDF and p(x) is the probability mass at x.
///
/// This is recommended for discrete/heavily-tied data where standard quantile
/// estimators may produce biased results.
///
/// # Reference
///
/// Geraci, M. & Jones, M. C. (2015). "Improved transformation-based quantile
/// regression." Canadian Journal of Statistics 43(1):118-132.
pub fn compute_midquantile(data: &mut [f64], p: f64) -> f64 {
    assert!(!data.is_empty(), "Cannot compute quantile of empty slice");
    assert!(
        (0.0..=1.0).contains(&p),
        "Quantile probability must be in [0, 1]"
    );

    let n = data.len();

    // Sort the data
    data.sort_by(|a, b| a.total_cmp(b));

    // Handle edge cases
    if n == 1 {
        return data[0];
    }

    // Compute mid-distribution CDF values for each unique value
    // F_mid(x) = F(x) - ½p(x) = (rank + 0.5 * count_at_x) / n - 0.5 * count_at_x / n
    //          = (rank - 0.5 * count_at_x + 0.5 * count_at_x) / n
    //          = (rank_start + 0.5 * count_at_x) / n
    // where rank_start is the 0-based index of first occurrence

    // Find where in the mid-CDF our target probability p falls
    let mut i = 0;
    while i < n {
        let value = data[i];

        // Count how many times this value appears
        let mut count = 1;
        while i + count < n && data[i + count] == value {
            count += 1;
        }

        // Mid-CDF at this value: (i + count/2 + 0.5) / n
        // This is the "center" of the step in the CDF
        let f_mid = (i as f64 + count as f64 / 2.0) / n as f64;

        // If p <= f_mid, return this value
        if p <= f_mid {
            return value;
        }

        i += count;
    }

    // If we get here, p is very close to 1, return the last value
    data[n - 1]
}

/// Compute all 9 deciles using mid-distribution quantiles for discrete data.
///
/// This is the discrete-mode equivalent of `compute_deciles()`, using
/// mid-distribution quantiles that properly handle tied values.
///
/// # Arguments
///
/// * `data` - Slice of timing measurements (will be cloned and sorted)
///
/// # Returns
///
/// A `Vector9` with mid-distribution decile values.
pub fn compute_midquantile_deciles(data: &[f64]) -> Vector9 {
    assert!(!data.is_empty(), "Cannot compute deciles of empty slice");

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    compute_midquantile_deciles_sorted(&sorted)
}

/// Compute all 9 deciles using mid-distribution quantiles from pre-sorted data.
///
/// More efficient when you already have sorted data.
pub fn compute_midquantile_deciles_sorted(sorted: &[f64]) -> Vector9 {
    assert!(!sorted.is_empty(), "Cannot compute deciles of empty slice");

    let n = sorted.len();
    let mut result = Vector9::zeros();

    for (i, &p) in DECILES.iter().enumerate() {
        // For each decile, compute mid-distribution quantile
        result[i] = midquantile_from_sorted(sorted, n, p);
    }

    result
}

/// Helper to compute mid-distribution quantile from sorted data.
fn midquantile_from_sorted(sorted: &[f64], n: usize, p: f64) -> f64 {
    if n == 1 {
        return sorted[0];
    }

    let mut i = 0;
    while i < n {
        let value = sorted[i];

        // Count how many times this value appears
        let mut count = 1;
        while i + count < n && sorted[i + count] == value {
            count += 1;
        }

        // Mid-CDF at this value
        let f_mid = (i as f64 + count as f64 / 2.0) / n as f64;

        if p <= f_mid {
            return value;
        }

        i += count;
    }

    sorted[n - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_quantile_median() {
        // With Type 2 quantiles: h = n * p + 0.5 = 5 * 0.5 + 0.5 = 3.0
        // floor(3) = 3, ceil(3) = 3 (both point to index 2 in 0-based)
        // So median = x[2] = 3.0
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let median = compute_quantile(&mut data, 0.5);
        assert!((median - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_quantile_extremes() {
        // Type 2 at p=0: h = n * 0 + 0.5 = 0.5
        // floor(0.5) = 0, ceil(0.5) = 1, both as 1-based then -1 for 0-based
        // floor_idx = 0 - 1 = -1 -> saturating_sub gives 0
        // ceil_idx = 1 - 1 = 0
        // So min = (x[0] + x[0]) / 2 = 1.0
        //
        // Type 2 at p=1: h = n * 1 + 0.5 = 5.5
        // floor(5.5) = 5, ceil(5.5) = 6, as 1-based indices
        // floor_idx = 5 - 1 = 4 (clamped to n-1 = 4)
        // ceil_idx = 6 - 1 = 5 (clamped to n-1 = 4)
        // So max = (x[4] + x[4]) / 2 = 5.0
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let min = compute_quantile(&mut data.clone(), 0.0);
        let max = compute_quantile(&mut data, 1.0);
        assert!((min - 1.0).abs() < 1e-10, "min was {}", min);
        assert!((max - 5.0).abs() < 1e-10, "max was {}", max);
    }

    #[test]
    fn test_compute_deciles_sorted() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let deciles = compute_deciles(&data);

        // Check that deciles are monotonically increasing
        for i in 1..9 {
            assert!(deciles[i] >= deciles[i - 1]);
        }
    }

    #[test]
    fn test_compute_deciles_fast_matches_sort() {
        // Test on sequential data
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let deciles_sort = compute_deciles(&data);
        let deciles_fast = compute_deciles_fast(&data);

        for i in 0..9 {
            let diff = (deciles_sort[i] - deciles_fast[i]).abs();
            assert!(
                diff < 1e-10,
                "Decile {} differs: sort={}, fast={}, diff={}",
                i,
                deciles_sort[i],
                deciles_fast[i],
                diff
            );
        }
    }

    #[test]
    fn test_compute_deciles_fast_random_data() {
        // Test on random-ish data
        let data: Vec<f64> = vec![
            3.7, 1.2, 9.5, 2.1, 7.3, 4.8, 6.2, 8.9, 1.5, 5.4, 2.7, 9.1, 3.3, 6.8, 4.5, 7.9, 2.4,
            8.3, 5.7, 1.9,
        ];
        let deciles_sort = compute_deciles(&data);
        let deciles_fast = compute_deciles_fast(&data);

        for i in 0..9 {
            let diff = (deciles_sort[i] - deciles_fast[i]).abs();
            assert!(
                diff < 1e-10,
                "Decile {} differs: sort={}, fast={}, diff={}",
                i,
                deciles_sort[i],
                deciles_fast[i],
                diff
            );
        }

        // Verify monotonicity
        for i in 1..9 {
            assert!(deciles_fast[i] >= deciles_fast[i - 1]);
        }
    }

    #[test]
    fn test_compute_deciles_fast_small_data() {
        // Test edge case: small dataset
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let deciles_sort = compute_deciles(&data);
        let deciles_fast = compute_deciles_fast(&data);

        for i in 0..9 {
            let diff = (deciles_sort[i] - deciles_fast[i]).abs();
            assert!(
                diff < 1e-10,
                "Decile {} differs: sort={}, fast={}, diff={}",
                i,
                deciles_sort[i],
                deciles_fast[i],
                diff
            );
        }
    }

    #[test]
    fn test_compute_deciles_fast_large_data() {
        // Test on larger dataset (typical bootstrap size)
        let data: Vec<f64> = (0..20000).map(|x| (x as f64 * 1.234) % 1000.0).collect();
        let deciles_sort = compute_deciles(&data);
        let deciles_fast = compute_deciles_fast(&data);

        for i in 0..9 {
            let diff = (deciles_sort[i] - deciles_fast[i]).abs();
            assert!(
                diff < 1e-8, // Slightly relaxed tolerance for large data
                "Decile {} differs: sort={}, fast={}, diff={}",
                i,
                deciles_sort[i],
                deciles_fast[i],
                diff
            );
        }
    }

    #[test]
    #[should_panic(expected = "Cannot compute quantile of empty slice")]
    fn test_empty_slice_panics() {
        let mut data: Vec<f64> = vec![];
        compute_quantile(&mut data, 0.5);
    }

    #[test]
    #[should_panic(expected = "Cannot compute deciles of empty slice")]
    fn test_compute_deciles_fast_empty_panics() {
        let data: Vec<f64> = vec![];
        compute_deciles_fast(&data);
    }

    // ========== Mid-distribution Quantile Tests ==========

    #[test]
    fn test_midquantile_no_ties() {
        // Without ties, mid-quantile should behave similarly to regular quantiles
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let median = compute_midquantile(&mut data, 0.5);

        // For 10 elements, mid-CDF at 5.0 is (4 + 0.5) / 10 = 0.45
        // mid-CDF at 6.0 is (5 + 0.5) / 10 = 0.55
        // So 0.5 should return 6.0
        assert!((median - 6.0).abs() < 1e-10, "Median was {}", median);
    }

    #[test]
    fn test_midquantile_with_ties() {
        // Data with ties - this is where mid-quantile shines
        let mut data = vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0];
        let median = compute_midquantile(&mut data, 0.5);

        // Value 1.0 appears 3 times: mid-CDF = (0 + 1.5) / 10 = 0.15
        // Value 2.0 appears 2 times: mid-CDF = (3 + 1) / 10 = 0.40
        // Value 3.0 appears 4 times: mid-CDF = (5 + 2) / 10 = 0.70
        // So 0.5 falls in the 3.0 range
        assert!((median - 3.0).abs() < 1e-10, "Median was {}", median);
    }

    #[test]
    fn test_midquantile_all_same() {
        // All values the same
        let mut data = vec![42.0; 100];
        let median = compute_midquantile(&mut data, 0.5);
        assert!((median - 42.0).abs() < 1e-10);

        let q10 = compute_midquantile(&mut data, 0.1);
        assert!((q10 - 42.0).abs() < 1e-10);

        let q90 = compute_midquantile(&mut data, 0.9);
        assert!((q90 - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_midquantile_deciles_discrete_data() {
        // Simulate discrete timer data with only a few unique values
        let data: Vec<f64> = (0..1000)
            .map(|i| ((i % 5) * 10) as f64) // Values: 0, 10, 20, 30, 40
            .collect();

        let deciles = compute_midquantile_deciles(&data);

        // Check monotonicity
        for i in 1..9 {
            assert!(
                deciles[i] >= deciles[i - 1],
                "Deciles should be monotonic: d[{}]={} < d[{}]={}",
                i - 1,
                deciles[i - 1],
                i,
                deciles[i]
            );
        }
    }

    #[test]
    fn test_midquantile_single_element() {
        let mut data = vec![42.0];
        let result = compute_midquantile(&mut data, 0.5);
        assert!((result - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_midquantile_deciles_matches_sorted() {
        let data: Vec<f64> = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0];

        let deciles1 = compute_midquantile_deciles(&data);

        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let deciles2 = compute_midquantile_deciles_sorted(&sorted);

        for i in 0..9 {
            assert!(
                (deciles1[i] - deciles2[i]).abs() < 1e-10,
                "Decile {} mismatch: {} vs {}",
                i,
                deciles1[i],
                deciles2[i]
            );
        }
    }
}
