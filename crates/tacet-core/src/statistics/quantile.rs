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
//! at discontinuities. See spec §3.1 for theoretical justification.
//!
//! # Input Requirements
//!
//! All input data must be finite (no NaN or infinity values). In debug builds,
//! this is checked via assertions. NaN values would produce meaningless statistical
//! results since `total_cmp` orders them but their presence indicates invalid data.
//!
//! # Reference
//!
//! Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages."
//! The American Statistician 50(4):361–365.

extern crate alloc;

use alloc::vec::Vec;

use crate::constants::DECILES;
use crate::math;
use crate::types::Vector9;

/// Compute Type 2 quantile indices for a decile using integer arithmetic.
///
/// For decile k (1-9), computes floor and ceil indices for:
/// ```text
/// h = n * (k/10) + 0.5 = (n*k + 5) / 10
/// ```
///
/// Returns (floor_idx, ceil_idx) as 0-based indices clamped to [0, n-1].
#[inline]
fn decile_indices(n: usize, k: usize) -> (usize, usize) {
    debug_assert!((1..=9).contains(&k), "decile k must be in 1..=9");
    debug_assert!(n > 0, "n must be positive");
    debug_assert!(
        n <= (usize::MAX - 5) / 9,
        "n too large, would overflow in index computation"
    );

    // h = (n*k + 5) / 10 using integer arithmetic
    let h_numerator = n * k + 5;
    let floor_h = h_numerator / 10; // Integer division = floor
    let has_fraction = !h_numerator.is_multiple_of(10);
    let ceil_h = if has_fraction { floor_h + 1 } else { floor_h };

    // Convert 1-based h to 0-based index, clamped to valid range
    let floor_idx = floor_h.saturating_sub(1).min(n - 1);
    let ceil_idx = ceil_h.saturating_sub(1).min(n - 1);

    (floor_idx, ceil_idx)
}

/// Debug assertion that all values in the slice are finite.
#[inline]
fn debug_assert_finite(data: &[f64]) {
    debug_assert!(
        data.iter().all(|x| x.is_finite()),
        "quantile input must be finite (no NaN or infinity)"
    );
}

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
    debug_assert_finite(data);

    let n = data.len();

    // Handle edge cases
    if n == 1 {
        return data[0];
    }

    // Type 2 quantile: h = n * p + 0.5
    let h = n as f64 * p + 0.5;

    // Convert to 0-based indices with bounds checking
    let floor_idx = (math::floor(h) as usize).saturating_sub(1).min(n - 1);
    let ceil_idx = (math::ceil(h) as usize).saturating_sub(1).min(n - 1);

    let cmp = |a: &f64, b: &f64| a.total_cmp(b);

    if floor_idx == ceil_idx {
        // Single index case - just select that element
        let (_, mid, _) = data.select_nth_unstable_by(floor_idx, cmp);
        return *mid;
    }

    // Need both elements - select the larger index first
    let (_, mid, _) = data.select_nth_unstable_by(ceil_idx, cmp);
    let ceil_val = *mid; // Copy out (borrow ends here under NLL)

    // Select floor only within the left partition - avoids touching ceil position
    let (_, mid, _) = data[..ceil_idx].select_nth_unstable_by(floor_idx, cmp);
    let floor_val = *mid;

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
    debug_assert_finite(data);

    // Sort once, then compute all quantiles from sorted data - O(n log n) total
    // Using unstable sort (faster, and stability doesn't matter for deciles)
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));

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
    debug_assert_finite(data);

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
    debug_assert_finite(data);

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
/// After this call, `data` will be partially reordered (elements at decile
/// positions will be correct, but full ordering is not guaranteed).
pub fn compute_deciles_inplace(data: &mut [f64]) -> Vector9 {
    assert!(!data.is_empty(), "Cannot compute deciles of empty slice");
    debug_assert_finite(data);

    let n = data.len();

    // For small arrays, just sort - the overhead of multi-select isn't worth it.
    // Benchmarks show multi-select wins at n >= 300; threshold of 200 is conservative.
    if n <= 200 {
        data.sort_unstable_by(|a, b| a.total_cmp(b));
        return compute_deciles_sorted(data);
    }

    // Precompute all 9 decile index pairs using integer arithmetic
    // This avoids floating-point rounding issues and guarantees monotonicity
    let mut decile_idx_pairs = [(0usize, 0usize); 9];
    for k in 1..=9 {
        decile_idx_pairs[k - 1] = decile_indices(n, k);
    }

    // Build deduplicated sorted index list for multi-select
    // Using a fixed-size array since we need at most 18 indices
    let mut indices = [0usize; 18];
    let mut num_indices = 0;

    for &(floor_idx, ceil_idx) in &decile_idx_pairs {
        // Add floor index if not already present (indices are monotonically non-decreasing)
        if num_indices == 0 || indices[num_indices - 1] != floor_idx {
            indices[num_indices] = floor_idx;
            num_indices += 1;
        }
        // Add ceil index if different from floor and not already present
        if ceil_idx != floor_idx && (num_indices == 0 || indices[num_indices - 1] != ceil_idx) {
            indices[num_indices] = ceil_idx;
            num_indices += 1;
        }
    }

    // Verify indices are sorted (guaranteed by integer arithmetic, but check in debug)
    debug_assert!(
        indices[..num_indices].windows(2).all(|w| w[0] < w[1]),
        "decile indices must be strictly increasing"
    );

    // Use multi-select to place all required indices in O(n) expected time
    multi_select(data, &indices[..num_indices]);

    // Read the decile values from the correctly-placed elements
    let mut result = Vector9::zeros();
    for (i, &(floor_idx, ceil_idx)) in decile_idx_pairs.iter().enumerate() {
        result[i] = (data[floor_idx] + data[ceil_idx]) / 2.0;
    }

    result
}

/// Multi-select algorithm: place multiple order statistics in their correct positions.
///
/// Uses recursive partitioning with `select_nth_unstable` to achieve O(n) expected
/// time complexity for selecting k order statistics, compared to O(n log n) for sorting.
///
/// After this function returns, for each index i in `indices`, `data[i]` contains
/// the value that would be at position i if the array were fully sorted.
///
/// # Arguments
///
/// * `data` - Mutable slice to partition
/// * `indices` - Sorted slice of indices to place (must be sorted in ascending order)
fn multi_select(data: &mut [f64], indices: &[usize]) {
    if indices.is_empty() {
        return;
    }
    // Allocation-free recursive implementation tracking [lo, hi) bounds
    multi_select_recursive(data, indices, 0, data.len());
}

/// Recursive helper for multi_select that avoids allocations.
///
/// Operates on the subslice `data[lo..hi]` and places elements at absolute indices
/// in `indices` (which must all be in `[lo, hi)`).
///
/// # Arguments
///
/// * `data` - The full data slice
/// * `indices` - Slice of absolute target indices (must be sorted, all in `[lo, hi)`)
/// * `lo` - Start of working region (inclusive)
/// * `hi` - End of working region (exclusive)
fn multi_select_recursive(data: &mut [f64], indices: &[usize], lo: usize, hi: usize) {
    // Base cases: nothing to do if no indices or region has 0-1 elements
    if indices.is_empty() || hi.saturating_sub(lo) <= 1 {
        return;
    }

    // Validate bounds (debug check)
    debug_assert!(indices[0] >= lo);
    debug_assert!(indices[indices.len() - 1] < hi);

    // Single index: just select it
    if indices.len() == 1 {
        let target = indices[0];
        let rel_idx = target - lo;
        data[lo..hi].select_nth_unstable_by(rel_idx, |a, b| a.total_cmp(b));
        return;
    }

    // Pick the middle target index as pivot
    let mid = indices.len() / 2;
    let pivot_abs = indices[mid];
    let pivot_rel = pivot_abs - lo;

    // Partition [lo, hi) around the pivot
    data[lo..hi].select_nth_unstable_by(pivot_rel, |a, b| a.total_cmp(b));

    // Recurse on left partition: indices in [lo, pivot_abs)
    // These are indices[0..mid]
    if mid > 0 {
        multi_select_recursive(data, &indices[..mid], lo, pivot_abs);
    }

    // Recurse on right partition: indices in (pivot_abs, hi)
    // These are indices[mid+1..]
    if mid + 1 < indices.len() {
        multi_select_recursive(data, &indices[mid + 1..], pivot_abs + 1, hi);
    }
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

    // Use integer arithmetic for index computation (avoids floating-point rounding)
    for k in 1..=9 {
        let (floor_idx, ceil_idx) = decile_indices(n, k);
        // Average the two values (handles the case where they're the same)
        result[k - 1] = (sorted[floor_idx] + sorted[ceil_idx]) / 2.0;
    }

    result
}

/// Compute mid-distribution quantiles for discrete data (spec §3.6).
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
    debug_assert_finite(data);

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
    debug_assert_finite(data);

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

    // ========== Multi-select (compute_deciles_inplace) Tests ==========

    #[test]
    fn test_multiselect_vs_sort_sequential() {
        // Sequential data
        let data: Vec<f64> = (1..=1000).map(|x| x as f64).collect();

        // Reference: sort-based
        let reference = compute_deciles(&data);

        // Test: multi-select based
        let mut test_data = data.clone();
        let result = compute_deciles_inplace(&mut test_data);

        for i in 0..9 {
            assert!(
                (reference[i] - result[i]).abs() < 1e-10,
                "Decile {} mismatch: sort={}, multiselect={}",
                i,
                reference[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_multiselect_vs_sort_random() {
        // Pseudo-random data
        let data: Vec<f64> = (0..2500)
            .map(|i| (i as f64 * 17.3 + 42.7) % 1000.0)
            .collect();

        let reference = compute_deciles(&data);
        let mut test_data = data.clone();
        let result = compute_deciles_inplace(&mut test_data);

        for i in 0..9 {
            assert!(
                (reference[i] - result[i]).abs() < 1e-10,
                "Decile {} mismatch: sort={}, multiselect={}",
                i,
                reference[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_multiselect_vs_sort_reversed() {
        // Reverse-sorted (adversarial for quickselect)
        let data: Vec<f64> = (0..1000).rev().map(|x| x as f64).collect();

        let reference = compute_deciles(&data);
        let mut test_data = data.clone();
        let result = compute_deciles_inplace(&mut test_data);

        for i in 0..9 {
            assert!(
                (reference[i] - result[i]).abs() < 1e-10,
                "Decile {} mismatch: sort={}, multiselect={}",
                i,
                reference[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_multiselect_vs_sort_all_equal() {
        // All equal values
        let data: Vec<f64> = vec![42.0; 500];

        let reference = compute_deciles(&data);
        let mut test_data = data.clone();
        let result = compute_deciles_inplace(&mut test_data);

        for i in 0..9 {
            assert!(
                (reference[i] - result[i]).abs() < 1e-10,
                "Decile {} mismatch: sort={}, multiselect={}",
                i,
                reference[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_multiselect_vs_sort_heavy_ties() {
        // Heavy ties (only 5 unique values)
        let data: Vec<f64> = (0..1000).map(|i| (i % 5) as f64 * 10.0).collect();

        let reference = compute_deciles(&data);
        let mut test_data = data.clone();
        let result = compute_deciles_inplace(&mut test_data);

        for i in 0..9 {
            assert!(
                (reference[i] - result[i]).abs() < 1e-10,
                "Decile {} mismatch: sort={}, multiselect={}",
                i,
                reference[i],
                result[i]
            );
        }
    }

    #[test]
    fn test_multiselect_vs_sort_small_sizes() {
        // Test various small sizes (edge cases)
        for n in [51, 52, 100, 101, 200] {
            let data: Vec<f64> = (0..n).map(|i| i as f64).collect();

            let reference = compute_deciles(&data);
            let mut test_data = data.clone();
            let result = compute_deciles_inplace(&mut test_data);

            for i in 0..9 {
                assert!(
                    (reference[i] - result[i]).abs() < 1e-10,
                    "n={}, Decile {} mismatch: sort={}, multiselect={}",
                    n,
                    i,
                    reference[i],
                    result[i]
                );
            }
        }
    }

    #[test]
    fn test_multiselect_vs_sort_bootstrap_size() {
        // Typical bootstrap sample size
        let data: Vec<f64> = (0..5000)
            .map(|i| 100.0 + ((i as f64 * std::f64::consts::PI).sin() * 50.0))
            .collect();

        let reference = compute_deciles(&data);
        let mut test_data = data.clone();
        let result = compute_deciles_inplace(&mut test_data);

        for i in 0..9 {
            assert!(
                (reference[i] - result[i]).abs() < 1e-8,
                "Decile {} mismatch: sort={}, multiselect={}",
                i,
                reference[i],
                result[i]
            );
        }
    }
}

/// Property-based tests using proptest
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating test data of various sizes
    fn data_strategy(min_size: usize, max_size: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(prop::num::f64::NORMAL, min_size..=max_size)
    }

    /// Strategy for data with potential duplicates (discrete-ish distribution)
    fn discrete_data_strategy(min_size: usize, max_size: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(0i32..100, min_size..=max_size)
            .prop_map(|v| v.into_iter().map(|x| x as f64).collect())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Multi-select must produce identical results to sort-based computation
        #[test]
        fn prop_multiselect_matches_sort(data in data_strategy(51, 10000)) {
            let reference = compute_deciles(&data);
            let mut test_data = data.clone();
            let result = compute_deciles_inplace(&mut test_data);

            for i in 0..9 {
                let diff = (reference[i] - result[i]).abs();
                prop_assert!(
                    diff < 1e-10,
                    "Decile {} mismatch: sort={}, multiselect={}, diff={}",
                    i, reference[i], result[i], diff
                );
            }
        }

        /// Multi-select works correctly with discrete data (many ties)
        #[test]
        fn prop_multiselect_discrete_data(data in discrete_data_strategy(51, 5000)) {
            let reference = compute_deciles(&data);
            let mut test_data = data.clone();
            let result = compute_deciles_inplace(&mut test_data);

            for i in 0..9 {
                let diff = (reference[i] - result[i]).abs();
                prop_assert!(
                    diff < 1e-10,
                    "Decile {} mismatch with discrete data: sort={}, multiselect={}",
                    i, reference[i], result[i]
                );
            }
        }

        /// Multi-select works on small arrays (falls back to sort)
        #[test]
        fn prop_multiselect_small_arrays(data in data_strategy(1, 50)) {
            let reference = compute_deciles(&data);
            let mut test_data = data.clone();
            let result = compute_deciles_inplace(&mut test_data);

            for i in 0..9 {
                let r = reference[i];
                let t = result[i];
                // Use total_cmp for exact equality (handles inf == inf), then check tolerance
                let equal = r.total_cmp(&t).is_eq() || (r - t).abs() < 1e-10;
                prop_assert!(
                    equal,
                    "Decile {} mismatch on small array (n={}): sort={}, multiselect={}",
                    i, data.len(), r, t
                );
            }
        }

        /// Multi-select preserves decile monotonicity
        #[test]
        fn prop_multiselect_monotonic(data in data_strategy(51, 5000)) {
            let mut test_data = data.clone();
            let result = compute_deciles_inplace(&mut test_data);

            for i in 1..9 {
                prop_assert!(
                    result[i] >= result[i - 1],
                    "Deciles not monotonic: d[{}]={} < d[{}]={}",
                    i - 1, result[i - 1], i, result[i]
                );
            }
        }

        /// Multi-select results are within data range
        #[test]
        fn prop_multiselect_within_range(data in data_strategy(51, 5000)) {
            let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let mut test_data = data.clone();
            let result = compute_deciles_inplace(&mut test_data);

            for i in 0..9 {
                prop_assert!(
                    result[i] >= min_val && result[i] <= max_val,
                    "Decile {} ({}) outside data range [{}, {}]",
                    i, result[i], min_val, max_val
                );
            }
        }

        /// compute_deciles_fast matches compute_deciles exactly
        #[test]
        fn prop_deciles_fast_matches_stable(data in data_strategy(1, 5000)) {
            let stable = compute_deciles(&data);
            let fast = compute_deciles_fast(&data);

            for i in 0..9 {
                let diff = (stable[i] - fast[i]).abs();
                prop_assert!(
                    diff < 1e-10,
                    "Decile {} mismatch between stable and fast: {} vs {}",
                    i, stable[i], fast[i]
                );
            }
        }

        /// compute_deciles_with_buffer matches compute_deciles
        #[test]
        fn prop_deciles_buffer_matches(data in data_strategy(1, 5000)) {
            let reference = compute_deciles(&data);
            let mut buffer = Vec::new();
            let result = compute_deciles_with_buffer(&data, &mut buffer);

            for i in 0..9 {
                let diff = (reference[i] - result[i]).abs();
                prop_assert!(
                    diff < 1e-10,
                    "Decile {} mismatch with buffer version: {} vs {}",
                    i, reference[i], result[i]
                );
            }
        }
    }

    /// Test with specific adversarial patterns
    #[test]
    fn test_multiselect_sorted_ascending() {
        let data: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let reference = compute_deciles(&data);
        let mut test_data = data.clone();
        let result = compute_deciles_inplace(&mut test_data);

        for i in 0..9 {
            assert!(
                (reference[i] - result[i]).abs() < 1e-10,
                "Sorted ascending: decile {} mismatch",
                i
            );
        }
    }

    #[test]
    fn test_multiselect_sorted_descending() {
        let data: Vec<f64> = (0..1000).rev().map(|x| x as f64).collect();
        let reference = compute_deciles(&data);
        let mut test_data = data.clone();
        let result = compute_deciles_inplace(&mut test_data);

        for i in 0..9 {
            assert!(
                (reference[i] - result[i]).abs() < 1e-10,
                "Sorted descending: decile {} mismatch",
                i
            );
        }
    }

    #[test]
    fn test_multiselect_all_same_value() {
        let data: Vec<f64> = vec![42.0; 1000];
        let reference = compute_deciles(&data);
        let mut test_data = data.clone();
        let result = compute_deciles_inplace(&mut test_data);

        for i in 0..9 {
            assert!(
                (reference[i] - result[i]).abs() < 1e-10,
                "All same value: decile {} mismatch",
                i
            );
        }
    }

    #[test]
    fn test_multiselect_two_values() {
        // Worst case for quickselect: two distinct values
        let data: Vec<f64> = (0..1000)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let reference = compute_deciles(&data);
        let mut test_data = data.clone();
        let result = compute_deciles_inplace(&mut test_data);

        for i in 0..9 {
            assert!(
                (reference[i] - result[i]).abs() < 1e-10,
                "Two values: decile {} mismatch",
                i
            );
        }
    }

    #[test]
    fn test_multiselect_boundary_size_51() {
        // Just above the threshold where multi-select kicks in
        let data: Vec<f64> = (0..51).map(|x| x as f64).collect();
        let reference = compute_deciles(&data);
        let mut test_data = data.clone();
        let result = compute_deciles_inplace(&mut test_data);

        for i in 0..9 {
            assert!(
                (reference[i] - result[i]).abs() < 1e-10,
                "Boundary size 51: decile {} mismatch",
                i
            );
        }
    }
}
