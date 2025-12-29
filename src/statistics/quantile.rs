//! Quantile computation using O(n) selection algorithms.
//!
//! This module provides efficient quantile computation using Rust's
//! `slice.select_nth_unstable()` which uses introselect for O(n) average time.

use crate::constants::DECILES;
use crate::types::Vector9;

/// Compute a single quantile from a mutable slice.
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

    // Compute the index for the quantile
    // Using the "R-7" quantile definition (linear interpolation)
    let h = (n - 1) as f64 * p;
    let h_floor = h.floor() as usize;
    let h_frac = h - h.floor();

    if h_floor >= n - 1 {
        // At or beyond the last element
        let (_, &mut max, _) = data.select_nth_unstable_by(n - 1, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        return max;
    }

    // Get the lower value using select_nth_unstable
    let (_, &mut lower, upper) = data.select_nth_unstable_by(h_floor, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });

    if h_frac == 0.0 {
        return lower;
    }

    // Find the minimum of the upper partition for interpolation
    let upper_min = upper
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(lower);

    // Linear interpolation between lower and upper
    lower + h_frac * (upper_min - lower)
}

/// Compute all 9 deciles [0.1, 0.2, ..., 0.9] from timing measurements.
///
/// Returns a Vector9 containing the quantile values at each decile.
/// The input slice is cloned internally to avoid mutation.
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

    let mut working_copy = data.to_vec();
    let mut result = Vector9::zeros();

    for (i, &p) in DECILES.iter().enumerate() {
        // TODO: Optimize by computing all quantiles in a single pass
        // Currently each quantile computation may partially reorder the data
        result[i] = compute_quantile(&mut working_copy, p);

        // Reset working copy for next quantile (inefficient but correct)
        working_copy.copy_from_slice(data);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_quantile_median() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let median = compute_quantile(&mut data, 0.5);
        assert!((median - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_quantile_extremes() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let min = compute_quantile(&mut data.clone(), 0.0);
        let max = compute_quantile(&mut data, 1.0);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max - 5.0).abs() < 1e-10);
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
    #[should_panic(expected = "Cannot compute quantile of empty slice")]
    fn test_empty_slice_panics() {
        let mut data: Vec<f64> = vec![];
        compute_quantile(&mut data, 0.5);
    }
}
