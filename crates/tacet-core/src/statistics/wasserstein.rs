//! Wasserstein distance computation for timing distribution comparison.
//!
//! This module implements the 1-Wasserstein distance (Earth Mover's Distance)
//! between timing distributions. W₁ provides an interpretable metric in
//! nanoseconds that measures the cost of transforming one distribution into
//! another.
//!
//! # Algorithm
//!
//! For two empirical distributions F and G with equal sample sizes n:
//! ```text
//! W₁(F, G) = (1/n) Σᵢ |F_sorted[i] - G_sorted[i]|
//! ```
//!
//! This is the average absolute difference between corresponding order statistics,
//! which equals the area between the CDFs.
//!
//! # Debiased Estimation
//!
//! Raw W₁ includes within-class noise. The debiased estimator removes this:
//! ```text
//! W₁_debiased = max(0, W₁(baseline, sample) - floor)
//! floor = median(W₁(B₁, B₂), W₁(S₁, S₂))
//! ```
//!
//! where B₁, B₂ are random splits of the baseline class and S₁, S₂ are
//! random splits of the sample class. The floor estimates the within-class
//! noise level.
//!
//! # Reference
//!
//! Villani, C. (2008). "Optimal Transport: Old and New."
//! Springer, Grundlehren der mathematischen Wissenschaften, Vol. 338.

extern crate alloc;

use alloc::vec::Vec;
use rand::{Rng, RngExt};

/// Compute the 1-Wasserstein distance (W₁) between two empirical distributions.
///
/// For two samples F and G of equal size n, this computes:
/// ```text
/// W₁(F, G) = (1/n) Σᵢ |F_sorted[i] - G_sorted[i]|
/// ```
///
/// This is the average absolute difference between order statistics,
/// which equals the area between the empirical CDFs.
///
/// # Arguments
///
/// * `baseline` - First sample (will be copied and sorted)
/// * `sample` - Second sample (will be copied and sorted)
///
/// # Returns
///
/// W₁ distance in nanoseconds. Returns 0.0 for empty inputs.
/// For unequal-sized inputs, truncates to the minimum length.
///
/// # Algorithm
///
/// 1. Sort both samples using unstable sort (O(n log n))
/// 2. Compute element-wise absolute differences (O(n))
/// 3. Return average difference
///
/// # Example
///
/// ```
/// use tacet_core::statistics::compute_w1_distance;
///
/// let baseline = vec![100.0, 102.0, 101.0];
/// let sample = vec![110.0, 112.0, 111.0];
///
/// let w1 = compute_w1_distance(&baseline, &sample);
/// assert!((w1 - 10.0).abs() < 1e-10); // Uniform shift of 10ns
/// ```
pub fn compute_w1_distance(baseline: &[f64], sample: &[f64]) -> f64 {
    // Handle edge cases
    if baseline.is_empty() || sample.is_empty() {
        return 0.0;
    }

    // Sort both arrays
    let mut baseline_sorted = baseline.to_vec();
    let mut sample_sorted = sample.to_vec();

    baseline_sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    sample_sorted.sort_unstable_by(|a, b| a.total_cmp(b));

    // For unequal sizes, truncate sorted arrays to the minimum length.
    // The block bootstrap can produce slightly unequal class sizes; returning
    // 0.0 would corrupt the variance estimate.
    let n = baseline_sorted.len().min(sample_sorted.len());

    // Compute average absolute difference between order statistics
    let mut sum = 0.0;
    for i in 0..n {
        sum += (baseline_sorted[i] - sample_sorted[i]).abs();
    }

    sum / (n as f64)
}

/// Compute debiased W₁ distance by subtracting within-class noise floor.
///
/// The raw W₁ distance includes both between-class differences and within-class
/// noise. This function estimates the noise floor by splitting each class into
/// two random halves and computing their W₁ distances. The median of these
/// within-class distances estimates the noise level.
///
/// # Algorithm
///
/// 1. Compute raw W₁(baseline, sample)
/// 2. Randomly split baseline into B₁ and B₂, compute W₁(B₁, B₂)
/// 3. Randomly split sample into S₁ and S₂, compute W₁(S₁, S₂)
/// 4. floor = median(W₁(B₁, B₂), W₁(S₁, S₂))
/// 5. Return max(0, raw - floor)
///
/// # Arguments
///
/// * `baseline` - Baseline class measurements
/// * `sample` - Sample class measurements
/// * `rng` - Random number generator for splitting
///
/// # Returns
///
/// Debiased W₁ distance in nanoseconds. Returns 0.0 for empty inputs or
/// samples too small to split (n < 4).
///
/// # Example
///
/// ```
/// use tacet_core::statistics::compute_w1_debiased;
/// use rand::SeedableRng;
/// use rand_xoshiro::Xoshiro256PlusPlus;
///
/// let baseline = vec![100.0; 100];
/// let sample = vec![110.0; 100];
/// let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
///
/// let w1 = compute_w1_debiased(&baseline, &sample, &mut rng);
/// // With constant values, within-class W₁ = 0, so debiased ≈ raw
/// assert!(w1 > 9.0 && w1 < 11.0);
/// ```
pub fn compute_w1_debiased(baseline: &[f64], sample: &[f64], rng: &mut impl Rng) -> f64 {
    // Handle edge cases
    if baseline.is_empty() || sample.is_empty() {
        return 0.0;
    }

    // Need at least 4 samples to split into halves of size 2+
    if baseline.len() < 4 || sample.len() < 4 {
        return 0.0;
    }

    // Compute raw W₁
    let raw_w1 = compute_w1_distance(baseline, sample);

    // Split baseline into two random halves
    let (baseline_1, baseline_2) = random_split(baseline, rng);
    let w1_baseline = compute_w1_distance(&baseline_1, &baseline_2);

    // Split sample into two random halves
    let (sample_1, sample_2) = random_split(sample, rng);
    let w1_sample = compute_w1_distance(&sample_1, &sample_2);

    // Compute median of within-class distances as noise floor
    let floor = median_of_two(w1_baseline, w1_sample);

    // Return debiased estimate (non-negative)
    (raw_w1 - floor).max(0.0)
}

/// Split a slice into two random halves without replacement.
///
/// Uses Fisher-Yates shuffle to randomly partition indices into two groups.
/// The halves will have sizes floor(n/2) and ceil(n/2).
///
/// # Arguments
///
/// * `data` - Input slice to split
/// * `rng` - Random number generator
///
/// # Returns
///
/// Tuple of (first_half, second_half) as new vectors.
fn random_split(data: &[f64], rng: &mut impl Rng) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    let half = n / 2;

    // Create shuffled indices
    let mut indices: Vec<usize> = (0..n).collect();

    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }

    // Split indices into two halves
    let mut first = Vec::with_capacity(half);
    let mut second = Vec::with_capacity(n - half);

    for i in 0..half {
        first.push(data[indices[i]]);
    }
    for i in half..n {
        second.push(data[indices[i]]);
    }

    (first, second)
}

/// Compute median of two values.
///
/// For two values, the median is their average.
fn median_of_two(a: f64, b: f64) -> f64 {
    (a + b) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_w1_identical_distributions() {
        // W₁ should be 0 for identical distributions
        let data = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let w1 = compute_w1_distance(&data, &data);
        assert!(
            w1.abs() < 1e-10,
            "W₁ should be 0 for identical distributions, got {}",
            w1
        );
    }

    #[test]
    fn test_w1_uniform_shift() {
        // W₁ should equal the shift amount for uniform shift
        let baseline = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let sample: Vec<f64> = baseline.iter().map(|x| x + 10.0).collect();

        let w1 = compute_w1_distance(&baseline, &sample);
        assert!(
            (w1 - 10.0).abs() < 1e-10,
            "W₁ should be 10.0 for uniform +10ns shift, got {}",
            w1
        );
    }

    #[test]
    fn test_w1_symmetric() {
        // W₁(A, B) should equal W₁(B, A)
        let baseline = vec![100.0, 105.0, 110.0, 115.0];
        let sample = vec![102.0, 107.0, 112.0, 117.0];

        let w1_ab = compute_w1_distance(&baseline, &sample);
        let w1_ba = compute_w1_distance(&sample, &baseline);

        assert!(
            (w1_ab - w1_ba).abs() < 1e-10,
            "W₁ should be symmetric, got {} vs {}",
            w1_ab,
            w1_ba
        );
    }

    #[test]
    fn test_w1_unsorted_input() {
        // W₁ should work correctly with unsorted input
        let baseline = vec![108.0, 100.0, 104.0, 106.0, 102.0];
        let sample = vec![118.0, 110.0, 114.0, 116.0, 112.0];

        let w1 = compute_w1_distance(&baseline, &sample);
        // After sorting: baseline = [100, 102, 104, 106, 108]
        //                sample = [110, 112, 114, 116, 118]
        // Differences: [10, 10, 10, 10, 10]
        // Average: 10
        assert!(
            (w1 - 10.0).abs() < 1e-10,
            "W₁ should handle unsorted input, got {}",
            w1
        );
    }

    #[test]
    fn test_w1_empty_inputs() {
        // Empty inputs should return 0
        let empty: Vec<f64> = vec![];
        let data = vec![100.0, 110.0];

        assert_eq!(compute_w1_distance(&empty, &data), 0.0);
        assert_eq!(compute_w1_distance(&data, &empty), 0.0);
        assert_eq!(compute_w1_distance(&empty, &empty), 0.0);
    }

    #[test]
    fn test_w1_size_mismatch() {
        // Size mismatch should return 0
        let short = vec![100.0, 110.0];
        let long = vec![100.0, 110.0, 120.0];

        assert_eq!(compute_w1_distance(&short, &long), 0.0);
    }

    #[test]
    fn test_w1_single_element() {
        // Single element should work
        let a = vec![100.0];
        let b = vec![110.0];

        let w1 = compute_w1_distance(&a, &b);
        assert!(
            (w1 - 10.0).abs() < 1e-10,
            "W₁ for single element should be absolute difference, got {}",
            w1
        );
    }

    #[test]
    fn test_w1_debiased_constant_values() {
        // With constant values, within-class W₁ = 0, so debiased ≈ raw
        let baseline = vec![100.0; 100];
        let sample = vec![110.0; 100];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let debiased = compute_w1_debiased(&baseline, &sample, &mut rng);
        let raw = compute_w1_distance(&baseline, &sample);

        assert!(
            (debiased - raw).abs() < 1e-10,
            "Debiased should equal raw for constant values (zero noise floor), got debiased={}, raw={}",
            debiased,
            raw
        );
    }

    #[test]
    fn test_w1_debiased_reduces_noise() {
        // With noisy data, debiased should be smaller than raw
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);

        // Generate noisy baseline and sample with small shift
        let mut baseline = Vec::with_capacity(100);
        let mut sample = Vec::with_capacity(100);

        for i in 0..100 {
            let noise = (i as f64 * 0.1) % 5.0; // Deterministic "noise"
            baseline.push(100.0 + noise);
            sample.push(102.0 + noise); // 2ns shift + same noise pattern
        }

        let raw = compute_w1_distance(&baseline, &sample);
        let debiased = compute_w1_debiased(&baseline, &sample, &mut rng);

        // Debiased should be positive but smaller than raw
        assert!(
            debiased >= 0.0,
            "Debiased W₁ should be non-negative, got {}",
            debiased
        );
        assert!(
            debiased <= raw,
            "Debiased W₁ should be <= raw W₁, got debiased={}, raw={}",
            debiased,
            raw
        );
    }

    #[test]
    fn test_w1_debiased_non_negative() {
        // Debiased W₁ should never be negative, even with high noise
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(456);

        // Generate high-noise data with no real shift
        let mut baseline = Vec::with_capacity(100);
        let mut sample = Vec::with_capacity(100);

        for i in 0..100 {
            let noise_b = ((i as f64 * 13.7) % 50.0) - 25.0;
            let noise_s = ((i as f64 * 17.3) % 50.0) - 25.0;
            baseline.push(100.0 + noise_b);
            sample.push(100.0 + noise_s);
        }

        let debiased = compute_w1_debiased(&baseline, &sample, &mut rng);

        assert!(
            debiased >= 0.0,
            "Debiased W₁ should be non-negative even with high noise, got {}",
            debiased
        );
    }

    #[test]
    fn test_w1_debiased_empty_inputs() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let empty: Vec<f64> = vec![];
        let data = vec![100.0; 10];

        assert_eq!(compute_w1_debiased(&empty, &data, &mut rng), 0.0);
        assert_eq!(compute_w1_debiased(&data, &empty, &mut rng), 0.0);
    }

    #[test]
    fn test_w1_debiased_too_small() {
        // Need at least 4 samples to split
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let small = vec![100.0, 110.0, 120.0];

        assert_eq!(compute_w1_debiased(&small, &small, &mut rng), 0.0);
    }

    #[test]
    fn test_random_split_preserves_size() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(789);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let (first, second) = random_split(&data, &mut rng);

        assert_eq!(first.len() + second.len(), data.len());
        assert_eq!(first.len(), 5);
        assert_eq!(second.len(), 5);
    }

    #[test]
    fn test_random_split_no_duplicates() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(999);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let (first, second) = random_split(&data, &mut rng);

        // Combine both halves and sort
        let mut combined = first.clone();
        combined.extend_from_slice(&second);
        combined.sort_by(|a, b| a.total_cmp(b));

        // Should match original when sorted
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.total_cmp(b));

        assert_eq!(
            combined, sorted_data,
            "Split should preserve all elements without duplication"
        );
    }

    #[test]
    fn test_median_of_two() {
        assert_eq!(median_of_two(10.0, 20.0), 15.0);
        assert_eq!(median_of_two(5.5, 5.5), 5.5);
        assert_eq!(median_of_two(0.0, 100.0), 50.0);
    }

    #[test]
    fn test_w1_triangle_inequality() {
        // W₁ satisfies triangle inequality: W₁(A,C) ≤ W₁(A,B) + W₁(B,C)
        let a = vec![100.0, 110.0, 120.0, 130.0];
        let b = vec![105.0, 115.0, 125.0, 135.0];
        let c = vec![110.0, 120.0, 130.0, 140.0];

        let w1_ac = compute_w1_distance(&a, &c);
        let w1_ab = compute_w1_distance(&a, &b);
        let w1_bc = compute_w1_distance(&b, &c);

        assert!(
            w1_ac <= w1_ab + w1_bc + 1e-10,
            "Triangle inequality violated: W₁(A,C)={} > W₁(A,B)={} + W₁(B,C)={}",
            w1_ac,
            w1_ab,
            w1_bc
        );
    }
}
