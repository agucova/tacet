//! Block bootstrap resampling for time series data.
//!
//! This module implements block bootstrap resampling which preserves
//! the autocorrelation structure of timing measurements. This is critical
//! for accurate covariance estimation when measurements have temporal
//! dependencies.

use rand::Rng;

/// Compute the optimal block size for block bootstrap.
///
/// Uses approximately sqrt(n) as the block size, which is a common
/// heuristic that balances between capturing autocorrelation structure
/// and having enough blocks for resampling variability.
///
/// # Arguments
///
/// * `n` - Number of samples in the original data
///
/// # Returns
///
/// The recommended block size, minimum 1.
pub fn compute_block_size(n: usize) -> usize {
    let block_size = (n as f64).sqrt().round() as usize;
    block_size.max(1)
}

/// Perform block bootstrap resampling on timing measurements.
///
/// Resamples measurements from a single class (fixed or random) while
/// preserving autocorrelation structure. Blocks of consecutive measurements
/// are sampled with replacement.
///
/// # Arguments
///
/// * `data` - Slice of timing measurements for one class
/// * `block_size` - Size of contiguous blocks to resample
/// * `rng` - Random number generator
///
/// # Returns
///
/// A new vector of resampled measurements with the same length as input.
///
/// # Algorithm
///
/// 1. Divide data into overlapping blocks of size `block_size`
/// 2. Sample blocks with replacement until we have n samples
/// 3. Truncate to exactly n samples
pub fn block_bootstrap_resample<R: Rng>(data: &[f64], block_size: usize, rng: &mut R) -> Vec<f64> {
    if data.is_empty() {
        return Vec::new();
    }

    let n = data.len();
    let block_size = block_size.max(1).min(n);

    // Number of possible starting positions for blocks
    let num_block_starts = n.saturating_sub(block_size) + 1;
    if num_block_starts == 0 {
        return data.to_vec();
    }

    let mut result = Vec::with_capacity(n);

    // Sample blocks until we have enough data
    while result.len() < n {
        // Random starting position for this block
        let start = rng.random_range(0..num_block_starts);
        let end = (start + block_size).min(n);

        // Add the block to our resampled data
        for i in start..end {
            if result.len() < n {
                result.push(data[i]);
            } else {
                break;
            }
        }
    }

    result
}

/// Perform stratified block bootstrap on interleaved measurements.
///
/// When measurements alternate between fixed and random classes,
/// this function resamples within each class separately to preserve
/// both the autocorrelation structure and class separation.
///
/// # Arguments
///
/// * `fixed_data` - Measurements for the fixed input class
/// * `random_data` - Measurements for the random input class
/// * `block_size` - Size of contiguous blocks to resample
/// * `rng` - Random number generator
///
/// # Returns
///
/// A tuple of (resampled_fixed, resampled_random) vectors.
pub fn stratified_block_bootstrap<R: Rng>(
    fixed_data: &[f64],
    random_data: &[f64],
    block_size: usize,
    rng: &mut R,
) -> (Vec<f64>, Vec<f64>) {
    // TODO: Consider implementing circular block bootstrap for better
    // edge handling (Politis & Romano, 1992)

    let resampled_fixed = block_bootstrap_resample(fixed_data, block_size, rng);
    let resampled_random = block_bootstrap_resample(random_data, block_size, rng);

    (resampled_fixed, resampled_random)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_block_size_computation() {
        assert_eq!(compute_block_size(100), 10);
        assert_eq!(compute_block_size(10000), 100);
        assert_eq!(compute_block_size(1), 1);
        assert_eq!(compute_block_size(0), 1); // Minimum is 1
    }

    #[test]
    fn test_bootstrap_preserves_length() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let resampled = block_bootstrap_resample(&data, 10, &mut rng);
        assert_eq!(resampled.len(), data.len());
    }

    #[test]
    fn test_bootstrap_samples_from_data() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let resampled = block_bootstrap_resample(&data, 10, &mut rng);

        // All resampled values should be from original data
        for val in &resampled {
            assert!(data.contains(val));
        }
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<f64> = vec![];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let resampled = block_bootstrap_resample(&data, 10, &mut rng);
        assert!(resampled.is_empty());
    }

    #[test]
    fn test_stratified_bootstrap() {
        let fixed: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let random: Vec<f64> = (50..100).map(|x| x as f64).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let (resampled_fixed, resampled_random) =
            stratified_block_bootstrap(&fixed, &random, 7, &mut rng);

        assert_eq!(resampled_fixed.len(), fixed.len());
        assert_eq!(resampled_random.len(), random.len());

        // Fixed resamples should come from fixed data
        for val in &resampled_fixed {
            assert!(fixed.contains(val));
        }

        // Random resamples should come from random data
        for val in &resampled_random {
            assert!(random.contains(val));
        }
    }
}
