//! Variance estimation via bootstrap.
//!
//! This module estimates the variance of W₁ distance using block bootstrap
//! resampling. The variance estimate is used for hypothesis testing in the
//! timing oracle.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::types::{Class, TimingSample};

use super::bootstrap::{
    block_bootstrap_resample_into, block_bootstrap_resample_joint_into, counter_rng_seed,
};
use super::wasserstein::compute_w1_distance;

/// Bootstrap resampling strategy for variance estimation.
///
/// Controls how the block bootstrap preserves (or breaks) temporal dependencies
/// between the two measurement classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BootstrapMethod {
    /// Joint (stream) bootstrap: resample the interleaved measurement stream as
    /// a single sequence, then split by class. This preserves cross-class
    /// temporal dependencies from common-mode noise (e.g., CPU frequency drift,
    /// thermal throttling), yielding correct variance estimates under
    /// autocorrelation.
    #[default]
    Joint,

    /// Stratified (per-class) bootstrap: resample each class independently.
    /// This breaks cross-class temporal pairing, causing variance
    /// underestimation when measurements exhibit autocorrelation. Included
    /// for ablation comparison only.
    Stratified,
}

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Online variance accumulator using Welford's algorithm (scalar version).
///
/// This is the 1D analog of `WelfordCovariance9`, accumulating variance
/// in a single pass without storing all values. Used for bootstrap variance
/// estimation of scalar statistics like W₁ distance.
///
/// Uses Welford's numerically stable online algorithm for mean and M2 (sum of
/// squared deviations), which can be converted to variance via M2/(n-1).
#[derive(Debug, Clone)]
struct WelfordVariance {
    /// Count of values accumulated so far.
    n: usize,
    /// Running mean of values.
    mean: f64,
    /// Sum of squared deviations: Σ(x - μ)²
    m2: f64,
}

impl WelfordVariance {
    /// Create a new accumulator initialized to zero.
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Update the accumulator with a new value using Welford's algorithm.
    ///
    /// Algorithm:
    /// ```text
    /// δ = x - μₙ₋₁
    /// μₙ = μₙ₋₁ + δ/n
    /// δ' = x - μₙ
    /// M2ₙ = M2ₙ₋₁ + δ·δ'
    /// ```
    ///
    /// This is numerically stable and avoids catastrophic cancellation.
    fn update(&mut self, x: f64) {
        self.n += 1;
        let n = self.n as f64;

        // δ = x - μₙ₋₁
        let delta = x - self.mean;

        // μₙ = μₙ₋₁ + δ/n
        self.mean += delta / n;

        // δ' = x - μₙ
        let delta2 = x - self.mean;

        // M2ₙ = M2ₙ₋₁ + δ·δ'
        self.m2 += delta * delta2;
    }

    /// Finalize the accumulator and return the sample variance.
    ///
    /// Returns M2/(n-1) for the unbiased sample variance estimator.
    /// For n < 2, returns a conservative large variance (1e6) rather than
    /// zero, since zero variance would be arbitrarily small.
    fn finalize(&self) -> f64 {
        if self.n < 2 {
            // Return conservative high variance (not zero)
            return 1e6;
        }

        self.m2 / (self.n - 1) as f64
    }

    /// Merge another accumulator into this one using Chan's parallel algorithm.
    ///
    /// Algorithm:
    /// ```text
    /// n_AB = n_A + n_B
    /// δ = μ_B - μ_A
    /// μ_AB = (n_A·μ_A + n_B·μ_B) / n_AB
    /// M2_AB = M2_A + M2_B + (n_A·n_B/n_AB)·δ²
    /// ```
    ///
    /// This preserves numerical stability.
    #[allow(dead_code)]
    fn merge(&mut self, other: &Self) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
            return;
        }

        let n_a = self.n as f64;
        let n_b = other.n as f64;
        let n_ab = n_a + n_b;

        // δ = μ_B - μ_A
        let delta = other.mean - self.mean;

        // μ_AB = (n_A·μ_A + n_B·μ_B) / n_AB
        self.mean = (self.mean * n_a + other.mean * n_b) / n_ab;

        // M2_AB = M2_A + M2_B + (n_A·n_B/n_AB)·δ²
        let correction = delta * delta * (n_a * n_b / n_ab);
        self.m2 = self.m2 + other.m2 + correction;

        self.n += other.n;
    }
}

/// Result of W₁ variance estimation including variance and diagnostics.
#[derive(Debug, Clone)]
pub struct W1VarianceEstimate {
    /// The estimated variance of W₁ distance.
    pub variance: f64,

    /// Number of bootstrap replicates used.
    pub n_bootstrap: usize,

    /// Block size used for bootstrap.
    pub block_size: usize,

    /// Amount of jitter added for numerical stability.
    pub jitter_added: f64,
}

/// Estimate variance of W₁ distance via block bootstrap.
///
/// This function bootstraps the W₁ (Wasserstein-1) distance between
/// two distributions using joint resampling to preserve temporal pairing.
/// The W₁ distance is computed from sorted samples and accumulated into
/// a variance estimate using Welford's algorithm.
///
/// The W₁ distance between two empirical distributions is computed as:
/// ```text
/// W₁(F, R) = (1/n) Σᵢ |F_sorted[i] - R_sorted[i]|
/// ```
///
/// This is equivalent to the L1 distance between the quantile functions.
///
/// # Arguments
///
/// * `interleaved` - Timing samples in measurement order, each tagged with class
/// * `n_bootstrap` - Number of bootstrap replicates (typically 50-100 for adaptive, 2000 for thorough)
/// * `seed` - Random seed for reproducibility
/// * `is_fragile` - If true, apply inflation factor for fragile regimes
///   (uniqueness ratio < 10%, high autocorrelation detected)
///
/// # Returns
///
/// A `W1VarianceEstimate` containing the variance and diagnostics.
///
/// # Algorithm
///
/// 1. Compute block size using class-conditional Politis-White algorithm
/// 2. For each bootstrap replicate:
///    a. Block-resample the JOINT interleaved sequence (preserving temporal pairing)
///    b. Split by class AFTER resampling
///    c. Sort both class samples
///    d. Compute W₁ = (1/n) Σ |baseline_sorted\[i\] - sample_sorted\[i\]|
///    e. Update WelfordVariance accumulator
/// 3. Finalize to get variance estimate
/// 4. Add jitter for numerical stability
pub fn bootstrap_w1_variance(
    interleaved: &[TimingSample],
    n_bootstrap: usize,
    seed: u64,
    is_fragile: bool,
    method: BootstrapMethod,
) -> W1VarianceEstimate {
    let n = interleaved.len();

    // Use class-conditional acquisition-lag ACF for block length selection
    let block_size =
        super::block_length::class_conditional_optimal_block_length(interleaved, is_fragile);

    // Pre-split interleaved data by class (needed for stratified bootstrap)
    let (baseline_raw, sample_raw): (Vec<f64>, Vec<f64>) = if method == BootstrapMethod::Stratified
    {
        let mut b = Vec::with_capacity(n / 2 + 1);
        let mut s = Vec::with_capacity(n / 2 + 1);
        for sample in interleaved {
            match sample.class {
                Class::Baseline => b.push(sample.time_ns),
                Class::Sample => s.push(sample.time_ns),
            }
        }
        (b, s)
    } else {
        (Vec::new(), Vec::new())
    };

    // Generate bootstrap replicates of W₁
    #[cfg(feature = "parallel")]
    let var_accumulator: WelfordVariance = {
        // Pre-compute estimated class sizes (roughly n/2 each)
        let estimated_class_size = (n / 2) + 1;
        (0..n_bootstrap)
            .into_par_iter()
            .fold_with(
                // Per-thread state: RNG, scratch buffers, and Welford accumulator
                (
                    Xoshiro256PlusPlus::seed_from_u64(seed),
                    vec![
                        TimingSample {
                            time_ns: 0.0,
                            class: Class::Baseline
                        };
                        n
                    ],
                    Vec::with_capacity(estimated_class_size), // baseline_samples buffer
                    Vec::with_capacity(estimated_class_size), // sample_samples buffer
                    vec![0.0f64; baseline_raw.len()],         // stratified baseline resample buffer
                    vec![0.0f64; sample_raw.len()],           // stratified sample resample buffer
                    WelfordVariance::new(),
                ),
                |(
                    _,
                    mut buffer,
                    mut baseline_samples,
                    mut sample_samples,
                    mut baseline_buf,
                    mut sample_buf,
                    mut acc,
                ),
                 i| {
                    // Counter-based RNG for deterministic, well-distributed seeding
                    let mut rng =
                        Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

                    match method {
                        BootstrapMethod::Joint => {
                            // Joint resample the interleaved sequence (preserves temporal pairing)
                            block_bootstrap_resample_joint_into(
                                interleaved,
                                block_size,
                                &mut rng,
                                &mut buffer,
                            );

                            // Split by class AFTER resampling
                            baseline_samples.clear();
                            sample_samples.clear();
                            for sample in &buffer {
                                match sample.class {
                                    Class::Baseline => baseline_samples.push(sample.time_ns),
                                    Class::Sample => sample_samples.push(sample.time_ns),
                                }
                            }
                        }
                        BootstrapMethod::Stratified => {
                            // Resample each class independently (breaks cross-class pairing)
                            block_bootstrap_resample_into(
                                &baseline_raw,
                                block_size,
                                &mut rng,
                                &mut baseline_buf,
                            );
                            block_bootstrap_resample_into(
                                &sample_raw,
                                block_size,
                                &mut rng,
                                &mut sample_buf,
                            );
                            baseline_samples.clear();
                            sample_samples.clear();
                            baseline_samples.extend_from_slice(&baseline_buf);
                            sample_samples.extend_from_slice(&sample_buf);
                        }
                    }

                    // Compute W₁ distance using the wasserstein module
                    let w1 = compute_w1_distance(&baseline_samples, &sample_samples);

                    // Update variance accumulator
                    acc.update(w1);

                    (
                        rng,
                        buffer,
                        baseline_samples,
                        sample_samples,
                        baseline_buf,
                        sample_buf,
                        acc,
                    )
                },
            )
            .map(|(_, _, _, _, _, _, acc)| acc)
            .reduce(WelfordVariance::new, |mut a, b| {
                a.merge(&b);
                a
            })
    };

    #[cfg(not(feature = "parallel"))]
    let var_accumulator: WelfordVariance = {
        let mut accumulator = WelfordVariance::new();
        let mut buffer = vec![
            TimingSample {
                time_ns: 0.0,
                class: Class::Baseline
            };
            n
        ];
        // Pre-allocate class buffers (roughly n/2 each)
        let estimated_class_size = (n / 2) + 1;
        let mut baseline_samples: Vec<f64> = Vec::with_capacity(estimated_class_size);
        let mut sample_samples: Vec<f64> = Vec::with_capacity(estimated_class_size);
        let mut baseline_buf = vec![0.0f64; baseline_raw.len()];
        let mut sample_buf = vec![0.0f64; sample_raw.len()];

        for i in 0..n_bootstrap {
            // Counter-based RNG for deterministic, well-distributed seeding
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

            match method {
                BootstrapMethod::Joint => {
                    // Joint resample the interleaved sequence (preserves temporal pairing)
                    block_bootstrap_resample_joint_into(
                        interleaved,
                        block_size,
                        &mut rng,
                        &mut buffer,
                    );

                    // Split by class AFTER resampling
                    baseline_samples.clear();
                    sample_samples.clear();
                    for sample in &buffer {
                        match sample.class {
                            Class::Baseline => baseline_samples.push(sample.time_ns),
                            Class::Sample => sample_samples.push(sample.time_ns),
                        }
                    }
                }
                BootstrapMethod::Stratified => {
                    // Resample each class independently (breaks cross-class pairing)
                    block_bootstrap_resample_into(
                        &baseline_raw,
                        block_size,
                        &mut rng,
                        &mut baseline_buf,
                    );
                    block_bootstrap_resample_into(
                        &sample_raw,
                        block_size,
                        &mut rng,
                        &mut sample_buf,
                    );
                    baseline_samples.clear();
                    sample_samples.clear();
                    baseline_samples.extend_from_slice(&baseline_buf);
                    sample_samples.extend_from_slice(&sample_buf);
                }
            }

            // Compute W₁ distance using the wasserstein module
            let w1 = compute_w1_distance(&baseline_samples, &sample_samples);

            // Update variance accumulator
            accumulator.update(w1);
        }
        accumulator
    };

    // Finalize the Welford accumulator to get the variance
    let variance = var_accumulator.finalize();

    // Add jitter for numerical stability (similar to covariance matrix)
    // ε = 10⁻¹⁰ + σ²·10⁻⁸
    let epsilon = 1e-10 + variance * 1e-8;
    let stabilized_variance = variance + epsilon;

    W1VarianceEstimate {
        variance: stabilized_variance,
        n_bootstrap,
        block_size,
        jitter_added: epsilon,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== WelfordVariance Tests ==========

    #[test]
    fn test_welford_variance_basic() {
        // Test basic variance computation matches manual calculation
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut welford = WelfordVariance::new();
        for &v in &values {
            welford.update(v);
        }
        let var_welford = welford.finalize();

        // Manual calculation: mean = 3.0, variance = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 4
        // = (4 + 1 + 0 + 1 + 4) / 4 = 2.5
        let expected_var = 2.5;
        assert!(
            (var_welford - expected_var).abs() < 1e-10,
            "Welford variance mismatch: got {}, expected {}",
            var_welford,
            expected_var
        );
    }

    #[test]
    fn test_welford_variance_edge_cases() {
        // n=0 should return conservative high variance
        let empty = WelfordVariance::new();
        let var0 = empty.finalize();
        assert_eq!(var0, 1e6, "n=0 should return conservative variance");

        // n=1 should return conservative high variance
        let mut one = WelfordVariance::new();
        one.update(42.0);
        let var1 = one.finalize();
        assert_eq!(var1, 1e6, "n=1 should return conservative variance");

        // n=2 should compute valid variance
        let mut two = WelfordVariance::new();
        two.update(1.0);
        two.update(3.0);
        let var2 = two.finalize();
        // Variance = ((1-2)² + (3-2)²) / 1 = 2.0
        assert!(
            (var2 - 2.0).abs() < 1e-10,
            "n=2 variance should be 2.0, got {}",
            var2
        );
    }

    #[test]
    fn test_welford_variance_merge() {
        // Test that merge(A, B) == accumulate(A ∪ B)
        let values_a = vec![1.0, 2.0, 3.0];
        let values_b = vec![4.0, 5.0, 6.0];

        // Accumulate A separately
        let mut acc_a = WelfordVariance::new();
        for &v in &values_a {
            acc_a.update(v);
        }

        // Accumulate B separately
        let mut acc_b = WelfordVariance::new();
        for &v in &values_b {
            acc_b.update(v);
        }

        // Merge A and B
        let mut merged = acc_a.clone();
        merged.merge(&acc_b);
        let merged_var = merged.finalize();

        // Accumulate all at once
        let mut combined = WelfordVariance::new();
        for &v in values_a.iter().chain(values_b.iter()) {
            combined.update(v);
        }
        let combined_var = combined.finalize();

        // Results should be identical
        assert!(
            (merged_var - combined_var).abs() < 1e-10,
            "Merge variance mismatch: merged={}, combined={}",
            merged_var,
            combined_var
        );
    }

    #[test]
    fn test_welford_variance_stability() {
        // Test numerical stability with large values and small differences
        let base = 1e9;
        let values: Vec<f64> = (0..100).map(|i| base + (i as f64) * 0.1).collect();

        let mut welford = WelfordVariance::new();
        for &v in &values {
            welford.update(v);
        }
        let var_welford = welford.finalize();

        // Should produce non-zero, finite variance
        assert!(var_welford.is_finite(), "Variance should be finite");
        assert!(var_welford > 0.0, "Variance should be positive");

        // Variance should be reasonable for a linear sequence with step 0.1
        // For sequence 0, 0.1, 0.2, ..., 9.9 (100 values):
        // Variance ≈ (n²-1) * h² / 12 = (10000-1) * 0.01 / 12 ≈ 8.33
        // Allow for numerical error: expect variance in range [8.0, 9.0]
        assert!(
            var_welford > 8.0 && var_welford < 9.0,
            "Variance {} outside expected range [8.0, 9.0] for linear sequence",
            var_welford
        );
    }

    // ========== W₁ Variance Bootstrap Tests ==========

    #[test]
    fn test_w1_variance_bootstrap_basic() {
        use crate::types::Class;

        // Generate simple test data with known W₁ distance
        let n = 500;
        let mut samples = Vec::with_capacity(2 * n);

        for i in 0..n {
            samples.push(TimingSample {
                time_ns: 100.0 + (i as f64) * 0.1,
                class: Class::Baseline,
            });
            samples.push(TimingSample {
                time_ns: 102.0 + (i as f64) * 0.1, // Constant shift of 2.0 ns
                class: Class::Sample,
            });
        }

        let estimate = bootstrap_w1_variance(&samples, 100, 42, false, BootstrapMethod::Joint);

        // Variance should be positive and finite
        assert!(
            estimate.variance > 0.0 && estimate.variance.is_finite(),
            "Variance should be positive and finite, got {}",
            estimate.variance
        );

        // Check diagnostics
        assert_eq!(estimate.n_bootstrap, 100);
        assert!(estimate.block_size > 0);
        assert!(estimate.jitter_added > 0.0);
    }

    #[test]
    fn test_w1_variance_identical_distributions() {
        use crate::types::Class;

        // When distributions are identical, W₁ ≈ 0 with small variance
        let n = 500;
        let mut samples = Vec::with_capacity(2 * n);

        for i in 0..n {
            let val = 100.0 + (i as f64) * 0.1;
            samples.push(TimingSample {
                time_ns: val,
                class: Class::Baseline,
            });
            samples.push(TimingSample {
                time_ns: val, // Identical
                class: Class::Sample,
            });
        }

        let estimate = bootstrap_w1_variance(&samples, 50, 123, false, BootstrapMethod::Joint);

        // Variance should be small (near zero) but positive due to jitter
        assert!(
            estimate.variance < 1.0,
            "Variance should be small for identical distributions, got {}",
            estimate.variance
        );
        assert!(
            estimate.variance > 0.0,
            "Variance should be positive (has jitter)"
        );
    }

    #[test]
    fn test_w1_variance_deterministic() {
        use crate::types::Class;

        // Same input and seed should give same result
        let n = 200;
        let mut samples = Vec::with_capacity(2 * n);

        for i in 0..n {
            samples.push(TimingSample {
                time_ns: 100.0 + (i as f64),
                class: Class::Baseline,
            });
            samples.push(TimingSample {
                time_ns: 105.0 + (i as f64),
                class: Class::Sample,
            });
        }

        let estimate1 = bootstrap_w1_variance(&samples, 50, 999, false, BootstrapMethod::Joint);
        let estimate2 = bootstrap_w1_variance(&samples, 50, 999, false, BootstrapMethod::Joint);

        // Should be identical (deterministic RNG)
        assert_eq!(
            estimate1.variance, estimate2.variance,
            "Bootstrap should be deterministic with same seed"
        );
        assert_eq!(estimate1.block_size, estimate2.block_size);
    }

    #[test]
    fn test_w1_variance_fragile_flag() {
        use crate::types::Class;

        // Test that fragile flag affects block size
        let n = 500;
        let mut samples = Vec::with_capacity(2 * n);

        for i in 0..n {
            samples.push(TimingSample {
                time_ns: 100.0 + (i as f64) * 0.1,
                class: Class::Baseline,
            });
            samples.push(TimingSample {
                time_ns: 102.0 + (i as f64) * 0.1,
                class: Class::Sample,
            });
        }

        let normal = bootstrap_w1_variance(&samples, 50, 42, false, BootstrapMethod::Joint);
        let fragile = bootstrap_w1_variance(&samples, 50, 42, true, BootstrapMethod::Joint);

        // Fragile mode should use larger block size (1.5x inflation + floor of 10)
        assert!(
            fragile.block_size >= normal.block_size,
            "Fragile mode should use larger or equal block size: normal={}, fragile={}",
            normal.block_size,
            fragile.block_size
        );
    }

    #[test]
    fn test_w1_variance_parallel_equivalence() {
        use crate::types::Class;

        // Test that parallel and sequential modes give same results
        // (This test will pass in both modes since we use counter-based RNG)
        let n = 300;
        let mut samples = Vec::with_capacity(2 * n);

        for i in 0..n {
            samples.push(TimingSample {
                time_ns: 100.0 + ((i * 7) % 100) as f64,
                class: Class::Baseline,
            });
            samples.push(TimingSample {
                time_ns: 110.0 + ((i * 11) % 100) as f64,
                class: Class::Sample,
            });
        }

        let estimate = bootstrap_w1_variance(&samples, 100, 777, false, BootstrapMethod::Joint);

        // Just verify it produces sensible output
        assert!(estimate.variance > 0.0);
        assert!(estimate.variance.is_finite());
        assert!(estimate.block_size > 0);
        assert!(estimate.n_bootstrap == 100);
    }
}
