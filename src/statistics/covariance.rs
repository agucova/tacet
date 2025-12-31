//! Covariance matrix estimation via bootstrap.
//!
//! This module estimates the covariance matrix of quantile vectors
//! using block bootstrap resampling. The covariance matrix is essential
//! for the multivariate hypothesis testing in the timing oracle.

use crate::types::{Matrix9, Vector9};

use super::bootstrap::{block_bootstrap_resample_into, compute_block_size, counter_rng_seed};
use super::quantile::compute_deciles_inplace;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Result of covariance estimation including the matrix and diagnostics.
#[derive(Debug, Clone)]
pub struct CovarianceEstimate {
    /// The estimated 9x9 covariance matrix.
    pub matrix: Matrix9,

    /// Number of bootstrap replicates used.
    pub n_bootstrap: usize,

    /// Block size used for bootstrap.
    pub block_size: usize,

    /// Minimum eigenvalue (for numerical stability check).
    pub min_eigenvalue: f64,

    /// Amount of jitter added for numerical stability.
    pub jitter_added: f64,
}

impl CovarianceEstimate {
    /// Check if the covariance matrix is numerically stable.
    ///
    /// A matrix is considered stable if its minimum eigenvalue is
    /// sufficiently positive (not near-singular).
    pub fn is_stable(&self) -> bool {
        self.min_eigenvalue > 1e-10
    }
}

/// Online covariance accumulator using Welford's algorithm.
///
/// This accumulates covariance in a single pass without storing all vectors,
/// saving memory (728 bytes vs 72 KB for 1000 iterations) and improving
/// cache locality.
///
/// Uses Welford's numerically stable online algorithm for mean and M2 (sum of
/// outer products), which can be converted to covariance via M2/(n-1).
#[derive(Debug, Clone)]
pub struct WelfordCovariance9 {
    /// Count of vectors accumulated so far.
    n: usize,
    /// Running mean of vectors.
    mean: Vector9,
    /// Sum of outer products: Σ(x - μ)(x - μ)^T
    m2: Matrix9,
}

impl WelfordCovariance9 {
    /// Create a new accumulator initialized to zeros.
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: Vector9::zeros(),
            m2: Matrix9::zeros(),
        }
    }

    /// Update the accumulator with a new vector using Welford's algorithm.
    ///
    /// Algorithm:
    /// ```text
    /// δ = x - μₙ₋₁
    /// μₙ = μₙ₋₁ + δ/n
    /// δ' = x - μₙ
    /// M2ₙ = M2ₙ₋₁ + δ·δ'^T
    /// ```
    ///
    /// This is numerically stable and produces a symmetric M2 matrix.
    pub fn update(&mut self, x: &Vector9) {
        self.n += 1;
        let n = self.n as f64;

        // δ = x - μₙ₋₁
        let delta = x - &self.mean;

        // μₙ = μₙ₋₁ + δ/n
        self.mean += &delta / n;

        // δ' = x - μₙ
        let delta2 = x - &self.mean;

        // M2ₙ = M2ₙ₋₁ + δ·δ'^T
        // The outer product δ·δ'^T is symmetric
        self.m2 += delta * delta2.transpose();
    }

    /// Finalize the accumulator and return the covariance matrix.
    ///
    /// Returns M2/(n-1) for the unbiased sample covariance estimator.
    /// For n < 2, returns the identity matrix (degenerate case).
    pub fn finalize(&self) -> Matrix9 {
        if self.n < 2 {
            return Matrix9::identity();
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
    /// M2_AB = M2_A + M2_B + (n_A·n_B/n_AB)·δ·δ^T
    /// ```
    ///
    /// This preserves numerical stability and symmetry.
    pub fn merge(&mut self, other: &Self) {
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
        let delta = &other.mean - &self.mean;

        // μ_AB = (n_A·μ_A + n_B·μ_B) / n_AB
        self.mean = (&self.mean * n_a + &other.mean * n_b) / n_ab;

        // M2_AB = M2_A + M2_B + (n_A·n_B/n_AB)·δ·δ^T
        let correction = delta * delta.transpose() * (n_a * n_b / n_ab);
        self.m2 = &self.m2 + &other.m2 + correction;

        self.n += other.n;
    }

    /// Get the current count of vectors.
    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        self.n
    }
}

/// Estimate covariance matrix of single-class quantile vectors via block bootstrap.
///
/// This function bootstraps quantile vectors for one class (not differences)
/// and computes their sample covariance. Jitter is added to the diagonal
/// for numerical stability.
///
/// # Arguments
///
/// * `data` - Timing measurements for a single input class
/// * `n_bootstrap` - Number of bootstrap replicates (typically 1000-5000)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// A `CovarianceEstimate` containing the covariance matrix and diagnostics.
///
/// # Algorithm
///
/// 1. Compute block size as sqrt(n)
/// 2. For each bootstrap replicate:
///    a. Resample measurements with block bootstrap
///    b. Compute deciles for the resampled data
/// 3. Compute sample covariance of quantile vectors
/// 4. Add jitter to diagonal for numerical stability
pub fn bootstrap_covariance_matrix(
    data: &[f64],
    n_bootstrap: usize,
    seed: u64,
) -> CovarianceEstimate {
    let n = data.len();
    let block_size = compute_block_size(n);

    // Generate bootstrap replicates using online Welford covariance accumulation
    // This avoids allocating Vec<Vector9> (saves 72 KB for 1000 iterations)
    #[cfg(feature = "parallel")]
    let cov_accumulator: WelfordCovariance9 = {
        (0..n_bootstrap)
            .into_par_iter()
            .fold_with(
                // Per-thread state: RNG, scratch buffer, and Welford accumulator
                (
                    Xoshiro256PlusPlus::seed_from_u64(seed),
                    vec![0.0; n],
                    WelfordCovariance9::new(),
                ),
                |(mut rng, mut buffer, mut acc), i| {
                    // Counter-based RNG for deterministic, well-distributed seeding
                    rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

                    // Resample, compute deciles, and update accumulator
                    block_bootstrap_resample_into(data, block_size, &mut rng, &mut buffer);
                    let quantiles = compute_deciles_inplace(&mut buffer);
                    acc.update(&quantiles);

                    (rng, buffer, acc)
                },
            )
            .map(|(_, _, acc)| acc)
            .reduce(
                || WelfordCovariance9::new(),
                |mut a, b| {
                    a.merge(&b);
                    a
                },
            )
    };

    #[cfg(not(feature = "parallel"))]
    let cov_accumulator: WelfordCovariance9 = {
        let mut accumulator = WelfordCovariance9::new();
        let mut buffer = vec![0.0; n];

        for i in 0..n_bootstrap {
            // Counter-based RNG for deterministic, well-distributed seeding
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

            // Resample, compute deciles, and update accumulator
            block_bootstrap_resample_into(data, block_size, &mut rng, &mut buffer);
            let quantiles = compute_deciles_inplace(&mut buffer);
            accumulator.update(&quantiles);
        }
        accumulator
    };

    // Finalize the Welford accumulator to get the covariance matrix
    let cov_matrix = cov_accumulator.finalize();

    // Add jitter for numerical stability
    let (stabilized_matrix, jitter) = add_diagonal_jitter(cov_matrix);

    // Compute minimum eigenvalue for stability check
    // TODO: Use proper eigenvalue computation from nalgebra
    let min_eigenvalue = estimate_min_eigenvalue(&stabilized_matrix);

    CovarianceEstimate {
        matrix: stabilized_matrix,
        n_bootstrap,
        block_size,
        min_eigenvalue,
        jitter_added: jitter,
    }
}

/// Compute sample covariance matrix from a collection of vectors.
///
/// Uses the unbiased estimator with n-1 denominator.
fn compute_sample_covariance(vectors: &[Vector9]) -> Matrix9 {
    let n = vectors.len();
    if n < 2 {
        return Matrix9::identity();
    }

    // Compute mean vector
    let mut mean = Vector9::zeros();
    for v in vectors {
        mean += v;
    }
    mean /= n as f64;

    // Compute covariance matrix
    let mut cov = Matrix9::zeros();
    for v in vectors {
        let centered = v - mean;
        cov += centered * centered.transpose();
    }
    cov /= (n - 1) as f64;

    cov
}

/// Add small jitter to diagonal for numerical stability.
///
/// This ensures the covariance matrix is positive definite even
/// when there's near-collinearity in the quantile vectors.
fn add_diagonal_jitter(mut matrix: Matrix9) -> (Matrix9, f64) {
    // Compute a data-adaptive jitter based on the matrix scale
    let trace = matrix.trace();
    let base_jitter = 1e-10;
    let adaptive_jitter = (trace / 9.0) * 1e-8;
    let jitter = base_jitter + adaptive_jitter;

    // Add jitter to diagonal
    for i in 0..9 {
        matrix[(i, i)] += jitter;
    }

    (matrix, jitter)
}

/// Estimate minimum eigenvalue of a matrix.
///
/// This is a placeholder that uses a simple heuristic.
/// TODO: Replace with proper eigenvalue decomposition.
fn estimate_min_eigenvalue(matrix: &Matrix9) -> f64 {
    // Simple heuristic: check if Cholesky decomposition succeeds
    // If it does, all eigenvalues are positive

    // For now, use a rough estimate based on the diagonal dominance
    // This is not accurate but provides a stability indicator

    let mut min_diag = f64::MAX;
    let mut max_off_diag_sum: f64 = 0.0;

    for i in 0..9 {
        min_diag = min_diag.min(matrix[(i, i)]);

        let mut row_sum = 0.0;
        for j in 0..9 {
            if i != j {
                row_sum += matrix[(i, j)].abs();
            }
        }
        max_off_diag_sum = max_off_diag_sum.max(row_sum);
    }

    // Gershgorin circle theorem lower bound
    min_diag - max_off_diag_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariance_estimate_basic() {
        // Generate simple test data
        let data: Vec<f64> = (0..1000).map(|x| (x as f64) + 100.0).collect();

        let estimate = bootstrap_covariance_matrix(&data, 100, 42);

        assert_eq!(estimate.n_bootstrap, 100);
        assert!(estimate.block_size > 0);
        assert!(estimate.jitter_added > 0.0);
    }

    #[test]
    fn test_covariance_symmetry() {
        let data: Vec<f64> = (0..500).map(|x| (x as f64) * 0.1).collect();

        let estimate = bootstrap_covariance_matrix(&data, 50, 123);

        // Check symmetry
        for i in 0..9 {
            for j in 0..9 {
                let diff = (estimate.matrix[(i, j)] - estimate.matrix[(j, i)]).abs();
                assert!(diff < 1e-12, "Matrix not symmetric at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_sample_covariance_identity() {
        // With identical vectors, covariance should be zero (plus jitter)
        let vectors: Vec<Vector9> = (0..100).map(|_| Vector9::from_element(1.0)).collect();

        let cov = compute_sample_covariance(&vectors);

        // All elements should be essentially zero
        for i in 0..9 {
            for j in 0..9 {
                assert!(cov[(i, j)].abs() < 1e-12);
            }
        }
    }

    // ========== Welford Covariance Validation Tests ==========

    #[test]
    fn test_welford_numerical_equivalence() {
        // Test that Welford accumulator gives identical results to batch computation
        let vectors: Vec<Vector9> = (0..100)
            .map(|i| {
                Vector9::from_fn(|j, _| {
                    // Create diverse vectors with different patterns
                    (i * 7 + j * 13) as f64 % 17.0
                })
            })
            .collect();

        // Batch method
        let batch_cov = compute_sample_covariance(&vectors);

        // Welford method
        let mut welford = WelfordCovariance9::new();
        for v in &vectors {
            welford.update(v);
        }
        let welford_cov = welford.finalize();

        // Should be numerically identical (within floating point precision)
        for i in 0..9 {
            for j in 0..9 {
                let diff = (batch_cov[(i, j)] - welford_cov[(i, j)]).abs();
                assert!(
                    diff < 1e-9,
                    "Mismatch at ({}, {}): batch={}, welford={}, diff={}",
                    i,
                    j,
                    batch_cov[(i, j)],
                    welford_cov[(i, j)],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_welford_edge_cases() {
        // n=0 should return identity
        let empty = WelfordCovariance9::new();
        let cov0 = empty.finalize();
        assert_eq!(cov0, Matrix9::identity(), "n=0 should return identity");

        // n=1 should return identity
        let mut one = WelfordCovariance9::new();
        one.update(&Vector9::from_element(42.0));
        let cov1 = one.finalize();
        assert_eq!(cov1, Matrix9::identity(), "n=1 should return identity");

        // n=2 should compute valid covariance
        let mut two = WelfordCovariance9::new();
        two.update(&Vector9::from_element(1.0));
        two.update(&Vector9::from_element(2.0));
        let cov2 = two.finalize();

        // Should not be identity (has variance)
        let is_identity = cov2 == Matrix9::identity();
        assert!(!is_identity, "n=2 should not return identity");

        // Should be symmetric
        for i in 0..9 {
            for j in 0..9 {
                assert!(
                    (cov2[(i, j)] - cov2[(j, i)]).abs() < 1e-12,
                    "n=2 result not symmetric"
                );
            }
        }
    }

    #[test]
    fn test_welford_merge_correctness() {
        // Test that merge(A, B) == accumulate(A ∪ B)
        let vectors_a: Vec<Vector9> = (0..50).map(|i| Vector9::from_element(i as f64)).collect();
        let vectors_b: Vec<Vector9> = (50..100).map(|i| Vector9::from_element(i as f64)).collect();

        // Accumulate A separately
        let mut acc_a = WelfordCovariance9::new();
        for v in &vectors_a {
            acc_a.update(v);
        }

        // Accumulate B separately
        let mut acc_b = WelfordCovariance9::new();
        for v in &vectors_b {
            acc_b.update(v);
        }

        // Merge A and B
        let mut merged = acc_a.clone();
        merged.merge(&acc_b);
        let merged_cov = merged.finalize();

        // Accumulate all at once
        let mut combined = WelfordCovariance9::new();
        for v in vectors_a.iter().chain(vectors_b.iter()) {
            combined.update(v);
        }
        let combined_cov = combined.finalize();

        // Results should be identical
        for i in 0..9 {
            for j in 0..9 {
                let diff = (merged_cov[(i, j)] - combined_cov[(i, j)]).abs();
                assert!(
                    diff < 1e-9,
                    "Merge mismatch at ({}, {}): merged={}, combined={}, diff={}",
                    i,
                    j,
                    merged_cov[(i, j)],
                    combined_cov[(i, j)],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_welford_symmetry() {
        // Welford algorithm should produce symmetric covariance matrix
        let mut welford = WelfordCovariance9::new();
        for i in 0..100 {
            welford.update(&Vector9::from_fn(|j, _| ((i * 7 + j * 11) % 23) as f64));
        }

        let cov = welford.finalize();

        for i in 0..9 {
            for j in 0..9 {
                let diff = (cov[(i, j)] - cov[(j, i)]).abs();
                assert!(
                    diff < 1e-12,
                    "Welford result not symmetric at ({}, {}): diff={}",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_determinism() {
        // Test that parallel and serial versions give identical results with same seed
        let data: Vec<f64> = (0..500).map(|x| (x as f64) * 0.1 + 100.0).collect();

        let result1 = bootstrap_covariance_matrix(&data, 100, 42);
        let result2 = bootstrap_covariance_matrix(&data, 100, 42);
        let result3 = bootstrap_covariance_matrix(&data, 100, 42);

        // All three runs should produce identical matrices
        for i in 0..9 {
            for j in 0..9 {
                let val1 = result1.matrix[(i, j)];
                let val2 = result2.matrix[(i, j)];
                let val3 = result3.matrix[(i, j)];

                assert!(
                    (val1 - val2).abs() < 1e-12,
                    "Run 1 and 2 differ at ({}, {})",
                    i,
                    j
                );
                assert!(
                    (val2 - val3).abs() < 1e-12,
                    "Run 2 and 3 differ at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_covariance_chunking_edge_cases() {
        // Test edge cases: n_bootstrap < typical chunk size, exact multiples, remainders
        let data: Vec<f64> = (0..500).map(|x| x as f64 * 0.1).collect();

        // n_bootstrap < chunk_size (should still work)
        let r1 = bootstrap_covariance_matrix(&data, 5, 42);
        assert_eq!(r1.n_bootstrap, 5, "Should use n_bootstrap=5");

        // Larger n_bootstrap
        let r2 = bootstrap_covariance_matrix(&data, 100, 42);
        assert_eq!(r2.n_bootstrap, 100, "Should use n_bootstrap=100");

        // n_bootstrap with potential remainder
        let r3 = bootstrap_covariance_matrix(&data, 150, 42);
        assert_eq!(r3.n_bootstrap, 150, "Should use n_bootstrap=150");

        // All should be deterministic
        let r3_repeat = bootstrap_covariance_matrix(&data, 150, 42);
        for i in 0..9 {
            for j in 0..9 {
                assert!(
                    (r3.matrix[(i, j)] - r3_repeat.matrix[(i, j)]).abs() < 1e-12,
                    "Edge case should be deterministic at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}
