//! Covariance matrix estimation via bootstrap.
//!
//! This module estimates the covariance matrix of quantile vectors
//! using block bootstrap resampling. The covariance matrix is essential
//! for the multivariate hypothesis testing in the timing oracle.

use crate::types::{Matrix9, Vector9};

use super::bootstrap::{block_bootstrap_resample, compute_block_size};
use super::quantile::compute_deciles;

use rand::SeedableRng;

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

/// Estimate covariance matrix of quantile vectors via block bootstrap.
///
/// This function bootstraps single-class quantile vectors (not differences)
/// and computes their sample covariance. Jitter is added to the diagonal
/// for numerical stability.
///
/// # Arguments
///
/// * `fixed_data` - Timing measurements for fixed input class
/// * `random_data` - Timing measurements for random input class
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
///    a. Resample fixed measurements with block bootstrap
///    b. Resample random measurements with block bootstrap
///    c. Compute deciles for each resampled class
///    d. Compute difference vector: q_fixed - q_random
/// 3. Compute sample covariance of difference vectors
/// 4. Add jitter to diagonal for numerical stability
pub fn bootstrap_covariance_matrix(
    fixed_data: &[f64],
    random_data: &[f64],
    n_bootstrap: usize,
    seed: u64,
) -> CovarianceEstimate {
    let n = fixed_data.len().min(random_data.len());
    let block_size = compute_block_size(n);

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Storage for bootstrap quantile difference vectors
    let mut bootstrap_diffs: Vec<Vector9> = Vec::with_capacity(n_bootstrap);

    // Generate bootstrap replicates
    for _ in 0..n_bootstrap {
        // Resample each class independently
        let resampled_fixed = block_bootstrap_resample(fixed_data, block_size, &mut rng);
        let resampled_random = block_bootstrap_resample(random_data, block_size, &mut rng);

        // Compute deciles for resampled data
        let q_fixed = compute_deciles(&resampled_fixed);
        let q_random = compute_deciles(&resampled_random);

        // Store the difference
        bootstrap_diffs.push(q_fixed - q_random);
    }

    // Compute sample covariance matrix
    let cov_matrix = compute_sample_covariance(&bootstrap_diffs);

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
        let fixed: Vec<f64> = (0..1000).map(|x| (x as f64) + 100.0).collect();
        let random: Vec<f64> = (0..1000).map(|x| x as f64).collect();

        let estimate = bootstrap_covariance_matrix(&fixed, &random, 100, 42);

        assert_eq!(estimate.n_bootstrap, 100);
        assert!(estimate.block_size > 0);
        assert!(estimate.jitter_added > 0.0);
    }

    #[test]
    fn test_covariance_symmetry() {
        let fixed: Vec<f64> = (0..500).map(|x| (x as f64) * 0.1).collect();
        let random: Vec<f64> = (0..500).map(|x| (x as f64) * 0.1 + 10.0).collect();

        let estimate = bootstrap_covariance_matrix(&fixed, &random, 50, 123);

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
}
