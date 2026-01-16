//! Covariance matrix estimation via bootstrap.
//!
//! This module estimates the covariance matrix of quantile vectors
//! using block bootstrap resampling. The covariance matrix is essential
//! for the multivariate hypothesis testing in the timing oracle.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::types::{Class, Matrix9, TimingSample, Vector9};

use super::block_length::{optimal_block_length, paired_optimal_block_length};
use super::bootstrap::{
    block_bootstrap_resample_into, block_bootstrap_resample_joint_into, counter_rng_seed,
};
use super::quantile::{compute_deciles_inplace, compute_midquantile_deciles};

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
    /// A matrix is stable if Cholesky decomposition succeeds, which
    /// is both necessary and sufficient for positive definiteness.
    /// The Gershgorin bound is only a lower bound on eigenvalues and
    /// can be negative even for positive definite matrices.
    pub fn is_stable(&self) -> bool {
        nalgebra::Cholesky::new(self.matrix).is_some()
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
        let delta = x - self.mean;

        // μₙ = μₙ₋₁ + δ/n
        self.mean += delta / n;

        // δ' = x - μₙ
        let delta2 = x - self.mean;

        // M2ₙ = M2ₙ₋₁ + δ·δ'^T
        // The outer product δ·δ'^T is symmetric
        self.m2 += delta * delta2.transpose();
    }

    /// Finalize the accumulator and return the covariance matrix.
    ///
    /// Returns M2/(n-1) for the unbiased sample covariance estimator.
    /// For n < 2, returns a conservative large-variance diagonal matrix
    /// (1e6 on diagonal) rather than identity, since "1 ns² variance"
    /// would be arbitrarily small.
    pub fn finalize(&self) -> Matrix9 {
        if self.n < 2 {
            // Return conservative high-variance diagonal (not identity)
            // This ensures MDE will be huge → priors dominated by min_effect_of_concern
            return Matrix9::from_diagonal(&Vector9::repeat(1e6));
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
        let delta = other.mean - self.mean;

        // μ_AB = (n_A·μ_A + n_B·μ_B) / n_AB
        self.mean = (self.mean * n_a + other.mean * n_b) / n_ab;

        // M2_AB = M2_A + M2_B + (n_A·n_B/n_AB)·δ·δ^T
        let correction = delta * delta.transpose() * (n_a * n_b / n_ab);
        self.m2 = self.m2 + other.m2 + correction;

        self.n += other.n;
    }

    /// Get the current count of vectors.
    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        self.n
    }
}

impl Default for WelfordCovariance9 {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate covariance matrix of single-class quantile vectors via block bootstrap.
///
/// This function bootstraps quantile vectors for one class (not differences)
/// and computes their sample covariance. Jitter is added to the diagonal
/// for numerical stability.
///
/// See spec section 2.6 (Covariance Estimation):
/// - Uses block bootstrap to preserve autocorrelation structure
/// - Block length from Politis-White algorithm
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
/// 1. Compute block size using Politis-White algorithm
/// 2. For each bootstrap replicate:
///    a. Resample measurements with block bootstrap
///    b. Compute deciles for the resampled data
/// 3. Compute sample covariance of quantile vectors using Welford's online algorithm
/// 4. Add jitter to diagonal for numerical stability
pub fn bootstrap_covariance_matrix(
    data: &[f64],
    n_bootstrap: usize,
    seed: u64,
) -> CovarianceEstimate {
    let n = data.len();
    // Use Politis-White algorithm for optimal block length selection (spec §2.6)
    let block_size = if n >= 10 {
        optimal_block_length(data).circular.ceil() as usize
    } else {
        // Fall back to simple formula for very small samples
        (1.3 * (n as f64).powf(1.0 / 3.0)).ceil() as usize
    }
    .max(1);

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
                |(_, mut buffer, mut acc), i| {
                    // Counter-based RNG for deterministic, well-distributed seeding
                    let mut rng =
                        Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

                    // Resample, compute deciles, and update accumulator
                    block_bootstrap_resample_into(data, block_size, &mut rng, &mut buffer);
                    let quantiles = compute_deciles_inplace(&mut buffer);
                    acc.update(&quantiles);

                    (rng, buffer, acc)
                },
            )
            .map(|(_, _, acc)| acc)
            .reduce(WelfordCovariance9::new, |mut a, b| {
                a.merge(&b);
                a
            })
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
    let min_eigenvalue = estimate_min_eigenvalue(&stabilized_matrix);

    CovarianceEstimate {
        matrix: stabilized_matrix,
        n_bootstrap,
        block_size,
        min_eigenvalue,
        jitter_added: jitter,
    }
}

fn resample_with_indices(data: &[f64], indices: &[usize], block_size: usize, buffer: &mut [f64]) {
    let n = buffer.len();
    let mut pos = 0;

    for &start in indices {
        for offset in 0..block_size {
            if pos >= n {
                break;
            }
            let idx = (start + offset) % data.len();
            buffer[pos] = data[idx];
            pos += 1;
        }
    }
}

/// Estimate covariance matrix of quantile differences Δ* = q_F* - q_R* via joint block bootstrap.
///
/// Uses joint resampling to preserve temporal pairing between fixed and random samples.
/// This captures cross-covariance Cov(q_F, q_R) > 0 from common-mode noise, giving the
/// correct (smaller) Var(Δ) and improving statistical power.
///
/// See spec section 2.6 (Covariance Estimation):
/// - Uses paired block bootstrap: same indices for both classes
/// - Block length from Politis-White algorithm
/// - Σ_rate = Σ_cal × n_cal (covariance scales as 1/n)
///
/// This function is designed to work with calibration sample counts (default 5000)
/// and supports the adaptive architecture by providing covariance estimates that
/// can be scaled via `compute_covariance_rate` and `scale_covariance_rate`.
///
/// # Arguments
///
/// * `interleaved` - Timing samples in measurement order, each tagged with class
/// * `n_bootstrap` - Number of bootstrap replicates (typically 50-100 for adaptive, 2000 for thorough)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// A `CovarianceEstimate` containing the covariance matrix of Δ* and diagnostics.
///
/// # Algorithm
///
/// 1. Compute block size using Politis-White algorithm (per-class, take max)
/// 2. For each bootstrap replicate:
///    a. Block-resample the JOINT interleaved sequence (preserving temporal pairing)
///    b. Split by class AFTER resampling
///    c. Compute q_F* and q_R* from the split data
///    d. Compute Δ* = q_F* - q_R*
/// 3. Compute sample covariance of Δ* vectors using Welford's online algorithm
/// 4. Add jitter to diagonal for numerical stability
pub fn bootstrap_difference_covariance(
    interleaved: &[TimingSample],
    n_bootstrap: usize,
    seed: u64,
) -> CovarianceEstimate {
    let n = interleaved.len();
    // Use Politis-White algorithm for optimal block length selection (spec §2.6)
    // CRITICAL: Spec §3.2.2 says compute per-class, not on interleaved sequence.
    let mut baseline_data = Vec::new();
    let mut sample_data = Vec::new();
    for s in interleaved {
        match s.class {
            Class::Baseline => baseline_data.push(s.time_ns),
            Class::Sample => sample_data.push(s.time_ns),
        }
    }

    let block_size = if baseline_data.len() >= 10 && sample_data.len() >= 10 {
        paired_optimal_block_length(&baseline_data, &sample_data)
    } else {
        // Fall back to simple formula for very small samples
        (1.3 * (n as f64).powf(1.0 / 3.0)).ceil() as usize
    }
    .max(1);

    // Generate bootstrap replicates of Δ* = q_F* - q_R* using joint resampling
    #[cfg(feature = "parallel")]
    let cov_accumulator: WelfordCovariance9 = {
        (0..n_bootstrap)
            .into_par_iter()
            .fold_with(
                // Per-thread state: RNG, scratch buffer for joint samples, and Welford accumulator
                (
                    Xoshiro256PlusPlus::seed_from_u64(seed),
                    vec![
                        TimingSample {
                            time_ns: 0.0,
                            class: Class::Baseline
                        };
                        n
                    ],
                    WelfordCovariance9::new(),
                ),
                |(_, mut buffer, mut acc), i| {
                    // Counter-based RNG for deterministic, well-distributed seeding
                    let mut rng =
                        Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

                    // Joint resample the interleaved sequence (preserves temporal pairing)
                    block_bootstrap_resample_joint_into(
                        interleaved,
                        block_size,
                        &mut rng,
                        &mut buffer,
                    );

                    // Split by class AFTER resampling
                    let mut baseline_samples: Vec<f64> = Vec::new();
                    let mut sample_samples: Vec<f64> = Vec::new();
                    for sample in &buffer {
                        match sample.class {
                            Class::Baseline => baseline_samples.push(sample.time_ns),
                            Class::Sample => sample_samples.push(sample.time_ns),
                        }
                    }

                    // Compute quantiles for each class
                    let q_baseline = compute_deciles_inplace(&mut baseline_samples);
                    let q_sample = compute_deciles_inplace(&mut sample_samples);

                    // Compute difference and update accumulator
                    let delta = q_baseline - q_sample;
                    acc.update(&delta);

                    (rng, buffer, acc)
                },
            )
            .map(|(_, _, acc)| acc)
            .reduce(WelfordCovariance9::new, |mut a, b| {
                a.merge(&b);
                a
            })
    };

    #[cfg(not(feature = "parallel"))]
    let cov_accumulator: WelfordCovariance9 = {
        let mut accumulator = WelfordCovariance9::new();
        let mut buffer = vec![
            TimingSample {
                time_ns: 0.0,
                class: Class::Baseline
            };
            n
        ];

        for i in 0..n_bootstrap {
            // Counter-based RNG for deterministic, well-distributed seeding
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

            // Joint resample the interleaved sequence (preserves temporal pairing)
            block_bootstrap_resample_joint_into(interleaved, block_size, &mut rng, &mut buffer);

            // Split by class AFTER resampling
            let mut baseline_samples: Vec<f64> = Vec::new();
            let mut sample_samples: Vec<f64> = Vec::new();
            for sample in &buffer {
                match sample.class {
                    Class::Baseline => baseline_samples.push(sample.time_ns),
                    Class::Sample => sample_samples.push(sample.time_ns),
                }
            }

            // Compute quantiles for each class
            let q_baseline = compute_deciles_inplace(&mut baseline_samples);
            let q_sample = compute_deciles_inplace(&mut sample_samples);

            // Compute difference and update accumulator
            let delta = q_baseline - q_sample;
            accumulator.update(&delta);
        }
        accumulator
    };

    // Finalize the Welford accumulator to get the covariance matrix
    let cov_matrix = cov_accumulator.finalize();

    // Add jitter for numerical stability
    let (stabilized_matrix, jitter) = add_diagonal_jitter(cov_matrix);

    // Compute minimum eigenvalue for stability check
    let min_eigenvalue = estimate_min_eigenvalue(&stabilized_matrix);

    CovarianceEstimate {
        matrix: stabilized_matrix,
        n_bootstrap,
        block_size,
        min_eigenvalue,
        jitter_added: jitter,
    }
}

/// Estimate covariance matrix of quantile differences Δ* = q_F* - q_R* in discrete mode.
///
/// Uses m-out-of-n paired block bootstrap on per-class sequences with mid-distribution
/// quantiles, then rescales by m/n (spec §2.4).
pub fn bootstrap_difference_covariance_discrete(
    baseline: &[f64],
    sample: &[f64],
    n_bootstrap: usize,
    seed: u64,
) -> CovarianceEstimate {
    let n = baseline.len().min(sample.len());
    let baseline = &baseline[..n];
    let sample = &sample[..n];

    let m = if n < 2000 {
        let half = (0.5 * n as f64).floor() as usize;
        half.max(200).min(n)
    } else {
        let m = (n as f64).powf(2.0 / 3.0).floor() as usize;
        m.max(400).min(n)
    };
    let mut block_size = if n >= 10 {
        paired_optimal_block_length(baseline, sample)
    } else {
        (1.3 * (n as f64).powf(1.0 / 3.0)).ceil().max(1.0) as usize
    };
    let max_block = (m / 5).max(1);
    block_size = block_size.min(max_block).max(1);

    #[cfg(feature = "parallel")]
    let cov_accumulator: WelfordCovariance9 = {
        (0..n_bootstrap)
            .into_par_iter()
            .fold_with(
                (
                    Xoshiro256PlusPlus::seed_from_u64(seed),
                    vec![0.0; m],
                    vec![0.0; m],
                    Vec::<usize>::with_capacity(m.div_ceil(block_size)),
                    WelfordCovariance9::new(),
                ),
                |(_, mut baseline_buf, mut sample_buf, mut indices, mut acc), i| {
                    let mut rng =
                        Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

                    indices.clear();
                    let n_blocks = m.div_ceil(block_size);
                    let max_start = n.saturating_sub(block_size);
                    for _ in 0..n_blocks {
                        indices.push(rng.random_range(0..=max_start));
                    }

                    resample_with_indices(baseline, &indices, block_size, &mut baseline_buf);
                    resample_with_indices(sample, &indices, block_size, &mut sample_buf);

                    let q_baseline = compute_midquantile_deciles(&baseline_buf);
                    let q_sample = compute_midquantile_deciles(&sample_buf);
                    let delta = q_baseline - q_sample;
                    acc.update(&delta);

                    (rng, baseline_buf, sample_buf, indices, acc)
                },
            )
            .map(|(_, _, _, _, acc)| acc)
            .reduce(WelfordCovariance9::new, |mut a, b| {
                a.merge(&b);
                a
            })
    };

    #[cfg(not(feature = "parallel"))]
    let cov_accumulator: WelfordCovariance9 = {
        let mut accumulator = WelfordCovariance9::new();
        let mut baseline_buf = vec![0.0; m];
        let mut sample_buf = vec![0.0; m];
        let mut indices = Vec::with_capacity(m.div_ceil(block_size));

        for i in 0..n_bootstrap {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

            indices.clear();
            let n_blocks = m.div_ceil(block_size);
            let max_start = n.saturating_sub(block_size);
            for _ in 0..n_blocks {
                indices.push(rng.random_range(0..=max_start));
            }

            resample_with_indices(baseline, &indices, block_size, &mut baseline_buf);
            resample_with_indices(sample, &indices, block_size, &mut sample_buf);

            let q_baseline = compute_midquantile_deciles(&baseline_buf);
            let q_sample = compute_midquantile_deciles(&sample_buf);
            let delta = q_baseline - q_sample;
            accumulator.update(&delta);
        }
        accumulator
    };

    // Finalize the Welford accumulator and rescale from m to n.
    let mut cov_matrix = cov_accumulator.finalize();
    if n > 0 {
        cov_matrix *= (m as f64) / (n as f64);
    }

    let (stabilized_matrix, jitter) = add_diagonal_jitter(cov_matrix);
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
/// For n < 2, returns conservative large-variance diagonal.
#[cfg(test)]
fn compute_sample_covariance(vectors: &[Vector9]) -> Matrix9 {
    let n = vectors.len();
    if n < 2 {
        return Matrix9::from_diagonal(&Vector9::repeat(1e6));
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

/// Apply variance floor based on timer resolution.
///
/// In idealized environments (simulators, deterministic operations), variance
/// can approach zero, causing numerical instability. The 1/12 factor is the
/// variance of a uniform distribution over one tick.
///
/// # Arguments
///
/// * `matrix` - Covariance matrix to apply floor to
/// * `timer_resolution_ns` - Timer resolution in nanoseconds
///
/// # Returns
///
/// The matrix with variance floor applied to diagonal elements.
pub fn apply_variance_floor(mut matrix: Matrix9, timer_resolution_ns: f64) -> Matrix9 {
    let floor = timer_resolution_ns.powi(2) / 12.0;
    for i in 0..9 {
        matrix[(i, i)] += floor;
    }
    matrix
}

/// Scale covariance matrix from calibration to inference sample sizes.
///
/// Σ₀ was estimated from calibration set (n_cal samples) but will be used
/// for inference set (n_inf samples). Quantile variance scales as 1/n,
/// so we must adjust.
///
/// See spec section 2.6 (Covariance Estimation): "Covariance scales as 1/n"
///
/// # Arguments
///
/// * `matrix` - Covariance matrix estimated from calibration set
/// * `n_calibration` - Number of samples in calibration set
/// * `n_inference` - Number of samples in inference set
///
/// # Returns
///
/// The scaled covariance matrix.
pub fn scale_covariance_for_inference(
    matrix: Matrix9,
    n_calibration: usize,
    n_inference: usize,
) -> Matrix9 {
    let scale = n_calibration as f64 / n_inference as f64;
    matrix * scale
}

/// Compute covariance rate from calibration covariance.
///
/// The covariance rate Σ_rate = Σ_cal × n_cal allows efficient
/// scaling during adaptive sampling: Σ_n = Σ_rate / n.
///
/// This is useful for adaptive sampling where the sample count
/// grows over multiple rounds. Rather than re-bootstrapping
/// covariance at each round, compute the rate once from
/// calibration data and scale as needed.
///
/// See spec section 2.6 (Covariance Estimation): "Covariance scales as 1/n"
///
/// # Arguments
///
/// * `covariance` - Covariance matrix estimated from calibration set
/// * `n_calibration` - Number of samples used for calibration
///
/// # Returns
///
/// The covariance rate matrix (Σ_rate = Σ_cal × n_cal).
pub fn compute_covariance_rate(covariance: &Matrix9, n_calibration: usize) -> Matrix9 {
    let scale = n_calibration as f64;
    covariance * scale
}

/// Scale covariance rate to get covariance for n samples.
///
/// Given a covariance rate Σ_rate (computed via `compute_covariance_rate`),
/// returns the covariance matrix for n samples: Σ_n = Σ_rate / n.
///
/// This enables efficient covariance estimation during adaptive sampling
/// without re-running the bootstrap at each sample count.
///
/// See spec section 2.6 (Covariance Estimation): "Covariance scales as 1/n"
///
/// # Arguments
///
/// * `rate` - Covariance rate matrix (Σ_rate = Σ_cal × n_cal)
/// * `n` - Number of samples for which to compute covariance
///
/// # Returns
///
/// The scaled covariance matrix Σ_n = Σ_rate / n.
///
/// # Panics
///
/// Panics if n is 0 (would cause division by zero).
pub fn scale_covariance_rate(rate: &Matrix9, n: usize) -> Matrix9 {
    assert!(n > 0, "Cannot scale covariance rate for 0 samples");
    let scale = 1.0 / (n as f64);
    rate * scale
}

/// Estimate minimum eigenvalue of a matrix.
///
/// This is a placeholder that uses a simple heuristic.
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
        // n=0 should return conservative high-variance diagonal
        let empty = WelfordCovariance9::new();
        let cov0 = empty.finalize();
        let expected_conservative = Matrix9::from_diagonal(&Vector9::repeat(1e6));
        assert_eq!(
            cov0, expected_conservative,
            "n=0 should return conservative diagonal"
        );

        // n=1 should return conservative high-variance diagonal
        let mut one = WelfordCovariance9::new();
        one.update(&Vector9::from_element(42.0));
        let cov1 = one.finalize();
        assert_eq!(
            cov1, expected_conservative,
            "n=1 should return conservative diagonal"
        );

        // n=2 should compute valid covariance
        let mut two = WelfordCovariance9::new();
        two.update(&Vector9::from_element(1.0));
        two.update(&Vector9::from_element(2.0));
        let cov2 = two.finalize();

        // Should not be the conservative fallback (has actual variance)
        assert!(
            cov2 != expected_conservative,
            "n=2 should not return conservative fallback"
        );

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

    // ========== Covariance Rate Tests ==========

    #[test]
    fn test_covariance_rate_roundtrip() {
        // compute_covariance_rate followed by scale_covariance_rate should
        // return to the original covariance when using the same n
        let original = Matrix9::from_fn(|i, j| {
            if i == j {
                10.0 + i as f64
            } else {
                (i as f64 - j as f64).abs() * 0.5
            }
        });

        let n_cal = 5000;
        let rate = compute_covariance_rate(&original, n_cal);
        let recovered = scale_covariance_rate(&rate, n_cal);

        for i in 0..9 {
            for j in 0..9 {
                let diff = (original[(i, j)] - recovered[(i, j)]).abs();
                assert!(
                    diff < 1e-10,
                    "Roundtrip failed at ({}, {}): original={}, recovered={}, diff={}",
                    i,
                    j,
                    original[(i, j)],
                    recovered[(i, j)],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_covariance_rate_scaling() {
        // Verify that Σ_n = Σ_rate / n produces correct scaling
        // If Σ_cal was estimated from n_cal samples, then:
        // - Σ_rate = Σ_cal * n_cal
        // - Σ_{2*n_cal} = Σ_rate / (2*n_cal) = Σ_cal / 2

        let sigma_cal = Matrix9::from_fn(|i, j| {
            if i == j {
                100.0 // Diagonal variance
            } else if (i as i32 - j as i32).abs() == 1 {
                50.0 // Adjacent covariance
            } else {
                10.0 // Other covariance
            }
        });

        let n_cal = 1000;
        let rate = compute_covariance_rate(&sigma_cal, n_cal);

        // Double the sample size should halve the covariance
        let sigma_2n = scale_covariance_rate(&rate, 2 * n_cal);
        for i in 0..9 {
            for j in 0..9 {
                let expected = sigma_cal[(i, j)] / 2.0;
                let actual = sigma_2n[(i, j)];
                let diff = (expected - actual).abs();
                assert!(
                    diff < 1e-10,
                    "2n scaling failed at ({}, {}): expected={}, actual={}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }

        // 10x samples should reduce covariance by 10x
        let sigma_10n = scale_covariance_rate(&rate, 10 * n_cal);
        for i in 0..9 {
            for j in 0..9 {
                let expected = sigma_cal[(i, j)] / 10.0;
                let actual = sigma_10n[(i, j)];
                let diff = (expected - actual).abs();
                assert!(
                    diff < 1e-10,
                    "10n scaling failed at ({}, {}): expected={}, actual={}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "Cannot scale covariance rate for 0 samples")]
    fn test_scale_covariance_rate_zero_panics() {
        let rate = Matrix9::identity();
        let _ = scale_covariance_rate(&rate, 0);
    }

    #[test]
    fn test_covariance_rate_preserves_symmetry() {
        // Symmetric covariance should remain symmetric after rate operations
        let symmetric = Matrix9::from_fn(|i, j| {
            if i == j {
                50.0
            } else {
                25.0 / (1.0 + (i as i32 - j as i32).abs() as f64)
            }
        });

        // Verify input is symmetric
        for i in 0..9 {
            for j in 0..9 {
                assert!(
                    (symmetric[(i, j)] - symmetric[(j, i)]).abs() < 1e-12,
                    "Input not symmetric"
                );
            }
        }

        let rate = compute_covariance_rate(&symmetric, 5000);
        let scaled = scale_covariance_rate(&rate, 10000);

        // Verify output is symmetric
        for i in 0..9 {
            for j in 0..9 {
                let diff = (scaled[(i, j)] - scaled[(j, i)]).abs();
                assert!(
                    diff < 1e-12,
                    "Rate operations broke symmetry at ({}, {}): diff={}",
                    i,
                    j,
                    diff
                );
            }
        }
    }
}
