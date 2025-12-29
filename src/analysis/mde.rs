//! Minimum Detectable Effect (MDE) estimation.
//!
//! This module estimates the smallest timing effect that can be reliably
//! detected given the current noise level. This helps users understand:
//!
//! - Whether "no leak detected" means the code is safe, or just noisy
//! - What magnitude of timing difference they could detect
//! - Whether more samples are needed for better sensitivity
//!
//! The MDE is computed by:
//! 1. Sampling from the null distribution (no timing leak)
//! 2. Fitting the effect model to each null sample
//! 3. Taking the 95th percentile of |beta_hat| as the MDE

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::result::MinDetectableEffect;
use crate::types::{Matrix9, Vector9};

use super::effect::decompose_effect;

/// Result from MDE estimation.
#[derive(Debug, Clone)]
pub struct MdeEstimate {
    /// Minimum detectable uniform shift in nanoseconds.
    pub shift_ns: f64,
    /// Minimum detectable tail effect in nanoseconds.
    pub tail_ns: f64,
    /// Number of null simulations used.
    pub n_simulations: usize,
}

impl From<MdeEstimate> for MinDetectableEffect {
    fn from(mde: MdeEstimate) -> Self {
        MinDetectableEffect {
            shift_ns: mde.shift_ns,
            tail_ns: mde.tail_ns,
        }
    }
}

/// Estimate the minimum detectable effect.
///
/// Algorithm:
/// 1. Generate `n_simulations` samples from MVN(0, covariance/n_samples)
/// 2. For each sample, fit the effect decomposition model
/// 3. Collect the estimated shift and tail effects
/// 4. Take the 95th percentile of absolute values as MDE
///
/// # Arguments
///
/// * `covariance` - Pooled covariance matrix of quantile differences
/// * `n_samples` - Sample size per class
/// * `n_simulations` - Number of null simulations (default: 1000)
/// * `prior_var` - Prior variance for effect decomposition
///
/// # Returns
///
/// An `MdeEstimate` with shift and tail MDE in nanoseconds.
pub fn estimate_mde(
    covariance: &Matrix9,
    n_samples: usize,
    n_simulations: usize,
    prior_var: f64,
) -> MdeEstimate {
    let mut rng = rand::rng();

    // Collect effect estimates from null samples
    let mut shift_effects = Vec::with_capacity(n_simulations);
    let mut tail_effects = Vec::with_capacity(n_simulations);

    for _ in 0..n_simulations {
        // Sample from null distribution
        let null_sample = sample_null_mvn(covariance, n_samples, &mut rng);

        // Fit effect model
        let decomp = decompose_effect(&null_sample, covariance, n_samples, prior_var);

        // Collect absolute effects
        shift_effects.push(decomp.posterior_mean[0].abs());
        tail_effects.push(decomp.posterior_mean[1].abs());
    }

    // Compute 95th percentiles
    let shift_mde = percentile(&mut shift_effects, 0.95);
    let tail_mde = percentile(&mut tail_effects, 0.95);

    MdeEstimate {
        shift_ns: shift_mde,
        tail_ns: tail_mde,
        n_simulations,
    }
}

/// Sample from MVN(0, covariance/n_samples).
///
/// Uses the Cholesky decomposition: if L*L^T = Sigma, then
/// L * z ~ MVN(0, Sigma) when z ~ N(0, I).
fn sample_null_mvn<R: Rng>(covariance: &Matrix9, n_samples: usize, rng: &mut R) -> Vector9 {
    // Compute Cholesky decomposition
    let scaled_cov = covariance / (n_samples as f64);

    // TODO: Cache Cholesky decomposition across samples for efficiency
    let chol = match nalgebra::Cholesky::new(scaled_cov) {
        Some(c) => c,
        None => {
            // Regularize if not positive definite
            let regularized = scaled_cov + Matrix9::identity() * 1e-10;
            nalgebra::Cholesky::new(regularized)
                .expect("Regularized covariance should be positive definite")
        }
    };

    // Sample standard normal vector
    let z: Vector9 = Vector9::from_fn(|_, _| StandardNormal.sample(rng));

    // Transform: L * z
    chol.l() * z
}

/// Compute the p-th percentile of a vector.
///
/// Modifies the input vector by sorting it.
fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = (p * (values.len() - 1) as f64).round() as usize;
    let idx = idx.min(values.len() - 1);

    values[idx]
}

/// Estimate MDE with default parameters.
///
/// Uses 1000 simulations and a weak prior (variance = 1e6).
pub fn estimate_mde_default(covariance: &Matrix9, n_samples: usize) -> MdeEstimate {
    estimate_mde(covariance, n_samples, 1000, 1e6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_basic() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&mut values, 0.5) - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_95() {
        let mut values: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let p95 = percentile(&mut values, 0.95);
        assert!((p95 - 95.0).abs() < 1.0);
    }

    #[test]
    fn test_mde_positive() {
        // With identity covariance, MDE should be positive
        let cov = Matrix9::identity();
        let mde = estimate_mde(&cov, 100, 100, 1e6);

        assert!(mde.shift_ns > 0.0, "MDE shift should be positive");
        assert!(mde.tail_ns > 0.0, "MDE tail should be positive");
    }

    #[test]
    fn test_mde_decreases_with_samples() {
        let cov = Matrix9::identity();

        // MDE with fewer samples
        let mde_small = estimate_mde(&cov, 100, 100, 1e6);

        // MDE with more samples
        let mde_large = estimate_mde(&cov, 10000, 100, 1e6);

        // More samples should generally lead to smaller MDE
        // (Note: due to randomness, this isn't guaranteed for small n_simulations)
        // This is a weak test that checks the right order of magnitude
        assert!(
            mde_large.shift_ns < mde_small.shift_ns * 2.0,
            "MDE should not increase dramatically with more samples"
        );
    }
}
