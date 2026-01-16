//! Posterior distribution representation for Bayesian inference.
//!
//! The posterior is Gaussian: β | Δ ~ N(β_mean, β_cov) where:
//! - β = (μ, τ) is the effect vector (shift, tail)
//! - Δ is the observed quantile difference vector
//!
//! See spec Section 2.5 for the Bayesian model details.

use crate::types::{Matrix2, Vector2};

/// Posterior distribution parameters for the effect vector β = (μ, τ).
///
/// The posterior is Gaussian: β | Δ ~ N(β_mean, β_cov) where:
/// - μ is the uniform shift component
/// - τ is the tail effect component
///
/// See spec Section 2.5 for the Bayesian model details.
#[derive(Clone, Debug)]
pub struct Posterior {
    /// Posterior mean of β = (μ, τ).
    pub beta_mean: Vector2,

    /// Posterior covariance of β.
    pub beta_cov: Matrix2,

    /// Leak probability: P(max_k |(Xβ)_k| > θ | Δ).
    /// Computed via Monte Carlo integration over the posterior.
    pub leak_probability: f64,

    /// Number of samples used in this posterior computation.
    pub n: usize,
}

impl Posterior {
    /// Create a new posterior with given parameters.
    pub fn new(beta_mean: Vector2, beta_cov: Matrix2, leak_probability: f64, n: usize) -> Self {
        Self {
            beta_mean,
            beta_cov,
            leak_probability,
            n,
        }
    }

    /// Get the shift component (μ) from the posterior mean.
    #[inline]
    pub fn shift_ns(&self) -> f64 {
        self.beta_mean[0]
    }

    /// Get the tail component (τ) from the posterior mean.
    #[inline]
    pub fn tail_ns(&self) -> f64 {
        self.beta_mean[1]
    }

    /// Get the standard error of the shift component.
    #[inline]
    pub fn shift_se(&self) -> f64 {
        self.beta_cov[(0, 0)].sqrt()
    }

    /// Get the standard error of the tail component.
    #[inline]
    pub fn tail_se(&self) -> f64 {
        self.beta_cov[(1, 1)].sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posterior_accessors() {
        let posterior = Posterior::new(
            Vector2::new(10.0, 5.0),
            Matrix2::new(4.0, 0.0, 0.0, 1.0),
            0.75,
            1000,
        );

        assert_eq!(posterior.shift_ns(), 10.0);
        assert_eq!(posterior.tail_ns(), 5.0);
        assert_eq!(posterior.shift_se(), 2.0); // sqrt(4.0)
        assert_eq!(posterior.tail_se(), 1.0); // sqrt(1.0)
        assert_eq!(posterior.leak_probability, 0.75);
        assert_eq!(posterior.n, 1000);
    }

    #[test]
    fn test_posterior_clone() {
        let posterior = Posterior::new(
            Vector2::new(5.0, 3.0),
            Matrix2::new(1.0, 0.0, 0.0, 1.0),
            0.5,
            500,
        );

        let cloned = posterior.clone();
        assert_eq!(cloned.shift_ns(), posterior.shift_ns());
        assert_eq!(cloned.tail_ns(), posterior.tail_ns());
        assert_eq!(cloned.leak_probability, posterior.leak_probability);
    }
}
