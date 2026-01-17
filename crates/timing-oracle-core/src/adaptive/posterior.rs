//! Posterior distribution representation for Bayesian inference.
//!
//! The posterior is Gaussian: β | Δ ~ N(β_mean, β_cov) where:
//! - β = (μ, τ) is the effect vector (shift, tail)
//! - Δ is the observed quantile difference vector
//!
//! See spec Section 2.5 for the Bayesian model details.

use crate::analysis::effect::classify_pattern;
use crate::math::sqrt;
use crate::result::{EffectEstimate, MeasurementQuality};
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

    /// Model fit Q statistic (spec §2.3.3, §2.6 Gate 8).
    ///
    /// Q = r' Σ_n^{-1} r where r = Δ - Xβ_post is the residual.
    /// Under the 2D model, Q ~ chi-squared(7). High values indicate
    /// the 2D (shift + tail) model doesn't fit the data well.
    pub model_fit_q: f64,
}

impl Posterior {
    /// Create a new posterior with given parameters.
    pub fn new(
        beta_mean: Vector2,
        beta_cov: Matrix2,
        leak_probability: f64,
        n: usize,
        model_fit_q: f64,
    ) -> Self {
        Self {
            beta_mean,
            beta_cov,
            leak_probability,
            n,
            model_fit_q,
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
        sqrt(self.beta_cov[(0, 0)])
    }

    /// Get the standard error of the tail component.
    #[inline]
    pub fn tail_se(&self) -> f64 {
        sqrt(self.beta_cov[(1, 1)])
    }

    /// Build an EffectEstimate from this posterior.
    ///
    /// This computes the credible interval and classifies the effect pattern
    /// based on the posterior mean and covariance.
    pub fn to_effect_estimate(&self) -> EffectEstimate {
        let pattern = classify_pattern(&self.beta_mean, &self.beta_cov);

        // Compute credible interval from posterior covariance
        let shift_std = self.shift_se();
        let tail_std = self.tail_se();
        let total_effect =
            sqrt(self.shift_ns() * self.shift_ns() + self.tail_ns() * self.tail_ns());
        let total_std = sqrt((shift_std * shift_std + tail_std * tail_std) / 2.0);

        let ci_low = (total_effect - 1.96 * total_std).max(0.0);
        let ci_high = total_effect + 1.96 * total_std;

        EffectEstimate {
            shift_ns: self.shift_ns(),
            tail_ns: self.tail_ns(),
            credible_interval_ns: (ci_low, ci_high),
            pattern,
            interpretation_caveat: None,
        }
    }

    /// Get measurement quality based on the effect standard error.
    ///
    /// Quality is determined by the minimum detectable effect (MDE),
    /// which is approximately 2x the effect standard error.
    pub fn measurement_quality(&self) -> MeasurementQuality {
        let effect_se = self.shift_se();
        MeasurementQuality::from_mde_ns(effect_se * 2.0)
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
            5.0, // model_fit_q
        );

        assert_eq!(posterior.shift_ns(), 10.0);
        assert_eq!(posterior.tail_ns(), 5.0);
        assert_eq!(posterior.shift_se(), 2.0); // sqrt(4.0)
        assert_eq!(posterior.tail_se(), 1.0); // sqrt(1.0)
        assert_eq!(posterior.leak_probability, 0.75);
        assert_eq!(posterior.n, 1000);
        assert_eq!(posterior.model_fit_q, 5.0);
    }

    #[test]
    fn test_posterior_clone() {
        let posterior = Posterior::new(
            Vector2::new(5.0, 3.0),
            Matrix2::new(1.0, 0.0, 0.0, 1.0),
            0.5,
            500,
            3.0, // model_fit_q
        );

        let cloned = posterior.clone();
        assert_eq!(cloned.shift_ns(), posterior.shift_ns());
        assert_eq!(cloned.tail_ns(), posterior.tail_ns());
        assert_eq!(cloned.leak_probability, posterior.leak_probability);
    }
}
