//! Posterior distribution representation for Bayesian inference.
//!
//! The posterior is Gaussian: δ | Δ ~ N(δ_post, Λ_post) where:
//! - δ ∈ ℝ⁹ is the per-decile timing difference vector
//! - Δ is the observed quantile difference vector
//!
//! For interpretability, the 9D posterior is projected to 2D (shift, tail)
//! using GLS: β_proj = (X'Σ_n⁻¹X)⁻¹ X'Σ_n⁻¹ δ_post

use crate::analysis::effect::classify_pattern;
use crate::math::sqrt;
use crate::result::{EffectEstimate, MeasurementQuality};
use crate::types::{Matrix2, Matrix9, Vector2, Vector9};

/// Posterior distribution parameters for the 9D effect vector δ.
///
/// The posterior is Gaussian: δ | Δ ~ N(δ_post, Λ_post) where each δ_k
/// represents the timing difference at decile k.
///
/// The 2D projection β_proj = (μ, τ) provides shift/tail decomposition.
#[derive(Clone, Debug)]
pub struct Posterior {
    /// 9D posterior mean δ_post in nanoseconds.
    pub delta_post: Vector9,

    /// 9D posterior covariance Λ_post.
    pub lambda_post: Matrix9,

    /// 2D GLS projection β = (μ, τ) for interpretability.
    /// - β[0] = μ (uniform shift)
    /// - β[1] = τ (tail effect)
    pub beta_proj: Vector2,

    /// 2D projection covariance.
    pub beta_proj_cov: Matrix2,

    /// Leak probability: P(max_k |δ_k| > θ | Δ).
    /// Computed via Monte Carlo integration over the 9D posterior.
    pub leak_probability: f64,

    /// Projection mismatch Q statistic.
    /// Q = r'Σ_n⁻¹r where r = δ_post - X β_proj.
    /// High values indicate the 2D summary is approximate.
    pub projection_mismatch_q: f64,

    /// Number of samples used in this posterior computation.
    pub n: usize,
}

impl Posterior {
    /// Create a new posterior with given parameters.
    pub fn new(
        delta_post: Vector9,
        lambda_post: Matrix9,
        beta_proj: Vector2,
        beta_proj_cov: Matrix2,
        leak_probability: f64,
        projection_mismatch_q: f64,
        n: usize,
    ) -> Self {
        Self {
            delta_post,
            lambda_post,
            beta_proj,
            beta_proj_cov,
            leak_probability,
            projection_mismatch_q,
            n,
        }
    }

    /// Get the shift component (μ) from the 2D projection.
    #[inline]
    pub fn shift_ns(&self) -> f64 {
        self.beta_proj[0]
    }

    /// Get the tail component (τ) from the 2D projection.
    #[inline]
    pub fn tail_ns(&self) -> f64 {
        self.beta_proj[1]
    }

    /// Get the standard error of the shift component.
    #[inline]
    pub fn shift_se(&self) -> f64 {
        sqrt(self.beta_proj_cov[(0, 0)])
    }

    /// Get the standard error of the tail component.
    #[inline]
    pub fn tail_se(&self) -> f64 {
        sqrt(self.beta_proj_cov[(1, 1)])
    }

    /// Get the max absolute effect across all deciles.
    pub fn max_effect_ns(&self) -> f64 {
        self.delta_post
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Build an EffectEstimate from this posterior.
    ///
    /// This computes the credible interval and classifies the effect pattern
    /// based on the 2D projection.
    pub fn to_effect_estimate(&self) -> EffectEstimate {
        let pattern = classify_pattern(&self.beta_proj, &self.beta_proj_cov);

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
        let delta_post =
            Vector9::from_row_slice(&[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        let lambda_post = Matrix9::identity();
        let beta_proj = Vector2::new(10.0, 0.0);
        let beta_proj_cov = Matrix2::new(4.0, 0.0, 0.0, 1.0);

        let posterior = Posterior::new(
            delta_post,
            lambda_post,
            beta_proj,
            beta_proj_cov,
            0.75,
            0.5,
            1000,
        );

        assert_eq!(posterior.shift_ns(), 10.0);
        assert_eq!(posterior.tail_ns(), 0.0);
        assert_eq!(posterior.shift_se(), 2.0); // sqrt(4.0)
        assert_eq!(posterior.tail_se(), 1.0); // sqrt(1.0)
        assert_eq!(posterior.leak_probability, 0.75);
        assert_eq!(posterior.n, 1000);
        assert!((posterior.max_effect_ns() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_posterior_clone() {
        let delta_post = Vector9::from_row_slice(&[5.0; 9]);
        let lambda_post = Matrix9::identity();
        let beta_proj = Vector2::new(5.0, 3.0);
        let beta_proj_cov = Matrix2::identity();

        let posterior = Posterior::new(
            delta_post,
            lambda_post,
            beta_proj,
            beta_proj_cov,
            0.5,
            1.0,
            500,
        );

        let cloned = posterior.clone();
        assert_eq!(cloned.shift_ns(), posterior.shift_ns());
        assert_eq!(cloned.tail_ns(), posterior.tail_ns());
        assert_eq!(cloned.leak_probability, posterior.leak_probability);
    }
}
