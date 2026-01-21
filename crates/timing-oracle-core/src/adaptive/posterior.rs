//! Posterior distribution representation for Bayesian inference.
//!
//! The posterior is Gaussian: δ | Δ ~ N(δ_post, Λ_post) where:
//! - δ ∈ ℝ⁹ is the per-decile timing difference vector
//! - Δ is the observed quantile difference vector
//!
//! For interpretability, the 9D posterior is projected to 2D (shift, tail)
//! using GLS: β_proj = (X'Σ_n⁻¹X)⁻¹ X'Σ_n⁻¹ δ_post

extern crate alloc;
use alloc::vec::Vec;

use crate::math::sqrt;
use crate::result::{EffectEstimate, EffectPattern, MeasurementQuality};
use crate::types::{Matrix2, Matrix9, Vector2, Vector9};

// Note: classify_pattern from analysis::effect is no longer used here.
// We now use classify_pattern_from_draws() which uses β draws directly.

/// Posterior distribution parameters for the 9D effect vector δ.
///
/// The posterior is Gaussian: δ | Δ ~ N(δ_post, Λ_post) where each δ_k
/// represents the timing difference at decile k.
///
/// The 2D projection β_proj = (μ, τ) provides shift/tail decomposition.
///
/// Uses Student's t prior (ν=4) via Gibbs sampling for robust inference.
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

    /// 2D projection covariance (empirical, from draws).
    pub beta_proj_cov: Matrix2,

    /// Retained β draws for dominance-based classification (spec §3.4.6).
    /// Each draw is β^(s) = A·δ^(s) where A is the GLS projection matrix.
    pub beta_draws: Vec<Vector2>,

    /// Leak probability: P(max_k |δ_k| > θ | Δ).
    /// Computed via Monte Carlo integration over the 9D posterior.
    pub leak_probability: f64,

    /// Projection mismatch Q statistic.
    /// Q = r'Σ_n⁻¹r where r = δ_post - X β_proj.
    /// High values indicate the 2D summary is approximate.
    pub projection_mismatch_q: f64,

    /// Number of samples used in this posterior computation.
    pub n: usize,

    // ==================== Gibbs sampler fields ====================

    /// Posterior mean of latent scale λ.
    /// `None` if using simple posterior (no Gibbs sampler).
    pub lambda_mean: Option<f64>,

    /// Whether the Gibbs sampler's lambda chain mixed well.
    /// `None` if using simple posterior.
    /// When `Some(false)`, indicates potential posterior unreliability.
    pub lambda_mixing_ok: Option<bool>,

    /// Posterior mean of likelihood precision κ.
    /// `None` if using simple posterior.
    pub kappa_mean: Option<f64>,

    /// Coefficient of variation of κ.
    /// `None` if using simple posterior.
    pub kappa_cv: Option<f64>,

    /// Effective sample size of κ chain.
    /// `None` if using simple posterior.
    pub kappa_ess: Option<f64>,

    /// Whether the Gibbs sampler's kappa chain mixed well.
    /// `None` if using simple posterior.
    pub kappa_mixing_ok: Option<bool>,
}

impl Posterior {
    /// Create a new posterior with given parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        delta_post: Vector9,
        lambda_post: Matrix9,
        beta_proj: Vector2,
        beta_proj_cov: Matrix2,
        beta_draws: Vec<Vector2>,
        leak_probability: f64,
        projection_mismatch_q: f64,
        n: usize,
    ) -> Self {
        Self {
            delta_post,
            lambda_post,
            beta_proj,
            beta_proj_cov,
            beta_draws,
            leak_probability,
            projection_mismatch_q,
            n,
            lambda_mean: None,        // v5.4: no Gibbs sampler
            lambda_mixing_ok: None,   // v5.4: no Gibbs sampler
            kappa_mean: None,         // v5.6: no Gibbs sampler
            kappa_cv: None,           // v5.6: no Gibbs sampler
            kappa_ess: None,          // v5.6: no Gibbs sampler
            kappa_mixing_ok: None,    // v5.6: no Gibbs sampler
        }
    }

    /// Create a new posterior with Gibbs sampler diagnostics (v5.4, v5.6).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_gibbs(
        delta_post: Vector9,
        lambda_post: Matrix9,
        beta_proj: Vector2,
        beta_proj_cov: Matrix2,
        beta_draws: Vec<Vector2>,
        leak_probability: f64,
        projection_mismatch_q: f64,
        n: usize,
        lambda_mean: f64,
        lambda_mixing_ok: bool,
        kappa_mean: f64,
        kappa_cv: f64,
        kappa_ess: f64,
        kappa_mixing_ok: bool,
    ) -> Self {
        Self {
            delta_post,
            lambda_post,
            beta_proj,
            beta_proj_cov,
            beta_draws,
            leak_probability,
            projection_mismatch_q,
            n,
            lambda_mean: Some(lambda_mean),
            lambda_mixing_ok: Some(lambda_mixing_ok),
            kappa_mean: Some(kappa_mean),
            kappa_cv: Some(kappa_cv),
            kappa_ess: Some(kappa_ess),
            kappa_mixing_ok: Some(kappa_mixing_ok),
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
    /// using dominance-based classification from the β draws (spec §3.4.6).
    pub fn to_effect_estimate(&self) -> EffectEstimate {
        // Classify pattern using dominance-based approach from draws
        let pattern = self.classify_pattern_from_draws();

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

    /// Classify effect pattern using dominance-based approach from β draws (spec §3.4.6).
    ///
    /// Uses posterior draws to compute dominance probabilities rather than
    /// relying on "statistical significance" (|effect| > 2×SE), which is
    /// brittle under covariance regularization.
    ///
    /// Classification rules (dominance is primary):
    /// - UniformShift: shift dominates tail (|μ| ≥ 5|τ| with ≥80% probability)
    /// - TailEffect: tail dominates shift (|τ| ≥ 5|μ| with ≥80% probability)
    /// - Mixed: both practically significant and neither dominates
    /// - Indeterminate: effect too small or uncertain to classify
    fn classify_pattern_from_draws(&self) -> EffectPattern {
        // Dominance ratio threshold: one effect must be 5x larger to dominate
        const DOMINANCE_RATIO: f64 = 5.0;
        // Probability threshold for dominance classification
        const DOMINANCE_PROB: f64 = 0.80;
        // Threshold for effect to be "practically significant" (absolute, in ns)
        // Effects smaller than this are considered noise
        const MIN_SIGNIFICANT_NS: f64 = 10.0;

        if self.beta_draws.is_empty() {
            // Fallback to point estimate if no draws available
            return self.classify_pattern_point_estimate();
        }

        let n = self.beta_draws.len() as f64;

        // Count draws where each condition holds
        let mut shift_significant_count = 0;
        let mut tail_significant_count = 0;
        let mut shift_dominates_count = 0;
        let mut tail_dominates_count = 0;

        for beta in &self.beta_draws {
            let shift_abs = beta[0].abs();
            let tail_abs = beta[1].abs();

            // Check if components are practically significant (absolute threshold)
            if shift_abs > MIN_SIGNIFICANT_NS {
                shift_significant_count += 1;
            }
            if tail_abs > MIN_SIGNIFICANT_NS {
                tail_significant_count += 1;
            }

            // Check dominance (with safety for near-zero values)
            if tail_abs < 1e-12 || shift_abs >= DOMINANCE_RATIO * tail_abs {
                shift_dominates_count += 1;
            }
            if shift_abs < 1e-12 || tail_abs >= DOMINANCE_RATIO * shift_abs {
                tail_dominates_count += 1;
            }
        }

        // Compute probabilities
        let p_shift_significant = shift_significant_count as f64 / n;
        let p_tail_significant = tail_significant_count as f64 / n;
        let p_shift_dominates = shift_dominates_count as f64 / n;
        let p_tail_dominates = tail_dominates_count as f64 / n;

        // Classification: dominance is the primary criterion
        // If one component dominates in most draws, that determines the pattern
        if p_shift_dominates >= DOMINANCE_PROB {
            EffectPattern::UniformShift
        } else if p_tail_dominates >= DOMINANCE_PROB {
            EffectPattern::TailEffect
        } else if p_shift_significant >= DOMINANCE_PROB && p_tail_significant >= DOMINANCE_PROB {
            // Neither dominates but both are significant -> Mixed
            EffectPattern::Mixed
        } else {
            EffectPattern::Indeterminate
        }
    }

    /// Fallback classification using point estimates when no draws available.
    fn classify_pattern_point_estimate(&self) -> EffectPattern {
        const DOMINANCE_RATIO: f64 = 5.0;

        let shift_abs = self.shift_ns().abs();
        let tail_abs = self.tail_ns().abs();
        let shift_se = self.shift_se();
        let tail_se = self.tail_se();

        // Check statistical significance (|effect| > 2×SE)
        let shift_significant = shift_abs > 2.0 * shift_se;
        let tail_significant = tail_abs > 2.0 * tail_se;

        match (shift_significant, tail_significant) {
            (true, false) => EffectPattern::UniformShift,
            (false, true) => EffectPattern::TailEffect,
            (true, true) => {
                // Both significant - check dominance
                if shift_abs > tail_abs * DOMINANCE_RATIO {
                    EffectPattern::UniformShift
                } else if tail_abs > shift_abs * DOMINANCE_RATIO {
                    EffectPattern::TailEffect
                } else {
                    EffectPattern::Mixed
                }
            }
            (false, false) => EffectPattern::Indeterminate,
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
            Vec::new(), // beta_draws
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
            Vec::new(), // beta_draws
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
