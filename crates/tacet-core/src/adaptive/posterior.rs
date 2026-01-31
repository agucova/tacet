//! Posterior distribution representation for Bayesian inference.
//!
//! The posterior is Gaussian: δ | Δ ~ N(δ_post, Λ_post) where:
//! - δ ∈ ℝ⁹ is the per-decile timing difference vector
//! - Δ is the observed quantile difference vector
//!
//! Effect estimation uses max_k |δ_k| as the primary metric (spec §5.2).

extern crate alloc;
use alloc::vec::Vec;

use crate::analysis::{compute_effect_estimate, compute_effect_estimate_analytical};
use crate::math::sqrt;
use crate::result::{EffectEstimate, MeasurementQuality};
use crate::types::{Matrix9, Vector9};

/// Posterior distribution parameters for the 9D effect vector δ.
///
/// The posterior is Gaussian: δ | Δ ~ N(δ_post, Λ_post) where each δ_k
/// represents the timing difference at decile k.
///
/// Uses Student's t prior (ν=4) via Gibbs sampling for robust inference.
#[derive(Clone, Debug)]
pub struct Posterior {
    /// 9D posterior mean δ_post in nanoseconds.
    pub delta_post: Vector9,

    /// 9D posterior covariance Λ_post.
    pub lambda_post: Matrix9,

    /// Retained δ draws from the Gibbs sampler.
    /// Used for effect estimation via `compute_effect_estimate`.
    pub delta_draws: Vec<Vector9>,

    /// Leak probability: P(max_k |δ_k| > θ | Δ).
    /// Computed via Monte Carlo integration over the 9D posterior.
    pub leak_probability: f64,

    /// Effect threshold used for leak probability computation.
    pub theta: f64,

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
        delta_draws: Vec<Vector9>,
        leak_probability: f64,
        theta: f64,
        n: usize,
    ) -> Self {
        Self {
            delta_post,
            lambda_post,
            delta_draws,
            leak_probability,
            theta,
            n,
            lambda_mean: None,      // v5.4: no Gibbs sampler
            lambda_mixing_ok: None, // v5.4: no Gibbs sampler
            kappa_mean: None,       // v5.6: no Gibbs sampler
            kappa_cv: None,         // v5.6: no Gibbs sampler
            kappa_ess: None,        // v5.6: no Gibbs sampler
            kappa_mixing_ok: None,  // v5.6: no Gibbs sampler
        }
    }

    /// Create a new posterior with Gibbs sampler diagnostics (v5.4, v5.6).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_gibbs(
        delta_post: Vector9,
        lambda_post: Matrix9,
        delta_draws: Vec<Vector9>,
        leak_probability: f64,
        theta: f64,
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
            delta_draws,
            leak_probability,
            theta,
            n,
            lambda_mean: Some(lambda_mean),
            lambda_mixing_ok: Some(lambda_mixing_ok),
            kappa_mean: Some(kappa_mean),
            kappa_cv: Some(kappa_cv),
            kappa_ess: Some(kappa_ess),
            kappa_mixing_ok: Some(kappa_mixing_ok),
        }
    }

    /// Get the max absolute effect across all deciles from posterior mean.
    pub fn max_effect_ns(&self) -> f64 {
        self.delta_post
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Build an EffectEstimate from this posterior.
    ///
    /// Uses delta draws if available, otherwise uses analytical approximation.
    pub fn to_effect_estimate(&self) -> EffectEstimate {
        if !self.delta_draws.is_empty() {
            compute_effect_estimate(&self.delta_draws, self.theta)
        } else {
            compute_effect_estimate_analytical(&self.delta_post, &self.lambda_post, self.theta)
        }
    }

    /// Get measurement quality based on the posterior uncertainty.
    ///
    /// Quality is determined by the minimum detectable effect (MDE),
    /// which is approximately the maximum marginal standard deviation.
    pub fn measurement_quality(&self) -> MeasurementQuality {
        // MDE is approximately max_k sqrt(λ_post[k,k])
        let max_se = (0..9)
            .map(|k| sqrt(self.lambda_post[(k, k)].max(1e-12)))
            .fold(0.0_f64, f64::max);
        MeasurementQuality::from_mde_ns(max_se * 2.0)
    }

    /// Convert to an FFI-friendly summary containing only scalar fields.
    pub fn to_summary(&self) -> crate::ffi_summary::PosteriorSummary {
        let effect = self.to_effect_estimate();

        crate::ffi_summary::PosteriorSummary {
            max_effect_ns: effect.max_effect_ns,
            ci_low_ns: effect.credible_interval_ns.0,
            ci_high_ns: effect.credible_interval_ns.1,
            leak_probability: self.leak_probability,
            n: self.n,
            lambda_mean: self.lambda_mean.unwrap_or(1.0),
            lambda_mixing_ok: self.lambda_mixing_ok.unwrap_or(true),
            kappa_mean: self.kappa_mean.unwrap_or(1.0),
            kappa_cv: self.kappa_cv.unwrap_or(0.0),
            kappa_ess: self.kappa_ess.unwrap_or(0.0),
            kappa_mixing_ok: self.kappa_mixing_ok.unwrap_or(true),
        }
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

        let posterior = Posterior::new(
            delta_post,
            lambda_post,
            Vec::new(), // delta_draws
            0.75,
            5.0, // theta
            1000,
        );

        assert_eq!(posterior.leak_probability, 0.75);
        assert_eq!(posterior.n, 1000);
        assert!((posterior.max_effect_ns() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_posterior_clone() {
        let delta_post = Vector9::from_row_slice(&[5.0; 9]);
        let lambda_post = Matrix9::identity();

        let posterior = Posterior::new(
            delta_post,
            lambda_post,
            Vec::new(), // delta_draws
            0.5,
            5.0, // theta
            500,
        );

        let cloned = posterior.clone();
        assert_eq!(cloned.leak_probability, posterior.leak_probability);
        assert_eq!(cloned.max_effect_ns(), posterior.max_effect_ns());
    }

    #[test]
    fn test_effect_estimate_from_draws() {
        let delta_post = Vector9::from_row_slice(&[10.0; 9]);
        let lambda_post = Matrix9::identity();

        // Create some sample draws
        let delta_draws: Vec<Vector9> = (0..100)
            .map(|_| Vector9::from_row_slice(&[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))
            .collect();

        let posterior = Posterior::new(delta_post, lambda_post, delta_draws, 0.99, 5.0, 1000);

        let effect = posterior.to_effect_estimate();
        assert!(effect.max_effect_ns > 9.0, "max effect should be around 10");
    }
}
