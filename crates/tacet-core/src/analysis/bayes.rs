//! Bayesian inference for timing leak detection using 9D quantile model.
//!
//! This module computes the posterior probability of a timing leak using:
//! - 9D Gaussian model over quantile differences δ ∈ ℝ⁹
//! - Conjugate Gaussian prior with shaped covariance
//! - Closed-form posterior (no MCMC required)
//! - Monte Carlo integration for leak probability
//!
//! ## Model (spec §3)
//!
//! Likelihood: Δ | δ ~ N(δ, Σ_n)
//!
//! where:
//! - Δ is the observed quantile differences (9-vector)
//! - δ is the true per-decile timing differences (9-vector)
//! - Σ_n is the covariance matrix scaled for sample size
//!
//! ## Prior
//!
//! δ ~ N(0, Λ₀), where Λ₀ = σ²_prior × S
//! S = Σ_rate / tr(Σ_rate) (shaped to match empirical covariance structure)
//!
//! ## Posterior
//!
//! δ | Δ ~ N(δ_post, Λ_post)
//!
//! Computed via stable Cholesky solves (no explicit matrix inversion).
//!
//! ## Leak Probability
//!
//! P(leak | Δ) = P(max_k |δ_k| > θ_eff | Δ)
//!
//! Computed via Monte Carlo: draw samples from posterior, count exceedances.

extern crate alloc;

use alloc::vec::Vec;
use core::f64::consts::PI;

use nalgebra::Cholesky;
use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::math;
use crate::types::{Matrix9, Vector9};

/// Number of Monte Carlo samples for leak probability estimation.
const N_MONTE_CARLO: usize = 1000;

/// Result from Bayesian analysis using 9D inference with Gibbs sampling (v5.4+).
///
/// The posterior is computed using a Student's t prior (ν=4) via Gibbs sampling,
/// which is robust to correlation-induced pathologies.
#[derive(Debug, Clone)]
pub struct BayesResult {
    /// Posterior probability of a significant leak: P(max_k |δ_k| > θ | Δ).
    pub leak_probability: f64,

    /// 9D posterior mean δ_post in nanoseconds.
    pub delta_post: Vector9,

    /// 9D posterior covariance Λ_post.
    pub lambda_post: Matrix9,

    /// Retained δ draws from the Gibbs sampler.
    /// Used for effect estimation via `compute_effect_estimate`.
    pub delta_draws: Vec<Vector9>,

    /// 95% credible interval for max effect magnitude: (2.5th, 97.5th) percentiles.
    pub effect_magnitude_ci: (f64, f64),

    /// Whether the computation encountered numerical issues.
    /// If true, the posterior is set to the prior (maximally uncertain).
    pub is_clamped: bool,

    /// Covariance matrix used for inference (after regularization).
    pub sigma_n: Matrix9,

    // ==================== v5.4 Gibbs sampler diagnostics ====================
    /// Posterior mean of latent scale λ (v5.4 Gibbs).
    /// Only populated when using `compute_bayes_gibbs()`.
    pub lambda_mean: f64,

    /// Posterior standard deviation of λ (v5.4 Gibbs).
    pub lambda_sd: f64,

    /// Coefficient of variation: λ_sd / λ_mean (v5.4 Gibbs).
    pub lambda_cv: f64,

    /// Effective sample size of λ chain (v5.4 Gibbs).
    pub lambda_ess: f64,

    /// True if mixing diagnostics pass: CV ≥ 0.1 AND ESS ≥ 20 (v5.4 Gibbs).
    pub lambda_mixing_ok: bool,

    // ==================== v5.6 Gibbs sampler kappa diagnostics ====================
    /// v5.6: Posterior mean of likelihood precision κ.
    pub kappa_mean: f64,

    /// v5.6: Posterior standard deviation of κ.
    pub kappa_sd: f64,

    /// v5.6: Coefficient of variation: κ_sd / κ_mean.
    pub kappa_cv: f64,

    /// v5.6: Effective sample size of κ chain.
    pub kappa_ess: f64,

    /// v5.6: Whether κ mixing diagnostics pass: CV ≥ 0.1 AND ESS ≥ 20.
    pub kappa_mixing_ok: bool,
}

/// Compute Bayesian posterior using Student's t prior with Gibbs sampling (v5.4).
///
/// This replaces the v5.2 mixture prior, using a Student's t prior (ν=4) that
/// is more robust to correlation-induced pathologies. The t-prior is represented
/// as a scale mixture of Gaussians and sampled via Gibbs.
///
/// # Arguments
///
/// * `delta` - Observed quantile differences (9-vector)
/// * `sigma_n` - Covariance matrix scaled for inference sample size (Σ_rate / n)
/// * `sigma_t` - Calibrated Student's t prior scale
/// * `l_r` - Cholesky factor of correlation matrix R
/// * `theta` - Minimum effect of concern (threshold)
/// * `seed` - Random seed for Gibbs sampling reproducibility
///
/// # Returns
///
/// `BayesResult` with posterior from Gibbs sampling, including lambda diagnostics.
pub fn compute_bayes_gibbs(
    delta: &Vector9,
    sigma_n: &Matrix9,
    sigma_t: f64,
    l_r: &Matrix9,
    theta: f64,
    seed: Option<u64>,
) -> BayesResult {
    use super::gibbs::run_gibbs_inference;

    let regularized = add_jitter(*sigma_n);
    let actual_seed = seed.unwrap_or(crate::constants::DEFAULT_SEED);

    // Run Gibbs sampler
    let gibbs_result = run_gibbs_inference(delta, &regularized, sigma_t, l_r, theta, actual_seed);

    BayesResult {
        leak_probability: gibbs_result.leak_probability,
        delta_post: gibbs_result.delta_post,
        lambda_post: gibbs_result.lambda_post,
        delta_draws: gibbs_result.delta_draws,
        effect_magnitude_ci: gibbs_result.effect_magnitude_ci,
        is_clamped: false,
        sigma_n: regularized,
        // v5.4 Gibbs diagnostics
        lambda_mean: gibbs_result.lambda_mean,
        lambda_sd: gibbs_result.lambda_sd,
        lambda_cv: gibbs_result.lambda_cv,
        lambda_ess: gibbs_result.lambda_ess,
        lambda_mixing_ok: gibbs_result.lambda_mixing_ok,
        // v5.6 kappa diagnostics
        kappa_mean: gibbs_result.kappa_mean,
        kappa_sd: gibbs_result.kappa_sd,
        kappa_cv: gibbs_result.kappa_cv,
        kappa_ess: gibbs_result.kappa_ess,
        kappa_mixing_ok: gibbs_result.kappa_mixing_ok,
    }
}

/// Sample from standard normal using Box-Muller transform.
fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    math::sqrt(-2.0 * math::ln(u1)) * math::cos(2.0 * PI * u2)
}

/// Result from max effect CI computation for Research mode.
#[derive(Debug, Clone)]
pub struct MaxEffectCI {
    /// Posterior mean of max_k |δ_k|.
    pub mean: f64,
    /// 95% credible interval for max_k |δ_k|: (2.5th, 97.5th percentile).
    pub ci: (f64, f64),
}

/// Compute 95% CI for max effect: max_k |δ_k|.
///
/// Used by Research mode for stopping conditions.
pub fn compute_max_effect_ci(delta_post: &Vector9, lambda_post: &Matrix9, seed: u64) -> MaxEffectCI {
    let chol = match Cholesky::new(*lambda_post) {
        Some(c) => c,
        None => {
            let jittered = add_jitter(*lambda_post);
            match Cholesky::new(jittered) {
                Some(c) => c,
                None => {
                    return MaxEffectCI {
                        mean: 0.0,
                        ci: (0.0, 0.0),
                    };
                }
            }
        }
    };
    let l = chol.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut max_effects = Vec::with_capacity(N_MONTE_CARLO);
    let mut sum = 0.0;

    for _ in 0..N_MONTE_CARLO {
        let mut z = Vector9::zeros();
        for i in 0..9 {
            z[i] = sample_standard_normal(&mut rng);
        }

        let delta_sample = delta_post + l * z;
        let max_effect = delta_sample.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        max_effects.push(max_effect);
        sum += max_effect;
    }

    let mean = sum / N_MONTE_CARLO as f64;

    max_effects.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = math::round((N_MONTE_CARLO as f64) * 0.025) as usize;
    let hi_idx = math::round((N_MONTE_CARLO as f64) * 0.975) as usize;
    let ci = (
        max_effects[lo_idx.min(N_MONTE_CARLO - 1)],
        max_effects[hi_idx.min(N_MONTE_CARLO - 1)],
    );

    MaxEffectCI { mean, ci }
}

/// Apply variance floor regularization for numerical stability.
///
/// Ensures minimum diagonal value of 1% of mean variance.
pub fn add_jitter(mut sigma: Matrix9) -> Matrix9 {
    let trace: f64 = (0..9).map(|i| sigma[(i, i)]).sum();
    let mean_var = trace / 9.0;

    let min_var = (0.01 * mean_var).max(1e-10);
    let jitter = 1e-10 + mean_var * 1e-8;

    for i in 0..9 {
        sigma[(i, i)] = sigma[(i, i)].max(min_var) + jitter;
    }
    sigma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_effect_ci_basic() {
        let delta_post = Vector9::from_row_slice(&[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let lambda_post = Matrix9::identity() * 1.0;

        let ci = compute_max_effect_ci(&delta_post, &lambda_post, 42);

        // Max effect should be around 10ns (the first quantile has mean 10)
        assert!(ci.mean > 8.0, "mean should be around 10, got {}", ci.mean);
        assert!(ci.ci.0 < ci.mean, "CI lower should be below mean");
        assert!(ci.ci.1 > ci.mean, "CI upper should be above mean");
    }

    #[test]
    fn test_gibbs_determinism() {
        use crate::adaptive::calibrate_t_prior_scale;

        let delta = Vector9::from_row_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
        let sigma_n = Matrix9::identity() * 100.0;
        let sigma_rate = sigma_n * 1000.0;
        let theta = 10.0;

        let (sigma_t, l_r) = calibrate_t_prior_scale(&sigma_rate, theta, 1000, false, 42);

        let result1 = compute_bayes_gibbs(&delta, &sigma_n, sigma_t, &l_r, theta, Some(42));
        let result2 = compute_bayes_gibbs(&delta, &sigma_n, sigma_t, &l_r, theta, Some(42));

        assert_eq!(
            result1.leak_probability, result2.leak_probability,
            "Same seed should give same result"
        );
    }
}
