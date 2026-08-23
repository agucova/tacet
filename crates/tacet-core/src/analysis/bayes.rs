//! Bayesian inference for timing leak detection.
//!
//! This module provides:
//! - 1D Bayesian inference on Wasserstein-1 distance
//! - Monte Carlo utilities for sampling

extern crate alloc;

use alloc::vec::Vec;
use core::f64::consts::PI;

use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::math;

/// Sample from standard normal using Box-Muller transform.
///
/// This is a public utility function that can be used by other modules
/// for Monte Carlo sampling from N(0, 1).
pub fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    math::sqrt(-2.0 * math::ln(u1)) * math::cos(2.0 * PI * u2)
}

/// Sample from a truncated normal distribution restricted to [0, ∞).
///
/// Uses the inverse CDF method (spec §3.4.4, implementation guide §5.2):
/// 1. Compute α = -μ/σ (standardized lower bound)
/// 2. Sample p ~ Uniform(Φ(α), 1)
/// 3. Return μ + σ·Φ⁻¹(p)
///
/// This is used for the half-t prior on δ, which requires δ ≥ 0
/// since W₁ distances are non-negative.
pub fn sample_truncated_normal_positive<R: Rng>(rng: &mut R, mu: f64, var: f64) -> f64 {
    let sigma = math::sqrt(var);
    if sigma < 1e-15 {
        return mu.max(0.0);
    }
    let alpha = -mu / sigma;
    let phi_alpha = math::normal_cdf(alpha);

    // Sample p ~ Uniform(Φ(α), 1)
    let u: f64 = rng.random();
    let p = phi_alpha + u * (1.0 - phi_alpha);

    // Clamp p to avoid numerical issues at the boundary
    let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);

    let result = mu + sigma * math::normal_quantile(p_clamped);
    // Ensure non-negative (guard against floating-point edge cases)
    result.max(0.0)
}

/// Result from 1D Bayesian inference on Wasserstein-1 distance.
///
/// Models: W₁_obs ~ t(δ, var_n, ν_ℓ), δ ~ half-t(ν=4, σ_t), δ ≥ 0
#[derive(Debug, Clone)]
pub struct BayesW1Result {
    /// Posterior probability of a significant leak: P(δ > θ | W₁_obs).
    pub leak_probability: f64,

    /// Posterior mean of δ (effect size in nanoseconds).
    pub w1_post: f64,

    /// Posterior variance of δ.
    pub var_post: f64,

    /// Posterior samples of δ from Gibbs sampler.
    pub w1_draws: Vec<f64>,

    /// 95% credible interval for δ: (2.5th, 97.5th percentile).
    pub credible_interval: (f64, f64),
}

/// Compute 1D Bayesian inference on Wasserstein-1 distance.
///
/// # Model
///
/// Likelihood: W₁_obs ~ t(δ, var_n, ν_ℓ) — Student-t for robustness
///
/// Prior: δ ~ half-t(ν=4, σ_t)
///
/// The Student-t likelihood is more robust to variance underestimation
/// than Normal likelihood. It's represented as:
/// - W₁_obs | δ, κ ~ N(δ, var_n/κ)
/// - κ ~ Gamma(ν_ℓ/2, ν_ℓ/2)
///   Marginalizing κ gives Student-t(δ, var_n, ν_ℓ).
///
/// The half-t prior is represented as a scale mixture of half-normals:
/// - λ ~ Gamma(ν/2, ν/2) = Gamma(2, 2)
/// - δ | λ ~ half-N(0, σ_t²/λ), i.e., N(0, σ_t²/λ) truncated to [0, ∞)
///
/// # Gibbs Sampling (spec §3.4.4)
///
/// 1. Sample κ ~ Gamma(ν_ℓ/2 + 1/2, ν_ℓ/2 + (W₁_obs - δ)²/(2·var_n))
/// 2. Sample λ ~ Gamma(ν/2 + 1/2, ν/2 + δ²/(2σ_t²))
/// 3. Sample δ | λ, κ ~ TruncatedNormal(μ_post, σ²_post, lower=0) where:
///    - posterior_precision = λ/σ_t² + κ/var_n
///    - μ_post = (W₁_obs · κ/var_n) / posterior_precision
///    - σ²_post = 1 / posterior_precision
///
/// The truncation to [0, ∞) enforces the non-negativity constraint from
/// the half-t prior, since W₁ distances are non-negative.
///
/// # Arguments
///
/// * `w1_obs` - Observed W₁ distance (can be negative from debiasing)
/// * `var_n` - Likelihood variance (W₁ variance scaled for sample size)
/// * `sigma_t` - Prior scale parameter
/// * `theta` - Minimum effect of concern (threshold)
/// * `seed` - Random seed for reproducibility
/// * `nu_likelihood` - Degrees of freedom for Student-t likelihood (default 4.0)
///
/// # Returns
///
/// `BayesW1Result` with posterior statistics and leak probability.
pub fn compute_bayes_1d(
    w1_obs: f64,
    var_n: f64,
    sigma_t: f64,
    theta: f64,
    seed: u64,
    nu_likelihood: f64,
) -> BayesW1Result {
    const N_ITER: usize = 5000;
    const BURN_IN: usize = 1000;
    const NU: f64 = 4.0; // Prior degrees of freedom (half-t prior)

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // Initialize δ to max(0, w1_obs) since δ ≥ 0 under the half-t prior
    let mut delta = w1_obs.max(0.0);
    // κ is sampled in each iteration (no initialization needed)
    let mut kappa;

    let mut delta_draws = Vec::with_capacity(N_ITER - BURN_IN);
    let mut leak_count = 0;

    // Gibbs sampling with Student-t likelihood via κ augmentation
    for iter in 0..N_ITER {
        // Step 1: Sample κ ~ Gamma(ν_ℓ/2 + 1/2, ν_ℓ/2 + (w1_obs - δ)²/(2*var_n))
        // This gives Student-t likelihood when marginalized
        let shape_kappa = nu_likelihood / 2.0 + 0.5;
        let residual = w1_obs - delta;
        let rate_kappa = nu_likelihood / 2.0 + (residual * residual) / (2.0 * var_n);
        kappa = sample_gamma(&mut rng, shape_kappa, rate_kappa);

        // Step 2: Sample λ ~ Gamma(ν/2 + 1/2, ν/2 + δ²/(2σ_t²))
        // This gives half-t prior when marginalized
        let shape_lambda = NU / 2.0 + 0.5;
        let rate_lambda = NU / 2.0 + (delta * delta) / (2.0 * sigma_t * sigma_t);
        let lambda = sample_gamma(&mut rng, shape_lambda, rate_lambda);

        // Step 3: Sample δ | λ, κ ~ TruncatedNormal(μ_post, σ²_post, lower=0)
        // The half-t prior constrains δ ≥ 0 (W₁ distances are non-negative).
        // Spec §3.4.4: "truncated normal, positive only"
        let posterior_precision = lambda / (sigma_t * sigma_t) + kappa / var_n;
        let posterior_mean = (w1_obs * kappa / var_n) / posterior_precision;
        let posterior_var = 1.0 / posterior_precision;

        delta = sample_truncated_normal_positive(&mut rng, posterior_mean, posterior_var);

        // Store samples after burn-in
        if iter >= BURN_IN {
            delta_draws.push(delta);

            // Count leak exceedances: P(δ > θ | data) per spec §3.4.5
            if delta > theta {
                leak_count += 1;
            }
        }
    }

    // Compute posterior statistics
    let n_kept = delta_draws.len();
    let leak_probability = leak_count as f64 / n_kept as f64;

    let w1_post = delta_draws.iter().sum::<f64>() / n_kept as f64;
    let var_post = delta_draws
        .iter()
        .map(|d| math::sq(d - w1_post))
        .sum::<f64>()
        / (n_kept - 1) as f64;

    // Compute 95% credible interval
    let mut sorted_draws = delta_draws.clone();
    sorted_draws.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = math::round(n_kept as f64 * 0.025) as usize;
    let hi_idx = math::round(n_kept as f64 * 0.975) as usize;
    let credible_interval = (
        sorted_draws[lo_idx.min(n_kept - 1)],
        sorted_draws[hi_idx.min(n_kept - 1)],
    );

    BayesW1Result {
        leak_probability,
        w1_post,
        var_post,
        w1_draws: delta_draws,
        credible_interval,
    }
}

/// Sample from Gamma(shape, rate) using Marsaglia-Tsang method.
///
/// Sample from Gamma(shape, rate) distribution.
///
/// This is a public utility function that can be used by other modules
/// for Monte Carlo sampling from Gamma distributions.
///
/// For shape ≥ 1, uses the Marsaglia-Tsang algorithm.
/// For shape < 1, uses the transformation: Gamma(shape, rate) = Gamma(shape+1, rate) × U^(1/shape).
pub fn sample_gamma<R: Rng>(rng: &mut R, shape: f64, rate: f64) -> f64 {
    if shape < 1.0 {
        // Use transformation for shape < 1
        let g = sample_gamma_marsaglia_tsang(rng, shape + 1.0);
        let u: f64 = rng.random();
        return g * math::pow(u, 1.0 / shape) / rate;
    }

    sample_gamma_marsaglia_tsang(rng, shape) / rate
}

/// Marsaglia-Tsang method for Gamma(shape, 1) with shape ≥ 1.
fn sample_gamma_marsaglia_tsang<R: Rng>(rng: &mut R, shape: f64) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / math::sqrt(9.0 * d);

    loop {
        let z = sample_standard_normal(rng);
        let v = math::cube(1.0 + c * z);

        if v <= 0.0 {
            continue;
        }

        let u: f64 = rng.random();
        let z_sq = z * z;

        // Accept if u < 1 - 0.0331 * z^4 (quick acceptance)
        if u < 1.0 - 0.0331 * z_sq * z_sq {
            return d * v;
        }

        // Reject if log(u) ≥ 0.5 * z^2 + d * (1 - v + log(v))
        if math::ln(u) < 0.5 * z_sq + d * (1.0 - v + math::ln(v)) {
            return d * v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayes_1d_determinism() {
        let w1_obs = 5.0;
        let var_n = 10.0;
        let sigma_t = 20.0;
        let theta = 3.0;

        let result1 = compute_bayes_1d(w1_obs, var_n, sigma_t, theta, 42, 4.0);
        let result2 = compute_bayes_1d(w1_obs, var_n, sigma_t, theta, 42, 4.0);

        assert_eq!(
            result1.leak_probability, result2.leak_probability,
            "Same seed should give same result"
        );
        assert_eq!(
            result1.w1_post, result2.w1_post,
            "Posterior mean should be deterministic"
        );
    }

    #[test]
    fn test_bayes_1d_handles_negative_w1() {
        // Negative W₁ can occur from debiasing; δ is still constrained ≥ 0
        // by the half-t prior (truncated normal sampling)
        let w1_obs = -3.0;
        let var_n = 5.0;
        let sigma_t = 10.0;
        let theta = 2.0;

        let result = compute_bayes_1d(w1_obs, var_n, sigma_t, theta, 42, 4.0);

        // δ draws are constrained to [0, ∞) so posterior mean must be ≥ 0
        assert!(
            result.w1_post >= 0.0,
            "Posterior mean should be non-negative (half-t prior), got {}",
            result.w1_post
        );
        // With negative observation and truncation, posterior should be near zero
        assert!(
            result.w1_post < 5.0,
            "Posterior should be small for negative observation, got {}",
            result.w1_post
        );
        assert!(
            result.leak_probability >= 0.0 && result.leak_probability <= 1.0,
            "Leak probability should be valid"
        );
        // All draws should be non-negative
        assert!(
            result.w1_draws.iter().all(|&d| d >= 0.0),
            "All δ draws should be non-negative under half-t prior"
        );
    }

    #[test]
    fn test_bayes_1d_threshold_sensitivity() {
        let w1_obs = 10.0;
        let var_n = 2.0;
        let sigma_t = 15.0;
        let seed = 42;

        // Small threshold should give high leak probability
        let result_small_theta = compute_bayes_1d(w1_obs, var_n, sigma_t, 1.0, seed, 4.0);

        // Large threshold should give low leak probability
        let result_large_theta = compute_bayes_1d(w1_obs, var_n, sigma_t, 20.0, seed, 4.0);

        assert!(
            result_small_theta.leak_probability > result_large_theta.leak_probability,
            "Smaller threshold should yield higher leak probability"
        );
    }

    #[test]
    fn test_bayes_1d_credible_interval() {
        let w1_obs = 5.0;
        let var_n = 10.0;
        let sigma_t = 20.0;
        let theta = 3.0;

        let result = compute_bayes_1d(w1_obs, var_n, sigma_t, theta, 42, 4.0);

        let (lo, hi) = result.credible_interval;

        // CI should be ordered
        assert!(lo < hi, "CI bounds should be ordered");

        // Posterior mean should be within CI (usually)
        assert!(
            result.w1_post >= lo && result.w1_post <= hi,
            "Posterior mean should be within 95% CI"
        );
    }

    #[test]
    fn test_student_t_robustness() {
        // Test that Student-t likelihood is more robust than Normal
        // by comparing with different nu_likelihood values
        let w1_obs = 10.0;
        let var_n = 5.0; // Underestimated variance
        let sigma_t = 20.0;
        let theta = 5.0;
        let seed = 42;

        // Student-t with df=4 (robust)
        let result_t4 = compute_bayes_1d(w1_obs, var_n, sigma_t, theta, seed, 4.0);

        // Student-t with df=100 (approximately Normal)
        let result_t100 = compute_bayes_1d(w1_obs, var_n, sigma_t, theta, seed, 100.0);

        // Both should give valid probabilities
        assert!(
            result_t4.leak_probability >= 0.0 && result_t4.leak_probability <= 1.0,
            "t(4) should give valid probability"
        );
        assert!(
            result_t100.leak_probability >= 0.0 && result_t100.leak_probability <= 1.0,
            "t(100) should give valid probability"
        );

        // Posterior means should be similar (same data)
        let mean_diff = (result_t4.w1_post - result_t100.w1_post).abs();
        assert!(
            mean_diff < 2.0,
            "Posterior means should be similar: t(4)={:.2}, t(100)={:.2}",
            result_t4.w1_post,
            result_t100.w1_post
        );
    }

    #[test]
    fn test_kappa_augmentation_convergence() {
        // Test that κ augmentation (Student-t likelihood) converges correctly
        // even with moderately underestimated variance
        let w1_obs = 15.0;
        let var_n = 3.0; // Intentionally underestimated
        let sigma_t = 25.0;
        let theta = 8.0;
        let seed = 123;

        let result = compute_bayes_1d(w1_obs, var_n, sigma_t, theta, seed, 4.0);

        // Should produce reasonable posterior estimate
        assert!(
            result.w1_post > 0.0 && result.w1_post < 30.0,
            "Posterior mean should be reasonable: {:.2}",
            result.w1_post
        );

        // Should have reasonable uncertainty
        assert!(
            result.var_post > 0.0 && result.var_post < 1000.0,
            "Posterior variance should be reasonable: {:.2}",
            result.var_post
        );

        // Should give high leak probability (w1_obs=15 >> theta=8)
        assert!(
            result.leak_probability > 0.7,
            "Should detect clear leak: P(leak)={:.3}",
            result.leak_probability
        );
    }
}
