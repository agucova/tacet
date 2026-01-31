//! Effect estimation from posterior samples (spec §5.2).
//!
//! This module computes effect estimates from the 9D posterior over quantile
//! differences. The primary metric is max_k |δ_k| - the maximum absolute
//! effect across all deciles.
//!
//! ## Effect Reporting (spec §5.2)
//!
//! - `max_effect_ns`: Posterior mean of max_k |δ_k|
//! - `credible_interval_ns`: 95% CI for max|δ|
//! - `top_quantiles`: Top 2-3 quantiles by exceedance probability

extern crate alloc;

use alloc::vec::Vec;

use crate::constants::DECILES;
use crate::math;
use crate::result::{EffectEstimate, TopQuantile};
use crate::types::{Matrix9, Vector9};

/// Per-quantile statistics: (index, quantile_p, mean_ns, ci95_ns, exceed_prob).
type QuantileStats = (usize, f64, f64, (f64, f64), f64);

/// Compute effect estimate from delta draws (spec §5.2).
///
/// Takes posterior samples of the 9D effect vector δ and computes:
/// - max_effect_ns: posterior mean of max_k |δ_k|
/// - credible_interval_ns: 95% CI for max|δ|
/// - top_quantiles: top 2-3 quantiles by exceedance probability
///
/// # Arguments
///
/// * `delta_draws` - Posterior samples of δ ∈ ℝ⁹
/// * `theta` - Threshold for exceedance probability computation
///
/// # Returns
///
/// An `EffectEstimate` with max effect and top quantiles.
pub fn compute_effect_estimate(delta_draws: &[Vector9], theta: f64) -> EffectEstimate {
    if delta_draws.is_empty() {
        return EffectEstimate::default();
    }

    let n = delta_draws.len();

    // Compute max|δ| for each draw
    let mut max_effects: Vec<f64> = Vec::with_capacity(n);
    for delta in delta_draws {
        let max_abs = delta.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        max_effects.push(max_abs);
    }

    // Posterior mean of max|δ|
    let max_effect_ns = max_effects.iter().sum::<f64>() / n as f64;

    // 95% credible interval (2.5th and 97.5th percentiles)
    max_effects.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = ((n as f64 * 0.025).round() as usize).min(n - 1);
    let hi_idx = ((n as f64 * 0.975).round() as usize).min(n - 1);
    let credible_interval_ns = (max_effects[lo_idx], max_effects[hi_idx]);

    // Compute top quantiles by exceedance probability
    let top_quantiles = compute_top_quantiles(delta_draws, theta);

    EffectEstimate {
        max_effect_ns,
        credible_interval_ns,
        top_quantiles,
    }
}

/// Compute top 2-3 quantiles by exceedance probability.
///
/// For each quantile k, computes:
/// - mean_ns: posterior mean δ_k
/// - ci95_ns: 95% marginal CI for δ_k
/// - exceed_prob: P(|δ_k| > θ | data)
///
/// Returns the top quantiles (up to 3) with exceed_prob > 0.5.
pub fn compute_top_quantiles(delta_draws: &[Vector9], theta: f64) -> Vec<TopQuantile> {
    if delta_draws.is_empty() {
        return Vec::new();
    }

    let n = delta_draws.len();

    // Compute per-quantile statistics
    let mut quantile_stats: Vec<QuantileStats> = Vec::with_capacity(9);

    for k in 0..9 {
        // Extract draws for quantile k
        let mut values: Vec<f64> = delta_draws.iter().map(|d| d[k]).collect();

        // Mean
        let mean = values.iter().sum::<f64>() / n as f64;

        // 95% CI
        values.sort_by(|a, b| a.total_cmp(b));
        let lo_idx = ((n as f64 * 0.025).round() as usize).min(n - 1);
        let hi_idx = ((n as f64 * 0.975).round() as usize).min(n - 1);
        let ci = (values[lo_idx], values[hi_idx]);

        // Exceedance probability: P(|δ_k| > θ | data)
        let exceed_count = delta_draws.iter().filter(|d| d[k].abs() > theta).count();
        let exceed_prob = exceed_count as f64 / n as f64;

        quantile_stats.push((k, DECILES[k], mean, ci, exceed_prob));
    }

    // Sort by exceedance probability (descending)
    quantile_stats.sort_by(|a, b| b.4.total_cmp(&a.4));

    // Take top 2-3 with exceed_prob > 0.5
    quantile_stats
        .into_iter()
        .filter(|(_, _, _, _, exceed_prob)| *exceed_prob > 0.5)
        .take(3)
        .map(|(_, quantile_p, mean_ns, ci95_ns, exceed_prob)| TopQuantile {
            quantile_p,
            mean_ns,
            ci95_ns,
            exceed_prob,
        })
        .collect()
}

/// Compute effect estimate from posterior mean and covariance (analytical).
///
/// This is a faster alternative to `compute_effect_estimate` when only the
/// posterior mean and covariance are available (no draws).
///
/// # Arguments
///
/// * `delta_post` - Posterior mean δ_post
/// * `lambda_post` - Posterior covariance Λ_post
/// * `theta` - Threshold for exceedance probability
///
/// # Returns
///
/// An `EffectEstimate` with approximate max effect (uses mean of |δ_post|).
pub fn compute_effect_estimate_analytical(
    delta_post: &Vector9,
    lambda_post: &Matrix9,
    theta: f64,
) -> EffectEstimate {
    // Max absolute effect from posterior mean
    let max_effect_ns = delta_post.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

    // Approximate CI using marginal variances
    // This is a rough approximation - the true CI requires sampling
    let max_k = delta_post
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        .map(|(k, _)| k)
        .unwrap_or(0);

    let se = math::sqrt(lambda_post[(max_k, max_k)].max(1e-12));
    let ci_low = (max_effect_ns - 1.96 * se).max(0.0);
    let ci_high = max_effect_ns + 1.96 * se;

    // Compute top quantiles analytically
    let top_quantiles = compute_top_quantiles_analytical(delta_post, lambda_post, theta);

    EffectEstimate {
        max_effect_ns,
        credible_interval_ns: (ci_low, ci_high),
        top_quantiles,
    }
}

/// Compute top quantiles analytically from posterior mean and covariance.
fn compute_top_quantiles_analytical(
    delta_post: &Vector9,
    lambda_post: &Matrix9,
    theta: f64,
) -> Vec<TopQuantile> {
    let mut quantile_stats: Vec<QuantileStats> = Vec::with_capacity(9);

    for k in 0..9 {
        let mean = delta_post[k];
        let se = math::sqrt(lambda_post[(k, k)].max(1e-12));

        // 95% CI
        let ci = (mean - 1.96 * se, mean + 1.96 * se);

        // Exceedance probability: P(|δ_k| > θ)
        // For Gaussian N(μ, σ²): P(|X| > θ) = 1 - Φ((θ-μ)/σ) + Φ((-θ-μ)/σ)
        let exceed_prob = compute_exceedance_prob(mean, se, theta);

        quantile_stats.push((k, DECILES[k], mean, ci, exceed_prob));
    }

    // Sort by exceedance probability (descending)
    quantile_stats.sort_by(|a, b| b.4.total_cmp(&a.4));

    // Take top 2-3 with exceed_prob > 0.5
    quantile_stats
        .into_iter()
        .filter(|(_, _, _, _, exceed_prob)| *exceed_prob > 0.5)
        .take(3)
        .map(|(_, quantile_p, mean_ns, ci95_ns, exceed_prob)| TopQuantile {
            quantile_p,
            mean_ns,
            ci95_ns,
            exceed_prob,
        })
        .collect()
}

/// Compute P(|X| > θ) for X ~ N(μ, σ²).
fn compute_exceedance_prob(mu: f64, sigma: f64, theta: f64) -> f64 {
    if sigma < 1e-12 {
        // Degenerate case
        return if mu.abs() > theta { 1.0 } else { 0.0 };
    }
    let phi_upper = math::normal_cdf((theta - mu) / sigma);
    let phi_lower = math::normal_cdf((-theta - mu) / sigma);
    1.0 - (phi_upper - phi_lower)
}

/// Apply variance floor regularization for numerical stability (spec §3.3.2).
///
/// When some quantiles have zero or near-zero variance (common in discrete mode
/// with ties), the covariance matrix becomes ill-conditioned.
///
/// Formula (spec §3.3.2):
///   σ²ᵢ ← max(σ²ᵢ, 0.01 × σ̄²) + ε
/// where σ̄² = tr(Σ)/9 and ε = 10⁻¹⁰ + σ̄² × 10⁻⁸
pub fn regularize_covariance(sigma: &Matrix9) -> Matrix9 {
    let trace: f64 = (0..9).map(|i| sigma[(i, i)]).sum();
    let mean_var = trace / 9.0;

    // Use 1% of mean variance as floor, with absolute minimum of 1e-10
    let min_var = (0.01 * mean_var).max(1e-10);

    // Also add small jitter proportional to scale for numerical stability
    let jitter = 1e-10 + mean_var * 1e-8;

    let mut regularized = *sigma;
    for i in 0..9 {
        regularized[(i, i)] = regularized[(i, i)].max(min_var) + jitter;
    }
    regularized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_estimate_basic() {
        // Create some sample draws
        let draws: Vec<Vector9> = (0..100)
            .map(|i| {
                let val = (i as f64) * 0.1;
                Vector9::from_row_slice(&[val, val, val, val, val, val, val, val, val])
            })
            .collect();

        let estimate = compute_effect_estimate(&draws, 5.0);

        // Max effect should be around 9.9 (max draw is 99 * 0.1 = 9.9)
        assert!(estimate.max_effect_ns > 4.0, "max effect should be significant");
        assert!(
            estimate.credible_interval_ns.0 < estimate.max_effect_ns,
            "CI lower should be below mean"
        );
        assert!(
            estimate.credible_interval_ns.1 > estimate.max_effect_ns,
            "CI upper should be above mean"
        );
    }

    #[test]
    fn test_effect_estimate_empty() {
        let estimate = compute_effect_estimate(&[], 5.0);
        assert_eq!(estimate.max_effect_ns, 0.0);
        assert!(estimate.top_quantiles.is_empty());
    }

    #[test]
    fn test_top_quantiles_threshold() {
        // Create draws where only the 90th percentile exceeds threshold
        let draws: Vec<Vector9> = (0..100)
            .map(|_| Vector9::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]))
            .collect();

        let top = compute_top_quantiles(&draws, 5.0);

        // Should have the 90th percentile (index 8, quantile_p = 0.9)
        assert!(!top.is_empty());
        assert!((top[0].quantile_p - 0.9).abs() < 0.01);
        assert!(top[0].exceed_prob > 0.99);
    }

    #[test]
    fn test_regularize_covariance() {
        let mut sigma = Matrix9::zeros();
        for i in 0..9 {
            sigma[(i, i)] = if i == 0 { 0.0 } else { 1.0 }; // First diagonal is zero
        }

        let regularized = regularize_covariance(&sigma);

        // All diagonal elements should be positive
        for i in 0..9 {
            assert!(
                regularized[(i, i)] > 0.0,
                "diagonal {} should be positive",
                i
            );
        }
    }

    #[test]
    fn test_exceedance_prob() {
        // Large mean should have high exceedance
        let prob_high = compute_exceedance_prob(100.0, 10.0, 50.0);
        assert!(prob_high > 0.99, "large mean should exceed threshold");

        // Small mean should have low exceedance for high threshold
        let prob_low = compute_exceedance_prob(1.0, 1.0, 50.0);
        assert!(prob_low < 0.01, "small mean should not exceed threshold");

        // Zero mean, threshold equals 2σ -> ~5% exceedance
        let prob_2sigma = compute_exceedance_prob(0.0, 1.0, 2.0);
        assert!(
            (prob_2sigma - 0.0455).abs() < 0.01,
            "2σ threshold should have ~4.5% exceedance"
        );
    }
}
