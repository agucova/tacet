//! Effect estimation from posterior samples (spec §5.2).
//!
//! This module computes effect estimates using the W₁ Wasserstein distance
//! decomposed into shift and tail components.
//!
//! ## Effect Reporting (spec §5.2)
//!
//! - W₁ Wasserstein distance between baseline and sample distributions
//! - Decomposition: W₁ ≈ shift + tail
//! - Quantile-level diagnostics (p50/p90/p95/p99)
//! - Pattern classification (uniform shift, tail effect, mixed, negligible)

extern crate alloc;

use alloc::vec::Vec;

use crate::result::{EffectEstimate, EffectPattern, QuantileShifts, TailDiagnostics};

/// Compute effect estimate from W₁ posterior samples (spec §5.2).
///
/// Takes posterior samples of W₁ (1D Wasserstein distance) and computes:
/// - max_effect_ns: posterior mean of W₁
/// - credible_interval_ns: 95% CI for W₁
/// - tail_diagnostics: None (not computed here)
///
/// # Arguments
///
/// * `w1_draws` - Posterior samples of W₁
///
/// # Returns
///
/// An `EffectEstimate` with W₁ effect and no tail diagnostics.
pub fn compute_effect_estimate(w1_draws: &[f64]) -> EffectEstimate {
    if w1_draws.is_empty() {
        return EffectEstimate::default();
    }

    let n = w1_draws.len();

    // Posterior mean of W₁
    let max_effect_ns = w1_draws.iter().sum::<f64>() / n as f64;

    // 95% credible interval (2.5th and 97.5th percentiles)
    let mut sorted = w1_draws.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let lo_idx = ((n as f64 * 0.025).round() as usize).min(n - 1);
    let hi_idx = ((n as f64 * 0.975).round() as usize).min(n - 1);
    let credible_interval_ns = (sorted[lo_idx], sorted[hi_idx]);

    EffectEstimate {
        max_effect_ns,
        credible_interval_ns,
        tail_diagnostics: None,
    }
}

/// Compute effect estimate from posterior mean and variance (analytical).
///
/// This is a faster alternative to `compute_effect_estimate` when only the
/// posterior mean and variance are available (no draws).
///
/// # Arguments
///
/// * `w1_post` - Posterior mean of W₁
/// * `var_post` - Posterior variance of W₁
/// * `_theta` - Threshold for exceedance probability (unused in 1D case)
///
/// # Returns
///
/// An `EffectEstimate` with approximate effect using mean and SE.
pub fn compute_effect_estimate_analytical(
    w1_post: f64,
    var_post: f64,
    _theta: f64,
) -> EffectEstimate {
    use crate::math::sqrt;

    // Effect is simply |W₁|
    let max_effect_ns = w1_post.abs();

    // Approximate CI using marginal variance
    let se = sqrt(var_post.max(1e-12));
    let ci_low = (max_effect_ns - 1.96 * se).max(0.0);
    let ci_high = max_effect_ns + 1.96 * se;

    EffectEstimate::new(max_effect_ns, (ci_low, ci_high))
}

/// Compute tail diagnostics from baseline and sample distributions.
///
/// Decomposes W₁ ≈ shift + tail and computes pattern labels.
///
/// # Algorithm
///
/// 1. Sort both distributions
/// 2. Compute rank-matched differences: dᵢ = baseline[i] - sample[i]
/// 3. shift = median(diffs)
/// 4. tail = mean(|dᵢ - shift|)
/// 5. tail_share = tail / (|shift| + tail)
/// 6. tail_slow_share = among top 5% (p95+), fraction of deviations that are slowdowns
/// 7. Compute quantile shifts (p50/p90/p95/p99)
/// 8. Pattern labeling based on tail_share
pub fn compute_tail_diagnostics(baseline: &[f64], sample: &[f64], w1_deb: f64) -> TailDiagnostics {
    // Handle edge cases
    if baseline.is_empty() || sample.is_empty() || baseline.len() != sample.len() {
        return TailDiagnostics {
            shift_ns: 0.0,
            tail_ns: 0.0,
            tail_share: 0.0,
            tail_slow_share: 0.5,
            quantile_shifts: QuantileShifts {
                p50_ns: 0.0,
                p90_ns: 0.0,
                p95_ns: 0.0,
                p99_ns: 0.0,
            },
            pattern_label: EffectPattern::Negligible,
        };
    }

    let n = baseline.len();

    // 1. Sort both distributions
    let mut baseline_sorted = baseline.to_vec();
    let mut sample_sorted = sample.to_vec();
    baseline_sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    sample_sorted.sort_unstable_by(|a, b| a.total_cmp(b));

    // 2. Compute rank-matched differences
    let diffs: Vec<f64> = baseline_sorted
        .iter()
        .zip(sample_sorted.iter())
        .map(|(b, s)| b - s)
        .collect();

    // 3. Shift = median(diffs)
    let mut diffs_copy = diffs.clone();
    let mid = n / 2;
    diffs_copy.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let shift_ns = if n.is_multiple_of(2) {
        (diffs_copy[mid - 1] + diffs_copy[mid]) / 2.0
    } else {
        diffs_copy[mid]
    };

    // 4. Tail = mean(|dᵢ - shift|)
    let tail_ns = diffs.iter().map(|&d| (d - shift_ns).abs()).sum::<f64>() / n as f64;

    // 5. Tail share
    let total_effect = shift_ns.abs() + tail_ns;
    let tail_share = if total_effect > 1e-12 {
        tail_ns / total_effect
    } else {
        0.0
    };

    // 6. Tail slow share: among tail deviations (top 5%), what fraction are slowdowns?
    // Define tail as top 5% of indices (p95 and above)
    let tail_start_idx = (0.95 * n as f64) as usize;
    let pos_tail_sum: f64 = diffs[tail_start_idx..]
        .iter()
        .map(|&d| (d - shift_ns).max(0.0))
        .sum();
    let total_tail_sum: f64 = diffs[tail_start_idx..]
        .iter()
        .map(|&d| (d - shift_ns).abs())
        .sum();
    let tail_slow_share = if total_tail_sum > 1e-12 {
        pos_tail_sum / total_tail_sum
    } else {
        0.5 // Symmetric if no tail
    };

    // 7. Compute quantile shifts
    let p90_idx = (0.90 * n as f64) as usize;
    let p95_idx = (0.95 * n as f64) as usize;
    let p99_idx = ((0.99 * n as f64) as usize).min(n - 1);

    let quantile_shifts = QuantileShifts {
        p50_ns: baseline_sorted[n / 2] - sample_sorted[n / 2],
        p90_ns: baseline_sorted[p90_idx] - sample_sorted[p90_idx],
        p95_ns: baseline_sorted[p95_idx] - sample_sorted[p95_idx],
        p99_ns: baseline_sorted[p99_idx] - sample_sorted[p99_idx],
    };

    // 8. Pattern labeling
    let pattern_label = if w1_deb.abs() < 1.0 {
        EffectPattern::Negligible
    } else if tail_share >= 0.5 {
        EffectPattern::TailEffect
    } else if tail_share < 0.3 {
        EffectPattern::UniformShift
    } else {
        EffectPattern::Mixed
    };

    TailDiagnostics {
        shift_ns,
        tail_ns,
        tail_share,
        tail_slow_share,
        quantile_shifts,
        pattern_label,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_estimate_basic() {
        // Create some sample W₁ draws
        let draws: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();

        let estimate = compute_effect_estimate(&draws);

        // Max effect should be around 4.95 (mean of 0..9.9)
        assert!(
            estimate.max_effect_ns > 4.0,
            "max effect should be significant"
        );
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
        let estimate = compute_effect_estimate(&[]);
        assert_eq!(estimate.max_effect_ns, 0.0);
        assert!(estimate.tail_diagnostics.is_none());
    }

    #[test]
    fn test_tail_diagnostics_uniform_shift() {
        // Create two distributions with a uniform 10ns shift
        let n = 1000;
        let baseline: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.1).collect();
        let sample: Vec<f64> = (0..n).map(|i| 90.0 + i as f64 * 0.1).collect();

        // W₁ is approximately 10ns (uniform shift)
        let w1_deb = 10.0;

        let diagnostics = compute_tail_diagnostics(&baseline, &sample, w1_deb);

        // Should detect uniform shift
        assert!(
            diagnostics.shift_ns > 9.0 && diagnostics.shift_ns < 11.0,
            "shift should be ~10ns, got {}",
            diagnostics.shift_ns
        );
        assert!(
            diagnostics.tail_ns < 1.0,
            "tail should be minimal for uniform shift, got {}",
            diagnostics.tail_ns
        );
        assert!(
            diagnostics.tail_share < 0.3,
            "tail_share should be low for uniform shift, got {}",
            diagnostics.tail_share
        );
        assert_eq!(
            diagnostics.pattern_label,
            EffectPattern::UniformShift,
            "should classify as uniform shift"
        );

        // Quantile shifts should all be similar
        assert!(
            (diagnostics.quantile_shifts.p50_ns - 10.0).abs() < 1.0,
            "p50 shift should be ~10ns"
        );
        assert!(
            (diagnostics.quantile_shifts.p90_ns - 10.0).abs() < 1.0,
            "p90 shift should be ~10ns"
        );
        assert!(
            (diagnostics.quantile_shifts.p95_ns - 10.0).abs() < 1.0,
            "p95 shift should be ~10ns"
        );
    }

    #[test]
    fn test_tail_diagnostics_tail_effect() {
        // Create distributions where only the tail differs
        let n = 1000;
        let mut baseline: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.01).collect();
        let sample: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.01).collect();

        // Add tail effect to top 10% of baseline
        for b in baseline.iter_mut().skip(n * 9 / 10) {
            *b += 50.0; // Add 50ns to tail
        }

        // W₁ should capture this tail effect
        let w1_deb = 5.0; // Approximate W₁ from 10% tail effect

        let diagnostics = compute_tail_diagnostics(&baseline, &sample, w1_deb);

        // Should detect tail effect
        assert!(
            diagnostics.shift_ns.abs() < 5.0,
            "shift should be small for tail effect, got {}",
            diagnostics.shift_ns
        );
        assert!(
            diagnostics.tail_ns > 4.0,
            "tail should be large for tail effect, got {}",
            diagnostics.tail_ns
        );
        assert!(
            diagnostics.tail_share > 0.5,
            "tail_share should be high for tail effect, got {}",
            diagnostics.tail_share
        );
        assert_eq!(
            diagnostics.pattern_label,
            EffectPattern::TailEffect,
            "should classify as tail effect"
        );

        // Tail slow share should indicate baseline is slower in tail
        assert!(
            diagnostics.tail_slow_share > 0.8,
            "tail_slow_share should indicate baseline slower, got {}",
            diagnostics.tail_slow_share
        );

        // Quantile shifts should show increasing effect at higher quantiles
        assert!(
            diagnostics.quantile_shifts.p99_ns > diagnostics.quantile_shifts.p50_ns,
            "p99 shift should be larger than p50 for tail effect"
        );
    }

    #[test]
    fn test_tail_diagnostics_mixed_pattern() {
        // Create distributions with both shift and tail effect
        let n = 1000;
        let mut baseline: Vec<f64> = (0..n).map(|i| 105.0 + i as f64 * 0.1).collect();
        let sample: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.1).collect();

        // Add larger tail effect to top 30% of baseline to create mixed pattern
        for b in baseline.iter_mut().skip(n * 7 / 10) {
            *b += 15.0;
        }

        // W₁ should capture both components
        let w1_deb = 10.0;

        let diagnostics = compute_tail_diagnostics(&baseline, &sample, w1_deb);

        // Should detect mixed pattern
        assert!(
            diagnostics.shift_ns > 2.0,
            "shift should be present, got {}",
            diagnostics.shift_ns
        );
        assert!(
            diagnostics.tail_ns > 2.0,
            "tail should be present, got {}",
            diagnostics.tail_ns
        );
        assert!(
            diagnostics.tail_share >= 0.3 && diagnostics.tail_share <= 0.5,
            "tail_share should be in mixed range, got {}",
            diagnostics.tail_share
        );
        assert_eq!(
            diagnostics.pattern_label,
            EffectPattern::Mixed,
            "should classify as mixed pattern"
        );
    }

    #[test]
    fn test_tail_diagnostics_negligible() {
        // Create identical distributions
        let n = 1000;
        let baseline: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.1).collect();
        let sample = baseline.clone();

        // W₁ is negligible
        let w1_deb = 0.5;

        let diagnostics = compute_tail_diagnostics(&baseline, &sample, w1_deb);

        // Should detect negligible effect
        assert_eq!(
            diagnostics.pattern_label,
            EffectPattern::Negligible,
            "should classify as negligible"
        );
        assert!(
            diagnostics.shift_ns.abs() < 1e-6,
            "shift should be near zero, got {}",
            diagnostics.shift_ns
        );
        assert!(
            diagnostics.tail_ns < 1e-6,
            "tail should be near zero, got {}",
            diagnostics.tail_ns
        );
    }

    #[test]
    fn test_tail_diagnostics_empty_input() {
        // Test empty arrays
        let diagnostics = compute_tail_diagnostics(&[], &[], 0.0);

        assert_eq!(diagnostics.shift_ns, 0.0);
        assert_eq!(diagnostics.tail_ns, 0.0);
        assert_eq!(diagnostics.tail_share, 0.0);
        assert_eq!(diagnostics.tail_slow_share, 0.5);
        assert_eq!(diagnostics.pattern_label, EffectPattern::Negligible);
    }

    #[test]
    fn test_tail_diagnostics_mismatched_lengths() {
        // Test mismatched array lengths
        let baseline = vec![1.0, 2.0, 3.0];
        let sample = vec![1.0, 2.0];

        let diagnostics = compute_tail_diagnostics(&baseline, &sample, 0.0);

        assert_eq!(diagnostics.shift_ns, 0.0);
        assert_eq!(diagnostics.tail_ns, 0.0);
        assert_eq!(diagnostics.pattern_label, EffectPattern::Negligible);
    }

    #[test]
    fn test_tail_diagnostics_single_element() {
        // Test single-element arrays
        let baseline = vec![100.0];
        let sample = vec![90.0];

        let diagnostics = compute_tail_diagnostics(&baseline, &sample, 10.0);

        // Should handle single element gracefully
        assert!(
            (diagnostics.shift_ns - 10.0).abs() < 1e-6,
            "shift should be 10ns for single element"
        );
        // With single element, tail should be zero
        assert!(
            diagnostics.tail_ns < 1e-6,
            "tail should be zero for single element"
        );
    }

    #[test]
    fn test_tail_diagnostics_symmetric_tail() {
        // Create distributions with symmetric QUANTILE deviations in tail
        // Key: W₁ operates on quantile-aligned differences, not sample identities
        // We need quantile crossings: sometimes baseline_q > sample_q, sometimes opposite
        let n = 1000;

        // Build distributions with alternating quantile dominance in tail
        let mut baseline: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.01).collect();
        let mut sample: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.01).collect();

        // Create quantile crossing in tail (p95+):
        // Baseline has higher odd quantiles, sample has higher even quantiles
        for i in (n * 95 / 100)..n {
            if i % 2 == 0 {
                sample[i] += 2.0; // Sample's even quantiles higher
            } else {
                baseline[i] += 2.0; // Baseline's odd quantiles higher
            }
        }

        let w1_deb = 2.0;

        let diagnostics = compute_tail_diagnostics(&baseline, &sample, w1_deb);

        // With quantile crossing, tail_slow_share should be ~0.5
        // (roughly balanced positive and negative quantile differences)
        assert!(
            (diagnostics.tail_slow_share - 0.5).abs() < 0.15,
            "tail_slow_share should be ~0.5 for symmetric quantile deviations, got {}",
            diagnostics.tail_slow_share
        );
    }
}
