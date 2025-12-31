//! Layer 1: RTLF-style CI gate with bounded false positive rate.
//!
//! This implements a frequentist screening layer that provides:
//! - Pass/fail decision suitable for CI pipelines
//! - Guaranteed false positive rate via Bonferroni correction
//! - Block-bootstrap-based threshold computation within each class
//!
//! The approach follows the methodology from:
//! Dunsche et al. (RTLF) for constant-time validation.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::result::CiGate;
use crate::statistics::{
    block_bootstrap_resample_into, compute_block_size, compute_deciles_inplace, counter_rng_seed,
};
use crate::types::Vector9;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Input data for CI gate analysis.
#[derive(Debug, Clone)]
pub struct CiGateInput<'a> {
    /// Observed quantile differences (fixed - random) for 9 deciles.
    pub observed_diff: Vector9,
    /// Fixed-class samples (nanoseconds).
    pub fixed_samples: &'a [f64],
    /// Random-class samples (nanoseconds).
    pub random_samples: &'a [f64],
    /// Significance level (e.g., 0.05 for 5% false positive rate).
    pub alpha: f64,
    /// Number of bootstrap iterations for thresholds.
    pub bootstrap_iterations: usize,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
    /// Timer resolution in nanoseconds (for quantization to integer ticks).
    pub timer_resolution_ns: f64,
}

/// Run the CI gate analysis.
///
/// This implements RTLF-style testing with:
/// 1. Bootstrap within each class separately to estimate null distribution
/// 2. Compute per-class thresholds at corrected alpha level
/// 3. Take maximum threshold across classes for each quantile
/// 4. Apply Bonferroni correction (alpha/9) for multiple testing
pub fn run_ci_gate(input: &CiGateInput<'_>) -> CiGate {
    // Bonferroni correction for 9 simultaneous tests
    let corrected_alpha = input.alpha / 9.0;

    // Compute thresholds via block bootstrap within each class
    let thresholds = compute_bootstrap_thresholds(
        input.fixed_samples,
        input.random_samples,
        corrected_alpha,
        input.bootstrap_iterations,
        input.seed,
    );

    // Convert observed differences to array
    let observed: [f64; 9] = input.observed_diff.as_slice().try_into().unwrap_or([0.0; 9]);

    // Convert thresholds to array and floor to timer resolution
    let timer_res = input.timer_resolution_ns;
    let thresholds_arr: [f64; 9] = thresholds
        .as_slice()
        .iter()
        .map(|&t| {
            // Floor threshold to at least 1 timer tick to handle quantization
            // When bootstrap finds zero variance (tied data), this prevents
            // rejecting any non-zero difference due to timer quantization noise
            t.max(timer_res)
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or([0.0; 9]);

    // Convert to integer ticks for comparison (eliminates float rounding errors)
    let passed = observed
        .iter()
        .zip(thresholds_arr.iter())
        .all(|(obs, thresh)| {
            // Convert both to integer ticks
            let obs_ticks = (obs.abs() / timer_res).round() as i64;
            let thresh_ticks = (thresh / timer_res).round() as u64;
            // Integer comparison avoids float precision issues
            obs_ticks.unsigned_abs() <= thresh_ticks
        });

    CiGate {
        alpha: input.alpha,
        passed,
        thresholds: thresholds_arr,
        observed,
    }
}

fn compute_bootstrap_thresholds(
    fixed_samples: &[f64],
    random_samples: &[f64],
    alpha: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> Vector9 {
    let base_seed = seed.unwrap_or(42);
    let n = fixed_samples.len().min(random_samples.len());
    let block_size = compute_block_size(n);

    // Parallel bootstrap for both fixed and random classes
    // Use map_init to create per-thread scratch buffers that are reused across iterations
    #[cfg(feature = "parallel")]
    let (fixed_diffs, random_diffs): (Vec<Vector9>, Vec<Vector9>) = {
        // Pre-allocate output vectors
        let mut fixed_out = vec![Vector9::zeros(); n_bootstrap];
        let mut random_out = vec![Vector9::zeros(); n_bootstrap];

        fixed_out
            .par_iter_mut()
            .zip(random_out.par_iter_mut())
            .enumerate()
            .map_init(
                || {
                    // Per-thread initialization: RNG and scratch buffers
                    // These are reused across all iterations assigned to this thread
                    (
                        Xoshiro256PlusPlus::seed_from_u64(base_seed),
                        vec![0.0; fixed_samples.len()],
                        vec![0.0; fixed_samples.len()],
                        vec![0.0; random_samples.len()],
                        vec![0.0; random_samples.len()],
                    )
                },
                |(rng, f1, f2, r1, r2), (i, (fixed_out, random_out))| {
                    // Use counter-based RNG for deterministic, well-distributed seeding
                    // SplitMix64 provides better statistical properties than simple addition
                    *rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(base_seed, i as u64));

                    // Reuse scratch buffers (no allocations in hot loop)
                    // Resample into buffers, sort in-place, compute deciles
                    block_bootstrap_resample_into(fixed_samples, block_size, rng, f1);
                    block_bootstrap_resample_into(fixed_samples, block_size, rng, f2);
                    *fixed_out = compute_deciles_inplace(f1) - compute_deciles_inplace(f2);

                    block_bootstrap_resample_into(random_samples, block_size, rng, r1);
                    block_bootstrap_resample_into(random_samples, block_size, rng, r2);
                    *random_out = compute_deciles_inplace(r1) - compute_deciles_inplace(r2);
                },
            )
            .count(); // Force execution

        (fixed_out, random_out)
    };

    #[cfg(not(feature = "parallel"))]
    let (fixed_diffs, random_diffs) = {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed);
        let mut fixed = Vec::with_capacity(n_bootstrap);
        let mut random = Vec::with_capacity(n_bootstrap);

        // Reusable buffers to eliminate allocations
        let mut f1 = vec![0.0; fixed_samples.len()];
        let mut f2 = vec![0.0; fixed_samples.len()];
        let mut r1 = vec![0.0; random_samples.len()];
        let mut r2 = vec![0.0; random_samples.len()];

        for _ in 0..n_bootstrap {
            block_bootstrap_resample_into(fixed_samples, block_size, &mut rng, &mut f1);
            block_bootstrap_resample_into(fixed_samples, block_size, &mut rng, &mut f2);
            fixed.push(compute_deciles_inplace(&mut f1) - compute_deciles_inplace(&mut f2));

            block_bootstrap_resample_into(random_samples, block_size, &mut rng, &mut r1);
            block_bootstrap_resample_into(random_samples, block_size, &mut rng, &mut r2);
            random.push(compute_deciles_inplace(&mut r1) - compute_deciles_inplace(&mut r2));
        }

        (fixed, random)
    };

    let threshold_quantile = 1.0 - alpha;
    let idx = ((n_bootstrap as f64) * threshold_quantile).ceil() as usize;
    let idx = idx.saturating_sub(1).min(n_bootstrap.saturating_sub(1));

    let mut thresholds = Vector9::zeros();
    for i in 0..9 {
        let mut abs_fixed: Vec<f64> = fixed_diffs.iter().map(|s| s[i].abs()).collect();
        abs_fixed.sort_by(|a, b| a.total_cmp(b));

        let mut abs_random: Vec<f64> = random_diffs.iter().map(|s| s[i].abs()).collect();
        abs_random.sort_by(|a, b| a.total_cmp(b));

        let thresh_fixed = abs_fixed[idx];
        let thresh_random = abs_random[idx];
        thresholds[i] = thresh_fixed.max(thresh_random);
    }

    thresholds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::compute_deciles;

    #[test]
    fn test_ci_gate_passes_on_identical_samples() {
        let fixed: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let random = fixed.clone();
        let observed = compute_deciles(&fixed) - compute_deciles(&random);

        let input = CiGateInput {
            observed_diff: observed,
            fixed_samples: &fixed,
            random_samples: &random,
            alpha: 0.05,
            bootstrap_iterations: 200,
            seed: Some(123),
            timer_resolution_ns: 1.0, // 1ns resolution for test
        };

        let result = run_ci_gate(&input);
        assert!(result.passed, "CI gate should pass when no difference observed");
    }

    #[test]
    fn test_ci_gate_fails_on_large_difference() {
        let fixed: Vec<f64> = (0..2000).map(|x| x as f64 + 1000.0).collect();
        let random: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let observed = compute_deciles(&fixed) - compute_deciles(&random);

        let input = CiGateInput {
            observed_diff: observed,
            fixed_samples: &fixed,
            random_samples: &random,
            alpha: 0.05,
            bootstrap_iterations: 200,
            seed: Some(123),
            timer_resolution_ns: 1.0, // 1ns resolution for test
        };

        let result = run_ci_gate(&input);
        assert!(!result.passed, "CI gate should fail on large difference");
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_ci_gate_chunked_determinism() {
        // Test that chunked parallel implementation produces identical results
        // across multiple runs with the same seed
        let fixed: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let random: Vec<f64> = (0..2000).map(|x| x as f64 + 0.5).collect();

        let result1 = compute_bootstrap_thresholds(&fixed, &random, 0.01, 1000, Some(42));
        let result2 = compute_bootstrap_thresholds(&fixed, &random, 0.01, 1000, Some(42));
        let result3 = compute_bootstrap_thresholds(&fixed, &random, 0.01, 1000, Some(42));

        // Verify all 9 quantile thresholds are identical
        for i in 0..9 {
            assert!(
                (result1[i] - result2[i]).abs() < 1e-12,
                "Run 1 vs 2 differ at quantile {}: {} vs {}",
                i,
                result1[i],
                result2[i]
            );
            assert!(
                (result2[i] - result3[i]).abs() < 1e-12,
                "Run 2 vs 3 differ at quantile {}: {} vs {}",
                i,
                result2[i],
                result3[i]
            );
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_ci_gate_chunking_edge_cases() {
        // Test edge cases: n_bootstrap < typical chunk size, exact multiples, remainders
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();

        // n_bootstrap < chunk_size (should still work with chunk_size = n_bootstrap)
        let r1 = compute_bootstrap_thresholds(&data, &data, 0.01, 5, Some(42));
        assert_eq!(r1.len(), 9, "Should return 9 quantiles even with n=5");

        // n_bootstrap = chunk_size (one chunk per core)
        let r2 = compute_bootstrap_thresholds(&data, &data, 0.01, 50, Some(42));
        assert_eq!(r2.len(), 9, "Should return 9 quantiles with n=50");

        // n_bootstrap % chunk_size != 0 (last chunk is partial)
        let r3 = compute_bootstrap_thresholds(&data, &data, 0.01, 150, Some(42));
        assert_eq!(r3.len(), 9, "Should return 9 quantiles with n=150");

        // All should be deterministic
        let r3_repeat = compute_bootstrap_thresholds(&data, &data, 0.01, 150, Some(42));
        for i in 0..9 {
            assert!(
                (r3[i] - r3_repeat[i]).abs() < 1e-12,
                "Edge case should be deterministic"
            );
        }
    }
}
