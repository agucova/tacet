//! Layer 1: SILENT-style CI gate with controlled false positive rate.
//!
//! This implements a frequentist screening layer that provides:
//! - Pass/fail decision suitable for CI pipelines
//! - Controlled false positive rate via studentized max-bootstrap thresholds
//! - Paired block-bootstrap-based threshold computation
//!
//! The approach follows the methodology from:
//! SILENT (Systematic Inference for LEverage of Nondiscriminatory Timing) which uses
//! studentized statistics with paired block resampling for proper FPR control at the
//! boundary of practical significance (effect = θ).
//!
//! ## SILENT Method (Spec §2.4)
//!
//! Key improvements over simple max|Δ|:
//! - **Studentized statistics**: Each quantile's contribution normalized by its variability
//! - **Paired block resampling**: Same indices for both classes preserve cross-covariance
//! - **Quantile filtering**: Remove high-variance quantiles, apply power-boost filter
//! - **Centered bootstrap**: Proper null distribution estimation
//! - **Finite-sample correction**: Removed in current implementation
//!
//! Test statistic: Q̂_max = max_{k ∈ K_sub} (A_k - θ) / σ̂_k
//! Bootstrap statistic: Q̂*_max = max_{k ∈ K_sub} (A*_k - A_k) / σ̂_k (centered)

use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::result::{CiGate, CiGateResult};
use crate::statistics::{compute_deciles_inplace, counter_rng_seed, paired_optimal_block_length};
use crate::types::Vector9;
use std::env;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Input data for CI gate analysis.
#[derive(Debug, Clone)]
pub struct CiGateInput<'a> {
    /// Observed quantile differences (baseline - sample) for 9 deciles.
    pub observed_diff: Vector9,
    /// Baseline-class samples (nanoseconds).
    pub baseline_samples: &'a [f64],
    /// Sample-class samples (nanoseconds).
    pub sample_samples: &'a [f64],
    /// Significance level (e.g., 0.05 for 5% false positive rate).
    pub alpha: f64,
    /// Number of bootstrap iterations for thresholds.
    pub bootstrap_iterations: usize,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
    /// Timer resolution in nanoseconds (for quantization to integer ticks).
    pub timer_resolution_ns: f64,
    /// Minimum effect of concern in nanoseconds (threshold floor for practical significance).
    pub min_effect_of_concern: f64,
    /// Whether to use discrete mode (m-out-of-n bootstrap with mid-quantiles).
    /// Triggered when min uniqueness ratio < 10% (spec §2.4).
    pub discrete_mode: bool,
}

/// Result of bootstrap variance estimation for SILENT.
#[derive(Debug, Clone)]
struct BootstrapVarianceResult {
    /// Per-quantile standard deviations (σ̂_k for k = 1..9)
    sigmas: [f64; 9],
    /// Bootstrap differences A*_k for each iteration (for computing critical value)
    bootstrap_diffs: Vec<[f64; 9]>,
}

/// Run the CI gate analysis using SILENT methodology (spec §2.4).
///
/// Algorithm (continuous mode):
/// 1. Compute observed A_k = |q_F(k) - q_R(k)| for each quantile k
/// 2. Compute beta corrections for finite-sample bias
/// 3. Estimate per-quantile variances via paired block bootstrap
/// 4. Filter quantiles (high-variance removal + power-boost filter)
/// 5. Compute centered, studentized bootstrap statistics
/// 6. Compare corrected, studentized test statistic against critical value
///
/// Algorithm (discrete mode, spec §2.4):
/// 1. Use mid-distribution quantiles instead of Type 2
/// 2. Use m-out-of-n bootstrap with m = ⌊n^(2/3)⌋
/// 3. Use non-studentized statistics with √m scaling
///
/// # SILENT FPR Guarantee
///
/// When true effect d ≤ θ:
///   lim sup P(Q̂_max > c*_{1-α}) ≤ α
///
/// with equality at d = θ (the least favorable point in the null).
pub fn run_ci_gate(input: &CiGateInput<'_>) -> CiGate {
    // Dispatch to discrete or continuous mode
    if input.discrete_mode {
        run_ci_gate_discrete(input)
    } else {
        run_ci_gate_continuous(input)
    }
}

/// Continuous mode CI gate (standard SILENT with studentized statistics).
fn run_ci_gate_continuous(input: &CiGateInput<'_>) -> CiGate {
    let base_seed = input.seed.unwrap_or(42);
    // theta is already clamped by the caller (oracle.rs) to ensure consistency with Bayesian layer
    let theta = input.min_effect_of_concern;
    let debug_pipeline = env::var("TO_DEBUG_PIPELINE").map(|v| v != "0").unwrap_or(false);

    // Convert observed differences to absolute values A_k
    let observed_abs: [f64; 9] = {
        let slice = input.observed_diff.as_slice();
        let mut arr = [0.0; 9];
        for i in 0..9 {
            arr[i] = slice[i].abs();
        }
        arr
    };

    // Compute max observed for reporting
    let max_observed = observed_abs.iter().cloned().fold(0.0_f64, f64::max);

    // Estimate bootstrap variances via paired block bootstrap
    let variance_result = estimate_bootstrap_variances(
        input.baseline_samples,
        input.sample_samples,
        input.bootstrap_iterations,
        base_seed,
    );

    // Filter quantiles (spec §2.4: high-variance removal + power-boost filter)
    let filtered_indices = filter_quantiles(&observed_abs, &variance_result.sigmas, theta, input.baseline_samples.len());

    // If no quantiles pass filtering, PASS the gate.
    // This happens when no quantile shows potential evidence of effect >= theta.
    // Under null (A_k ≈ 0), the power-boost filter correctly identifies that
    // all quantiles are consistent with effects < theta, so we should pass.
    // P-value is 1.0 (no evidence against null).
    if filtered_indices.is_empty() {
        return CiGate {
            alpha: input.alpha,
            result: CiGateResult::Pass,
            threshold: theta,
            max_observed,
            observed: {
                let slice = input.observed_diff.as_slice();
                [slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7], slice[8]]
            },
            p_value: 1.0,
        };
    }

    // Compute test statistic: Q̂_max = max_{k ∈ K_sub} (A_k - θ) / σ̂_k
    let q_max = compute_studentized_max(
        &observed_abs,
        &variance_result.sigmas,
        &filtered_indices,
        theta,
    );

    // Compute centered, studentized bootstrap statistics and critical value
    let (critical_value, p_value) = compute_critical_value_and_pvalue(
        &observed_abs,
        &variance_result,
        &filtered_indices,
        input.alpha,
        q_max,
    );

    if debug_pipeline {
        eprintln!(
            "[DEBUG_PIPELINE] ci_gate_continuous theta={:.2} observed_abs={:?}",
            theta,
            observed_abs
        );
        eprintln!(
            "[DEBUG_PIPELINE] ci_gate_continuous sigmas={:?}",
            variance_result.sigmas
        );
        eprintln!(
            "[DEBUG_PIPELINE] ci_gate_continuous filtered_indices={:?}",
            filtered_indices
        );
        eprintln!(
            "[DEBUG_PIPELINE] ci_gate_continuous q_max={:.4} critical_value={:.4}",
            q_max,
            critical_value
        );
    }

    // Decision: reject (fail) if Q̂_max > c*_{1-α}
    let passed = q_max <= critical_value;

    // Compute threshold in ns for reporting (convert back from studentized units)
    // Find the quantile that achieved max, compute what A_k would need to be
    let threshold = theta + critical_value * variance_result.sigmas.iter().cloned().fold(0.0_f64, f64::max);

    CiGate {
        alpha: input.alpha,
        result: if passed { CiGateResult::Pass } else { CiGateResult::Fail },
        threshold: threshold.max(theta),
        max_observed,
        observed: {
            let slice = input.observed_diff.as_slice();
            [slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7], slice[8]]
        },
        p_value,
    }
}

/// Discrete mode CI gate (m-out-of-n bootstrap with √m scaling and √n test statistic, spec §2.4).
///
/// Key differences from continuous mode:
/// - Uses mid-distribution quantiles that handle ties correctly
/// - Uses m-out-of-n bootstrap with m = ⌊n^(2/3)⌋
/// - Uses non-studentized statistics with √m scaling
fn run_ci_gate_discrete(input: &CiGateInput<'_>) -> CiGate {
    use crate::statistics::compute_midquantile_deciles;

    let base_seed = input.seed.unwrap_or(42);
    // Theta is already clamped by caller (oracle.rs) per spec §2.4.
    // Research mode (θ=0) is preserved to detect any statistical difference.
    let theta = input.min_effect_of_concern;
    let n = input.baseline_samples.len().min(input.sample_samples.len());

    // Compute resample size m (spec §2.4)
    let m = discrete_resample_size(n);
    let block_size = discrete_block_size(input.baseline_samples, input.sample_samples, n, m);

    // Compute observed quantiles using mid-distribution quantiles
    let q_baseline = compute_midquantile_deciles(input.baseline_samples);
    let q_sample = compute_midquantile_deciles(input.sample_samples);

    // Compute observed A_k = |q_F(k) - q_R(k)|
    let observed_abs: [f64; 9] = {
        let mut arr = [0.0; 9];
        for i in 0..9 {
            arr[i] = (q_baseline[i] - q_sample[i]).abs();
        }
        arr
    };

    // Compute max observed for reporting
    let max_observed = observed_abs.iter().cloned().fold(0.0_f64, f64::max);

    // m-out-of-n paired block bootstrap with √m scaling (spec §2.4)
    let bootstrap_stats = run_m_out_of_n_bootstrap(
        input.baseline_samples,
        input.sample_samples,
        m,
        block_size,
        input.bootstrap_iterations,
        base_seed,
    );

    // Estimate per-quantile sigmas for high-variance filtering
    let sigmas = bootstrap_sigmas(&bootstrap_stats);
    let filtered_indices = filter_quantiles_discrete(&sigmas);

    // If no quantiles pass filtering, PASS (consistent with continuous mode)
    // P-value is 1.0 (no evidence against null).
    if filtered_indices.is_empty() {
        return CiGate {
            alpha: input.alpha,
            result: CiGateResult::Pass,
            threshold: theta,
            max_observed,
            observed: {
                let slice = input.observed_diff.as_slice();
                [slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7], slice[8]]
            },
            p_value: 1.0,
        };
    }

    // Test statistic: √n × max_{k in K_sub}(A_k - θ) (non-studentized)
    let sqrt_n = (n as f64).sqrt();
    let t_observed = sqrt_n
        * filtered_indices
            .iter()
            .map(|&k| observed_abs[k] - theta)
            .fold(f64::NEG_INFINITY, f64::max);

    // Compute centered bootstrap statistics: √m × max_{k in K_sub}(A*_k - A_k)
    let sqrt_m = (m as f64).sqrt();
    let mut centered_stats: Vec<f64> = bootstrap_stats
        .iter()
        .map(|a_star| {
            let max_a_star = filtered_indices
                .iter()
                .map(|&k| a_star[k] - observed_abs[k])
                .fold(f64::NEG_INFINITY, f64::max);
            sqrt_m * max_a_star
        })
        .collect();

    // Compute p-value before sorting: P(centered_stat >= t_observed)
    let exceedances = centered_stats.iter().filter(|&&s| s >= t_observed).count();
    let p_value = (exceedances as f64) / (centered_stats.len() as f64);

    // Compute critical value
    centered_stats.sort_by(|a, b| a.total_cmp(b));
    let idx = ((centered_stats.len() as f64) * (1.0 - input.alpha)).ceil() as usize;
    let idx = idx.saturating_sub(1).min(centered_stats.len().saturating_sub(1));
    let critical_value = centered_stats.get(idx).copied().unwrap_or(0.0);

    // Decision: reject (fail) if t_observed > critical_value
    let passed = t_observed <= critical_value;

    // Threshold in ns
    let threshold = theta + critical_value / sqrt_n;

    CiGate {
        alpha: input.alpha,
        result: if passed { CiGateResult::Pass } else { CiGateResult::Fail },
        threshold: threshold.max(theta),
        max_observed,
        observed: {
            let slice = input.observed_diff.as_slice();
            [slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7], slice[8]]
        },
        p_value,
    }
}

/// Run m-out-of-n paired block bootstrap for discrete mode.
///
/// Resamples m observations using the same block indices for both classes,
/// computes mid-distribution quantiles, and returns the absolute differences.
fn run_m_out_of_n_bootstrap(
    baseline: &[f64],
    sample: &[f64],
    m: usize,
    block_size: usize,
    n_bootstrap: usize,
    seed: u64,
) -> Vec<[f64; 9]> {
    use crate::statistics::compute_midquantile_deciles;

    let n = baseline.len().min(sample.len());
    let block_size = block_size.max(1);

    #[cfg(feature = "parallel")]
    let result: Vec<[f64; 9]> = crate::thread_pool::install(|| {
        (0..n_bootstrap)
            .into_par_iter()
            .map_init(
                || {
                    (
                        Xoshiro256PlusPlus::seed_from_u64(seed),
                        vec![0.0; m],
                        vec![0.0; m],
                    )
                },
                |(rng, baseline_buf, sample_buf), i| {
                    *rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

                    // Resample m observations from both classes with the same block indices
                    let n_blocks = m.div_ceil(block_size);
                    let max_start = n.saturating_sub(block_size);
                    let mut indices = Vec::with_capacity(n_blocks);
                    for _ in 0..n_blocks {
                        indices.push(rng.random_range(0..=max_start));
                    }
                    resample_with_indices(baseline, &indices, block_size, baseline_buf);
                    resample_with_indices(sample, &indices, block_size, sample_buf);

                    // Compute mid-distribution quantiles
                    let q_baseline = compute_midquantile_deciles(baseline_buf);
                    let q_sample = compute_midquantile_deciles(sample_buf);

                    // Compute A*_k = |q*_F(k) - q*_R(k)|
                    let mut diffs = [0.0; 9];
                    for k in 0..9 {
                        diffs[k] = (q_baseline[k] - q_sample[k]).abs();
                    }
                    diffs
                },
            )
            .collect()
    });

    #[cfg(not(feature = "parallel"))]
    let result: Vec<[f64; 9]> = {
        let mut results = Vec::with_capacity(n_bootstrap);
        let mut baseline_buf = vec![0.0; m];
        let mut sample_buf = vec![0.0; m];

        for i in 0..n_bootstrap {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

            // Resample m observations from both classes with the same block indices
            let n_blocks = m.div_ceil(block_size);
            let max_start = n.saturating_sub(block_size);
            let mut indices = Vec::with_capacity(n_blocks);
            for _ in 0..n_blocks {
                indices.push(rng.gen_range(0..=max_start));
            }
            resample_with_indices(baseline, &indices, block_size, &mut baseline_buf);
            resample_with_indices(sample, &indices, block_size, &mut sample_buf);

            // Compute mid-distribution quantiles
            let q_baseline = compute_midquantile_deciles(&baseline_buf);
            let q_sample = compute_midquantile_deciles(&sample_buf);

            // Compute A*_k
            let mut diffs = [0.0; 9];
            for k in 0..9 {
                diffs[k] = (q_baseline[k] - q_sample[k]).abs();
            }
            results.push(diffs);
        }
        results
    };

    result
}

fn discrete_resample_size(n: usize) -> usize {
    if n < 2_000 {
        let m = (n / 2).max(200);
        m.min(n).max(1)
    } else {
        let m = (n as f64).powf(2.0 / 3.0).floor() as usize;
        let m = m.max(400);
        m.min(n).max(1)
    }
}

fn discrete_block_size(baseline: &[f64], sample: &[f64], n: usize, m: usize) -> usize {
    let block_size = if n >= 10 {
        paired_optimal_block_length(baseline, sample)
    } else {
        (1.3 * (n as f64).powf(1.0 / 3.0)).ceil().max(1.0) as usize
    };
    let cap = (m / 5).max(1);
    block_size.min(cap).max(1)
}

fn bootstrap_sigmas(bootstrap_diffs: &[[f64; 9]]) -> [f64; 9] {
    let n_bootstrap = bootstrap_diffs.len().max(1);
    let mut sigmas = [0.0; 9];
    for k in 0..9 {
        let mean: f64 = bootstrap_diffs.iter().map(|d| d[k]).sum::<f64>() / n_bootstrap as f64;
        let variance: f64 = if n_bootstrap > 1 {
            bootstrap_diffs
                .iter()
                .map(|d| (d[k] - mean).powi(2))
                .sum::<f64>()
                / (n_bootstrap - 1) as f64
        } else {
            0.0
        };
        sigmas[k] = variance.sqrt().max(1e-12);
    }
    sigmas
}

fn filter_quantiles_discrete(sigmas: &[f64; 9]) -> Vec<usize> {
    let mean_variance: f64 = sigmas.iter().map(|s| s * s).sum::<f64>() / 9.0;
    let variance_threshold = 5.0 * mean_variance;

    (0..9)
        .filter(|&k| {
            let variance = sigmas[k] * sigmas[k];
            variance <= variance_threshold
        })
        .collect()
}

/// Estimate per-quantile variances via paired block bootstrap (spec §2.4).
///
/// Uses paired resampling: the same block indices are used for both classes,
/// preserving cross-covariance structure from common-mode drift.
fn estimate_bootstrap_variances(
    baseline_samples: &[f64],
    sample_samples: &[f64],
    n_bootstrap: usize,
    seed: u64,
) -> BootstrapVarianceResult {
    let n = baseline_samples.len().min(sample_samples.len());
    // Use Politis-White algorithm for optimal block length selection (spec §2.6)
    // Compute separately for each class and take maximum to preserve dependence in both
    let block_size = if n >= 10 {
        paired_optimal_block_length(baseline_samples, sample_samples)
    } else {
        // Fall back to simple formula for very small samples
        (1.3 * (n as f64).powf(1.0 / 3.0)).ceil().max(1.0) as usize
    };

    // Generate block indices once, then compute A*_k for each iteration
    #[cfg(feature = "parallel")]
    let bootstrap_diffs: Vec<[f64; 9]> = crate::thread_pool::install(|| {
        (0..n_bootstrap)
            .into_par_iter()
            .map_init(
                || {
                    // Per-thread: RNG and scratch buffers
                    (
                        Xoshiro256PlusPlus::seed_from_u64(seed),
                        vec![0.0; n], // baseline resample
                        vec![0.0; n], // sample resample
                        Vec::<usize>::with_capacity(n), // block indices
                    )
                },
                |(rng, baseline_buf, sample_buf, indices), i| {
                    *rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

                    // Generate block starting indices
                    indices.clear();
                    let n_blocks = n.div_ceil(block_size);
                    let max_start = n.saturating_sub(block_size);
                    for _ in 0..n_blocks {
                        indices.push(rng.random_range(0..=max_start));
                    }

                    // Resample BOTH classes using SAME block indices (paired resampling)
                    resample_with_indices(baseline_samples, indices, block_size, baseline_buf);
                    resample_with_indices(sample_samples, indices, block_size, sample_buf);

                    // Compute quantile differences A*_k = |q*_F(k) - q*_R(k)|
                    let q_baseline = compute_deciles_inplace(baseline_buf);
                    let q_sample = compute_deciles_inplace(sample_buf);

                    let mut diffs = [0.0; 9];
                    for k in 0..9 {
                        diffs[k] = (q_baseline[k] - q_sample[k]).abs();
                    }
                    diffs
                },
            )
            .collect()
    });

    #[cfg(not(feature = "parallel"))]
    let bootstrap_diffs: Vec<[f64; 9]> = {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut baseline_buf = vec![0.0; n];
        let mut sample_buf = vec![0.0; n];
        let mut indices = Vec::with_capacity((n + block_size - 1) / block_size);
        let mut diffs = Vec::with_capacity(n_bootstrap);

        for i in 0..n_bootstrap {
            rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

            // Generate block starting indices
            indices.clear();
            let n_blocks = (n + block_size - 1) / block_size;
            let max_start = n.saturating_sub(block_size);
            for _ in 0..n_blocks {
                indices.push(rng.gen_range(0..=max_start));
            }

            // Resample BOTH classes using SAME block indices (paired resampling)
            resample_with_indices(baseline_samples, &indices, block_size, &mut baseline_buf);
            resample_with_indices(sample_samples, &indices, block_size, &mut sample_buf);

            // Compute quantile differences A*_k
            let q_baseline = compute_deciles_inplace(&mut baseline_buf);
            let q_sample = compute_deciles_inplace(&mut sample_buf);

            let mut diff = [0.0; 9];
            for k in 0..9 {
                diff[k] = (q_baseline[k] - q_sample[k]).abs();
            }
            diffs.push(diff);
        }
        diffs
    };

    // Compute per-quantile standard deviations
    let mut sigmas = [0.0; 9];
    for k in 0..9 {
        let mean: f64 = bootstrap_diffs.iter().map(|d| d[k]).sum::<f64>() / n_bootstrap as f64;
        let variance: f64 = bootstrap_diffs.iter().map(|d| (d[k] - mean).powi(2)).sum::<f64>() / (n_bootstrap - 1) as f64;
        sigmas[k] = variance.sqrt().max(1e-12); // Floor to avoid division by zero
    }

    BootstrapVarianceResult {
        sigmas,
        bootstrap_diffs,
    }
}

/// Resample using block bootstrap with given starting indices.
fn resample_with_indices(data: &[f64], indices: &[usize], block_size: usize, buffer: &mut [f64]) {
    let n = buffer.len();
    let mut pos = 0;

    for &start in indices {
        for offset in 0..block_size {
            if pos >= n {
                break;
            }
            let idx = (start + offset) % data.len();
            buffer[pos] = data[idx];
            pos += 1;
        }
    }
}

/// Filter quantiles per SILENT spec §2.4.
///
/// Step 1: Remove high-variance quantiles (σ²_k > 5 × mean variance)
/// Step 2: Power-boost filter (keep quantiles likely to contribute to detection)
fn filter_quantiles(
    observed_abs: &[f64; 9],
    sigmas: &[f64; 9],
    theta: f64,
    n: usize,
) -> Vec<usize> {
    // Step 1: High-variance filter
    // K_sub = { k : σ²_k ≤ 5 × mean(σ²) }
    let mean_variance: f64 = sigmas.iter().map(|s| s * s).sum::<f64>() / 9.0;
    let variance_threshold = 5.0 * mean_variance;

    let mut k_sub: Vec<usize> = (0..9)
        .filter(|&k| {
            let variance = sigmas[k] * sigmas[k];
            variance <= variance_threshold
        })
        .collect();

    // If all quantiles filtered, return empty (will pass the gate)
    if k_sub.is_empty() {
        return k_sub;
    }

    // Step 2: Power-boost filter (K_sub^max)
    // Keep quantiles where: A_k + 30 × σ_k × √((log n)^(3/2) / n) ≥ θ
    let n_f = n as f64;
    let log_n = n_f.ln();
    let slack = 30.0 * (log_n.powf(1.5) / n_f).sqrt();

    k_sub.retain(|&k| {
        let a_k = observed_abs[k];
        let sigma_k = sigmas[k];
        a_k + slack * sigma_k >= theta
    });

    k_sub
}

/// Compute studentized max statistic:
/// Q̂_max = max_{k ∈ K_sub} (A_k - θ) / σ̂_k
fn compute_studentized_max(
    observed_abs: &[f64; 9],
    sigmas: &[f64; 9],
    filtered_indices: &[usize],
    theta: f64,
) -> f64 {
    filtered_indices
        .iter()
        .map(|&k| (observed_abs[k] - theta) / sigmas[k])
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Compute critical value c*_{1-α} and p-value from centered, studentized bootstrap distribution.
///
/// Bootstrap statistic: Q̂*_max = max_{k ∈ K_sub} (A*_k - A_k) / σ̂_k
/// Note: We subtract observed A_k (not θ) to center the bootstrap distribution around 0.
///
/// Returns (critical_value, p_value) where p_value = P(Q̂*_max >= Q̂_max).
fn compute_critical_value_and_pvalue(
    observed_abs: &[f64; 9],
    variance_result: &BootstrapVarianceResult,
    filtered_indices: &[usize],
    alpha: f64,
    q_max: f64,
) -> (f64, f64) {
    if filtered_indices.is_empty() {
        return (f64::INFINITY, 1.0); // No rejection possible, p-value = 1
    }

    // Compute centered, studentized bootstrap statistics
    let mut q_star_values: Vec<f64> = variance_result
        .bootstrap_diffs
        .iter()
        .map(|a_star| {
            // Q̂*_max = max_{k ∈ K_sub} (A*_k - A_k) / σ̂_k
            filtered_indices
                .iter()
                .map(|&k| (a_star[k] - observed_abs[k]) / variance_result.sigmas[k])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();

    // Compute p-value before sorting: P(Q̂*_max >= Q̂_max)
    let exceedances = q_star_values.iter().filter(|&&q| q >= q_max).count();
    let p_value = (exceedances as f64) / (q_star_values.len() as f64);

    // Sort to find (1-α) quantile
    q_star_values.sort_by(|a, b| a.total_cmp(b));

    let n = q_star_values.len();
    let idx = ((n as f64) * (1.0 - alpha)).ceil() as usize;
    let idx = idx.saturating_sub(1).min(n.saturating_sub(1));

    let critical_value = q_star_values[idx];

    (critical_value, p_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::compute_deciles;

    #[test]
    fn test_ci_gate_passes_on_identical_samples() {
        let baseline: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let sample = baseline.clone();
        let observed = compute_deciles(&baseline) - compute_deciles(&sample);

        let input = CiGateInput {
            observed_diff: observed,
            baseline_samples: &baseline,
            sample_samples: &sample,
            alpha: 0.05,
            bootstrap_iterations: 200,
            seed: Some(123),
            timer_resolution_ns: 1.0, // 1ns resolution for test
            min_effect_of_concern: 0.0,
            discrete_mode: false,
        };

        let result = run_ci_gate(&input);
        assert!(result.passed(), "CI gate should pass when no difference observed");
    }

    #[test]
    fn test_ci_gate_fails_on_large_difference() {
        // Use random samples to add variability that paired bootstrap can detect
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(42);

        // Baseline has different distribution than sample (shifted by 1000)
        let baseline: Vec<f64> = (0..2000)
            .map(|_| 1000.0 + rng.random::<f64>() * 100.0)
            .collect();
        let sample: Vec<f64> = (0..2000)
            .map(|_| rng.random::<f64>() * 100.0)
            .collect();

        let observed = compute_deciles(&baseline) - compute_deciles(&sample);

        let input = CiGateInput {
            observed_diff: observed,
            baseline_samples: &baseline,
            sample_samples: &sample,
            alpha: 0.05,
            bootstrap_iterations: 200,
            seed: Some(123),
            timer_resolution_ns: 1.0, // 1ns resolution for test
            min_effect_of_concern: 0.0,
            discrete_mode: false,
        };

        let result = run_ci_gate(&input);
        assert!(!result.passed(), "CI gate should fail on large difference (~1000ns)");
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_ci_gate_max_bootstrap_determinism() {
        // Test that parallel implementation produces identical results
        // across multiple runs with the same seed
        let baseline: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let sample: Vec<f64> = (0..2000).map(|x| x as f64 + 0.5).collect();
        let observed = compute_deciles(&baseline) - compute_deciles(&sample);

        let input = CiGateInput {
            observed_diff: observed,
            baseline_samples: &baseline,
            sample_samples: &sample,
            alpha: 0.01,
            bootstrap_iterations: 200,
            seed: Some(42),
            timer_resolution_ns: 1.0,
            min_effect_of_concern: 0.0,
            discrete_mode: false,
        };

        let result1 = run_ci_gate(&input);
        let result2 = run_ci_gate(&input);
        let result3 = run_ci_gate(&input);

        // Verify results are identical across runs
        assert_eq!(result1.result, result2.result, "Result should be deterministic");
        assert_eq!(result2.result, result3.result, "Result should be deterministic");
        assert!(
            (result1.threshold - result2.threshold).abs() < 1e-10,
            "Threshold should be deterministic"
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_ci_gate_max_bootstrap_edge_cases() {
        // Test edge cases: n_bootstrap < typical chunk size, exact multiples, remainders
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let observed = compute_deciles(&data) - compute_deciles(&data);

        // n_bootstrap < chunk_size (should still work)
        let input = CiGateInput {
            observed_diff: observed,
            baseline_samples: &data,
            sample_samples: &data,
            alpha: 0.01,
            bootstrap_iterations: 5,
            seed: Some(42),
            timer_resolution_ns: 1.0,
            min_effect_of_concern: 0.0,
            discrete_mode: false,
        };
        let r1 = run_ci_gate(&input);
        assert!(r1.threshold >= 0.0, "Threshold should be non-negative");

        // n_bootstrap = chunk_size (one chunk per core)
        let input2 = CiGateInput {
            bootstrap_iterations: 50,
            ..input.clone()
        };
        let r2 = run_ci_gate(&input2);
        assert!(r2.threshold >= 0.0, "Threshold should be non-negative");
    }

    #[test]
    fn test_ci_gate_new_fields() {
        // Test that the new CiGate struct has correct fields
        let baseline: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let sample = baseline.clone();
        let observed = compute_deciles(&baseline) - compute_deciles(&sample);

        let input = CiGateInput {
            observed_diff: observed,
            baseline_samples: &baseline,
            sample_samples: &sample,
            alpha: 0.05,
            bootstrap_iterations: 200,
            seed: Some(123),
            timer_resolution_ns: 1.0,
            min_effect_of_concern: 0.0,
            discrete_mode: false,
        };

        let result = run_ci_gate(&input);

        // Check new fields exist and are sensible
        assert!(result.threshold >= 0.0, "threshold should be non-negative");
        assert!(result.max_observed >= 0.0, "max_observed should be non-negative");
        assert_eq!(result.observed.len(), 9, "observed should have 9 elements");
    }

    #[test]
    fn test_filter_quantiles_removes_high_variance() {
        // Simulate one quantile with much higher variance
        let observed_abs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut sigmas = [1.0; 9];
        sigmas[8] = 10.0; // p90 has 10x higher sigma

        let filtered = filter_quantiles(&observed_abs, &sigmas, 0.5, 1000);

        // p90 (index 8) should be filtered out due to high variance
        assert!(!filtered.contains(&8), "High variance quantile should be filtered");
        assert!(filtered.len() < 9, "Should filter at least one quantile");
    }
}
