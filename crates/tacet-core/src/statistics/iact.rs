//! Integrated Autocorrelation Time (IACT) estimation using Geyer's Initial Monotone Sequence algorithm.
//!
//! This module implements the Geyer IMS algorithm as specified in the implementation guide §3,
//! which provides a robust estimate of the effective sample size for autocorrelated data.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::math;
use crate::types::{Class, TimingSample};

/// Result of IACT computation
#[derive(Debug, Clone)]
pub struct IactResult {
    /// IACT estimate (taû)
    pub tau: f64,
    /// Truncation point in the monotone sequence
    pub m_trunc: usize,
    /// Maximum lag computed for ACF
    pub max_lag: usize,
    /// Warnings encountered during computation
    pub warnings: Vec<IactWarning>,
}

/// Warnings that may be emitted during IACT computation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IactWarning {
    /// Insufficient samples for reliable IACT estimation
    InsufficientSamples { n: usize },
    /// Data has zero variance
    ZeroVariance,
    /// All consecutive pairs were non-positive
    AllPairsNonPositive,
    /// Upper bound (Stan's safeguard) was applied
    UpperBoundApplied { tau_uncapped: f64, bound: f64 },
}

// Manual Eq/Ord implementations for sorting/dedup
impl Eq for IactWarning {}

impl PartialOrd for IactWarning {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IactWarning {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use IactWarning::*;
        match (self, other) {
            (InsufficientSamples { n: n1 }, InsufficientSamples { n: n2 }) => n1.cmp(n2),
            (InsufficientSamples { .. }, _) => core::cmp::Ordering::Less,
            (_, InsufficientSamples { .. }) => core::cmp::Ordering::Greater,
            (ZeroVariance, ZeroVariance) => core::cmp::Ordering::Equal,
            (ZeroVariance, _) => core::cmp::Ordering::Less,
            (_, ZeroVariance) => core::cmp::Ordering::Greater,
            (AllPairsNonPositive, AllPairsNonPositive) => core::cmp::Ordering::Equal,
            (AllPairsNonPositive, _) => core::cmp::Ordering::Less,
            (_, AllPairsNonPositive) => core::cmp::Ordering::Greater,
            (UpperBoundApplied { .. }, UpperBoundApplied { .. }) => core::cmp::Ordering::Equal,
        }
    }
}

/// Geyer's Initial Monotone Sequence algorithm for IACT estimation.
///
/// Implements the algorithm from the implementation guide §3.1:
/// 1. Edge case: n < 20 → return tau=1.0
/// 2. Edge case: variance = 0 → return tau=1.0
/// 3. Compute ACF ρ[k] for k=0..min(n/4, 1000)
/// 4. Form pairs Γ[m] = ρ[2m] + ρ[2m+1]
/// 5. Enforce monotonicity: Γ[m] = min(Γ[m], Γ[m-1])
/// 6. Truncate at first Γ[m] ≤ 0
/// 7. Compute tau = -1 + 2·sum(Γ[0..m_trunc])
/// 8. Clamp: max(tau, 1.0), min(tau, n·log₁₀(n))
///
/// # Arguments
///
/// * `u` - Scalar time series data
///
/// # Returns
///
/// `IactResult` containing the IACT estimate and diagnostic information
pub fn geyer_ims_iact(u: &[f64]) -> IactResult {
    let n = u.len();
    let mut warnings = Vec::new();

    // Edge case: n < 20
    if n < 20 {
        warnings.push(IactWarning::InsufficientSamples { n });
        return IactResult {
            tau: 1.0,
            m_trunc: 0,
            max_lag: 0,
            warnings,
        };
    }

    // Compute mean and center data
    let mean = u.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = u.iter().map(|&x| x - mean).collect();

    // Compute variance
    let var = centered.iter().map(|x| x * x).sum::<f64>() / n as f64;

    // Edge case: zero variance
    if var < 1e-12 {
        warnings.push(IactWarning::ZeroVariance);
        return IactResult {
            tau: 1.0,
            m_trunc: 0,
            max_lag: 0,
            warnings,
        };
    }

    // Step 1: Compute autocorrelations ρ[k] for k=0..max_lag
    let max_lag = (n / 4).min(1000);
    let mut rho = vec![0.0; max_lag + 1];
    rho[0] = 1.0; // ρ[0] = 1 by definition

    for k in 1..=max_lag {
        if k >= n {
            break;
        }
        let cross_product: f64 = centered[k..]
            .iter()
            .zip(centered[..n - k].iter())
            .map(|(&a, &b)| a * b)
            .sum();
        rho[k] = cross_product / (n as f64 * var);
    }

    // Step 2: Form consecutive pairs Γ[m] = ρ[2m] + ρ[2m+1]
    let m_max = (max_lag - 1) / 2;
    let mut gamma = vec![0.0; m_max + 1];
    for (m, g) in gamma.iter_mut().enumerate() {
        let idx1 = 2 * m;
        let idx2 = 2 * m + 1;
        if idx2 <= max_lag {
            *g = rho[idx1] + rho[idx2];
        }
    }

    // Step 3: Enforce monotonicity (sequential)
    for m in 1..=m_max {
        gamma[m] = gamma[m].min(gamma[m - 1]);
    }

    // Step 4: Truncation at first Γ[m] ≤ 0
    let mut m_trunc = 0;
    for (m, &g) in gamma.iter().enumerate().skip(1) {
        if g <= 0.0 {
            break;
        }
        m_trunc = m;
    }

    // Edge case: all pairs non-positive
    if m_trunc == 0 && gamma[0] <= 0.0 {
        warnings.push(IactWarning::AllPairsNonPositive);
        return IactResult {
            tau: 1.0,
            m_trunc: 0,
            max_lag,
            warnings,
        };
    }

    // Step 5: Compute tau = -1 + 2·sum(Γ[0..m_trunc])
    let gamma_sum: f64 = gamma[0..=m_trunc].iter().sum();
    let mut tau = (-1.0 + 2.0 * gamma_sum).max(1.0);

    // Step 6: Upper bound (Stan's safeguard)
    let upper_bound = (n as f64) * math::log10(n as f64);
    if tau > upper_bound {
        warnings.push(IactWarning::UpperBoundApplied {
            tau_uncapped: tau,
            bound: upper_bound,
        });
        tau = upper_bound;
    }

    IactResult {
        tau,
        m_trunc,
        max_lag,
        warnings,
    }
}

/// Compute IACT for timing data using scalarization via indicator series.
///
/// Implements the scalarization approach from implementation guide §3.2:
/// 1. Separate stream by class (Fixed/Random)
/// 2. For each quantile p ∈ {0.1, 0.2, ..., 0.9}:
///    - Compute quantile q_p for each class
///    - Create indicator series z_i = 1 if y_i ≤ q_p else 0
///    - Compute IACT on indicator series
/// 3. Return max(tau_F, tau_R) across all quantiles
///
/// # Arguments
///
/// * `stream` - Interleaved timing samples with class labels
///
/// # Returns
///
/// `IactResult` with combined IACT estimate
pub fn timing_iact_combined(stream: &[TimingSample]) -> IactResult {
    const QUANTILES: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    // Separate by class
    let f_samples: Vec<f64> = stream
        .iter()
        .filter(|s| s.class == Class::Baseline)
        .map(|s| s.time_ns)
        .collect();

    let r_samples: Vec<f64> = stream
        .iter()
        .filter(|s| s.class == Class::Sample)
        .map(|s| s.time_ns)
        .collect();

    if f_samples.len() < 20 || r_samples.len() < 20 {
        return IactResult {
            tau: 1.0,
            m_trunc: 0,
            max_lag: 0,
            warnings: vec![IactWarning::InsufficientSamples {
                n: f_samples.len().min(r_samples.len()),
            }],
        };
    }

    let mut tau_f: f64 = 1.0;
    let mut tau_r: f64 = 1.0;
    let mut all_warnings = Vec::new();

    // Compute IACT for each quantile indicator
    for &p in &QUANTILES {
        // Fixed class
        let q_f = compute_quantile(&f_samples, p);
        let z_f: Vec<f64> = f_samples
            .iter()
            .map(|&y| if y <= q_f { 1.0 } else { 0.0 })
            .collect();
        let result_f = geyer_ims_iact(&z_f);
        tau_f = f64::max(tau_f, result_f.tau);
        all_warnings.extend(result_f.warnings);

        // Random class
        let q_r = compute_quantile(&r_samples, p);
        let z_r: Vec<f64> = r_samples
            .iter()
            .map(|&y| if y <= q_r { 1.0 } else { 0.0 })
            .collect();
        let result_r = geyer_ims_iact(&z_r);
        tau_r = f64::max(tau_r, result_r.tau);
        all_warnings.extend(result_r.warnings);
    }

    // Conservative combination
    let tau_combined = f64::max(tau_f, tau_r);

    // Deduplicate warnings
    all_warnings.sort_unstable();
    all_warnings.dedup();

    IactResult {
        tau: tau_combined,
        m_trunc: 0, // Not meaningful for combined result
        max_lag: 0, // Not meaningful for combined result
        warnings: all_warnings,
    }
}

/// Compute IACT directly on timing differences between baseline and sample.
///
/// This approach computes the IACT on timing differences while respecting
/// temporal adjacency. Instead of pairing samples by within-class index,
/// we use a sliding window to pair each sample with the nearest baseline
/// measurement in time, preserving the dependence structure.
///
/// # Algorithm
///
/// 1. Scan through the interleaved stream maintaining a window
/// 2. For each sample, pair with the nearest baseline (within window)
/// 3. Compute difference: sample - baseline for each pair
/// 4. Apply Geyer's IMS IACT algorithm to the difference series
///
/// This avoids the pairing scrambling issue that occurs when randomly
/// interleaved samples are separated and paired by within-class order.
///
/// # Arguments
///
/// * `stream` - Interleaved timing samples with class labels
///
/// # Returns
///
/// `IactResult` with IACT estimate from the timing difference series
pub fn timing_iact_direct(stream: &[TimingSample]) -> IactResult {
    // Use a sliding window approach to pair samples with nearest baselines
    // Window size: how many samples to look back for pairing
    const WINDOW_SIZE: usize = 10;

    let mut differences = Vec::new();
    let mut recent_baselines: Vec<(usize, f64)> = Vec::new(); // (index, time_ns)

    for (idx, sample) in stream.iter().enumerate() {
        match sample.class {
            Class::Baseline => {
                // Add to recent baselines window
                recent_baselines.push((idx, sample.time_ns));
                // Keep only recent samples within window
                if recent_baselines.len() > WINDOW_SIZE {
                    recent_baselines.remove(0);
                }
            }
            Class::Sample => {
                // Pair this sample with the nearest baseline in the window
                if let Some(&(_, baseline_time)) = recent_baselines.last() {
                    // Use most recent baseline (closest in time)
                    differences.push(sample.time_ns - baseline_time);
                }
            }
        }
    }

    // Need at least 20 paired differences
    if differences.len() < 20 {
        return IactResult {
            tau: 1.0,
            m_trunc: 0,
            max_lag: 0,
            warnings: vec![IactWarning::InsufficientSamples {
                n: differences.len(),
            }],
        };
    }

    // Apply Geyer's IMS IACT to the difference series
    geyer_ims_iact(&differences)
}

/// Compute IACT using per-quantile indicators with robust aggregation.
///
/// This method is more principled than DirectDifferences for quantile-based
/// inference because it targets the correct object: the dependence structure
/// of quantile influence functions (indicator-like processes).
///
/// Unlike GeyersIMS which uses max() aggregation (overly conservative),
/// this uses median aggregation which is robust to outlier IACT estimates
/// while still respecting the per-quantile dependence structure.
///
/// # Algorithm
///
/// 1. For each of 9 quantiles (0.1, 0.2, ..., 0.9):
///    - Compute indicator series: u_t(p) = 𝟙{y_t ≤ q_p}
///    - Apply Geyer's IMS IACT to the indicator series
///    - Do this separately for baseline and sample classes
/// 2. Aggregate using median instead of max:
///    - τ_baseline = median({τ̂₁, ..., τ̂₉})
///    - τ_sample = median({τ̂₁, ..., τ̂₉})
/// 3. Conservative combination: τ_combined = max(τ_baseline, τ_sample)
///
/// This avoids the "max of 9 noisy things" pathology while preserving
/// the theoretical correctness of indicator-based quantile dependence.
///
/// # Arguments
///
/// * `stream` - Interleaved timing samples with class labels
///
/// # Returns
///
/// `IactResult` with median-aggregated IACT estimate
pub fn timing_iact_per_quantile(stream: &[TimingSample]) -> IactResult {
    const QUANTILES: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    // Separate by class
    let f_samples: Vec<f64> = stream
        .iter()
        .filter(|s| s.class == Class::Baseline)
        .map(|s| s.time_ns)
        .collect();

    let r_samples: Vec<f64> = stream
        .iter()
        .filter(|s| s.class == Class::Sample)
        .map(|s| s.time_ns)
        .collect();

    if f_samples.len() < 20 || r_samples.len() < 20 {
        return IactResult {
            tau: 1.0,
            m_trunc: 0,
            max_lag: 0,
            warnings: vec![IactWarning::InsufficientSamples {
                n: f_samples.len().min(r_samples.len()),
            }],
        };
    }

    let mut tau_f_vec: Vec<f64> = Vec::new();
    let mut tau_r_vec: Vec<f64> = Vec::new();
    let mut all_warnings = Vec::new();

    // Compute IACT for each quantile indicator
    for &p in &QUANTILES {
        // Baseline class
        let q_f = compute_quantile(&f_samples, p);
        let z_f: Vec<f64> = f_samples
            .iter()
            .map(|&y| if y <= q_f { 1.0 } else { 0.0 })
            .collect();
        let result_f = geyer_ims_iact(&z_f);
        tau_f_vec.push(result_f.tau);
        all_warnings.extend(result_f.warnings);

        // Sample class
        let q_r = compute_quantile(&r_samples, p);
        let z_r: Vec<f64> = r_samples
            .iter()
            .map(|&y| if y <= q_r { 1.0 } else { 0.0 })
            .collect();
        let result_r = geyer_ims_iact(&z_r);
        tau_r_vec.push(result_r.tau);
        all_warnings.extend(result_r.warnings);
    }

    // Robust aggregation: use median instead of max
    let tau_f = compute_median(&mut tau_f_vec);
    let tau_r = compute_median(&mut tau_r_vec);

    // Conservative combination (still use max across classes)
    let tau_combined = f64::max(tau_f, tau_r);

    // Deduplicate warnings
    all_warnings.sort_unstable();
    all_warnings.dedup();

    IactResult {
        tau: tau_combined,
        m_trunc: 0, // Not meaningful for combined result
        max_lag: 0, // Not meaningful for combined result
        warnings: all_warnings,
    }
}

/// Compute median of a vector in-place.
fn compute_median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

/// Compute quantile using type 2 quantiles (inverse empirical CDF with averaging).
///
/// This matches the quantile computation used elsewhere in the codebase.
fn compute_quantile(sorted_or_unsorted: &[f64], p: f64) -> f64 {
    if sorted_or_unsorted.is_empty() {
        return 0.0;
    }

    let mut data = sorted_or_unsorted.to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let n = data.len();
    let h = (n as f64) * p + 0.5;

    let lo = (math::floor(h) as usize).clamp(1, n);
    let hi = (math::ceil(h) as usize).clamp(1, n);

    (data[lo - 1] + data[hi - 1]) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_iid_data() {
        // White noise should have tau ≈ 1
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let data: Vec<f64> = (0..500).map(|_| rng.random()).collect();
        let result = geyer_ims_iact(&data);
        assert!(
            result.tau >= 1.0 && result.tau < 2.0,
            "IID data should have tau close to 1, got {}",
            result.tau
        );
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_ar1_moderate() {
        // AR(1) with phi=0.5: theoretical tau = (1+phi)/(1-phi) = 3.0
        let phi = 0.5;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let mut data = Vec::with_capacity(1000);
        let mut x = 0.0;
        for _ in 0..1000 {
            x = phi * x + rng.random::<f64>() - 0.5;
            data.push(x);
        }
        let result = geyer_ims_iact(&data);
        // Allow 30% tolerance
        assert!(
            result.tau >= 2.1 && result.tau <= 3.9,
            "AR(1) with phi=0.5 should have tau ~3.0, got {}",
            result.tau
        );
    }

    #[test]
    fn test_ar1_strong() {
        // AR(1) with phi=0.9: theoretical tau = 19.0
        let phi = 0.9;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(456);
        let mut data = Vec::with_capacity(2000);
        let mut x = 0.0;
        for _ in 0..2000 {
            x = phi * x + rng.random::<f64>() - 0.5;
            data.push(x);
        }
        let result = geyer_ims_iact(&data);
        assert!(
            result.tau > 10.0,
            "Strong AR(1) should have high IACT, got {}",
            result.tau
        );
    }

    #[test]
    fn test_edge_case_small_n() {
        let data = vec![1.0, 2.0, 3.0]; // n=3 < 20
        let result = geyer_ims_iact(&data);
        assert_eq!(result.tau, 1.0);
        assert!(matches!(
            result.warnings[0],
            IactWarning::InsufficientSamples { .. }
        ));
    }

    #[test]
    fn test_edge_case_zero_variance() {
        let data = vec![5.0; 100];
        let result = geyer_ims_iact(&data);
        assert_eq!(result.tau, 1.0);
        assert!(matches!(result.warnings[0], IactWarning::ZeroVariance));
    }

    #[test]
    fn test_negative_autocorrelation() {
        // Alternating sequence: -1, 1, -1, 1, ...
        let data: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let result = geyer_ims_iact(&data);
        assert!(result.tau >= 1.0); // Should be clamped
    }

    #[test]
    fn test_quantile_computation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q50 = compute_quantile(&data, 0.5);
        assert!(
            (q50 - 3.0).abs() < 0.1,
            "Median should be ~3.0, got {}",
            q50
        );

        let q90 = compute_quantile(&data, 0.9);
        assert!(q90 > 4.0, "90th percentile should be >4.0, got {}", q90);
    }

    #[test]
    fn test_timing_iact_direct_iid() {
        use crate::types::Class;

        // Create IID baseline and sample with no timing difference
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(456);
        let n = 500;

        let mut stream = Vec::with_capacity(n * 2);
        for _ in 0..n {
            let time_ns = rng.random::<f64>() * 100.0; // Random timing
            stream.push(TimingSample {
                time_ns,
                class: Class::Baseline,
            });
            stream.push(TimingSample {
                time_ns: time_ns + rng.random::<f64>() * 2.0, // Small random difference
                class: Class::Sample,
            });
        }

        let result = timing_iact_direct(&stream);

        // IID differences should have tau ≈ 1
        assert!(result.tau >= 1.0, "IACT should be >= 1.0");
        assert!(
            result.tau < 2.0,
            "IID data should have low IACT, got {}",
            result.tau
        );
        assert!(result.warnings.is_empty(), "Should have no warnings");
    }

    #[test]
    fn test_timing_iact_direct_with_shift() {
        use crate::types::Class;

        // Create timing data with consistent shift (no autocorrelation in differences)
        let n = 500;
        let shift = 50.0; // Fixed 50ns shift

        let mut stream = Vec::with_capacity(n * 2);
        for i in 0..n {
            let time_ns = (i as f64) * 10.0; // Linear time progression
            stream.push(TimingSample {
                time_ns,
                class: Class::Baseline,
            });
            stream.push(TimingSample {
                time_ns: time_ns + shift, // Consistent shift
                class: Class::Sample,
            });
        }

        let result = timing_iact_direct(&stream);

        // Constant differences should have tau ≈ 1 (no autocorrelation in differences)
        assert!(result.tau >= 1.0, "IACT should be >= 1.0");
        assert!(
            result.tau < 3.0,
            "Constant shift should have low IACT, got {}",
            result.tau
        );
    }

    #[test]
    fn test_timing_iact_direct_insufficient_samples() {
        use crate::types::Class;

        // Create stream with < 20 samples
        let stream = vec![
            TimingSample {
                time_ns: 1.0,
                class: Class::Baseline,
            },
            TimingSample {
                time_ns: 2.0,
                class: Class::Sample,
            },
            TimingSample {
                time_ns: 3.0,
                class: Class::Baseline,
            },
        ];

        let result = timing_iact_direct(&stream);

        assert_eq!(
            result.tau, 1.0,
            "Should return tau=1.0 for insufficient samples"
        );
        assert!(
            !result.warnings.is_empty(),
            "Should have InsufficientSamples warning"
        );
    }

    #[test]
    fn test_timing_iact_per_quantile_iid() {
        use crate::types::Class;

        // Create IID baseline and sample with no timing difference
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(789);
        let n = 500;

        let mut stream = Vec::with_capacity(n * 2);
        for _ in 0..n {
            stream.push(TimingSample {
                time_ns: rng.random::<f64>() * 100.0,
                class: Class::Baseline,
            });
            stream.push(TimingSample {
                time_ns: rng.random::<f64>() * 100.0,
                class: Class::Sample,
            });
        }

        let result = timing_iact_per_quantile(&stream);

        // IID data should have low IACT
        assert!(result.tau >= 1.0, "IACT should be >= 1.0");
        assert!(
            result.tau < 3.0,
            "IID data should have low IACT, got {}",
            result.tau
        );
    }

    #[test]
    fn test_timing_iact_per_quantile_vs_max() {
        use crate::types::Class;

        // Create data where one quantile has high IACT but most have low IACT
        // This simulates the sparse outlier case
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(999);
        let n = 500;

        let mut stream = Vec::with_capacity(n * 2);
        for i in 0..n {
            let baseline_time = (i as f64) * 10.0;
            let sample_time = if i % 20 == 0 {
                // 5% of samples have large effect with autocorrelation
                baseline_time + 500.0 + ((i / 20) as f64) * 50.0
            } else {
                // 95% of samples have no effect
                baseline_time + rng.random::<f64>() * 2.0
            };

            stream.push(TimingSample {
                time_ns: baseline_time,
                class: Class::Baseline,
            });
            stream.push(TimingSample {
                time_ns: sample_time,
                class: Class::Sample,
            });
        }

        let result_per_quantile = timing_iact_per_quantile(&stream);
        let result_max = timing_iact_combined(&stream);

        // Per-quantile (median) should be less conservative than max
        assert!(
            result_per_quantile.tau <= result_max.tau,
            "Median aggregation should be <= max aggregation: {} vs {}",
            result_per_quantile.tau,
            result_max.tau
        );
    }

    #[test]
    fn test_timing_iact_per_quantile_insufficient_samples() {
        use crate::types::Class;

        let stream = vec![
            TimingSample {
                time_ns: 1.0,
                class: Class::Baseline,
            },
            TimingSample {
                time_ns: 2.0,
                class: Class::Sample,
            },
        ];

        let result = timing_iact_per_quantile(&stream);

        assert_eq!(
            result.tau, 1.0,
            "Should return tau=1.0 for insufficient samples"
        );
        assert!(
            !result.warnings.is_empty(),
            "Should have InsufficientSamples warning"
        );
    }
}
