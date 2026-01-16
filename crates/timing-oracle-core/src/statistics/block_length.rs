//! Optimal block length estimation using Politis-White (2004) with
//! Patton-Politis-White (2009) correction.
//!
//! This module implements data-adaptive block length selection for time-series
//! bootstraps. The algorithm analyzes the autocorrelation structure of the data
//! to determine the optimal block size, rather than using a fixed rule like n^(1/3).
//!
//! # Algorithm Overview
//!
//! 1. Find the lag `m` where autocorrelations become insignificant
//! 2. Use a flat-top kernel to estimate the long-run variance and its derivative
//! 3. Compute optimal block lengths that minimize MSE of the bootstrap variance estimator
//!
//! # References
//!
//! - Politis, D. N., & White, H. (2004). Automatic Block-Length Selection for
//!   the Dependent Bootstrap. Econometric Reviews, 23(1), 53-70.
//! - Patton, A., Politis, D. N., & White, H. (2009). Correction to "Automatic
//!   Block-Length Selection for the Dependent Bootstrap". Econometric Reviews, 28(4), 372-375.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// Result of optimal block length estimation.
#[derive(Debug, Clone, Copy)]
pub struct OptimalBlockLength {
    /// Optimal block length for stationary bootstrap (exponentially distributed blocks).
    pub stationary: f64,
    /// Optimal block length for circular block bootstrap (fixed-size blocks with wrap-around).
    pub circular: f64,
}

/// Estimate optimal block lengths for a single time series.
///
/// Implements the Politis-White (2004) algorithm with Patton-Politis-White (2009)
/// corrections for automatic block length selection.
///
/// # Arguments
///
/// * `x` - A slice of time series observations (minimum 10 required)
///
/// # Returns
///
/// `OptimalBlockLength` containing estimates for both stationary and circular bootstraps.
///
/// # Panics
///
/// Panics if `x.len() < 10` (need sufficient data for autocorrelation estimation).
///
/// # Algorithm
///
/// The algorithm proceeds in three phases:
///
/// 1. **Find truncation lag `m`**: Scan autocorrelations to find the first lag where
///    `k_n` consecutive autocorrelations are all insignificant (within ±2√(log₁₀(n)/n)).
///    This determines the "memory" of the process.
///
/// 2. **Estimate spectral quantities**: Using a flat-top (trapezoidal) kernel,
///    estimate the long-run variance σ² and its derivative g.
///
/// 3. **Compute optimal block length**: The MSE-optimal block length is:
///    ```text
///    b_opt = ((2 * g²) / d)^(1/3) * n^(1/3)
///    ```
///    where d = 2σ⁴ for stationary bootstrap and d = (4/3)σ⁴ for circular.
pub fn optimal_block_length(x: &[f64]) -> OptimalBlockLength {
    let n = x.len();
    assert!(
        n >= 10,
        "Need at least 10 observations for block length estimation"
    );

    // =========================================================================
    // Step 1: Center the data
    // =========================================================================
    let mean = x.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = x.iter().map(|&xi| xi - mean).collect();

    // =========================================================================
    // Step 2: Compute tuning parameters
    // =========================================================================

    // Maximum allowed block length: min(3√n, n/3)
    // Prevents blocks from being too large relative to sample size
    let max_block_length = (3.0 * (n as f64).sqrt()).min(n as f64 / 3.0).ceil();

    // k_n: number of consecutive insignificant autocorrelations needed
    // Scales slowly with n: max(5, log₁₀(n))
    let consecutive_insignificant_needed = 5.max((n as f64).log10() as usize);

    // m_max: maximum lag to consider for autocorrelation truncation
    // Roughly √n + k_n to ensure we explore enough lags
    let max_lag = (n as f64).sqrt().ceil() as usize + consecutive_insignificant_needed;

    // Critical value for insignificance test: ±2√(log₁₀(n)/n)
    // Conservative bound that scales appropriately with sample size
    let insignificance_threshold = 2.0 * ((n as f64).log10() / n as f64).sqrt();

    // =========================================================================
    // Step 3: Compute autocovariances and find truncation lag
    // =========================================================================

    // Storage for autocovariances γ(k) and |ρ(k)| (absolute autocorrelations)
    let mut autocovariances = vec![0.0; max_lag + 1];
    let mut abs_autocorrelations = vec![0.0; max_lag + 1];

    // Track when we first find k_n consecutive insignificant autocorrelations
    let mut first_insignificant_run_start: Option<usize> = None;

    for lag in 0..=max_lag {
        // Need at least lag+1 observations for the cross-product
        if lag + 1 >= n {
            break;
        }

        // Compute variance of the leading and trailing segments
        // These are used to normalize the cross-product into a correlation
        let leading_segment = &centered[lag + 1..]; // x_{lag+1}, ..., x_n
        let trailing_segment = &centered[..n - lag - 1]; // x_1, ..., x_{n-lag-1}

        let variance_leading: f64 = leading_segment.iter().map(|e| e * e).sum();
        let variance_trailing: f64 = trailing_segment.iter().map(|e| e * e).sum();

        // Cross-product: Σ_{k=lag}^{n-1} x_k * x_{k-lag}
        let cross_product: f64 = centered[lag..]
            .iter()
            .zip(centered[..n - lag].iter())
            .map(|(&a, &b)| a * b)
            .sum();

        // Store autocovariance: γ(lag) = (1/n) * Σ (x_t - μ)(x_{t-lag} - μ)
        autocovariances[lag] = cross_product / n as f64;

        // Store absolute autocorrelation: |ρ(lag)| = |cross_product| / √(var_lead * var_trail)
        let denominator = (variance_leading * variance_trailing).sqrt();
        abs_autocorrelations[lag] = if denominator > 0.0 {
            cross_product.abs() / denominator
        } else {
            0.0
        };

        // Check if we've found k_n consecutive insignificant autocorrelations
        if lag >= consecutive_insignificant_needed && first_insignificant_run_start.is_none() {
            let recent_autocorrelations =
                &abs_autocorrelations[lag - consecutive_insignificant_needed..lag];
            let all_insignificant = recent_autocorrelations
                .iter()
                .all(|&r| r < insignificance_threshold);

            if all_insignificant {
                // The run of insignificant autocorrelations starts k_n lags ago
                first_insignificant_run_start = Some(lag - consecutive_insignificant_needed);
            }
        }
    }

    // =========================================================================
    // Step 4: Determine truncation lag m
    // =========================================================================

    // If we found a run of insignificant autocorrelations, use 2 * (start of run)
    // Otherwise, fall back to max_lag
    let truncation_lag = match first_insignificant_run_start {
        Some(start) => (2 * start.max(1)).min(max_lag),
        None => max_lag,
    };

    // =========================================================================
    // Step 5: Compute spectral quantities using flat-top kernel
    // =========================================================================

    // g: weighted sum of lag * autocovariance (related to derivative of spectrum)
    // long_run_variance: weighted sum of autocovariances (spectrum at frequency 0)

    let mut g = 0.0; // Σ λ(k/m) * k * γ(k) for k ≠ 0
    let mut long_run_variance = autocovariances[0]; // Start with γ(0)

    for (lag, &acv) in autocovariances[1..=truncation_lag].iter().enumerate() {
        let lag = lag + 1; // Adjust since we started from index 1

        // Flat-top (trapezoidal) kernel:
        //   λ(x) = 1           if |x| ≤ 0.5
        //   λ(x) = 2(1 - |x|)  if 0.5 < |x| ≤ 1
        //   λ(x) = 0           otherwise
        let kernel_arg = lag as f64 / truncation_lag as f64;
        let kernel_weight = if kernel_arg <= 0.5 {
            1.0
        } else {
            2.0 * (1.0 - kernel_arg)
        };

        // g accumulates kernel-weighted lag * autocovariance
        // Factor of 2 accounts for both positive and negative lags (symmetry)
        g += 2.0 * kernel_weight * lag as f64 * acv;

        // Long-run variance accumulates kernel-weighted autocovariances
        long_run_variance += 2.0 * kernel_weight * acv;
    }

    // =========================================================================
    // Step 6: Compute optimal block lengths
    // =========================================================================

    // The MSE-optimal block length formula:
    //   b_opt = ((2 * g²) / d)^(1/3) * n^(1/3)
    //
    // where d depends on the bootstrap type:
    //   - Stationary bootstrap: d = 2 * σ⁴
    //   - Circular block bootstrap: d = (4/3) * σ⁴

    let variance_squared = long_run_variance.powi(2);

    // Constants for each bootstrap type
    let d_stationary = 2.0 * variance_squared;
    let d_circular = (4.0 / 3.0) * variance_squared;

    // Compute block lengths, handling degenerate cases
    let n_cuberoot = (n as f64).powf(1.0 / 3.0);

    let block_stationary = if d_stationary > 0.0 {
        let ratio = (2.0 * g.powi(2)) / d_stationary;
        ratio.powf(1.0 / 3.0) * n_cuberoot
    } else {
        // Degenerate case: no dependence or zero variance
        1.0
    };

    let block_circular = if d_circular > 0.0 {
        let ratio = (2.0 * g.powi(2)) / d_circular;
        ratio.powf(1.0 / 3.0) * n_cuberoot
    } else {
        1.0
    };

    // Apply upper bound to prevent unreasonably large blocks
    OptimalBlockLength {
        stationary: block_stationary.min(max_block_length),
        circular: block_circular.min(max_block_length),
    }
}

/// Compute optimal block length for paired time series (for timing oracle).
///
/// When analyzing timing differences between two classes, we need a block length
/// that accounts for the dependence structure in both series. This function
/// computes optimal block lengths for each series and returns the maximum,
/// ensuring we adequately capture the temporal dependence in both.
///
/// # Arguments
///
/// * `baseline` - Timing measurements for baseline class
/// * `sample` - Timing measurements for sample class
///
/// # Returns
///
/// The ceiling of the maximum circular bootstrap block length from both series.
/// Uses the circular bootstrap estimate as it's more appropriate for the
/// fixed-block resampling used in timing oracle.
pub fn paired_optimal_block_length(baseline: &[f64], sample: &[f64]) -> usize {
    let opt_baseline = optimal_block_length(baseline);
    let opt_sample = optimal_block_length(sample);

    // Take the maximum to ensure we capture dependence in both series
    // Use circular estimate since our bootstrap uses fixed-size blocks
    let max_circular = opt_baseline.circular.max(opt_sample.circular);

    // Return ceiling, with minimum of 1
    max_circular.ceil().max(1.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// Generate an AR(1) process: x_t = φ * x_{t-1} + ε_t
    fn generate_ar1(n: usize, phi: f64, seed: u64) -> Vec<f64> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        let mut x = vec![0.0; n];
        x[0] = rng.random::<f64>() - 0.5;

        for i in 1..n {
            let innovation = rng.random::<f64>() - 0.5;
            x[i] = phi * x[i - 1] + innovation;
        }

        x
    }

    #[test]
    fn test_iid_data_small_block() {
        // IID data should have small optimal block length (close to 1)
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let x: Vec<f64> = (0..500).map(|_| rng.random::<f64>()).collect();

        let opt = optimal_block_length(&x);

        // IID data should give small block lengths
        assert!(
            opt.stationary < 10.0,
            "IID stationary block {} should be small",
            opt.stationary
        );
        assert!(
            opt.circular < 10.0,
            "IID circular block {} should be small",
            opt.circular
        );
    }

    #[test]
    fn test_ar1_moderate_dependence() {
        // AR(1) with φ=0.5 should have moderate block length
        let x = generate_ar1(500, 0.5, 123);

        let opt = optimal_block_length(&x);

        // Moderate dependence: expect blocks in range [3, 30]
        assert!(
            opt.stationary > 2.0 && opt.stationary < 40.0,
            "AR(1) φ=0.5 stationary block {} outside expected range",
            opt.stationary
        );
    }

    #[test]
    fn test_ar1_strong_dependence() {
        // AR(1) with φ=0.9 should have larger block length
        let x = generate_ar1(500, 0.9, 456);

        let opt = optimal_block_length(&x);

        // Strong dependence: expect larger blocks than moderate case
        assert!(
            opt.stationary > 5.0,
            "AR(1) φ=0.9 stationary block {} should be substantial",
            opt.stationary
        );
    }

    #[test]
    fn test_stationary_vs_circular() {
        // Circular block length should be larger than stationary
        // (stationary uses d=2σ⁴, circular uses d=(4/3)σ⁴)
        // Same numerator, smaller denominator for circular → larger block
        let x = generate_ar1(500, 0.6, 789);

        let opt = optimal_block_length(&x);

        // Due to the formula, circular should be (2 / (4/3))^(1/3) ≈ 1.14× larger
        let expected_ratio = (2.0_f64 / (4.0 / 3.0)).powf(1.0 / 3.0);

        let actual_ratio = opt.circular / opt.stationary;

        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "Circular/stationary ratio {} should be ~{}",
            actual_ratio,
            expected_ratio
        );
    }

    #[test]
    fn test_paired_optimal_takes_max() {
        // Paired function should return max of both series
        let x = generate_ar1(500, 0.9, 111); // High dependence
        let y = generate_ar1(500, 0.3, 222); // Low dependence

        let paired = paired_optimal_block_length(&x, &y);

        let opt_x = optimal_block_length(&x);
        let opt_y = optimal_block_length(&y);

        let expected = opt_x.circular.max(opt_y.circular).ceil() as usize;

        assert_eq!(
            paired, expected,
            "Paired block length {} should equal max of individual circular estimates {}",
            paired, expected
        );
    }

    #[test]
    fn test_minimum_sample_size() {
        // Should work with minimum sample size
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(999);
        let x: Vec<f64> = (0..10).map(|_| rng.random::<f64>()).collect();

        let opt = optimal_block_length(&x);

        // Just verify it returns something reasonable
        assert!(opt.stationary >= 1.0, "Block length should be at least 1");
        assert!(
            opt.circular <= 10.0,
            "Block length should not exceed sample size"
        );
    }

    #[test]
    #[should_panic(expected = "Need at least 10 observations")]
    fn test_insufficient_samples_panics() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let _ = optimal_block_length(&x);
    }

    #[test]
    fn test_constant_series() {
        // Constant series has zero autocovariance for lag > 0
        let x = vec![42.0; 100];

        let opt = optimal_block_length(&x);

        // Should fall back to 1 (degenerate case)
        assert_eq!(opt.stationary, 1.0, "Constant series should give block = 1");
        assert_eq!(opt.circular, 1.0, "Constant series should give block = 1");
    }

    #[test]
    fn test_deterministic_results() {
        // Same input should give same output
        let x = generate_ar1(500, 0.5, 42);

        let opt1 = optimal_block_length(&x);
        let opt2 = optimal_block_length(&x);

        assert_eq!(opt1.stationary, opt2.stationary, "Should be deterministic");
        assert_eq!(opt1.circular, opt2.circular, "Should be deterministic");
    }
}
