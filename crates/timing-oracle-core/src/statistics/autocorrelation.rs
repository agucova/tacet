//! Autocorrelation function (ACF) computation.
//!
//! This module computes autocorrelation coefficients for timing measurements.
//! Autocorrelation is important for:
//! - Determining appropriate block sizes for bootstrap
//! - Detecting non-stationarity in measurements
//! - Validating the assumption of weak dependence

extern crate alloc;

use alloc::vec::Vec;

/// Compute lag-1 autocorrelation of a time series.
///
/// The lag-1 autocorrelation measures the correlation between
/// consecutive observations. High values (> 0.3) indicate significant
/// temporal dependence that should be accounted for in bootstrap.
///
/// # Arguments
///
/// * `data` - Slice of timing measurements in temporal order
///
/// # Returns
///
/// The lag-1 autocorrelation coefficient in [-1, 1], or 0.0 if
/// the data has fewer than 2 elements or zero variance.
///
/// # Formula
///
/// ```text
/// r_1 = sum((x_t - mean) * (x_{t+1} - mean)) / sum((x_t - mean)^2)
/// ```
pub fn lag1_autocorrelation(data: &[f64]) -> f64 {
    compute_lag_autocorrelation(data, 1)
}

/// Compute lag-2 autocorrelation of a time series.
///
/// The lag-2 autocorrelation measures the correlation between
/// observations separated by 2 time steps. This is useful for
/// detecting higher-order temporal patterns.
///
/// # Arguments
///
/// * `data` - Slice of timing measurements in temporal order
///
/// # Returns
///
/// The lag-2 autocorrelation coefficient in [-1, 1], or 0.0 if
/// the data has fewer than 3 elements or zero variance.
pub fn lag2_autocorrelation(data: &[f64]) -> f64 {
    compute_lag_autocorrelation(data, 2)
}

/// Compute autocorrelation at a specified lag.
///
/// # Arguments
///
/// * `data` - Slice of measurements in temporal order
/// * `lag` - The lag at which to compute autocorrelation
///
/// # Returns
///
/// The autocorrelation coefficient at the specified lag.
pub fn compute_lag_autocorrelation(data: &[f64], lag: usize) -> f64 {
    let n = data.len();

    // Need at least lag + 1 observations
    if n <= lag {
        return 0.0;
    }

    // Compute mean
    let mean: f64 = data.iter().sum::<f64>() / n as f64;

    // Compute variance (denominator)
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();

    if variance == 0.0 {
        return 0.0;
    }

    // Compute lagged covariance (numerator)
    let mut lagged_cov = 0.0;
    for t in 0..(n - lag) {
        lagged_cov += (data[t] - mean) * (data[t + lag] - mean);
    }

    // Return autocorrelation coefficient
    lagged_cov / variance
}

/// Estimate the dependence length m using the spec's ACF heuristic (§3.2.2).
///
/// Find the smallest h where |ρ(h)| < 2/sqrt(n).
///
/// # Arguments
///
/// * `data` - Subsequence for one class (already extracted and demeaned)
/// * `max_lag` - Maximum lag to search
///
/// # Returns
///
/// The estimated dependence length m_c.
pub fn estimate_dependence_length(data: &[f64], max_lag: usize) -> usize {
    let n = data.len();
    if n == 0 { return 1; }
    let threshold = 2.0 / (n as f64).sqrt();

    for h in 1..=max_lag.min(n - 1) {
        let rho = compute_lag_autocorrelation(data, h);
        if rho.abs() < threshold {
            return h;
        }
    }

    max_lag
}

/// Compute autocorrelation on the full interleaved sequence.
///
/// When measurements alternate between fixed and random classes,
/// this function computes the autocorrelation on the full interleaved
/// sequence, which may capture cross-class dependencies.
///
/// # Arguments
///
/// * `fixed` - Timing measurements for fixed class
/// * `random` - Timing measurements for random class
/// * `lag` - The lag at which to compute autocorrelation
///
/// # Returns
///
/// The autocorrelation coefficient of the interleaved sequence.
#[cfg(test)]
pub fn interleaved_autocorrelation(fixed: &[f64], random: &[f64], lag: usize) -> f64 {
    // Interleave the sequences: F0, R0, F1, R1, F2, R2, ...
    let n = fixed.len().min(random.len());
    let mut interleaved = Vec::with_capacity(2 * n);

    for i in 0..n {
        interleaved.push(fixed[i]);
        interleaved.push(random[i]);
    }

    compute_lag_autocorrelation(&interleaved, lag)
}

/// Compute multiple autocorrelation coefficients up to a maximum lag.
///
/// # Arguments
///
/// * `data` - Slice of measurements in temporal order
/// * `max_lag` - Maximum lag to compute (inclusive)
///
/// # Returns
///
/// A vector of autocorrelation coefficients for lags 1 through max_lag.
pub fn autocorrelation_function(data: &[f64], max_lag: usize) -> Vec<f64> {
    (1..=max_lag)
        .map(|lag| compute_lag_autocorrelation(data, lag))
        .collect()
}

/// Estimate effective sample size given autocorrelation.
///
/// When data is autocorrelated, the effective sample size is smaller
/// than the actual sample size. This function estimates the effective
/// sample size using the lag-1 autocorrelation.
///
/// # Arguments
///
/// * `n` - Actual sample size
/// * `rho1` - Lag-1 autocorrelation coefficient
///
/// # Returns
///
/// The estimated effective sample size.
///
/// # Formula
///
/// ```text
/// n_eff = n * (1 - rho1) / (1 + rho1)
/// ```
///
/// This is a first-order approximation assuming AR(1) dependence.
#[cfg(test)]
pub fn effective_sample_size(n: usize, rho1: f64) -> f64 {
    // Clamp rho1 to valid range to avoid division issues
    let rho1 = rho1.clamp(-0.99, 0.99);

    let n = n as f64;
    n * (1.0 - rho1) / (1.0 + rho1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lag1_independent_data() {
        // For truly independent data, autocorrelation should be near zero
        // Using a better pseudo-random sequence with more mixing
        let data: Vec<f64> = (0..1000)
            .map(|i| {
                let mut x = i as u64;
                x = x.wrapping_mul(0x5851F42D4C957F2D).wrapping_add(0x14057B7EF767814F);
                x ^= x >> 33;
                x = x.wrapping_mul(0xC4CEB9FE1A85EC53);
                (x as f64) / (u64::MAX as f64)
            })
            .collect();

        let acf = lag1_autocorrelation(&data);
        // For a well-mixed sequence, autocorrelation should be relatively low
        assert!(
            acf.abs() < 0.3,
            "Expected low autocorrelation, got {}",
            acf
        );
    }

    #[test]
    fn test_lag1_correlated_data() {
        // Strongly correlated data: x_t = x_{t-1} + small noise
        let mut data = vec![0.0];
        for i in 1..1000 {
            data.push(data[i - 1] + ((i % 10) as f64 - 5.0) * 0.01);
        }

        let acf = lag1_autocorrelation(&data);
        assert!(
            acf > 0.9,
            "Expected high autocorrelation, got {}",
            acf
        );
    }

    #[test]
    fn test_constant_data() {
        let data = vec![5.0; 100];
        let acf = lag1_autocorrelation(&data);
        assert_eq!(acf, 0.0); // Zero variance means zero autocorrelation
    }

    #[test]
    fn test_alternating_data() {
        // Alternating data should have negative lag-1 autocorrelation
        let data: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

        let acf = lag1_autocorrelation(&data);
        assert!(acf < -0.9, "Expected negative autocorrelation, got {}", acf);
    }

    #[test]
    fn test_short_data() {
        assert_eq!(lag1_autocorrelation(&[1.0]), 0.0);
        assert_eq!(lag2_autocorrelation(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_effective_sample_size() {
        // No autocorrelation: effective size equals actual size
        assert!((effective_sample_size(100, 0.0) - 100.0).abs() < 1e-10);

        // Positive autocorrelation: effective size is smaller
        let ess_pos = effective_sample_size(100, 0.5);
        assert!(ess_pos < 100.0);
        assert!((ess_pos - 100.0 * (1.0 - 0.5) / (1.0 + 0.5)).abs() < 1e-10);

        // Negative autocorrelation: effective size is larger
        let ess_neg = effective_sample_size(100, -0.5);
        assert!(ess_neg > 100.0);
    }

    #[test]
    fn test_interleaved_autocorrelation() {
        // Two sequences with similar trends should create positive autocorrelation
        // when interleaved, even though adjacent values jump between sequences
        let fixed: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let random: Vec<f64> = (0..50).map(|x| (x + 50) as f64).collect();

        let acf = interleaved_autocorrelation(&fixed, &random, 2);
        // At lag-2, we compare fixed[i] with fixed[i+1] and random[i] with random[i+1]
        // Both are increasing sequences, so lag-2 autocorrelation should be high
        assert!(acf > 0.5, "Expected high lag-2 autocorrelation, got {}", acf);
    }

    #[test]
    fn test_acf_function() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let acf = autocorrelation_function(&data, 5);

        assert_eq!(acf.len(), 5);
        // All should be positive for monotonic data
        for (i, &r) in acf.iter().enumerate() {
            assert!(r > 0.0, "Expected positive ACF at lag {}, got {}", i + 1, r);
        }
    }
}
