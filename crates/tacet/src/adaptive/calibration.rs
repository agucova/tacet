//! Calibration phase for adaptive sampling.
//!
//! This module provides a thin wrapper around `tacet_core::adaptive::calibrate()`
//! that adds:
//! - Time measurement using `std::time::Instant`
//! - Platform-specific preflight checks (system configuration)
//!
//! For no_std environments, use `tacet_core::adaptive::calibrate()` directly.

use std::time::Instant;

use crate::preflight::system_check;

// Re-export types from tacet-core for API compatibility
pub use tacet_core::adaptive::{Calibration, CalibrationConfig, CalibrationError};

/// Run calibration phase to estimate covariance and set priors.
///
/// This wrapper around `tacet_core::adaptive::calibrate()` adds:
/// - Time measurement using `std::time::Instant` to compute throughput
/// - Platform-specific preflight checks (system configuration)
///
/// For no_std environments (WASM, embedded), use `tacet_core::adaptive::calibrate()`
/// directly and provide the throughput measurement externally.
///
/// # Arguments
///
/// * `baseline_samples` - Pre-collected baseline timing samples (in native units)
/// * `sample_samples` - Pre-collected sample timing samples (in native units)
/// * `ns_per_tick` - Conversion factor from native units to nanoseconds
/// * `config` - Calibration configuration
///
/// # Returns
///
/// A `Calibration` struct with all computed quantities, or a `CalibrationError`.
pub fn calibrate(
    baseline_samples: &[u64],
    sample_samples: &[u64],
    ns_per_tick: f64,
    config: &CalibrationConfig,
) -> Result<Calibration, CalibrationError> {
    let start = Instant::now();
    let n = baseline_samples.len().min(sample_samples.len());

    // Call core calibration with placeholder throughput
    // (we'll update it after measuring elapsed time)
    let mut calibration = tacet_core::adaptive::calibrate(
        baseline_samples,
        sample_samples,
        ns_per_tick,
        config,
        1_000_000.0, // Placeholder, will be updated
    )?;

    // Update throughput based on actual elapsed time
    let elapsed = start.elapsed().as_secs_f64();
    if elapsed > 0.0 {
        calibration.samples_per_second = n as f64 / elapsed;
    }

    // Add platform-specific preflight checks (system configuration)
    // Core calibration already ran the no_std-compatible checks
    if !config.skip_preflight {
        for warning in system_check() {
            // System warnings are informational and don't affect validity
            // but we need to extend the preflight result to include them
            // For now, just track them separately in a warning log
            // TODO: Add system warnings to a dedicated field
            let _ = warning; // Suppress unused warning
        }
    }

    Ok(calibration)
}

/// Estimate calibration samples needed for desired MDE.
///
/// Since MDE scales as 1/sqrt(n), to halve MDE you need 4x samples.
///
/// # Arguments
///
/// * `current_mde` - Current MDE from calibration
/// * `target_mde` - Desired MDE
/// * `current_n` - Current sample count
///
/// # Returns
///
/// Estimated samples needed to achieve target MDE.
#[allow(dead_code)]
pub fn estimate_samples_for_mde(current_mde: f64, target_mde: f64, current_n: usize) -> usize {
    if target_mde >= current_mde || target_mde <= 0.0 {
        return current_n;
    }

    // MDE scales as 1/sqrt(n), so n scales as (MDE_current / MDE_target)^2
    let scale = (current_mde / target_mde).powi(2);
    ((current_n as f64) * scale).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_samples_for_mde() {
        // To halve MDE, need 4x samples
        let estimate = estimate_samples_for_mde(10.0, 5.0, 1000);
        assert_eq!(estimate, 4000);

        // To get 1/4 MDE, need 16x samples
        let estimate = estimate_samples_for_mde(10.0, 2.5, 1000);
        assert_eq!(estimate, 16000);
    }

    #[test]
    fn test_estimate_samples_target_already_met() {
        let estimate = estimate_samples_for_mde(5.0, 10.0, 1000);
        assert_eq!(estimate, 1000); // Already at target or better
    }

    #[test]
    fn test_calibration_config_default() {
        let config = CalibrationConfig::default();
        assert_eq!(config.calibration_samples, 5000);
        assert_eq!(config.bootstrap_iterations, 200);
    }

    #[test]
    fn test_calibration_basic() {
        // Generate some mock timing data with noise
        let baseline: Vec<u64> = (0..1000).map(|i| 1000 + (i % 10)).collect();
        let sample: Vec<u64> = (0..1000).map(|i| 1005 + (i % 10)).collect();

        let config = CalibrationConfig {
            calibration_samples: 1000,
            bootstrap_iterations: 50, // Fewer for test speed
            timer_resolution_ns: 1.0,
            theta_ns: 100.0,
            alpha: 0.01,
            seed: 42,
            skip_preflight: true, // Skip in tests for speed
            force_discrete_mode: false,
        };

        let result = calibrate(&baseline, &sample, 1.0, &config);

        assert!(
            result.is_ok(),
            "Calibration should succeed: {:?}",
            result.err()
        );
        let cal = result.unwrap();

        assert!(
            cal.sigma_rate.trace() > 0.0,
            "Sigma rate should be positive"
        );
        assert!(cal.block_length >= 1, "Block length should be at least 1");
        assert!(
            cal.prior_cov_marginal[(0, 0)] > 0.0,
            "Prior marginal variance should be positive"
        );
        assert!(cal.sigma_t > 0.0, "Sigma t should be positive");
        assert!(
            cal.samples_per_second > 0.0,
            "Throughput should be positive"
        );
    }

    #[test]
    fn test_calibration_too_few_samples() {
        let baseline: Vec<u64> = vec![1000, 1001, 1002];
        let sample: Vec<u64> = vec![1005, 1006, 1007];

        let config = CalibrationConfig::default();
        let result = calibrate(&baseline, &sample, 1.0, &config);

        assert!(matches!(
            result,
            Err(CalibrationError::TooFewSamples { .. })
        ));
    }
}
