//! Calibration phase for adaptive sampling.
//!
//! The calibration phase collects initial samples and estimates key quantities
//! needed for the adaptive loop (spec Section 2.3, Section 2.6):
//!
//! - **Covariance rate** (Sigma_rate = Sigma_cal * n_cal): Allows efficient covariance
//!   scaling without re-bootstrapping. For n samples, Sigma_n = Sigma_rate / n.
//!
//! - **Block length**: Optimal block length for bootstrap via Politis-White algorithm,
//!   accounting for autocorrelation in timing measurements.
//!
//! - **Prior scale**: Based on MDE, sets the Bayesian prior variance for beta.
//!
//! - **Discrete mode detection**: If < 10% unique values, use mid-quantile estimators.

use std::time::Instant;

use crate::analysis::mde::estimate_mde;
use crate::preflight::{run_all_checks, PreflightResult};
use crate::statistics::{bootstrap_difference_covariance, paired_optimal_block_length};
use crate::types::{Class, Matrix2, Matrix9, TimingSample};

/// Calibration results from the initial measurement phase.
///
/// These values are used to configure the adaptive sampling loop.
#[derive(Debug, Clone)]
pub struct Calibration {
    /// Covariance "rate" - multiply by 1/n to get Sigma_n for n samples.
    /// Computed as Sigma_cal * n_cal where Sigma_cal is calibration covariance.
    /// This allows O(1) covariance scaling as samples accumulate.
    pub sigma_rate: Matrix9,

    /// Block length from Politis-White algorithm (spec Section 2.6).
    /// Used for block bootstrap to preserve autocorrelation structure.
    pub block_length: usize,

    /// Prior covariance for beta = (mu, tau).
    /// Set to diag(sigma_mu^2, sigma_tau^2) where sigma = max(2*MDE, theta).
    pub prior_cov: Matrix2,

    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,

    /// Measured throughput for time estimation (samples per second).
    pub samples_per_second: f64,

    /// Whether discrete mode is active (< 10% unique values).
    /// When true, use mid-quantile estimators and m-out-of-n bootstrap.
    pub discrete_mode: bool,

    /// The theta threshold being used (in nanoseconds).
    pub theta_ns: f64,

    /// Number of calibration samples collected per class.
    pub calibration_samples: usize,

    /// Minimum detectable effect from calibration (for prior setting).
    pub mde_shift_ns: f64,

    /// Minimum detectable tail effect from calibration.
    pub mde_tail_ns: f64,

    /// Results of preflight checks run during calibration.
    pub preflight_result: PreflightResult,
}

/// Errors that can occur during calibration.
#[derive(Debug, Clone)]
pub enum CalibrationError {
    /// Too few samples collected for reliable calibration.
    TooFewSamples {
        /// Number of samples actually collected.
        collected: usize,
        /// Minimum required samples.
        minimum: usize,
    },

    /// Covariance estimation failed (e.g., singular matrix).
    CovarianceEstimationFailed {
        /// Reason for failure.
        reason: String,
    },

    /// A preflight check failed before calibration.
    PreflightCheckFailed {
        /// Which check failed.
        check: String,
        /// Error message.
        message: String,
    },

    /// Timer is too coarse to measure this operation.
    TimerTooCoarse {
        /// Timer resolution in nanoseconds.
        resolution_ns: f64,
        /// Measured operation time in nanoseconds.
        operation_ns: f64,
    },
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrationError::TooFewSamples { collected, minimum } => {
                write!(
                    f,
                    "Too few samples: collected {}, need at least {}",
                    collected, minimum
                )
            }
            CalibrationError::CovarianceEstimationFailed { reason } => {
                write!(f, "Covariance estimation failed: {}", reason)
            }
            CalibrationError::PreflightCheckFailed { check, message } => {
                write!(f, "Preflight check '{}' failed: {}", check, message)
            }
            CalibrationError::TimerTooCoarse {
                resolution_ns,
                operation_ns,
            } => {
                write!(
                    f,
                    "Timer resolution ({:.1}ns) too coarse for operation ({:.1}ns)",
                    resolution_ns, operation_ns
                )
            }
        }
    }
}

impl std::error::Error for CalibrationError {}

/// Configuration for calibration phase.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Number of samples to collect per class during calibration.
    pub calibration_samples: usize,

    /// Number of bootstrap iterations for covariance estimation.
    pub bootstrap_iterations: usize,

    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,

    /// Theta threshold in nanoseconds.
    pub theta_ns: f64,

    /// Alpha level for MDE computation.
    pub alpha: f64,

    /// Random seed for bootstrap and preflight randomization.
    pub seed: u64,

    /// Optional: time to generate baseline inputs (nanoseconds).
    pub baseline_gen_time_ns: Option<f64>,

    /// Optional: time to generate sample inputs (nanoseconds).
    pub sample_gen_time_ns: Option<f64>,

    /// Whether to skip preflight checks.
    pub skip_preflight: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            calibration_samples: 5000,
            bootstrap_iterations: 200, // Fewer iterations for calibration phase
            timer_resolution_ns: 1.0,
            theta_ns: 100.0,
            alpha: 0.01,
            seed: 42,
            baseline_gen_time_ns: None,
            sample_gen_time_ns: None,
            skip_preflight: false,
        }
    }
}

/// Run calibration phase to estimate covariance and set priors.
///
/// This collects `calibration_samples` and estimates:
/// - Covariance rate (Sigma_rate = Sigma_cal * n_cal)
/// - Optimal block length via Politis-White
/// - Prior scale from MDE
/// - Timer resolution and throughput
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
    let n = baseline_samples.len().min(sample_samples.len());

    // Check minimum sample requirement
    const MIN_CALIBRATION_SAMPLES: usize = 100;
    if n < MIN_CALIBRATION_SAMPLES {
        return Err(CalibrationError::TooFewSamples {
            collected: n,
            minimum: MIN_CALIBRATION_SAMPLES,
        });
    }

    let start = Instant::now();

    // Convert to nanoseconds for analysis
    let baseline_ns: Vec<f64> = baseline_samples[..n]
        .iter()
        .map(|&t| t as f64 * ns_per_tick)
        .collect();
    let sample_ns: Vec<f64> = sample_samples[..n]
        .iter()
        .map(|&t| t as f64 * ns_per_tick)
        .collect();

    // Check discrete mode (spec Section 2.4): < 10% unique values
    let unique_baseline = count_unique(&baseline_ns);
    let unique_sample = count_unique(&sample_ns);
    let min_uniqueness = (unique_baseline as f64 / n as f64).min(unique_sample as f64 / n as f64);
    let discrete_mode = min_uniqueness < 0.10;

    // Compute block length (spec Section 2.6)
    let block_length = if n >= 10 {
        paired_optimal_block_length(&baseline_ns, &sample_ns)
    } else {
        1
    };

    // Create interleaved TimingSample sequence for joint bootstrap
    let interleaved: Vec<TimingSample> = baseline_ns
        .iter()
        .zip(sample_ns.iter())
        .flat_map(|(&b, &s)| {
            [
                TimingSample {
                    time_ns: b,
                    class: Class::Baseline,
                },
                TimingSample {
                    time_ns: s,
                    class: Class::Sample,
                },
            ]
        })
        .collect();

    // Bootstrap covariance estimation
    let cov_estimate =
        bootstrap_difference_covariance(&interleaved, config.bootstrap_iterations, config.seed);

    // Check covariance validity
    if !cov_estimate.is_stable() {
        return Err(CalibrationError::CovarianceEstimationFailed {
            reason: "Covariance matrix is not positive definite".to_string(),
        });
    }

    // Compute sigma rate: Sigma_rate = Sigma_cal * n_cal
    // For n samples, Sigma_n = Sigma_rate / n
    let sigma_rate = cov_estimate.matrix * (n as f64);

    // Compute MDE for prior setting (spec Section 2.7)
    let mde = estimate_mde(&cov_estimate.matrix, config.alpha);

    // Set prior covariance: sigma proportional to theta (spec Section 2.5)
    //
    // Previously we used max(2*MDE, theta), but this caused poor calibration:
    // when MDE >> theta (noisy data, strict threshold), the prior became too
    // diffuse and P(|β| > θ | prior) ≈ 99%, biasing toward "leak detected"
    // even before seeing data.
    //
    // Using sigma = 2*theta ensures P(|β| > θ | prior) ≈ 62%, representing
    // reasonable uncertainty. The variance ratio quality gate (Gate 1) catches
    // cases where posterior ≈ prior (data uninformative).
    let prior_sigma_shift = config.theta_ns * 2.0;
    let prior_sigma_tail = config.theta_ns * 2.0;
    let prior_cov = Matrix2::new(
        prior_sigma_shift.powi(2),
        0.0,
        0.0,
        prior_sigma_tail.powi(2),
    );

    // Compute throughput
    let elapsed = start.elapsed();
    let samples_per_second = if elapsed.as_secs_f64() > 0.0 {
        n as f64 / elapsed.as_secs_f64()
    } else {
        1_000_000.0 // Conservative default if measurement was instant
    };

    // Run preflight checks (unless skipped)
    let preflight_result = if config.skip_preflight {
        PreflightResult::new()
    } else {
        run_all_checks(
            &baseline_ns,
            &sample_ns,
            config.baseline_gen_time_ns,
            config.sample_gen_time_ns,
            config.timer_resolution_ns,
            config.seed,
        )
    };

    Ok(Calibration {
        sigma_rate,
        block_length,
        prior_cov,
        timer_resolution_ns: config.timer_resolution_ns,
        samples_per_second,
        discrete_mode,
        theta_ns: config.theta_ns,
        calibration_samples: n,
        mde_shift_ns: mde.shift_ns,
        mde_tail_ns: mde.tail_ns,
        preflight_result,
    })
}

/// Count unique values in a slice (for discrete mode detection).
fn count_unique(values: &[f64]) -> usize {
    use std::collections::HashSet;

    // Discretize to avoid floating point comparison issues
    // Use 0.001ns buckets (well below any meaningful timing difference)
    let buckets: HashSet<i64> = values.iter().map(|&v| (v * 1000.0) as i64).collect();
    buckets.len()
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
    fn test_count_unique() {
        let values = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
        assert_eq!(count_unique(&values), 4);
    }

    #[test]
    fn test_count_unique_continuous() {
        // Continuous data should have many unique values
        let values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
        assert_eq!(count_unique(&values), 1000);
    }

    #[test]
    fn test_count_unique_discrete() {
        // Discrete timer: only 10 unique values among 1000 samples
        let values: Vec<f64> = (0..1000).map(|i| (i % 10) as f64 * 41.0).collect();
        assert_eq!(count_unique(&values), 10);
    }

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
    fn test_calibration_error_display() {
        let err = CalibrationError::TooFewSamples {
            collected: 50,
            minimum: 100,
        };
        assert!(err.to_string().contains("50"));
        assert!(err.to_string().contains("100"));

        let err = CalibrationError::PreflightCheckFailed {
            check: "sanity".to_string(),
            message: "timer broken".to_string(),
        };
        assert!(err.to_string().contains("sanity"));
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
            baseline_gen_time_ns: None,
            sample_gen_time_ns: None,
            skip_preflight: true, // Skip in tests for speed
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
            cal.prior_cov[(0, 0)] > 0.0,
            "Prior variance should be positive"
        );
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
