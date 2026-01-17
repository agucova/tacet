//! Calibration data for adaptive sampling (no_std compatible).
//!
//! This module defines the calibration results needed for the adaptive sampling loop.
//! The actual calibration computation lives in the `timing-oracle` crate; this module
//! provides the data structures that can be used in no_std environments.
//!
//! For no_std compatibility:
//! - No `std::time::Instant` (throughput measured by caller)
//! - No preflight checks (those require std features)
//! - Uses only `alloc` for heap allocation

use crate::constants::DEFAULT_SEED;
use crate::types::{Matrix2, Matrix9};

use super::CalibrationSnapshot;

/// Calibration results from the initial measurement phase (no_std compatible).
///
/// This struct contains the essential statistical data needed for the adaptive
/// sampling loop. It is designed for use in no_std environments like SGX enclaves.
///
/// For full calibration with preflight checks, see `timing_oracle::Calibration`.
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
    /// Set to diag(sigma_mu^2, sigma_tau^2) where sigma = 1.12 * theta.
    pub prior_cov: Matrix2,

    /// The theta threshold being used (in nanoseconds).
    pub theta_ns: f64,

    /// Number of calibration samples collected per class.
    pub calibration_samples: usize,

    /// Whether discrete mode is active (< 10% unique values).
    /// When true, use mid-quantile estimators and m-out-of-n bootstrap.
    pub discrete_mode: bool,

    /// Minimum detectable effect (shift component) from calibration.
    pub mde_shift_ns: f64,

    /// Minimum detectable effect (tail component) from calibration.
    pub mde_tail_ns: f64,

    /// Statistics snapshot from calibration phase for drift detection.
    /// Used to compare against post-test statistics (spec Section 2.6, Gate 6).
    pub calibration_snapshot: CalibrationSnapshot,

    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,

    /// Measured throughput for time estimation (samples per second).
    /// The caller is responsible for measuring this during calibration.
    pub samples_per_second: f64,

    // === v4.1 additions ===
    /// Floor-rate constant (spec Section 2.3.4).
    /// Computed once at calibration: 95th percentile of max|Z_k| where Z ~ N(0, Σ_pred,rate).
    /// Used for analytical theta_floor computation: theta_floor_stat(n) = c_floor / sqrt(n).
    pub c_floor: f64,

    /// Model mismatch threshold (spec Section 2.3.3, 2.6 Gate 8).
    /// 99th percentile of bootstrap Q* distribution, or chi-squared(7, 0.99) ≈ 18.48 fallback.
    pub q_thresh: f64,

    /// Timer resolution floor component (spec Section 2.3.4).
    /// theta_tick = (1 tick in ns) / K where K is the batch size.
    pub theta_tick: f64,

    /// Effective threshold for this run (spec Section 2.3.4).
    /// theta_eff = max(theta_user, theta_floor) or just theta_floor in research mode.
    pub theta_eff: f64,

    /// Initial measurement floor at calibration time.
    pub theta_floor_initial: f64,

    /// Deterministic RNG seed used for this run.
    pub rng_seed: u64,
}

impl Calibration {
    /// Create a new Calibration with all required fields.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sigma_rate: Matrix9,
        block_length: usize,
        prior_cov: Matrix2,
        theta_ns: f64,
        calibration_samples: usize,
        discrete_mode: bool,
        mde_shift_ns: f64,
        mde_tail_ns: f64,
        calibration_snapshot: CalibrationSnapshot,
        timer_resolution_ns: f64,
        samples_per_second: f64,
        // v4.1 fields
        c_floor: f64,
        q_thresh: f64,
        theta_tick: f64,
        theta_eff: f64,
        theta_floor_initial: f64,
        rng_seed: u64,
    ) -> Self {
        Self {
            sigma_rate,
            block_length,
            prior_cov,
            theta_ns,
            calibration_samples,
            discrete_mode,
            mde_shift_ns,
            mde_tail_ns,
            calibration_snapshot,
            timer_resolution_ns,
            samples_per_second,
            // v4.1 fields
            c_floor,
            q_thresh,
            theta_tick,
            theta_eff,
            theta_floor_initial,
            rng_seed,
        }
    }

    /// Scale sigma_rate to get covariance for n samples.
    ///
    /// Per spec Section 2.8: Σ_n = Σ_rate / n
    pub fn covariance_for_n(&self, n: usize) -> Matrix9 {
        if n == 0 {
            return self.sigma_rate; // Avoid division by zero
        }
        self.sigma_rate / (n as f64)
    }

    /// Estimate time to collect `n` additional samples based on calibration throughput.
    ///
    /// Returns estimated seconds. Caller should add any overhead.
    pub fn estimate_collection_time_secs(&self, n: usize) -> f64 {
        if self.samples_per_second <= 0.0 {
            return 0.0;
        }
        n as f64 / self.samples_per_second
    }
}

/// Configuration for calibration phase (no_std compatible).
///
/// This contains parameters for calibration that don't require std features.
/// For full configuration with preflight options, see `timing_oracle::CalibrationConfig`.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Number of samples to collect per class during calibration.
    pub calibration_samples: usize,

    /// Number of bootstrap iterations for covariance estimation.
    pub bootstrap_iterations: usize,

    /// Theta threshold in nanoseconds.
    pub theta_ns: f64,

    /// Alpha level for MDE computation.
    pub alpha: f64,

    /// Random seed for bootstrap.
    pub seed: u64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            calibration_samples: 5000,
            bootstrap_iterations: 2000, // Full bootstrap for accurate covariance
            theta_ns: 100.0,
            alpha: 0.01,
            seed: DEFAULT_SEED,
        }
    }
}

/// Prior scale factor for setting prior covariance.
///
/// The prior scale is calibrated so that P(max_k |(Xβ)_k| > θ | prior) ≈ 62%.
/// With β ~ N(0, σ²I₂) and the 9×2 design matrix X, σ = 1.12θ achieves this.
/// See bayes.rs for derivation.
pub const PRIOR_SCALE_FACTOR: f64 = 1.12;

/// Compute prior covariance from effective theta threshold.
///
/// **Important:** Pass `theta_eff` (effective threshold), not `theta_user`.
/// Per spec §2.3.4, the prior should be based on the effective threshold
/// which accounts for the measurement floor.
///
/// Returns a 2x2 diagonal matrix with variance = (1.12 * theta)² for both
/// the shift (mu) and tail (tau) components.
pub fn compute_prior_cov(theta_eff_ns: f64) -> Matrix2 {
    let sigma = theta_eff_ns * PRIOR_SCALE_FACTOR;
    Matrix2::new(sigma * sigma, 0.0, 0.0, sigma * sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::StatsSnapshot;

    fn make_test_calibration() -> Calibration {
        let snapshot = CalibrationSnapshot::new(
            StatsSnapshot {
                count: 1000,
                mean: 100.0,
                variance: 25.0,
                autocorr_lag1: 0.1,
            },
            StatsSnapshot {
                count: 1000,
                mean: 105.0,
                variance: 30.0,
                autocorr_lag1: 0.12,
            },
        );

        Calibration::new(
            Matrix9::identity() * 1000.0, // sigma_rate
            10,                           // block_length
            compute_prior_cov(100.0),     // prior_cov
            100.0,                        // theta_ns
            5000,                         // calibration_samples
            false,                        // discrete_mode
            5.0,                          // mde_shift_ns
            10.0,                         // mde_tail_ns
            snapshot,                     // calibration_snapshot
            1.0,                          // timer_resolution_ns
            10000.0,                      // samples_per_second
            // v4.1 fields
            10.0,  // c_floor
            18.48, // q_thresh (chi-squared(7, 0.99) fallback)
            0.001, // theta_tick
            100.0, // theta_eff
            0.1,   // theta_floor_initial
            42,    // rng_seed
        )
    }

    #[test]
    fn test_covariance_scaling() {
        let cal = make_test_calibration();

        // At n=1000, covariance should be sigma_rate / 1000
        let cov_1000 = cal.covariance_for_n(1000);
        assert!((cov_1000[(0, 0)] - 1.0).abs() < 1e-10);

        // At n=2000, covariance should be half
        let cov_2000 = cal.covariance_for_n(2000);
        assert!((cov_2000[(0, 0)] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_zero_n() {
        let cal = make_test_calibration();
        let cov = cal.covariance_for_n(0);
        // Should return sigma_rate unchanged (avoid NaN)
        assert!((cov[(0, 0)] - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_collection_time() {
        let cal = make_test_calibration();

        // 10000 samples/sec -> 1000 samples takes 0.1 sec
        let time = cal.estimate_collection_time_secs(1000);
        assert!((time - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_compute_prior_cov() {
        let prior = compute_prior_cov(100.0);

        // sigma = 1.12 * 100 = 112
        // variance = 112^2 = 12544
        let expected_var = (100.0 * PRIOR_SCALE_FACTOR).powi(2);
        assert!((prior[(0, 0)] - expected_var).abs() < 1e-10);
        assert!((prior[(1, 1)] - expected_var).abs() < 1e-10);
        assert!(prior[(0, 1)].abs() < 1e-10);
        assert!(prior[(1, 0)].abs() < 1e-10);
    }

    #[test]
    fn test_calibration_config_default() {
        let config = CalibrationConfig::default();
        assert_eq!(config.calibration_samples, 5000);
        assert_eq!(config.bootstrap_iterations, 2000);
        assert!((config.theta_ns - 100.0).abs() < 1e-10);
    }
}
