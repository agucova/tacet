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

extern crate alloc;

use alloc::vec::Vec;
use core::f64::consts::PI;

use nalgebra::Cholesky;
use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::constants::DEFAULT_SEED;
use crate::math;
use crate::types::{Matrix9, Vector9};

use super::CalibrationSnapshot;

/// Conservative prior scale factor used as fallback.
/// Chosen to give higher exceedance probability than the target 62%.
const CONSERVATIVE_PRIOR_SCALE: f64 = 1.5;

/// Target prior exceedance probability.
const TARGET_EXCEEDANCE: f64 = 0.62;

/// Number of Monte Carlo samples for prior calibration.
const PRIOR_CALIBRATION_SAMPLES: usize = 50_000;

/// Maximum iterations for prior calibration root-finding.
const MAX_CALIBRATION_ITERATIONS: usize = 20;

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

    /// Block length from Politis-White algorithm.
    /// Used for block bootstrap to preserve autocorrelation structure.
    pub block_length: usize,

    /// 9D prior covariance for δ = (δ₁, ..., δ₉).
    /// Λ₀ = σ²_prior × S where S = Σ_rate / tr(Σ_rate) (shaped).
    pub prior_cov_9d: Matrix9,

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
    pub calibration_snapshot: CalibrationSnapshot,

    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,

    /// Measured throughput for time estimation (samples per second).
    /// The caller is responsible for measuring this during calibration.
    pub samples_per_second: f64,

    /// Floor-rate constant.
    /// Computed once at calibration: 95th percentile of max_k|Z_k| where Z ~ N(0, Σ_rate).
    /// Used for analytical theta_floor computation: theta_floor_stat(n) = c_floor / sqrt(n).
    pub c_floor: f64,

    /// Projection mismatch threshold.
    /// 99th percentile of bootstrap Q_proj distribution.
    pub projection_mismatch_thresh: f64,

    /// Timer resolution floor component.
    /// theta_tick = (1 tick in ns) / K where K is the batch size.
    pub theta_tick: f64,

    /// Effective threshold for this run.
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
        prior_cov_9d: Matrix9,
        theta_ns: f64,
        calibration_samples: usize,
        discrete_mode: bool,
        mde_shift_ns: f64,
        mde_tail_ns: f64,
        calibration_snapshot: CalibrationSnapshot,
        timer_resolution_ns: f64,
        samples_per_second: f64,
        c_floor: f64,
        projection_mismatch_thresh: f64,
        theta_tick: f64,
        theta_eff: f64,
        theta_floor_initial: f64,
        rng_seed: u64,
    ) -> Self {
        Self {
            sigma_rate,
            block_length,
            prior_cov_9d,
            theta_ns,
            calibration_samples,
            discrete_mode,
            mde_shift_ns,
            mde_tail_ns,
            calibration_snapshot,
            timer_resolution_ns,
            samples_per_second,
            c_floor,
            projection_mismatch_thresh,
            theta_tick,
            theta_eff,
            theta_floor_initial,
            rng_seed,
        }
    }

    /// Scale sigma_rate to get covariance for n samples.
    ///
    /// Σ_n = Σ_rate / n
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

/// Compute shaped 9D prior covariance.
///
/// Λ₀ = σ²_prior × S where S = Σ_rate / tr(Σ_rate)
///
/// The shaped prior matches the empirical covariance structure while
/// maintaining the desired exceedance probability.
pub fn compute_prior_cov_9d(sigma_rate: &Matrix9, sigma_prior: f64) -> Matrix9 {
    let trace: f64 = (0..9).map(|i| sigma_rate[(i, i)]).sum();
    if trace < 1e-12 {
        // Fallback to identity if sigma_rate is degenerate
        return Matrix9::identity() * (sigma_prior * sigma_prior / 9.0);
    }

    // S = Σ_rate / tr(Σ_rate), so tr(S) = 1
    let s = sigma_rate / trace;

    // Apply shrinkage if needed for numerical stability
    let s = apply_prior_shrinkage(&s);

    // Λ₀ = σ²_prior × S
    s * (sigma_prior * sigma_prior)
}

/// Apply shrinkage to the shape matrix for numerical stability.
///
/// S ← (1-λ)S + λ × I_9/9 where λ is chosen based on conditioning.
fn apply_prior_shrinkage(s: &Matrix9) -> Matrix9 {
    // Check conditioning via diagonal variance ratio
    let diag: Vec<f64> = (0..9).map(|i| s[(i, i)]).collect();
    let max_diag = diag.iter().cloned().fold(0.0_f64, f64::max);
    let min_diag = diag.iter().cloned().fold(f64::INFINITY, f64::min);

    if min_diag < 1e-12 || max_diag / min_diag.max(1e-12) > 100.0 {
        // Poor conditioning: apply shrinkage
        let lambda = 0.1;
        let identity_scaled = Matrix9::identity() / 9.0;
        s * (1.0 - lambda) + identity_scaled * lambda
    } else {
        *s
    }
}

/// Calibrate σ_prior so that P(max_k |δ_k| > θ_eff | δ ~ N(0, Λ₀)) = π₀.
///
/// Uses binary search to find the scale factor.
/// Falls back to CONSERVATIVE_PRIOR_SCALE (1.5) with warning if calibration fails.
///
/// # Arguments
/// * `sigma_rate` - Covariance rate matrix from calibration
/// * `theta_eff` - Effective threshold in nanoseconds
/// * `seed` - Deterministic RNG seed
///
/// # Returns
/// The calibrated σ_prior value.
pub fn calibrate_prior_scale(sigma_rate: &Matrix9, theta_eff: f64, seed: u64) -> f64 {
    // Binary search for sigma_prior
    let mut lo = theta_eff * 0.1; // Lower bound
    let mut hi = theta_eff * 5.0; // Upper bound

    for _ in 0..MAX_CALIBRATION_ITERATIONS {
        let mid = (lo + hi) / 2.0;
        let lambda0 = compute_prior_cov_9d(sigma_rate, mid);
        let exceedance = compute_prior_exceedance(&lambda0, theta_eff, seed);

        if (exceedance - TARGET_EXCEEDANCE).abs() < 0.01 {
            return mid; // Close enough
        }

        if exceedance > TARGET_EXCEEDANCE {
            // Too much exceedance -> reduce scale
            hi = mid;
        } else {
            // Too little exceedance -> increase scale
            lo = mid;
        }
    }

    // Fallback to conservative value
    theta_eff * CONSERVATIVE_PRIOR_SCALE
}

/// Compute P(max_k |δ_k| > θ | δ ~ N(0, Λ₀)) via Monte Carlo.
fn compute_prior_exceedance(lambda0: &Matrix9, theta: f64, seed: u64) -> f64 {
    let chol = match Cholesky::new(*lambda0) {
        Some(c) => c,
        None => return 0.5, // Neutral if decomposition fails
    };
    let l = chol.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut count = 0usize;

    for _ in 0..PRIOR_CALIBRATION_SAMPLES {
        // Sample z ~ N(0, I_9)
        let mut z = Vector9::zeros();
        for i in 0..9 {
            z[i] = sample_standard_normal(&mut rng);
        }

        // Transform to δ ~ N(0, Λ₀)
        let delta = l * z;

        // Check if max_k |δ_k| > θ
        let max_effect = delta.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if max_effect > theta {
            count += 1;
        }
    }

    count as f64 / PRIOR_CALIBRATION_SAMPLES as f64
}

/// Compute floor-rate constant c_floor for 9D model.
///
/// c_floor = q_95(max_k |Z_k|) where Z ~ N(0, Σ_rate)
///
/// Used for theta_floor computation: theta_floor_stat(n) = c_floor / sqrt(n)
pub fn compute_c_floor_9d(sigma_rate: &Matrix9, seed: u64) -> f64 {
    let chol = match Cholesky::new(*sigma_rate) {
        Some(c) => c,
        None => {
            // Fallback: use trace-based approximation
            let trace: f64 = (0..9).map(|i| sigma_rate[(i, i)]).sum();
            return math::sqrt(trace / 9.0) * 2.5; // Approximate 95th percentile
        }
    };
    let l = chol.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut max_effects = Vec::with_capacity(PRIOR_CALIBRATION_SAMPLES);

    for _ in 0..PRIOR_CALIBRATION_SAMPLES {
        let mut z = Vector9::zeros();
        for i in 0..9 {
            z[i] = sample_standard_normal(&mut rng);
        }

        let sample = l * z;
        let max_effect = sample.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        max_effects.push(max_effect);
    }

    // 95th percentile
    max_effects.sort_by(|a, b| a.total_cmp(b));
    let idx = (PRIOR_CALIBRATION_SAMPLES as f64 * 0.95) as usize;
    max_effects[idx.min(PRIOR_CALIBRATION_SAMPLES - 1)]
}

/// Sample from standard normal using Box-Muller transform.
fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    math::sqrt(-2.0 * math::ln(u1.max(1e-12))) * math::cos(2.0 * PI * u2)
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

        let sigma_rate = Matrix9::identity() * 1000.0;
        let prior_cov = compute_prior_cov_9d(&sigma_rate, 100.0 * 1.12);

        Calibration::new(
            sigma_rate, 10, prior_cov, 100.0, 5000, false, 5.0, 10.0, snapshot, 1.0, 10000.0, 10.0,
            18.48, 0.001, 100.0, 0.1, 42,
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
    fn test_compute_prior_cov_9d_trace_one() {
        // Use identity matrix so trace = 9 (not 9*9=81)
        let sigma_rate = Matrix9::identity();
        let prior = compute_prior_cov_9d(&sigma_rate, 10.0);

        // S = I / trace(I) = I / 9
        // Λ₀ = σ²_prior × S = 100 × I / 9 = I × 11.11
        // Each diagonal should be ~11.11 (σ² / 9 = 100/9)
        let expected = 100.0 / 9.0;
        for i in 0..9 {
            assert!(
                (prior[(i, i)] - expected).abs() < 1.0,
                "Diagonal {} was {}, expected ~{}",
                i,
                prior[(i, i)],
                expected
            );
        }
    }

    #[test]
    fn test_prior_exceedance_calibration() {
        let sigma_rate = Matrix9::identity() * 100.0;
        let theta_eff = 10.0;

        let sigma_prior = calibrate_prior_scale(&sigma_rate, theta_eff, 42);

        // Check that calibration gives reasonable value
        assert!(
            sigma_prior > theta_eff * 0.5,
            "sigma_prior should be > theta_eff/2"
        );
        assert!(
            sigma_prior < theta_eff * 3.0,
            "sigma_prior should be < 3*theta_eff"
        );

        // Verify exceedance is near target
        let lambda0 = compute_prior_cov_9d(&sigma_rate, sigma_prior);
        let exceedance = compute_prior_exceedance(&lambda0, theta_eff, 42);
        assert!(
            (exceedance - TARGET_EXCEEDANCE).abs() < 0.05,
            "Exceedance {} should be near {}",
            exceedance,
            TARGET_EXCEEDANCE
        );
    }

    #[test]
    fn test_c_floor_computation() {
        let sigma_rate = Matrix9::identity() * 100.0;
        let c_floor = compute_c_floor_9d(&sigma_rate, 42);

        // c_floor should be roughly sqrt(100) * 2 to 3 for 95th percentile of max
        assert!(c_floor > 15.0, "c_floor {} should be > 15", c_floor);
        assert!(c_floor < 40.0, "c_floor {} should be < 40", c_floor);
    }

    #[test]
    fn test_calibration_config_default() {
        let config = CalibrationConfig::default();
        assert_eq!(config.calibration_samples, 5000);
        assert_eq!(config.bootstrap_iterations, 2000);
        assert!((config.theta_ns - 100.0).abs() < 1e-10);
    }
}
