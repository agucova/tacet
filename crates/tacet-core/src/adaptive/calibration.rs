//! Calibration data and computation for adaptive sampling (no_std compatible).
//!
//! This module defines the calibration results needed for the adaptive sampling loop,
//! as well as the core calibration computation that can run in no_std environments.
//!
//! For no_std compatibility:
//! - No `std::time::Instant` (throughput measured by caller)
//! - Core preflight checks are no_std compatible
//! - Uses only `alloc` for heap allocation
//!
//! # Usage
//!
//! The `calibrate()` function runs the calibration phase using pre-collected samples.
//! Unlike the `tacet` crate version, it takes `samples_per_second` as a parameter
//! instead of measuring it internally (to avoid `std::time::Instant`).

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::analysis::bayes::{sample_gamma, sample_standard_normal};
use crate::constants::DEFAULT_SEED;
use crate::preflight::{run_core_checks, PreflightResult};
use crate::statistics::{
    bootstrap_w1_variance, compute_w1_distance, timing_iact_combined, timing_iact_direct,
    timing_iact_per_quantile, AcquisitionStream, OnlineStats,
};
use crate::types::{Class, TimingSample};

use super::CalibrationSnapshot;

/// Degrees of freedom for half-t prior (W₁ distance).
pub const NU: f64 = 4.0;

/// Calibration results from the initial measurement phase (no_std compatible).
///
/// This struct contains the essential statistical data needed for the adaptive
/// sampling loop. It is designed for use in no_std environments like SGX enclaves.
///
/// For full calibration with preflight checks, see `tacet::Calibration`.
///
/// ## Half-t Prior (W₁ distance)
///
/// The prior is a half-t distribution with ν=4 degrees of freedom:
/// δ ~ half-t(ν=4, σ_t)
///
/// This is implemented via scale mixture: λ ~ Gamma(ν/2, ν/2), δ = |σ_t/√λ · z| where z ~ N(0,1).
/// The marginal variance is (ν/(ν-2)) σ_t² = 2σ_t² for ν=4.
#[derive(Debug, Clone)]
pub struct Calibration {
    /// Variance "rate" - multiply by 1/n to get var_n for n samples (W₁ distance).
    /// Computed as var_cal * n_cal where var_cal is calibration variance.
    /// This allows O(1) variance scaling as samples accumulate.
    pub var_rate: f64,

    /// Block length from Politis-White algorithm.
    /// Used for block bootstrap to preserve autocorrelation structure.
    pub block_length: usize,

    /// IACT (Integrated Autocorrelation Time) estimate.
    /// Used when iact_method = GeyersIMS for effective sample size computation.
    pub iact: f64,

    /// Method used for IACT computation.
    pub iact_method: crate::types::IactMethod,

    /// Calibrated half-t prior scale (W₁ distance).
    /// This is the σ_t in δ ~ half-t(ν=4, σ_t).
    pub sigma_t: f64,

    /// The theta threshold being used (in nanoseconds).
    pub theta_ns: f64,

    /// Number of calibration samples collected per class.
    pub calibration_samples: usize,

    /// Whether discrete mode is active (< 10% unique values).
    /// When true, use mid-quantile estimators and m-out-of-n bootstrap.
    pub discrete_mode: bool,

    /// Minimum detectable effect in nanoseconds.
    pub mde_ns: f64,

    /// Statistics snapshot from calibration phase for drift detection.
    pub calibration_snapshot: CalibrationSnapshot,

    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,

    /// Measured throughput for time estimation (samples per second).
    /// The caller is responsible for measuring this during calibration.
    pub samples_per_second: f64,

    /// Floor-rate constant (W₁ distance).
    /// Computed once at calibration: 95th percentile of |Z| where Z ~ N(0, var_rate).
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

    /// Batch size K for adaptive batching.
    /// When K > 1, samples contain K iterations worth of timing.
    /// Effect estimates must be divided by K to report per-call differences.
    pub batch_k: u32,

    /// Results of core preflight checks run during calibration (no_std compatible).
    /// Platform-specific checks (like system configuration) are handled by the
    /// `tacet` crate's wrapper.
    pub preflight_result: PreflightResult,
}

impl Calibration {
    /// Create a new Calibration with half-t prior (W₁ distance).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        var_rate: f64,
        block_length: usize,
        iact: f64,
        iact_method: crate::types::IactMethod,
        sigma_t: f64,
        theta_ns: f64,
        calibration_samples: usize,
        discrete_mode: bool,
        mde_ns: f64,
        calibration_snapshot: CalibrationSnapshot,
        timer_resolution_ns: f64,
        samples_per_second: f64,
        c_floor: f64,
        projection_mismatch_thresh: f64,
        theta_tick: f64,
        theta_eff: f64,
        theta_floor_initial: f64,
        rng_seed: u64,
        batch_k: u32,
    ) -> Self {
        Self {
            var_rate,
            block_length,
            iact,
            iact_method,
            sigma_t,
            theta_ns,
            calibration_samples,
            discrete_mode,
            mde_ns,
            calibration_snapshot,
            timer_resolution_ns,
            samples_per_second,
            c_floor,
            projection_mismatch_thresh,
            theta_tick,
            theta_eff,
            theta_floor_initial,
            rng_seed,
            batch_k,
            preflight_result: PreflightResult::new(),
        }
    }

    /// Create a new Calibration with all fields including preflight results.
    #[allow(clippy::too_many_arguments)]
    pub fn with_preflight(
        var_rate: f64,
        block_length: usize,
        iact: f64,
        iact_method: crate::types::IactMethod,
        sigma_t: f64,
        theta_ns: f64,
        calibration_samples: usize,
        discrete_mode: bool,
        mde_ns: f64,
        calibration_snapshot: CalibrationSnapshot,
        timer_resolution_ns: f64,
        samples_per_second: f64,
        c_floor: f64,
        projection_mismatch_thresh: f64,
        theta_tick: f64,
        theta_eff: f64,
        theta_floor_initial: f64,
        rng_seed: u64,
        batch_k: u32,
        preflight_result: PreflightResult,
    ) -> Self {
        Self {
            var_rate,
            block_length,
            iact,
            iact_method,
            sigma_t,
            theta_ns,
            calibration_samples,
            discrete_mode,
            mde_ns,
            calibration_snapshot,
            timer_resolution_ns,
            samples_per_second,
            c_floor,
            projection_mismatch_thresh,
            theta_tick,
            theta_eff,
            theta_floor_initial,
            rng_seed,
            batch_k,
            preflight_result,
        }
    }

    /// Compute the dynamic measurement floor at sample size n (spec §3.3.3).
    ///
    /// theta_floor(n) = max(theta_tick, c_floor / sqrt(n_blocks(n)))
    ///
    /// This decreases as sample size grows, reflecting improved measurement precision.
    pub fn theta_floor_at(&self, n: usize) -> f64 {
        let n_blocks = if self.block_length > 0 {
            (n / self.block_length).max(1)
        } else {
            n.max(1)
        };
        libm::fmax(self.c_floor / (n_blocks as f64).sqrt(), self.theta_tick)
    }

    /// Compute the dynamic effective threshold at sample size n (spec §3.3.3).
    ///
    /// theta_eff(n) = max(theta_user, theta_floor(n))
    ///
    /// Implementations MUST recompute this after each batch during adaptive sampling.
    pub fn theta_eff_at(&self, n: usize, theta_user: f64) -> f64 {
        if theta_user > 0.0 {
            libm::fmax(theta_user, self.theta_floor_at(n))
        } else {
            self.theta_floor_at(n)
        }
    }

    /// Compute effective sample size accounting for dependence (spec §3.3.2 v5.6).
    ///
    /// Under strong temporal dependence, n samples do not provide n independent observations.
    /// The number of effectively independent blocks.
    ///
    /// n_blocks = max(1, floor(n / block_length))
    ///
    /// where block_length (PolitisWhite) or IACT (GeyersIMS) is the estimated dependence length.
    pub fn n_blocks(&self, n: usize) -> usize {
        use crate::types::IactMethod;
        match self.iact_method {
            IactMethod::PolitisWhite => {
                if self.block_length == 0 {
                    return n.max(1);
                }
                (n / self.block_length).max(1)
            }
            IactMethod::GeyersIMS | IactMethod::DirectDifferences | IactMethod::PerQuantile => {
                if self.iact <= 1.0 {
                    return n.max(1);
                }
                ((n as f64) / self.iact).floor().max(1.0) as usize
            }
        }
    }

    /// Scale var_rate to get variance for n samples (W₁ distance).
    ///
    /// var_n = var_rate / n
    ///
    /// Since var_rate is computed via block bootstrap (which preserves temporal
    /// dependence), it already represents the long-run variance rate. No additional
    /// IACT scaling is needed—the block bootstrap already accounts for autocorrelation.
    pub fn variance_for_n(&self, n: usize) -> f64 {
        if n == 0 {
            return self.var_rate; // Avoid division by zero
        }
        self.var_rate / (n as f64)
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

    /// Convert to an FFI-friendly summary containing only scalar fields.
    pub fn to_summary(&self) -> crate::ffi_summary::CalibrationSummary {
        crate::ffi_summary::CalibrationSummary {
            block_length: self.block_length,
            calibration_samples: self.calibration_samples,
            discrete_mode: self.discrete_mode,
            timer_resolution_ns: self.timer_resolution_ns,
            theta_ns: self.theta_ns,
            theta_eff: self.theta_eff,
            theta_floor_initial: self.theta_floor_initial,
            theta_tick: self.theta_tick,
            mde_ns: self.mde_ns,
            samples_per_second: self.samples_per_second,
        }
    }
}

/// Configuration for calibration phase (no_std compatible).
///
/// This contains parameters for calibration that don't require std features.
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

    /// Random seed for bootstrap.
    pub seed: u64,

    /// Whether to skip preflight checks.
    pub skip_preflight: bool,

    /// Force discrete mode regardless of uniqueness ratio.
    pub force_discrete_mode: bool,

    /// IACT computation method (PolitisWhite or GeyersIMS).
    pub iact_method: crate::types::IactMethod,

    /// Bootstrap resampling method (Joint or Stratified).
    pub bootstrap_method: crate::statistics::BootstrapMethod,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            calibration_samples: 5000,
            bootstrap_iterations: 200, // Fewer iterations for calibration phase
            timer_resolution_ns: 1.0,
            theta_ns: 100.0,
            alpha: 0.01,
            seed: DEFAULT_SEED,
            skip_preflight: false,
            force_discrete_mode: false,
            iact_method: crate::types::IactMethod::default(),
            bootstrap_method: crate::statistics::BootstrapMethod::default(),
        }
    }
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

impl core::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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

/// Run calibration phase to estimate covariance and set priors (no_std compatible).
///
/// This is the core calibration function that can run in no_std environments.
/// Unlike the `tacet` crate version, throughput (`samples_per_second`) must be
/// provided by the caller since `std::time::Instant` is not available.
///
/// # Arguments
///
/// * `baseline_samples` - Pre-collected baseline timing samples (in native units)
/// * `sample_samples` - Pre-collected sample timing samples (in native units)
/// * `ns_per_tick` - Conversion factor from native units to nanoseconds
/// * `config` - Calibration configuration
/// * `samples_per_second` - Throughput measured by caller (e.g., via performance.now())
///
/// # Returns
///
/// A `Calibration` struct with all computed quantities, or a `CalibrationError`.
pub fn calibrate(
    baseline_samples: &[u64],
    sample_samples: &[u64],
    ns_per_tick: f64,
    config: &CalibrationConfig,
    samples_per_second: f64,
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

    // Convert to nanoseconds for analysis
    let baseline_ns: Vec<f64> = baseline_samples[..n]
        .iter()
        .map(|&t| t as f64 * ns_per_tick)
        .collect();
    let sample_ns: Vec<f64> = sample_samples[..n]
        .iter()
        .map(|&t| t as f64 * ns_per_tick)
        .collect();

    // Check discrete mode (spec §3.6): < 10% unique values
    let unique_baseline = count_unique(&baseline_ns);
    let unique_sample = count_unique(&sample_ns);
    let min_uniqueness = (unique_baseline as f64 / n as f64).min(unique_sample as f64 / n as f64);
    let discrete_mode = config.force_discrete_mode || min_uniqueness < 0.10;

    // Create acquisition stream for joint bootstrap (spec Section 2.3.1)
    let mut acquisition_stream = AcquisitionStream::with_capacity(2 * n);
    acquisition_stream.push_batch_interleaved(&baseline_ns, &sample_ns);
    let interleaved = acquisition_stream.to_timing_samples();

    // Bootstrap variance estimation for W₁ distance
    let var_estimate = bootstrap_w1_variance(
        &interleaved,
        config.bootstrap_iterations,
        config.seed,
        false, // is_fragile
        config.bootstrap_method,
    );

    // Check variance validity (must be positive and finite)
    if !var_estimate.variance.is_finite() || var_estimate.variance <= 0.0 {
        return Err(CalibrationError::CovarianceEstimationFailed {
            reason: String::from("W₁ variance estimate is not positive and finite"),
        });
    }

    // Compute variance rate: var_rate = var_cal * n_cal
    let var_rate = var_estimate.variance * (n as f64);

    // Invariant check: var_rate / n should equal var_estimate.variance
    // This validates that our variance scaling is consistent (v6.0)
    #[cfg(debug_assertions)]
    {
        let var_n_at_cal = var_rate / (n as f64);
        let diff = (var_n_at_cal - var_estimate.variance).abs();
        assert!(
            diff < 1e-6,
            "Variance scaling invariant violated: difference = {} (expected < 1e-6). \
             This indicates var_rate scaling is incorrect.",
            diff
        );
    }

    // Compute IACT based on method
    let (block_length, iact, iact_method) = match config.iact_method {
        crate::types::IactMethod::PolitisWhite => (
            var_estimate.block_size,
            1.0,
            crate::types::IactMethod::PolitisWhite,
        ),
        crate::types::IactMethod::GeyersIMS => {
            let iact_result = timing_iact_combined(&interleaved);

            // Warnings are captured in iact_result but don't affect calibration
            // They will be logged by the caller if needed

            (
                var_estimate.block_size,
                iact_result.tau,
                crate::types::IactMethod::GeyersIMS,
            )
        }
        crate::types::IactMethod::DirectDifferences => {
            let iact_result = timing_iact_direct(&interleaved);

            // Warnings are captured in iact_result but don't affect calibration
            // They will be logged by the caller if needed

            (
                var_estimate.block_size,
                iact_result.tau,
                crate::types::IactMethod::DirectDifferences,
            )
        }
        crate::types::IactMethod::PerQuantile => {
            let iact_result = timing_iact_per_quantile(&interleaved);

            // Warnings are captured in iact_result but don't affect calibration
            // They will be logged by the caller if needed

            (
                var_estimate.block_size,
                iact_result.tau,
                crate::types::IactMethod::PerQuantile,
            )
        }
    };

    // Compute MDE from variance estimate (1D W₁ distance)
    // MDE ≈ z_α/2 * sqrt(var) for standard normal quantile z_α/2
    let z_alpha = 1.96; // 95% confidence for alpha=0.05
    let mde_ns = z_alpha * var_estimate.variance.sqrt();

    // Run preflight checks (unless skipped)
    let preflight_result = if config.skip_preflight {
        PreflightResult::new()
    } else {
        run_core_checks(
            &baseline_ns,
            &sample_ns,
            config.timer_resolution_ns,
            config.seed,
        )
    };

    // Compute calibration statistics snapshot for drift detection
    let calibration_snapshot = compute_calibration_snapshot(&baseline_ns, &sample_ns);

    // Compute floor-rate constant from null distribution (v7.1 Fix 2)
    // c_floor = q95 of null W₁ distribution, scaled by sqrt(n_blocks)
    let null_cal = calibrate_floor_from_null(
        &interleaved,
        block_length,
        config.bootstrap_iterations,
        config.seed,
    );
    let c_floor = null_cal.c_floor;

    // Timer tick floor
    let theta_tick = config.timer_resolution_ns;

    // Initial measurement floor at calibration sample count
    let n_blocks_cal = if block_length > 0 {
        (n / block_length).max(1)
    } else {
        n.max(1)
    };
    let theta_floor_initial = (c_floor / (n_blocks_cal as f64).sqrt()).max(theta_tick);

    // Effective threshold: max(user threshold, measurement floor)
    let theta_eff = if config.theta_ns > 0.0 {
        config.theta_ns.max(theta_floor_initial)
    } else {
        theta_floor_initial
    };

    // Half-t prior (ν=4) calibration for W₁ distance (v7.1 Fix 4)
    // Prior targets theta_user (threat model), not theta_eff (noise-elevated)
    let sigma_t = calibrate_halft_prior_scale_1d(var_rate, config.theta_ns, n, config.seed);

    // Projection mismatch is not applicable to W₁ distance (no projection model)
    // Use placeholder value for compatibility with diagnostics
    let projection_mismatch_thresh = 0.0;

    Ok(Calibration::with_preflight(
        var_rate,
        block_length,
        iact,
        iact_method,
        sigma_t,
        config.theta_ns,
        n,
        discrete_mode,
        mde_ns,
        calibration_snapshot,
        config.timer_resolution_ns,
        samples_per_second,
        c_floor,
        projection_mismatch_thresh,
        theta_tick,
        theta_eff,
        theta_floor_initial,
        config.seed,
        1, // batch_k = 1 (no batching), updated by collector
        preflight_result,
    ))
}

/// Count unique values in a slice (for discrete mode detection).
fn count_unique(values: &[f64]) -> usize {
    use alloc::collections::BTreeSet;

    // Discretize to avoid floating point comparison issues
    // Use 0.001ns buckets (well below any meaningful timing difference)
    let buckets: BTreeSet<i64> = values.iter().map(|&v| (v * 1000.0) as i64).collect();
    buckets.len()
}

/// Compute calibration statistics snapshot for drift detection.
fn compute_calibration_snapshot(baseline_ns: &[f64], sample_ns: &[f64]) -> CalibrationSnapshot {
    let mut baseline_stats = OnlineStats::new();
    let mut sample_stats = OnlineStats::new();

    for &t in baseline_ns {
        baseline_stats.update(t);
    }
    for &t in sample_ns {
        sample_stats.update(t);
    }

    CalibrationSnapshot::new(baseline_stats.finalize(), sample_stats.finalize())
}

// DELETED: compute_prior_cov_9d() - 9D version (no 1D equivalent needed, prior is scalar)

// DELETED: compute_correlation_matrix() - 9D helper
// DELETED: estimate_condition_number() - 9D helper
// DELETED: is_fragile_regime() - 9D helper
// DELETED: apply_correlation_regularization() - 9D helper
// DELETED: compute_median_se() - 9D helper

// DELETED: calibrate_t_prior_scale() - 9D version (replaced by calibrate_halft_prior_scale_1d)

// DELETED: precompute_t_prior_effects() - 9D helper (replaced by 1D calibration)


/// Calibrate half-t prior scale to achieve target exceedance probability (W₁ distance).
///
/// Finds σ_t such that P(|δ| > theta_user | δ ~ half-t(ν=4, σ_t)) ≈ 0.62
///
/// **IMPORTANT (v7.1 Fix 4)**: Prior targets theta_user (threat-model threshold),
/// NOT theta_eff (noise-elevated threshold). The prior encodes the threat model,
/// while theta_eff = theta_user + theta_floor is for certifiability.
///
/// Uses binary search with Monte Carlo exceedance estimation. The half-t distribution
/// is sampled via scale mixture:
/// - λ ~ Gamma(ν/2, ν/2) = Gamma(2, 2) for ν=4
/// - z ~ N(0, 1)
/// - δ = |σ_t/√λ · z|
///
/// # Arguments
/// * `var_rate` - Variance rate (scalar) from bootstrap (unused, kept for API compatibility)
/// * `theta_user` - User-specified threat-model threshold in nanoseconds
/// * `n_cal` - Number of calibration samples (unused, kept for API compatibility)
/// * `seed` - Deterministic RNG seed
///
/// # Returns
/// Calibrated prior scale σ_t achieving target 62% exceedance probability.
///
/// # Example
/// ```
/// use tacet_core::adaptive::calibrate_halft_prior_scale_1d;
///
/// let var_rate = 1000.0;
/// let theta_user = 100.0;
/// let n_cal = 5000;
/// let seed = 42;
///
/// let sigma_t = calibrate_halft_prior_scale_1d(var_rate, theta_user, n_cal, seed);
/// assert!(sigma_t > 0.0);
/// ```
pub fn calibrate_halft_prior_scale_1d(
    var_rate: f64,
    theta_user: f64,
    n_cal: usize,
    seed: u64,
) -> f64 {
    calibrate_halft_prior_scale_1d_with_target(var_rate, theta_user, n_cal, seed, 0.62, NU)
}

/// Variant of `calibrate_halft_prior_scale_1d` that accepts `target_exceedance`
/// and `nu_prior` as parameters, for hyperparameter-sensitivity ablation studies.
/// The default values (0.62, 4.0) match the spec; deviations should only be used
/// for research.
pub fn calibrate_halft_prior_scale_1d_with_target(
    _var_rate: f64,
    theta_user: f64,
    _n_cal: usize,
    seed: u64,
    target_exceedance: f64,
    nu_prior: f64,
) -> f64 {
    const TOLERANCE: f64 = 0.02;
    const N_SAMPLES: usize = 10_000;
    let nu = nu_prior;

    // Binary search bounds
    let mut sigma_low = theta_user / 10.0;
    let mut sigma_high = theta_user * 10.0;

    // Use RNG for Monte Carlo
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // Binary search loop (max 30 iterations)
    for _ in 0..30 {
        let sigma_mid = (sigma_low + sigma_high) / 2.0;

        // Estimate P(|δ| > theta_user) via Monte Carlo
        // Sample δ ~ half-t(ν, σ_mid) via scale mixture:
        //   λ ~ Gamma(ν/2, ν/2), z ~ N(0,1), δ = |σ/√λ · z|
        let mut exceed_count = 0;
        for _ in 0..N_SAMPLES {
            let lambda = sample_gamma(&mut rng, nu / 2.0, nu / 2.0);
            let z = sample_standard_normal(&mut rng);
            let delta = (sigma_mid / lambda.sqrt() * z).abs();

            if delta > theta_user {
                exceed_count += 1;
            }
        }

        let exceedance_prob = exceed_count as f64 / N_SAMPLES as f64;

        // Check convergence
        if (exceedance_prob - target_exceedance).abs() < TOLERANCE {
            return sigma_mid;
        }

        // Adjust bounds
        if exceedance_prob < target_exceedance {
            sigma_low = sigma_mid;
        } else {
            sigma_high = sigma_mid;
        }
    }

    // Return midpoint if didn't converge
    (sigma_low + sigma_high) / 2.0
}

/// Result from null floor calibration.
///
/// Contains the floor statistic (95th percentile) plus the mean and variance
/// of the null W₁ replicates in rate form. The variance can be used as a floor
/// on the bootstrap variance to prevent posterior overconfidence under dependence.
#[derive(Debug, Clone, Copy)]
pub struct NullCalibration {
    /// 95th percentile of scaled null W₁ replicates (c_floor in rate form: ns·√n_blocks).
    pub c_floor: f64,
    /// Mean of scaled null W₁ replicates (null bias in rate form).
    pub null_mean_rate: f64,
    /// Variance of scaled null W₁ replicates (null variance in rate form).
    pub null_var_rate: f64,
}

/// Calibrate floor constant c_floor from null distribution of raw W₁.
///
/// Uses within-class splits to generate null W₁ replicates under "no leak"
/// assumption, then takes 95th percentile scaled by sqrt(n_blocks).
///
/// At runtime: theta_floor(n) = max(theta_tick, c_floor / sqrt(n_blocks(n)))
///
/// Also returns the mean and variance of the null replicates (in rate form)
/// for use as a variance floor in Bayesian inference.
///
/// # Arguments
/// * `interleaved` - Calibration samples (both classes interleaved)
/// * `block_length` - Block size for computing n_blocks
/// * `n_bootstrap` - Number of bootstrap replicates (typically 200-500)
/// * `seed` - Deterministic RNG seed
///
/// # Returns
/// `NullCalibration` with 95th percentile, mean, and variance of scaled null W₁.
pub fn calibrate_floor_from_null(
    interleaved: &[TimingSample],
    block_length: usize,
    n_bootstrap: usize,
    seed: u64,
) -> NullCalibration {
    use rand::seq::SliceRandom;

    let n = interleaved.len();
    let n_per_class = n / 2;

    // Compute n_blocks at calibration size for scaling
    let n_blocks_cal = if block_length > 0 {
        (n_per_class / block_length).max(1)
    } else {
        n_per_class.max(1)
    };
    let sqrt_n_blocks_cal = (n_blocks_cal as f64).sqrt();

    // Split interleaved samples by class
    let mut baseline_samples: Vec<f64> = Vec::with_capacity(n_per_class);
    let mut sample_samples: Vec<f64> = Vec::with_capacity(n_per_class);

    for ts in interleaved {
        match ts.class {
            Class::Baseline => baseline_samples.push(ts.time_ns),
            Class::Sample => sample_samples.push(ts.time_ns),
        }
    }

    let n_baseline = baseline_samples.len();
    let n_sample = sample_samples.len();

    // Generate null W₁ replicates via within-class random splits.
    let mut null_replicates: Vec<f64> = Vec::with_capacity(n_bootstrap);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    for _ in 0..n_bootstrap {
        // Randomly shuffle baseline indices for split
        let mut baseline_indices: Vec<usize> = (0..n_baseline).collect();
        baseline_indices.shuffle(&mut rng);
        let split_baseline = n_baseline / 2;

        let mut b1: Vec<f64> = baseline_indices[..split_baseline]
            .iter()
            .map(|&i| baseline_samples[i])
            .collect();
        let mut b2: Vec<f64> = baseline_indices[split_baseline..]
            .iter()
            .map(|&i| baseline_samples[i])
            .collect();

        // Randomly shuffle sample indices for split
        let mut sample_indices: Vec<usize> = (0..n_sample).collect();
        sample_indices.shuffle(&mut rng);
        let split_sample = n_sample / 2;

        let mut s1: Vec<f64> = sample_indices[..split_sample]
            .iter()
            .map(|&i| sample_samples[i])
            .collect();
        let mut s2: Vec<f64> = sample_indices[split_sample..]
            .iter()
            .map(|&i| sample_samples[i])
            .collect();

        // Compute W₁ for both within-class splits
        let w1_baseline = compute_w1_distance(&mut b1, &mut b2);
        let w1_sample = compute_w1_distance(&mut s1, &mut s2);

        // Average the two null replicates and scale by sqrt(n_blocks_cal).
        // Correction: null W₁ replicates use half-samples (m = n_per_class/2),
        // but the floor guards a decision made with the full n_per_class samples.
        // Since W₁ variance ∝ 1/√m, the half-sample W₁ overestimates by √2.
        let avg_w1 = (w1_baseline + w1_sample) / 2.0;
        let scaled_w1 = avg_w1 * sqrt_n_blocks_cal / std::f64::consts::SQRT_2;

        null_replicates.push(scaled_w1);
    }

    // Compute null replicate statistics before partial sorting
    let null_mean_rate = null_replicates.iter().sum::<f64>() / n_bootstrap as f64;
    let null_var_rate = if n_bootstrap > 1 {
        null_replicates
            .iter()
            .map(|x| (x - null_mean_rate).powi(2))
            .sum::<f64>()
            / (n_bootstrap - 1) as f64
    } else {
        0.0
    };

    // Return 95th percentile
    let percentile_idx = ((n_bootstrap as f64 * 0.95) as usize).min(n_bootstrap - 1);
    null_replicates.select_nth_unstable_by(percentile_idx, |a, b| a.total_cmp(b));

    NullCalibration {
        c_floor: null_replicates[percentile_idx],
        null_mean_rate,
        null_var_rate,
    }
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

        let var_rate = 1000.0;
        let theta_eff = 100.0;

        Calibration::new(
            var_rate,                               // var_rate
            10,                                     // block_length
            1.0,                                    // iact
            crate::types::IactMethod::PolitisWhite, // iact_method
            100.0,                                  // sigma_t
            theta_eff,                              // theta_ns
            5000,                                   // calibration_samples
            false,                                  // discrete_mode
            5.0,                                    // mde_ns
            snapshot,                               // calibration_snapshot
            1.0,                                    // timer_resolution_ns
            10000.0,                                // samples_per_second
            10.0,                                   // c_floor
            18.48,                                  // projection_mismatch_thresh
            0.001,                                  // theta_tick
            theta_eff,                              // theta_eff
            0.1,                                    // theta_floor_initial
            42,                                     // rng_seed
            1,                                      // batch_k
        )
    }

    #[test]
    fn test_variance_scaling() {
        let cal = make_test_calibration();
        // make_test_calibration uses var_rate = 1000.0

        // v6.0: variance_for_n uses raw n (block bootstrap already accounts for dependence)
        // At n=1000: var_n = var_rate / n = 1000 / 1000 = 1.0
        let var_1000 = cal.variance_for_n(1000);
        assert!(
            (var_1000 - 1.0).abs() < 1e-10,
            "expected 1.0, got {}",
            var_1000
        );

        // At n=2000: var_n = 1000 / 2000 = 0.5
        let var_2000 = cal.variance_for_n(2000);
        assert!(
            (var_2000 - 0.5).abs() < 1e-10,
            "expected 0.5, got {}",
            var_2000
        );
    }

    #[test]
    fn test_n_blocks() {
        let cal = make_test_calibration();
        // make_test_calibration uses block_length = 10

        // n_blocks = max(1, floor(n / block_length))
        assert_eq!(cal.n_blocks(100), 10);
        assert_eq!(cal.n_blocks(1000), 100);
        assert_eq!(cal.n_blocks(10), 1);
        assert_eq!(cal.n_blocks(5), 1); // Clamped to 1 when n < block_length
        assert_eq!(cal.n_blocks(0), 1); // Edge case: n=0 returns 1
    }

    #[test]
    fn test_variance_zero_n() {
        let cal = make_test_calibration();
        let var = cal.variance_for_n(0);
        // Should return var_rate unchanged (avoid NaN)
        assert!((var - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_collection_time() {
        let cal = make_test_calibration();

        // 10000 samples/sec -> 1000 samples takes 0.1 sec
        let time = cal.estimate_collection_time_secs(1000);
        assert!((time - 0.1).abs() < 1e-10);
    }

    // DELETED: test_compute_prior_cov_9d_unit_diagonal() - 9D test
    // DELETED: test_c_floor_computation() - 9D test

    #[test]
    fn test_calibration_config_default() {
        let config = CalibrationConfig::default();
        assert_eq!(config.calibration_samples, 5000);
        assert_eq!(config.bootstrap_iterations, 200);
        assert!((config.theta_ns - 100.0).abs() < 1e-10);
        assert!((config.timer_resolution_ns - 1.0).abs() < 1e-10);
        assert!(!config.skip_preflight);
        assert!(!config.force_discrete_mode);
    }

    // DELETED: 9D t-prior test helper functions and tests
    // - reference_t_prior_exceedance()
    // - optimized_t_prior_exceedance()
    // - test_t_prior_precompute_exceedance_matches_reference()
    // - test_t_prior_exceedance_monotonicity()
    // - test_calibrate_t_prior_scale_finds_target_exceedance()
    // - test_calibration_determinism()
    // - test_precomputed_effects_distribution()
    // - bench_calibration_timing()

    // =========================================================================
    // W₁ distance (1D) helper function tests
    // =========================================================================

    #[test]
    fn test_calibrate_halft_prior_scale_1d_basic() {
        // Test basic calibration produces reasonable scale
        let var_rate = 100.0;
        let theta_eff = 10.0;
        let n_cal = 5000;
        let seed = 42;

        let sigma_t = calibrate_halft_prior_scale_1d(var_rate, theta_eff, n_cal, seed);

        // Should be positive and in reasonable range relative to theta
        assert!(sigma_t > 0.0, "sigma_t should be positive");
        assert!(
            sigma_t > theta_eff * 0.05,
            "sigma_t {} should be > theta * 0.05 = {}",
            sigma_t,
            theta_eff * 0.05
        );
        assert!(
            sigma_t < theta_eff * 50.0,
            "sigma_t {} should be < theta * 50 = {}",
            sigma_t,
            theta_eff * 50.0
        );
    }

    #[test]
    fn test_calibrate_halft_prior_scale_1d_achieves_target() {
        // Test that calibrated scale achieves target exceedance probability
        let var_rate = 100.0;
        let theta_eff = 10.0;
        let n_cal = 5000;
        let seed = 42;

        let sigma_t = calibrate_halft_prior_scale_1d(var_rate, theta_eff, n_cal, seed);

        // Verify exceedance probability via MC
        const N_VERIFY: usize = 50_000;
        const NU: f64 = 4.0;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + 1);
        let mut exceed_count = 0;

        for _ in 0..N_VERIFY {
            let lambda = sample_gamma(&mut rng, NU / 2.0, NU / 2.0);
            let z = sample_standard_normal(&mut rng);
            let delta = (sigma_t / lambda.sqrt() * z).abs();

            if delta > theta_eff {
                exceed_count += 1;
            }
        }

        let exceedance = exceed_count as f64 / N_VERIFY as f64;

        // Should be within 0.05 of target 0.62
        assert!(
            (exceedance - 0.62).abs() < 0.05,
            "Exceedance {} should be near 0.62",
            exceedance
        );
    }

    #[test]
    fn test_calibrate_halft_prior_scale_1d_determinism() {
        // Same seed should give same result
        let var_rate = 100.0;
        let theta_eff = 10.0;
        let n_cal = 5000;
        let seed = 12345;

        let sigma_1 = calibrate_halft_prior_scale_1d(var_rate, theta_eff, n_cal, seed);
        let sigma_2 = calibrate_halft_prior_scale_1d(var_rate, theta_eff, n_cal, seed);

        assert!(
            (sigma_1 - sigma_2).abs() < 1e-10,
            "Same seed should give same sigma_t: {} vs {}",
            sigma_1,
            sigma_2
        );
    }

    #[test]
    fn test_calibrate_halft_prior_scale_1d_scale_invariance() {
        // Scaling var_rate and theta_eff by same factor should scale sigma_t proportionally
        let var_rate = 100.0;
        let theta_eff = 10.0;
        let n_cal = 5000;
        let seed = 42;

        let sigma_1 = calibrate_halft_prior_scale_1d(var_rate, theta_eff, n_cal, seed);
        let sigma_2 = calibrate_halft_prior_scale_1d(var_rate * 4.0, theta_eff * 2.0, n_cal, seed);

        // sigma should scale roughly with sqrt(var_rate) ~ theta
        // But due to discrete binary search, won't be exact
        let expected_ratio = 2.0;
        let actual_ratio = sigma_2 / sigma_1;

        assert!(
            (actual_ratio - expected_ratio).abs() < 0.5,
            "Ratio {} should be near {}",
            actual_ratio,
            expected_ratio
        );
    }

    #[test]
    fn test_halft_prior_scale_1d_monotonicity() {
        // Increasing theta_eff should increase sigma_t (for fixed exceedance target)
        let var_rate = 100.0;
        let n_cal = 5000;
        let seed = 42;

        let sigma_10 = calibrate_halft_prior_scale_1d(var_rate, 10.0, n_cal, seed);
        let sigma_20 = calibrate_halft_prior_scale_1d(var_rate, 20.0, n_cal, seed);
        let sigma_50 = calibrate_halft_prior_scale_1d(var_rate, 50.0, n_cal, seed);

        assert!(
            sigma_20 > sigma_10,
            "sigma_t should increase with theta: {} vs {}",
            sigma_20,
            sigma_10
        );
        assert!(
            sigma_50 > sigma_20,
            "sigma_t should increase with theta: {} vs {}",
            sigma_50,
            sigma_20
        );
    }

    // DELETED: test_halft_vs_9d_scale_relationship() - test comparing 1D vs 9D

    // =========================================================================
    // calibrate_floor_from_null() tests
    // =========================================================================

    #[test]
    fn test_calibrate_floor_from_null_basic() {
        // Test basic calibration produces positive value
        let n = 500;
        let mut samples = Vec::with_capacity(2 * n);

        for i in 0..n {
            samples.push(TimingSample {
                time_ns: 100.0 + (i as f64) * 0.1,
                class: Class::Baseline,
            });
            samples.push(TimingSample {
                time_ns: 100.0 + (i as f64) * 0.1, // Identical (null)
                class: Class::Sample,
            });
        }

        let block_length = 10;
        let n_bootstrap = 200;
        let seed = 42;

        let null_cal = calibrate_floor_from_null(&samples, block_length, n_bootstrap, seed);
        let c_floor = null_cal.c_floor;

        // Should be positive and finite
        assert!(c_floor > 0.0, "c_floor should be positive");
        assert!(c_floor.is_finite(), "c_floor should be finite");

        // Null variance should also be non-negative and finite
        assert!(
            null_cal.null_var_rate >= 0.0 && null_cal.null_var_rate.is_finite(),
            "null_var_rate should be non-negative and finite"
        );
    }

    #[test]
    fn test_calibrate_floor_from_null_determinism() {
        // Same seed should give same result
        let n = 300;
        let mut samples = Vec::with_capacity(2 * n);

        for i in 0..n {
            samples.push(TimingSample {
                time_ns: 100.0 + (i as f64),
                class: Class::Baseline,
            });
            samples.push(TimingSample {
                time_ns: 105.0 + (i as f64),
                class: Class::Sample,
            });
        }

        let block_length = 8;
        let n_bootstrap = 150;
        let seed = 12345;

        let null_cal_1 = calibrate_floor_from_null(&samples, block_length, n_bootstrap, seed);
        let null_cal_2 = calibrate_floor_from_null(&samples, block_length, n_bootstrap, seed);

        assert!(
            (null_cal_1.c_floor - null_cal_2.c_floor).abs() < 1e-10,
            "Same seed should give same c_floor: {} vs {}",
            null_cal_1.c_floor,
            null_cal_2.c_floor
        );
    }

    #[test]
    fn test_calibrate_floor_from_null_null_hypothesis() {
        // Under null hypothesis (identical distributions), c_floor should reflect noise
        let n = 400;
        let mut samples = Vec::with_capacity(2 * n);

        for i in 0..n {
            let val = 100.0 + (i as f64) * 0.05;
            samples.push(TimingSample {
                time_ns: val,
                class: Class::Baseline,
            });
            samples.push(TimingSample {
                time_ns: val, // Identical
                class: Class::Sample,
            });
        }

        let block_length = 12;
        let n_bootstrap = 200;
        let seed = 999;

        let null_cal = calibrate_floor_from_null(&samples, block_length, n_bootstrap, seed);
        let c_floor = null_cal.c_floor;

        // Under null, should be relatively small (reflects sampling noise only)
        assert!(
            c_floor < 10.0,
            "c_floor under null should be small, got {}",
            c_floor
        );
    }

    #[test]
    fn test_calibrate_floor_from_null_scaling() {
        // Test that block length affects scaling correctly
        let n = 500;
        let mut samples = Vec::with_capacity(2 * n);

        for i in 0..n {
            samples.push(TimingSample {
                time_ns: 100.0 + (i as f64) * 0.1,
                class: Class::Baseline,
            });
            samples.push(TimingSample {
                time_ns: 102.0 + (i as f64) * 0.1,
                class: Class::Sample,
            });
        }

        let n_bootstrap = 200;
        let seed = 42;

        let null_cal_small = calibrate_floor_from_null(&samples, 5, n_bootstrap, seed);
        let null_cal_large = calibrate_floor_from_null(&samples, 25, n_bootstrap, seed);

        // Both should be positive
        assert!(null_cal_small.c_floor > 0.0);
        assert!(null_cal_large.c_floor > 0.0);

        // Larger block length means fewer blocks (n_blocks = n / block_length)
        // c_floor is scaled by sqrt(n_blocks), so fewer blocks -> smaller sqrt(n_blocks) -> smaller c_floor
        // For block_length=5: n_blocks = 250/5 = 50, sqrt = 7.07
        // For block_length=25: n_blocks = 250/25 = 10, sqrt = 3.16
        // So small_block should have larger c_floor
        assert!(
            null_cal_small.c_floor > null_cal_large.c_floor,
            "Smaller block length (more blocks) should yield larger c_floor due to sqrt(n_blocks) scaling: small={}, large={}",
            null_cal_small.c_floor,
            null_cal_large.c_floor
        );
    }

    #[test]
    fn test_calibrate_floor_from_null_percentile_property() {
        // Verify that c_floor represents a high percentile of the null distribution
        let n = 400;
        let mut samples = Vec::with_capacity(2 * n);

        for i in 0..n {
            samples.push(TimingSample {
                time_ns: 100.0 + (i as f64) * 0.1,
                class: Class::Baseline,
            });
            samples.push(TimingSample {
                time_ns: 100.0 + (i as f64) * 0.1, // Null
                class: Class::Sample,
            });
        }

        let block_length = 10;
        let n_bootstrap = 1000; // More samples for better percentile estimate
        let seed = 777;

        let null_cal = calibrate_floor_from_null(&samples, block_length, n_bootstrap, seed);
        let c_floor = null_cal.c_floor;

        // Generate fresh null replicates and check that c_floor is indeed high percentile
        use rand::seq::SliceRandom;
        let n_per_class = n;
        let n_blocks_cal = (n_per_class / block_length).max(1);
        let sqrt_n_blocks_cal = (n_blocks_cal as f64).sqrt();

        let mut baseline_samples: Vec<f64> = Vec::new();
        let mut sample_samples: Vec<f64> = Vec::new();
        for ts in &samples {
            match ts.class {
                Class::Baseline => baseline_samples.push(ts.time_ns),
                Class::Sample => sample_samples.push(ts.time_ns),
            }
        }

        let n_baseline = baseline_samples.len();
        let n_sample = sample_samples.len();

        let mut null_replicates: Vec<f64> = Vec::new();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + 1);

        for _ in 0..500 {
            let mut baseline_indices: Vec<usize> = (0..n_baseline).collect();
            baseline_indices.shuffle(&mut rng);
            let split = n_baseline / 2;
            let mut b1: Vec<f64> = baseline_indices[..split]
                .iter()
                .map(|&i| baseline_samples[i])
                .collect();
            let mut b2: Vec<f64> = baseline_indices[split..]
                .iter()
                .map(|&i| baseline_samples[i])
                .collect();

            let mut sample_indices: Vec<usize> = (0..n_sample).collect();
            sample_indices.shuffle(&mut rng);
            let split = n_sample / 2;
            let mut s1: Vec<f64> = sample_indices[..split]
                .iter()
                .map(|&i| sample_samples[i])
                .collect();
            let mut s2: Vec<f64> = sample_indices[split..]
                .iter()
                .map(|&i| sample_samples[i])
                .collect();

            let w1_b = compute_w1_distance(&mut b1, &mut b2);
            let w1_s = compute_w1_distance(&mut s1, &mut s2);
            let avg_w1 = (w1_b + w1_s) / 2.0;
            null_replicates.push(avg_w1 * sqrt_n_blocks_cal);
        }

        null_replicates.sort_by(|a, b| a.total_cmp(b));
        let median = null_replicates[null_replicates.len() / 2];

        // c_floor should be well above median (since it's 95th percentile)
        assert!(
            c_floor > median,
            "c_floor {} should be above median {} of null distribution",
            c_floor,
            median
        );
    }
}
