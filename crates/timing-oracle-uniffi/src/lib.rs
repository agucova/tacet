//! UniFFI bindings for timing-oracle statistical analysis.
//!
//! This crate provides cross-language bindings for timing-oracle using Mozilla's UniFFI.
//! It exposes the statistical analysis functionality to Go, C++, and JavaScript (via WASM).
//!
//! # Design
//!
//! The measurement loop should be implemented natively in each target language to avoid
//! FFI overhead during timing-critical code. Only the statistical analysis crosses the
//! FFI boundary, which is not timing-sensitive.
//!
//! # Supported Languages
//!
//! - Go (via uniffi-bindgen-go)
//! - C++ (via uniffi-bindgen-cpp)
//! - JavaScript/TypeScript (via uniffi-bindgen-js with WASM)

use std::sync::{Arc, RwLock};

use timing_oracle::adaptive::{calibrate, CalibrationConfig};
use timing_oracle_core::adaptive::{
    adaptive_step, AdaptiveOutcome, AdaptiveState as CoreAdaptiveState,
    Calibration as CoreCalibration, InconclusiveReason as CoreInconclusiveReason,
    AdaptiveStepConfig, StepResult,
};
use timing_oracle_core::math::sqrt;
use timing_oracle_core::result::{
    EffectPattern as CoreEffectPattern, Exploitability as CoreExploitability,
    MeasurementQuality as CoreMeasurementQuality,
};

uniffi::setup_scaffolding!();

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during timing analysis.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum TimingOracleError {
    #[error("Null pointer provided")]
    NullPointer,

    #[error("Insufficient samples: need at least {minimum}, got {actual}")]
    InsufficientSamples { minimum: u64, actual: u64 },

    #[error("Calibration failed: {message}")]
    CalibrationFailed { message: String },

    #[error("Analysis failed: {message}")]
    AnalysisFailed { message: String },

    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },
}

// =============================================================================
// Enums - Simple enums that map directly to core types
// =============================================================================

/// Attacker model determines the minimum effect threshold (theta) for leak detection.
///
/// Choose based on your threat model - this is the most important configuration choice.
#[derive(Debug, Clone, Copy, PartialEq, Default, uniffi::Enum)]
pub enum AttackerModel {
    /// theta = 0.6 ns (~2 cycles @ 3GHz) - SGX, cross-VM, containers
    SharedHardware,
    /// theta = 3.3 ns (~10 cycles) - Post-quantum crypto (KyberSlash-class)
    PostQuantum,
    /// theta = 100 ns - LAN, HTTP/2 (Timeless Timing Attacks)
    #[default]
    AdjacentNetwork,
    /// theta = 50 us - General internet
    RemoteNetwork,
    /// theta -> 0 - Detect any difference (not for CI)
    Research,
    /// User-specified threshold
    Custom { threshold_ns: f64 },
}

impl AttackerModel {
    /// Convert to threshold in nanoseconds.
    pub fn to_threshold_ns(&self) -> f64 {
        match self {
            AttackerModel::SharedHardware => 0.6,
            AttackerModel::PostQuantum => 3.3,
            AttackerModel::AdjacentNetwork => 100.0,
            AttackerModel::RemoteNetwork => 50_000.0,
            AttackerModel::Research => 0.0,
            AttackerModel::Custom { threshold_ns } => *threshold_ns,
        }
    }
}

/// Test outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum Outcome {
    /// No timing leak detected within threshold theta.
    Pass,
    /// Timing leak detected exceeding threshold theta.
    Fail,
    /// Could not reach a decision (quality gate triggered).
    Inconclusive,
    /// Operation too fast to measure reliably.
    Unmeasurable,
}

/// Reason for inconclusive result.
#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum InconclusiveReason {
    /// No inconclusive reason (not applicable).
    None,
    /// Posterior approx equals prior after calibration; data not informative.
    DataTooNoisy { message: String, guidance: String },
    /// Posterior stopped updating despite new data.
    NotLearning { message: String, guidance: String },
    /// Estimated time to decision exceeds budget.
    WouldTakeTooLong {
        estimated_time_secs: f64,
        samples_needed: u64,
        guidance: String,
    },
    /// Time budget exhausted.
    TimeBudgetExceeded {
        current_probability: f64,
        samples_collected: u64,
    },
    /// Sample limit reached.
    SampleBudgetExceeded {
        current_probability: f64,
        samples_collected: u64,
    },
    /// Measurement conditions changed during test.
    ConditionsChanged { message: String, guidance: String },
    /// Threshold was elevated and pass criterion was met at effective threshold.
    ThresholdElevated {
        theta_user: f64,
        theta_eff: f64,
        leak_probability_at_eff: f64,
        achievable_at_max: bool,
        message: String,
        guidance: String,
    },
}

/// Pattern of timing effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum EffectPattern {
    /// Uniform shift across all quantiles.
    UniformShift,
    /// Effect concentrated in tails (upper quantiles).
    TailEffect,
    /// Both shift and tail components present.
    Mixed,
    /// Cannot determine pattern.
    Indeterminate,
}

impl From<CoreEffectPattern> for EffectPattern {
    fn from(p: CoreEffectPattern) -> Self {
        match p {
            CoreEffectPattern::UniformShift => EffectPattern::UniformShift,
            CoreEffectPattern::TailEffect => EffectPattern::TailEffect,
            CoreEffectPattern::Mixed => EffectPattern::Mixed,
            CoreEffectPattern::Indeterminate => EffectPattern::Indeterminate,
        }
    }
}

/// Exploitability assessment (for Fail outcomes).
///
/// Based on Timeless Timing Attacks (Van Goethem et al., 2020) research.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum Exploitability {
    /// < 10 ns - Requires shared hardware (SGX, containers) to exploit.
    SharedHardwareOnly,
    /// 10-100 ns - Exploitable via HTTP/2 request multiplexing.
    Http2Multiplexing,
    /// 100 ns - 10 us - Exploitable with standard remote timing.
    StandardRemote,
    /// > 10 us - Obvious leak, trivially exploitable.
    ObviousLeak,
}

impl From<CoreExploitability> for Exploitability {
    fn from(e: CoreExploitability) -> Self {
        match e {
            CoreExploitability::SharedHardwareOnly => Exploitability::SharedHardwareOnly,
            CoreExploitability::Http2Multiplexing => Exploitability::Http2Multiplexing,
            CoreExploitability::StandardRemote => Exploitability::StandardRemote,
            CoreExploitability::ObviousLeak => Exploitability::ObviousLeak,
        }
    }
}

impl Exploitability {
    /// Compute exploitability from effect magnitude in nanoseconds.
    pub fn from_effect_ns(effect_ns: f64) -> Self {
        CoreExploitability::from_effect_ns(effect_ns).into()
    }
}

/// Measurement quality assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum MeasurementQuality {
    /// MDE < 5 ns - Excellent measurement precision.
    Excellent,
    /// MDE 5-20 ns - Good precision.
    Good,
    /// MDE 20-100 ns - Poor precision.
    Poor,
    /// MDE > 100 ns - Too noisy for reliable detection.
    TooNoisy,
}

impl From<CoreMeasurementQuality> for MeasurementQuality {
    fn from(q: CoreMeasurementQuality) -> Self {
        match q {
            CoreMeasurementQuality::Excellent => MeasurementQuality::Excellent,
            CoreMeasurementQuality::Good => MeasurementQuality::Good,
            CoreMeasurementQuality::Poor => MeasurementQuality::Poor,
            CoreMeasurementQuality::TooNoisy => MeasurementQuality::TooNoisy,
        }
    }
}

// =============================================================================
// Structs - Records that hold data
// =============================================================================

/// Credible interval bounds.
///
/// UniFFI doesn't support tuples, so we use a struct instead of (f64, f64).
#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct CredibleInterval {
    /// Lower bound of the 95% credible interval.
    pub low: f64,
    /// Upper bound of the 95% credible interval.
    pub high: f64,
}

/// Effect size estimate.
#[derive(Debug, Clone, uniffi::Record)]
pub struct EffectEstimate {
    /// Uniform shift component in nanoseconds.
    pub shift_ns: f64,
    /// Tail effect component in nanoseconds.
    pub tail_ns: f64,
    /// 95% credible interval for total effect.
    pub credible_interval: CredibleInterval,
    /// Pattern of the effect.
    pub pattern: EffectPattern,
    /// Interpretation caveat if model fit is poor.
    pub interpretation_caveat: Option<String>,
}

impl EffectEstimate {
    /// Compute the total effect magnitude (L2 norm of shift and tail).
    pub fn total_effect_ns(&self) -> f64 {
        sqrt(self.shift_ns * self.shift_ns + self.tail_ns * self.tail_ns)
    }
}

/// Diagnostics information for debugging and quality assessment.
#[derive(Debug, Clone, uniffi::Record)]
pub struct Diagnostics {
    // Core diagnostics
    /// Block length used for bootstrap resampling.
    pub dependence_length: u64,
    /// Effective sample size accounting for autocorrelation.
    pub effective_sample_size: u64,
    /// Ratio of post-test variance to calibration variance.
    pub stationarity_ratio: f64,
    /// Whether stationarity check passed.
    pub stationarity_ok: bool,
    /// Projection mismatch Q statistic.
    pub projection_mismatch_q: f64,
    /// Whether projection mismatch is acceptable.
    pub projection_mismatch_ok: bool,

    // Timer diagnostics
    /// Whether discrete mode was used (low timer resolution).
    pub discrete_mode: bool,
    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,

    // Gibbs sampler lambda diagnostics (v5.4)
    /// Total number of Gibbs iterations.
    pub gibbs_iters_total: u64,
    /// Number of burn-in iterations.
    pub gibbs_burnin: u64,
    /// Number of retained samples.
    pub gibbs_retained: u64,
    /// Posterior mean of latent scale lambda.
    pub lambda_mean: f64,
    /// Posterior standard deviation of lambda.
    pub lambda_sd: f64,
    /// Coefficient of variation of lambda.
    pub lambda_cv: f64,
    /// Effective sample size of lambda chain.
    pub lambda_ess: f64,
    /// Whether lambda chain mixed well.
    pub lambda_mixing_ok: bool,

    // Gibbs sampler kappa diagnostics (v5.6)
    /// Posterior mean of likelihood precision kappa.
    pub kappa_mean: f64,
    /// Posterior standard deviation of kappa.
    pub kappa_sd: f64,
    /// Coefficient of variation of kappa.
    pub kappa_cv: f64,
    /// Effective sample size of kappa chain.
    pub kappa_ess: f64,
    /// Whether kappa chain mixed well.
    pub kappa_mixing_ok: bool,
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self {
            dependence_length: 0,
            effective_sample_size: 0,
            stationarity_ratio: 1.0,
            stationarity_ok: true,
            projection_mismatch_q: 0.0,
            projection_mismatch_ok: true,
            discrete_mode: false,
            timer_resolution_ns: 0.0,
            gibbs_iters_total: 256,
            gibbs_burnin: 64,
            gibbs_retained: 192,
            lambda_mean: 1.0,
            lambda_sd: 0.0,
            lambda_cv: 0.0,
            lambda_ess: 0.0,
            lambda_mixing_ok: true,
            kappa_mean: 1.0,
            kappa_sd: 0.0,
            kappa_cv: 0.0,
            kappa_ess: 0.0,
            kappa_mixing_ok: true,
        }
    }
}

/// Configuration for the timing analysis.
#[derive(Debug, Clone, uniffi::Record)]
pub struct Config {
    /// Attacker model to use (determines threshold theta).
    pub attacker_model: AttackerModel,
    /// Maximum samples per class. 0 = default (100,000).
    pub max_samples: u64,
    /// Time budget in seconds. 0 = default (30s).
    pub time_budget_secs: f64,
    /// Pass threshold for leak probability. Default: 0.05.
    pub pass_threshold: f64,
    /// Fail threshold for leak probability. Default: 0.95.
    pub fail_threshold: f64,
    /// Random seed. 0 = use system entropy.
    pub seed: u64,
    /// Timer frequency in Hz (for converting ticks to nanoseconds).
    pub timer_frequency_hz: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            attacker_model: AttackerModel::AdjacentNetwork,
            max_samples: 100_000,
            time_budget_secs: 30.0,
            pass_threshold: 0.05,
            fail_threshold: 0.95,
            seed: 0,
            timer_frequency_hz: 1_000_000_000,
        }
    }
}

impl Config {
    /// Get the effective theta threshold in nanoseconds.
    pub fn theta_ns(&self) -> f64 {
        self.attacker_model.to_threshold_ns()
    }

    /// Get nanoseconds per tick based on timer frequency.
    pub fn ns_per_tick(&self) -> f64 {
        if self.timer_frequency_hz == 0 {
            1.0
        } else {
            1_000_000_000.0 / (self.timer_frequency_hz as f64)
        }
    }
}

/// Analysis result.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AnalysisResult {
    /// Test outcome.
    pub outcome: Outcome,
    /// Leak probability: P(max_k |(X*beta)_k| > theta | data).
    pub leak_probability: f64,
    /// Effect size estimate.
    pub effect: EffectEstimate,
    /// Measurement quality.
    pub quality: MeasurementQuality,
    /// Number of samples used per class.
    pub samples_used: u64,
    /// Time spent in seconds.
    pub elapsed_secs: f64,
    /// Exploitability (only valid if outcome == Fail).
    pub exploitability: Exploitability,
    /// Inconclusive reason (only valid if outcome == Inconclusive).
    pub inconclusive_reason: InconclusiveReason,
    /// Minimum detectable effect (shift) in nanoseconds.
    pub mde_shift_ns: f64,
    /// Minimum detectable effect (tail) in nanoseconds.
    pub mde_tail_ns: f64,
    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,
    /// User's requested threshold (theta) in nanoseconds.
    pub theta_user_ns: f64,
    /// Effective threshold after floor adjustment in nanoseconds.
    pub theta_eff_ns: f64,
    /// Measurement floor (theta_floor) in nanoseconds.
    pub theta_floor_ns: f64,
    /// Threshold at which decision was made.
    pub decision_threshold_ns: f64,
    /// Recommendation string (empty if not applicable).
    pub recommendation: String,
    /// Detailed diagnostics.
    pub diagnostics: Diagnostics,
}

// =============================================================================
// Opaque Objects - Calibration and State handles
// =============================================================================

/// Opaque calibration state.
///
/// Created by `calibrate_samples()`, used in `adaptive_step_batch()`.
#[derive(uniffi::Object)]
pub struct Calibration {
    inner: CoreCalibration,
}

impl Calibration {
    fn new(inner: CoreCalibration) -> Self {
        Self { inner }
    }
}

/// Adaptive sampling state.
///
/// Tracks cumulative samples and posterior evolution between adaptive steps.
/// Uses interior mutability (RwLock) to allow mutation through shared references,
/// which is required for UniFFI's object model.
#[derive(uniffi::Object)]
pub struct AdaptiveState {
    /// The core adaptive state, wrapped in RwLock for interior mutability.
    inner: RwLock<CoreAdaptiveState>,
}

#[uniffi::export]
impl AdaptiveState {
    /// Create a new adaptive state.
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(CoreAdaptiveState::new()),
        }
    }

    /// Get total baseline samples collected.
    pub fn total_baseline(&self) -> u64 {
        self.inner.read().unwrap().baseline_samples.len() as u64
    }

    /// Get total sample class samples collected.
    pub fn total_sample(&self) -> u64 {
        self.inner.read().unwrap().sample_samples.len() as u64
    }

    /// Get current leak probability estimate.
    /// Returns 0.5 if no posterior has been computed yet.
    pub fn current_probability(&self) -> f64 {
        self.inner
            .read()
            .unwrap()
            .current_posterior()
            .map(|p| p.leak_probability)
            .unwrap_or(0.5)
    }

    /// Get the number of batches collected so far.
    pub fn batch_count(&self) -> u64 {
        self.inner.read().unwrap().batch_count as u64
    }
}

impl Default for AdaptiveState {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of an adaptive step.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum AdaptiveStepResult {
    /// Continue sampling - no decision yet.
    Continue {
        /// Current leak probability estimate.
        current_probability: f64,
        /// Samples collected per class so far.
        samples_per_class: u64,
    },
    /// Decision reached - result is available.
    Decision {
        /// The final analysis result.
        result: AnalysisResult,
    },
}

// =============================================================================
// Exported Functions
// =============================================================================

/// Create a default configuration for the given attacker model.
#[uniffi::export]
pub fn config_default(model: AttackerModel) -> Config {
    Config {
        attacker_model: model,
        ..Config::default()
    }
}

/// Create a configuration for adjacent network attacker (100ns threshold).
#[uniffi::export]
pub fn config_adjacent_network() -> Config {
    config_default(AttackerModel::AdjacentNetwork)
}

/// Create a configuration for shared hardware attacker (0.6ns threshold).
#[uniffi::export]
pub fn config_shared_hardware() -> Config {
    config_default(AttackerModel::SharedHardware)
}

/// Create a configuration for remote network attacker (50us threshold).
#[uniffi::export]
pub fn config_remote_network() -> Config {
    config_default(AttackerModel::RemoteNetwork)
}

/// Run calibration phase on initial timing samples.
///
/// This function performs the calibration phase:
/// 1. Computes quantile differences between baseline and sample classes
/// 2. Estimates covariance matrix via block bootstrap
/// 3. Sets up prior distribution for Bayesian inference
/// 4. Returns calibration state for use in adaptive steps
///
/// # Arguments
///
/// * `baseline` - Baseline timing samples (raw ticks/cycles)
/// * `sample` - Sample class timing samples (raw ticks/cycles)
/// * `config` - Analysis configuration
///
/// # Returns
///
/// Calibration state on success, error on failure.
#[uniffi::export]
pub fn calibrate_samples(
    baseline: Vec<u64>,
    sample: Vec<u64>,
    config: Config,
) -> Result<Arc<Calibration>, TimingOracleError> {
    if baseline.len() < 100 || sample.len() < 100 {
        return Err(TimingOracleError::InsufficientSamples {
            minimum: 100,
            actual: baseline.len().min(sample.len()) as u64,
        });
    }

    let theta_ns = config.theta_ns();
    let ns_per_tick = config.ns_per_tick();
    let seed = if config.seed == 0 {
        baseline.iter().take(10).sum::<u64>() ^ 0x12345678
    } else {
        config.seed
    };

    let cal_config = CalibrationConfig {
        calibration_samples: baseline.len().min(sample.len()),
        bootstrap_iterations: 200,
        timer_resolution_ns: ns_per_tick,
        theta_ns,
        alpha: 0.01,
        seed,
        skip_preflight: true,
    };

    match calibrate(&baseline, &sample, ns_per_tick, &cal_config) {
        Ok(cal) => {
            let core_cal = CoreCalibration::new(
                cal.sigma_rate,
                cal.block_length,
                cal.sigma_t,
                cal.l_r,
                cal.theta_ns,
                cal.calibration_samples,
                cal.discrete_mode,
                cal.mde_shift_ns,
                cal.mde_tail_ns,
                cal.calibration_snapshot.clone(),
                cal.timer_resolution_ns,
                cal.samples_per_second,
                cal.c_floor,
                cal.projection_mismatch_thresh,
                cal.theta_tick,
                cal.theta_eff,
                cal.theta_floor_initial,
                cal.rng_seed,
                cal.batch_k,
            );
            Ok(Arc::new(Calibration::new(core_cal)))
        }
        Err(e) => Err(TimingOracleError::CalibrationFailed {
            message: format!("{:?}", e),
        }),
    }
}

/// Run one adaptive step with a new batch of samples.
///
/// This function adds a batch of samples to the adaptive state and runs
/// one step of the adaptive sampling algorithm. The state is updated in place.
///
/// # Arguments
///
/// * `calibration` - Calibration state from `calibrate_samples()`
/// * `state` - Adaptive state (will be mutated internally)
/// * `baseline` - New baseline timing samples for this batch
/// * `sample` - New sample class timing samples for this batch
/// * `config` - Analysis configuration
/// * `elapsed_secs` - Elapsed time in seconds since test start (measured by caller)
///
/// # Returns
///
/// - `Continue`: Keep sampling, no decision yet
/// - `Decision`: Analysis complete, result available
///
/// # Example (pseudocode)
///
/// ```ignore
/// let calibration = calibrate_samples(initial_baseline, initial_sample, config)?;
/// let state = AdaptiveState::new();
/// let start = Instant::now();
///
/// loop {
///     let (baseline_batch, sample_batch) = collect_samples(batch_size);
///     let elapsed = start.elapsed().as_secs_f64();
///
///     match adaptive_step_batch(&calibration, &state, baseline_batch, sample_batch, config, elapsed)? {
///         AdaptiveStepResult::Continue { current_probability, .. } => {
///             // Keep sampling
///         }
///         AdaptiveStepResult::Decision { result } => {
///             return Ok(result);
///         }
///     }
/// }
/// ```
#[uniffi::export]
pub fn adaptive_step_batch(
    calibration: &Calibration,
    state: &AdaptiveState,
    baseline: Vec<u64>,
    sample: Vec<u64>,
    config: Config,
    elapsed_secs: f64,
) -> Result<AdaptiveStepResult, TimingOracleError> {
    // Lock the state for writing
    let mut inner_state = state.inner.write().map_err(|_| TimingOracleError::AnalysisFailed {
        message: "Failed to acquire state lock".to_string(),
    })?;

    // Add the batch to state
    inner_state.add_batch(baseline, sample);

    // Create step config
    let step_config = AdaptiveStepConfig {
        pass_threshold: config.pass_threshold,
        fail_threshold: config.fail_threshold,
        time_budget_secs: config.time_budget_secs,
        max_samples: config.max_samples as usize,
        theta_ns: config.theta_ns(),
        seed: config.seed,
        ..AdaptiveStepConfig::default()
    };

    let ns_per_tick = config.ns_per_tick();

    // Run adaptive step
    let step = adaptive_step(
        &calibration.inner,
        &mut inner_state,
        ns_per_tick,
        elapsed_secs,
        &step_config,
    );

    match step {
        StepResult::Continue { posterior, samples_per_class } => {
            Ok(AdaptiveStepResult::Continue {
                current_probability: posterior.leak_probability,
                samples_per_class: samples_per_class as u64,
            })
        }
        StepResult::Decision(outcome) => {
            let result = build_result_from_outcome(&outcome, &config, Some(&calibration.inner));
            Ok(AdaptiveStepResult::Decision { result })
        }
    }
}

/// Run complete analysis on pre-collected timing data.
///
/// This is a convenience function that runs calibration and adaptive analysis
/// in a single call. Use the separate `calibrate_samples()` and `adaptive_step_batch()`
/// functions for incremental analysis.
///
/// # Arguments
///
/// * `baseline` - All baseline timing samples
/// * `sample` - All sample class timing samples
/// * `config` - Analysis configuration
///
/// # Returns
///
/// Analysis result on success, error on failure.
#[uniffi::export]
pub fn analyze(
    baseline: Vec<u64>,
    sample: Vec<u64>,
    config: Config,
) -> Result<AnalysisResult, TimingOracleError> {
    if baseline.len() < 100 || sample.len() < 100 {
        return Err(TimingOracleError::InsufficientSamples {
            minimum: 100,
            actual: baseline.len().min(sample.len()) as u64,
        });
    }

    let theta_ns = config.theta_ns();
    let ns_per_tick = config.ns_per_tick();
    let seed = if config.seed == 0 {
        baseline.iter().take(10).sum::<u64>() ^ 0x12345678
    } else {
        config.seed
    };

    // Split into calibration and adaptive samples
    let cal_samples = 5000.min(baseline.len() / 2);

    let cal_config = CalibrationConfig {
        calibration_samples: cal_samples,
        bootstrap_iterations: 200,
        timer_resolution_ns: ns_per_tick,
        theta_ns,
        alpha: 0.01,
        seed,
        skip_preflight: true,
    };

    // Run calibration
    let cal = match calibrate(&baseline[..cal_samples], &sample[..cal_samples], ns_per_tick, &cal_config) {
        Ok(c) => c,
        Err(e) => {
            return Err(TimingOracleError::CalibrationFailed {
                message: format!("{:?}", e),
            });
        }
    };

    // Convert to core Calibration
    let core_cal = CoreCalibration::new(
        cal.sigma_rate,
        cal.block_length,
        cal.sigma_t,
        cal.l_r,
        cal.theta_ns,
        cal.calibration_samples,
        cal.discrete_mode,
        cal.mde_shift_ns,
        cal.mde_tail_ns,
        cal.calibration_snapshot.clone(),
        cal.timer_resolution_ns,
        cal.samples_per_second,
        cal.c_floor,
        cal.projection_mismatch_thresh,
        cal.theta_tick,
        cal.theta_eff,
        cal.theta_floor_initial,
        cal.rng_seed,
        cal.batch_k,
    );

    // Run adaptive loop
    let mut state = CoreAdaptiveState::new();
    let step_config = AdaptiveStepConfig {
        pass_threshold: config.pass_threshold,
        fail_threshold: config.fail_threshold,
        time_budget_secs: config.time_budget_secs,
        max_samples: config.max_samples as usize,
        theta_ns,
        seed,
        ..AdaptiveStepConfig::default()
    };

    let batch_size = 1000;
    let mut elapsed_secs = 0.0;
    let time_per_batch = 0.01;

    for (b_chunk, s_chunk) in baseline.chunks(batch_size).zip(sample.chunks(batch_size)) {
        state.add_batch(b_chunk.to_vec(), s_chunk.to_vec());
        elapsed_secs += time_per_batch;

        let step = adaptive_step(&core_cal, &mut state, ns_per_tick, elapsed_secs, &step_config);

        if let StepResult::Decision(outcome) = step {
            let mut result = build_result_from_outcome(&outcome, &config, Some(&core_cal));
            result.mde_shift_ns = cal.mde_shift_ns;
            result.mde_tail_ns = cal.mde_tail_ns;
            result.theta_user_ns = theta_ns;
            result.theta_eff_ns = cal.theta_eff;
            return Ok(result);
        }
    }

    // Exhausted samples without decision
    let outcome = AdaptiveOutcome::Inconclusive {
        reason: CoreInconclusiveReason::SampleBudgetExceeded {
            current_probability: state
                .current_posterior()
                .map(|p| p.leak_probability)
                .unwrap_or(0.5),
            samples_collected: state.n_total(),
        },
        posterior: state.current_posterior().cloned(),
        samples_per_class: state.n_total(),
        elapsed_secs,
    };

    let mut result = build_result_from_outcome(&outcome, &config, Some(&core_cal));
    result.mde_shift_ns = cal.mde_shift_ns;
    result.mde_tail_ns = cal.mde_tail_ns;
    result.theta_user_ns = theta_ns;
    result.theta_eff_ns = cal.theta_eff;
    Ok(result)
}

/// Get the library version.
#[uniffi::export]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// =============================================================================
// Internal Helper Functions
// =============================================================================

fn classify_effect_pattern(shift_ns: f64, tail_ns: f64) -> EffectPattern {
    let shift_abs = shift_ns.abs();
    let tail_abs = tail_ns.abs();

    if shift_abs < 1.0 && tail_abs < 1.0 {
        EffectPattern::Indeterminate
    } else if shift_abs > tail_abs * 2.0 {
        EffectPattern::UniformShift
    } else if tail_abs > shift_abs * 2.0 {
        EffectPattern::TailEffect
    } else {
        EffectPattern::Mixed
    }
}

fn build_diagnostics_from_posterior(
    posterior: Option<&timing_oracle_core::adaptive::Posterior>,
    calibration: Option<&CoreCalibration>,
) -> Diagnostics {
    let mut diag = Diagnostics::default();

    if let Some(p) = posterior {
        diag.effective_sample_size = p.n as u64;
        diag.projection_mismatch_q = p.projection_mismatch_q;
        diag.projection_mismatch_ok = calibration
            .map(|c| p.projection_mismatch_q <= c.projection_mismatch_thresh)
            .unwrap_or(true);

        diag.lambda_mean = p.lambda_mean.unwrap_or(1.0);
        diag.lambda_mixing_ok = p.lambda_mixing_ok.unwrap_or(true);
        diag.kappa_mean = p.kappa_mean.unwrap_or(1.0);
        diag.kappa_cv = p.kappa_cv.unwrap_or(0.0);
        diag.kappa_ess = p.kappa_ess.unwrap_or(0.0);
        diag.kappa_mixing_ok = p.kappa_mixing_ok.unwrap_or(true);
    }

    if let Some(cal) = calibration {
        diag.dependence_length = cal.block_length as u64;
        diag.discrete_mode = cal.discrete_mode;
        diag.timer_resolution_ns = cal.timer_resolution_ns;
    }

    diag
}

fn build_result_from_outcome(
    outcome: &AdaptiveOutcome,
    config: &Config,
    calibration: Option<&CoreCalibration>,
) -> AnalysisResult {
    match outcome {
        AdaptiveOutcome::LeakDetected {
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            let shift_ns = posterior.beta_proj[0];
            let tail_ns = posterior.beta_proj[1];
            let total_effect = sqrt(shift_ns * shift_ns + tail_ns * tail_ns);
            let ci_width = 1.96 * sqrt(posterior.beta_proj_cov[(0, 0)] + posterior.beta_proj_cov[(1, 1)]);

            AnalysisResult {
                outcome: Outcome::Fail,
                leak_probability: posterior.leak_probability,
                effect: EffectEstimate {
                    shift_ns,
                    tail_ns,
                    credible_interval: CredibleInterval {
                        low: total_effect - ci_width,
                        high: total_effect + ci_width,
                    },
                    pattern: classify_effect_pattern(shift_ns, tail_ns),
                    interpretation_caveat: None,
                },
                quality: MeasurementQuality::Good,
                samples_used: *samples_per_class as u64,
                elapsed_secs: *elapsed_secs,
                exploitability: Exploitability::from_effect_ns(total_effect),
                inconclusive_reason: InconclusiveReason::None,
                mde_shift_ns: 0.0,
                mde_tail_ns: 0.0,
                timer_resolution_ns: calibration.map(|c| c.timer_resolution_ns).unwrap_or(0.0),
                theta_user_ns: config.theta_ns(),
                theta_eff_ns: calibration.map(|c| c.theta_eff).unwrap_or(config.theta_ns()),
                theta_floor_ns: calibration.map(|c| c.theta_floor_initial).unwrap_or(0.0),
                decision_threshold_ns: calibration.map(|c| c.theta_eff).unwrap_or(config.theta_ns()),
                recommendation: String::new(),
                diagnostics: build_diagnostics_from_posterior(Some(posterior), calibration),
            }
        }

        AdaptiveOutcome::NoLeakDetected {
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            let shift_ns = posterior.beta_proj[0];
            let tail_ns = posterior.beta_proj[1];
            let total_effect = sqrt(shift_ns * shift_ns + tail_ns * tail_ns);
            let ci_width = 1.96 * sqrt(posterior.beta_proj_cov[(0, 0)] + posterior.beta_proj_cov[(1, 1)]);

            AnalysisResult {
                outcome: Outcome::Pass,
                leak_probability: posterior.leak_probability,
                effect: EffectEstimate {
                    shift_ns,
                    tail_ns,
                    credible_interval: CredibleInterval {
                        low: total_effect - ci_width,
                        high: total_effect + ci_width,
                    },
                    pattern: classify_effect_pattern(shift_ns, tail_ns),
                    interpretation_caveat: None,
                },
                quality: MeasurementQuality::Good,
                samples_used: *samples_per_class as u64,
                elapsed_secs: *elapsed_secs,
                exploitability: Exploitability::SharedHardwareOnly,
                inconclusive_reason: InconclusiveReason::None,
                mde_shift_ns: 0.0,
                mde_tail_ns: 0.0,
                timer_resolution_ns: calibration.map(|c| c.timer_resolution_ns).unwrap_or(0.0),
                theta_user_ns: config.theta_ns(),
                theta_eff_ns: calibration.map(|c| c.theta_eff).unwrap_or(config.theta_ns()),
                theta_floor_ns: calibration.map(|c| c.theta_floor_initial).unwrap_or(0.0),
                decision_threshold_ns: calibration.map(|c| c.theta_eff).unwrap_or(config.theta_ns()),
                recommendation: String::new(),
                diagnostics: build_diagnostics_from_posterior(Some(posterior), calibration),
            }
        }

        AdaptiveOutcome::ThresholdElevated {
            posterior,
            theta_user,
            theta_eff,
            achievable_at_max,
            samples_per_class,
            elapsed_secs,
            ..
        } => {
            let shift_ns = posterior.beta_proj[0];
            let tail_ns = posterior.beta_proj[1];
            let total_effect = sqrt(shift_ns * shift_ns + tail_ns * tail_ns);
            let ci_width = 1.96 * sqrt(posterior.beta_proj_cov[(0, 0)] + posterior.beta_proj_cov[(1, 1)]);

            let guidance = if *achievable_at_max {
                format!(
                    "Threshold elevated from {:.0}ns to {:.1}ns. More samples could achieve the requested threshold.",
                    theta_user, theta_eff
                )
            } else {
                format!(
                    "Threshold elevated from {:.0}ns to {:.1}ns. Use a cycle counter (PMU timer) for better resolution.",
                    theta_user, theta_eff
                )
            };

            AnalysisResult {
                outcome: Outcome::Inconclusive,
                leak_probability: posterior.leak_probability,
                effect: EffectEstimate {
                    shift_ns,
                    tail_ns,
                    credible_interval: CredibleInterval {
                        low: total_effect - ci_width,
                        high: total_effect + ci_width,
                    },
                    pattern: classify_effect_pattern(shift_ns, tail_ns),
                    interpretation_caveat: None,
                },
                quality: MeasurementQuality::Good,
                samples_used: *samples_per_class as u64,
                elapsed_secs: *elapsed_secs,
                exploitability: Exploitability::SharedHardwareOnly,
                inconclusive_reason: InconclusiveReason::ThresholdElevated {
                    theta_user: *theta_user,
                    theta_eff: *theta_eff,
                    leak_probability_at_eff: posterior.leak_probability,
                    achievable_at_max: *achievable_at_max,
                    message: guidance.clone(),
                    guidance: guidance.clone(),
                },
                mde_shift_ns: 0.0,
                mde_tail_ns: 0.0,
                timer_resolution_ns: calibration.map(|c| c.timer_resolution_ns).unwrap_or(0.0),
                theta_user_ns: *theta_user,
                theta_eff_ns: *theta_eff,
                theta_floor_ns: calibration.map(|c| c.theta_floor_initial).unwrap_or(0.0),
                decision_threshold_ns: *theta_eff,
                recommendation: guidance,
                diagnostics: build_diagnostics_from_posterior(Some(posterior), calibration),
            }
        }

        AdaptiveOutcome::Inconclusive {
            reason,
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            let (shift_ns, tail_ns) = posterior
                .as_ref()
                .map(|p| (p.beta_proj[0], p.beta_proj[1]))
                .unwrap_or((0.0, 0.0));

            let (inconclusive_reason, recommendation) = convert_inconclusive_reason(reason);

            AnalysisResult {
                outcome: Outcome::Inconclusive,
                leak_probability: posterior.as_ref().map(|p| p.leak_probability).unwrap_or(0.5),
                effect: EffectEstimate {
                    shift_ns,
                    tail_ns,
                    credible_interval: CredibleInterval { low: 0.0, high: 0.0 },
                    pattern: classify_effect_pattern(shift_ns, tail_ns),
                    interpretation_caveat: None,
                },
                quality: MeasurementQuality::Poor,
                samples_used: *samples_per_class as u64,
                elapsed_secs: *elapsed_secs,
                exploitability: Exploitability::SharedHardwareOnly,
                inconclusive_reason,
                mde_shift_ns: 0.0,
                mde_tail_ns: 0.0,
                timer_resolution_ns: calibration.map(|c| c.timer_resolution_ns).unwrap_or(0.0),
                theta_user_ns: config.theta_ns(),
                theta_eff_ns: calibration.map(|c| c.theta_eff).unwrap_or(config.theta_ns()),
                theta_floor_ns: calibration.map(|c| c.theta_floor_initial).unwrap_or(0.0),
                decision_threshold_ns: calibration.map(|c| c.theta_eff).unwrap_or(config.theta_ns()),
                recommendation,
                diagnostics: build_diagnostics_from_posterior(posterior.as_ref(), calibration),
            }
        }
    }
}

fn convert_inconclusive_reason(reason: &CoreInconclusiveReason) -> (InconclusiveReason, String) {
    match reason {
        CoreInconclusiveReason::DataTooNoisy { message, guidance, .. } => (
            InconclusiveReason::DataTooNoisy {
                message: message.clone(),
                guidance: guidance.clone(),
            },
            guidance.clone(),
        ),
        CoreInconclusiveReason::NotLearning { message, guidance, .. } => (
            InconclusiveReason::NotLearning {
                message: message.clone(),
                guidance: guidance.clone(),
            },
            guidance.clone(),
        ),
        CoreInconclusiveReason::WouldTakeTooLong {
            estimated_time_secs,
            samples_needed,
            guidance,
            ..
        } => (
            InconclusiveReason::WouldTakeTooLong {
                estimated_time_secs: *estimated_time_secs,
                samples_needed: *samples_needed as u64,
                guidance: guidance.clone(),
            },
            guidance.clone(),
        ),
        CoreInconclusiveReason::TimeBudgetExceeded {
            current_probability,
            samples_collected,
            ..
        } => (
            InconclusiveReason::TimeBudgetExceeded {
                current_probability: *current_probability,
                samples_collected: *samples_collected as u64,
            },
            "Increase time budget or reduce threshold".to_string(),
        ),
        CoreInconclusiveReason::SampleBudgetExceeded {
            current_probability,
            samples_collected,
            ..
        } => (
            InconclusiveReason::SampleBudgetExceeded {
                current_probability: *current_probability,
                samples_collected: *samples_collected as u64,
            },
            "Increase sample budget or reduce threshold".to_string(),
        ),
        CoreInconclusiveReason::ConditionsChanged { message, guidance, .. } => (
            InconclusiveReason::ConditionsChanged {
                message: message.clone(),
                guidance: guidance.clone(),
            },
            guidance.clone(),
        ),
        CoreInconclusiveReason::ThresholdElevated {
            theta_user,
            theta_eff,
            leak_probability_at_eff,
            achievable_at_max,
            message,
            guidance,
            ..
        } => (
            InconclusiveReason::ThresholdElevated {
                theta_user: *theta_user,
                theta_eff: *theta_eff,
                leak_probability_at_eff: *leak_probability_at_eff,
                achievable_at_max: *achievable_at_max,
                message: message.clone(),
                guidance: guidance.clone(),
            },
            guidance.clone(),
        ),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = config_default(AttackerModel::AdjacentNetwork);
        assert!((config.theta_ns() - 100.0).abs() < 1e-10);
        assert!((config.pass_threshold - 0.05).abs() < 1e-10);
        assert!((config.fail_threshold - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_attacker_thresholds() {
        assert!((AttackerModel::SharedHardware.to_threshold_ns() - 0.6).abs() < 1e-10);
        assert!((AttackerModel::AdjacentNetwork.to_threshold_ns() - 100.0).abs() < 1e-10);
        assert!((AttackerModel::RemoteNetwork.to_threshold_ns() - 50_000.0).abs() < 1e-10);
        assert!(
            (AttackerModel::Custom { threshold_ns: 42.0 }.to_threshold_ns() - 42.0).abs() < 1e-10
        );
    }

    #[test]
    fn test_effect_pattern_classification() {
        assert_eq!(
            classify_effect_pattern(100.0, 10.0),
            EffectPattern::UniformShift
        );
        assert_eq!(
            classify_effect_pattern(10.0, 100.0),
            EffectPattern::TailEffect
        );
        assert_eq!(classify_effect_pattern(50.0, 50.0), EffectPattern::Mixed);
        assert_eq!(
            classify_effect_pattern(0.1, 0.1),
            EffectPattern::Indeterminate
        );
    }

    #[test]
    fn test_exploitability_thresholds() {
        assert_eq!(
            Exploitability::from_effect_ns(5.0),
            Exploitability::SharedHardwareOnly
        );
        assert_eq!(
            Exploitability::from_effect_ns(50.0),
            Exploitability::Http2Multiplexing
        );
        assert_eq!(
            Exploitability::from_effect_ns(1000.0),
            Exploitability::StandardRemote
        );
        assert_eq!(
            Exploitability::from_effect_ns(50_000.0),
            Exploitability::ObviousLeak
        );
    }

    #[test]
    fn test_adaptive_state_new() {
        let state = AdaptiveState::new();
        assert_eq!(state.total_baseline(), 0);
        assert_eq!(state.total_sample(), 0);
        assert_eq!(state.batch_count(), 0);
        assert!((state.current_probability() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_state_is_send_sync() {
        // This test verifies that AdaptiveState is Send + Sync,
        // which is required for UniFFI objects.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AdaptiveState>();
    }

    #[test]
    fn test_calibration_is_send_sync() {
        // This test verifies that Calibration is Send + Sync.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Calibration>();
    }
}
