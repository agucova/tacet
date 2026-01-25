//! C-compatible type definitions for timing-oracle.

use timing_oracle_core::adaptive::{AdaptiveState, Calibration};
use timing_oracle_core::constants::{
    DEFAULT_FAIL_THRESHOLD, DEFAULT_MAX_SAMPLES, DEFAULT_PASS_THRESHOLD, DEFAULT_TIME_BUDGET_SECS,
};
use timing_oracle_core::result::{Exploitability, MeasurementQuality};
use timing_oracle_core::types::AttackerModel;

// ============================================================================
// Attacker Models
// ============================================================================

/// Attacker model presets defining the timing threshold (theta).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToAttackerModel {
    /// theta = 0.6 ns (~2 cycles @ 3GHz) - SGX, cross-VM, containers
    SharedHardware = 0,
    /// theta = 3.3 ns (~10 cycles) - Post-quantum crypto
    PostQuantum = 1,
    /// theta = 100 ns - LAN, HTTP/2 (Timeless Timing Attacks)
    AdjacentNetwork = 2,
    /// theta = 50 us - General internet
    RemoteNetwork = 3,
    /// theta -> 0 - Detect any difference (not for CI)
    Research = 4,
}

impl From<ToAttackerModel> for AttackerModel {
    fn from(model: ToAttackerModel) -> Self {
        match model {
            ToAttackerModel::SharedHardware => AttackerModel::SharedHardware,
            ToAttackerModel::PostQuantum => AttackerModel::PostQuantumSentinel,
            ToAttackerModel::AdjacentNetwork => AttackerModel::AdjacentNetwork,
            ToAttackerModel::RemoteNetwork => AttackerModel::RemoteNetwork,
            ToAttackerModel::Research => AttackerModel::Research,
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for timing analysis.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ToConfig {
    /// Attacker model (determines threshold theta).
    pub attacker_model: ToAttackerModel,
    /// Custom threshold in nanoseconds (only if attacker_model would be Custom).
    /// Set to 0.0 to use the attacker model's default threshold.
    pub custom_threshold_ns: f64,
    /// Maximum samples per class. 0 = default (100,000).
    pub max_samples: u64,
    /// Time budget in seconds. 0 = default (30s).
    pub time_budget_secs: f64,
    /// Pass threshold for P(leak). Default: 0.05.
    pub pass_threshold: f64,
    /// Fail threshold for P(leak). Default: 0.95.
    pub fail_threshold: f64,
    /// Random seed. 0 = use default seed.
    pub seed: u64,
    /// Timer frequency in Hz (for converting ticks to nanoseconds).
    /// 0 = assume 1 tick = 1 ns.
    pub timer_frequency_hz: u64,
}

impl Default for ToConfig {
    fn default() -> Self {
        Self {
            attacker_model: ToAttackerModel::AdjacentNetwork,
            custom_threshold_ns: 0.0,
            max_samples: DEFAULT_MAX_SAMPLES as u64,
            time_budget_secs: DEFAULT_TIME_BUDGET_SECS as f64,
            pass_threshold: DEFAULT_PASS_THRESHOLD,
            fail_threshold: DEFAULT_FAIL_THRESHOLD,
            seed: 0,
            timer_frequency_hz: 0,
        }
    }
}

impl ToConfig {
    /// Get the effective threshold in nanoseconds.
    pub fn threshold_ns(&self) -> f64 {
        if self.custom_threshold_ns > 0.0 {
            self.custom_threshold_ns
        } else {
            let model: AttackerModel = self.attacker_model.into();
            model.to_threshold_ns()
        }
    }
}

// ============================================================================
// Outcome Types
// ============================================================================

/// Test outcome.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToOutcome {
    /// No timing leak detected within the threshold.
    Pass = 0,
    /// Timing leak detected exceeding the threshold.
    Fail = 1,
    /// Could not reach a decision within budget.
    Inconclusive = 2,
    /// Operation too fast to measure reliably.
    Unmeasurable = 3,
}

/// Reason for inconclusive result.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToInconclusiveReason {
    None = 0,
    DataTooNoisy = 1,
    NotLearning = 2,
    WouldTakeTooLong = 3,
    TimeBudgetExceeded = 4,
    SampleBudgetExceeded = 5,
    ConditionsChanged = 6,
    ThresholdElevated = 7,
}

/// Exploitability assessment.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToExploitability {
    /// < 10 ns - Requires shared hardware to exploit.
    SharedHardwareOnly = 0,
    /// 10-100 ns - Exploitable via HTTP/2 multiplexing.
    Http2Multiplexing = 1,
    /// 100 ns - 10 us - Exploitable with standard remote timing.
    StandardRemote = 2,
    /// > 10 us - Obvious leak, trivially exploitable.
    ObviousLeak = 3,
}

impl From<Exploitability> for ToExploitability {
    fn from(e: Exploitability) -> Self {
        match e {
            Exploitability::SharedHardwareOnly => ToExploitability::SharedHardwareOnly,
            Exploitability::Http2Multiplexing => ToExploitability::Http2Multiplexing,
            Exploitability::StandardRemote => ToExploitability::StandardRemote,
            Exploitability::ObviousLeak => ToExploitability::ObviousLeak,
        }
    }
}

/// Measurement quality assessment.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToMeasurementQuality {
    /// MDE < 5 ns - Excellent precision.
    Excellent = 0,
    /// MDE 5-20 ns - Good for most use cases.
    Good = 1,
    /// MDE 20-100 ns - Limited precision.
    Poor = 2,
    /// MDE > 100 ns - Too noisy for reliable detection.
    TooNoisy = 3,
}

impl From<MeasurementQuality> for ToMeasurementQuality {
    fn from(q: MeasurementQuality) -> Self {
        match q {
            MeasurementQuality::Excellent => ToMeasurementQuality::Excellent,
            MeasurementQuality::Good => ToMeasurementQuality::Good,
            MeasurementQuality::Poor => ToMeasurementQuality::Poor,
            MeasurementQuality::TooNoisy => ToMeasurementQuality::TooNoisy,
        }
    }
}

/// Effect pattern.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToEffectPattern {
    /// Constant timing difference across all quantiles.
    UniformShift = 0,
    /// Timing difference concentrated in upper quantiles.
    TailEffect = 1,
    /// Both shift and tail components present.
    Mixed = 2,
    /// Pattern cannot be determined.
    Indeterminate = 3,
}

// ============================================================================
// Diagnostics
// ============================================================================

/// Diagnostics information for debugging and quality assessment.
///
/// Mirrors the diagnostics available in UniFFI bindings.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ToDiagnostics {
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
    /// Whether discrete mode was used (low timer resolution).
    pub discrete_mode: bool,
    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,
    /// Posterior mean of latent scale lambda.
    pub lambda_mean: f64,
    /// Posterior standard deviation of lambda.
    pub lambda_sd: f64,
    /// Effective sample size of lambda chain.
    pub lambda_ess: f64,
    /// Whether lambda chain mixed well.
    pub lambda_mixing_ok: bool,
    /// Posterior mean of likelihood precision kappa.
    pub kappa_mean: f64,
    /// Coefficient of variation of kappa.
    pub kappa_cv: f64,
    /// Effective sample size of kappa chain.
    pub kappa_ess: f64,
    /// Whether kappa chain mixed well.
    pub kappa_mixing_ok: bool,
}

impl Default for ToDiagnostics {
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
            lambda_mean: 1.0,
            lambda_sd: 0.0,
            lambda_ess: 0.0,
            lambda_mixing_ok: true,
            kappa_mean: 1.0,
            kappa_cv: 0.0,
            kappa_ess: 0.0,
            kappa_mixing_ok: true,
        }
    }
}

// ============================================================================
// Effect Estimate
// ============================================================================

/// Effect size estimate.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ToEffect {
    /// Uniform shift component in nanoseconds.
    pub shift_ns: f64,
    /// Tail effect component in nanoseconds.
    pub tail_ns: f64,
    /// 95% credible interval lower bound.
    pub ci_low_ns: f64,
    /// 95% credible interval upper bound.
    pub ci_high_ns: f64,
    /// Effect pattern.
    pub pattern: ToEffectPattern,
}

impl Default for ToEffect {
    fn default() -> Self {
        Self {
            shift_ns: 0.0,
            tail_ns: 0.0,
            ci_low_ns: 0.0,
            ci_high_ns: 0.0,
            pattern: ToEffectPattern::Indeterminate,
        }
    }
}

// ============================================================================
// Analysis Result
// ============================================================================

/// Complete analysis result.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ToResult {
    /// Test outcome.
    pub outcome: ToOutcome,
    /// Leak probability: P(max effect > theta | data).
    pub leak_probability: f64,
    /// Effect size estimate.
    pub effect: ToEffect,
    /// Measurement quality.
    pub quality: ToMeasurementQuality,
    /// Number of samples used per class.
    pub samples_used: u64,
    /// Elapsed time in seconds.
    pub elapsed_secs: f64,
    /// Exploitability (only meaningful if outcome == Fail).
    pub exploitability: ToExploitability,
    /// Inconclusive reason (only meaningful if outcome == Inconclusive).
    pub inconclusive_reason: ToInconclusiveReason,
    /// Minimum detectable effect (shift) in nanoseconds.
    pub mde_shift_ns: f64,
    /// Minimum detectable effect (tail) in nanoseconds.
    pub mde_tail_ns: f64,
    /// User's requested threshold in nanoseconds.
    pub theta_user_ns: f64,
    /// Effective threshold after floor adjustment.
    pub theta_eff_ns: f64,
    /// Measurement floor in nanoseconds.
    pub theta_floor_ns: f64,
    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,
    /// Threshold at which decision was made in nanoseconds.
    pub decision_threshold_ns: f64,
    /// Detailed diagnostics.
    pub diagnostics: ToDiagnostics,
}

impl Default for ToResult {
    fn default() -> Self {
        Self {
            outcome: ToOutcome::Inconclusive,
            leak_probability: 0.5,
            effect: ToEffect::default(),
            quality: ToMeasurementQuality::Good,
            samples_used: 0,
            elapsed_secs: 0.0,
            exploitability: ToExploitability::SharedHardwareOnly,
            inconclusive_reason: ToInconclusiveReason::None,
            mde_shift_ns: 0.0,
            mde_tail_ns: 0.0,
            theta_user_ns: 0.0,
            theta_eff_ns: 0.0,
            theta_floor_ns: 0.0,
            timer_resolution_ns: 0.0,
            decision_threshold_ns: 0.0,
            diagnostics: ToDiagnostics::default(),
        }
    }
}

// ============================================================================
// Error Handling
// ============================================================================

/// Error codes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToError {
    /// No error.
    Ok = 0,
    /// Null pointer passed.
    NullPointer = 1,
    /// Invalid configuration.
    InvalidConfig = 2,
    /// Calibration failed.
    CalibrationFailed = 3,
    /// Analysis failed.
    AnalysisFailed = 4,
    /// Not enough samples.
    NotEnoughSamples = 5,
}

// ============================================================================
// Opaque Handle Types
// ============================================================================

/// Opaque handle to calibration data.
/// Created by `to_calibrate()`, freed by `to_calibration_free()`.
pub struct ToCalibration {
    pub(crate) inner: Calibration,
    pub(crate) ns_per_tick: f64,
}

/// Opaque handle to adaptive state.
/// Created by `to_state_new()`, freed by `to_state_free()`.
pub struct ToState {
    pub(crate) inner: AdaptiveState,
}

// ============================================================================
// Step Result
// ============================================================================

/// Result of an adaptive step.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ToStepResult {
    /// Whether a decision was reached.
    pub has_decision: bool,
    /// Current leak probability estimate.
    pub leak_probability: f64,
    /// Number of samples used so far.
    pub samples_used: u64,
    /// Elapsed time in seconds (only set when has_decision is true).
    pub elapsed_secs: f64,
    /// The result (only valid if has_decision is true).
    pub result: ToResult,
}

impl Default for ToStepResult {
    fn default() -> Self {
        Self {
            has_decision: false,
            leak_probability: 0.5,
            samples_used: 0,
            elapsed_secs: 0.0,
            result: ToResult::default(),
        }
    }
}

// ============================================================================
// Callback Function Types
// ============================================================================

/// Callback function type for collecting timing samples.
///
/// The callback is invoked by `to_test()` to collect batches of timing samples.
/// The user must fill `baseline_out` and `sample_out` with `count` samples each.
///
/// # Parameters
/// - `baseline_out`: Array to fill with baseline timing samples (in timer ticks)
/// - `sample_out`: Array to fill with sample timing samples (in timer ticks)
/// - `count`: Number of samples to collect for each class
/// - `user_ctx`: User context pointer passed to `to_test()`
///
/// # Example
///
/// ```c
/// void my_collect(uint64_t* baseline, uint64_t* sample, size_t count, void* ctx) {
///     for (size_t i = 0; i < count; i++) {
///         baseline[i] = measure_baseline_operation();
///         sample[i] = measure_sample_operation();
///     }
/// }
/// ```
pub type ToCollectFn =
    Option<unsafe extern "C" fn(baseline_out: *mut u64, sample_out: *mut u64, count: usize, user_ctx: *mut std::ffi::c_void)>;
