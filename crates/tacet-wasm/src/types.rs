//! WASM-compatible type definitions.
//!
//! These types mirror the core types but are designed for wasm-bindgen FFI.
//! Uses tsify-next for automatic TypeScript type generation.

use serde::{Deserialize, Serialize};
use tacet_core::constants::{
    DEFAULT_FAIL_THRESHOLD, DEFAULT_MAX_SAMPLES, DEFAULT_PASS_THRESHOLD, DEFAULT_TIME_BUDGET_SECS,
};
use tacet_core::result::{
    EffectPattern as CoreEffectPattern, Exploitability as CoreExploitability,
    MeasurementQuality as CoreMeasurementQuality,
};
use tacet_core::types::AttackerModel as CoreAttackerModel;
use tsify::Tsify;

/// Attacker model determines the minimum effect threshold (theta) for leak detection.
#[derive(Tsify, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub enum AttackerModel {
    /// theta = 0.6 ns (~2 cycles @ 3GHz) - SGX, cross-VM, containers
    SharedHardware,
    /// theta = 3.3 ns (~10 cycles) - Post-quantum crypto
    PostQuantum,
    /// theta = 100 ns - LAN, HTTP/2 (Timeless Timing Attacks)
    #[default]
    AdjacentNetwork,
    /// theta = 50 us - General internet
    RemoteNetwork,
    /// theta -> 0 - Detect any difference (not for CI)
    Research,
}

impl AttackerModel {
    /// Convert to core AttackerModel type.
    pub fn to_core(&self) -> CoreAttackerModel {
        match self {
            AttackerModel::SharedHardware => CoreAttackerModel::SharedHardware,
            AttackerModel::PostQuantum => CoreAttackerModel::PostQuantumSentinel,
            AttackerModel::AdjacentNetwork => CoreAttackerModel::AdjacentNetwork,
            AttackerModel::RemoteNetwork => CoreAttackerModel::RemoteNetwork,
            AttackerModel::Research => CoreAttackerModel::Research,
        }
    }

    /// Convert to threshold in nanoseconds.
    pub fn to_threshold_ns(&self) -> f64 {
        self.to_core().to_threshold_ns()
    }
}

/// Test outcome.
#[derive(Tsify, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub enum Outcome {
    /// No timing leak detected within threshold theta.
    Pass = 0,
    /// Timing leak detected exceeding threshold theta.
    Fail = 1,
    /// Could not reach a decision.
    Inconclusive = 2,
    /// Operation too fast to measure reliably.
    Unmeasurable = 3,
}

/// Reason for inconclusive result.
#[derive(Tsify, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub enum InconclusiveReason {
    /// Not applicable (outcome is not Inconclusive).
    None = 0,
    /// Posterior approximately equals prior after calibration.
    DataTooNoisy = 1,
    /// Posterior stopped updating despite new data.
    NotLearning = 2,
    /// Estimated time to decision exceeds budget.
    WouldTakeTooLong = 3,
    /// Time budget exhausted.
    TimeBudgetExceeded = 4,
    /// Sample limit reached.
    SampleBudgetExceeded = 5,
    /// Measurement conditions changed during test.
    ConditionsChanged = 6,
    /// Threshold was elevated due to measurement noise.
    ThresholdElevated = 7,
}

/// Pattern of timing effect.
#[derive(Tsify, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub enum EffectPattern {
    /// Uniform shift across all quantiles.
    UniformShift = 0,
    /// Effect concentrated in tails (upper quantiles).
    TailEffect = 1,
    /// Both shift and tail components present.
    Mixed = 2,
    /// Cannot determine pattern.
    Indeterminate = 3,
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

/// Exploitability assessment.
#[derive(Tsify, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub enum Exploitability {
    /// < 10 ns - Requires shared hardware to exploit.
    SharedHardwareOnly = 0,
    /// 10-100 ns - Exploitable via HTTP/2 request multiplexing.
    Http2Multiplexing = 1,
    /// 100 ns - 10 us - Exploitable with standard remote timing.
    StandardRemote = 2,
    /// > 10 us - Obvious leak, trivially exploitable.
    ObviousLeak = 3,
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
#[derive(Tsify, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub enum MeasurementQuality {
    /// MDE < 5 ns - Excellent measurement precision.
    Excellent = 0,
    /// MDE 5-20 ns - Good precision.
    Good = 1,
    /// MDE 20-100 ns - Poor precision.
    Poor = 2,
    /// MDE > 100 ns - Too noisy for reliable detection.
    TooNoisy = 3,
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

/// Effect size estimate.
#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct EffectEstimate {
    /// Uniform shift component in nanoseconds.
    pub shift_ns: f64,
    /// Tail effect component in nanoseconds.
    pub tail_ns: f64,
    /// Lower bound of 95% credible interval for total effect.
    pub credible_interval_low: f64,
    /// Upper bound of 95% credible interval for total effect.
    pub credible_interval_high: f64,
    /// Pattern of the effect.
    pub pattern: EffectPattern,
    /// Interpretation caveat if model fit is poor.
    pub interpretation_caveat: Option<String>,
}

impl EffectEstimate {
    /// Compute the total effect magnitude (L2 norm of shift and tail).
    pub fn total_effect_ns(&self) -> f64 {
        (self.shift_ns * self.shift_ns + self.tail_ns * self.tail_ns).sqrt()
    }
}

/// Diagnostics information for debugging and quality assessment.
#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostics {
    /// Block length used for bootstrap resampling.
    pub dependence_length: u32,
    /// Effective sample size accounting for autocorrelation.
    pub effective_sample_size: u32,
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
    /// Whether lambda chain mixed well.
    pub lambda_mixing_ok: bool,
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
            lambda_mean: 1.0,
            lambda_mixing_ok: true,
        }
    }
}

/// Configuration for the timing analysis.
#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    /// Attacker model to use (determines threshold theta).
    pub attacker_model: AttackerModel,
    /// Maximum samples per class. 0 = default (100,000).
    pub max_samples: u32,
    /// Time budget in milliseconds. 0 = default (30,000 ms).
    pub time_budget_ms: u32,
    /// Pass threshold for leak probability. Default: 0.05.
    pub pass_threshold: f64,
    /// Fail threshold for leak probability. Default: 0.95.
    pub fail_threshold: f64,
    /// Random seed. None = use system entropy.
    pub seed: Option<u32>,
    /// Custom threshold in nanoseconds (overrides attacker_model).
    pub custom_threshold_ns: Option<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            attacker_model: AttackerModel::AdjacentNetwork,
            max_samples: DEFAULT_MAX_SAMPLES as u32,
            time_budget_ms: (DEFAULT_TIME_BUDGET_SECS * 1000) as u32,
            pass_threshold: DEFAULT_PASS_THRESHOLD,
            fail_threshold: DEFAULT_FAIL_THRESHOLD,
            seed: None,
            custom_threshold_ns: None,
        }
    }
}

impl Config {
    /// Get the effective theta threshold in nanoseconds.
    pub fn theta_ns(&self) -> f64 {
        self.custom_threshold_ns
            .unwrap_or_else(|| self.attacker_model.to_threshold_ns())
    }

    /// Get time budget in seconds.
    pub fn time_budget_secs(&self) -> f64 {
        self.time_budget_ms as f64 / 1000.0
    }
}

/// Analysis result.
#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
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
    pub samples_used: u32,
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
    /// Recommendation string (empty if not applicable).
    pub recommendation: String,
    /// Detailed diagnostics.
    pub diagnostics: Diagnostics,
}

/// Timer calibration info.
#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct TimerInfo {
    /// Cycles per nanosecond.
    pub cycles_per_ns: f64,
    /// Timer resolution in nanoseconds.
    pub resolution_ns: f64,
    /// Timer frequency in Hz.
    pub frequency_hz: f64,
}

/// Result of an adaptive step.
#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct AdaptiveStepResult {
    /// Whether a decision was reached.
    pub is_decision: bool,
    /// Current leak probability estimate.
    pub current_probability: f64,
    /// Samples collected per class so far.
    pub samples_per_class: u32,
    /// The final result (only valid if is_decision is true).
    pub result: Option<AnalysisResult>,
}

/// Calibration handle for adaptive analysis.
/// Contains opaque calibration state from the calibration phase.
#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationHandle {
    /// Serialized calibration state (opaque to JS).
    pub data: Vec<u8>,
    /// Nanoseconds per timer tick.
    pub ns_per_tick: f64,
}

/// Adaptive state handle for the adaptive sampling loop.
/// Contains opaque state from the adaptive phase.
#[derive(Tsify, Debug, Clone, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct AdaptiveStateHandle {
    /// Serialized adaptive state (opaque to JS).
    pub data: Vec<u8>,
}
