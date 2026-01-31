//! napi-compatible type definitions.
//!
//! These types mirror the core types but are designed for napi-rs FFI.

use napi_derive::napi;
use tacet_core::constants::{
    DEFAULT_FAIL_THRESHOLD, DEFAULT_MAX_SAMPLES, DEFAULT_PASS_THRESHOLD, DEFAULT_TIME_BUDGET_SECS,
};
use tacet_core::result::{Exploitability as CoreExploitability, MeasurementQuality as CoreMeasurementQuality};
use tacet_core::types::AttackerModel as CoreAttackerModel;

/// Attacker model determines the minimum effect threshold (theta) for leak detection.
///
/// Cycle-based thresholds use a 5 GHz reference frequency (conservative).
#[napi]
#[derive(Debug, PartialEq, Default)]
pub enum AttackerModel {
    /// theta = 0.4 ns (~2 cycles @ 5 GHz) - SGX, cross-VM, containers
    SharedHardware,
    /// theta = 2.0 ns (~10 cycles @ 5 GHz) - Post-quantum crypto
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
#[napi]
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// No timing leak detected within threshold theta.
    Pass,
    /// Timing leak detected exceeding threshold theta.
    Fail,
    /// Could not reach a decision.
    Inconclusive,
    /// Operation too fast to measure reliably.
    Unmeasurable,
}

/// Reason for inconclusive result.
#[napi]
#[derive(Debug, PartialEq, Eq)]
pub enum InconclusiveReason {
    /// Not applicable (outcome is not Inconclusive).
    None,
    /// Posterior approximately equals prior after calibration.
    DataTooNoisy,
    /// Posterior stopped updating despite new data.
    NotLearning,
    /// Estimated time to decision exceeds budget.
    WouldTakeTooLong,
    /// Time budget exhausted.
    TimeBudgetExceeded,
    /// Sample limit reached.
    SampleBudgetExceeded,
    /// Measurement conditions changed during test.
    ConditionsChanged,
    /// Threshold was elevated due to measurement noise.
    ThresholdElevated,
}


/// Exploitability assessment.
#[napi]
#[derive(Debug, PartialEq, Eq)]
pub enum Exploitability {
    /// < 10 ns - Requires shared hardware to exploit.
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
#[napi]
#[derive(Debug, PartialEq, Eq)]
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

/// Effect size estimate.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct EffectEstimate {
    /// Maximum effect in nanoseconds: max_k |delta_k|.
    pub max_effect_ns: f64,
    /// Lower bound of 95% credible interval for max effect.
    pub ci_low_ns: f64,
    /// Upper bound of 95% credible interval for max effect.
    pub ci_high_ns: f64,
}

impl EffectEstimate {
    /// Get the maximum effect magnitude.
    pub fn total_effect_ns(&self) -> f64 {
        self.max_effect_ns
    }
}

/// Diagnostics information for debugging and quality assessment.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Diagnostics {
    /// Block length used for bootstrap resampling.
    pub dependence_length: u32,
    /// Effective sample size accounting for autocorrelation.
    pub effective_sample_size: u32,
    /// Ratio of post-test variance to calibration variance.
    pub stationarity_ratio: f64,
    /// Whether stationarity check passed.
    pub stationarity_ok: bool,
    /// Whether discrete mode was used (low timer resolution).
    pub discrete_mode: bool,
    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,
    /// Posterior mean of latent scale lambda.
    pub lambda_mean: f64,
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

impl Default for Diagnostics {
    fn default() -> Self {
        Self {
            dependence_length: 0,
            effective_sample_size: 0,
            stationarity_ratio: 1.0,
            stationarity_ok: true,
            discrete_mode: false,
            timer_resolution_ns: 0.0,
            lambda_mean: 1.0,
            lambda_mixing_ok: true,
            kappa_mean: 1.0,
            kappa_cv: 0.0,
            kappa_ess: 0.0,
            kappa_mixing_ok: true,
        }
    }
}

/// Configuration for the timing analysis.
#[napi(object)]
#[derive(Debug, Clone)]
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
#[napi(object)]
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Test outcome.
    pub outcome: Outcome,
    /// Leak probability: P(max_k |delta_k| > theta | data).
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
    /// Minimum detectable effect in nanoseconds.
    pub mde_ns: f64,
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
#[napi(object)]
#[derive(Debug, Clone)]
pub struct TimerInfo {
    /// Cycles per nanosecond.
    pub cycles_per_ns: f64,
    /// Timer resolution in nanoseconds.
    pub resolution_ns: f64,
    /// Timer frequency in Hz.
    pub frequency_hz: f64,
}
