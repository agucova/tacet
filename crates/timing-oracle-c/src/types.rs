//! FFI-safe type definitions for timing-oracle-c
//!
//! All types in this module are #[repr(C)] for ABI compatibility.

use std::ffi::c_char;

/// Attacker model determines the minimum effect threshold (θ) for leak detection.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToAttackerModel {
    /// θ = 0.6 ns (~2 cycles @ 3GHz) - SGX, cross-VM, containers
    SharedHardware,
    /// θ = 3.3 ns (~10 cycles) - Post-quantum crypto (KyberSlash-class)
    PostQuantum,
    /// θ = 100 ns - LAN, HTTP/2 (Timeless Timing Attacks)
    #[default]
    AdjacentNetwork,
    /// θ = 50 μs - General internet
    RemoteNetwork,
    /// θ → 0 - Detect any difference (not for CI)
    Research,
    /// User-specified threshold
    Custom,
}

impl ToAttackerModel {
    /// Convert to threshold in nanoseconds.
    pub fn to_threshold_ns(&self, custom_threshold: f64) -> f64 {
        match self {
            ToAttackerModel::SharedHardware => 0.6,
            ToAttackerModel::PostQuantum => 3.3,
            ToAttackerModel::AdjacentNetwork => 100.0,
            ToAttackerModel::RemoteNetwork => 50_000.0,
            ToAttackerModel::Research => 0.0,
            ToAttackerModel::Custom => custom_threshold,
        }
    }
}

/// Configuration for the timing test.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ToConfig {
    /// Attacker model to use.
    pub attacker_model: ToAttackerModel,
    /// Custom threshold in nanoseconds (only used if attacker_model == Custom).
    pub custom_threshold_ns: f64,
    /// Maximum samples per class. 0 = default (100,000).
    pub max_samples: usize,
    /// Time budget in seconds. 0 = default (30s).
    pub time_budget_secs: f64,
    /// Pass threshold for leak probability. Default: 0.05.
    pub pass_threshold: f64,
    /// Fail threshold for leak probability. Default: 0.95.
    pub fail_threshold: f64,
    /// Random seed. 0 = system entropy.
    pub seed: u64,
}

impl Default for ToConfig {
    fn default() -> Self {
        Self {
            attacker_model: ToAttackerModel::default(),
            custom_threshold_ns: 0.0,
            max_samples: 0,
            time_budget_secs: 0.0,
            pass_threshold: 0.05,
            fail_threshold: 0.95,
            seed: 0,
        }
    }
}

/// Test outcome.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToOutcome {
    /// No timing leak detected within threshold θ.
    Pass,
    /// Timing leak detected exceeding threshold θ.
    Fail,
    /// Could not reach a decision (quality gate triggered).
    Inconclusive,
    /// Operation too fast to measure reliably.
    Unmeasurable,
}

/// Reason for inconclusive result.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToInconclusiveReason {
    /// Posterior ≈ prior after calibration; data not informative.
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
}

/// Pattern of timing effect.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToEffectPattern {
    /// Uniform shift across all quantiles.
    UniformShift,
    /// Effect concentrated in tails (upper quantiles).
    TailEffect,
    /// Both shift and tail components present.
    Mixed,
    /// Cannot determine pattern.
    Indeterminate,
}

/// Exploitability assessment (for Fail outcomes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToExploitability {
    /// < 100 ns - Negligible practical impact.
    Negligible,
    /// 100-500 ns - Possible on LAN with many measurements.
    PossibleLan,
    /// 500 ns - 20 μs - Likely exploitable on LAN.
    LikelyLan,
    /// > 20 μs - Potentially exploitable remotely.
    PossibleRemote,
}

/// Measurement quality assessment.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToQuality {
    /// MDE < 5 ns.
    Excellent,
    /// MDE 5-20 ns.
    Good,
    /// MDE 20-100 ns.
    Poor,
    /// MDE > 100 ns.
    TooNoisy,
}

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
    /// Pattern of the effect.
    pub pattern: ToEffectPattern,
}

/// Test result.
#[repr(C)]
pub struct ToResult {
    /// Test outcome.
    pub outcome: ToOutcome,

    /// Leak probability: P(max_k |(Xβ)_k| > θ | data).
    pub leak_probability: f64,

    /// Effect size estimate.
    pub effect: ToEffect,

    /// Measurement quality.
    pub quality: ToQuality,

    /// Number of samples used per class.
    pub samples_used: usize,

    /// Time spent in seconds.
    pub elapsed_secs: f64,

    /// Exploitability (only valid if outcome == Fail).
    pub exploitability: ToExploitability,

    /// Inconclusive reason (only valid if outcome == Inconclusive).
    pub inconclusive_reason: ToInconclusiveReason,

    /// Measured operation time in ns (only valid if outcome == Unmeasurable).
    pub operation_ns: f64,

    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,

    /// Recommendation string (owned, must be freed; null if not applicable).
    pub recommendation: *const c_char,

    /// Timer name (static string, do not free).
    pub timer_name: *const c_char,

    /// Platform description (static string, do not free).
    pub platform: *const c_char,

    /// Whether adaptive batching was used.
    pub adaptive_batching_used: bool,
}

impl Default for ToResult {
    fn default() -> Self {
        Self {
            outcome: ToOutcome::Pass,
            leak_probability: 0.0,
            effect: ToEffect {
                shift_ns: 0.0,
                tail_ns: 0.0,
                ci_low_ns: 0.0,
                ci_high_ns: 0.0,
                pattern: ToEffectPattern::Indeterminate,
            },
            quality: ToQuality::Excellent,
            samples_used: 0,
            elapsed_secs: 0.0,
            exploitability: ToExploitability::Negligible,
            inconclusive_reason: ToInconclusiveReason::DataTooNoisy,
            operation_ns: 0.0,
            timer_resolution_ns: 0.0,
            recommendation: std::ptr::null(),
            timer_name: std::ptr::null(),
            platform: std::ptr::null(),
            adaptive_batching_used: false,
        }
    }
}
