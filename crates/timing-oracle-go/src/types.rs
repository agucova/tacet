//! FFI-safe type definitions for timing-oracle-go.
//!
//! All types in this module are #[repr(C)] for ABI compatibility with Go via CGo.

use core::ffi::c_char;
use core::ptr;

/// Attacker model determines the minimum effect threshold (theta) for leak detection.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToGoAttackerModel {
    /// theta = 0.6 ns (~2 cycles @ 3GHz) - SGX, cross-VM, containers
    SharedHardware = 0,
    /// theta = 3.3 ns (~10 cycles) - Post-quantum crypto (KyberSlash-class)
    PostQuantum = 1,
    /// theta = 100 ns - LAN, HTTP/2 (Timeless Timing Attacks)
    #[default]
    AdjacentNetwork = 2,
    /// theta = 50 us - General internet
    RemoteNetwork = 3,
    /// theta -> 0 - Detect any difference (not for CI)
    Research = 4,
    /// User-specified threshold via custom_threshold_ns
    Custom = 5,
}

impl ToGoAttackerModel {
    /// Convert to threshold in nanoseconds.
    pub fn to_threshold_ns(&self, custom_threshold: f64) -> f64 {
        match self {
            ToGoAttackerModel::SharedHardware => 0.6,
            ToGoAttackerModel::PostQuantum => 3.3,
            ToGoAttackerModel::AdjacentNetwork => 100.0,
            ToGoAttackerModel::RemoteNetwork => 50_000.0,
            ToGoAttackerModel::Research => 0.0,
            ToGoAttackerModel::Custom => custom_threshold,
        }
    }
}

/// Configuration for the timing analysis.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ToGoConfig {
    /// Attacker model to use (determines threshold theta).
    pub attacker_model: ToGoAttackerModel,
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
    /// Random seed. 0 = use system entropy.
    pub seed: u64,
    /// Timer frequency in Hz (for converting ticks to nanoseconds).
    pub timer_frequency_hz: u64,
}

impl Default for ToGoConfig {
    fn default() -> Self {
        Self {
            attacker_model: ToGoAttackerModel::default(),
            custom_threshold_ns: 0.0,
            max_samples: 100_000,
            time_budget_secs: 30.0,
            pass_threshold: 0.05,
            fail_threshold: 0.95,
            seed: 0,
            timer_frequency_hz: 1_000_000_000, // 1 GHz default (1 tick = 1ns)
        }
    }
}

impl ToGoConfig {
    /// Get the effective theta threshold in nanoseconds.
    pub fn theta_ns(&self) -> f64 {
        self.attacker_model
            .to_threshold_ns(self.custom_threshold_ns)
    }

    /// Get nanoseconds per tick based on timer frequency.
    pub fn ns_per_tick(&self) -> f64 {
        if self.timer_frequency_hz == 0 {
            1.0 // Default to 1ns per tick
        } else {
            1_000_000_000.0 / (self.timer_frequency_hz as f64)
        }
    }
}

/// Test outcome.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToGoOutcome {
    /// No timing leak detected within threshold theta.
    Pass = 0,
    /// Timing leak detected exceeding threshold theta.
    Fail = 1,
    /// Could not reach a decision (quality gate triggered).
    Inconclusive = 2,
    /// Operation too fast to measure reliably.
    Unmeasurable = 3,
}

/// Reason for inconclusive result.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToGoInconclusiveReason {
    /// No inconclusive reason (not applicable).
    None = 0,
    /// Posterior approx equals prior after calibration; data not informative.
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
    /// Requested threshold is below measurement floor.
    ThresholdUnachievable = 7,
    /// Model doesn't fit the data - 2D model insufficient.
    ModelMismatch = 8,
}

/// Pattern of timing effect.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToGoEffectPattern {
    /// Uniform shift across all quantiles.
    UniformShift = 0,
    /// Effect concentrated in tails (upper quantiles).
    TailEffect = 1,
    /// Both shift and tail components present.
    Mixed = 2,
    /// Cannot determine pattern.
    Indeterminate = 3,
}

/// Exploitability assessment (for Fail outcomes).
///
/// Based on Timeless Timing Attacks (Van Goethem et al., 2020) research.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToGoExploitability {
    /// < 10 ns - Requires shared hardware (SGX, containers) to exploit.
    SharedHardwareOnly = 0,
    /// 10-100 ns - Exploitable via HTTP/2 request multiplexing.
    Http2Multiplexing = 1,
    /// 100 ns - 10 us - Exploitable with standard remote timing.
    StandardRemote = 2,
    /// > 10 us - Obvious leak, trivially exploitable.
    ObviousLeak = 3,
}

/// Measurement quality assessment.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToGoQuality {
    /// MDE < 5 ns - Excellent measurement precision.
    Excellent = 0,
    /// MDE 5-20 ns - Good precision.
    Good = 1,
    /// MDE 20-100 ns - Poor precision.
    Poor = 2,
    /// MDE > 100 ns - Too noisy for reliable detection.
    TooNoisy = 3,
}

/// Effect size estimate.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ToGoEffect {
    /// Uniform shift component in nanoseconds.
    pub shift_ns: f64,
    /// Tail effect component in nanoseconds.
    pub tail_ns: f64,
    /// 95% credible interval lower bound.
    pub ci_low_ns: f64,
    /// 95% credible interval upper bound.
    pub ci_high_ns: f64,
    /// Pattern of the effect.
    pub pattern: ToGoEffectPattern,
}

impl Default for ToGoEffect {
    fn default() -> Self {
        Self {
            shift_ns: 0.0,
            tail_ns: 0.0,
            ci_low_ns: 0.0,
            ci_high_ns: 0.0,
            pattern: ToGoEffectPattern::Indeterminate,
        }
    }
}

/// Analysis result.
#[repr(C)]
pub struct ToGoResult {
    /// Test outcome.
    pub outcome: ToGoOutcome,

    /// Leak probability: P(max_k |(X*beta)_k| > theta | data).
    pub leak_probability: f64,

    /// Effect size estimate.
    pub effect: ToGoEffect,

    /// Measurement quality.
    pub quality: ToGoQuality,

    /// Number of samples used per class.
    pub samples_used: usize,

    /// Time spent in seconds.
    pub elapsed_secs: f64,

    /// Exploitability (only valid if outcome == Fail).
    pub exploitability: ToGoExploitability,

    /// Inconclusive reason (only valid if outcome == Inconclusive).
    pub inconclusive_reason: ToGoInconclusiveReason,

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

    /// Recommendation string (owned, must be freed via togo_result_free; NULL if not applicable).
    pub recommendation: *const c_char,
}

impl Default for ToGoResult {
    fn default() -> Self {
        Self {
            outcome: ToGoOutcome::Pass,
            leak_probability: 0.0,
            effect: ToGoEffect::default(),
            quality: ToGoQuality::Excellent,
            samples_used: 0,
            elapsed_secs: 0.0,
            exploitability: ToGoExploitability::SharedHardwareOnly,
            inconclusive_reason: ToGoInconclusiveReason::None,
            mde_shift_ns: 0.0,
            mde_tail_ns: 0.0,
            timer_resolution_ns: 0.0,
            theta_user_ns: 0.0,
            theta_eff_ns: 0.0,
            recommendation: ptr::null(),
        }
    }
}

/// Opaque calibration state handle.
///
/// Created by togo_calibrate(), freed by togo_calibration_free().
#[repr(C)]
pub struct ToGoCalibration {
    /// Opaque pointer to internal Calibration struct.
    pub ptr: *mut core::ffi::c_void,
}

impl Default for ToGoCalibration {
    fn default() -> Self {
        Self {
            ptr: ptr::null_mut(),
        }
    }
}

/// Adaptive sampling state.
///
/// Tracks cumulative samples and posterior evolution between adaptive steps.
#[repr(C)]
pub struct ToGoAdaptiveState {
    /// Total baseline samples collected.
    pub total_baseline: usize,
    /// Total sample class samples collected.
    pub total_sample: usize,
    /// Current leak probability estimate.
    pub current_probability: f64,
    /// Opaque pointer to internal AdaptiveState struct.
    pub ptr: *mut core::ffi::c_void,
}

impl Default for ToGoAdaptiveState {
    fn default() -> Self {
        Self {
            total_baseline: 0,
            total_sample: 0,
            current_probability: 0.5,
            ptr: ptr::null_mut(),
        }
    }
}

// =============================================================================
// Conversions from timing-oracle-core types to FFI types
// =============================================================================

impl From<timing_oracle_core::result::EffectPattern> for ToGoEffectPattern {
    fn from(pattern: timing_oracle_core::result::EffectPattern) -> Self {
        use timing_oracle_core::result::EffectPattern;
        match pattern {
            EffectPattern::UniformShift => ToGoEffectPattern::UniformShift,
            EffectPattern::TailEffect => ToGoEffectPattern::TailEffect,
            EffectPattern::Mixed => ToGoEffectPattern::Mixed,
            EffectPattern::Indeterminate => ToGoEffectPattern::Indeterminate,
        }
    }
}

impl From<timing_oracle_core::result::MeasurementQuality> for ToGoQuality {
    fn from(quality: timing_oracle_core::result::MeasurementQuality) -> Self {
        use timing_oracle_core::result::MeasurementQuality;
        match quality {
            MeasurementQuality::Excellent => ToGoQuality::Excellent,
            MeasurementQuality::Good => ToGoQuality::Good,
            MeasurementQuality::Poor => ToGoQuality::Poor,
            MeasurementQuality::TooNoisy => ToGoQuality::TooNoisy,
        }
    }
}

impl From<timing_oracle_core::result::Exploitability> for ToGoExploitability {
    fn from(exploit: timing_oracle_core::result::Exploitability) -> Self {
        use timing_oracle_core::result::Exploitability;
        match exploit {
            Exploitability::SharedHardwareOnly => ToGoExploitability::SharedHardwareOnly,
            Exploitability::Http2Multiplexing => ToGoExploitability::Http2Multiplexing,
            Exploitability::StandardRemote => ToGoExploitability::StandardRemote,
            Exploitability::ObviousLeak => ToGoExploitability::ObviousLeak,
        }
    }
}

/// Compute exploitability from effect magnitude.
///
/// Thresholds based on Timeless Timing Attacks (Van Goethem et al., 2020):
/// - < 10 ns: SharedHardwareOnly (SGX, containers, shared cache)
/// - 10-100 ns: Http2Multiplexing (concurrent request timing)
/// - 100 ns - 10 μs: StandardRemote (traditional remote timing)
/// - > 10 μs: ObviousLeak (trivially exploitable)
pub fn exploitability_from_effect_ns(effect_ns: f64) -> ToGoExploitability {
    let effect_ns = effect_ns.abs();
    if effect_ns < 10.0 {
        ToGoExploitability::SharedHardwareOnly
    } else if effect_ns < 100.0 {
        ToGoExploitability::Http2Multiplexing
    } else if effect_ns < 10_000.0 {
        ToGoExploitability::StandardRemote
    } else {
        ToGoExploitability::ObviousLeak
    }
}

/// Compute quality from MDE.
pub fn quality_from_mde_ns(mde_ns: f64) -> ToGoQuality {
    if mde_ns < 5.0 {
        ToGoQuality::Excellent
    } else if mde_ns < 20.0 {
        ToGoQuality::Good
    } else if mde_ns < 100.0 {
        ToGoQuality::Poor
    } else {
        ToGoQuality::TooNoisy
    }
}
