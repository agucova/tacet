//! FFI-safe type definitions for timing-oracle-c
//!
//! All types in this module are #[repr(C)] for ABI compatibility.

use core::ffi::c_char;
use core::ptr;

/// Maximum number of warnings that can be stored in diagnostics.
pub const TO_MAX_WARNINGS: usize = 8;

/// Maximum length of a warning message.
pub const TO_MAX_WARNING_LEN: usize = 128;

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

/// Timer preference for measurements.
///
/// Controls which timer implementation to use:
/// - Auto: Try PMU first (perf/kperf), fall back to standard (rdtsc/cntvct)
/// - Standard: Always use standard timer
/// - PreferPmu: Use PMU timer or fail if unavailable
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToTimerPref {
    /// Try PMU first, fall back to standard timer.
    #[default]
    Auto,
    /// Always use standard timer (rdtsc/cntvct_el0).
    Standard,
    /// Require PMU timer, fail if unavailable.
    PreferPmu,
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
    /// Timer preference. Default: Auto.
    pub timer_preference: ToTimerPref,
    /// Calibration samples per class. 0 = default (5,000).
    pub calibration_samples: usize,
    /// Batch size for adaptive sampling. 0 = default (1,000).
    pub batch_size: usize,
    /// Bootstrap iterations for covariance estimation. 0 = default (2,000).
    pub bootstrap_iterations: usize,
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
            timer_preference: ToTimerPref::default(),
            calibration_samples: 0,
            batch_size: 0,
            bootstrap_iterations: 0,
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
    /// Research mode result (spec §3.6) - returned when attacker_model is Research.
    Research,
}

/// Status of a research mode run (spec §3.6).
///
/// Research mode (AttackerModel::Research) doesn't make Pass/Fail decisions.
/// Instead, it characterizes the timing behavior with respect to the measurement floor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToResearchStatus {
    /// CI clearly above θ_floor — timing difference detected.
    EffectDetected,
    /// CI clearly below θ_floor — no timing difference above noise.
    NoEffectDetected,
    /// Hit timer resolution limit; θ_floor is as good as it gets.
    ResolutionLimitReached,
    /// Data quality issue detected (see inconclusive_reason for details).
    QualityIssue,
    /// Ran out of time/samples before reaching conclusion.
    BudgetExhausted,
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
    /// Threshold was elevated and pass criterion was met at effective threshold (v5.5).
    ThresholdElevated,
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
    /// < 10 ns - Requires shared hardware (SGX, containers) to exploit.
    SharedHardwareOnly,
    /// 10-100 ns - Exploitable via HTTP/2 request multiplexing.
    Http2Multiplexing,
    /// 100 ns - 10 μs - Exploitable with standard remote timing.
    StandardRemote,
    /// > 10 μs - Obvious leak, trivially exploitable.
    ObviousLeak,
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
    /// Whether there is a projection mismatch caveat.
    /// When true, the 2D effect summary may not fully capture the timing difference.
    pub has_interpretation_caveat: bool,
    /// Top quantile indices that contribute most to mismatch (0-8 for deciles 1-9).
    /// Only valid when has_interpretation_caveat is true.
    /// Set to 255 for unused slots.
    pub top_quantiles: [u8; 3],
}

/// Diagnostics information for the test result.
///
/// Contains detailed information about the measurement process for debugging
/// and quality assessment (spec §2.8).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ToDiagnostics {
    /// Block length used for bootstrap resampling.
    pub dependence_length: usize,

    /// Effective sample size accounting for autocorrelation.
    pub effective_sample_size: usize,

    /// Ratio of post-test variance to calibration variance.
    /// Values outside [0.5, 2.0] indicate drift.
    pub stationarity_ratio: f64,

    /// Whether stationarity check passed.
    pub stationarity_ok: bool,

    /// Projection mismatch Q statistic.
    /// High values indicate 2D summary is approximate.
    pub projection_mismatch_q: f64,

    /// Threshold for projection mismatch (calibrated at 99th percentile).
    pub projection_mismatch_threshold: f64,

    /// Whether projection mismatch is acceptable.
    pub projection_mismatch_ok: bool,

    /// Outlier rate for baseline class (fraction filtered).
    pub outlier_rate_baseline: f64,

    /// Outlier rate for sample class (fraction filtered).
    pub outlier_rate_sample: f64,

    /// Whether outlier rates are balanced between classes.
    pub outlier_asymmetry_ok: bool,

    /// Whether discrete mode was used (low timer resolution).
    pub discrete_mode: bool,

    /// Fraction of duplicate timing values.
    pub duplicate_fraction: f64,

    /// Whether preflight checks passed.
    pub preflight_ok: bool,

    /// Number of samples used in calibration phase.
    pub calibration_samples: usize,

    /// Total time spent in seconds.
    pub total_time_secs: f64,

    /// RNG seed used for bootstrap (0 if random).
    pub seed: u64,

    /// User-specified threshold (theta_user) in nanoseconds.
    pub theta_user: f64,

    /// Effective threshold (theta_eff) in nanoseconds.
    /// May be higher than theta_user due to measurement floor.
    pub theta_eff: f64,

    /// Measurement floor (theta_floor) in nanoseconds.
    /// Minimum detectable effect given current noise.
    pub theta_floor: f64,

    /// Number of warnings.
    pub warning_count: usize,

    /// Warning messages (null-terminated, fixed-size buffers).
    pub warnings: [[c_char; TO_MAX_WARNING_LEN]; TO_MAX_WARNINGS],

    // =========================================================================
    // v5.4 Gibbs sampler λ (lambda) diagnostics
    // =========================================================================

    /// Total number of Gibbs iterations.
    pub gibbs_iters_total: usize,

    /// Number of burn-in iterations.
    pub gibbs_burnin: usize,

    /// Number of retained samples.
    pub gibbs_retained: usize,

    /// Posterior mean of latent scale λ.
    pub lambda_mean: f64,

    /// Posterior standard deviation of λ.
    pub lambda_sd: f64,

    /// Coefficient of variation of λ (λ_sd / λ_mean).
    pub lambda_cv: f64,

    /// Effective sample size of λ chain.
    pub lambda_ess: f64,

    /// Whether λ chain mixed well (CV ≥ 0.1 AND ESS ≥ 20).
    pub lambda_mixing_ok: bool,

    // =========================================================================
    // v5.6 Gibbs sampler κ (kappa) diagnostics
    // =========================================================================

    /// Posterior mean of likelihood precision κ.
    pub kappa_mean: f64,

    /// Posterior standard deviation of κ.
    pub kappa_sd: f64,

    /// Coefficient of variation of κ (kappa_sd / kappa_mean).
    pub kappa_cv: f64,

    /// Effective sample size of κ chain.
    pub kappa_ess: f64,

    /// Whether κ chain mixed well (CV ≥ 0.1 AND ESS ≥ 20).
    pub kappa_mixing_ok: bool,
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

    /// Detailed diagnostics (spec §2.8).
    pub diagnostics: ToDiagnostics,

    /// Threshold at which decision was made (θ_eff at decision time).
    /// Only valid for Pass/Fail outcomes.
    pub decision_threshold_ns: f64,

    // Research mode fields (only valid if outcome == Research)
    /// Research mode status (only valid if outcome == Research).
    pub research_status: ToResearchStatus,

    /// Maximum effect across quantiles in nanoseconds (only valid if outcome == Research).
    pub max_effect_ns: f64,

    /// 95% CI lower bound for maximum effect (only valid if outcome == Research).
    pub max_effect_ci_low: f64,

    /// 95% CI upper bound for maximum effect (only valid if outcome == Research).
    pub max_effect_ci_high: f64,

    /// Whether the effect is detectable: CI lower bound > theta_floor (only valid if outcome == Research).
    pub research_detectable: bool,

    /// Whether model mismatch was detected in research mode (only valid if outcome == Research).
    pub research_model_mismatch: bool,
}

impl Default for ToDiagnostics {
    fn default() -> Self {
        Self {
            dependence_length: 0,
            effective_sample_size: 0,
            stationarity_ratio: 1.0,
            stationarity_ok: true,
            projection_mismatch_q: 0.0,
            projection_mismatch_threshold: 18.48, // chi-squared(7, 0.99)
            projection_mismatch_ok: true,
            outlier_rate_baseline: 0.0,
            outlier_rate_sample: 0.0,
            outlier_asymmetry_ok: true,
            discrete_mode: false,
            duplicate_fraction: 0.0,
            preflight_ok: true,
            calibration_samples: 0,
            total_time_secs: 0.0,
            seed: 0,
            theta_user: 0.0,
            theta_eff: 0.0,
            theta_floor: 0.0,
            warning_count: 0,
            warnings: [[0; TO_MAX_WARNING_LEN]; TO_MAX_WARNINGS],
            // v5.4 Gibbs sampler λ diagnostics
            gibbs_iters_total: 256,
            gibbs_burnin: 64,
            gibbs_retained: 192,
            lambda_mean: 1.0,
            lambda_sd: 0.0,
            lambda_cv: 0.0,
            lambda_ess: 0.0,
            lambda_mixing_ok: true,
            // v5.6 Gibbs sampler κ diagnostics
            kappa_mean: 1.0,
            kappa_sd: 0.0,
            kappa_cv: 0.0,
            kappa_ess: 0.0,
            kappa_mixing_ok: true,
        }
    }
}

impl Default for ToEffect {
    fn default() -> Self {
        Self {
            shift_ns: 0.0,
            tail_ns: 0.0,
            ci_low_ns: 0.0,
            ci_high_ns: 0.0,
            pattern: ToEffectPattern::Indeterminate,
            has_interpretation_caveat: false,
            top_quantiles: [255, 255, 255],
        }
    }
}

impl Default for ToResult {
    fn default() -> Self {
        Self {
            outcome: ToOutcome::Pass,
            leak_probability: 0.0,
            effect: ToEffect::default(),
            quality: ToQuality::Excellent,
            samples_used: 0,
            elapsed_secs: 0.0,
            exploitability: ToExploitability::SharedHardwareOnly,
            inconclusive_reason: ToInconclusiveReason::DataTooNoisy,
            operation_ns: 0.0,
            timer_resolution_ns: 0.0,
            recommendation: ptr::null(),
            timer_name: ptr::null(),
            platform: ptr::null(),
            adaptive_batching_used: false,
            diagnostics: ToDiagnostics::default(),
            decision_threshold_ns: 0.0,
            // Research mode defaults
            research_status: ToResearchStatus::BudgetExhausted,
            max_effect_ns: 0.0,
            max_effect_ci_low: 0.0,
            max_effect_ci_high: 0.0,
            research_detectable: false,
            research_model_mismatch: false,
        }
    }
}

// =============================================================================
// Conversions from timing-oracle-core types to FFI types
// =============================================================================

impl From<timing_oracle_core::result::EffectPattern> for ToEffectPattern {
    fn from(pattern: timing_oracle_core::result::EffectPattern) -> Self {
        use timing_oracle_core::result::EffectPattern;
        match pattern {
            EffectPattern::UniformShift => ToEffectPattern::UniformShift,
            EffectPattern::TailEffect => ToEffectPattern::TailEffect,
            EffectPattern::Mixed => ToEffectPattern::Mixed,
            EffectPattern::Indeterminate => ToEffectPattern::Indeterminate,
        }
    }
}

impl From<timing_oracle_core::result::MeasurementQuality> for ToQuality {
    fn from(quality: timing_oracle_core::result::MeasurementQuality) -> Self {
        use timing_oracle_core::result::MeasurementQuality;
        match quality {
            MeasurementQuality::Excellent => ToQuality::Excellent,
            MeasurementQuality::Good => ToQuality::Good,
            MeasurementQuality::Poor => ToQuality::Poor,
            MeasurementQuality::TooNoisy => ToQuality::TooNoisy,
        }
    }
}

impl From<timing_oracle_core::result::EffectEstimate> for ToEffect {
    fn from(estimate: timing_oracle_core::result::EffectEstimate) -> Self {
        ToEffect {
            shift_ns: estimate.shift_ns,
            tail_ns: estimate.tail_ns,
            ci_low_ns: estimate.credible_interval_ns.0,
            ci_high_ns: estimate.credible_interval_ns.1,
            pattern: estimate.pattern.into(),
            has_interpretation_caveat: estimate.interpretation_caveat.is_some(),
            top_quantiles: [255, 255, 255], // Will be populated by caller if needed
        }
    }
}
