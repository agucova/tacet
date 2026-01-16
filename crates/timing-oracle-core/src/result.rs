//! Result types for adaptive Bayesian timing analysis.
//!
//! See spec Section 4.1 (Result Types) for the full specification.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use serde::{Deserialize, Serialize};

// ============================================================================
// Outcome - The top-level result type
// ============================================================================

/// Top-level outcome of a timing test.
///
/// The adaptive Bayesian oracle returns one of four outcomes:
/// - `Pass`: No timing leak detected (leak_probability < pass_threshold)
/// - `Fail`: Timing leak confirmed (leak_probability > fail_threshold)
/// - `Inconclusive`: Cannot reach a definitive conclusion
/// - `Unmeasurable`: Operation too fast to measure on this platform
///
/// See spec Section 4.1 (Result Types).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum Outcome {
    /// No timing leak detected.
    ///
    /// The posterior probability of a timing leak is below the pass threshold
    /// (default 0.05), meaning we're confident there is no exploitable leak.
    Pass {
        /// Posterior probability of timing leak: P(effect > theta | data).
        /// Will be < pass_threshold (default 0.05).
        leak_probability: f64,

        /// Effect size estimate (shift and tail components).
        effect: EffectEstimate,

        /// Number of samples used in the analysis.
        samples_used: usize,

        /// Measurement quality assessment.
        quality: MeasurementQuality,

        /// Diagnostic information for debugging.
        diagnostics: Diagnostics,
    },

    /// Timing leak confirmed.
    ///
    /// The posterior probability of a timing leak exceeds the fail threshold
    /// (default 0.95), meaning we're confident there is an exploitable leak.
    Fail {
        /// Posterior probability of timing leak: P(effect > theta | data).
        /// Will be > fail_threshold (default 0.95).
        leak_probability: f64,

        /// Effect size estimate (shift and tail components).
        effect: EffectEstimate,

        /// Exploitability assessment based on effect magnitude.
        exploitability: Exploitability,

        /// Number of samples used in the analysis.
        samples_used: usize,

        /// Measurement quality assessment.
        quality: MeasurementQuality,

        /// Diagnostic information for debugging.
        diagnostics: Diagnostics,
    },

    /// Cannot reach a definitive conclusion.
    ///
    /// The posterior probability is between pass_threshold and fail_threshold,
    /// or the analysis hit a limit (timeout, sample budget, noise).
    Inconclusive {
        /// Reason why the result is inconclusive.
        reason: InconclusiveReason,

        /// Current posterior probability of timing leak.
        leak_probability: f64,

        /// Effect size estimate (may have wide credible intervals).
        effect: EffectEstimate,

        /// Number of samples used in the analysis.
        samples_used: usize,

        /// Measurement quality assessment.
        quality: MeasurementQuality,

        /// Diagnostic information for debugging.
        diagnostics: Diagnostics,
    },

    /// Operation too fast to measure reliably on this platform.
    ///
    /// The operation completes faster than the timer's resolution allows
    /// for meaningful measurement, even with adaptive batching.
    Unmeasurable {
        /// Estimated operation duration in nanoseconds.
        operation_ns: f64,

        /// Minimum measurable duration on this platform.
        threshold_ns: f64,

        /// Platform description (e.g., "Apple Silicon (cntvct)").
        platform: String,

        /// Suggested actions to make the operation measurable.
        recommendation: String,
    },
}

// ============================================================================
// InconclusiveReason - Why we couldn't reach a conclusion
// ============================================================================

/// Reason why a timing test result is inconclusive.
///
/// See spec Section 4.1 (Result Types).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InconclusiveReason {
    /// Data is too noisy to reach a conclusion.
    ///
    /// The measurement noise is high enough that we cannot distinguish
    /// between "no leak" and "small leak" with the available samples.
    DataTooNoisy {
        /// Human-readable explanation.
        message: String,
        /// Suggested actions to improve measurement quality.
        guidance: String,
    },

    /// Posterior is not converging toward either threshold.
    ///
    /// After collecting samples, the leak probability remains in the
    /// inconclusive range and isn't trending toward pass or fail.
    NotLearning {
        /// Human-readable explanation.
        message: String,
        /// Suggested actions.
        guidance: String,
    },

    /// Reaching a conclusion would take too long.
    ///
    /// Based on current convergence rate, reaching the pass or fail
    /// threshold would exceed the configured time budget.
    WouldTakeTooLong {
        /// Estimated time in seconds to reach a conclusion.
        estimated_time_secs: f64,
        /// Estimated samples needed to reach a conclusion.
        samples_needed: usize,
        /// Suggested actions.
        guidance: String,
    },

    /// Time budget exhausted.
    ///
    /// The configured time limit was reached before the posterior
    /// converged to a conclusive result.
    Timeout {
        /// Posterior probability at timeout.
        current_probability: f64,
        /// Number of samples collected before timeout.
        samples_collected: usize,
    },

    /// Sample budget exhausted.
    ///
    /// The maximum number of samples was collected without reaching
    /// a conclusive result.
    SampleBudgetExceeded {
        /// Posterior probability when budget was exhausted.
        current_probability: f64,
        /// Number of samples collected.
        samples_collected: usize,
    },
}

// ============================================================================
// EffectEstimate - Decomposed timing effect
// ============================================================================

/// Estimated timing effect decomposed into shift and tail components.
///
/// The effect is decomposed using a 2-component linear model:
/// - **Shift**: Uniform timing difference across all quantiles (e.g., different code path)
/// - **Tail**: Upper quantiles shift more than lower (e.g., cache misses)
///
/// See spec Section 2.5 (Bayesian Inference).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectEstimate {
    /// Uniform shift in nanoseconds.
    ///
    /// Positive value means the baseline (fixed) class is slower.
    /// This captures effects like branch timing where all operations
    /// take a fixed additional time.
    pub shift_ns: f64,

    /// Tail effect in nanoseconds.
    ///
    /// Positive value means the baseline class has a heavier upper tail.
    /// This captures effects like cache misses that occur probabilistically.
    pub tail_ns: f64,

    /// 95% credible interval for the total effect magnitude in nanoseconds.
    ///
    /// This is a Bayesian credible interval, not a frequentist confidence interval.
    /// There is a 95% posterior probability that the true effect lies within this range.
    pub credible_interval_ns: (f64, f64),

    /// Classification of the dominant effect pattern.
    pub pattern: EffectPattern,
}

impl EffectEstimate {
    /// Compute the total effect magnitude (L2 norm of shift and tail).
    #[cfg(feature = "std")]
    pub fn total_effect_ns(&self) -> f64 {
        (self.shift_ns.powi(2) + self.tail_ns.powi(2)).sqrt()
    }

    /// Compute the total effect magnitude (L2 norm of shift and tail).
    #[cfg(not(feature = "std"))]
    pub fn total_effect_ns(&self) -> f64 {
        libm::sqrt(self.shift_ns * self.shift_ns + self.tail_ns * self.tail_ns)
    }

    /// Check if the effect is negligible (both components near zero).
    pub fn is_negligible(&self, threshold_ns: f64) -> bool {
        self.shift_ns.abs() < threshold_ns && self.tail_ns.abs() < threshold_ns
    }
}

impl Default for EffectEstimate {
    fn default() -> Self {
        Self {
            shift_ns: 0.0,
            tail_ns: 0.0,
            credible_interval_ns: (0.0, 0.0),
            pattern: EffectPattern::Indeterminate,
        }
    }
}

// ============================================================================
// EffectPattern - Classification of timing effect type
// ============================================================================

/// Pattern of timing difference.
///
/// Classifies the dominant type of timing difference based on the
/// relative magnitudes of shift and tail components.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum EffectPattern {
    /// Uniform shift across all quantiles.
    ///
    /// All quantiles shift by approximately the same amount.
    /// Typical cause: branch on secret data, different code path.
    UniformShift,

    /// Primarily affects upper tail.
    ///
    /// Upper quantiles (e.g., 80th, 90th percentile) shift more than
    /// lower quantiles. Typical cause: cache misses, memory access patterns.
    TailEffect,

    /// Mixed pattern with both shift and tail components.
    ///
    /// Both uniform shift and tail effect are significant.
    Mixed,

    /// Neither shift nor tail is statistically significant.
    ///
    /// The effect magnitude is below the detection threshold or
    /// uncertainty is too high to classify.
    #[default]
    Indeterminate,
}

// ============================================================================
// Exploitability - Risk assessment
// ============================================================================

/// Exploitability assessment based on effect magnitude.
///
/// Based on Crosby et al. (2009) thresholds for timing attack feasibility.
/// These thresholds are heuristics for risk prioritization, not guarantees.
///
/// See spec Section 5.4 (Exploitability).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Exploitability {
    /// Effect < 100 ns: Would require impractical number of measurements.
    ///
    /// Exploiting this leak over a network would require millions of
    /// measurements, making it impractical for most attackers.
    Negligible,

    /// 100-500 ns: Possible on local network with ~100k measurements.
    ///
    /// An attacker on the same LAN could potentially exploit this
    /// leak with sufficient measurement time.
    PossibleLAN,

    /// 500 ns - 20 us: Likely exploitable on local network.
    ///
    /// This leak is readily exploitable by an attacker on the local
    /// network with moderate effort.
    LikelyLAN,

    /// > 20 us: Possibly exploitable over internet.
    ///
    /// This leak is large enough that it may be exploitable even
    /// across the internet, depending on network conditions.
    PossibleRemote,
}

impl Exploitability {
    /// Determine exploitability from effect size in nanoseconds.
    pub fn from_effect_ns(effect_ns: f64) -> Self {
        let effect_ns = effect_ns.abs();
        if effect_ns < 100.0 {
            Exploitability::Negligible
        } else if effect_ns < 500.0 {
            Exploitability::PossibleLAN
        } else if effect_ns < 20_000.0 {
            Exploitability::LikelyLAN
        } else {
            Exploitability::PossibleRemote
        }
    }
}

// ============================================================================
// MeasurementQuality - Assessment of measurement reliability
// ============================================================================

/// Measurement quality assessment based on noise level.
///
/// Quality is determined primarily by the minimum detectable effect (MDE)
/// relative to the configured threshold.
///
/// See spec Section 5.5 (Quality Assessment).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MeasurementQuality {
    /// Low noise, high confidence (MDE < 5 ns).
    Excellent,

    /// Normal noise levels (MDE 5-20 ns).
    Good,

    /// High noise, results less reliable (MDE 20-100 ns).
    Poor,

    /// Cannot produce meaningful results (MDE > 100 ns).
    TooNoisy,
}

impl MeasurementQuality {
    /// Determine quality from minimum detectable effect.
    ///
    /// Invalid MDE values (less than or equal to 0 or non-finite) indicate a measurement problem
    /// and are classified as `TooNoisy`.
    ///
    /// Very small MDE (< 0.01 ns) also indicates timer resolution issues
    /// where most samples have identical values.
    pub fn from_mde_ns(mde_ns: f64) -> Self {
        // Invalid MDE indicates measurement failure
        if mde_ns <= 0.01 || !mde_ns.is_finite() {
            return MeasurementQuality::TooNoisy;
        }

        if mde_ns < 5.0 {
            MeasurementQuality::Excellent
        } else if mde_ns < 20.0 {
            MeasurementQuality::Good
        } else if mde_ns < 100.0 {
            MeasurementQuality::Poor
        } else {
            MeasurementQuality::TooNoisy
        }
    }
}

// ============================================================================
// Diagnostics - Detailed diagnostic information
// ============================================================================

/// Diagnostic information for debugging and analysis.
///
/// See spec Section 4.1 (Result Types).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostics {
    /// Block size used for bootstrap (Politis-White automatic selection).
    pub dependence_length: usize,

    /// Effective sample size accounting for autocorrelation (ESS approx n / dependence_length).
    pub effective_sample_size: usize,

    /// Non-stationarity: ratio of inference to calibration variance.
    /// Values 0.5-2.0 are normal; >5.0 indicates non-stationarity.
    pub stationarity_ratio: f64,

    /// True if stationarity ratio is within acceptable bounds (0.5-2.0).
    pub stationarity_ok: bool,

    /// Model fit: chi-squared statistic for residuals.
    /// Should be approximately chi-squared with 7 degrees of freedom under correct model.
    pub model_fit_chi2: f64,

    /// True if chi-squared is within acceptable bounds (p > 0.01 under chi-squared with 7 df).
    pub model_fit_ok: bool,

    /// Outlier rate for baseline class (fraction trimmed).
    pub outlier_rate_baseline: f64,

    /// Outlier rate for sample class (fraction trimmed).
    pub outlier_rate_sample: f64,

    /// True if outlier rates are symmetric (both <1%, ratio <3x, diff <2%).
    pub outlier_asymmetry_ok: bool,

    /// Whether discrete timer mode was used (low timer resolution).
    pub discrete_mode: bool,

    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,

    /// Fraction of samples with duplicate timing values (0.0-1.0).
    pub duplicate_fraction: f64,

    /// True if preflight checks passed (sanity, generator, system).
    pub preflight_ok: bool,

    /// Number of samples used for calibration (covariance estimation).
    pub calibration_samples: usize,

    /// Total time spent on the analysis in seconds.
    pub total_time_secs: f64,

    /// Human-readable warnings (empty if all checks pass).
    pub warnings: Vec<String>,

    /// Quality issues detected during measurement.
    pub quality_issues: Vec<QualityIssue>,
}

impl Diagnostics {
    /// Create diagnostics indicating all checks passed.
    ///
    /// Uses placeholder values for numeric fields; prefer constructing
    /// explicitly with actual measured values.
    pub fn all_ok() -> Self {
        Self {
            dependence_length: 1,
            effective_sample_size: 0,
            stationarity_ratio: 1.0,
            stationarity_ok: true,
            model_fit_chi2: 0.0,
            model_fit_ok: true,
            outlier_rate_baseline: 0.0,
            outlier_rate_sample: 0.0,
            outlier_asymmetry_ok: true,
            discrete_mode: false,
            timer_resolution_ns: 1.0,
            duplicate_fraction: 0.0,
            preflight_ok: true,
            calibration_samples: 0,
            total_time_secs: 0.0,
            warnings: Vec::new(),
            quality_issues: Vec::new(),
        }
    }

    /// Check if all diagnostics are OK.
    pub fn all_checks_passed(&self) -> bool {
        self.stationarity_ok && self.model_fit_ok && self.outlier_asymmetry_ok && self.preflight_ok
    }
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self::all_ok()
    }
}

// ============================================================================
// QualityIssue - Specific quality problems
// ============================================================================

/// A specific quality issue detected during measurement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QualityIssue {
    /// Issue code for programmatic handling.
    pub code: IssueCode,

    /// Human-readable description of the issue.
    pub message: String,

    /// Suggested actions to address the issue.
    pub guidance: String,
}

/// Issue codes for programmatic handling of quality problems.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IssueCode {
    /// High autocorrelation reduces effective sample size.
    HighDependence,

    /// Effective sample size is too low for reliable inference.
    LowEffectiveSamples,

    /// Timing distribution appears to drift during measurement.
    StationaritySuspect,

    /// Timer has low resolution, using discrete mode.
    DiscreteTimer,

    /// Sample count is small for discrete mode bootstrap.
    SmallSampleDiscrete,

    /// Generator cost differs between classes.
    HighGeneratorCost,

    /// Low entropy in random inputs (possible API misuse).
    LowUniqueInputs,

    /// Some quantiles were filtered from analysis.
    QuantilesFiltered,

    /// Threshold was clamped to timer resolution.
    ThresholdClamped,

    /// High fraction of samples were winsorized.
    HighWinsorRate,
}

// ============================================================================
// MinDetectableEffect - Sensitivity information
// ============================================================================

/// Minimum detectable effect at current noise level.
///
/// The MDE tells you the smallest effect that could be reliably detected
/// given the measurement noise. If MDE > threshold, a "pass" result means
/// insufficient sensitivity, not necessarily safety.
///
/// See spec Section 2.7 (Minimum Detectable Effect).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinDetectableEffect {
    /// Minimum detectable uniform shift in nanoseconds.
    pub shift_ns: f64,

    /// Minimum detectable tail effect in nanoseconds.
    pub tail_ns: f64,
}

impl Default for MinDetectableEffect {
    fn default() -> Self {
        Self {
            shift_ns: f64::INFINITY,
            tail_ns: f64::INFINITY,
        }
    }
}

// ============================================================================
// BatchingInfo - Metadata about batching
// ============================================================================

/// Information about batching configuration used during collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingInfo {
    /// Whether batching was enabled.
    pub enabled: bool,

    /// Iterations per batch (1 if batching disabled).
    pub k: u32,

    /// Effective ticks per batch measurement.
    pub ticks_per_batch: f64,

    /// Explanation of why batching was enabled/disabled.
    pub rationale: String,

    /// Whether the operation was too fast to measure reliably.
    pub unmeasurable: Option<UnmeasurableInfo>,
}

/// Information about why an operation is unmeasurable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnmeasurableInfo {
    /// Estimated operation duration in nanoseconds.
    pub operation_ns: f64,

    /// Minimum measurable threshold in nanoseconds.
    pub threshold_ns: f64,

    /// Ticks per call (below MIN_TICKS_SINGLE_CALL).
    pub ticks_per_call: f64,
}

// ============================================================================
// Metadata - Runtime information
// ============================================================================

/// Metadata for debugging and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// Samples per class after outlier filtering.
    pub samples_per_class: usize,

    /// Cycles per nanosecond (for conversion).
    pub cycles_per_ns: f64,

    /// Timer type used.
    pub timer: String,

    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,

    /// Batching configuration and rationale.
    pub batching: BatchingInfo,

    /// Total runtime in seconds.
    pub runtime_secs: f64,
}

// ============================================================================
// UnreliablePolicy - How to handle unreliable results
// ============================================================================

/// Policy for handling unreliable measurements in test assertions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UnreliablePolicy {
    /// Log warning and skip assertions. Test passes.
    /// Use when: noisy CI, parallel tests, "some coverage is better than none".
    #[default]
    FailOpen,

    /// Panic. Test fails.
    /// Use when: security-critical code, dedicated quiet CI runners.
    FailClosed,
}

impl UnreliablePolicy {
    /// Get policy from environment variable, or use default.
    ///
    /// Checks `TIMING_ORACLE_UNRELIABLE_POLICY`:
    /// - "fail_open" or "skip" → FailOpen
    /// - "fail_closed" or "panic" → FailClosed
    /// - unset or other → default
    #[cfg(feature = "std")]
    pub fn from_env_or(default: Self) -> Self {
        match std::env::var("TIMING_ORACLE_UNRELIABLE_POLICY")
            .ok()
            .as_deref()
        {
            Some("fail_open") | Some("skip") => Self::FailOpen,
            Some("fail_closed") | Some("panic") => Self::FailClosed,
            _ => default,
        }
    }

    /// Get policy from environment variable, or use default.
    ///
    /// In no_std mode, always returns the default.
    #[cfg(not(feature = "std"))]
    pub fn from_env_or(default: Self) -> Self {
        default
    }
}

// ============================================================================
// Outcome implementation
// ============================================================================

impl Outcome {
    /// Check if the test passed (no timing leak detected).
    pub fn passed(&self) -> bool {
        matches!(self, Outcome::Pass { .. })
    }

    /// Check if the test failed (timing leak detected).
    pub fn failed(&self) -> bool {
        matches!(self, Outcome::Fail { .. })
    }

    /// Check if the result is conclusive (either Pass or Fail).
    pub fn is_conclusive(&self) -> bool {
        matches!(self, Outcome::Pass { .. } | Outcome::Fail { .. })
    }

    /// Check if the operation was measurable.
    pub fn is_measurable(&self) -> bool {
        !matches!(self, Outcome::Unmeasurable { .. })
    }

    /// Get the leak probability if available.
    pub fn leak_probability(&self) -> Option<f64> {
        match self {
            Outcome::Pass { leak_probability, .. } => Some(*leak_probability),
            Outcome::Fail { leak_probability, .. } => Some(*leak_probability),
            Outcome::Inconclusive { leak_probability, .. } => Some(*leak_probability),
            Outcome::Unmeasurable { .. } => None,
        }
    }

    /// Get the effect estimate if available.
    pub fn effect(&self) -> Option<&EffectEstimate> {
        match self {
            Outcome::Pass { effect, .. } => Some(effect),
            Outcome::Fail { effect, .. } => Some(effect),
            Outcome::Inconclusive { effect, .. } => Some(effect),
            Outcome::Unmeasurable { .. } => None,
        }
    }

    /// Get the measurement quality if available.
    pub fn quality(&self) -> Option<MeasurementQuality> {
        match self {
            Outcome::Pass { quality, .. } => Some(*quality),
            Outcome::Fail { quality, .. } => Some(*quality),
            Outcome::Inconclusive { quality, .. } => Some(*quality),
            Outcome::Unmeasurable { .. } => None,
        }
    }

    /// Get the diagnostics if available.
    pub fn diagnostics(&self) -> Option<&Diagnostics> {
        match self {
            Outcome::Pass { diagnostics, .. } => Some(diagnostics),
            Outcome::Fail { diagnostics, .. } => Some(diagnostics),
            Outcome::Inconclusive { diagnostics, .. } => Some(diagnostics),
            Outcome::Unmeasurable { .. } => None,
        }
    }

    /// Get the number of samples used if available.
    pub fn samples_used(&self) -> Option<usize> {
        match self {
            Outcome::Pass { samples_used, .. } => Some(*samples_used),
            Outcome::Fail { samples_used, .. } => Some(*samples_used),
            Outcome::Inconclusive { samples_used, .. } => Some(*samples_used),
            Outcome::Unmeasurable { .. } => None,
        }
    }

    /// Check if the measurement is reliable enough for assertions.
    ///
    /// Returns `true` if:
    /// - Test is conclusive (Pass or Fail), AND
    /// - Quality is not TooNoisy, OR posterior is very conclusive (< 0.1 or > 0.9)
    ///
    /// The key insight: a very conclusive posterior is trustworthy even with noisy
    /// measurements - the signal overcame the noise.
    pub fn is_reliable(&self) -> bool {
        match self {
            Outcome::Unmeasurable { .. } => false,
            Outcome::Inconclusive { .. } => false,
            Outcome::Pass { quality, leak_probability, .. } => {
                *quality != MeasurementQuality::TooNoisy || *leak_probability < 0.01
            }
            Outcome::Fail { quality, leak_probability, .. } => {
                *quality != MeasurementQuality::TooNoisy || *leak_probability > 0.99
            }
        }
    }

    /// Unwrap a Pass result, panicking otherwise.
    pub fn unwrap_pass(self) -> (f64, EffectEstimate, MeasurementQuality, Diagnostics) {
        match self {
            Outcome::Pass { leak_probability, effect, quality, diagnostics, .. } => {
                (leak_probability, effect, quality, diagnostics)
            }
            _ => panic!("Expected Pass outcome, got {:?}", self),
        }
    }

    /// Unwrap a Fail result, panicking otherwise.
    pub fn unwrap_fail(self) -> (f64, EffectEstimate, Exploitability, MeasurementQuality, Diagnostics) {
        match self {
            Outcome::Fail { leak_probability, effect, exploitability, quality, diagnostics, .. } => {
                (leak_probability, effect, exploitability, quality, diagnostics)
            }
            _ => panic!("Expected Fail outcome, got {:?}", self),
        }
    }

    /// Handle unreliable results according to policy.
    ///
    /// Returns `Some(self)` if the result is reliable.
    /// For unreliable results:
    /// - `FailOpen`: prints warning, returns `None`
    /// - `FailClosed`: panics
    ///
    /// # Example
    ///
    /// ```ignore
    /// let outcome = oracle.test(...);
    /// if let Some(result) = outcome.handle_unreliable("test_name", UnreliablePolicy::FailOpen) {
    ///     assert!(result.passed());
    /// }
    /// ```
    #[cfg(feature = "std")]
    pub fn handle_unreliable(self, test_name: &str, policy: UnreliablePolicy) -> Option<Self> {
        if self.is_reliable() {
            return Some(self);
        }

        let reason = match &self {
            Outcome::Unmeasurable { recommendation, .. } => {
                format!("unmeasurable: {}", recommendation)
            }
            Outcome::Inconclusive { reason, .. } => {
                format!("inconclusive: {:?}", reason)
            }
            Outcome::Pass { quality, .. } | Outcome::Fail { quality, .. } => {
                format!("unreliable quality: {:?}", quality)
            }
        };

        match policy {
            UnreliablePolicy::FailOpen => {
                eprintln!("[SKIPPED] {}: {} (fail-open policy)", test_name, reason);
                None
            }
            UnreliablePolicy::FailClosed => {
                panic!(
                    "[FAILED] {}: {} (fail-closed policy)",
                    test_name, reason
                );
            }
        }
    }

    /// Handle unreliable results according to policy (no_std version).
    ///
    /// In no_std mode, this always panics on unreliable results with FailClosed,
    /// and returns None with FailOpen (no printing).
    #[cfg(not(feature = "std"))]
    pub fn handle_unreliable(self, _test_name: &str, policy: UnreliablePolicy) -> Option<Self> {
        if self.is_reliable() {
            return Some(self);
        }

        match policy {
            UnreliablePolicy::FailOpen => None,
            UnreliablePolicy::FailClosed => {
                panic!("Unreliable result with fail-closed policy");
            }
        }
    }
}

// ============================================================================
// Display implementations
// ============================================================================

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Pass { leak_probability, effect, samples_used, .. } => {
                write!(
                    f,
                    "PASS: P(leak)={:.1}%, effect={:.1}ns shift/{:.1}ns tail, {} samples",
                    leak_probability * 100.0,
                    effect.shift_ns,
                    effect.tail_ns,
                    samples_used
                )
            }
            Outcome::Fail { leak_probability, effect, exploitability, samples_used, .. } => {
                write!(
                    f,
                    "FAIL: P(leak)={:.1}%, effect={:.1}ns shift/{:.1}ns tail, {:?}, {} samples",
                    leak_probability * 100.0,
                    effect.shift_ns,
                    effect.tail_ns,
                    exploitability,
                    samples_used
                )
            }
            Outcome::Inconclusive { reason, leak_probability, samples_used, .. } => {
                let reason_str = match reason {
                    InconclusiveReason::DataTooNoisy { .. } => "data too noisy",
                    InconclusiveReason::NotLearning { .. } => "not learning",
                    InconclusiveReason::WouldTakeTooLong { .. } => "would take too long",
                    InconclusiveReason::Timeout { .. } => "timeout",
                    InconclusiveReason::SampleBudgetExceeded { .. } => "budget exceeded",
                };
                write!(
                    f,
                    "INCONCLUSIVE ({}): P(leak)={:.1}%, {} samples",
                    reason_str,
                    leak_probability * 100.0,
                    samples_used
                )
            }
            Outcome::Unmeasurable { operation_ns, threshold_ns, platform, .. } => {
                write!(
                    f,
                    "UNMEASURABLE: {:.1}ns operation < {:.1}ns threshold ({})",
                    operation_ns, threshold_ns, platform
                )
            }
        }
    }
}

impl fmt::Display for EffectPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EffectPattern::UniformShift => write!(f, "uniform shift"),
            EffectPattern::TailEffect => write!(f, "tail effect"),
            EffectPattern::Mixed => write!(f, "mixed"),
            EffectPattern::Indeterminate => write!(f, "indeterminate"),
        }
    }
}

impl fmt::Display for Exploitability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Exploitability::Negligible => write!(f, "negligible"),
            Exploitability::PossibleLAN => write!(f, "possible LAN"),
            Exploitability::LikelyLAN => write!(f, "likely LAN"),
            Exploitability::PossibleRemote => write!(f, "possible remote"),
        }
    }
}

impl fmt::Display for MeasurementQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeasurementQuality::Excellent => write!(f, "excellent"),
            MeasurementQuality::Good => write!(f, "good"),
            MeasurementQuality::Poor => write!(f, "poor"),
            MeasurementQuality::TooNoisy => write!(f, "too noisy"),
        }
    }
}
