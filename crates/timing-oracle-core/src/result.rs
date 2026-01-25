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
#[derive(Clone, Serialize, Deserialize)]
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

        /// User's requested threshold in nanoseconds.
        theta_user: f64,

        /// Effective threshold used for inference (may be elevated due to measurement floor).
        theta_eff: f64,

        /// Measurement floor at final sample count.
        theta_floor: f64,
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

        /// User's requested threshold in nanoseconds.
        theta_user: f64,

        /// Effective threshold used for inference (may be elevated due to measurement floor).
        theta_eff: f64,

        /// Measurement floor at final sample count.
        theta_floor: f64,
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

        /// User's requested threshold in nanoseconds.
        theta_user: f64,

        /// Effective threshold used for inference (may be elevated due to measurement floor).
        theta_eff: f64,

        /// Measurement floor at final sample count.
        theta_floor: f64,
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

    /// Research mode result.
    ///
    /// Returned when using `AttackerModel::Research`. Unlike Pass/Fail/Inconclusive
    /// which make threshold-based decisions, research mode characterizes the
    /// timing behavior relative to the measurement floor using CI-based semantics.
    ///
    /// See `ResearchOutcome` for details on the stopping conditions.
    Research(ResearchOutcome),
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
    TimeBudgetExceeded {
        /// Posterior probability when budget was exhausted.
        current_probability: f64,
        /// Number of samples collected.
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

    /// Measurement conditions changed during the test.
    ///
    /// Detected by comparing calibration statistics with post-test statistics.
    /// This can indicate environmental interference (CPU frequency scaling,
    /// concurrent processes, etc.) that invalidates the covariance estimate.
    /// See spec Section 2.6, Gate 6.
    ConditionsChanged {
        /// Human-readable explanation.
        message: String,
        /// Suggested actions.
        guidance: String,
    },

    /// Threshold was elevated and pass criterion was met at effective threshold.
    ///
    /// The measurement floor exceeded the user's requested threshold, so inference
    /// was performed at an elevated effective threshold. The posterior probability
    /// dropped below pass_threshold at θ_eff, but since θ_eff > θ_user + ε, we
    /// cannot guarantee the user's original requirement is met.
    ///
    /// This is NOT a quality gate failure - it's a semantic constraint: Pass requires
    /// both P < pass_threshold AND θ_eff ≤ θ_user + ε.
    ///
    /// See spec Section 3.5.3 (v5.5 Threshold Elevation Decision Rule).
    ThresholdElevated {
        /// User's requested threshold in nanoseconds (θ_user).
        theta_user: f64,
        /// Effective threshold used for inference (θ_eff = max(θ_user, θ_floor)).
        theta_eff: f64,
        /// Posterior probability at θ_eff (was < pass_threshold).
        leak_probability_at_eff: f64,
        /// True: P(leak > θ_eff) < pass_threshold (pass criterion met at elevated threshold).
        meets_pass_criterion_at_eff: bool,
        /// True: θ_floor at max_samples would be ≤ θ_user + ε (more samples could achieve user threshold).
        achievable_at_max: bool,
        /// Human-readable explanation.
        message: String,
        /// Suggested actions.
        guidance: String,
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
    /// Positive value means the sample class is slower (timing leak detected).
    /// Negative value means the sample class is faster (no leak, or unusual).
    /// This captures effects like branch timing where all operations
    /// take a fixed additional time.
    pub shift_ns: f64,

    /// Tail effect in nanoseconds.
    ///
    /// Positive value means the sample class has a heavier upper tail.
    /// This captures effects like cache misses that occur probabilistically.
    pub tail_ns: f64,

    /// 95% credible interval for the total effect magnitude in nanoseconds.
    ///
    /// This is a Bayesian credible interval, not a frequentist confidence interval.
    /// There is a 95% posterior probability that the true effect lies within this range.
    pub credible_interval_ns: (f64, f64),

    /// Classification of the dominant effect pattern.
    pub pattern: EffectPattern,

    /// When Some, the (μ, τ) decomposition may be unreliable.
    ///
    /// This is set when model fit is poor (Q > q_thresh), indicating that the
    /// observed quantile pattern is not well-explained by the shift+tail basis.
    /// The shift_ns and tail_ns values should be interpreted with caution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interpretation_caveat: Option<String>,
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
            interpretation_caveat: None,
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
/// These thresholds are heuristics based on academic research for risk
/// prioritization, not guarantees. The thresholds reflect modern attack
/// techniques including HTTP/2 multiplexing (Timeless Timing Attacks) and
/// shared-hardware attacks (KyberSlash, Flush+Reload).
///
/// See spec Section 5.4 (Exploitability).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Exploitability {
    /// Effect < 10 ns: Requires shared hardware to exploit.
    ///
    /// Only exploitable by attackers with physical co-location: SGX enclaves,
    /// hyperthreading on same core, containers on same host, or cross-VM on
    /// shared cache. Remote exploitation is impractical.
    ///
    /// References: KyberSlash (2024), Flush+Reload, Prime+Probe literature
    SharedHardwareOnly,

    /// 10-100 ns: Exploitable via HTTP/2 request multiplexing.
    ///
    /// Requires ~100k concurrent HTTP/2 requests to exploit. The "Timeless
    /// Timing Attacks" technique eliminates network jitter by sending requests
    /// that arrive simultaneously, making response order reveal timing differences.
    ///
    /// Reference: Van Goethem et al., "Timeless Timing Attacks" (USENIX Security 2020)
    Http2Multiplexing,

    /// 100 ns - 10 μs: Exploitable with standard remote timing.
    ///
    /// Requires ~1k-10k requests using traditional timing techniques.
    /// Exploitable on LAN with any protocol, or over internet with HTTP/2.
    ///
    /// References: Crosby et al. (2009), Brumley & Boneh (2005)
    StandardRemote,

    /// > 10 μs: Obvious timing leak, trivially exploitable.
    ///
    /// Detectable with < 100 requests. Exploitable over the internet even
    /// with high-jitter connections using traditional timing techniques.
    ObviousLeak,
}

impl Exploitability {
    /// Determine exploitability from effect size in nanoseconds.
    ///
    /// Thresholds are based on:
    /// - < 10 ns: Below HTTP/2 timing precision, requires shared hardware
    /// - 10-100 ns: Within HTTP/2 "Timeless Timing Attacks" range
    /// - 100 ns - 10 μs: Standard remote timing attack range
    /// - > 10 μs: Trivially observable
    pub fn from_effect_ns(effect_ns: f64) -> Self {
        let effect_ns = effect_ns.abs();
        if effect_ns < 10.0 {
            Exploitability::SharedHardwareOnly
        } else if effect_ns < 100.0 {
            Exploitability::Http2Multiplexing
        } else if effect_ns < 10_000.0 {
            Exploitability::StandardRemote
        } else {
            Exploitability::ObviousLeak
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
// ResearchOutcome - Result type for research mode
// ============================================================================

/// Status of a research mode run.
///
/// Research mode (AttackerModel::Research) doesn't make Pass/Fail decisions.
/// Instead, it characterizes the timing behavior with respect to the measurement floor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResearchStatus {
    /// CI clearly above θ_floor — timing difference detected.
    ///
    /// The 95% credible interval lower bound is clearly above the measurement
    /// floor (> 1.1 × θ_floor), indicating a confidently detectable effect.
    EffectDetected,

    /// CI clearly below θ_floor — no timing difference above noise.
    ///
    /// The 95% credible interval upper bound is clearly below the measurement
    /// floor (< 0.9 × θ_floor), indicating no detectable effect.
    NoEffectDetected,

    /// Hit timer resolution limit; θ_floor is as good as it gets.
    ///
    /// Further sampling won't improve the measurement floor because we've
    /// hit the fundamental timer tick resolution.
    ResolutionLimitReached,

    /// Data quality issue detected.
    ///
    /// A quality gate triggered during research mode. Unlike standard mode,
    /// this doesn't block the result but is reported for transparency.
    QualityIssue(InconclusiveReason),

    /// Ran out of time/samples before reaching conclusion.
    ///
    /// The budget was exhausted before the CI could confidently settle
    /// above or below the measurement floor.
    BudgetExhausted,
}

/// Research mode outcome (spec v4.1 research mode).
///
/// This struct is returned when using `AttackerModel::Research`. Unlike the
/// standard `Outcome` which makes Pass/Fail decisions, research mode characterizes
/// the timing behavior relative to the measurement floor.
///
/// Key differences from standard mode:
/// - No Pass/Fail verdict (no threshold comparison)
/// - Reports measurement floor (`theta_floor`) at final sample size
/// - `detectable` field indicates if CI lower bound > floor
/// - `model_mismatch` is non-blocking (tracked but doesn't stop analysis)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchOutcome {
    /// Research outcome status.
    pub status: ResearchStatus,

    /// Maximum effect across quantiles: max_k |(Xβ)_k| in nanoseconds.
    /// This is the posterior mean of the maximum absolute predicted effect.
    pub max_effect_ns: f64,

    /// 95% credible interval for maximum effect: (2.5th, 97.5th percentile).
    pub max_effect_ci: (f64, f64),

    /// Measurement floor at final sample size.
    /// This is the minimum detectable effect given measurement noise.
    pub theta_floor: f64,

    /// True if the effect is detectable: CI lower bound > theta_floor.
    pub detectable: bool,

    /// True if model mismatch was detected (Q > q_thresh).
    /// In research mode, this is non-blocking but adds a caveat to interpretation.
    pub model_mismatch: bool,

    /// Effect size estimate with decomposition.
    /// If `model_mismatch` is true, `interpretation_caveat` will be set.
    pub effect: EffectEstimate,

    /// Number of samples used.
    pub samples_used: usize,

    /// Measurement quality assessment.
    pub quality: MeasurementQuality,

    /// Diagnostic information.
    pub diagnostics: Diagnostics,
}

impl ResearchOutcome {
    /// Check if a timing effect was confidently detected.
    pub fn is_effect_detected(&self) -> bool {
        matches!(self.status, ResearchStatus::EffectDetected)
    }

    /// Check if no effect was confidently detected.
    pub fn is_no_effect_detected(&self) -> bool {
        matches!(self.status, ResearchStatus::NoEffectDetected)
    }

    /// Check if the resolution limit was reached.
    pub fn is_resolution_limit_reached(&self) -> bool {
        matches!(self.status, ResearchStatus::ResolutionLimitReached)
    }

    /// Check if there was a quality issue.
    pub fn has_quality_issue(&self) -> bool {
        matches!(self.status, ResearchStatus::QualityIssue(_))
    }

    /// Get the effect estimate.
    pub fn effect(&self) -> &EffectEstimate {
        &self.effect
    }

    /// Get the measurement quality.
    pub fn quality(&self) -> MeasurementQuality {
        self.quality
    }

    /// Get the diagnostics.
    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }
}

impl fmt::Display for ResearchStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResearchStatus::EffectDetected => write!(f, "effect detected"),
            ResearchStatus::NoEffectDetected => write!(f, "no effect detected"),
            ResearchStatus::ResolutionLimitReached => write!(f, "resolution limit reached"),
            ResearchStatus::QualityIssue(reason) => write!(f, "quality issue: {}", reason),
            ResearchStatus::BudgetExhausted => write!(f, "budget exhausted"),
        }
    }
}

impl fmt::Display for ResearchOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Research Mode: {}", self.status)?;
        writeln!(
            f,
            "  Max effect: {:.2}ns (CI: {:.2}-{:.2}ns)",
            self.max_effect_ns, self.max_effect_ci.0, self.max_effect_ci.1
        )?;
        writeln!(f, "  Measurement floor: {:.2}ns", self.theta_floor)?;
        writeln!(
            f,
            "  Detectable: {}",
            if self.detectable { "yes" } else { "no" }
        )?;
        if self.model_mismatch {
            writeln!(f, "  Warning: model mismatch detected")?;
        }
        writeln!(f, "  Samples: {}", self.samples_used)?;
        writeln!(f, "  Quality: {}", self.quality)?;
        Ok(())
    }
}

// ============================================================================
// TopQuantile - Information about significant quantiles
// ============================================================================

/// Information about a significant quantile (for projection mismatch reporting).
///
/// When the 2D (shift, tail) projection doesn't fit the data well, this struct
/// provides information about which individual quantiles are driving the leak
/// detection. This helps diagnose effects that don't fit the shift+tail model
/// (e.g., effects concentrated at a single quantile).
///
/// See spec Section 7.5 (Per-Quantile Exceedance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopQuantile {
    /// Quantile probability (e.g., 0.9 for 90th percentile).
    pub quantile_p: f64,

    /// Posterior mean δ_k in nanoseconds.
    pub mean_ns: f64,

    /// 95% marginal credible interval (lower, upper) in nanoseconds.
    pub ci95_ns: (f64, f64),

    /// P(|δ_k| > θ_eff | Δ) - per-quantile exceedance probability.
    ///
    /// This is the probability that this individual quantile's effect
    /// exceeds the threshold, computed from the marginal posterior.
    pub exceed_prob: f64,
}

impl TopQuantile {
    /// Create a new TopQuantile entry.
    pub fn new(quantile_p: f64, mean_ns: f64, ci95_ns: (f64, f64), exceed_prob: f64) -> Self {
        Self {
            quantile_p,
            mean_ns,
            ci95_ns,
            exceed_prob,
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

    /// Projection mismatch Q statistic.
    ///
    /// Measures how well the 2D (shift, tail) model fits the 9D quantile differences.
    /// A high value indicates the effect is concentrated at specific quantiles
    /// rather than following the shift+tail pattern.
    pub projection_mismatch_q: f64,

    /// Bootstrap-calibrated threshold for projection mismatch Q statistic.
    /// Q > threshold indicates the 2D projection may be unreliable.
    pub projection_mismatch_threshold: f64,

    /// True if projection fits well (Q <= threshold).
    ///
    /// When false, the shift_ns and tail_ns estimates should be interpreted
    /// with caution; use top_quantiles for more detailed information.
    pub projection_mismatch_ok: bool,

    /// Top quantiles by exceedance probability (when projection mismatch detected).
    ///
    /// When projection_mismatch_ok is false, this field contains detailed
    /// information about which quantiles drive the leak detection. This helps
    /// diagnose effects that don't fit the shift+tail model (e.g., effects
    /// concentrated at a single quantile).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_quantiles: Option<Vec<TopQuantile>>,

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

    /// Preflight warnings from calibration phase.
    ///
    /// These warnings are categorized by severity:
    /// - `Informational`: Sampling efficiency issues (results still valid)
    /// - `ResultUndermining`: Statistical assumption violations (results may be unreliable)
    pub preflight_warnings: Vec<PreflightWarningInfo>,

    // =========================================================================
    // Reproduction info (for verbose/debug output)
    // =========================================================================
    /// Measurement seed used for reproducibility.
    pub seed: Option<u64>,

    /// Attacker model name (e.g., "AdjacentNetwork", "SharedHardware").
    pub attacker_model: Option<String>,

    /// Effect threshold (theta) in nanoseconds.
    pub threshold_ns: f64,

    /// Timer implementation name (e.g., "rdtsc", "cntvct_el0", "kperf").
    pub timer_name: String,

    /// Platform description (e.g., "macos-aarch64").
    pub platform: String,

    /// Reason the timer fell back from high-precision PMU (if applicable).
    ///
    /// Used to generate context-aware recommendations in output.
    /// - "concurrent access": kperf locked by another process
    /// - "no sudo": not running with elevated privileges
    /// - "unavailable": PMU init failed despite privileges
    /// - None: using high-precision timer or x86_64 (rdtsc is already ~0.3ns)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timer_fallback_reason: Option<String>,

    // =========================================================================
    // v5.4 Gibbs sampler diagnostics
    // =========================================================================

    /// v5.4: Total number of Gibbs iterations.
    pub gibbs_iters_total: usize,

    /// v5.4: Number of burn-in iterations.
    pub gibbs_burnin: usize,

    /// v5.4: Number of retained samples.
    pub gibbs_retained: usize,

    /// v5.4: Posterior mean of latent scale λ.
    pub lambda_mean: f64,

    /// v5.4: Posterior standard deviation of λ.
    pub lambda_sd: f64,

    /// v5.4: Coefficient of variation of λ (λ_sd / λ_mean).
    pub lambda_cv: f64,

    /// v5.4: Effective sample size of λ chain.
    pub lambda_ess: f64,

    /// v5.4: Whether λ chain mixed well (CV ≥ 0.1 AND ESS ≥ 20).
    pub lambda_mixing_ok: bool,

    // =========================================================================
    // v5.6 Gibbs sampler κ (kappa) diagnostics - robust t-likelihood
    // =========================================================================

    /// v5.6: Posterior mean of likelihood precision κ.
    pub kappa_mean: f64,

    /// v5.6: Posterior standard deviation of κ.
    pub kappa_sd: f64,

    /// v5.6: Coefficient of variation of κ (kappa_sd / kappa_mean).
    pub kappa_cv: f64,

    /// v5.6: Effective sample size of κ chain.
    pub kappa_ess: f64,

    /// v5.6: Whether κ chain mixed well (CV ≥ 0.1 AND ESS ≥ 20).
    pub kappa_mixing_ok: bool,
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
            projection_mismatch_q: 0.0,
            projection_mismatch_threshold: 18.48, // chi-squared(7, 0.99) as default
            projection_mismatch_ok: true,
            top_quantiles: None,
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
            preflight_warnings: Vec::new(),
            seed: None,
            attacker_model: None,
            threshold_ns: 0.0,
            timer_name: String::new(),
            platform: String::new(),
            timer_fallback_reason: None,
            // v5.4 Gibbs sampler diagnostics
            gibbs_iters_total: 256,
            gibbs_burnin: 64,
            gibbs_retained: 192,
            lambda_mean: 1.0,
            lambda_sd: 0.0,
            lambda_cv: 0.0,
            lambda_ess: 0.0,
            lambda_mixing_ok: true,
            // v5.6 kappa diagnostics
            kappa_mean: 1.0,
            kappa_sd: 0.0,
            kappa_cv: 0.0,
            kappa_ess: 0.0,
            kappa_mixing_ok: true,
        }
    }

    /// Check if all diagnostics are OK.
    pub fn all_checks_passed(&self) -> bool {
        self.stationarity_ok
            && self.projection_mismatch_ok
            && self.outlier_asymmetry_ok
            && self.preflight_ok
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

    /// User's threshold was elevated due to measurement floor.
    ThresholdElevated,

    /// Slab component dominates posterior (v5.2 mixture prior).
    ///
    /// The wide "slab" prior component has higher posterior weight than the
    /// narrow component, indicating evidence strongly favors a large effect
    /// (well above the threshold). This is informational, not a problem.
    ///
    /// DEPRECATED: v5.4 uses Student's t prior with Gibbs sampling instead
    /// of mixture prior. This code is kept for backwards compatibility.
    SlabDominant,

    /// v5.4: Gibbs sampler's lambda chain did not mix well.
    ///
    /// The latent scale variable λ showed poor mixing (CV < 0.1 or ESS < 20),
    /// indicating the posterior may be unreliable. This typically occurs with
    /// very small or very large effects where the posterior is concentrated.
    LambdaMixingPoor,

    /// v5.6: Gibbs sampler's kappa chain did not mix well.
    ///
    /// The likelihood precision variable κ showed poor mixing (CV < 0.1 or ESS < 20),
    /// indicating the posterior may be unreliable.
    KappaMixingPoor,

    /// v5.6: Likelihood covariance was inflated (kappa_mean < 0.3).
    ///
    /// The robust t-likelihood inflated covariance by ~1/κ_mean to accommodate
    /// data that doesn't match the estimated Σₙ. Effect estimates remain valid
    /// but uncertainty was increased for robustness.
    LikelihoodInflated,
}

// ============================================================================
// PreflightWarning - Preflight check results
// ============================================================================

/// Severity of a preflight warning.
///
/// This distinction is critical for interpreting results:
///
/// - **Informational**: Affects sampling efficiency but not result validity.
///   The Bayesian posterior is still trustworthy; you just needed more samples
///   to reach the same confidence level. Examples: high autocorrelation,
///   coarse timer resolution, suboptimal CPU governor.
///
/// - **ResultUndermining**: Violates statistical assumptions the Bayesian model
///   relies on. The posterior confidence may be misplaced because the model's
///   assumptions don't hold. Examples: non-monotonic timer (measurements are
///   garbage), severe non-stationarity (distribution changed during measurement),
///   broken harness with mutable state (Fixed-vs-Fixed inconsistency).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PreflightSeverity {
    /// Sampling efficiency issue - doesn't invalidate results.
    ///
    /// These warnings indicate that the measurement setup is suboptimal and
    /// required more samples to reach a conclusion, but the Bayesian posterior
    /// is still valid. The result can be trusted.
    ///
    /// Examples:
    /// - High autocorrelation (reduces effective sample size)
    /// - Coarse timer resolution (requires more samples)
    /// - Suboptimal CPU governor (adds variance)
    /// - Generator cost asymmetry (may inflate differences but doesn't invalidate)
    Informational,

    /// Statistical assumption violation - undermines result confidence.
    ///
    /// These warnings indicate that fundamental assumptions of the Bayesian
    /// model may be violated. Even if the posterior appears confident, that
    /// confidence may be misplaced.
    ///
    /// Examples:
    /// - Non-monotonic timer (measurements are meaningless)
    /// - Severe non-stationarity (distribution changed during measurement)
    /// - Fixed-vs-Fixed inconsistency with randomization (likely mutable state bug)
    ResultUndermining,
}

/// Category of preflight check.
///
/// Used for organizing warnings in output and for programmatic filtering.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PreflightCategory {
    /// Timer sanity checks (monotonicity, basic functionality).
    ///
    /// **Severity**: ResultUndermining if failed - measurements are unreliable.
    TimerSanity,

    /// Fixed-vs-Fixed internal consistency check.
    ///
    /// **Severity**: ResultUndermining if triggered - may indicate mutable state
    /// captured in test closure, or severe environmental interference.
    /// Note: May be intentional for FPR validation testing.
    Sanity,

    /// Autocorrelation in timing samples.
    ///
    /// **Severity**: Informational - reduces effective sample size but the
    /// block bootstrap accounts for this.
    Autocorrelation,

    /// System configuration (CPU governor, turbo boost, etc.).
    ///
    /// **Severity**: Informational - suboptimal config adds variance but
    /// doesn't invalidate results.
    System,

    /// Timer resolution and precision.
    ///
    /// **Severity**: Informational - coarse timers require more samples but
    /// adaptive batching compensates for this.
    Resolution,

    /// Stationarity of timing distribution.
    ///
    /// **Severity**: ResultUndermining if severely violated - indicates the
    /// timing distribution changed during measurement.
    Stationarity,
}

/// Information about a preflight warning.
///
/// Preflight warnings are collected during the calibration phase and reported
/// to help users understand measurement quality and potential issues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflightWarningInfo {
    /// Category of the check that generated this warning.
    pub category: PreflightCategory,

    /// Severity of this warning.
    ///
    /// - `Informational`: Sampling efficiency issue, results still valid.
    /// - `ResultUndermining`: Statistical assumption violation, results may be unreliable.
    pub severity: PreflightSeverity,

    /// Human-readable description of the warning.
    pub message: String,

    /// Optional guidance for addressing the issue.
    pub guidance: Option<String>,
}

impl PreflightWarningInfo {
    /// Create a new preflight warning.
    pub fn new(
        category: PreflightCategory,
        severity: PreflightSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            category,
            severity,
            message: message.into(),
            guidance: None,
        }
    }

    /// Create a new preflight warning with guidance.
    pub fn with_guidance(
        category: PreflightCategory,
        severity: PreflightSeverity,
        message: impl Into<String>,
        guidance: impl Into<String>,
    ) -> Self {
        Self {
            category,
            severity,
            message: message.into(),
            guidance: Some(guidance.into()),
        }
    }

    /// Check if this warning undermines result confidence.
    pub fn is_result_undermining(&self) -> bool {
        self.severity == PreflightSeverity::ResultUndermining
    }
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
    ///
    /// Returns `None` for `Unmeasurable` and `Research` (research mode uses CI, not probability).
    pub fn leak_probability(&self) -> Option<f64> {
        match self {
            Outcome::Pass {
                leak_probability, ..
            } => Some(*leak_probability),
            Outcome::Fail {
                leak_probability, ..
            } => Some(*leak_probability),
            Outcome::Inconclusive {
                leak_probability, ..
            } => Some(*leak_probability),
            Outcome::Unmeasurable { .. } => None,
            Outcome::Research(_) => None, // Research mode uses CI-based semantics
        }
    }

    /// Get the effect estimate if available.
    pub fn effect(&self) -> Option<&EffectEstimate> {
        match self {
            Outcome::Pass { effect, .. } => Some(effect),
            Outcome::Fail { effect, .. } => Some(effect),
            Outcome::Inconclusive { effect, .. } => Some(effect),
            Outcome::Unmeasurable { .. } => None,
            Outcome::Research(res) => Some(&res.effect),
        }
    }

    /// Get the measurement quality if available.
    pub fn quality(&self) -> Option<MeasurementQuality> {
        match self {
            Outcome::Pass { quality, .. } => Some(*quality),
            Outcome::Fail { quality, .. } => Some(*quality),
            Outcome::Inconclusive { quality, .. } => Some(*quality),
            Outcome::Unmeasurable { .. } => None,
            Outcome::Research(res) => Some(res.quality),
        }
    }

    /// Get the diagnostics if available.
    pub fn diagnostics(&self) -> Option<&Diagnostics> {
        match self {
            Outcome::Pass { diagnostics, .. } => Some(diagnostics),
            Outcome::Fail { diagnostics, .. } => Some(diagnostics),
            Outcome::Inconclusive { diagnostics, .. } => Some(diagnostics),
            Outcome::Unmeasurable { .. } => None,
            Outcome::Research(res) => Some(&res.diagnostics),
        }
    }

    /// Get the number of samples used if available.
    pub fn samples_used(&self) -> Option<usize> {
        match self {
            Outcome::Pass { samples_used, .. } => Some(*samples_used),
            Outcome::Fail { samples_used, .. } => Some(*samples_used),
            Outcome::Inconclusive { samples_used, .. } => Some(*samples_used),
            Outcome::Unmeasurable { .. } => None,
            Outcome::Research(res) => Some(res.samples_used),
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
    ///
    /// For Research mode, reliability is based on whether the CI is clearly above
    /// or below the measurement floor.
    pub fn is_reliable(&self) -> bool {
        match self {
            Outcome::Unmeasurable { .. } => false,
            Outcome::Inconclusive { .. } => false,
            Outcome::Pass {
                quality,
                leak_probability,
                ..
            } => *quality != MeasurementQuality::TooNoisy || *leak_probability < 0.01,
            Outcome::Fail {
                quality,
                leak_probability,
                ..
            } => *quality != MeasurementQuality::TooNoisy || *leak_probability > 0.99,
            Outcome::Research(res) => {
                // Research mode is reliable if we reached a confident conclusion
                matches!(
                    res.status,
                    ResearchStatus::EffectDetected | ResearchStatus::NoEffectDetected
                )
            }
        }
    }

    /// Unwrap a Pass result, panicking otherwise.
    pub fn unwrap_pass(self) -> (f64, EffectEstimate, MeasurementQuality, Diagnostics) {
        match self {
            Outcome::Pass {
                leak_probability,
                effect,
                quality,
                diagnostics,
                ..
            } => (leak_probability, effect, quality, diagnostics),
            _ => panic!("Expected Pass outcome, got {:?}", self),
        }
    }

    /// Unwrap a Fail result, panicking otherwise.
    pub fn unwrap_fail(
        self,
    ) -> (
        f64,
        EffectEstimate,
        Exploitability,
        MeasurementQuality,
        Diagnostics,
    ) {
        match self {
            Outcome::Fail {
                leak_probability,
                effect,
                exploitability,
                quality,
                diagnostics,
                ..
            } => (
                leak_probability,
                effect,
                exploitability,
                quality,
                diagnostics,
            ),
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
            Outcome::Research(research) => {
                format!("research mode: {:?}", research.status)
            }
        };

        match policy {
            UnreliablePolicy::FailOpen => {
                eprintln!("[SKIPPED] {}: {} (fail-open policy)", test_name, reason);
                None
            }
            UnreliablePolicy::FailClosed => {
                panic!("[FAILED] {}: {} (fail-closed policy)", test_name, reason);
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
        write!(f, "{}", crate::formatting::format_outcome_plain(self))
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
            Exploitability::SharedHardwareOnly => write!(f, "shared hardware only"),
            Exploitability::Http2Multiplexing => write!(f, "HTTP/2 multiplexing"),
            Exploitability::StandardRemote => write!(f, "standard remote"),
            Exploitability::ObviousLeak => write!(f, "obvious leak"),
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

impl fmt::Display for InconclusiveReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InconclusiveReason::DataTooNoisy { message, guidance } => {
                write!(f, "Data too noisy: {}\n  \u{2192} {}", message, guidance)
            }
            InconclusiveReason::NotLearning { message, guidance } => {
                write!(f, "Not learning: {}\n  \u{2192} {}", message, guidance)
            }
            InconclusiveReason::WouldTakeTooLong {
                estimated_time_secs,
                samples_needed,
                guidance,
            } => {
                write!(
                    f,
                    "Would take too long: ~{:.0}s / {} samples needed\n  \u{2192} {}",
                    estimated_time_secs, samples_needed, guidance
                )
            }
            InconclusiveReason::TimeBudgetExceeded { .. } => {
                write!(f, "Time budget exceeded")
            }
            InconclusiveReason::SampleBudgetExceeded { .. } => {
                write!(f, "Sample budget exceeded")
            }
            InconclusiveReason::ConditionsChanged { message, guidance } => {
                write!(
                    f,
                    "Conditions changed: {}\n  \u{2192} {}",
                    message, guidance
                )
            }
            InconclusiveReason::ThresholdElevated {
                theta_user,
                theta_eff,
                leak_probability_at_eff,
                achievable_at_max,
                guidance,
                ..
            } => {
                let achievability = if *achievable_at_max {
                    "achievable with more samples"
                } else {
                    "not achievable at max samples"
                };
                write!(
                    f,
                    "Threshold elevated: requested {:.1}ns, used {:.1}ns (P={:.1}% at θ_eff, {})\n  \u{2192} {}",
                    theta_user, theta_eff, leak_probability_at_eff * 100.0, achievability, guidance
                )
            }
        }
    }
}

// ============================================================================
// Debug implementation for Outcome
// ============================================================================

impl fmt::Debug for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", crate::formatting::format_debug_summary_plain(self))
    }
}
