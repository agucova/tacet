//! Test result types and related structures.

use serde::{Deserialize, Serialize};

/// Complete result from a timing analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Probability of timing leak given data and model (0.0 to 1.0).
    pub leak_probability: f64,

    /// Bayes factor (BF₁₀): evidence ratio for timing leak vs no leak.
    /// Values > 10 indicate strong evidence for leak, < 0.1 strong evidence against.
    pub bayes_factor: f64,

    /// Effect size estimate (present if leak_probability > 0.5).
    pub effect: Option<Effect>,

    /// Exploitability assessment (heuristic).
    pub exploitability: Exploitability,

    /// Minimum detectable effect given noise level.
    pub min_detectable_effect: MinDetectableEffect,

    /// CI gate result.
    pub ci_gate: CiGate,

    /// Measurement quality assessment.
    pub quality: MeasurementQuality,

    /// Fraction of samples trimmed as outliers.
    pub outlier_fraction: f64,

    /// Diagnostic checks (stationarity, model fit, outlier asymmetry).
    pub diagnostics: Diagnostics,

    /// Metadata for debugging.
    pub metadata: Metadata,
}

/// Effect size decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Effect {
    /// Uniform shift in nanoseconds (positive = fixed class slower).
    pub shift_ns: f64,

    /// Tail effect in nanoseconds (positive = fixed has heavier upper tail).
    pub tail_ns: f64,

    /// 95% credible interval for total effect magnitude.
    pub credible_interval_ns: (f64, f64),

    /// Dominant pattern description.
    pub pattern: EffectPattern,
}

/// Pattern of timing difference.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EffectPattern {
    /// Uniform shift across all quantiles (e.g., branch).
    UniformShift,
    /// Primarily affects upper tail (e.g., cache misses).
    TailEffect,
    /// Mixed pattern.
    Mixed,
    /// Neither shift nor tail is statistically significant.
    Indeterminate,
}

/// Minimum detectable effect at current noise level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinDetectableEffect {
    /// Minimum detectable uniform shift in nanoseconds.
    pub shift_ns: f64,
    /// Minimum detectable tail effect in nanoseconds.
    pub tail_ns: f64,
}

/// CI gate result for pass/fail decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiGate {
    /// The alpha level used.
    pub alpha: f64,
    /// Whether the test passed (no leak detected at this alpha).
    pub passed: bool,
    /// Bootstrap threshold for max statistic (single value).
    pub threshold: f64,
    /// Observed max|Δ_p| statistic.
    pub max_observed: f64,
    /// Per-quantile observed differences (for diagnostics).
    pub observed: [f64; 9],
}

/// Exploitability assessment based on effect magnitude.
///
/// Based on Crosby et al. (2009) thresholds.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Exploitability {
    /// < 100 ns: Would require impractical number of measurements.
    Negligible,
    /// 100-500 ns: Possible on local network with ~100k measurements.
    PossibleLAN,
    /// 500 ns - 20 us: Likely exploitable on local network.
    LikelyLAN,
    /// > 20 us: Possibly exploitable over internet.
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

/// Measurement quality assessment based on noise level.
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
    /// Invalid MDE values (≤ 0 or non-finite) indicate a measurement problem
    /// and are classified as `TooNoisy`.
    ///
    /// Very small MDE (< 0.01 ns) also indicates timer resolution issues
    /// where most samples have identical values.
    pub fn from_mde_ns(mde_ns: f64) -> Self {
        // Invalid MDE indicates measurement failure (e.g., timer resolution too coarse)
        // Very small MDE (< 0.01ns) also indicates collapsed measurement data
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

/// Reason why a quantile was filtered from the CI gate test.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FilterReason {
    /// Variance exceeds 5× mean variance threshold.
    HighVariance,
    /// Near-zero variance (< 1e-10) would cause division issues.
    NearZeroVariance,
    /// Failed power-boost filter (unlikely to contribute to detection).
    PowerBoostFilter,
}

/// Information about a filtered quantile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilteredQuantile {
    /// Quantile index (0-8, corresponding to 10th-90th percentile).
    pub index: usize,
    /// Reason for filtering.
    pub reason: FilterReason,
}

/// Diagnostic checks for result reliability (spec §2.8, §4.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostics {
    /// Block size used for bootstrap (Politis-White automatic selection).
    pub dependence_length: usize,
    /// Effective sample size accounting for autocorrelation (ESS ≈ n / dependence_length).
    pub effective_sample_size: usize,

    /// Non-stationarity: ratio of inference to calibration variance.
    /// Values 0.5-2.0 are normal; >5.0 indicates non-stationarity.
    pub stationarity_ratio: f64,
    /// True if stationarity ratio is within acceptable bounds (0.5-2.0).
    pub stationarity_ok: bool,

    /// Model fit: chi-squared statistic for residuals.
    /// Should be approximately χ²₇ under correct model.
    pub model_fit_chi2: f64,
    /// True if chi² ≤ 18.5 (p > 0.01 under χ²₇).
    pub model_fit_ok: bool,

    /// Outlier rate for fixed class (fraction trimmed).
    pub outlier_rate_fixed: f64,
    /// Outlier rate for random class (fraction trimmed).
    pub outlier_rate_random: f64,
    /// True if outlier rates are symmetric (both <1%, ratio <3×, diff <2%).
    pub outlier_asymmetry_ok: bool,

    /// Quantiles filtered from CI gate analysis with reasons.
    pub filtered_quantiles: Vec<FilteredQuantile>,
    /// Whether discrete timer mode was used (low timer resolution).
    pub discrete_mode: bool,
    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,
    /// Fraction of samples with duplicate timing values (0.0-1.0).
    pub duplicate_fraction: f64,

    /// True if preflight checks passed (sanity, generator, system).
    /// False indicates measurement harness issues or environmental interference.
    pub preflight_ok: bool,

    /// Human-readable warnings (empty if all checks pass).
    pub warnings: Vec<String>,
}

impl Diagnostics {
    /// Create diagnostics indicating all checks passed.
    ///
    /// Uses placeholder values for numeric fields; prefer constructing
    /// explicitly with actual measured values.
    pub fn all_ok() -> Self {
        Self {
            dependence_length: 1,
            effective_sample_size: 0, // Should be set to actual sample count
            stationarity_ratio: 1.0,
            stationarity_ok: true,
            model_fit_chi2: 0.0,
            model_fit_ok: true,
            outlier_rate_fixed: 0.0,
            outlier_rate_random: 0.0,
            outlier_asymmetry_ok: true,
            filtered_quantiles: Vec::new(),
            discrete_mode: false,
            timer_resolution_ns: 1.0,
            duplicate_fraction: 0.0,
            preflight_ok: true,
            warnings: Vec::new(),
        }
    }

    /// Check if all diagnostics are OK.
    pub fn all_checks_passed(&self) -> bool {
        self.stationarity_ok && self.model_fit_ok && self.outlier_asymmetry_ok && self.preflight_ok
    }
}

// ============================================================================
// Outcome and Reliability Handling
// ============================================================================

/// Top-level outcome of a timing test.
///
/// Distinguishes between successful analysis and unmeasurable operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)] // Completed path is hot; avoid Box indirection
pub enum Outcome {
    /// Analysis completed successfully.
    Completed(TestResult),

    /// Operation too fast to measure reliably on this platform.
    Unmeasurable {
        /// Estimated operation duration in nanoseconds.
        operation_ns: f64,
        /// Minimum measurable duration on this platform.
        threshold_ns: f64,
        /// Platform description (e.g., "Apple Silicon (cntvct)").
        platform: String,
        /// Suggested actions.
        recommendation: String,
    },
}

impl Outcome {
    /// Check if the measurement is reliable enough for assertions.
    ///
    /// Returns `true` if:
    /// - Test completed successfully, AND
    /// - MDE is valid (> 0.01ns, indicating timer can resolve operations), AND
    /// - Either quality is acceptable, OR posterior is conclusive (< 0.1 or > 0.9)
    ///
    /// The key insight: a conclusive posterior (near 0% or 100%) is trustworthy
    /// even with noisy measurements—the signal overcame the noise. Only unreliable
    /// when noise prevents reaching any conclusion.
    ///
    /// However, if MDE is 0.0 or very small, this indicates timer resolution failure
    /// (most samples identical), making any posterior meaningless regardless of value.
    pub fn is_reliable(&self) -> bool {
        match self {
            Outcome::Unmeasurable { .. } => false,
            Outcome::Completed(result) => {
                // MDE of 0.0 or very small indicates timer resolution failure
                // In this case, even "conclusive" posteriors are meaningless garbage
                let mde_valid = result.min_detectable_effect.shift_ns > 0.01
                    && result.min_detectable_effect.shift_ns.is_finite();

                if !mde_valid {
                    return false;
                }

                // Conclusive posteriors are trustworthy even if noisy
                result.leak_probability < 0.1
                    || result.leak_probability > 0.9
                    || result.quality != MeasurementQuality::TooNoisy
            }
        }
    }

    /// Handle unreliable measurements according to policy.
    ///
    /// - `FailOpen`: logs `[SKIPPED]` message, returns `None`
    /// - `FailClosed`: panics with error message
    ///
    /// Returns `Some(result)` if measurement is reliable.
    pub fn handle_unreliable(
        self,
        test_name: &str,
        policy: UnreliablePolicy,
    ) -> Option<TestResult> {
        if self.is_reliable() {
            return Some(self.unwrap_completed());
        }

        // Format message based on Outcome variant
        let message = match &self {
            Outcome::Unmeasurable {
                operation_ns,
                threshold_ns,
                platform,
                ..
            } => {
                format!(
                    "unmeasurable: ~{:.0}ns operation, {:.0}ns threshold ({})",
                    operation_ns, threshold_ns, platform
                )
            }
            Outcome::Completed(result) => {
                format!(
                    "inconclusive: P(leak)={:.0}%, MDE={:.1}ns shift/{:.1}ns tail",
                    result.leak_probability * 100.0,
                    result.min_detectable_effect.shift_ns,
                    result.min_detectable_effect.tail_ns
                )
            }
        };

        match policy {
            UnreliablePolicy::FailOpen => {
                eprintln!("[SKIPPED] {}: {}", test_name, message);
                None
            }
            UnreliablePolicy::FailClosed => {
                panic!("[UNRELIABLE] {}: {}", test_name, message);
            }
        }
    }

    /// Unwrap a completed result, panicking if unmeasurable.
    pub fn unwrap_completed(self) -> TestResult {
        match self {
            Outcome::Completed(result) => result,
            Outcome::Unmeasurable { platform, .. } => {
                panic!("Test was unmeasurable on {}", platform)
            }
        }
    }

    /// Returns the result if completed, `None` if unmeasurable.
    pub fn completed(self) -> Option<TestResult> {
        match self {
            Outcome::Completed(result) => Some(result),
            Outcome::Unmeasurable { .. } => None,
        }
    }
}

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
    /// Get policy from environment, with fallback default.
    ///
    /// Reads `TIMING_ORACLE_UNRELIABLE_POLICY` environment variable:
    /// - `"fail_open"` → `FailOpen`
    /// - `"fail_closed"` → `FailClosed`
    /// - anything else → `default`
    pub fn from_env_or(default: Self) -> Self {
        match std::env::var("TIMING_ORACLE_UNRELIABLE_POLICY").as_deref() {
            Ok("fail_open") => UnreliablePolicy::FailOpen,
            Ok("fail_closed") => UnreliablePolicy::FailClosed,
            _ => default,
        }
    }
}

impl TestResult {
    /// Returns true if the test could detect effects of the given size.
    pub fn can_detect(&self, effect_ns: f64) -> bool {
        self.min_detectable_effect.shift_ns <= effect_ns
    }
}
