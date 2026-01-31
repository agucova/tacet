//! FFI-friendly summary types for tacet bindings.
//!
//! This module provides scalar-only summary types that can be easily converted
//! to FFI structures in both NAPI and C bindings. These types extract the
//! essential information from internal types like `Posterior`, `Calibration`,
//! and `AdaptiveOutcome` without exposing nalgebra matrices.
//!
//! The goal is to:
//! 1. Centralize conversion logic that was duplicated in bindings
//! 2. Use canonical effect pattern classification (dominance-based from draws)
//! 3. Provide consistent behavior across all FFI boundaries

extern crate alloc;
use alloc::string::String;

use crate::result::{Exploitability, MeasurementQuality};

// ============================================================================
// Outcome Type Enum
// ============================================================================

/// FFI-friendly outcome type discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OutcomeType {
    /// No timing leak detected.
    Pass = 0,
    /// Timing leak detected.
    Fail = 1,
    /// Could not reach a decision.
    Inconclusive = 2,
    /// Threshold was elevated beyond tolerance.
    ThresholdElevated = 3,
}

// ============================================================================
// Inconclusive Reason Kind
// ============================================================================

/// FFI-friendly inconclusive reason discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum InconclusiveReasonKind {
    /// Not applicable (outcome is not Inconclusive).
    #[default]
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

// ============================================================================
// Posterior Summary
// ============================================================================

/// FFI-friendly summary of a Posterior distribution (all scalars).
///
/// Extracts all relevant information from a `Posterior` struct without
/// exposing nalgebra matrices.
#[derive(Debug, Clone)]
pub struct PosteriorSummary {
    /// Maximum effect across all deciles: max_k |δ_k| in nanoseconds.
    pub max_effect_ns: f64,
    /// 95% credible interval lower bound for max effect.
    pub ci_low_ns: f64,
    /// 95% credible interval upper bound for max effect.
    pub ci_high_ns: f64,
    /// Posterior probability of timing leak: P(max_k |δ_k| > θ | data).
    pub leak_probability: f64,
    /// Number of samples used in this posterior computation.
    pub n: usize,
    /// Posterior mean of latent scale λ.
    pub lambda_mean: f64,
    /// Whether the λ chain mixed well.
    pub lambda_mixing_ok: bool,
    /// Posterior mean of likelihood precision κ.
    pub kappa_mean: f64,
    /// Coefficient of variation of κ.
    pub kappa_cv: f64,
    /// Effective sample size of κ chain.
    pub kappa_ess: f64,
    /// Whether the κ chain mixed well.
    pub kappa_mixing_ok: bool,
}

impl PosteriorSummary {
    /// Get the maximum effect magnitude.
    pub fn total_effect_ns(&self) -> f64 {
        self.max_effect_ns
    }

    /// Determine measurement quality from the credible interval width.
    pub fn measurement_quality(&self) -> MeasurementQuality {
        // MDE is approximately half the CI width
        let ci_width = self.ci_high_ns - self.ci_low_ns;
        MeasurementQuality::from_mde_ns(ci_width / 2.0)
    }

    /// Determine exploitability from the maximum effect magnitude.
    pub fn exploitability(&self) -> Exploitability {
        Exploitability::from_effect_ns(self.max_effect_ns)
    }
}

// ============================================================================
// Calibration Summary
// ============================================================================

/// FFI-friendly summary of Calibration data.
///
/// Contains the essential scalar fields from `Calibration` needed by bindings.
#[derive(Debug, Clone)]
pub struct CalibrationSummary {
    /// Block length from Politis-White algorithm.
    pub block_length: usize,
    /// Number of calibration samples collected per class.
    pub calibration_samples: usize,
    /// Whether discrete mode is active (< 10% unique values).
    pub discrete_mode: bool,
    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,
    /// User's requested threshold in nanoseconds.
    pub theta_ns: f64,
    /// Effective threshold used for inference.
    pub theta_eff: f64,
    /// Initial measurement floor at calibration time.
    pub theta_floor_initial: f64,
    /// Timer tick floor component.
    pub theta_tick: f64,
    /// Minimum detectable effect in nanoseconds.
    pub mde_ns: f64,
    /// Measured throughput (samples per second).
    pub samples_per_second: f64,
}

// ============================================================================
// Effect Summary
// ============================================================================

/// FFI-friendly effect estimate.
///
/// Contains the timing effect with credible interval.
#[derive(Debug, Clone)]
pub struct EffectSummary {
    /// Maximum effect in nanoseconds: max_k |δ_k|.
    pub max_effect_ns: f64,
    /// 95% credible interval lower bound for max effect.
    pub ci_low_ns: f64,
    /// 95% credible interval upper bound for max effect.
    pub ci_high_ns: f64,
}

impl EffectSummary {
    /// Get the maximum effect magnitude.
    pub fn total_effect_ns(&self) -> f64 {
        self.max_effect_ns
    }
}

impl Default for EffectSummary {
    fn default() -> Self {
        Self {
            max_effect_ns: 0.0,
            ci_low_ns: 0.0,
            ci_high_ns: 0.0,
        }
    }
}

// ============================================================================
// Diagnostics Summary
// ============================================================================

/// FFI-friendly diagnostics.
///
/// Contains scalar diagnostic information from posterior and calibration.
#[derive(Debug, Clone)]
pub struct DiagnosticsSummary {
    /// Block length used for bootstrap resampling.
    pub dependence_length: usize,
    /// Effective sample size accounting for autocorrelation.
    pub effective_sample_size: usize,
    /// Ratio of post-test variance to calibration variance.
    pub stationarity_ratio: f64,
    /// Whether stationarity check passed.
    pub stationarity_ok: bool,
    /// Whether discrete mode was used (low timer resolution).
    pub discrete_mode: bool,
    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,
    /// Posterior mean of latent scale λ.
    pub lambda_mean: f64,
    /// Whether λ chain mixed well.
    pub lambda_mixing_ok: bool,
    /// Posterior mean of likelihood precision κ.
    pub kappa_mean: f64,
    /// Coefficient of variation of κ.
    pub kappa_cv: f64,
    /// Effective sample size of κ chain.
    pub kappa_ess: f64,
    /// Whether κ chain mixed well.
    pub kappa_mixing_ok: bool,
}

impl Default for DiagnosticsSummary {
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

// ============================================================================
// Outcome Summary
// ============================================================================

/// FFI-friendly outcome summary.
///
/// Contains all information needed by bindings to construct their result types.
#[derive(Debug, Clone)]
pub struct OutcomeSummary {
    /// Outcome type discriminant.
    pub outcome_type: OutcomeType,
    /// Posterior probability of timing leak.
    pub leak_probability: f64,
    /// Number of samples used per class.
    pub samples_per_class: usize,
    /// Elapsed time in seconds.
    pub elapsed_secs: f64,
    /// Effect estimate.
    pub effect: EffectSummary,
    /// Measurement quality assessment.
    pub quality: MeasurementQuality,
    /// Exploitability assessment.
    pub exploitability: Exploitability,
    /// Inconclusive reason (if outcome_type is Inconclusive).
    pub inconclusive_reason: InconclusiveReasonKind,
    /// Human-readable recommendation or guidance.
    pub recommendation: String,
    /// User's requested threshold in nanoseconds.
    pub theta_user: f64,
    /// Effective threshold used for inference.
    pub theta_eff: f64,
    /// Measurement floor in nanoseconds.
    pub theta_floor: f64,
    /// Timer tick floor component.
    pub theta_tick: f64,
    /// Whether the user's threshold is achievable at max_samples.
    pub achievable_at_max: bool,
    /// Diagnostics information.
    pub diagnostics: DiagnosticsSummary,
    /// Minimum detectable effect in nanoseconds.
    pub mde_ns: f64,
}

impl OutcomeSummary {
    /// Check if the outcome is conclusive (Pass or Fail).
    pub fn is_conclusive(&self) -> bool {
        matches!(self.outcome_type, OutcomeType::Pass | OutcomeType::Fail)
    }

    /// Check if a leak was detected.
    pub fn is_leak_detected(&self) -> bool {
        matches!(self.outcome_type, OutcomeType::Fail)
    }
}
