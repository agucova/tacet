//! Test result types and related structures.

use serde::{Deserialize, Serialize};

/// Complete result from a timing analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Probability of timing leak given data and model (0.0 to 1.0).
    pub leak_probability: f64,

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
    /// Per-quantile thresholds used (for debugging).
    pub thresholds: [f64; 9],
    /// Per-quantile observed differences.
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
    pub fn from_mde_ns(mde_ns: f64) -> Self {
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

/// Metadata for debugging and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// Samples per class after outlier filtering.
    pub samples_per_class: usize,
    /// Cycles per nanosecond (for conversion).
    pub cycles_per_ns: f64,
    /// Timer type used.
    pub timer: String,
    /// Total runtime in seconds.
    pub runtime_secs: f64,
}
