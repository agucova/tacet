//! Power analysis report structures.
//!
//! This module defines the output types for power side-channel analysis,
//! including leak probability, effect estimates, and localization information.

use serde::{Deserialize, Serialize};

use super::config::FeatureFamily;
use super::dataset::{PowerUnits, StageId};

/// Dimension regime classification.
///
/// Based on the ratio r = d / n_eff, determines the statistical regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Regime {
    /// r ≤ 0.05: Covariance estimation is well-conditioned.
    WellConditioned,
    /// 0.05 < r ≤ 0.15: Moderate dimension, some shrinkage may help.
    Stressed,
    /// r > 0.15: High dimension relative to samples, heavy shrinkage needed.
    Overstressed,
}

impl Regime {
    /// Determine regime from dimension ratio.
    pub fn from_ratio(r: f64) -> Self {
        if r <= 0.05 {
            Regime::WellConditioned
        } else if r <= 0.15 {
            Regime::Stressed
        } else {
            Regime::Overstressed
        }
    }
}

impl std::fmt::Display for Regime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Regime::WellConditioned => write!(f, "Well-conditioned"),
            Regime::Stressed => write!(f, "Stressed"),
            Regime::Overstressed => write!(f, "Overstressed"),
        }
    }
}

/// Information about feature dimension and statistical regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionInfo {
    /// Feature dimension (d).
    pub d: usize,
    /// Effective sample size (n_eff).
    pub n_eff: f64,
    /// Dimension ratio (r = d / n_eff).
    pub r: f64,
    /// Statistical regime.
    pub regime: Regime,
    /// Shrinkage parameter used (if any).
    pub shrinkage_lambda: Option<f64>,
}

impl DimensionInfo {
    /// Create dimension info from components.
    pub fn new(d: usize, n_eff: f64) -> Self {
        let r = d as f64 / n_eff;
        Self {
            d,
            n_eff,
            r,
            regime: Regime::from_ratio(r),
            shrinkage_lambda: None,
        }
    }

    /// Set the shrinkage parameter.
    pub fn with_shrinkage(mut self, lambda: f64) -> Self {
        self.shrinkage_lambda = Some(lambda);
        self
    }
}

/// A hotspot identifying a feature with high leakage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureHotspot {
    /// Feature index (0-based).
    pub feature_index: usize,
    /// Partition index within the trace.
    pub partition_index: usize,
    /// Feature family component (e.g., "median", "P10", "P90" for Robust3).
    pub component: Option<String>,
    /// Effect magnitude (in original units).
    pub effect_magnitude: f64,
    /// Contribution to overall leak probability.
    pub contribution: f64,
    /// 95% credible interval for effect.
    pub credible_interval: (f64, f64),
}

/// Report for a single stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageReport {
    /// Stage identifier.
    pub stage_id: StageId,
    /// Leak probability for this stage.
    pub leak_probability: f64,
    /// Maximum effect magnitude in this stage.
    pub max_effect: f64,
    /// Top feature hotspots in this stage.
    pub hotspots: Vec<FeatureHotspot>,
    /// Number of features in this stage.
    pub num_features: usize,
}

/// Diagnostics for power analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerDiagnostics {
    /// Number of Fixed class traces used.
    pub n_fixed: usize,
    /// Number of Random class traces used.
    pub n_random: usize,
    /// Total samples used in analysis.
    pub n_total: usize,
    /// Effective sample size (accounting for autocorrelation).
    pub n_eff: f64,
    /// IACT for Fixed class.
    pub iact_fixed: f64,
    /// IACT for Random class.
    pub iact_random: f64,
    /// Combined IACT.
    pub iact_combined: f64,
    /// Noise floor estimate (θ_floor).
    pub theta_floor: f64,
    /// Block length used for bootstrap.
    pub block_length: usize,
    /// Number of Gibbs samples.
    pub gibbs_samples: usize,
    /// Gibbs sampler burn-in.
    pub gibbs_burnin: usize,
    /// Convergence diagnostic (R-hat or similar).
    pub convergence: Option<f64>,
    /// Warnings generated during analysis.
    pub warnings: Vec<String>,
}

impl Default for PowerDiagnostics {
    fn default() -> Self {
        Self {
            n_fixed: 0,
            n_random: 0,
            n_total: 0,
            n_eff: 0.0,
            iact_fixed: 1.0,
            iact_random: 1.0,
            iact_combined: 1.0,
            theta_floor: 0.0,
            block_length: 1,
            gibbs_samples: 0,
            gibbs_burnin: 0,
            convergence: None,
            warnings: Vec::new(),
        }
    }
}

/// Outcome of power analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerOutcome {
    /// No leakage detected.
    Pass {
        /// Posterior probability of leak.
        leak_probability: f64,
        /// Maximum observed effect.
        max_effect: f64,
    },
    /// Leakage detected.
    Fail {
        /// Posterior probability of leak.
        leak_probability: f64,
        /// Maximum observed effect.
        max_effect: f64,
        /// 95% credible interval for max effect.
        max_effect_ci95: (f64, f64),
    },
    /// Could not reach a conclusion.
    Inconclusive {
        /// Reason for inconclusive result.
        reason: String,
        /// Current posterior probability.
        leak_probability: f64,
    },
}

impl PowerOutcome {
    /// Check if the outcome is a pass.
    pub fn is_pass(&self) -> bool {
        matches!(self, PowerOutcome::Pass { .. })
    }

    /// Check if the outcome is a fail.
    pub fn is_fail(&self) -> bool {
        matches!(self, PowerOutcome::Fail { .. })
    }

    /// Check if the outcome is conclusive (Pass or Fail).
    pub fn is_conclusive(&self) -> bool {
        !matches!(self, PowerOutcome::Inconclusive { .. })
    }

    /// Get the leak probability.
    pub fn leak_probability(&self) -> f64 {
        match self {
            PowerOutcome::Pass { leak_probability, .. } => *leak_probability,
            PowerOutcome::Fail { leak_probability, .. } => *leak_probability,
            PowerOutcome::Inconclusive { leak_probability, .. } => *leak_probability,
        }
    }
}

/// Complete power analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Analysis outcome.
    pub outcome: PowerOutcome,
    /// Maximum effect magnitude across all features.
    pub max_effect: f64,
    /// 95% credible interval for maximum effect.
    pub max_effect_ci95: (f64, f64),
    /// Posterior probability of leak.
    pub leak_probability: f64,
    /// Noise floor threshold (θ_floor).
    pub theta_floor: f64,
    /// Floor multiplier used.
    pub floor_multiplier: f64,
    /// Effective threshold (θ_floor × floor_multiplier).
    pub theta_eff: f64,
    /// Units of the power measurements.
    pub units: PowerUnits,
    /// Feature family used.
    pub feature_family: FeatureFamily,
    /// Dimension information.
    pub dimension: DimensionInfo,
    /// Top feature hotspots (sorted by contribution).
    pub top_features: Vec<FeatureHotspot>,
    /// Per-stage reports (if stage-wise analysis was performed).
    pub stages: Option<Vec<StageReport>>,
    /// Analysis diagnostics.
    pub diagnostics: PowerDiagnostics,
}

impl Report {
    /// Create a new report with the given outcome.
    pub fn new(outcome: PowerOutcome, dimension: DimensionInfo) -> Self {
        let (leak_probability, max_effect, max_effect_ci95) = match &outcome {
            PowerOutcome::Pass { leak_probability, max_effect } => {
                (*leak_probability, *max_effect, (0.0, 0.0))
            }
            PowerOutcome::Fail { leak_probability, max_effect, max_effect_ci95 } => {
                (*leak_probability, *max_effect, *max_effect_ci95)
            }
            PowerOutcome::Inconclusive { leak_probability, .. } => {
                (*leak_probability, 0.0, (0.0, 0.0))
            }
        };

        Self {
            outcome,
            max_effect,
            max_effect_ci95,
            leak_probability,
            theta_floor: 0.0,
            floor_multiplier: 5.0,
            theta_eff: 0.0,
            units: PowerUnits::default(),
            feature_family: FeatureFamily::default(),
            dimension,
            top_features: Vec::new(),
            stages: None,
            diagnostics: PowerDiagnostics::default(),
        }
    }

    /// Check if the analysis detected a leak.
    pub fn is_leaky(&self) -> bool {
        self.outcome.is_fail()
    }

    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        match &self.outcome {
            PowerOutcome::Pass { leak_probability, .. } => {
                format!(
                    "PASS: No power leakage detected (P={:.1}%, max_effect={:.3} {})",
                    leak_probability * 100.0,
                    self.max_effect,
                    self.units
                )
            }
            PowerOutcome::Fail { leak_probability, max_effect, .. } => {
                format!(
                    "FAIL: Power leakage detected (P={:.1}%, max_effect={:.3} {})",
                    leak_probability * 100.0,
                    max_effect,
                    self.units
                )
            }
            PowerOutcome::Inconclusive { reason, leak_probability } => {
                format!(
                    "INCONCLUSIVE: {} (P={:.1}%)",
                    reason,
                    leak_probability * 100.0
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_from_ratio() {
        assert_eq!(Regime::from_ratio(0.01), Regime::WellConditioned);
        assert_eq!(Regime::from_ratio(0.05), Regime::WellConditioned);
        assert_eq!(Regime::from_ratio(0.10), Regime::Stressed);
        assert_eq!(Regime::from_ratio(0.15), Regime::Stressed);
        assert_eq!(Regime::from_ratio(0.20), Regime::Overstressed);
    }

    #[test]
    fn test_dimension_info() {
        let info = DimensionInfo::new(32, 100.0);
        assert_eq!(info.d, 32);
        assert_eq!(info.n_eff, 100.0);
        assert!((info.r - 0.32).abs() < 1e-6);
        assert_eq!(info.regime, Regime::Overstressed);
    }

    #[test]
    fn test_power_outcome() {
        let pass = PowerOutcome::Pass {
            leak_probability: 0.01,
            max_effect: 0.5,
        };
        assert!(pass.is_pass());
        assert!(!pass.is_fail());
        assert!(pass.is_conclusive());

        let fail = PowerOutcome::Fail {
            leak_probability: 0.99,
            max_effect: 10.0,
            max_effect_ci95: (8.0, 12.0),
        };
        assert!(!fail.is_pass());
        assert!(fail.is_fail());
        assert!(fail.is_conclusive());

        let inconclusive = PowerOutcome::Inconclusive {
            reason: "Not enough data".to_string(),
            leak_probability: 0.5,
        };
        assert!(!inconclusive.is_pass());
        assert!(!inconclusive.is_fail());
        assert!(!inconclusive.is_conclusive());
    }

    #[test]
    fn test_report_summary() {
        let dim = DimensionInfo::new(32, 1000.0);
        let report = Report::new(
            PowerOutcome::Fail {
                leak_probability: 0.95,
                max_effect: 5.0,
                max_effect_ci95: (4.0, 6.0),
            },
            dim,
        );

        let summary = report.summary();
        assert!(summary.contains("FAIL"));
        assert!(summary.contains("95.0%"));
    }
}
