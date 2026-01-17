//! Generator cost check.
//!
//! This check ensures that the input generators for fixed and random
//! classes have similar overhead. If they differ significantly, it
//! could introduce systematic bias in the timing measurements.
//!
//! **Severity**: Informational
//!
//! Generator cost asymmetry may inflate timing differences but doesn't
//! invalidate the statistical analysis. The Bayesian model's assumptions
//! are still valid; you just need to interpret the results knowing that
//! some of the measured difference may come from generator overhead.

extern crate alloc;

use alloc::string::String;

use crate::result::{PreflightCategory, PreflightSeverity, PreflightWarningInfo};

/// Warning from the generator cost check.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub enum GeneratorWarning {
    /// Generator costs differ significantly between classes.
    ///
    /// **Severity**: Informational
    ///
    /// This may inflate timing differences but doesn't invalidate the
    /// statistical analysis. Consider using pre-generated inputs to
    /// equalize overhead if this is a concern.
    CostMismatch {
        /// Time to generate fixed inputs (nanoseconds).
        fixed_time_ns: f64,
        /// Time to generate random inputs (nanoseconds).
        random_time_ns: f64,
        /// Percentage difference.
        difference_percent: f64,
    },

    /// One of the generators has suspiciously high cost.
    ///
    /// **Severity**: Informational
    ///
    /// High generator cost adds measurement overhead but doesn't
    /// invalidate the statistical analysis.
    HighCost {
        /// Which class has high cost.
        class: GeneratorClass,
        /// Generator time (nanoseconds).
        time_ns: f64,
        /// Threshold that was exceeded.
        threshold_ns: f64,
    },
}

/// Identifies which generator class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub enum GeneratorClass {
    /// Fixed input generator.
    Fixed,
    /// Random input generator.
    Random,
}

impl GeneratorWarning {
    /// Check if this warning undermines result confidence.
    ///
    /// Generator warnings are always informational - they may affect
    /// sampling efficiency but don't invalidate results.
    pub fn is_result_undermining(&self) -> bool {
        false
    }

    /// Get the severity of this warning.
    pub fn severity(&self) -> PreflightSeverity {
        // All generator warnings are informational
        PreflightSeverity::Informational
    }

    /// Get a human-readable description of the warning.
    pub fn description(&self) -> String {
        match self {
            GeneratorWarning::CostMismatch {
                fixed_time_ns,
                random_time_ns,
                difference_percent: _,
            } => {
                let ratio = random_time_ns / fixed_time_ns.max(1.0);
                alloc::format!(
                    "Generator cost asymmetry: baseline={:.0}ns, sample={:.0}ns ({:.1}x). \
                     This may inflate timing differences. Consider using \
                     pre-generated inputs to equalize overhead.",
                    fixed_time_ns, random_time_ns, ratio
                )
            }
            GeneratorWarning::HighCost {
                class,
                time_ns,
                threshold_ns,
            } => {
                let class_name = match class {
                    GeneratorClass::Fixed => "Baseline",
                    GeneratorClass::Random => "Sample",
                };
                alloc::format!(
                    "{} generator has high overhead: {:.0}ns (threshold: {:.0}ns). \
                     This may dominate measurement noise.",
                    class_name, time_ns, threshold_ns
                )
            }
        }
    }

    /// Get guidance for addressing this warning.
    pub fn guidance(&self) -> Option<String> {
        match self {
            GeneratorWarning::CostMismatch { .. } => Some(
                "Consider pre-generating inputs before measurement to equalize overhead, \
                 or verify that the detected difference accounts for expected timing differences."
                    .into(),
            ),
            GeneratorWarning::HighCost { .. } => Some(
                "Consider pre-generating inputs to reduce overhead, \
                 or increasing the number of samples to compensate."
                    .into(),
            ),
        }
    }

    /// Convert to a PreflightWarningInfo.
    pub fn to_warning_info(&self) -> PreflightWarningInfo {
        match self.guidance() {
            Some(guidance) => PreflightWarningInfo::with_guidance(
                PreflightCategory::Generator,
                self.severity(),
                self.description(),
                guidance,
            ),
            None => PreflightWarningInfo::new(
                PreflightCategory::Generator,
                self.severity(),
                self.description(),
            ),
        }
    }
}

/// Threshold for percentage difference to trigger warning.
const MISMATCH_THRESHOLD_PERCENT: f64 = 10.0;

/// Threshold for absolute generator cost to be considered "high" (in ns).
const HIGH_COST_THRESHOLD_NS: f64 = 1000.0;

/// Perform generator cost check.
///
/// Compares the generation time for fixed and random inputs.
/// Returns a warning if they differ by more than 10%.
///
/// # Arguments
///
/// * `fixed_gen_time_ns` - Average time to generate a fixed input
/// * `random_gen_time_ns` - Average time to generate a random input
///
/// # Returns
///
/// `Some(GeneratorWarning)` if an issue is detected, `None` otherwise.
pub fn generator_cost_check(
    fixed_gen_time_ns: f64,
    random_gen_time_ns: f64,
) -> Option<GeneratorWarning> {
    // Avoid division by zero
    let max_time = fixed_gen_time_ns.max(random_gen_time_ns);
    if max_time < 1e-10 {
        return None;
    }

    // Check for high absolute cost first
    if fixed_gen_time_ns > HIGH_COST_THRESHOLD_NS {
        return Some(GeneratorWarning::HighCost {
            class: GeneratorClass::Fixed,
            time_ns: fixed_gen_time_ns,
            threshold_ns: HIGH_COST_THRESHOLD_NS,
        });
    }

    if random_gen_time_ns > HIGH_COST_THRESHOLD_NS {
        return Some(GeneratorWarning::HighCost {
            class: GeneratorClass::Random,
            time_ns: random_gen_time_ns,
            threshold_ns: HIGH_COST_THRESHOLD_NS,
        });
    }

    // Calculate percentage difference relative to the larger value
    let diff = (fixed_gen_time_ns - random_gen_time_ns).abs();
    let difference_percent = (diff / max_time) * 100.0;

    if difference_percent > MISMATCH_THRESHOLD_PERCENT {
        Some(GeneratorWarning::CostMismatch {
            fixed_time_ns: fixed_gen_time_ns,
            random_time_ns: random_gen_time_ns,
            difference_percent,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_warning_for_similar_costs() {
        let result = generator_cost_check(100.0, 105.0);
        assert!(result.is_none(), "5% difference should not trigger warning");
    }

    #[test]
    fn test_warning_for_large_difference() {
        let result = generator_cost_check(100.0, 150.0);
        assert!(
            matches!(result, Some(GeneratorWarning::CostMismatch { .. })),
            "50% difference should trigger warning"
        );

        if let Some(GeneratorWarning::CostMismatch {
            difference_percent, ..
        }) = result
        {
            assert!(difference_percent > 30.0);
        }
    }

    #[test]
    fn test_high_cost_warning() {
        let result = generator_cost_check(2000.0, 100.0);
        assert!(
            matches!(
                result,
                Some(GeneratorWarning::HighCost {
                    class: GeneratorClass::Fixed,
                    ..
                })
            ),
            "High fixed cost should trigger warning"
        );
    }

    #[test]
    fn test_zero_costs() {
        let result = generator_cost_check(0.0, 0.0);
        assert!(result.is_none(), "Zero costs should not cause issues");
    }

    #[test]
    fn test_severity() {
        let mismatch = GeneratorWarning::CostMismatch {
            fixed_time_ns: 100.0,
            random_time_ns: 200.0,
            difference_percent: 50.0,
        };
        assert_eq!(mismatch.severity(), PreflightSeverity::Informational);
        assert!(!mismatch.is_result_undermining());

        let high_cost = GeneratorWarning::HighCost {
            class: GeneratorClass::Fixed,
            time_ns: 2000.0,
            threshold_ns: 1000.0,
        };
        assert_eq!(high_cost.severity(), PreflightSeverity::Informational);
        assert!(!high_cost.is_result_undermining());
    }
}
