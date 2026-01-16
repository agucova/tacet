//! Generator cost check.
//!
//! This check ensures that the input generators for fixed and random
//! classes have similar overhead. If they differ significantly, it
//! could introduce systematic bias in the timing measurements.

use serde::{Deserialize, Serialize};

/// Warning from the generator cost check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneratorWarning {
    /// Generator costs differ significantly between classes.
    ///
    /// This is a critical warning as it can introduce systematic bias.
    CostMismatch {
        /// Time to generate fixed inputs (nanoseconds).
        fixed_time_ns: f64,
        /// Time to generate random inputs (nanoseconds).
        random_time_ns: f64,
        /// Percentage difference.
        difference_percent: f64,
    },

    /// One of the generators has suspiciously high cost.
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum GeneratorClass {
    /// Fixed input generator.
    Fixed,
    /// Random input generator.
    Random,
}

impl GeneratorWarning {
    /// Check if this warning indicates a critical issue.
    pub fn is_critical(&self) -> bool {
        matches!(self, GeneratorWarning::CostMismatch { .. })
    }

    /// Get a human-readable description of the warning.
    pub fn description(&self) -> String {
        match self {
            GeneratorWarning::CostMismatch {
                fixed_time_ns,
                random_time_ns,
                difference_percent,
            } => {
                format!(
                    "Generator cost mismatch: fixed={:.1}ns, random={:.1}ns \
                     ({:.1}% difference). This may introduce systematic bias. \
                     Consider equalizing generator overhead or accounting for it.",
                    fixed_time_ns, random_time_ns, difference_percent
                )
            }
            GeneratorWarning::HighCost {
                class,
                time_ns,
                threshold_ns,
            } => {
                let class_name = match class {
                    GeneratorClass::Fixed => "Fixed",
                    GeneratorClass::Random => "Random",
                };
                format!(
                    "{} generator has high overhead: {:.1}ns (threshold: {:.1}ns). \
                     This may dominate measurement noise.",
                    class_name, time_ns, threshold_ns
                )
            }
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
pub fn generator_cost_check(fixed_gen_time_ns: f64, random_gen_time_ns: f64) -> Option<GeneratorWarning> {
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

/// Measure generator cost by running the generator multiple times.
///
/// TODO: Implement actual timing measurement.
/// This would be called by the harness to measure generator overhead.
///
/// # Arguments
///
/// * `generator` - Closure that generates an input
/// * `iterations` - Number of iterations to average over
///
/// # Returns
///
/// Average time per generation in nanoseconds.
#[allow(dead_code)]
pub fn measure_generator_cost<F, T>(mut generator: F, iterations: usize) -> f64
where
    F: FnMut() -> T,
{
    if iterations == 0 {
        return 0.0;
    }

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(generator());
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / iterations as f64
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
    fn test_warning_description() {
        let warning = GeneratorWarning::CostMismatch {
            fixed_time_ns: 100.0,
            random_time_ns: 200.0,
            difference_percent: 50.0,
        };
        let desc = warning.description();
        assert!(desc.contains("100.0ns"));
        assert!(desc.contains("200.0ns"));
        assert!(desc.contains("50.0%"));
    }
}
