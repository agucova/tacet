//! JSON serialization for timing analysis results.

use crate::result::TestResult;

/// Serialize a TestResult to a compact JSON string.
///
/// # Errors
///
/// Returns an error if serialization fails (should not happen for TestResult).
pub fn to_json(result: &TestResult) -> Result<String, serde_json::Error> {
    serde_json::to_string(result)
}

/// Serialize a TestResult to a pretty-printed JSON string.
///
/// # Errors
///
/// Returns an error if serialization fails (should not happen for TestResult).
pub fn to_json_pretty(result: &TestResult) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{
        CiGate, Effect, EffectPattern, Exploitability, MeasurementQuality, Metadata,
        MinDetectableEffect,
    };

    fn make_test_result() -> TestResult {
        TestResult {
            leak_probability: 0.85,
            effect: Some(Effect {
                shift_ns: 150.0,
                tail_ns: 25.0,
                credible_interval_ns: (100.0, 200.0),
                pattern: EffectPattern::UniformShift,
            }),
            exploitability: Exploitability::PossibleLAN,
            min_detectable_effect: MinDetectableEffect {
                shift_ns: 10.0,
                tail_ns: 15.0,
            },
            ci_gate: CiGate {
                alpha: 0.001,
                passed: false,
                thresholds: [0.0; 9],
                observed: [0.0; 9],
            },
            quality: MeasurementQuality::Good,
            outlier_fraction: 0.02,
            metadata: Metadata {
                samples_per_class: 10000,
                cycles_per_ns: 3.0,
                timer: "rdtsc".to_string(),
                runtime_secs: 1.5,
            },
        }
    }

    #[test]
    fn test_to_json() {
        let result = make_test_result();
        let json = to_json(&result).unwrap();
        assert!(json.contains("\"leak_probability\":0.85"));
        assert!(json.contains("\"shift_ns\":150.0"));
    }

    #[test]
    fn test_to_json_pretty() {
        let result = make_test_result();
        let json = to_json_pretty(&result).unwrap();
        assert!(json.contains('\n')); // Pretty print has newlines
        assert!(json.contains("leak_probability"));
    }
}
