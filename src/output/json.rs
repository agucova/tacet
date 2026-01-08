//! JSON serialization for timing analysis results.

use serde::Serialize;

use crate::result::{TestResult, UnmeasurableInfo};

#[derive(Serialize)]
struct JsonOutcome<'a> {
    status: &'a str,
    unmeasurable: Option<&'a UnmeasurableInfo>,
    result: &'a TestResult,
}

/// Serialize a TestResult to a compact JSON string.
///
/// # Errors
///
/// Returns an error if serialization fails (should not happen for TestResult).
pub fn to_json(result: &TestResult) -> Result<String, serde_json::Error> {
    let unmeasurable = result.metadata.batching.unmeasurable.as_ref();
    let status = if unmeasurable.is_some() {
        "unmeasurable"
    } else {
        "completed"
    };
    let outcome = JsonOutcome {
        status,
        unmeasurable,
        result,
    };
    serde_json::to_string(&outcome)
}

/// Serialize a TestResult to a pretty-printed JSON string.
///
/// # Errors
///
/// Returns an error if serialization fails (should not happen for TestResult).
pub fn to_json_pretty(result: &TestResult) -> Result<String, serde_json::Error> {
    let unmeasurable = result.metadata.batching.unmeasurable.as_ref();
    let status = if unmeasurable.is_some() {
        "unmeasurable"
    } else {
        "completed"
    };
    let outcome = JsonOutcome {
        status,
        unmeasurable,
        result,
    };
    serde_json::to_string_pretty(&outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{
        CiGate, CiGateResult, Diagnostics, Effect, EffectPattern, Exploitability, MeasurementQuality, Metadata,
        MinDetectableEffect,
    };

    fn make_test_result() -> TestResult {
        TestResult {
            leak_probability: 0.85,
            effect: Some(Effect {
                shift_ns: 150.0,
                tail_ns: 25.0,
                shift_ci_ns: (100.0, 200.0),
                tail_ci_ns: (5.0, 45.0),
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
                result: CiGateResult::Fail,
                threshold: 0.0,
                max_observed: 0.0,
                observed: [0.0; 9],
                p_value: 0.0001,
            },
            quality: MeasurementQuality::Good,
            outlier_fraction: 0.02,
            diagnostics: Diagnostics::all_ok(),
            metadata: Metadata {
                samples_per_class: 10000,
                cycles_per_ns: 3.0,
                timer: "rdtsc".to_string(),
                timer_resolution_ns: 0.33,
                batching: crate::result::BatchingInfo {
                    enabled: false,
                    k: 1,
                    ticks_per_batch: 1.0,
                    rationale: "No batching".to_string(),
                    unmeasurable: None,
                },
                runtime_secs: 1.5,
            },
        }
    }

    #[test]
    fn test_to_json() {
        let result = make_test_result();
        let json = to_json(&result).unwrap();
        assert!(json.contains("\"status\":\"completed\""));
        assert!(json.contains("\"leak_probability\":0.85"));
        assert!(json.contains("\"shift_ns\":150.0"));
    }

    #[test]
    fn test_to_json_pretty() {
        let result = make_test_result();
        let json = to_json_pretty(&result).unwrap();
        assert!(json.contains('\n')); // Pretty print has newlines
        assert!(json.contains("\"status\": \"completed\""));
        assert!(json.contains("leak_probability"));
    }
}
