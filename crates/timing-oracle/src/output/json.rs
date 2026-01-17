//! JSON serialization for timing analysis results.

use crate::result::Outcome;

/// Serialize an Outcome to a compact JSON string.
///
/// # Errors
///
/// Returns an error if serialization fails (should not happen for Outcome).
pub fn to_json(outcome: &Outcome) -> Result<String, serde_json::Error> {
    serde_json::to_string(outcome)
}

/// Serialize an Outcome to a pretty-printed JSON string.
///
/// # Errors
///
/// Returns an error if serialization fails (should not happen for Outcome).
pub fn to_json_pretty(outcome: &Outcome) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{
        Diagnostics, EffectEstimate, EffectPattern, Exploitability, InconclusiveReason,
        MeasurementQuality,
    };

    fn make_pass_outcome() -> Outcome {
        Outcome::Pass {
            leak_probability: 0.02,
            effect: EffectEstimate {
                shift_ns: 5.0,
                tail_ns: 2.0,
                credible_interval_ns: (0.0, 10.0),
                pattern: EffectPattern::Indeterminate,
                interpretation_caveat: None,
            },
            samples_used: 10000,
            quality: MeasurementQuality::Good,
            diagnostics: Diagnostics::all_ok(),
            theta_user: 100.0,
            theta_eff: 100.0,
            theta_floor: 0.0,
        }
    }

    fn make_fail_outcome() -> Outcome {
        Outcome::Fail {
            leak_probability: 0.98,
            effect: EffectEstimate {
                shift_ns: 150.0,
                tail_ns: 25.0,
                credible_interval_ns: (100.0, 200.0),
                pattern: EffectPattern::UniformShift,
                interpretation_caveat: None,
            },
            exploitability: Exploitability::Http2Multiplexing,
            samples_used: 10000,
            quality: MeasurementQuality::Good,
            diagnostics: Diagnostics::all_ok(),
            theta_user: 100.0,
            theta_eff: 100.0,
            theta_floor: 0.0,
        }
    }

    fn make_inconclusive_outcome() -> Outcome {
        Outcome::Inconclusive {
            reason: InconclusiveReason::TimeBudgetExceeded {
                current_probability: 0.5,
                samples_collected: 50000,
            },
            leak_probability: 0.5,
            effect: EffectEstimate::default(),
            samples_used: 50000,
            quality: MeasurementQuality::Good,
            diagnostics: Diagnostics::all_ok(),
            theta_user: 100.0,
            theta_eff: 100.0,
            theta_floor: 0.0,
        }
    }

    fn make_unmeasurable_outcome() -> Outcome {
        Outcome::Unmeasurable {
            operation_ns: 0.5,
            threshold_ns: 10.0,
            platform: "macos (cntvct)".to_string(),
            recommendation: "Run with sudo for cycle counting".to_string(),
        }
    }

    #[test]
    fn test_to_json_pass() {
        let outcome = make_pass_outcome();
        let json = to_json(&outcome).unwrap();
        assert!(json.contains("Pass"));
        assert!(json.contains("\"leak_probability\":0.02"));
    }

    #[test]
    fn test_to_json_fail() {
        let outcome = make_fail_outcome();
        let json = to_json(&outcome).unwrap();
        assert!(json.contains("Fail"));
        assert!(json.contains("\"leak_probability\":0.98"));
        assert!(json.contains("\"shift_ns\":150.0"));
    }

    #[test]
    fn test_to_json_inconclusive() {
        let outcome = make_inconclusive_outcome();
        let json = to_json(&outcome).unwrap();
        assert!(json.contains("Inconclusive"));
        assert!(json.contains("TimeBudgetExceeded"));
    }

    #[test]
    fn test_to_json_unmeasurable() {
        let outcome = make_unmeasurable_outcome();
        let json = to_json(&outcome).unwrap();
        assert!(json.contains("Unmeasurable"));
        assert!(json.contains("operation_ns"));
    }

    #[test]
    fn test_to_json_pretty() {
        let outcome = make_pass_outcome();
        let json = to_json_pretty(&outcome).unwrap();
        assert!(json.contains('\n')); // Pretty print has newlines
        assert!(json.contains("leak_probability"));
    }
}
