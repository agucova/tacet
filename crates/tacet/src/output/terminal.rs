//! Terminal output formatting with colors and box drawing.
//!
//! This module re-exports the formatting functions from `tacet-core`.
//! When the `ansi` feature is enabled on `tacet-core` (which it is by
//! default when using `tacet`), the output includes ANSI color codes
//! automatically.

use colored::Colorize;

use crate::result::{Diagnostics, IssueCode, Outcome};

/// Format an Outcome for human-readable terminal output.
///
/// Uses ANSI colors and a spec-aligned layout for clear presentation.
/// Colors are included automatically via the `ansi` feature on `tacet-core`.
pub fn format_outcome(outcome: &Outcome) -> String {
    tacet_core::formatting::format_outcome_plain(outcome)
}

/// Format a compact debug summary for test assertions.
///
/// This provides a skimmable overview of key metrics and warnings that help
/// diagnose why a test failed. Designed to be included in assertion panic messages.
///
/// # Example output
///
/// ```text
/// ┌─ Debug Summary ────────────────────────────────────────
/// │ P(leak) = 45.2%
/// │ Effect  = 12.3ns shift + 3.1ns tail (Mixed)
/// │ Quality = Poor (ESS: 2,500 / 50,000 raw)
/// │
/// │ ⚠ Warnings:
/// │   • HighDependence: block length 47
/// │   • StationaritySuspect: variance ratio 3.2x
/// │
/// │ Diagnostics:
/// │   Timer: 41.7ns resolution
/// │   Model fit: χ² = 24.1 (FAIL)
/// │   Outliers: 0.1% / 0.2%
/// │   Runtime: 12.3s
/// └────────────────────────────────────────────────────────
/// ```
pub fn format_debug_summary(outcome: &Outcome) -> String {
    tacet_core::formatting::format_debug_summary_plain(outcome)
}

/// Format a detailed diagnostics section for verbose output.
///
/// This provides comprehensive diagnostic information for debugging
/// timing oracle implementation issues.
pub fn format_diagnostics_section(diagnostics: &Diagnostics) -> String {
    let mut out = String::new();
    let sep = "\u{2500}".repeat(62);

    out.push('\n');
    out.push_str(&sep);
    out.push_str("\n\n");
    out.push_str("  Measurement Diagnostics\n\n");

    // Dependence and ESS
    out.push_str(&format!(
        "    Dependence:   block length {} (ESS: {} / {} raw)\n",
        diagnostics.dependence_length,
        diagnostics.effective_sample_size,
        diagnostics.calibration_samples
    ));

    // Projection fit
    let projection_status = if diagnostics.projection_mismatch_ok {
        "OK".green().to_string()
    } else {
        "Mismatch".red().to_string()
    };
    out.push_str(&format!(
        "    Projection:   Q = {:.1} ({})\n",
        diagnostics.projection_mismatch_q, projection_status
    ));

    // Outliers
    out.push_str(&format!(
        "    Outliers:     baseline {:.2}%, sample {:.2}%",
        diagnostics.outlier_rate_baseline * 100.0,
        diagnostics.outlier_rate_sample * 100.0,
    ));
    if !diagnostics.outlier_asymmetry_ok {
        out.push_str(&format!(" {}", "(asymmetric)".red()));
    }
    out.push('\n');

    // Calibration
    out.push_str(&format!(
        "    Calibration:  {} samples\n",
        diagnostics.calibration_samples
    ));

    // Runtime
    out.push_str(&format!(
        "    Runtime:      {:.1}s\n",
        diagnostics.total_time_secs
    ));

    // Warnings
    if !diagnostics.warnings.is_empty() {
        out.push_str(&format!("\n  {} Warnings\n", "\u{26A0}".yellow()));
        for warning in &diagnostics.warnings {
            out.push_str(&format!("    \u{2022} {}\n", warning));
        }
    }

    // Quality issues with guidance
    if !diagnostics.quality_issues.is_empty() {
        out.push_str(&format!("\n  {} Quality Issues\n", "\u{26A0}".yellow()));
        for issue in &diagnostics.quality_issues {
            let code_str = format!("{}: ", format_issue_code(issue.code));
            out.push_str(&format!(
                "    \u{2022} {}{}\n",
                code_str.bold(),
                issue.message
            ));
            out.push_str(&format!("      \u{2192} {}\n", issue.guidance.dimmed()));
        }
    }

    out
}

/// Check if an environment variable is set to a truthy value.
///
/// Returns true if the variable is set to "1", "true", or "yes" (case-insensitive).
/// Returns false if unset, empty, "0", "false", or "no".
fn env_is_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Check if verbose output is enabled via environment variable.
pub fn is_verbose() -> bool {
    env_is_truthy("TIMING_ORACLE_VERBOSE")
}

/// Check if debug output is enabled via environment variable.
#[allow(dead_code)]
pub fn is_debug() -> bool {
    env_is_truthy("TIMING_ORACLE_DEBUG")
}

/// Format IssueCode for display.
fn format_issue_code(code: IssueCode) -> &'static str {
    match code {
        IssueCode::HighDependence => "HighDependence",
        IssueCode::LowEffectiveSamples => "LowEffectiveSamples",
        IssueCode::StationaritySuspect => "StationaritySuspect",
        IssueCode::DiscreteTimer => "DiscreteTimer",
        IssueCode::SmallSampleDiscrete => "SmallSampleDiscrete",
        IssueCode::HighGeneratorCost => "HighGeneratorCost",
        IssueCode::LowUniqueInputs => "LowUniqueInputs",
        IssueCode::QuantilesFiltered => "QuantilesFiltered",
        IssueCode::ThresholdClamped => "ThresholdClamped",
        IssueCode::HighWinsorRate => "HighWinsorRate",
        IssueCode::ThresholdElevated => "ThresholdElevated",
        IssueCode::SlabDominant => "SlabDominant",
        IssueCode::LambdaMixingPoor => "LambdaMixingPoor",
        IssueCode::KappaMixingPoor => "KappaMixingPoor",
        IssueCode::LikelihoodInflated => "LikelihoodInflated",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{
        Diagnostics, EffectEstimate, EffectPattern, Exploitability, MeasurementQuality,
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

    #[test]
    fn test_format_pass_outcome() {
        let outcome = make_pass_outcome();
        let output = format_outcome(&outcome);
        assert!(output.contains("tacet"));
        assert!(output.contains("No timing leak detected"));
        assert!(output.contains("2.0%")); // 0.02 * 100
    }

    #[test]
    fn test_format_fail_outcome() {
        let outcome = make_fail_outcome();
        let output = format_outcome(&outcome);
        assert!(output.contains("Timing leak detected"));
        assert!(output.contains("98.0%")); // 0.98 * 100
        assert!(output.contains("Effect:"));
        assert!(output.contains("Exploitability"));
    }

    #[test]
    fn test_format_unmeasurable() {
        let outcome = Outcome::Unmeasurable {
            operation_ns: 0.5,
            threshold_ns: 10.0,
            platform: "macos (cntvct)".to_string(),
            recommendation: "Run with sudo".to_string(),
        };
        let output = format_outcome(&outcome);
        assert!(output.contains("too fast to measure"));
        assert!(output.contains("unmeasurable"));
    }
}
