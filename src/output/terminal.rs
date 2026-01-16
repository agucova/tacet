//! Terminal output formatting with colors and box drawing.

use colored::Colorize;

use crate::result::{EffectPattern, Exploitability, MeasurementQuality, Outcome};

/// Format an Outcome for human-readable terminal output.
///
/// Uses ANSI colors and a spec-aligned layout for clear presentation.
pub fn format_outcome(outcome: &Outcome) -> String {
    let mut output = String::new();
    let sep = "\u{2500}".repeat(62);

    output.push_str("timing-oracle\n");
    output.push_str(&sep);
    output.push('\n');
    output.push('\n');

    match outcome {
        Outcome::Pass {
            leak_probability,
            effect,
            samples_used,
            quality,
            diagnostics,
        } => {
            output.push_str(&format!("  Samples: {} per class\n", samples_used));
            output.push_str(&format!("  Quality: {}\n", format_quality(*quality)));
            output.push('\n');

            output.push_str(&format!(
                "  {}\n\n",
                "\u{2713} No timing leak detected".green().bold()
            ));

            let prob_pct = leak_probability * 100.0;
            output.push_str(&format!("    Probability of leak: {:.1}%\n", prob_pct));

            let magnitude = effect.total_effect_ns();
            output.push_str(&format!(
                "    Effect: {:.1} ns {}\n",
                magnitude,
                format_pattern(effect.pattern),
            ));
            output.push_str(&format!(
                "      Shift: {:.1} ns\n",
                effect.shift_ns,
            ));
            output.push_str(&format!(
                "      Tail:  {:.1} ns\n",
                effect.tail_ns,
            ));
            output.push_str(&format!(
                "      95% CI: {:.1}–{:.1} ns\n",
                effect.credible_interval_ns.0,
                effect.credible_interval_ns.1,
            ));
        }

        Outcome::Fail {
            leak_probability,
            effect,
            exploitability,
            samples_used,
            quality,
            diagnostics,
        } => {
            output.push_str(&format!("  Samples: {} per class\n", samples_used));
            output.push_str(&format!("  Quality: {}\n", format_quality(*quality)));
            output.push('\n');

            output.push_str(&format!(
                "  {}\n\n",
                "\u{26A0} Timing leak detected".yellow().bold()
            ));

            let prob_pct = leak_probability * 100.0;
            output.push_str(&format!("    Probability of leak: {:.1}%\n", prob_pct));

            let magnitude = effect.total_effect_ns();
            output.push_str(&format!(
                "    Effect: {:.1} ns {}\n",
                magnitude,
                format_pattern(effect.pattern),
            ));
            output.push_str(&format!(
                "      Shift: {:.1} ns\n",
                effect.shift_ns,
            ));
            output.push_str(&format!(
                "      Tail:  {:.1} ns\n",
                effect.tail_ns,
            ));
            output.push_str(&format!(
                "      95% CI: {:.1}–{:.1} ns\n",
                effect.credible_interval_ns.0,
                effect.credible_interval_ns.1,
            ));

            output.push('\n');
            output.push_str("    Exploitability (heuristic):\n");
            let (lan, internet) = exploitability_lines(*exploitability);
            output.push_str(&format!("      Local network:  {}\n", lan));
            output.push_str(&format!("      Internet:       {}\n", internet));
        }

        Outcome::Inconclusive {
            reason,
            leak_probability,
            effect,
            samples_used,
            quality,
            diagnostics,
        } => {
            output.push_str(&format!("  Samples: {} per class\n", samples_used));
            output.push_str(&format!("  Quality: {}\n", format_quality(*quality)));
            output.push('\n');

            output.push_str(&format!("  {}\n", "? Inconclusive".cyan().bold()));
            output.push_str(&format!("    {:?}\n\n", reason));

            let prob_pct = leak_probability * 100.0;
            output.push_str(&format!(
                "    Current probability of leak: {:.1}%\n",
                prob_pct
            ));
        }

        Outcome::Unmeasurable {
            operation_ns,
            threshold_ns,
            platform,
            recommendation,
        } => {
            output.push_str(&format!(
                "  {}\n\n",
                "\u{26A0} Operation too fast to measure reliably"
                    .yellow()
                    .bold()
            ));
            output.push_str(&format!(
                "    Estimated duration: ~{:.1} ns\n",
                operation_ns
            ));
            output.push_str(&format!(
                "    Minimum measurable: ~{:.1} ns\n",
                threshold_ns
            ));
            output.push_str(&format!("    Platform: {}\n", platform));
            output.push('\n');
            output.push_str(&format!("    Recommendation: {}\n", recommendation));
            output.push('\n');
            output.push_str(&sep);
            output.push('\n');
            output.push_str(
                "Note: Results are unmeasurable at this resolution; no leak probability is reported.\n",
            );
            return output;
        }
    }

    output.push('\n');
    output.push_str(&sep);
    output.push('\n');

    if matches!(outcome, Outcome::Fail { .. }) {
        output.push_str(
            "Note: Exploitability is a heuristic estimate based on effect magnitude.\n",
        );
    }

    output
}

/// Format MeasurementQuality for display.
fn format_quality(quality: MeasurementQuality) -> String {
    match quality {
        MeasurementQuality::Excellent => "Excellent".green().to_string(),
        MeasurementQuality::Good => "Good".green().to_string(),
        MeasurementQuality::Poor => "Poor".yellow().to_string(),
        MeasurementQuality::TooNoisy => "Too Noisy".red().to_string(),
    }
}

/// Format EffectPattern for display.
fn format_pattern(pattern: EffectPattern) -> &'static str {
    match pattern {
        EffectPattern::UniformShift => "UniformShift",
        EffectPattern::TailEffect => "TailEffect",
        EffectPattern::Mixed => "Mixed",
        EffectPattern::Indeterminate => "Indeterminate",
    }
}

fn exploitability_lines(exploit: Exploitability) -> (String, String) {
    match exploit {
        Exploitability::Negligible => (
            "Negligible".green().to_string(),
            "Unlikely".green().to_string(),
        ),
        Exploitability::PossibleLAN => (
            "Possible (~10\u{2075} queries)".yellow().to_string(),
            "Unlikely".green().to_string(),
        ),
        Exploitability::LikelyLAN => (
            "Likely (~10\u{2074} queries)".red().to_string(),
            "Unlikely".yellow().to_string(),
        ),
        Exploitability::PossibleRemote => (
            "Likely".red().to_string(),
            "Possible".red().bold().to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{Diagnostics, EffectEstimate, InconclusiveReason};

    fn make_pass_outcome() -> Outcome {
        Outcome::Pass {
            leak_probability: 0.02,
            effect: EffectEstimate {
                shift_ns: 5.0,
                tail_ns: 2.0,
                credible_interval_ns: (0.0, 10.0),
                pattern: EffectPattern::Indeterminate,
            },
            samples_used: 10000,
            quality: MeasurementQuality::Good,
            diagnostics: Diagnostics::all_ok(),
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
            },
            exploitability: Exploitability::PossibleLAN,
            samples_used: 10000,
            quality: MeasurementQuality::Good,
            diagnostics: Diagnostics::all_ok(),
        }
    }

    #[test]
    fn test_format_pass_outcome() {
        let outcome = make_pass_outcome();
        let output = format_outcome(&outcome);
        assert!(output.contains("timing-oracle"));
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
