//! Terminal output formatting with colors and box drawing.

use colored::Colorize;

use crate::result::{EffectPattern, Exploitability, MeasurementQuality, TestResult};

/// Format a TestResult for human-readable terminal output.
///
/// Uses ANSI colors and a spec-aligned layout for clear presentation.
pub fn format_result(result: &TestResult) -> String {
    let mut output = String::new();
    let sep = "\u{2500}".repeat(62);

    output.push_str("timing-oracle\n");
    output.push_str(&sep);
    output.push('\n');
    output.push('\n');

    output.push_str(&format!(
        "  Samples: {} per class\n",
        result.metadata.samples_per_class
    ));
    if let Some(unmeasurable) = &result.metadata.batching.unmeasurable {
        output.push('\n');
        output.push_str(&format!(
            "  {}\n\n",
            "\u{26A0} Operation too fast to measure reliably"
                .yellow()
                .bold()
        ));
        output.push_str(&format!(
            "    Estimated duration: ~{:.0} ns\n",
            unmeasurable.operation_ns
        ));
        output.push_str(&format!(
            "    Timer resolution:   {:.1} ns ({})\n",
            result.metadata.timer_resolution_ns, result.metadata.timer
        ));
        output.push_str(&format!(
            "    Minimum measurable: ~{:.0} ns on this platform\n",
            unmeasurable.threshold_ns
        ));
        output.push('\n');
        output.push_str(&sep);
        output.push('\n');
        output.push_str(
            "Note: Results are unmeasurable at this resolution; no leak probability is reported.\n",
        );
        return output;
    }

    output.push_str(&format!("  Quality: {}\n", format_quality(result.quality)));
    output.push_str(&format!(
        "  Min detectable effect: {:.1} ns (shift), {:.1} ns (tail)\n",
        result.min_detectable_effect.shift_ns, result.min_detectable_effect.tail_ns
    ));
    output.push('\n');

    use crate::result::CiGateResult;
    match &result.ci_gate.result {
        CiGateResult::Pass => {
            output.push_str(&format!("  {}\n\n", "\u{2713} No timing leak detected".green().bold()));
        }
        CiGateResult::Fail => {
            output.push_str(&format!("  {}\n\n", "\u{26A0} Timing leak detected".yellow().bold()));
        }
        CiGateResult::Inconclusive { reason } => {
            output.push_str(&format!("  {}\n", "? Inconclusive".cyan().bold()));
            output.push_str(&format!("    {}\n\n", reason));
        }
    }

    let prob_pct = result.leak_probability * 100.0;
    output.push_str(&format!(
        "    Probability of leak: {:.0}%\n",
        prob_pct
    ));

    if let Some(ref effect) = result.effect {
        let magnitude = (effect.shift_ns.powi(2) + effect.tail_ns.powi(2)).sqrt();
        output.push_str(&format!(
            "    Effect: {:.1} ns {}\n",
            magnitude,
            format_pattern(effect.pattern),
        ));
        output.push_str(&format!(
            "      Shift: {:.1} ns (95% CI: {:.1}–{:.1} ns)\n",
            effect.shift_ns,
            effect.shift_ci_ns.0,
            effect.shift_ci_ns.1
        ));
        output.push_str(&format!(
            "      Tail:  {:.1} ns (95% CI: {:.1}–{:.1} ns)\n",
            effect.tail_ns,
            effect.tail_ci_ns.0,
            effect.tail_ci_ns.1
        ));
    }

    output.push('\n');
    output.push_str("    Exploitability (heuristic):\n");
    let (lan, internet) = exploitability_lines(result.exploitability);
    output.push_str(&format!("      Local network:  {}\n", lan));
    output.push_str(&format!("      Internet:       {}\n", internet));
    output.push('\n');

    output.push_str(&sep);
    output.push('\n');

    output.push_str(
        "Note: Exploitability is a heuristic estimate based on effect magnitude.\n",
    );

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
    use crate::result::{CiGate, CiGateResult, Diagnostics, Effect, Metadata, MinDetectableEffect};

    fn make_test_result(passed: bool, leak_probability: f64) -> TestResult {
        TestResult {
            leak_probability,
            effect: if leak_probability > 0.5 {
                Some(Effect {
                    shift_ns: 150.0,
                    tail_ns: 25.0,
                    shift_ci_ns: (100.0, 200.0),
                    tail_ci_ns: (5.0, 45.0),
                    credible_interval_ns: (100.0, 200.0),
                    pattern: EffectPattern::UniformShift,
                })
            } else {
                None
            },
            exploitability: Exploitability::PossibleLAN,
            min_detectable_effect: MinDetectableEffect {
                shift_ns: 10.0,
                tail_ns: 15.0,
            },
            ci_gate: CiGate {
                alpha: 0.001,
                result: if passed { CiGateResult::Pass } else { CiGateResult::Fail },
                threshold: 0.0,
                max_observed: 0.0,
                observed: [0.0; 9],
                p_value: if passed { 0.5 } else { 0.0001 },
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
    fn test_format_passing_result() {
        let result = make_test_result(true, 0.1);
        let output = format_result(&result);
        assert!(output.contains("timing-oracle"));
        assert!(output.contains("Probability of leak: 10%"));
    }

    #[test]
    fn test_format_failing_result() {
        let result = make_test_result(false, 0.95);
        let output = format_result(&result);
        assert!(output.contains("Timing leak detected"));
        assert!(output.contains("Probability of leak: 95%"));
        assert!(output.contains("Effect:"));
    }
}
