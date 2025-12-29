//! Terminal output formatting with colors and box drawing.

use colored::Colorize;

use crate::result::{EffectPattern, Exploitability, MeasurementQuality, TestResult};

/// Format a TestResult for human-readable terminal output.
///
/// Uses ANSI colors and Unicode box drawing for clear presentation.
/// Includes a warning symbol for detected leaks and checkmark for passes.
pub fn format_result(result: &TestResult) -> String {
    let mut output = String::new();

    // Header with pass/fail indicator
    let header = if result.ci_gate.passed {
        format!("{} {}", "\u{2713}".green().bold(), "PASS".green().bold())
    } else {
        format!(
            "{} {}",
            "\u{26A0}".yellow().bold(),
            "TIMING LEAK DETECTED".red().bold()
        )
    };

    output.push_str(&format_box_top());
    output.push_str(&format_box_line(&header));
    output.push_str(&format_box_separator());

    // Leak probability
    let prob_pct = result.leak_probability * 100.0;
    let prob_str = format!("Leak Probability: {:.1}%", prob_pct);
    let prob_colored = if prob_pct > 50.0 {
        prob_str.red()
    } else if prob_pct > 20.0 {
        prob_str.yellow()
    } else {
        prob_str.green()
    };
    output.push_str(&format_box_line(&prob_colored.to_string()));

    // Measurement quality
    let quality_str = format!("Quality: {}", format_quality(result.quality));
    output.push_str(&format_box_line(&quality_str));

    // Sample count
    let samples_str = format!(
        "Samples: {} per class",
        result.metadata.samples_per_class
    );
    output.push_str(&format_box_line(&samples_str));

    // Minimum detectable effect
    let mde_str = format!(
        "MDE: {:.1} ns shift, {:.1} ns tail",
        result.min_detectable_effect.shift_ns, result.min_detectable_effect.tail_ns
    );
    output.push_str(&format_box_line(&mde_str));

    output.push_str(&format_box_separator());

    // Effect details (if present)
    if let Some(ref effect) = result.effect {
        let effect_header = "Effect Size:".bold().to_string();
        output.push_str(&format_box_line(&effect_header));

        let shift_str = format!("  Shift: {:.1} ns", effect.shift_ns);
        output.push_str(&format_box_line(&shift_str));

        let tail_str = format!("  Tail:  {:.1} ns", effect.tail_ns);
        output.push_str(&format_box_line(&tail_str));

        let ci_str = format!(
            "  95% CI: [{:.1}, {:.1}] ns",
            effect.credible_interval_ns.0, effect.credible_interval_ns.1
        );
        output.push_str(&format_box_line(&ci_str));

        let pattern_str = format!("  Pattern: {}", format_pattern(effect.pattern));
        output.push_str(&format_box_line(&pattern_str));

        output.push_str(&format_box_separator());
    }

    // Exploitability
    let exploit_str = format!(
        "Exploitability: {}",
        format_exploitability(result.exploitability)
    );
    output.push_str(&format_box_line(&exploit_str));

    // Outlier fraction
    let outlier_str = format!("Outliers Trimmed: {:.1}%", result.outlier_fraction * 100.0);
    output.push_str(&format_box_line(&outlier_str));

    output.push_str(&format_box_bottom());

    // Disclaimer for exploitability
    output.push_str(&format!(
        "\n{}\n",
        "Note: Exploitability is a heuristic estimate based on effect magnitude."
            .dimmed()
            .italic()
    ));
    output.push_str(&format!(
        "{}\n",
        "Actual exploitability depends on network conditions and attacker capabilities."
            .dimmed()
            .italic()
    ));

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
        EffectPattern::UniformShift => "Uniform Shift (e.g., branch)",
        EffectPattern::TailEffect => "Tail Effect (e.g., cache miss)",
        EffectPattern::Mixed => "Mixed",
    }
}

/// Format Exploitability for display.
fn format_exploitability(exploit: Exploitability) -> String {
    match exploit {
        Exploitability::Negligible => "Negligible (<100 ns)".green().to_string(),
        Exploitability::PossibleLAN => "Possible on LAN (100-500 ns)".yellow().to_string(),
        Exploitability::LikelyLAN => "Likely on LAN (500 ns - 20 us)".red().to_string(),
        Exploitability::PossibleRemote => "Possible Remote (>20 us)".red().bold().to_string(),
    }
}

// Box drawing helpers

const BOX_WIDTH: usize = 60;

fn format_box_top() -> String {
    format!("\u{250C}{}\u{2510}\n", "\u{2500}".repeat(BOX_WIDTH))
}

fn format_box_bottom() -> String {
    format!("\u{2514}{}\u{2518}\n", "\u{2500}".repeat(BOX_WIDTH))
}

fn format_box_separator() -> String {
    format!("\u{251C}{}\u{2524}\n", "\u{2500}".repeat(BOX_WIDTH))
}

fn format_box_line(content: &str) -> String {
    // Strip ANSI codes for length calculation
    let visible_len = strip_ansi_codes(content).chars().count();
    let padding = if visible_len < BOX_WIDTH - 2 {
        BOX_WIDTH - 2 - visible_len
    } else {
        0
    };
    format!(
        "\u{2502} {}{} \u{2502}\n",
        content,
        " ".repeat(padding)
    )
}

/// Strip ANSI escape codes for accurate length calculation.
fn strip_ansi_codes(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip until 'm' (end of ANSI sequence)
            while let Some(&next) = chars.peek() {
                chars.next();
                if next == 'm' {
                    break;
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{CiGate, Effect, Metadata, MinDetectableEffect};

    fn make_test_result(passed: bool, leak_probability: f64) -> TestResult {
        TestResult {
            leak_probability,
            effect: if leak_probability > 0.5 {
                Some(Effect {
                    shift_ns: 150.0,
                    tail_ns: 25.0,
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
                passed,
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
    fn test_format_passing_result() {
        let result = make_test_result(true, 0.1);
        let output = format_result(&result);
        assert!(output.contains("PASS"));
        assert!(output.contains("10.0%"));
    }

    #[test]
    fn test_format_failing_result() {
        let result = make_test_result(false, 0.95);
        let output = format_result(&result);
        assert!(output.contains("TIMING LEAK DETECTED"));
        assert!(output.contains("95.0%"));
        assert!(output.contains("150.0 ns")); // Effect shift
    }

    #[test]
    fn test_strip_ansi_codes() {
        let colored = "\x1b[32mgreen\x1b[0m";
        assert_eq!(strip_ansi_codes(colored), "green");
    }
}
