//! Terminal output formatting with colors and box drawing.

use colored::Colorize;

use crate::result::{
    Diagnostics, EffectPattern, Exploitability, IssueCode, MeasurementQuality, Outcome,
};
use timing_oracle_core::result::{PreflightCategory, PreflightSeverity, PreflightWarningInfo};

/// Default terminal width for text wrapping.
const DEFAULT_WRAP_WIDTH: usize = 72;

/// Wrap text to fit within a given width, with indentation for continuation lines.
///
/// The first line has no indent applied (caller handles prefix like bullets).
/// Continuation lines get the specified indent to align with the first line's content.
///
/// # Arguments
/// * `text` - The text to wrap
/// * `width` - Maximum line width
/// * `first_line_used` - Characters already used on first line (e.g., "    • " = 6)
/// * `cont_indent` - Indentation string for continuation lines
fn wrap_text(text: &str, width: usize, first_line_used: usize, cont_indent: &str) -> String {
    let first_available = width.saturating_sub(first_line_used);
    let cont_available = width.saturating_sub(cont_indent.len());

    if first_available == 0 || cont_available == 0 {
        return text.to_string();
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return String::new();
    }

    let mut lines = Vec::new();
    let mut current_line = String::new();
    let mut is_first_line = true;

    for word in words {
        let available = if is_first_line {
            first_available
        } else {
            cont_available
        };

        if current_line.is_empty() {
            current_line = word.to_string();
        } else if current_line.len() + 1 + word.len() <= available {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push((is_first_line, current_line));
            is_first_line = false;
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push((is_first_line, current_line));
    }

    lines
        .into_iter()
        .map(|(is_first, line)| {
            if is_first {
                line
            } else {
                format!("{}{}", cont_indent, line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

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
            ..
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
            output.push_str(&format!(
                "    95% CI: {:.1}–{:.1} ns\n",
                effect.credible_interval_ns.0, effect.credible_interval_ns.1,
            ));

            // Show preflight warnings (Measurement Notes)
            if !diagnostics.preflight_warnings.is_empty() {
                if is_verbose() || is_debug() {
                    output.push_str(&format_preflight_validation(diagnostics));
                } else {
                    output.push_str(&format_measurement_notes(&diagnostics.preflight_warnings));
                }
            }

            // Include diagnostics in verbose mode
            if is_verbose() || is_debug() {
                output.push_str(&format_diagnostics_section(diagnostics));
                output.push_str(&format_reproduction_line(diagnostics));
            }

            // Include extended debug info
            if is_debug() {
                output.push_str(&format_debug_environment(diagnostics));
            }
        }

        Outcome::Fail {
            leak_probability,
            effect,
            exploitability,
            samples_used,
            quality,
            diagnostics,
            ..
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
            output.push_str(&format!("      Shift: {:.1} ns\n", effect.shift_ns,));
            output.push_str(&format!("      Tail:  {:.1} ns\n", effect.tail_ns,));
            output.push_str(&format!(
                "      95% CI: {:.1}–{:.1} ns\n",
                effect.credible_interval_ns.0, effect.credible_interval_ns.1,
            ));

            output.push('\n');
            output.push_str("    Exploitability (heuristic):\n");
            let (vector, queries) = exploitability_lines(*exploitability);
            output.push_str(&format!("      Attack vector:  {}\n", vector));
            output.push_str(&format!("      Queries needed: {}\n", queries));

            // Show preflight warnings (Measurement Notes)
            if !diagnostics.preflight_warnings.is_empty() {
                if is_verbose() || is_debug() {
                    output.push_str(&format_preflight_validation(diagnostics));
                } else {
                    output.push_str(&format_measurement_notes(&diagnostics.preflight_warnings));
                }
            }

            // Include diagnostics in verbose mode
            if is_verbose() || is_debug() {
                output.push_str(&format_diagnostics_section(diagnostics));
                output.push_str(&format_reproduction_line(diagnostics));
            }

            // Include extended debug info
            if is_debug() {
                output.push_str(&format_debug_environment(diagnostics));
            }
        }

        Outcome::Inconclusive {
            reason,
            leak_probability,
            effect,
            samples_used,
            quality,
            diagnostics,
            ..
        } => {
            output.push_str(&format!("  Samples: {} per class\n", samples_used));
            output.push_str(&format!("  Quality: {}\n", format_quality(*quality)));
            output.push('\n');

            output.push_str(&format!("  {}\n", "? Inconclusive".cyan().bold()));
            // Use Display instead of Debug for cleaner output
            output.push_str(&format!("    {}\n\n", reason));

            let prob_pct = leak_probability * 100.0;
            output.push_str(&format!(
                "    Current probability of leak: {:.1}%\n",
                prob_pct
            ));

            // Show effect estimate even for inconclusive
            let magnitude = effect.total_effect_ns();
            output.push_str(&format!(
                "    Effect estimate: {:.1} ns {}\n",
                magnitude,
                format_pattern(effect.pattern),
            ));

            // Show "Why This May Have Happened" section with system diagnostics
            output.push_str(&format_inconclusive_diagnostics(diagnostics));

            // Show detailed preflight in verbose mode
            if is_verbose() || is_debug() {
                output.push_str(&format_preflight_validation(diagnostics));
                output.push_str(&format_diagnostics_section(diagnostics));
                output.push_str(&format_reproduction_line(diagnostics));
            }

            // Include extended debug info
            if is_debug() {
                output.push_str(&format_debug_environment(diagnostics));
            }
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

        Outcome::Research(research) => {
            output.push_str(&format!("  Samples: {} per class\n", research.samples_used));
            output.push_str(&format!(
                "  Quality: {}\n",
                format_quality(research.quality)
            ));
            output.push('\n');

            // Status-based header
            let status_line = match &research.status {
                timing_oracle_core::result::ResearchStatus::EffectDetected => {
                    "\u{1F50D} Effect Detected".green().bold()
                }
                timing_oracle_core::result::ResearchStatus::NoEffectDetected => {
                    "\u{2713} No Effect Detected".cyan().bold()
                }
                timing_oracle_core::result::ResearchStatus::ResolutionLimitReached => {
                    "\u{26A0} Resolution Limit Reached".yellow().bold()
                }
                timing_oracle_core::result::ResearchStatus::QualityIssue(_) => {
                    "? Quality Issue".yellow().bold()
                }
                timing_oracle_core::result::ResearchStatus::BudgetExhausted => {
                    "? Budget Exhausted".cyan().bold()
                }
            };
            output.push_str(&format!("  {}\n\n", status_line));

            // Max effect and CI
            output.push_str(&format!(
                "    Max effect: {:.1} ns\n",
                research.max_effect_ns
            ));
            output.push_str(&format!(
                "    95% CI: [{:.1}, {:.1}] ns\n",
                research.max_effect_ci.0, research.max_effect_ci.1
            ));
            output.push_str(&format!(
                "    Measurement floor: {:.1} ns\n",
                research.theta_floor
            ));
            output.push_str(&format!(
                "    Detectable: {}\n",
                if research.detectable { "yes" } else { "no" }
            ));

            // Effect decomposition
            output.push('\n');
            output.push_str(&format!(
                "    Effect: {:.1} ns {}\n",
                research.effect.total_effect_ns(),
                format_pattern(research.effect.pattern),
            ));
            output.push_str(&format!(
                "      Shift: {:.1} ns\n",
                research.effect.shift_ns
            ));
            output.push_str(&format!("      Tail:  {:.1} ns\n", research.effect.tail_ns));

            // Model mismatch warning
            if research.model_mismatch {
                output.push('\n');
                output.push_str(&format!(
                    "    {}\n",
                    "\u{26A0} Model mismatch detected - interpret with caution".yellow()
                ));
                if let Some(ref caveat) = research.effect.interpretation_caveat {
                    output.push_str(&format!("      {}\n", caveat));
                }
            }

            // Diagnostics
            if is_verbose() || is_debug() {
                output.push_str(&format_diagnostics_section(&research.diagnostics));
                output.push_str(&format_reproduction_line(&research.diagnostics));
            }

            if is_debug() {
                output.push_str(&format_debug_environment(&research.diagnostics));
            }
        }
    }

    output.push('\n');
    output.push_str(&sep);
    output.push('\n');

    if matches!(outcome, Outcome::Fail { .. }) {
        output
            .push_str("Note: Exploitability is a heuristic estimate based on effect magnitude.\n");
    }

    output
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
    let mut out = String::new();

    out.push_str("\u{250C}\u{2500} Debug Summary ");
    out.push_str(&"\u{2500}".repeat(40));
    out.push('\n');

    match outcome {
        Outcome::Pass {
            leak_probability,
            effect,
            quality,
            samples_used,
            diagnostics,
            ..
        }
        | Outcome::Fail {
            leak_probability,
            effect,
            quality,
            samples_used,
            diagnostics,
            ..
        }
        | Outcome::Inconclusive {
            leak_probability,
            effect,
            quality,
            samples_used,
            diagnostics,
            ..
        } => {
            // Core metrics
            out.push_str(&format!(
                "\u{2502} P(leak) = {:.1}%\n",
                leak_probability * 100.0
            ));
            out.push_str(&format!(
                "\u{2502} Effect  = {:.1}ns shift + {:.1}ns tail ({})\n",
                effect.shift_ns,
                effect.tail_ns,
                format_pattern(effect.pattern)
            ));

            // Quality with ESS context
            let ess = diagnostics.effective_sample_size;
            let raw = *samples_used;
            let efficiency = if raw > 0 {
                (ess as f64 / raw as f64 * 100.0).round() as usize
            } else {
                0
            };
            out.push_str(&format!(
                "\u{2502} Quality = {} (ESS: {} / {} raw, {}%)\n",
                format_quality_plain(*quality),
                ess,
                raw,
                efficiency
            ));

            // Warnings section (if any)
            if !diagnostics.warnings.is_empty() || !diagnostics.quality_issues.is_empty() {
                out.push_str("\u{2502}\n");
                out.push_str(&format!("\u{2502} {} Warnings:\n", "\u{26A0}".yellow()));

                for warning in &diagnostics.warnings {
                    out.push_str(&format!("\u{2502}   \u{2022} {}\n", warning));
                }
                for issue in &diagnostics.quality_issues {
                    out.push_str(&format!(
                        "\u{2502}   \u{2022} {:?}: {}\n",
                        issue.code, issue.message
                    ));
                }
            }

            // Key diagnostics
            out.push_str("\u{2502}\n");
            out.push_str("\u{2502} Diagnostics:\n");
            out.push_str(&format!(
                "\u{2502}   Timer: {:.1}ns resolution{}\n",
                diagnostics.timer_resolution_ns,
                if diagnostics.discrete_mode {
                    " (discrete)"
                } else {
                    ""
                }
            ));

            // Projection fit with pass/fail indicator
            let projection_status = if diagnostics.projection_mismatch_ok {
                "OK".green().to_string()
            } else {
                "MISMATCH".red().to_string()
            };
            out.push_str(&format!(
                "\u{2502}   Projection: Q = {:.1} ({})\n",
                diagnostics.projection_mismatch_q, projection_status
            ));

            // Stationarity
            let stationarity_status = if diagnostics.stationarity_ok {
                format!("{:.1}x", diagnostics.stationarity_ratio)
            } else {
                format!("{:.1}x {}", diagnostics.stationarity_ratio, "DRIFT".red())
            };
            out.push_str(&format!(
                "\u{2502}   Stationarity: {}\n",
                stationarity_status
            ));

            // Outliers
            out.push_str(&format!(
                "\u{2502}   Outliers: {:.1}% / {:.1}%{}\n",
                diagnostics.outlier_rate_baseline * 100.0,
                diagnostics.outlier_rate_sample * 100.0,
                if !diagnostics.outlier_asymmetry_ok {
                    " (asymmetric!)".red().to_string()
                } else {
                    String::new()
                }
            ));

            out.push_str(&format!(
                "\u{2502}   Runtime: {:.1}s\n",
                diagnostics.total_time_secs
            ));
        }

        Outcome::Unmeasurable {
            operation_ns,
            threshold_ns,
            platform,
            recommendation,
        } => {
            out.push_str(&format!(
                "\u{2502} {} Operation unmeasurable\n",
                "\u{26A0}".yellow()
            ));
            out.push_str(&format!("\u{2502}   Operation: ~{:.1}ns\n", operation_ns));
            out.push_str(&format!("\u{2502}   Threshold: ~{:.1}ns\n", threshold_ns));
            out.push_str(&format!("\u{2502}   Platform: {}\n", platform));
            out.push_str(&format!("\u{2502}   Tip: {}\n", recommendation));
        }

        Outcome::Research(research) => {
            // Status line
            let status_str = match &research.status {
                timing_oracle_core::result::ResearchStatus::EffectDetected => "Effect Detected",
                timing_oracle_core::result::ResearchStatus::NoEffectDetected => {
                    "No Effect Detected"
                }
                timing_oracle_core::result::ResearchStatus::ResolutionLimitReached => {
                    "Resolution Limit"
                }
                timing_oracle_core::result::ResearchStatus::QualityIssue(_) => "Quality Issue",
                timing_oracle_core::result::ResearchStatus::BudgetExhausted => "Budget Exhausted",
            };
            out.push_str(&format!("\u{2502} Status = {}\n", status_str));

            // Max effect and CI
            out.push_str(&format!(
                "\u{2502} Max Effect = {:.1}ns (CI: [{:.1}, {:.1}])\n",
                research.max_effect_ns, research.max_effect_ci.0, research.max_effect_ci.1
            ));
            out.push_str(&format!(
                "\u{2502} Floor = {:.1}ns, Detectable = {}\n",
                research.theta_floor,
                if research.detectable { "yes" } else { "no" }
            ));

            // Effect decomposition
            out.push_str(&format!(
                "\u{2502} Effect  = {:.1}ns shift + {:.1}ns tail ({})\n",
                research.effect.shift_ns,
                research.effect.tail_ns,
                format_pattern(research.effect.pattern)
            ));

            // Quality with ESS context
            let ess = research.diagnostics.effective_sample_size;
            let raw = research.samples_used;
            let efficiency = if raw > 0 {
                (ess as f64 / raw as f64 * 100.0).round() as usize
            } else {
                0
            };
            out.push_str(&format!(
                "\u{2502} Quality = {} (ESS: {} / {} raw, {}%)\n",
                format_quality_plain(research.quality),
                ess,
                raw,
                efficiency
            ));

            // Model mismatch warning
            if research.model_mismatch {
                out.push_str("\u{2502}\n");
                out.push_str(&format!(
                    "\u{2502} {} Model mismatch detected\n",
                    "\u{26A0}".yellow()
                ));
            }

            // Warnings section (if any)
            if !research.diagnostics.warnings.is_empty()
                || !research.diagnostics.quality_issues.is_empty()
            {
                out.push_str("\u{2502}\n");
                out.push_str(&format!("\u{2502} {} Warnings:\n", "\u{26A0}".yellow()));

                for warning in &research.diagnostics.warnings {
                    out.push_str(&format!("\u{2502}   \u{2022} {}\n", warning));
                }
                for issue in &research.diagnostics.quality_issues {
                    out.push_str(&format!(
                        "\u{2502}   \u{2022} {:?}: {}\n",
                        issue.code, issue.message
                    ));
                }
            }

            // Key diagnostics
            out.push_str("\u{2502}\n");
            out.push_str("\u{2502} Diagnostics:\n");
            out.push_str(&format!(
                "\u{2502}   Timer: {:.1}ns resolution{}\n",
                research.diagnostics.timer_resolution_ns,
                if research.diagnostics.discrete_mode {
                    " (discrete)"
                } else {
                    ""
                }
            ));
            out.push_str(&format!(
                "\u{2502}   Runtime: {:.1}s\n",
                research.diagnostics.total_time_secs
            ));
        }
    }

    out.push('\u{2514}');
    out.push_str(&"\u{2500}".repeat(55));
    out.push('\n');

    out
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
            // First line: "    • Code: message" - calculate used width for first line
            let first_used = 6 + code_str.len(); // "    • " + "Code: "
            let cont_indent = format!("      {}", " ".repeat(code_str.len()));
            let wrapped_msg =
                wrap_text(&issue.message, DEFAULT_WRAP_WIDTH, first_used, &cont_indent);
            out.push_str(&format!(
                "    \u{2022} {}{}\n",
                code_str.bold(),
                wrapped_msg
            ));
            // Guidance: "      → tip" (8 chars used on first line)
            let wrapped_guidance = wrap_text(&issue.guidance, DEFAULT_WRAP_WIDTH, 8, "        ");
            out.push_str(&format!("      \u{2192} {}\n", wrapped_guidance.dimmed()));
        }
    }

    out
}

/// Format the reproduction line for verbose/debug output.
///
/// This provides the information needed to reproduce this exact result.
fn format_reproduction_line(diagnostics: &Diagnostics) -> String {
    let mut parts = Vec::new();

    // Attacker model
    if let Some(ref model) = diagnostics.attacker_model {
        parts.push(format!("model={}", model));
    }

    // Seed
    if let Some(seed) = diagnostics.seed {
        parts.push(format!("seed={}", seed));
    }

    // Threshold
    if diagnostics.threshold_ns > 0.0 {
        parts.push(format!("\u{03B8}={:.0}ns", diagnostics.threshold_ns));
    }

    // Timer
    if !diagnostics.timer_name.is_empty() {
        parts.push(format!("timer={}", diagnostics.timer_name));
    }

    if parts.is_empty() {
        return String::new();
    }

    format!("\n  Reproduce: {}\n", parts.join(", ").dimmed())
}

/// Format extended debug environment information for bug reports.
///
/// This provides comprehensive context for reproducing issues.
fn format_debug_environment(diagnostics: &Diagnostics) -> String {
    let mut out = String::new();
    let sep = "\u{2500}".repeat(62);

    out.push('\n');
    out.push_str(&sep);
    out.push_str("\n\n");
    out.push_str("  Debug Information\n\n");

    // Environment
    out.push_str("    Environment:\n");
    out.push_str(&format!(
        "      Platform:       {}\n",
        if diagnostics.platform.is_empty() {
            format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
        } else {
            diagnostics.platform.clone()
        }
    ));
    out.push_str(&format!(
        "      Rust version:   {}\n",
        env!("CARGO_PKG_RUST_VERSION")
    ));
    out.push_str(&format!(
        "      Package:        timing-oracle v{}\n",
        env!("CARGO_PKG_VERSION")
    ));

    // Configuration
    out.push_str("\n    Configuration:\n");
    if let Some(ref model) = diagnostics.attacker_model {
        out.push_str(&format!("      Attacker model: {}\n", model));
    }
    out.push_str(&format!(
        "      Threshold (\u{03B8}):  {:.1} ns\n",
        diagnostics.threshold_ns
    ));
    if let Some(seed) = diagnostics.seed {
        out.push_str(&format!("      Seed:           {}\n", seed));
    }
    if !diagnostics.timer_name.is_empty() {
        out.push_str(&format!(
            "      Timer:          {}\n",
            diagnostics.timer_name
        ));
    }
    out.push_str(&format!(
        "      Resolution:     {:.1} ns\n",
        diagnostics.timer_resolution_ns
    ));
    if diagnostics.discrete_mode {
        out.push_str("      Discrete mode:  enabled\n");
    }

    // Statistical summary
    out.push_str("\n    Statistical Summary:\n");
    out.push_str(&format!(
        "      Calibration:    {} samples\n",
        diagnostics.calibration_samples
    ));
    out.push_str(&format!(
        "      Block length:   {}\n",
        diagnostics.dependence_length
    ));
    out.push_str(&format!(
        "      ESS:            {}\n",
        diagnostics.effective_sample_size
    ));
    out.push_str(&format!(
        "      Stationarity:   {:.2}x {}\n",
        diagnostics.stationarity_ratio,
        if diagnostics.stationarity_ok {
            "OK"
        } else {
            "Suspect"
        }
    ));
    out.push_str(&format!(
        "      Projection:     Q={:.1} {}\n",
        diagnostics.projection_mismatch_q,
        if diagnostics.projection_mismatch_ok {
            "OK"
        } else {
            "Mismatch"
        }
    ));

    // Bug report hint
    out.push_str("\n    ");
    out.push_str(
        &"For bug reports, include this output with TIMING_ORACLE_DEBUG=1"
            .dimmed()
            .to_string(),
    );
    out.push('\n');

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
pub fn is_debug() -> bool {
    env_is_truthy("TIMING_ORACLE_DEBUG")
}

/// Format MeasurementQuality for display (with colors).
fn format_quality(quality: MeasurementQuality) -> String {
    match quality {
        MeasurementQuality::Excellent => "Excellent".green().to_string(),
        MeasurementQuality::Good => "Good".green().to_string(),
        MeasurementQuality::Poor => "Poor".yellow().to_string(),
        MeasurementQuality::TooNoisy => "High Noise".yellow().to_string(),
    }
}

/// Format MeasurementQuality for display (plain text, no colors).
fn format_quality_plain(quality: MeasurementQuality) -> &'static str {
    match quality {
        MeasurementQuality::Excellent => "Excellent",
        MeasurementQuality::Good => "Good",
        MeasurementQuality::Poor => "Poor",
        MeasurementQuality::TooNoisy => "HighNoise",
    }
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

/// Format the "Measurement Notes" section for preflight warnings.
///
/// This section appears in normal (non-verbose) output when there are
/// preflight warnings to display.
fn format_measurement_notes(warnings: &[PreflightWarningInfo]) -> String {
    if warnings.is_empty() {
        return String::new();
    }

    let mut out = String::new();

    // Check if any warnings are result-undermining
    let has_critical = warnings
        .iter()
        .any(|w| w.severity == PreflightSeverity::ResultUndermining);

    if has_critical {
        out.push_str(&format!("\n  {} Measurement Notes:\n", "\u{26A0}".yellow()));
    } else {
        out.push_str("\n  Measurement Notes:\n");
    }

    for warning in warnings {
        let bullet = match warning.severity {
            PreflightSeverity::ResultUndermining => "\u{2022}".red().to_string(),
            PreflightSeverity::Informational => "\u{2022}".to_string(),
        };
        // Wrap message: first line has "    • " prefix (6 chars), continuation aligns
        let wrapped_msg = wrap_text(&warning.message, DEFAULT_WRAP_WIDTH, 6, "      ");
        out.push_str(&format!("    {} {}\n", bullet, wrapped_msg));
        if let Some(guidance) = &warning.guidance {
            // Guidance is indented 6 chars on all lines
            let wrapped_guidance = wrap_text(guidance, DEFAULT_WRAP_WIDTH, 6, "      ");
            out.push_str(&format!("      {}\n", wrapped_guidance.dimmed()));
        }
    }

    out
}

/// Format the "Why This May Have Happened" section for Inconclusive outcomes.
///
/// This provides actionable guidance based on system warnings.
fn format_inconclusive_diagnostics(diagnostics: &Diagnostics) -> String {
    let mut out = String::new();

    // Filter for system-related warnings that explain the inconclusiveness
    let system_config: Vec<&PreflightWarningInfo> = diagnostics
        .preflight_warnings
        .iter()
        .filter(|w| w.category == PreflightCategory::System)
        .collect();
    let resolution: Vec<&PreflightWarningInfo> = diagnostics
        .preflight_warnings
        .iter()
        .filter(|w| w.category == PreflightCategory::Resolution)
        .collect();

    // Only print header if there's actual content to display
    if system_config.is_empty() && resolution.is_empty() {
        return out;
    }

    out.push_str("\n  \u{2139} Why This May Have Happened:\n");

    if !system_config.is_empty() {
        out.push_str("\n    System Configuration:\n");
        for warning in system_config {
            // "      • " = 8 chars
            let wrapped = wrap_text(&warning.message, DEFAULT_WRAP_WIDTH, 8, "        ");
            out.push_str(&format!("      \u{2022} {}\n", wrapped));
        }
    }

    if !resolution.is_empty() {
        out.push_str("\n    Timer Resolution:\n");
        for warning in resolution {
            // "      • " = 8 chars
            let wrapped = wrap_text(&warning.message, DEFAULT_WRAP_WIDTH, 8, "        ");
            out.push_str(&format!("      \u{2022} {}\n", wrapped));
            if let Some(guidance) = &warning.guidance {
                // "      → Tip: " = 13 chars
                let wrapped_tip = wrap_text(guidance, DEFAULT_WRAP_WIDTH, 13, "             ");
                out.push_str(&format!("      \u{2192} Tip: {}\n", wrapped_tip.dimmed()));
            }
        }
    }

    out
}

/// Format detailed preflight validation section for verbose mode.
fn format_preflight_validation(diagnostics: &Diagnostics) -> String {
    let mut out = String::new();
    let sep = "\u{2500}".repeat(62);

    out.push('\n');
    out.push_str(&sep);
    out.push_str("\n\n");
    out.push_str("  Preflight Checks\n");

    // Group warnings by category
    let sanity: Vec<_> = diagnostics
        .preflight_warnings
        .iter()
        .filter(|w| w.category == PreflightCategory::Sanity)
        .collect();
    let timer_sanity: Vec<_> = diagnostics
        .preflight_warnings
        .iter()
        .filter(|w| w.category == PreflightCategory::TimerSanity)
        .collect();
    let autocorr: Vec<_> = diagnostics
        .preflight_warnings
        .iter()
        .filter(|w| w.category == PreflightCategory::Autocorrelation)
        .collect();
    let system: Vec<_> = diagnostics
        .preflight_warnings
        .iter()
        .filter(|w| w.category == PreflightCategory::System)
        .collect();
    let resolution: Vec<_> = diagnostics
        .preflight_warnings
        .iter()
        .filter(|w| w.category == PreflightCategory::Resolution)
        .collect();

    // Result Integrity section
    out.push_str("\n    Result Integrity:\n");

    // Sanity check
    if sanity.is_empty() {
        out.push_str(&format!("      Sanity (F-vs-F):    {}\n", "OK".green()));
    } else {
        for w in &sanity {
            out.push_str(&format!(
                "      Sanity (F-vs-F):    {}\n",
                format_severity_indicator(w.severity)
            ));
            // 8 chars indent for message
            let wrapped = wrap_text(&w.message, DEFAULT_WRAP_WIDTH, 8, "        ");
            out.push_str(&format!("        {}\n", wrapped));
        }
    }

    // Timer sanity
    if timer_sanity.is_empty() {
        out.push_str(&format!("      Timer monotonic:    {}\n", "OK".green()));
    } else {
        for w in &timer_sanity {
            out.push_str(&format!(
                "      Timer monotonic:    {}\n",
                format_severity_indicator(w.severity)
            ));
            // 8 chars indent for message
            let wrapped = wrap_text(&w.message, DEFAULT_WRAP_WIDTH, 8, "        ");
            out.push_str(&format!("        {}\n", wrapped));
        }
    }

    // Stationarity
    let stationarity_status = if diagnostics.stationarity_ok {
        format!("{} {:.2}x", "OK".green(), diagnostics.stationarity_ratio)
    } else {
        format!(
            "{} {:.2}x",
            "Suspect".yellow(),
            diagnostics.stationarity_ratio
        )
    };
    out.push_str(&format!(
        "      Stationarity:       {}\n",
        stationarity_status
    ));

    // Sampling Efficiency section
    out.push_str("\n    Sampling Efficiency:\n");

    // Autocorrelation
    if autocorr.is_empty() {
        out.push_str(&format!("      Autocorrelation:    {}\n", "OK".green()));
    } else {
        for w in &autocorr {
            out.push_str(&format!(
                "      Autocorrelation:    {}\n",
                format_severity_indicator(w.severity)
            ));
            // 8 chars indent
            let wrapped = wrap_text(&w.message, DEFAULT_WRAP_WIDTH, 8, "        ");
            out.push_str(&format!("        {}\n", wrapped));
        }
    }

    // Resolution
    let timer_name = if diagnostics.timer_name.is_empty() {
        String::new()
    } else {
        format!(" ({})", diagnostics.timer_name)
    };
    if resolution.is_empty() {
        out.push_str(&format!(
            "      Timer resolution:   {} {:.1}ns{}\n",
            "OK".green(),
            diagnostics.timer_resolution_ns,
            timer_name
        ));
    } else {
        for w in &resolution {
            out.push_str(&format!(
                "      Timer resolution:   {} {:.1}ns{}\n",
                format_severity_indicator(w.severity),
                diagnostics.timer_resolution_ns,
                timer_name
            ));
            // 8 chars indent
            let wrapped = wrap_text(&w.message, DEFAULT_WRAP_WIDTH, 8, "        ");
            out.push_str(&format!("        {}\n", wrapped));
        }
    }

    // System Configuration section
    out.push_str("\n    System:\n");
    if system.is_empty() {
        out.push_str(&format!("      Configuration:      {}\n", "OK".green()));
    } else {
        for w in &system {
            // "      ⚠ " = 8 chars visual width
            let wrapped = wrap_text(&w.message, DEFAULT_WRAP_WIDTH, 8, "        ");
            out.push_str(&format!("      {} {}\n", "\u{26A0}".yellow(), wrapped));
            if let Some(guidance) = &w.guidance {
                // 8 chars indent
                let wrapped_guidance = wrap_text(guidance, DEFAULT_WRAP_WIDTH, 8, "        ");
                out.push_str(&format!("        {}\n", wrapped_guidance.dimmed()));
            }
        }
    }

    out
}

/// Format severity indicator for verbose output.
fn format_severity_indicator(severity: PreflightSeverity) -> String {
    match severity {
        PreflightSeverity::ResultUndermining => "WARNING".red().to_string(),
        PreflightSeverity::Informational => "INFO".yellow().to_string(),
    }
}

fn exploitability_lines(exploit: Exploitability) -> (String, String) {
    match exploit {
        Exploitability::SharedHardwareOnly => (
            "Shared hardware (SGX, containers)".green().to_string(),
            "~1k on same core".green().to_string(),
        ),
        Exploitability::Http2Multiplexing => (
            "HTTP/2 multiplexing".yellow().to_string(),
            "~100k concurrent".yellow().to_string(),
        ),
        Exploitability::StandardRemote => (
            "Standard remote timing".red().to_string(),
            "~1k-10k".red().to_string(),
        ),
        Exploitability::ObviousLeak => (
            "Any (trivially observable)".red().bold().to_string(),
            "<100".red().bold().to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{Diagnostics, EffectEstimate};

    #[test]
    fn test_wrap_text_short() {
        // Short text that fits on one line
        let result = wrap_text("Hello world", 72, 6, "      ");
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_wrap_text_long() {
        // Long text that needs to wrap
        let text = "This is a very long message that should wrap to multiple lines because it exceeds the maximum width";
        let result = wrap_text(text, 40, 6, "      ");
        let lines: Vec<&str> = result.lines().collect();
        assert!(lines.len() >= 2, "Should wrap to multiple lines");
        // First line should not exceed available width (40 - 6 = 34)
        assert!(lines[0].len() <= 34, "First line too long: {}", lines[0]);
        // Continuation lines should have proper indent
        for line in lines.iter().skip(1) {
            assert!(line.starts_with("      "), "Missing indent: '{}'", line);
        }
    }

    #[test]
    fn test_wrap_text_preserves_indent() {
        let text = "Short message here plus some more words to wrap";
        let result = wrap_text(text, 30, 8, "        ");
        let lines: Vec<&str> = result.lines().collect();
        // Continuation lines should all start with the indent
        for line in lines.iter().skip(1) {
            assert!(
                line.starts_with("        "),
                "Line missing indent: '{}'",
                line
            );
        }
    }

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
