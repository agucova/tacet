//! Formatting helpers for Outcome display.
//!
//! These functions produce output that can be used directly by Display/Debug
//! implementations. When the `std` feature is enabled, output includes ANSI
//! color codes (via the `colored` crate). Without `std`, output is plain text.

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write;

use crate::colors::{bold, bold_cyan, bold_green, bold_red, bold_yellow, dim, green, red, yellow};
use crate::result::{
    Diagnostics, EffectEstimate, Exploitability, MeasurementQuality, Outcome, PreflightCategory,
    PreflightSeverity, ResearchOutcome, ResearchStatus,
};

/// Separator line used in output.
pub const SEPARATOR: &str = "──────────────────────────────────────────────────────────────";

/// Default terminal width for text wrapping.
const DEFAULT_WRAP_WIDTH: usize = 72;

// ============================================================================
// Main formatting functions
// ============================================================================

/// Format an Outcome for human-readable output (no colors).
///
/// This produces the full output including all diagnostic sections.
pub fn format_outcome_plain(outcome: &Outcome) -> String {
    let mut out = String::new();

    writeln!(out, "tacet").unwrap();
    writeln!(out, "{}", SEPARATOR).unwrap();
    writeln!(out).unwrap();

    match outcome {
        Outcome::Pass {
            leak_probability,
            effect,
            samples_used,
            quality,
            diagnostics,
            ..
        } => {
            format_header(&mut out, *samples_used, *quality);
            writeln!(out, "  {}", bold_green("\u{2713} No timing leak detected")).unwrap();
            writeln!(out).unwrap();
            format_pass_body(&mut out, *leak_probability, effect);
            format_preflight_notes(&mut out, diagnostics);
            format_diagnostics_section(&mut out, diagnostics);
            format_reproduction_line(&mut out, diagnostics);
            format_debug_environment(&mut out, diagnostics);
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
            format_header(&mut out, *samples_used, *quality);
            writeln!(out, "  {}", bold_yellow("\u{26A0} Timing leak detected")).unwrap();
            writeln!(out).unwrap();
            format_fail_body(&mut out, *leak_probability, effect, *exploitability);
            format_preflight_notes(&mut out, diagnostics);
            format_diagnostics_section(&mut out, diagnostics);
            format_reproduction_line(&mut out, diagnostics);
            format_debug_environment(&mut out, diagnostics);
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
            format_header(&mut out, *samples_used, *quality);
            writeln!(out, "  {}", bold_cyan("? Inconclusive")).unwrap();
            writeln!(out, "    {}", reason).unwrap();
            writeln!(out).unwrap();
            format_inconclusive_body(&mut out, *leak_probability, effect);
            format_inconclusive_diagnostics(&mut out, diagnostics);
            format_preflight_validation(&mut out, diagnostics);
            format_diagnostics_section(&mut out, diagnostics);
            format_reproduction_line(&mut out, diagnostics);
            format_debug_environment(&mut out, diagnostics);
        }

        Outcome::Unmeasurable {
            operation_ns,
            threshold_ns,
            platform,
            recommendation,
        } => {
            writeln!(
                out,
                "  {}",
                bold_yellow("\u{26A0} Operation too fast to measure reliably")
            )
            .unwrap();
            writeln!(out).unwrap();
            writeln!(out, "    Estimated duration: ~{:.1} ns", operation_ns).unwrap();
            writeln!(out, "    Minimum measurable: ~{:.1} ns", threshold_ns).unwrap();
            writeln!(out, "    Platform: {}", platform).unwrap();
            writeln!(out).unwrap();
            writeln!(out, "    Recommendation: {}", recommendation).unwrap();
            writeln!(out).unwrap();
            writeln!(out, "{}", SEPARATOR).unwrap();
            write!(
                out,
                "Note: Results are unmeasurable at this resolution; no leak probability is reported."
            )
            .unwrap();
            return out;
        }

        Outcome::Research(research) => {
            format_research_outcome(&mut out, research);
        }
    }

    writeln!(out).unwrap();
    writeln!(out, "{}", SEPARATOR).unwrap();

    if matches!(outcome, Outcome::Fail { .. }) {
        write!(
            out,
            "Note: Exploitability is a heuristic estimate based on effect magnitude."
        )
        .unwrap();
    }

    out
}

/// Format a compact debug summary (no colors).
pub fn format_debug_summary_plain(outcome: &Outcome) -> String {
    let mut out = String::new();

    writeln!(
        out,
        "\u{250C}\u{2500} Debug Summary {}",
        "\u{2500}".repeat(40)
    )
    .unwrap();

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
            let outcome_type = match outcome {
                Outcome::Pass { .. } => "PASS",
                Outcome::Fail { .. } => "FAIL",
                Outcome::Inconclusive { .. } => "INCONCLUSIVE",
                _ => unreachable!(),
            };
            writeln!(out, "\u{2502} Outcome = {}", outcome_type).unwrap();
            format_debug_core_metrics(
                &mut out,
                *leak_probability,
                effect,
                *quality,
                *samples_used,
                diagnostics,
            );
            format_debug_warnings(&mut out, diagnostics);
            format_debug_diagnostics(&mut out, diagnostics);
        }

        Outcome::Unmeasurable {
            operation_ns,
            threshold_ns,
            platform,
            recommendation,
        } => {
            writeln!(out, "\u{2502} Outcome = UNMEASURABLE").unwrap();
            writeln!(out, "\u{2502}   Operation: ~{:.1}ns", operation_ns).unwrap();
            writeln!(out, "\u{2502}   Threshold: ~{:.1}ns", threshold_ns).unwrap();
            writeln!(out, "\u{2502}   Platform: {}", platform).unwrap();
            writeln!(out, "\u{2502}   Tip: {}", recommendation).unwrap();
        }

        Outcome::Research(research) => {
            format_debug_research(&mut out, research);
        }
    }

    write!(out, "\u{2514}{}", "\u{2500}".repeat(55)).unwrap();

    out
}

// ============================================================================
// Section formatting helpers
// ============================================================================

fn format_header(out: &mut String, samples_used: usize, quality: MeasurementQuality) {
    writeln!(out, "  Samples: {} per class", samples_used).unwrap();
    writeln!(out, "  Quality: {}", format_quality_colored(quality)).unwrap();
    writeln!(out).unwrap();
}

/// Format quality with colors (when std feature is enabled).
fn format_quality_colored(quality: MeasurementQuality) -> String {
    match quality {
        MeasurementQuality::Excellent => green("Excellent"),
        MeasurementQuality::Good => green("Good"),
        MeasurementQuality::Poor => yellow("Poor"),
        MeasurementQuality::TooNoisy => yellow("too noisy"),
    }
}

fn format_pass_body(out: &mut String, leak_probability: f64, effect: &EffectEstimate) {
    writeln!(
        out,
        "    Probability of leak: {:.1}%",
        leak_probability * 100.0
    )
    .unwrap();
    writeln!(
        out,
        "    95% CI: {:.1}\u{2013}{:.1} ns",
        effect.credible_interval_ns.0, effect.credible_interval_ns.1
    )
    .unwrap();
}

fn format_fail_body(
    out: &mut String,
    leak_probability: f64,
    effect: &EffectEstimate,
    exploitability: Exploitability,
) {
    writeln!(
        out,
        "    Probability of leak: {:.1}%",
        leak_probability * 100.0
    )
    .unwrap();

    writeln!(
        out,
        "    Max effect: {:.1} ns [CI: {:.1}\u{2013}{:.1}]",
        effect.max_effect_ns,
        effect.credible_interval_ns.0,
        effect.credible_interval_ns.1
    )
    .unwrap();

    // Show top quantiles if available
    if !effect.top_quantiles.is_empty() {
        writeln!(out, "      Hotspots:").unwrap();
        for tq in &effect.top_quantiles {
            writeln!(
                out,
                "        \u{2192} p{:.0} ({:.1}ns, {:.0}% confident)",
                tq.quantile_p * 100.0,
                tq.mean_ns,
                tq.exceed_prob * 100.0
            )
            .unwrap();
        }
    }

    writeln!(out).unwrap();
    writeln!(out, "    Exploitability (heuristic):").unwrap();
    let (vector, queries) = exploitability_info_colored(exploitability);
    writeln!(out, "      Attack vector:  {}", vector).unwrap();
    writeln!(out, "      Queries needed: {}", queries).unwrap();
}

fn format_inconclusive_body(out: &mut String, leak_probability: f64, effect: &EffectEstimate) {
    writeln!(
        out,
        "    Current probability of leak: {:.1}%",
        leak_probability * 100.0
    )
    .unwrap();

    writeln!(
        out,
        "    Max effect estimate: {:.1} ns [CI: {:.1}\u{2013}{:.1}]",
        effect.max_effect_ns,
        effect.credible_interval_ns.0,
        effect.credible_interval_ns.1
    )
    .unwrap();
}

fn format_research_outcome(out: &mut String, research: &ResearchOutcome) {
    writeln!(out, "  Samples: {} per class", research.samples_used).unwrap();
    writeln!(
        out,
        "  Quality: {}",
        format_quality_colored(research.quality)
    )
    .unwrap();
    writeln!(out).unwrap();

    let status_line = match &research.status {
        ResearchStatus::EffectDetected => bold_green("\u{1F50D} Effect Detected"),
        ResearchStatus::NoEffectDetected => bold_cyan("\u{2713} No Effect Detected"),
        ResearchStatus::ResolutionLimitReached => bold_yellow("\u{26A0} Resolution Limit Reached"),
        ResearchStatus::QualityIssue(_) => bold_yellow("? Quality Issue"),
        ResearchStatus::BudgetExhausted => bold_cyan("? Budget Exhausted"),
    };
    writeln!(out, "  {}", status_line).unwrap();
    writeln!(out).unwrap();

    writeln!(out, "    Max effect: {:.1} ns", research.max_effect_ns).unwrap();
    writeln!(
        out,
        "    95% CI: [{:.1}, {:.1}] ns",
        research.max_effect_ci.0, research.max_effect_ci.1
    )
    .unwrap();
    writeln!(out, "    Measurement floor: {:.1} ns", research.theta_floor).unwrap();
    writeln!(
        out,
        "    Detectable: {}",
        if research.detectable { "yes" } else { "no" }
    )
    .unwrap();

    writeln!(out).unwrap();
    writeln!(
        out,
        "    Max effect: {:.1} ns [CI: {:.1}\u{2013}{:.1}]",
        research.effect.max_effect_ns,
        research.effect.credible_interval_ns.0,
        research.effect.credible_interval_ns.1
    )
    .unwrap();

    // Show top quantiles if available
    if !research.effect.top_quantiles.is_empty() {
        writeln!(out, "      Hotspots:").unwrap();
        for tq in &research.effect.top_quantiles {
            writeln!(
                out,
                "        \u{2192} p{:.0} ({:.1}ns, {:.0}% confident)",
                tq.quantile_p * 100.0,
                tq.mean_ns,
                tq.exceed_prob * 100.0
            )
            .unwrap();
        }
    }

    if research.model_mismatch {
        writeln!(out).unwrap();
        writeln!(
            out,
            "    {} Model mismatch detected \u{2013} interpret with caution",
            yellow("\u{26A0}")
        )
        .unwrap();
    }

    // Include diagnostics
    format_diagnostics_section(out, &research.diagnostics);
    format_reproduction_line(out, &research.diagnostics);
    format_debug_environment(out, &research.diagnostics);
}

// ============================================================================
// Diagnostics formatting
// ============================================================================

/// Format the detailed diagnostics section.
pub fn format_diagnostics_section(out: &mut String, diagnostics: &Diagnostics) {
    writeln!(out).unwrap();
    writeln!(out, "{}", SEPARATOR).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "  Measurement Diagnostics").unwrap();
    writeln!(out).unwrap();

    writeln!(
        out,
        "    Dependence:   block length {} (ESS: {} / {} raw)",
        diagnostics.dependence_length,
        diagnostics.effective_sample_size,
        diagnostics.calibration_samples
    )
    .unwrap();

    let stationarity_status = if diagnostics.stationarity_ok {
        green("OK")
    } else {
        red("Suspect")
    };
    writeln!(
        out,
        "    Stationarity: {:.2}x ({})",
        diagnostics.stationarity_ratio, stationarity_status
    )
    .unwrap();

    let mut outlier_line = format!(
        "    Outliers:     baseline {:.2}%, sample {:.2}%",
        diagnostics.outlier_rate_baseline * 100.0,
        diagnostics.outlier_rate_sample * 100.0
    );
    if !diagnostics.outlier_asymmetry_ok {
        outlier_line.push_str(&format!(" ({})", red("asymmetric")));
    }
    writeln!(out, "{}", outlier_line).unwrap();

    writeln!(
        out,
        "    Calibration:  {} samples",
        diagnostics.calibration_samples
    )
    .unwrap();

    writeln!(out, "    Runtime:      {:.1}s", diagnostics.total_time_secs).unwrap();

    // Warnings
    if !diagnostics.warnings.is_empty() {
        writeln!(out).unwrap();
        writeln!(out, "  {} Warnings", yellow("\u{26A0}")).unwrap();
        for warning in &diagnostics.warnings {
            writeln!(out, "    \u{2022} {}", warning).unwrap();
        }
    }

    // Quality issues with guidance
    if !diagnostics.quality_issues.is_empty() {
        writeln!(out).unwrap();
        writeln!(out, "  {} Quality Issues", yellow("\u{26A0}")).unwrap();
        for issue in &diagnostics.quality_issues {
            let wrapped_msg = wrap_text(
                &issue.message,
                DEFAULT_WRAP_WIDTH,
                20,
                "                    ",
            );
            writeln!(
                out,
                "    \u{2022} {}: {}",
                bold(&format!("{:?}", issue.code)),
                wrapped_msg
            )
            .unwrap();
            let wrapped_guidance = wrap_text(&issue.guidance, DEFAULT_WRAP_WIDTH, 8, "        ");
            writeln!(out, "      \u{2192} {}", dim(&wrapped_guidance)).unwrap();
        }
    }
}

/// Format reproduction line for output.
pub fn format_reproduction_line(out: &mut String, diagnostics: &Diagnostics) {
    let mut parts = Vec::new();

    if let Some(ref model) = diagnostics.attacker_model {
        parts.push(format!("model={}", model));
    }
    if diagnostics.threshold_ns > 0.0 {
        parts.push(format!("\u{03B8}={:.0}ns", diagnostics.threshold_ns));
    }
    if !diagnostics.timer_name.is_empty() {
        parts.push(format!("timer={}", diagnostics.timer_name));
    }

    if !parts.is_empty() {
        writeln!(out).unwrap();
        writeln!(out, "  Reproduce: {}", parts.join(", ")).unwrap();
    }
}

/// Format extended debug environment information.
pub fn format_debug_environment(out: &mut String, diagnostics: &Diagnostics) {
    writeln!(out).unwrap();
    writeln!(out, "{}", SEPARATOR).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "  Debug Information").unwrap();
    writeln!(out).unwrap();

    // Environment
    writeln!(out, "    Environment:").unwrap();
    let platform = if diagnostics.platform.is_empty() {
        #[cfg(feature = "std")]
        {
            format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
        }
        #[cfg(not(feature = "std"))]
        {
            "unknown".to_string()
        }
    } else {
        diagnostics.platform.clone()
    };
    writeln!(out, "      Platform:       {}", platform).unwrap();
    writeln!(out, "      Rust version:   1.80").unwrap();
    writeln!(out, "      Package:        tacet v0.1.5").unwrap();

    // Configuration
    writeln!(out).unwrap();
    writeln!(out, "    Configuration:").unwrap();
    if let Some(ref model) = diagnostics.attacker_model {
        writeln!(out, "      Attacker model: {}", model).unwrap();
    }
    writeln!(
        out,
        "      Threshold (\u{03B8}):  {:.1} ns",
        diagnostics.threshold_ns
    )
    .unwrap();
    if !diagnostics.timer_name.is_empty() {
        // Format timer line with fallback reason if present
        let timer_display = match &diagnostics.timer_fallback_reason {
            Some(reason) => format!("{} (fallback: {})", diagnostics.timer_name, reason),
            None => diagnostics.timer_name.clone(),
        };
        writeln!(out, "      Timer:          {}", timer_display).unwrap();
    }
    writeln!(
        out,
        "      Resolution:     {:.1} ns",
        diagnostics.timer_resolution_ns
    )
    .unwrap();
    if diagnostics.discrete_mode {
        writeln!(out, "      Discrete mode:  enabled").unwrap();
    }

    // Statistical summary
    writeln!(out).unwrap();
    writeln!(out, "    Statistical Summary:").unwrap();
    writeln!(
        out,
        "      Calibration:    {} samples",
        diagnostics.calibration_samples
    )
    .unwrap();
    writeln!(
        out,
        "      Block length:   {}",
        diagnostics.dependence_length
    )
    .unwrap();
    writeln!(
        out,
        "      ESS:            {}",
        diagnostics.effective_sample_size
    )
    .unwrap();
    writeln!(
        out,
        "      Stationarity:   {:.2}x {}",
        diagnostics.stationarity_ratio,
        if diagnostics.stationarity_ok {
            "OK"
        } else {
            "Suspect"
        }
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    For bug reports, re-run with TIMING_ORACLE_DEBUG=1 and include both outputs"
    )
    .unwrap();
}

/// Format preflight validation section for verbose output.
pub fn format_preflight_validation(out: &mut String, diagnostics: &Diagnostics) {
    writeln!(out).unwrap();
    writeln!(out, "{}", SEPARATOR).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "  Preflight Checks").unwrap();

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
    writeln!(out).unwrap();
    writeln!(out, "    Result Integrity:").unwrap();

    if sanity.is_empty() {
        writeln!(out, "      Sanity (F-vs-F):    OK").unwrap();
    } else {
        for w in &sanity {
            writeln!(
                out,
                "      Sanity (F-vs-F):    {}",
                format_severity(w.severity)
            )
            .unwrap();
            writeln!(out, "        {}", w.message).unwrap();
        }
    }

    if timer_sanity.is_empty() {
        writeln!(out, "      Timer monotonic:    OK").unwrap();
    } else {
        for w in &timer_sanity {
            writeln!(
                out,
                "      Timer monotonic:    {}",
                format_severity(w.severity)
            )
            .unwrap();
            writeln!(out, "        {}", w.message).unwrap();
        }
    }

    let stationarity_status = if diagnostics.stationarity_ok {
        format!("OK {:.2}x", diagnostics.stationarity_ratio)
    } else {
        format!("Suspect {:.2}x", diagnostics.stationarity_ratio)
    };
    writeln!(out, "      Stationarity:       {}", stationarity_status).unwrap();

    // Sampling Efficiency section
    writeln!(out).unwrap();
    writeln!(out, "    Sampling Efficiency:").unwrap();

    if autocorr.is_empty() {
        writeln!(out, "      Autocorrelation:    OK").unwrap();
    } else {
        for w in &autocorr {
            writeln!(
                out,
                "      Autocorrelation:    {}",
                format_severity(w.severity)
            )
            .unwrap();
            writeln!(out, "        {}", w.message).unwrap();
        }
    }

    let timer_name = if diagnostics.timer_name.is_empty() {
        String::new()
    } else {
        format!(" ({})", diagnostics.timer_name)
    };
    if resolution.is_empty() {
        writeln!(
            out,
            "      Timer resolution:   OK {:.1}ns{}",
            diagnostics.timer_resolution_ns, timer_name
        )
        .unwrap();
    } else {
        for w in &resolution {
            writeln!(
                out,
                "      Timer resolution:   {} {:.1}ns{}",
                format_severity(w.severity),
                diagnostics.timer_resolution_ns,
                timer_name
            )
            .unwrap();
            writeln!(out, "        {}", w.message).unwrap();
        }
    }

    // System Configuration section
    writeln!(out).unwrap();
    writeln!(out, "    System:").unwrap();
    if system.is_empty() {
        writeln!(out, "      Configuration:      OK").unwrap();
    } else {
        for w in &system {
            writeln!(out, "      \u{26A0} {}", w.message).unwrap();
            if let Some(guidance) = &w.guidance {
                writeln!(out, "        {}", guidance).unwrap();
            }
        }
    }
}

/// Format measurement notes for non-verbose output.
fn format_preflight_notes(out: &mut String, diagnostics: &Diagnostics) {
    if diagnostics.preflight_warnings.is_empty() {
        return;
    }

    let has_critical = diagnostics
        .preflight_warnings
        .iter()
        .any(|w| w.severity == PreflightSeverity::ResultUndermining);

    writeln!(out).unwrap();
    if has_critical {
        writeln!(out, "  \u{26A0} Measurement Notes:").unwrap();
    } else {
        writeln!(out, "  Measurement Notes:").unwrap();
    }

    for warning in &diagnostics.preflight_warnings {
        let bullet = match warning.severity {
            PreflightSeverity::ResultUndermining => "\u{2022}",
            PreflightSeverity::Informational => "\u{2022}",
        };
        writeln!(out, "    {} {}", bullet, warning.message).unwrap();

        // For resolution warnings, provide context-aware guidance based on fallback reason
        let guidance = if warning.category == PreflightCategory::Resolution {
            resolution_guidance_for_fallback(diagnostics.timer_fallback_reason.as_deref())
        } else {
            warning.guidance.clone()
        };

        if let Some(g) = guidance {
            writeln!(out, "      {}", g).unwrap();
        }
    }
}

/// Generate context-aware guidance for resolution warnings based on timer fallback reason.
fn resolution_guidance_for_fallback(fallback_reason: Option<&str>) -> Option<String> {
    match fallback_reason {
        Some("concurrent access") => Some(
            "Cycle-accurate timing is locked by another process. \
             If using cargo test, run with --test-threads=1."
                .into(),
        ),
        Some("no sudo") => {
            Some("Run with sudo to enable cycle-accurate timing (~0.3ns resolution).".into())
        }
        Some("unavailable") => Some(
            "Cycle-accurate timing unavailable. Consider increasing max_samples \
             or testing at a higher abstraction level."
                .into(),
        ),
        Some("user requested") => Some(
            "System timer was explicitly requested. For better resolution, \
             use TimerSpec::Auto or TimerSpec::RequireCycleAccurate."
                .into(),
        ),
        // x86_64 or already using cycle-accurate timer
        None => Some(
            "Timer resolution may be limiting measurement quality. \
             Consider increasing max_samples or time_budget."
                .into(),
        ),
        // Unknown fallback reason
        Some(_) => Some("Timer resolution may be limiting measurement quality.".into()),
    }
}

/// Format "Why This May Have Happened" section for Inconclusive outcomes.
fn format_inconclusive_diagnostics(out: &mut String, diagnostics: &Diagnostics) {
    let system_config: Vec<_> = diagnostics
        .preflight_warnings
        .iter()
        .filter(|w| w.category == PreflightCategory::System)
        .collect();
    let resolution: Vec<_> = diagnostics
        .preflight_warnings
        .iter()
        .filter(|w| w.category == PreflightCategory::Resolution)
        .collect();

    if system_config.is_empty() && resolution.is_empty() {
        return;
    }

    writeln!(out).unwrap();
    writeln!(out, "  \u{2139} Why This May Have Happened:").unwrap();

    if !system_config.is_empty() {
        writeln!(out).unwrap();
        writeln!(out, "    System Configuration:").unwrap();
        for warning in system_config {
            writeln!(out, "      \u{2022} {}", warning.message).unwrap();
        }
    }

    if !resolution.is_empty() {
        writeln!(out).unwrap();
        writeln!(out, "    Timer Resolution:").unwrap();
        for warning in resolution {
            writeln!(out, "      \u{2022} {}", warning.message).unwrap();
            if let Some(guidance) = &warning.guidance {
                writeln!(out, "      \u{2192} Tip: {}", guidance).unwrap();
            }
        }
    }
}

// ============================================================================
// Debug summary helpers
// ============================================================================

fn format_debug_core_metrics(
    out: &mut String,
    leak_probability: f64,
    effect: &EffectEstimate,
    quality: MeasurementQuality,
    samples_used: usize,
    diagnostics: &Diagnostics,
) {
    writeln!(out, "\u{2502} P(leak) = {:.1}%", leak_probability * 100.0).unwrap();
    writeln!(
        out,
        "\u{2502} Effect  = {:.1}ns (CI: [{:.1}, {:.1}])",
        effect.max_effect_ns, effect.credible_interval_ns.0, effect.credible_interval_ns.1
    )
    .unwrap();

    let ess = diagnostics.effective_sample_size;
    let efficiency = if samples_used > 0 {
        libm::round(ess as f64 / samples_used as f64 * 100.0) as usize
    } else {
        0
    };
    writeln!(
        out,
        "\u{2502} Quality = {} (ESS: {} / {} raw, {}%)",
        format_quality_colored(quality),
        ess,
        samples_used,
        efficiency
    )
    .unwrap();
}

fn format_debug_warnings(out: &mut String, diagnostics: &Diagnostics) {
    if diagnostics.warnings.is_empty() && diagnostics.quality_issues.is_empty() {
        return;
    }

    writeln!(out, "\u{2502}").unwrap();
    writeln!(out, "\u{2502} {} Warnings:", yellow("\u{26A0}")).unwrap();

    for warning in &diagnostics.warnings {
        writeln!(out, "\u{2502}   \u{2022} {}", warning).unwrap();
    }
    for issue in &diagnostics.quality_issues {
        writeln!(
            out,
            "\u{2502}   \u{2022} {:?}: {}",
            issue.code, issue.message
        )
        .unwrap();
    }
}

fn format_debug_diagnostics(out: &mut String, diagnostics: &Diagnostics) {
    writeln!(out, "\u{2502}").unwrap();
    writeln!(out, "\u{2502} Diagnostics:").unwrap();

    // Format timer line with fallback reason if present
    let timer_suffix = match &diagnostics.timer_fallback_reason {
        Some(reason) => format!(" (fallback: {})", reason),
        None => {
            if diagnostics.discrete_mode {
                " (discrete)".to_string()
            } else {
                String::new()
            }
        }
    };
    writeln!(
        out,
        "\u{2502}   Timer: {}{}",
        diagnostics.timer_name, timer_suffix
    )
    .unwrap();

    let stationarity_status = if diagnostics.stationarity_ok {
        format!("{:.1}x", diagnostics.stationarity_ratio)
    } else {
        format!("{:.1}x {}", diagnostics.stationarity_ratio, red("DRIFT"))
    };
    writeln!(out, "\u{2502}   Stationarity: {}", stationarity_status).unwrap();

    let outlier_note = if !diagnostics.outlier_asymmetry_ok {
        format!(" ({})", red("asymmetric!"))
    } else {
        String::new()
    };
    writeln!(
        out,
        "\u{2502}   Outliers: {:.1}% / {:.1}%{}",
        diagnostics.outlier_rate_baseline * 100.0,
        diagnostics.outlier_rate_sample * 100.0,
        outlier_note
    )
    .unwrap();

    writeln!(
        out,
        "\u{2502}   Runtime: {:.1}s",
        diagnostics.total_time_secs
    )
    .unwrap();
}

fn format_debug_research(out: &mut String, research: &ResearchOutcome) {
    let status_str = match &research.status {
        ResearchStatus::EffectDetected => green("Effect Detected"),
        ResearchStatus::NoEffectDetected => green("No Effect Detected"),
        ResearchStatus::ResolutionLimitReached => yellow("Resolution Limit"),
        ResearchStatus::QualityIssue(_) => yellow("Quality Issue"),
        ResearchStatus::BudgetExhausted => yellow("Budget Exhausted"),
    };
    writeln!(out, "\u{2502} Outcome = RESEARCH").unwrap();
    writeln!(out, "\u{2502} Status = {}", status_str).unwrap();
    writeln!(
        out,
        "\u{2502} Max Effect = {:.1}ns (CI: [{:.1}, {:.1}])",
        research.max_effect_ns, research.max_effect_ci.0, research.max_effect_ci.1
    )
    .unwrap();
    writeln!(
        out,
        "\u{2502} Floor = {:.1}ns, Detectable = {}",
        research.theta_floor,
        if research.detectable { "yes" } else { "no" }
    )
    .unwrap();
    writeln!(
        out,
        "\u{2502} Effect = {:.1}ns (CI: [{:.1}, {:.1}])",
        research.effect.max_effect_ns,
        research.effect.credible_interval_ns.0,
        research.effect.credible_interval_ns.1
    )
    .unwrap();

    let ess = research.diagnostics.effective_sample_size;
    let raw = research.samples_used;
    let efficiency = if raw > 0 {
        libm::round(ess as f64 / raw as f64 * 100.0) as usize
    } else {
        0
    };
    writeln!(
        out,
        "\u{2502} Quality = {} (ESS: {} / {} raw, {}%)",
        format_quality_colored(research.quality),
        ess,
        raw,
        efficiency
    )
    .unwrap();

    if research.model_mismatch {
        writeln!(out, "\u{2502}").unwrap();
        writeln!(
            out,
            "\u{2502} {} Model mismatch detected",
            yellow("\u{26A0}")
        )
        .unwrap();
    }

    format_debug_warnings(out, &research.diagnostics);

    writeln!(out, "\u{2502}").unwrap();
    writeln!(out, "\u{2502} Diagnostics:").unwrap();
    writeln!(
        out,
        "\u{2502}   Timer: {:.1}ns resolution",
        research.diagnostics.timer_resolution_ns
    )
    .unwrap();
    writeln!(
        out,
        "\u{2502}   Runtime: {:.1}s",
        research.diagnostics.total_time_secs
    )
    .unwrap();
}

// ============================================================================
// Utility functions
// ============================================================================

/// Get exploitability info as plain text.
pub fn exploitability_info(exploit: Exploitability) -> (&'static str, &'static str) {
    match exploit {
        Exploitability::SharedHardwareOnly => {
            ("Shared hardware (SGX, containers)", "~1k on same core")
        }
        Exploitability::Http2Multiplexing => ("HTTP/2 multiplexing", "~100k concurrent"),
        Exploitability::StandardRemote => ("Standard remote timing", "~1k-10k"),
        Exploitability::ObviousLeak => ("Any (trivially observable)", "<100"),
    }
}

/// Get exploitability info with colors.
fn exploitability_info_colored(exploit: Exploitability) -> (String, String) {
    match exploit {
        Exploitability::SharedHardwareOnly => (
            green("Shared hardware (SGX, containers)"),
            green("~1k on same core"),
        ),
        Exploitability::Http2Multiplexing => {
            (yellow("HTTP/2 multiplexing"), yellow("~100k concurrent"))
        }
        Exploitability::StandardRemote => (red("Standard remote timing"), red("~1k-10k")),
        Exploitability::ObviousLeak => (bold_red("Any (trivially observable)"), bold_red("<100")),
    }
}

fn format_severity(severity: PreflightSeverity) -> &'static str {
    match severity {
        PreflightSeverity::ResultUndermining => "WARNING",
        PreflightSeverity::Informational => "INFO",
    }
}

/// Wrap text to fit within a given width.
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
