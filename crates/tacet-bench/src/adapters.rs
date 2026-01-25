//! Tool adapters for benchmark comparison.
//!
//! This module provides a unified interface for comparing tacet against
//! other timing side-channel detection tools (dudect, RTLF, etc.).
//!
//! # Design
//!
//! Each tool gets data in its preferred format:
//! - tacet: interleaved samples (exploits temporal correlations)
//! - dudect/RTLF: blocked samples (baseline first, then test)
//!
//! The `ToolAdapter` trait provides a common interface for analysis, and
//! each adapter handles format conversion if needed.

use crate::{BlockedData, GeneratedDataset};
use rand::prelude::*;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tacet::{AttackerModel, Class, Outcome, TimingOracle};

/// Standardized outcome category for cross-tool comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeCategory {
    /// Tool determined no timing leak exists.
    Pass,
    /// Tool determined a timing leak exists.
    Fail,
    /// Tool could not reach a decision (insufficient samples, unmeasurable, etc.).
    Inconclusive,
    /// An error occurred during analysis.
    Error,
}

impl OutcomeCategory {
    /// Convert to lowercase string for CSV output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Inconclusive => "inconclusive",
            Self::Error => "error",
        }
    }
}

/// Result from a timing analysis tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Whether the tool detected a timing leak.
    pub detected_leak: bool,
    /// Number of samples used in the analysis.
    pub samples_used: usize,
    /// Time taken for analysis in milliseconds.
    pub decision_time_ms: u64,
    /// Raw leak probability (if available).
    pub leak_probability: Option<f64>,
    /// Tool-specific status message.
    pub status: String,
    /// Standardized outcome category for cross-tool comparison.
    pub outcome: OutcomeCategory,
}

/// Common interface for timing side-channel detection tools.
pub trait ToolAdapter: Send + Sync {
    /// Get the tool's name.
    fn name(&self) -> &str;

    /// Analyze blocked data (baseline then test).
    ///
    /// This is the baseline interface that all tools support.
    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult;

    /// Analyze interleaved data.
    ///
    /// Default implementation converts to blocked format.
    /// Tools that benefit from interleaved data should override this.
    fn analyze_interleaved(&self, data: &[(Class, u64)]) -> ToolResult {
        let blocked = split_interleaved(data);
        self.analyze_blocked(&blocked)
    }

    /// Analyze a complete generated dataset.
    ///
    /// Uses interleaved data if the tool supports it, otherwise blocked.
    fn analyze(&self, dataset: &GeneratedDataset) -> ToolResult {
        if self.uses_interleaved() {
            self.analyze_interleaved(&dataset.interleaved)
        } else {
            self.analyze_blocked(&dataset.blocked)
        }
    }

    /// Whether this tool benefits from interleaved data.
    fn uses_interleaved(&self) -> bool {
        false
    }

    /// Whether this tool supports configurable attacker models.
    ///
    /// Only tacet currently supports this. Other tools ignore
    /// the attacker_model parameter.
    fn supports_attacker_model(&self) -> bool {
        false
    }

    /// Analyze with a specific attacker model.
    ///
    /// For tools that support attacker models (tacet), this overrides
    /// the default attacker model for this single analysis.
    /// For other tools, this is equivalent to `analyze()`.
    fn analyze_with_attacker_model(
        &self,
        dataset: &GeneratedDataset,
        _attacker_model: Option<AttackerModel>,
    ) -> ToolResult {
        // Default: ignore attacker_model, use regular analyze
        self.analyze(dataset)
    }
}

/// Convert interleaved data to blocked format.
pub fn split_interleaved(data: &[(Class, u64)]) -> BlockedData {
    let mut baseline = Vec::new();
    let mut test = Vec::new();

    for (class, value) in data {
        match class {
            Class::Baseline => baseline.push(*value),
            Class::Sample => test.push(*value),
        }
    }

    BlockedData { baseline, test }
}

// =============================================================================
// tacet adapter
// =============================================================================

/// Adapter for tacet.
///
/// Uses the full Bayesian analysis pipeline with 9D posterior estimation.
/// For benchmark comparisons, uses `analyze_raw_samples` which takes blocked data
/// but still applies tacet's superior statistical methodology.
#[derive(Debug, Clone)]
pub struct TimingOracleAdapter {
    /// Attacker model to use for analysis.
    pub attacker_model: AttackerModel,
    /// Time budget for analysis (optional, default 60s).
    pub time_budget: Option<Duration>,
    /// Maximum samples to use.
    pub max_samples: Option<usize>,
}

impl Default for TimingOracleAdapter {
    fn default() -> Self {
        Self {
            attacker_model: AttackerModel::AdjacentNetwork,
            time_budget: Some(Duration::from_secs(60)),
            max_samples: None,
        }
    }
}

impl TimingOracleAdapter {
    /// Create with a specific attacker model.
    pub fn with_attacker_model(model: AttackerModel) -> Self {
        Self {
            attacker_model: model,
            ..Default::default()
        }
    }

    /// Set time budget.
    pub fn time_budget(mut self, duration: Duration) -> Self {
        self.time_budget = Some(duration);
        self
    }

    /// Set max samples.
    pub fn max_samples(mut self, n: usize) -> Self {
        self.max_samples = Some(n);
        self
    }
}

impl ToolAdapter for TimingOracleAdapter {
    fn name(&self) -> &str {
        "tacet"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        let start = Instant::now();

        // Convert cycles to nanoseconds (assume ~1 cycle/ns for synthetic data)
        // The synthetic data generator uses log-normal values representing cycles
        let baseline_ns: Vec<f64> = data.baseline.iter().map(|&c| c as f64).collect();
        let test_ns: Vec<f64> = data.test.iter().map(|&c| c as f64).collect();

        let mut oracle = TimingOracle::for_attacker(self.attacker_model);

        if let Some(budget) = self.time_budget {
            oracle = oracle.time_budget(budget);
        }
        if let Some(max) = self.max_samples {
            oracle = oracle.max_samples(max);
        }

        let outcome = oracle.analyze_raw_samples(&baseline_ns, &test_ns);
        let elapsed_ms = start.elapsed().as_millis() as u64;

        match outcome {
            Outcome::Pass {
                leak_probability,
                samples_used,
                ..
            } => ToolResult {
                detected_leak: false,
                samples_used,
                decision_time_ms: elapsed_ms,
                leak_probability: Some(leak_probability),
                status: format!("Pass (P={:.1}%)", leak_probability * 100.0),
                outcome: OutcomeCategory::Pass,
            },
            Outcome::Fail {
                leak_probability,
                samples_used,
                exploitability,
                ..
            } => ToolResult {
                detected_leak: true,
                samples_used,
                decision_time_ms: elapsed_ms,
                leak_probability: Some(leak_probability),
                status: format!(
                    "Fail (P={:.1}%, {:?})",
                    leak_probability * 100.0,
                    exploitability
                ),
                outcome: OutcomeCategory::Fail,
            },
            Outcome::Inconclusive {
                leak_probability,
                samples_used,
                reason,
                ..
            } => {
                // Treat inconclusive as no leak for benchmark purposes
                // (conservative - minimizes FPR)
                ToolResult {
                    detected_leak: false,
                    samples_used,
                    decision_time_ms: elapsed_ms,
                    leak_probability: Some(leak_probability),
                    status: format!("Inconclusive: {:?}", reason),
                    outcome: OutcomeCategory::Inconclusive,
                }
            }
            Outcome::Unmeasurable { recommendation, .. } => ToolResult {
                detected_leak: false,
                samples_used: 0,
                decision_time_ms: elapsed_ms,
                leak_probability: None,
                status: format!("Unmeasurable: {}", recommendation),
                outcome: OutcomeCategory::Inconclusive,
            },
            Outcome::Research(research) => ToolResult {
                detected_leak: matches!(
                    research.status,
                    tacet::result::ResearchStatus::EffectDetected
                ),
                samples_used: research.samples_used,
                decision_time_ms: elapsed_ms,
                leak_probability: None,
                status: format!("Research: {:?}", research.status),
                outcome: if matches!(
                    research.status,
                    tacet::result::ResearchStatus::EffectDetected
                ) {
                    OutcomeCategory::Fail
                } else {
                    OutcomeCategory::Pass
                },
            },
        }
    }

    // Note: tacet currently uses blocked data for analyze_raw_samples.
    // A future enhancement could add analyze_interleaved_samples to preserve
    // temporal correlation information for covariance estimation.
    fn uses_interleaved(&self) -> bool {
        false // Use blocked for now; the statistical model is the main advantage
    }

    fn supports_attacker_model(&self) -> bool {
        true
    }

    fn analyze_with_attacker_model(
        &self,
        dataset: &GeneratedDataset,
        attacker_model: Option<AttackerModel>,
    ) -> ToolResult {
        // Use provided attacker model, or fall back to self's default
        let actual_model = attacker_model.unwrap_or(self.attacker_model);

        let start = Instant::now();

        let baseline_ns: Vec<f64> = dataset.blocked.baseline.iter().map(|&c| c as f64).collect();
        let test_ns: Vec<f64> = dataset.blocked.test.iter().map(|&c| c as f64).collect();

        let mut oracle = TimingOracle::for_attacker(actual_model);

        if let Some(budget) = self.time_budget {
            oracle = oracle.time_budget(budget);
        }
        if let Some(max) = self.max_samples {
            oracle = oracle.max_samples(max);
        }

        let outcome = oracle.analyze_raw_samples(&baseline_ns, &test_ns);
        let elapsed_ms = start.elapsed().as_millis() as u64;

        match outcome {
            Outcome::Pass {
                leak_probability,
                samples_used,
                ..
            } => ToolResult {
                detected_leak: false,
                samples_used,
                decision_time_ms: elapsed_ms,
                leak_probability: Some(leak_probability),
                status: format!("Pass (P={:.1}%)", leak_probability * 100.0),
                outcome: OutcomeCategory::Pass,
            },
            Outcome::Fail {
                leak_probability,
                samples_used,
                exploitability,
                ..
            } => ToolResult {
                detected_leak: true,
                samples_used,
                decision_time_ms: elapsed_ms,
                leak_probability: Some(leak_probability),
                status: format!(
                    "Fail (P={:.1}%, {:?})",
                    leak_probability * 100.0,
                    exploitability
                ),
                outcome: OutcomeCategory::Fail,
            },
            Outcome::Inconclusive {
                leak_probability,
                samples_used,
                reason,
                ..
            } => ToolResult {
                detected_leak: false,
                samples_used,
                decision_time_ms: elapsed_ms,
                leak_probability: Some(leak_probability),
                status: format!("Inconclusive: {:?}", reason),
                outcome: OutcomeCategory::Inconclusive,
            },
            Outcome::Unmeasurable { recommendation, .. } => ToolResult {
                detected_leak: false,
                samples_used: 0,
                decision_time_ms: elapsed_ms,
                leak_probability: None,
                status: format!("Unmeasurable: {}", recommendation),
                outcome: OutcomeCategory::Inconclusive,
            },
            Outcome::Research(research) => ToolResult {
                detected_leak: matches!(
                    research.status,
                    tacet::result::ResearchStatus::EffectDetected
                ),
                samples_used: research.samples_used,
                decision_time_ms: elapsed_ms,
                leak_probability: None,
                status: format!("Research: {:?}", research.status),
                outcome: if matches!(
                    research.status,
                    tacet::result::ResearchStatus::EffectDetected
                ) {
                    OutcomeCategory::Fail
                } else {
                    OutcomeCategory::Pass
                },
            },
        }
    }
}

// =============================================================================
// dudect adapter (Welch's t-test)
// =============================================================================

/// Adapter implementing dudect's statistical methodology.
///
/// Implements Welch's t-test directly, matching dudect's core algorithm.
/// This allows analyzing pre-collected timing data without requiring
/// the dudect binary.
///
/// Reference: "Dude, is my code constant time?" (Reparaz et al., 2016)
/// https://eprint.iacr.org/2016/1123.pdf
#[derive(Debug, Clone)]
pub struct DudectAdapter {
    /// Significance threshold for t-test (default: 4.5).
    ///
    /// From the dudect paper: "A t value larger than 4.5 provides strong
    /// statistical evidence that the distributions are different."
    pub t_threshold: f64,
}

impl Default for DudectAdapter {
    fn default() -> Self {
        Self {
            t_threshold: 4.5, // Standard dudect threshold
        }
    }
}

impl DudectAdapter {
    /// Create with custom t-test threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            t_threshold: threshold,
        }
    }

    /// Set t-test threshold.
    pub fn t_threshold(mut self, threshold: f64) -> Self {
        self.t_threshold = threshold;
        self
    }
}

impl ToolAdapter for DudectAdapter {
    fn name(&self) -> &str {
        "dudect"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        use crate::dudect_stats::update_ct_stats;

        let start = Instant::now();

        let n1 = data.baseline.len();
        let n2 = data.test.len();

        if n1 < 2 || n2 < 2 {
            return ToolResult {
                detected_leak: false,
                samples_used: n1 + n2,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: "Insufficient samples for t-test".to_string(),
                outcome: OutcomeCategory::Inconclusive,
            };
        }

        // Use the actual dudect statistical methodology with percentile cropping
        let (summary, _ctx) = update_ct_stats(None, &(data.baseline.clone(), data.test.clone()));
        let detected = summary.max_t.abs() > self.t_threshold;

        ToolResult {
            detected_leak: detected,
            samples_used: summary.sample_size,
            decision_time_ms: start.elapsed().as_millis() as u64,
            leak_probability: None, // dudect doesn't give probabilities
            status: format!(
                "max|t|={:.2}, tau={:.4}, threshold={:.1}",
                summary.max_t.abs(),
                summary.max_tau,
                self.t_threshold
            ),
            outcome: if detected {
                OutcomeCategory::Fail
            } else {
                OutcomeCategory::Pass
            },
        }
    }

    fn uses_interleaved(&self) -> bool {
        false // dudect uses blocked data
    }
}

// =============================================================================
// RTLF adapter (direct Rscript)
// =============================================================================

/// Adapter for RTLF (Dunsche et al., USENIX Security 2024).
///
/// Runs RTLF analysis directly. If using devenv, `rtlf` is available in PATH.
/// Otherwise, install R with packages: tidyverse, optparse, jsonlite.
///
/// **Note:** RTLF requires at least 100 samples per class for its bootstrap test.
/// Smaller datasets will fail with "Insufficient data" error.
#[derive(Debug, Clone)]
pub struct RtlfAdapter {
    /// Command to run RTLF (default: "rtlf" wrapper from devenv).
    pub command: String,
    /// Significance level alpha (default: 0.09 as in RTLF paper).
    pub alpha: f64,
}

impl Default for RtlfAdapter {
    fn default() -> Self {
        Self {
            // Default uses the rtlf wrapper from devenv
            command: "rtlf".to_string(),
            alpha: 0.09, // RTLF paper default
        }
    }
}

impl RtlfAdapter {
    /// Create with custom command/script path.
    pub fn with_command(cmd: impl Into<String>) -> Self {
        Self {
            command: cmd.into(),
            ..Default::default()
        }
    }

    /// Set significance level.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

impl ToolAdapter for RtlfAdapter {
    fn name(&self) -> &str {
        "rtlf"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        let start = Instant::now();

        // Create temporary directory for input/output
        let temp_dir = match tempfile::tempdir() {
            Ok(dir) => dir,
            Err(e) => {
                return ToolResult {
                    detected_leak: false,
                    samples_used: 0,
                    decision_time_ms: start.elapsed().as_millis() as u64,
                    leak_probability: None,
                    status: format!("Failed to create temp dir: {}", e),
                    outcome: OutcomeCategory::Error,
                };
            }
        };

        // Write input CSV in RTLF format
        let input_file = temp_dir.path().join("input.csv");
        let output_file = temp_dir.path().join("output.json");

        if let Err(e) = write_rtlf_csv(&input_file, data) {
            return ToolResult {
                detected_leak: false,
                samples_used: 0,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: format!("Failed to write input: {}", e),
                outcome: OutcomeCategory::Error,
            };
        }

        // Run RTLF
        let result = run_rtlf(&self.command, &input_file, &output_file, self.alpha);

        match result {
            Ok((detected, p_value)) => {
                let samples_used = data.baseline.len() + data.test.len();
                ToolResult {
                    detected_leak: detected,
                    samples_used,
                    decision_time_ms: start.elapsed().as_millis() as u64,
                    leak_probability: Some(1.0 - p_value),
                    status: format!("p={:.4}, alpha={:.2}", p_value, self.alpha),
                    outcome: if detected {
                        OutcomeCategory::Fail
                    } else {
                        OutcomeCategory::Pass
                    },
                }
            }
            Err(e) => ToolResult {
                detected_leak: false,
                samples_used: 0,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: format!("Error: {}", e),
                outcome: OutcomeCategory::Error,
            },
        }
    }

    fn uses_interleaved(&self) -> bool {
        false // RTLF uses blocked data
    }
}

/// Run RTLF analysis.
fn run_rtlf(
    command: &str,
    input_file: &Path,
    output_file: &Path,
    alpha: f64,
) -> Result<(bool, f64), String> {
    let output = Command::new(command)
        .arg("-i")
        .arg(input_file)
        .arg("-o")
        .arg(output_file)
        .arg("-a")
        .arg(format!("{}", alpha))
        .arg("-q") // Quiet mode
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run RTLF ({}): {}. Is rtlf in PATH?", command, e))?;

    // RTLF uses custom exit codes:
    // - 10: No difference detected (success, no leak)
    // - 11: Difference detected (success, leak found)
    // - 1: Error
    let exit_code = output.status.code().unwrap_or(1);

    if exit_code == 1 {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "RTLF error: stderr={}, stdout={}",
            stderr.trim(),
            stdout.trim()
        ));
    }

    // Exit codes 10 and 11 are success - try to parse JSON output
    if output_file.exists() {
        let json_content = std::fs::read_to_string(output_file)
            .map_err(|e| format!("Failed to read output: {}", e))?;
        parse_rtlf_json(&json_content, alpha)
    } else {
        // Fallback: interpret exit code directly
        // 11 = difference detected, 10 = no difference
        let detected = exit_code == 11;
        let p_value = if detected { 0.0 } else { 1.0 };
        Ok((detected, p_value))
    }
}

/// Parse RTLF JSON output.
fn parse_rtlf_json(json_content: &str, _alpha: f64) -> Result<(bool, f64), String> {
    // RTLF JSON structure:
    // {
    //   "metadata": {
    //     "difference_detected": true/false,
    //     "exit_code": 10 (no diff) or 11 (diff detected),
    //     "significant_deciles": [10, 20, ...],
    //     ...
    //   },
    //   ...
    // }
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_content) {
        // Check metadata.difference_detected first (most reliable)
        if let Some(metadata) = parsed.get("metadata") {
            if let Some(detected) = metadata
                .get("difference_detected")
                .and_then(|v| v.as_bool())
            {
                // RTLF doesn't output p-values, just the decision
                let p_value = if detected { 0.0 } else { 1.0 };
                return Ok((detected, p_value));
            }

            // Fallback to exit_code
            if let Some(exit_code) = metadata.get("exit_code").and_then(|v| v.as_i64()) {
                let detected = exit_code == 11;
                let p_value = if detected { 0.0 } else { 1.0 };
                return Ok((detected, p_value));
            }
        }

        // Legacy format: check for Decision field
        if let Some(decision) = parsed.get("Decision").or(parsed.get("decision")) {
            if let Some(decision_str) = decision.as_str() {
                let detected = decision_str.to_lowercase().contains("significant")
                    || decision_str.to_lowercase().contains("detected");
                let p_value = if detected { 0.0 } else { 1.0 };
                return Ok((detected, p_value));
            }
        }
    }

    Err("Could not parse RTLF JSON output".to_string())
}

// =============================================================================
// SILENT adapter
// =============================================================================

/// Adapter for SILENT (arXiv 2024).
///
/// SILENT uses bootstrap-based statistical testing with quantile analysis.
/// If using devenv, `silent` is available in PATH.
#[derive(Debug, Clone)]
pub struct SilentAdapter {
    /// Command to run SILENT (default: "silent" wrapper from devenv).
    pub command: String,
    /// False positive rate alpha (default: 0.1 as recommended).
    pub alpha: f64,
    /// Number of bootstrap samples (default: 1000).
    pub bootstrap_samples: usize,
    /// Minimum detectable effect size in cycles/ns (default: 100).
    pub delta: f64,
}

impl Default for SilentAdapter {
    fn default() -> Self {
        Self {
            command: "silent".to_string(),
            alpha: 0.1,
            bootstrap_samples: 1000,
            delta: 100.0,
        }
    }
}

impl SilentAdapter {
    /// Create with custom command/script path.
    pub fn with_command(cmd: impl Into<String>) -> Self {
        Self {
            command: cmd.into(),
            ..Default::default()
        }
    }

    /// Set false positive rate.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set bootstrap samples.
    pub fn bootstrap_samples(mut self, b: usize) -> Self {
        self.bootstrap_samples = b;
        self
    }

    /// Set minimum detectable effect size.
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }
}

impl ToolAdapter for SilentAdapter {
    fn name(&self) -> &str {
        "silent"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        let start = Instant::now();

        // Create temporary directory for input/output
        let temp_dir = match tempfile::tempdir() {
            Ok(dir) => dir,
            Err(e) => {
                return ToolResult {
                    detected_leak: false,
                    samples_used: 0,
                    decision_time_ms: start.elapsed().as_millis() as u64,
                    leak_probability: None,
                    status: format!("Failed to create temp dir: {}", e),
                    outcome: OutcomeCategory::Error,
                };
            }
        };

        // Write input CSV in SILENT format (same as RTLF)
        let input_file = temp_dir.path().join("input.csv");
        let output_dir = temp_dir.path();

        if let Err(e) = write_rtlf_csv(&input_file, data) {
            return ToolResult {
                detected_leak: false,
                samples_used: 0,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: format!("Failed to write input: {}", e),
                outcome: OutcomeCategory::Error,
            };
        }

        // Run SILENT
        let result = run_silent(
            &self.command,
            self.alpha,
            &input_file,
            output_dir,
            self.bootstrap_samples,
            self.delta,
        );

        match result {
            Ok((detected, stat)) => {
                let samples_used = data.baseline.len() + data.test.len();
                ToolResult {
                    detected_leak: detected,
                    samples_used,
                    decision_time_ms: start.elapsed().as_millis() as u64,
                    leak_probability: None, // SILENT doesn't output probabilities
                    status: format!(
                        "stat={:.3}, alpha={:.2}, delta={:.0}",
                        stat, self.alpha, self.delta
                    ),
                    outcome: if detected {
                        OutcomeCategory::Fail
                    } else {
                        OutcomeCategory::Pass
                    },
                }
            }
            Err(e) => ToolResult {
                detected_leak: false,
                samples_used: 0,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: format!("Error: {}", e),
                outcome: OutcomeCategory::Error,
            },
        }
    }

    fn uses_interleaved(&self) -> bool {
        false // SILENT uses blocked data
    }
}

/// Run SILENT analysis.
fn run_silent(
    command: &str,
    alpha: f64,
    input_file: &Path,
    output_dir: &Path,
    bootstrap_samples: usize,
    delta: f64,
) -> Result<(bool, f64), String> {
    // SILENT.R <alpha> <input CSV> <output folder> <B> <Delta>
    let output = Command::new(command)
        .arg(format!("{}", alpha))
        .arg(input_file)
        .arg(output_dir)
        .arg(format!("{}", bootstrap_samples))
        .arg(format!("{}", delta))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| {
            format!(
                "Failed to run SILENT ({}): {}. Is silent in PATH?",
                command, e
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("SILENT failed: {}", stderr));
    }

    // Parse JSON output file
    let base_name = input_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("input");
    let json_file = output_dir.join(format!("{}_summary_results.json", base_name));

    if json_file.exists() {
        let json_content = std::fs::read_to_string(&json_file)
            .map_err(|e| format!("Failed to read output: {}", e))?;
        parse_silent_json(&json_content)
    } else {
        // Fallback: parse stdout for decision
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.contains("Rejected") {
            Ok((true, 0.0))
        } else if stdout.contains("Failed to reject") {
            Ok((false, 0.0))
        } else {
            Err("Could not parse SILENT output".to_string())
        }
    }
}

/// Parse SILENT JSON output.
fn parse_silent_json(json_content: &str) -> Result<(bool, f64), String> {
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_content) {
        if let Some(test_result) = parsed.get("test_result") {
            let decision = test_result
                .get("decision")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let detected = decision.to_lowercase().contains("rejected")
                && !decision.to_lowercase().contains("failed");

            let stat = test_result
                .get("adjusted_statistic")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);

            return Ok((detected, stat));
        }
    }
    Err("Could not parse SILENT JSON output".to_string())
}

// =============================================================================
// RTLF adapter (Docker/Podman-based)
// =============================================================================

/// Adapter for RTLF using Docker or Podman container.
///
/// Use this if you don't want to install R locally.
#[derive(Debug, Clone)]
pub struct RtlfDockerAdapter {
    /// Docker image name.
    pub image: String,
    /// Significance level alpha (default: 0.09 as in RTLF paper).
    pub alpha: f64,
}

impl Default for RtlfDockerAdapter {
    fn default() -> Self {
        Self {
            image: "rtlf:latest".to_string(),
            alpha: 0.09, // RTLF paper default
        }
    }
}

impl RtlfDockerAdapter {
    /// Create with custom Docker image.
    pub fn with_image(image: impl Into<String>) -> Self {
        Self {
            image: image.into(),
            ..Default::default()
        }
    }

    /// Set significance level.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

impl ToolAdapter for RtlfDockerAdapter {
    fn name(&self) -> &str {
        "rtlf"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        let start = Instant::now();

        // Create temporary directory for input/output
        let temp_dir = std::env::temp_dir().join(format!("rtlf_{}", std::process::id()));
        if let Err(e) = std::fs::create_dir_all(&temp_dir) {
            return ToolResult {
                detected_leak: false,
                samples_used: 0,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: format!("Failed to create temp dir: {}", e),
                outcome: OutcomeCategory::Error,
            };
        }

        // Write input CSV in RTLF format
        let input_file = temp_dir.join("input.csv");
        if let Err(e) = write_rtlf_csv(&input_file, data) {
            let _ = std::fs::remove_dir_all(&temp_dir);
            return ToolResult {
                detected_leak: false,
                samples_used: 0,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: format!("Failed to write input: {}", e),
                outcome: OutcomeCategory::Error,
            };
        }

        // Run RTLF Docker container
        let result = run_rtlf_docker(&self.image, &temp_dir, self.alpha);

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);

        match result {
            Ok((detected, p_value)) => {
                let samples_used = data.baseline.len() + data.test.len();
                ToolResult {
                    detected_leak: detected,
                    samples_used,
                    decision_time_ms: start.elapsed().as_millis() as u64,
                    leak_probability: Some(1.0 - p_value), // Convert p-value to "leak probability"
                    status: format!("p={:.4}, alpha={:.2}", p_value, self.alpha),
                    outcome: if detected {
                        OutcomeCategory::Fail
                    } else {
                        OutcomeCategory::Pass
                    },
                }
            }
            Err(e) => ToolResult {
                detected_leak: false,
                samples_used: 0,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: format!("Error: {}", e),
                outcome: OutcomeCategory::Error,
            },
        }
    }

    fn uses_interleaved(&self) -> bool {
        false // RTLF uses blocked data
    }
}

/// Write data in RTLF CSV format.
fn write_rtlf_csv(path: &Path, data: &BlockedData) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    // RTLF expects: V1,V2 header, then CLASS,VALUE rows
    // Uses X/Y as class names (matching RTLF examples)
    writeln!(file, "V1,V2")?;

    for &value in &data.baseline {
        writeln!(file, "X,{}", value)?;
    }
    for &value in &data.test {
        writeln!(file, "Y,{}", value)?;
    }

    Ok(())
}

/// Run RTLF analysis in Docker container.
fn run_rtlf_docker(image: &str, work_dir: &Path, alpha: f64) -> Result<(bool, f64), String> {
    // Run Docker container with mounted volume
    let output = Command::new("docker")
        .arg("run")
        .arg("--rm")
        .arg("-v")
        .arg(format!("{}:/data", work_dir.display()))
        .arg(image)
        .arg("--input")
        .arg("/data/input.csv")
        .arg("--alpha")
        .arg(format!("{}", alpha))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run Docker: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("RTLF failed: {}", stderr));
    }

    // Parse RTLF output for result
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_rtlf_output(&stdout, alpha)
}

/// Parse RTLF output.
fn parse_rtlf_output(output: &str, alpha: f64) -> Result<(bool, f64), String> {
    // RTLF outputs p-value; leak detected if p < alpha
    for line in output.lines() {
        let line_lower = line.to_lowercase();
        if line_lower.contains("p-value") || line_lower.contains("pvalue") {
            // Extract p-value from line
            for word in line.split_whitespace() {
                if let Ok(p) = word
                    .trim_matches(|c: char| !c.is_numeric() && c != '.')
                    .parse::<f64>()
                {
                    if (0.0..=1.0).contains(&p) {
                        return Ok((p < alpha, p));
                    }
                }
            }
        }
    }

    // Try to find result file
    Err("Could not parse RTLF output".to_string())
}

// =============================================================================
// TVLA adapter (Test Vector Leakage Assessment)
// =============================================================================

/// Adapter for Timing-TVLA (simplified TVLA for timing side channels).
///
/// This is a simplified adaptation of TVLA (Test Vector Leakage Assessment)
/// for timing analysis. Unlike power TVLA which analyzes multi-dimensional
/// traces, this performs a single Welch's t-test on scalar timing measurements
/// with a threshold of |t| > 4.5.
///
/// **Important limitations** (see SILENT paper, arXiv:2504.19821):
/// - Assumes Gaussian distributions (timing data is typically log-normal)
/// - Assumes independent observations (sequential measurements are correlated)
/// - No formal false positive rate control (4.5 threshold is arbitrary)
///
/// For rigorous analysis, prefer RTLF, SILENT, or tacet which use
/// non-parametric methods with formal statistical guarantees.
///
/// Reference: "A testing methodology for side channel resistance validation"
/// (Goodwill et al., NIST 2011) - originally for power analysis
#[derive(Debug, Clone)]
pub struct TimingTvlaAdapter {
    /// Significance threshold for t-test (default: 4.5).
    pub t_threshold: f64,
}

impl Default for TimingTvlaAdapter {
    fn default() -> Self {
        Self {
            t_threshold: 4.5, // Standard TVLA threshold
        }
    }
}

impl TimingTvlaAdapter {
    /// Create with custom t-test threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            t_threshold: threshold,
        }
    }
}

impl ToolAdapter for TimingTvlaAdapter {
    fn name(&self) -> &str {
        "timing-tvla"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        let start = Instant::now();

        let n1 = data.baseline.len();
        let n2 = data.test.len();

        if n1 < 2 || n2 < 2 {
            return ToolResult {
                detected_leak: false,
                samples_used: n1 + n2,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: "Insufficient samples for t-test".to_string(),
                outcome: OutcomeCategory::Inconclusive,
            };
        }

        // Compute Welch's t-test (single test on all data, unlike DudeCT)
        let t_stat = welch_t_test(&data.baseline, &data.test);
        let detected = t_stat.abs() > self.t_threshold;

        ToolResult {
            detected_leak: detected,
            samples_used: n1 + n2,
            decision_time_ms: start.elapsed().as_millis() as u64,
            leak_probability: None, // TVLA doesn't give probabilities
            status: format!("|t|={:.2}, threshold={:.1}", t_stat.abs(), self.t_threshold),
            outcome: if detected {
                OutcomeCategory::Fail
            } else {
                OutcomeCategory::Pass
            },
        }
    }

    fn uses_interleaved(&self) -> bool {
        false
    }
}

/// Compute Welch's t-statistic for two samples.
fn welch_t_test(sample1: &[u64], sample2: &[u64]) -> f64 {
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    // Compute means
    let mean1: f64 = sample1.iter().map(|&x| x as f64).sum::<f64>() / n1;
    let mean2: f64 = sample2.iter().map(|&x| x as f64).sum::<f64>() / n2;

    // Compute variances
    let var1: f64 = sample1
        .iter()
        .map(|&x| (x as f64 - mean1).powi(2))
        .sum::<f64>()
        / (n1 - 1.0);
    let var2: f64 = sample2
        .iter()
        .map(|&x| (x as f64 - mean2).powi(2))
        .sum::<f64>()
        / (n2 - 1.0);

    // Welch's t-statistic
    let num = mean1 - mean2;
    let den = (var1 / n1 + var2 / n2).sqrt();

    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

// =============================================================================
// Kolmogorov-Smirnov two-sample test adapter
// =============================================================================

/// Adapter implementing the two-sample Kolmogorov-Smirnov test.
///
/// The KS test compares the empirical cumulative distribution functions (ECDFs)
/// of two samples. The test statistic D is the maximum absolute difference
/// between the two ECDFs.
///
/// This is a non-parametric test that makes no assumptions about the underlying
/// distributions. It is sensitive to differences in location, scale, and shape.
///
/// # References
///
/// - Kolmogorov, A. (1933). "Sulla determinazione empirica di una legge di distribuzione"
/// - Smirnov, N. (1948). "Table for estimating the goodness of fit of empirical distributions"
#[derive(Debug, Clone)]
pub struct KsTestAdapter {
    /// Significance level (default: 0.05).
    pub alpha: f64,
}

impl Default for KsTestAdapter {
    fn default() -> Self {
        Self { alpha: 0.05 }
    }
}

impl KsTestAdapter {
    /// Create with custom significance level.
    pub fn with_alpha(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl ToolAdapter for KsTestAdapter {
    fn name(&self) -> &str {
        "ks-test"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        let start = Instant::now();

        let n1 = data.baseline.len();
        let n2 = data.test.len();

        if n1 < 2 || n2 < 2 {
            return ToolResult {
                detected_leak: false,
                samples_used: n1 + n2,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: "Insufficient samples for KS test".to_string(),
                outcome: OutcomeCategory::Inconclusive,
            };
        }

        // Compute KS statistic and approximate p-value
        let (d_stat, p_value) = ks_two_sample(&data.baseline, &data.test);
        let detected = p_value < self.alpha;

        ToolResult {
            detected_leak: detected,
            samples_used: n1 + n2,
            decision_time_ms: start.elapsed().as_millis() as u64,
            leak_probability: Some(1.0 - p_value),
            status: format!("D={:.4}, p={:.4}, alpha={:.2}", d_stat, p_value, self.alpha),
            outcome: if detected {
                OutcomeCategory::Fail
            } else {
                OutcomeCategory::Pass
            },
        }
    }

    fn uses_interleaved(&self) -> bool {
        false
    }
}

/// Compute the two-sample Kolmogorov-Smirnov test statistic and p-value.
///
/// Returns (D, p-value) where D is the KS statistic.
fn ks_two_sample(sample1: &[u64], sample2: &[u64]) -> (f64, f64) {
    let n1 = sample1.len();
    let n2 = sample2.len();

    // Sort both samples
    let mut s1: Vec<f64> = sample1.iter().map(|&x| x as f64).collect();
    let mut s2: Vec<f64> = sample2.iter().map(|&x| x as f64).collect();
    s1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s2.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Merge all unique values to evaluate ECDFs
    let mut all_values: Vec<f64> = s1.iter().chain(s2.iter()).copied().collect();
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_values.dedup();

    // Compute maximum ECDF difference
    let mut d_max = 0.0f64;
    for &x in &all_values {
        // ECDF values at x
        let f1 = s1.iter().filter(|&&v| v <= x).count() as f64 / n1 as f64;
        let f2 = s2.iter().filter(|&&v| v <= x).count() as f64 / n2 as f64;
        let diff = (f1 - f2).abs();
        if diff > d_max {
            d_max = diff;
        }
    }

    // Compute approximate p-value using asymptotic distribution
    // The statistic D * sqrt(n1*n2/(n1+n2)) follows the Kolmogorov distribution
    let n = (n1 * n2) as f64 / (n1 + n2) as f64;
    let z = d_max * n.sqrt();

    // Approximate p-value using the asymptotic formula
    // P(D > d) ≈ 2 * sum_{k=1}^{inf} (-1)^{k-1} * exp(-2 * k^2 * z^2)
    let p_value = ks_p_value(z);

    (d_max, p_value)
}

/// Compute the p-value for the Kolmogorov distribution.
///
/// Uses the asymptotic series: P(K > z) = 2 * sum_{k=1}^{inf} (-1)^{k-1} * exp(-2*k^2*z^2)
fn ks_p_value(z: f64) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }
    if z > 3.0 {
        // For large z, p-value is essentially 0
        return 2.0 * (-2.0 * z * z).exp();
    }

    let mut p = 0.0;
    for k in 1..=100 {
        let term = (-2.0 * (k as f64).powi(2) * z * z).exp();
        if k % 2 == 1 {
            p += term;
        } else {
            p -= term;
        }
        if term < 1e-12 {
            break;
        }
    }
    (2.0 * p).clamp(0.0, 1.0)
}

// =============================================================================
// Anderson-Darling two-sample test adapter
// =============================================================================

/// Adapter implementing the two-sample Anderson-Darling test.
///
/// The Anderson-Darling test is similar to Kolmogorov-Smirnov but gives more
/// weight to the tails of the distribution, making it more sensitive to
/// differences in the tails (which is often where timing leaks manifest).
///
/// This implements the k-sample Anderson-Darling test for k=2.
///
/// # References
///
/// - Anderson, T. W.; Darling, D. A. (1952). "Asymptotic theory of certain
///   goodness-of-fit criteria based on stochastic processes"
/// - Scholz, F. W.; Stephens, M. A. (1987). "K-sample Anderson-Darling tests"
#[derive(Debug, Clone)]
pub struct AndersonDarlingAdapter {
    /// Significance level (default: 0.05).
    pub alpha: f64,
}

impl Default for AndersonDarlingAdapter {
    fn default() -> Self {
        Self { alpha: 0.05 }
    }
}

impl AndersonDarlingAdapter {
    /// Create with custom significance level.
    pub fn with_alpha(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl ToolAdapter for AndersonDarlingAdapter {
    fn name(&self) -> &str {
        "ad-test"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        let start = Instant::now();

        let n1 = data.baseline.len();
        let n2 = data.test.len();

        if n1 < 2 || n2 < 2 {
            return ToolResult {
                detected_leak: false,
                samples_used: n1 + n2,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: "Insufficient samples for AD test".to_string(),
                outcome: OutcomeCategory::Inconclusive,
            };
        }

        // Compute AD statistic and approximate p-value
        let (a2_stat, p_value) = ad_two_sample(&data.baseline, &data.test);
        let detected = p_value < self.alpha;

        ToolResult {
            detected_leak: detected,
            samples_used: n1 + n2,
            decision_time_ms: start.elapsed().as_millis() as u64,
            leak_probability: Some(1.0 - p_value),
            status: format!(
                "A²={:.4}, p={:.4}, alpha={:.2}",
                a2_stat, p_value, self.alpha
            ),
            outcome: if detected {
                OutcomeCategory::Fail
            } else {
                OutcomeCategory::Pass
            },
        }
    }

    fn uses_interleaved(&self) -> bool {
        false
    }
}

/// Compute the two-sample Anderson-Darling test statistic and approximate p-value.
///
/// Uses the Scholz-Stephens k-sample formulation for k=2.
fn ad_two_sample(sample1: &[u64], sample2: &[u64]) -> (f64, f64) {
    let n1 = sample1.len();
    let n2 = sample2.len();
    let n = n1 + n2;

    // Combine and sort all samples, tracking which sample each came from
    let mut combined: Vec<(f64, usize)> = sample1
        .iter()
        .map(|&x| (x as f64, 0))
        .chain(sample2.iter().map(|&x| (x as f64, 1)))
        .collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Shuffle tied elements to avoid bias from stable sort grouping sample1 before sample2.
    // Without this, heavily quantized data causes systematic deviation in cumulative counts.
    // Use deterministic seed based on data for reproducibility.
    let seed = sample1
        .iter()
        .take(10)
        .fold(0u64, |acc, &x| acc.wrapping_add(x));
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && combined[j].0 == combined[i].0 {
            j += 1;
        }
        // Shuffle elements within this tie group
        if j - i > 1 {
            combined[i..j].shuffle(&mut rng);
        }
        i = j;
    }

    // Handle ties by averaging ranks (midrank method)
    let mut ranks: Vec<(f64, usize)> = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && combined[j].0 == combined[i].0 {
            j += 1;
        }
        // Midrank for tied values
        let midrank = (i + j + 1) as f64 / 2.0;
        for &(_, idx) in combined[i..j].iter() {
            ranks.push((midrank, idx));
        }
        i = j;
    }

    // Compute the k-sample AD statistic (k=2)
    // A² = (1/N) * sum_{i=1}^{k} (1/n_i) * sum_{j=1}^{N-1} (M_ij - j*n_i/N)² / (j*(N-j))
    // where M_ij is the number of observations from sample i in the first j pooled observations

    let n1_f = n1 as f64;
    let n2_f = n2 as f64;
    let n_f = n as f64;

    // Count cumulative observations from each sample
    let mut m1 = 0.0; // cumulative count from sample 1
    let mut sum = 0.0;

    for (j, rank) in ranks.iter().take(n - 1).enumerate() {
        if rank.1 == 0 {
            m1 += 1.0;
        }
        let m2 = (j + 1) as f64 - m1;

        let j_f = (j + 1) as f64;
        let denom = j_f * (n_f - j_f);

        if denom > 0.0 {
            // Contribution from sample 1
            let expected1 = j_f * n1_f / n_f;
            let term1 = (m1 - expected1).powi(2) / denom;

            // Contribution from sample 2
            let expected2 = j_f * n2_f / n_f;
            let term2 = (m2 - expected2).powi(2) / denom;

            sum += term1 / n1_f + term2 / n2_f;
        }
    }

    let a2 = sum * n_f;

    // Compute approximate p-value using the asymptotic distribution
    // The standardized statistic follows approximately a known distribution
    // Use the Scholz-Stephens approximation for k=2
    let p_value = ad_p_value(a2, n1, n2);

    (a2, p_value)
}

/// Approximate p-value for the two-sample Anderson-Darling test.
///
/// Uses interpolation of critical values from Scholz & Stephens (1987).
fn ad_p_value(a2: f64, n1: usize, n2: usize) -> f64 {
    let n = n1 + n2;

    // Standardize the statistic
    // A²* = A² / (1 + 0.75/n + 2.25/n²) approximately
    let h = 1.0 + 0.75 / n as f64 + 2.25 / (n as f64).powi(2);
    let a2_star = a2 / h;

    // Approximate p-value using critical value interpolation
    // Critical values for α: 0.25, 0.10, 0.05, 0.025, 0.01
    // From Scholz & Stephens (1987) Table 1 for k=2
    let critical_values = [
        (0.25, 1.248),
        (0.10, 1.933),
        (0.05, 2.492),
        (0.025, 3.070),
        (0.01, 3.857),
        (0.005, 4.500),
        (0.001, 5.800),
    ];

    // Interpolate/extrapolate p-value
    if a2_star <= critical_values[0].1 {
        // Below smallest critical value - p > 0.25
        // Linear extrapolation
        return (0.5 - (a2_star / critical_values[0].1) * 0.25).clamp(0.25, 1.0);
    }

    for i in 0..critical_values.len() - 1 {
        let (p1, c1): (f64, f64) = critical_values[i];
        let (p2, c2): (f64, f64) = critical_values[i + 1];

        if a2_star <= c2 {
            // Linear interpolation in log-p space
            let t = (a2_star - c1) / (c2 - c1);
            let log_p = p1.ln() * (1.0 - t) + p2.ln() * t;
            return log_p.exp();
        }
    }

    // Beyond largest critical value - very small p
    // Exponential extrapolation
    let (p_last, c_last): (f64, f64) = critical_values[critical_values.len() - 1];
    let excess = a2_star - c_last;
    (p_last * (-excess * 0.5).exp()).clamp(0.0, p_last)
}

// =============================================================================
// Mona adapter (Crosby box test)
// =============================================================================

/// Adapter implementing Crosby's box test for timing side-channel detection.
///
/// The box test iterates over all possible percentile ranges [i%, j%] and checks
/// if the distributions are non-overlapping within any range. If `max(A) < min(B)`
/// or `max(B) < min(A)` for any percentile box, a timing leak is detected.
///
/// This is a pure Rust implementation matching the algorithm from:
///
/// # References
///
/// **Original paper:**
/// - S. A. Crosby, D. S. Wallach, and R. H. Riedi. "Opportunities and Limits of
///   Remote Timing Attacks." ACM Transactions on Information and System Security,
///   Vol. 12, No. 3, Article 17, January 2009.
///   <https://www.cs.rice.edu/~dwallach/pub/crosby-timing2009.pdf>
///
/// **Reference implementation:**
/// - `BoxTest.java` from mona-timing-report by Sebastian Schinzel (seecurity)
///   <https://github.com/seecurity/mona-timing-report/blob/master/src/de/fau/pi1/timerReporter/evaluation/BoxTest.java>
///
/// The Java implementation iterates `for (int i = 0; i < 100; ++i)` for lower bounds
/// and `for (int j = (i + 1); j <= 100; ++j)` for upper bounds, checking if
/// `isSignificantlySmaller(upperTimeA, lowerTimeB)` returns true when the upper
/// time measurement from one dataset is less than the lower time from the other.
#[derive(Debug, Clone)]
pub struct MonaAdapter {
    /// Minimum box size in percentile points (default: 1).
    /// The original Mona uses boxes of size 1% or larger.
    pub min_box_size: usize,
}

impl Default for MonaAdapter {
    fn default() -> Self {
        Self {
            // The original Mona uses 1% minimum, but this has very high FPR with
            // small sample sizes due to the large number of box comparisons (10,000).
            // We use 5% as a more robust default that still matches the spirit
            // of the test while reducing spurious detections.
            min_box_size: 5,
        }
    }
}

impl MonaAdapter {
    /// Create with custom minimum box size.
    pub fn with_min_box_size(size: usize) -> Self {
        Self { min_box_size: size }
    }
}

impl ToolAdapter for MonaAdapter {
    fn name(&self) -> &str {
        "mona"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        let start = Instant::now();

        let n1 = data.baseline.len();
        let n2 = data.test.len();

        if n1 < 100 || n2 < 100 {
            return ToolResult {
                detected_leak: false,
                samples_used: n1 + n2,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: "Insufficient samples for box test (need >= 100 per class)".to_string(),
                outcome: OutcomeCategory::Inconclusive,
            };
        }

        let (detected, best_box) = crosby_box_test(&data.baseline, &data.test, self.min_box_size);

        ToolResult {
            detected_leak: detected,
            samples_used: n1 + n2,
            decision_time_ms: start.elapsed().as_millis() as u64,
            leak_probability: None, // Box test doesn't give probabilities
            status: if detected {
                format!(
                    "non-overlapping box found at [{}-{}%]",
                    best_box.0, best_box.1
                )
            } else {
                "no non-overlapping box found".to_string()
            },
            outcome: if detected {
                OutcomeCategory::Fail
            } else {
                OutcomeCategory::Pass
            },
        }
    }

    fn uses_interleaved(&self) -> bool {
        false
    }
}

/// Crosby's box test: check if any percentile box shows non-overlapping distributions.
///
/// Algorithm (from BoxTest.java):
/// 1. Sort both timing arrays
/// 2. Iterate over all percentile box ranges: `i` from 0-99, `j` from `i+min_size` to 100
/// 3. For each box [i%, j%]:
///    - Extract values in that percentile range from both classes
///    - Check if max(class_A) < min(class_B) OR max(class_B) < min(class_A)
///    - If true → boxes don't overlap → leak detected
/// 4. Return (true, box) on first non-overlapping box found
///
/// Returns (detected, (lower_percentile, upper_percentile)) where the box is the
/// first non-overlapping range found, or (false, (0, 0)) if none found.
fn crosby_box_test(
    baseline: &[u64],
    sample: &[u64],
    min_box_size: usize,
) -> (bool, (usize, usize)) {
    let mut a: Vec<u64> = baseline.to_vec();
    let mut b: Vec<u64> = sample.to_vec();
    a.sort_unstable();
    b.sort_unstable();

    let len_a = a.len();
    let len_b = b.len();

    // Iterate all percentile boxes [i%, j%]
    // Matches BoxTest.java: for (int i = 0; i < 100; ++i) { for (int j = (i + 1); j <= 100; ++j)
    for i in 0..100 {
        for j in (i + min_box_size)..=100 {
            // Convert percentiles to indices
            let lo_a = i * len_a / 100;
            let hi_a = j * len_a / 100;
            let lo_b = i * len_b / 100;
            let hi_b = j * len_b / 100;

            // Skip empty or single-element boxes
            if lo_a >= hi_a || lo_b >= hi_b {
                continue;
            }

            // Get min/max within the box for each distribution
            // Since arrays are sorted: min is at lo, max is at hi-1
            let min_a = a[lo_a];
            let max_a = a[hi_a - 1];
            let min_b = b[lo_b];
            let max_b = b[hi_b - 1];

            // Check for non-overlapping boxes (Crosby's criterion)
            // BoxTest.java: isSignificantlySmaller(upperTimeA, lowerTimeB)
            if max_a < min_b || max_b < min_a {
                return (true, (i, j));
            }
        }
    }

    (false, (0, 0))
}

// =============================================================================
// Native RTLF adapter (Rust implementation)
// =============================================================================

/// Native Rust implementation of RTLF (Dunsche et al., USENIX Security 2024).
///
/// This is a faithful pure-Rust reimplementation of the RTLF algorithm, matching
/// the reference R implementation at `rtlf.R` lines 405-493.
///
/// **Algorithm overview:**
/// 1. Compute 9 quantiles (deciles: 10%, 20%, ..., 90%) for both samples
/// 2. Compute absolute difference at each decile: `|q_baseline - q_test|`
/// 3. **Within-group bootstrap** (key insight from rtlf.R:432-452):
///    - For baseline: resample baseline twice, compute `|q_b1 - q_b2|`
///    - For test: resample test twice, compute `|q_t1 - q_t2|`
///    - This estimates sampling variability assuming each group is homogeneous
/// 4. Threshold = MAX(baseline_threshold, test_threshold) at each decile
///    (rtlf.R:473-478, Bonferroni-corrected at 1 - α/9)
/// 5. Significant if observed difference exceeds threshold at any decile
///
/// This implementation avoids the R subprocess overhead of `RtlfAdapter`.
///
/// Reference: "With Great Power Come Great Side Channels: Statistical Timing
/// Side-Channel Analyses with Bounded Type-1 Errors" (USENIX Security 2024)
/// Source: <https://github.com/Fraunhofer-AISEC/RTLF>
#[derive(Debug, Clone)]
pub struct RtlfNativeAdapter {
    /// Significance level alpha (default: 0.09 as in RTLF paper).
    pub alpha: f64,
    /// Number of bootstrap iterations (default: 10000).
    pub bootstrap_iterations: usize,
}

impl Default for RtlfNativeAdapter {
    fn default() -> Self {
        Self {
            alpha: 0.09, // RTLF paper default (split across 9 deciles = 1% each)
            bootstrap_iterations: 10_000,
        }
    }
}

impl RtlfNativeAdapter {
    /// Create with custom alpha.
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha,
            ..Default::default()
        }
    }

    /// Set number of bootstrap iterations.
    pub fn bootstrap_iterations(mut self, n: usize) -> Self {
        self.bootstrap_iterations = n;
        self
    }
}

impl ToolAdapter for RtlfNativeAdapter {
    fn name(&self) -> &str {
        "rtlf-native"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        let start = Instant::now();

        let n1 = data.baseline.len();
        let n2 = data.test.len();

        if n1 < 100 || n2 < 100 {
            return ToolResult {
                detected_leak: false,
                samples_used: n1 + n2,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: "Insufficient samples for RTLF (need >= 100 per class)".to_string(),
                outcome: OutcomeCategory::Inconclusive,
            };
        }

        let (detected, significant_deciles, max_excess) = rtlf_bootstrap_test(
            &data.baseline,
            &data.test,
            self.alpha,
            self.bootstrap_iterations,
        );

        ToolResult {
            detected_leak: detected,
            samples_used: n1 + n2,
            decision_time_ms: start.elapsed().as_millis() as u64,
            leak_probability: if detected {
                Some(1.0 - self.alpha)
            } else {
                Some(self.alpha)
            },
            status: if detected {
                format!(
                    "significant at deciles {:?}, max_excess={:.2}",
                    significant_deciles, max_excess
                )
            } else {
                format!("no significant deciles, max_excess={:.2}", max_excess)
            },
            outcome: if detected {
                OutcomeCategory::Fail
            } else {
                OutcomeCategory::Pass
            },
        }
    }

    fn uses_interleaved(&self) -> bool {
        false
    }
}

/// RTLF bootstrap test for quantile differences.
///
/// Implements the RTLF algorithm from Dunsche et al. (USENIX Security 2024).
/// Key insight: RTLF uses **within-group bootstrap** to estimate sampling variability,
/// NOT pooled permutation. For each group, it resamples twice from that group alone
/// and computes quantile differences. The threshold is the MAX of both group thresholds.
///
/// Reference: rtlf.R lines 432-493 (bootstrap1 and autotest functions)
///
/// Returns (detected, significant_deciles, max_excess_ratio).
///
/// Note: We don't parallelize within RTLF because the outer benchmark loop already
/// parallelizes at the (dataset × tool) level, providing good CPU utilization.
fn rtlf_bootstrap_test(
    baseline: &[u64],
    test: &[u64],
    alpha: f64,
    bootstrap_iters: usize,
) -> (bool, Vec<usize>, f64) {
    use rand::prelude::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    let n = baseline.len().min(test.len());

    // Decile probabilities: 0.1, 0.2, ..., 0.9
    let decile_probs: Vec<f64> = (1..=9).map(|i| i as f64 * 0.1).collect();

    // Compute observed quantile differences
    let baseline_quantiles = compute_quantiles(baseline, &decile_probs);
    let test_quantiles = compute_quantiles(test, &decile_probs);

    let observed_diffs: Vec<f64> = baseline_quantiles
        .iter()
        .zip(test_quantiles.iter())
        .map(|(b, t)| (*b as f64 - *t as f64).abs())
        .collect();

    // RTLF within-group bootstrap (rtlf.R:432-452, bootstrap1 function)
    // Pre-allocate all buffers for reuse across iterations
    let mut bootstrap_diffs_baseline: Vec<Vec<f64>> = (0..9)
        .map(|_| Vec::with_capacity(bootstrap_iters))
        .collect();
    let mut bootstrap_diffs_test: Vec<Vec<f64>> = (0..9)
        .map(|_| Vec::with_capacity(bootstrap_iters))
        .collect();

    let mut rng = SmallRng::seed_from_u64(0x5deece66d);
    let mut b1: Vec<u64> = vec![0; n];
    let mut b2: Vec<u64> = vec![0; n];
    let mut t1: Vec<u64> = vec![0; n];
    let mut t2: Vec<u64> = vec![0; n];
    let mut sort_buf: Vec<u64> = Vec::with_capacity(n);
    let mut q_b1 = [0u64; 9];
    let mut q_b2 = [0u64; 9];
    let mut q_t1 = [0u64; 9];
    let mut q_t2 = [0u64; 9];

    for _ in 0..bootstrap_iters {
        // Bootstrap WITHIN baseline: resample baseline twice
        for i in 0..n {
            b1[i] = baseline[rng.gen_range(0..baseline.len())];
            b2[i] = baseline[rng.gen_range(0..baseline.len())];
        }
        compute_quantiles_inplace(&b1, &decile_probs, &mut sort_buf, &mut q_b1);
        compute_quantiles_inplace(&b2, &decile_probs, &mut sort_buf, &mut q_b2);

        // Bootstrap WITHIN test: resample test twice
        for i in 0..n {
            t1[i] = test[rng.gen_range(0..test.len())];
            t2[i] = test[rng.gen_range(0..test.len())];
        }
        compute_quantiles_inplace(&t1, &decile_probs, &mut sort_buf, &mut q_t1);
        compute_quantiles_inplace(&t2, &decile_probs, &mut sort_buf, &mut q_t2);

        // Compute quantile differences for this iteration
        for i in 0..9 {
            bootstrap_diffs_baseline[i].push((q_b1[i] as f64 - q_b2[i] as f64).abs());
            bootstrap_diffs_test[i].push((q_t1[i] as f64 - q_t2[i] as f64).abs());
        }
    }

    // Compute critical thresholds at (1 - alpha/9) for each decile
    // This implements the Bonferroni-style correction across 9 deciles
    let alpha_per_decile = alpha / 9.0;
    let threshold_percentile = 1.0 - alpha_per_decile;
    let threshold_idx =
        ((bootstrap_iters as f64 * threshold_percentile) as usize).min(bootstrap_iters - 1);

    let mut significant_deciles = Vec::new();
    let mut max_excess = 0.0f64;

    for i in 0..9 {
        // Sort bootstrap distributions for each group
        bootstrap_diffs_baseline[i]
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        bootstrap_diffs_test[i]
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Threshold from each group (rtlf.R:473-474)
        let thresh_baseline = bootstrap_diffs_baseline[i][threshold_idx];
        let thresh_test = bootstrap_diffs_test[i][threshold_idx];

        // RTLF uses MAX of both thresholds (rtlf.R:477-478)
        let threshold = thresh_baseline.max(thresh_test);

        let excess = if threshold > 0.0 {
            observed_diffs[i] / threshold
        } else if observed_diffs[i] > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        if excess > max_excess {
            max_excess = excess;
        }

        // Significant if observed exceeds threshold (rtlf.R:485-488)
        if observed_diffs[i] > threshold {
            significant_deciles.push((i + 1) * 10); // Convert to percentile (10, 20, ..., 90)
        }
    }

    let detected = !significant_deciles.is_empty();
    (detected, significant_deciles, max_excess)
}

/// Compute quantiles at specified probabilities.
fn compute_quantiles(data: &[u64], probs: &[f64]) -> Vec<u64> {
    let mut sorted: Vec<u64> = data.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();

    probs
        .iter()
        .map(|&p| {
            // Type 2 quantile (RTLF uses this)
            let idx = ((n as f64) * p).ceil() as usize;
            let idx = idx.saturating_sub(1).min(n - 1);
            sorted[idx]
        })
        .collect()
}

/// Compute quantiles using a pre-allocated buffer to avoid allocation.
fn compute_quantiles_inplace(data: &[u64], probs: &[f64], buf: &mut Vec<u64>, result: &mut [u64]) {
    buf.clear();
    buf.extend_from_slice(data);
    buf.sort_unstable();
    let n = buf.len();

    for (i, &p) in probs.iter().enumerate() {
        let idx = ((n as f64) * p).ceil() as usize;
        let idx = idx.saturating_sub(1).min(n - 1);
        result[i] = buf[idx];
    }
}

// =============================================================================
// Native SILENT adapter (Rust implementation)
// =============================================================================

/// Native Rust implementation of SILENT (Deininger et al., arXiv:2504.19821).
///
/// # WARNING: NOT FAITHFUL TO ORIGINAL ALGORITHM
///
/// This implementation is **NOT accurate** to the original SILENT methodology.
/// Use [`SilentAdapter`] (which calls the R reference implementation) for
/// accurate results in paper comparisons.
///
/// **Key differences from real SILENT:**
/// - Uses **mean difference** instead of quantile comparison (91 quantiles)
/// - Uses **IID bootstrap** instead of block bootstrap with m-dependence
/// - Missing **variance normalization** per quantile
/// - Missing **two-stage filtering** (variance + relevance filters)
/// - Wrong **test statistic** (CI-based instead of max quantile test)
///
/// This adapter exists only for running benchmarks without R installed.
/// For publication-quality comparisons, use `SilentAdapter` instead.
///
/// Reference: "SILENT: A Practical Method for Verifying Constant-Time Code"
/// (arXiv:2504.19821)
#[derive(Debug, Clone)]
pub struct SilentNativeAdapter {
    /// Significance level alpha (default: 0.05).
    pub alpha: f64,
    /// Practical significance threshold Δ in nanoseconds (default: 0.5 of σ).
    /// If None, uses 0.5 * pooled_std as in the paper.
    pub delta: Option<f64>,
    /// Number of bootstrap iterations (default: 1000, as in the SILENT paper).
    pub bootstrap_iterations: usize,
}

impl Default for SilentNativeAdapter {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            delta: None, // Will compute as 0.5 * pooled_std
            bootstrap_iterations: 1_000, // SILENT paper uses B=1000
        }
    }
}

impl SilentNativeAdapter {
    /// Create with custom alpha.
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha,
            ..Default::default()
        }
    }

    /// Set practical significance threshold Δ.
    pub fn delta(mut self, delta: f64) -> Self {
        self.delta = Some(delta);
        self
    }

    /// Set number of bootstrap iterations.
    pub fn bootstrap_iterations(mut self, n: usize) -> Self {
        self.bootstrap_iterations = n;
        self
    }
}

impl ToolAdapter for SilentNativeAdapter {
    fn name(&self) -> &str {
        "silent-native"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        let start = Instant::now();

        let n1 = data.baseline.len();
        let n2 = data.test.len();

        if n1 < 30 || n2 < 30 {
            return ToolResult {
                detected_leak: false,
                samples_used: n1 + n2,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: "Insufficient samples for SILENT (need >= 30 per class)".to_string(),
                outcome: OutcomeCategory::Inconclusive,
            };
        }

        let (detected, ci_lower, observed_diff, delta_used) = silent_bootstrap_test(
            &data.baseline,
            &data.test,
            self.alpha,
            self.delta,
            self.bootstrap_iterations,
        );

        ToolResult {
            detected_leak: detected,
            samples_used: n1 + n2,
            decision_time_ms: start.elapsed().as_millis() as u64,
            leak_probability: if detected {
                Some(1.0 - self.alpha)
            } else {
                Some(self.alpha)
            },
            status: format!(
                "diff={:.1}ns, CI_lower={:.1}ns, Δ={:.1}ns",
                observed_diff, ci_lower, delta_used
            ),
            outcome: if detected {
                OutcomeCategory::Fail
            } else {
                OutcomeCategory::Pass
            },
        }
    }

    fn uses_interleaved(&self) -> bool {
        false
    }
}

/// SILENT studentized bootstrap test.
///
/// Returns (detected, ci_lower, observed_diff, delta_used).
fn silent_bootstrap_test(
    baseline: &[u64],
    test: &[u64],
    alpha: f64,
    delta: Option<f64>,
    bootstrap_iters: usize,
) -> (bool, f64, f64, f64) {
    use rand::prelude::*;

    let mut rng = rand::thread_rng();

    // Convert to f64 for calculations
    let x: Vec<f64> = baseline.iter().map(|&v| v as f64).collect();
    let y: Vec<f64> = test.iter().map(|&v| v as f64).collect();

    let n_x = x.len();
    let n_y = y.len();

    // Compute observed statistics
    let mean_x: f64 = x.iter().sum::<f64>() / n_x as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n_y as f64;
    let observed_diff = (mean_x - mean_y).abs();

    // Compute pooled standard deviation for delta
    let var_x: f64 = x.iter().map(|v| (v - mean_x).powi(2)).sum::<f64>() / (n_x - 1) as f64;
    let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / (n_y - 1) as f64;
    let pooled_std = ((var_x + var_y) / 2.0).sqrt();

    // Delta: use provided value or default to 0.5 * pooled_std
    let delta_used = delta.unwrap_or(0.5 * pooled_std);

    // Studentized bootstrap
    // We compute the studentized statistic: T = (diff - observed_diff) / SE(diff)
    let mut bootstrap_t_stats: Vec<f64> = Vec::with_capacity(bootstrap_iters);

    for _ in 0..bootstrap_iters {
        // Resample with replacement from each group
        let x_boot: Vec<f64> = (0..n_x).map(|_| x[rng.gen_range(0..n_x)]).collect();
        let y_boot: Vec<f64> = (0..n_y).map(|_| y[rng.gen_range(0..n_y)]).collect();

        let mean_x_boot: f64 = x_boot.iter().sum::<f64>() / n_x as f64;
        let mean_y_boot: f64 = y_boot.iter().sum::<f64>() / n_y as f64;
        let diff_boot = (mean_x_boot - mean_y_boot).abs();

        // Compute SE of the bootstrap difference
        let var_x_boot: f64 = x_boot
            .iter()
            .map(|v| (v - mean_x_boot).powi(2))
            .sum::<f64>()
            / (n_x - 1) as f64;
        let var_y_boot: f64 = y_boot
            .iter()
            .map(|v| (v - mean_y_boot).powi(2))
            .sum::<f64>()
            / (n_y - 1) as f64;
        let se_boot = (var_x_boot / n_x as f64 + var_y_boot / n_y as f64).sqrt();

        if se_boot > 1e-10 {
            // Studentized statistic: how many SEs away from observed
            let t_stat = (diff_boot - observed_diff) / se_boot;
            bootstrap_t_stats.push(t_stat);
        }
    }

    if bootstrap_t_stats.is_empty() {
        return (false, 0.0, observed_diff, delta_used);
    }

    // Sort bootstrap t-statistics
    bootstrap_t_stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Get percentiles for CI
    // For lower bound of CI, we need the (1 - alpha) percentile of t-stats
    let upper_idx = ((bootstrap_t_stats.len() as f64) * (1.0 - alpha)) as usize;
    let upper_idx = upper_idx.min(bootstrap_t_stats.len() - 1);
    let t_upper = bootstrap_t_stats[upper_idx];

    // Original SE
    let se_orig = (var_x / n_x as f64 + var_y / n_y as f64).sqrt();

    // CI lower bound: observed - t_upper * SE
    // Note: for absolute difference, we want lower bound > delta
    let ci_lower = observed_diff - t_upper * se_orig;

    // Significant if CI lower bound exceeds delta
    let detected = ci_lower > delta_used;

    (detected, ci_lower, observed_diff, delta_used)
}

// =============================================================================
// tlsfuzzer adapter (Python-based)
// =============================================================================

/// Adapter for tlsfuzzer timing analysis.
///
/// tlsfuzzer uses multiple statistical tests (Wilcoxon, Sign test, t-test,
/// Friedman test) for timing analysis. This adapter calls the Python script
/// via subprocess.
///
/// Reference: https://github.com/tlsfuzzer/tlsfuzzer
#[derive(Debug, Clone)]
pub struct TlsfuzzerAdapter {
    /// Path to Python executable (default: "python3").
    pub python: String,
    /// Path to tlsfuzzer analysis script (default: uses installed tlsfuzzer).
    pub script_path: Option<String>,
    /// Significance level alpha (default: 0.05).
    pub alpha: f64,
}

impl Default for TlsfuzzerAdapter {
    fn default() -> Self {
        Self {
            python: "python3".to_string(),
            script_path: None,
            alpha: 0.05,
        }
    }
}

impl TlsfuzzerAdapter {
    /// Create with custom Python path.
    pub fn with_python(python: impl Into<String>) -> Self {
        Self {
            python: python.into(),
            ..Default::default()
        }
    }

    /// Set script path.
    pub fn script_path(mut self, path: impl Into<String>) -> Self {
        self.script_path = Some(path.into());
        self
    }

    /// Set significance level.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

impl ToolAdapter for TlsfuzzerAdapter {
    fn name(&self) -> &str {
        "tlsfuzzer"
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        let start = Instant::now();

        // Create temporary directory for input/output
        let temp_dir = match tempfile::tempdir() {
            Ok(dir) => dir,
            Err(e) => {
                return ToolResult {
                    detected_leak: false,
                    samples_used: 0,
                    decision_time_ms: start.elapsed().as_millis() as u64,
                    leak_probability: None,
                    status: format!("Failed to create temp dir: {}", e),
                    outcome: OutcomeCategory::Error,
                };
            }
        };

        // Write input CSV in tlsfuzzer format (timing.csv)
        let input_file = temp_dir.path().join("timing.csv");

        if let Err(e) = write_tlsfuzzer_csv(&input_file, data) {
            return ToolResult {
                detected_leak: false,
                samples_used: 0,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: format!("Failed to write input: {}", e),
                outcome: OutcomeCategory::Error,
            };
        }

        // Run tlsfuzzer analysis
        let result = run_tlsfuzzer(
            &self.python,
            self.script_path.as_deref(),
            &input_file,
            self.alpha,
        );

        match result {
            Ok((detected, p_value, test_name)) => {
                let samples_used = data.baseline.len() + data.test.len();
                ToolResult {
                    detected_leak: detected,
                    samples_used,
                    decision_time_ms: start.elapsed().as_millis() as u64,
                    leak_probability: Some(1.0 - p_value),
                    status: format!("{}(p={:.4}), alpha={:.2}", test_name, p_value, self.alpha),
                    outcome: if detected {
                        OutcomeCategory::Fail
                    } else {
                        OutcomeCategory::Pass
                    },
                }
            }
            Err(e) => ToolResult {
                detected_leak: false,
                samples_used: 0,
                decision_time_ms: start.elapsed().as_millis() as u64,
                leak_probability: None,
                status: format!("Error: {}", e),
                outcome: OutcomeCategory::Error,
            },
        }
    }

    fn uses_interleaved(&self) -> bool {
        false
    }
}

/// Write data in tlsfuzzer timing.csv format.
fn write_tlsfuzzer_csv(path: &Path, data: &BlockedData) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    // tlsfuzzer expects: header row with class names, then timing values
    // Format: class1,class2 (column per class)
    writeln!(file, "baseline,sample")?;

    let max_len = data.baseline.len().max(data.test.len());
    for i in 0..max_len {
        let v1 = data
            .baseline
            .get(i)
            .map(|&v| v.to_string())
            .unwrap_or_default();
        let v2 = data.test.get(i).map(|&v| v.to_string()).unwrap_or_default();
        writeln!(file, "{},{}", v1, v2)?;
    }

    Ok(())
}

/// Run tlsfuzzer analysis.
fn run_tlsfuzzer(
    python: &str,
    script_path: Option<&str>,
    input_file: &Path,
    alpha: f64,
) -> Result<(bool, f64, String), String> {
    // Use the tlsfuzzer.tlsfuzzer.analysis module if no script specified
    let script = script_path
        .map(String::from)
        .unwrap_or_else(|| "-m tlsfuzzer.analysis".to_string());

    let mut cmd = Command::new(python);

    if script.starts_with("-m") {
        // Module invocation
        cmd.arg("-m").arg("tlsfuzzer.analysis");
    } else {
        cmd.arg(&script);
    }

    let output = cmd
        .arg("-o")
        .arg(input_file.parent().unwrap_or(Path::new(".")))
        .arg("--no-ecdf-plot")
        .arg("--no-scatter-plot")
        .arg("--no-conf-interval-plot")
        .arg("timing.csv")
        .current_dir(input_file.parent().unwrap_or(Path::new(".")))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run tlsfuzzer: {}. Is tlsfuzzer installed?", e))?;

    // tlsfuzzer exit codes:
    // 0 = no leak detected
    // 1 = leak detected (this is NOT an error!)
    // Other codes = actual error
    //
    // IMPORTANT: Python also returns exit code 1 for import errors!
    // We must check stderr to distinguish real errors from leak detection.
    let exit_code = output.status.code().unwrap_or(2);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check for Python/import errors in stderr
    if !stderr.is_empty()
        && (stderr.contains("ModuleNotFoundError")
            || stderr.contains("ImportError")
            || stderr.contains("Error while finding module")
            || stderr.contains("Traceback"))
    {
        return Err(format!(
            "tlsfuzzer failed: {}. Is tlsfuzzer installed?",
            stderr.lines().last().unwrap_or(&stderr)
        ));
    }

    if exit_code > 1 {
        return Err(format!("tlsfuzzer failed (exit {}): {}", exit_code, stderr));
    }

    // Also verify we got actual output (not just an error that returned exit 1)
    if stdout.is_empty() && exit_code == 1 {
        return Err(format!(
            "tlsfuzzer produced no output (exit 1). stderr: {}",
            stderr
        ));
    }

    // Parse output for results
    parse_tlsfuzzer_output(&stdout, alpha, exit_code == 1)
}

/// Parse tlsfuzzer analysis output.
fn parse_tlsfuzzer_output(
    output: &str,
    alpha: f64,
    exit_code_detected: bool,
) -> Result<(bool, f64, String), String> {
    // tlsfuzzer outputs results in various formats:
    // "Sign test: p-value: 0.1234"
    // "Sign test mean p-value(p=0.5067)"
    // "Wilcoxon signed-rank test: p-value: 0.0567"
    // "VULNERABLE" when leak detected

    let mut min_p_value = 1.0;
    let mut test_name = "unknown".to_string();

    for line in output.lines() {
        let line_lower = line.to_lowercase();
        if line_lower.contains("p-value") || line_lower.contains("pvalue") {
            // Try format: "Test name(p=0.1234)" or "Test name p-value(p=0.1234)"
            if let Some(start) = line.find("(p=") {
                if let Some(end) = line[start..].find(')') {
                    let p_str = &line[start + 3..start + end];
                    if let Ok(p) = p_str.parse::<f64>() {
                        if p < min_p_value {
                            min_p_value = p;
                            // Extract test name (everything before the p-value part)
                            test_name = line[..start].trim().to_string();
                        }
                    }
                }
            }

            // Also try format: "Test name: p-value: 0.1234"
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() >= 2 {
                let name = parts[0].trim();
                // Try to extract p-value from last part
                if let Some(p_str) = parts.last() {
                    if let Ok(p) = p_str.trim().parse::<f64>() {
                        if p < min_p_value {
                            min_p_value = p;
                            test_name = name.to_string();
                        }
                    }
                }
            }
        }
    }

    // Check for explicit VULNERABLE/PASS keywords
    let output_lower = output.to_lowercase();
    let explicit_vulnerable = output_lower.contains("vulnerable");
    let explicit_fail = output_lower.contains("fail") && !output_lower.contains("failed to");

    // Detect leak if: exit code says so, p-value below alpha, or explicit vulnerable keyword
    let detected =
        exit_code_detected || min_p_value < alpha || explicit_vulnerable || explicit_fail;

    if min_p_value < 1.0 || exit_code_detected || explicit_vulnerable {
        Ok((detected, min_p_value, test_name))
    } else {
        Err("Could not parse tlsfuzzer output".to_string())
    }
}

// =============================================================================
// Stub adapters for tools that aren't yet implemented
// =============================================================================

/// Stub adapter that always returns "not available".
#[derive(Debug, Clone)]
pub struct StubAdapter {
    name: String,
}

impl StubAdapter {
    /// Create a stub for a tool that isn't available.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl ToolAdapter for StubAdapter {
    fn name(&self) -> &str {
        &self.name
    }

    fn analyze_blocked(&self, data: &BlockedData) -> ToolResult {
        ToolResult {
            detected_leak: false,
            samples_used: data.baseline.len() + data.test.len(),
            decision_time_ms: 0,
            leak_probability: None,
            status: format!("{} not available", self.name),
            outcome: OutcomeCategory::Error,
        }
    }
}

// =============================================================================
// Utility functions
// =============================================================================

/// Load interleaved data from CSV file.
pub fn load_interleaved_csv(path: &Path) -> std::io::Result<Vec<(Class, u64)>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for line in reader.lines().skip(1) {
        // Skip header
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let class = match parts[0].trim() {
                "BASELINE" => Class::Baseline,
                "MODIFIED" | "SAMPLE" => Class::Sample,
                _ => continue,
            };
            if let Ok(value) = parts[1].trim().parse::<u64>() {
                data.push((class, value));
            }
        }
    }

    Ok(data)
}

/// Load blocked data from CSV file.
pub fn load_blocked_csv(path: &Path) -> std::io::Result<BlockedData> {
    let interleaved = load_interleaved_csv(path)?;
    Ok(split_interleaved(&interleaved))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{generate_dataset, EffectType, SyntheticConfig};

    #[test]
    fn test_split_interleaved() {
        let interleaved = vec![
            (Class::Baseline, 100),
            (Class::Sample, 200),
            (Class::Baseline, 150),
            (Class::Sample, 250),
        ];

        let blocked = split_interleaved(&interleaved);
        assert_eq!(blocked.baseline, vec![100, 150]);
        assert_eq!(blocked.test, vec![200, 250]);
    }

    #[test]
    fn test_tacet_adapter_null() {
        let config = SyntheticConfig {
            samples_per_class: 1000,
            effect: EffectType::Null,
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = TimingOracleAdapter::default().time_budget(Duration::from_secs(10));
        let result = adapter.analyze(&dataset);

        // Null effect should (usually) not detect leak
        // This is probabilistic, so we just check the result is valid
        assert!(result.samples_used > 0);
        assert!(result.leak_probability.is_some());
    }

    #[test]
    fn test_tacet_adapter_shift() {
        let config = SyntheticConfig {
            samples_per_class: 5000,
            effect: EffectType::Shift { percent: 10.0 }, // Strong shift
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = TimingOracleAdapter::default().time_budget(Duration::from_secs(30));
        let result = adapter.analyze(&dataset);

        // Strong shift should (usually) detect leak
        // This is probabilistic, so we just check the result is valid
        assert!(result.samples_used > 0);
        println!(
            "Shift 10%: detected={}, status={}",
            result.detected_leak, result.status
        );
    }

    #[test]
    fn test_stub_adapter() {
        let adapter = StubAdapter::new("test-tool");
        let data = BlockedData {
            baseline: vec![100, 200, 300],
            test: vec![110, 210, 310],
        };

        let result = adapter.analyze_blocked(&data);
        assert!(!result.detected_leak);
        assert_eq!(result.samples_used, 6);
        assert!(result.status.contains("not available"));
    }

    #[test]
    fn test_dudect_stats_different_distributions() {
        use crate::dudect_stats::update_ct_stats;

        // Two samples with clearly different means
        let sample1: Vec<u64> = vec![100, 102, 98, 101, 99];
        let sample2: Vec<u64> = vec![200, 198, 202, 199, 201];

        let (summary, _) = update_ct_stats(None, &(sample1, sample2));

        // max_t should be large (distributions are different)
        assert!(
            summary.max_t.abs() > 10.0,
            "|t|={} should be large",
            summary.max_t.abs()
        );
    }

    #[test]
    fn test_dudect_stats_same_distribution() {
        use crate::dudect_stats::update_ct_stats;

        // Two samples from same distribution should have small t
        let sample1: Vec<u64> = vec![100, 102, 98, 101, 99, 100, 100, 101, 99, 100];
        let sample2: Vec<u64> = vec![101, 99, 100, 102, 98, 100, 101, 99, 100, 100];

        let (summary, _) = update_ct_stats(None, &(sample1, sample2));

        // max_t should be small (close to 0)
        assert!(
            summary.max_t.abs() < 3.0,
            "|t|={} should be small for same distribution",
            summary.max_t.abs()
        );
    }

    #[test]
    fn test_dudect_adapter_null() {
        let config = SyntheticConfig {
            samples_per_class: 1000,
            effect: EffectType::Null,
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = DudectAdapter::default();
        let result = adapter.analyze(&dataset);

        // Null effect should (usually) not detect leak with t-threshold 4.5
        assert!(result.samples_used > 0);
        assert!(result.status.contains("max|t|="));
        println!(
            "Dudect null: detected={}, status={}",
            result.detected_leak, result.status
        );
    }

    #[test]
    fn test_dudect_adapter_strong_shift() {
        let config = SyntheticConfig {
            samples_per_class: 5000,
            effect: EffectType::Shift { percent: 20.0 }, // Very strong shift
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = DudectAdapter::default();
        let result = adapter.analyze(&dataset);

        // Strong shift should detect leak
        assert!(result.samples_used > 0);
        println!(
            "Dudect shift 20%: detected={}, status={}",
            result.detected_leak, result.status
        );
        // With a 20% shift and 5000 samples, we should definitely detect it
        assert!(
            result.detected_leak,
            "20% shift should be detected, got: {}",
            result.status
        );
    }

    // =========================================================================
    // TVLA tests
    // =========================================================================

    #[test]
    fn test_tvla_welch_t_test() {
        // Two samples with clearly different means
        let sample1: Vec<u64> = vec![100, 102, 98, 101, 99, 103, 97, 100, 101, 99];
        let sample2: Vec<u64> = vec![200, 198, 202, 199, 201, 203, 197, 200, 201, 199];

        let t = welch_t_test(&sample1, &sample2);

        // t should be large (distributions are different)
        assert!(t.abs() > 10.0, "|t|={} should be large", t.abs());
    }

    #[test]
    fn test_timing_tvla_adapter_null() {
        let config = SyntheticConfig {
            samples_per_class: 1000,
            effect: EffectType::Null,
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = TimingTvlaAdapter::default();
        let result = adapter.analyze(&dataset);

        // Null effect should (usually) not detect leak
        assert!(result.samples_used > 0);
        assert!(result.status.contains("|t|="));
        println!(
            "Timing-TVLA null: detected={}, status={}",
            result.detected_leak, result.status
        );
    }

    #[test]
    fn test_timing_tvla_adapter_strong_shift() {
        let config = SyntheticConfig {
            samples_per_class: 5000,
            effect: EffectType::Shift { percent: 20.0 },
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = TimingTvlaAdapter::default();
        let result = adapter.analyze(&dataset);

        // Strong shift should detect leak
        assert!(result.samples_used > 0);
        println!(
            "Timing-TVLA shift 20%: detected={}, status={}",
            result.detected_leak, result.status
        );
        assert!(result.detected_leak, "20% shift should be detected");
    }

    // =========================================================================
    // KS test adapter tests
    // =========================================================================

    #[test]
    fn test_ks_test_identical_samples() {
        // Two samples from the same distribution should have D ≈ 0
        let sample: Vec<u64> = (0..1000).map(|i| 1000 + i).collect();
        let d = ks_two_sample(&sample, &sample).0;
        assert!(d < 0.01, "Identical samples should have D ≈ 0, got {}", d);
    }

    #[test]
    fn test_ks_test_different_samples() {
        // Two clearly different distributions
        let sample1: Vec<u64> = (0..1000).map(|i| 1000 + i).collect();
        let sample2: Vec<u64> = (0..1000).map(|i| 2000 + i).collect();
        let (d, p) = ks_two_sample(&sample1, &sample2);
        assert!(
            d > 0.9,
            "Non-overlapping samples should have D ≈ 1, got {}",
            d
        );
        assert!(
            p < 0.001,
            "Non-overlapping samples should have tiny p-value, got {}",
            p
        );
    }

    #[test]
    fn test_ks_adapter_null() {
        let config = SyntheticConfig {
            samples_per_class: 1000,
            effect: EffectType::Null,
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = KsTestAdapter::default();
        let result = adapter.analyze(&dataset);

        assert!(result.samples_used > 0);
        assert!(result.status.contains("D="));
        println!(
            "KS test null: detected={}, status={}",
            result.detected_leak, result.status
        );
    }

    #[test]
    fn test_ks_adapter_strong_shift() {
        let config = SyntheticConfig {
            samples_per_class: 5000,
            effect: EffectType::Shift { percent: 20.0 },
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = KsTestAdapter::default();
        let result = adapter.analyze(&dataset);

        assert!(result.samples_used > 0);
        println!(
            "KS test shift 20%: detected={}, status={}",
            result.detected_leak, result.status
        );
        assert!(result.detected_leak, "20% shift should be detected");
    }

    // =========================================================================
    // Anderson-Darling test adapter tests
    // =========================================================================

    #[test]
    fn test_ad_test_identical_samples() {
        // Two samples from the same distribution should have small A²
        let sample: Vec<u64> = (0..1000).map(|i| 1000 + i).collect();
        let (a2, _) = ad_two_sample(&sample, &sample);
        assert!(
            a2 < 1.0,
            "Identical samples should have small A², got {}",
            a2
        );
    }

    #[test]
    fn test_ad_test_different_samples() {
        // Two clearly different distributions
        let sample1: Vec<u64> = (0..1000).map(|i| 1000 + i).collect();
        let sample2: Vec<u64> = (0..1000).map(|i| 2000 + i).collect();
        let (a2, p) = ad_two_sample(&sample1, &sample2);
        assert!(
            a2 > 10.0,
            "Non-overlapping samples should have large A², got {}",
            a2
        );
        assert!(
            p < 0.001,
            "Non-overlapping samples should have tiny p-value, got {}",
            p
        );
    }

    #[test]
    fn test_ad_test_heavily_quantized_same_distribution() {
        // Heavily quantized data (simulating coarse timer resolution) from the same distribution.
        // This tests the fix for the stable sort bug that caused 100% FPR with tied values.
        // Before the fix, A² would be astronomically large due to systematic bias.
        let n = 5000;
        // Only 5 unique values, each appearing 1000 times per sample
        let values = [1000u64, 1042, 1084, 1126, 1168];
        let sample1: Vec<u64> = (0..n).map(|i| values[i % values.len()]).collect();
        let sample2: Vec<u64> = (0..n).map(|i| values[i % values.len()]).collect();

        let (a2, p) = ad_two_sample(&sample1, &sample2);
        // With identical distributions and proper tie handling, p-value should be high
        assert!(
            p > 0.01,
            "Heavily quantized same-distribution data should not be rejected at α=0.01, got A²={:.2}, p={:.6}",
            a2, p
        );
    }

    #[test]
    fn test_ad_adapter_null() {
        let config = SyntheticConfig {
            samples_per_class: 1000,
            effect: EffectType::Null,
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = AndersonDarlingAdapter::default();
        let result = adapter.analyze(&dataset);

        assert!(result.samples_used > 0);
        assert!(result.status.contains("A²="));
        println!(
            "AD test null: detected={}, status={}",
            result.detected_leak, result.status
        );
    }

    #[test]
    fn test_ad_adapter_strong_shift() {
        let config = SyntheticConfig {
            samples_per_class: 5000,
            effect: EffectType::Shift { percent: 20.0 },
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = AndersonDarlingAdapter::default();
        let result = adapter.analyze(&dataset);

        assert!(result.samples_used > 0);
        println!(
            "AD test shift 20%: detected={}, status={}",
            result.detected_leak, result.status
        );
        assert!(result.detected_leak, "20% shift should be detected");
    }

    // =========================================================================
    // Mona (Crosby box test) tests
    // =========================================================================

    #[test]
    fn test_crosby_box_test_non_overlapping() {
        // Two completely separated distributions - should detect
        let baseline: Vec<u64> = (100..200).collect(); // 100-199
        let sample: Vec<u64> = (300..400).collect(); // 300-399

        let (detected, box_range) = crosby_box_test(&baseline, &sample, 1);

        assert!(
            detected,
            "Completely separated distributions should be detected"
        );
        println!(
            "Non-overlapping: detected at box [{}-{}%]",
            box_range.0, box_range.1
        );
    }

    #[test]
    fn test_crosby_box_test_overlapping() {
        // Two identical distributions - should NOT detect
        let baseline: Vec<u64> = (100..200).collect();
        let sample: Vec<u64> = (100..200).collect();

        let (detected, _) = crosby_box_test(&baseline, &sample, 1);

        assert!(!detected, "Identical distributions should not be detected");
    }

    #[test]
    fn test_crosby_box_test_partial_separation() {
        // Distributions that are separated only in tails
        // baseline: 100-200, sample: 150-250 (overlap in middle, but tails differ)
        let baseline: Vec<u64> = (100..200).collect();
        let sample: Vec<u64> = (150..250).collect();

        let (detected, box_range) = crosby_box_test(&baseline, &sample, 1);

        // The lower percentiles of baseline (100-149) don't overlap with
        // lower percentiles of sample (150-199), so should detect
        assert!(
            detected,
            "Partially separated distributions should be detected in tails"
        );
        println!(
            "Partial separation: detected at box [{}-{}%]",
            box_range.0, box_range.1
        );
    }

    #[test]
    fn test_mona_adapter_null() {
        // The Crosby box test can have false positives with small box sizes (1%)
        // because with finite samples, random data may occasionally have
        // non-overlapping percentile ranges by chance. This is a known property
        // of the test - the SILENT paper notes this sensitivity.
        //
        // We test FPR empirically over multiple seeds rather than asserting
        // no false positives on a single run.
        let mut false_positives = 0;
        let trials = 20;

        for seed in 0..trials {
            let config = SyntheticConfig {
                samples_per_class: 1000,
                effect: EffectType::Null,
                seed,
                ..Default::default()
            };
            let dataset = generate_dataset(&config);

            let adapter = MonaAdapter::default();
            let result = adapter.analyze(&dataset);

            if result.detected_leak {
                false_positives += 1;
            }
        }

        let fpr = false_positives as f64 / trials as f64;
        println!(
            "Mona box test FPR: {}/{} = {:.1}%",
            false_positives,
            trials,
            fpr * 100.0
        );

        // Box test should have reasonable FPR (< 50% on null data)
        // Higher FPR than other tests is expected for this non-parametric method
        assert!(
            fpr < 0.5,
            "Box test FPR should be < 50%, got {:.1}%",
            fpr * 100.0
        );
    }

    #[test]
    fn test_mona_adapter_strong_shift() {
        // For the box test to detect, we need distributions that are truly
        // non-overlapping in at least one percentile range. A mean shift in
        // log-normal data might not create complete separation, so we use
        // a very strong effect.
        let config = SyntheticConfig {
            samples_per_class: 1000,
            effect: EffectType::Shift { percent: 200.0 }, // Very strong shift
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = MonaAdapter::default();
        let result = adapter.analyze(&dataset);

        assert!(result.samples_used > 0);
        println!(
            "Mona shift 200%: detected={}, status={}",
            result.detected_leak, result.status
        );
        // With a 200% shift, there should be some non-overlapping percentile range
    }

    // =========================================================================
    // tlsfuzzer tests (requires Python + tlsfuzzer installed)
    // =========================================================================

    #[test]
    #[ignore] // Requires tlsfuzzer Python package installed
    fn test_tlsfuzzer_adapter() {
        let config = SyntheticConfig {
            samples_per_class: 1000,
            effect: EffectType::Shift { percent: 20.0 },
            seed: 42,
            ..Default::default()
        };
        let dataset = generate_dataset(&config);

        let adapter = TlsfuzzerAdapter::default();
        let result = adapter.analyze(&dataset);

        println!(
            "tlsfuzzer: detected={}, status={}",
            result.detected_leak, result.status
        );
        // Just check it runs without crashing when tlsfuzzer is available
    }
}
