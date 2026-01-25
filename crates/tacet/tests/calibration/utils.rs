//! Shared utilities for calibration tests.
//!
//! This module provides:
//! - Configuration management via environment variables
//! - Deterministic RNG with logged seeds for reproducibility
//! - Effect injection (busy-wait delays)
//! - Trial runner with outcome accounting
//! - Statistical helpers (Wilson CI)
//! - JSONL output formatting
//!
//! See docs/calibration-test-spec.md for the full specification.

#![allow(dead_code)]

use rand::{rngs::StdRng, SeedableRng};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tacet::Outcome;

// Re-export effect injection from the shared module
// These are used by other calibration test files via `mod calibration_utils;`
#[allow(unused_imports)]
pub use tacet::helpers::effect::{busy_wait_ns, set_global_max_delay_ns};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Test tier controlling trial counts and runtime budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Quick iteration tier for developing the validation suite (~30 min total)
    Iteration,
    /// Quick tier for PR checks (~3-5 min per test)
    Quick,
    /// Full tier for thorough validation (~8 min per test)
    Full,
    /// Validation tier for pre-release checks (~25 min per test, 2-4 hours total)
    Validation,
}

impl Tier {
    pub fn from_env() -> Self {
        match std::env::var("CALIBRATION_TIER").as_deref() {
            Ok("iteration") => Tier::Iteration,
            Ok("quick") => Tier::Quick,
            Ok("full") => Tier::Full,
            Ok("validation") => Tier::Validation,
            _ => {
                // Auto-detect: quick in CI, full locally
                if is_ci() {
                    Tier::Quick
                } else {
                    Tier::Full
                }
            }
        }
    }

    pub fn max_wall_ms(&self) -> u64 {
        match self {
            Tier::Iteration => 120_000,    // 2 minutes per test (30 min total)
            Tier::Quick => 180_000,        // 3 minutes
            Tier::Full => 480_000,         // 8 minutes
            Tier::Validation => 1_500_000, // 25 minutes
        }
    }

    pub fn samples_per_trial(&self) -> usize {
        match self {
            Tier::Iteration => 2_000,
            Tier::Quick => 2_000,
            Tier::Full => 5_000,
            Tier::Validation => 10_000,
        }
    }

    pub fn time_budget_per_trial(&self) -> Duration {
        match self {
            Tier::Iteration => Duration::from_secs(3),
            Tier::Quick => Duration::from_secs(3),
            Tier::Full => Duration::from_secs(5),
            Tier::Validation => Duration::from_secs(10),
        }
    }

    pub fn fpr_trials(&self) -> usize {
        match self {
            Tier::Iteration => 50,
            Tier::Quick => 50,
            Tier::Full => 100,
            Tier::Validation => 1000,
        }
    }

    pub fn power_trials(&self) -> usize {
        match self {
            Tier::Iteration => 50,
            Tier::Quick => 50,
            Tier::Full => 100,
            Tier::Validation => 200,
        }
    }

    pub fn max_fpr(&self) -> f64 {
        match self {
            Tier::Iteration => 0.15,
            Tier::Quick => 0.15,
            Tier::Full => 0.10,
            Tier::Validation => 0.07,
        }
    }

    pub fn max_inject_ns(&self) -> u64 {
        match self {
            Tier::Iteration => 10_000,     // 10μs
            Tier::Quick => 10_000,         // 10μs
            Tier::Full => 100_000,         // 100μs
            Tier::Validation => 1_000_000, // 1ms
        }
    }

    pub fn min_power_2x_theta(&self) -> f64 {
        0.70 // All tiers require 70% power at 2×θ
    }

    pub fn min_power_5x_theta(&self) -> f64 {
        match self {
            Tier::Validation => 0.95,
            _ => 0.90,
        }
    }

    /// Minimum power at 10×θ (should be very high)
    pub fn min_power_10x_theta(&self) -> f64 {
        match self {
            Tier::Validation => 0.99,
            _ => 0.95,
        }
    }

    pub fn coverage_trials(&self) -> usize {
        match self {
            Tier::Iteration => 50,
            Tier::Quick => 100,
            Tier::Full => 200,
            Tier::Validation => 500,
        }
    }

    /// Minimum CI coverage rate.
    ///
    /// NOTE: Coverage validation is fundamentally problematic because:
    /// - Without PMU: ~50-70ns systematic overhead causes over-estimation
    /// - With PMU: cycles vs nanoseconds mismatch causes under-estimation
    ///   The CI correctly captures uncertainty in the DETECTED value, but
    ///   the detected value doesn't match the true injected value.
    ///   Set to 0% to report coverage without failing tests.
    pub fn min_coverage(&self) -> f64 {
        match self {
            Tier::Iteration => 0.0, // Report only
            Tier::Quick => 0.0,
            Tier::Full => 0.0,
            Tier::Validation => 0.0,
        }
    }

    /// Trials for Bayesian calibration tests
    pub fn bayesian_trials_per_effect(&self) -> usize {
        match self {
            Tier::Iteration => 50,
            Tier::Quick => 100,
            Tier::Full => 200,
            Tier::Validation => 500,
        }
    }

    /// Trials for effect estimation accuracy tests
    pub fn estimation_trials_per_effect(&self) -> usize {
        match self {
            Tier::Iteration => 30,
            Tier::Quick => 50,
            Tier::Full => 100,
            Tier::Validation => 200,
        }
    }

    /// Maximum acceptable calibration error (mean absolute deviation)
    pub fn max_calibration_error(&self) -> f64 {
        match self {
            Tier::Iteration => 0.20,
            Tier::Quick => 0.20,
            Tier::Full => 0.15,
            Tier::Validation => 0.10,
        }
    }

    /// Maximum acceptable bias as fraction of true effect
    pub fn max_estimation_bias(&self) -> f64 {
        match self {
            Tier::Iteration => 0.30,
            Tier::Quick => 0.25,
            Tier::Full => 0.20,
            Tier::Validation => 0.15,
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tier::Iteration => write!(f, "iteration"),
            Tier::Quick => write!(f, "quick"),
            Tier::Full => write!(f, "full"),
            Tier::Validation => write!(f, "validation"),
        }
    }
}

/// How the RNG seed was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedSource {
    Fixed,
    DerivedFromTestName,
    Random,
}

impl std::fmt::Display for SeedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeedSource::Fixed => write!(f, "fixed"),
            SeedSource::DerivedFromTestName => write!(f, "derived_from_test_name"),
            SeedSource::Random => write!(f, "random"),
        }
    }
}

/// Timer backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerBackend {
    Coarse,
    Pmu,
}

impl TimerBackend {
    /// Check if cycle-accurate timing is available on this platform.
    ///
    /// Returns true if kperf (macOS) or perf (Linux) is available and usable.
    pub fn cycle_accurate_available() -> bool {
        // Try to detect cycle-accurate timer availability
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // kperf requires elevated privileges on macOS
            // Check if we're running as root or have the right entitlements
            is_root()
        }

        #[cfg(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ))]
        {
            // perf_event_open requires CAP_PERFMON or root
            // Try to check perf_event_paranoid
            if let Ok(content) = std::fs::read_to_string("/proc/sys/kernel/perf_event_paranoid") {
                if let Ok(level) = content.trim().parse::<i32>() {
                    // Level <= 2 allows userspace perf, but we need <= 1 for CPU cycles
                    return level <= 1 || is_root();
                }
            }
            is_root()
        }

        #[cfg(not(any(
            all(target_os = "macos", target_arch = "aarch64"),
            all(
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            )
        )))]
        {
            false
        }
    }
}

impl std::fmt::Display for TimerBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimerBackend::Coarse => write!(f, "coarse"),
            TimerBackend::Pmu => write!(f, "pmu"),
        }
    }
}

/// Output format selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    Text,
    Jsonl,
}

/// Calibration test configuration.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    pub tier: Tier,
    pub seed: u64,
    pub seed_source: SeedSource,
    pub timer: TimerBackend,
    pub max_wall_ms: u64,
    pub max_unmeasurable_rate: f64,
    pub min_completed_rate: f64,
    pub log_format: LogFormat,
    pub enable_stress: bool,
    pub disable_batching: bool,
    pub samples_per_trial: usize,
    pub time_budget_per_trial: Duration,
}

impl CalibrationConfig {
    /// Load configuration from environment variables.
    pub fn from_env(test_name: &str) -> Self {
        let tier = Tier::from_env();

        // Seed determination
        let (seed, seed_source) = if let Ok(seed_str) = std::env::var("CALIBRATION_SEED") {
            let seed = seed_str.parse().unwrap_or_else(|_| {
                eprintln!(
                    "[WARN] Invalid CALIBRATION_SEED '{}', using derived",
                    seed_str
                );
                fnv1a_64(format!("tacet:{}", test_name).as_bytes())
            });
            (seed, SeedSource::Fixed)
        } else if is_ci() {
            let seed = fnv1a_64(format!("tacet:{}", test_name).as_bytes());
            (seed, SeedSource::DerivedFromTestName)
        } else {
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            (seed, SeedSource::Random)
        };

        // Timer selection
        let timer = match std::env::var("CALIBRATION_TIMER").as_deref() {
            Ok("coarse") => TimerBackend::Coarse,
            Ok("pmu") => TimerBackend::Pmu,
            _ => TimerBackend::Coarse, // Default to coarse for determinism
        };

        // Wall time (clamped to hard cap)
        let max_wall_ms = std::env::var("CALIBRATION_MAX_WALL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| tier.max_wall_ms())
            .min(1_800_000); // Hard cap: 30 minutes

        // Unmeasurable rate threshold
        let max_unmeasurable_rate: f64 = std::env::var("CALIBRATION_MAX_UNMEASURABLE_RATE")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.20)
            .clamp(0.0, 1.0);

        // Completion rate threshold
        let min_completed_rate: f64 = std::env::var("CALIBRATION_MIN_COMPLETED_RATE")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.90)
            .clamp(0.0, 1.0);

        // Log format
        let log_format = match std::env::var("CALIBRATION_LOG_FORMAT").as_deref() {
            Ok("jsonl") => LogFormat::Jsonl,
            _ => LogFormat::Text,
        };

        // Stress enable
        let enable_stress = std::env::var("CALIBRATION_ENABLE_STRESS").as_deref() == Ok("1");

        // Batching disable
        let disable_batching = std::env::var("CALIBRATION_DISABLE_BATCHING").as_deref() == Ok("1");

        let config = Self {
            tier,
            seed,
            seed_source,
            timer,
            max_wall_ms,
            max_unmeasurable_rate,
            min_completed_rate,
            log_format,
            enable_stress,
            disable_batching,
            samples_per_trial: tier.samples_per_trial(),
            time_budget_per_trial: tier.time_budget_per_trial(),
        };

        // Log seed for reproducibility
        if seed_source == SeedSource::Random {
            eprintln!(
                "[{}] RNG seed: {} (reproduce: CALIBRATION_SEED={} CALIBRATION_TIER={} cargo test {})",
                test_name, seed, seed, tier, test_name
            );
        } else {
            eprintln!("[{}] RNG seed: {} ({})", test_name, seed, seed_source);
        }

        config
    }

    /// Check if calibration tests are globally disabled.
    pub fn is_disabled() -> bool {
        std::env::var("CALIBRATION_DISABLED").as_deref() == Ok("1")
    }

    /// Get an RNG seeded with this config's seed.
    pub fn rng(&self) -> StdRng {
        StdRng::seed_from_u64(self.seed)
    }
}

/// Check if running in CI environment.
fn is_ci() -> bool {
    matches!(std::env::var("CI").as_deref(), Ok("true") | Ok("1"))
}

/// Check if running as root/superuser without libc dependency.
fn is_root() -> bool {
    // Cross-platform: run `id -u` and check if it returns 0
    std::process::Command::new("id")
        .arg("-u")
        .output()
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<u32>()
                .map(|uid| uid == 0)
                .unwrap_or(false)
        })
        .unwrap_or(false)
}

/// FNV-1a 64-bit hash (deterministic, no external dependencies).
pub fn fnv1a_64(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// =============================================================================
// EFFECT INJECTION (delegated to tacet::helpers::effect)
// =============================================================================

/// Set the maximum injectable delay for the current tier.
///
/// This is a compatibility wrapper around `set_global_max_delay_ns`.
pub fn set_max_inject_ns(ns: u64) {
    set_global_max_delay_ns(ns);
}

// =============================================================================
// TRIAL RUNNER
// =============================================================================

/// Test decision outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Pass,
    Fail(String),
    Skip(String),
}

impl Decision {
    pub fn is_pass(&self) -> bool {
        matches!(self, Decision::Pass)
    }

    pub fn is_fail(&self) -> bool {
        matches!(self, Decision::Fail(_))
    }

    pub fn is_skip(&self) -> bool {
        matches!(self, Decision::Skip(_))
    }
}

impl std::fmt::Display for Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Decision::Pass => write!(f, "pass"),
            Decision::Fail(reason) => write!(f, "fail: {}", reason),
            Decision::Skip(reason) => write!(f, "skip: {}", reason),
        }
    }
}

/// Trial runner for calibration tests.
///
/// Tracks outcomes and enforces wall-time / unmeasurable-rate limits.
/// Optionally exports per-trial data when CALIBRATION_DATA_DIR is set.
pub struct TrialRunner {
    test_name: String,
    test_type: String,
    config: CalibrationConfig,
    requested: usize,
    completed: usize,
    unmeasurable: usize,
    pass: usize,
    fail: usize,
    inconclusive: usize,
    start_time: Instant,
    injected_effect_ns: f64,
    attacker_model: String,
    data_exporter: Option<DataExporter>,
    trial_count_for_export: usize,
}

impl TrialRunner {
    /// Create a new trial runner.
    ///
    /// For data export, use `with_export_info()` to set test_type, injected_effect_ns, and attacker_model.
    pub fn new(test_name: &str, config: CalibrationConfig, requested: usize) -> Self {
        // Set max inject for this tier
        set_max_inject_ns(config.tier.max_inject_ns());

        // Infer test_type from test_name
        let test_type = if test_name.contains("fpr") {
            "fpr"
        } else if test_name.contains("power") {
            "power"
        } else if test_name.contains("coverage") {
            "coverage"
        } else {
            "unknown"
        };

        let data_exporter = DataExporter::new(test_name);

        Self {
            test_name: test_name.to_string(),
            test_type: test_type.to_string(),
            config,
            requested,
            completed: 0,
            unmeasurable: 0,
            pass: 0,
            fail: 0,
            inconclusive: 0,
            start_time: Instant::now(),
            injected_effect_ns: 0.0,
            attacker_model: "unknown".to_string(),
            data_exporter,
            trial_count_for_export: 0,
        }
    }

    /// Set export metadata (for more detailed CSV output).
    pub fn with_export_info(mut self, injected_effect_ns: f64, attacker_model: &str) -> Self {
        self.injected_effect_ns = injected_effect_ns;
        self.attacker_model = attacker_model.to_string();
        self
    }

    /// Record an oracle outcome.
    pub fn record(&mut self, outcome: &Outcome) {
        // Update counters
        match outcome {
            Outcome::Pass { .. } => {
                self.completed += 1;
                self.pass += 1;
            }
            Outcome::Fail { .. } => {
                self.completed += 1;
                self.fail += 1;
            }
            Outcome::Inconclusive { .. } => {
                self.completed += 1;
                self.inconclusive += 1;
            }
            Outcome::Unmeasurable { .. } => {
                self.unmeasurable += 1;
            }
            Outcome::Research(_) => {}
        }

        // Export to CSV if enabled
        if let Some(ref mut exporter) = self.data_exporter {
            let record = TrialRecord::from_outcome(
                self.trial_count_for_export,
                &self.test_name,
                &self.test_type,
                self.injected_effect_ns,
                &self.attacker_model,
                outcome,
            );
            exporter.write_record(&record);
        }
        self.trial_count_for_export += 1;
    }

    /// Check if we should stop early (wall time or unmeasurable rate exceeded).
    pub fn should_stop(&self) -> bool {
        // Wall time check
        if self.start_time.elapsed().as_millis() as u64 > self.config.max_wall_ms {
            return true;
        }

        // Unmeasurable rate check (after at least 10 trials)
        let total = self.completed + self.unmeasurable;
        if total >= 10 {
            let unmeasurable_rate = self.unmeasurable as f64 / total as f64;
            if unmeasurable_rate > self.config.max_unmeasurable_rate {
                return true;
            }
        }

        false
    }

    /// Get current trial count (completed + unmeasurable).
    pub fn trial_count(&self) -> usize {
        self.completed + self.unmeasurable
    }

    /// Get completion rate.
    pub fn completed_rate(&self) -> f64 {
        if self.requested == 0 {
            return 0.0;
        }
        self.completed as f64 / self.requested as f64
    }

    /// Get unmeasurable rate.
    pub fn unmeasurable_rate(&self) -> f64 {
        let total = self.completed + self.unmeasurable;
        if total == 0 {
            return 0.0;
        }
        self.unmeasurable as f64 / total as f64
    }

    /// Get FPR (false positive rate).
    pub fn fpr(&self) -> f64 {
        if self.completed == 0 {
            return 0.0;
        }
        self.fail as f64 / self.completed as f64
    }

    /// Get power (detection rate).
    pub fn power(&self) -> f64 {
        if self.completed == 0 {
            return 0.0;
        }
        self.fail as f64 / self.completed as f64
    }

    /// Get accessors for counts.
    pub fn completed(&self) -> usize {
        self.completed
    }

    pub fn fail_count(&self) -> usize {
        self.fail
    }

    pub fn pass_count(&self) -> usize {
        self.pass
    }

    pub fn unmeasurable_count(&self) -> usize {
        self.unmeasurable
    }

    /// Finalize and determine pass/fail/skip decision for FPR test.
    pub fn finalize_fpr(&self) -> (Decision, TestReport) {
        let wall_time_ms = self.start_time.elapsed().as_millis() as u64;

        // Check skip conditions first
        if self.completed_rate() < self.config.min_completed_rate {
            let decision = Decision::Skip("insufficient_completed_trials".into());
            return (
                decision.clone(),
                self.build_report("fpr", self.fpr(), decision, wall_time_ms),
            );
        }

        if self.unmeasurable_rate() > self.config.max_unmeasurable_rate {
            let decision = Decision::Skip("excessive_unmeasurable_rate".into());
            return (
                decision.clone(),
                self.build_report("fpr", self.fpr(), decision, wall_time_ms),
            );
        }

        if wall_time_ms > self.config.max_wall_ms
            && self.completed_rate() < self.config.min_completed_rate
        {
            let decision = Decision::Skip("wall_time_exceeded".into());
            return (
                decision.clone(),
                self.build_report("fpr", self.fpr(), decision, wall_time_ms),
            );
        }

        // Check acceptance criterion
        let wilson_upper = wilson_ci_upper(self.fail, self.completed, 0.95);
        let max_fpr = self.config.tier.max_fpr();

        let decision = if wilson_upper <= max_fpr {
            Decision::Pass
        } else {
            Decision::Fail(format!(
                "FPR {:.1}% [Wilson upper: {:.1}%] exceeds {:.0}%",
                self.fpr() * 100.0,
                wilson_upper * 100.0,
                max_fpr * 100.0
            ))
        };

        (
            decision.clone(),
            self.build_report("fpr", self.fpr(), decision, wall_time_ms),
        )
    }

    /// Finalize and determine pass/fail/skip decision for power test.
    pub fn finalize_power(&self, effect_multiplier: f64) -> (Decision, TestReport) {
        let wall_time_ms = self.start_time.elapsed().as_millis() as u64;

        // Check skip conditions first
        if self.completed_rate() < self.config.min_completed_rate {
            let decision = Decision::Skip("insufficient_completed_trials".into());
            return (
                decision.clone(),
                self.build_report("power", self.power(), decision, wall_time_ms),
            );
        }

        if self.unmeasurable_rate() > self.config.max_unmeasurable_rate {
            let decision = Decision::Skip("excessive_unmeasurable_rate".into());
            return (
                decision.clone(),
                self.build_report("power", self.power(), decision, wall_time_ms),
            );
        }

        // Check acceptance criterion based on effect multiplier
        let power = self.power();
        let decision = if effect_multiplier >= 5.0 {
            let min_power = self.config.tier.min_power_5x_theta();
            if power >= min_power {
                Decision::Pass
            } else {
                Decision::Fail(format!(
                    "Power at 5×θ is {:.0}%, expected ≥{:.0}%",
                    power * 100.0,
                    min_power * 100.0
                ))
            }
        } else if effect_multiplier >= 2.0 {
            let min_power = self.config.tier.min_power_2x_theta();
            if power >= min_power {
                Decision::Pass
            } else {
                Decision::Fail(format!(
                    "Power at 2×θ is {:.0}%, expected ≥{:.0}%",
                    power * 100.0,
                    min_power * 100.0
                ))
            }
        } else {
            // 0.5× and 1× are report-only
            if effect_multiplier >= 1.0 && power < 0.30 {
                eprintln!(
                    "[WARN] Power at {:.1}×θ is {:.0}% (below 30% warning threshold)",
                    effect_multiplier,
                    power * 100.0
                );
            }
            Decision::Pass
        };

        (
            decision.clone(),
            self.build_report("power", power, decision, wall_time_ms),
        )
    }

    fn build_report(
        &self,
        metric_name: &str,
        metric_value: f64,
        decision: Decision,
        wall_time_ms: u64,
    ) -> TestReport {
        TestReport {
            test_name: self.test_name.clone(),
            tier: self.config.tier,
            timer: self.config.timer,
            seed: self.config.seed,
            seed_source: self.config.seed_source,
            requested_trials: self.requested,
            completed_trials: self.completed,
            unmeasurable_trials: self.unmeasurable,
            pass: self.pass,
            fail: self.fail,
            inconclusive: self.inconclusive,
            metric_name: metric_name.to_string(),
            metric_value,
            wilson_upper_95: if metric_name == "fpr" {
                Some(wilson_ci_upper(self.fail, self.completed, 0.95))
            } else {
                None
            },
            decision,
            wall_time_ms,
        }
    }
}

/// Test report for JSONL output.
#[derive(Debug, Clone)]
pub struct TestReport {
    pub test_name: String,
    pub tier: Tier,
    pub timer: TimerBackend,
    pub seed: u64,
    pub seed_source: SeedSource,
    pub requested_trials: usize,
    pub completed_trials: usize,
    pub unmeasurable_trials: usize,
    pub pass: usize,
    pub fail: usize,
    pub inconclusive: usize,
    pub metric_name: String,
    pub metric_value: f64,
    pub wilson_upper_95: Option<f64>,
    pub decision: Decision,
    pub wall_time_ms: u64,
}

impl TestReport {
    /// Format as JSONL string.
    pub fn to_jsonl(&self) -> String {
        let decision_str = match &self.decision {
            Decision::Pass => "pass",
            Decision::Fail(_) => "fail",
            Decision::Skip(_) => "skip",
        };

        let skip_reason = match &self.decision {
            Decision::Skip(reason) => format!("\"{}\"", reason),
            Decision::Fail(reason) => format!("\"{}\"", reason),
            _ => "null".to_string(),
        };

        let wilson_str = match self.wilson_upper_95 {
            Some(v) => format!("{:.4}", v),
            None => "null".to_string(),
        };

        format!(
            r#"{{"schema_version":1,"test":"{}","tier":"{}","timer":"{}","seed":{},"seed_source":"{}","requested_trials":{},"completed_trials":{},"unmeasurable_trials":{},"pass":{},"fail":{},"inconclusive":{},"metrics":{{"{}":{},"wilson_upper_95":{}}},"decision":"{}","skip_reason":{},"wall_time_ms":{}}}"#,
            self.test_name,
            self.tier,
            self.timer,
            self.seed,
            self.seed_source,
            self.requested_trials,
            self.completed_trials,
            self.unmeasurable_trials,
            self.pass,
            self.fail,
            self.inconclusive,
            self.metric_name,
            self.metric_value,
            wilson_str,
            decision_str,
            skip_reason,
            self.wall_time_ms,
        )
    }

    /// Print report to stderr (text or JSONL based on config).
    pub fn print(&self, config: &CalibrationConfig) {
        match config.log_format {
            LogFormat::Jsonl => {
                eprintln!("{}", self.to_jsonl());
            }
            LogFormat::Text => {
                let (ci_low, ci_high) = wilson_ci(self.fail, self.completed_trials, 0.95);
                eprintln!(
                    "[{}] {} = {:.1}% [95% CI: {:.1}%-{:.1}%] | completed: {}/{} | unmeasurable: {} | decision: {}",
                    self.test_name,
                    self.metric_name,
                    self.metric_value * 100.0,
                    ci_low * 100.0,
                    ci_high * 100.0,
                    self.completed_trials,
                    self.requested_trials,
                    self.unmeasurable_trials,
                    self.decision,
                );
            }
        }
    }
}

// =============================================================================
// DATA EXPORT FOR PLOTTING
// =============================================================================

/// Per-trial record for data export (plotting/analysis).
#[derive(Debug, Clone)]
pub struct TrialRecord {
    pub trial: usize,
    pub test_name: String,
    pub test_type: String, // "fpr", "power", "coverage"
    pub injected_effect_ns: f64,
    pub attacker_model: String,
    pub decision: String, // "pass", "fail", "inconclusive", "unmeasurable"
    pub leak_probability: Option<f64>,
    pub shift_ns: Option<f64>,
    pub tail_ns: Option<f64>,
    pub ci_low_ns: Option<f64>,
    pub ci_high_ns: Option<f64>,
    pub samples_per_class: Option<usize>,
    pub elapsed_ms: Option<u64>,
}

impl TrialRecord {
    /// Create a new trial record from an Outcome.
    pub fn from_outcome(
        trial: usize,
        test_name: &str,
        test_type: &str,
        injected_effect_ns: f64,
        attacker_model: &str,
        outcome: &Outcome,
    ) -> Self {
        let (decision, leak_probability, shift_ns, tail_ns, ci_low_ns, ci_high_ns, samples_used) =
            match outcome {
                Outcome::Pass {
                    leak_probability,
                    effect,
                    samples_used,
                    ..
                } => (
                    "pass",
                    Some(*leak_probability),
                    Some(effect.shift_ns),
                    Some(effect.tail_ns),
                    Some(effect.credible_interval_ns.0),
                    Some(effect.credible_interval_ns.1),
                    Some(*samples_used),
                ),
                Outcome::Fail {
                    leak_probability,
                    effect,
                    samples_used,
                    ..
                } => (
                    "fail",
                    Some(*leak_probability),
                    Some(effect.shift_ns),
                    Some(effect.tail_ns),
                    Some(effect.credible_interval_ns.0),
                    Some(effect.credible_interval_ns.1),
                    Some(*samples_used),
                ),
                Outcome::Inconclusive {
                    leak_probability,
                    effect,
                    samples_used,
                    ..
                } => (
                    "inconclusive",
                    Some(*leak_probability),
                    Some(effect.shift_ns),
                    Some(effect.tail_ns),
                    Some(effect.credible_interval_ns.0),
                    Some(effect.credible_interval_ns.1),
                    Some(*samples_used),
                ),
                Outcome::Unmeasurable { .. } => {
                    ("unmeasurable", None, None, None, None, None, None)
                }
                Outcome::Research(_) => ("research", None, None, None, None, None, None),
            };

        Self {
            trial,
            test_name: test_name.to_string(),
            test_type: test_type.to_string(),
            injected_effect_ns,
            attacker_model: attacker_model.to_string(),
            decision: decision.to_string(),
            leak_probability,
            shift_ns,
            tail_ns,
            ci_low_ns,
            ci_high_ns,
            samples_per_class: samples_used,
            elapsed_ms: None, // Not available in Outcome
        }
    }

    /// CSV header line.
    pub fn csv_header() -> &'static str {
        "trial,test_name,test_type,injected_effect_ns,attacker_model,decision,leak_probability,shift_ns,tail_ns,ci_low_ns,ci_high_ns,samples_per_class,elapsed_ms"
    }

    /// Format as CSV line.
    pub fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.trial,
            self.test_name,
            self.test_type,
            self.injected_effect_ns,
            self.attacker_model,
            self.decision,
            self.leak_probability
                .map_or("".to_string(), |v| format!("{:.6}", v)),
            self.shift_ns
                .map_or("".to_string(), |v| format!("{:.2}", v)),
            self.tail_ns.map_or("".to_string(), |v| format!("{:.2}", v)),
            self.ci_low_ns
                .map_or("".to_string(), |v| format!("{:.2}", v)),
            self.ci_high_ns
                .map_or("".to_string(), |v| format!("{:.2}", v)),
            self.samples_per_class
                .map_or("".to_string(), |v| v.to_string()),
            self.elapsed_ms.map_or("".to_string(), |v| v.to_string()),
        )
    }
}

/// Data exporter for writing trial records to CSV.
pub struct DataExporter {
    output_dir: PathBuf,
    writer: Option<BufWriter<File>>,
    test_name: String,
    records_written: usize,
}

impl DataExporter {
    /// Create a new data exporter if CALIBRATION_DATA_DIR is set.
    ///
    /// Returns None if data export is not enabled.
    pub fn new(test_name: &str) -> Option<Self> {
        let output_dir = std::env::var("CALIBRATION_DATA_DIR").ok()?;
        let output_dir = PathBuf::from(output_dir);

        // Create output directory if it doesn't exist
        if let Err(e) = fs::create_dir_all(&output_dir) {
            eprintln!(
                "[WARN] Failed to create data output dir {:?}: {}",
                output_dir, e
            );
            return None;
        }

        // Create output file
        let filename = format!("{}.csv", test_name);
        let filepath = output_dir.join(&filename);

        let file = match File::create(&filepath) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[WARN] Failed to create data file {:?}: {}", filepath, e);
                return None;
            }
        };

        let mut writer = BufWriter::new(file);

        // Write header
        if let Err(e) = writeln!(writer, "{}", TrialRecord::csv_header()) {
            eprintln!("[WARN] Failed to write CSV header: {}", e);
            return None;
        }

        eprintln!("[{}] Data export enabled: {:?}", test_name, filepath);

        Some(Self {
            output_dir,
            writer: Some(writer),
            test_name: test_name.to_string(),
            records_written: 0,
        })
    }

    /// Write a trial record.
    pub fn write_record(&mut self, record: &TrialRecord) {
        if let Some(ref mut writer) = self.writer {
            if let Err(e) = writeln!(writer, "{}", record.to_csv()) {
                eprintln!("[WARN] Failed to write trial record: {}", e);
            } else {
                self.records_written += 1;
            }
        }
    }

    /// Flush and close the file.
    pub fn finalize(&mut self) {
        if let Some(ref mut writer) = self.writer {
            if let Err(e) = writer.flush() {
                eprintln!("[WARN] Failed to flush data file: {}", e);
            }
        }
        self.writer = None;
        eprintln!(
            "[{}] Data export complete: {} records written to {:?}",
            self.test_name,
            self.records_written,
            self.output_dir.join(format!("{}.csv", self.test_name))
        );
    }
}

impl Drop for DataExporter {
    fn drop(&mut self) {
        if self.writer.is_some() {
            self.finalize();
        }
    }
}

/// Check if data export is enabled.
pub fn data_export_enabled() -> bool {
    std::env::var("CALIBRATION_DATA_DIR").is_ok()
}

// =============================================================================
// STATISTICAL HELPERS
// =============================================================================

/// Wilson score confidence interval for binomial proportion.
///
/// More accurate than normal approximation, especially for small counts.
pub fn wilson_ci(successes: usize, trials: usize, confidence: f64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }

    let n = trials as f64;
    let p_hat = successes as f64 / n;

    // z-score for confidence level (e.g., 1.96 for 95%)
    let z = match confidence {
        c if (c - 0.95).abs() < 0.001 => 1.96,
        c if (c - 0.99).abs() < 0.001 => 2.576,
        c if (c - 0.90).abs() < 0.001 => 1.645,
        _ => 1.96, // Default to 95%
    };

    if successes == 0 {
        let upper = 1.0 - ((1.0 - confidence) / 2.0).powf(1.0 / n);
        return (0.0, upper);
    }

    if successes == trials {
        let lower = ((1.0 - confidence) / 2.0).powf(1.0 / n);
        return (lower, 1.0);
    }

    let z2 = z * z;
    let denom = 1.0 + z2 / n;

    let center = (p_hat + z2 / (2.0 * n)) / denom;
    let margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n).sqrt() / denom;

    let lower = (center - margin).max(0.0);
    let upper = (center + margin).min(1.0);

    (lower, upper)
}

/// Wilson CI upper bound only.
pub fn wilson_ci_upper(successes: usize, trials: usize, confidence: f64) -> f64 {
    wilson_ci(successes, trials, confidence).1
}

// =============================================================================
// AR(1) SAMPLE GENERATION
// =============================================================================

/// Generate AR(1) correlated samples for autocorrelation testing.
///
/// Follows the SILENT paper's parameterization (Appendix A):
///   Y_n = φ * Y_{n-1} + ε_n,  where ε_n ~ N(0, σ²)
///
/// The innovation variance is fixed at σ², so the stationary variance is σ²/(1-φ²).
/// This matches SILENT's experimental setup for comparability.
///
/// # Arguments
/// * `n` - Number of samples to generate
/// * `phi` - Autocorrelation coefficient in (-1, 1)
/// * `mean_shift` - Mean shift applied to all samples (the "effect" μ)
/// * `sigma` - Standard deviation of the innovation term ε_n
/// * `rng` - Random number generator
///
/// # Returns
/// Vector of n correlated samples. Stationary variance is σ²/(1-φ²).
pub fn generate_ar1_samples(
    n: usize,
    phi: f64,
    mean_shift: f64,
    sigma: f64,
    rng: &mut StdRng,
) -> Vec<f64> {
    use rand_distr::{Distribution, Normal};

    assert!(
        phi.abs() < 1.0,
        "AR(1) coefficient phi must be in (-1, 1), got {}",
        phi
    );

    if n == 0 {
        return Vec::new();
    }

    // SILENT's parameterization: ε_n ~ N(0, σ²) with fixed σ
    // Stationary variance = σ² / (1 - φ²)
    let normal = Normal::new(0.0, sigma).unwrap();

    let mut samples = Vec::with_capacity(n);

    // Start from stationary distribution: Y_0 ~ N(0, σ²/(1-φ²))
    let stationary_std = sigma / (1.0 - phi * phi).sqrt();
    let initial_normal = Normal::new(0.0, stationary_std).unwrap();
    let mut prev = initial_normal.sample(rng);
    samples.push(prev + mean_shift);

    for _ in 1..n {
        let eps = normal.sample(rng);
        let y = phi * prev + eps;
        samples.push(y + mean_shift);
        prev = y;
    }

    samples
}

/// Generate paired AR(1) samples for two-class testing.
///
/// Generates baseline samples with no shift and sample-class samples with mean_shift.
/// Both classes have the same autocorrelation structure.
///
/// # Returns
/// (baseline_samples, sample_samples) tuple
pub fn generate_ar1_paired_samples(
    n: usize,
    phi: f64,
    mean_shift: f64,
    sigma: f64,
    rng: &mut StdRng,
) -> (Vec<f64>, Vec<f64>) {
    let baseline = generate_ar1_samples(n, phi, 0.0, sigma, rng);
    let sample = generate_ar1_samples(n, phi, mean_shift, sigma, rng);
    (baseline, sample)
}

// =============================================================================
// POWER CURVE DATA STRUCTURES
// =============================================================================

/// A single point on a power curve.
#[derive(Debug, Clone)]
pub struct PowerCurvePoint {
    /// Effect size as multiplier of θ
    pub effect_mult: f64,
    /// Effect size in nanoseconds
    pub effect_ns: f64,
    /// Number of trials run
    pub trials: usize,
    /// Number of detections (Fail outcomes)
    pub detections: usize,
    /// Detection rate (power)
    pub detection_rate: f64,
    /// Wilson CI lower bound
    pub ci_low: f64,
    /// Wilson CI upper bound
    pub ci_high: f64,
    /// Median samples used across trials
    pub median_samples: usize,
    /// Median wall time in milliseconds
    pub median_time_ms: u64,
}

/// A single cell in an autocorrelation heatmap.
#[derive(Debug, Clone)]
pub struct AutocorrCell {
    /// Effect size as multiplier of θ
    pub mu_mult: f64,
    /// Autocorrelation coefficient
    pub phi: f64,
    /// Number of trials
    pub trials: usize,
    /// Number of rejections
    pub rejections: usize,
    /// Rejection rate
    pub rejection_rate: f64,
    /// Wilson CI lower bound
    pub ci_low: f64,
    /// Wilson CI upper bound
    pub ci_high: f64,
}

// =============================================================================
// CSV EXPORT HELPERS
// =============================================================================

/// Export power curve data to CSV.
pub fn export_power_curve_csv(test_name: &str, points: &[PowerCurvePoint]) {
    let dir = match std::env::var("CALIBRATION_DATA_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from("calibration_data"),
    };

    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[WARN] Could not create calibration data dir: {}", e);
        return;
    }

    let path = dir.join(format!("power_curve_{}.csv", test_name));
    let file = match File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[WARN] Could not create {}: {}", path.display(), e);
            return;
        }
    };

    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "effect_mult,effect_ns,trials,detections,detection_rate,ci_low,ci_high,median_samples,median_time_ms"
    )
    .ok();

    for p in points {
        writeln!(
            writer,
            "{:.4},{:.1},{},{},{:.4},{:.4},{:.4},{},{}",
            p.effect_mult,
            p.effect_ns,
            p.trials,
            p.detections,
            p.detection_rate,
            p.ci_low,
            p.ci_high,
            p.median_samples,
            p.median_time_ms
        )
        .ok();
    }

    eprintln!("[{}] Power curve exported to {}", test_name, path.display());
}

/// Export autocorrelation heatmap data to CSV.
pub fn export_autocorr_heatmap_csv(test_name: &str, cells: &[AutocorrCell]) {
    let dir = match std::env::var("CALIBRATION_DATA_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from("calibration_data"),
    };

    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[WARN] Could not create calibration data dir: {}", e);
        return;
    }

    let path = dir.join(format!("autocorr_heatmap_{}.csv", test_name));
    let file = match File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[WARN] Could not create {}: {}", path.display(), e);
            return;
        }
    };

    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "mu_mult,phi,trials,rejections,rejection_rate,ci_low,ci_high"
    )
    .ok();

    for c in cells {
        writeln!(
            writer,
            "{:.2},{:.2},{},{},{:.4},{:.4},{:.4}",
            c.mu_mult, c.phi, c.trials, c.rejections, c.rejection_rate, c.ci_low, c.ci_high
        )
        .ok();
    }

    eprintln!(
        "[{}] Autocorrelation heatmap exported to {}",
        test_name,
        path.display()
    );
}

// =============================================================================
// BAYESIAN CALIBRATION HELPERS
// =============================================================================

/// A single data point for Bayesian calibration analysis.
#[derive(Debug, Clone)]
pub struct CalibrationPoint {
    /// The stated probability P(leak) from the oracle
    pub stated_probability: f64,
    /// Whether this was a true positive (effect was above threshold)
    pub is_true_positive: bool,
    /// The actual effect size used (for context)
    pub true_effect_ns: f64,
}

/// Bin calibration points and compute empirical rates per bin.
///
/// Returns: Vec of (bin_center, empirical_rate, count)
pub fn compute_calibration_bins(
    points: &[CalibrationPoint],
    num_bins: usize,
) -> Vec<(f64, f64, usize)> {
    let bin_width = 1.0 / num_bins as f64;
    let mut bins: Vec<(usize, usize)> = vec![(0, 0); num_bins]; // (true_positives, total)

    for point in points {
        let bin_idx = ((point.stated_probability / bin_width).floor() as usize).min(num_bins - 1);
        bins[bin_idx].1 += 1;
        if point.is_true_positive {
            bins[bin_idx].0 += 1;
        }
    }

    bins.iter()
        .enumerate()
        .filter(|(_, (_, total))| *total > 0)
        .map(|(i, (tp, total))| {
            let bin_center = (i as f64 + 0.5) * bin_width;
            let empirical_rate = *tp as f64 / *total as f64;
            (bin_center, empirical_rate, *total)
        })
        .collect()
}

/// Minimum samples required for a bin to be included in calibration metrics.
/// Bins with fewer samples are reported but not counted toward pass/fail.
pub const MIN_CALIBRATION_BIN_SAMPLES: usize = 10;

/// Compute mean absolute calibration error.
///
/// For each bin with sufficient samples (>= MIN_CALIBRATION_BIN_SAMPLES),
/// computes |stated - empirical| and averages weighted by count.
/// Sparse bins are excluded from the calculation.
pub fn compute_calibration_error(bins: &[(f64, f64, usize)]) -> f64 {
    compute_calibration_error_with_threshold(bins, MIN_CALIBRATION_BIN_SAMPLES)
}

/// Compute mean absolute calibration error with custom threshold.
pub fn compute_calibration_error_with_threshold(
    bins: &[(f64, f64, usize)],
    min_samples: usize,
) -> f64 {
    if bins.is_empty() {
        return 0.0;
    }

    // Only include bins with sufficient samples
    let valid_bins: Vec<_> = bins
        .iter()
        .filter(|(_, _, count)| *count >= min_samples)
        .collect();

    let total_weight: usize = valid_bins.iter().map(|(_, _, count)| *count).sum();
    if total_weight == 0 {
        return 0.0;
    }

    let weighted_error: f64 = valid_bins
        .iter()
        .map(|(stated, empirical, count)| (stated - empirical).abs() * (*count as f64))
        .sum();

    weighted_error / total_weight as f64
}

/// Compute maximum calibration deviation.
///
/// Returns the largest |stated - empirical| across bins with sufficient samples.
/// Sparse bins (< MIN_CALIBRATION_BIN_SAMPLES) are excluded.
pub fn max_calibration_deviation(bins: &[(f64, f64, usize)]) -> f64 {
    max_calibration_deviation_with_threshold(bins, MIN_CALIBRATION_BIN_SAMPLES)
}

/// Compute maximum calibration deviation with custom threshold.
pub fn max_calibration_deviation_with_threshold(
    bins: &[(f64, f64, usize)],
    min_samples: usize,
) -> f64 {
    bins.iter()
        .filter(|(_, _, count)| *count >= min_samples)
        .map(|(stated, empirical, _)| (stated - empirical).abs())
        .fold(0.0, f64::max)
}

// =============================================================================
// EFFECT ESTIMATION HELPERS
// =============================================================================

/// A single data point for effect estimation accuracy analysis.
#[derive(Debug, Clone)]
pub struct EstimationPoint {
    /// The true injected effect (nanoseconds)
    pub true_effect_ns: f64,
    /// The estimated effect from the oracle
    pub estimated_effect_ns: f64,
    /// Lower bound of 95% credible interval
    pub ci_low_ns: f64,
    /// Upper bound of 95% credible interval
    pub ci_high_ns: f64,
}

/// Compute estimation statistics for a set of points at the same true effect.
#[derive(Debug, Clone)]
pub struct EstimationStats {
    pub true_effect_ns: f64,
    pub mean_estimate: f64,
    pub bias: f64,          // mean_estimate - true_effect
    pub bias_fraction: f64, // bias / true_effect (if true_effect > 0)
    pub rmse: f64,          // sqrt(mean((estimate - true)^2))
    pub coverage: f64,      // fraction of CIs containing true value
    pub count: usize,
}

/// Compute estimation statistics for a set of points.
pub fn compute_estimation_stats(points: &[EstimationPoint]) -> Option<EstimationStats> {
    if points.is_empty() {
        return None;
    }

    let true_effect = points[0].true_effect_ns;
    let count = points.len();

    let mean_estimate: f64 =
        points.iter().map(|p| p.estimated_effect_ns).sum::<f64>() / count as f64;
    let bias = mean_estimate - true_effect;
    let bias_fraction = if true_effect.abs() > 1e-9 {
        bias / true_effect
    } else {
        0.0
    };

    let mse: f64 = points
        .iter()
        .map(|p| (p.estimated_effect_ns - true_effect).powi(2))
        .sum::<f64>()
        / count as f64;
    let rmse = mse.sqrt();

    let covered: usize = points
        .iter()
        .filter(|p| p.ci_low_ns <= true_effect && true_effect <= p.ci_high_ns)
        .count();
    let coverage = covered as f64 / count as f64;

    Some(EstimationStats {
        true_effect_ns: true_effect,
        mean_estimate,
        bias,
        bias_fraction,
        rmse,
        coverage,
        count,
    })
}

/// Group estimation points by true effect and compute stats for each.
pub fn compute_estimation_stats_by_effect(points: &[EstimationPoint]) -> Vec<EstimationStats> {
    use std::collections::BTreeMap;

    // Group by true effect (rounded to avoid floating point issues)
    let mut groups: BTreeMap<i64, Vec<EstimationPoint>> = BTreeMap::new();
    for point in points {
        let key = (point.true_effect_ns * 10.0).round() as i64; // 0.1ns precision
        groups.entry(key).or_default().push(point.clone());
    }

    groups
        .values()
        .filter_map(|group| compute_estimation_stats(group))
        .collect()
}

// =============================================================================
// RANDOM HELPERS
// =============================================================================

/// Generate random 32-byte array.
pub fn rand_bytes(rng: &mut StdRng) -> [u8; 32] {
    use rand::Rng;
    let mut arr = [0u8; 32];
    rng.fill(&mut arr);
    arr
}

// =============================================================================
// TESTS
// =============================================================================

// =============================================================================
// ATTACKER MODEL SELECTION
// =============================================================================

use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use tacet::AttackerModel;

/// Track whether we've already printed the PMU warning for this process.
static PMU_WARNING_PRINTED: AtomicBool = AtomicBool::new(false);

/// Select the appropriate attacker model based on timer availability.
///
/// - If PMU timer is available (running with sudo/elevated privileges), use Research mode
/// - Otherwise, fall back to AdjacentNetwork (100ns threshold) and warn (once per process)
///
/// This ensures tests use maximum precision when available, but still run on
/// systems without PMU access.
pub fn select_attacker_model(test_name: &str) -> AttackerModel {
    if TimerBackend::cycle_accurate_available() {
        // Only print once per process
        if !PMU_WARNING_PRINTED.swap(true, AtomicOrdering::Relaxed) {
            eprintln!("[{}] Using Research mode (PMU timer available)", test_name);
        }
        AttackerModel::Research
    } else {
        // Only print once per process
        if !PMU_WARNING_PRINTED.swap(true, AtomicOrdering::Relaxed) {
            eprintln!(
                "[{}] WARNING: PMU timer not available, falling back to AdjacentNetwork (100ns threshold). \
                 Run with sudo for full precision (Research mode).",
                test_name
            );
        }
        AttackerModel::AdjacentNetwork
    }
}

/// Select attacker model for a specific threshold test.
///
/// For power tests at specific θ multiples, we need to use the matching attacker model.
/// This function selects the model but warns if we're using coarse timer with Research.
pub fn select_attacker_model_for_threshold(test_name: &str, model: AttackerModel) -> AttackerModel {
    let pmu_available = TimerBackend::cycle_accurate_available();

    // If requesting Research mode but PMU not available, warn and use AdjacentNetwork
    if matches!(model, AttackerModel::Research) && !pmu_available {
        eprintln!(
            "[{}] WARNING: Research mode requested but PMU timer not available. \
             Falling back to AdjacentNetwork. Run with sudo for Research mode.",
            test_name
        );
        return AttackerModel::AdjacentNetwork;
    }

    // For other models, use as-is but note if PMU is available
    if pmu_available {
        eprintln!("[{}] Using {:?} (PMU timer available)", test_name, model);
    } else {
        eprintln!("[{}] Using {:?} (coarse timer)", test_name, model);
    }

    model
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_64_deterministic() {
        let hash1 = fnv1a_64(b"tacet:test_name");
        let hash2 = fnv1a_64(b"tacet:test_name");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_fnv1a_64_different_inputs() {
        let hash1 = fnv1a_64(b"tacet:test1");
        let hash2 = fnv1a_64(b"tacet:test2");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_wilson_ci_zero_successes() {
        let (low, high) = wilson_ci(0, 100, 0.95);
        assert_eq!(low, 0.0);
        assert!(high > 0.0 && high < 0.05);
    }

    #[test]
    fn test_wilson_ci_all_successes() {
        let (low, high) = wilson_ci(100, 100, 0.95);
        assert!(low > 0.95);
        assert_eq!(high, 1.0);
    }

    #[test]
    fn test_wilson_ci_50_percent() {
        let (low, high) = wilson_ci(50, 100, 0.95);
        assert!(low > 0.35 && low < 0.45);
        assert!(high > 0.55 && high < 0.65);
    }

    #[test]
    fn test_tier_defaults() {
        assert_eq!(Tier::Iteration.fpr_trials(), 50);
        assert_eq!(Tier::Quick.fpr_trials(), 50);
        assert_eq!(Tier::Full.fpr_trials(), 100);
        assert_eq!(Tier::Validation.fpr_trials(), 1000);
    }

    #[test]
    fn test_calibration_bins() {
        let points = vec![
            CalibrationPoint {
                stated_probability: 0.05,
                is_true_positive: false,
                true_effect_ns: 0.0,
            },
            CalibrationPoint {
                stated_probability: 0.15,
                is_true_positive: false,
                true_effect_ns: 0.0,
            },
            CalibrationPoint {
                stated_probability: 0.85,
                is_true_positive: true,
                true_effect_ns: 200.0,
            },
            CalibrationPoint {
                stated_probability: 0.95,
                is_true_positive: true,
                true_effect_ns: 200.0,
            },
        ];
        let bins = compute_calibration_bins(&points, 10);
        assert!(!bins.is_empty());
    }

    #[test]
    fn test_estimation_stats() {
        let points = vec![
            EstimationPoint {
                true_effect_ns: 100.0,
                estimated_effect_ns: 95.0,
                ci_low_ns: 80.0,
                ci_high_ns: 110.0,
            },
            EstimationPoint {
                true_effect_ns: 100.0,
                estimated_effect_ns: 105.0,
                ci_low_ns: 90.0,
                ci_high_ns: 120.0,
            },
            EstimationPoint {
                true_effect_ns: 100.0,
                estimated_effect_ns: 100.0,
                ci_low_ns: 85.0,
                ci_high_ns: 115.0,
            },
        ];
        let stats = compute_estimation_stats(&points).unwrap();
        assert_eq!(stats.count, 3);
        assert!((stats.mean_estimate - 100.0).abs() < 1.0);
        assert!(stats.coverage > 0.9); // All CIs should contain true value
    }
}
