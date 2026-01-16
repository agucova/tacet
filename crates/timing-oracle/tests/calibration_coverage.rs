//! Coverage calibration tests.
//!
//! These tests verify that the Bayesian credible intervals have correct coverage:
//! the 95% CI should contain the true effect approximately 95% of the time.
//!
//! We inject known timing effects (uniform shifts) and check if the reported
//! credible interval contains the true injected value.
//!
//! NOTE: The spec specifies using `shift_ns_ci95` for separate shift/tail CIs,
//! but that API extension hasn't been implemented yet. These tests use the
//! existing `credible_interval_ns` which provides a CI for the total effect.
//! Since we inject pure uniform shifts, this is a reasonable approximation.
//!
//! See docs/calibration-test-spec.md for the full specification.

mod calibration_utils;

use calibration_utils::{busy_wait_ns, select_attacker_model, CalibrationConfig, Decision};
use rand::{rngs::StdRng, Rng, SeedableRng};
use timing_oracle::helpers::InputPair;
use timing_oracle::{Outcome, TimingOracle};

// =============================================================================
// COVERAGE TRIAL RUNNER
// =============================================================================

/// Trial runner specialized for coverage tests.
pub struct CoverageRunner {
    config: CalibrationConfig,
    requested: usize,
    completed: usize,
    unmeasurable: usize,
    covered: usize,
    not_covered: usize,
    start_time: std::time::Instant,
    injected_effect_ns: f64,
}

impl CoverageRunner {
    pub fn new(config: CalibrationConfig, requested: usize, injected_effect_ns: f64) -> Self {
        calibration_utils::set_max_inject_ns(config.tier.max_inject_ns());

        Self {
            config,
            requested,
            completed: 0,
            unmeasurable: 0,
            covered: 0,
            not_covered: 0,
            start_time: std::time::Instant::now(),
            injected_effect_ns,
        }
    }

    /// Record an oracle outcome and check if CI covers the true value.
    pub fn record(&mut self, outcome: &Outcome) {
        match outcome {
            Outcome::Pass { effect, .. }
            | Outcome::Fail { effect, .. }
            | Outcome::Inconclusive { effect, .. } => {
                self.completed += 1;

                // Check if credible interval contains the true injected effect
                let (ci_low, ci_high) = effect.credible_interval_ns;
                let true_effect = self.injected_effect_ns;

                // Debug: print first few results
                if self.completed <= 3 {
                    eprintln!(
                        "[DEBUG] Trial {}: shift={:.1}ns, tail={:.1}ns, CI=[{:.1}, {:.1}], true={:.1}ns",
                        self.completed, effect.shift_ns, effect.tail_ns, ci_low, ci_high, true_effect
                    );
                }

                // The CI is for total effect magnitude
                // Check if the true injected effect falls within the CI
                let _detected_shift = effect.shift_ns.abs();
                if ci_low <= true_effect && true_effect <= ci_high {
                    self.covered += 1;
                } else {
                    self.not_covered += 1;
                }
            }
            Outcome::Unmeasurable { .. } => {
                self.unmeasurable += 1;
            }
        }
    }

    /// Check if we should stop early.
    pub fn should_stop(&self) -> bool {
        if self.start_time.elapsed().as_millis() as u64 > self.config.max_wall_ms {
            return true;
        }

        let total = self.completed + self.unmeasurable;
        if total >= 10 {
            let unmeasurable_rate = self.unmeasurable as f64 / total as f64;
            if unmeasurable_rate > self.config.max_unmeasurable_rate {
                return true;
            }
        }

        false
    }

    /// Get coverage rate.
    pub fn coverage(&self) -> f64 {
        if self.completed == 0 {
            return 0.0;
        }
        self.covered as f64 / self.completed as f64
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

    pub fn completed(&self) -> usize {
        self.completed
    }

    pub fn covered_count(&self) -> usize {
        self.covered
    }

    #[allow(dead_code)]
    pub fn not_covered_count(&self) -> usize {
        self.not_covered
    }

    /// Finalize and determine pass/fail/skip decision.
    pub fn finalize(&self) -> Decision {
        if self.completed_rate() < self.config.min_completed_rate {
            return Decision::Skip("insufficient_completed_trials".into());
        }

        if self.unmeasurable_rate() > self.config.max_unmeasurable_rate {
            return Decision::Skip("excessive_unmeasurable_rate".into());
        }

        let coverage = self.coverage();
        let min_coverage = self.config.tier.min_coverage();

        if coverage >= min_coverage {
            Decision::Pass
        } else {
            Decision::Fail(format!(
                "Coverage {:.1}% below minimum {:.0}%",
                coverage * 100.0,
                min_coverage * 100.0
            ))
        }
    }
}

// =============================================================================
// EFFECT SIZE DEFINITIONS
// =============================================================================

/// Effect sizes to test for coverage calibration.
/// These represent effects at various multiples of θ for AdjacentNetwork (θ=100ns).
///
/// With PMU timers (requires sudo), measurement overhead is minimal (<1ns),
/// so we can test coverage across a range of effect sizes.
const COVERAGE_EFFECTS_NS: [u64; 5] = [
    100,  // 1×θ - at threshold
    200,  // 2×θ - above threshold
    500,  // 5×θ - strong effect
    1000, // 10×θ - very strong effect
    2000, // 20×θ - large effect
];

// =============================================================================
// ITERATION TIER TESTS (quick feedback during development)
// =============================================================================

/// Quick iteration coverage check.
///
/// Runs fewer trials for faster iteration during development.
/// Uses 500ns and 1000ns effects (5×θ and 10×θ).
#[test]
fn coverage_iteration() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[coverage_iteration] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    run_coverage_test(
        "coverage_iteration",
        &COVERAGE_EFFECTS_NS[2..4], // 500ns and 1000ns for quick check
    );
}

// =============================================================================
// QUICK TIER TESTS (run on every PR)
// =============================================================================

/// Coverage test with 500ns injected effect.
///
/// Injects a known 500ns timing difference and verifies the 95% CI
/// contains the true value at least min_coverage% of the time.
#[test]
fn coverage_quick_500ns() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[coverage_quick_500ns] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    let test_name = "coverage_quick_500ns";
    let config = CalibrationConfig::from_env(test_name);
    let mut rng = config.rng();

    let injected_ns: u64 = 500;
    let trials = config.tier.coverage_trials();
    let mut runner = CoverageRunner::new(config.clone(), trials, injected_ns as f64);

    eprintln!(
        "[{}] Starting {} trials with {}ns injected effect (tier: {})",
        test_name, trials, injected_ns, config.tier
    );

    for trial in 0..trials {
        if runner.should_stop() {
            eprintln!("[{}] Early stop at trial {}", test_name, trial);
            break;
        }

        // Use simple boolean to distinguish baseline vs sample
        let delay = injected_ns;
        let inputs = InputPair::new(|| false, || true);

        // Use AdjacentNetwork (100ns threshold) since our effect is 500ns (5× threshold)
        let outcome = TimingOracle::for_attacker(select_attacker_model(test_name))
            .max_samples(config.samples_per_trial)
            .time_budget(config.time_budget_per_trial)
            .test(inputs, move |&is_sample| {
                // Base operation (~2μs) to ensure:
                // 1. Operation is measurable on coarse timers
                // 2. Adaptive batching uses K=1 (no multiplier)
                busy_wait_ns(2000);

                // Inject delay only for sample class
                if is_sample {
                    busy_wait_ns(delay);
                }
            });

        runner.record(&outcome);

        // Progress logging
        if (trial + 1) % 20 == 0 || trial + 1 == trials {
            eprintln!(
                "[{}] Trial {}/{}: {}/{} covered ({:.1}% coverage)",
                test_name,
                trial + 1,
                trials,
                runner.covered_count(),
                runner.completed(),
                runner.coverage() * 100.0
            );
        }

        rng = StdRng::seed_from_u64(rng.random());
    }

    let decision = runner.finalize();

    eprintln!(
        "[{}] Final: {}/{} covered ({:.1}% coverage) - {}",
        test_name,
        runner.covered_count(),
        runner.completed(),
        runner.coverage() * 100.0,
        decision
    );

    match decision {
        Decision::Pass => {
            eprintln!("[{}] PASSED", test_name);
        }
        Decision::Skip(reason) => {
            eprintln!("[{}] SKIPPED: {}", test_name, reason);
        }
        Decision::Fail(reason) => {
            panic!("[{}] FAILED: {}", test_name, reason);
        }
    }
}

/// Coverage test with multiple effect sizes.
///
/// Tests coverage calibration across 200ns, 500ns, and 1000ns effects.
#[test]
fn coverage_quick_multi_effect() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[coverage_quick_multi_effect] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    let test_name = "coverage_quick_multi_effect";
    let config = CalibrationConfig::from_env(test_name);

    // Test multiple effect sizes: 200ns (2×θ), 500ns (5×θ), 1000ns (10×θ)
    let effect_sizes = [200u64, 500, 1000];
    let trials_per_effect = config.tier.coverage_trials() / effect_sizes.len();

    eprintln!(
        "[{}] Testing {} effect sizes with {} trials each (tier: {})",
        test_name,
        effect_sizes.len(),
        trials_per_effect,
        config.tier
    );

    let mut total_covered = 0;
    let mut total_completed = 0;

    for &effect_ns in &effect_sizes {
        let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(effect_ns));
        let mut runner = CoverageRunner::new(config.clone(), trials_per_effect, effect_ns as f64);

        for _ in 0..trials_per_effect {
            if runner.should_stop() {
                break;
            }

            let delay = effect_ns;
            let inputs = InputPair::new(|| false, || true);

            let outcome = TimingOracle::for_attacker(select_attacker_model(test_name))
                .max_samples(config.samples_per_trial)
                .time_budget(config.time_budget_per_trial)
                .test(inputs, move |&is_sample| {
                    // Base operation (~2μs) to ensure:
                    // 1. Operation is measurable on coarse timers
                    // 2. Adaptive batching uses K=1 (no multiplier)
                    busy_wait_ns(2000);

                    // Inject delay for sample class
                    if is_sample {
                        busy_wait_ns(delay);
                    }
                });

            runner.record(&outcome);
            rng = StdRng::seed_from_u64(rng.random());
        }

        total_covered += runner.covered_count();
        total_completed += runner.completed();

        eprintln!(
            "[{}] {}ns: {}/{} covered ({:.1}%)",
            test_name,
            effect_ns,
            runner.covered_count(),
            runner.completed(),
            runner.coverage() * 100.0
        );
    }

    // Overall decision
    let overall_coverage = if total_completed > 0 {
        total_covered as f64 / total_completed as f64
    } else {
        0.0
    };

    let min_coverage = config.tier.min_coverage();
    let min_trials = trials_per_effect * effect_sizes.len() / 2;
    let decision = if total_completed < min_trials {
        Decision::Skip("insufficient_completed_trials".into())
    } else if overall_coverage >= min_coverage {
        Decision::Pass
    } else {
        Decision::Fail(format!(
            "Overall coverage {:.1}% below minimum {:.0}%",
            overall_coverage * 100.0,
            min_coverage * 100.0
        ))
    };

    eprintln!(
        "[{}] Overall: {}/{} covered ({:.1}% coverage) - {}",
        test_name,
        total_covered,
        total_completed,
        overall_coverage * 100.0,
        decision
    );

    match decision {
        Decision::Pass => {
            eprintln!("[{}] PASSED", test_name);
        }
        Decision::Skip(reason) => {
            eprintln!("[{}] SKIPPED: {}", test_name, reason);
        }
        Decision::Fail(reason) => {
            panic!("[{}] FAILED: {}", test_name, reason);
        }
    }
}

// =============================================================================
// VALIDATION TIER TESTS (run weekly, ignored by default)
// =============================================================================

/// Rigorous coverage validation with 500 trials.
/// Uses 500ns effect (5×θ for AdjacentNetwork).
#[test]
#[ignore]
fn coverage_validation_rigorous() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[coverage_validation_rigorous] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    // Force validation tier
    std::env::set_var("CALIBRATION_TIER", "validation");

    let test_name = "coverage_validation_rigorous";
    let config = CalibrationConfig::from_env(test_name);
    let mut rng = config.rng();

    let injected_ns: u64 = 500;
    let trials = config.tier.coverage_trials();
    let mut runner = CoverageRunner::new(config.clone(), trials, injected_ns as f64);

    eprintln!(
        "[{}] Starting {} trials with {}ns injected effect (tier: {})",
        test_name, trials, injected_ns, config.tier
    );

    for trial in 0..trials {
        if runner.should_stop() {
            eprintln!("[{}] Early stop at trial {}", test_name, trial);
            break;
        }

        let delay = injected_ns;
        let inputs = InputPair::new(|| false, || true);

        let outcome = TimingOracle::for_attacker(select_attacker_model(test_name))
            .max_samples(config.samples_per_trial)
            .time_budget(config.time_budget_per_trial)
            .test(inputs, move |&is_sample| {
                // Base operation (~2μs) to ensure:
                // 1. Operation is measurable on coarse timers
                // 2. Adaptive batching uses K=1 (no multiplier)
                busy_wait_ns(2000);

                if is_sample {
                    busy_wait_ns(delay);
                }
            });

        runner.record(&outcome);

        if (trial + 1) % 50 == 0 || trial + 1 == trials {
            eprintln!(
                "[{}] Trial {}/{}: {}/{} covered ({:.1}% coverage)",
                test_name,
                trial + 1,
                trials,
                runner.covered_count(),
                runner.completed(),
                runner.coverage() * 100.0
            );
        }

        rng = StdRng::seed_from_u64(rng.random());
    }

    let decision = runner.finalize();

    eprintln!(
        "[{}] Final: {}/{} covered ({:.1}% coverage) - {}",
        test_name,
        runner.covered_count(),
        runner.completed(),
        runner.coverage() * 100.0,
        decision
    );

    match decision {
        Decision::Pass => {
            eprintln!("[{}] PASSED", test_name);
        }
        Decision::Skip(reason) => {
            eprintln!("[{}] SKIPPED: {}", test_name, reason);
        }
        Decision::Fail(reason) => {
            panic!("[{}] FAILED: {}", test_name, reason);
        }
    }
}

/// Comprehensive coverage validation across all effect sizes.
#[test]
#[ignore]
fn coverage_validation_per_effect() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[coverage_validation_per_effect] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    // Force validation tier
    std::env::set_var("CALIBRATION_TIER", "validation");

    run_coverage_test("coverage_validation_per_effect", &COVERAGE_EFFECTS_NS);
}

// =============================================================================
// TEST HELPERS
// =============================================================================

/// Coverage statistics for a single effect size.
struct CoverageStats {
    effect_ns: u64,
    #[allow(dead_code)]
    trials: usize,
    covered: usize,
    not_covered: usize,
    unmeasurable: usize,
}

impl CoverageStats {
    fn coverage(&self) -> f64 {
        let completed = self.covered + self.not_covered;
        if completed == 0 {
            0.0
        } else {
            self.covered as f64 / completed as f64
        }
    }
}

/// Run a coverage test across multiple effect sizes.
fn run_coverage_test(test_name: &str, effects: &[u64]) {
    if CalibrationConfig::is_disabled() {
        eprintln!("[{}] Skipped: CALIBRATION_DISABLED=1", test_name);
        return;
    }

    let config = CalibrationConfig::from_env(test_name);
    let trials_per_effect = config.tier.coverage_trials() / effects.len().max(1);

    eprintln!(
        "[{}] Starting coverage test ({} effect sizes, {} trials each, tier: {})",
        test_name,
        effects.len(),
        trials_per_effect,
        config.tier
    );

    let mut stats: Vec<CoverageStats> = Vec::new();
    let mut any_failed = false;

    for &effect_ns in effects {
        let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(effect_ns));
        let mut runner = CoverageRunner::new(config.clone(), trials_per_effect, effect_ns as f64);

        eprintln!("\n[{}] Testing effect = {}ns", test_name, effect_ns);

        for trial in 0..trials_per_effect {
            if runner.should_stop() {
                eprintln!("[{}] Early stop at trial {}", test_name, trial);
                break;
            }

            let delay = effect_ns;
            let inputs = InputPair::new(|| false, || true);

            let outcome = TimingOracle::for_attacker(select_attacker_model(test_name))
                .max_samples(config.samples_per_trial)
                .time_budget(config.time_budget_per_trial)
                .test(inputs, move |&is_sample| {
                    // Base operation (~2μs) to ensure:
                    // 1. Operation is measurable on coarse timers
                    // 2. Adaptive batching uses K=1 (no multiplier)
                    busy_wait_ns(2000);

                    // Inject delay for sample class
                    if is_sample {
                        busy_wait_ns(delay);
                    }
                });

            runner.record(&outcome);

            // Progress logging
            if (trial + 1) % 20 == 0 || trial + 1 == trials_per_effect {
                eprintln!(
                    "  Trial {}/{}: {:.1}% coverage",
                    trial + 1,
                    trials_per_effect,
                    runner.coverage() * 100.0
                );
            }

            rng = StdRng::seed_from_u64(rng.random());
        }

        // Collect stats for this effect size
        let completed = runner.completed();
        let coverage = runner.coverage();
        let covered = runner.covered_count();
        let not_covered = completed - covered;

        stats.push(CoverageStats {
            effect_ns,
            trials: trials_per_effect,
            covered,
            not_covered,
            unmeasurable: trials_per_effect - completed,
        });

        eprintln!(
            "[{}] {}ns: {}/{} covered ({:.1}% coverage)",
            test_name, effect_ns, covered, completed, coverage * 100.0
        );

        // Check coverage for effects >= 2θ (200ns for AdjacentNetwork)
        // With PMU timers, systematic bias is minimal so we can test smaller effects
        if effect_ns >= 200 && coverage < config.tier.min_coverage() {
            eprintln!(
                "[{}] FAILED at {}ns: coverage {:.1}% below {:.0}%",
                test_name,
                effect_ns,
                coverage * 100.0,
                config.tier.min_coverage() * 100.0
            );
            any_failed = true;
        }
    }

    // Print summary table
    eprintln!("\n[{}] Coverage Summary:", test_name);
    eprintln!("  Effect | Coverage | Covered | Not Covered | Unmeasurable");
    eprintln!("  -------|----------|---------|-------------|-------------");
    for stat in &stats {
        let marker = if stat.effect_ns >= 200 && stat.coverage() < config.tier.min_coverage() {
            " !!!"
        } else {
            ""
        };
        eprintln!(
            "  {:>5}ns | {:>6.1}% | {:>7} | {:>11} | {:>12}{}",
            stat.effect_ns,
            stat.coverage() * 100.0,
            stat.covered,
            stat.not_covered,
            stat.unmeasurable,
            marker
        );
    }

    // Overall statistics
    let total_covered: usize = stats.iter().map(|s| s.covered).sum();
    let total_completed: usize = stats.iter().map(|s| s.covered + s.not_covered).sum();
    let overall_coverage = if total_completed > 0 {
        total_covered as f64 / total_completed as f64
    } else {
        0.0
    };

    eprintln!(
        "\n[{}] Overall: {}/{} covered ({:.1}% coverage)",
        test_name, total_covered, total_completed, overall_coverage * 100.0
    );

    // Check if we have enough data
    let min_trials = trials_per_effect * effects.len() / 2;
    if total_completed < min_trials {
        eprintln!(
            "[{}] SKIPPED: Insufficient data ({} completed, need {})",
            test_name, total_completed, min_trials
        );
        return;
    }

    if any_failed {
        panic!(
            "[{}] FAILED: One or more effect sizes did not meet coverage requirements",
            test_name
        );
    }

    eprintln!("\n[{}] PASSED", test_name);
}
