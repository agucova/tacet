//! Power calibration tests.
//!
//! These tests verify that the timing oracle has adequate power to detect
//! timing differences at various effect sizes relative to the attacker model threshold θ.
//!
//! See docs/calibration-test-spec.md for the full specification.

use crate::calibration_utils;

use calibration_utils::{
    busy_wait_ns, init_effect_injection, CalibrationConfig, Decision, TimerBackend, TrialRunner,
};
use tacet::helpers::InputPair;
use tacet::{AttackerModel, TimingOracle};

// =============================================================================
// EFFECT SIZE DEFINITIONS
// =============================================================================

/// Effect sizes to test for each attacker model.
/// Format: (multiplier, effect_ns)
struct EffectSizes {
    model_name: &'static str,
    attacker_model: AttackerModel,
    theta_ns: f64,
    effects: [(f64, u64); 5], // (multiplier, ns) for 0.5×, 1×, 2×, 5×, 10×
}

const RESEARCH_EFFECTS: EffectSizes = EffectSizes {
    model_name: "Research",
    attacker_model: AttackerModel::Research,
    theta_ns: 50.0, // Nominal value for Research mode
    effects: [(0.5, 25), (1.0, 50), (2.0, 100), (5.0, 250), (10.0, 500)],
};

const ADJACENT_NETWORK_EFFECTS: EffectSizes = EffectSizes {
    model_name: "AdjacentNetwork",
    attacker_model: AttackerModel::AdjacentNetwork,
    theta_ns: 100.0,
    effects: [(0.5, 50), (1.0, 100), (2.0, 200), (5.0, 500), (10.0, 1000)],
};

const REMOTE_NETWORK_EFFECTS: EffectSizes = EffectSizes {
    model_name: "RemoteNetwork",
    attacker_model: AttackerModel::RemoteNetwork,
    theta_ns: 50_000.0, // 50μs
    effects: [
        (0.5, 25_000),
        (1.0, 50_000),
        (2.0, 100_000),
        (5.0, 250_000),
        (10.0, 500_000),
    ],
};

// PostQuantumSentinel has θ = 2.0ns (~10 cycles @ 5 GHz) which is below Instant precision,
// so we use larger multipliers (requires PMU timer)
const PQ_SENTINEL_EFFECTS: EffectSizes = EffectSizes {
    model_name: "PostQuantumSentinel",
    attacker_model: AttackerModel::PostQuantumSentinel,
    theta_ns: 2.0,
    effects: [
        (5.0, 10),     // ~5×θ ≈ 10ns (minimum measurable)
        (15.0, 30),    // ~15×θ
        (50.0, 100),   // ~50×θ
        (150.0, 300),  // ~150×θ
        (500.0, 1000), // ~500×θ (1μs)
    ],
};

// SharedHardware has θ = 0.4ns (~2 cycles @ 5 GHz) which requires PMU timer
const SHARED_HARDWARE_EFFECTS: EffectSizes = EffectSizes {
    model_name: "SharedHardware",
    attacker_model: AttackerModel::SharedHardware,
    theta_ns: 0.4,
    effects: [
        (15.0, 6),      // ~15×θ ≈ 6ns
        (75.0, 30),     // ~75×θ
        (150.0, 60),    // ~150×θ
        (750.0, 300),   // ~750×θ
        (1500.0, 600),  // ~1500×θ
    ],
};

// =============================================================================
// ITERATION TIER TESTS (quick feedback during development)
// =============================================================================

/// Quick iteration power test at 5×θ for AdjacentNetwork.
///
/// Runs fewer trials for faster iteration.
#[test]
fn power_iteration_adjacent_network() {
    run_power_test(
        "power_iteration_adjacent_network",
        &ADJACENT_NETWORK_EFFECTS,
        3, // Index of 5× effect
    );
}

// =============================================================================
// QUICK TIER TESTS (run on every PR)
// =============================================================================

/// Power test at 2×θ for AdjacentNetwork model.
///
/// This is the primary power validation test. At 2×θ, we expect ≥70% power.
#[test]
fn power_quick_2x_theta_adjacent_network() {
    if std::env::var("CALIBRATION_TIER").as_deref() == Ok("iteration") {
        eprintln!("[power_quick_2x_theta_adjacent_network] Skipped: iteration tier");
        return;
    }
    run_power_test(
        "power_quick_2x_theta_adjacent_network",
        &ADJACENT_NETWORK_EFFECTS,
        2, // Index of 2× effect in the effects array
    );
}

/// Power test at 5×θ for AdjacentNetwork model.
///
/// At 5×θ, we expect ≥90% power (≥95% for validation tier).
#[test]
fn power_quick_5x_theta_adjacent_network() {
    if std::env::var("CALIBRATION_TIER").as_deref() == Ok("iteration") {
        eprintln!("[power_quick_5x_theta_adjacent_network] Skipped: iteration tier");
        return;
    }
    run_power_test(
        "power_quick_5x_theta_adjacent_network",
        &ADJACENT_NETWORK_EFFECTS,
        3, // Index of 5× effect
    );
}

/// Power test at 10×θ for AdjacentNetwork model.
///
/// At 10×θ, we expect ≥95% power (≥99% for validation tier).
#[test]
fn power_quick_10x_theta_adjacent_network() {
    if std::env::var("CALIBRATION_TIER").as_deref() == Ok("iteration") {
        eprintln!("[power_quick_10x_theta_adjacent_network] Skipped: iteration tier");
        return;
    }
    run_power_test(
        "power_quick_10x_theta_adjacent_network",
        &ADJACENT_NETWORK_EFFECTS,
        4, // Index of 10× effect
    );
}

/// Power test at 2×θ for Research model (θ=50ns nominal).
#[test]
fn power_quick_2x_theta_research() {
    if std::env::var("CALIBRATION_TIER").as_deref() == Ok("iteration") {
        eprintln!("[power_quick_2x_theta_research] Skipped: iteration tier");
        return;
    }
    run_power_test("power_quick_2x_theta_research", &RESEARCH_EFFECTS, 2);
}

// =============================================================================
// VALIDATION TIER TESTS (run weekly, ignored by default)
// =============================================================================

/// Full power curve for AdjacentNetwork model.
///
/// Tests 0.5×, 1×, 2×, 5× effects and reports power at each level.
#[test]
#[ignore]
fn power_validation_curve_adjacent_network() {
    run_power_curve(
        "power_validation_curve_adjacent_network",
        &ADJACENT_NETWORK_EFFECTS,
    );
}

/// Full power curve for Research model.
#[test]
#[ignore]
fn power_validation_curve_research() {
    run_power_curve("power_validation_curve_research", &RESEARCH_EFFECTS);
}

/// Full power curve for RemoteNetwork model.
///
/// Uses larger delays (25-250μs) appropriate for WAN scenarios.
#[test]
#[ignore]
fn power_validation_curve_remote_network() {
    run_power_curve(
        "power_validation_curve_remote_network",
        &REMOTE_NETWORK_EFFECTS,
    );
}

/// Power curve for PostQuantumSentinel model.
///
/// Note: Uses higher multipliers because θ = 2.0ns is below Instant precision.
#[test]
#[ignore]
fn power_validation_curve_pq_sentinel() {
    run_power_curve("power_validation_curve_pq_sentinel", &PQ_SENTINEL_EFFECTS);
}

// =============================================================================
// PMU-SPECIFIC TESTS (require elevated privileges)
// =============================================================================

/// Power curve for SharedHardware model with PMU timer.
///
/// SharedHardware has θ = 0.4ns (~2 cycles @ 5 GHz) which is only measurable with PMU timers.
/// Skip if PMU is not available.
#[test]
#[ignore]
fn power_validation_curve_shared_hardware_pmu() {
    if !TimerBackend::cycle_accurate_available() {
        eprintln!("[power_validation_curve_shared_hardware_pmu] Skipped: PMU timer not available (run with sudo)");
        return;
    }

    std::env::set_var("CALIBRATION_TIER", "validation");
    run_power_curve(
        "power_validation_curve_shared_hardware_pmu",
        &SHARED_HARDWARE_EFFECTS,
    );
}

/// Power curve for PostQuantumSentinel with PMU timer.
///
/// More accurate results with PMU timer for small effects.
#[test]
#[ignore]
fn power_validation_curve_pq_sentinel_pmu() {
    if !TimerBackend::cycle_accurate_available() {
        eprintln!("[power_validation_curve_pq_sentinel_pmu] Skipped: PMU timer not available (run with sudo)");
        return;
    }

    std::env::set_var("CALIBRATION_TIER", "validation");
    run_power_curve(
        "power_validation_curve_pq_sentinel_pmu",
        &PQ_SENTINEL_EFFECTS,
    );
}

// =============================================================================
// TEST HELPERS
// =============================================================================

/// Run a single power test at a specific effect size.
fn run_power_test(test_name: &str, effect_sizes: &EffectSizes, effect_index: usize) {
    if CalibrationConfig::is_disabled() {
        eprintln!("[{}] Skipped: CALIBRATION_DISABLED=1", test_name);
        return;
    }

    // Initialize effect injection calibration before measurements
    init_effect_injection();

    let config = CalibrationConfig::from_env(test_name);
    let (multiplier, effect_ns) = effect_sizes.effects[effect_index];

    let trials = config.tier.power_trials();
    let mut runner = TrialRunner::new(test_name, config.clone(), trials);

    eprintln!(
        "[{}] Starting {} trials at {:.1}×θ ({:.0}ns) for {} (tier: {})",
        test_name, trials, multiplier, effect_ns, effect_sizes.model_name, config.tier
    );

    for trial in 0..trials {
        if runner.should_stop() {
            eprintln!("[{}] Early stop at trial {}", test_name, trial);
            break;
        }

        // Pass effect directly: 0 for baseline, effect_ns for sample
        let inputs = InputPair::new(|| 0u64, || effect_ns);

        let outcome = TimingOracle::for_attacker(effect_sizes.attacker_model)
            .max_samples(config.samples_per_trial)
            .time_budget(config.time_budget_per_trial)
            .test(inputs, |&effect| {
                busy_wait_ns(effect);
                std::hint::black_box(0);
            });

        runner.record(&outcome);

        // Debug: print first few outcomes
        if trial < 3 {
            eprintln!("[{}] Trial {} outcome: {:?}", test_name, trial, outcome);
        }

        // Progress logging
        if (trial + 1) % 10 == 0 || trial + 1 == trials {
            eprintln!(
                "[{}] Trial {}/{}: {} detections ({:.0}% power)",
                test_name,
                trial + 1,
                trials,
                runner.fail_count(),
                runner.power() * 100.0
            );
        }
    }

    let (decision, report) = runner.finalize_power(multiplier);
    report.print(&config);

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

/// Run a full power curve (all effect sizes) for an attacker model.
fn run_power_curve(test_name: &str, effect_sizes: &EffectSizes) {
    if CalibrationConfig::is_disabled() {
        eprintln!("[{}] Skipped: CALIBRATION_DISABLED=1", test_name);
        return;
    }

    // Initialize effect injection calibration before measurements
    init_effect_injection();

    // Force validation tier for curve tests
    std::env::set_var("CALIBRATION_TIER", "validation");

    let config = CalibrationConfig::from_env(test_name);
    let trials = config.tier.power_trials();

    eprintln!(
        "[{}] Power curve for {} (θ={:.1}ns, {} trials per effect)",
        test_name, effect_sizes.model_name, effect_sizes.theta_ns, trials
    );

    let mut results = Vec::new();
    let mut any_failed = false;

    for &(multiplier, effect_ns) in &effect_sizes.effects {
        let sub_test_name = format!("{}_{:.1}x", test_name, multiplier);
        let mut runner = TrialRunner::new(&sub_test_name, config.clone(), trials);

        eprintln!(
            "\n[{}] Testing {:.1}×θ ({:.0}ns)",
            test_name, multiplier, effect_ns
        );

        for trial in 0..trials {
            if runner.should_stop() {
                eprintln!("[{}] Early stop at trial {}", sub_test_name, trial);
                break;
            }

            // Pass effect directly: 0 for baseline, effect_ns for sample
            let inputs = InputPair::new(|| 0u64, || effect_ns);

            let outcome = TimingOracle::for_attacker(effect_sizes.attacker_model)
                .max_samples(config.samples_per_trial)
                .time_budget(config.time_budget_per_trial)
                .test(inputs, move |&effect| {
                    busy_wait_ns(effect);
                    std::hint::black_box(0);
                });

            runner.record(&outcome);

            // Progress logging every 20 trials
            if (trial + 1) % 20 == 0 || trial + 1 == trials {
                eprintln!(
                    "  Trial {}/{}: {:.0}% power",
                    trial + 1,
                    trials,
                    runner.power() * 100.0
                );
            }
        }

        let (decision, _report) = runner.finalize_power(multiplier);
        let power = runner.power();

        results.push((multiplier, power, decision.clone()));

        if decision.is_fail() {
            any_failed = true;
        }
    }

    // Print summary
    eprintln!(
        "\n[{}] Power Curve Summary for {}:",
        test_name, effect_sizes.model_name
    );
    eprintln!("  Effect | Power | Decision");
    eprintln!("  -------|-------|----------");
    for (multiplier, power, decision) in &results {
        let decision_str = match decision {
            Decision::Pass => "PASS",
            Decision::Skip(_) => "SKIP",
            Decision::Fail(_) => "FAIL",
        };
        eprintln!(
            "  {:.1}×θ  | {:.0}%  | {}",
            multiplier,
            power * 100.0,
            decision_str
        );
    }

    // Check monotonicity (power should generally increase with effect size)
    for i in 1..results.len() {
        let (mult_prev, power_prev, _) = results[i - 1];
        let (mult_curr, power_curr, _) = results[i];
        if power_curr < power_prev - 0.15 {
            eprintln!(
                "[WARN] Non-monotonic power: {:.1}×θ ({:.0}%) < {:.1}×θ ({:.0}%)",
                mult_curr,
                power_curr * 100.0,
                mult_prev,
                power_prev * 100.0
            );
        }
    }

    if any_failed {
        panic!(
            "[{}] FAILED: One or more effect sizes did not meet power requirements",
            test_name
        );
    }

    eprintln!("\n[{}] PASSED", test_name);
}
