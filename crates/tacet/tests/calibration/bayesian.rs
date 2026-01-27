//! Bayesian calibration tests.
//!
//! These tests verify that the oracle's stated probabilities match empirical frequencies.
//! When the oracle reports P(leak) = X%, approximately X% of those cases should be true positives.
//!
//! This is a key property of well-calibrated Bayesian inference: stated credences should
//! match long-run frequencies.

use crate::calibration_utils;

use calibration_utils::{
    busy_wait_ns, compute_calibration_bins, compute_calibration_error, init_effect_injection,
    max_calibration_deviation, CalibrationConfig, CalibrationPoint, TimerBackend, TrialRunner,
    MIN_CALIBRATION_BIN_SAMPLES,
};
use tacet::helpers::InputPair;
use tacet::{AttackerModel, Outcome, TimingOracle};

// =============================================================================
// EFFECT SIZE DEFINITIONS
// =============================================================================

/// Effect sizes for calibration testing.
/// We test at 0 (null), θ (boundary), and 2θ (above threshold) to get a range of probabilities.
struct CalibrationEffects {
    model_name: &'static str,
    attacker_model: AttackerModel,
    /// (multiplier, effect_ns, expected_is_true_positive)
    effects: [(f64, u64, bool); 3],
}

const ADJACENT_NETWORK_CALIBRATION: CalibrationEffects = CalibrationEffects {
    model_name: "AdjacentNetwork",
    attacker_model: AttackerModel::AdjacentNetwork,
    effects: [
        (0.0, 0, false),  // Null: expect low P, not true positive
        (1.0, 100, true), // At threshold (100ns): ~50% P, is true positive
        (2.0, 200, true), // Above threshold: high P, is true positive
    ],
};

const REMOTE_NETWORK_CALIBRATION: CalibrationEffects = CalibrationEffects {
    model_name: "RemoteNetwork",
    attacker_model: AttackerModel::RemoteNetwork,
    effects: [
        (0.0, 0, false),
        (1.0, 50_000, true),  // At threshold (50μs)
        (2.0, 100_000, true), // Above threshold
    ],
};

// =============================================================================
// ITERATION TIER TESTS (quick feedback, ~5-10 min)
// =============================================================================

/// Quick Bayesian calibration check for AdjacentNetwork.
///
/// Runs fewer trials for faster iteration during development.
#[test]
fn bayesian_calibration_iteration_adjacent_network() {
    run_calibration_test(
        "bayesian_calibration_iteration_adjacent_network",
        &ADJACENT_NETWORK_CALIBRATION,
        false, // Not PMU-required
    );
}

// =============================================================================
// QUICK TIER TESTS (run on every PR, ~5-10 min)
// =============================================================================

/// Bayesian calibration for AdjacentNetwork model.
///
/// This is the primary calibration validation test.
#[test]
fn bayesian_calibration_quick_adjacent_network() {
    // Force quick tier if iteration is set
    if std::env::var("CALIBRATION_TIER").as_deref() == Ok("iteration") {
        eprintln!("[bayesian_calibration_quick_adjacent_network] Skipped: iteration tier");
        return;
    }

    run_calibration_test(
        "bayesian_calibration_quick_adjacent_network",
        &ADJACENT_NETWORK_CALIBRATION,
        false,
    );
}

// =============================================================================
// VALIDATION TIER TESTS (thorough, ~30-60 min each)
// =============================================================================

/// Comprehensive Bayesian calibration for AdjacentNetwork.
#[test]
#[ignore]
fn bayesian_calibration_validation_adjacent_network() {
    std::env::set_var("CALIBRATION_TIER", "validation");
    run_calibration_test(
        "bayesian_calibration_validation_adjacent_network",
        &ADJACENT_NETWORK_CALIBRATION,
        false,
    );
}

/// Comprehensive Bayesian calibration for RemoteNetwork.
#[test]
#[ignore]
fn bayesian_calibration_validation_remote_network() {
    std::env::set_var("CALIBRATION_TIER", "validation");
    run_calibration_test(
        "bayesian_calibration_validation_remote_network",
        &REMOTE_NETWORK_CALIBRATION,
        false,
    );
}

/// Bayesian calibration with PMU timer for maximum precision.
#[test]
#[ignore]
fn bayesian_calibration_validation_pmu() {
    if !TimerBackend::cycle_accurate_available() {
        eprintln!("[bayesian_calibration_validation_pmu] Skipped: PMU timer not available");
        return;
    }

    std::env::set_var("CALIBRATION_TIER", "validation");
    run_calibration_test(
        "bayesian_calibration_validation_pmu",
        &ADJACENT_NETWORK_CALIBRATION,
        true, // PMU required
    );
}

// =============================================================================
// TEST RUNNER
// =============================================================================

fn run_calibration_test(test_name: &str, calibration: &CalibrationEffects, _pmu_required: bool) {
    init_effect_injection();

    if CalibrationConfig::is_disabled() {
        eprintln!("[{}] Skipped: CALIBRATION_DISABLED=1", test_name);
        return;
    }

    let config = CalibrationConfig::from_env(test_name);
    let trials_per_effect = config.tier.bayesian_trials_per_effect();

    eprintln!(
        "[{}] Starting Bayesian calibration for {} (tier: {}, {} trials per effect)",
        test_name, calibration.model_name, config.tier, trials_per_effect
    );

    let mut all_points: Vec<CalibrationPoint> = Vec::new();
    let mut any_failed = false;

    for &(multiplier, effect_ns, is_true_positive) in &calibration.effects {
        let sub_test_name = format!("{}_{:.1}x", test_name, multiplier);
        let mut runner = TrialRunner::new(&sub_test_name, config.clone(), trials_per_effect)
            .with_export_info(effect_ns as f64, calibration.model_name);

        eprintln!(
            "\n[{}] Testing {:.1}×θ ({:.0}ns) - expect is_true_positive={}",
            test_name, multiplier, effect_ns, is_true_positive
        );

        for trial in 0..trials_per_effect {
            if runner.should_stop() {
                eprintln!("[{}] Early stop at trial {}", sub_test_name, trial);
                break;
            }

            // Use simple boolean to distinguish baseline vs sample
            let inputs = InputPair::new(|| false, || true);
            let effect = effect_ns;

            let outcome = TimingOracle::for_attacker(calibration.attacker_model)
                .max_samples(config.samples_per_trial)
                .time_budget(config.time_budget_per_trial)
                .test(inputs, move |&is_sample| {
                    // Base operation (~2μs) to ensure:
                    // 1. Operation is measurable on coarse timers
                    // 2. Adaptive batching uses K=1 (no multiplier)
                    busy_wait_ns(2000);

                    // Inject additional delay only for sample class and effect > 0
                    if is_sample && effect > 0 {
                        busy_wait_ns(effect);
                    }
                });

            runner.record(&outcome);

            // Extract probability and record calibration point
            if let Some(leak_probability) = extract_leak_probability(&outcome) {
                all_points.push(CalibrationPoint {
                    stated_probability: leak_probability,
                    is_true_positive,
                    true_effect_ns: effect_ns as f64,
                });
            }

            // Progress logging
            if (trial + 1) % 20 == 0 || trial + 1 == trials_per_effect {
                eprintln!(
                    "  Trial {}/{}: {} points collected",
                    trial + 1,
                    trials_per_effect,
                    all_points.len()
                );
            }
        }

        // Check completion rate
        let completion_rate = runner.completed() as f64 / trials_per_effect as f64;
        if completion_rate < config.min_completed_rate {
            eprintln!(
                "[WARN] Low completion rate at {:.1}×θ: {:.0}%",
                multiplier,
                completion_rate * 100.0
            );
        }
    }

    // Compute calibration metrics
    eprintln!("\n[{}] Computing calibration metrics...", test_name);

    let bins = compute_calibration_bins(&all_points, 10);
    let calibration_error = compute_calibration_error(&bins);
    let max_deviation = max_calibration_deviation(&bins);

    // Print calibration curve
    eprintln!("\n[{}] Calibration Curve:", test_name);
    eprintln!(
        "  (bins with <{} samples are excluded from metrics)",
        MIN_CALIBRATION_BIN_SAMPLES
    );
    eprintln!("  Stated P | Empirical P | Count | Deviation");
    eprintln!("  ---------|-------------|-------|----------");
    for (stated, empirical, count) in &bins {
        let deviation = (stated - empirical).abs();
        let sparse_marker = if *count < MIN_CALIBRATION_BIN_SAMPLES {
            " (sparse, excluded)"
        } else {
            ""
        };
        let deviation_marker = if *count >= MIN_CALIBRATION_BIN_SAMPLES && deviation > 0.20 {
            " !!!"
        } else {
            ""
        };
        eprintln!(
            "  {:.0}%-{:.0}%   | {:.1}%        | {:>5} | {:.1}%{}{}",
            (stated - 0.05) * 100.0,
            (stated + 0.05) * 100.0,
            empirical * 100.0,
            count,
            deviation * 100.0,
            sparse_marker,
            deviation_marker
        );
    }

    eprintln!(
        "\n[{}] Mean calibration error: {:.1}% (max allowed: {:.0}%)",
        test_name,
        calibration_error * 100.0,
        config.tier.max_calibration_error() * 100.0
    );
    eprintln!(
        "[{}] Max deviation: {:.1}% (max allowed: 25%)",
        test_name,
        max_deviation * 100.0
    );

    // Check acceptance criteria
    if calibration_error > config.tier.max_calibration_error() {
        eprintln!(
            "[{}] FAILED: Calibration error {:.1}% exceeds {:.0}%",
            test_name,
            calibration_error * 100.0,
            config.tier.max_calibration_error() * 100.0
        );
        any_failed = true;
    }

    if max_deviation > 0.25 {
        eprintln!(
            "[{}] FAILED: Max deviation {:.1}% exceeds 25%",
            test_name,
            max_deviation * 100.0
        );
        any_failed = true;
    }

    // Skip if insufficient data
    if all_points.len() < 30 {
        eprintln!(
            "[{}] SKIPPED: Insufficient data ({} points)",
            test_name,
            all_points.len()
        );
        return;
    }

    if any_failed {
        panic!("[{}] FAILED: Bayesian calibration check failed", test_name);
    }

    eprintln!(
        "\n[{}] PASSED: Bayesian calibration within acceptable bounds",
        test_name
    );
}

/// Extract leak probability from an Outcome.
fn extract_leak_probability(outcome: &Outcome) -> Option<f64> {
    match outcome {
        Outcome::Pass {
            leak_probability, ..
        }
        | Outcome::Fail {
            leak_probability, ..
        }
        | Outcome::Inconclusive {
            leak_probability, ..
        } => Some(*leak_probability),
        Outcome::Unmeasurable { .. } | Outcome::Research(_) => None,
    }
}
