//! Fine-grained power curve generation.
//!
//! Generates publication-quality power curves showing detection rate
//! vs. effect size with Wilson confidence intervals.
//!
//! Key outputs:
//! - Detection rate at each effect size (0, 0.25θ, 0.5θ, ..., 10θ)
//! - 95% Wilson CI for each detection rate
//! - Median samples used and wall time
//!
//! These curves allow direct comparison with other tools' power characteristics.

mod calibration_utils;

use calibration_utils::{
    busy_wait_ns, export_power_curve_csv, wilson_ci, CalibrationConfig, PowerCurvePoint,
};
use std::time::Instant;
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};

// =============================================================================
// EFFECT SIZE CONFIGURATIONS
// =============================================================================

/// Fine-grained effect multipliers for publication-quality curves.
const FINE_MULTIPLIERS: [f64; 11] = [
    0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0,
];

/// Quick multipliers for iteration/PR checks.
const QUICK_MULTIPLIERS: [f64; 6] = [0.0, 0.5, 1.0, 1.5, 2.0, 5.0];

/// Iteration multipliers for fast feedback.
const ITERATION_MULTIPLIERS: [f64; 3] = [0.0, 1.0, 2.0];

// =============================================================================
// ITERATION TIER TESTS
// =============================================================================

/// Quick power curve check during development.
#[test]
fn power_curve_iteration() {
    run_power_curve(
        "power_curve_iteration",
        AttackerModel::AdjacentNetwork,
        100.0, // θ = 100ns
        &ITERATION_MULTIPLIERS,
    );
}

// =============================================================================
// QUICK TIER TESTS
// =============================================================================

/// Power curve for AdjacentNetwork model (PR checks).
#[test]
fn power_curve_quick_adjacent_network() {
    if std::env::var("CALIBRATION_TIER").as_deref() == Ok("iteration") {
        eprintln!("[power_curve_quick_adjacent_network] Skipped: iteration tier");
        return;
    }

    run_power_curve(
        "power_curve_quick_adjacent_network",
        AttackerModel::AdjacentNetwork,
        100.0,
        &QUICK_MULTIPLIERS,
    );
}

// =============================================================================
// VALIDATION TIER TESTS
// =============================================================================

/// Full power curve for AdjacentNetwork model.
///
/// Generates 11-point curve from 0 to 10×θ with Wilson CIs.
#[test]
#[ignore]
fn power_curve_validation_adjacent_network() {
    std::env::set_var("CALIBRATION_TIER", "validation");
    run_power_curve(
        "power_curve_validation_adjacent_network",
        AttackerModel::AdjacentNetwork,
        100.0,
        &FINE_MULTIPLIERS,
    );
}

/// Full power curve for RemoteNetwork model.
///
/// Uses larger effects appropriate for WAN scenarios (θ = 50μs).
#[test]
#[ignore]
fn power_curve_validation_remote_network() {
    std::env::set_var("CALIBRATION_TIER", "validation");
    run_power_curve(
        "power_curve_validation_remote_network",
        AttackerModel::RemoteNetwork,
        50_000.0, // θ = 50μs
        &FINE_MULTIPLIERS,
    );
}

/// Power curve for Research model (most sensitive).
#[test]
#[ignore]
fn power_curve_validation_research() {
    std::env::set_var("CALIBRATION_TIER", "validation");
    run_power_curve(
        "power_curve_validation_research",
        AttackerModel::Research,
        50.0, // Nominal θ for Research
        &FINE_MULTIPLIERS,
    );
}

// =============================================================================
// TEST RUNNER
// =============================================================================

fn run_power_curve(test_name: &str, model: AttackerModel, theta_ns: f64, multipliers: &[f64]) {
    if CalibrationConfig::is_disabled() {
        eprintln!("[{}] Skipped: CALIBRATION_DISABLED=1", test_name);
        return;
    }

    let config = CalibrationConfig::from_env(test_name);
    let trials = config.tier.power_trials();

    eprintln!(
        "[{}] Power curve for {:?} (θ={:.0}ns, {} trials/point, {} points)",
        test_name,
        model,
        theta_ns,
        trials,
        multipliers.len()
    );

    let mut points: Vec<PowerCurvePoint> = Vec::new();

    for &mult in multipliers {
        let effect_ns = (theta_ns * mult) as u64;

        eprintln!(
            "\n[{}] Testing {:.2}×θ = {}ns",
            test_name, mult, effect_ns
        );

        let mut detections = 0;
        let mut samples_used: Vec<usize> = Vec::new();
        let mut times_ms: Vec<u64> = Vec::new();

        for trial in 0..trials {
            let start = Instant::now();

            let inputs = InputPair::new(|| false, || true);
            let effect = effect_ns;

            let outcome = TimingOracle::for_attacker(model)
                .max_samples(config.samples_per_trial)
                .time_budget(config.time_budget_per_trial)
                .test(inputs, move |&should_delay| {
                    // Base operation (~2μs) for measurability
                    busy_wait_ns(2000);
                    if should_delay {
                        busy_wait_ns(effect);
                    }
                });

            let elapsed_ms = start.elapsed().as_millis() as u64;
            times_ms.push(elapsed_ms);

            match &outcome {
                Outcome::Fail { samples_used: n, .. } => {
                    detections += 1;
                    samples_used.push(*n);
                }
                // Count INCONCLUSIVE with high leak probability as detection
                // (oracle detected effect but needs more samples to be confident)
                Outcome::Inconclusive {
                    samples_used: n,
                    leak_probability,
                    ..
                } if *leak_probability >= 0.95 => {
                    detections += 1;
                    samples_used.push(*n);
                }
                Outcome::Pass { samples_used: n, .. }
                | Outcome::Inconclusive { samples_used: n, .. } => {
                    samples_used.push(*n);
                }
                Outcome::Unmeasurable { .. } | Outcome::Research(_) => {
                    // Don't count these toward samples
                }
            }

            // Progress every 20 trials
            if (trial + 1) % 20 == 0 || trial + 1 == trials {
                let rate = detections as f64 / (trial + 1) as f64;
                eprintln!(
                    "  Trial {}/{}: {:.0}% detection rate",
                    trial + 1,
                    trials,
                    rate * 100.0
                );
            }
        }

        // Compute statistics
        let rate = detections as f64 / trials as f64;
        let (ci_low, ci_high) = wilson_ci(detections, trials, 0.95);

        samples_used.sort();
        times_ms.sort();

        let median_samples = if samples_used.is_empty() {
            0
        } else {
            samples_used[samples_used.len() / 2]
        };

        let median_time_ms = if times_ms.is_empty() {
            0
        } else {
            times_ms[times_ms.len() / 2]
        };

        points.push(PowerCurvePoint {
            effect_mult: mult,
            effect_ns: effect_ns as f64,
            trials,
            detections,
            detection_rate: rate,
            ci_low,
            ci_high,
            median_samples,
            median_time_ms,
        });
    }

    // Export to CSV
    export_power_curve_csv(test_name, &points);

    // Print summary table
    eprintln!("\n[{}] Power Curve Summary:", test_name);
    eprintln!(
        "  {:>6} | {:>8} | {:>6} | {:>12} | {:>8} | {:>8}",
        "Effect", "Rate", "N", "95% CI", "Samples", "Time"
    );
    eprintln!(
        "  {:->6}-+-{:->8}-+-{:->6}-+-{:->12}-+-{:->8}-+-{:->8}",
        "", "", "", "", "", ""
    );

    for p in &points {
        eprintln!(
            "  {:>5.2}×θ | {:>6.1}% | {:>6} | [{:>4.1}%, {:>4.1}%] | {:>8} | {:>6}ms",
            p.effect_mult,
            p.detection_rate * 100.0,
            p.trials,
            p.ci_low * 100.0,
            p.ci_high * 100.0,
            p.median_samples,
            p.median_time_ms
        );
    }

    // Validation checks
    let mut any_failed = false;

    // Check FPR at 0 effect
    if let Some(p) = points.iter().find(|p| p.effect_mult == 0.0) {
        let max_fpr = config.tier.max_fpr();
        if p.detection_rate > max_fpr {
            eprintln!(
                "\n[{}] WARNING: FPR {:.1}% exceeds {:.0}% at 0×θ",
                test_name,
                p.detection_rate * 100.0,
                max_fpr * 100.0
            );
            any_failed = true;
        }
    }

    // Check power at 2×θ (should be ≥70%)
    if let Some(p) = points.iter().find(|p| (p.effect_mult - 2.0).abs() < 0.01) {
        let min_power = config.tier.min_power_2x_theta();
        if p.detection_rate < min_power {
            eprintln!(
                "\n[{}] WARNING: Power {:.1}% below {:.0}% at 2×θ",
                test_name,
                p.detection_rate * 100.0,
                min_power * 100.0
            );
            any_failed = true;
        }
    }

    // Check power at 5×θ (should be ≥90%)
    if let Some(p) = points.iter().find(|p| (p.effect_mult - 5.0).abs() < 0.01) {
        let min_power = config.tier.min_power_5x_theta();
        if p.detection_rate < min_power {
            eprintln!(
                "\n[{}] WARNING: Power {:.1}% below {:.0}% at 5×θ",
                test_name,
                p.detection_rate * 100.0,
                min_power * 100.0
            );
            any_failed = true;
        }
    }

    // Check monotonicity (power should generally increase with effect size)
    for i in 1..points.len() {
        let prev = &points[i - 1];
        let curr = &points[i];
        // Allow 15% non-monotonicity due to noise
        if curr.detection_rate < prev.detection_rate - 0.15 {
            eprintln!(
                "\n[{}] WARNING: Non-monotonic power: {:.2}×θ ({:.0}%) < {:.2}×θ ({:.0}%)",
                test_name,
                curr.effect_mult,
                curr.detection_rate * 100.0,
                prev.effect_mult,
                prev.detection_rate * 100.0
            );
        }
    }

    if any_failed {
        eprintln!("\n[{}] FAILED: Power curve checks failed", test_name);
        // Don't panic for quick/iteration tiers, only validation
        if config.tier == calibration_utils::Tier::Validation {
            panic!("[{}] FAILED: Power curve validation failed", test_name);
        }
    } else {
        eprintln!("\n[{}] PASSED", test_name);
    }
}
