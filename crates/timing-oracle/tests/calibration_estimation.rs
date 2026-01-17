//! Effect estimation accuracy tests.
//!
//! These tests verify that the oracle's effect estimates (shift_ns, tail_ns) accurately
//! reflect the true injected timing differences.
//!
//! Key metrics:
//! - **Bias**: Is the mean estimate close to the true value?
//! - **RMSE**: How much variance is there in the estimates?
//! - **Coverage**: Do 95% CIs contain the true value ~95% of the time?

mod calibration_utils;

use calibration_utils::{
    busy_wait_ns, compute_estimation_stats_by_effect, CalibrationConfig, EstimationPoint,
    TimerBackend, TrialRunner,
};
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};

// =============================================================================
// EFFECT SIZE DEFINITIONS
// =============================================================================

/// Effect sizes to test for estimation accuracy.
/// We use a range of effect sizes to verify accuracy across the spectrum.
const ESTIMATION_EFFECTS_NS: [u64; 5] = [
    50,     // Small effect (0.5× AdjacentNetwork θ)
    100,    // At threshold
    200,    // 2× threshold
    500,    // 5× threshold
    1000,   // 10× threshold (1μs)
];

// =============================================================================
// ITERATION TIER TESTS
// =============================================================================

/// Quick estimation accuracy check.
///
/// Uses 200ns and 500ns effects (2×θ and 5×θ for AdjacentNetwork).
/// With PMU timers, measurement overhead is minimal.
#[test]
fn estimation_accuracy_iteration() {
    run_estimation_test(
        "estimation_accuracy_iteration",
        AttackerModel::AdjacentNetwork,
        &ESTIMATION_EFFECTS_NS[2..4], // 200ns and 500ns for quick check
    );
}

// =============================================================================
// QUICK TIER TESTS
// =============================================================================

/// Estimation accuracy for AdjacentNetwork model.
#[test]
fn estimation_accuracy_quick_adjacent_network() {
    if std::env::var("CALIBRATION_TIER").as_deref() == Ok("iteration") {
        eprintln!("[estimation_accuracy_quick_adjacent_network] Skipped: iteration tier");
        return;
    }

    run_estimation_test(
        "estimation_accuracy_quick_adjacent_network",
        AttackerModel::AdjacentNetwork,
        &ESTIMATION_EFFECTS_NS[..4], // 50ns through 500ns
    );
}

// =============================================================================
// VALIDATION TIER TESTS
// =============================================================================

/// Comprehensive estimation accuracy for AdjacentNetwork.
#[test]
#[ignore]
fn estimation_accuracy_validation_adjacent_network() {
    std::env::set_var("CALIBRATION_TIER", "validation");
    run_estimation_test(
        "estimation_accuracy_validation_adjacent_network",
        AttackerModel::AdjacentNetwork,
        &ESTIMATION_EFFECTS_NS,
    );
}

/// Estimation accuracy with PMU timer for maximum precision.
#[test]
#[ignore]
fn estimation_accuracy_validation_pmu() {
    if !TimerBackend::pmu_available() {
        eprintln!("[estimation_accuracy_validation_pmu] Skipped: PMU timer not available");
        return;
    }

    std::env::set_var("CALIBRATION_TIER", "validation");
    run_estimation_test(
        "estimation_accuracy_validation_pmu",
        AttackerModel::AdjacentNetwork,
        &ESTIMATION_EFFECTS_NS,
    );
}

/// Estimation accuracy for RemoteNetwork (larger effects).
#[test]
#[ignore]
fn estimation_accuracy_validation_remote_network() {
    std::env::set_var("CALIBRATION_TIER", "validation");

    // Use larger effects appropriate for RemoteNetwork (θ = 50μs)
    let effects: [u64; 4] = [25_000, 50_000, 100_000, 250_000];
    run_estimation_test(
        "estimation_accuracy_validation_remote_network",
        AttackerModel::RemoteNetwork,
        &effects,
    );
}

// =============================================================================
// TEST RUNNER
// =============================================================================

fn run_estimation_test(
    test_name: &str,
    attacker_model: AttackerModel,
    effects: &[u64],
) {
    if CalibrationConfig::is_disabled() {
        eprintln!("[{}] Skipped: CALIBRATION_DISABLED=1", test_name);
        return;
    }

    let config = CalibrationConfig::from_env(test_name);
    let trials_per_effect = config.tier.estimation_trials_per_effect();

    eprintln!(
        "[{}] Starting estimation accuracy test (tier: {}, {} trials per effect)",
        test_name, config.tier, trials_per_effect
    );

    let mut all_points: Vec<EstimationPoint> = Vec::new();
    let mut any_failed = false;

    let model_name = format!("{:?}", attacker_model);
    for &effect_ns in effects {
        let sub_test_name = format!("{}_{}ns", test_name, effect_ns);
        let mut runner = TrialRunner::new(&sub_test_name, config.clone(), trials_per_effect)
            .with_export_info(effect_ns as f64, &model_name);

        eprintln!("\n[{}] Testing effect = {}ns", test_name, effect_ns);

        for trial in 0..trials_per_effect {
            if runner.should_stop() {
                eprintln!("[{}] Early stop at trial {}", sub_test_name, trial);
                break;
            }

            // Use simple boolean to distinguish baseline vs sample
            let inputs = InputPair::new(|| false, || true);
            let effect = effect_ns;

            let outcome = TimingOracle::for_attacker(attacker_model)
                .max_samples(config.samples_per_trial)
                .time_budget(config.time_budget_per_trial)
                .test(inputs, move |&should_delay| {
                    // Base operation (~2μs) to ensure:
                    // 1. Operation is measurable on coarse timers
                    // 2. Adaptive batching uses K=1 (no multiplier)
                    // Uses busy_wait for precise, constant-time delay.
                    busy_wait_ns(2000);

                    // Inject additional delay only for sample class
                    if should_delay {
                        busy_wait_ns(effect);
                    }
                });

            runner.record(&outcome);

            // Extract estimation data
            if let Some(point) = extract_estimation_point(&outcome, effect_ns as f64) {
                all_points.push(point);
            }

            // Progress logging
            if (trial + 1) % 20 == 0 || trial + 1 == trials_per_effect {
                eprintln!(
                    "  Trial {}/{}: {} points collected",
                    trial + 1,
                    trials_per_effect,
                    all_points.iter().filter(|p| (p.true_effect_ns - effect_ns as f64).abs() < 1.0).count()
                );
            }
        }
    }

    // Compute estimation statistics
    eprintln!("\n[{}] Computing estimation statistics...", test_name);

    let stats_by_effect = compute_estimation_stats_by_effect(&all_points);

    // Print results table
    eprintln!("\n[{}] Estimation Accuracy Summary:", test_name);
    eprintln!("  True Effect | Mean Est. | Bias     | Bias %   | RMSE     | Coverage | N");
    eprintln!("  ------------|-----------|----------|----------|----------|----------|----");

    for stats in &stats_by_effect {
        let bias_pct = if stats.true_effect_ns > 0.0 {
            format!("{:>7.1}%", stats.bias_fraction * 100.0)
        } else {
            "   N/A".to_string()
        };

        let bias_marker = if stats.bias_fraction.abs() > config.tier.max_estimation_bias() {
            " !!!"
        } else {
            ""
        };

        eprintln!(
            "  {:>10.0}ns | {:>7.1}ns | {:>7.1}ns | {} | {:>7.1}ns | {:>7.1}% | {:>3}{}",
            stats.true_effect_ns,
            stats.mean_estimate,
            stats.bias,
            bias_pct,
            stats.rmse,
            stats.coverage * 100.0,
            stats.count,
            bias_marker
        );

        // Check acceptance criteria for effects >= 2θ (200ns for AdjacentNetwork)
        // Note: We only check bias, not CI coverage. The oracle's CI is for the
        // detected effect magnitude, not accounting for systematic measurement bias.
        // CI coverage is tested separately in calibration_coverage.rs.
        if stats.true_effect_ns >= 200.0 {
            if stats.bias_fraction.abs() > config.tier.max_estimation_bias() {
                eprintln!(
                    "[{}] FAILED: Bias {:.1}% at {}ns exceeds {:.0}%",
                    test_name,
                    stats.bias_fraction * 100.0,
                    stats.true_effect_ns,
                    config.tier.max_estimation_bias() * 100.0
                );
                any_failed = true;
            }
        }
    }

    // Overall summary
    let total_points: usize = stats_by_effect.iter().map(|s| s.count).sum();
    let avg_bias: f64 = stats_by_effect
        .iter()
        .filter(|s| s.true_effect_ns > 0.0)
        .map(|s| s.bias_fraction.abs())
        .sum::<f64>()
        / stats_by_effect.iter().filter(|s| s.true_effect_ns > 0.0).count().max(1) as f64;
    let avg_coverage: f64 = stats_by_effect
        .iter()
        .map(|s| s.coverage)
        .sum::<f64>()
        / stats_by_effect.len().max(1) as f64;

    eprintln!("\n[{}] Overall:", test_name);
    eprintln!("  Total points: {}", total_points);
    eprintln!("  Average |bias|: {:.1}%", avg_bias * 100.0);
    eprintln!("  Average coverage: {:.1}%", avg_coverage * 100.0);

    // Skip if insufficient data
    if total_points < 20 {
        eprintln!(
            "[{}] SKIPPED: Insufficient data ({} points)",
            test_name, total_points
        );
        return;
    }

    if any_failed {
        panic!("[{}] FAILED: Estimation accuracy check failed", test_name);
    }

    eprintln!("\n[{}] PASSED: Estimation accuracy within acceptable bounds", test_name);
}

/// Extract estimation data from an Outcome.
///
/// NOTE: shift_ns can be negative (when sample is slower than baseline).
/// We use absolute value since we're testing magnitude estimation accuracy.
/// The CI is already expressed as magnitude (positive values).
fn extract_estimation_point(outcome: &Outcome, true_effect_ns: f64) -> Option<EstimationPoint> {
    match outcome {
        Outcome::Pass { effect, .. }
        | Outcome::Fail { effect, .. }
        | Outcome::Inconclusive { effect, .. } => {
            // Use absolute shift_ns as the primary estimate (uniform shift component)
            // shift_ns is negative when sample is slower, but we care about magnitude
            Some(EstimationPoint {
                true_effect_ns,
                estimated_effect_ns: effect.shift_ns.abs(),
                ci_low_ns: effect.credible_interval_ns.0,
                ci_high_ns: effect.credible_interval_ns.1,
            })
        }
        Outcome::Unmeasurable { .. } | Outcome::Research(_) => None,
    }
}
