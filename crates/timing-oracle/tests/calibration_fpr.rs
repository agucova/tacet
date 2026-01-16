//! False Positive Rate (FPR) calibration tests.
//!
//! These tests verify that the timing oracle's FPR is bounded under the null hypothesis
//! (no timing difference between classes).
//!
//! NOTE: Quick/Full tier tests use AdjacentNetwork (100ns threshold) because Research mode's
//! threshold (~42ns when clamped to timer resolution) is too close to system noise levels
//! on coarse timers. Validation tier tests with PMU timers can use Research mode.
//!
//! See docs/calibration-test-spec.md for the full specification.

mod calibration_utils;

use calibration_utils::{
    rand_bytes, select_attacker_model, CalibrationConfig, Decision, TrialRunner,
};
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, TimingOracle};

// =============================================================================
// QUICK TIER TESTS (run on every PR)
// =============================================================================

/// FPR test with random-vs-random data (true null hypothesis).
///
/// Both classes generate random data, so there should be no timing difference.
/// This is the most common FPR validation scenario.
#[test]
fn fpr_quick_random_vs_random() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[fpr_quick_random_vs_random] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    let test_name = "fpr_quick_random_vs_random";
    let config = CalibrationConfig::from_env(test_name);
    let mut rng = config.rng();

    let trials = config.tier.fpr_trials();
    let mut runner = TrialRunner::new(test_name, config.clone(), trials);

    eprintln!(
        "[{}] Starting {} trials (tier: {})",
        test_name, trials, config.tier
    );

    for trial in 0..trials {
        if runner.should_stop() {
            eprintln!("[{}] Early stop at trial {}", test_name, trial);
            break;
        }

        // Both classes use random data - true null hypothesis
        let mut rng_clone = rng.clone();
        let inputs = InputPair::new(move || rand_bytes(&mut rng_clone), {
            let mut rng2 = rng.clone();
            move || rand_bytes(&mut rng2)
        });

        // Select attacker model based on timer availability
        let attacker_model = select_attacker_model(test_name);
        let outcome = TimingOracle::for_attacker(attacker_model)
            .max_samples(config.samples_per_trial)
            .time_budget(config.time_budget_per_trial)
            .test(inputs, |data| {
                // Do a measurable operation that takes ~400ns
                // This is constant-time and identical for both classes
                let mut acc: u64 = 0;
                for _ in 0..100 {
                    for &b in data.iter() {
                        acc = acc.wrapping_add(std::hint::black_box(b) as u64);
                    }
                }
                std::hint::black_box(acc);
            });

        runner.record(&outcome);

        // Progress logging
        if (trial + 1) % 10 == 0 || trial + 1 == trials {
            eprintln!(
                "[{}] Trial {}/{}: {} failures ({:.1}% FPR)",
                test_name,
                trial + 1,
                trials,
                runner.fail_count(),
                runner.fpr() * 100.0
            );
        }

        // Advance RNG state for next trial
        rng = StdRng::seed_from_u64(rng.random());
    }

    let (decision, report) = runner.finalize_fpr();
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

/// FPR test with fixed-vs-fixed identical data.
///
/// Both classes use the exact same fixed data (all zeros).
/// This is an even stricter null hypothesis test.
#[test]
fn fpr_quick_fixed_vs_fixed() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[fpr_quick_fixed_vs_fixed] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    let test_name = "fpr_quick_fixed_vs_fixed";
    let config = CalibrationConfig::from_env(test_name);

    let trials = config.tier.fpr_trials();
    let mut runner = TrialRunner::new(test_name, config.clone(), trials);

    eprintln!(
        "[{}] Starting {} trials (tier: {})",
        test_name, trials, config.tier
    );

    for trial in 0..trials {
        if runner.should_stop() {
            eprintln!("[{}] Early stop at trial {}", test_name, trial);
            break;
        }

        // Both classes use identical fixed data
        let inputs = InputPair::new(|| [0u8; 32], || [0u8; 32]);

        // Select attacker model based on timer availability
        let attacker_model = select_attacker_model(test_name);
        let outcome = TimingOracle::for_attacker(attacker_model)
            .max_samples(config.samples_per_trial)
            .time_budget(config.time_budget_per_trial)
            .test(inputs, |data| {
                // Do a measurable operation that takes ~400ns
                // This is constant-time and identical for both classes
                let mut acc: u64 = 0;
                for _ in 0..100 {
                    for &b in data.iter() {
                        acc = acc.wrapping_add(std::hint::black_box(b) as u64);
                    }
                }
                std::hint::black_box(acc);
            });

        runner.record(&outcome);

        // Progress logging
        if (trial + 1) % 10 == 0 || trial + 1 == trials {
            eprintln!(
                "[{}] Trial {}/{}: {} failures ({:.1}% FPR)",
                test_name,
                trial + 1,
                trials,
                runner.fail_count(),
                runner.fpr() * 100.0
            );
        }
    }

    let (decision, report) = runner.finalize_fpr();
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

// =============================================================================
// VALIDATION TIER TESTS (run weekly, ignored by default)
// =============================================================================

/// Rigorous FPR validation with 500 trials.
///
/// This test provides high statistical confidence but takes longer to run.
/// Run with: cargo test fpr_validation_rigorous -- --ignored
#[test]
#[ignore]
fn fpr_validation_rigorous() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[fpr_validation_rigorous] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    // Force validation tier for this test
    std::env::set_var("CALIBRATION_TIER", "validation");

    let test_name = "fpr_validation_rigorous";
    let config = CalibrationConfig::from_env(test_name);
    let mut rng = config.rng();

    let trials = config.tier.fpr_trials();
    let mut runner = TrialRunner::new(test_name, config.clone(), trials);

    eprintln!(
        "[{}] Starting {} trials (tier: {})",
        test_name, trials, config.tier
    );

    for trial in 0..trials {
        if runner.should_stop() {
            eprintln!("[{}] Early stop at trial {}", test_name, trial);
            break;
        }

        let mut rng_clone = rng.clone();
        let inputs = InputPair::new(move || rand_bytes(&mut rng_clone), {
            let mut rng2 = rng.clone();
            move || rand_bytes(&mut rng2)
        });

        let outcome = TimingOracle::for_attacker(AttackerModel::Research)
            .max_samples(config.samples_per_trial)
            .time_budget(config.time_budget_per_trial)
            .test(inputs, |data| {
                // Do a measurable operation that takes ~1-2μs
                // This is constant-time and identical for both classes
                // (just iterating over bytes multiple times, no data-dependent timing)
                let mut acc: u64 = 0;
                for _ in 0..100 {
                    for &b in data.iter() {
                        acc = acc.wrapping_add(std::hint::black_box(b) as u64);
                    }
                }
                std::hint::black_box(acc);
            });

        runner.record(&outcome);

        // Progress logging every 50 trials
        if (trial + 1) % 50 == 0 || trial + 1 == trials {
            eprintln!(
                "[{}] Trial {}/{}: {} failures ({:.1}% FPR)",
                test_name,
                trial + 1,
                trials,
                runner.fail_count(),
                runner.fpr() * 100.0
            );
        }

        rng = StdRng::seed_from_u64(rng.random());
    }

    let (decision, report) = runner.finalize_fpr();
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

/// FPR test per AttackerModel to ensure thresholds don't affect null behavior.
#[test]
#[ignore]
fn fpr_validation_per_attacker_model() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[fpr_validation_per_attacker_model] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    let test_name = "fpr_validation_per_attacker_model";
    let config = CalibrationConfig::from_env(test_name);

    let attacker_models = [
        ("Research", AttackerModel::Research),
        ("PostQuantumSentinel", AttackerModel::PostQuantumSentinel),
        ("AdjacentNetwork", AttackerModel::AdjacentNetwork),
        ("RemoteNetwork", AttackerModel::RemoteNetwork),
    ];

    let trials_per_model = 50;

    for (model_name, attacker_model) in attacker_models {
        let mut rng = config.rng();
        let sub_test_name = format!("{}_{}", test_name, model_name);
        let mut runner = TrialRunner::new(&sub_test_name, config.clone(), trials_per_model);

        eprintln!(
            "[{}] Testing {} ({} trials)",
            test_name, model_name, trials_per_model
        );

        for trial in 0..trials_per_model {
            if runner.should_stop() {
                break;
            }

            let mut rng_clone = rng.clone();
            let inputs = InputPair::new(move || rand_bytes(&mut rng_clone), {
                let mut rng2 = rng.clone();
                move || rand_bytes(&mut rng2)
            });

            let outcome = TimingOracle::for_attacker(attacker_model)
                .max_samples(config.samples_per_trial)
                .time_budget(config.time_budget_per_trial)
                .test(inputs, |data| {
                    // Do a measurable operation - XOR all bytes to produce a single result
                    // This is constant-time and identical for both classes
                    let result: u8 = data.iter().fold(0u8, |acc, &b| acc ^ b);
                    std::hint::black_box(result);
                });

            runner.record(&outcome);
            rng = StdRng::seed_from_u64(rng.random());
        }

        let (decision, report) = runner.finalize_fpr();

        eprintln!(
            "[{}] {}: {} failures / {} completed ({:.1}% FPR) - {}",
            test_name,
            model_name,
            runner.fail_count(),
            runner.completed(),
            runner.fpr() * 100.0,
            decision
        );

        if decision.is_fail() {
            panic!("[{}] {} FAILED: {:?}", test_name, model_name, decision);
        }
    }

    eprintln!(
        "[{}] PASSED: All attacker models have bounded FPR",
        test_name
    );
}

// Need to import these for the rng operations
use rand::{rngs::StdRng, Rng, SeedableRng};
