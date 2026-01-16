//! Calibration tests to verify statistical properties.
//!
//! These tests validate that the timing oracle's statistical machinery
//! is properly calibrated:
//!
//! - FPR (False Positive Rate) tests: Verify the oracle doesn't incorrectly detect leaks
//! - Bayesian calibration: Verify posterior doesn't concentrate falsely
//!
//! Run with: cargo nextest run --profile calibration
//! Expected runtime: ~5-7 minutes

use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};
use std::time::Duration;

const MAX_SAMPLES: usize = 5_000;

// =============================================================================
// CORE FPR CALIBRATION (DEFAULT PROFILE)
// =============================================================================

/// Verify false positive rate is bounded.
///
/// Run 100 trials on pure noise data (random vs random) and check that
/// the failure rate (Fail outcomes) is low.
/// This is a quick sanity check included in the default test suite.
#[test]
fn fpr_calibration() {
    const TRIALS: usize = 100;

    eprintln!("\n[fpr] Starting {} trials", TRIALS);

    let mut failures = 0;
    let mut completed_trials = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        // Use Research mode (theta=0) to test raw statistical calibration
        let outcome = TimingOracle::for_attacker(AttackerModel::Research)
            .max_samples(MAX_SAMPLES)
            .time_budget(Duration::from_secs(5))
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        match &outcome {
            Outcome::Pass { .. } => {
                completed_trials += 1;
            }
            Outcome::Fail { leak_probability, .. } => {
                completed_trials += 1;
                failures += 1;
                eprintln!(
                    "[fpr] Trial {}: FALSE POSITIVE (P={:.1}%)",
                    trial + 1, leak_probability * 100.0
                );
            }
            Outcome::Inconclusive { .. } => {
                completed_trials += 1;
                // Inconclusive is acceptable for null hypothesis data
            }
            Outcome::Unmeasurable { .. } => {
                // Skip unmeasurable trials
            }
        }

        if (trial + 1) % 25 == 0 && completed_trials > 0 {
            let rate = failures as f64 / completed_trials as f64;
            eprintln!(
                "[fpr] Trial {}/{}: {} failures (rate={:.1}%)",
                trial + 1, TRIALS, failures, rate * 100.0
            );
        }
    }

    // Skip if no trials completed (e.g., permission denied on kperf)
    if completed_trials == 0 {
        eprintln!("[fpr] Skipping: all trials were unmeasurable");
        return;
    }

    let failure_rate = failures as f64 / completed_trials as f64;
    let (ci_low, ci_high) = wilson_ci(failures, completed_trials);

    eprintln!(
        "[fpr] Complete: {} failures out of {} trials, rate={:.1}% [95% CI: {:.1}%-{:.1}%]",
        failures, completed_trials, failure_rate * 100.0, ci_low * 100.0, ci_high * 100.0
    );

    // With default thresholds (pass < 0.05, fail > 0.95), we expect very few
    // false positives on null data. Allow up to 10% failure rate to account
    // for statistical variability.
    assert!(
        failure_rate <= 0.10,
        "FPR {} exceeds 10% tolerance",
        failure_rate
    );
}

/// Verify Bayesian layer doesn't over-concentrate on high probabilities for null data.
#[test]
fn bayesian_calibration() {
    const TRIALS: usize = 100;

    eprintln!("\n[bayesian] Starting {} trials", TRIALS);

    let mut high_prob_count = 0;
    let mut completed_trials = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        let outcome = TimingOracle::for_attacker(AttackerModel::Research)
            .max_samples(MAX_SAMPLES)
            .time_budget(Duration::from_secs(5))
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        match &outcome {
            Outcome::Pass { leak_probability, .. }
            | Outcome::Fail { leak_probability, .. }
            | Outcome::Inconclusive { leak_probability, .. } => {
                completed_trials += 1;
                if *leak_probability > 0.9 {
                    high_prob_count += 1;
                }
            }
            Outcome::Unmeasurable { .. } => {
                // Skip unmeasurable trials
            }
        }

        if (trial + 1) % 25 == 0 && completed_trials > 0 {
            let rate = high_prob_count as f64 / completed_trials as f64;
            eprintln!(
                "[bayesian] Trial {}/{}: {} high-prob (rate={:.1}%)",
                trial + 1, TRIALS, high_prob_count, rate * 100.0
            );
        }
    }

    // Skip if no trials completed (e.g., permission denied on kperf)
    if completed_trials == 0 {
        eprintln!("[bayesian] Skipping: all trials were unmeasurable");
        return;
    }

    let high_prob_rate = high_prob_count as f64 / completed_trials as f64;
    eprintln!(
        "[bayesian] Complete: {} high-prob out of {} trials, rate={:.1}% (limit=10%)",
        high_prob_count, completed_trials, high_prob_rate * 100.0
    );

    assert!(
        high_prob_rate < 0.1,
        "Too many high probabilities on null data: {}",
        high_prob_rate
    );
}

fn rand_bytes() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

// =============================================================================
// FIXED-VS-FIXED FPR VALIDATION
// =============================================================================

/// Verify FPR with Fixed-vs-Fixed identical data (true null hypothesis).
///
/// This tests the "true null" scenario where both classes use exactly the
/// same data, ensuring that the oracle doesn't falsely detect differences
/// when there are none.
#[test]
fn fpr_fixed_vs_fixed() {
    const TRIALS: usize = 100;

    eprintln!(
        "\n[fpr_fixed_vs_fixed] Starting {} trials",
        TRIALS
    );

    let mut failures = 0;
    let mut completed_trials = 0;

    for trial in 0..TRIALS {
        // Both classes use IDENTICAL fixed data (all zeros)
        let inputs = InputPair::new(
            || [0u8; 32], // Fixed: all zeros
            || [0u8; 32], // Fixed: all zeros (identical!)
        );

        let outcome = TimingOracle::for_attacker(AttackerModel::Research)
            .max_samples(MAX_SAMPLES)
            .time_budget(Duration::from_secs(5))
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        match &outcome {
            Outcome::Pass { .. } => {
                completed_trials += 1;
            }
            Outcome::Fail { leak_probability, .. } => {
                completed_trials += 1;
                failures += 1;
                eprintln!(
                    "[fpr_fixed_vs_fixed] Trial {}: FALSE POSITIVE (P={:.1}%)",
                    trial + 1, leak_probability * 100.0
                );
            }
            Outcome::Inconclusive { .. } => {
                completed_trials += 1;
            }
            Outcome::Unmeasurable { .. } => {
                // Skip unmeasurable trials
            }
        }

        if (trial + 1) % 25 == 0 && completed_trials > 0 {
            let rate = failures as f64 / completed_trials as f64;
            eprintln!(
                "[fpr_fixed_vs_fixed] Trial {}/{}: {} failures (rate={:.1}%)",
                trial + 1, TRIALS, failures, rate * 100.0
            );
        }
    }

    if completed_trials == 0 {
        eprintln!("[fpr_fixed_vs_fixed] Skipping: all trials were unmeasurable");
        return;
    }

    let failure_rate = failures as f64 / completed_trials as f64;
    let (ci_low, ci_high) = wilson_ci(failures, completed_trials);

    eprintln!(
        "[fpr_fixed_vs_fixed] Complete: {} failures out of {} trials, rate={:.1}% [95% CI: {:.1}%-{:.1}%]",
        failures, completed_trials, failure_rate * 100.0, ci_low * 100.0, ci_high * 100.0
    );

    assert!(
        failure_rate <= 0.10,
        "Fixed-vs-Fixed FPR {:.1}% exceeds 10% tolerance",
        failure_rate * 100.0
    );

    eprintln!("[fpr_fixed_vs_fixed] PASSED: Fixed-vs-Fixed FPR is properly bounded");
}

// =============================================================================
// BAYESIAN PRIOR CALIBRATION
// =============================================================================

/// Test Bayesian inference with different prior values.
///
/// Verifies that the prior affects posterior without causing false positives.
/// With 50 trials per prior, we can detect false positive rates > 15%.
#[test]
fn bayesian_prior_sweep() {
    const TRIALS_PER_PRIOR: usize = 50;

    let priors = [0.5, 0.75, 0.9];

    eprintln!("\n[prior_sweep] Testing {} prior values, {} trials each",
              priors.len(), TRIALS_PER_PRIOR);

    let mut any_completed = false;

    for &prior in &priors {
        let mut high_prob_count = 0;
        let mut completed_trials = 0;

        for trial in 0..TRIALS_PER_PRIOR {
            let inputs = InputPair::new(rand_bytes, rand_bytes);

            let outcome = TimingOracle::for_attacker(AttackerModel::Research)
                .max_samples(MAX_SAMPLES)
                .time_budget(Duration::from_secs(5))
                .prior_no_leak(prior)
                .test(inputs, |data| {
                    std::hint::black_box(data);
                });

            match &outcome {
                Outcome::Pass { leak_probability, .. }
                | Outcome::Fail { leak_probability, .. }
                | Outcome::Inconclusive { leak_probability, .. } => {
                    completed_trials += 1;
                    if *leak_probability > 0.9 {
                        high_prob_count += 1;
                    }
                }
                Outcome::Unmeasurable { .. } => {
                    // Skip unmeasurable trials
                }
            }

            if (trial + 1) % 25 == 0 && trial < TRIALS_PER_PRIOR - 1 && completed_trials > 0 {
                let rate = high_prob_count as f64 / completed_trials as f64;
                eprintln!(
                    "  prior={:.2}: {}/{} trials, {} high-prob ({:.0}%)",
                    prior, trial + 1, TRIALS_PER_PRIOR, high_prob_count, rate * 100.0
                );
            }
        }

        // Skip this prior if no trials completed
        if completed_trials == 0 {
            eprintln!("[prior_sweep] prior={:.2}: skipped (all trials unmeasurable)", prior);
            continue;
        }

        any_completed = true;
        let high_prob_rate = high_prob_count as f64 / completed_trials as f64;
        let (ci_low, ci_high) = wilson_ci(high_prob_count, completed_trials);

        eprintln!(
            "[prior_sweep] prior={:.2}: {} high-prob out of {} trials ({:.1}%) [95% CI: {:.1}%-{:.1}%]",
            prior, high_prob_count, completed_trials, high_prob_rate * 100.0, ci_low * 100.0, ci_high * 100.0
        );

        // Even with different priors, null data shouldn't produce many high posteriors
        assert!(
            high_prob_rate < 0.15,
            "Prior {} produced too many false high probabilities: {:.1}%",
            prior, high_prob_rate * 100.0
        );
    }

    if !any_completed {
        eprintln!("[prior_sweep] Skipping: all trials were unmeasurable");
        return;
    }

    eprintln!("[prior_sweep] PASSED: All priors properly calibrated");
}

// =============================================================================
// RIGOROUS 1,000-TRIAL FPR VALIDATION (IGNORED BY DEFAULT)
// =============================================================================

/// Rigorous FPR validation with 1,000 trials.
///
/// This test provides rigorous validation of the FPR guarantee but takes
/// several minutes to run. It is marked #[ignore] and should be run manually
/// for thorough validation:
///
/// ```bash
/// cargo test fpr_1k_trials -- --ignored
/// ```
#[test]
#[ignore]
fn fpr_1k_trials() {
    const TRIALS: usize = 1_000;

    eprintln!(
        "\n[fpr_1k] Starting {} trials",
        TRIALS
    );
    eprintln!("[fpr_1k] Expected ~50 failures or fewer (5% + margin)");

    let mut failures = 0;
    let mut completed_trials = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        // Use Research mode (theta=0) to measure true FPR
        let outcome = TimingOracle::for_attacker(AttackerModel::Research)
            .max_samples(MAX_SAMPLES)
            .time_budget(Duration::from_secs(5))
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        match &outcome {
            Outcome::Pass { .. } | Outcome::Inconclusive { .. } => {
                completed_trials += 1;
            }
            Outcome::Fail { .. } => {
                completed_trials += 1;
                failures += 1;
            }
            Outcome::Unmeasurable { .. } => {
                // Skip
            }
        }

        if (trial + 1) % 100 == 0 && completed_trials > 0 {
            let rate = failures as f64 / completed_trials as f64;
            eprintln!(
                "[fpr_1k] Trial {}/{}: {} failures ({:.2}%)",
                trial + 1,
                TRIALS,
                failures,
                rate * 100.0
            );
        }
    }

    if completed_trials == 0 {
        eprintln!("[fpr_1k] Skipping: all trials were unmeasurable");
        return;
    }

    let (ci_low, ci_high) = wilson_ci(failures, completed_trials);

    eprintln!(
        "\n[fpr_1k] Complete: {} failures out of {} trials",
        failures, completed_trials
    );
    eprintln!(
        "[fpr_1k] Rate: {:.2}% [95% CI: {:.2}%-{:.2}%]",
        failures as f64 / completed_trials as f64 * 100.0,
        ci_low * 100.0,
        ci_high * 100.0
    );

    // Allow 10% failure rate as tolerance
    let max_allowed_rate = 0.10;
    let failure_rate = failures as f64 / completed_trials as f64;
    assert!(
        failure_rate <= max_allowed_rate,
        "Failures {} ({:.2}%) exceeds {:.0}% tolerance",
        failures,
        failure_rate * 100.0,
        max_allowed_rate * 100.0
    );

    eprintln!("[fpr_1k] PASSED: Rigorous FPR validation complete");
}

// =============================================================================
// HELPERS
// =============================================================================

/// Wilson score confidence interval for binomial proportion.
///
/// More accurate than normal approximation, especially for small counts.
fn wilson_ci(successes: usize, trials: usize) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }

    let n = trials as f64;
    let p_hat = successes as f64 / n;

    if successes == 0 {
        let upper = 1.0 - (0.025_f64).powf(1.0 / n);
        return (0.0, upper);
    }

    if successes == trials {
        let lower = (0.025_f64).powf(1.0 / n);
        return (lower, 1.0);
    }

    let z = 1.96; // 95% CI
    let z2 = z * z;
    let denom = 1.0 + z2 / n;

    let center = (p_hat + z2 / (2.0 * n)) / denom;
    let margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n).sqrt() / denom;

    let lower = (center - margin).max(0.0);
    let upper = (center + margin).min(1.0);

    (lower, upper)
}
