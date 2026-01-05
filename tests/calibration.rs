//! Calibration tests to verify statistical properties.
//!
//! These tests validate that the timing oracle's statistical machinery
//! is properly calibrated:
//!
//! - FPR (False Positive Rate) tests: Verify CI gate rejects at ≤ 2×alpha
//! - Bayesian calibration: Verify posterior doesn't concentrate falsely
//!
//! Run with: cargo nextest run --profile calibration
//! Expected runtime: ~5-7 minutes

use timing_oracle::helpers::InputPair;
use timing_oracle::{Outcome, TimingOracle};

const SAMPLES: usize = 5_000;

// =============================================================================
// CORE FPR CALIBRATION (DEFAULT PROFILE)
// =============================================================================

/// Verify CI gate false positive rate is bounded at alpha=0.01.
///
/// Run 100 trials on pure noise data and check rejection rate <= 2*alpha.
/// This is a quick sanity check included in the default test suite.
#[test]
fn ci_gate_fpr_calibration() {
    const TRIALS: usize = 100;
    const ALPHA: f64 = 0.01;

    eprintln!("\n[ci_gate_fpr] Starting {} trials (alpha={})", TRIALS, ALPHA);

    let mut rejections = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        let outcome = TimingOracle::quick()
            .samples(SAMPLES)
            .alpha(ALPHA)
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        if let Outcome::Completed(result) = outcome {
            if !result.ci_gate.passed {
                rejections += 1;
            }

            if (trial + 1) % 25 == 0 {
                let rate = rejections as f64 / (trial + 1) as f64;
                eprintln!(
                    "[ci_gate_fpr] Trial {}/{}: {} rejections (rate={:.1}%)",
                    trial + 1, TRIALS, rejections, rate * 100.0
                );
            }
        }
    }

    let rejection_rate = rejections as f64 / TRIALS as f64;
    let (ci_low, ci_high) = wilson_ci(rejections, TRIALS);

    eprintln!(
        "[ci_gate_fpr] Complete: {} rejections, rate={:.1}% [95% CI: {:.1}%-{:.1}%] (limit={:.1}%)",
        rejections, rejection_rate * 100.0, ci_low * 100.0, ci_high * 100.0, 2.0 * ALPHA * 100.0
    );

    assert!(
        rejection_rate <= 2.0 * ALPHA,
        "FPR {} exceeds 2*alpha={}",
        rejection_rate, 2.0 * ALPHA
    );
}

/// Verify Bayesian layer doesn't over-concentrate on high probabilities for null data.
#[test]
fn bayesian_calibration() {
    const TRIALS: usize = 100;

    eprintln!("\n[bayesian] Starting {} trials", TRIALS);

    let mut high_prob_count = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        let outcome = TimingOracle::quick()
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        if let Outcome::Completed(result) = outcome {
            if result.leak_probability > 0.9 {
                high_prob_count += 1;
            }

            if (trial + 1) % 25 == 0 {
                let rate = high_prob_count as f64 / (trial + 1) as f64;
                eprintln!(
                    "[bayesian] Trial {}/{}: {} high-prob (rate={:.1}%)",
                    trial + 1, TRIALS, high_prob_count, rate * 100.0
                );
            }
        }
    }

    let high_prob_rate = high_prob_count as f64 / TRIALS as f64;
    eprintln!(
        "[bayesian] Complete: {} high-prob, rate={:.1}% (limit=10%)",
        high_prob_count, high_prob_rate * 100.0
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
// RIGOROUS FPR CALIBRATION (CALIBRATION PROFILE)
// =============================================================================

/// Rigorous FPR test at alpha=0.001 with 500 trials.
///
/// With 500 trials:
/// - Expected rejections: 0.5 (at nominal α)
/// - 95% CI width for 0% FPR: ±0.7%
/// - Can reliably detect FPR > 0.4% (2×α)
///
/// This test verifies the CI gate is properly calibrated at strict alpha levels.
#[test]
fn fpr_alpha_001() {
    const TRIALS: usize = 500;
    const ALPHA: f64 = 0.001;

    eprintln!("\n[fpr_alpha_001] Starting {} trials (alpha={})", TRIALS, ALPHA);

    let mut rejections = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        let outcome = TimingOracle::quick()
            .samples(SAMPLES)
            .alpha(ALPHA)
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        if let Outcome::Completed(result) = outcome {
            if !result.ci_gate.passed {
                rejections += 1;
            }
        }

        if (trial + 1) % 100 == 0 {
            let rate = rejections as f64 / (trial + 1) as f64;
            eprintln!(
                "[fpr_alpha_001] Trial {}/{}: {} rejections (rate={:.2}%)",
                trial + 1, TRIALS, rejections, rate * 100.0
            );
        }
    }

    let rejection_rate = rejections as f64 / TRIALS as f64;
    let (ci_low, ci_high) = wilson_ci(rejections, TRIALS);

    eprintln!(
        "[fpr_alpha_001] Complete: {} rejections, rate={:.2}% [95% CI: {:.2}%-{:.2}%] (limit={:.2}%)",
        rejections, rejection_rate * 100.0, ci_low * 100.0, ci_high * 100.0, 2.0 * ALPHA * 100.0
    );

    // Allow 2x alpha for statistical tolerance
    // With 500 trials, we have tight enough CI to detect violations
    assert!(
        rejection_rate <= 2.0 * ALPHA,
        "FPR {:.3}% exceeds 2*alpha={:.2}%",
        rejection_rate * 100.0, 2.0 * ALPHA * 100.0
    );

    eprintln!("[fpr_alpha_001] PASSED: FPR is properly bounded");
}

/// Rigorous FPR test at alpha=0.01 with 300 trials.
///
/// With 300 trials:
/// - Expected rejections: 3 (at nominal α)
/// - 95% CI width for 1% FPR: ±1.1%
/// - Can reliably detect FPR > 2% (2×α)
#[test]
fn fpr_alpha_01() {
    const TRIALS: usize = 300;
    const ALPHA: f64 = 0.01;

    eprintln!("\n[fpr_alpha_01] Starting {} trials (alpha={})", TRIALS, ALPHA);

    let mut rejections = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        let outcome = TimingOracle::quick()
            .samples(SAMPLES)
            .alpha(ALPHA)
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        if let Outcome::Completed(result) = outcome {
            if !result.ci_gate.passed {
                rejections += 1;
            }
        }

        if (trial + 1) % 75 == 0 {
            let rate = rejections as f64 / (trial + 1) as f64;
            eprintln!(
                "[fpr_alpha_01] Trial {}/{}: {} rejections (rate={:.2}%)",
                trial + 1, TRIALS, rejections, rate * 100.0
            );
        }
    }

    let rejection_rate = rejections as f64 / TRIALS as f64;
    let (ci_low, ci_high) = wilson_ci(rejections, TRIALS);

    eprintln!(
        "[fpr_alpha_01] Complete: {} rejections, rate={:.2}% [95% CI: {:.2}%-{:.2}%] (limit={:.2}%)",
        rejections, rejection_rate * 100.0, ci_low * 100.0, ci_high * 100.0, 2.0 * ALPHA * 100.0
    );

    assert!(
        rejection_rate <= 2.0 * ALPHA,
        "FPR {:.2}% exceeds 2*alpha={:.1}%",
        rejection_rate * 100.0, 2.0 * ALPHA * 100.0
    );

    eprintln!("[fpr_alpha_01] PASSED: FPR is properly bounded");
}

/// Rigorous FPR test at alpha=0.05 with 200 trials.
///
/// With 200 trials:
/// - Expected rejections: 10 (at nominal α)
/// - 95% CI width for 5% FPR: ±3%
/// - Can reliably detect FPR > 10% (2×α)
#[test]
fn fpr_alpha_005() {
    const TRIALS: usize = 200;
    const ALPHA: f64 = 0.05;

    eprintln!("\n[fpr_alpha_005] Starting {} trials (alpha={})", TRIALS, ALPHA);

    let mut rejections = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        let outcome = TimingOracle::quick()
            .samples(SAMPLES)
            .alpha(ALPHA)
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        if let Outcome::Completed(result) = outcome {
            if !result.ci_gate.passed {
                rejections += 1;
            }
        }

        if (trial + 1) % 50 == 0 {
            let rate = rejections as f64 / (trial + 1) as f64;
            eprintln!(
                "[fpr_alpha_005] Trial {}/{}: {} rejections (rate={:.1}%)",
                trial + 1, TRIALS, rejections, rate * 100.0
            );
        }
    }

    let rejection_rate = rejections as f64 / TRIALS as f64;
    let (ci_low, ci_high) = wilson_ci(rejections, TRIALS);

    eprintln!(
        "[fpr_alpha_005] Complete: {} rejections, rate={:.1}% [95% CI: {:.1}%-{:.1}%] (limit={:.1}%)",
        rejections, rejection_rate * 100.0, ci_low * 100.0, ci_high * 100.0, 2.0 * ALPHA * 100.0
    );

    assert!(
        rejection_rate <= 2.0 * ALPHA,
        "FPR {:.1}% exceeds 2*alpha={:.0}%",
        rejection_rate * 100.0, 2.0 * ALPHA * 100.0
    );

    eprintln!("[fpr_alpha_005] PASSED: FPR is properly bounded");
}

// =============================================================================
// BAYESIAN PRIOR CALIBRATION
// =============================================================================

/// Test Bayesian inference with different prior values.
///
/// Verifies that the prior affects posterior without causing false positives.
/// With 100 trials per prior, we can detect false positive rates > 10%.
#[test]
fn bayesian_prior_sweep() {
    const TRIALS_PER_PRIOR: usize = 100;

    let priors = [0.5, 0.75, 0.9, 0.95];

    eprintln!("\n[prior_sweep] Testing {} prior values, {} trials each",
              priors.len(), TRIALS_PER_PRIOR);

    for &prior in &priors {
        let mut high_prob_count = 0;

        for trial in 0..TRIALS_PER_PRIOR {
            let inputs = InputPair::new(rand_bytes, rand_bytes);

            let outcome = TimingOracle::quick()
                .samples(SAMPLES)
                .prior_no_leak(prior)
                .test(inputs, |data| {
                    std::hint::black_box(data);
                });

            if let Outcome::Completed(result) = outcome {
                if result.leak_probability > 0.9 {
                    high_prob_count += 1;
                }
            }

            if (trial + 1) % 25 == 0 && trial < TRIALS_PER_PRIOR - 1 {
                let rate = high_prob_count as f64 / (trial + 1) as f64;
                eprintln!(
                    "  prior={:.2}: {}/{} trials, {} high-prob ({:.0}%)",
                    prior, trial + 1, TRIALS_PER_PRIOR, high_prob_count, rate * 100.0
                );
            }
        }

        let high_prob_rate = high_prob_count as f64 / TRIALS_PER_PRIOR as f64;
        let (ci_low, ci_high) = wilson_ci(high_prob_count, TRIALS_PER_PRIOR);

        eprintln!(
            "[prior_sweep] prior={:.2}: {} high-prob ({:.1}%) [95% CI: {:.1}%-{:.1}%]",
            prior, high_prob_count, high_prob_rate * 100.0, ci_low * 100.0, ci_high * 100.0
        );

        // Even with different priors, null data shouldn't produce many high posteriors
        assert!(
            high_prob_rate < 0.15,
            "Prior {} produced too many false high probabilities: {:.1}%",
            prior, high_prob_rate * 100.0
        );
    }

    eprintln!("[prior_sweep] PASSED: All priors properly calibrated");
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
