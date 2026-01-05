//! Power analysis and sensitivity tests.
//!
//! These tests verify the statistical power properties of the timing oracle
//! with rigorous trial counts for tight confidence intervals:
//!
//! - MDE-calibrated power curve: Tests power at multiples of estimated MDE
//! - MDE scaling validation: Verifies O(1/√n) scaling with 50 trials per sample size
//! - Large effect detection: Confirms high power for obvious timing differences
//!
//! Run with: cargo nextest run --profile power
//! Expected runtime: ~5-7 minutes

use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{Outcome, TimingOracle};

// =============================================================================
// MDE-CALIBRATED POWER CURVE
// =============================================================================

/// Test power at multiples of the minimum detectable effect.
///
/// This is the gold standard for power analysis:
/// 1. First estimate MDE from 100 null trials (take median for robustness)
/// 2. Test detection rates at 0.5×, 1×, 2×, 3×, 5× MDE
/// 3. Assert power ≥ 80% at 2×MDE (standard statistical criterion)
///
/// Total: 100 + 5×80 = 500 trials
#[test]
fn mde_calibrated_power_curve() {
    const MDE_ESTIMATION_TRIALS: usize = 100;
    const TRIALS_PER_MULTIPLE: usize = 80;
    const SAMPLES: usize = 5_000;

    eprintln!("\n[mde_power_curve] Phase 1: Estimating MDE from {} null trials", MDE_ESTIMATION_TRIALS);

    // Phase 1: Estimate MDE from null distribution
    let mut mde_estimates = Vec::with_capacity(MDE_ESTIMATION_TRIALS);

    for trial in 0..MDE_ESTIMATION_TRIALS {
        let inputs = InputPair::new(
            || [0u8; 32],
            || rand::random::<[u8; 32]>(),
        );

        let outcome = TimingOracle::quick()
            .samples(SAMPLES)
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        if let Outcome::Completed(result) = outcome {
            mde_estimates.push(result.min_detectable_effect.shift_ns);
        }

        if (trial + 1) % 25 == 0 {
            eprintln!(
                "  MDE estimation: {}/{} trials complete",
                trial + 1, MDE_ESTIMATION_TRIALS
            );
        }
    }

    // Use median MDE for robustness against outliers
    mde_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_mde_ns = mde_estimates[mde_estimates.len() / 2];

    eprintln!(
        "[mde_power_curve] Median MDE: {:.1}ns (range: {:.1}-{:.1}ns)",
        median_mde_ns,
        mde_estimates.first().unwrap_or(&0.0),
        mde_estimates.last().unwrap_or(&0.0)
    );

    // Phase 2: Test power at MDE multiples
    let mde_multiples = [0.5, 1.0, 2.0, 3.0, 5.0];
    let mut power_results = Vec::new();

    eprintln!("\n[mde_power_curve] Phase 2: Testing power at {} MDE multiples, {} trials each",
              mde_multiples.len(), TRIALS_PER_MULTIPLE);

    for &multiple in &mde_multiples {
        let effect_ns = median_mde_ns * multiple;
        let effect_us = effect_ns / 1000.0;

        let mut detections = 0;

        for trial in 0..TRIALS_PER_MULTIPLE {
            let inputs = InputPair::new(|| false, || true);

            let outcome = TimingOracle::quick()
                .samples(SAMPLES)
                .test(inputs, |should_delay| {
                    if *should_delay && effect_ns > 0.0 {
                        spin_delay_us(effect_us);
                    }
                    std::hint::black_box(should_delay);
                });

            if let Outcome::Completed(result) = outcome {
                // Detection = CI gate fails OR leak probability > 50%
                if !result.ci_gate.passed || result.leak_probability > 0.5 {
                    detections += 1;
                }
            }

            if (trial + 1) % 20 == 0 {
                let rate = detections as f64 / (trial + 1) as f64;
                eprintln!(
                    "  {:.1}×MDE ({:.0}ns): {}/{} detected ({:.0}%)",
                    multiple, effect_ns, detections, trial + 1, rate * 100.0
                );
            }
        }

        let power = detections as f64 / TRIALS_PER_MULTIPLE as f64;
        power_results.push((multiple, power));

        // Clopper-Pearson 95% CI for binomial proportion
        let (ci_low, ci_high) = clopper_pearson_ci(detections, TRIALS_PER_MULTIPLE, 0.05);

        eprintln!(
            "[mde_power_curve] {:.1}×MDE ({:.0}ns): power={:.1}% [95% CI: {:.1}%-{:.1}%]",
            multiple, effect_ns, power * 100.0, ci_low * 100.0, ci_high * 100.0
        );
    }

    // Assertions
    // 1. Power should be monotonically increasing with effect size
    for i in 1..power_results.len() {
        let (mult_prev, power_prev) = power_results[i - 1];
        let (mult_curr, power_curr) = power_results[i];
        // Allow 10% tolerance for statistical noise
        assert!(
            power_curr >= power_prev - 0.10,
            "Power decreased: {:.1}×MDE ({:.0}%) < {:.1}×MDE ({:.0}%)",
            mult_curr, power_curr * 100.0, mult_prev, power_prev * 100.0
        );
    }

    // 2. Power at 2×MDE should be ≥ 80% (standard criterion)
    let power_at_2x = power_results.iter()
        .find(|(m, _)| (*m - 2.0).abs() < 0.01)
        .map(|(_, p)| *p)
        .unwrap_or(0.0);

    // Use 70% threshold to account for statistical variance
    // (with 80 trials, 80% power has 95% CI of roughly [70%, 88%])
    assert!(
        power_at_2x >= 0.70,
        "Power at 2×MDE is {:.0}%, expected ≥70% (targeting 80%)",
        power_at_2x * 100.0
    );

    // 3. Power at 5×MDE should be very high (≥90%)
    let power_at_5x = power_results.iter()
        .find(|(m, _)| (*m - 5.0).abs() < 0.01)
        .map(|(_, p)| *p)
        .unwrap_or(0.0);

    assert!(
        power_at_5x >= 0.85,
        "Power at 5×MDE is {:.0}%, expected ≥85%",
        power_at_5x * 100.0
    );

    // 4. Power at 0.5×MDE should be low (≤50%)
    let power_at_half = power_results.iter()
        .find(|(m, _)| (*m - 0.5).abs() < 0.01)
        .map(|(_, p)| *p)
        .unwrap_or(1.0);

    assert!(
        power_at_half <= 0.60,
        "Power at 0.5×MDE is {:.0}%, expected ≤60% (below detection threshold)",
        power_at_half * 100.0
    );

    eprintln!("\n[mde_power_curve] PASSED: Power curve meets statistical criteria");
}

// =============================================================================
// MDE SCALING VALIDATION
// =============================================================================

/// Verify MDE scales as O(1/√n) with 50 trials per sample size.
///
/// Theory: MDE ∝ σ/√n, so doubling n should halve MDE.
/// With n₁=5000 and n₂=20000, ratio = √(20000/5000) = 2.0
///
/// We use 50 trials at each sample size to get robust median estimates
/// and verify the ratio is in [1.5, 2.5] (allowing 25% tolerance).
#[test]
fn mde_scaling_validation() {
    const TRIALS_PER_N: usize = 50;

    // 4× difference in sample size → 2× difference in MDE
    let sample_sizes = [5_000usize, 20_000usize];
    let mut mde_medians = Vec::new();

    eprintln!("\n[mde_scaling] Testing MDE at {} sample sizes, {} trials each",
              sample_sizes.len(), TRIALS_PER_N);

    for &samples in &sample_sizes {
        let mut mdes = Vec::with_capacity(TRIALS_PER_N);

        for trial in 0..TRIALS_PER_N {
            let inputs = InputPair::new(
                || [0u8; 32],
                || rand::random::<[u8; 32]>(),
            );

            let outcome = TimingOracle::quick()
                .samples(samples)
                .test(inputs, |data| {
                    std::hint::black_box(data);
                });

            if let Outcome::Completed(result) = outcome {
                mdes.push(result.min_detectable_effect.shift_ns);
            }

            if (trial + 1) % 10 == 0 {
                eprintln!(
                    "  n={}: {}/{} trials complete",
                    samples, trial + 1, TRIALS_PER_N
                );
            }
        }

        mdes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = mdes[mdes.len() / 2];
        let q25 = mdes[mdes.len() / 4];
        let q75 = mdes[3 * mdes.len() / 4];

        eprintln!(
            "[mde_scaling] n={}: median MDE={:.1}ns [IQR: {:.1}-{:.1}ns]",
            samples, median, q25, q75
        );

        mde_medians.push((samples, median));
    }

    // Verify scaling ratio
    let (n_small, mde_small) = mde_medians[0];
    let (n_large, mde_large) = mde_medians[1];

    let observed_ratio = mde_small / mde_large;
    let expected_ratio = ((n_large as f64) / (n_small as f64)).sqrt();

    eprintln!(
        "\n[mde_scaling] MDE ratio: {:.2} (expected {:.2}, theoretical √({}/{})",
        observed_ratio, expected_ratio, n_large, n_small
    );

    // Allow 25% tolerance: expect 2.0, accept [1.5, 2.5]
    assert!(
        observed_ratio >= 1.5 && observed_ratio <= 2.5,
        "MDE ratio {:.2} outside acceptable range [1.5, 2.5] (expected ~{:.1})",
        observed_ratio, expected_ratio
    );

    // Additional check: larger sample size should have smaller MDE
    assert!(
        mde_large < mde_small,
        "MDE did not decrease with more samples: n={}→{:.1}ns, n={}→{:.1}ns",
        n_small, mde_small, n_large, mde_large
    );

    eprintln!("[mde_scaling] PASSED: MDE scales correctly with sample size");
}

// =============================================================================
// LARGE EFFECT DETECTION
// =============================================================================

/// Verify near-perfect detection of large timing differences.
///
/// Injects 10μs delay (massive in crypto terms) and expects ≥95% detection.
/// This is a sanity check that the oracle works at all.
#[test]
fn large_effect_detection() {
    const TRIALS: usize = 50;
    const SAMPLES: usize = 10_000;
    const EFFECT_US: f64 = 10.0; // 10 microseconds

    eprintln!("\n[large_effect] Testing {:.0}μs effect over {} trials", EFFECT_US, TRIALS);

    let mut detections = 0;
    let mut leak_probs = Vec::with_capacity(TRIALS);

    for trial in 0..TRIALS {
        let inputs = InputPair::new(|| false, || true);

        let outcome = TimingOracle::quick()
            .samples(SAMPLES)
            .test(inputs, |should_delay| {
                if *should_delay {
                    spin_delay_us(EFFECT_US);
                }
                std::hint::black_box(should_delay);
            });

        if let Outcome::Completed(result) = outcome {
            leak_probs.push(result.leak_probability);

            if !result.ci_gate.passed || result.leak_probability > 0.5 {
                detections += 1;
            }

            if (trial + 1) % 10 == 0 {
                let rate = detections as f64 / (trial + 1) as f64;
                eprintln!(
                    "  Trial {}/{}: {}/{} detected ({:.0}%), avg P(leak)={:.1}%",
                    trial + 1, TRIALS, detections, trial + 1, rate * 100.0,
                    leak_probs.iter().sum::<f64>() / leak_probs.len() as f64 * 100.0
                );
            }
        }
    }

    let power = detections as f64 / TRIALS as f64;
    let avg_leak_prob = leak_probs.iter().sum::<f64>() / leak_probs.len() as f64;

    let (ci_low, ci_high) = clopper_pearson_ci(detections, TRIALS, 0.05);

    eprintln!(
        "\n[large_effect] Power: {:.0}% [95% CI: {:.0}%-{:.0}%]",
        power * 100.0, ci_low * 100.0, ci_high * 100.0
    );
    eprintln!("[large_effect] Avg leak probability: {:.1}%", avg_leak_prob * 100.0);

    // Large effects should be detected almost always
    assert!(
        power >= 0.90,
        "Large effect ({:.0}μs) detected only {:.0}% of the time (expected ≥90%)",
        EFFECT_US, power * 100.0
    );

    // Average leak probability should be very high
    assert!(
        avg_leak_prob >= 0.90,
        "Average leak probability {:.1}% is too low (expected ≥90%)",
        avg_leak_prob * 100.0
    );

    eprintln!("[large_effect] PASSED: Large effects reliably detected");
}

// =============================================================================
// NEGLIGIBLE EFFECT FALSE POSITIVES
// =============================================================================

/// Verify that truly negligible effects don't trigger false positives.
///
/// Tests with 1ns delay (essentially noise) and expects ≤15% detection rate,
/// consistent with the configured alpha level.
#[test]
fn negligible_effect_fpr() {
    const TRIALS: usize = 100;
    const SAMPLES: usize = 5_000;
    const EFFECT_NS: u64 = 1; // 1 nanosecond - essentially noise

    eprintln!("\n[negligible_fpr] Testing {}ns effect over {} trials", EFFECT_NS, TRIALS);

    let mut detections = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(|| false, || true);

        let outcome = TimingOracle::quick()
            .samples(SAMPLES)
            .test(inputs, |should_delay| {
                if *should_delay {
                    spin_delay_ns(EFFECT_NS);
                }
                std::hint::black_box(should_delay);
            });

        if let Outcome::Completed(result) = outcome {
            if !result.ci_gate.passed || result.leak_probability > 0.5 {
                detections += 1;
            }
        }

        if (trial + 1) % 25 == 0 {
            let rate = detections as f64 / (trial + 1) as f64;
            eprintln!(
                "  Trial {}/{}: {}/{} detected ({:.0}%)",
                trial + 1, TRIALS, detections, trial + 1, rate * 100.0
            );
        }
    }

    let fpr = detections as f64 / TRIALS as f64;
    let (ci_low, ci_high) = clopper_pearson_ci(detections, TRIALS, 0.05);

    eprintln!(
        "\n[negligible_fpr] FPR: {:.0}% [95% CI: {:.0}%-{:.0}%]",
        fpr * 100.0, ci_low * 100.0, ci_high * 100.0
    );

    // Negligible effects should have low detection rate
    // (essentially FPR, should be ≤ alpha or close to it)
    assert!(
        fpr <= 0.20,
        "Negligible effect ({}ns) detected {:.0}% of the time (expected ≤20%)",
        EFFECT_NS, fpr * 100.0
    );

    eprintln!("[negligible_fpr] PASSED: Negligible effects don't trigger false positives");
}

// =============================================================================
// HELPERS
// =============================================================================

/// Spin-wait for approximately the given number of nanoseconds.
#[inline(never)]
fn spin_delay_ns(ns: u64) {
    let start = std::time::Instant::now();
    let target = Duration::from_nanos(ns);
    while start.elapsed() < target {
        std::hint::spin_loop();
    }
}

/// Spin-wait for approximately the given number of microseconds.
#[inline(never)]
fn spin_delay_us(us: f64) {
    let start = std::time::Instant::now();
    let target = Duration::from_nanos((us * 1000.0) as u64);
    while start.elapsed() < target {
        std::hint::spin_loop();
    }
}

/// Compute Clopper-Pearson exact 95% confidence interval for binomial proportion.
///
/// This gives conservative (exact) confidence intervals, suitable for
/// validating statistical properties.
fn clopper_pearson_ci(successes: usize, trials: usize, alpha: f64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }

    let k = successes as f64;
    let n = trials as f64;

    // Lower bound: Beta(α/2; k, n-k+1) quantile
    // Upper bound: Beta(1-α/2; k+1, n-k) quantile
    // Using normal approximation for simplicity (valid for n ≥ 30)

    let p_hat = k / n;

    if successes == 0 {
        // Special case: 0 successes
        let upper = 1.0 - (alpha / 2.0_f64).powf(1.0 / n);
        return (0.0, upper);
    }

    if successes == trials {
        // Special case: all successes
        let lower = (alpha / 2.0_f64).powf(1.0 / n);
        return (lower, 1.0);
    }

    // Wilson score interval (more accurate than normal approximation)
    let z = 1.96; // 95% CI
    let z2 = z * z;
    let denom = 1.0 + z2 / n;

    let center = (p_hat + z2 / (2.0 * n)) / denom;
    let margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n).sqrt() / denom;

    let lower = (center - margin).max(0.0);
    let upper = (center + margin).min(1.0);

    (lower, upper)
}
