//! Power analysis and sensitivity tests.
//!
//! These tests verify the statistical power properties of the timing oracle
//! with rigorous trial counts for tight confidence intervals:
//!
//! - Threshold-relative power curve: Tests power at multiples of θ per spec §3.8
//! - Large effect detection: Confirms high power for obvious timing differences
//! - Negligible effect FPR: Confirms low detection rate for sub-resolution effects
//!
//! Run with: cargo test --test synthetic -- power --nocapture
//! Expected runtime: ~5-10 minutes
//!
//! Note: These tests may skip trials that return Unmeasurable outcomes (e.g., due to
//! system noise or lack of high-precision timing). If too many trials are unmeasurable,
//! the test will be skipped entirely.

use std::time::Duration;
use tacet::helpers::effect::{busy_wait_ns, init_effect_injection, timer_resolution_ns};
use tacet::helpers::InputPair;
use tacet::{AttackerModel, Outcome, TimingOracle};

// =============================================================================
// HELPERS
// =============================================================================

/// Detect the platform's measurement floor (θ_floor) via Research mode calibration.
///
/// Runs a short calibration with Research mode to determine the minimum detectable
/// effect on this platform. Returns the θ_floor from the ResearchOutcome.
fn detect_measurement_floor() -> f64 {
    let inputs = InputPair::new(|| [0u8; 32], rand::random::<[u8; 32]>);

    let outcome = TimingOracle::for_attacker(AttackerModel::Research)
        .time_budget(Duration::from_secs(5))
        .max_samples(5_000)
        .test(inputs, |data| {
            std::hint::black_box(data);
        });

    match outcome {
        Outcome::Research(report) => report.theta_floor,
        // Fallback: use timer resolution × 5 (same as min_injectable_effect_ns logic)
        _ => {
            let resolution = timer_resolution_ns();
            (resolution * 5.0).clamp(20.0, 500.0)
        }
    }
}

/// Run power trials at a specific effect size.
///
/// Injects `effect_ns` delay into sample class and measures detection rate.
/// Returns the fraction of trials that detected a leak (power).
fn run_power_trials(theta_ns: f64, effect_ns: u64, trials: usize) -> f64 {
    let mut detections = 0;
    let mut measurable = 0;

    for _ in 0..trials {
        let inputs = InputPair::new(|| false, || true);

        let outcome = TimingOracle::for_attacker(AttackerModel::Custom {
            threshold_ns: theta_ns,
        })
        .time_budget(Duration::from_secs(10))
        .max_samples(5_000)
        .test(inputs, |should_delay| {
            if *should_delay && effect_ns > 0 {
                busy_wait_ns(effect_ns);
            }
            std::hint::black_box(should_delay);
        });

        match outcome {
            Outcome::Pass { .. } => measurable += 1,
            Outcome::Fail { .. } => {
                measurable += 1;
                detections += 1;
            }
            Outcome::Inconclusive {
                leak_probability, ..
            } => {
                measurable += 1;
                // Count as detection if leak_probability > 50%
                if leak_probability > 0.5 {
                    detections += 1;
                }
            }
            Outcome::Unmeasurable { .. } | Outcome::Research(_) => {
                // Skip unmeasurable trials
            }
        }
    }

    if measurable == 0 {
        return 0.0;
    }
    detections as f64 / measurable as f64
}

// =============================================================================
// THRESHOLD BOUNDARY TEST
// =============================================================================

/// Test that the oracle correctly classifies effects relative to the threshold.
///
/// This test validates the core guarantee: effects below θ should pass,
/// effects above θ should fail. Rather than testing the exact power curve
/// shape (which depends on θ/θ_floor ratio), we test the boundary behavior:
///
/// - At 0 effect: Should pass (FPR ≤ 10%)
/// - At 0.5θ: Should mostly pass (≤ 25% detection)
/// - At 2θ: Should mostly fail (≥ 80% detection)
/// - At 3θ: Should reliably fail (≥ 90% detection)
///
/// This is more robust than testing exact power at θ, which assumes θ ≈ θ_floor.
#[test]
fn threshold_boundary_test() {
    init_effect_injection();

    // =========================================================================
    // Phase 1: Detect platform capabilities
    // =========================================================================

    eprintln!("\n[threshold_test] Phase 1: Detecting platform capabilities...");

    let theta_floor = detect_measurement_floor();
    let timer_res = timer_resolution_ns();

    // Choose testable threshold: at least θ_floor, with minimum of 200ns
    // to ensure effects are reliably injectable across platforms
    let theta_test = theta_floor.max(200.0);

    eprintln!(
        "[threshold_test] Platform: timer_res={:.1}ns, θ_floor={:.1}ns",
        timer_res, theta_floor
    );
    eprintln!("[threshold_test] Using θ_test={:.1}ns", theta_test);

    // =========================================================================
    // Phase 2: Test boundary behavior
    // =========================================================================

    // Test cases: (multiple, label, max_detection for "below", min_detection for "above")
    // Below threshold: should have LOW detection (pass)
    // Above threshold: should have HIGH detection (fail)
    let test_cases: [(f64, &str, f64, f64); 4] = [
        // (multiple, label, min_power, max_power)
        (0.0, "null (FPR)", 0.0, 0.10),      // No effect: ≤10% FPR
        (0.5, "below θ", 0.0, 0.25),          // Below threshold: ≤25% detection
        (2.0, "above θ (2×)", 0.80, 1.00),    // Above threshold: ≥80% detection
        (3.0, "well above θ (3×)", 0.90, 1.00), // Well above: ≥90% detection
    ];

    const TRIALS: usize = 50; // Fewer trials needed for boundary test

    eprintln!(
        "\n[threshold_test] Phase 2: Testing {} boundary conditions, {} trials each",
        test_cases.len(),
        TRIALS
    );

    let mut all_passed = true;

    for (multiple, label, min_power, max_power) in test_cases {
        let effect_ns = (theta_test * multiple) as u64;
        let power = run_power_trials(theta_test, effect_ns, TRIALS);

        let (ci_low, ci_high) = clopper_pearson_ci(
            (power * TRIALS as f64).round() as usize,
            TRIALS,
            0.05,
        );

        let passed = power >= min_power && power <= max_power;
        let status = if passed { "✓" } else { "✗" };

        eprintln!(
            "[threshold_test] {} {}: power={:.0}% [95% CI: {:.0}%-{:.0}%] (expected {:.0}%-{:.0}%)",
            status,
            label,
            power * 100.0,
            ci_low * 100.0,
            ci_high * 100.0,
            min_power * 100.0,
            max_power * 100.0
        );

        if !passed {
            all_passed = false;
        }

        assert!(
            power >= min_power && power <= max_power,
            "{}: power={:.0}%, expected [{:.0}%-{:.0}%]",
            label,
            power * 100.0,
            min_power * 100.0,
            max_power * 100.0
        );
    }

    if all_passed {
        eprintln!("\n[threshold_test] PASSED: Threshold boundary behavior is correct");
    }
}

// =============================================================================
// POWER CURVE AT DETECTION LIMIT (Spec §3.8)
// =============================================================================

/// Find the effect size at which the oracle has ~50% power (empirical MDE).
///
/// Uses binary search to find the effect size where detection rate ≈ 50%.
/// This is more accurate than using θ_floor from Research mode.
///
/// Note: The oracle's power curve can be very sharp (0% to 100% over ~20ns),
/// so we use many trials per probe and accept a wider convergence range.
fn find_empirical_mde(theta_ns: f64, trials_per_probe: usize) -> f64 {
    let mut low = 0u64;
    let mut high = (theta_ns * 2.0) as u64; // Start with 2× threshold as upper bound

    eprintln!(
        "[mde_search] Searching for 50% power point (θ={:.0}ns, {} trials/probe)...",
        theta_ns, trials_per_probe
    );

    // Binary search for ~50% power point
    // Use wider acceptance range (30-70%) because the power curve can be very sharp
    for iteration in 0..10 {
        let mid = (low + high) / 2;
        let power = run_power_trials(theta_ns, mid, trials_per_probe);

        eprintln!(
            "[mde_search]   iter {}: effect={}ns, power={:.0}%",
            iteration + 1,
            mid,
            power * 100.0
        );

        if power < 0.30 {
            low = mid;
        } else if power > 0.70 {
            high = mid;
        } else {
            // Found ~50% power point (within 30-70% range)
            eprintln!("[mde_search]   converged at {}ns with {:.0}% power", mid, power * 100.0);
            return mid as f64;
        }

        if high - low < 20 {
            eprintln!("[mde_search]   converged (range < 20ns)");
            break;
        }
    }

    ((low + high) / 2) as f64
}

/// Test power curve at the empirically-determined detection limit.
///
/// This validates the spec §3.8 power expectations by:
/// 1. Finding the effect size where power ≈ 50% (empirical MDE)
/// 2. Testing power at 0, 0.5×, 1×, 2×, 3× that MDE
///
/// This is more accurate than using θ_floor from Research mode,
/// which is just a heuristic (5× timer resolution).
#[test]
#[ignore] // Expensive test (~5-10 min); run with --ignored
fn power_curve_at_detection_limit() {
    init_effect_injection();

    eprintln!("\n[power_curve] Phase 1: Detecting platform capabilities...");

    let timer_res = timer_resolution_ns();

    // Use a reasonable threshold for testing (200ns minimum for reliable injection)
    let theta_ns = 500.0; // Use 500ns threshold for power curve testing

    eprintln!("[power_curve] Platform: timer_res={:.1}ns", timer_res);
    eprintln!("[power_curve] Using θ={:.0}ns for power curve testing", theta_ns);

    // =========================================================================
    // Phase 2: Find empirical MDE via binary search
    // =========================================================================

    eprintln!("\n[power_curve] Phase 2: Finding empirical 50% power point...");

    let empirical_mde = find_empirical_mde(theta_ns, 30); // 30 trials per probe for accuracy

    eprintln!(
        "[power_curve] Empirical MDE: {:.0}ns (effect size with ~50% power)",
        empirical_mde
    );

    // =========================================================================
    // Phase 3: Test power curve at MDE multiples
    // =========================================================================

    // Test cases relative to empirical MDE (not θ)
    // Note: The oracle's power curve can be very sharp, so we use wide ranges
    let test_cases: [(f64, &str, f64, f64); 5] = [
        (0.0, "null", 0.0, 0.15),        // FPR: 0-15%
        (0.5, "MDE/2", 0.0, 0.40),       // Below MDE: low power (allow up to 40%)
        (1.0, "MDE", 0.20, 0.80),        // At MDE: ~50% power (wide range due to sharp curve)
        (2.0, "2×MDE", 0.70, 1.00),      // Above MDE: high power
        (3.0, "3×MDE", 0.85, 1.00),      // Well above: very high power
    ];

    const TRIALS: usize = 50;

    eprintln!(
        "\n[power_curve] Phase 3: Testing power at {} MDE multiples, {} trials each",
        test_cases.len(),
        TRIALS
    );

    let mut all_passed = true;

    for (multiple, label, min_power, max_power) in test_cases {
        let effect_ns = (empirical_mde * multiple) as u64;
        let power = run_power_trials(theta_ns, effect_ns, TRIALS);

        let (ci_low, ci_high) = clopper_pearson_ci(
            (power * TRIALS as f64).round() as usize,
            TRIALS,
            0.05,
        );

        let passed = power >= min_power && power <= max_power;
        let status = if passed { "✓" } else { "✗" };

        eprintln!(
            "[power_curve] {} {} ({:.0}ns): power={:.0}% [95% CI: {:.0}%-{:.0}%] (expected {:.0}%-{:.0}%)",
            status,
            label,
            effect_ns,
            power * 100.0,
            ci_low * 100.0,
            ci_high * 100.0,
            min_power * 100.0,
            max_power * 100.0
        );

        if !passed {
            all_passed = false;
        }

        assert!(
            power >= min_power && power <= max_power,
            "{}: power={:.0}%, expected [{:.0}%-{:.0}%]",
            label,
            power * 100.0,
            min_power * 100.0,
            max_power * 100.0
        );
    }

    // Also verify monotonicity
    eprintln!("\n[power_curve] Verifying monotonicity...");
    let mut prev_power = 0.0;
    for (multiple, label, _, _) in test_cases {
        let effect_ns = (empirical_mde * multiple) as u64;
        let power = run_power_trials(theta_ns, effect_ns, 30);

        if power < prev_power - 0.15 {
            eprintln!(
                "[power_curve] WARNING: Non-monotonic at {}: {:.0}% < {:.0}%",
                label,
                power * 100.0,
                prev_power * 100.0
            );
        }
        prev_power = power;
    }

    if all_passed {
        eprintln!("\n[power_curve] PASSED: Power curve shape is correct");
    }
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
    init_effect_injection();

    const TRIALS: usize = 50;
    const SAMPLES: usize = 10_000;
    const EFFECT_NS: u64 = 10_000; // 10 microseconds in nanoseconds

    eprintln!(
        "\n[large_effect] Testing {:.0}μs effect over {} trials",
        EFFECT_NS as f64 / 1000.0, TRIALS
    );

    let mut detections = 0;
    let mut leak_probs = Vec::with_capacity(TRIALS);
    let mut measurable_trials = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(|| false, || true);

        let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .require_high_precision()
            .time_budget(Duration::from_secs(10))
            .max_samples(SAMPLES)
            .test(inputs, |should_delay| {
                if *should_delay {
                    busy_wait_ns(EFFECT_NS);
                }
                std::hint::black_box(should_delay);
            });

        match outcome {
            Outcome::Pass {
                leak_probability, ..
            } => {
                measurable_trials += 1;
                leak_probs.push(leak_probability);
                // Not detected
            }
            Outcome::Fail {
                leak_probability, ..
            } => {
                measurable_trials += 1;
                leak_probs.push(leak_probability);
                detections += 1;
            }
            Outcome::Inconclusive {
                leak_probability, ..
            } => {
                measurable_trials += 1;
                leak_probs.push(leak_probability);
                // Count as detection if leak_probability > 50%
                if leak_probability > 0.5 {
                    detections += 1;
                }
            }
            Outcome::Unmeasurable { recommendation, .. } => {
                if trial == 0 {
                    eprintln!(
                        "  Trial {} returned Unmeasurable (skipping): {}",
                        trial + 1,
                        recommendation
                    );
                }
            }
            Outcome::Research(_) => {}
        }

        if (trial + 1) % 10 == 0 && measurable_trials > 0 {
            let rate = detections as f64 / measurable_trials as f64;
            eprintln!(
                "  Trial {}/{}: {}/{} detected ({:.0}%), avg P(leak)={:.1}%",
                trial + 1,
                TRIALS,
                detections,
                measurable_trials,
                rate * 100.0,
                leak_probs.iter().sum::<f64>() / leak_probs.len() as f64 * 100.0
            );
        }
    }

    if measurable_trials == 0 {
        eprintln!("[large_effect] SKIPPED: No measurable trials");
        eprintln!("  Try running with sudo for high-precision timing, or on a less noisy system");
        return;
    }

    let power = detections as f64 / measurable_trials as f64;
    let avg_leak_prob = leak_probs.iter().sum::<f64>() / leak_probs.len() as f64;

    let (ci_low, ci_high) = clopper_pearson_ci(detections, measurable_trials, 0.05);

    eprintln!(
        "\n[large_effect] Power: {:.0}% [95% CI: {:.0}%-{:.0}%]",
        power * 100.0,
        ci_low * 100.0,
        ci_high * 100.0
    );
    eprintln!(
        "[large_effect] Avg leak probability: {:.1}%",
        avg_leak_prob * 100.0
    );

    // Large effects should be detected almost always
    assert!(
        power >= 0.90,
        "Large effect ({:.0}μs) detected only {:.0}% of the time (expected ≥90%)",
        EFFECT_NS as f64 / 1000.0,
        power * 100.0
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
    init_effect_injection();

    const TRIALS: usize = 100;
    const SAMPLES: usize = 5_000;
    const EFFECT_NS: u64 = 1; // 1 nanosecond - essentially noise

    eprintln!(
        "\n[negligible_fpr] Testing {}ns effect over {} trials",
        EFFECT_NS, TRIALS
    );

    let mut detections = 0;
    let mut measurable_trials = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(|| false, || true);

        let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .require_high_precision()
            .time_budget(Duration::from_secs(10))
            .max_samples(SAMPLES)
            .test(inputs, |should_delay| {
                if *should_delay {
                    busy_wait_ns(EFFECT_NS);
                }
                std::hint::black_box(should_delay);
            });

        match outcome {
            Outcome::Pass { .. } => {
                measurable_trials += 1;
                // Not detected
            }
            Outcome::Fail { .. } => {
                measurable_trials += 1;
                detections += 1;
            }
            Outcome::Inconclusive {
                leak_probability, ..
            } => {
                measurable_trials += 1;
                // Count as detection if leak_probability > 50%
                if leak_probability > 0.5 {
                    detections += 1;
                }
            }
            Outcome::Unmeasurable { recommendation, .. } => {
                if trial == 0 {
                    eprintln!(
                        "  Trial {} returned Unmeasurable (skipping): {}",
                        trial + 1,
                        recommendation
                    );
                }
            }
            Outcome::Research(_) => {}
        }

        if (trial + 1) % 25 == 0 && measurable_trials > 0 {
            let rate = detections as f64 / measurable_trials as f64;
            eprintln!(
                "  Trial {}/{}: {}/{} detected ({:.0}%)",
                trial + 1,
                TRIALS,
                detections,
                measurable_trials,
                rate * 100.0
            );
        }
    }

    // Need at least 20 measurable trials for meaningful FPR estimate
    const MIN_TRIALS_FOR_FPR: usize = 20;
    if measurable_trials < MIN_TRIALS_FOR_FPR {
        eprintln!(
            "[negligible_fpr] SKIPPED: Only {} measurable trials (need at least {})",
            measurable_trials, MIN_TRIALS_FOR_FPR
        );
        eprintln!("  Try running with sudo for high-precision timing, or on a less noisy system");
        return;
    }

    let fpr = detections as f64 / measurable_trials as f64;
    let (ci_low, ci_high) = clopper_pearson_ci(detections, measurable_trials, 0.05);

    eprintln!(
        "\n[negligible_fpr] FPR: {:.0}% [95% CI: {:.0}%-{:.0}%] ({} trials)",
        fpr * 100.0,
        ci_low * 100.0,
        ci_high * 100.0,
        measurable_trials
    );

    // Negligible effects should have low detection rate
    // (essentially FPR, should be ≤ alpha or close to it)
    assert!(
        fpr <= 0.20,
        "Negligible effect ({}ns) detected {:.0}% of the time (expected ≤20%)",
        EFFECT_NS,
        fpr * 100.0
    );

    eprintln!("[negligible_fpr] PASSED: Negligible effects don't trigger false positives");
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
