//! Integration tests comparing Politis-White vs Geyer's IMS methods.
//!
//! These tests validate that both IACT estimation methods maintain statistical
//! properties (type-1 error control, power to detect leaks) and produce similar
//! outcomes on IID-like data.

use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{assert_leak_detected, AttackerModel, IactMethod, Outcome, TimingOracle};

// =============================================================================
// TEST 1: IID CONSISTENCY
// =============================================================================

/// Both methods should produce similar outcomes on IID-like data.
///
/// Uses a simple constant-time operation (array sum) with random data.
/// Both PolitisWhite and GeyersIMS should pass (or be inconclusive) since
/// there is no timing leak.
#[test]
fn test_iid_consistency() {
    // Test with Politis-White (current default)
    let inputs_pw = InputPair::new(|| [0u8; 32], rand::random::<[u8; 32]>);
    let outcome_politis_white = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .iact_method(IactMethod::PolitisWhite)
        .time_budget(Duration::from_secs(10))
        .test(inputs_pw, |data| {
            // Constant-time operation: array sum
            let _: u64 = data.iter().map(|&x| x as u64).sum();
        });

    // Test with Geyer's IMS
    let inputs_geyers = InputPair::new(|| [0u8; 32], rand::random::<[u8; 32]>);
    let outcome_geyers = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .iact_method(IactMethod::GeyersIMS)
        .time_budget(Duration::from_secs(10))
        .test(inputs_geyers, |data| {
            // Constant-time operation: array sum
            let _: u64 = data.iter().map(|&x| x as u64).sum();
        });

    // Print outcomes for debugging
    eprintln!("\n[test_iid_consistency] Politis-White:");
    eprintln!("{}", tacet::output::format_outcome(&outcome_politis_white));
    eprintln!("\n[test_iid_consistency] Geyer's IMS:");
    eprintln!("{}", tacet::output::format_outcome(&outcome_geyers));

    // Both should pass or be inconclusive on constant-time operation
    // Neither should fail (that would be a false positive)
    match &outcome_politis_white {
        Outcome::Fail { .. } => {
            panic!("Politis-White false positive on constant-time operation");
        }
        Outcome::Pass { .. } | Outcome::Inconclusive { .. } => {
            eprintln!("[test_iid_consistency] Politis-White: OK");
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!(
                "[SKIPPED] test_iid_consistency (Politis-White): {}",
                recommendation
            );
            return;
        }
        Outcome::Research(_) => {
            eprintln!("[SKIPPED] test_iid_consistency: Research mode not expected");
            return;
        }
    }

    match &outcome_geyers {
        Outcome::Fail { .. } => {
            panic!("Geyer's IMS false positive on constant-time operation");
        }
        Outcome::Pass { .. } | Outcome::Inconclusive { .. } => {
            eprintln!("[test_iid_consistency] Geyer's IMS: OK");
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!(
                "[SKIPPED] test_iid_consistency (Geyer's IMS): {}",
                recommendation
            );
        }
        Outcome::Research(_) => {
            eprintln!("[SKIPPED] test_iid_consistency: Research mode not expected");
        }
    }
}

// =============================================================================
// TEST 2: AUTOCORRELATION ROBUSTNESS WITH GEYER'S
// =============================================================================

/// Verify GeyersIMS maintains type-1 error control with autocorrelated data.
///
/// Uses a constant-time operation with autocorrelated timing (intentional
/// measurement artifacts). Geyer's IMS should handle this without false positives.
#[test]
fn test_autocorr_robustness_geyers() {
    let inputs = InputPair::new(|| [0u8; 32], rand::random::<[u8; 32]>);

    // Use Geyer's IMS which is designed to handle autocorrelation
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .iact_method(IactMethod::GeyersIMS)
        .time_budget(Duration::from_secs(15))
        .test(inputs, |data| {
            // Constant-time operation with intentional memory access patterns
            // that may introduce autocorrelation in timing measurements
            let mut acc: u64 = 0;
            for &byte in data.iter() {
                // Access pattern that may cause cache effects (but constant-time)
                acc = acc.wrapping_add(byte as u64);
                // Add some busywork to make the operation measurable
                for _ in 0..10 {
                    acc = acc.wrapping_add(1);
                }
            }
            std::hint::black_box(acc);
        });

    // Print outcome for debugging
    eprintln!("\n[test_autocorr_robustness_geyers]:");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    // Should not false positive (Fail) on constant-time operation
    // even with potential autocorrelation
    match &outcome {
        Outcome::Fail {
            leak_probability, ..
        } => {
            panic!(
                "Geyer's IMS false positive with autocorrelation: P={:.1}%",
                leak_probability * 100.0
            );
        }
        Outcome::Pass { .. } => {
            eprintln!("[test_autocorr_robustness_geyers] PASSED: No false positive");
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            eprintln!(
                "[test_autocorr_robustness_geyers] INCONCLUSIVE: P={:.1}% (acceptable)",
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!(
                "[SKIPPED] test_autocorr_robustness_geyers: {}",
                recommendation
            );
        }
        Outcome::Research(_) => {
            eprintln!("[SKIPPED] test_autocorr_robustness_geyers: Research mode not expected");
        }
    }
}

// =============================================================================
// TEST 3: GEYER'S DETECTS KNOWN LEAK
// =============================================================================

/// Verify GeyersIMS detects early-exit comparison leak.
///
/// This is a known leaky operation (early-exit on mismatch). Both methods
/// should detect the leak, but this specifically validates that Geyer's IMS
/// maintains detection power.
#[test]
fn test_geyers_detects_leak() {
    let secret = [0u8; 512];

    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);

    // Use Geyer's IMS
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .iact_method(IactMethod::GeyersIMS)
        .pass_threshold(0.01) // Harder to falsely pass
        .fail_threshold(0.85) // Quick to detect leak
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            std::hint::black_box(early_exit_compare(&secret, data));
        });

    // Print outcome for debugging
    eprintln!("\n[test_geyers_detects_leak]:");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    // Skip ONLY if unmeasurable
    if let Outcome::Unmeasurable { recommendation, .. } = &outcome {
        eprintln!("[SKIPPED] test_geyers_detects_leak: {}", recommendation);
        return;
    }

    // Should detect the known leak
    assert_leak_detected!(outcome);
}

// =============================================================================
// TEST 4: POLITIS-WHITE DETECTS KNOWN LEAK
// =============================================================================

/// Verify PolitisWhite detects early-exit comparison leak.
///
/// This serves as a baseline comparison to test_geyers_detects_leak,
/// ensuring both methods have similar detection power.
#[test]
fn test_politis_white_detects_leak() {
    let secret = [0u8; 512];

    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);

    // Use Politis-White (current default)
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .iact_method(IactMethod::PolitisWhite)
        .pass_threshold(0.01) // Harder to falsely pass
        .fail_threshold(0.85) // Quick to detect leak
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            std::hint::black_box(early_exit_compare(&secret, data));
        });

    // Print outcome for debugging
    eprintln!("\n[test_politis_white_detects_leak]:");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    // Skip ONLY if unmeasurable
    if let Outcome::Unmeasurable { recommendation, .. } = &outcome {
        eprintln!(
            "[SKIPPED] test_politis_white_detects_leak: {}",
            recommendation
        );
        return;
    }

    // Should detect the known leak
    assert_leak_detected!(outcome);
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Early-exit comparison (known leaky).
fn early_exit_compare(a: &[u8], b: &[u8]) -> bool {
    for i in 0..a.len().min(b.len()) {
        if a[i] != b[i] {
            return false; // Early exit leaks timing
        }
    }
    a.len() == b.len()
}

/// Generate random 512-byte array.
fn rand_bytes_512() -> [u8; 512] {
    let mut arr = [0u8; 512];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
