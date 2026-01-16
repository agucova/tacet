//! Tests that must not false-positive on constant-time code.
//!
//! These tests verify that the oracle does NOT detect timing leaks in
//! properly implemented constant-time code.
//!
//! NOTE: Simple XOR operations can leak timing information due to CPU
//! microarchitectural effects (store buffer optimizations, cache behavior).
//! XORing zeros vs random data shows ~40ns timing difference on most platforms.
//!
//! CI Configuration for constant-time tests:
//! - pass_threshold(0.15): Quick to pass (we expect no leaks)
//! - fail_threshold(0.99): Very hard to falsely fail (low FPR for ~80 tests)
//! - time_budget(30s): Generous ceiling

use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{assert_no_timing_leak, AttackerModel, Outcome, TimingOracle};

/// Test that XOR-based comparison is not flagged.
///
/// Both classes use identical code paths - only the data differs.
#[test]
fn no_false_positive_xor_compare() {
    let secret = [0xABu8; 32];

    // Pre-generate inputs using InputPair
    let inputs = InputPair::new(|| [0xABu8; 32], rand_bytes);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            constant_time_compare(&secret, data);
        });

    // Print the outcome for debugging
    eprintln!("\n[no_false_positive_xor_compare]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Skip if unmeasurable or inconclusive
    match &outcome {
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!(
                "[SKIPPED] no_false_positive_xor_compare: {}",
                recommendation
            );
            return;
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("[SKIPPED] no_false_positive_xor_compare: {}", reason);
            return;
        }
        _ => {}
    }

    // For constant-time code, we expect Pass (no leak) - uses new macro with rich diagnostics on failure
    assert_no_timing_leak!(outcome);
}

fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    let mut acc = 0u8;
    for i in 0..a.len().min(b.len()) {
        acc |= a[i] ^ b[i];
    }
    acc == 0 && a.len() == b.len()
}

fn rand_bytes() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

// ============================================================================
// Tests with default thresholds (0.05/0.95) for edge case coverage
// ============================================================================

/// Test XOR-based comparison with default thresholds (0.05/0.95).
///
/// This test verifies constant-time code at stricter default thresholds,
/// which may catch edge cases that the relaxed CI thresholds miss.
#[test]
fn no_false_positive_xor_compare_default_thresholds() {
    let secret = [0xABu8; 32];

    let inputs = InputPair::new(|| [0xABu8; 32], rand_bytes);

    // Use default thresholds (0.05 pass, 0.95 fail) - no custom thresholds
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            constant_time_compare(&secret, data);
        });

    eprintln!("\n[no_false_positive_xor_compare_default_thresholds]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    match &outcome {
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!(
                "[SKIPPED] no_false_positive_xor_compare_default_thresholds: {}",
                recommendation
            );
            return;
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!(
                "[SKIPPED] no_false_positive_xor_compare_default_thresholds: {}",
                reason
            );
            return;
        }
        _ => {}
    }

    assert_no_timing_leak!(outcome);
}

/// Test bitwise OR accumulator with default thresholds.
///
/// This tests a common constant-time pattern: accumulating differences
/// with bitwise OR, then checking if the result is zero.
#[test]
fn no_false_positive_bitwise_accumulator_default_thresholds() {
    let inputs = InputPair::new(|| [0u8; 32], rand_bytes);

    // Use default thresholds - no custom pass_threshold or fail_threshold
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            // Constant-time accumulator pattern
            let mut acc = 0u8;
            for &byte in data.iter() {
                acc |= byte;
            }
            std::hint::black_box(acc);
        });

    eprintln!("\n[no_false_positive_bitwise_accumulator_default_thresholds]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    match &outcome {
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!(
                "[SKIPPED] no_false_positive_bitwise_accumulator_default_thresholds: {}",
                recommendation
            );
            return;
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!(
                "[SKIPPED] no_false_positive_bitwise_accumulator_default_thresholds: {}",
                reason
            );
            return;
        }
        _ => {}
    }

    assert_no_timing_leak!(outcome);
}
