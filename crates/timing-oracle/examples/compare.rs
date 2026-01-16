//! Example comparing constant-time vs non-constant-time implementations.
//!
//! This demonstrates the CORRECT way to test operations:
//! - Pre-generate all inputs before measurement using InputPair
//! - The test closure receives a reference to the input
//! - Only the input data differs between fixed and random classes

use std::time::Duration;
use timing_oracle::{
    helpers::InputPair, timing_test_checked, AttackerModel, Outcome, TimingOracle,
};

fn main() {
    println!("Comparing constant-time vs variable-time implementations\n");

    let secret = [0xABu8; 32];

    // Test 1: Variable-time comparison (should detect leak)
    println!("Testing variable-time comparison...");
    let outcome = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .time_budget(Duration::from_secs(30)),
        baseline: || [0xABu8; 32],  // Matches secret (early exit on first byte)
        sample: || rand_bytes(),   // Likely differs early
        measure: |data| {
            variable_time_compare(&secret, data);
        },
    };

    match outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            println!("  Result: PASS (unexpected!)");
            println!("  Leak probability: {:.1}%\n", leak_probability * 100.0);
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            println!("  Result: FAIL (expected)");
            println!("  Leak probability: {:.1}%", leak_probability * 100.0);
            println!("  Exploitability: {:?}\n", exploitability);
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            println!("  Result: INCONCLUSIVE");
            println!("  Leak probability: {:.1}%\n", leak_probability * 100.0);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            println!("  Could not measure: {}\n", recommendation);
        }
    }

    // Test 2: Constant-time comparison (should not detect leak)
    println!("Testing constant-time comparison...");
    let inputs = InputPair::new(|| [0xABu8; 32], rand_bytes);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            constant_time_compare(&secret, data);
        });

    match outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            println!("  Result: PASS (expected)");
            println!("  Leak probability: {:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            println!("  Result: FAIL (unexpected!)");
            println!("  Leak probability: {:.1}%", leak_probability * 100.0);
            println!("  Exploitability: {:?}", exploitability);
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            println!("  Result: INCONCLUSIVE");
            println!("  Leak probability: {:.1}%", leak_probability * 100.0);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            println!("  Could not measure: {}", recommendation);
        }
    }
}

/// Variable-time comparison with early exit (leaky).
fn variable_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if a[i] != b[i] {
            return false;
        }
    }
    true
}

/// Constant-time comparison using XOR accumulator.
fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc = 0u8;
    for i in 0..a.len() {
        acc |= a[i] ^ b[i];
    }
    acc == 0
}

fn rand_bytes() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
