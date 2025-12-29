//! Example comparing constant-time vs non-constant-time implementations.

use timing_oracle::TimingOracle;

fn main() {
    println!("Comparing constant-time vs variable-time implementations\n");

    let secret = [0xABu8; 32];

    // Test 1: Variable-time comparison (should detect leak)
    println!("Testing variable-time comparison...");
    let result = TimingOracle::new()
        .samples(50_000)
        .test(
            || variable_time_compare(&secret, &[0xAB; 32]),
            || variable_time_compare(&secret, &rand_bytes()),
        );

    println!("  Leak probability: {:.1}%", result.leak_probability * 100.0);
    println!("  CI gate passed: {}\n", result.ci_gate.passed);

    // Test 2: Constant-time comparison (should not detect leak)
    println!("Testing constant-time comparison...");
    let result = TimingOracle::new()
        .samples(50_000)
        .test(
            || constant_time_compare(&secret, &[0xAB; 32]),
            || constant_time_compare(&secret, &rand_bytes()),
        );

    println!("  Leak probability: {:.1}%", result.leak_probability * 100.0);
    println!("  CI gate passed: {}", result.ci_gate.passed);
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
