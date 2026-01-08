//! Tests that must not false-positive on constant-time code.
//!
//! These tests verify that the oracle does NOT detect timing leaks in
//! properly implemented constant-time code.
//!
//! NOTE: Simple XOR operations can leak timing information due to CPU
//! microarchitectural effects (store buffer optimizations, cache behavior).
//! XORing zeros vs random data shows ~40ns timing difference on most platforms.

use timing_oracle::helpers::InputPair;
use timing_oracle::{skip_if_unreliable, TimingOracle};

/// Test that XOR-based comparison is not flagged.
///
/// Both classes use identical code paths - only the data differs.
#[test]
fn no_false_positive_xor_compare() {
    let secret = [0xABu8; 32];

    // Pre-generate inputs using InputPair
    let inputs = InputPair::new(|| [0xABu8; 32], rand_bytes);

    // Use the new API: test(inputs, operation)
    let outcome = TimingOracle::new().samples(100_000).test(inputs, |data| {
        constant_time_compare(&secret, data);
    });

    let result = skip_if_unreliable!(outcome, "no_false_positive_xor_compare");

    assert!(
        result.leak_probability < 0.5,
        "Should not detect leak in constant-time code, got {}",
        result.leak_probability
    );
    assert!(
        result.ci_gate.passed(),
        "CI gate should pass for constant-time code"
    );
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
