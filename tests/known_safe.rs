//! Tests that must not false-positive on constant-time code.

use timing_oracle::TimingOracle;

/// Test that XOR-based comparison is not flagged.
#[test]
#[ignore = "requires full implementation"]
fn no_false_positive_xor_compare() {
    let secret = [0xABu8; 32];

    let result = TimingOracle::new()
        .samples(100_000)
        .test(
            || constant_time_compare(&secret, &[0xAB; 32]),
            || constant_time_compare(&secret, &rand_bytes()),
        );

    assert!(
        result.leak_probability < 0.5,
        "Should not detect leak in constant-time code, got {}",
        result.leak_probability
    );
    assert!(
        result.ci_gate.passed,
        "CI gate should pass for constant-time code"
    );
}

/// Test that simple XOR operation is not flagged.
#[test]
#[ignore = "requires full implementation"]
fn no_false_positive_xor() {
    let result = TimingOracle::new()
        .samples(100_000)
        .test(
            || xor_bytes(&[0u8; 32], &[0u8; 32]),
            || xor_bytes(&rand_bytes(), &rand_bytes()),
        );

    assert!(
        result.ci_gate.passed,
        "CI gate should pass for XOR operation"
    );
}

fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    let mut acc = 0u8;
    for i in 0..a.len().min(b.len()) {
        acc |= a[i] ^ b[i];
    }
    acc == 0 && a.len() == b.len()
}

fn xor_bytes(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for i in 0..32 {
        result[i] = a[i] ^ b[i];
    }
    result
}

fn rand_bytes() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
