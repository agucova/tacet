//! Tests that must detect known timing leaks.

use timing_oracle::TimingOracle;

/// Test that early-exit comparison is detected as leaky.
#[test]
#[ignore = "requires full implementation"]
fn detects_early_exit_comparison() {
    let secret = [0u8; 32];

    let result = TimingOracle::new()
        .samples(100_000)
        .test(
            || early_exit_compare(&secret, &[0u8; 32]),
            || early_exit_compare(&secret, &rand_bytes()),
        );

    assert!(
        result.leak_probability > 0.9,
        "Should detect leak with high probability, got {}",
        result.leak_probability
    );
    assert!(
        !result.ci_gate.passed,
        "CI gate should fail for leaky code"
    );
}

/// Test that branch-based timing is detected.
#[test]
#[ignore = "requires full implementation"]
fn detects_branch_timing() {
    let result = TimingOracle::new()
        .samples(100_000)
        .test(
            || branch_on_zero(0),
            || branch_on_zero(rand::random::<u8>() | 1), // Never zero
        );

    assert!(
        result.leak_probability > 0.9,
        "Should detect branch timing leak"
    );
}

fn early_exit_compare(a: &[u8], b: &[u8]) -> bool {
    for i in 0..a.len().min(b.len()) {
        if a[i] != b[i] {
            return false;
        }
    }
    a.len() == b.len()
}

fn branch_on_zero(x: u8) -> u8 {
    if x == 0 {
        // Simulate expensive operation
        std::hint::black_box(0u8);
        for _ in 0..1000 {
            std::hint::black_box(0u8);
        }
        0
    } else {
        x
    }
}

fn rand_bytes() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
