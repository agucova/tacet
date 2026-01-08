//! Tests that must detect known timing leaks.

use timing_oracle::{skip_if_unreliable, timing_test_checked, TimingOracle};

/// Test that early-exit comparison is detected as leaky.
///
/// Uses a larger array (512 bytes) to ensure the operation is measurable
/// with coarse timers (~41ns resolution on Apple Silicon).
#[test]
fn detects_early_exit_comparison() {
    let secret = [0u8; 512];

    // Use the timing_test_checked! macro for explicit Outcome handling
    let outcome = timing_test_checked! {
        oracle: TimingOracle::new().samples(100_000),
        baseline: || [0u8; 512],
        sample: || rand_bytes_512(),
        measure: |data| {
            early_exit_compare(&secret, data);
        },
    };

    let result = skip_if_unreliable!(outcome, "detects_early_exit_comparison");

    assert!(
        result.leak_probability > 0.9,
        "Should detect leak with high probability, got {}",
        result.leak_probability
    );
    assert!(
        !result.ci_gate.passed(),
        "CI gate should fail for leaky code"
    );
}

/// Test that branch-based timing is detected.
#[test]
fn detects_branch_timing() {
    // Use the timing_test_checked! macro
    // Fixed: 0 (triggers expensive branch)
    // Random: never zero (skips expensive branch)
    let outcome = timing_test_checked! {
        oracle: TimingOracle::new().samples(100_000),
        baseline: || 0u8,
        sample: || rand::random::<u8>() | 1,
        measure: |x| {
            branch_on_zero(*x);
        },
    };

    let result = skip_if_unreliable!(outcome, "detects_branch_timing");

    assert!(
        result.leak_probability > 0.9,
        "Should detect branch timing leak, got {}",
        result.leak_probability
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

fn rand_bytes_512() -> [u8; 512] {
    let mut arr = [0u8; 512];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
