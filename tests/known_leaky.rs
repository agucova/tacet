//! Tests that must detect known timing leaks.

use timing_oracle::{AttackerModel, Outcome, TimingOracle};
use timing_oracle::helpers::InputPair;

/// Test that early-exit comparison is detected as leaky.
///
/// Uses a larger array (512 bytes) to ensure the operation is measurable
/// with coarse timers (~41ns resolution on Apple Silicon).
#[test]
fn detects_early_exit_comparison() {
    let secret = [0u8; 512];

    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(100_000)
        .test(inputs, |data| {
            early_exit_compare(&secret, data);
        });

    // Skip if unmeasurable or inconclusive
    match &outcome {
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("[SKIPPED] detects_early_exit_comparison: {}", recommendation);
            return;
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("[SKIPPED] detects_early_exit_comparison: {:?}", reason);
            return;
        }
        _ => {}
    }

    // For known leaky code, we expect Fail
    let leak_prob = outcome.leak_probability().unwrap_or(0.0);
    assert!(
        leak_prob > 0.9,
        "Should detect leak with high probability, got {}",
        leak_prob
    );
    assert!(
        matches!(outcome, Outcome::Fail { .. }),
        "Should fail for leaky code, got {:?}",
        outcome
    );
}

/// Test that branch-based timing is detected.
#[test]
fn detects_branch_timing() {
    // Baseline: 0 (triggers expensive branch)
    // Sample: never zero (skips expensive branch)
    let inputs = InputPair::new(|| 0u8, || rand::random::<u8>() | 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(100_000)
        .test(inputs, |x| {
            branch_on_zero(*x);
        });

    // Skip if unmeasurable or inconclusive
    match &outcome {
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("[SKIPPED] detects_branch_timing: {}", recommendation);
            return;
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("[SKIPPED] detects_branch_timing: {:?}", reason);
            return;
        }
        _ => {}
    }

    let leak_prob = outcome.leak_probability().unwrap_or(0.0);
    assert!(
        leak_prob > 0.9,
        "Should detect branch timing leak, got {}",
        leak_prob
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
