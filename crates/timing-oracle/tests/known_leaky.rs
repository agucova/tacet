//! Tests that must detect known timing leaks.
//!
//! CI Configuration for leak detection tests:
//! - pass_threshold(0.01): Very hard to falsely pass (we expect leaks)
//! - fail_threshold(0.85): Quick to detect leaks
//! - time_budget(30s): Generous ceiling

use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{assert_leak_detected, AttackerModel, Outcome, TimingOracle};

/// Test that early-exit comparison is detected as leaky.
///
/// Uses a larger array (512 bytes) to ensure the operation is measurable
/// with coarse timers (~41ns resolution on Apple Silicon).
#[test]
fn detects_early_exit_comparison() {
    let secret = [0u8; 512];

    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            early_exit_compare(&secret, data);
        });

    // Print the outcome for debugging
    eprintln!("\n[detects_early_exit_comparison]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Skip ONLY if unmeasurable (operation too fast for this platform)
    // Inconclusive should NOT be skipped - if we can't detect a known leak, that's a test failure
    if let Outcome::Unmeasurable { recommendation, .. } = &outcome {
        eprintln!(
            "[SKIPPED] detects_early_exit_comparison: {}",
            recommendation
        );
        return;
    }

    // For known leaky code, we expect Fail - uses new macro with rich diagnostics on failure
    assert_leak_detected!(outcome);
}

/// Test that branch-based timing is detected.
#[test]
fn detects_branch_timing() {
    // Baseline: 0 (triggers expensive branch)
    // Sample: never zero (skips expensive branch)
    let inputs = InputPair::new(|| 0u8, || rand::random::<u8>() | 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |x| {
            branch_on_zero(*x);
        });

    // Print the outcome for debugging
    eprintln!("\n[detects_branch_timing]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Skip ONLY if unmeasurable (operation too fast for this platform)
    // Inconclusive should NOT be skipped - if we can't detect a known leak, that's a test failure
    if let Outcome::Unmeasurable { recommendation, .. } = &outcome {
        eprintln!("[SKIPPED] detects_branch_timing: {}", recommendation);
        return;
    }

    // For known leaky code, we expect Fail - uses new macro with rich diagnostics on failure
    assert_leak_detected!(outcome);
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
