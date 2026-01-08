//! Simple example demonstrating basic timing-oracle usage.
//!
//! This demonstrates the CORRECT way to test operations:
//! - Pre-generate all inputs before measurement
//! - The test closure executes identical code for fixed and random inputs
//! - Only the input data differs

use timing_oracle::{helpers::InputPair, timing_test_checked, Outcome, TimingOracle};

fn main() {
    println!("timing-oracle simple example\n");

    // Example: Testing a potentially leaky comparison
    let secret = [0u8; 32];

    // Pre-generate inputs using InputPair
    let inputs = InputPair::new(
        || [0u8; 32], // Baseline: all zeros (same as secret)
        || {
            let mut arr = [0u8; 32];
            for i in 0..32 {
                arr[i] = rand::random();
            }
            arr
        },
    );

    // Simple API with default config
    let outcome = TimingOracle::new().test(inputs, |data| {
        compare_bytes(&secret, data);
    });

    let result = match outcome {
        Outcome::Completed(r) => r,
        Outcome::Unmeasurable {
            recommendation, ..
        } => {
            println!("Could not measure: {}", recommendation);
            return;
        }
    };

    println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
    println!("CI gate passed: {}", result.ci_gate.passed());
    println!("Quality: {:?}", result.quality);
    println!(
        "Timer: {} ({:.1}ns resolution)",
        result.metadata.timer, result.metadata.timer_resolution_ns
    );
    println!(
        "Batching: enabled={}, K={}, ticks_per_batch={:.1}",
        result.metadata.batching.enabled,
        result.metadata.batching.k,
        result.metadata.batching.ticks_per_batch
    );
    println!("Rationale: {}", result.metadata.batching.rationale);
    println!(
        "MDE shift: {:.2}ns, tail: {:.2}ns",
        result.min_detectable_effect.shift_ns, result.min_detectable_effect.tail_ns
    );

    // Using timing_test_checked! macro with custom config
    let outcome = timing_test_checked! {
        oracle: TimingOracle::new().samples(10_000).alpha(0.01),
        baseline: || [0u8; 32],
        sample: || {
            let mut arr = [0u8; 32];
            for i in 0..32 {
                arr[i] = rand::random();
            }
            arr
        },
        measure: |data| {
            compare_bytes(&secret, data);
        },
    };

    if let Outcome::Completed(result) = outcome {
        println!("\nWith custom config:");
        println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
    }
}

/// Non-constant-time comparison (intentionally leaky for demo).
fn compare_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if a[i] != b[i] {
            return false; // Early exit - timing leak!
        }
    }
    true
}
