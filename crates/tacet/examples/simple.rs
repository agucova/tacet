//! Simple example demonstrating basic tacet usage.
//!
//! This demonstrates the CORRECT way to test operations:
//! - Pre-generate all inputs before measurement
//! - The test closure executes identical code for fixed and random inputs
//! - Only the input data differs

use std::time::Duration;
use tacet::{helpers::InputPair, timing_test_checked, AttackerModel, Outcome, TimingOracle};

fn main() {
    println!("tacet simple example\n");

    // Example: Testing a potentially leaky comparison
    let secret = [0u8; 32];

    // Pre-generate inputs using InputPair
    let inputs = InputPair::new(
        || [0u8; 32], // Baseline: all zeros (same as secret)
        || {
            let mut arr = [0u8; 32];
            for item in &mut arr {
                *item = rand::random();
            }
            arr
        },
    );

    // Simple API with attacker model
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            compare_bytes(&secret, data);
        });

    match &outcome {
        Outcome::Pass {
            leak_probability,
            quality,
            diagnostics,
            ..
        } => {
            println!("Result: PASS");
            println!("Leak probability: {:.1}%", leak_probability * 100.0);
            println!("Quality: {:?}", quality);
            println!("Timer resolution: {:.1}ns", diagnostics.timer_resolution_ns);
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            effect,
            ..
        } => {
            println!("Result: FAIL");
            println!("Leak probability: {:.1}%", leak_probability * 100.0);
            println!("Exploitability: {:?}", exploitability);
            println!(
                "Effect: shift={:.1}ns, tail={:.1}ns",
                effect.shift_ns, effect.tail_ns
            );
        }
        Outcome::Inconclusive {
            reason,
            leak_probability,
            ..
        } => {
            println!("Result: INCONCLUSIVE");
            println!("Leak probability: {:.1}%", leak_probability * 100.0);
            println!("Reason: {:?}", reason);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            println!("Could not measure: {}", recommendation);
            return;
        }
        Outcome::Research(research) => {
            println!("Result: RESEARCH MODE");
            println!("Status: {:?}", research.status);
            println!("Max effect: {:.1}ns", research.max_effect_ns);
            return;
        }
    }

    // Using timing_test_checked! macro with custom config
    let outcome = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .time_budget(Duration::from_secs(10)),
        baseline: || [0u8; 32],
        sample: || {
            let mut arr = [0u8; 32];
            for item in &mut arr {
                *item = rand::random();
            }
            arr
        },
        measure: |data| {
            compare_bytes(&secret, data);
        },
    };

    if let Outcome::Pass {
        leak_probability, ..
    }
    | Outcome::Fail {
        leak_probability, ..
    } = outcome
    {
        println!("\nWith custom config:");
        println!("Leak probability: {:.1}%", leak_probability * 100.0);
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
