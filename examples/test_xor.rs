//! Example: XOR constant-time test.
//!
//! XOR operations should be constant-time - this test verifies
//! that no timing leak is detected with adaptive batching.

use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};

fn main() {
    // Create InputPair for tuples of two arrays
    let inputs = InputPair::new(
        || ([0u8; 32], [0u8; 32]),
        || (rand_bytes(), rand_bytes()),
    );

    println!("Testing XOR operation for timing leaks...");
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |(a, b)| {
            std::hint::black_box(xor_bytes(a, b));
        });

    match outcome {
        Outcome::Pass {
            leak_probability,
            quality,
            diagnostics,
            ..
        } => {
            println!("Result: PASS (expected for XOR)");
            println!("Leak probability: {:.4}%", leak_probability * 100.0);
            println!("Quality: {:?}", quality);
            println!(
                "Timer resolution: {:.1}ns",
                diagnostics.timer_resolution_ns
            );
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            println!("Result: FAIL (unexpected for XOR!)");
            println!("Leak probability: {:.4}%", leak_probability * 100.0);
            println!("Exploitability: {:?}", exploitability);
        }
        Outcome::Inconclusive {
            leak_probability,
            reason,
            ..
        } => {
            println!("Result: INCONCLUSIVE");
            println!("Leak probability: {:.4}%", leak_probability * 100.0);
            println!("Reason: {:?}", reason);
        }
        Outcome::Unmeasurable {
            recommendation, ..
        } => {
            println!("Could not measure: {}", recommendation);
        }
    }
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
