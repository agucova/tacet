//! Demonstrates the new probability clamping behavior.
//!
//! This example shows how extremely obvious timing leaks now report 99.99%
//! probability instead of 100%, staying philosophically sound.

use timing_oracle::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};
use std::thread;
use std::time::Duration;

fn main() {
    println!("Demonstrating probability clamping\n");

    // Create an EXTREMELY obvious timing leak (1ms vs 0ms)
    let inputs = InputPair::new(|| true, || false);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(10))
        .test(inputs, |input| {
            if *input {
                // Extremely obvious delay
                thread::sleep(Duration::from_micros(100));
            }
        });

    match outcome {
        Outcome::Pass { leak_probability, .. } => {
            println!("Results for obvious timing leak (100us difference):");
            println!("  Result: PASS (unexpected!)");
            println!("  Leak probability: {:.2}%", leak_probability * 100.0);
        }
        Outcome::Fail { leak_probability, .. } => {
            println!("Results for obvious timing leak (100us difference):");
            println!("  Result: FAIL (expected)");
            println!("  Leak probability: {:.2}%", leak_probability * 100.0);

            if leak_probability >= 0.9999 {
                println!("\n  Probability capped at 99.99% (not 100%)");
                println!("  This represents overwhelming evidence while staying");
                println!("  philosophically honest - we never claim P=1.0");
            } else {
                println!("\n  Probability: {:.4}", leak_probability);
            }
        }
        Outcome::Inconclusive { leak_probability, reason, .. } => {
            println!("Results: INCONCLUSIVE");
            println!("  Reason: {:?}", reason);
            println!("  Leak probability: {:.2}%", leak_probability * 100.0);
        }
        Outcome::Unmeasurable { operation_ns, threshold_ns, .. } => {
            println!("Results: UNMEASURABLE");
            println!("  Operation: {:.1}ns, threshold: {:.1}ns", operation_ns, threshold_ns);
        }
    }
}
