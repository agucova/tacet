//! Tests that must not false-positive on constant-time code.
use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};

fn main() {
    test_no_false_positive_xor_compare();
}

fn test_no_false_positive_xor_compare() {
    let secret = [0xABu8; 32];

    // Pre-generate inputs using InputPair
    let inputs = InputPair::new(|| [0xABu8; 32], rand_bytes);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .max_samples(100_000)
        .test(inputs, |data| {
            constant_time_compare(&secret, data);
        });

    let leak_probability = match &outcome {
        Outcome::Pass { leak_probability, .. } => *leak_probability,
        Outcome::Fail { leak_probability, .. } => *leak_probability,
        Outcome::Inconclusive { leak_probability, .. } => *leak_probability,
        Outcome::Unmeasurable { .. } => {
            println!("Operation too fast to measure");
            return;
        }
    };

    let passed = outcome.passed();

    println!("Leak probability: {}", leak_probability);
    println!("Test passed: {}", passed);

    if leak_probability > 0.5 {
        println!(
            "ERROR: False positive! leak_probability={}",
            leak_probability
        );
    } else {
        println!("OK: No false positive detected");
    }
    if !passed {
        println!("ERROR: Test failed!");
    } else {
        println!("OK: Test passed");
    }
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
