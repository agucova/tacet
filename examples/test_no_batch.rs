// Quick test to inspect adaptive batching behavior and false positives
use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};

fn main() {
    let secret = [0xABu8; 32];
    let inputs = InputPair::new(|| [0xABu8; 32], rand_bytes);

    // Test with adaptive batching
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(100_000)
        .time_budget(Duration::from_secs(60))
        .test(inputs, |input| {
            std::hint::black_box(constant_time_compare(&secret, input));
        });

    println!("=== Adaptive batching ===");
    match outcome {
        Outcome::Pass { leak_probability, diagnostics, .. } => {
            println!("Result: PASS");
            println!("Leak probability: {:.4}", leak_probability);
            println!("Discrete mode: {}", diagnostics.discrete_mode);
        }
        Outcome::Fail { leak_probability, diagnostics, .. } => {
            println!("Result: FAIL");
            println!("Leak probability: {:.4}", leak_probability);
            println!("Discrete mode: {}", diagnostics.discrete_mode);
        }
        Outcome::Inconclusive { leak_probability, reason, .. } => {
            println!("Result: INCONCLUSIVE");
            println!("Leak probability: {:.4}", leak_probability);
            println!("Reason: {:?}", reason);
        }
        Outcome::Unmeasurable { operation_ns, threshold_ns, .. } => {
            println!("Result: UNMEASURABLE");
            println!("Operation: {:.1}ns, threshold: {:.1}ns", operation_ns, threshold_ns);
        }
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
