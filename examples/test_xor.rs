use timing_oracle::helpers::InputPair;
use timing_oracle::{Outcome, TimingOracle};

fn main() {
    // Create InputPair for tuples of two arrays
    let inputs = InputPair::new(
        || ([0u8; 32], [0u8; 32]),
        || (rand_bytes(), rand_bytes()),
    );

    println!("Testing no_false_positive_xor with adaptive batching...");
    let outcome = TimingOracle::new()
        .samples(100_000)
        .test(inputs, |(a, b)| {
            std::hint::black_box(xor_bytes(a, b));
        });

    let result = match outcome {
        Outcome::Completed(r) => r,
        Outcome::Unmeasurable { recommendation, .. } => {
            println!("Could not measure: {}", recommendation);
            return;
        }
    };

    println!("CI gate passed: {}", result.ci_gate.passed());
    println!("Leak probability: {:.4}", result.leak_probability);
    println!("Batching: enabled={}, K={}, ticks_per_batch={:.1}",
        result.metadata.batching.enabled,
        result.metadata.batching.k,
        result.metadata.batching.ticks_per_batch);
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
