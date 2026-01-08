// Quick test to inspect adaptive batching behavior and false positives
use timing_oracle::helpers::InputPair;
use timing_oracle::TimingOracle;

fn main() {
    let secret = [0xABu8; 32];
    let inputs = InputPair::new(|| [0xABu8; 32], rand_bytes);

    // Test with adaptive batching
    let result = TimingOracle::new()
        .samples(100_000)
        .test(inputs, |input| {
            std::hint::black_box(constant_time_compare(&secret, input));
        })
        .unwrap_completed();

    println!("=== Adaptive batching ===");
    println!("Leak probability: {:.4}", result.leak_probability);
    println!("CI gate passed: {}", result.ci_gate.passed());
    println!("Batching: enabled={}, K={}, ticks_per_batch={:.1}",
        result.metadata.batching.enabled,
        result.metadata.batching.k,
        result.metadata.batching.ticks_per_batch);
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
