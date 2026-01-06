use std::time::Instant;
use timing_oracle::helpers::InputPair;
use timing_oracle::TimingOracle;

fn early_exit_compare(a: &[u8], b: &[u8]) -> bool {
    for i in 0..a.len().min(b.len()) {
        if a[i] != b[i] {
            return false;
        }
    }
    a.len() == b.len()
}

fn rand_bytes_512() -> [u8; 512] {
    let mut input = [0u8; 512];
    for byte in &mut input {
        *byte = rand::random();
    }
    input
}

fn main() {
    // Run with balanced samples (20k) but CORRECT bootstrap iterations (10k/2k)
    let secret = [0u8; 512];

    println!("Starting benchmark with 20k samples, 10k CI bootstrap, 2k cov bootstrap...");
    let start = Instant::now();

    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);

    let result = TimingOracle::new()
        .samples(20_000)  // Use balanced sample count
        // Uses default ci_bootstrap_iterations: 10_000
        // Uses default cov_bootstrap_iterations: 2_000
        .test(inputs, |input| {
            std::hint::black_box(early_exit_compare(&secret, input));
        })
        .unwrap_completed();

    let total_time = start.elapsed();

    println!("\nTotal execution time: {:.2}s", total_time.as_secs_f64());
    println!("Leak probability: {:.3}", result.leak_probability);
}
