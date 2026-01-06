use std::time::Instant;
use timing_oracle::{helpers::InputPair, TimingOracle};

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
    // Run a representative leaky test
    let secret = [0u8; 512];

    println!("Starting profiling run with balanced preset...");
    let start = Instant::now();

    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);
    let outcome = TimingOracle::balanced()
        .test(inputs, |input| {
            std::hint::black_box(early_exit_compare(&secret, input));
        });

    let total_time = start.elapsed();

    let result = outcome.unwrap_completed();
    println!("\nTotal execution time: {:.2}s", total_time.as_secs_f64());
    println!("Leak probability: {:.3}", result.leak_probability);
}
