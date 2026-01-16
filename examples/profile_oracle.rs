use std::time::{Duration, Instant};
use timing_oracle::{helpers::InputPair, AttackerModel, TimingOracle};

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
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .max_samples(20_000)
        .test(inputs, |input| {
            std::hint::black_box(early_exit_compare(&secret, input));
        });

    let total_time = start.elapsed();

    println!("\nTotal execution time: {:.2}s", total_time.as_secs_f64());
    if let Some(leak_prob) = outcome.leak_probability() {
        println!("Leak probability: {:.3}", leak_prob);
    } else {
        println!("Unmeasurable");
    }
}
