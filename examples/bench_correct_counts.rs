use std::time::{Duration, Instant};
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};

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
    // Run with balanced samples (20k)
    let secret = [0u8; 512];

    println!("Starting benchmark with adaptive sampling...");
    let start = Instant::now();

    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(20_000)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |input| {
            std::hint::black_box(early_exit_compare(&secret, input));
        });

    let total_time = start.elapsed();

    println!("\nTotal execution time: {:.2}s", total_time.as_secs_f64());
    match outcome {
        Outcome::Pass { leak_probability, .. } => {
            println!("Result: PASS");
            println!("Leak probability: {:.3}", leak_probability);
        }
        Outcome::Fail { leak_probability, .. } => {
            println!("Result: FAIL");
            println!("Leak probability: {:.3}", leak_probability);
        }
        Outcome::Inconclusive { leak_probability, reason, .. } => {
            println!("Result: INCONCLUSIVE ({:?})", reason);
            println!("Leak probability: {:.3}", leak_probability);
        }
        Outcome::Unmeasurable { operation_ns, threshold_ns, .. } => {
            println!("Result: UNMEASURABLE (op={:.1}ns, thresh={:.1}ns)", operation_ns, threshold_ns);
        }
    }
}
