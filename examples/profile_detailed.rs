//! Detailed profiling with stage-level timing instrumentation.
//!
//! This manually instruments the oracle to measure time spent in each stage.
//!
//! NOTE: This example uses internal APIs that may have changed with the
//! adaptive sampling rewrite. Some statistics functions may no longer be
//! publicly exported. This is a demonstration of profiling methodology.

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

fn main() {
    println!("=== Timing Oracle Profiling ===\n");

    let secret = [0u8; 512];
    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);

    println!("Testing early-exit comparison with adaptive sampling...\n");

    let start = Instant::now();
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .max_samples(20_000)
        .test(inputs, |input| {
            std::hint::black_box(early_exit_compare(&secret, input));
        });

    let total_time = start.elapsed();

    println!("Total execution time: {:.3}s\n", total_time.as_secs_f64());

    match outcome {
        Outcome::Pass { leak_probability, samples_used, diagnostics, .. } => {
            println!("Result: PASS");
            println!("  Leak probability: {:.3}", leak_probability);
            println!("  Samples used: {}", samples_used);
            println!("  Discrete mode: {}", diagnostics.discrete_mode);
            println!("  Timer resolution: {:.1}ns", diagnostics.timer_resolution_ns);
        }
        Outcome::Fail { leak_probability, samples_used, diagnostics, exploitability, .. } => {
            println!("Result: FAIL");
            println!("  Leak probability: {:.3}", leak_probability);
            println!("  Samples used: {}", samples_used);
            println!("  Exploitability: {:?}", exploitability);
            println!("  Discrete mode: {}", diagnostics.discrete_mode);
            println!("  Timer resolution: {:.1}ns", diagnostics.timer_resolution_ns);
        }
        Outcome::Inconclusive { leak_probability, reason, samples_used, diagnostics, .. } => {
            println!("Result: INCONCLUSIVE");
            println!("  Reason: {:?}", reason);
            println!("  Leak probability: {:.3}", leak_probability);
            println!("  Samples used: {}", samples_used);
            println!("  Timer resolution: {:.1}ns", diagnostics.timer_resolution_ns);
        }
        Outcome::Unmeasurable { operation_ns, threshold_ns, platform, .. } => {
            println!("Result: UNMEASURABLE");
            println!("  Operation: {:.1}ns", operation_ns);
            println!("  Threshold: {:.1}ns", threshold_ns);
            println!("  Platform: {}", platform);
        }
    }
}

fn rand_bytes_512() -> [u8; 512] {
    let mut arr = [0u8; 512];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
