//! Profiling harness for measuring timing-oracle performance.
//!
//! Run with: cargo run --release --features parallel --example profile

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
    let mut arr = [0u8; 512];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

fn main() {
    let secret = [0u8; 512];

    println!("=== Timing Oracle Performance Profiling ===\n");

    // Test 1: Balanced preset (20k samples)
    println!("Test 1: Balanced preset (20k samples, 100 CI bootstrap, 50 cov bootstrap)");
    let start = Instant::now();
    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .max_samples(20_000)
        .test(inputs, |input| {
            std::hint::black_box(early_exit_compare(&secret, input));
        });
    let balanced_time = start.elapsed();

    if let Some(leak_prob) = outcome.leak_probability() {
        println!("  Total time: {:.2}s", balanced_time.as_secs_f64());
        println!("  Leak probability: {:.3}", leak_prob);
        println!("  Samples used: {:?}", outcome.samples_used());
    } else {
        println!("  Unmeasurable");
    }
    println!();

    // Test 2: Quick preset (5k samples)
    println!("Test 2: Quick preset (5k samples, 50 CI bootstrap, 50 cov bootstrap)");
    let start = Instant::now();
    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(10))
        .max_samples(5_000)
        .test(inputs, |input| {
            std::hint::black_box(early_exit_compare(&secret, input));
        });
    let quick_time = start.elapsed();

    if let Some(leak_prob) = outcome.leak_probability() {
        println!("  Total time: {:.2}s", quick_time.as_secs_f64());
        println!("  Leak probability: {:.3}", leak_prob);
        println!("  Samples used: {:?}", outcome.samples_used());
    } else {
        println!("  Unmeasurable");
    }
    println!();

    // Test 3: Default preset (100k samples)
    println!("Test 3: Default preset (100k samples, 100 CI bootstrap, 50 cov bootstrap)");
    let start = Instant::now();
    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(60))
        .max_samples(100_000)
        .test(inputs, |input| {
            std::hint::black_box(early_exit_compare(&secret, input));
        });
    let default_time = start.elapsed();

    if let Some(leak_prob) = outcome.leak_probability() {
        println!("  Total time: {:.2}s", default_time.as_secs_f64());
        println!("  Leak probability: {:.3}", leak_prob);
        println!("  Samples used: {:?}", outcome.samples_used());
    } else {
        println!("  Unmeasurable");
    }
    println!();

    println!("=== Summary ===");
    println!("Quick (5k):     {:.2}s", quick_time.as_secs_f64());
    println!("Balanced (20k): {:.2}s", balanced_time.as_secs_f64());
    println!("Default (100k): {:.2}s", default_time.as_secs_f64());
    println!();
    println!("Time per sample:");
    println!("  Quick:    {:.2}μs", quick_time.as_micros() as f64 / 5_000.0);
    println!("  Balanced: {:.2}μs", balanced_time.as_micros() as f64 / 20_000.0);
    println!("  Default:  {:.2}μs", default_time.as_micros() as f64 / 100_000.0);
}
