use std::time::{Duration, Instant};
use timing_oracle::{helpers::InputPair, AttackerModel, TimingOracle};

fn main() {
    println!("=== Performance Baseline (With Correct Bootstrap Counts) ===\n");

    // Simulating a timing leak via different iteration counts
    // Fixed: 1000 iterations, Random: 1100 iterations
    fn do_work(iterations: &u32) {
        let mut x = 0u64;
        for i in 0..*iterations {
            x = x.wrapping_add(std::hint::black_box(i as u64));
        }
        std::hint::black_box(x);
    }

    // Test 1: Quick preset (5k samples, 50 CI bootstrap, 50 cov bootstrap)
    println!("Testing Quick preset (5k samples, 50 CI bootstrap, 50 cov bootstrap)...");
    let inputs = InputPair::new(|| 1000u32, || 1100u32);
    let start = Instant::now();
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(10))
        .max_samples(5_000)
        .test(inputs, do_work);
    let quick_time = start.elapsed();
    if let Some(leak_prob) = outcome.leak_probability() {
        println!("  Time: {:>8.3}s", quick_time.as_secs_f64());
        println!("  Leak prob: {:.3}", leak_prob);
    }
    println!();

    // Test 2: Balanced preset (20k samples, 100 CI bootstrap, 50 cov bootstrap)
    println!("Testing Balanced preset (20k samples, 100 CI bootstrap, 50 cov bootstrap)...");
    let inputs = InputPair::new(|| 1000u32, || 1100u32);
    let start = Instant::now();
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .max_samples(20_000)
        .test(inputs, do_work);
    let balanced_time = start.elapsed();
    if let Some(leak_prob) = outcome.leak_probability() {
        println!("  Time: {:>8.3}s", balanced_time.as_secs_f64());
        println!("  Leak prob: {:.3}", leak_prob);
    }
    println!();

    // Test 3: Default preset (100k samples, 10k CI bootstrap, 2k cov bootstrap)
    println!("Testing Default preset (100k samples, 10k CI bootstrap, 2k cov bootstrap)...");
    let inputs = InputPair::new(|| 1000u32, || 1100u32);
    let start = Instant::now();
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(60))
        .max_samples(100_000)
        .test(inputs, do_work);
    let default_time = start.elapsed();
    if let Some(leak_prob) = outcome.leak_probability() {
        println!("  Time: {:>8.3}s", default_time.as_secs_f64());
        println!("  Leak prob: {:.3}", leak_prob);
    }
    println!();

    println!("=== Summary ===");
    println!("Quick:    {:>8.3}s", quick_time.as_secs_f64());
    println!("Balanced: {:>8.3}s", balanced_time.as_secs_f64());
    println!("Default:  {:>8.3}s", default_time.as_secs_f64());
    println!("\nNote: Using adaptive sampling with time budgets");
}
