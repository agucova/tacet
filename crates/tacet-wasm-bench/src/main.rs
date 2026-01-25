//! Standalone benchmark for comparing native vs WASM performance.
//!
//! This benchmark simulates the core statistical operations in tacet:
//! - Sorting for quantile computation
//! - Bootstrap resampling
//! - Mean/variance calculation
//! - Covariance matrix operations
//!
//! Run native:
//!   cargo run --release -p tacet-wasm-bench
//!
//! Run WASM:
//!   cargo build --release --target wasm32-wasip1 -p tacet-wasm-bench
//!   wasmtime target/wasm32-wasip1/release/wasm-bench.wasm

use std::time::Instant;

/// Simple xorshift PRNG (no external dependencies)
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

fn main() {
    println!("Tacet WASM Performance Benchmark");
    println!("=================================\n");

    let n_samples = 5000;
    let bootstrap_iterations = 200;

    // Generate synthetic timing data
    let mut rng = Rng::new(12345);
    let mut baseline: Vec<u64> = Vec::with_capacity(n_samples);
    let mut sample: Vec<u64> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        baseline.push(1000 + (rng.next_u64() % 100));
        sample.push(1000 + (rng.next_u64() % 100));
    }

    println!("Generated {} samples per class\n", n_samples);

    // Benchmark 1: Sorting (for quantile computation)
    println!("1. Sorting (1000 iterations of {} samples):", n_samples);
    let start = Instant::now();
    for _ in 0..1000 {
        let mut sorted = baseline.clone();
        sorted.sort_unstable();
        std::hint::black_box(&sorted);
    }
    let elapsed = start.elapsed();
    println!("   Elapsed: {:?}", elapsed);

    // Benchmark 2: Bootstrap resampling with replacement
    println!(
        "\n2. Bootstrap resampling ({} iterations, {} samples):",
        bootstrap_iterations, n_samples
    );
    let start = Instant::now();
    let mut bootstrap_means: Vec<f64> = Vec::with_capacity(bootstrap_iterations);
    for _ in 0..bootstrap_iterations {
        let mut sum: u64 = 0;
        for _ in 0..n_samples {
            let idx = rng.next_usize(n_samples);
            sum += baseline[idx];
        }
        bootstrap_means.push(sum as f64 / n_samples as f64);
    }
    let elapsed = start.elapsed();
    println!("   Elapsed: {:?}", elapsed);
    println!(
        "   Mean of bootstrap means: {:.2}",
        bootstrap_means.iter().sum::<f64>() / bootstrap_iterations as f64
    );

    // Benchmark 3: Difference statistics (mean, variance, std dev)
    println!("\n3. Difference statistics:");
    let start = Instant::now();
    let mut diffs: Vec<f64> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        diffs.push(sample[i] as f64 - baseline[i] as f64);
    }

    let mean: f64 = diffs.iter().sum::<f64>() / n_samples as f64;
    let variance: f64 =
        diffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_samples - 1) as f64;
    let std_dev = variance.sqrt();

    let elapsed = start.elapsed();
    println!("   Elapsed: {:?}", elapsed);
    println!("   Mean diff: {:.2} ns, Std dev: {:.2} ns", mean, std_dev);

    // Benchmark 4: Covariance/posterior update simulation
    println!("\n4. Covariance matrix operations (1000 iterations):");
    let start = Instant::now();
    let mut result = 0.0f64;
    for _ in 0..1000 {
        // Simulate 2x2 covariance matrix operations
        let sigma = [[variance, variance * 0.1], [variance * 0.1, variance]];

        // Invert 2x2 matrix
        let det = sigma[0][0] * sigma[1][1] - sigma[0][1] * sigma[1][0];
        let inv = [
            [sigma[1][1] / det, -sigma[0][1] / det],
            [-sigma[1][0] / det, sigma[0][0] / det],
        ];

        // Compute quadratic form
        let x = [mean, std_dev];
        let quad =
            x[0] * inv[0][0] * x[0] + 2.0 * x[0] * inv[0][1] * x[1] + x[1] * inv[1][1] * x[1];
        result += quad;
    }
    std::hint::black_box(result);
    let elapsed = start.elapsed();
    println!("   Elapsed: {:?}", elapsed);

    // Benchmark 5: Full calibration simulation
    println!("\n5. Full calibration simulation:");
    let start = Instant::now();

    // Sort both arrays (for quantiles)
    let mut b_sorted = baseline.clone();
    let mut s_sorted = sample.clone();
    b_sorted.sort_unstable();
    s_sorted.sort_unstable();

    // Bootstrap for shift estimation
    let mut shift_samples: Vec<f64> = Vec::with_capacity(bootstrap_iterations);
    for _ in 0..bootstrap_iterations {
        let mut b_sum: u64 = 0;
        let mut s_sum: u64 = 0;
        for _ in 0..n_samples {
            let b_idx = rng.next_usize(n_samples);
            let s_idx = rng.next_usize(n_samples);
            b_sum += baseline[b_idx];
            s_sum += sample[s_idx];
        }
        let b_mean = b_sum as f64 / n_samples as f64;
        let s_mean = s_sum as f64 / n_samples as f64;
        shift_samples.push(s_mean - b_mean);
    }

    // Compute shift estimate (median and CI)
    shift_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let shift_median = shift_samples[bootstrap_iterations / 2];
    let shift_ci_low = shift_samples[bootstrap_iterations * 5 / 100];
    let shift_ci_high = shift_samples[bootstrap_iterations * 95 / 100];

    let elapsed = start.elapsed();
    println!("   Elapsed: {:?}", elapsed);
    println!(
        "   Shift estimate: {:.2} ns (95% CI: [{:.2}, {:.2}])",
        shift_median, shift_ci_low, shift_ci_high
    );

    // Summary
    println!("\n=================================");
    println!("Benchmark complete.");
}
