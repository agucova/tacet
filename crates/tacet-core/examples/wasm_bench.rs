//! Quick benchmark to compare native vs WASM performance.
//!
//! Run native:
//!   cargo run --example wasm_bench --release
//!
//! Run WASM:
//!   cargo build --example wasm_bench --release --target wasm32-wasip1
//!   wasmtime --dir=. target/wasm32-wasip1/release/examples/wasm_bench.wasm

use std::time::Instant;

fn main() {
    println!("Tacet WASM Performance Benchmark");
    println!("=================================\n");

    // Generate synthetic timing data (simulating what JS would collect)
    let n_samples = 5000;
    let mut baseline: Vec<u64> = Vec::with_capacity(n_samples);
    let mut sample: Vec<u64> = Vec::with_capacity(n_samples);

    // Simple PRNG (xorshift) to avoid rand dependency issues in WASM
    let mut rng_state: u64 = 12345;
    let mut next_u64 = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    // Generate baseline: ~1000ns with noise
    for _ in 0..n_samples {
        let noise = (next_u64() % 100) as u64;
        baseline.push(1000 + noise);
    }

    // Generate sample: ~1000ns with noise (no leak scenario)
    for _ in 0..n_samples {
        let noise = (next_u64() % 100) as u64;
        sample.push(1000 + noise);
    }

    println!("Generated {} samples per class", n_samples);

    // Benchmark 1: Quantile computation
    println!("\n1. Quantile computation (1000 iterations):");
    let start = Instant::now();
    for _ in 0..1000 {
        let mut sorted = baseline.clone();
        sorted.sort_unstable();
        let _q50 = sorted[sorted.len() / 2];
        let _q99 = sorted[sorted.len() * 99 / 100];
    }
    let elapsed = start.elapsed();
    println!("   Elapsed: {:?}", elapsed);

    // Benchmark 2: Bootstrap-like resampling
    println!("\n2. Bootstrap resampling (200 iterations, {} samples):", n_samples);
    let start = Instant::now();
    let mut bootstrap_means: Vec<f64> = Vec::with_capacity(200);
    for i in 0..200 {
        // Resample with replacement using simple modular indexing
        let mut sum: u64 = 0;
        for j in 0..n_samples {
            let idx = ((i * 7 + j * 13) % n_samples) as usize;
            sum += baseline[idx];
        }
        bootstrap_means.push(sum as f64 / n_samples as f64);
    }
    let elapsed = start.elapsed();
    println!("   Elapsed: {:?}", elapsed);
    println!("   Mean of bootstrap means: {:.2}", bootstrap_means.iter().sum::<f64>() / 200.0);

    // Benchmark 3: Difference computation and statistics
    println!("\n3. Difference statistics:");
    let start = Instant::now();
    let mut diffs: Vec<f64> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        diffs.push(sample[i] as f64 - baseline[i] as f64);
    }

    // Mean
    let mean: f64 = diffs.iter().sum::<f64>() / n_samples as f64;

    // Variance
    let variance: f64 = diffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_samples - 1) as f64;
    let std_dev = variance.sqrt();

    let elapsed = start.elapsed();
    println!("   Elapsed: {:?}", elapsed);
    println!("   Mean diff: {:.2} ns, Std dev: {:.2} ns", mean, std_dev);

    // Benchmark 4: Simulated Bayesian update (matrix-like operations)
    println!("\n4. Simulated covariance/posterior update (100 iterations):");
    let start = Instant::now();
    for _ in 0..100 {
        // Simulate 2x2 covariance matrix operations
        let sigma = [[variance, variance * 0.1], [variance * 0.1, variance]];

        // Invert 2x2 matrix
        let det = sigma[0][0] * sigma[1][1] - sigma[0][1] * sigma[1][0];
        let _inv = [
            [sigma[1][1] / det, -sigma[0][1] / det],
            [-sigma[1][0] / det, sigma[0][0] / det],
        ];

        // Compute quadratic form (simplified posterior calculation)
        let x = [mean, std_dev];
        let _quad = x[0] * _inv[0][0] * x[0] + 2.0 * x[0] * _inv[0][1] * x[1] + x[1] * _inv[1][1] * x[1];
    }
    let elapsed = start.elapsed();
    println!("   Elapsed: {:?}", elapsed);

    // Total benchmark
    println!("\n5. Combined benchmark (simulating full calibration):");
    let start = Instant::now();

    // Sort both arrays
    let mut b_sorted = baseline.clone();
    let mut s_sorted = sample.clone();
    b_sorted.sort_unstable();
    s_sorted.sort_unstable();

    // 200 bootstrap iterations
    let mut shift_samples: Vec<f64> = Vec::with_capacity(200);
    for i in 0..200 {
        let mut b_sum: u64 = 0;
        let mut s_sum: u64 = 0;
        for j in 0..n_samples {
            let idx = ((i * 7 + j * 13) % n_samples) as usize;
            b_sum += baseline[idx];
            s_sum += sample[idx];
        }
        let b_mean = b_sum as f64 / n_samples as f64;
        let s_mean = s_sum as f64 / n_samples as f64;
        shift_samples.push(s_mean - b_mean);
    }

    // Compute shift estimate
    shift_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let shift_median = shift_samples[100];
    let shift_ci_low = shift_samples[5];
    let shift_ci_high = shift_samples[195];

    let elapsed = start.elapsed();
    println!("   Elapsed: {:?}", elapsed);
    println!("   Shift estimate: {:.2} ns (95% CI: [{:.2}, {:.2}])", shift_median, shift_ci_low, shift_ci_high);

    println!("\n=================================");
    println!("Benchmark complete.");
}
