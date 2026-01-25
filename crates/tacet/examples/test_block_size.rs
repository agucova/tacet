//! Test mbedTLS with different block sizes to see impact on theta_floor

use std::path::Path;

use nalgebra::{SMatrix, SVector};
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use tacet::data::load_two_column_csv;

type Vector9 = SVector<f64, 9>;
type Matrix9 = SMatrix<f64, 9, 9>;

fn main() {
    let silent_path = Path::new("/Users/agucova/repos/SILENT/examples");

    println!("=== mbedTLS (500k samples, extreme autocorrelation) ===");
    analyze_dataset(
        &silent_path.join("paper-mbedtls/Wrong_second_byte_(0x02_set_to_0x17)vsCorrectly_formatted_PKCS#1_PMS_message.csv"),
        "BASELINE", "MODIFIED",
        1.0 / 3.0,
        None, // use all samples
    );

    println!("\n=== KyberSlash (5k samples, real timing leak) ===");
    analyze_dataset(
        &silent_path.join("paper-kyberslash/timing_measurements_1_comparison_-72_72.csv"),
        "X",
        "Y",
        1.0 / 3.0,
        None,
    );
}

fn analyze_dataset(path: &Path, label1: &str, label2: &str, ns_factor: f64, max_n: Option<usize>) {
    let data = load_two_column_csv(path, true, label1, label2).expect("Failed to load data");
    let (mut baseline_ns, mut test_ns) = data.to_nanoseconds(ns_factor);

    if let Some(max) = max_n {
        baseline_ns.truncate(max);
        test_ns.truncate(max);
    }

    let n = baseline_ns.len();
    println!("Samples: {}", n);

    let max_diff = compute_max_quantile_diff(&baseline_ns, &test_ns);
    println!("Max quantile diff: {:.1} ns", max_diff);

    // Compute autocorrelation at lag 1
    let acf1 = compute_lag1_acf(&baseline_ns);
    println!("Lag-1 ACF: {:.3}", acf1);

    // Compute what the ACTUAL code would use for m and block_size
    let m = if n < 2000 {
        let half = (0.5 * n as f64) as usize;
        half.max(200).min(n)
    } else {
        let m_calc = (n as f64).powf(2.0 / 3.0) as usize;
        m_calc.max(400).min(n)
    };
    let max_block = (m / 5).max(1);
    println!("m-out-of-n: m={}, max_block={}", m, max_block);

    println!();
    println!(
        "  {:>5}  {:>12}  {:>12}  {:>12}  {:>10}",
        "block", "Σ_rate[0,0]", "Σ_rate[4,4]", "θ_floor", "ratio"
    );

    // Test the actual max_block and a few smaller values
    let test_blocks: Vec<usize> = vec![10, 20, 50, max_block.min(100), max_block]
        .into_iter()
        .filter(|&b| b <= n / 2)
        .collect();
    let test_blocks: Vec<usize> = test_blocks
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    for block_size in test_blocks {
        let cov = bootstrap_with_block_size(&baseline_ns, &test_ns, block_size, 500, 42);
        let sigma_rate = cov * (n as f64);
        let c_floor = compute_c_floor_9d(&sigma_rate, 42);
        let theta_floor = c_floor / (n as f64).sqrt();

        let ratio = theta_floor / max_diff;
        let marker = if block_size == max_block {
            " <-- actual max"
        } else {
            ""
        };

        println!(
            "  {:5}  {:12.2e}  {:12.2e}  {:12.1}  {:10.2}x{}",
            block_size,
            sigma_rate[(0, 0)],
            sigma_rate[(4, 4)],
            theta_floor,
            ratio,
            marker
        );
    }
}

fn compute_lag1_acf(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-10 {
        return 0.0;
    }
    let cov: f64 = data
        .windows(2)
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum::<f64>()
        / (n - 1) as f64;
    cov / var
}

fn compute_max_quantile_diff(baseline: &[f64], test: &[f64]) -> f64 {
    let n = baseline.len();
    let mut b = baseline.to_vec();
    let mut t = test.to_vec();
    b.sort_by(|a, c| a.partial_cmp(c).unwrap());
    t.sort_by(|a, c| a.partial_cmp(c).unwrap());

    let mut max_diff = 0.0f64;
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let idx = (n as f64 * p) as usize;
        let diff = (b[idx] - t[idx]).abs();
        max_diff = max_diff.max(diff);
    }
    max_diff
}

fn bootstrap_with_block_size(
    baseline: &[f64],
    test: &[f64],
    block_size: usize,
    n_bootstrap: usize,
    seed: u64,
) -> Matrix9 {
    let n = baseline.len().min(test.len());

    // Welford accumulator
    let mut count = 0usize;
    let mut mean = Vector9::zeros();
    let mut m2 = Matrix9::zeros();

    for i in 0..n_bootstrap {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(i as u64));

        // Block bootstrap with paired indices
        let mut baseline_buf = Vec::with_capacity(n);
        let mut test_buf = Vec::with_capacity(n);

        while baseline_buf.len() < n {
            let start = rng.random_range(0..n.saturating_sub(block_size).max(1));
            for offset in 0..block_size {
                if baseline_buf.len() >= n {
                    break;
                }
                let idx = (start + offset) % n;
                baseline_buf.push(baseline[idx]);
                test_buf.push(test[idx]);
            }
        }

        // Compute deciles
        let q_b = compute_deciles(&mut baseline_buf);
        let q_t = compute_deciles(&mut test_buf);
        let delta = q_b - q_t;

        // Welford update
        count += 1;
        let d1 = delta - mean;
        mean += d1 / (count as f64);
        let d2 = delta - mean;
        m2 += d1 * d2.transpose();
    }

    if count < 2 {
        return Matrix9::identity() * 1e6;
    }

    m2 / ((count - 1) as f64)
}

fn compute_deciles(data: &mut [f64]) -> Vector9 {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len();
    Vector9::from_fn(|i, _| {
        let p = (i + 1) as f64 / 10.0;
        let idx = ((n as f64) * p) as usize;
        data[idx.min(n - 1)]
    })
}

fn compute_c_floor_9d(sigma_rate: &Matrix9, seed: u64) -> f64 {
    // Sample max|Z_k| where Z ~ N(0, Σ_rate)
    let chol = match nalgebra::Cholesky::new(*sigma_rate) {
        Some(c) => c,
        None => return f64::INFINITY,
    };
    let l = chol.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let n_samples = 10_000;
    let mut max_values = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // z ~ N(0, I)
        let z = Vector9::from_fn(|_, _| sample_standard_normal(&mut rng));
        // x = L * z ~ N(0, Σ_rate)
        let x = l * z;
        let max_abs = x.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        max_values.push(max_abs);
    }

    max_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95_idx = (n_samples as f64 * 0.95) as usize;
    max_values[p95_idx]
}

fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    // Box-Muller transform
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
