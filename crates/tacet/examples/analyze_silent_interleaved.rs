//! Analyze SILENT paper datasets with simulated interleaving.
//!
//! The SILENT datasets are collected in blocks (all baseline, then all test).
//! This example simulates interleaved acquisition by:
//! 1. Splitting each class into contiguous time blocks of length L
//! 2. Pairing blocks by position (early with early, late with late)
//! 3. Constructing pseudo-interleaved sequences: B1, M1, B2, M2, ...
//!
//! Run with: TIMING_ORACLE_DEBUG=1 cargo run --example analyze_silent_interleaved

use std::path::Path;
use tacet::adaptive::single_pass::{analyze_single_pass, SinglePassConfig};
use tacet::data::{load_two_column_csv, TimeUnit};
use tacet::Outcome;

/// Simulate interleaved acquisition from blocked data.
///
/// Takes two arrays collected as blocks (all baseline, then all test) and
/// reconstructs a pseudo-interleaved acquisition stream by:
/// 1. Splitting each into blocks of length `block_size`
/// 2. Pairing blocks by position (early with early)
/// 3. Interleaving: B1, M1, B2, M2, ...
///
/// Returns (interleaved_baseline, interleaved_test) where samples are reordered
/// to simulate interleaved acquisition.
fn simulate_interleaving(
    baseline: &[f64],
    test: &[f64],
    block_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = baseline.len().min(test.len());
    let num_blocks = n / block_size;

    if num_blocks == 0 {
        // Not enough data for blocking, return as-is
        return (baseline.to_vec(), test.to_vec());
    }

    let mut interleaved_baseline = Vec::with_capacity(num_blocks * block_size);
    let mut interleaved_test = Vec::with_capacity(num_blocks * block_size);

    // Pair blocks by position and interleave
    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = start + block_size;

        // Add baseline block, then test block (simulating B_i, T_i alternation)
        interleaved_baseline.extend_from_slice(&baseline[start..end]);
        interleaved_test.extend_from_slice(&test[start..end]);
    }

    (interleaved_baseline, interleaved_test)
}

/// Estimate optimal block length using a simple lag-1 autocorrelation heuristic.
/// Returns approximately 2 * (first zero-crossing of ACF) or a default.
fn estimate_block_length(data: &[f64]) -> usize {
    let n = data.len();
    if n < 100 {
        return 10;
    }

    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-10 {
        return 10;
    }

    // Find first negative ACF (rough decorrelation lag)
    let max_lag = (n / 4).min(1000);
    let mut first_negative = max_lag;

    for lag in 1..max_lag {
        let cov: f64 = data.windows(2)
            .take(n - lag)
            .enumerate()
            .filter(|(i, _)| *i + lag < n)
            .map(|(i, _)| (data[i] - mean) * (data[i + lag] - mean))
            .sum::<f64>() / (n - lag) as f64;

        let acf = cov / var;
        if acf < 0.0 {
            first_negative = lag;
            break;
        }
    }

    // Use 2x first negative as block length, with bounds
    let block_len = (2 * first_negative).max(10).min(n / 10);
    block_len
}

/// Compute decile differences (baseline - test) for each quantile
fn compute_decile_diffs(baseline: &[f64], test: &[f64]) -> [f64; 9] {
    let mut b = baseline.to_vec();
    let mut t = test.to_vec();
    b.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut diffs = [0.0; 9];
    for (i, p) in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].iter().enumerate() {
        let idx_b = ((b.len() as f64 * p) as usize).min(b.len() - 1);
        let idx_t = ((t.len() as f64 * p) as usize).min(t.len() - 1);
        diffs[i] = b[idx_b] - t[idx_t];
    }
    diffs
}

fn main() {
    println!("=== SILENT Dataset Analysis with Simulated Interleaving ===\n");
    println!("The SILENT datasets are collected in blocks (all baseline, then all test).");
    println!("This violates the interleaved acquisition assumption in our bootstrap.");
    println!("We simulate interleaving by pairing time-matched blocks.\n");

    let silent_path = Path::new("/Users/agucova/repos/SILENT/examples");

    // 1. KyberSlash
    println!("{}", "=".repeat(70));
    println!("1. KyberSlash");
    println!("   Original: 5000 X samples, then 5000 Y samples (BLOCKED)");
    println!("{}", "=".repeat(70));

    analyze_with_interleaving(
        &silent_path.join("paper-kyberslash/timing_measurements_1_comparison_-72_72.csv"),
        "X", "Y",
        TimeUnit::Cycles,
        Some(3.0),
        3.3,
        None, // Use all samples
    );

    // 2. Web App (already roughly interleaved, but let's test anyway)
    println!("\n{}", "=".repeat(70));
    println!("2. Web Application");
    println!("   Original: Roughly interleaved (median run length ~2)");
    println!("{}", "=".repeat(70));

    analyze_with_interleaving(
        &silent_path.join("paper-web-application/web-diff-lan-10k.csv"),
        "X", "Y",
        TimeUnit::Nanoseconds,
        None,
        100.0,
        None,
    );

    // 3. mbedTLS
    println!("\n{}", "=".repeat(70));
    println!("3. mbedTLS Bleichenbacher");
    println!("   Original: 500k BASELINE, then 500k MODIFIED (BLOCKED)");
    println!("{}", "=".repeat(70));

    analyze_with_interleaving(
        &silent_path.join("paper-mbedtls/Wrong_second_byte_(0x02_set_to_0x17)vsCorrectly_formatted_PKCS#1_PMS_message.csv"),
        "BASELINE", "MODIFIED",
        TimeUnit::Cycles,
        Some(3.0),
        1.67,
        Some(50_000),
    );

    println!("\n=== Analysis Complete ===");
}

fn analyze_with_interleaving(
    path: &Path,
    baseline_label: &str,
    test_label: &str,
    unit: TimeUnit,
    cpu_freq_ghz: Option<f64>,
    theta_ns: f64,
    max_samples: Option<usize>,
) {
    println!("\nLoading: {}", path.file_name().unwrap().to_string_lossy());

    // Load data
    let data = load_two_column_csv(path, true, baseline_label, test_label);
    let data = match data {
        Ok(d) => d,
        Err(e) => {
            println!("ERROR: {}", e);
            return;
        }
    };

    let ns_per_unit = unit.ns_per_unit(cpu_freq_ghz);
    let (mut baseline_ns, mut test_ns) = data.to_nanoseconds(ns_per_unit);

    // Truncate if needed
    if let Some(max) = max_samples {
        if baseline_ns.len() > max {
            baseline_ns.truncate(max);
            test_ns.truncate(max);
            println!("Truncated to {} samples per class", max);
        }
    }

    let n = baseline_ns.len().min(test_ns.len());
    println!("Samples: {} per class", n);

    // Estimate block length from baseline data
    let est_block = estimate_block_length(&baseline_ns);
    println!("Estimated decorrelation block length: {}", est_block);

    // Use a block size that's reasonable for interleaving
    // We want enough blocks to get good mixing, but large enough to preserve temporal structure
    let block_size = est_block.max(50).min(n / 20);
    let num_blocks = n / block_size;
    println!("Using block size: {} ({} blocks)", block_size, num_blocks);

    // ===== ORIGINAL (BLOCKED) ANALYSIS =====
    println!("\n--- ORIGINAL (blocked acquisition) ---");

    let decile_diffs = compute_decile_diffs(&baseline_ns, &test_ns);
    println!("Raw quantile diffs: [{:+.1}, {:+.1}, {:+.1}, ..., {:+.1}]",
        decile_diffs[0], decile_diffs[1], decile_diffs[2], decile_diffs[8]);

    let config = SinglePassConfig {
        theta_ns,
        pass_threshold: 0.05,
        fail_threshold: 0.95,
        bootstrap_iterations: 2000,
        timer_resolution_ns: 1.0,
        seed: 0xDEADBEEF,
        max_variance_ratio: 0.95,
    };

    let result_original = analyze_single_pass(&baseline_ns, &test_ns, &config);

    println!("P(leak > {:.1}ns): {:.1}%", theta_ns, result_original.leak_probability * 100.0);
    println!("Quality: {:?}", result_original.quality);
    print_outcome(&result_original.outcome, theta_ns);

    // ===== SIMULATED INTERLEAVING =====
    println!("\n--- SIMULATED INTERLEAVING (block size = {}) ---", block_size);

    let (interleaved_baseline, interleaved_test) =
        simulate_interleaving(&baseline_ns, &test_ns, block_size);

    println!("Interleaved samples: {} per class", interleaved_baseline.len());

    let decile_diffs_int = compute_decile_diffs(&interleaved_baseline, &interleaved_test);
    println!("Raw quantile diffs: [{:+.1}, {:+.1}, {:+.1}, ..., {:+.1}]",
        decile_diffs_int[0], decile_diffs_int[1], decile_diffs_int[2], decile_diffs_int[8]);

    let result_interleaved = analyze_single_pass(&interleaved_baseline, &interleaved_test, &config);

    println!("P(leak > {:.1}ns): {:.1}%", theta_ns, result_interleaved.leak_probability * 100.0);
    println!("Quality: {:?}", result_interleaved.quality);
    print_outcome(&result_interleaved.outcome, theta_ns);

    // ===== COMPARISON =====
    println!("\n--- COMPARISON ---");
    println!("                    Original    Interleaved");
    println!("  P(leak):          {:>6.1}%     {:>6.1}%",
        result_original.leak_probability * 100.0,
        result_interleaved.leak_probability * 100.0);
    println!("  Shift estimate:   {:>6.1}ns    {:>6.1}ns",
        result_original.effect_estimate.shift_ns,
        result_interleaved.effect_estimate.shift_ns);
}

fn print_outcome(outcome: &Outcome, theta_ns: f64) {
    match outcome {
        Outcome::Pass { .. } => {
            println!("RESULT: PASS (no leak > {:.1}ns)", theta_ns);
        }
        Outcome::Fail { exploitability, .. } => {
            println!("RESULT: FAIL - {:?}", exploitability);
        }
        Outcome::Inconclusive { reason, .. } => {
            println!("RESULT: INCONCLUSIVE - {}", reason);
        }
        _ => {
            println!("RESULT: {:?}", outcome);
        }
    }
}
