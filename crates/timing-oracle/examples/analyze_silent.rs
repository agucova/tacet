//! Analyze SILENT paper datasets with timing-oracle.
//!
//! This example loads the CSV files from the SILENT paper and runs them through
//! timing-oracle's single-pass analysis to replicate their detection scenarios.
//!
//! Run with: cargo run --example analyze_silent

use std::path::Path;
use timing_oracle::adaptive::single_pass::{analyze_single_pass, SinglePassConfig};
use timing_oracle::data::{load_silent_csv, load_two_column_csv, TimeUnit};
use timing_oracle::Outcome;

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
    println!("=== SILENT Paper Dataset Analysis ===\n");
    println!("Testing with multiple thresholds to see posterior behavior\n");

    // Path to SILENT repo
    let silent_path = Path::new("/Users/agucova/repos/SILENT/examples");

    // 1. KyberSlash dataset
    println!("1. KyberSlash (timing_measurements_1_comparison_-72_72.csv)");
    println!("   Known vulnerability: Timing leak in Kyber decapsulation\n");

    // Test with PostQuantumSentinel threshold (3.3ns)
    println!("   --- PostQuantumSentinel (theta = 3.3ns) ---");
    analyze_dataset(
        &silent_path.join("paper-kyberslash/timing_measurements_1_comparison_-72_72.csv"),
        DatasetType::Silent,
        TimeUnit::Cycles,
        Some(3.0),
        3.3, // PostQuantumSentinel threshold
    );
    println!();

    // 2. Web Application dataset
    println!("\n2. Web Application (web-diff-lan-10k.csv)");
    println!("   Known vulnerability: Timing difference in web app response\n");

    // Test with SILENT's delta=5 threshold
    println!("   --- SILENT's threshold (theta = 5ns) ---");
    analyze_dataset(
        &silent_path.join("paper-web-application/web-diff-lan-10k.csv"),
        DatasetType::Silent,
        TimeUnit::Nanoseconds,
        None,
        5.0,
    );
    println!();

    // Also test with AdjacentNetwork threshold (100ns)
    println!("   --- AdjacentNetwork (theta = 100ns) ---");
    analyze_dataset(
        &silent_path.join("paper-web-application/web-diff-lan-10k.csv"),
        DatasetType::Silent,
        TimeUnit::Nanoseconds,
        None,
        100.0,
    );
    println!();

    // 3. mbedTLS PKCS#1 dataset (use 50k subset for speed)
    println!("\n3. mbedTLS Bleichenbacher (PKCS#1 padding oracle)");
    println!("   Known vulnerability: Timing difference based on padding validity\n");

    // Test with AdjacentNetwork threshold (100ns)
    println!("   --- AdjacentNetwork (theta = 100ns) ---");
    analyze_dataset_subset(
        &silent_path.join("paper-mbedtls/Wrong_second_byte_(0x02_set_to_0x17)vsCorrectly_formatted_PKCS#1_PMS_message.csv"),
        DatasetType::CustomLabels("BASELINE", "MODIFIED"),
        TimeUnit::Cycles,
        Some(3.0),
        100.0,
        50_000,
    );
    println!();

    println!("\n=== Analysis Complete ===");
}

enum DatasetType<'a> {
    Silent,
    CustomLabels(&'a str, &'a str),
}

fn analyze_dataset(
    path: &Path,
    dataset_type: DatasetType,
    unit: TimeUnit,
    cpu_freq_ghz: Option<f64>,
    theta_ns: f64,
) {
    analyze_dataset_subset(path, dataset_type, unit, cpu_freq_ghz, theta_ns, usize::MAX);
}

fn analyze_dataset_subset(
    path: &Path,
    dataset_type: DatasetType,
    unit: TimeUnit,
    cpu_freq_ghz: Option<f64>,
    theta_ns: f64,
    max_samples: usize,
) {
    println!("   Loading: {}", path.file_name().unwrap().to_string_lossy());

    // Load the data
    let data = match dataset_type {
        DatasetType::Silent => load_silent_csv(path),
        DatasetType::CustomLabels(baseline, test) => {
            load_two_column_csv(path, true, baseline, test)
        }
    };

    let data = match data {
        Ok(d) => d,
        Err(e) => {
            println!("   ERROR loading: {}", e);
            return;
        }
    };

    println!(
        "   Samples: {} baseline, {} test",
        data.baseline_samples.len(),
        data.test_samples.len()
    );

    // Convert to nanoseconds
    let ns_per_unit = unit.ns_per_unit(cpu_freq_ghz);
    let (mut baseline_ns, mut test_ns) = data.to_nanoseconds(ns_per_unit);

    // Truncate to max_samples if specified
    if max_samples < baseline_ns.len() {
        baseline_ns.truncate(max_samples);
        test_ns.truncate(max_samples);
        println!("   (Truncated to {} samples per class)", max_samples);
    }

    // Compute basic statistics
    let baseline_mean: f64 = baseline_ns.iter().sum::<f64>() / baseline_ns.len() as f64;
    let test_mean: f64 = test_ns.iter().sum::<f64>() / test_ns.len() as f64;
    let diff_ns = test_mean - baseline_mean;

    println!(
        "   Mean baseline: {:.2} ns, Mean test: {:.2} ns",
        baseline_mean, test_mean
    );
    println!("   Raw mean difference: {:.2} ns", diff_ns);

    // Compute and display raw quantile differences
    let decile_diffs = compute_decile_diffs(&baseline_ns, &test_ns);
    println!("   Raw quantile differences (baseline - test):");
    println!(
        "     10%: {:+.1}ns  50%: {:+.1}ns  90%: {:+.1}ns",
        decile_diffs[0], decile_diffs[4], decile_diffs[8]
    );
    println!(
        "     All: [{:+.1}, {:+.1}, {:+.1}, {:+.1}, {:+.1}, {:+.1}, {:+.1}, {:+.1}, {:+.1}]",
        decile_diffs[0], decile_diffs[1], decile_diffs[2], decile_diffs[3], decile_diffs[4],
        decile_diffs[5], decile_diffs[6], decile_diffs[7], decile_diffs[8]
    );

    // Run single-pass analysis
    let config = SinglePassConfig {
        theta_ns,
        pass_threshold: 0.05,
        fail_threshold: 0.95,
        bootstrap_iterations: 2000,
        seed: 0xDEADBEEF,
    };

    let result = analyze_single_pass(&baseline_ns, &test_ns, &config);

    println!("   Analysis time: {:?}", result.analysis_time);
    println!(
        "   Leak probability P(effect > {}ns): {:.1}%",
        theta_ns, result.leak_probability * 100.0
    );
    println!("   ---");
    println!("   Effect decomposition:");
    println!("     Shift (uniform):  {:.2} ns", result.effect_estimate.shift_ns);
    println!("     Tail (asymmetric): {:.2} ns", result.effect_estimate.tail_ns);
    println!(
        "     95% CI: [{:.2}, {:.2}] ns",
        result.effect_estimate.credible_interval_ns.0,
        result.effect_estimate.credible_interval_ns.1
    );
    println!("     Pattern: {:?}", result.effect_estimate.pattern);
    println!("   Quality: {:?}", result.quality);

    match &result.outcome {
        Outcome::Pass { .. } => {
            println!("   RESULT: PASS (no timing leak detected above {}ns)", theta_ns);
        }
        Outcome::Fail {
            exploitability, ..
        } => {
            println!(
                "   RESULT: FAIL - Timing leak detected! Exploitability: {:?}",
                exploitability
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            println!("   RESULT: INCONCLUSIVE - {}", reason);
        }
        _ => {
            println!("   RESULT: {:?}", result.outcome);
        }
    }
}
