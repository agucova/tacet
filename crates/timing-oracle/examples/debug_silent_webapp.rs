//! Debug analysis of SILENT web app dataset
//!
//! Run with: cargo run --example debug_silent_webapp

use std::path::Path;
use timing_oracle::data::{load_silent_csv, TimeUnit};
use timing_oracle_core::adaptive::{calibrate_prior_scale, compute_prior_cov_9d};
use timing_oracle_core::analysis::compute_bayes_factor;
use timing_oracle_core::statistics::{bootstrap_difference_covariance_discrete, compute_deciles_inplace};

fn main() {
    println!("=== Debug: SILENT Web App Dataset ===\n");

    let path = Path::new("/Users/agucova/repos/SILENT/examples/paper-web-application/web-diff-lan-10k.csv");

    // Load data
    let data = load_silent_csv(path).expect("Failed to load CSV");
    println!("Loaded {} baseline, {} test samples",
        data.baseline_samples.len(), data.test_samples.len());

    // Convert to nanoseconds (data is already in ns)
    let ns_per_unit = TimeUnit::Nanoseconds.ns_per_unit(None);
    let (baseline_ns, test_ns) = data.to_nanoseconds(ns_per_unit);
    let n = baseline_ns.len();

    // Basic statistics
    let baseline_mean: f64 = baseline_ns.iter().sum::<f64>() / n as f64;
    let test_mean: f64 = test_ns.iter().sum::<f64>() / n as f64;
    let baseline_std: f64 = (baseline_ns.iter().map(|x| (x - baseline_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let test_std: f64 = (test_ns.iter().map(|x| (x - test_mean).powi(2)).sum::<f64>() / n as f64).sqrt();

    println!("\n--- Basic Statistics ---");
    println!("Baseline: mean={:.1}ns, std={:.1}ns", baseline_mean, baseline_std);
    println!("Test:     mean={:.1}ns, std={:.1}ns", test_mean, test_std);
    println!("Diff:     {:.1}ns", test_mean - baseline_mean);
    println!("Cohen's d: {:.3}", (test_mean - baseline_mean) / ((baseline_std + test_std) / 2.0));

    // Compute quantiles
    let mut b_sorted = baseline_ns.clone();
    let mut t_sorted = test_ns.clone();
    let q_baseline = compute_deciles_inplace(&mut b_sorted);
    let q_test = compute_deciles_inplace(&mut t_sorted);
    let delta_hat = q_baseline - q_test;

    println!("\n--- Quantile Analysis ---");
    println!("Deciles (10%, 20%, ..., 90%):");
    println!("  Baseline: [{:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}]",
        q_baseline[0], q_baseline[1], q_baseline[2], q_baseline[3], q_baseline[4],
        q_baseline[5], q_baseline[6], q_baseline[7], q_baseline[8]);
    println!("  Test:     [{:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}]",
        q_test[0], q_test[1], q_test[2], q_test[3], q_test[4],
        q_test[5], q_test[6], q_test[7], q_test[8]);
    println!("  Delta:    [{:+.0}, {:+.0}, {:+.0}, {:+.0}, {:+.0}, {:+.0}, {:+.0}, {:+.0}, {:+.0}]",
        delta_hat[0], delta_hat[1], delta_hat[2], delta_hat[3], delta_hat[4],
        delta_hat[5], delta_hat[6], delta_hat[7], delta_hat[8]);

    // Bootstrap covariance
    println!("\n--- Bootstrap Covariance ---");
    let cov_estimate = bootstrap_difference_covariance_discrete(&baseline_ns, &test_ns, 2000, 0xDEADBEEF);
    let sigma = cov_estimate.matrix;

    println!("Diagonal (variances): [{:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}]",
        sigma[(0,0)], sigma[(1,1)], sigma[(2,2)], sigma[(3,3)], sigma[(4,4)],
        sigma[(5,5)], sigma[(6,6)], sigma[(7,7)], sigma[(8,8)]);
    println!("Trace: {:.0}", sigma.trace());
    println!("Mean variance: {:.0}", sigma.trace() / 9.0);

    // Standard errors for each quantile difference
    let ses: Vec<f64> = (0..9).map(|i| sigma[(i,i)].sqrt()).collect();
    println!("SEs:    [{:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}, {:.0}]",
        ses[0], ses[1], ses[2], ses[3], ses[4], ses[5], ses[6], ses[7], ses[8]);

    // Z-scores (delta / SE)
    let zscores: Vec<f64> = (0..9).map(|i| delta_hat[i] / ses[i]).collect();
    println!("Z-scores: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
        zscores[0], zscores[1], zscores[2], zscores[3], zscores[4],
        zscores[5], zscores[6], zscores[7], zscores[8]);

    // Detect discrete mode (< 10% unique values)
    let discrete_mode = {
        let unique: std::collections::HashSet<i64> = baseline_ns.iter().map(|&v| v as i64).collect();
        (unique.len() as f64 / n as f64) < 0.10
    };

    // Test with multiple thresholds
    println!("\n--- Bayesian Analysis at Different Thresholds ---");

    for theta in [5.0, 100.0, 1000.0, 5000.0, 10000.0] {
        let sigma_rate = sigma * (n as f64);
        let prior_scale = calibrate_prior_scale(&sigma_rate, theta, n, discrete_mode, 0xDEADBEEF);
        let prior_cov = compute_prior_cov_9d(&sigma_rate, prior_scale, discrete_mode);

        let bayes = compute_bayes_factor(&delta_hat, &sigma, &prior_cov, theta, Some(0xDEADBEEF));

        println!("\ntheta = {}ns:", theta);
        println!("  Prior scale: {:.2}", prior_scale);
        println!("  Leak probability: {:.1}%", bayes.leak_probability * 100.0);
        println!("  Beta (shift, tail): ({:.2}, {:.2})", bayes.beta_proj[0], bayes.beta_proj[1]);
        println!("  Effect magnitude CI: [{:.1}, {:.1}]",
            bayes.effect_magnitude_ci.0, bayes.effect_magnitude_ci.1);
        println!("  Projection mismatch Q: {:.4}", bayes.projection_mismatch_q);
    }

    // Check for potential issues
    println!("\n--- Diagnostic Checks ---");

    // 1. Uniqueness ratio (discrete mode detection)
    let unique_baseline: std::collections::HashSet<i64> = baseline_ns.iter().map(|&v| v as i64).collect();
    let unique_test: std::collections::HashSet<i64> = test_ns.iter().map(|&v| v as i64).collect();
    let uniqueness_baseline = unique_baseline.len() as f64 / n as f64;
    let uniqueness_test = unique_test.len() as f64 / n as f64;
    println!("Uniqueness ratio: baseline={:.1}%, test={:.1}%",
        uniqueness_baseline * 100.0, uniqueness_test * 100.0);

    // 2. Check for extreme outliers
    let baseline_median = q_baseline[4];
    let test_median = q_test[4];
    let baseline_iqr = q_baseline[6] - q_baseline[2];
    let test_iqr = q_test[6] - q_test[2];
    let baseline_outliers = baseline_ns.iter().filter(|&&x|
        x < baseline_median - 3.0 * baseline_iqr || x > baseline_median + 3.0 * baseline_iqr).count();
    let test_outliers = test_ns.iter().filter(|&&x|
        x < test_median - 3.0 * test_iqr || x > test_median + 3.0 * test_iqr).count();
    println!("Outliers (>3 IQR): baseline={} ({:.1}%), test={} ({:.1}%)",
        baseline_outliers, baseline_outliers as f64 / n as f64 * 100.0,
        test_outliers, test_outliers as f64 / n as f64 * 100.0);

    // 3. Signal to noise ratio
    let effect_magnitude = delta_hat.iter().map(|x| x.abs()).sum::<f64>() / 9.0;
    let noise_magnitude = ses.iter().sum::<f64>() / 9.0;
    println!("Mean |effect|: {:.1}ns, Mean SE: {:.1}ns, SNR: {:.2}",
        effect_magnitude, noise_magnitude, effect_magnitude / noise_magnitude);

    println!("\n=== Done ===");
}
