//! Diagnostic checks for SILENT web app dataset
//!
//! Three checks suggested by spec author:
//! 1. Flat-ish prior sanity check - verify P(m(δ)>θ) ≈ 1 with wide prior
//! 2. Check Σ_n matches displayed SEs
//! 3. Check effective sample size / dependence
//!
//! Run with: cargo run --example diagnostic_webapp

use std::path::Path;
use timing_oracle::data::{load_silent_csv, TimeUnit};
use timing_oracle_core::adaptive::{calibrate_prior_scale, compute_prior_cov_9d};
use timing_oracle_core::analysis::compute_bayes_factor;
use timing_oracle_core::statistics::{
    bootstrap_difference_covariance_discrete, compute_deciles_inplace,
};

fn main() {
    println!("=== Diagnostic Checks: SILENT Web App Dataset ===\n");

    let path =
        Path::new("/Users/agucova/repos/SILENT/examples/paper-web-application/web-diff-lan-10k.csv");

    // Load data
    let data = load_silent_csv(path).expect("Failed to load CSV");
    println!(
        "Loaded {} baseline, {} test samples",
        data.baseline_samples.len(),
        data.test_samples.len()
    );

    // Convert to nanoseconds
    let ns_per_unit = TimeUnit::Nanoseconds.ns_per_unit(None);
    let (baseline_ns, test_ns) = data.to_nanoseconds(ns_per_unit);
    let n = baseline_ns.len();

    // Compute quantile differences
    let mut b_sorted = baseline_ns.clone();
    let mut t_sorted = test_ns.clone();
    let q_baseline = compute_deciles_inplace(&mut b_sorted);
    let q_test = compute_deciles_inplace(&mut t_sorted);
    let delta_hat = q_baseline - q_test;

    println!("\n--- Raw Data Summary ---");
    println!(
        "Delta (baseline - test): [{:+.0}, {:+.0}, {:+.0}, {:+.0}, {:+.0}, {:+.0}, {:+.0}, {:+.0}, {:+.0}]",
        delta_hat[0], delta_hat[1], delta_hat[2], delta_hat[3], delta_hat[4],
        delta_hat[5], delta_hat[6], delta_hat[7], delta_hat[8]
    );

    // Bootstrap covariance
    let cov_estimate =
        bootstrap_difference_covariance_discrete(&baseline_ns, &test_ns, 2000, 0xDEADBEEF);
    let sigma_n = cov_estimate.matrix; // This is Σ_n (covariance of delta_hat)
    let sigma_rate = sigma_n * (n as f64); // Σ_rate = Σ_n * n

    // Detect discrete mode (< 10% unique values)
    let discrete_mode = {
        let unique: std::collections::HashSet<i64> = baseline_ns.iter().map(|&v| v as i64).collect();
        (unique.len() as f64 / n as f64) < 0.10
    };

    println!("\n========================================");
    println!("CHECK 1: Σ_n matches displayed SEs?");
    println!("========================================");
    println!("\nComparing sqrt(Σ_n[k,k]) vs reported SE[k]:");
    println!("k   | sqrt(Σ_n[k,k]) | SE from diag | Match?");
    println!("----|----------------|--------------|-------");

    let mut se_match = true;
    for k in 0..9 {
        let sigma_n_kk = sigma_n[(k, k)];
        let se_from_sigma_n = sigma_n_kk.sqrt();
        let se_reported = sigma_n_kk.sqrt(); // Same source, should match by definition

        // The question is: are we using the SAME covariance for SE display and inference?
        let matches = (se_from_sigma_n - se_reported).abs() < 1e-6;
        if !matches {
            se_match = false;
        }
        println!(
            "{:3} | {:14.2} | {:12.2} | {}",
            k,
            se_from_sigma_n,
            se_reported,
            if matches { "✓" } else { "✗" }
        );
    }

    println!("\nSE check: {}", if se_match { "PASS" } else { "FAIL" });
    println!("\nNote: Both come from same bootstrap covariance matrix.");
    println!("The real question is whether inference uses this same Σ_n.");

    // Check what compute_bayes_factor receives
    println!("\nVerifying compute_bayes_factor inputs:");
    println!("  delta_hat = {:?}", delta_hat.as_slice());
    println!("  sigma (Σ_n) diagonal: [{:.0}, {:.0}, {:.0}, ..., {:.0}]",
        sigma_n[(0,0)], sigma_n[(1,1)], sigma_n[(2,2)], sigma_n[(8,8)]);

    println!("\n========================================");
    println!("CHECK 2: Flat prior sanity check");
    println!("========================================");
    println!("\nWith a VERY wide prior, P(max|δ| > θ) should be ~1 for θ=100ns");
    println!("(Because the observed effect is ~13,000ns)\n");

    // Test with increasingly wide priors
    for prior_scale_mult in [1.0, 10.0, 100.0, 1000.0] {
        let theta = 100.0;

        // Manually create a wide prior
        // Normal calibration gives σ_prior ~ theta, so multiply by our factor
        let base_sigma_prior = calibrate_prior_scale(&sigma_rate, theta, n, discrete_mode, 0xDEADBEEF);
        let wide_sigma_prior = base_sigma_prior * prior_scale_mult;
        let wide_prior_cov = compute_prior_cov_9d(&sigma_rate, wide_sigma_prior, discrete_mode);

        let bayes = compute_bayes_factor(&delta_hat, &sigma_n, &wide_prior_cov, theta, Some(0xDEADBEEF));

        println!(
            "Prior scale = {:.1}x base ({:.1}ns): P(leak) = {:.1}%, beta = ({:.2}, {:.2})",
            prior_scale_mult,
            wide_sigma_prior,
            bayes.leak_probability * 100.0,
            bayes.beta_proj[0],
            bayes.beta_proj[1]
        );
    }

    // Also try with "no prior" - just use delta_hat directly
    println!("\n--- Direct frequentist check (no Bayesian shrinkage) ---");
    let max_effect = delta_hat.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let max_se = (0..9).map(|i| sigma_n[(i,i)].sqrt()).fold(0.0_f64, f64::max);
    println!("Max |δ_hat|: {:.1}ns", max_effect);
    println!("Max SE: {:.1}ns", max_se);
    println!("If we just ask 'is max|δ_hat| > 100ns?': {}", if max_effect > 100.0 { "YES" } else { "NO" });

    // Compute frequentist confidence interval
    let z_975 = 1.96;
    for k in 0..9 {
        let se = sigma_n[(k,k)].sqrt();
        let ci_lo = delta_hat[k] - z_975 * se;
        let ci_hi = delta_hat[k] + z_975 * se;
        if k == 0 || k == 4 || k == 8 {
            println!("  δ_{} = {:.0} ± {:.0} (95% CI: [{:.0}, {:.0}])",
                (k + 1) * 10, delta_hat[k], z_975 * se, ci_lo, ci_hi);
        }
    }

    println!("\n========================================");
    println!("CHECK 3: Effective sample size / dependence");
    println!("========================================");

    // Block length from bootstrap
    println!("\nBlock length from bootstrap: {}", cov_estimate.block_size);

    // Compute lag-1 autocorrelation for both series
    let acf1_baseline = compute_lag1_autocorr(&baseline_ns);
    let acf1_test = compute_lag1_autocorr(&test_ns);

    println!("Lag-1 autocorrelation:");
    println!("  Baseline: {:.3}", acf1_baseline);
    println!("  Test:     {:.3}", acf1_test);

    // Effective sample size approximation: n_eff ≈ n * (1 - ρ) / (1 + ρ)
    let ess_baseline = n as f64 * (1.0 - acf1_baseline) / (1.0 + acf1_baseline);
    let ess_test = n as f64 * (1.0 - acf1_test) / (1.0 + acf1_test);

    println!("Effective sample size (approx):");
    println!("  Baseline: {:.0} (of {})", ess_baseline, n);
    println!("  Test:     {:.0} (of {})", ess_test, n);

    // Uniqueness ratio (discrete mode indicator)
    let unique_baseline: std::collections::HashSet<i64> =
        baseline_ns.iter().map(|&v| v as i64).collect();
    let unique_test: std::collections::HashSet<i64> = test_ns.iter().map(|&v| v as i64).collect();
    let uniqueness_baseline = unique_baseline.len() as f64 / n as f64;
    let uniqueness_test = unique_test.len() as f64 / n as f64;

    println!("Uniqueness ratio (discrete if <10%):");
    println!(
        "  Baseline: {:.1}% ({} unique values)",
        uniqueness_baseline * 100.0,
        unique_baseline.len()
    );
    println!(
        "  Test:     {:.1}% ({} unique values)",
        uniqueness_test * 100.0,
        unique_test.len()
    );

    println!("\n========================================");
    println!("CHECK 4: Posterior mean computation trace");
    println!("========================================");

    // Let's trace through the Bayesian computation manually
    let theta = 100.0;
    let sigma_prior = calibrate_prior_scale(&sigma_rate, theta, n, discrete_mode, 0xDEADBEEF);
    let prior_cov = compute_prior_cov_9d(&sigma_rate, sigma_prior, discrete_mode);

    println!("\nFor θ = 100ns:");
    println!("  Calibrated σ_prior: {:.2}ns", sigma_prior);

    // Prior precision
    println!("\n  Prior covariance Λ₀ diagonal:");
    println!("    [{:.1}, {:.1}, {:.1}, ..., {:.1}]",
        prior_cov[(0,0)], prior_cov[(1,1)], prior_cov[(2,2)], prior_cov[(8,8)]);

    // Data precision
    println!("  Data covariance Σ_n diagonal:");
    println!("    [{:.0}, {:.0}, {:.0}, ..., {:.0}]",
        sigma_n[(0,0)], sigma_n[(1,1)], sigma_n[(2,2)], sigma_n[(8,8)]);

    // Compare magnitudes
    let prior_precision_0 = 1.0 / prior_cov[(0,0)];
    let data_precision_0 = 1.0 / sigma_n[(0,0)];
    println!("\n  Precision comparison (for quantile 0):");
    println!("    Prior precision: {:.6}", prior_precision_0);
    println!("    Data precision:  {:.6}", data_precision_0);
    println!("    Ratio (prior/data): {:.2}", prior_precision_0 / data_precision_0);

    if prior_precision_0 > data_precision_0 * 10.0 {
        println!("    ⚠️  Prior precision >> Data precision: prior dominates!");
    } else if data_precision_0 > prior_precision_0 * 10.0 {
        println!("    ✓ Data precision >> Prior precision: data dominates");
    } else {
        println!("    ~ Similar magnitudes: mixed influence");
    }

    // Run actual Bayes computation
    let bayes = compute_bayes_factor(&delta_hat, &sigma_n, &prior_cov, theta, Some(0xDEADBEEF));

    println!("\n  Bayes factor result:");
    println!("    Leak probability: {:.1}%", bayes.leak_probability * 100.0);
    println!("    Beta (shift, tail): ({:.2}, {:.2})", bayes.beta_proj[0], bayes.beta_proj[1]);
    println!("    Effect CI: [{:.1}, {:.1}]ns", bayes.effect_magnitude_ci.0, bayes.effect_magnitude_ci.1);

    println!("\n=== Diagnostics Complete ===");
}

fn compute_lag1_autocorr(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }

    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-12 {
        return 0.0;
    }

    let cov: f64 = data
        .windows(2)
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum::<f64>()
        / (n - 1) as f64;

    cov / var
}
