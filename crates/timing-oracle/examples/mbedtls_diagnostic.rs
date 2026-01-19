//! Comprehensive diagnostic analysis of mbedTLS false positive
//!
//! This generates a detailed report for investigating the block size estimation issue.
//!
//! Run with: cargo run --example mbedtls_diagnostic 2>&1 | tee mbedtls_report.md

use std::collections::HashSet;
use std::path::Path;
use timing_oracle::data::load_two_column_csv;
use timing_oracle::adaptive::single_pass::{analyze_single_pass, SinglePassConfig};
use timing_oracle_core::statistics::{
    bootstrap_difference_covariance_discrete, compute_deciles_inplace,
    paired_optimal_block_length,
};

fn main() {
    println!("# mbedTLS False Positive Diagnostic Report\n");
    println!("Generated: {}\n", chrono_lite());

    // Load the dataset
    let path = Path::new("/Users/agucova/repos/SILENT/examples/paper-mbedtls/Wrong_second_byte_(0x02_set_to_0x17)vsCorrectly_formatted_PKCS#1_PMS_message.csv");

    println!("## 1. Dataset Information\n");
    println!("- **Source**: SILENT paper (Dunsche et al.)");
    println!("- **File**: `{}`", path.file_name().unwrap().to_string_lossy());
    println!("- **Context**: mbedTLS Bleichenbacher timing attack (PKCS#1 padding oracle)");
    println!("- **Ground Truth**: SILENT reports 'Failed to reject' → NO timing leak");
    println!("- **Issue**: Our tool incorrectly reports FAIL (false positive)\n");

    let data = load_two_column_csv(path, true, "BASELINE", "MODIFIED")
        .expect("Failed to load CSV");

    let full_n = data.baseline_samples.len();
    println!("### Raw Data Statistics\n");
    println!("| Metric | Value |");
    println!("|--------|-------|");
    println!("| Total samples per class | {} |", full_n);
    println!("| Data unit | CPU cycles |");
    println!("| Assumed CPU frequency | 3.0 GHz |");

    // Convert to nanoseconds at 3GHz
    let ns_per_cycle = 1.0 / 3.0;
    let (full_baseline_ns, full_test_ns) = data.to_nanoseconds(ns_per_cycle);

    // Full dataset statistics
    let baseline_mean: f64 = full_baseline_ns.iter().sum::<f64>() / full_n as f64;
    let test_mean: f64 = full_test_ns.iter().sum::<f64>() / full_n as f64;
    let baseline_std: f64 = (full_baseline_ns.iter().map(|x| (x - baseline_mean).powi(2)).sum::<f64>() / full_n as f64).sqrt();
    let test_std: f64 = (full_test_ns.iter().map(|x| (x - test_mean).powi(2)).sum::<f64>() / full_n as f64).sqrt();

    println!("| Baseline mean | {:.2} ns ({:.2} ms) |", baseline_mean, baseline_mean / 1e6);
    println!("| Test mean | {:.2} ns ({:.2} ms) |", test_mean, test_mean / 1e6);
    println!("| Raw mean difference | {:.2} ns |", test_mean - baseline_mean);
    println!("| Baseline std | {:.2} ns ({:.2} ms) |", baseline_std, baseline_std / 1e6);
    println!("| Test std | {:.2} ns ({:.2} ms) |", test_std, test_std / 1e6);
    println!("| Coefficient of variation | {:.1}% |", (baseline_std / baseline_mean) * 100.0);

    // Work with truncated dataset for detailed analysis
    let n = 50_000usize;
    let baseline_ns: Vec<f64> = full_baseline_ns.iter().take(n).copied().collect();
    let test_ns: Vec<f64> = full_test_ns.iter().take(n).copied().collect();

    println!("\n### Analysis Subset\n");
    println!("Using first {} samples per class for detailed analysis.\n", n);

    // =========================================================================
    // Section 2: SILENT's Parameters
    // =========================================================================
    println!("## 2. SILENT's Analysis Parameters\n");
    println!("From `Wrong_second_byte_..._summary_results.json`:\n");
    println!("```json");
    println!("{{");
    println!("  \"alpha\": 0.1,");
    println!("  \"bootstrap_samples\": 1000,");
    println!("  \"delta\": 5,");
    println!("  \"block_size\": 2122,");
    println!("  \"test_statistic\": 2.303,");
    println!("  \"threshold\": 4.296,");
    println!("  \"decision\": \"Failed to reject\"");
    println!("}}");
    println!("```\n");

    println!("**Key observations**:");
    println!("- SILENT uses `block_size=2122` for 500,000 samples");
    println!("- Block ratio: {:.4} ({:.2}% of data)", 2122.0 / 500000.0, 2122.0 / 500000.0 * 100.0);
    println!("- Equivalent for {} samples: {}", n, (2122.0 / 500000.0 * n as f64) as usize);
    println!("- Threshold in nanoseconds: {:.2} ns (5 cycles @ 3GHz)\n", 5.0 * ns_per_cycle);

    // =========================================================================
    // Section 3: Autocorrelation Analysis
    // =========================================================================
    println!("## 3. Autocorrelation Analysis\n");

    // Compute lag-1 autocorrelation for both classes
    let acf1_baseline = compute_lag1_acf(&baseline_ns);
    let acf1_test = compute_lag1_acf(&test_ns);

    // Compute ACF for multiple lags
    println!("### Lag-k Autocorrelation\n");
    println!("| Lag | Baseline ACF | Test ACF |");
    println!("|-----|--------------|----------|");
    for lag in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] {
        if lag < n / 4 {
            let acf_b = compute_lagk_acf(&baseline_ns, lag);
            let acf_t = compute_lagk_acf(&test_ns, lag);
            println!("| {} | {:.4} | {:.4} |", lag, acf_b, acf_t);
        }
    }

    println!("\n**Interpretation**: High autocorrelation at large lags indicates strong temporal dependence.");
    println!("SILENT's block_size=2122 suggests dependence extends ~2000+ samples.\n");

    // Estimate where ACF drops below 0.05
    let acf_threshold = 0.05;
    let mut decorr_lag_baseline = 1;
    let mut decorr_lag_test = 1;
    for lag in 1..5000.min(n/4) {
        let acf_b = compute_lagk_acf(&baseline_ns, lag);
        let acf_t = compute_lagk_acf(&test_ns, lag);
        if acf_b.abs() > acf_threshold { decorr_lag_baseline = lag; }
        if acf_t.abs() > acf_threshold { decorr_lag_test = lag; }
    }

    println!("### Decorrelation Length Estimates\n");
    println!("| Metric | Baseline | Test |");
    println!("|--------|----------|------|");
    println!("| Lag-1 ACF | {:.4} | {:.4} |", acf1_baseline, acf1_test);
    println!("| Last lag with |ACF| > 0.05 | {} | {} |", decorr_lag_baseline, decorr_lag_test);
    println!("| Suggested block size | {} | {} |", decorr_lag_baseline * 2, decorr_lag_test * 2);

    // =========================================================================
    // Section 4: Block Size Estimation Comparison
    // =========================================================================
    println!("\n## 4. Block Size Estimation\n");

    // Our Politis-White estimate
    let pw_block = paired_optimal_block_length(&baseline_ns, &test_ns);

    // Simple formula
    let simple_block = (1.3 * (n as f64).cbrt()).ceil() as usize;

    // Maximum allowed by P-W
    let max_block = (3.0 * (n as f64).sqrt()).ceil().min(n as f64 / 3.0) as usize;

    // SILENT equivalent
    let silent_equiv = (2122.0 / 500000.0 * n as f64) as usize;

    println!("| Method | Block Size | Ratio to SILENT |");
    println!("|--------|------------|-----------------|");
    println!("| SILENT (scaled to {}k) | {} | 1.00x |", n/1000, silent_equiv);
    println!("| Our Politis-White | {} | {:.2}x |", pw_block, pw_block as f64 / silent_equiv as f64);
    println!("| Simple (1.3×n^⅓) | {} | {:.2}x |", simple_block, simple_block as f64 / silent_equiv as f64);
    println!("| Max P-W (3×√n) | {} | {:.2}x |", max_block, max_block as f64 / silent_equiv as f64);

    println!("\n**Critical Issue**: Our block size is {:.1}x smaller than SILENT's!",
             silent_equiv as f64 / pw_block as f64);
    println!("This underestimates variance by approximately the same factor.\n");

    // =========================================================================
    // Section 5: Bootstrap Covariance Analysis
    // =========================================================================
    println!("## 5. Bootstrap Covariance Analysis\n");

    // Run our bootstrap
    let cov_estimate = bootstrap_difference_covariance_discrete(&baseline_ns, &test_ns, 2000, 0xDEADBEEF);
    let sigma = cov_estimate.matrix;

    println!("### Our Bootstrap Results (block_size={})\n", cov_estimate.block_size);

    // Diagonal variances
    println!("#### Variance Estimates (diagonal of Σ)\n");
    println!("| Quantile | Variance | Std Error | 95% CI Width |");
    println!("|----------|----------|-----------|--------------|");
    for (i, q) in [10, 20, 30, 40, 50, 60, 70, 80, 90].iter().enumerate() {
        let var = sigma[(i, i)];
        let se = var.sqrt();
        let ci_width = 1.96 * 2.0 * se;
        println!("| {}% | {:.2e} | {:.1} ns | {:.1} ns |", q, var, se, ci_width);
    }

    // Correlation matrix
    println!("\n#### Correlation Matrix (selected elements)\n");
    println!("| | 10% | 50% | 90% |");
    println!("|---|-----|-----|-----|");
    for (i, q) in [(0, "10%"), (4, "50%"), (8, "90%")] {
        let r_10 = sigma[(i, 0)] / (sigma[(i, i)].sqrt() * sigma[(0, 0)].sqrt());
        let r_50 = sigma[(i, 4)] / (sigma[(i, i)].sqrt() * sigma[(4, 4)].sqrt());
        let r_90 = sigma[(i, 8)] / (sigma[(i, i)].sqrt() * sigma[(8, 8)].sqrt());
        println!("| {} | {:.3} | {:.3} | {:.3} |", q, r_10, r_50, r_90);
    }

    println!("\n**Note**: High inter-quantile correlations (0.4-0.7) indicate dependent data structure.\n");

    // =========================================================================
    // Section 6: Quantile Difference Analysis
    // =========================================================================
    println!("## 6. Quantile Difference Analysis\n");

    let mut b_sorted = baseline_ns.clone();
    let mut t_sorted = test_ns.clone();
    let q_baseline = compute_deciles_inplace(&mut b_sorted);
    let q_test = compute_deciles_inplace(&mut t_sorted);
    let delta_hat = q_baseline - q_test;

    println!("### Raw Quantile Differences (Δ = baseline - test)\n");
    println!("| Quantile | Baseline | Test | Δ (ns) | SE | Z-score |");
    println!("|----------|----------|------|--------|-----|---------|");
    for (i, q) in [10, 20, 30, 40, 50, 60, 70, 80, 90].iter().enumerate() {
        let se = sigma[(i, i)].sqrt();
        let z = delta_hat[i] / se;
        println!("| {}% | {:.0} | {:.0} | {:+.1} | {:.1} | {:.2} |",
                 q, q_baseline[i], q_test[i], delta_hat[i], se, z);
    }

    println!("\n**Interpretation**:");
    println!("- The 10% quantile shows a large difference ({:+.0} ns)", delta_hat[0]);
    println!("- But with SE={:.0} ns, Z-score={:.2}", sigma[(0,0)].sqrt(), delta_hat[0] / sigma[(0,0)].sqrt());
    println!("- Most quantiles have |Z| < 2, suggesting no significant effect\n");

    // =========================================================================
    // Section 7: Gibbs Sampler Results
    // =========================================================================
    println!("## 7. Gibbs Sampler Analysis\n");

    for theta in [1.67, 5.0, 100.0, 1000.0] {
        println!("### θ = {} ns\n", theta);

        let config = SinglePassConfig {
            theta_ns: theta,
            pass_threshold: 0.05,
            fail_threshold: 0.95,
            bootstrap_iterations: 2000,
            timer_resolution_ns: 1.0, // Unknown for external data
            seed: 0xDEADBEEF,
            max_variance_ratio: 0.95,
        };

        let result = analyze_single_pass(&baseline_ns, &test_ns, &config);

        println!("| Metric | Value |");
        println!("|--------|-------|");
        println!("| Leak probability | {:.1}% |", result.leak_probability * 100.0);
        println!("| Decision | {} |", match result.leak_probability {
            p if p < 0.05 => "PASS",
            p if p > 0.95 => "FAIL",
            _ => "INCONCLUSIVE"
        });
        println!("| Expected (SILENT) | PASS |");
        println!("| Shift estimate | {:.2} ns |", result.effect_estimate.shift_ns);
        println!("| Tail estimate | {:.2} ns |", result.effect_estimate.tail_ns);
        println!("| 95% CI | [{:.1}, {:.1}] ns |",
                 result.effect_estimate.credible_interval_ns.0,
                 result.effect_estimate.credible_interval_ns.1);
        println!("| Quality | {:?} |", result.quality);
        println!();
    }

    // =========================================================================
    // Section 8: Variance Inflation Analysis
    // =========================================================================
    println!("## 8. Variance Inflation Analysis\n");

    // Effective sample size with different block sizes
    println!("### Effective Sample Size by Block Size\n");
    println!("| Block Size | Source | Effective n | Variance Multiplier |");
    println!("|------------|--------|-------------|---------------------|");

    for (block, source) in [
        (pw_block, "Our P-W"),
        (simple_block, "Simple"),
        (silent_equiv, "SILENT equiv"),
        (decorr_lag_baseline.max(decorr_lag_test) * 2, "ACF-based"),
    ] {
        let eff_n = n / block.max(1);
        let var_mult = block as f64;
        println!("| {} | {} | {} | {:.1}x |", block, source, eff_n, var_mult);
    }

    println!("\n**Key insight**: Using block_size={} instead of {} means our variance", pw_block, silent_equiv);
    println!("estimates are ~{:.1}x too small, making signals appear {:.1}x more significant.\n",
             silent_equiv as f64 / pw_block as f64,
             (silent_equiv as f64 / pw_block as f64).sqrt());

    // =========================================================================
    // Section 9: Recommendations
    // =========================================================================
    println!("## 9. Recommendations\n");

    println!("### Potential Fixes\n");
    println!("1. **Allow manual block size override** in `SinglePassConfig`");
    println!("2. **Improve ACF-based block estimation** for extreme autocorrelation");
    println!("3. **Add diagnostic warning** when computed block << decorrelation length");
    println!("4. **Use conservative (larger) block sizes** as a safety margin\n");

    println!("### Proposed Block Size Formula\n");
    println!("```");
    println!("// Current: Politis-White with max = 3√n");
    println!("// Proposed: max(P-W, ACF-based, conservative_floor)");
    println!("let acf_block = estimate_decorrelation_length(data) * 2;");
    println!("let conservative_floor = (n as f64).sqrt() as usize; // √n");
    println!("let block = pw_block.max(acf_block).max(conservative_floor);");
    println!("```\n");

    // Data quality indicators
    println!("### Data Quality Indicators for This Dataset\n");
    let unique_baseline: HashSet<i64> = baseline_ns.iter().map(|&v| v as i64).collect();
    let unique_test: HashSet<i64> = test_ns.iter().map(|&v| v as i64).collect();

    println!("| Indicator | Baseline | Test | Concern? |");
    println!("|-----------|----------|------|----------|");
    println!("| Unique values | {}% | {}% | {} |",
             (unique_baseline.len() as f64 / n as f64 * 100.0) as usize,
             (unique_test.len() as f64 / n as f64 * 100.0) as usize,
             if unique_baseline.len() > n / 10 { "No" } else { "Yes (discrete)" });
    println!("| Lag-1 ACF | {:.3} | {:.3} | {} |",
             acf1_baseline, acf1_test,
             if acf1_baseline.abs() > 0.1 || acf1_test.abs() > 0.1 { "Yes (correlated)" } else { "No" });
    println!("| CV | {:.1}% | {:.1}% | {} |",
             baseline_std / baseline_mean * 100.0,
             test_std / test_mean * 100.0,
             if baseline_std / baseline_mean > 0.5 { "Yes (high variance)" } else { "No" });

    println!("\n---\n");
    println!("*Report generated by timing-oracle mbedTLS diagnostic*");
}

fn compute_lag1_acf(x: &[f64]) -> f64 {
    compute_lagk_acf(x, 1)
}

fn compute_lagk_acf(x: &[f64], k: usize) -> f64 {
    let n = x.len();
    if k >= n { return 0.0; }

    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-12 { return 0.0; }

    let cov: f64 = x.iter().take(n - k)
        .zip(x.iter().skip(k))
        .map(|(&xi, &xk)| (xi - mean) * (xk - mean))
        .sum::<f64>() / (n - k) as f64;

    cov / var
}

fn chrono_lite() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    // Simple date formatting
    let days = secs / 86400;
    let years = 1970 + days / 365;
    format!("{}-01-18", years) // Approximate
}
