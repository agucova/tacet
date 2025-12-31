//! Direct comparison of Monte Carlo vs Analytical MDE methods.
//!
//! This benchmark runs both MDE estimation methods on identical covariance
//! matrices to measure the actual speedup and verify statistical equivalence.

use timing_oracle::analysis::{analytical_mde, estimate_mde_monte_carlo};
use timing_oracle::statistics::bootstrap_covariance_matrix;
use std::time::Instant;

fn main() {
    println!("=== MDE Method Comparison ===\n");

    // Generate realistic covariance data using Default preset
    println!("Generating realistic covariance matrix...");
    let samples: Vec<f64> = (0..100_000)
        .map(|i| {
            // Simulate realistic timing data with some variance
            let base = 100.0;
            let noise = (i as f64 * 0.123).sin() * 5.0;
            base + noise
        })
        .collect();

    let cov_result = bootstrap_covariance_matrix(&samples, 2_000, 42);
    let covariance = &cov_result.matrix;
    println!("Covariance matrix generated (2000 bootstrap iterations)\n");

    // Parameters matching Default preset
    let n_simulations = 100;
    let prior_sigmas = (1e6, 1e6);

    // Warmup runs
    println!("Running warmup iterations...");
    for _ in 0..3 {
        let _ = estimate_mde_monte_carlo(covariance, n_simulations, prior_sigmas);
        let _ = analytical_mde(covariance);
    }

    // Benchmark Monte Carlo method
    println!("\nBenchmarking Monte Carlo MDE ({} simulations)...", n_simulations);
    let mut mc_times = Vec::new();
    let mut mc_results = Vec::new();

    for run in 0..10 {
        let start = Instant::now();
        let result = estimate_mde_monte_carlo(covariance, n_simulations, prior_sigmas);
        let elapsed = start.elapsed();
        mc_times.push(elapsed);
        mc_results.push((result.shift_ns, result.tail_ns));

        if run < 3 {
            println!("  Run {}: {:?} (shift={:.2}ns, tail={:.2}ns)",
                run + 1, elapsed, result.shift_ns, result.tail_ns);
        }
    }

    let mc_mean_time = mc_times.iter().sum::<std::time::Duration>() / mc_times.len() as u32;
    let mc_mean_shift = mc_results.iter().map(|(s, _)| s).sum::<f64>() / mc_results.len() as f64;
    let mc_mean_tail = mc_results.iter().map(|(_, t)| t).sum::<f64>() / mc_results.len() as f64;

    println!("  Mean time: {:?}", mc_mean_time);
    println!("  Mean MDE: shift={:.2}ns, tail={:.2}ns\n", mc_mean_shift, mc_mean_tail);

    // Benchmark Analytical method
    println!("Benchmarking Analytical MDE...");
    let mut analytical_times = Vec::new();
    let mut analytical_results = Vec::new();

    for run in 0..10 {
        let start = Instant::now();
        let (shift, tail) = analytical_mde(covariance);
        let elapsed = start.elapsed();
        analytical_times.push(elapsed);
        analytical_results.push((shift, tail));

        if run < 3 {
            println!("  Run {}: {:?} (shift={:.2}ns, tail={:.2}ns)",
                run + 1, elapsed, shift, tail);
        }
    }

    let analytical_mean_time = analytical_times.iter().sum::<std::time::Duration>()
        / analytical_times.len() as u32;
    let analytical_mean_shift = analytical_results.iter().map(|(s, _)| s).sum::<f64>()
        / analytical_results.len() as f64;
    let analytical_mean_tail = analytical_results.iter().map(|(_, t)| t).sum::<f64>()
        / analytical_results.len() as f64;

    println!("  Mean time: {:?}", analytical_mean_time);
    println!("  Mean MDE: shift={:.2}ns, tail={:.2}ns\n",
        analytical_mean_shift, analytical_mean_tail);

    // Compare results
    println!("=== Comparison ===");
    let speedup = mc_mean_time.as_secs_f64() / analytical_mean_time.as_secs_f64();
    println!("Speedup: {:.1}× faster (MC: {:?} → Analytical: {:?})",
        speedup, mc_mean_time, analytical_mean_time);

    let shift_diff_pct = ((analytical_mean_shift - mc_mean_shift).abs() / mc_mean_shift) * 100.0;
    let tail_diff_pct = ((analytical_mean_tail - mc_mean_tail).abs() / mc_mean_tail) * 100.0;

    println!("\nStatistical Equivalence:");
    println!("  Shift MDE: {:.2}ns (MC) vs {:.2}ns (Analytical) → {:.1}% difference",
        mc_mean_shift, analytical_mean_shift, shift_diff_pct);
    println!("  Tail MDE:  {:.2}ns (MC) vs {:.2}ns (Analytical) → {:.1}% difference",
        mc_mean_tail, analytical_mean_tail, tail_diff_pct);

    if shift_diff_pct < 20.0 && tail_diff_pct < 20.0 {
        println!("  ✓ Results are statistically equivalent (< 20% difference)");
    } else {
        println!("  ⚠ Results differ by > 20% (unexpected!)");
    }

    // Estimate impact on full Default preset
    println!("\n=== Impact on Default Preset ===");
    let default_total_time = 1.324; // seconds (from previous benchmark)
    let mde_calls_per_test = 1; // Called once per oracle.test()

    let mc_contribution = mc_mean_time.as_secs_f64() * mde_calls_per_test as f64;
    let analytical_contribution = analytical_mean_time.as_secs_f64() * mde_calls_per_test as f64;
    let savings = mc_contribution - analytical_contribution;

    println!("Default preset baseline: {:.3}s", default_total_time);
    println!("MDE contribution (MC): {:.3}s ({:.1}%)",
        mc_contribution, (mc_contribution / default_total_time) * 100.0);
    println!("MDE contribution (Analytical): {:.3}s ({:.1}%)",
        analytical_contribution, (analytical_contribution / default_total_time) * 100.0);
    println!("Expected savings: {:.3}s ({:.1}% of total)",
        savings, (savings / default_total_time) * 100.0);
    println!("Expected new total: {:.3}s", default_total_time - savings);

    if savings < 0.010 {
        println!("\n⚠ MDE is not a significant bottleneck (< 10ms savings)");
        println!("   Other components dominate Default preset runtime.");
    }
}
