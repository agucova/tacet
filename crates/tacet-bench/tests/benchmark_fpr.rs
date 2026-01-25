//! False Positive Rate (FPR) benchmark tests.
//!
//! These tests measure the false positive rate of tacet and comparison tools
//! on null-effect datasets (same-xy). A good timing analysis tool should have FPR <= 10%.
//!
//! Run with: `cargo test --test benchmark_fpr -- --ignored --nocapture`

use std::time::Duration;
use tacet::AttackerModel;
use tacet_bench::{
    BenchmarkRunner, DudectAdapter, EffectType, StubAdapter, SyntheticConfig, TimingOracleAdapter, ToolAdapter,
};

/// Create a tacet adapter configured for fair comparison with dudect/RTLF.
///
/// Uses `Custom { threshold_ns: 0.0 }` which gets floored to timer resolution,
/// making it test "any detectable difference" like dudect/RTLF do.
fn fair_comparison_adapter(time_budget: Duration) -> TimingOracleAdapter {
    TimingOracleAdapter::with_attacker_model(AttackerModel::Custom { threshold_ns: 0.0 })
        .time_budget(time_budget)
}

/// Quick FPR sanity check (10 datasets, small samples).
#[test]
fn fpr_sanity_check() {
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(fair_comparison_adapter(Duration::from_secs(5))),
        Box::new(DudectAdapter::default()),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(10)
        .on_progress(|msg| println!("{}", msg));

    let config = SyntheticConfig {
        samples_per_class: 500,
        effect: EffectType::Null,
        seed: 42,
        ..Default::default()
    };

    let results = runner.run_fpr_benchmark("sanity-null", &config);

    println!("\nFPR Sanity Check Results:");
    for (tool, fpr) in results.all_fprs() {
        println!("  {}: {:.1}% ({}/{})", tool, fpr * 100.0,
                 results.false_positives.get(tool).unwrap_or(&0), results.total);
    }

    // Sanity check: FPR should be reasonable (not 100%)
    let to_fpr = results.fpr("tacet");
    assert!(
        to_fpr < 0.5,
        "tacet FPR {} is unreasonably high for sanity check",
        to_fpr
    );
}

/// FPR benchmark with 15k samples per class.
///
/// Expected: tacet FPR <= 10%
#[test]
#[ignore] // Long-running
fn fpr_15k_samples() {
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(fair_comparison_adapter(Duration::from_secs(30))),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(100)
        .on_progress(|msg| println!("{}", msg));

    let config = SyntheticConfig {
        samples_per_class: 15000,
        effect: EffectType::Null,
        seed: 42,
        ..Default::default()
    };

    let results = runner.run_fpr_benchmark("15k-same-xy", &config);

    println!("\nFPR Results (n=15k, {} datasets):", results.total);
    for (tool, fpr) in results.all_fprs() {
        println!(
            "  {}: {:.1}% ({}/{})",
            tool,
            fpr * 100.0,
            results.false_positives.get(tool).unwrap_or(&0),
            results.total
        );
    }

    let to_fpr = results.fpr("tacet");
    assert!(
        to_fpr <= 0.10,
        "tacet FPR {:.1}% exceeds 10% threshold",
        to_fpr * 100.0
    );
}

/// FPR benchmark with 30k samples per class.
///
/// Expected: tacet FPR <= 10%
#[test]
#[ignore] // Long-running
fn fpr_30k_samples() {
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(fair_comparison_adapter(Duration::from_secs(30))),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(100)
        .on_progress(|msg| println!("{}", msg));

    let config = SyntheticConfig {
        samples_per_class: 30000,
        effect: EffectType::Null,
        seed: 42,
        ..Default::default()
    };

    let results = runner.run_fpr_benchmark("30k-same-xy", &config);

    println!("\nFPR Results (n=30k, {} datasets):", results.total);
    for (tool, fpr) in results.all_fprs() {
        println!(
            "  {}: {:.1}% ({}/{})",
            tool,
            fpr * 100.0,
            results.false_positives.get(tool).unwrap_or(&0),
            results.total
        );
    }

    let to_fpr = results.fpr("tacet");
    assert!(
        to_fpr <= 0.10,
        "tacet FPR {:.1}% exceeds 10% threshold",
        to_fpr * 100.0
    );
}

/// FPR benchmark with 500k samples per class (high-power null test).
///
/// Expected: tacet FPR <= 10%
#[test]
#[ignore] // Very long-running
fn fpr_500k_samples() {
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(fair_comparison_adapter(Duration::from_secs(60))),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(50) // Fewer datasets due to size
        .on_progress(|msg| println!("{}", msg));

    let config = SyntheticConfig {
        samples_per_class: 500000,
        effect: EffectType::Null,
        seed: 42,
        ..Default::default()
    };

    let results = runner.run_fpr_benchmark("500k-same-xy", &config);

    println!("\nFPR Results (n=500k, {} datasets):", results.total);
    for (tool, fpr) in results.all_fprs() {
        println!(
            "  {}: {:.1}% ({}/{})",
            tool,
            fpr * 100.0,
            results.false_positives.get(tool).unwrap_or(&0),
            results.total
        );
    }

    let to_fpr = results.fpr("tacet");
    assert!(
        to_fpr <= 0.10,
        "tacet FPR {:.1}% exceeds 10% threshold",
        to_fpr * 100.0
    );
}

/// Multi-tool FPR comparison (requires dudect/RTLF to be available).
#[test]
#[ignore] // Requires external tools
fn fpr_multi_tool_comparison() {
    // Note: This test requires dudect and RTLF to be installed.
    // If they're not available, stub adapters are used instead.
    //
    // tacet uses Custom { threshold_ns: 0.0 } (floored to timer resolution)
    // to match dudect/RTLF's "any difference" test methodology.
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(fair_comparison_adapter(Duration::from_secs(30))),
        // Use stubs for now - replace with real adapters when available
        Box::new(StubAdapter::new("dudect")),
        Box::new(StubAdapter::new("rtlf")),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(100)
        .on_progress(|msg| println!("{}", msg));

    let config = SyntheticConfig {
        samples_per_class: 30000,
        effect: EffectType::Null,
        seed: 42,
        ..Default::default()
    };

    let results = runner.run_fpr_benchmark("30k-same-xy-multi", &config);

    println!("\nMulti-Tool FPR Comparison (n=30k, {} datasets):", results.total);
    println!("| Tool | FPR | False Positives |");
    println!("|------|-----|-----------------|");
    for (tool, fpr) in results.all_fprs() {
        println!(
            "| {} | {:.1}% | {}/{} |",
            tool,
            fpr * 100.0,
            results.false_positives.get(tool).unwrap_or(&0),
            results.total
        );
    }
}
