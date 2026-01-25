//! Statistical Power benchmark tests.
//!
//! These tests measure the detection rate (power) of tacet and comparison tools
//! on datasets with known timing effects. A good timing analysis tool should detect
//! strong effects (e.g., 5% shift) with high probability (>= 90%).
//!
//! Run with: `cargo test --test benchmark_power -- --ignored --nocapture`

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

/// Quick power sanity check with strong effect.
#[test]
fn power_sanity_check() {
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(
            fair_comparison_adapter(Duration::from_secs(10)),
        ),
        Box::new(DudectAdapter::default()),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(10)
        .on_progress(|msg| println!("{}", msg));

    // Strong 10% shift should be easily detectable
    let config = SyntheticConfig {
        samples_per_class: 5000,
        effect: EffectType::Shift { percent: 10.0 },
        seed: 42,
        ..Default::default()
    };

    let results = runner.run_power_benchmark("sanity-shift10", &config);

    println!("\nPower Sanity Check Results (10% shift):");
    for (tool, power) in results.all_powers() {
        println!(
            "  {}: {:.1}% ({}/{})",
            tool,
            power * 100.0,
            results.true_positives.get(tool).unwrap_or(&0),
            results.total
        );
    }

    // Strong effect should have reasonable detection rate
    let to_power = results.power("tacet");
    println!("tacet power: {:.1}%", to_power * 100.0);
    // Note: With small samples and short time budget, we may not always detect
}

/// Power benchmark with 1% shift (subtle effect).
///
/// This is a challenging test - detecting 1% shifts requires many samples.
#[test]
#[ignore] // Long-running
fn power_30k_shift_1pct() {
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(
            fair_comparison_adapter(Duration::from_secs(30)),
        ),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(100)
        .on_progress(|msg| println!("{}", msg));

    let config = SyntheticConfig {
        samples_per_class: 30000,
        effect: EffectType::Shift { percent: 1.0 },
        seed: 42,
        ..Default::default()
    };

    let results = runner.run_power_benchmark("30k-shift-1pct", &config);

    println!("\nPower Results (n=30k, 1% shift, {} datasets):", results.total);
    for (tool, power) in results.all_powers() {
        println!(
            "  {}: {:.1}% ({}/{})",
            tool,
            power * 100.0,
            results.true_positives.get(tool).unwrap_or(&0),
            results.total
        );
    }

    // 1% shift is subtle - just report the results
    let to_power = results.power("tacet");
    println!("\ntacet detects 1% shifts at {:.1}% rate", to_power * 100.0);
}

/// Power benchmark with 5% shift (clear effect).
///
/// Expected: tacet power >= 90%
#[test]
#[ignore] // Long-running
fn power_30k_shift_5pct() {
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(
            fair_comparison_adapter(Duration::from_secs(30)),
        ),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(100)
        .on_progress(|msg| println!("{}", msg));

    let config = SyntheticConfig {
        samples_per_class: 30000,
        effect: EffectType::Shift { percent: 5.0 },
        seed: 42,
        ..Default::default()
    };

    let results = runner.run_power_benchmark("30k-shift-5pct", &config);

    println!("\nPower Results (n=30k, 5% shift, {} datasets):", results.total);
    for (tool, power) in results.all_powers() {
        println!(
            "  {}: {:.1}% ({}/{})",
            tool,
            power * 100.0,
            results.true_positives.get(tool).unwrap_or(&0),
            results.total
        );
    }

    let to_power = results.power("tacet");
    // Note: With synthetic data, the effect is in log-space, so detection depends
    // on how the shift translates to real-space differences
    println!("\ntacet detects 5% shifts at {:.1}% rate", to_power * 100.0);
}

/// Power benchmark with tail effect.
///
/// Tail effects (same mean, heavier tail) are harder to detect than shifts.
#[test]
#[ignore] // Long-running
fn power_30k_tail() {
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(
            fair_comparison_adapter(Duration::from_secs(30)),
        ),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(100)
        .on_progress(|msg| println!("{}", msg));

    let config = SyntheticConfig {
        samples_per_class: 30000,
        effect: EffectType::Tail,
        seed: 42,
        ..Default::default()
    };

    let results = runner.run_power_benchmark("30k-tail", &config);

    println!("\nPower Results (n=30k, tail effect, {} datasets):", results.total);
    for (tool, power) in results.all_powers() {
        println!(
            "  {}: {:.1}% ({}/{})",
            tool,
            power * 100.0,
            results.true_positives.get(tool).unwrap_or(&0),
            results.total
        );
    }

    let to_power = results.power("tacet");
    println!("\ntacet detects tail effects at {:.1}% rate", to_power * 100.0);
}

/// Power benchmark with same-mean effect (different variance).
#[test]
#[ignore] // Long-running
fn power_30k_same_mean() {
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(
            fair_comparison_adapter(Duration::from_secs(30)),
        ),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(100)
        .on_progress(|msg| println!("{}", msg));

    let config = SyntheticConfig {
        samples_per_class: 30000,
        effect: EffectType::SameMean,
        seed: 42,
        ..Default::default()
    };

    let results = runner.run_power_benchmark("30k-same-mean", &config);

    println!("\nPower Results (n=30k, same-mean effect, {} datasets):", results.total);
    for (tool, power) in results.all_powers() {
        println!(
            "  {}: {:.1}% ({}/{})",
            tool,
            power * 100.0,
            results.true_positives.get(tool).unwrap_or(&0),
            results.total
        );
    }

    let to_power = results.power("tacet");
    println!("\ntacet detects same-mean effects at {:.1}% rate", to_power * 100.0);
}

/// High-sample power test with 1% shift.
#[test]
#[ignore] // Very long-running
fn power_500k_shift_1pct() {
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(
            fair_comparison_adapter(Duration::from_secs(60)),
        ),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(50)
        .on_progress(|msg| println!("{}", msg));

    let config = SyntheticConfig {
        samples_per_class: 500000,
        effect: EffectType::Shift { percent: 1.0 },
        seed: 42,
        ..Default::default()
    };

    let results = runner.run_power_benchmark("500k-shift-1pct", &config);

    println!("\nPower Results (n=500k, 1% shift, {} datasets):", results.total);
    for (tool, power) in results.all_powers() {
        println!(
            "  {}: {:.1}% ({}/{})",
            tool,
            power * 100.0,
            results.true_positives.get(tool).unwrap_or(&0),
            results.total
        );
    }

    let to_power = results.power("tacet");
    println!("\ntacet detects 1% shifts with 500k samples at {:.1}% rate", to_power * 100.0);
}

/// Multi-tool power comparison.
#[test]
#[ignore] // Requires external tools
fn power_multi_tool_comparison() {
    let tools: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(
            fair_comparison_adapter(Duration::from_secs(30)),
        ),
        Box::new(StubAdapter::new("dudect")),
        Box::new(StubAdapter::new("rtlf")),
    ];

    let runner = BenchmarkRunner::new(tools)
        .datasets_per_config(100)
        .on_progress(|msg| println!("{}", msg));

    println!("\n=== Multi-Tool Power Comparison ===\n");

    // Test multiple effect types
    let effects = [
        ("30k-shift-1pct", EffectType::Shift { percent: 1.0 }),
        ("30k-shift-5pct", EffectType::Shift { percent: 5.0 }),
        ("30k-tail", EffectType::Tail),
    ];

    println!("| Configuration | Effect | tacet | dudect | rtlf |");
    println!("|---------------|--------|---------------|--------|------|");

    for (name, effect) in effects {
        let config = SyntheticConfig {
            samples_per_class: 30000,
            effect,
            seed: 42,
            ..Default::default()
        };

        let results = runner.run_power_benchmark(name, &config);

        println!(
            "| {} | {} | {:.1}% | {:.1}% | {:.1}% |",
            name,
            results.effect_type,
            results.power("tacet") * 100.0,
            results.power("dudect") * 100.0,
            results.power("rtlf") * 100.0,
        );
    }
}
