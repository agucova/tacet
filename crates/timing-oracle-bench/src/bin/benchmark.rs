//! CLI tool for running timing side-channel benchmark suites.
//!
//! # Usage
//!
//! ```bash
//! # Run quick benchmark with all native tools
//! cargo run --bin benchmark -- --preset quick
//!
//! # Run medium benchmark with specific tools
//! cargo run --bin benchmark -- --preset medium --tools dudect,ks-test
//!
//! # Run thorough benchmark with output directory
//! cargo run --bin benchmark -- --preset thorough --output ./results/
//!
//! # Custom configuration
//! cargo run --bin benchmark -- \
//!   --samples 20000 \
//!   --datasets 50 \
//!   --effects "0,0.5,1,2,5" \
//!   --patterns "null,shift,tail"
//! ```

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use timing_oracle_bench::adapters::ToolAdapter;
use timing_oracle_bench::output::{to_markdown, write_csv, write_summary_csv};
use timing_oracle_bench::sweep::{SweepConfig, SweepRunner};
use timing_oracle_bench::{
    AndersonDarlingAdapter, DudectAdapter, EffectPattern, KsTestAdapter, MonaAdapter,
    NoiseModel, RtlfAdapter, RtlfNativeAdapter, SilentAdapter, SilentNativeAdapter,
    TimingOracleAdapter, TimingTvlaAdapter, TlsfuzzerAdapter,
};

/// Timing side-channel detection tool benchmark suite
#[derive(Parser, Debug)]
#[command(name = "benchmark")]
#[command(about = "Run comprehensive benchmarks comparing timing analysis tools")]
#[command(version)]
struct Args {
    /// Preset configuration: quick (~5min), medium (~30min), thorough (~3h), fine-threshold (~2-4h)
    #[arg(short, long, default_value = "quick")]
    preset: String,

    /// Output directory for results
    #[arg(short, long, default_value = ".")]
    output: PathBuf,

    /// Specific tools to benchmark (comma-separated)
    /// Available: timing-oracle, dudect, timing-tvla, ks-test, ad-test, mona
    #[arg(short, long)]
    tools: Option<String>,

    /// Custom samples per class (overrides preset)
    #[arg(long)]
    samples: Option<usize>,

    /// Custom datasets per configuration point (overrides preset)
    #[arg(long)]
    datasets: Option<usize>,

    /// Custom effect multipliers (comma-separated, e.g., "0,0.5,1,2,5")
    #[arg(long)]
    effects: Option<String>,

    /// Custom effect patterns (comma-separated: null,shift,tail,variance,bimodal,quantized)
    #[arg(long)]
    patterns: Option<String>,

    /// Custom noise models (comma-separated: iid, ar1-0.3, ar1-0.8, ar1-n0.5 for negative)
    #[arg(long)]
    noise: Option<String>,

    /// Skip writing CSV output files
    #[arg(long)]
    no_csv: bool,

    /// Show verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Suppress timing-oracle debug output (cleaner progress bar)
    #[arg(short, long)]
    quiet: bool,

    /// Use realistic timing (actual measurements) instead of synthetic generation
    ///
    /// In realistic mode, the benchmark executes actual timed operations using
    /// busy_wait_ns to inject controlled delays. This captures real platform
    /// noise, timer quantization, and measurement overhead that synthetic
    /// data cannot replicate.
    #[arg(long)]
    realistic: bool,

    /// Base operation time in nanoseconds for realistic mode (default: 1000 = 1μs)
    #[arg(long, default_value = "1000")]
    realistic_base_ns: u64,
}

fn main() {
    let args = Args::parse();

    // Suppress timing-oracle debug output for cleaner progress bar
    if args.quiet {
        std::env::set_var("TIMING_ORACLE_QUIET", "1");
    }

    // Parse preset
    let mut config = match args.preset.to_lowercase().as_str() {
        "quick" => SweepConfig::quick(),
        "medium" => SweepConfig::medium(),
        "thorough" => SweepConfig::thorough(),
        "fine-threshold" | "fine_threshold" | "finethreshold" => SweepConfig::fine_threshold(),
        _ => {
            eprintln!(
                "Unknown preset '{}'. Available: quick, medium, thorough, fine-threshold",
                args.preset
            );
            std::process::exit(1);
        }
    };

    // Apply custom overrides
    if let Some(samples) = args.samples {
        config.samples_per_class = samples;
    }
    if let Some(datasets) = args.datasets {
        config.datasets_per_point = datasets;
    }
    if let Some(effects) = &args.effects {
        config.effect_multipliers = effects
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
    }
    if let Some(patterns) = &args.patterns {
        config.effect_patterns = patterns
            .split(',')
            .filter_map(|s| parse_pattern(s.trim()))
            .collect();
    }
    if let Some(noise) = &args.noise {
        config.noise_models = noise
            .split(',')
            .filter_map(|s| parse_noise(s.trim()))
            .collect();
    }

    // Apply realistic mode
    if args.realistic {
        config.use_realistic = true;
        config.realistic_base_ns = args.realistic_base_ns;
    }

    // Select tools
    let tools: Vec<Box<dyn ToolAdapter>> = if let Some(tool_list) = &args.tools {
        if tool_list.to_lowercase() == "all" {
            // All tools for paper comparison
            // Uses RtlfNativeAdapter (faithful Rust port) and SilentAdapter (R reference)
            vec![
                Box::new(TimingOracleAdapter::default()),
                Box::new(DudectAdapter::default()),
                Box::new(TimingTvlaAdapter::default()),
                Box::new(KsTestAdapter::default()),
                Box::new(AndersonDarlingAdapter::default()),
                Box::new(MonaAdapter::default()),
                Box::new(RtlfNativeAdapter::default()), // Native Rust RTLF (faithful port)
                Box::new(SilentAdapter::default()),     // External R SILENT (reference impl)
                Box::new(TlsfuzzerAdapter::default()),  // External Python tlsfuzzer
            ]
        } else if tool_list.to_lowercase() == "native" {
            // All native Rust tools (no external dependencies)
            vec![
                Box::new(TimingOracleAdapter::default()),
                Box::new(DudectAdapter::default()),
                Box::new(TimingTvlaAdapter::default()),
                Box::new(KsTestAdapter::default()),
                Box::new(AndersonDarlingAdapter::default()),
                Box::new(MonaAdapter::default()),
                Box::new(RtlfNativeAdapter::default()),
                Box::new(SilentNativeAdapter::default()),
            ]
        } else {
            tool_list
                .split(',')
                .filter_map(|s| create_tool(s.trim()))
                .collect()
        }
    } else {
        // Default: all native tools (including native RTLF/SILENT)
        vec![
            Box::new(TimingOracleAdapter::default()),
            Box::new(DudectAdapter::default()),
            Box::new(TimingTvlaAdapter::default()),
            Box::new(KsTestAdapter::default()),
            Box::new(AndersonDarlingAdapter::default()),
            Box::new(MonaAdapter::default()),
            Box::new(RtlfNativeAdapter::default()),
            Box::new(SilentNativeAdapter::default()),
        ]
    };

    if tools.is_empty() {
        eprintln!("No valid tools selected. Available: dudect, timing-tvla, ks-test, ad-test, mona");
        std::process::exit(1);
    }

    // Create output directory
    if !args.output.exists() {
        if let Err(e) = fs::create_dir_all(&args.output) {
            eprintln!("Failed to create output directory: {}", e);
            std::process::exit(1);
        }
    }

    // Print configuration
    println!("=== Timing Tool Benchmark Suite ===\n");
    println!("Configuration:");
    println!("  Preset: {}", config.preset.name());
    println!("  Samples per class: {}", config.samples_per_class);
    println!("  Datasets per point: {}", config.datasets_per_point);
    println!("  Effect multipliers: {:?}", config.effect_multipliers);
    println!(
        "  Effect patterns: {:?}",
        config.effect_patterns.iter().map(|p| p.name()).collect::<Vec<_>>()
    );
    println!(
        "  Noise models: {:?}",
        config.noise_models.iter().map(|n| n.name()).collect::<Vec<_>>()
    );
    if config.use_realistic {
        println!("  Mode: REALISTIC (actual timed operations)");
        println!("  Base operation: {} ns", config.realistic_base_ns);
    } else {
        println!("  Mode: synthetic");
    }
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    let num_tools = tools.len();

    println!("  Tools: {:?}", tool_names);
    println!("  Total points: {}", config.total_points());
    println!("  Total datasets: {}", config.total_datasets());
    println!(
        "  Total tool runs: {}",
        config.total_datasets() * num_tools
    );
    println!();

    // Create runner
    let runner = SweepRunner::new(tools);

    // Set up progress bar
    let total_work = config.total_datasets() * runner.num_tools();
    let progress_bar = ProgressBar::new(total_work as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) ETA: {eta} | {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    progress_bar.enable_steady_tick(std::time::Duration::from_millis(100));

    let last_update = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    // Run benchmark
    let results = runner.run(&config, |progress, task| {
        let current = (progress * total_work as f64) as u64;
        let last = last_update.load(Ordering::Relaxed);
        if current > last {
            last_update.store(current, Ordering::Relaxed);
            progress_bar.set_position(current);
            progress_bar.set_message(task.to_string());
        }
    });

    progress_bar.finish_with_message("Complete!");

    let elapsed = start.elapsed();
    println!("\nCompleted in {:.1}s\n", elapsed.as_secs_f64());

    // Write outputs
    if !args.no_csv {
        let csv_path = args.output.join("benchmark_results.csv");
        if let Err(e) = write_csv(&results, &csv_path) {
            eprintln!("Failed to write CSV: {}", e);
        } else {
            println!("Wrote raw results to: {}", csv_path.display());
        }

        let summary_path = args.output.join("benchmark_summary.csv");
        if let Err(e) = write_summary_csv(&results, &summary_path) {
            eprintln!("Failed to write summary CSV: {}", e);
        } else {
            println!("Wrote summary to: {}", summary_path.display());
        }
    }

    // Write markdown report
    let md_path = args.output.join("benchmark_report.md");
    let report = to_markdown(&results);
    if let Err(e) = fs::write(&md_path, &report) {
        eprintln!("Failed to write markdown report: {}", e);
    } else {
        println!("Wrote report to: {}", md_path.display());
    }

    // Print summary
    println!("\n{}", report);
}

fn parse_pattern(s: &str) -> Option<EffectPattern> {
    match s.to_lowercase().as_str() {
        "null" => Some(EffectPattern::Null),
        "shift" => Some(EffectPattern::Shift),
        "tail" => Some(EffectPattern::Tail),
        "variance" => Some(EffectPattern::Variance),
        "bimodal" => Some(EffectPattern::bimodal_default()),
        "quantized" => Some(EffectPattern::quantized_default()),
        _ => None,
    }
}

fn parse_noise(s: &str) -> Option<NoiseModel> {
    let s = s.to_lowercase();
    if s == "iid" {
        return Some(NoiseModel::IID);
    }
    // Support both positive and negative phi:
    // ar1-0.5 for positive, ar1-n0.5 or ar1--0.5 for negative
    if let Some(phi_str) = s.strip_prefix("ar1-") {
        // Handle "ar1-n0.5" format for negative
        let (phi_str, negative) = if let Some(rest) = phi_str.strip_prefix('n') {
            (rest, true)
        } else {
            (phi_str, false)
        };
        if let Ok(mut phi) = phi_str.parse::<f64>() {
            if negative {
                phi = -phi;
            }
            // Valid range is (-1, 1) exclusive
            if phi.abs() < 1.0 {
                return Some(NoiseModel::AR1 { phi });
            }
        }
    }
    None
}

fn create_tool(name: &str) -> Option<Box<dyn ToolAdapter>> {
    match name.to_lowercase().as_str() {
        // Native adapters (always available)
        "timing-oracle" | "to" => Some(Box::new(TimingOracleAdapter::default())),
        "dudect" => Some(Box::new(DudectAdapter::default())),
        "timing-tvla" | "tvla" => Some(Box::new(TimingTvlaAdapter::default())),
        "ks-test" | "ks" => Some(Box::new(KsTestAdapter::default())),
        "ad-test" | "ad" | "anderson-darling" => Some(Box::new(AndersonDarlingAdapter::default())),
        "mona" => Some(Box::new(MonaAdapter::default())),
        // Native implementations of RTLF and SILENT (pure Rust, no external deps)
        "rtlf-native" | "rtlf-rs" => Some(Box::new(RtlfNativeAdapter::default())),
        "silent-native" | "silent-rs" => Some(Box::new(SilentNativeAdapter::default())),
        // External adapters (require external tools in PATH)
        "rtlf" => Some(Box::new(RtlfAdapter::default())),
        "silent" => Some(Box::new(SilentAdapter::default())),
        "tlsfuzzer" => Some(Box::new(TlsfuzzerAdapter::default())),
        // Shorthand for all native + external
        "all" => None, // Handled separately
        _ => None,
    }
}
