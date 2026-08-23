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

use tacet_bench::adapters::ToolAdapter;
use tacet_bench::checkpoint::IncrementalCsvWriter;
use tacet_bench::output::{to_markdown, write_csv, write_summary_csv};
use tacet_bench::process_pool::{ProcessConfig, ProcessPool};
use tacet_bench::sweep::{SweepConfig, SweepRunner};
use tacet_bench::{
    AndersonDarlingAdapter, DudectAdapter, EffectPattern, KsTestAdapter, MonaAdapter, NoiseModel,
    RtlfAdapter, RtlfNativeAdapter, SilentAdapter, SilentNativeAdapter, TimingOracleAdapter,
    TimingTvlaAdapter, TlsfuzzerAdapter,
};

/// Timing side-channel detection tool benchmark suite
#[derive(Parser, Debug)]
#[command(name = "benchmark")]
#[command(about = "Run comprehensive benchmarks comparing timing analysis tools")]
#[command(version)]
struct Args {
    /// Preset configuration: quick (~5min), medium (~30min), thorough (~3h),
    /// fine-threshold (~2-4h), threshold-relative (~1-2h)
    #[arg(short, long, default_value = "quick")]
    preset: String,

    /// Output directory for results
    #[arg(short, long, default_value = ".")]
    output: PathBuf,

    /// Specific tools to benchmark (comma-separated)
    /// Available: tacet, dudect, timing-tvla, ks-test, ad-test, mona
    /// Special: "all" (all tools), "native" (Rust-only), "threshold-relative" (multi-model)
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

    /// Suppress tacet debug output (cleaner progress bar)
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

    /// Disable incremental CSV writes (write all results at the end instead).
    ///
    /// By default, results are written to disk immediately after each tool
    /// completes, enabling survival of interrupted runs. Use this flag to
    /// disable incremental writes and write all results only at the end.
    #[arg(long)]
    no_incremental: bool,

    /// Resume from an interrupted benchmark.
    ///
    /// Loads existing results from the output CSV and skips completed work items.
    /// Implies --incremental.
    #[arg(long)]
    resume: bool,

    /// Disable persistent process pools for R and Python tools.
    ///
    /// By default, subprocess-based tools (silent, tlsfuzzer, rtlf) use persistent
    /// process pools to avoid interpreter startup overhead on each analysis.
    /// This flag disables that optimization, falling back to subprocess-per-call.
    #[arg(long)]
    no_persistent_processes: bool,
}

fn main() {
    let args = Args::parse();

    // Suppress tacet debug output for cleaner progress bar
    if args.quiet {
        std::env::set_var("TIMING_ORACLE_QUIET", "1");
    }

    // Parse bootstrap method from env var (for ablation studies)
    let bootstrap_method = match std::env::var("TACET_BOOTSTRAP_METHOD").ok().as_deref() {
        Some("stratified") | Some("per-class") => tacet::BootstrapMethod::Stratified,
        _ => tacet::BootstrapMethod::Joint,
    };

    // Parse preset
    let mut config = match args.preset.to_lowercase().as_str() {
        "quick" => SweepConfig::quick(),
        "medium" => SweepConfig::medium(),
        "thorough" => SweepConfig::thorough(),
        "fine-threshold" | "fine_threshold" | "finethreshold" => SweepConfig::fine_threshold(),
        "threshold-relative" | "threshold_relative" | "thresholdrelative" => {
            SweepConfig::threshold_relative()
        }
        "shared-hardware-stress" | "shared_hardware_stress" | "sharedhardwarestress" => {
            SweepConfig::shared_hardware_stress()
        }
        _ => {
            eprintln!(
                "Unknown preset '{}'. Available: quick, medium, thorough, fine-threshold, threshold-relative, shared-hardware-stress",
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

    // Create process pools for subprocess-based tools (R, Python)
    // Pools are required for R-based tools (SILENT, RTLF)
    let (r_pool, python_pool) = if args.no_persistent_processes {
        eprintln!("Error: --no-persistent-processes is no longer supported");
        eprintln!("R-based tools (silent, rtlf) require persistent process pools");
        std::process::exit(1);
    } else {
        let r_pool = create_r_pool().unwrap_or_else(|| {
            eprintln!("Error: Failed to create R process pool");
            eprintln!("Make sure R is installed and the worker script is available");
            std::process::exit(1);
        });
        let python_pool = create_python_pool().unwrap_or_else(|| {
            eprintln!("Error: Failed to create Python process pool");
            eprintln!("Make sure Python and tlsfuzzer are installed");
            std::process::exit(1);
        });

        if args.verbose {
            println!("  Process pools: R=enabled, Python=enabled");
        }

        (r_pool, python_pool)
    };

    // Select tools
    let tools: Vec<Box<dyn ToolAdapter>> = if let Some(tool_list) = &args.tools {
        if tool_list.to_lowercase() == "all" {
            // All tools for paper comparison
            // Uses R-based RTLF/SILENT (reference implementations) via persistent pools
            vec![
                Box::new(TimingOracleAdapter::default().bootstrap_method(bootstrap_method)),
                Box::new(DudectAdapter::default()),
                Box::new(TimingTvlaAdapter::default()),
                Box::new(KsTestAdapter::default()),
                Box::new(AndersonDarlingAdapter::default()),
                Box::new(MonaAdapter::default()),
                Box::new(SilentAdapter::default().with_pool(r_pool.clone())),
                Box::new(RtlfAdapter::default().with_pool(r_pool.clone())),
                Box::new(TlsfuzzerAdapter::default().with_pool(python_pool.clone())),
            ]
        } else if tool_list.to_lowercase() == "native" {
            // All native Rust tools (no external dependencies)
            vec![
                Box::new(TimingOracleAdapter::default().bootstrap_method(bootstrap_method)),
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
                .filter_map(|s| create_tool(s.trim(), &r_pool, &python_pool, bootstrap_method))
                .collect()
        }
    } else {
        // Default: all tools (same as --tools all)
        vec![
            Box::new(TimingOracleAdapter::default().bootstrap_method(bootstrap_method)),
            Box::new(DudectAdapter::default()),
            Box::new(TimingTvlaAdapter::default()),
            Box::new(KsTestAdapter::default()),
            Box::new(AndersonDarlingAdapter::default()),
            Box::new(MonaAdapter::default()),
            Box::new(SilentAdapter::default().with_pool(r_pool.clone())),
            Box::new(RtlfAdapter::default().with_pool(r_pool.clone())),
            Box::new(TlsfuzzerAdapter::default().with_pool(python_pool.clone())),
        ]
    };

    if tools.is_empty() {
        eprintln!(
            "No valid tools selected. Available: dudect, timing-tvla, ks-test, ad-test, mona"
        );
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
        config
            .effect_patterns
            .iter()
            .map(|p| p.name())
            .collect::<Vec<_>>()
    );
    println!(
        "  Noise models: {:?}",
        config
            .noise_models
            .iter()
            .map(|n| n.name())
            .collect::<Vec<_>>()
    );
    if config.use_realistic {
        println!("  Mode: REALISTIC (actual timed operations)");
        println!("  Base operation: {} ns", config.realistic_base_ns);
    } else {
        println!("  Mode: synthetic");
    }
    if args.resume {
        println!("  Checkpoint: RESUME (loading existing results)");
    } else if !args.no_incremental {
        println!("  Checkpoint: INCREMENTAL (writing results as they complete)");
    }
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    let num_tools = tools.len();

    println!("  Tools: {:?}", tool_names);
    println!("  Total points: {}", config.total_points());
    println!("  Total datasets: {}", config.total_datasets());
    println!("  Total tool runs: {}", config.total_datasets() * num_tools);

    // Show batch info for memory-constrained users
    let batch_size = 500; // DEFAULT_BATCH_SIZE from sweep.rs
    let num_batches = config.total_datasets().div_ceil(batch_size);
    if num_batches > 1 {
        let estimated_mem_mb = (batch_size * config.samples_per_class * 16) / (1024 * 1024);
        println!(
            "  Batched: {} batches of {} datasets (~{}MB peak memory)",
            num_batches,
            batch_size,
            estimated_mem_mb.max(160)
        );
    }
    println!();

    // Create runner
    let runner = SweepRunner::new(tools);

    // Set up checkpoint for incremental writes and resumability
    // Incremental is the default; --no-incremental disables it
    let use_checkpoint = !args.no_incremental || args.resume;
    let checkpoint_path = args.output.join("benchmark_results.csv");
    let checkpoint = if use_checkpoint {
        match IncrementalCsvWriter::new(&checkpoint_path, args.resume) {
            Ok(writer) => {
                let resumed = writer.resumed_count;
                if resumed > 0 {
                    println!("  Resuming: {} results loaded from checkpoint", resumed);
                }
                Some(Arc::new(writer))
            }
            Err(e) => {
                eprintln!("Failed to create checkpoint file: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // Set up progress bar
    // Note: indicatif's {eta} produces garbage values at position 0 (division by zero).
    // We show our own ETA in the message field once we have enough data points.
    let total_work = config.total_datasets() * runner.num_tools();
    let progress_bar = ProgressBar::new(total_work as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    progress_bar.enable_steady_tick(std::time::Duration::from_millis(100));
    progress_bar.set_message("warming up...");

    // Set initial progress position if resuming
    if let Some(ref cp) = checkpoint {
        progress_bar.set_position(cp.resumed_count as u64);
    }

    let last_update = Arc::new(AtomicU64::new(
        checkpoint
            .as_ref()
            .map(|c| c.resumed_count as u64)
            .unwrap_or(0),
    ));
    let start = Instant::now();

    // Helper to format ETA from seconds
    fn format_eta(secs: f64) -> String {
        if secs < 60.0 {
            format!("{:.0}s", secs)
        } else if secs < 3600.0 {
            format!("{}m {}s", (secs / 60.0) as u64, (secs % 60.0) as u64)
        } else {
            format!(
                "{}h {}m",
                (secs / 3600.0) as u64,
                ((secs % 3600.0) / 60.0) as u64
            )
        }
    }

    // Helper to build progress message with ETA
    let make_message = |progress: f64, task: &str, elapsed_secs: f64| -> String {
        // Only show ETA after 5% progress to get stable estimates
        if progress > 0.05 {
            let eta_secs = elapsed_secs / progress * (1.0 - progress);
            format!("ETA: {} | {}", format_eta(eta_secs), task)
        } else if progress > 0.0 {
            format!("estimating... | {}", task)
        } else {
            "warming up...".to_string()
        }
    };

    // Run benchmark (with or without checkpoint)
    let results = if let Some(cp) = checkpoint.clone() {
        runner.run_with_checkpoint(&config, Some(cp), |progress, task| {
            let current = (progress * total_work as f64) as u64;
            let last = last_update.load(Ordering::Relaxed);
            if current > last {
                last_update.store(current, Ordering::Relaxed);
                progress_bar.set_position(current);
                let msg = make_message(progress, task, start.elapsed().as_secs_f64());
                progress_bar.set_message(msg);
            }
        })
    } else {
        runner.run(&config, |progress, task| {
            let current = (progress * total_work as f64) as u64;
            let last = last_update.load(Ordering::Relaxed);
            if current > last {
                last_update.store(current, Ordering::Relaxed);
                progress_bar.set_position(current);
                let msg = make_message(progress, task, start.elapsed().as_secs_f64());
                progress_bar.set_message(msg);
            }
        })
    };

    progress_bar.finish_with_message("Complete!");

    let elapsed = start.elapsed();
    println!("\nCompleted in {:.1}s\n", elapsed.as_secs_f64());

    // Write outputs
    if !args.no_csv {
        // Skip raw CSV if checkpoint was used (already written incrementally)
        if !use_checkpoint {
            let csv_path = args.output.join("benchmark_results.csv");
            if let Err(e) = write_csv(&results, &csv_path) {
                eprintln!("Failed to write CSV: {}", e);
            } else {
                println!("Wrote raw results to: {}", csv_path.display());
            }
        } else {
            println!(
                "Raw results already written to: {}",
                checkpoint_path.display()
            );
        }

        // Summary is always written at the end (needs all results for aggregation)
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

fn create_tool(
    name: &str,
    r_pool: &Arc<ProcessPool>,
    python_pool: &Arc<ProcessPool>,
    bootstrap_method: tacet::BootstrapMethod,
) -> Option<Box<dyn ToolAdapter>> {
    match name.to_lowercase().as_str() {
        // Native adapters (always available)
        "tacet" | "to" => Some(Box::new(
            TimingOracleAdapter::default().bootstrap_method(bootstrap_method),
        )),
        "dudect" => Some(Box::new(DudectAdapter::default())),
        "timing-tvla" | "tvla" => Some(Box::new(TimingTvlaAdapter::default())),
        "ks-test" | "ks" => Some(Box::new(KsTestAdapter::default())),
        "ad-test" | "ad" | "anderson-darling" => Some(Box::new(AndersonDarlingAdapter::default())),
        "mona" => Some(Box::new(MonaAdapter::default())),
        // Native implementations of RTLF and SILENT (pure Rust, no external deps)
        "rtlf-native" | "rtlf-rs" => Some(Box::new(RtlfNativeAdapter::default())),
        "silent-native" | "silent-rs" => Some(Box::new(SilentNativeAdapter::default())),
        // R-based adapters (use persistent pools)
        "rtlf" => Some(Box::new(RtlfAdapter::default().with_pool(r_pool.clone()))),
        "silent" => Some(Box::new(SilentAdapter::default().with_pool(r_pool.clone()))),
        // Python-based tlsfuzzer (uses persistent pool)
        "tlsfuzzer" => Some(Box::new(
            TlsfuzzerAdapter::default().with_pool(python_pool.clone()),
        )),
        // Shorthand for all native + external
        "all" => None, // Handled separately
        _ => None,
    }
}

/// Create an R process pool for SILENT and RTLF.
///
/// Returns None if the worker script is not available or R is not installed.
fn create_r_pool() -> Option<Arc<ProcessPool>> {
    // Look for the worker script relative to the executable or in known locations
    let script_path = find_script("r-persistent-worker.R")?;

    // Find SILENT and RTLF script paths
    // These are installed via nix devenv, so we look for them via the wrapper commands
    let silent_path = find_r_tool_script("silent", "share/silent/scripts/SILENT.R");
    let rtlf_path = find_r_tool_script("rtlf", "share/rtlf/rtlf.R");

    // Create pool with ~92% of CPU count (leave headroom for system, rayon, I/O)
    let cpus = num_cpus::get();
    let pool_size = ((cpus * 92) / 100).max(2);
    let config =
        ProcessConfig::r_worker(&script_path, silent_path.as_deref(), rtlf_path.as_deref());

    let pool = ProcessPool::new(config, pool_size);
    Some(Arc::new(pool))
}

/// Find an R tool's script path by inspecting the wrapper command.
///
/// The nix devenv creates wrapper scripts that call Rscript with the script path.
/// We parse the wrapper to extract the actual script location.
fn find_r_tool_script(command: &str, relative_path: &str) -> Option<String> {
    // First, try to find the command in PATH
    let which_output = std::process::Command::new("which")
        .arg(command)
        .output()
        .ok()?;

    if !which_output.status.success() {
        return None;
    }

    let wrapper_path = String::from_utf8_lossy(&which_output.stdout)
        .trim()
        .to_string();
    if wrapper_path.is_empty() {
        return None;
    }

    // Read the wrapper script to find the actual R script path
    let wrapper_content = std::fs::read_to_string(&wrapper_path).ok()?;

    // The wrapper looks like: exec Rscript ... "$out/share/silent/scripts/SILENT.R"
    // or contains --add-flags with the path
    // We look for a nix store path containing our relative_path
    for line in wrapper_content.lines() {
        // Look for the relative path in the wrapper
        if let Some(idx) = line.find(relative_path) {
            // Find the start of the nix store path (starts with /nix/store/)
            let search_start = line[..idx]
                .rfind("/nix/store/")
                .or_else(|| line[..idx].rfind("/"))
                .unwrap_or(0);
            // Find the end (next quote or whitespace)
            let path_start = &line[search_start..];
            let path_end = path_start
                .find(|c: char| c == '"' || c == '\'' || c.is_whitespace())
                .map(|i| search_start + i)
                .unwrap_or(line.len());
            let path = line[search_start..path_end].trim_matches(|c| c == '"' || c == '\'');
            if std::path::Path::new(path).exists() {
                return Some(path.to_string());
            }
        }
    }

    // Fallback: derive path from wrapper location
    // Wrapper is at /nix/store/HASH-tool/bin/tool
    // Script is at /nix/store/HASH-tool/share/...
    let wrapper_path = std::path::Path::new(&wrapper_path);
    if let Some(bin_dir) = wrapper_path.parent() {
        if let Some(tool_dir) = bin_dir.parent() {
            let script_path = tool_dir.join(relative_path);
            if script_path.exists() {
                return Some(script_path.to_string_lossy().to_string());
            }
        }
    }

    None
}

/// Create a Python process pool for tlsfuzzer.
///
/// Returns None if the worker script is not available or Python is not installed.
fn create_python_pool() -> Option<Arc<ProcessPool>> {
    // Look for the worker script relative to the executable or in known locations
    let script_path = find_script("python-persistent-worker.py")?;

    // Python workers use ~150-200MB each with numpy/pandas/tlsfuzzer loaded
    // Use 1/3 of CPUs for Python (R tools like SILENT/RTLF run more frequently)
    // No cap - on large instances with enough RAM, scale up
    let pool_size = (num_cpus::get() / 3).max(2);
    let config = ProcessConfig::python_worker(&script_path);

    let pool = ProcessPool::new(config, pool_size);
    Some(Arc::new(pool))
}

/// Find a worker script in known locations.
fn find_script(name: &str) -> Option<String> {
    // Check relative to executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            // Check scripts/ relative to binary
            let script_path = exe_dir.join("scripts").join(name);
            if script_path.exists() {
                return Some(script_path.to_string_lossy().to_string());
            }

            // Check ../../scripts/ (for development builds)
            let script_path = exe_dir.join("../../scripts").join(name);
            if script_path.exists() {
                return Some(
                    script_path
                        .canonicalize()
                        .ok()?
                        .to_string_lossy()
                        .to_string(),
                );
            }
        }
    }

    // Check current directory
    let script_path = PathBuf::from("scripts").join(name);
    if script_path.exists() {
        return Some(
            script_path
                .canonicalize()
                .ok()?
                .to_string_lossy()
                .to_string(),
        );
    }

    // Check workspace root (for cargo run)
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let workspace_root = PathBuf::from(manifest_dir).join("../../scripts").join(name);
        if workspace_root.exists() {
            return Some(
                workspace_root
                    .canonicalize()
                    .ok()?
                    .to_string_lossy()
                    .to_string(),
            );
        }
    }

    None
}
