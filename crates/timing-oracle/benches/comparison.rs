//! Comparison benchmark suite entry point.
//!
//! Run with:
//! ```bash
//! cargo bench --bench comparison
//! ```
//!
//! Or with custom configuration:
//! ```bash
//! cargo bench --bench comparison -- --samples 50000 --trials 20
//! ```

mod comp;

use comp::{
    run_detection_comparison, run_efficiency_analysis, run_roc_analysis, BenchmarkConfig,
};
use comp::report::print_sample_efficiency;

fn main() {
    // Parse command-line arguments (simple version)
    let args: Vec<String> = std::env::args().collect();

    let mut config = BenchmarkConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--samples" => {
                if i + 1 < args.len() {
                    config.samples = args[i + 1].parse().unwrap_or(20_000);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--trials" => {
                if i + 1 < args.len() {
                    config.detection_trials = args[i + 1].parse().unwrap_or(10);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--no-roc" => {
                config.run_roc = false;
                i += 1;
            }
            "--no-efficiency" => {
                config.run_efficiency = false;
                i += 1;
            }
            "--json" => {
                config.export_json = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    config.json_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    // Print header
    println!("════════════════════════════════════════════");
    println!("  Timing Oracle Comparison Benchmarks");
    println!("════════════════════════════════════════════\n");

    println!("Configuration:");
    println!("  Samples per run: {}", config.samples);
    println!("  Detection trials: {}", config.detection_trials);
    if config.run_roc {
        println!("  ROC trials per case: {}", config.roc_trials_per_case);
    }
    if config.run_efficiency {
        println!(
            "  Efficiency trials per size: {}",
            config.efficiency_trials_per_size
        );
    }
    println!();

    // Section 1: Detection Rate Comparison
    println!("\n[1/3] Detection Rate Comparison");
    println!("─────────────────────────────────\n");

    let detection_results = run_detection_comparison(&config);
    detection_results.print_terminal_report();

    // Section 2: ROC Curve Analysis
    if config.run_roc {
        println!("\n\n[2/3] ROC Curve Analysis");
        println!("─────────────────────────\n");

        let roc_results = run_roc_analysis(&config);

        // Print ROC summary
        let mut full_results = comp::report::BenchmarkResults::new();
        for roc in roc_results {
            full_results.add_roc_curve(roc);
        }
        full_results.print_terminal_report();
    }

    // Section 3: Sample Efficiency Analysis
    if config.run_efficiency {
        println!("\n\n[3/3] Sample Efficiency Analysis");
        println!("─────────────────────────────────\n");

        let efficiency_results = run_efficiency_analysis(&config);
        print_sample_efficiency(&efficiency_results, config.efficiency_trials_per_size);
    }

    println!("\n════════════════════════════════════════════");
    println!("  Benchmark Complete");
    println!("════════════════════════════════════════════\n");
}
