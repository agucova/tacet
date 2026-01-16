//! Comparison benchmark suite for timing-oracle vs other tools.
//!
//! This benchmark compares timing-oracle against dudect-bencher using
//! standardized test cases and metrics including:
//! - Detection rate (TPR) on known-leaky code
//! - False positive rate on known-safe code
//! - Sample efficiency
//! - ROC curve analysis
//!
//! Run with: `cargo bench --bench comparison`

pub mod adapters;
pub mod metrics;
pub mod report;
pub mod test_cases;

use adapters::dudect_adapter::DudectDetector;
use adapters::timing_oracle_adapter::TimingOracleDetector;
use adapters::Detector;
use metrics::{
    generate_roc_curve, measure_detection_rate, measure_false_positive_rate,
    measure_sample_efficiency, measure_sample_efficiency_stats, SampleEfficiencyStats,
};
use report::{create_tool_result, BenchmarkResults, TestCaseComparison};
use std::collections::HashMap;
use test_cases::{all_test_cases, leaky_test_cases, safe_test_cases, TestCase};

/// Configuration for benchmark run
pub struct BenchmarkConfig {
    /// Number of samples per detection run
    pub samples: usize,
    /// Number of trials for detection rate measurement
    pub detection_trials: usize,
    /// Number of trials per threshold for ROC curves
    pub roc_trials_per_case: usize,
    /// Sample sizes to test for efficiency analysis
    pub efficiency_sample_sizes: Vec<usize>,
    /// Number of trials per sample size for efficiency analysis
    pub efficiency_trials_per_size: usize,
    /// Whether to run ROC analysis
    pub run_roc: bool,
    /// Whether to run sample efficiency analysis
    pub run_efficiency: bool,
    /// Whether to export JSON results
    pub export_json: bool,
    /// Path to JSON output file
    pub json_path: Option<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            samples: 20_000, // Balanced preset
            detection_trials: 10,
            roc_trials_per_case: 5,
            efficiency_sample_sizes: vec![1_000, 5_000, 10_000, 20_000, 50_000],
            efficiency_trials_per_size: 10,
            run_roc: true,
            run_efficiency: true,
            export_json: false,
            json_path: None,
        }
    }
}

/// Run the full comparison benchmark
pub fn run_comparison_benchmark(config: BenchmarkConfig) {
    println!("\n🔬 Starting Timing Analysis Comparison Benchmark");
    println!("================================================\n");
    println!("Configuration:");
    println!("  Samples per run: {}", config.samples);
    println!("  Detection trials: {}", config.detection_trials);
    println!("  ROC trials per case: {}", config.roc_trials_per_case);
    println!();

    let mut results = BenchmarkResults::new();

    // Initialize detectors
    let timing_oracle = TimingOracleDetector::new();
    let dudect = DudectDetector::new();

    let detectors: Vec<Box<dyn Detector>> = vec![Box::new(timing_oracle), Box::new(dudect)];

    // Run tests on all test cases
    let test_cases = all_test_cases();

    for test_case in &test_cases {
        println!(
            "\n📝 Testing: {} (expected: {})",
            test_case.name(),
            if test_case.expected_leaky() {
                "LEAKY"
            } else {
                "SAFE"
            }
        );
        println!("─────────────────────────────────────────");

        let mut tool_results = Vec::new();

        for detector in &detectors {
            println!("\n  Tool: {}", detector.name());

            // Measure detection rate
            let detection_result = if test_case.expected_leaky() {
                measure_detection_rate(
                    detector.as_ref(),
                    test_case.as_ref(),
                    config.samples,
                    config.detection_trials,
                )
            } else {
                measure_false_positive_rate(
                    detector.as_ref(),
                    test_case.as_ref(),
                    config.samples,
                    config.detection_trials,
                )
            };

            println!(
                "    Detection rate: {:.1}%",
                detection_result.detection_rate * 100.0
            );
            println!("    Avg confidence: {:.3}", detection_result.avg_confidence);
            println!(
                "    Avg time: {:.2}s",
                detection_result.avg_duration.as_secs_f64()
            );

            // Measure sample efficiency (only on leaky cases)
            let efficiency_result = if test_case.expected_leaky() {
                let eff = measure_sample_efficiency(detector.as_ref(), test_case.as_ref());
                if let Some(min_samples) = eff.min_samples_to_detect {
                    println!("    Min samples to detect: {}", min_samples);
                } else {
                    println!("    Min samples to detect: Not detected");
                }
                eff
            } else {
                // For safe cases, don't measure efficiency
                metrics::SampleEfficiencyResult {
                    min_samples_to_detect: None,
                    tested_sample_sizes: vec![],
                    detection_results: vec![],
                }
            };

            tool_results.push(create_tool_result(
                detector.name(),
                &detection_result,
                &efficiency_result,
            ));
        }

        results.add_test_case(TestCaseComparison {
            test_case_name: test_case.name().to_string(),
            expected_leaky: test_case.expected_leaky(),
            results: tool_results,
        });
    }

    // Run ROC curve analysis
    if config.run_roc {
        println!("\n\n📊 Running ROC Curve Analysis");
        println!("─────────────────────────────────────────");

        let leaky_cases = leaky_test_cases();
        let safe_cases = safe_test_cases();

        let leaky_refs: Vec<&dyn TestCase> = leaky_cases.iter().map(|b| b.as_ref()).collect();
        let safe_refs: Vec<&dyn TestCase> = safe_cases.iter().map(|b| b.as_ref()).collect();

        for detector in &detectors {
            println!("\n  Generating ROC curve for: {}", detector.name());

            let thresholds = if detector.name() == "timing-oracle" {
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            } else {
                vec![2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
            };

            let roc = generate_roc_curve(
                detector.as_ref(),
                &leaky_refs,
                &safe_refs,
                &thresholds,
                config.samples,
                config.roc_trials_per_case,
            );

            println!("    AUC: {:.3}", roc.auc);
            results.add_roc_curve(roc);
        }
    }

    // Print terminal report
    println!("\n\n");
    results.print_terminal_report();

    // Export JSON if requested
    if config.export_json {
        let json_path = config
            .json_path
            .unwrap_or_else(|| "comparison_results.json".to_string());

        match results.to_json() {
            Ok(json) => {
                if let Err(e) = std::fs::write(&json_path, json) {
                    eprintln!("Failed to write JSON results: {}", e);
                } else {
                    println!("\n✅ Results exported to: {}", json_path);
                }
            }
            Err(e) => eprintln!("Failed to serialize results to JSON: {}", e),
        }
    }

    println!("\n✅ Benchmark complete!");
}

/// Run detection rate comparison (Section 1)
pub fn run_detection_comparison(config: &BenchmarkConfig) -> BenchmarkResults {
    let mut results = BenchmarkResults::new();

    // Initialize detectors
    let timing_oracle = TimingOracleDetector::new();
    let dudect = DudectDetector::new();

    let detectors: Vec<Box<dyn Detector>> = vec![Box::new(timing_oracle), Box::new(dudect)];

    // Run tests on all test cases
    let test_cases = all_test_cases();

    for test_case in &test_cases {
        eprintln!(
            "\n📝 Testing: {} (expected: {})",
            test_case.name(),
            if test_case.expected_leaky() {
                "LEAKY"
            } else {
                "SAFE"
            }
        );

        let mut tool_results = Vec::new();

        for detector in &detectors {
            eprintln!("  Tool: {}", detector.name());

            // Measure detection rate
            let detection_result = if test_case.expected_leaky() {
                measure_detection_rate(
                    detector.as_ref(),
                    test_case.as_ref(),
                    config.samples,
                    config.detection_trials,
                )
            } else {
                measure_false_positive_rate(
                    detector.as_ref(),
                    test_case.as_ref(),
                    config.samples,
                    config.detection_trials,
                )
            };

            eprintln!(
                "    Detection rate: {:.1}%",
                detection_result.detection_rate * 100.0
            );
            eprintln!("    Avg samples: {}", detection_result.avg_samples_used);
            eprintln!(
                "    Time/sample: {:.1}ns",
                detection_result.avg_time_per_sample.as_nanos()
            );

            // Measure sample efficiency (only on leaky cases, for min_samples_to_detect field)
            let efficiency_result = if test_case.expected_leaky() {
                measure_sample_efficiency(detector.as_ref(), test_case.as_ref())
            } else {
                metrics::SampleEfficiencyResult {
                    min_samples_to_detect: None,
                    tested_sample_sizes: vec![],
                    detection_results: vec![],
                }
            };

            tool_results.push(create_tool_result(
                detector.name(),
                &detection_result,
                &efficiency_result,
            ));
        }

        results.add_test_case(TestCaseComparison {
            test_case_name: test_case.name().to_string(),
            expected_leaky: test_case.expected_leaky(),
            results: tool_results,
        });
    }

    results
}

/// Run ROC curve analysis (Section 2)
pub fn run_roc_analysis(config: &BenchmarkConfig) -> Vec<metrics::RocCurveResult> {
    let mut results = Vec::new();

    // Initialize detectors
    let timing_oracle = TimingOracleDetector::new();
    let dudect = DudectDetector::new();

    let detectors: Vec<Box<dyn Detector>> = vec![Box::new(timing_oracle), Box::new(dudect)];

    let leaky_cases = leaky_test_cases();
    let safe_cases = safe_test_cases();

    let leaky_refs: Vec<&dyn TestCase> = leaky_cases.iter().map(|b| b.as_ref()).collect();
    let safe_refs: Vec<&dyn TestCase> = safe_cases.iter().map(|b| b.as_ref()).collect();

    for detector in &detectors {
        eprintln!("\n  Generating ROC curve for: {}", detector.name());

        let thresholds = if detector.name() == "timing-oracle" {
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        } else {
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        };

        let roc = generate_roc_curve(
            detector.as_ref(),
            &leaky_refs,
            &safe_refs,
            &thresholds,
            config.samples,
            config.roc_trials_per_case,
        );

        eprintln!("    AUC: {:.3}", roc.auc);
        results.push(roc);
    }

    results
}

/// Run sample efficiency analysis (Section 3)
pub fn run_efficiency_analysis(
    config: &BenchmarkConfig,
) -> HashMap<String, HashMap<String, SampleEfficiencyStats>> {
    let mut results: HashMap<String, HashMap<String, SampleEfficiencyStats>> = HashMap::new();

    // Initialize detectors
    let timing_oracle = TimingOracleDetector::new();
    let dudect = DudectDetector::new();

    let detectors: Vec<(&str, Box<dyn Detector>)> = vec![
        ("timing-oracle", Box::new(timing_oracle)),
        ("dudect-bencher", Box::new(dudect)),
    ];

    // Only test on known-leaky cases
    let test_cases = leaky_test_cases();

    for (detector_name, detector) in &detectors {
        eprintln!("\n  Analyzing sample efficiency for: {}", detector_name);

        let mut test_case_stats: HashMap<String, SampleEfficiencyStats> = HashMap::new();

        for test_case in &test_cases {
            let stats = measure_sample_efficiency_stats(
                detector.as_ref(),
                test_case.as_ref(),
                &config.efficiency_sample_sizes,
                config.efficiency_trials_per_size,
            );

            test_case_stats.insert(test_case.name().to_string(), stats);
        }

        results.insert(detector_name.to_string(), test_case_stats);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_comparison() {
        let config = BenchmarkConfig {
            samples: 5_000,
            detection_trials: 2,
            roc_trials_per_case: 2,
            run_roc: false,
            export_json: false,
            json_path: None,
        };

        run_comparison_benchmark(config);
    }
}
