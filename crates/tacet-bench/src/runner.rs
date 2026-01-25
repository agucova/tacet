//! Benchmark runner infrastructure.
//!
//! This module provides the main runner for comparing tacet against
//! other timing side-channel detection tools. It orchestrates data generation,
//! tool execution, and result aggregation.
//!
//! # Benchmarks
//!
//! - **FPR (False Positive Rate)**: Measure false positives on null-effect data
//! - **Power**: Measure detection rate on known-effect data

use crate::adapters::{ToolAdapter, ToolResult};
use crate::{
    generate_dataset, load_interleaved_csv, EffectType, GeneratedDataset, SyntheticConfig,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Type alias for progress callback to reduce type complexity.
type ProgressCallback = Box<dyn Fn(&str) + Send + Sync>;

// =============================================================================
// Result types
// =============================================================================

/// Results from a False Positive Rate benchmark.
#[derive(Debug, Clone)]
pub struct FprResults {
    /// Number of false positives per tool.
    pub false_positives: HashMap<String, usize>,
    /// Total number of datasets tested.
    pub total: usize,
    /// Individual results per tool per dataset.
    pub details: HashMap<String, Vec<ToolResult>>,
}

impl FprResults {
    /// Get the FPR for a specific tool.
    pub fn fpr(&self, tool: &str) -> f64 {
        let fp = *self.false_positives.get(tool).unwrap_or(&0);
        fp as f64 / self.total.max(1) as f64
    }

    /// Get all FPRs as a sorted list.
    pub fn all_fprs(&self) -> Vec<(&str, f64)> {
        let mut results: Vec<_> = self
            .false_positives
            .iter()
            .map(|(tool, &fp)| (tool.as_str(), fp as f64 / self.total.max(1) as f64))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

/// Results from a Power (detection rate) benchmark.
#[derive(Debug, Clone)]
pub struct PowerResults {
    /// Number of true positives (correctly detected leaks) per tool.
    pub true_positives: HashMap<String, usize>,
    /// Total number of datasets with known leaks.
    pub total: usize,
    /// Effect type used in this benchmark.
    pub effect_type: String,
    /// Individual results per tool per dataset.
    pub details: HashMap<String, Vec<ToolResult>>,
}

impl PowerResults {
    /// Get the power (detection rate) for a specific tool.
    pub fn power(&self, tool: &str) -> f64 {
        let tp = *self.true_positives.get(tool).unwrap_or(&0);
        tp as f64 / self.total.max(1) as f64
    }

    /// Get all powers as a sorted list (highest first).
    pub fn all_powers(&self) -> Vec<(&str, f64)> {
        let mut results: Vec<_> = self
            .true_positives
            .iter()
            .map(|(tool, &tp)| (tool.as_str(), tp as f64 / self.total.max(1) as f64))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

/// Combined benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// FPR results by configuration name.
    pub fpr_results: HashMap<String, FprResults>,
    /// Power results by configuration name.
    pub power_results: HashMap<String, PowerResults>,
    /// Total benchmark time.
    pub total_time: Duration,
}

// =============================================================================
// Benchmark runner
// =============================================================================

/// Main benchmark runner.
///
/// Coordinates running multiple timing analysis tools against synthetic
/// and/or real-world datasets.
pub struct BenchmarkRunner {
    /// List of tools to compare.
    tools: Vec<Box<dyn ToolAdapter>>,
    /// Base directory for synthetic data (if pre-generated).
    synthetic_dir: Option<PathBuf>,
    /// Number of datasets to use per configuration.
    datasets_per_config: usize,
    /// Progress callback.
    progress_callback: Option<ProgressCallback>,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner.
    pub fn new(tools: Vec<Box<dyn ToolAdapter>>) -> Self {
        Self {
            tools,
            synthetic_dir: None,
            datasets_per_config: 100,
            progress_callback: None,
        }
    }

    /// Set the directory containing pre-generated synthetic data.
    pub fn synthetic_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.synthetic_dir = Some(path.into());
        self
    }

    /// Set number of datasets per configuration.
    pub fn datasets_per_config(mut self, n: usize) -> Self {
        self.datasets_per_config = n;
        self
    }

    /// Set progress callback.
    pub fn on_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    fn log_progress(&self, msg: &str) {
        if let Some(ref callback) = self.progress_callback {
            callback(msg);
        }
    }

    // =========================================================================
    // FPR Benchmark
    // =========================================================================

    /// Run FPR benchmark on null-effect datasets.
    ///
    /// Returns the false positive rate for each tool.
    pub fn run_fpr_benchmark(
        &self,
        config_name: &str,
        base_config: &SyntheticConfig,
    ) -> FprResults {
        assert!(
            matches!(base_config.effect, EffectType::Null),
            "FPR benchmark requires null effect"
        );

        let mut false_positives: HashMap<String, usize> = HashMap::new();
        let mut details: HashMap<String, Vec<ToolResult>> = HashMap::new();

        // Initialize counters
        for tool in &self.tools {
            false_positives.insert(tool.name().to_string(), 0);
            details.insert(tool.name().to_string(), Vec::new());
        }

        self.log_progress(&format!(
            "Running FPR benchmark: {} ({} datasets)",
            config_name, self.datasets_per_config
        ));

        for i in 0..self.datasets_per_config {
            // Generate or load dataset
            let dataset = self.get_dataset(config_name, base_config, i);

            for tool in &self.tools {
                let result = tool.analyze(&dataset);

                if result.detected_leak {
                    *false_positives.get_mut(tool.name()).unwrap() += 1;
                }

                details.get_mut(tool.name()).unwrap().push(result);
            }

            if (i + 1) % 10 == 0 {
                self.log_progress(&format!(
                    "  {}/{} datasets complete",
                    i + 1,
                    self.datasets_per_config
                ));
            }
        }

        FprResults {
            false_positives,
            total: self.datasets_per_config,
            details,
        }
    }

    // =========================================================================
    // Power Benchmark
    // =========================================================================

    /// Run power benchmark on datasets with known effects.
    ///
    /// Returns the detection rate for each tool.
    pub fn run_power_benchmark(
        &self,
        config_name: &str,
        base_config: &SyntheticConfig,
    ) -> PowerResults {
        assert!(
            !matches!(base_config.effect, EffectType::Null),
            "Power benchmark requires non-null effect"
        );

        let mut true_positives: HashMap<String, usize> = HashMap::new();
        let mut details: HashMap<String, Vec<ToolResult>> = HashMap::new();

        // Initialize counters
        for tool in &self.tools {
            true_positives.insert(tool.name().to_string(), 0);
            details.insert(tool.name().to_string(), Vec::new());
        }

        let effect_name = base_config.effect.name();
        self.log_progress(&format!(
            "Running power benchmark: {} ({} datasets, effect={})",
            config_name, self.datasets_per_config, effect_name
        ));

        for i in 0..self.datasets_per_config {
            let dataset = self.get_dataset(config_name, base_config, i);

            for tool in &self.tools {
                let result = tool.analyze(&dataset);

                if result.detected_leak {
                    *true_positives.get_mut(tool.name()).unwrap() += 1;
                }

                details.get_mut(tool.name()).unwrap().push(result);
            }

            if (i + 1) % 10 == 0 {
                self.log_progress(&format!(
                    "  {}/{} datasets complete",
                    i + 1,
                    self.datasets_per_config
                ));
            }
        }

        PowerResults {
            true_positives,
            total: self.datasets_per_config,
            effect_type: effect_name,
            details,
        }
    }

    // =========================================================================
    // Full benchmark suite
    // =========================================================================

    /// Run the full benchmark suite.
    ///
    /// Runs FPR and power benchmarks on standard configurations.
    pub fn run_full_suite(&self, configs: &[(String, SyntheticConfig)]) -> BenchmarkReport {
        let start = Instant::now();
        let mut fpr_results = HashMap::new();
        let mut power_results = HashMap::new();

        for (name, config) in configs {
            match config.effect {
                EffectType::Null => {
                    let result = self.run_fpr_benchmark(name, config);
                    fpr_results.insert(name.clone(), result);
                }
                _ => {
                    let power = self.run_power_benchmark(name, config);
                    power_results.insert(name.clone(), power);
                }
            }
        }

        BenchmarkReport {
            fpr_results,
            power_results,
            total_time: start.elapsed(),
        }
    }

    // =========================================================================
    // Dataset loading
    // =========================================================================

    /// Get a dataset (generate or load from disk).
    fn get_dataset(
        &self,
        config_name: &str,
        base_config: &SyntheticConfig,
        index: usize,
    ) -> GeneratedDataset {
        // Try to load from disk first
        if let Some(ref dir) = self.synthetic_dir {
            let interleaved_path = dir
                .join(config_name)
                .join(format!("{}_interleaved.csv", index));
            if interleaved_path.exists() {
                if let Ok(interleaved) = load_interleaved_csv(&interleaved_path) {
                    let blocked = crate::split_interleaved(&interleaved);
                    return GeneratedDataset {
                        interleaved,
                        blocked,
                    };
                }
            }
        }

        // Generate fresh dataset
        let config = SyntheticConfig {
            seed: base_config.seed.wrapping_add(index as u64),
            ..base_config.clone()
        };
        generate_dataset(&config)
    }
}

// =============================================================================
// Report formatting
// =============================================================================

impl BenchmarkReport {
    /// Generate a markdown report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str("# tacet Benchmark Results\n\n");
        md.push_str(&format!(
            "Total benchmark time: {:.1} seconds\n\n",
            self.total_time.as_secs_f64()
        ));

        // FPR Results
        if !self.fpr_results.is_empty() {
            md.push_str("## False Positive Rate (Null Hypothesis)\n\n");
            md.push_str("| Configuration | ");
            let tools: Vec<_> = self
                .fpr_results
                .values()
                .next()
                .map(|r| r.false_positives.keys().collect())
                .unwrap_or_default();
            for tool in &tools {
                md.push_str(&format!("{} | ", tool));
            }
            md.push('\n');
            md.push('|');
            md.push_str(&"-|".repeat(tools.len() + 1));
            md.push('\n');

            for (name, results) in &self.fpr_results {
                md.push_str(&format!("| {} | ", name));
                for tool in &tools {
                    let fpr = results.fpr(tool) * 100.0;
                    md.push_str(&format!("{:.1}% | ", fpr));
                }
                md.push('\n');
            }
            md.push('\n');
        }

        // Power Results
        if !self.power_results.is_empty() {
            md.push_str("## Statistical Power (Leak Detection)\n\n");
            md.push_str("| Configuration | Effect | ");
            let tools: Vec<_> = self
                .power_results
                .values()
                .next()
                .map(|r| r.true_positives.keys().collect())
                .unwrap_or_default();
            for tool in &tools {
                md.push_str(&format!("{} | ", tool));
            }
            md.push('\n');
            md.push('|');
            md.push_str(&"-|".repeat(tools.len() + 2));
            md.push('\n');

            for (name, results) in &self.power_results {
                md.push_str(&format!("| {} | {} | ", name, results.effect_type));
                for tool in &tools {
                    let power = results.power(tool) * 100.0;
                    md.push_str(&format!("{:.1}% | ", power));
                }
                md.push('\n');
            }
            md.push('\n');
        }

        md
    }

    /// Write report to file.
    pub fn write_to_file(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_markdown())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::StubAdapter;

    #[test]
    fn test_fpr_results() {
        let mut fp = HashMap::new();
        fp.insert("tool1".to_string(), 5);
        fp.insert("tool2".to_string(), 10);

        let results = FprResults {
            false_positives: fp,
            total: 100,
            details: HashMap::new(),
        };

        assert_eq!(results.fpr("tool1"), 0.05);
        assert_eq!(results.fpr("tool2"), 0.10);
        assert_eq!(results.fpr("unknown"), 0.0);
    }

    #[test]
    fn test_power_results() {
        let mut tp = HashMap::new();
        tp.insert("tool1".to_string(), 95);
        tp.insert("tool2".to_string(), 80);

        let results = PowerResults {
            true_positives: tp,
            total: 100,
            effect_type: "shift-5pct".to_string(),
            details: HashMap::new(),
        };

        assert_eq!(results.power("tool1"), 0.95);
        assert_eq!(results.power("tool2"), 0.80);
    }

    #[test]
    fn test_benchmark_runner_small() {
        let tools: Vec<Box<dyn ToolAdapter>> = vec![
            Box::new(StubAdapter::new("stub1")),
            Box::new(StubAdapter::new("stub2")),
        ];

        let runner = BenchmarkRunner::new(tools).datasets_per_config(3);

        let config = SyntheticConfig {
            samples_per_class: 100,
            effect: EffectType::Null,
            seed: 42,
            ..Default::default()
        };

        let results = runner.run_fpr_benchmark("test-null", &config);
        assert_eq!(results.total, 3);
        // Stub adapters never detect leaks
        assert_eq!(results.fpr("stub1"), 0.0);
        assert_eq!(results.fpr("stub2"), 0.0);
    }

    #[test]
    fn test_report_markdown() {
        let mut fpr = HashMap::new();
        fpr.insert("tool1".to_string(), 5);

        let fpr_results = FprResults {
            false_positives: fpr,
            total: 100,
            details: HashMap::new(),
        };

        let mut fpr_map = HashMap::new();
        fpr_map.insert("15k-same-xy".to_string(), fpr_results);

        let report = BenchmarkReport {
            fpr_results: fpr_map,
            power_results: HashMap::new(),
            total_time: Duration::from_secs(60),
        };

        let md = report.to_markdown();
        assert!(md.contains("False Positive Rate"));
        assert!(md.contains("15k-same-xy"));
    }
}
