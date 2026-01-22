//! Benchmark sweep infrastructure for comprehensive tool comparison.
//!
//! This module provides:
//! - `SweepConfig`: Configuration for benchmark sweeps with presets
//! - `SweepRunner`: Parallel execution of benchmarks across configurations
//! - `SweepResults`: Collection and aggregation of benchmark results
//!
//! # Example
//!
//! ```ignore
//! use timing_oracle_bench::sweep::{SweepConfig, SweepRunner};
//! use timing_oracle_bench::{DudectAdapter, KsTestAdapter};
//!
//! let config = SweepConfig::quick();
//! let runner = SweepRunner::new(vec![
//!     Box::new(DudectAdapter::default()),
//!     Box::new(KsTestAdapter::default()),
//! ]);
//!
//! let results = runner.run(&config, |progress| {
//!     println!("Progress: {:.1}%", progress * 100.0);
//! });
//!
//! println!("{}", results.to_markdown());
//! ```

use crate::adapters::ToolAdapter;
use crate::checkpoint::{IncrementalCsvWriter, WorkItemKey};
use crate::realistic::{collect_realistic_dataset, realistic_to_generated, RealisticConfig};
use crate::synthetic::{generate_benchmark_dataset, BenchmarkConfig, EffectPattern, NoiseModel};
use crate::GeneratedDataset;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use timing_oracle::BenchmarkEffect;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Preset levels for benchmark detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkPreset {
    /// Quick check: ~5 minutes, minimal coverage
    Quick,
    /// Medium detail: ~30 minutes, good coverage
    Medium,
    /// Thorough: ~3 hours, comprehensive coverage
    Thorough,
    /// Fine threshold: Focused on graduation around timing-oracle's 100ns threshold
    /// Inspired by SILENT paper (arXiv:2504.19821) heatmap methodology
    FineThreshold,
}

impl BenchmarkPreset {
    /// Get a short name for this preset
    pub fn name(&self) -> &'static str {
        match self {
            BenchmarkPreset::Quick => "quick",
            BenchmarkPreset::Medium => "medium",
            BenchmarkPreset::Thorough => "thorough",
            BenchmarkPreset::FineThreshold => "fine-threshold",
        }
    }
}

/// Configuration for a benchmark sweep.
///
/// Defines what combinations of effect sizes, patterns, and noise models to test.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// Preset level (informational)
    pub preset: BenchmarkPreset,
    /// Number of samples per class for each dataset
    pub samples_per_class: usize,
    /// Number of datasets to generate per configuration point
    pub datasets_per_point: usize,
    /// Effect size multipliers (relative to σ)
    pub effect_multipliers: Vec<f64>,
    /// Effect patterns to test
    pub effect_patterns: Vec<EffectPattern>,
    /// Noise models to test
    pub noise_models: Vec<NoiseModel>,
    /// Use realistic timing (actual measurements) instead of synthetic generation
    pub use_realistic: bool,
    /// Base operation time in nanoseconds (for realistic mode)
    pub realistic_base_ns: u64,
}

impl SweepConfig {
    /// Quick preset: minimal coverage for fast feedback (~5 min)
    ///
    /// Scaled similar to SILENT paper (μ ∈ [0, 1.0] with σ=10 → [0, 0.1σ] in our terms).
    /// With σ = 100μs: 0.02σ = 2μs, 0.05σ = 5μs, 0.1σ = 10μs
    ///
    /// - Effect sizes: 0, 2μs, 5μs, 10μs (0-0.1σ range)
    /// - 2 patterns: Null, Shift
    /// - 3 noise models: IID, AR(0.5), AR(-0.5)
    /// - 20 datasets per point
    pub fn quick() -> Self {
        Self {
            preset: BenchmarkPreset::Quick,
            samples_per_class: 5_000,
            datasets_per_point: 20,
            // SILENT-like scale: [0, 0.1σ] = [0, 10μs]
            // 0.02σ = 2μs, 0.05σ = 5μs, 0.1σ = 10μs
            effect_multipliers: vec![0.0, 0.02, 0.05, 0.1],
            effect_patterns: vec![EffectPattern::Null, EffectPattern::Shift],
            noise_models: vec![
                NoiseModel::IID,
                NoiseModel::AR1 { phi: 0.5 },
                NoiseModel::AR1 { phi: -0.5 },
            ],
            use_realistic: false,
            realistic_base_ns: 1000, // 1μs default
        }
    }

    /// Medium preset: good coverage for development (~30 min)
    ///
    /// Scaled similar to SILENT paper with finer granularity.
    /// With σ = 100μs: range [0, 0.1σ] = [0, 10μs] with 6 points
    ///
    /// - Effect sizes: 0, 1μs, 2μs, 4μs, 6μs, 8μs, 10μs (decile-like in 0-0.1σ)
    /// - 4 patterns: Null, Shift, Tail, Bimodal
    /// - 5 noise models: IID, AR(±0.3), AR(±0.6)
    /// - 50 datasets per point
    pub fn medium() -> Self {
        Self {
            preset: BenchmarkPreset::Medium,
            samples_per_class: 10_000,
            datasets_per_point: 50,
            // SILENT-like scale with finer granularity
            // [0, 0.1σ] = [0, 10μs] with ~2μs steps
            effect_multipliers: vec![0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
            effect_patterns: vec![
                EffectPattern::Null,
                EffectPattern::Shift,
                EffectPattern::Tail,
                EffectPattern::bimodal_default(),
            ],
            noise_models: vec![
                NoiseModel::AR1 { phi: -0.6 },
                NoiseModel::AR1 { phi: -0.3 },
                NoiseModel::IID,
                NoiseModel::AR1 { phi: 0.3 },
                NoiseModel::AR1 { phi: 0.6 },
            ],
            use_realistic: false,
            realistic_base_ns: 1000,
        }
    }

    /// Thorough preset: comprehensive coverage for publication (~3 hours)
    ///
    /// Scaled similar to SILENT paper with full granularity.
    /// SILENT uses μ ∈ [0, 1.0] with deciles and Φ ∈ [-0.9, 0.9].
    /// With our σ = 100μs: [0, 0.1σ] = [0, 10μs] with decile granularity
    ///
    /// - Effect sizes: 11 points in [0, 0.1σ] = [0, 10μs] (deciles)
    /// - 6 patterns: Null, Shift, Tail, Variance, Bimodal, Quantized
    /// - 9 noise models: IID, AR(±0.2), AR(±0.4), AR(±0.6), AR(±0.8)
    /// - 100 datasets per point
    pub fn thorough() -> Self {
        Self {
            preset: BenchmarkPreset::Thorough,
            samples_per_class: 20_000,
            datasets_per_point: 100,
            // SILENT-like scale: [0, 0.1σ] with deciles (11 points)
            // 0.01σ = 1μs increments up to 0.1σ = 10μs
            effect_multipliers: vec![
                0.0,    // 0μs
                0.01,   // 1μs
                0.02,   // 2μs
                0.03,   // 3μs
                0.04,   // 4μs
                0.05,   // 5μs (Δ in SILENT = 0.5)
                0.06,   // 6μs
                0.07,   // 7μs
                0.08,   // 8μs
                0.09,   // 9μs
                0.1,    // 10μs (μ=1.0 in SILENT)
            ],
            effect_patterns: vec![
                EffectPattern::Null,
                EffectPattern::Shift,
                EffectPattern::Tail,
                EffectPattern::Variance,
                EffectPattern::bimodal_default(),
                EffectPattern::quantized_default(),
            ],
            // Full autocorrelation range like SILENT: Φ ∈ [-0.8, 0.8]
            noise_models: vec![
                NoiseModel::AR1 { phi: -0.8 },
                NoiseModel::AR1 { phi: -0.6 },
                NoiseModel::AR1 { phi: -0.4 },
                NoiseModel::AR1 { phi: -0.2 },
                NoiseModel::IID,
                NoiseModel::AR1 { phi: 0.2 },
                NoiseModel::AR1 { phi: 0.4 },
                NoiseModel::AR1 { phi: 0.6 },
                NoiseModel::AR1 { phi: 0.8 },
            ],
            use_realistic: false,
            realistic_base_ns: 1000,
        }
    }

    /// Fine threshold preset: Focused graduation around timing-oracle's 100ns threshold
    ///
    /// Inspired by SILENT paper (arXiv:2504.19821) heatmap methodology.
    /// Key differences from other presets:
    /// - Fine-grained effect sizes around the 100ns threshold (0-500ns range)
    /// - Both positive AND negative autocorrelation (SILENT tests Φ ∈ [-0.9, 0.9])
    /// - Focuses on shift pattern only (most informative for threshold analysis)
    /// - 50 datasets per point for tight confidence intervals
    ///
    /// Estimated runtime: ~2-4 hours with 9 tools
    ///
    /// Effect size mapping (σ = 100μs = 100,000ns):
    /// - 0.0001σ = 10ns
    /// - 0.0005σ = 50ns
    /// - 0.001σ = 100ns (timing-oracle threshold)
    /// - 0.002σ = 200ns
    /// - 0.005σ = 500ns
    pub fn fine_threshold() -> Self {
        Self {
            preset: BenchmarkPreset::FineThreshold,
            samples_per_class: 10_000,
            datasets_per_point: 50,
            // Fine-grained effect sizes around 100ns threshold
            // σ = 100,000ns, so these map to:
            // 0ns, 10ns, 20ns, 40ns, 60ns, 80ns, 100ns, 120ns, 150ns, 200ns, 300ns, 500ns
            effect_multipliers: vec![
                0.0,      // 0ns (FPR test)
                0.0001,   // 10ns
                0.0002,   // 20ns
                0.0004,   // 40ns
                0.0006,   // 60ns
                0.0008,   // 80ns
                0.001,    // 100ns (threshold)
                0.0012,   // 120ns
                0.0015,   // 150ns
                0.002,    // 200ns
                0.003,    // 300ns
                0.005,    // 500ns
            ],
            effect_patterns: vec![EffectPattern::Shift],
            // Both positive AND negative autocorrelation (like SILENT)
            noise_models: vec![
                NoiseModel::AR1 { phi: -0.8 },
                NoiseModel::AR1 { phi: -0.6 },
                NoiseModel::AR1 { phi: -0.4 },
                NoiseModel::AR1 { phi: -0.2 },
                NoiseModel::IID,
                NoiseModel::AR1 { phi: 0.2 },
                NoiseModel::AR1 { phi: 0.4 },
                NoiseModel::AR1 { phi: 0.6 },
                NoiseModel::AR1 { phi: 0.8 },
            ],
            use_realistic: false,
            realistic_base_ns: 1000,
        }
    }

    /// Calculate total number of configuration points
    pub fn total_points(&self) -> usize {
        self.effect_multipliers.len() * self.effect_patterns.len() * self.noise_models.len()
    }

    /// Calculate total number of datasets to generate
    pub fn total_datasets(&self) -> usize {
        self.total_points() * self.datasets_per_point
    }

    /// Iterate over all configuration points
    pub fn iter_configs(&self) -> impl Iterator<Item = (EffectPattern, f64, NoiseModel)> + '_ {
        self.effect_patterns.iter().flat_map(move |&pattern| {
            self.effect_multipliers.iter().flat_map(move |&mult| {
                self.noise_models
                    .iter()
                    .map(move |&noise| (pattern, mult, noise))
            })
        })
    }
}

/// Single benchmark result from one tool on one dataset.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Tool name
    pub tool: String,
    /// Preset used
    pub preset: String,
    /// Effect pattern
    pub effect_pattern: String,
    /// Effect size (σ multiplier)
    pub effect_sigma_mult: f64,
    /// Noise model
    pub noise_model: String,
    /// Dataset ID within this config point
    pub dataset_id: usize,
    /// Samples per class
    pub samples_per_class: usize,
    /// Whether leak was detected
    pub detected: bool,
    /// Test statistic (tool-specific)
    pub statistic: Option<f64>,
    /// P-value if available
    pub p_value: Option<f64>,
    /// Analysis time in milliseconds
    pub time_ms: u64,
    /// Samples actually used (for adaptive tools)
    pub samples_used: Option<usize>,
}

/// Collection of all benchmark results.
#[derive(Debug, Clone)]
pub struct SweepResults {
    /// All individual results
    pub results: Vec<BenchmarkResult>,
    /// Configuration used
    pub config: SweepConfig,
    /// Total execution time
    pub total_time: Duration,
}

impl SweepResults {
    /// Create new empty results
    pub fn new(config: SweepConfig) -> Self {
        Self {
            results: Vec::new(),
            config,
            total_time: Duration::ZERO,
        }
    }

    /// Add a result
    pub fn push(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Extend with multiple results
    pub fn extend(&mut self, results: impl IntoIterator<Item = BenchmarkResult>) {
        self.results.extend(results);
    }

    /// Get results for a specific tool
    pub fn for_tool(&self, tool: &str) -> Vec<&BenchmarkResult> {
        self.results.iter().filter(|r| r.tool == tool).collect()
    }

    /// Get unique tool names
    pub fn tools(&self) -> Vec<String> {
        let mut tools: Vec<String> = self.results.iter().map(|r| r.tool.clone()).collect();
        tools.sort();
        tools.dedup();
        tools
    }

    /// Calculate detection rate for a specific configuration
    pub fn detection_rate(
        &self,
        tool: &str,
        pattern: &str,
        effect_mult: f64,
        noise: &str,
    ) -> Option<f64> {
        let matching: Vec<&BenchmarkResult> = self
            .results
            .iter()
            .filter(|r| {
                r.tool == tool
                    && r.effect_pattern == pattern
                    && (r.effect_sigma_mult - effect_mult).abs() < 0.001
                    && r.noise_model == noise
            })
            .collect();

        if matching.is_empty() {
            return None;
        }

        let detected_count = matching.iter().filter(|r| r.detected).count();
        Some(detected_count as f64 / matching.len() as f64)
    }

    /// Calculate Wilson confidence interval for a proportion
    fn wilson_ci(successes: usize, total: usize, z: f64) -> (f64, f64) {
        if total == 0 {
            return (0.0, 1.0);
        }

        let n = total as f64;
        let p_hat = successes as f64 / n;
        let z2 = z * z;

        let center = (p_hat + z2 / (2.0 * n)) / (1.0 + z2 / n);
        let margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n).sqrt() / (1.0 + z2 / n);

        ((center - margin).max(0.0), (center + margin).min(1.0))
    }

    /// Get summary statistics for all configuration points
    pub fn summarize(&self) -> Vec<PointSummary> {
        let mut summaries = Vec::new();

        for tool in self.tools() {
            for (pattern, mult, noise) in self.config.iter_configs() {
                // In realistic mode, noise_model has "-realistic" suffix
                let expected_noise = if self.config.use_realistic {
                    format!("{}-realistic", noise.name())
                } else {
                    noise.name()
                };

                let matching: Vec<&BenchmarkResult> = self
                    .results
                    .iter()
                    .filter(|r| {
                        r.tool == tool
                            && r.effect_pattern == pattern.name()
                            && (r.effect_sigma_mult - mult).abs() < 0.001
                            && r.noise_model == expected_noise
                    })
                    .collect();

                if matching.is_empty() {
                    continue;
                }

                let detected_count = matching.iter().filter(|r| r.detected).count();
                let n = matching.len();
                let rate = detected_count as f64 / n as f64;
                let (ci_low, ci_high) = Self::wilson_ci(detected_count, n, 1.96);

                let mut times: Vec<u64> = matching.iter().map(|r| r.time_ms).collect();
                times.sort();
                let median_time_ms = times[times.len() / 2];

                let samples_used: Vec<usize> = matching
                    .iter()
                    .filter_map(|r| r.samples_used)
                    .collect();
                let median_samples = if samples_used.is_empty() {
                    None
                } else {
                    let mut sorted = samples_used.clone();
                    sorted.sort();
                    Some(sorted[sorted.len() / 2])
                };

                summaries.push(PointSummary {
                    tool: tool.clone(),
                    effect_pattern: pattern.name().to_string(),
                    effect_sigma_mult: mult,
                    noise_model: expected_noise,
                    n_datasets: n,
                    detection_rate: rate,
                    ci_low,
                    ci_high,
                    median_time_ms,
                    median_samples,
                });
            }
        }

        summaries
    }
}

/// Summary statistics for one configuration point.
#[derive(Debug, Clone)]
pub struct PointSummary {
    /// Tool name
    pub tool: String,
    /// Effect pattern
    pub effect_pattern: String,
    /// Effect size (σ multiplier)
    pub effect_sigma_mult: f64,
    /// Noise model
    pub noise_model: String,
    /// Number of datasets tested
    pub n_datasets: usize,
    /// Detection rate (FPR when mult=0, Power when mult>0)
    pub detection_rate: f64,
    /// Wilson 95% CI lower bound
    pub ci_low: f64,
    /// Wilson 95% CI upper bound
    pub ci_high: f64,
    /// Median analysis time in ms
    pub median_time_ms: u64,
    /// Median samples used (for adaptive tools)
    pub median_samples: Option<usize>,
}

/// Progress callback type
pub type ProgressCallback = Box<dyn Fn(f64, &str) + Send + Sync>;

/// Benchmark sweep runner.
///
/// Executes benchmark configurations across multiple tools in parallel.
pub struct SweepRunner {
    /// Tools to benchmark
    tools: Vec<Box<dyn ToolAdapter>>,
}

impl SweepRunner {
    /// Create a new sweep runner with the given tools
    pub fn new(tools: Vec<Box<dyn ToolAdapter>>) -> Self {
        Self { tools }
    }

    /// Get the number of tools
    pub fn num_tools(&self) -> usize {
        self.tools.len()
    }

    /// Get the tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Run the benchmark sweep with optional progress callback.
    ///
    /// # Arguments
    /// * `config` - Sweep configuration
    /// * `progress` - Optional callback receiving (progress_fraction, current_task)
    ///
    /// # Returns
    /// Aggregated results from all tools and configurations
    ///
    /// # Parallelization Strategy
    /// Uses (dataset × tool) level parallelism for maximum CPU utilization.
    /// Each work item is a single (dataset, tool) pair, allowing tools to run
    /// in parallel across different datasets.
    pub fn run<F>(&self, config: &SweepConfig, mut progress: F) -> SweepResults
    where
        F: FnMut(f64, &str),
    {
        use std::sync::Arc;

        let start = Instant::now();
        let mut results = SweepResults::new(config.clone());

        let total_work = config.total_datasets() * self.tools.len();
        let completed = AtomicUsize::new(0);

        // Step 1: Collect dataset configs
        let dataset_configs: Vec<(EffectPattern, f64, NoiseModel, usize)> = config
            .iter_configs()
            .flat_map(|(pattern, mult, noise)| {
                (0..config.datasets_per_point).map(move |dataset_id| (pattern, mult, noise, dataset_id))
            })
            .collect();

        // Step 2: Generate all datasets first (parallel for synthetic, sequential for realistic)
        progress(0.0, "Generating datasets...");

        #[cfg(feature = "parallel")]
        let datasets: Vec<(EffectPattern, f64, NoiseModel, usize, GeneratedDataset)> = if config.use_realistic {
            // Sequential for realistic mode (PMU timer requires exclusive access)
            dataset_configs
                .iter()
                .map(|&(pattern, mult, noise, dataset_id)| {
                    let dataset = self.generate_dataset(config, pattern, mult, noise, dataset_id);
                    (pattern, mult, noise, dataset_id, dataset)
                })
                .collect()
        } else {
            // Parallel dataset generation for synthetic mode
            dataset_configs
                .par_iter()
                .map(|&(pattern, mult, noise, dataset_id)| {
                    let dataset = self.generate_dataset(config, pattern, mult, noise, dataset_id);
                    (pattern, mult, noise, dataset_id, dataset)
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let datasets: Vec<(EffectPattern, f64, NoiseModel, usize, GeneratedDataset)> = dataset_configs
            .iter()
            .map(|&(pattern, mult, noise, dataset_id)| {
                let dataset = self.generate_dataset(config, pattern, mult, noise, dataset_id);
                (pattern, mult, noise, dataset_id, dataset)
            })
            .collect();

        // Wrap datasets in Arc for sharing across threads
        let datasets: Vec<(EffectPattern, f64, NoiseModel, usize, Arc<GeneratedDataset>)> = datasets
            .into_iter()
            .map(|(p, m, n, id, d)| (p, m, n, id, Arc::new(d)))
            .collect();

        progress(0.05, "Running tools...");

        // Step 3: Create (dataset × tool) work items
        let work_items: Vec<(usize, usize)> = (0..datasets.len())
            .flat_map(|dataset_idx| {
                (0..self.tools.len()).map(move |tool_idx| (dataset_idx, tool_idx))
            })
            .collect();

        // Step 4: Process all (dataset × tool) pairs in parallel
        #[cfg(feature = "parallel")]
        let all_results: Vec<BenchmarkResult> = if config.use_realistic {
            // Sequential for realistic mode
            work_items
                .iter()
                .map(|&(dataset_idx, tool_idx)| {
                    let (pattern, mult, noise, dataset_id, dataset) = &datasets[dataset_idx];
                    let result = self.run_tool(config, *pattern, *mult, *noise, *dataset_id, dataset, tool_idx);
                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    progress(done as f64 / total_work as f64, &format!("{}-{:.2}σ-{}", pattern.name(), mult, noise.name()));
                    result
                })
                .collect()
        } else {
            // Parallel for synthetic mode - this is where we get the speedup!
            work_items
                .par_iter()
                .map(|&(dataset_idx, tool_idx)| {
                    let (pattern, mult, noise, dataset_id, dataset) = &datasets[dataset_idx];
                    let result = self.run_tool(config, *pattern, *mult, *noise, *dataset_id, dataset, tool_idx);
                    completed.fetch_add(1, Ordering::Relaxed);
                    result
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let all_results: Vec<BenchmarkResult> = work_items
            .iter()
            .map(|&(dataset_idx, tool_idx)| {
                let (pattern, mult, noise, dataset_id, dataset) = &datasets[dataset_idx];
                let result = self.run_tool(config, *pattern, *mult, *noise, *dataset_id, dataset, tool_idx);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                progress(done as f64 / total_work as f64, &format!("{}-{:.2}σ-{}", pattern.name(), mult, noise.name()));
                result
            })
            .collect();

        // Collect results
        for result in all_results {
            results.results.push(result);
        }

        // Report final progress
        progress(1.0, "Complete");

        results.total_time = start.elapsed();
        results
    }

    /// Run the benchmark sweep with incremental checkpoint support.
    ///
    /// This method enables resumability by:
    /// 1. Writing results to CSV as they complete (via IncrementalCsvWriter)
    /// 2. Skipping work items that are already in the checkpoint file
    ///
    /// # Arguments
    /// * `config` - Sweep configuration
    /// * `checkpoint` - Optional checkpoint writer for incremental saves and resume
    /// * `progress` - Optional callback receiving (progress_fraction, current_task)
    ///
    /// # Returns
    /// Aggregated results from all tools and configurations (including resumed results)
    pub fn run_with_checkpoint<F>(
        &self,
        config: &SweepConfig,
        checkpoint: Option<Arc<IncrementalCsvWriter>>,
        mut progress: F,
    ) -> SweepResults
    where
        F: FnMut(f64, &str),
    {
        let start = Instant::now();
        let mut results = SweepResults::new(config.clone());

        // Step 1: Collect dataset configs
        let dataset_configs: Vec<(EffectPattern, f64, NoiseModel, usize)> = config
            .iter_configs()
            .flat_map(|(pattern, mult, noise)| {
                (0..config.datasets_per_point).map(move |dataset_id| (pattern, mult, noise, dataset_id))
            })
            .collect();

        // Step 2: Generate all datasets first (parallel for synthetic, sequential for realistic)
        progress(0.0, "Generating datasets...");

        #[cfg(feature = "parallel")]
        let datasets: Vec<(EffectPattern, f64, NoiseModel, usize, GeneratedDataset)> = if config.use_realistic {
            dataset_configs
                .iter()
                .map(|&(pattern, mult, noise, dataset_id)| {
                    let dataset = self.generate_dataset(config, pattern, mult, noise, dataset_id);
                    (pattern, mult, noise, dataset_id, dataset)
                })
                .collect()
        } else {
            dataset_configs
                .par_iter()
                .map(|&(pattern, mult, noise, dataset_id)| {
                    let dataset = self.generate_dataset(config, pattern, mult, noise, dataset_id);
                    (pattern, mult, noise, dataset_id, dataset)
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let datasets: Vec<(EffectPattern, f64, NoiseModel, usize, GeneratedDataset)> = dataset_configs
            .iter()
            .map(|&(pattern, mult, noise, dataset_id)| {
                let dataset = self.generate_dataset(config, pattern, mult, noise, dataset_id);
                (pattern, mult, noise, dataset_id, dataset)
            })
            .collect();

        // Wrap datasets in Arc for sharing across threads
        let datasets: Vec<(EffectPattern, f64, NoiseModel, usize, Arc<GeneratedDataset>)> = datasets
            .into_iter()
            .map(|(p, m, n, id, d)| (p, m, n, id, Arc::new(d)))
            .collect();

        progress(0.05, "Running tools...");

        // Step 3: Create (dataset × tool) work items, filtering already-completed items
        let all_work_items: Vec<(usize, usize)> = (0..datasets.len())
            .flat_map(|dataset_idx| {
                (0..self.tools.len()).map(move |tool_idx| (dataset_idx, tool_idx))
            })
            .collect();

        let total_work = all_work_items.len();
        let resumed_count = checkpoint.as_ref().map(|c| c.resumed_count).unwrap_or(0);

        // Filter out completed work items if resuming
        let work_items: Vec<(usize, usize)> = if let Some(ref writer) = checkpoint {
            all_work_items
                .into_iter()
                .filter(|&(dataset_idx, tool_idx)| {
                    let (pattern, mult, noise, dataset_id, _) = &datasets[dataset_idx];
                    let noise_name = if config.use_realistic {
                        format!("{}-realistic", noise.name())
                    } else {
                        noise.name()
                    };
                    let key = WorkItemKey::new(
                        self.tools[tool_idx].name(),
                        &pattern.name(),
                        *mult,
                        &noise_name,
                        *dataset_id,
                    );
                    !writer.is_completed(&key)
                })
                .collect()
        } else {
            all_work_items
        };

        let remaining_work = work_items.len();
        if resumed_count > 0 {
            progress(
                resumed_count as f64 / total_work as f64,
                &format!("Resuming ({} already complete)...", resumed_count),
            );
        }

        let completed = AtomicUsize::new(resumed_count);

        // Step 4: Process all (dataset × tool) pairs
        #[cfg(feature = "parallel")]
        let all_results: Vec<BenchmarkResult> = if config.use_realistic {
            // Sequential for realistic mode
            work_items
                .iter()
                .map(|&(dataset_idx, tool_idx)| {
                    let (pattern, mult, noise, dataset_id, dataset) = &datasets[dataset_idx];
                    let result = self.run_tool(config, *pattern, *mult, *noise, *dataset_id, dataset, tool_idx);

                    // Write incrementally if checkpoint enabled
                    if let Some(ref writer) = checkpoint {
                        if let Err(e) = writer.write_result(&result) {
                            eprintln!("Warning: Failed to write checkpoint: {}", e);
                        }
                    }

                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    progress(
                        done as f64 / total_work as f64,
                        &format!("{}-{:.2}σ-{}", pattern.name(), mult, noise.name()),
                    );
                    result
                })
                .collect()
        } else {
            // Parallel for synthetic mode
            work_items
                .par_iter()
                .map(|&(dataset_idx, tool_idx)| {
                    let (pattern, mult, noise, dataset_id, dataset) = &datasets[dataset_idx];
                    let result = self.run_tool(config, *pattern, *mult, *noise, *dataset_id, dataset, tool_idx);

                    // Write incrementally if checkpoint enabled
                    if let Some(ref writer) = checkpoint {
                        if let Err(e) = writer.write_result(&result) {
                            eprintln!("Warning: Failed to write checkpoint: {}", e);
                        }
                    }

                    completed.fetch_add(1, Ordering::Relaxed);
                    result
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let all_results: Vec<BenchmarkResult> = work_items
            .iter()
            .map(|&(dataset_idx, tool_idx)| {
                let (pattern, mult, noise, dataset_id, dataset) = &datasets[dataset_idx];
                let result = self.run_tool(config, *pattern, *mult, *noise, *dataset_id, dataset, tool_idx);

                // Write incrementally if checkpoint enabled
                if let Some(ref writer) = checkpoint {
                    if let Err(e) = writer.write_result(&result) {
                        eprintln!("Warning: Failed to write checkpoint: {}", e);
                    }
                }

                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                progress(
                    done as f64 / total_work as f64,
                    &format!("{}-{:.2}σ-{}", pattern.name(), mult, noise.name()),
                );
                result
            })
            .collect();

        // Collect results
        for result in all_results {
            results.results.push(result);
        }

        // Report final progress
        if remaining_work > 0 {
            progress(1.0, "Complete");
        } else {
            progress(1.0, "All work already complete (nothing to do)");
        }

        results.total_time = start.elapsed();
        results
    }

    /// Generate a single dataset for the given configuration.
    fn generate_dataset(
        &self,
        config: &SweepConfig,
        pattern: EffectPattern,
        mult: f64,
        noise: NoiseModel,
        dataset_id: usize,
    ) -> GeneratedDataset {
        if config.use_realistic {
            // In realistic mode, convert sigma multiplier to nanoseconds
            let sigma_ns = 100_000.0;
            let delay_ns = (mult * sigma_ns) as u64;

            let effect = match pattern {
                EffectPattern::Null => BenchmarkEffect::Null,
                EffectPattern::Shift => BenchmarkEffect::FixedDelay { delay_ns },
                EffectPattern::Tail => BenchmarkEffect::TailEffect {
                    base_delay_ns: delay_ns,
                    tail_prob: 0.05,
                    tail_mult: 5.0,
                },
                EffectPattern::Variance => BenchmarkEffect::VariableDelay {
                    mean_ns: delay_ns,
                    std_ns: delay_ns / 2,
                },
                EffectPattern::Bimodal { slow_prob, slow_mult } => BenchmarkEffect::Bimodal {
                    slow_prob,
                    slow_delay_ns: (delay_ns as f64 * slow_mult) as u64,
                },
                EffectPattern::Quantized { quantum_ns: _ } => BenchmarkEffect::FixedDelay { delay_ns },
            };

            let realistic_config = RealisticConfig {
                samples_per_class: config.samples_per_class,
                effect,
                seed: 42 + dataset_id as u64,
                base_operation_ns: config.realistic_base_ns,
                warmup_iterations: 1000,
            };

            realistic_to_generated(&collect_realistic_dataset(&realistic_config))
        } else {
            let bench_config = BenchmarkConfig {
                samples_per_class: config.samples_per_class,
                effect_pattern: pattern,
                effect_sigma_mult: mult,
                noise_model: noise,
                seed: 42 + dataset_id as u64,
                ..Default::default()
            };
            generate_benchmark_dataset(&bench_config)
        }
    }

    /// Run a single tool on a dataset.
    fn run_tool(
        &self,
        config: &SweepConfig,
        pattern: EffectPattern,
        mult: f64,
        noise: NoiseModel,
        dataset_id: usize,
        dataset: &GeneratedDataset,
        tool_idx: usize,
    ) -> BenchmarkResult {
        let tool = &self.tools[tool_idx];
        let tool_start = Instant::now();
        let result = tool.analyze(dataset);
        let time_ms = tool_start.elapsed().as_millis() as u64;

        let noise_name = if config.use_realistic {
            format!("{}-realistic", noise.name())
        } else {
            noise.name()
        };

        BenchmarkResult {
            tool: tool.name().to_string(),
            preset: config.preset.name().to_string(),
            effect_pattern: pattern.name().to_string(),
            effect_sigma_mult: mult,
            noise_model: noise_name,
            dataset_id,
            samples_per_class: config.samples_per_class,
            detected: result.detected_leak,
            statistic: result.leak_probability,
            p_value: None,
            time_ms,
            samples_used: Some(result.samples_used),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{DudectAdapter, KsTestAdapter};

    #[test]
    fn test_sweep_config_quick() {
        let config = SweepConfig::quick();
        assert_eq!(config.samples_per_class, 5_000);
        assert_eq!(config.datasets_per_point, 20);
        assert_eq!(config.effect_multipliers.len(), 4); // [0, 0.02, 0.05, 0.1]
        assert_eq!(config.effect_patterns.len(), 2);
        assert_eq!(config.noise_models.len(), 3); // [IID, AR(0.5), AR(-0.5)]
    }

    #[test]
    fn test_sweep_config_total_points() {
        let config = SweepConfig::quick();
        // 4 multipliers * 2 patterns * 3 noise = 24 points
        assert_eq!(config.total_points(), 24);
        // 24 points * 20 datasets = 480 total
        assert_eq!(config.total_datasets(), 480);
    }

    #[test]
    fn test_sweep_runner_small() {
        // Very small test to verify runner works
        let config = SweepConfig {
            preset: BenchmarkPreset::Quick,
            samples_per_class: 100,
            datasets_per_point: 2,
            effect_multipliers: vec![0.0, 1.0],
            effect_patterns: vec![EffectPattern::Null, EffectPattern::Shift],
            noise_models: vec![NoiseModel::IID],
            use_realistic: false,
            realistic_base_ns: 1000,
        };

        let runner = SweepRunner::new(vec![
            Box::new(DudectAdapter::default()),
            Box::new(KsTestAdapter::default()),
        ]);

        let results = runner.run(&config, |_progress, _task| {});

        // 2 mult * 2 patterns * 1 noise * 2 datasets * 2 tools = 16 results
        assert_eq!(results.results.len(), 16);
        assert_eq!(results.tools().len(), 2);
    }

    #[test]
    fn test_wilson_ci() {
        // Test edge cases
        let (low, high) = SweepResults::wilson_ci(0, 0, 1.96);
        assert_eq!((low, high), (0.0, 1.0));

        // Test with some successes
        let (low, high) = SweepResults::wilson_ci(5, 10, 1.96);
        assert!(low > 0.2);
        assert!(high < 0.8);
        assert!(low < 0.5);
        assert!(high > 0.5);
    }
}
