//! Benchmark comparison infrastructure for tacet.
//!
//! This crate provides tools for comparing tacet against other
//! timing side-channel detection tools (dudect, RTLF, SILENT, etc.) using both
//! synthetic and real-world datasets.
//!
//! # Modules
//!
//! - [`synthetic`]: Generate synthetic timing datasets with known statistical properties
//! - [`realistic`]: Generate realistic timing datasets using actual timed operations
//! - [`adapters`]: Tool adapters for different timing analysis tools
//! - [`runner`]: Legacy benchmark runner for FPR, power, and efficiency tests
//! - [`sweep`]: New comprehensive benchmark sweep infrastructure
//! - [`output`]: CSV and markdown report generation
//! - [`checkpoint`]: Incremental CSV writing and checkpoint-based resumability
//!
//! # Quick Start
//!
//! ```ignore
//! use tacet_bench::{
//!     sweep::{SweepConfig, SweepRunner},
//!     DudectAdapter, KsTestAdapter, AndersonDarlingAdapter,
//! };
//!
//! // Create runner with tools to compare
//! let runner = SweepRunner::new(vec![
//!     Box::new(DudectAdapter::default()),
//!     Box::new(KsTestAdapter::default()),
//!     Box::new(AndersonDarlingAdapter::default()),
//! ]);
//!
//! // Run quick benchmark
//! let config = SweepConfig::quick();
//! let results = runner.run(&config, |progress, task| {
//!     println!("{:.0}% - {}", progress * 100.0, task);
//! });
//!
//! // Output results
//! println!("{}", tacet_bench::output::to_markdown(&results));
//! ```
//!
//! # Legacy API
//!
//! The original API using `SyntheticConfig` and `BenchmarkRunner` is still available:
//!
//! ```ignore
//! use tacet_bench::{
//!     SyntheticConfig, EffectType, generate_dataset,
//!     BenchmarkRunner, TimingOracleAdapter, ToolAdapter,
//! };
//!
//! let config = SyntheticConfig {
//!     samples_per_class: 30000,
//!     effect: EffectType::Shift { percent: 5.0 },
//!     seed: 42,
//!     ..Default::default()
//! };
//!
//! let dataset = generate_dataset(&config);
//! let adapter = TimingOracleAdapter::default();
//! let result = adapter.analyze(&dataset);
//! println!("Detected leak: {}", result.detected_leak);
//! ```

pub mod adapters;
pub mod checkpoint;
pub mod dudect_stats;
pub mod output;
pub mod realistic;
pub mod runner;
mod semaphore;
pub mod sweep;
pub mod synthetic;

pub use adapters::{
    load_blocked_csv, load_interleaved_csv, split_interleaved, AndersonDarlingAdapter,
    DudectAdapter, KsTestAdapter, MonaAdapter, RtlfAdapter, RtlfDockerAdapter, RtlfNativeAdapter,
    SilentAdapter, SilentNativeAdapter, StubAdapter, TimingOracleAdapter, TimingTvlaAdapter,
    TlsfuzzerAdapter, ToolAdapter, ToolResult,
};
pub use checkpoint::{IncrementalCsvWriter, WorkItemKey};
pub use realistic::{
    collect_realistic_dataset, realistic_to_generated, standard_realistic_configs,
    RealisticBlockedData, RealisticConfig, RealisticDataset,
};
pub use runner::{BenchmarkReport, BenchmarkRunner, FprResults, PowerResults};
pub use sweep::{
    BenchmarkPreset, BenchmarkResult, PointSummary, SweepConfig, SweepResults, SweepRunner,
};
pub use synthetic::{
    // New benchmark suite types
    generate_benchmark_dataset,
    generate_benchmark_suite,
    generate_dataset,
    standard_configs,
    write_blocked_csv,
    write_interleaved_csv,
    BenchmarkConfig,
    BlockedData,
    EffectPattern,
    EffectType,
    GeneratedDataset,
    NoiseModel,
    SyntheticConfig,
};
