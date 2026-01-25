//! Power/EM side-channel analysis module.
//!
//! This module provides statistical methodology for detecting power and electromagnetic
//! side-channel leakage in cryptographic implementations, using the same Bayesian
//! framework as timing analysis but adapted for power trace data.
//!
//! # Overview
//!
//! Power side-channel analysis uses TVLA-style (Test Vector Leakage Assessment)
//! fixed-vs-random methodology:
//!
//! - **Fixed class**: Traces captured with fixed input (e.g., all-zeros key)
//! - **Random class**: Traces captured with random inputs
//!
//! The module extracts statistical features from power traces and uses Bayesian
//! inference to determine if there's a detectable difference between classes.
//!
//! # Feature Families
//!
//! Three feature extraction methods are supported:
//!
//! - **Mean**: Per-partition mean (standard TVLA)
//! - **Robust3**: Median, 10th, and 90th percentiles (robust to outliers)
//! - **CenteredSquare**: Centered variance per partition (detects second-order leakage)
//!
//! # Example
//!
//! ```ignore
//! use tacet::power::{Dataset, Trace, Class, Config, analyze};
//!
//! // Load or create traces
//! let traces = vec![
//!     Trace::new(Class::Fixed, vec![0.1, 0.2, 0.3, ...]),
//!     Trace::new(Class::Random, vec![0.15, 0.25, 0.35, ...]),
//!     // ...
//! ];
//!
//! let dataset = Dataset::new(traces);
//! let config = Config::default();
//! let report = analyze(&dataset, &config);
//!
//! println!("Leak probability: {:.1}%", report.leak_probability * 100.0);
//! ```

mod analyze;
mod config;
mod dataset;
mod features;
mod preprocessing;
mod report;

pub use analyze::analyze;
pub use config::{Config, FeatureFamily, PartitionConfig, PreprocessingConfig};
pub use dataset::{Class, Dataset, Marker, Meta, PowerUnits, StageId, Trace};
pub use features::{
    compute_class_difference, extract_features, extract_trace_features, ExtractedFeatures,
    TraceFeatures,
};
pub use report::{
    DimensionInfo, FeatureHotspot, PowerDiagnostics, PowerOutcome, Regime, Report, StageReport,
};
