//! Adapters for different timing analysis tools.
//!
//! Each adapter wraps a tool and provides a uniform interface for
//! running timing analysis and extracting results.

pub mod timing_oracle_adapter;
pub mod dudect_adapter;
pub mod dudect_template;
pub mod dudect_parser;

use std::time::Duration;

/// Result of a timing analysis detection run
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Whether a leak was detected according to the tool's threshold
    pub detected_leak: bool,

    /// Tool-specific confidence metric (probability for timing-oracle, t-statistic for dudect)
    pub confidence_metric: f64,

    /// Number of samples actually used
    pub samples_used: usize,

    /// Wall-clock time taken
    pub duration: Duration,

    /// Optional raw tool-specific data for detailed analysis
    pub raw_data: Option<RawData>,
}

/// Raw data from different tools
#[derive(Debug, Clone)]
pub enum RawData {
    TimingOracle {
        leak_probability: f64,
        ci_gate_passed: bool,
    },
    Dudect {
        max_t: f64,
        max_tau: f64,
    },
}

use std::any::Any;

/// Trait for timing analysis tool adapters
pub trait Detector {
    /// Name of the detector
    fn name(&self) -> &str;

    /// Run detection with given number of samples
    fn detect(&self, fixed: &dyn Fn(), random: &dyn Fn(), samples: usize) -> DetectionResult;

    /// Get default threshold for binary classification
    fn default_threshold(&self) -> f64;

    /// Check if confidence metric exceeds threshold
    fn exceeds_threshold(&self, confidence_metric: f64, threshold: f64) -> bool;

    /// Downcast to Any for runtime type checking
    fn as_any(&self) -> &dyn Any;
}
