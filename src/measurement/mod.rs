//! Measurement infrastructure for timing analysis.
//!
//! This module provides:
//! - High-resolution cycle counting with platform-specific implementations
//! - Sample collection with randomized interleaved design
//! - Symmetric outlier filtering for robust analysis

mod collector;
mod outlier;
mod timer;

pub use collector::{Collector, Sample};
pub use outlier::{filter_outliers, OutlierStats};
pub use timer::{black_box, cycles_per_ns, rdtsc, Timer};
