//! # timing-oracle
//!
//! Detect timing side channels in cryptographic code.
//!
//! This crate provides statistical methodology for detecting timing variations
//! between two input classes (fixed vs random), outputting:
//! - Probability of timing leak (0.0-1.0)
//! - Effect size estimates in nanoseconds
//! - CI gate pass/fail with bounded false positive rate
//! - Exploitability assessment
//!
//! ## Quick Start
//!
//! ```ignore
//! use timing_oracle::{TimingOracle, test};
//!
//! // Simple API
//! let result = test(
//!     || my_function(&fixed_input),
//!     || my_function(&random_input()),
//! );
//!
//! println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

// Core modules
mod config;
mod constants;
mod oracle;
mod result;
mod types;

// Functional modules
pub mod analysis;
pub mod measurement;
pub mod output;
pub mod preflight;
pub mod statistics;

// Re-exports for public API
pub use config::Config;
pub use constants::{B_TAIL, DECILES, LOG_2PI, ONES};
pub use oracle::TimingOracle;
pub use result::{
    CiGate, Effect, EffectPattern, Exploitability, MeasurementQuality, Metadata,
    MinDetectableEffect, TestResult,
};
pub use types::{Class, Matrix9, Matrix9x2, Vector9};

/// Convenience function for simple timing tests with default configuration.
///
/// This runs a timing analysis comparing a fixed input operation against
/// a random input operation.
///
/// # Arguments
///
/// * `fixed` - Closure that executes with a fixed (potentially vulnerable) input
/// * `random` - Closure that executes with random inputs
///
/// # Returns
///
/// A `TestResult` containing leak probability, effect estimates, and CI gate result.
pub fn test<F, R, T>(fixed: F, random: R) -> TestResult
where
    F: FnMut() -> T,
    R: FnMut() -> T,
{
    TimingOracle::new().test(fixed, random)
}
