//! # timing-oracle
//!
//! Detect timing side channels in cryptographic code.
//!
//! This crate provides adaptive Bayesian methodology for detecting timing variations
//! between two input classes (baseline vs sample), outputting:
//! - Posterior probability of timing leak (0.0-1.0)
//! - Effect size estimates in nanoseconds (shift and tail components)
//! - Pass/Fail/Inconclusive decisions with bounded FPR
//! - Exploitability assessment
//!
//! ## Common Pitfall: Side-Effects in Closures
//!
//! The closures you provide must execute **identical code paths**.
//! Only the input *data* should differ - not the operations performed.
//!
//! ```ignore
//! // WRONG - Sample closure has extra RNG/allocation overhead
//! TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).test(
//!     InputPair::new(|| my_op(&[0u8; 32]), || my_op(&rand::random())),
//!     |_| {},  // RNG called during measurement!
//! );
//!
//! // CORRECT - Pre-generate inputs, both closures identical
//! use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair};
//! let inputs = InputPair::new(|| [0u8; 32], || rand::random());
//! TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).test(inputs, |data| {
//!     my_op(data);
//! });
//! ```
//!
//! See the `helpers` module for utilities that make this pattern easier.
//!
//! ## Quick Start
//!
//! ```ignore
//! use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair, Outcome};
//!
//! // Builder API with InputPair
//! let inputs = InputPair::new(|| [0u8; 32], || rand::random());
//! let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
//!     .test(inputs, |data| {
//!         my_function(data);
//!     });
//!
//! match outcome {
//!     Outcome::Pass { leak_probability, .. } => {
//!         println!("No leak detected: P={:.1}%", leak_probability * 100.0);
//!     }
//!     Outcome::Fail { leak_probability, exploitability, .. } => {
//!         println!("Leak detected: P={:.1}%, {:?}", leak_probability * 100.0, exploitability);
//!     }
//!     Outcome::Inconclusive { reason, .. } => {
//!         println!("Inconclusive: {:?}", reason);
//!     }
//!     Outcome::Unmeasurable { recommendation, .. } => {
//!         println!("Skipping: {}", recommendation);
//!     }
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

// Core modules
pub mod adaptive;
mod config;
mod constants;
mod oracle;
mod result;
mod thread_pool;
mod types;

// Functional modules
pub mod analysis;
pub mod helpers;
pub mod measurement;
pub mod output;
pub mod preflight;
pub mod statistics;

// Re-exports for public API
pub use config::{Config, IterationsPerSample};
pub use constants::{B_TAIL, DECILES, LOG_2PI, ONES};
pub use measurement::{BoxedTimer, Timer, TimerSpec};
pub use oracle::{compute_min_uniqueness_ratio, TimingOracle};
pub use result::{
    BatchingInfo, Diagnostics, EffectEstimate, EffectPattern, Exploitability, InconclusiveReason,
    IssueCode, MeasurementQuality, Metadata, MinDetectableEffect, Outcome, QualityIssue,
    UnmeasurableInfo, UnreliablePolicy,
};
pub use types::{AttackerModel, Class, TimingSample};

// Re-export helpers for convenience
pub use helpers::InputPair;

// ============================================================================
// Assertion Macros
// ============================================================================

/// Assert that the result indicates constant-time behavior.
/// Panics on Fail or Inconclusive with detailed diagnostic output.
///
/// # Example
///
/// ```ignore
/// use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair, assert_constant_time};
///
/// let inputs = InputPair::new(|| [0u8; 32], || rand::random());
/// let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
///     .test(inputs, |data| my_crypto_function(data));
/// assert_constant_time!(outcome);
/// ```
#[macro_export]
macro_rules! assert_constant_time {
    ($outcome:expr) => {
        match &$outcome {
            $crate::Outcome::Pass { .. } => {}
            $crate::Outcome::Fail { .. } => {
                let summary = $crate::output::format_debug_summary(&$outcome);
                panic!("Timing leak detected!\n\n{}", summary,);
            }
            $crate::Outcome::Inconclusive {
                reason,
                leak_probability,
                ..
            } => {
                let summary = $crate::output::format_debug_summary(&$outcome);
                panic!(
                    "Could not confirm constant-time (P={:.1}%): {}\n\n{}",
                    leak_probability * 100.0,
                    reason,
                    summary,
                );
            }
            $crate::Outcome::Unmeasurable { recommendation, .. } => {
                let summary = $crate::output::format_debug_summary(&$outcome);
                panic!(
                    "Cannot measure operation: {}\n\n{}",
                    recommendation, summary
                );
            }
        }
    };
}

/// Assert that no timing leak was detected.
/// Panics only on Fail (lenient - allows Inconclusive and Pass).
/// Includes detailed diagnostic output on failure.
///
/// # Example
///
/// ```ignore
/// use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair, assert_no_timing_leak};
///
/// let inputs = InputPair::new(|| [0u8; 32], || rand::random());
/// let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
///     .test(inputs, |data| my_crypto_function(data));
/// assert_no_timing_leak!(outcome);
/// ```
#[macro_export]
macro_rules! assert_no_timing_leak {
    ($outcome:expr) => {
        if let $crate::Outcome::Fail { .. } = &$outcome {
            let summary = $crate::output::format_debug_summary(&$outcome);
            panic!("Timing leak detected!\n\n{}", summary,);
        }
    };
}

/// Assert that a timing leak WAS detected (for testing known-leaky code).
/// Panics on Pass with detailed diagnostic output showing why no leak was found.
///
/// # Example
///
/// ```ignore
/// use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair, assert_leak_detected};
///
/// let inputs = InputPair::new(|| [0u8; 32], || rand::random());
/// let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
///     .test(inputs, |data| leaky_function(data));
/// assert_leak_detected!(outcome);
/// ```
#[macro_export]
macro_rules! assert_leak_detected {
    ($outcome:expr) => {
        match &$outcome {
            $crate::Outcome::Fail { .. } => {}
            $crate::Outcome::Pass {
                leak_probability, ..
            } => {
                let summary = $crate::output::format_debug_summary(&$outcome);
                panic!(
                    "Expected timing leak but got Pass (P={:.1}%)\n\n{}",
                    leak_probability * 100.0,
                    summary,
                );
            }
            $crate::Outcome::Inconclusive {
                reason,
                leak_probability,
                ..
            } => {
                let summary = $crate::output::format_debug_summary(&$outcome);
                panic!(
                    "Expected timing leak but got Inconclusive (P={:.1}%): {}\n\n{}",
                    leak_probability * 100.0,
                    reason,
                    summary,
                );
            }
            $crate::Outcome::Unmeasurable { recommendation, .. } => {
                let summary = $crate::output::format_debug_summary(&$outcome);
                panic!(
                    "Expected timing leak but operation unmeasurable: {}\n\n{}",
                    recommendation, summary,
                );
            }
        }
    };
}

// ============================================================================
// Reliability Handling Macros
// ============================================================================

/// Skip test if measurement is unreliable (fail-open).
///
/// Prints `[SKIPPED]` message and returns early if unreliable.
/// Returns `TestResult` if reliable.
///
/// # Example
/// ```ignore
/// use timing_oracle::{TimingOracle, InputPair, skip_if_unreliable};
///
/// #[test]
/// fn test_aes() {
///     let inputs = InputPair::new(|| [0u8; 16], || rand::random());
///     let outcome = TimingOracle::new().test(inputs, |data| encrypt(data));
///     let result = skip_if_unreliable!(outcome, "test_aes");
///     assert!(result.leak_probability < 0.1);
/// }
/// ```
#[macro_export]
macro_rules! skip_if_unreliable {
    ($outcome:expr, $name:expr) => {
        match $outcome.handle_unreliable($name, $crate::UnreliablePolicy::FailOpen) {
            Some(result) => result,
            None => return,
        }
    };
}

/// Require measurement to be reliable (fail-closed).
///
/// Panics if unreliable. Returns `TestResult` if reliable.
///
/// # Example
/// ```ignore
/// use timing_oracle::{TimingOracle, InputPair, require_reliable};
///
/// #[test]
/// fn test_aes_critical() {
///     let inputs = InputPair::new(|| [0u8; 16], || rand::random());
///     let outcome = TimingOracle::new().test(inputs, |data| encrypt(data));
///     let result = require_reliable!(outcome, "test_aes_critical");
///     assert!(result.leak_probability < 0.1);
/// }
/// ```
#[macro_export]
macro_rules! require_reliable {
    ($outcome:expr, $name:expr) => {
        match $outcome.handle_unreliable($name, $crate::UnreliablePolicy::FailClosed) {
            Some(result) => result,
            None => unreachable!(),
        }
    };
}

// Re-export the timing_test! and timing_test_checked! proc macros when the macros feature is enabled
#[cfg(feature = "macros")]
pub use timing_oracle_macros::{timing_test, timing_test_checked};
