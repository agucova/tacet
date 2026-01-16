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
mod types;
mod thread_pool;

// Functional modules
pub mod analysis;
pub mod measurement;
pub mod output;
pub mod preflight;
pub mod statistics;
pub mod helpers;

// Re-exports for public API
pub use config::{Config, IterationsPerSample};
pub use constants::{B_TAIL, DECILES, LOG_2PI, ONES};
pub use measurement::{BoxedTimer, Timer, TimerSpec};
pub use oracle::{compute_min_uniqueness_ratio, TimingOracle};
pub use result::{
    BatchingInfo, Diagnostics, EffectEstimate, EffectPattern, Exploitability,
    InconclusiveReason, IssueCode, MeasurementQuality, Metadata, MinDetectableEffect,
    Outcome, QualityIssue, UnmeasurableInfo, UnreliablePolicy,
};
pub use types::{AttackerModel, Class, TimingSample};

// Re-export helpers for convenience
pub use helpers::InputPair;

// ============================================================================
// Assertion Macros
// ============================================================================

/// Assert that the result indicates constant-time behavior.
/// Panics on Fail or Inconclusive.
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
        match $outcome {
            $crate::Outcome::Pass { .. } => {}
            $crate::Outcome::Fail { leak_probability, effect, .. } => {
                panic!(
                    "Timing leak detected: P={:.1}%, shift={:.1}ns, tail={:.1}ns",
                    leak_probability * 100.0,
                    effect.shift_ns,
                    effect.tail_ns,
                );
            }
            $crate::Outcome::Inconclusive { reason, leak_probability, .. } => {
                panic!(
                    "Could not confirm constant-time (P={:.1}%): {:?}",
                    leak_probability * 100.0,
                    reason,
                );
            }
            $crate::Outcome::Unmeasurable { recommendation, .. } => {
                panic!("Cannot measure operation: {}", recommendation);
            }
        }
    };
}

/// Assert that no timing leak was detected.
/// Panics only on Fail (lenient - allows Inconclusive and Pass).
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
        if let $crate::Outcome::Fail { leak_probability, effect, .. } = $outcome {
            panic!(
                "Timing leak detected: P={:.1}%, shift={:.1}ns, tail={:.1}ns",
                leak_probability * 100.0,
                effect.shift_ns,
                effect.tail_ns,
            );
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
