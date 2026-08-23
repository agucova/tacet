//! Core statistical analysis for timing side-channel detection.
//!
//! This crate provides the fundamental statistical algorithms for timing oracle,
//! designed to work in `no_std` environments (embedded, WASM, SGX) with only
//! an allocator.
//!
//! # Features
//!
//! - `std` (default): Enable standard library support for convenience. Also
//!   pulls in `colored`, so Display/Debug output carries ANSI colors with
//!   automatic `NO_COLOR`/`TERM`/TTY detection; without `std` the same output
//!   is emitted as plain text.
//! - `parallel`: Enable parallel bootstrap using rayon (requires `std`)
//!
//! # Usage
//!
//! This crate is typically used through the main `tacet` crate, which
//! provides measurement collection, orchestration, and output formatting.
//! However, it can be used directly for embedded or no_std scenarios.
//!
//! ```ignore
//! use tacet_core::{
//!     analysis::{compute_bayes_gibbs, estimate_mde},
//!     statistics::bootstrap_covariance_matrix,
//!     types::{Class, TimingSample},
//! };
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod adaptive;
pub mod analysis;
pub mod colors;
pub mod constants;
pub mod ffi_summary;
pub mod formatting;
pub mod histogram;
pub mod math;
pub mod orchestration;
pub mod preflight;
pub mod result;
pub mod statistics;
pub mod timer;
pub mod types;

// Re-export commonly used items at crate root
pub use ffi_summary::{
    CalibrationSummary, DiagnosticsSummary, EffectSummary, InconclusiveReasonKind, OutcomeSummary,
    OutcomeType, PosteriorSummary,
};
pub use result::{
    EffectEstimate, EffectPattern, Exploitability, MeasurementQuality, MinDetectableEffect,
    Outcome, QuantileShifts, ResearchOutcome, ResearchStatus, TailDiagnostics, TopQuantile,
    UnreliablePolicy,
};
pub use types::{AttackerModel, Class, TimingSample};
