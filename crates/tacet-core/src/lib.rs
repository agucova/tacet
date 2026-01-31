//! Core statistical analysis for timing side-channel detection.
//!
//! This crate provides the fundamental statistical algorithms for timing oracle,
//! designed to work in `no_std` environments (embedded, WASM, SGX) with only
//! an allocator.
//!
//! # Features
//!
//! - `std` (default): Enable standard library support for convenience
//! - `parallel`: Enable parallel bootstrap using rayon (requires `std`)
//! - `ansi`: Enable ANSI colors in Display/Debug output (no_std compatible)
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
//!     statistics::{bootstrap_covariance_matrix, compute_deciles},
//!     types::{Matrix9, Vector9, Class, TimingSample},
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
    EffectEstimate, Exploitability, MeasurementQuality, MinDetectableEffect, Outcome,
    ResearchOutcome, ResearchStatus, TopQuantile, UnreliablePolicy,
};
pub use types::{AttackerModel, Class, Matrix9, TimingSample, Vector9};
