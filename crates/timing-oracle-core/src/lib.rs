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
//!
//! # Usage
//!
//! This crate is typically used through the main `timing-oracle` crate, which
//! provides measurement collection, orchestration, and output formatting.
//! However, it can be used directly for embedded or no_std scenarios.
//!
//! ```ignore
//! use timing_oracle_core::{
//!     analysis::{compute_bayes_factor, estimate_mde},
//!     statistics::{bootstrap_covariance_matrix, compute_deciles},
//!     types::{Matrix9, Vector9, Class, TimingSample},
//! };
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod analysis;
pub mod constants;
pub mod result;
pub mod statistics;
pub mod types;

// Re-export commonly used items at crate root
pub use result::{
    EffectEstimate, EffectPattern, Exploitability, MeasurementQuality,
    MinDetectableEffect, Outcome, UnreliablePolicy,
};
pub use types::{AttackerModel, Class, Matrix9, TimingSample, Vector9};
