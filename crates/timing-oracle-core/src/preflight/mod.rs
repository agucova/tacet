//! Preflight checks to validate measurement setup before analysis.
//!
//! This module provides diagnostic checks that help identify common issues
//! with timing measurement setups that could lead to false positives or
//! unreliable results.
//!
//! # no_std Support
//!
//! All checks in this module are no_std compatible. Platform-specific checks
//! (like filesystem-based system configuration checks) remain in the main
//! timing-oracle crate.
//!
//! # Checks Provided
//!
//! - **Sanity Check**: Fixed-vs-Fixed comparison to detect broken harness
//! - **Autocorrelation**: Detects periodic interference patterns
//! - **Resolution**: Detects timer resolution issues
//! - **Generator Cost**: Ensures input generators have similar overhead

mod autocorr;
mod generator;
mod resolution;
mod sanity;

pub use autocorr::{autocorrelation_check, compute_acf, AutocorrWarning};
pub use generator::{generator_cost_check, GeneratorClass, GeneratorWarning};
pub use resolution::{resolution_check, ResolutionWarning};
pub use sanity::{sanity_check, SanityWarning};
