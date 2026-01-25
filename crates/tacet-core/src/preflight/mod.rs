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
//! tacet crate.
//!
//! # Checks Provided
//!
//! - **Sanity Check**: Fixed-vs-Fixed comparison to detect broken harness
//! - **Autocorrelation**: Detects periodic interference patterns
//! - **Resolution**: Detects timer resolution issues
//!
//! # Aggregation
//!
//! The `PreflightResult` and `PreflightWarnings` types aggregate results from
//! all core checks. Use `run_core_checks()` to run all no_std-compatible checks.

extern crate alloc;

use alloc::vec::Vec;

mod autocorr;
mod resolution;
mod sanity;

pub use autocorr::{autocorrelation_check, compute_acf, AutocorrWarning};
pub use resolution::{resolution_check, ResolutionWarning};
pub use sanity::{sanity_check, SanityWarning};

/// Result of running core preflight checks (no_std compatible).
///
/// This struct aggregates warnings from all core checks. Platform-specific
/// checks (like system configuration) are handled by the `tacet` crate.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct PreflightResult {
    /// All warnings collected from preflight checks.
    pub warnings: PreflightWarnings,

    /// Whether any critical warnings were found.
    pub has_critical: bool,

    /// Whether the measurement setup is considered valid.
    pub is_valid: bool,
}

impl PreflightResult {
    /// Create a new empty preflight result.
    pub fn new() -> Self {
        Self {
            warnings: PreflightWarnings::default(),
            has_critical: false,
            is_valid: true,
        }
    }

    /// Add a sanity warning.
    pub fn add_sanity_warning(&mut self, warning: SanityWarning) {
        if warning.is_result_undermining() {
            self.has_critical = true;
            self.is_valid = false;
        }
        self.warnings.sanity.push(warning);
    }

    /// Add an autocorrelation warning.
    pub fn add_autocorr_warning(&mut self, warning: AutocorrWarning) {
        self.warnings.autocorr.push(warning);
    }

    /// Add a resolution warning.
    pub fn add_resolution_warning(&mut self, warning: ResolutionWarning) {
        if warning.is_result_undermining() {
            self.has_critical = true;
            self.is_valid = false;
        }
        self.warnings.resolution.push(warning);
    }

    /// Check if there are any warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.sanity.is_empty()
            || !self.warnings.autocorr.is_empty()
            || !self.warnings.resolution.is_empty()
    }
}

/// Collection of warnings from core preflight checks (no_std compatible).
///
/// Platform-specific warnings (like system configuration) are not included
/// here; they are added by the `tacet` crate's preflight module.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct PreflightWarnings {
    /// Warnings from sanity check (Fixed-vs-Fixed).
    pub sanity: Vec<SanityWarning>,

    /// Warnings from autocorrelation check.
    pub autocorr: Vec<AutocorrWarning>,

    /// Warnings from timer resolution check.
    pub resolution: Vec<ResolutionWarning>,
}

impl PreflightWarnings {
    /// Create an empty warnings collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get total number of warnings.
    pub fn count(&self) -> usize {
        self.sanity.len() + self.autocorr.len() + self.resolution.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }
}

/// Run core preflight checks (no_std compatible).
///
/// This runs the three core checks that don't require std:
/// - Resolution check (timer quantization)
/// - Sanity check (Fixed-vs-Fixed)
/// - Autocorrelation check
///
/// Platform-specific checks (like CPU governor) are handled by `tacet::preflight::run_all_checks()`.
///
/// # Arguments
///
/// * `fixed_samples` - Timing samples from fixed input class (baseline), in nanoseconds
/// * `random_samples` - Timing samples from random input class (sample), in nanoseconds
/// * `timer_resolution_ns` - Timer resolution in nanoseconds
/// * `seed` - Seed for reproducible randomization in sanity check
///
/// # Returns
///
/// A `PreflightResult` containing all warnings and validity assessment.
pub fn run_core_checks(
    fixed_samples: &[f64],
    random_samples: &[f64],
    timer_resolution_ns: f64,
    seed: u64,
) -> PreflightResult {
    let mut result = PreflightResult::new();

    // Run resolution check (quantization)
    if let Some(warning) = resolution_check(fixed_samples, timer_resolution_ns) {
        result.add_resolution_warning(warning);
    }

    // Run sanity check (Fixed-vs-Fixed) with randomization
    if let Some(warning) = sanity_check(fixed_samples, timer_resolution_ns, seed) {
        result.add_sanity_warning(warning);
    }

    // Run autocorrelation check
    if let Some(warning) = autocorrelation_check(fixed_samples, random_samples) {
        result.add_autocorr_warning(warning);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preflight_result_default() {
        let result = PreflightResult::new();
        assert!(result.is_valid);
        assert!(!result.has_critical);
        assert!(!result.has_warnings());
    }

    #[test]
    fn test_warnings_count() {
        let mut warnings = PreflightWarnings::new();
        assert_eq!(warnings.count(), 0);
        assert!(warnings.is_empty());

        warnings.sanity.push(SanityWarning::BrokenHarness {
            variance_ratio: 7.5,
        });
        assert_eq!(warnings.count(), 1);
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_run_core_checks_empty() {
        // Empty samples should not crash
        let result = run_core_checks(&[], &[], 1.0, 42);
        // With empty data, preflight should be valid (no data to fail on)
        assert!(result.is_valid);
    }

    #[test]
    fn test_run_core_checks_normal_data() {
        // Generate some reasonable timing data
        let fixed: Vec<f64> = (0..1000).map(|i| 100.0 + (i % 10) as f64).collect();
        let random: Vec<f64> = (0..1000).map(|i| 105.0 + (i % 10) as f64).collect();

        let result = run_core_checks(&fixed, &random, 1.0, 42);
        // Should pass with reasonable data
        assert!(result.is_valid);
    }
}
