//! Preflight checks to validate measurement setup before analysis.
//!
//! This module provides diagnostic checks that help identify common issues
//! with timing measurement setups that could lead to false positives or
//! unreliable results.
//!
//! # Checks Performed
//!
//! - **Sanity Check**: Fixed-vs-Fixed comparison to detect broken harness
//! - **Autocorrelation**: Detects periodic interference patterns
//! - **Resolution**: Detects timer resolution issues
//! - **System**: Platform-specific checks (e.g., CPU governor on Linux)
//!
//! # Core vs Platform-specific Checks
//!
//! Most preflight checks are no_std compatible and live in `tacet-core`.
//! This module re-exports them and adds platform-specific checks:
//!
//! - **From core (no_std)**: sanity_check, autocorrelation_check, resolution_check,
//!   PreflightResult, PreflightWarnings, run_core_checks
//! - **Platform-specific (std)**: system_check, timer_sanity_check

// Re-export core preflight checks and types (no_std compatible)
pub use tacet_core::preflight::{
    autocorrelation_check, compute_acf, resolution_check, run_core_checks, sanity_check,
    AutocorrWarning, PreflightResult as CorePreflightResult,
    PreflightWarnings as CorePreflightWarnings, ResolutionWarning, SanityWarning,
};

// Platform-specific checks that require std
mod resolution;
mod system;

pub use resolution::timer_sanity_check;
pub use system::{system_check, SystemWarning};

use serde::{Deserialize, Serialize};

/// Result of running all preflight checks (including platform-specific).
///
/// This extends `tacet_core::preflight::PreflightResult` with platform-specific
/// system warnings that require std (e.g., reading /sys/devices on Linux).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

    /// Create from a core preflight result.
    pub fn from_core(core: CorePreflightResult) -> Self {
        Self {
            warnings: PreflightWarnings::from_core(core.warnings),
            has_critical: core.has_critical,
            is_valid: core.is_valid,
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

    /// Add a system warning.
    pub fn add_system_warning(&mut self, warning: SystemWarning) {
        self.warnings.system.push(warning);
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
            || !self.warnings.system.is_empty()
            || !self.warnings.resolution.is_empty()
    }
}

/// Collection of all warnings from preflight checks (including platform-specific).
///
/// This extends `tacet_core::preflight::PreflightWarnings` with platform-specific
/// system warnings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreflightWarnings {
    /// Warnings from sanity check (Fixed-vs-Fixed).
    pub sanity: Vec<SanityWarning>,

    /// Warnings from autocorrelation check.
    pub autocorr: Vec<AutocorrWarning>,

    /// Warnings from system checks (platform-specific).
    pub system: Vec<SystemWarning>,

    /// Warnings from timer resolution check.
    pub resolution: Vec<ResolutionWarning>,
}

impl PreflightWarnings {
    /// Create an empty warnings collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from core warnings.
    pub fn from_core(core: CorePreflightWarnings) -> Self {
        Self {
            sanity: core.sanity,
            autocorr: core.autocorr,
            system: Vec::new(),
            resolution: core.resolution,
        }
    }

    /// Get total number of warnings.
    pub fn count(&self) -> usize {
        self.sanity.len() + self.autocorr.len() + self.system.len() + self.resolution.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }
}

/// Run all preflight checks and collect warnings.
///
/// This runs both the core no_std-compatible checks and platform-specific
/// system checks.
///
/// # Arguments
///
/// * `fixed_samples` - Timing samples from fixed input class (baseline)
/// * `random_samples` - Timing samples from random input class (sample)
/// * `timer_resolution_ns` - Timer resolution in nanoseconds
/// * `seed` - Seed for reproducible randomization in sanity check
///
/// # Returns
///
/// A `PreflightResult` containing all warnings and validity assessment.
pub fn run_all_checks(
    fixed_samples: &[f64],
    random_samples: &[f64],
    timer_resolution_ns: f64,
    seed: u64,
) -> PreflightResult {
    // Run core checks
    let core_result = run_core_checks(fixed_samples, random_samples, timer_resolution_ns, seed);
    let mut result = PreflightResult::from_core(core_result);

    // Run platform-specific system checks
    for warning in system_check() {
        result.add_system_warning(warning);
    }

    result
}

/// Run timer sanity check separately (requires timer access).
///
/// This checks that the timer is monotonic by taking consecutive timestamps.
/// Should be called early in measurement setup where the timer is available.
///
/// # Arguments
///
/// * `timer` - The timer to check
///
/// # Returns
///
/// A `PreflightResult` with just the timer sanity check result.
pub fn run_timer_sanity_check(timer: &crate::measurement::BoxedTimer) -> PreflightResult {
    let mut result = PreflightResult::new();

    if let Some(warning) = timer_sanity_check(timer) {
        result.add_resolution_warning(warning.to_resolution_warning());
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
    fn test_from_core() {
        let mut core = CorePreflightResult::new();
        core.has_critical = true;
        core.is_valid = false;

        let result = PreflightResult::from_core(core);
        assert!(result.has_critical);
        assert!(!result.is_valid);
    }
}
