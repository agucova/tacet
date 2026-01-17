//! Preflight checks to validate measurement setup before analysis.
//!
//! This module provides diagnostic checks that help identify common issues
//! with timing measurement setups that could lead to false positives or
//! unreliable results.
//!
//! # Checks Performed
//!
//! - **Sanity Check**: Fixed-vs-Fixed comparison to detect broken harness
//! - **Generator Cost**: Ensures input generators have similar overhead
//! - **Autocorrelation**: Detects periodic interference patterns
//! - **Resolution**: Detects timer resolution issues
//! - **System**: Platform-specific checks (e.g., CPU governor on Linux)
//!
//! # Core vs Platform-specific Checks
//!
//! Most preflight checks are no_std compatible and live in `timing-oracle-core`.
//! This module re-exports them and adds platform-specific checks:
//!
//! - **From core (no_std)**: sanity_check, autocorrelation_check, resolution_check, generator_cost_check
//! - **Platform-specific (std)**: system_check, timer_sanity_check, measure_generator_cost

// Re-export core preflight checks (no_std compatible)
pub use timing_oracle_core::preflight::{
    autocorrelation_check, compute_acf, generator_cost_check, resolution_check, sanity_check,
    AutocorrWarning, GeneratorClass, GeneratorWarning, ResolutionWarning, SanityWarning,
};

// Platform-specific checks that require std
mod generator;
mod resolution;
mod system;

pub use generator::measure_generator_cost;
pub use resolution::timer_sanity_check;
pub use system::{system_check, SystemWarning};

use serde::{Deserialize, Serialize};

/// Result of running all preflight checks.
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

    /// Add a sanity warning.
    pub fn add_sanity_warning(&mut self, warning: SanityWarning) {
        if warning.is_result_undermining() {
            self.has_critical = true;
            self.is_valid = false;
        }
        self.warnings.sanity.push(warning);
    }

    /// Add a generator warning.
    pub fn add_generator_warning(&mut self, warning: GeneratorWarning) {
        if warning.is_result_undermining() {
            self.has_critical = true;
        }
        self.warnings.generator.push(warning);
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
            || !self.warnings.generator.is_empty()
            || !self.warnings.autocorr.is_empty()
            || !self.warnings.system.is_empty()
            || !self.warnings.resolution.is_empty()
    }
}

/// Collection of all warnings from preflight checks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreflightWarnings {
    /// Warnings from sanity check (Fixed-vs-Fixed).
    pub sanity: Vec<SanityWarning>,

    /// Warnings from generator cost check.
    pub generator: Vec<GeneratorWarning>,

    /// Warnings from autocorrelation check.
    pub autocorr: Vec<AutocorrWarning>,

    /// Warnings from system checks.
    pub system: Vec<SystemWarning>,

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
        self.sanity.len() + self.generator.len() + self.autocorr.len() + self.system.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }
}

/// Run all preflight checks and collect warnings.
///
/// # Arguments
///
/// * `fixed_samples` - Timing samples from fixed input class (baseline)
/// * `random_samples` - Timing samples from random input class (sample)
/// * `fixed_gen_time_ns` - Optional: time to generate fixed inputs
/// * `random_gen_time_ns` - Optional: time to generate random inputs
/// * `timer_resolution_ns` - Timer resolution in nanoseconds
/// * `seed` - Seed for reproducible randomization in sanity check
///
/// # Returns
///
/// A `PreflightResult` containing all warnings and validity assessment.
pub fn run_all_checks(
    fixed_samples: &[f64],
    random_samples: &[f64],
    fixed_gen_time_ns: Option<f64>,
    random_gen_time_ns: Option<f64>,
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

    // Run generator cost check if timing data available
    if let (Some(fixed_time), Some(random_time)) = (fixed_gen_time_ns, random_gen_time_ns) {
        if let Some(warning) = generator_cost_check(fixed_time, random_time) {
            result.add_generator_warning(warning);
        }
    }

    // Run autocorrelation check
    if let Some(warning) = autocorrelation_check(fixed_samples, random_samples) {
        result.add_autocorr_warning(warning);
    }

    // Run system checks
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
}
