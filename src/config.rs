//! Configuration for adaptive Bayesian timing analysis.
//!
//! See spec Section 4.2 (Configuration) for the full specification.

use std::time::Duration;

use crate::types::AttackerModel;

/// Configuration options for `TimingOracle`.
///
/// The adaptive Bayesian oracle uses these settings to control the
/// analysis behavior, thresholds, and resource limits.
///
/// See spec Section 4.2 (Configuration).
#[derive(Debug, Clone)]
pub struct Config {
    // =========================================================================
    // Decision thresholds (new for adaptive Bayesian)
    // =========================================================================

    /// Threshold for declaring "Pass" (no leak detected).
    ///
    /// If the posterior probability of a timing leak falls below this threshold,
    /// the test passes. Default: 0.05 (5%).
    ///
    /// Lower values require more confidence to pass (more conservative).
    pub pass_threshold: f64,

    /// Threshold for declaring "Fail" (leak detected).
    ///
    /// If the posterior probability of a timing leak exceeds this threshold,
    /// the test fails. Default: 0.95 (95%).
    ///
    /// Higher values require more confidence to fail (more conservative).
    pub fail_threshold: f64,

    // =========================================================================
    // Resource limits
    // =========================================================================

    /// Maximum time budget for the analysis.
    ///
    /// The oracle will stop collecting samples and return Inconclusive if this
    /// time limit is reached. Default: 60 seconds.
    pub time_budget: Duration,

    /// Maximum number of samples to collect per class.
    ///
    /// The oracle will stop and return Inconclusive if this limit is reached
    /// without achieving a conclusive result. Default: 1,000,000.
    pub max_samples: usize,

    /// Number of samples to collect per batch during adaptive sampling.
    ///
    /// Larger batches are more efficient but less responsive to early stopping.
    /// Default: 1,000.
    pub batch_size: usize,

    /// Number of samples for initial calibration (covariance estimation).
    ///
    /// This fixed number of samples is collected before the adaptive phase
    /// begins. Used to estimate the noise covariance matrix. Default: 5,000.
    ///
    /// Note: This is a fixed overhead, not prominently configurable.
    pub calibration_samples: usize,

    // =========================================================================
    // Effect thresholds (attacker model)
    // =========================================================================

    /// Minimum effect size we care about in nanoseconds.
    ///
    /// Effects smaller than this won't trigger high posterior probabilities
    /// even if statistically detectable. This encodes practical relevance.
    ///
    /// Note: When `attacker_model` is set, this value may be overridden
    /// at runtime based on the attacker model's threshold.
    ///
    /// Default: 10.0 ns.
    pub min_effect_of_concern_ns: f64,

    /// Attacker model preset.
    ///
    /// When set, the attacker model's threshold is used instead of
    /// `min_effect_of_concern_ns`. The threshold is computed at runtime
    /// based on the timer's resolution and CPU frequency.
    ///
    /// See [`AttackerModel`] for available presets.
    ///
    /// Default: None (uses min_effect_of_concern_ns).
    pub attacker_model: Option<AttackerModel>,

    /// Optional hard effect threshold in nanoseconds for reporting/panic.
    ///
    /// If the detected effect exceeds this threshold, the result is flagged
    /// prominently. Default: None.
    pub effect_threshold_ns: Option<f64>,

    // =========================================================================
    // Measurement configuration
    // =========================================================================

    /// Warmup iterations before measurement.
    ///
    /// These iterations warm CPU caches, stabilize frequency scaling, and
    /// trigger any JIT compilation before actual measurement begins.
    /// Default: 1,000.
    pub warmup: usize,

    /// Percentile for outlier winsorization.
    ///
    /// Samples beyond this percentile are capped (not dropped) to reduce
    /// the impact of extreme outliers while preserving information about
    /// tail-heavy distributions. Set to 1.0 to disable.
    ///
    /// Default: 0.9999 (99.99th percentile).
    pub outlier_percentile: f64,

    /// Iterations per timing sample.
    ///
    /// When set to `Auto`, the library detects timer resolution and
    /// automatically batches iterations when needed for coarse timers.
    /// Set to a specific value to override auto-detection.
    ///
    /// Default: Auto.
    pub iterations_per_sample: IterationsPerSample,

    // =========================================================================
    // Bayesian inference configuration
    // =========================================================================

    /// Prior probability of no leak.
    ///
    /// This is the prior belief that the code under test is constant-time.
    /// Higher values make the test more conservative (harder to fail).
    ///
    /// Default: 0.75 (75% prior belief in no leak).
    pub prior_no_leak: f64,

    /// Bootstrap iterations for covariance estimation.
    ///
    /// Used during the calibration phase to estimate the noise covariance
    /// matrix via block bootstrap. More iterations give better estimates
    /// but take longer.
    ///
    /// Default: 2,000.
    pub cov_bootstrap_iterations: usize,

    // =========================================================================
    // Sample splitting
    // =========================================================================

    /// Fraction of samples held out for calibration/preflight.
    ///
    /// In non-adaptive mode, this fraction of samples is used for covariance
    /// estimation. In adaptive mode, this is less relevant since calibration
    /// is a fixed upfront cost.
    ///
    /// Default: 0.3 (30% for calibration).
    pub calibration_fraction: f32,

    // =========================================================================
    // Optional limits and debugging
    // =========================================================================

    /// Optional guardrail for max duration in milliseconds (legacy).
    ///
    /// Prefer using `time_budget` instead. This is kept for backwards
    /// compatibility but will be removed in a future version.
    #[deprecated(since = "0.2.0", note = "Use time_budget instead")]
    pub max_duration_ms: Option<u64>,

    /// Optional deterministic seed for measurement randomness.
    ///
    /// When set, the measurement order (interleaving of classes) is
    /// deterministic, which can help with debugging and reproducibility.
    ///
    /// Default: None (random seed).
    pub measurement_seed: Option<u64>,

    /// Force discrete mode for testing.
    ///
    /// When true, discrete mode (m-out-of-n bootstrap with mid-quantiles)
    /// is used regardless of timer resolution. This is primarily for
    /// testing the discrete mode code path on machines with high-resolution timers.
    ///
    /// In production, discrete mode is triggered automatically when the
    /// minimum uniqueness ratio < 10% (per spec Section 2.4).
    ///
    /// Default: false.
    pub force_discrete_mode: bool,
}

impl Default for Config {
    fn default() -> Self {
        #[allow(deprecated)]
        Self {
            // Decision thresholds
            pass_threshold: 0.05,
            fail_threshold: 0.95,

            // Resource limits
            time_budget: Duration::from_secs(60),
            max_samples: 1_000_000,
            batch_size: 1_000,
            calibration_samples: 5_000,

            // Effect thresholds
            min_effect_of_concern_ns: 10.0,
            attacker_model: None,
            effect_threshold_ns: None,

            // Measurement configuration
            warmup: 1_000,
            outlier_percentile: 0.9999,
            iterations_per_sample: IterationsPerSample::Auto,

            // Bayesian inference
            prior_no_leak: 0.75,
            cov_bootstrap_iterations: 2_000,

            // Sample splitting
            calibration_fraction: 0.3,

            // Optional limits
            max_duration_ms: None,
            measurement_seed: None,
            force_discrete_mode: false,
        }
    }
}

impl Config {
    /// Create a new configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a balanced configuration for CI use.
    ///
    /// Uses moderate resource limits suitable for continuous integration:
    /// - 30 second time budget
    /// - 100,000 max samples
    /// - 500 sample batches
    pub fn balanced() -> Self {
        Self {
            time_budget: Duration::from_secs(30),
            max_samples: 100_000,
            batch_size: 500,
            ..Default::default()
        }
    }

    /// Create a quick configuration for development.
    ///
    /// Uses minimal resource limits for rapid iteration:
    /// - 10 second time budget
    /// - 20,000 max samples
    /// - 200 sample batches
    pub fn quick() -> Self {
        Self {
            time_budget: Duration::from_secs(10),
            max_samples: 20_000,
            batch_size: 200,
            calibration_samples: 2_000,
            cov_bootstrap_iterations: 500,
            ..Default::default()
        }
    }

    /// Create a thorough configuration for detailed analysis.
    ///
    /// Uses generous resource limits for thorough investigation:
    /// - 5 minute time budget
    /// - 10,000,000 max samples
    /// - 2,000 sample batches
    pub fn thorough() -> Self {
        Self {
            time_budget: Duration::from_secs(300),
            max_samples: 10_000_000,
            batch_size: 2_000,
            calibration_samples: 10_000,
            cov_bootstrap_iterations: 5_000,
            ..Default::default()
        }
    }

    /// Create a configuration for calibration/validation tests.
    ///
    /// Uses minimal settings for running many trials:
    /// - 5 second time budget
    /// - 5,000 max samples
    /// - 100 sample batches
    pub fn calibration() -> Self {
        Self {
            time_budget: Duration::from_secs(5),
            max_samples: 5_000,
            batch_size: 100,
            calibration_samples: 1_000,
            cov_bootstrap_iterations: 200,
            ..Default::default()
        }
    }

    // =========================================================================
    // Builder methods
    // =========================================================================

    /// Set the pass threshold.
    pub fn pass_threshold(mut self, threshold: f64) -> Self {
        assert!(threshold > 0.0 && threshold < 1.0, "pass_threshold must be in (0, 1)");
        assert!(threshold < self.fail_threshold, "pass_threshold must be < fail_threshold");
        self.pass_threshold = threshold;
        self
    }

    /// Set the fail threshold.
    pub fn fail_threshold(mut self, threshold: f64) -> Self {
        assert!(threshold > 0.0 && threshold < 1.0, "fail_threshold must be in (0, 1)");
        assert!(threshold > self.pass_threshold, "fail_threshold must be > pass_threshold");
        self.fail_threshold = threshold;
        self
    }

    /// Set the time budget.
    pub fn time_budget(mut self, budget: Duration) -> Self {
        self.time_budget = budget;
        self
    }

    /// Set the time budget in seconds.
    pub fn time_budget_secs(mut self, secs: u64) -> Self {
        self.time_budget = Duration::from_secs(secs);
        self
    }

    /// Set the maximum number of samples.
    pub fn max_samples(mut self, max: usize) -> Self {
        assert!(max > 0, "max_samples must be positive");
        self.max_samples = max;
        self
    }

    /// Set the batch size for adaptive sampling.
    pub fn batch_size(mut self, size: usize) -> Self {
        assert!(size > 0, "batch_size must be positive");
        self.batch_size = size;
        self
    }

    /// Set the number of calibration samples.
    pub fn calibration_samples(mut self, samples: usize) -> Self {
        assert!(samples > 0, "calibration_samples must be positive");
        self.calibration_samples = samples;
        self
    }

    /// Set the minimum effect of concern in nanoseconds.
    pub fn min_effect_ns(mut self, ns: f64) -> Self {
        assert!(ns >= 0.0, "min_effect_ns must be non-negative");
        self.min_effect_of_concern_ns = ns;
        self
    }

    /// Set the attacker model.
    pub fn attacker_model(mut self, model: AttackerModel) -> Self {
        self.attacker_model = Some(model);
        self
    }

    /// Set the warmup iterations.
    pub fn warmup(mut self, iterations: usize) -> Self {
        self.warmup = iterations;
        self
    }

    /// Set the outlier percentile.
    pub fn outlier_percentile(mut self, percentile: f64) -> Self {
        assert!(percentile > 0.0 && percentile <= 1.0, "outlier_percentile must be in (0, 1]");
        self.outlier_percentile = percentile;
        self
    }

    /// Set the iterations per sample.
    pub fn iterations_per_sample(mut self, iterations: IterationsPerSample) -> Self {
        self.iterations_per_sample = iterations;
        self
    }

    /// Set the prior probability of no leak.
    pub fn prior_no_leak(mut self, prior: f64) -> Self {
        assert!(prior > 0.0 && prior < 1.0, "prior_no_leak must be in (0, 1)");
        self.prior_no_leak = prior;
        self
    }

    /// Set the covariance bootstrap iterations.
    pub fn cov_bootstrap_iterations(mut self, iterations: usize) -> Self {
        assert!(iterations > 0, "cov_bootstrap_iterations must be positive");
        self.cov_bootstrap_iterations = iterations;
        self
    }

    /// Set the calibration fraction.
    pub fn calibration_fraction(mut self, fraction: f32) -> Self {
        assert!(fraction > 0.0 && fraction < 1.0, "calibration_fraction must be in (0, 1)");
        self.calibration_fraction = fraction;
        self
    }

    /// Set a deterministic seed for measurement.
    pub fn seed(mut self, seed: u64) -> Self {
        self.measurement_seed = Some(seed);
        self
    }

    /// Force discrete mode for testing.
    pub fn force_discrete_mode(mut self, force: bool) -> Self {
        self.force_discrete_mode = force;
        self
    }

    // =========================================================================
    // Resolution methods
    // =========================================================================

    /// Resolve the minimum effect of concern in nanoseconds.
    ///
    /// If an attacker model is set, returns its threshold in nanoseconds.
    /// Otherwise, returns the manually configured `min_effect_of_concern_ns`.
    ///
    /// # Arguments
    ///
    /// * `_cpu_freq_ghz` - Deprecated, kept for API compatibility
    /// * `_timer_resolution_ns` - Deprecated, kept for API compatibility
    ///
    /// # Returns
    ///
    /// The resolved threshold in nanoseconds.
    pub fn resolve_min_effect_ns(
        &self,
        _cpu_freq_ghz: Option<f64>,
        _timer_resolution_ns: Option<f64>,
    ) -> f64 {
        if let Some(model) = &self.attacker_model {
            model.to_threshold_ns()
        } else {
            self.min_effect_of_concern_ns
        }
    }

    /// Check if the configuration is valid.
    ///
    /// Returns an error message if the configuration is invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.pass_threshold <= 0.0 || self.pass_threshold >= 1.0 {
            return Err("pass_threshold must be in (0, 1)".to_string());
        }
        if self.fail_threshold <= 0.0 || self.fail_threshold >= 1.0 {
            return Err("fail_threshold must be in (0, 1)".to_string());
        }
        if self.pass_threshold >= self.fail_threshold {
            return Err("pass_threshold must be < fail_threshold".to_string());
        }
        if self.max_samples == 0 {
            return Err("max_samples must be positive".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be positive".to_string());
        }
        if self.calibration_samples == 0 {
            return Err("calibration_samples must be positive".to_string());
        }
        Ok(())
    }
}

/// Configuration for iterations per timing sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IterationsPerSample {
    /// Automatically detect based on timer resolution.
    ///
    /// On ARM64 with coarse timers (~40ns on Apple Silicon, Ampere Altra),
    /// this will batch multiple iterations per sample for reliable timing.
    /// On x86 or ARMv8.6+ (~1ns resolution), this typically uses 1 iteration.
    #[default]
    Auto,

    /// Use exactly N iterations per sample.
    ///
    /// The measured time will be divided by N to get per-iteration timing.
    Fixed(usize),
}

impl IterationsPerSample {
    /// Resolve the iterations count for a given timer.
    ///
    /// For `Auto`, uses the timer's resolution to suggest iterations.
    /// For `Fixed(n)`, returns `n`.
    pub fn resolve(&self, timer: &crate::measurement::Timer) -> usize {
        match self {
            Self::Auto => {
                // Target 10ns effective resolution for statistical reliability
                timer.suggested_iterations(10.0)
            }
            Self::Fixed(n) => *n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.pass_threshold, 0.05);
        assert_eq!(config.fail_threshold, 0.95);
        assert_eq!(config.time_budget, Duration::from_secs(60));
        assert_eq!(config.max_samples, 1_000_000);
        assert_eq!(config.batch_size, 1_000);
        assert_eq!(config.calibration_samples, 5_000);
    }

    #[test]
    fn test_preset_configs() {
        let balanced = Config::balanced();
        assert_eq!(balanced.time_budget, Duration::from_secs(30));
        assert_eq!(balanced.max_samples, 100_000);

        let quick = Config::quick();
        assert_eq!(quick.time_budget, Duration::from_secs(10));
        assert_eq!(quick.max_samples, 20_000);

        let thorough = Config::thorough();
        assert_eq!(thorough.time_budget, Duration::from_secs(300));
        assert_eq!(thorough.max_samples, 10_000_000);
    }

    #[test]
    fn test_builder_methods() {
        let config = Config::new()
            .pass_threshold(0.01)
            .fail_threshold(0.99)
            .time_budget_secs(120)
            .max_samples(500_000)
            .batch_size(2_000);

        assert_eq!(config.pass_threshold, 0.01);
        assert_eq!(config.fail_threshold, 0.99);
        assert_eq!(config.time_budget, Duration::from_secs(120));
        assert_eq!(config.max_samples, 500_000);
        assert_eq!(config.batch_size, 2_000);
    }

    #[test]
    fn test_validation() {
        let valid = Config::default();
        assert!(valid.validate().is_ok());

        let mut invalid = Config::default();
        invalid.pass_threshold = 0.0;
        assert!(invalid.validate().is_err());

        let mut invalid = Config::default();
        invalid.pass_threshold = 0.99;
        invalid.fail_threshold = 0.01;
        assert!(invalid.validate().is_err());
    }

    #[test]
    #[should_panic]
    fn test_invalid_pass_threshold() {
        Config::new().pass_threshold(1.5);
    }

    #[test]
    #[should_panic]
    fn test_invalid_threshold_order() {
        Config::new().pass_threshold(0.5).fail_threshold(0.4);
    }
}
