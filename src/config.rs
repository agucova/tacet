//! Configuration for timing analysis.

/// Configuration options for `TimingOracle`.
#[derive(Debug, Clone)]
pub struct Config {
    /// Samples per class (default: 100,000).
    pub samples: usize,

    /// Warmup iterations before measurement (default: 1,000).
    pub warmup: usize,

    /// False positive rate for CI gate (default: 0.01).
    pub ci_alpha: f64,

    /// Minimum effect size of concern in nanoseconds (default: 10.0).
    ///
    /// Effects smaller than this won't trigger high posterior probabilities
    /// even if statistically detectable. This encodes practical relevance.
    pub min_effect_of_concern_ns: f64,

    /// Bootstrap iterations for CI thresholds (default: 10,000).
    pub ci_bootstrap_iterations: usize,

    /// Bootstrap iterations for covariance estimation (default: 2,000).
    pub cov_bootstrap_iterations: usize,

    /// Percentile for outlier filtering (default: 0.999).
    /// Set to 1.0 to disable filtering.
    pub outlier_percentile: f64,

    /// Prior probability of no leak (default: 0.75).
    pub prior_no_leak: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            samples: 100_000,
            warmup: 1_000,
            ci_alpha: 0.01,
            min_effect_of_concern_ns: 10.0,
            ci_bootstrap_iterations: 10_000,
            cov_bootstrap_iterations: 2_000,
            outlier_percentile: 0.999,
            prior_no_leak: 0.75,
        }
    }
}
