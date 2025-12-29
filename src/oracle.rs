//! Main `TimingOracle` entry point and builder.

#[allow(unused_imports)]
use rand::Rng;

use crate::config::Config;
use crate::result::TestResult;

/// Main entry point for timing analysis.
///
/// Use the builder pattern to configure and run timing tests.
///
/// # Example
///
/// ```ignore
/// use timing_oracle::TimingOracle;
///
/// let result = TimingOracle::new()
///     .samples(50_000)
///     .ci_alpha(0.001)
///     .test(
///         || my_function(&fixed_input),
///         || my_function(&random_input()),
///     );
/// ```
#[derive(Debug, Clone)]
pub struct TimingOracle {
    config: Config,
}

impl Default for TimingOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl TimingOracle {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    /// Set samples per class.
    pub fn samples(mut self, n: usize) -> Self {
        self.config.samples = n;
        self
    }

    /// Set warmup iterations.
    pub fn warmup(mut self, n: usize) -> Self {
        self.config.warmup = n;
        self
    }

    /// Set CI false positive rate.
    pub fn ci_alpha(mut self, alpha: f64) -> Self {
        self.config.ci_alpha = alpha;
        self
    }

    /// Set minimum effect of concern in nanoseconds.
    pub fn min_effect_of_concern(mut self, ns: f64) -> Self {
        self.config.min_effect_of_concern_ns = ns;
        self
    }

    /// Set bootstrap iterations for CI thresholds.
    pub fn ci_bootstrap_iterations(mut self, n: usize) -> Self {
        self.config.ci_bootstrap_iterations = n;
        self
    }

    /// Set bootstrap iterations for covariance estimation.
    pub fn cov_bootstrap_iterations(mut self, n: usize) -> Self {
        self.config.cov_bootstrap_iterations = n;
        self
    }

    /// Set outlier filtering percentile.
    pub fn outlier_percentile(mut self, p: f64) -> Self {
        self.config.outlier_percentile = p;
        self
    }

    /// Set prior probability of no leak.
    pub fn prior_no_leak(mut self, p: f64) -> Self {
        self.config.prior_no_leak = p;
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Run test with simple closures.
    ///
    /// # Arguments
    ///
    /// * `fixed` - Closure that executes the operation with a fixed input
    /// * `random` - Closure that executes the operation with random inputs
    ///
    /// # Returns
    ///
    /// A `TestResult` containing the analysis results.
    pub fn test<F, R, T>(self, mut fixed: F, mut random: R) -> TestResult
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        // Delegate to test_with_state with trivial state
        self.test_with_state(
            || (),
            |_| (),
            |_, _rng| (),
            |_, _input| {
                // Alternate between fixed and random based on which input we got
                // This is a simplified version - the actual implementation will
                // track which class is being measured
            },
        )
    }

    /// Run test with setup and state.
    ///
    /// This variant allows more control over test setup and input generation.
    ///
    /// # Type Parameters
    ///
    /// * `S` - State type created by setup
    /// * `I` - Input type produced by generators
    /// * `F` - Fixed input generator
    /// * `R` - Random input generator
    /// * `E` - Executor that runs the operation under test
    ///
    /// # Arguments
    ///
    /// * `setup` - Creates initial state (called once)
    /// * `fixed_input` - Generates the fixed input
    /// * `random_input` - Generates random inputs
    /// * `execute` - Runs the operation under test with given input
    pub fn test_with_state<S, F, R, I, E>(
        self,
        setup: impl FnOnce() -> S,
        mut fixed_input: F,
        mut random_input: R,
        mut execute: E,
    ) -> TestResult
    where
        F: FnMut(&mut S) -> I,
        R: FnMut(&mut S, &mut dyn DynRng) -> I,
        E: FnMut(&mut S, I),
    {
        let _state = setup();

        // TODO: Implement the full measurement and analysis pipeline:
        // 1. Pre-flight checks
        // 2. Warmup
        // 3. Interleaved measurement
        // 4. Outlier filtering
        // 5. Quantile computation
        // 6. CI Gate (Layer 1)
        // 7. Bayesian inference (Layer 2)
        // 8. Result assembly

        // Placeholder result for now
        use crate::result::*;

        TestResult {
            leak_probability: 0.0,
            effect: None,
            exploitability: Exploitability::Negligible,
            min_detectable_effect: MinDetectableEffect {
                shift_ns: 0.0,
                tail_ns: 0.0,
            },
            ci_gate: CiGate {
                alpha: self.config.ci_alpha,
                passed: true,
                thresholds: [0.0; 9],
                observed: [0.0; 9],
            },
            quality: MeasurementQuality::Good,
            outlier_fraction: 0.0,
            metadata: Metadata {
                samples_per_class: self.config.samples,
                cycles_per_ns: 1.0,
                timer: "placeholder".to_string(),
                runtime_secs: 0.0,
            },
        }
    }
}

/// Trait alias for RNG that can be used as trait object.
///
/// This wraps `rand::RngCore` to provide a dyn-compatible interface.
pub trait DynRng {
    /// Generate a random u64.
    fn next_u64(&mut self) -> u64;
}

impl<T: rand::RngCore> DynRng for T {
    fn next_u64(&mut self) -> u64 {
        rand::RngCore::next_u64(self)
    }
}
