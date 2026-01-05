//! Main `TimingOracle` entry point and builder.

use std::env;
use std::time::Instant;

#[allow(unused_imports)]
use rand::Rng;
use rand::seq::SliceRandom;

use std::hash::Hash;

use crate::analysis::{compute_bayes_factor, compute_diagnostics, decompose_effect, estimate_mde, run_ci_gate, CiGateInput};
use crate::config::Config;
use crate::helpers::InputPair;
use crate::measurement::{filter_outliers, BoxedTimer, TimerSpec};
use crate::preflight::run_all_checks;
use crate::result::{
    BatchingInfo, CiGate, Diagnostics, Effect, Exploitability, MeasurementQuality, Metadata, MinDetectableEffect,
    Outcome, TestResult,
};
use crate::statistics::{
    apply_variance_floor, bootstrap_difference_covariance, compute_deciles_fast,
    scale_covariance_for_inference,
};
use crate::types::{Class, Vector9};

/// Main entry point for timing analysis.
///
/// Use the builder pattern to configure and run timing tests.
///
/// # Example
///
/// ```ignore
/// use timing_oracle::{TimingOracle, helpers::InputPair};
///
/// let inputs = InputPair::new(
///     || [0u8; 32],          // baseline: returns constant value
///     || rand::random(),     // sample: generates varied values
/// );
///
/// let result = TimingOracle::new()
///     .samples(50_000)
///     .alpha(0.001)
///     .test(inputs, |data| my_function(data));
/// ```
///
/// # Automatic PMU Detection
///
/// When running with sudo/root privileges, the library automatically uses
/// cycle-accurate PMU timing (kperf on macOS, perf_event on Linux).
/// No code changes needed - just run with sudo.
///
/// To explicitly control timer selection:
///
/// ```ignore
/// use timing_oracle::{TimingOracle, TimerSpec};
///
/// // Force standard timer (no PMU attempt)
/// let result = TimingOracle::new()
///     .timer_spec(TimerSpec::Standard)
///     .test(...);
/// ```
#[derive(Debug, Clone)]
pub struct TimingOracle {
    config: Config,
    /// Timer specification (Auto by default - tries PMU first).
    timer_spec: TimerSpec,
}

struct PipelineInputs {
    baseline_cycles: Vec<u64>,
    sample_cycles: Vec<u64>,
    interleaved_cycles: Vec<u64>,
    interleaved_classes: Vec<Class>,
    baseline_gen_time_ns: Option<f64>,
    sample_gen_time_ns: Option<f64>,
    batching: BatchingInfo,
}

impl Default for TimingOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl TimingOracle {
    /// Create with default configuration.
    ///
    /// Uses `TimerSpec::Auto` which automatically detects and uses PMU timing
    /// (kperf/perf_event) when running with sufficient privileges.
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            timer_spec: TimerSpec::Auto,
        }
    }

    /// Create with fast configuration for testing/calibration.
    ///
    /// Uses reduced sample counts, warmup, and bootstrap iterations
    /// for faster execution while maintaining statistical validity.
    ///
    /// Settings:
    /// - 5,000 samples (vs 100,000 default)
    /// - 50 warmup iterations (vs 1,000 default)
    /// - 50 covariance bootstrap iterations (vs 50 default)
    /// - 50 CI bootstrap iterations (vs 100 default)
    pub fn quick() -> Self {
        Self {
            config: Config {
                samples: 5_000,
                warmup: 50,
                cov_bootstrap_iterations: 50,
                ci_bootstrap_iterations: 50,
                ..Config::default()
            },
            timer_spec: TimerSpec::Auto,
        }
    }

    /// Create with balanced configuration for production use.
    ///
    /// Faster than default while maintaining good statistical power.
    /// Recommended for most timing tests where ~1-2 second runtime is acceptable.
    ///
    /// Settings:
    /// - 20,000 samples (vs 100,000 default, 5x faster)
    /// - 200 warmup iterations (vs 1,000 default)
    /// - 50 covariance bootstrap iterations (same as updated default)
    /// - 100 CI bootstrap iterations (same as updated default)
    pub fn balanced() -> Self {
        Self {
            config: Config {
                samples: 20_000,
                warmup: 200,
                cov_bootstrap_iterations: 50,
                ci_bootstrap_iterations: 100,
                ..Config::default()
            },
            timer_spec: TimerSpec::Auto,
        }
    }

    /// Create with minimal configuration for calibration tests.
    ///
    /// Optimized for running many trials (100+) in calibration suites.
    /// Uses minimum sample sizes that maintain statistical validity.
    ///
    /// Settings:
    /// - 2,000 samples (minimal for valid statistics)
    /// - 20 warmup iterations
    /// - 20 covariance bootstrap iterations
    /// - 30 CI bootstrap iterations
    pub fn calibration() -> Self {
        Self {
            config: Config {
                samples: 2_000,
                warmup: 20,
                cov_bootstrap_iterations: 20,
                ci_bootstrap_iterations: 30,
                ..Config::default()
            },
            timer_spec: TimerSpec::Auto,
        }
    }

    /// Set the timer specification.
    ///
    /// Controls which timer implementation is used:
    /// - `TimerSpec::Auto` (default): Try PMU first, fall back to standard
    /// - `TimerSpec::Standard`: Always use standard timer (rdtsc/cntvct_el0)
    /// - `TimerSpec::PreferPmu`: Prefer PMU, fall back if unavailable
    ///
    /// # Example
    ///
    /// ```ignore
    /// use timing_oracle::{TimingOracle, TimerSpec};
    ///
    /// // Force standard timer (no PMU attempt)
    /// let result = TimingOracle::new()
    ///     .timer_spec(TimerSpec::Standard)
    ///     .test(...);
    /// ```
    pub fn timer_spec(mut self, spec: TimerSpec) -> Self {
        self.timer_spec = spec;
        self
    }

    /// Use the standard timer only (no PMU attempt).
    ///
    /// Shorthand for `.timer_spec(TimerSpec::Standard)`.
    pub fn standard_timer(self) -> Self {
        self.timer_spec(TimerSpec::Standard)
    }

    /// Prefer PMU timer with fallback to standard.
    ///
    /// Shorthand for `.timer_spec(TimerSpec::PreferPmu)`.
    pub fn prefer_pmu(self) -> Self {
        self.timer_spec(TimerSpec::PreferPmu)
    }

    /// Set samples per class.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    pub fn samples(mut self, n: usize) -> Self {
        assert!(n > 0, "samples must be > 0 (got {})", n);
        self.config.samples = n;
        self
    }

    /// Set warmup iterations.
    pub fn warmup(mut self, n: usize) -> Self {
        self.config.warmup = n;
        self
    }

    /// Set CI false positive rate (default: 0.01).
    ///
    /// Lower values reduce false positives but require more samples to detect real leaks.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is not in the range (0, 1).
    pub fn alpha(mut self, alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "alpha must be in (0, 1), got {}",
            alpha
        );
        self.config.ci_alpha = alpha;
        self
    }

    /// Set minimum effect of concern in nanoseconds (default: 10.0).
    ///
    /// Effects smaller than this are considered negligible. Used for:
    /// - Setting the Bayesian prior scale
    /// - Determining exploitability classification
    ///
    /// # Panics
    ///
    /// Panics if `ns` is negative or NaN.
    pub fn min_effect_ns(mut self, ns: f64) -> Self {
        assert!(
            ns >= 0.0 && !ns.is_nan(),
            "min_effect_ns must be >= 0, got {}",
            ns
        );
        self.config.min_effect_of_concern_ns = ns;
        self
    }

    /// Optional hard effect threshold in nanoseconds for reporting/panic.
    ///
    /// # Panics
    ///
    /// Panics if `ns` is not positive or is NaN.
    pub fn effect_threshold_ns(mut self, ns: f64) -> Self {
        assert!(
            ns > 0.0 && !ns.is_nan(),
            "effect_threshold_ns must be > 0, got {}",
            ns
        );
        self.config.effect_threshold_ns = Some(ns);
        self
    }

    /// Set bootstrap iterations for CI thresholds.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    pub fn ci_bootstrap_iterations(mut self, n: usize) -> Self {
        assert!(n > 0, "ci_bootstrap_iterations must be > 0, got {}", n);
        self.config.ci_bootstrap_iterations = n;
        self
    }

    /// Set bootstrap iterations for covariance estimation.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    pub fn cov_bootstrap_iterations(mut self, n: usize) -> Self {
        assert!(n > 0, "cov_bootstrap_iterations must be > 0, got {}", n);
        self.config.cov_bootstrap_iterations = n;
        self
    }

    /// Set outlier filtering percentile.
    ///
    /// Must be in the range (0, 1]. Set to 1.0 to disable filtering.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in the range (0, 1].
    pub fn outlier_percentile(mut self, p: f64) -> Self {
        assert!(
            p > 0.0 && p <= 1.0,
            "outlier_percentile must be in (0, 1], got {}",
            p
        );
        self.config.outlier_percentile = p;
        self
    }

    /// Set prior probability of no leak.
    ///
    /// Must be in the range (0, 1).
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in the range (0, 1).
    pub fn prior_no_leak(mut self, p: f64) -> Self {
        assert!(
            p > 0.0 && p < 1.0,
            "prior_no_leak must be in (0, 1), got {}",
            p
        );
        self.config.prior_no_leak = p;
        self
    }

    /// Set calibration fraction for sample splitting.
    ///
    /// Must be in the range (0, 1). The remaining fraction is used for inference.
    ///
    /// # Panics
    ///
    /// Panics if `frac` is not in the range (0, 1).
    pub fn calibration_fraction(mut self, frac: f32) -> Self {
        assert!(
            frac > 0.0 && frac < 1.0,
            "calibration_fraction must be in (0, 1), got {}",
            frac
        );
        self.config.calibration_fraction = frac;
        self
    }

    /// Set maximum duration guardrail (milliseconds).
    pub fn max_duration_ms(mut self, ms: u64) -> Self {
        self.config.max_duration_ms = Some(ms);
        self
    }

    /// Set deterministic measurement seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.measurement_seed = Some(seed);
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Merge configuration from environment variables.
    ///
    /// Reads the following environment variables to override settings:
    /// - `TO_SAMPLES`: Number of samples per class
    /// - `TO_ALPHA`: CI false positive rate (e.g., "0.01")
    /// - `TO_MIN_EFFECT_NS`: Minimum effect of concern in nanoseconds
    /// - `TO_EFFECT_THRESHOLD_NS`: Hard effect threshold in nanoseconds
    /// - `TO_CALIBRATION_FRAC`: Calibration fraction (e.g., "0.3")
    /// - `TO_MAX_DURATION_MS`: Maximum duration guardrail in milliseconds
    /// - `TO_SEED`: Deterministic measurement seed
    ///
    /// # Example
    ///
    /// ```ignore
    /// use timing_oracle::TimingOracle;
    ///
    /// // In CI, set TO_SAMPLES=50000 to increase samples
    /// let oracle = TimingOracle::balanced().from_env();
    /// ```
    pub fn from_env(mut self) -> Self {
        if let Some(samples) = parse_usize_env("TO_SAMPLES") {
            self = self.samples(samples);
        }
        if let Some(alpha) = parse_f64_env("TO_ALPHA") {
            self = self.alpha(alpha);
        }
        if let Some(prior) = parse_f64_env("TO_MIN_EFFECT_NS") {
            self = self.min_effect_ns(prior);
        }
        if let Some(threshold) = parse_f64_env("TO_EFFECT_THRESHOLD_NS") {
            self = self.effect_threshold_ns(threshold);
        }
        if let Some(frac) = parse_f32_env("TO_CALIBRATION_FRAC") {
            self = self.calibration_fraction(frac);
        }
        if let Some(ms) = parse_u64_env("TO_MAX_DURATION_MS") {
            self = self.max_duration_ms(ms);
        }
        if let Some(seed) = parse_u64_env("TO_SEED") {
            self = self.seed(seed);
        }
        self
    }

    /// Run a timing test with pre-generated inputs.
    ///
    /// This is the primary API for timing tests. It handles input pre-generation
    /// internally to ensure accurate measurements without generator overhead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use timing_oracle::{TimingOracle, helpers::InputPair};
    ///
    /// let inputs = InputPair::new([0u8; 32], || rand::random());
    /// let result = TimingOracle::new().test(inputs, |data| {
    ///     my_crypto_function(data);
    /// });
    /// ```
    ///
    /// # How It Works
    ///
    /// 1. Pre-generates all baseline and sample inputs before measurement
    /// 2. Runs warmup iterations
    /// 3. Measures the operation with randomized interleaving
    /// 4. Only the `operation` closure is timed (not input generation)
    ///
    /// # Arguments
    ///
    /// * `inputs` - An `InputPair` containing the baseline and sample generators
    /// * `operation` - Closure that performs the operation under test
    ///
    /// # Returns
    ///
    /// An `Outcome` which is either `Completed(TestResult)` or `Unmeasurable` if
    /// the operation is too fast to measure reliably.
    pub fn test<T, F1, F2, F>(self, inputs: InputPair<T, F1, F2>, mut operation: F) -> Outcome
    where
        T: Clone + Hash,
        F1: FnMut() -> T,
        F2: FnMut() -> T,
        F: FnMut(&T),
    {
        let start_time = Instant::now();
        let mut rng = rand::rng();

        // Step 1: Create timer based on spec (auto-detects PMU if available)
        let mut timer = self.timer_spec.create_timer();

        // Step 2: Pre-generate ALL inputs before measurement (critical for accuracy)
        let baseline_gen_start = Instant::now();
        let baseline_inputs: Vec<T> = (0..self.config.samples)
            .map(|_| inputs.baseline())
            .collect();
        let baseline_gen_time_ns = baseline_gen_start.elapsed().as_nanos() as f64 / self.config.samples as f64;

        let sample_gen_start = Instant::now();
        let sample_inputs: Vec<T> = (0..self.config.samples)
            .map(|_| {
                let value = inputs.generate_sample();
                // Track for anomaly detection during pre-generation
                inputs.track_value(&value);
                value
            })
            .collect();
        let sample_gen_time_ns = sample_gen_start.elapsed().as_nanos() as f64 / self.config.samples as f64;

        // Note: With InputPair API, inputs are pre-generated BEFORE measurement,
        // so generator overhead does not affect timing results. The spec's 10% abort
        // threshold (§3.2) was for the old API where generators could run inside the
        // timed region. With InputPair, this is a non-issue by design.

        // Step 4: Warmup
        for i in 0..self.config.warmup.min(self.config.samples) {
            std::hint::black_box(operation(&baseline_inputs[i % baseline_inputs.len()]));
            std::hint::black_box(operation(&sample_inputs[i % sample_inputs.len()]));
        }

        // Step 5: Pilot phase to check measurability
        const PILOT_SAMPLES: usize = 50;
        let mut pilot_cycles = Vec::with_capacity(PILOT_SAMPLES * 2);

        for i in 0..PILOT_SAMPLES.min(self.config.samples) {
            let cycles = timer.measure_cycles(|| {
                std::hint::black_box(operation(&baseline_inputs[i]));
            });
            pilot_cycles.push(cycles);

            let cycles = timer.measure_cycles(|| {
                std::hint::black_box(operation(&sample_inputs[i]));
            });
            pilot_cycles.push(cycles);
        }

        // Check if operation is measurable
        pilot_cycles.sort_unstable();
        let median_cycles = pilot_cycles[pilot_cycles.len() / 2];
        let median_ns = timer.cycles_to_ns(median_cycles);
        let resolution_ns = timer.resolution_ns();
        let ticks_per_call = median_ns / resolution_ns;

        if ticks_per_call < crate::measurement::MIN_TICKS_SINGLE_CALL {
            let threshold_ns = resolution_ns * crate::measurement::MIN_TICKS_SINGLE_CALL;

            // Return early for unmeasurable operations
            let platform = format!(
                "{} ({}, {:.1}ns resolution)",
                std::env::consts::OS,
                timer.name(),
                timer.resolution_ns()
            );

            // Recommend PMU if using standard timer on ARM64
            let recommendation = if timer.name() == "cntvct_el0" {
                #[cfg(target_os = "macos")]
                {
                    "Run with sudo to enable kperf cycle counting (~0.3ns resolution), or increase operation complexity"
                }
                #[cfg(target_os = "linux")]
                {
                    "Run with sudo to enable perf_event cycle counting (~0.3ns resolution), or increase operation complexity"
                }
                #[cfg(not(any(target_os = "macos", target_os = "linux")))]
                {
                    "Increase operation complexity (current operation is too fast to measure reliably)"
                }
            } else {
                "Increase operation complexity (current operation is too fast to measure reliably)"
            };

            return Outcome::Unmeasurable {
                operation_ns: median_ns,
                threshold_ns,
                platform: platform.to_string(),
                recommendation: recommendation.to_string(),
            };
        }

        let batching = crate::result::BatchingInfo {
            enabled: false,
            k: 1,
            ticks_per_batch: ticks_per_call,
            rationale: format!("Pre-generated inputs ({:.1} ticks/call)", ticks_per_call),
            unmeasurable: None,
        };

        // Step 6: Create randomized schedule (interleaves baseline and sample measurements)
        let mut schedule: Vec<(Class, usize)> = Vec::with_capacity(self.config.samples * 2);
        for i in 0..self.config.samples {
            schedule.push((Class::Baseline, i));
            schedule.push((Class::Sample, i));
        }
        schedule.shuffle(&mut rng);

        // Step 7: Collect timing samples (only operation is timed)
        let mut baseline_cycles = Vec::with_capacity(self.config.samples);
        let mut sample_cycles = Vec::with_capacity(self.config.samples);
        let mut interleaved_cycles = Vec::with_capacity(self.config.samples * 2);
        let mut interleaved_classes = Vec::with_capacity(self.config.samples * 2);

        for (class, idx) in schedule {
            match class {
                Class::Baseline => {
                    let cycles = timer.measure_cycles(|| {
                        std::hint::black_box(operation(&baseline_inputs[idx]));
                    });
                    baseline_cycles.push(cycles);
                    interleaved_cycles.push(cycles);
                    interleaved_classes.push(Class::Baseline);
                }
                Class::Sample => {
                    let cycles = timer.measure_cycles(|| {
                        std::hint::black_box(operation(&sample_inputs[idx]));
                    });
                    sample_cycles.push(cycles);
                    interleaved_cycles.push(cycles);
                    interleaved_classes.push(Class::Sample);
                }
            }
        }

        // Step 8: Check for anomalies after measurement
        if let Some(warning) = inputs.check_anomaly() {
            eprintln!("[timing-oracle] {}", warning);
        }

        // Step 9: Run analysis pipeline
        let result = self.run_pipeline(
            PipelineInputs {
                baseline_cycles,
                sample_cycles,
                interleaved_cycles,
                interleaved_classes,
                baseline_gen_time_ns: Some(baseline_gen_time_ns),
                sample_gen_time_ns: Some(sample_gen_time_ns),
                batching,
            },
            &timer,
            start_time,
        );

        Outcome::Completed(result)
    }

    /// Run test with setup and state.
    ///
    /// This variant allows more control over test setup and input generation.
    ///
    /// # Type Parameters
    ///
    /// * `S` - State type created by setup
    /// * `I` - Input type produced by generators
    /// * `B` - Baseline input generator
    /// * `R` - Sample input generator
    /// * `E` - Executor that runs the operation under test
    ///
    /// # Arguments
    ///
    /// * `setup` - Creates initial state (called once)
    /// * `baseline_input` - Generates the baseline input
    /// * `sample_input` - Generates sample inputs
    /// * `execute` - Runs the operation under test with given input
    pub fn test_with_state<S, B, R, I, E>(
        self,
        setup: impl FnOnce() -> S,
        mut baseline_input: B,
        mut sample_input: R,
        mut execute: E,
    ) -> crate::result::Outcome
    where
        B: FnMut(&mut S) -> I,
        R: FnMut(&mut S, &mut rand::rngs::ThreadRng) -> I,
        E: FnMut(&mut S, I),
        I: Clone,
    {
        let mut rng = rand::rng();
        self.test_with_state_rng(
            setup,
            &mut rng,
            &mut baseline_input,
            &mut sample_input,
            &mut execute,
        )
    }

    /// Run test with setup and state, using a caller-provided RNG.
    pub fn test_with_state_rng<S, B, R, I, E, RNG>(
        self,
        setup: impl FnOnce() -> S,
        rng: &mut RNG,
        baseline_input: &mut B,
        sample_input: &mut R,
        execute: &mut E,
    ) -> crate::result::Outcome
    where
        RNG: rand::Rng + ?Sized,
        B: FnMut(&mut S) -> I,
        R: FnMut(&mut S, &mut RNG) -> I,
        E: FnMut(&mut S, I),
        I: Clone,
    {
        let start_time = Instant::now();

        // Create state
        let mut state = setup();

        // Create timer based on spec (auto-detects PMU if available)
        let mut timer = self.timer_spec.create_timer();

        // Pre-generate all inputs to avoid borrow conflicts
        let baseline_gen_start = Instant::now();
        let baseline_inputs: Vec<I> = (0..self.config.samples)
            .map(|_| baseline_input(&mut state))
            .collect();
        let baseline_gen_time_ns = if self.config.samples > 0 {
            baseline_gen_start.elapsed().as_nanos() as f64 / self.config.samples as f64
        } else {
            0.0
        };

        let sample_gen_start = Instant::now();
        let sample_inputs: Vec<I> = (0..self.config.samples)
            .map(|_| sample_input(&mut state, rng))
            .collect();
        let sample_gen_time_ns = if self.config.samples > 0 {
            sample_gen_start.elapsed().as_nanos() as f64 / self.config.samples as f64
        } else {
            0.0
        };

        // Run warmup
        for _ in 0..self.config.warmup {
            if let Some(input) = baseline_inputs.first() {
                execute(&mut state, input.clone());
            }
            if let Some(input) = sample_inputs.first() {
                execute(&mut state, input.clone());
            }
        }

        // Pilot phase: check measurability
        // Run a small sample to verify the operation is measurable
        const PILOT_SAMPLES: usize = 50;
        let mut pilot_cycles = Vec::with_capacity(PILOT_SAMPLES * 2);

        for i in 0..PILOT_SAMPLES.min(self.config.samples) {
            let cycles = timer.measure_cycles(|| {
                execute(&mut state, baseline_inputs[i].clone());
            });
            pilot_cycles.push(cycles);

            let cycles = timer.measure_cycles(|| {
                execute(&mut state, sample_inputs[i].clone());
            });
            pilot_cycles.push(cycles);
        }

        // Check if operation is measurable
        pilot_cycles.sort_unstable();
        let median_cycles = pilot_cycles[pilot_cycles.len() / 2];
        let median_ns = timer.cycles_to_ns(median_cycles);
        let resolution_ns = timer.resolution_ns();
        let ticks_per_call = median_ns / resolution_ns;

        let batching = if ticks_per_call < crate::measurement::MIN_TICKS_SINGLE_CALL {
            // Operation is too fast to measure reliably
            let threshold_ns = resolution_ns * crate::measurement::MIN_TICKS_SINGLE_CALL;

            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let suggestion = ". On macOS, run with sudo to enable kperf cycle counting (~1ns resolution)";
            #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
            let suggestion = ". Run with sudo and --features perf for cycle-accurate timing";
            #[cfg(not(target_arch = "aarch64"))]
            let suggestion = "";

            crate::result::BatchingInfo {
                enabled: false,
                k: 1,
                ticks_per_batch: ticks_per_call,
                rationale: format!(
                    "UNMEASURABLE: {:.1} ticks/call < {:.0} minimum (op ~{:.0}ns, threshold ~{:.0}ns){}",
                    ticks_per_call,
                    crate::measurement::MIN_TICKS_SINGLE_CALL,
                    median_ns,
                    threshold_ns,
                    suggestion
                ),
                unmeasurable: Some(crate::result::UnmeasurableInfo {
                    operation_ns: median_ns,
                    threshold_ns,
                    ticks_per_call,
                }),
            }
        } else {
            // Measurable - proceed without batching (stateful operations can't batch)
            crate::result::BatchingInfo {
                enabled: false,
                k: 1,
                ticks_per_batch: ticks_per_call,
                rationale: format!(
                    "test_with_state: no batching for stateful operations ({:.1} ticks/call)",
                    ticks_per_call
                ),
                unmeasurable: None,
            }
        };

        // Create randomized schedule (interleaves baseline and sample measurements)
        let mut schedule: Vec<(crate::types::Class, usize)> =
            Vec::with_capacity(self.config.samples * 2);
        for i in 0..self.config.samples {
            schedule.push((crate::types::Class::Baseline, i));
            schedule.push((crate::types::Class::Sample, i));
        }
        schedule.shuffle(rng);

        // Collect timing samples
        let mut baseline_cycles = Vec::with_capacity(self.config.samples);
        let mut sample_cycles = Vec::with_capacity(self.config.samples);
        let mut interleaved_cycles = Vec::with_capacity(self.config.samples * 2);
        let mut interleaved_classes = Vec::with_capacity(self.config.samples * 2);

        for (class, idx) in schedule {
            match class {
                crate::types::Class::Baseline => {
                    let cycles = timer.measure_cycles(|| {
                        execute(&mut state, baseline_inputs[idx].clone());
                    });
                    baseline_cycles.push(cycles);
                    interleaved_cycles.push(cycles);
                    interleaved_classes.push(crate::types::Class::Baseline);
                }
                crate::types::Class::Sample => {
                    let cycles = timer.measure_cycles(|| {
                        execute(&mut state, sample_inputs[idx].clone());
                    });
                    sample_cycles.push(cycles);
                    interleaved_cycles.push(cycles);
                    interleaved_classes.push(crate::types::Class::Sample);
                }
            }
        }

        // Run the analysis pipeline
        let result = self.run_pipeline(
            PipelineInputs {
                baseline_cycles,
                sample_cycles,
                interleaved_cycles,
                interleaved_classes,
                baseline_gen_time_ns: Some(baseline_gen_time_ns),
                sample_gen_time_ns: Some(sample_gen_time_ns),
                batching: batching.clone(),
            },
            &timer,
            start_time,
        );

        // Wrap result in Outcome enum based on measurability
        if let Some(unmeasurable_info) = &batching.unmeasurable {
            // Operation was too fast to measure reliably
            let platform = format!(
                "{} ({}, {:.1}ns resolution)",
                std::env::consts::OS,
                timer.name(),
                timer.resolution_ns()
            );

            // Recommend PMU if using standard timer on ARM64
            let recommendation = if timer.name() == "cntvct_el0" {
                #[cfg(target_os = "macos")]
                {
                    "Run with sudo to enable kperf cycle counting (~0.3ns resolution), or increase operation complexity"
                }
                #[cfg(target_os = "linux")]
                {
                    "Run with sudo to enable perf_event cycle counting (~0.3ns resolution), or increase operation complexity"
                }
                #[cfg(not(any(target_os = "macos", target_os = "linux")))]
                {
                    "Increase operation complexity (current operation is too fast to measure reliably)"
                }
            } else {
                "Increase operation complexity (current operation is too fast to measure reliably)"
            };

            crate::result::Outcome::Unmeasurable {
                operation_ns: unmeasurable_info.operation_ns,
                threshold_ns: unmeasurable_info.threshold_ns,
                platform,
                recommendation: recommendation.to_string(),
            }
        } else {
            // Measurement was successful
            crate::result::Outcome::Completed(result)
        }
    }

    /// Run the full analysis pipeline on collected samples.
    fn run_pipeline(
        &self,
        inputs: PipelineInputs,
        timer: &BoxedTimer,
        start_time: Instant,
    ) -> TestResult {
        use crate::types::TimingSample;

        let PipelineInputs {
            baseline_cycles,
            sample_cycles,
            interleaved_cycles,
            interleaved_classes,
            baseline_gen_time_ns,
            sample_gen_time_ns,
            batching,
        } = inputs;
        if batching.unmeasurable.is_some() {
            let runtime_secs = start_time.elapsed().as_secs_f64();
            let timer_name = timer.name();

            return TestResult {
                leak_probability: 0.0,
                bayes_factor: 1.0, // Neutral evidence
                effect: None,
                exploitability: Exploitability::Negligible,
                min_detectable_effect: MinDetectableEffect {
                    shift_ns: 0.0,
                    tail_ns: 0.0,
                },
                ci_gate: CiGate {
                    alpha: self.config.ci_alpha,
                    passed: true,
                    threshold: 0.0,
                    max_observed: 0.0,
                    observed: [0.0; 9],
                },
                quality: MeasurementQuality::TooNoisy,
                outlier_fraction: 0.0,
                diagnostics: Diagnostics::all_ok(),
                metadata: Metadata {
                    samples_per_class: baseline_cycles.len().min(sample_cycles.len()),
                    cycles_per_ns: timer.cycles_per_ns(),
                    timer: timer_name.to_string(),
                    timer_resolution_ns: timer.resolution_ns(),
                    batching,
                    runtime_secs,
                },
            };
        }

        // Get timer name for metadata
        let timer_name = timer.name();

        // Step 3a: Filter out zero measurements (kperf read failures)
        // kperf returns 0 on counter reset/read failure, which pollutes lower quantiles
        let baseline_nonzero: Vec<u64> = baseline_cycles.iter().copied().filter(|&c| c > 0).collect();
        let sample_nonzero: Vec<u64> = sample_cycles.iter().copied().filter(|&c| c > 0).collect();

        // If too many zeros were filtered, warn but continue
        let baseline_zeros = baseline_cycles.len() - baseline_nonzero.len();
        let sample_zeros = sample_cycles.len() - sample_nonzero.len();
        if baseline_zeros > baseline_cycles.len() / 10 || sample_zeros > sample_cycles.len() / 10 {
            eprintln!(
                "Warning: Filtered {} baseline and {} sample zero measurements (>10%). \
                 This may indicate kperf instability.",
                baseline_zeros, sample_zeros
            );
        }

        // Step 3b: Outlier filtering (pooled symmetric)
        let (filtered_baseline, filtered_sample, outlier_stats) =
            filter_outliers(&baseline_nonzero, &sample_nonzero, self.config.outlier_percentile);

        // Also filter interleaved samples using the same threshold
        let outlier_threshold = outlier_stats.threshold;
        let filtered_interleaved: Vec<(u64, Class)> = interleaved_cycles
            .iter()
            .zip(interleaved_classes.iter())
            .filter(|(&c, _)| c > 0 && c <= outlier_threshold)
            .map(|(&c, &class)| (c, class))
            .collect();

        eprintln!("[DEBUG] After outlier filtering ({}): baseline={} (removed {}), sample={} (removed {}), interleaved={}",
            self.config.outlier_percentile,
            filtered_baseline.len(),
            baseline_nonzero.len() - filtered_baseline.len(),
            filtered_sample.len(),
            sample_nonzero.len() - filtered_sample.len(),
            filtered_interleaved.len());

        // Convert cycles to nanoseconds
        let baseline_ns: Vec<f64> = filtered_baseline
            .iter()
            .map(|&c| timer.cycles_to_ns(c))
            .collect();
        let sample_ns: Vec<f64> = filtered_sample
            .iter()
            .map(|&c| timer.cycles_to_ns(c))
            .collect();

        // Convert interleaved to TimingSample (nanoseconds with class labels)
        let interleaved_samples: Vec<TimingSample> = filtered_interleaved
            .iter()
            .map(|&(c, class)| TimingSample {
                time_ns: timer.cycles_to_ns(c),
                class,
            })
            .collect();

        // Step 4: Compute full-sample quantile differences for CI gate
        let q_baseline_full = compute_deciles_fast(&baseline_ns);
        let q_sample_full = compute_deciles_fast(&sample_ns);
        let delta_full: Vector9 = q_baseline_full - q_sample_full;

        // Step 5: Sample splitting (calibration/inference)
        let n = baseline_ns.len().min(sample_ns.len());

        // Check minimum sample requirements (spec §2.3):
        // - ≥ 200: Normal 30/70 split
        // - 100–199: Warning; normal split with reduced reliability
        // - 50–99: Warning; use 50/50 split
        // - < 50: Return Unmeasurable (statistics meaningless)
        const MIN_SAMPLES_UNMEASURABLE: usize = 50;
        const MIN_SAMPLES_WARNING: usize = 100;
        const MIN_SAMPLES_NORMAL: usize = 200;

        if n < MIN_SAMPLES_UNMEASURABLE {
            // Too few samples for any meaningful statistics - return minimal result
            let runtime_secs = start_time.elapsed().as_secs_f64();
            let timer_name = if cfg!(target_arch = "x86_64") {
                "rdtsc"
            } else if cfg!(target_arch = "aarch64") {
                "cntvct_el0"
            } else {
                "Instant"
            };

            eprintln!(
                "[ERROR] Only {} samples after filtering (minimum: {}). \
                 Statistics are meaningless with so few samples.",
                n, MIN_SAMPLES_UNMEASURABLE
            );

            return TestResult {
                leak_probability: 0.5, // Maximum uncertainty
                bayes_factor: 1.0, // Neutral evidence
                effect: None,
                exploitability: Exploitability::Negligible,
                min_detectable_effect: MinDetectableEffect {
                    shift_ns: f64::INFINITY,
                    tail_ns: f64::INFINITY,
                },
                ci_gate: CiGate {
                    alpha: self.config.ci_alpha,
                    passed: true, // Don't fail CI on unmeasurable
                    threshold: f64::INFINITY,
                    max_observed: 0.0,
                    observed: [0.0; 9],
                },
                quality: MeasurementQuality::TooNoisy,
                outlier_fraction: outlier_stats.outlier_fraction,
                diagnostics: Diagnostics::all_ok(),
                metadata: Metadata {
                    samples_per_class: n,
                    cycles_per_ns: timer.cycles_per_ns(),
                    timer: timer_name.to_string(),
                    timer_resolution_ns: timer.resolution_ns(),
                    batching,
                    runtime_secs,
                },
            };
        }

        // Determine split strategy based on sample count
        let use_half_split = n < MIN_SAMPLES_WARNING;
        let calib_fraction = if use_half_split { 0.5 } else { self.config.calibration_fraction as f64 };

        if n < MIN_SAMPLES_WARNING {
            eprintln!(
                "[WARNING] Only {} samples (recommended: ≥{}). Using 50/50 split.",
                n, MIN_SAMPLES_NORMAL
            );
        } else if n < MIN_SAMPLES_NORMAL {
            eprintln!(
                "[WARNING] Only {} samples (recommended: ≥{}). Reduced reliability.",
                n, MIN_SAMPLES_NORMAL
            );
        }

        // Split interleaved samples for calibration (preserves temporal order for joint resampling)
        let n_interleaved = interleaved_samples.len();
        let n_calib_interleaved = ((n_interleaved as f64) * calib_fraction).round() as usize;

        let (calib_interleaved, infer_interleaved) = {
            let (calib, infer) = interleaved_samples.split_at(n_calib_interleaved);
            (calib.to_vec(), infer.to_vec())
        };

        // Also split baseline/sample for other computations
        let n_calib = ((n as f64) * calib_fraction).round() as usize;
        let (_, i_b) = split_calibration_temporal(&baseline_ns, n_calib);
        let (_, i_s) = split_calibration_temporal(&sample_ns, n_calib);
        let (infer_baseline, infer_sample) = (i_b, i_s);

        eprintln!("[DEBUG] After calibration split ({:.0}% calib, {:.0}% infer): calib_interleaved={}, infer_baseline={}, infer_sample={}",
            calib_fraction * 100.0,
            (1.0 - calib_fraction) * 100.0,
            calib_interleaved.len(),
            infer_baseline.len(),
            infer_sample.len());

        // Run preflight checks
        let interleaved_ns: Vec<f64> = interleaved_samples
            .iter()
            .map(|s| s.time_ns)
            .collect();
        let _preflight = run_all_checks(
            &baseline_ns,
            &sample_ns,
            &interleaved_ns,
            baseline_gen_time_ns,
            sample_gen_time_ns,
            timer.resolution_ns(),
        );

        // CALIBRATION PHASE
        // Step 6: Estimate covariance of Δ* = q_F* - q_R* via joint bootstrap
        // Joint resampling preserves temporal pairing for correct Var(Δ) estimation
        let calib_cov_estimate = bootstrap_difference_covariance(
            &calib_interleaved,
            self.config.cov_bootstrap_iterations,
            42,
        );
        // Save calibration covariance for diagnostics (before scaling)
        let calib_cov = calib_cov_estimate.matrix;

        // Compute inference covariance for diagnostics stationarity check
        let infer_cov_estimate = bootstrap_difference_covariance(
            &infer_interleaved,
            self.config.cov_bootstrap_iterations,
            42,
        );
        let infer_cov = infer_cov_estimate.matrix;

        // Step 6b: Scale covariance from calibration to inference sample sizes
        // Quantile variance scales as 1/n
        let n_calib = calib_interleaved.len();
        let n_infer = infer_baseline.len() + infer_sample.len();
        let scaled_cov = scale_covariance_for_inference(calib_cov, n_calib, n_infer);

        // Step 6c: Apply variance floor for numerical stability
        let pooled_cov = apply_variance_floor(scaled_cov, timer.resolution_ns());

        // Step 7: Estimate MDE from covariance (spec §2.7)
        let mde_estimate = estimate_mde(&pooled_cov, self.config.ci_alpha);

        // Prior scales from calibration (spec: max(2*MDE, min_effect_of_concern))
        let min_effect = self.config.min_effect_of_concern_ns;
        let prior_sigmas = (
            (2.0 * mde_estimate.shift_ns).max(min_effect),
            (2.0 * mde_estimate.tail_ns).max(min_effect),
        );

        // INFERENCE PHASE
        // Step 8: Compute quantile difference vector from inference data
        let q_baseline = compute_deciles_fast(&infer_baseline);
        let q_sample = compute_deciles_fast(&infer_sample);
        let delta_infer: Vector9 = q_baseline - q_sample;

        // Use scaled covariance for inference
        let cov_estimate = pooled_cov;

        // Step 9: Run CI Gate (Layer 1)
        let ci_gate_input = CiGateInput {
            observed_diff: delta_full,
            baseline_samples: &baseline_ns,
            sample_samples: &sample_ns,
            alpha: self.config.ci_alpha,
            bootstrap_iterations: self.config.ci_bootstrap_iterations,
            seed: self.config.measurement_seed,
            timer_resolution_ns: timer.resolution_ns(),
            min_effect_of_concern: self.config.min_effect_of_concern_ns,
        };
        let ci_gate = run_ci_gate(&ci_gate_input);

        // Step 10: Compute Bayes factor and posterior probability (Layer 2)
        let bayes_result =
            compute_bayes_factor(&delta_infer, &cov_estimate, prior_sigmas, self.config.prior_no_leak);
        let leak_probability = bayes_result.posterior_probability;

        // Step 11: Effect decomposition (if leak detected)
        // Also compute for diagnostics even if not returning effect
        let decomp = decompose_effect(
            &delta_infer,
            &cov_estimate,
            prior_sigmas,
        );
        // Convert Vector2 to [f64; 2] for diagnostics
        let posterior_mean = [decomp.posterior_mean[0], decomp.posterior_mean[1]];

        let effect = if leak_probability > 0.5 || !ci_gate.passed {
            // Scale batch-total effects to per-call effects by dividing by K
            let k_scale = batching.k as f64;
            Some(Effect {
                shift_ns: decomp.posterior_mean[0] / k_scale,
                tail_ns: decomp.posterior_mean[1] / k_scale,
                credible_interval_ns: (
                    decomp.effect_magnitude_ci.0 / k_scale,
                    decomp.effect_magnitude_ci.1 / k_scale,
                ),
                pattern: decomp.pattern,
            })
        } else {
            None
        };

        // Step 12: Exploitability assessment
        let effect_magnitude = effect
            .as_ref()
            .map(|e| (e.shift_ns.powi(2) + e.tail_ns.powi(2)).sqrt())
            .unwrap_or(0.0);
        let exploitability = Exploitability::from_effect_ns(effect_magnitude);

        // Step 13: Measurement quality assessment
        // MDE also needs to be scaled to per-call
        let k_scale = batching.k as f64;
        let scaled_mde = MinDetectableEffect {
            shift_ns: mde_estimate.shift_ns / k_scale,
            tail_ns: mde_estimate.tail_ns / k_scale,
        };
        let quality = MeasurementQuality::from_mde_ns(scaled_mde.shift_ns);

        let runtime_secs = start_time.elapsed().as_secs_f64();

        // Step 14: Compute Bayes factor from log BF (clamped for numerical stability)
        let bayes_factor = bayes_result.log_bayes_factor.exp().clamp(1e-100, 1e100);

        // Step 15: Compute diagnostics (stationarity, model fit, outlier asymmetry)
        let diagnostics = compute_diagnostics(
            &calib_cov,
            &infer_cov,
            &delta_infer,
            &posterior_mean,
            &outlier_stats,
        );

        // Step 16: Assemble and return TestResult
        TestResult {
            leak_probability,
            bayes_factor,
            effect,
            exploitability,
            min_detectable_effect: scaled_mde,
            ci_gate: CiGate {
                alpha: ci_gate.alpha,
                passed: ci_gate.passed,
                threshold: ci_gate.threshold,
                max_observed: ci_gate.max_observed,
                observed: ci_gate.observed,
            },
            quality,
            outlier_fraction: outlier_stats.outlier_fraction,
            diagnostics,
            metadata: Metadata {
                samples_per_class: infer_baseline.len().min(infer_sample.len()),
                cycles_per_ns: timer.cycles_per_ns(),
                timer: timer_name.to_string(),
                timer_resolution_ns: timer.resolution_ns(),
                batching,
                runtime_secs,
            },
        }
    }
}

/// Split samples temporally for calibration/inference.
///
/// CRITICAL: This must preserve temporal order for block bootstrap to work correctly.
/// Shuffling before block bootstrap destroys autocorrelation structure and causes
/// Σ₀ underestimation, leading to overconfident results.
///
/// Returns: (calibration_samples, inference_samples)
/// - calibration: first n_calib samples (in temporal order)
/// - inference: remaining samples (in temporal order)
fn split_calibration_temporal(
    data: &[f64],
    n_calib: usize,
) -> (Vec<f64>, Vec<f64>) {
    if n_calib == 0 || data.is_empty() {
        return (Vec::new(), data.to_vec());
    }

    let n_calib = n_calib.min(data.len());
    let (calib, infer) = data.split_at(n_calib);

    (calib.to_vec(), infer.to_vec())
}

// Environment variable parsing helpers
fn parse_usize_env(key: &str) -> Option<usize> {
    env::var(key).ok()?.parse().ok()
}

fn parse_u64_env(key: &str) -> Option<u64> {
    env::var(key).ok()?.parse().ok()
}

fn parse_f64_env(key: &str) -> Option<f64> {
    env::var(key).ok()?.parse().ok()
}

fn parse_f32_env(key: &str) -> Option<f32> {
    env::var(key).ok()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oracle_default_config() {
        let oracle = TimingOracle::new();
        assert_eq!(oracle.config().samples, 100_000);
        assert_eq!(oracle.config().warmup, 1_000);
        assert_eq!(oracle.config().ci_alpha, 0.01);
    }

    #[test]
    fn test_oracle_builder() {
        let oracle = TimingOracle::new()
            .samples(50_000)
            .warmup(500)
            .alpha(0.05)
            .min_effect_ns(5.0)
            .prior_no_leak(0.9);

        assert_eq!(oracle.config().samples, 50_000);
        assert_eq!(oracle.config().warmup, 500);
        assert_eq!(oracle.config().ci_alpha, 0.05);
        assert_eq!(oracle.config().min_effect_of_concern_ns, 5.0);
        assert_eq!(oracle.config().prior_no_leak, 0.9);
    }

    #[test]
    fn test_oracle_quick() {
        let oracle = TimingOracle::quick();
        assert_eq!(oracle.config().samples, 5_000);
        assert_eq!(oracle.config().warmup, 50);
        assert_eq!(oracle.config().cov_bootstrap_iterations, 50);
        assert_eq!(oracle.config().ci_bootstrap_iterations, 50);
    }

    #[test]
    fn test_oracle_timer_spec() {
        // Test default is Auto
        let oracle = TimingOracle::quick();
        assert_eq!(oracle.timer_spec, TimerSpec::Auto);

        // Test standard_timer() helper
        let oracle = TimingOracle::quick().standard_timer();
        assert_eq!(oracle.timer_spec, TimerSpec::Standard);

        // Test prefer_pmu() helper
        let oracle = TimingOracle::quick().prefer_pmu();
        assert_eq!(oracle.timer_spec, TimerSpec::PreferPmu);

        // Test explicit timer_spec()
        let oracle = TimingOracle::quick().timer_spec(TimerSpec::Standard);
        assert_eq!(oracle.timer_spec, TimerSpec::Standard);
    }
}
