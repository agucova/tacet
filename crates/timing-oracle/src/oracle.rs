//! Main `TimingOracle` entry point and builder.
//!
//! This module implements the adaptive Bayesian timing oracle (spec Section 2).
//! The oracle uses a two-phase approach:
//!
//! 1. **Calibration phase**: Collect initial samples to estimate covariance and set priors
//! 2. **Adaptive loop**: Collect batches until decision thresholds are reached
//!
//! The oracle returns one of four outcomes:
//! - `Pass`: No timing leak detected (leak_probability < pass_threshold)
//! - `Fail`: Timing leak confirmed (leak_probability > fail_threshold)
//! - `Inconclusive`: Cannot reach a definitive conclusion
//! - `Unmeasurable`: Operation too fast to measure on this platform

use std::env;
use std::hash::Hash;
use std::time::{Duration, Instant};

use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::adaptive::{
    calibrate, run_adaptive, AdaptiveConfig, AdaptiveOutcome, AdaptiveState, Calibration,
    CalibrationConfig, InconclusiveReason as AdaptiveInconclusiveReason,
};
use crate::analysis::classify_pattern;
use crate::config::Config;
use crate::constants::DEFAULT_SEED;
use crate::helpers::InputPair;
use crate::measurement::{BoxedTimer, TimerSpec};
use crate::analysis::compute_max_effect_ci;
use crate::result::{
    BatchingInfo, Diagnostics, EffectEstimate, Exploitability, InconclusiveReason, IssueCode,
    MeasurementQuality, Outcome, QualityIssue, ResearchOutcome, ResearchStatus,
};
use crate::types::{AttackerModel, Class};

/// Main entry point for adaptive Bayesian timing analysis.
///
/// Use the builder pattern to configure and run timing tests. The oracle
/// uses a two-phase approach:
///
/// 1. **Calibration**: Collect initial samples to estimate covariance and priors
/// 2. **Adaptive loop**: Collect batches until decision thresholds are reached
///
/// # Example
///
/// ```ignore
/// use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair, Outcome};
///
/// let inputs = InputPair::new(
///     || [0u8; 32],          // baseline: returns constant value
///     || rand::random(),     // sample: generates varied values
/// );
///
/// // Choose attacker model based on your threat scenario
/// let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
///     .test(inputs, |data| my_function(data));
///
/// match outcome {
///     Outcome::Pass { leak_probability, .. } => {
///         println!("No leak detected (P={:.1}%)", leak_probability * 100.0);
///     }
///     Outcome::Fail { leak_probability, exploitability, .. } => {
///         println!("Leak detected! P={:.1}%, {:?}", leak_probability * 100.0, exploitability);
///     }
///     Outcome::Inconclusive { reason, .. } => {
///         println!("Inconclusive: {:?}", reason);
///     }
///     Outcome::Unmeasurable { recommendation, .. } => {
///         println!("Operation too fast: {}", recommendation);
///     }
/// }
/// ```
///
/// # Automatic PMU Detection
///
/// When running with sudo/root privileges, the library automatically uses
/// cycle-accurate PMU timing (kperf on macOS, perf_event on Linux).
/// No code changes needed - just run with sudo.
///
/// # Attacker Model Presets
///
/// Choose the appropriate attacker model for your threat scenario:
///
/// | Preset | theta | Use case |
/// |--------|-------|----------|
/// | `SharedHardware` | ~0.6ns | SGX, cross-VM, containers |
/// | `AdjacentNetwork` | 100ns | LAN, HTTP/2 endpoints |
/// | `RemoteNetwork` | 50us | Public APIs, general internet |
/// | `Research` | 0 | Academic analysis (not for CI) |
#[derive(Debug, Clone)]
pub struct TimingOracle {
    config: Config,
    /// Timer specification (Auto by default - tries PMU first).
    timer_spec: TimerSpec,
}

impl TimingOracle {
    /// Create with an attacker model preset.
    ///
    /// The attacker model determines the minimum effect threshold (theta) that
    /// is considered practically significant. Different attacker models
    /// represent different threat scenarios with varying capabilities.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use timing_oracle::{TimingOracle, AttackerModel};
    ///
    /// // For public APIs exposed to the internet
    /// let oracle = TimingOracle::for_attacker(AttackerModel::RemoteNetwork);
    ///
    /// // For internal LAN services or HTTP/2 endpoints
    /// let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork);
    ///
    /// // For SGX enclaves or shared hosting (most strict)
    /// let oracle = TimingOracle::for_attacker(AttackerModel::SharedHardware);
    /// ```
    ///
    /// # Presets
    ///
    /// | Preset | theta | Use case |
    /// |--------|-------|----------|
    /// | `SharedHardware` | ~0.6ns | SGX, cross-VM, containers |
    /// | `AdjacentNetwork` | 100ns | LAN, HTTP/2 endpoints |
    /// | `RemoteNetwork` | 50us | Public APIs, general internet |
    /// | `Research` | 0 | Academic analysis (not for CI) |
    pub fn for_attacker(model: AttackerModel) -> Self {
        Self {
            config: Config {
                attacker_model: Some(model),
                ..Config::default()
            },
            timer_spec: TimerSpec::Auto,
        }
    }

    // =========================================================================
    // Builder methods
    // =========================================================================

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
    /// use timing_oracle::{TimingOracle, AttackerModel, TimerSpec};
    ///
    /// // Force standard timer (no PMU attempt)
    /// let result = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
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

    /// Set the time budget for the adaptive sampling loop.
    ///
    /// The oracle will stop and return `Inconclusive` if this time limit is
    /// reached without achieving a conclusive result.
    ///
    /// Default: 60 seconds
    ///
    /// # Example
    ///
    /// ```ignore
    /// use timing_oracle::{TimingOracle, AttackerModel};
    /// use std::time::Duration;
    ///
    /// let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    ///     .time_budget(Duration::from_secs(30));
    /// ```
    pub fn time_budget(mut self, duration: Duration) -> Self {
        self.config.time_budget = duration;
        self
    }

    /// Set the time budget in seconds.
    ///
    /// Convenience method for `.time_budget(Duration::from_secs(secs))`.
    pub fn time_budget_secs(mut self, secs: u64) -> Self {
        self.config.time_budget = Duration::from_secs(secs);
        self
    }

    /// Set the maximum number of samples per class.
    ///
    /// The oracle will stop and return `Inconclusive` if this limit is reached
    /// without achieving a conclusive result.
    ///
    /// Default: 1,000,000
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    pub fn max_samples(mut self, n: usize) -> Self {
        assert!(n > 0, "max_samples must be > 0 (got {})", n);
        self.config.max_samples = n;
        self
    }

    /// Set the batch size for adaptive sampling.
    ///
    /// Larger batches are more efficient but less responsive to early stopping.
    ///
    /// Default: 1,000
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    pub fn batch_size(mut self, n: usize) -> Self {
        assert!(n > 0, "batch_size must be > 0 (got {})", n);
        self.config.batch_size = n;
        self
    }

    /// Set the number of calibration samples.
    ///
    /// These samples are collected at the start to estimate the covariance
    /// matrix and set Bayesian priors. This is a fixed overhead.
    ///
    /// Default: 5,000
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0.
    pub fn calibration_samples(mut self, n: usize) -> Self {
        assert!(n > 0, "calibration_samples must be > 0 (got {})", n);
        self.config.calibration_samples = n;
        self
    }

    /// Set the pass threshold for leak probability.
    ///
    /// If the posterior probability of a timing leak falls below this threshold,
    /// the test passes. Default: 0.05 (5%).
    ///
    /// Lower values require more confidence to pass (more conservative).
    ///
    /// # Panics
    ///
    /// Panics if `threshold` is not in (0, 1) or >= fail_threshold.
    pub fn pass_threshold(mut self, threshold: f64) -> Self {
        assert!(
            threshold > 0.0 && threshold < 1.0,
            "pass_threshold must be in (0, 1), got {}",
            threshold
        );
        assert!(
            threshold < self.config.fail_threshold,
            "pass_threshold must be < fail_threshold"
        );
        self.config.pass_threshold = threshold;
        self
    }

    /// Set the fail threshold for leak probability.
    ///
    /// If the posterior probability of a timing leak exceeds this threshold,
    /// the test fails. Default: 0.95 (95%).
    ///
    /// Higher values require more confidence to fail (more conservative).
    ///
    /// # Panics
    ///
    /// Panics if `threshold` is not in (0, 1) or <= pass_threshold.
    pub fn fail_threshold(mut self, threshold: f64) -> Self {
        assert!(
            threshold > 0.0 && threshold < 1.0,
            "fail_threshold must be in (0, 1), got {}",
            threshold
        );
        assert!(
            threshold > self.config.pass_threshold,
            "fail_threshold must be > pass_threshold"
        );
        self.config.fail_threshold = threshold;
        self
    }

    /// Set warmup iterations.
    ///
    /// Warmup iterations warm CPU caches, stabilize frequency scaling, and
    /// trigger any JIT compilation before measurement begins.
    ///
    /// Default: 1,000
    pub fn warmup(mut self, n: usize) -> Self {
        self.config.warmup = n;
        self
    }

    /// Set bootstrap iterations for covariance estimation.
    ///
    /// Used during calibration to estimate the noise covariance matrix.
    /// More iterations give better estimates but take longer.
    ///
    /// Default: 2,000
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

    /// Set deterministic measurement seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.measurement_seed = Some(seed);
        self
    }

    /// Force discrete mode for testing.
    ///
    /// When set to `true`, the oracle uses discrete mode (m-out-of-n bootstrap
    /// with mid-quantiles) regardless of actual timer resolution. This is
    /// primarily useful for testing the discrete mode code path on machines
    /// with high-resolution timers.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use timing_oracle::{TimingOracle, AttackerModel};
    ///
    /// let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    ///     .force_discrete_mode(true)  // Force discrete mode for testing
    ///     .test(inputs, |data| operation(data));
    /// ```
    pub fn force_discrete_mode(mut self, force: bool) -> Self {
        self.config.force_discrete_mode = force;
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Merge configuration from environment variables.
    ///
    /// Reads the following environment variables to override settings:
    /// - `TO_TIME_BUDGET_SECS`: Time budget in seconds
    /// - `TO_MAX_SAMPLES`: Maximum samples per class
    /// - `TO_BATCH_SIZE`: Batch size for adaptive sampling
    /// - `TO_CALIBRATION_SAMPLES`: Number of calibration samples
    /// - `TO_PASS_THRESHOLD`: Pass threshold (e.g., "0.05")
    /// - `TO_FAIL_THRESHOLD`: Fail threshold (e.g., "0.95")
    /// - `TO_MIN_EFFECT_NS`: Minimum effect of concern in nanoseconds
    /// - `TO_SEED`: Deterministic measurement seed
    ///
    /// # Example
    ///
    /// ```ignore
    /// use timing_oracle::{TimingOracle, AttackerModel};
    ///
    /// // In CI, set TO_TIME_BUDGET_SECS=120 to increase time budget
    /// let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).from_env();
    /// ```
    pub fn from_env(mut self) -> Self {
        if let Some(secs) = parse_u64_env("TO_TIME_BUDGET_SECS") {
            self.config.time_budget = Duration::from_secs(secs);
        }
        if let Some(n) = parse_usize_env("TO_MAX_SAMPLES") {
            self.config.max_samples = n;
        }
        if let Some(n) = parse_usize_env("TO_BATCH_SIZE") {
            self.config.batch_size = n;
        }
        if let Some(n) = parse_usize_env("TO_CALIBRATION_SAMPLES") {
            self.config.calibration_samples = n;
        }
        if let Some(p) = parse_f64_env("TO_PASS_THRESHOLD") {
            if p > 0.0 && p < 1.0 && p < self.config.fail_threshold {
                self.config.pass_threshold = p;
            }
        }
        if let Some(p) = parse_f64_env("TO_FAIL_THRESHOLD") {
            if p > 0.0 && p < 1.0 && p > self.config.pass_threshold {
                self.config.fail_threshold = p;
            }
        }
        if let Some(ns) = parse_f64_env("TO_MIN_EFFECT_NS") {
            if ns >= 0.0 {
                self.config.min_effect_of_concern_ns = ns;
            }
        }
        if let Some(seed) = parse_u64_env("TO_SEED") {
            self.config.measurement_seed = Some(seed);
        }
        self
    }

    // =========================================================================
    // Main test method
    // =========================================================================

    /// Run a timing test with pre-generated inputs.
    ///
    /// This is the primary API for timing tests. It handles input pre-generation
    /// internally to ensure accurate measurements without generator overhead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair};
    ///
    /// let inputs = InputPair::new([0u8; 32], || rand::random());
    /// let result = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    ///     .test(inputs, |data| {
    ///         my_crypto_function(data);
    ///     });
    /// ```
    ///
    /// # How It Works
    ///
    /// 1. Pre-generates all baseline and sample inputs before measurement
    /// 2. Runs warmup iterations
    /// 3. Calibration phase: collects samples to estimate covariance
    /// 4. Adaptive phase: collects batches until a decision is reached
    ///
    /// # Arguments
    ///
    /// * `inputs` - An `InputPair` containing the baseline and sample generators
    /// * `operation` - Closure that performs the operation under test
    ///
    /// # Returns
    ///
    /// An `Outcome` which is one of:
    /// - `Pass`: No timing leak detected
    /// - `Fail`: Timing leak confirmed
    /// - `Inconclusive`: Cannot reach a definitive conclusion
    /// - `Unmeasurable`: Operation too fast to measure reliably
    pub fn test<T, F1, F2, F>(self, inputs: InputPair<T, F1, F2>, mut operation: F) -> Outcome
    where
        T: Clone + Hash,
        F1: FnMut() -> T,
        F2: FnMut() -> T,
        F: FnMut(&T),
    {
        let start_time = Instant::now();
        let mut rng: rand::rngs::StdRng = if let Some(seed) = self.config.measurement_seed {
            SeedableRng::seed_from_u64(seed)
        } else {
            SeedableRng::from_rng(&mut rand::rng())
        };

        // Step 1: Create timer based on spec (auto-detects PMU if available)
        let mut timer = self.timer_spec.create_timer();

        // Resolve the theta threshold based on attacker model and timer
        let raw_theta_ns = self
            .config
            .resolve_min_effect_ns(Some(timer.cycles_per_ns()), Some(timer.resolution_ns()));
        // Clamp theta to at least timer resolution to avoid degenerate priors
        // (Research mode returns 0, which causes zero prior covariance and Cholesky failure)
        let theta_ns = raw_theta_ns.max(timer.resolution_ns());

        // Step 2: Pre-generate ALL inputs before measurement (critical for accuracy)
        // We need enough for calibration + max adaptive samples
        let total_samples_needed = self.config.calibration_samples + self.config.max_samples;

        let baseline_gen_start = Instant::now();
        let baseline_inputs: Vec<T> = (0..total_samples_needed)
            .map(|_| inputs.baseline())
            .collect();
        let baseline_gen_time_ns =
            baseline_gen_start.elapsed().as_nanos() as f64 / total_samples_needed as f64;

        let sample_gen_start = Instant::now();
        let sample_inputs: Vec<T> = (0..total_samples_needed)
            .map(|_| {
                let value = inputs.generate_sample();
                inputs.track_value(&value);
                value
            })
            .collect();
        let sample_gen_time_ns =
            sample_gen_start.elapsed().as_nanos() as f64 / total_samples_needed as f64;

        // Step 3: Warmup
        for i in 0..self.config.warmup.min(total_samples_needed) {
            operation(&baseline_inputs[i % baseline_inputs.len()]);
            std::hint::black_box(());
            operation(&sample_inputs[i % sample_inputs.len()]);
            std::hint::black_box(());
        }

        // Step 4: Pilot phase to check measurability
        const PILOT_SAMPLES: usize = 100;
        let mut pilot_cycles = Vec::with_capacity(PILOT_SAMPLES * 2);

        for i in 0..PILOT_SAMPLES.min(total_samples_needed) {
            let cycles = timer.measure_cycles(|| {
                operation(&baseline_inputs[i]);
                std::hint::black_box(());
            });
            pilot_cycles.push(cycles);

            let cycles = timer.measure_cycles(|| {
                operation(&sample_inputs[i]);
                std::hint::black_box(());
            });
            pilot_cycles.push(cycles);
        }

        // Check if operation is measurable and select batching
        pilot_cycles.sort_unstable();
        let median_cycles = pilot_cycles[pilot_cycles.len() / 2];
        let median_ns = timer.cycles_to_ns(median_cycles);
        let resolution_ns = timer.resolution_ns();
        let ticks_per_call = median_ns / resolution_ns;

        if ticks_per_call <= 0.0 || !ticks_per_call.is_finite() {
            let threshold_ns = resolution_ns * crate::measurement::MIN_TICKS_SINGLE_CALL;
            let platform = format!(
                "{} ({}, {:.1}ns resolution)",
                std::env::consts::OS,
                timer.name(),
                timer.resolution_ns()
            );
            return Outcome::Unmeasurable {
                operation_ns: median_ns,
                threshold_ns,
                platform,
                recommendation:
                    "Timer returned non-finite measurements; retry on a more stable system."
                        .to_string(),
            };
        }

        // Determine batching K value
        let (k, _batching): (u32, BatchingInfo) = match self.config.iterations_per_sample {
            crate::config::IterationsPerSample::Fixed(k) => {
                let k = k.max(1) as u32;
                (
                    k,
                    BatchingInfo {
                        enabled: k > 1,
                        k,
                        ticks_per_batch: ticks_per_call * k as f64,
                        rationale: format!("fixed batching K={}", k),
                        unmeasurable: None,
                    },
                )
            }
            crate::config::IterationsPerSample::Auto => {
                if ticks_per_call >= crate::measurement::TARGET_TICKS_PER_BATCH {
                    (
                        1,
                        BatchingInfo {
                            enabled: false,
                            k: 1,
                            ticks_per_batch: ticks_per_call,
                            rationale: format!(
                                "no batching needed ({:.1} ticks/call)",
                                ticks_per_call
                            ),
                            unmeasurable: None,
                        },
                    )
                } else {
                    let k_raw =
                        (crate::measurement::TARGET_TICKS_PER_BATCH / ticks_per_call).ceil() as u32;
                    let k = k_raw.clamp(1, crate::measurement::MAX_BATCH_SIZE);
                    let ticks_per_batch = ticks_per_call * k as f64;
                    let partial = ticks_per_batch < crate::measurement::TARGET_TICKS_PER_BATCH;

                    if partial {
                        // Operation is unmeasurable even with max batching
                        let platform = format!(
                            "{} ({}, {:.1}ns resolution)",
                            std::env::consts::OS,
                            timer.name(),
                            timer.resolution_ns()
                        );

                        #[cfg(target_os = "macos")]
                        let suggestion = "Run with sudo to enable kperf cycle counting (~0.3ns resolution), or increase operation complexity";
                        #[cfg(target_os = "linux")]
                        let suggestion = "Run with sudo to enable perf_event cycle counting (~0.3ns resolution), or increase operation complexity";
                        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
                        let suggestion = "Increase operation complexity (current operation is too fast to measure reliably)";

                        return Outcome::Unmeasurable {
                            operation_ns: median_ns,
                            threshold_ns: resolution_ns
                                * crate::measurement::TARGET_TICKS_PER_BATCH
                                / k as f64,
                            platform,
                            recommendation: suggestion.to_string(),
                        };
                    }

                    (
                        k,
                        BatchingInfo {
                            enabled: k > 1,
                            k,
                            ticks_per_batch,
                            rationale: format!("K={} ({:.1} ticks/batch)", k, ticks_per_batch),
                            unmeasurable: None,
                        },
                    )
                }
            }
        };

        // Step 5: CALIBRATION PHASE - Collect calibration samples
        // Cap calibration samples to at most 50% of max_samples to ensure room for inference
        let n_cal = self
            .config
            .calibration_samples
            .min(self.config.max_samples / 2);

        let mut calibration_baseline_cycles = Vec::with_capacity(n_cal);
        let mut calibration_sample_cycles = Vec::with_capacity(n_cal);

        // Create interleaved schedule for calibration
        let mut cal_schedule: Vec<(Class, usize)> = Vec::with_capacity(n_cal * 2);
        for i in 0..n_cal {
            cal_schedule.push((Class::Baseline, i));
            cal_schedule.push((Class::Sample, i));
        }
        cal_schedule.shuffle(&mut rng);

        for (class, idx) in cal_schedule {
            match class {
                Class::Baseline => {
                    let cycles = timer.measure_cycles(|| {
                        for _ in 0..k {
                            operation(&baseline_inputs[idx]);
                            std::hint::black_box(());
                        }
                    });
                    calibration_baseline_cycles.push(cycles);
                }
                Class::Sample => {
                    let cycles = timer.measure_cycles(|| {
                        for _ in 0..k {
                            operation(&sample_inputs[idx]);
                            std::hint::black_box(());
                        }
                    });
                    calibration_sample_cycles.push(cycles);
                }
            }
        }

        // Check if preflight should be skipped
        let skip_preflight = std::env::var("TIMING_ORACLE_SKIP_PREFLIGHT").is_ok();

        // Perform calibration
        let ns_per_tick = timer.resolution_ns();
        let cal_config = CalibrationConfig {
            calibration_samples: n_cal,
            bootstrap_iterations: self.config.cov_bootstrap_iterations.min(200), // Fewer for calibration
            timer_resolution_ns: ns_per_tick,
            theta_ns,
            alpha: 0.01,
            seed: self.config.measurement_seed.unwrap_or(DEFAULT_SEED),
            baseline_gen_time_ns: Some(baseline_gen_time_ns),
            sample_gen_time_ns: Some(sample_gen_time_ns),
            skip_preflight,
        };

        let calibration = match calibrate(
            &calibration_baseline_cycles,
            &calibration_sample_cycles,
            1.0 / timer.cycles_per_ns(), // ns per cycle
            &cal_config,
        ) {
            Ok(cal) => cal,
            Err(e) => {
                // Calibration failed - return Inconclusive
                let diagnostics = Diagnostics {
                    calibration_samples: n_cal,
                    total_time_secs: start_time.elapsed().as_secs_f64(),
                    warnings: vec![format!("Calibration failed: {}", e)],
                    ..Diagnostics::default()
                };
                return Outcome::Inconclusive {
                    reason: InconclusiveReason::DataTooNoisy {
                        message: format!("Calibration failed: {}", e),
                        guidance: "Try increasing calibration_samples or reducing system noise"
                            .to_string(),
                    },
                    leak_probability: 0.5,
                    effect: EffectEstimate::default(),
                    samples_used: n_cal,
                    quality: MeasurementQuality::TooNoisy,
                    diagnostics,
                    theta_user: theta_ns,
                    theta_eff: theta_ns,
                    theta_floor: 0.0, // Unknown during calibration failure
                };
            }
        };

        // Step 6: RESEARCH MODE CHECK
        // If using AttackerModel::Research, run research-specific loop
        if matches!(self.config.attacker_model, Some(AttackerModel::Research)) {
            return self.run_research_mode(
                calibration,
                &calibration_baseline_cycles,
                &calibration_sample_cycles,
                &baseline_inputs,
                &sample_inputs,
                n_cal,
                k,
                &mut timer,
                &mut operation,
                &mut rng,
                total_samples_needed,
                start_time,
            );
        }

        // Step 7: ADAPTIVE PHASE - Collect samples until decision
        let adaptive_config = AdaptiveConfig {
            batch_size: self.config.batch_size,
            pass_threshold: self.config.pass_threshold,
            fail_threshold: self.config.fail_threshold,
            time_budget: self.config.time_budget,
            max_samples: self.config.max_samples,
            theta_ns,
            seed: self.config.measurement_seed.unwrap_or(DEFAULT_SEED),
            ..AdaptiveConfig::default()
        };

        let mut adaptive_state = AdaptiveState::with_capacity(self.config.max_samples);

        // Add calibration samples to adaptive state (they count toward total)
        adaptive_state.add_batch(
            calibration_baseline_cycles.clone(),
            calibration_sample_cycles.clone(),
        );

        // Create stationarity tracker for drift detection (spec Section 3.2.1)
        let ns_per_cycle = 1.0 / timer.cycles_per_ns();
        let tracker_seed = self.config.measurement_seed.unwrap_or(DEFAULT_SEED).wrapping_add(0xDEAD);
        let mut stationarity_tracker = crate::analysis::StationarityTracker::new(
            self.config.max_samples * 2, // baseline + sample
            tracker_seed,
        );

        // Add calibration samples to stationarity tracker
        for &cycles in &calibration_baseline_cycles {
            stationarity_tracker.push(cycles as f64 * ns_per_cycle);
        }
        for &cycles in &calibration_sample_cycles {
            stationarity_tracker.push(cycles as f64 * ns_per_cycle);
        }

        // Adaptive loop
        let mut input_idx = n_cal; // Start after calibration samples
        loop {
            // Check time budget
            if adaptive_state.elapsed() > self.config.time_budget {
                let posterior = adaptive_state.current_posterior();
                let leak_probability = posterior.map(|p| p.leak_probability).unwrap_or(0.5);
                let stationarity = stationarity_tracker.compute();

                return self.build_inconclusive_outcome(
                    InconclusiveReason::TimeBudgetExceeded {
                        current_probability: leak_probability,
                        samples_collected: adaptive_state.n_total(),
                    },
                    &adaptive_state,
                    &calibration,
                    &timer,
                    start_time,
                    theta_ns,
                    stationarity,
                );
            }

            // Check sample budget
            if adaptive_state.n_total() >= self.config.max_samples {
                let posterior = adaptive_state.current_posterior();
                let leak_probability = posterior.map(|p| p.leak_probability).unwrap_or(0.5);
                let stationarity = stationarity_tracker.compute();

                return self.build_inconclusive_outcome(
                    InconclusiveReason::SampleBudgetExceeded {
                        current_probability: leak_probability,
                        samples_collected: adaptive_state.n_total(),
                    },
                    &adaptive_state,
                    &calibration,
                    &timer,
                    start_time,
                    theta_ns,
                    stationarity,
                );
            }

            // Collect a batch
            let batch_size = self.config.batch_size.min(total_samples_needed - input_idx);
            if batch_size == 0 {
                // Out of pre-generated inputs
                let posterior = adaptive_state.current_posterior();
                let leak_probability = posterior.map(|p| p.leak_probability).unwrap_or(0.5);
                let stationarity = stationarity_tracker.compute();

                return self.build_inconclusive_outcome(
                    InconclusiveReason::SampleBudgetExceeded {
                        current_probability: leak_probability,
                        samples_collected: adaptive_state.n_total(),
                    },
                    &adaptive_state,
                    &calibration,
                    &timer,
                    start_time,
                    theta_ns,
                    stationarity,
                );
            }

            let mut batch_baseline = Vec::with_capacity(batch_size);
            let mut batch_sample = Vec::with_capacity(batch_size);

            // Create interleaved schedule for batch
            let mut batch_schedule: Vec<(Class, usize)> = Vec::with_capacity(batch_size * 2);
            for i in 0..batch_size {
                let global_idx = input_idx + i;
                batch_schedule.push((Class::Baseline, global_idx));
                batch_schedule.push((Class::Sample, global_idx));
            }
            batch_schedule.shuffle(&mut rng);

            for (class, idx) in batch_schedule {
                match class {
                    Class::Baseline => {
                        let cycles = timer.measure_cycles(|| {
                            for _ in 0..k {
                                operation(&baseline_inputs[idx]);
                                std::hint::black_box(());
                            }
                        });
                        batch_baseline.push(cycles);
                        stationarity_tracker.push(cycles as f64 * ns_per_cycle);
                    }
                    Class::Sample => {
                        let cycles = timer.measure_cycles(|| {
                            for _ in 0..k {
                                operation(&sample_inputs[idx]);
                                std::hint::black_box(());
                            }
                        });
                        batch_sample.push(cycles);
                        stationarity_tracker.push(cycles as f64 * ns_per_cycle);
                    }
                }
            }

            input_idx += batch_size;
            adaptive_state.add_batch(batch_baseline, batch_sample);

            // Run one step of adaptive analysis
            let outcome = run_adaptive(
                &calibration,
                &mut adaptive_state,
                1.0 / timer.cycles_per_ns(),
                &adaptive_config,
            );

            match outcome {
                AdaptiveOutcome::LeakDetected {
                    posterior,
                    samples_per_class,
                    elapsed: _,
                } => {
                    let stationarity = stationarity_tracker.compute();
                    return self.build_fail_outcome(
                        &posterior,
                        samples_per_class,
                        &calibration,
                        &timer,
                        start_time,
                        theta_ns,
                        stationarity,
                    );
                }
                AdaptiveOutcome::NoLeakDetected {
                    posterior,
                    samples_per_class,
                    elapsed: _,
                } => {
                    let stationarity = stationarity_tracker.compute();
                    return self.build_pass_outcome(
                        &posterior,
                        samples_per_class,
                        &calibration,
                        &timer,
                        start_time,
                        theta_ns,
                        stationarity,
                    );
                }
                AdaptiveOutcome::Continue { posterior, .. } => {
                    // Quality gates passed but no decision yet - continue collecting samples
                    adaptive_state.update_posterior(posterior);
                    continue;
                }
                AdaptiveOutcome::Inconclusive { reason, .. } => {
                    // Real stop condition (DataTooNoisy, NotLearning, WouldTakeTooLong, Timeout)
                    let result_reason = convert_adaptive_reason(&reason);
                    let stationarity = stationarity_tracker.compute();
                    return self.build_inconclusive_outcome(
                        result_reason,
                        &adaptive_state,
                        &calibration,
                        &timer,
                        start_time,
                        theta_ns,
                        stationarity,
                    );
                }
            }
        }
    }

    // =========================================================================
    // Outcome builders
    // =========================================================================

    fn build_pass_outcome(
        &self,
        posterior: &Posterior,
        samples_used: usize,
        calibration: &Calibration,
        timer: &BoxedTimer,
        start_time: Instant,
        theta_ns: f64,
        stationarity: Option<crate::analysis::StationarityResult>,
    ) -> Outcome {
        let effect = build_effect_estimate(posterior, theta_ns);
        let quality = MeasurementQuality::from_mde_ns(calibration.mde_shift_ns);
        let diagnostics =
            build_diagnostics(calibration, timer, start_time, &self.config, theta_ns, posterior.model_fit_q, stationarity);

        Outcome::Pass {
            leak_probability: posterior.leak_probability,
            effect,
            samples_used,
            quality,
            diagnostics,
            theta_user: theta_ns,
            theta_eff: calibration.theta_eff,
            theta_floor: (calibration.c_floor / (samples_used as f64).sqrt())
                .max(calibration.theta_tick),
        }
    }

    fn build_fail_outcome(
        &self,
        posterior: &Posterior,
        samples_used: usize,
        calibration: &Calibration,
        timer: &BoxedTimer,
        start_time: Instant,
        theta_ns: f64,
        stationarity: Option<crate::analysis::StationarityResult>,
    ) -> Outcome {
        let effect = build_effect_estimate(posterior, theta_ns);
        let exploitability = Exploitability::from_effect_ns(effect.total_effect_ns());
        let quality = MeasurementQuality::from_mde_ns(calibration.mde_shift_ns);
        let diagnostics =
            build_diagnostics(calibration, timer, start_time, &self.config, theta_ns, posterior.model_fit_q, stationarity);

        Outcome::Fail {
            leak_probability: posterior.leak_probability,
            effect,
            exploitability,
            samples_used,
            quality,
            diagnostics,
            theta_user: theta_ns,
            theta_eff: calibration.theta_eff,
            theta_floor: (calibration.c_floor / (samples_used as f64).sqrt())
                .max(calibration.theta_tick),
        }
    }

    fn build_inconclusive_outcome(
        &self,
        reason: InconclusiveReason,
        state: &AdaptiveState,
        calibration: &Calibration,
        timer: &BoxedTimer,
        start_time: Instant,
        theta_ns: f64,
        stationarity: Option<crate::analysis::StationarityResult>,
    ) -> Outcome {
        let posterior = state.current_posterior();
        let leak_probability = posterior.map(|p| p.leak_probability).unwrap_or(0.5);
        let effect = posterior
            .map(|p| build_effect_estimate(p, theta_ns))
            .unwrap_or_default();
        let quality = MeasurementQuality::from_mde_ns(calibration.mde_shift_ns);
        let model_fit_q = posterior.map(|p| p.model_fit_q).unwrap_or(0.0);
        let diagnostics =
            build_diagnostics(calibration, timer, start_time, &self.config, theta_ns, model_fit_q, stationarity);

        Outcome::Inconclusive {
            reason,
            leak_probability,
            effect,
            samples_used: state.n_total(),
            quality,
            diagnostics,
            theta_user: theta_ns,
            theta_eff: calibration.theta_eff,
            theta_floor: (calibration.c_floor / (state.n_total() as f64).sqrt())
                .max(calibration.theta_tick),
        }
    }

    // =========================================================================
    // Research mode implementation
    // =========================================================================

    /// Run research mode loop (spec v4.1).
    ///
    /// Research mode uses CI-based stopping conditions instead of Bayesian thresholds:
    /// - `CI.lower > 1.1 * theta_floor` → EffectDetected
    /// - `CI.upper < 0.9 * theta_floor` → NoEffectDetected
    /// - `theta_floor <= theta_tick * 1.01` → ResolutionLimitReached
    /// - Quality gates → QualityIssue
    /// - Budget exhausted → BudgetExhausted
    #[allow(clippy::too_many_arguments)]
    fn run_research_mode<T, F, R>(
        &self,
        calibration: Calibration,
        calibration_baseline_cycles: &[u64],
        calibration_sample_cycles: &[u64],
        baseline_inputs: &[T],
        sample_inputs: &[T],
        n_cal: usize,
        k: u32,
        timer: &mut BoxedTimer,
        operation: &mut F,
        rng: &mut R,
        total_samples_needed: usize,
        start_time: Instant,
    ) -> Outcome
    where
        T: Clone + Hash,
        F: FnMut(&T),
        R: rand::Rng,
    {
        use crate::adaptive::{run_adaptive, AdaptiveConfig, AdaptiveOutcome};

        // Set up state with calibration samples
        let mut adaptive_state = AdaptiveState::with_capacity(self.config.max_samples);
        adaptive_state.add_batch(
            calibration_baseline_cycles.to_vec(),
            calibration_sample_cycles.to_vec(),
        );

        // Use a minimal theta for the adaptive machinery (we don't use thresholds in Research mode)
        let theta_ns = timer.resolution_ns();

        let adaptive_config = AdaptiveConfig {
            batch_size: self.config.batch_size,
            pass_threshold: 0.0,  // Not used in research mode
            fail_threshold: 1.0,  // Not used in research mode
            time_budget: self.config.time_budget,
            max_samples: self.config.max_samples,
            theta_ns,
            seed: self.config.measurement_seed.unwrap_or(DEFAULT_SEED),
            ..AdaptiveConfig::default()
        };

        let mut input_idx = n_cal;

        loop {
            // Check time budget
            if adaptive_state.elapsed() > self.config.time_budget {
                return self.build_research_outcome(
                    ResearchStatus::BudgetExhausted,
                    &adaptive_state,
                    &calibration,
                    timer,
                    start_time,
                );
            }

            // Check sample budget
            if adaptive_state.n_total() >= self.config.max_samples {
                return self.build_research_outcome(
                    ResearchStatus::BudgetExhausted,
                    &adaptive_state,
                    &calibration,
                    timer,
                    start_time,
                );
            }

            // Collect a batch
            let batch_size = self.config.batch_size.min(total_samples_needed - input_idx);
            if batch_size == 0 {
                return self.build_research_outcome(
                    ResearchStatus::BudgetExhausted,
                    &adaptive_state,
                    &calibration,
                    timer,
                    start_time,
                );
            }

            let mut batch_baseline = Vec::with_capacity(batch_size);
            let mut batch_sample = Vec::with_capacity(batch_size);

            // Create interleaved schedule for batch
            let mut batch_schedule: Vec<(Class, usize)> = Vec::with_capacity(batch_size * 2);
            for i in 0..batch_size {
                let global_idx = input_idx + i;
                batch_schedule.push((Class::Baseline, global_idx));
                batch_schedule.push((Class::Sample, global_idx));
            }
            batch_schedule.shuffle(rng);

            for (class, idx) in batch_schedule {
                match class {
                    Class::Baseline => {
                        let cycles = timer.measure_cycles(|| {
                            for _ in 0..k {
                                operation(&baseline_inputs[idx]);
                                std::hint::black_box(());
                            }
                        });
                        batch_baseline.push(cycles);
                    }
                    Class::Sample => {
                        let cycles = timer.measure_cycles(|| {
                            for _ in 0..k {
                                operation(&sample_inputs[idx]);
                                std::hint::black_box(());
                            }
                        });
                        batch_sample.push(cycles);
                    }
                }
            }

            input_idx += batch_size;
            adaptive_state.add_batch(batch_baseline, batch_sample);

            // Run one step of adaptive analysis to get posterior
            let outcome = run_adaptive(
                &calibration,
                &mut adaptive_state,
                1.0 / timer.cycles_per_ns(),
                &adaptive_config,
            );

            // Extract posterior from outcome
            let posterior = match &outcome {
                AdaptiveOutcome::Continue { posterior, .. } => posterior,
                AdaptiveOutcome::LeakDetected { posterior, .. } => posterior,
                AdaptiveOutcome::NoLeakDetected { posterior, .. } => posterior,
                AdaptiveOutcome::Inconclusive { reason, .. } => {
                    // Quality gate failed - return with QualityIssue
                    // Convert adaptive reason to our InconclusiveReason
                    let inconclusive_reason = convert_adaptive_reason(&reason);
                    return self.build_research_outcome(
                        ResearchStatus::QualityIssue(inconclusive_reason),
                        &adaptive_state,
                        &calibration,
                        timer,
                        start_time,
                    );
                }
            };

            // Compute theta_floor at current sample size
            let n = adaptive_state.n_total() as f64;
            let theta_floor = (calibration.c_floor / n.sqrt()).max(calibration.theta_tick);

            // Check resolution limit: theta_floor <= theta_tick * 1.01
            if theta_floor <= calibration.theta_tick * 1.01 {
                return self.build_research_outcome(
                    ResearchStatus::ResolutionLimitReached,
                    &adaptive_state,
                    &calibration,
                    timer,
                    start_time,
                );
            }

            // Compute max effect CI
            let max_effect_ci = compute_max_effect_ci(
                &posterior.beta_mean,
                &posterior.beta_cov,
                self.config.measurement_seed.unwrap_or(DEFAULT_SEED),
            );

            // Check stopping conditions
            // EffectDetected: CI.lower > 1.1 * theta_floor
            if max_effect_ci.ci.0 > 1.1 * theta_floor {
                return self.build_research_outcome(
                    ResearchStatus::EffectDetected,
                    &adaptive_state,
                    &calibration,
                    timer,
                    start_time,
                );
            }

            // NoEffectDetected: CI.upper < 0.9 * theta_floor
            if max_effect_ci.ci.1 < 0.9 * theta_floor {
                return self.build_research_outcome(
                    ResearchStatus::NoEffectDetected,
                    &adaptive_state,
                    &calibration,
                    timer,
                    start_time,
                );
            }

            // Continue collecting samples
            adaptive_state.update_posterior(posterior.clone());
        }
    }

    /// Build a ResearchOutcome from current state.
    fn build_research_outcome(
        &self,
        status: ResearchStatus,
        state: &AdaptiveState,
        calibration: &Calibration,
        timer: &BoxedTimer,
        start_time: Instant,
    ) -> Outcome {
        let posterior = state.current_posterior();
        let theta_ns = timer.resolution_ns();

        // Compute theta_floor at final sample size
        let n = state.n_total() as f64;
        let theta_floor = (calibration.c_floor / n.sqrt()).max(calibration.theta_tick);

        // Compute max effect CI
        let (max_effect_ns, max_effect_ci, detectable) = if let Some(p) = posterior {
            let ci = compute_max_effect_ci(
                &p.beta_mean,
                &p.beta_cov,
                self.config.measurement_seed.unwrap_or(DEFAULT_SEED),
            );
            let detectable = ci.ci.0 > theta_floor;
            (ci.mean, ci.ci, detectable)
        } else {
            (0.0, (0.0, 0.0), false)
        };

        // Check model mismatch (non-blocking in research mode)
        let model_mismatch = posterior
            .map(|p| p.model_fit_q > calibration.q_thresh)
            .unwrap_or(false);

        // Build effect estimate
        let effect = posterior
            .map(|p| build_effect_estimate(p, theta_ns))
            .unwrap_or_default();

        let quality = MeasurementQuality::from_mde_ns(calibration.mde_shift_ns);
        let model_fit_q = posterior.map(|p| p.model_fit_q).unwrap_or(0.0);
        // Research mode doesn't track stationarity separately (would need refactoring)
        let diagnostics =
            build_diagnostics(calibration, timer, start_time, &self.config, theta_ns, model_fit_q, None);

        Outcome::Research(ResearchOutcome {
            status,
            max_effect_ns,
            max_effect_ci,
            theta_floor,
            detectable,
            model_mismatch,
            effect,
            samples_used: state.n_total(),
            quality,
            diagnostics,
        })
    }
}

// =============================================================================
// Helper functions
// =============================================================================

use crate::adaptive::Posterior;

/// Build an EffectEstimate from a posterior.
fn build_effect_estimate(posterior: &Posterior, _theta_ns: f64) -> EffectEstimate {
    let pattern = classify_pattern(&posterior.beta_mean, &posterior.beta_cov);

    // Compute credible interval from posterior covariance
    let shift_std = posterior.shift_se();
    let tail_std = posterior.tail_se();
    let total_effect = (posterior.shift_ns().powi(2) + posterior.tail_ns().powi(2)).sqrt();
    let total_std = ((shift_std.powi(2) + tail_std.powi(2)) / 2.0).sqrt();

    let ci_low = (total_effect - 1.96 * total_std).max(0.0);
    let ci_high = total_effect + 1.96 * total_std;

    EffectEstimate {
        shift_ns: posterior.shift_ns(),
        tail_ns: posterior.tail_ns(),
        credible_interval_ns: (ci_low, ci_high),
        pattern,
        interpretation_caveat: None, // Set when model fit is poor
    }
}

/// Build diagnostics from calibration, timer, and config info.
fn build_diagnostics(
    calibration: &Calibration,
    timer: &BoxedTimer,
    start_time: Instant,
    config: &Config,
    theta_ns: f64,
    model_fit_q: f64,
    stationarity: Option<crate::analysis::StationarityResult>,
) -> Diagnostics {
    // Convert preflight warnings to PreflightWarningInfo
    let preflight = &calibration.preflight_result;
    let mut preflight_warnings = Vec::new();

    for warning in &preflight.warnings.sanity {
        preflight_warnings.push(warning.to_warning_info());
    }
    for warning in &preflight.warnings.generator {
        preflight_warnings.push(warning.to_warning_info());
    }
    for warning in &preflight.warnings.autocorr {
        preflight_warnings.push(warning.to_warning_info());
    }
    for warning in &preflight.warnings.system {
        preflight_warnings.push(warning.to_warning_info());
    }
    for warning in &preflight.warnings.resolution {
        preflight_warnings.push(warning.to_warning_info());
    }

    // Format attacker model name
    let attacker_model = config.attacker_model.as_ref().map(|m| format!("{:?}", m));

    // Build platform string
    let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);

    // Check for ThresholdElevated: user requested threshold lower than measurement floor
    let mut quality_issues = Vec::new();
    if calibration.theta_ns > 0.0 && calibration.theta_eff > calibration.theta_ns {
        quality_issues.push(QualityIssue {
            code: IssueCode::ThresholdElevated,
            message: format!(
                "Requested threshold {} ns elevated to {} ns due to measurement floor",
                calibration.theta_ns, calibration.theta_eff
            ),
            guidance: format!(
                "Reporting probabilities for theta_eff = {:.1} ns. \
                 For better resolution, use a cycle counter (PmuTimer on macOS, LinuxPerfTimer on Linux) \
                 or increase sample count.",
                calibration.theta_eff
            ),
        });
    }

    Diagnostics {
        dependence_length: calibration.block_length,
        effective_sample_size: calibration.calibration_samples / calibration.block_length.max(1),
        // Use stationarity result if available, otherwise assume no drift
        stationarity_ratio: stationarity.map(|s| s.ratio).unwrap_or(1.0),
        stationarity_ok: stationarity.map(|s| s.ok).unwrap_or(true),
        model_fit_chi2: model_fit_q,
        model_fit_threshold: calibration.q_thresh,
        model_fit_ok: model_fit_q <= calibration.q_thresh,
        outlier_rate_baseline: 0.0,
        outlier_rate_sample: 0.0,
        outlier_asymmetry_ok: true,
        discrete_mode: calibration.discrete_mode,
        timer_resolution_ns: timer.resolution_ns(),
        duplicate_fraction: 0.0,
        preflight_ok: preflight.is_valid,
        calibration_samples: calibration.calibration_samples,
        total_time_secs: start_time.elapsed().as_secs_f64(),
        warnings: Vec::new(),
        quality_issues,
        preflight_warnings,
        seed: config.measurement_seed,
        attacker_model,
        threshold_ns: theta_ns,
        timer_name: timer.name().to_string(),
        platform,
    }
}

/// Convert adaptive module's InconclusiveReason to result module's InconclusiveReason.
fn convert_adaptive_reason(reason: &AdaptiveInconclusiveReason) -> InconclusiveReason {
    match reason {
        AdaptiveInconclusiveReason::DataTooNoisy {
            message, guidance, ..
        } => InconclusiveReason::DataTooNoisy {
            message: message.clone(),
            guidance: guidance.clone(),
        },
        AdaptiveInconclusiveReason::NotLearning {
            message, guidance, ..
        } => InconclusiveReason::NotLearning {
            message: message.clone(),
            guidance: guidance.clone(),
        },
        AdaptiveInconclusiveReason::WouldTakeTooLong {
            estimated_time_secs,
            samples_needed,
            guidance,
        } => InconclusiveReason::WouldTakeTooLong {
            estimated_time_secs: *estimated_time_secs,
            samples_needed: *samples_needed,
            guidance: guidance.clone(),
        },
        AdaptiveInconclusiveReason::TimeBudgetExceeded {
            current_probability,
            samples_collected,
            ..
        } => InconclusiveReason::TimeBudgetExceeded {
            current_probability: *current_probability,
            samples_collected: *samples_collected,
        },
        AdaptiveInconclusiveReason::SampleBudgetExceeded {
            current_probability,
            samples_collected,
        } => InconclusiveReason::SampleBudgetExceeded {
            current_probability: *current_probability,
            samples_collected: *samples_collected,
        },
        AdaptiveInconclusiveReason::ConditionsChanged {
            message, guidance, ..
        } => InconclusiveReason::ConditionsChanged {
            message: message.clone(),
            guidance: guidance.clone(),
        },
        AdaptiveInconclusiveReason::ThresholdUnachievable {
            theta_user,
            best_achievable,
            message,
            guidance,
        } => InconclusiveReason::ThresholdUnachievable {
            theta_user: *theta_user,
            best_achievable: *best_achievable,
            message: message.clone(),
            guidance: guidance.clone(),
        },
        AdaptiveInconclusiveReason::ModelMismatch {
            q_statistic,
            q_threshold,
            message,
            guidance,
        } => InconclusiveReason::ModelMismatch {
            q_statistic: *q_statistic,
            q_threshold: *q_threshold,
            message: message.clone(),
            guidance: guidance.clone(),
        },
    }
}

/// Compute minimum uniqueness ratio for discrete mode detection.
pub fn compute_min_uniqueness_ratio(baseline: &[f64], sample: &[f64]) -> f64 {
    use std::collections::HashSet;

    let unique_baseline: HashSet<i64> = baseline.iter().map(|&v| (v * 1000.0) as i64).collect();
    let unique_sample: HashSet<i64> = sample.iter().map(|&v| (v * 1000.0) as i64).collect();

    let ratio_baseline = unique_baseline.len() as f64 / baseline.len().max(1) as f64;
    let ratio_sample = unique_sample.len() as f64 / sample.len().max(1) as f64;

    ratio_baseline.min(ratio_sample)
}

// =============================================================================
// Environment variable parsing helpers
// =============================================================================

fn parse_usize_env(name: &str) -> Option<usize> {
    env::var(name).ok()?.parse().ok()
}

fn parse_u64_env(name: &str) -> Option<u64> {
    env::var(name).ok()?.parse().ok()
}

fn parse_f64_env(name: &str) -> Option<f64> {
    env::var(name).ok()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_for_attacker() {
        let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork);
        assert!(oracle.config.attacker_model.is_some());
    }

    #[test]
    fn test_builder_methods() {
        let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .time_budget_secs(30)
            .max_samples(50_000)
            .batch_size(500)
            .pass_threshold(0.01)
            .fail_threshold(0.99);

        assert_eq!(oracle.config.time_budget, Duration::from_secs(30));
        assert_eq!(oracle.config.max_samples, 50_000);
        assert_eq!(oracle.config.batch_size, 500);
        assert_eq!(oracle.config.pass_threshold, 0.01);
        assert_eq!(oracle.config.fail_threshold, 0.99);
    }

    #[test]
    fn test_compute_min_uniqueness_ratio() {
        // Continuous data should have high uniqueness
        let continuous: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
        let ratio = compute_min_uniqueness_ratio(&continuous, &continuous);
        assert!(ratio > 0.9);

        // Discrete data with few unique values
        let discrete: Vec<f64> = (0..1000).map(|i| (i % 5) as f64).collect();
        let ratio = compute_min_uniqueness_ratio(&discrete, &discrete);
        assert!(ratio < 0.1);
    }
}
