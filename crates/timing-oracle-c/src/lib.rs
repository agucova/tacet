//! C/C++ FFI wrapper for timing-oracle.
//!
//! This crate provides a C-compatible API for the timing-oracle library,
//! enabling timing side-channel detection from C, C++, and other languages.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     User Code (C/C++)                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    timing_oracle.h                          │
//! │              (C header with FFI declarations)               │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    libtiming_oracle                         │
//! │  ┌──────────────────────┬──────────────────────────────┐   │
//! │  │   timing-oracle-c    │     to_measure.c             │   │
//! │  │   (Rust FFI layer)   │  (C measurement loop)        │   │
//! │  │         │            │         │                    │   │
//! │  └─────────┼────────────┴─────────┼────────────────────┘   │
//! │            ▼                      ▼                        │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │              timing-oracle-core                      │   │
//! │  │    (All statistical logic, no_std compatible)       │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Safety
//!
//! All public functions use `catch_unwind` to prevent Rust panics from
//! propagating across the FFI boundary.

use std::ffi::{c_char, c_void, CString};
use std::panic::catch_unwind;
use std::ptr;

pub mod types;

use types::*;

// Link to the C measurement loop
extern "C" {
    fn to_collect_batch(
        generator: GeneratorFn,
        operation: OperationFn,
        ctx: *mut c_void,
        input_buffer: *mut u8,
        input_size: usize,
        schedule: *const bool,
        count: usize,
        out_timings: *mut u64,
        batch_k: usize,
    ) -> usize;

    fn to_get_timer_name() -> *const c_char;
    fn to_get_timer_frequency() -> u64;
    #[allow(dead_code)] // Part of C API, may be used for calibration
    fn to_read_timer() -> u64;
}

/// Callback type for generating input data.
pub type GeneratorFn = unsafe extern "C" fn(
    context: *mut c_void,
    is_baseline: bool,
    output: *mut u8,
    output_size: usize,
);

/// Callback type for the operation to time.
pub type OperationFn = unsafe extern "C" fn(
    context: *mut c_void,
    input: *const u8,
    input_size: usize,
);

// ============================================================================
// Public C API
// ============================================================================

/// Create a default configuration for the given attacker model.
#[no_mangle]
pub extern "C" fn to_config_default(model: ToAttackerModel) -> ToConfig {
    ToConfig {
        attacker_model: model,
        ..ToConfig::default()
    }
}

/// Run a timing test.
///
/// # Safety
///
/// - `config` must be a valid pointer or null (uses defaults if null)
/// - `generator` and `operation` must be valid function pointers
/// - `ctx` can be null if not needed by callbacks
/// - `input_size` must match the buffer size expected by callbacks
#[no_mangle]
pub unsafe extern "C" fn to_test(
    config: *const ToConfig,
    input_size: usize,
    generator: GeneratorFn,
    operation: OperationFn,
    ctx: *mut c_void,
) -> ToResult {
    let result = catch_unwind(|| {
        let config = if config.is_null() {
            ToConfig::default()
        } else {
            (*config).clone()
        };

        run_test_impl(&config, input_size, generator, operation, ctx)
    });

    match result {
        Ok(outcome) => outcome,
        Err(_) => {
            // Panic occurred - return an error result
            let msg = CString::new("Internal error: panic in timing-oracle").unwrap();
            ToResult {
                outcome: ToOutcome::Inconclusive,
                recommendation: msg.into_raw(),
                ..ToResult::default()
            }
        }
    }
}

/// Free a result's owned resources.
///
/// # Safety
///
/// - `result` must be a valid pointer to a ToResult
/// - Only call this once per result
#[no_mangle]
pub unsafe extern "C" fn to_result_free(result: *mut ToResult) {
    if result.is_null() {
        return;
    }

    let r = &mut *result;
    if !r.recommendation.is_null() {
        // Reconstruct and drop the CString to free memory
        drop(CString::from_raw(r.recommendation as *mut c_char));
        r.recommendation = ptr::null();
    }
}

/// Get string representation of outcome.
#[no_mangle]
pub extern "C" fn to_outcome_str(outcome: ToOutcome) -> *const c_char {
    match outcome {
        ToOutcome::Pass => c"Pass".as_ptr(),
        ToOutcome::Fail => c"Fail".as_ptr(),
        ToOutcome::Inconclusive => c"Inconclusive".as_ptr(),
        ToOutcome::Unmeasurable => c"Unmeasurable".as_ptr(),
    }
}

/// Get string representation of effect pattern.
#[no_mangle]
pub extern "C" fn to_effect_pattern_str(pattern: ToEffectPattern) -> *const c_char {
    match pattern {
        ToEffectPattern::UniformShift => c"UniformShift".as_ptr(),
        ToEffectPattern::TailEffect => c"TailEffect".as_ptr(),
        ToEffectPattern::Mixed => c"Mixed".as_ptr(),
        ToEffectPattern::Indeterminate => c"Indeterminate".as_ptr(),
    }
}

/// Get string representation of exploitability.
#[no_mangle]
pub extern "C" fn to_exploitability_str(exploitability: ToExploitability) -> *const c_char {
    match exploitability {
        ToExploitability::Negligible => c"Negligible".as_ptr(),
        ToExploitability::PossibleLan => c"PossibleLAN".as_ptr(),
        ToExploitability::LikelyLan => c"LikelyLAN".as_ptr(),
        ToExploitability::PossibleRemote => c"PossibleRemote".as_ptr(),
    }
}

/// Get string representation of quality.
#[no_mangle]
pub extern "C" fn to_quality_str(quality: ToQuality) -> *const c_char {
    match quality {
        ToQuality::Excellent => c"Excellent".as_ptr(),
        ToQuality::Good => c"Good".as_ptr(),
        ToQuality::Poor => c"Poor".as_ptr(),
        ToQuality::TooNoisy => c"TooNoisy".as_ptr(),
    }
}

/// Get string representation of inconclusive reason.
#[no_mangle]
pub extern "C" fn to_inconclusive_reason_str(reason: ToInconclusiveReason) -> *const c_char {
    match reason {
        ToInconclusiveReason::DataTooNoisy => c"DataTooNoisy".as_ptr(),
        ToInconclusiveReason::NotLearning => c"NotLearning".as_ptr(),
        ToInconclusiveReason::WouldTakeTooLong => c"WouldTakeTooLong".as_ptr(),
        ToInconclusiveReason::TimeBudgetExceeded => c"TimeBudgetExceeded".as_ptr(),
        ToInconclusiveReason::SampleBudgetExceeded => c"SampleBudgetExceeded".as_ptr(),
        ToInconclusiveReason::ConditionsChanged => c"ConditionsChanged".as_ptr(),
    }
}

/// Get the library version.
#[no_mangle]
pub extern "C" fn to_version() -> *const c_char {
    // Include version from Cargo.toml
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Get the timer name being used.
#[no_mangle]
pub extern "C" fn to_timer_name() -> *const c_char {
    unsafe { to_get_timer_name() }
}

/// Get the timer frequency in Hz.
#[no_mangle]
pub extern "C" fn to_timer_frequency() -> u64 {
    unsafe { to_get_timer_frequency() }
}

// ============================================================================
// Internal implementation
// ============================================================================

/// Batch size for adaptive loop.
const BATCH_SIZE: usize = 1000;

/// Calibration samples.
const CALIBRATION_SAMPLES: usize = 5000;

/// Bootstrap iterations for covariance estimation.
const BOOTSTRAP_ITERATIONS: usize = 2000;

/// Minimum timer ticks per measurement for reliable timing.
const MIN_TICKS_PER_MEASUREMENT: u64 = 50;

/// Maximum batch K for adaptive batching.
const MAX_BATCH_K: usize = 20;

fn run_test_impl(
    config: &ToConfig,
    input_size: usize,
    generator: GeneratorFn,
    operation: OperationFn,
    ctx: *mut c_void,
) -> ToResult {
    use timing_oracle_core::adaptive::{
        adaptive_step, AdaptiveOutcome, AdaptiveStepConfig, InconclusiveReason, StepResult,
    };

    let theta_ns = config.attacker_model.to_threshold_ns(config.custom_threshold_ns);
    let max_samples = if config.max_samples == 0 {
        100_000
    } else {
        config.max_samples
    };
    let time_budget_secs = if config.time_budget_secs == 0.0 {
        30.0
    } else {
        config.time_budget_secs
    };

    // Get timer info
    let timer_freq = unsafe { to_get_timer_frequency() };
    let ns_per_tick = if timer_freq > 0 {
        1_000_000_000.0 / timer_freq as f64
    } else {
        1.0 // Assume nanoseconds if frequency unknown
    };
    let timer_resolution_ns = ns_per_tick;

    // Allocate input buffer
    let mut input_buffer = vec![0u8; input_size];

    let start_time = std::time::Instant::now();

    // ==========================================================================
    // Phase 0: Pilot phase - detect unmeasurable operations and select batch_k
    // ==========================================================================

    let batch_k = run_pilot_phase(
        generator,
        operation,
        ctx,
        &mut input_buffer,
        input_size,
        timer_resolution_ns,
    );

    // If batch_k is None, operation is too fast
    let batch_k = match batch_k {
        Some(k) => k,
        None => {
            let msg = CString::new(
                "Operation completes faster than timer resolution. \
                 Consider using a higher-precision timer or batching operations.",
            )
            .unwrap();
            return ToResult {
                outcome: ToOutcome::Unmeasurable,
                operation_ns: 0.0, // Too fast to measure
                timer_resolution_ns,
                recommendation: msg.into_raw(),
                timer_name: unsafe { to_get_timer_name() },
                platform: c"".as_ptr(),
                ..ToResult::default()
            };
        }
    };

    // ==========================================================================
    // Phase 1: Calibration - collect initial samples, estimate covariance
    // ==========================================================================

    let (calibration, calibration_state) = match run_calibration_phase(
        generator,
        operation,
        ctx,
        &mut input_buffer,
        input_size,
        batch_k,
        ns_per_tick,
        theta_ns,
        config.seed,
        start_time,
    ) {
        Ok(result) => result,
        Err(result) => return result,
    };

    // ==========================================================================
    // Phase 2: Adaptive loop - collect batches until decision or quality gate
    // ==========================================================================

    let mut state = calibration_state;
    let step_config = AdaptiveStepConfig::with_theta(theta_ns)
        .pass_threshold(config.pass_threshold)
        .fail_threshold(config.fail_threshold)
        .time_budget_secs(time_budget_secs)
        .max_samples(max_samples);

    loop {
        let elapsed_secs = start_time.elapsed().as_secs_f64();

        // Check if we've exceeded time budget before collecting more
        if elapsed_secs > time_budget_secs {
            let posterior = state.current_posterior().cloned();
            let current_probability = posterior.as_ref().map(|p| p.leak_probability).unwrap_or(0.5);
            return build_result(
                AdaptiveOutcome::Inconclusive {
                    reason: InconclusiveReason::TimeBudgetExceeded {
                        current_probability,
                        samples_collected: state.n_total(),
                        elapsed_secs,
                    },
                    posterior,
                    samples_per_class: state.n_total(),
                    elapsed_secs,
                },
                &calibration,
                timer_resolution_ns,
                batch_k,
            );
        }

        // Check sample budget
        if state.n_total() >= max_samples {
            let posterior = state.current_posterior().cloned();
            let current_probability = posterior.as_ref().map(|p| p.leak_probability).unwrap_or(0.5);
            return build_result(
                AdaptiveOutcome::Inconclusive {
                    reason: InconclusiveReason::SampleBudgetExceeded {
                        current_probability,
                        samples_collected: state.n_total(),
                    },
                    posterior,
                    samples_per_class: state.n_total(),
                    elapsed_secs,
                },
                &calibration,
                timer_resolution_ns,
                batch_k,
            );
        }

        // Run adaptive step
        let result = adaptive_step(&calibration, &mut state, ns_per_tick, elapsed_secs, &step_config);

        match result {
            StepResult::Decision(outcome) => {
                return build_result(outcome, &calibration, timer_resolution_ns, batch_k);
            }
            StepResult::Continue { .. } => {
                // Collect another batch
                collect_batch(
                    &mut state,
                    generator,
                    operation,
                    ctx,
                    &mut input_buffer,
                    input_size,
                    batch_k,
                    ns_per_tick,
                    BATCH_SIZE,
                );
            }
        }
    }
}

/// Run pilot phase to detect unmeasurable operations and select batch_k.
///
/// Returns `Some(batch_k)` if measurable, `None` if too fast even with max batching.
fn run_pilot_phase(
    generator: GeneratorFn,
    operation: OperationFn,
    ctx: *mut c_void,
    input_buffer: &mut [u8],
    input_size: usize,
    _timer_resolution_ns: f64,
) -> Option<usize> {
    const PILOT_SAMPLES: usize = 100;

    // Start with batch_k = 1
    let mut timings = vec![0u64; PILOT_SAMPLES * 2];
    let mut schedule = vec![false; PILOT_SAMPLES * 2];
    for i in 0..PILOT_SAMPLES {
        schedule[i * 2] = true;
        schedule[i * 2 + 1] = false;
    }

    for batch_k in 1..=MAX_BATCH_K {
        unsafe {
            to_collect_batch(
                generator,
                operation,
                ctx,
                input_buffer.as_mut_ptr(),
                input_size,
                schedule.as_ptr(),
                PILOT_SAMPLES * 2,
                timings.as_mut_ptr(),
                batch_k,
            );
        }

        // Compute median timing
        let mut sorted = timings.clone();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2];

        // If median >= MIN_TICKS, this batch_k is sufficient
        if median >= MIN_TICKS_PER_MEASUREMENT {
            return Some(batch_k);
        }
    }

    // Even max batching isn't enough
    None
}

/// Run calibration phase to estimate covariance and set priors.
fn run_calibration_phase(
    generator: GeneratorFn,
    operation: OperationFn,
    ctx: *mut c_void,
    input_buffer: &mut [u8],
    input_size: usize,
    batch_k: usize,
    ns_per_tick: f64,
    theta_ns: f64,
    seed: u64,
    start_time: std::time::Instant,
) -> Result<
    (
        timing_oracle_core::adaptive::Calibration,
        timing_oracle_core::adaptive::AdaptiveState,
    ),
    ToResult,
> {
    use timing_oracle_core::adaptive::{
        AdaptiveState, Calibration, CalibrationSnapshot, PRIOR_SCALE_FACTOR,
    };
    use timing_oracle_core::statistics::{bootstrap_difference_covariance, OnlineStats};
    use timing_oracle_core::types::{Class, Matrix2, TimingSample};

    // Create interleaved schedule
    let mut schedule = vec![false; CALIBRATION_SAMPLES * 2];
    for i in 0..CALIBRATION_SAMPLES {
        schedule[i * 2] = true;
        schedule[i * 2 + 1] = false;
    }

    // Collect calibration samples
    let mut raw_timings = vec![0u64; CALIBRATION_SAMPLES * 2];

    unsafe {
        to_collect_batch(
            generator,
            operation,
            ctx,
            input_buffer.as_mut_ptr(),
            input_size,
            schedule.as_ptr(),
            CALIBRATION_SAMPLES * 2,
            raw_timings.as_mut_ptr(),
            batch_k,
        );
    }

    // Separate baseline and sample timings
    let mut baseline_timings_raw = Vec::with_capacity(CALIBRATION_SAMPLES);
    let mut sample_timings_raw = Vec::with_capacity(CALIBRATION_SAMPLES);
    let mut baseline_timings_ns = Vec::with_capacity(CALIBRATION_SAMPLES);
    let mut sample_timings_ns = Vec::with_capacity(CALIBRATION_SAMPLES);

    let mut baseline_stats = OnlineStats::new();
    let mut sample_stats = OnlineStats::new();

    for (i, &timing) in raw_timings.iter().enumerate() {
        let timing_ns = timing as f64 * ns_per_tick;
        if schedule[i] {
            baseline_timings_raw.push(timing);
            baseline_timings_ns.push(timing_ns);
            baseline_stats.update(timing_ns);
        } else {
            sample_timings_raw.push(timing);
            sample_timings_ns.push(timing_ns);
            sample_stats.update(timing_ns);
        }
    }

    // Create calibration snapshot for drift detection
    let calibration_snapshot = CalibrationSnapshot::new(
        baseline_stats.finalize(),
        sample_stats.finalize(),
    );

    // Bootstrap for covariance estimation
    let interleaved: Vec<TimingSample> = baseline_timings_ns
        .iter()
        .zip(sample_timings_ns.iter())
        .flat_map(|(&b, &s)| {
            [
                TimingSample {
                    time_ns: b,
                    class: Class::Baseline,
                },
                TimingSample {
                    time_ns: s,
                    class: Class::Sample,
                },
            ]
        })
        .collect();

    let cov_estimate = bootstrap_difference_covariance(&interleaved, BOOTSTRAP_ITERATIONS, seed);

    if !cov_estimate.is_stable() {
        let msg = CString::new("Covariance matrix is not stable; try more samples").unwrap();
        return Err(ToResult {
            outcome: ToOutcome::Inconclusive,
            inconclusive_reason: ToInconclusiveReason::DataTooNoisy,
            recommendation: msg.into_raw(),
            timer_resolution_ns: ns_per_tick,
            timer_name: unsafe { to_get_timer_name() },
            elapsed_secs: start_time.elapsed().as_secs_f64(),
            samples_used: CALIBRATION_SAMPLES,
            ..ToResult::default()
        });
    }

    // Compute sigma rate: Sigma_rate = Sigma_cal * n_cal
    let sigma_rate = cov_estimate.matrix * (CALIBRATION_SAMPLES as f64);

    // Compute prior covariance
    let prior_sigma = theta_ns * PRIOR_SCALE_FACTOR;
    let prior_cov = Matrix2::new(prior_sigma.powi(2), 0.0, 0.0, prior_sigma.powi(2));

    // Estimate MDE from calibration data (simplified)
    let mde_shift_ns = 2.0 * cov_estimate.matrix[(0, 0)].sqrt() / (CALIBRATION_SAMPLES as f64).sqrt();
    let mde_tail_ns = 2.0 * cov_estimate.matrix[(8, 8)].sqrt() / (CALIBRATION_SAMPLES as f64).sqrt();

    let elapsed_secs = start_time.elapsed().as_secs_f64();
    let samples_per_second = CALIBRATION_SAMPLES as f64 / elapsed_secs;

    let calibration = Calibration::new(
        sigma_rate,
        10, // block_length (default)
        prior_cov,
        theta_ns,
        CALIBRATION_SAMPLES,
        false, // discrete_mode (could detect from data)
        mde_shift_ns,
        mde_tail_ns,
        calibration_snapshot,
        ns_per_tick,
        samples_per_second,
    );

    // Initialize adaptive state with calibration samples
    let mut state = AdaptiveState::with_capacity(100_000);
    state.add_batch_with_conversion(baseline_timings_raw, sample_timings_raw, ns_per_tick);

    Ok((calibration, state))
}

/// Collect a batch of samples and add to state.
fn collect_batch(
    state: &mut timing_oracle_core::adaptive::AdaptiveState,
    generator: GeneratorFn,
    operation: OperationFn,
    ctx: *mut c_void,
    input_buffer: &mut [u8],
    input_size: usize,
    batch_k: usize,
    ns_per_tick: f64,
    batch_size: usize,
) {
    // Create interleaved schedule
    let mut schedule = vec![false; batch_size * 2];
    for i in 0..batch_size {
        schedule[i * 2] = true;
        schedule[i * 2 + 1] = false;
    }

    let mut raw_timings = vec![0u64; batch_size * 2];

    unsafe {
        to_collect_batch(
            generator,
            operation,
            ctx,
            input_buffer.as_mut_ptr(),
            input_size,
            schedule.as_ptr(),
            batch_size * 2,
            raw_timings.as_mut_ptr(),
            batch_k,
        );
    }

    // Separate baseline and sample timings
    let mut baseline = Vec::with_capacity(batch_size);
    let mut sample = Vec::with_capacity(batch_size);

    for (i, &timing) in raw_timings.iter().enumerate() {
        if schedule[i] {
            baseline.push(timing);
        } else {
            sample.push(timing);
        }
    }

    state.add_batch_with_conversion(baseline, sample, ns_per_tick);
}

/// Convert AdaptiveOutcome to ToResult.
fn build_result(
    outcome: timing_oracle_core::adaptive::AdaptiveOutcome,
    _calibration: &timing_oracle_core::adaptive::Calibration,
    timer_resolution_ns: f64,
    batch_k: usize,
) -> ToResult {
    use timing_oracle_core::adaptive::AdaptiveOutcome;

    let timer_name = unsafe { to_get_timer_name() };

    match outcome {
        AdaptiveOutcome::LeakDetected {
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            // Determine exploitability
            let max_effect = posterior.beta_mean[0]
                .abs()
                .max(posterior.beta_mean[1].abs());
            let exploitability = if max_effect < 100.0 {
                ToExploitability::Negligible
            } else if max_effect < 500.0 {
                ToExploitability::PossibleLan
            } else if max_effect < 20_000.0 {
                ToExploitability::LikelyLan
            } else {
                ToExploitability::PossibleRemote
            };

            ToResult {
                outcome: ToOutcome::Fail,
                leak_probability: posterior.leak_probability,
                effect: build_effect(&posterior),
                quality: build_quality(&posterior),
                samples_used: samples_per_class,
                elapsed_secs,
                exploitability,
                inconclusive_reason: ToInconclusiveReason::DataTooNoisy, // Not used
                operation_ns: 0.0,
                timer_resolution_ns,
                recommendation: ptr::null(),
                timer_name,
                platform: c"".as_ptr(),
                adaptive_batching_used: batch_k > 1,
            }
        }

        AdaptiveOutcome::NoLeakDetected {
            posterior,
            samples_per_class,
            elapsed_secs,
        } => ToResult {
            outcome: ToOutcome::Pass,
            leak_probability: posterior.leak_probability,
            effect: build_effect(&posterior),
            quality: build_quality(&posterior),
            samples_used: samples_per_class,
            elapsed_secs,
            exploitability: ToExploitability::Negligible,
            inconclusive_reason: ToInconclusiveReason::DataTooNoisy, // Not used
            operation_ns: 0.0,
            timer_resolution_ns,
            recommendation: ptr::null(),
            timer_name,
            platform: c"".as_ptr(),
            adaptive_batching_used: batch_k > 1,
        },

        AdaptiveOutcome::Inconclusive {
            reason,
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            let (c_reason, recommendation) = convert_reason(&reason);

            let (leak_probability, effect, quality) = if let Some(ref p) = posterior {
                (p.leak_probability, build_effect(p), build_quality(p))
            } else {
                (
                    0.5,
                    ToEffect {
                        shift_ns: 0.0,
                        tail_ns: 0.0,
                        ci_low_ns: 0.0,
                        ci_high_ns: 0.0,
                        pattern: ToEffectPattern::Indeterminate,
                    },
                    ToQuality::TooNoisy,
                )
            };

            let recommendation_ptr = recommendation
                .map(|s| CString::new(s).unwrap().into_raw() as *const c_char)
                .unwrap_or(ptr::null());

            ToResult {
                outcome: ToOutcome::Inconclusive,
                leak_probability,
                effect,
                quality,
                samples_used: samples_per_class,
                elapsed_secs,
                exploitability: ToExploitability::Negligible,
                inconclusive_reason: c_reason,
                operation_ns: 0.0,
                timer_resolution_ns,
                recommendation: recommendation_ptr,
                timer_name,
                platform: c"".as_ptr(),
                adaptive_batching_used: batch_k > 1,
            }
        }
    }
}

fn build_effect(posterior: &timing_oracle_core::adaptive::Posterior) -> ToEffect {
    let shift = posterior.beta_mean[0];
    let tail = posterior.beta_mean[1];
    let shift_se = posterior.beta_cov[(0, 0)].sqrt();

    let pattern = if shift.abs() > 2.0 * tail.abs() {
        ToEffectPattern::UniformShift
    } else if tail.abs() > 2.0 * shift.abs() {
        ToEffectPattern::TailEffect
    } else if shift.abs() > 1.0 || tail.abs() > 1.0 {
        ToEffectPattern::Mixed
    } else {
        ToEffectPattern::Indeterminate
    };

    ToEffect {
        shift_ns: shift,
        tail_ns: tail,
        ci_low_ns: shift - 1.96 * shift_se,
        ci_high_ns: shift + 1.96 * shift_se,
        pattern,
    }
}

fn build_quality(posterior: &timing_oracle_core::adaptive::Posterior) -> ToQuality {
    let effect_se = posterior.beta_cov[(0, 0)].sqrt();
    if effect_se < 5.0 {
        ToQuality::Excellent
    } else if effect_se < 20.0 {
        ToQuality::Good
    } else if effect_se < 100.0 {
        ToQuality::Poor
    } else {
        ToQuality::TooNoisy
    }
}

fn convert_reason(
    reason: &timing_oracle_core::adaptive::InconclusiveReason,
) -> (ToInconclusiveReason, Option<String>) {
    use timing_oracle_core::adaptive::InconclusiveReason;

    match reason {
        InconclusiveReason::DataTooNoisy { guidance, .. } => {
            (ToInconclusiveReason::DataTooNoisy, Some(guidance.clone()))
        }
        InconclusiveReason::NotLearning { guidance, .. } => {
            (ToInconclusiveReason::NotLearning, Some(guidance.clone()))
        }
        InconclusiveReason::WouldTakeTooLong { guidance, .. } => {
            (ToInconclusiveReason::WouldTakeTooLong, Some(guidance.clone()))
        }
        InconclusiveReason::TimeBudgetExceeded { .. } => {
            (ToInconclusiveReason::TimeBudgetExceeded, None)
        }
        InconclusiveReason::SampleBudgetExceeded { .. } => {
            (ToInconclusiveReason::SampleBudgetExceeded, None)
        }
        InconclusiveReason::ConditionsChanged { message, guidance, .. } => {
            (ToInconclusiveReason::ConditionsChanged, Some(format!("{} {}", message, guidance)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = to_config_default(ToAttackerModel::AdjacentNetwork);
        assert_eq!(config.attacker_model, ToAttackerModel::AdjacentNetwork);
        assert!((config.pass_threshold - 0.05).abs() < 1e-10);
        assert!((config.fail_threshold - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_outcome_str() {
        let s = to_outcome_str(ToOutcome::Pass);
        assert!(!s.is_null());
        let s = to_outcome_str(ToOutcome::Fail);
        assert!(!s.is_null());
    }

    #[test]
    fn test_attacker_model_thresholds() {
        assert!((ToAttackerModel::SharedHardware.to_threshold_ns(0.0) - 0.6).abs() < 1e-10);
        assert!((ToAttackerModel::AdjacentNetwork.to_threshold_ns(0.0) - 100.0).abs() < 1e-10);
        assert!((ToAttackerModel::RemoteNetwork.to_threshold_ns(0.0) - 50_000.0).abs() < 1e-10);
        assert!((ToAttackerModel::Custom.to_threshold_ns(42.0) - 42.0).abs() < 1e-10);
    }
}
