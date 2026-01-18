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
//! When built with the `std` feature (default), all public functions use
//! `catch_unwind` to prevent Rust panics from propagating across the FFI
//! boundary. In `no_std` mode, panics will abort.
//!
//! # Features
//!
//! - `std` (default): Enables `std::time::Instant` for time tracking and
//!   `catch_unwind` for panic safety. Use `to_test()` for automatic time tracking.
//! - Without `std`: Use `to_test_with_time()` and provide elapsed time externally.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use core::ffi::{c_char, c_void};
use core::ptr;

#[cfg(feature = "std")]
use std::ffi::CString;

#[cfg(feature = "std")]
use std::panic::catch_unwind;

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

// Math helper for no_std - use libm directly
fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

pub mod types;

// Re-export all types at crate root for convenience
pub use types::*;

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

    /// Initialize the timer subsystem with specified preference.
    /// Returns 0 on success, -1 if PREFER_PMU but PMU unavailable.
    fn to_timer_init(pref: i32) -> i32;

    /// Clean up the timer subsystem, releasing any resources.
    fn to_timer_cleanup();

    /// Get the currently active timer type.
    #[allow(dead_code)] // Part of C API
    fn to_get_timer_type() -> i32;

    /// Get cycles per nanosecond ratio for the active timer.
    #[allow(dead_code)] // Part of C API
    fn to_get_cycles_per_ns() -> f64;
}

/// Callback type for generating input data.
pub type GeneratorFn = unsafe extern "C" fn(
    context: *mut c_void,
    is_baseline: bool,
    output: *mut u8,
    output_size: usize,
);

/// Callback type for the operation to time.
pub type OperationFn =
    unsafe extern "C" fn(context: *mut c_void, input: *const u8, input_size: usize);

/// Convert ToTimerPref to C enum value.
fn timer_pref_to_c(pref: ToTimerPref) -> i32 {
    match pref {
        ToTimerPref::Auto => 0,
        ToTimerPref::Standard => 1,
        ToTimerPref::PreferPmu => 2,
    }
}

/// Initialize the timer subsystem based on config preference.
/// Returns Ok(()) on success, Err(Box<ToResult>) if PMU was required but unavailable.
unsafe fn init_timer(config: &ToConfig) -> Result<(), Box<ToResult>> {
    let pref = timer_pref_to_c(config.timer_preference);
    let result = to_timer_init(pref);

    if result != 0 && config.timer_preference == ToTimerPref::PreferPmu {
        return Err(Box::new(ToResult {
            outcome: ToOutcome::Inconclusive,
            inconclusive_reason: ToInconclusiveReason::ThresholdUnachievable,
            recommendation: make_recommendation(
                "PMU timer requested but unavailable. On Linux, try: sudo or \
                 echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid. \
                 On macOS, try: sudo",
            ),
            ..ToResult::default()
        }));
    }

    Ok(())
}

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

/// Run a timing test with automatic time tracking (requires `std` feature).
///
/// This is the standard API for most use cases. For `no_std` environments,
/// use `to_test_with_time()` instead.
///
/// # Safety
///
/// - `config` must be a valid pointer or null (uses defaults if null)
/// - `generator` and `operation` must be valid function pointers
/// - `ctx` can be null if not needed by callbacks
/// - `input_size` must match the buffer size expected by callbacks
#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn to_test(
    config: *const ToConfig,
    input_size: usize,
    generator: GeneratorFn,
    operation: OperationFn,
    ctx: *mut c_void,
) -> ToResult {
    let config = if config.is_null() {
        ToConfig::default()
    } else {
        (*config).clone()
    };

    // Initialize timer based on config preference
    if let Err(result) = init_timer(&config) {
        return *result;
    }

    let result = catch_unwind(|| {
        let start_time = std::time::Instant::now();
        run_test_impl(&config, input_size, generator, operation, ctx, || {
            start_time.elapsed().as_secs_f64()
        })
    });

    // Always cleanup timer, even if panic occurred
    to_timer_cleanup();

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

/// Run a timing test with caller-provided time tracking (no_std compatible).
///
/// This API is available in both `std` and `no_std` builds. The caller is
/// responsible for tracking elapsed time and providing it via the `elapsed_secs`
/// pointer. The library will read this value whenever it needs the current time.
///
/// # Usage
///
/// ```c
/// double elapsed = 0.0;
/// // ... start your timer ...
///
/// // Before calling, update elapsed_secs to current elapsed time
/// elapsed = get_elapsed_time(); // Your time function
/// ToResult result = to_test_with_time(&config, size, gen, op, ctx, &elapsed);
/// ```
///
/// Note: For the adaptive loop to work correctly, you should update `elapsed_secs`
/// before each call. In practice, since this is a single blocking call, you only
/// need to provide the initial elapsed time (usually 0.0).
///
/// # Safety
///
/// - `config` must be a valid pointer or null (uses defaults if null)
/// - `generator` and `operation` must be valid function pointers
/// - `ctx` can be null if not needed by callbacks
/// - `input_size` must match the buffer size expected by callbacks
/// - `elapsed_secs` must be a valid pointer to a f64
#[no_mangle]
pub unsafe extern "C" fn to_test_with_time(
    config: *const ToConfig,
    input_size: usize,
    generator: GeneratorFn,
    operation: OperationFn,
    ctx: *mut c_void,
    elapsed_secs: *const f64,
) -> ToResult {
    let config = if config.is_null() {
        ToConfig::default()
    } else {
        (*config).clone()
    };

    // Initialize timer based on config preference
    if let Err(result) = init_timer(&config) {
        return *result;
    }

    #[cfg(feature = "std")]
    {
        let result = catch_unwind(|| {
            run_test_impl(&config, input_size, generator, operation, ctx, || {
                if elapsed_secs.is_null() {
                    0.0
                } else {
                    *elapsed_secs
                }
            })
        });

        // Always cleanup timer, even if panic occurred
        to_timer_cleanup();

        match result {
            Ok(outcome) => outcome,
            Err(_) => {
                let msg = CString::new("Internal error: panic in timing-oracle").unwrap();
                ToResult {
                    outcome: ToOutcome::Inconclusive,
                    recommendation: msg.into_raw(),
                    ..ToResult::default()
                }
            }
        }
    }

    #[cfg(not(feature = "std"))]
    {
        // No panic catching in no_std - panics will abort
        let result = run_test_impl(&config, input_size, generator, operation, ctx, || {
            if elapsed_secs.is_null() {
                0.0
            } else {
                *elapsed_secs
            }
        });

        to_timer_cleanup();
        result
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
        #[cfg(feature = "std")]
        {
            // Reconstruct and drop the CString to free memory
            drop(CString::from_raw(r.recommendation as *mut c_char));
        }
        #[cfg(not(feature = "std"))]
        {
            // In no_std mode, recommendation points to static strings, so nothing to free
            // Just null it out for safety
        }
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
        ToOutcome::Research => c"Research".as_ptr(),
    }
}

/// Get string representation of research status.
#[no_mangle]
pub extern "C" fn to_research_status_str(status: ToResearchStatus) -> *const c_char {
    match status {
        ToResearchStatus::EffectDetected => c"EffectDetected".as_ptr(),
        ToResearchStatus::NoEffectDetected => c"NoEffectDetected".as_ptr(),
        ToResearchStatus::ResolutionLimitReached => c"ResolutionLimitReached".as_ptr(),
        ToResearchStatus::QualityIssue => c"QualityIssue".as_ptr(),
        ToResearchStatus::BudgetExhausted => c"BudgetExhausted".as_ptr(),
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
        ToExploitability::SharedHardwareOnly => c"SharedHardwareOnly".as_ptr(),
        ToExploitability::Http2Multiplexing => c"Http2Multiplexing".as_ptr(),
        ToExploitability::StandardRemote => c"StandardRemote".as_ptr(),
        ToExploitability::ObviousLeak => c"ObviousLeak".as_ptr(),
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
        ToInconclusiveReason::ThresholdUnachievable => c"ThresholdUnachievable".as_ptr(),
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

/// Create a recommendation string.
///
/// In std mode, returns a CString that must be freed by the caller.
/// In no_std mode, returns a static string.
#[cfg(feature = "std")]
fn make_recommendation(msg: &str) -> *const c_char {
    CString::new(msg).unwrap().into_raw()
}

#[cfg(not(feature = "std"))]
fn make_recommendation(_msg: &str) -> *const c_char {
    // In no_std mode, return a generic static string
    // The specific message is lost, but the caller gets a valid pointer
    c"See documentation for recommendations".as_ptr()
}

/// Create a recommendation from an optional String.
///
/// In std mode, converts the string to CString.
/// In no_std mode, returns a static string or null.
#[cfg(feature = "std")]
fn make_recommendation_from_option(msg: Option<String>) -> *const c_char {
    msg.map(|s| CString::new(s).unwrap().into_raw() as *const c_char)
        .unwrap_or(ptr::null())
}

#[cfg(not(feature = "std"))]
fn make_recommendation_from_option(msg: Option<String>) -> *const c_char {
    if msg.is_some() {
        c"See documentation for recommendations".as_ptr()
    } else {
        ptr::null()
    }
}

/// Default batch size for adaptive loop.
const DEFAULT_BATCH_SIZE: usize = 1000;

/// Default calibration samples.
const DEFAULT_CALIBRATION_SAMPLES: usize = 5000;

/// Default bootstrap iterations for covariance estimation.
const DEFAULT_BOOTSTRAP_ITERATIONS: usize = 2000;

/// Minimum timer ticks per measurement for reliable timing.
const MIN_TICKS_PER_MEASUREMENT: u64 = 50;

/// Maximum batch K for adaptive batching.
const MAX_BATCH_K: usize = 20;

fn run_test_impl<F: Fn() -> f64>(
    config: &ToConfig,
    input_size: usize,
    generator: GeneratorFn,
    operation: OperationFn,
    ctx: *mut c_void,
    get_elapsed_secs: F,
) -> ToResult {
    use timing_oracle_core::adaptive::{
        adaptive_step, AdaptiveOutcome, AdaptiveStepConfig, InconclusiveReason, StepResult,
    };

    let theta_ns = config
        .attacker_model
        .to_threshold_ns(config.custom_threshold_ns);
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
    let calibration_samples = if config.calibration_samples == 0 {
        DEFAULT_CALIBRATION_SAMPLES
    } else {
        config.calibration_samples
    };
    let batch_size = if config.batch_size == 0 {
        DEFAULT_BATCH_SIZE
    } else {
        config.batch_size
    };
    let bootstrap_iterations = if config.bootstrap_iterations == 0 {
        DEFAULT_BOOTSTRAP_ITERATIONS
    } else {
        config.bootstrap_iterations
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
            return ToResult {
                outcome: ToOutcome::Unmeasurable,
                operation_ns: 0.0, // Too fast to measure
                timer_resolution_ns,
                recommendation: make_recommendation(
                    "Operation completes faster than timer resolution. \
                     Consider using a higher-precision timer or batching operations.",
                ),
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
        calibration_samples,
        bootstrap_iterations,
        &get_elapsed_secs,
    ) {
        Ok(result) => result,
        Err(result) => return *result,
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

    // Drift detection state
    let drift_thresholds = timing_oracle_core::adaptive::DriftThresholds::default();
    let mut last_stationarity_ratio = 1.0;
    let mut last_stationarity_ok = true;

    loop {
        let elapsed_secs = get_elapsed_secs();

        // ==========================================================================
        // Quality Gate 7: Condition drift check (spec §3.5.2)
        // ==========================================================================
        if let Some(current_snapshot) = state.get_stats_snapshot() {
            let drift = timing_oracle_core::adaptive::ConditionDrift::compute(
                &calibration.calibration_snapshot,
                &current_snapshot,
            );

            // Track the maximum variance ratio for diagnostics
            last_stationarity_ratio = drift
                .variance_ratio_baseline
                .max(drift.variance_ratio_sample);
            last_stationarity_ok = !drift.is_significant(&drift_thresholds);

            if !last_stationarity_ok {
                let posterior = state.current_posterior().cloned();
                let drift_message = drift.description(&drift_thresholds);

                return build_result(
                    AdaptiveOutcome::Inconclusive {
                        reason: InconclusiveReason::ConditionsChanged {
                            message: String::from("Measurement conditions changed during test"),
                            guidance: String::from(
                                "Ensure stable environment: disable CPU frequency scaling, \
                                 minimize concurrent processes, use performance CPU governor",
                            ),
                            drift_description: drift_message,
                        },
                        posterior,
                        samples_per_class: state.n_total(),
                        elapsed_secs,
                    },
                    &calibration,
                    timer_resolution_ns,
                    batch_k,
                    last_stationarity_ratio,
                    last_stationarity_ok,
                );
            }
        }

        // Check if we've exceeded time budget before collecting more
        if elapsed_secs > time_budget_secs {
            let posterior = state.current_posterior().cloned();
            let current_probability = posterior
                .as_ref()
                .map(|p| p.leak_probability)
                .unwrap_or(0.5);
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
                last_stationarity_ratio,
                last_stationarity_ok,
            );
        }

        // Check sample budget
        if state.n_total() >= max_samples {
            let posterior = state.current_posterior().cloned();
            let current_probability = posterior
                .as_ref()
                .map(|p| p.leak_probability)
                .unwrap_or(0.5);
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
                last_stationarity_ratio,
                last_stationarity_ok,
            );
        }

        // Run adaptive step
        let result = adaptive_step(
            &calibration,
            &mut state,
            ns_per_tick,
            elapsed_secs,
            &step_config,
        );

        match result {
            StepResult::Decision(outcome) => {
                return build_result(
                    outcome,
                    &calibration,
                    timer_resolution_ns,
                    batch_k,
                    last_stationarity_ratio,
                    last_stationarity_ok,
                );
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
                    batch_size,
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
    let mut schedule = [false; PILOT_SAMPLES * 2];
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
#[allow(clippy::too_many_arguments)]
fn run_calibration_phase<F: Fn() -> f64>(
    generator: GeneratorFn,
    operation: OperationFn,
    ctx: *mut c_void,
    input_buffer: &mut [u8],
    input_size: usize,
    batch_k: usize,
    ns_per_tick: f64,
    theta_ns: f64,
    seed: u64,
    calibration_samples: usize,
    bootstrap_iterations: usize,
    get_elapsed_secs: &F,
) -> Result<
    (
        timing_oracle_core::adaptive::Calibration,
        timing_oracle_core::adaptive::AdaptiveState,
    ),
    Box<ToResult>,
> {
    use timing_oracle_core::adaptive::{
        calibrate_prior_scale, compute_prior_cov_9d, AdaptiveState, Calibration,
        CalibrationSnapshot,
    };
    use timing_oracle_core::statistics::{
        bootstrap_difference_covariance, bootstrap_difference_covariance_discrete,
        compute_min_uniqueness_ratio, OnlineStats, DISCRETE_MODE_THRESHOLD,
    };
    use timing_oracle_core::types::{Class, TimingSample};

    // Create interleaved schedule
    let mut schedule = vec![false; calibration_samples * 2];
    for i in 0..calibration_samples {
        schedule[i * 2] = true;
        schedule[i * 2 + 1] = false;
    }

    // Collect calibration samples
    let mut raw_timings = vec![0u64; calibration_samples * 2];

    unsafe {
        to_collect_batch(
            generator,
            operation,
            ctx,
            input_buffer.as_mut_ptr(),
            input_size,
            schedule.as_ptr(),
            calibration_samples * 2,
            raw_timings.as_mut_ptr(),
            batch_k,
        );
    }

    // Separate baseline and sample timings
    let mut baseline_timings_raw = Vec::with_capacity(calibration_samples);
    let mut sample_timings_raw = Vec::with_capacity(calibration_samples);
    let mut baseline_timings_ns = Vec::with_capacity(calibration_samples);
    let mut sample_timings_ns = Vec::with_capacity(calibration_samples);

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
    let calibration_snapshot =
        CalibrationSnapshot::new(baseline_stats.finalize(), sample_stats.finalize());

    // Detect discrete mode (spec §3.7): triggered when < 10% of values are unique
    let min_uniqueness = compute_min_uniqueness_ratio(&baseline_timings_ns, &sample_timings_ns);
    let discrete_mode = min_uniqueness < DISCRETE_MODE_THRESHOLD;

    // Select appropriate bootstrap method based on discrete mode
    let cov_estimate = if discrete_mode {
        // Discrete mode: use m-out-of-n bootstrap on per-class sequences
        bootstrap_difference_covariance_discrete(
            &baseline_timings_ns,
            &sample_timings_ns,
            bootstrap_iterations,
            seed,
        )
    } else {
        // Normal mode: use block bootstrap on interleaved stream
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
        bootstrap_difference_covariance(&interleaved, bootstrap_iterations, seed, false)
    };

    if !cov_estimate.is_stable() {
        return Err(Box::new(ToResult {
            outcome: ToOutcome::Inconclusive,
            inconclusive_reason: ToInconclusiveReason::DataTooNoisy,
            recommendation: make_recommendation(
                "Covariance matrix is not stable; try more samples",
            ),
            timer_resolution_ns: ns_per_tick,
            timer_name: unsafe { to_get_timer_name() },
            elapsed_secs: get_elapsed_secs(),
            samples_used: calibration_samples,
            ..ToResult::default()
        }));
    }

    // Compute sigma rate: Sigma_rate = Sigma_cal * n_cal
    let sigma_rate = cov_estimate.matrix * (calibration_samples as f64);

    // Estimate MDE from calibration data (simplified)
    let n_sqrt = sqrt(calibration_samples as f64);
    let mde_shift_ns = 2.0 * sqrt(cov_estimate.matrix[(0, 0)]) / n_sqrt;
    let mde_tail_ns = 2.0 * sqrt(cov_estimate.matrix[(8, 8)]) / n_sqrt;

    let elapsed_secs = get_elapsed_secs();
    let samples_per_second = calibration_samples as f64 / elapsed_secs;

    // v4.1: Compute floor-rate constant and effective threshold
    // For C API, use simplified defaults since we don't have full bootstrap
    let c_floor = 10.0; // Conservative default
    let q_thresh = 18.48; // chi-squared(7, 0.99) fallback
    let theta_tick = ns_per_tick / 20.0; // Assume batch size of 20
    let theta_floor_initial = if c_floor / n_sqrt > theta_tick {
        c_floor / n_sqrt
    } else {
        theta_tick
    };
    let theta_eff = if theta_ns > theta_floor_initial {
        theta_ns
    } else {
        theta_floor_initial
    };
    let rng_seed = 0x74696D696E67u64; // "timing" in ASCII

    // Compute 9D prior covariance using calibrated scale factor
    let sigma_prior = calibrate_prior_scale(&sigma_rate, theta_eff, rng_seed);
    let prior_cov_9d = compute_prior_cov_9d(&sigma_rate, sigma_prior);

    let calibration = Calibration::new(
        sigma_rate,
        10, // block_length (default)
        prior_cov_9d,
        theta_ns,
        calibration_samples,
        discrete_mode, // Detected from uniqueness ratio
        mde_shift_ns,
        mde_tail_ns,
        calibration_snapshot,
        ns_per_tick,
        samples_per_second,
        c_floor,
        q_thresh, // projection_mismatch_thresh
        theta_tick,
        theta_eff,
        theta_floor_initial,
        rng_seed,
    );

    // Initialize adaptive state with calibration samples
    let mut state = AdaptiveState::with_capacity(100_000);
    state.add_batch_with_conversion(baseline_timings_raw, sample_timings_raw, ns_per_tick);

    Ok((calibration, state))
}

/// Collect a batch of samples and add to state.
#[allow(clippy::too_many_arguments)]
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

/// Build diagnostics from calibration data and posterior.
fn build_diagnostics(
    calibration: &timing_oracle_core::adaptive::Calibration,
    posterior: Option<&timing_oracle_core::adaptive::Posterior>,
    elapsed_secs: f64,
    stationarity_ratio: f64,
    stationarity_ok: bool,
) -> ToDiagnostics {
    let projection_mismatch_q = posterior.map(|p| p.projection_mismatch_q).unwrap_or(0.0);
    let projection_mismatch_ok = projection_mismatch_q <= calibration.projection_mismatch_thresh;

    ToDiagnostics {
        dependence_length: calibration.block_length,
        effective_sample_size: posterior.map(|p| p.n).unwrap_or(0),
        stationarity_ratio,
        stationarity_ok,
        projection_mismatch_q,
        projection_mismatch_threshold: calibration.projection_mismatch_thresh,
        projection_mismatch_ok,
        outlier_rate_baseline: 0.0, // Not tracked currently
        outlier_rate_sample: 0.0,   // Not tracked currently
        outlier_asymmetry_ok: true, // Assume OK since not tracked
        discrete_mode: calibration.discrete_mode,
        duplicate_fraction: 0.0, // Not tracked currently
        preflight_ok: true,      // No preflight checks yet
        calibration_samples: calibration.calibration_samples,
        total_time_secs: elapsed_secs,
        seed: calibration.rng_seed,
        theta_user: calibration.theta_ns,
        theta_eff: calibration.theta_eff,
        theta_floor: calibration.theta_floor_initial,
        warning_count: 0,
        warnings: [[0; TO_MAX_WARNING_LEN]; TO_MAX_WARNINGS],
    }
}

/// Build effect with projection mismatch info from posterior.
fn build_effect_with_mismatch(
    posterior: &timing_oracle_core::adaptive::Posterior,
    calibration: &timing_oracle_core::adaptive::Calibration,
) -> ToEffect {
    let mut effect: ToEffect = posterior.to_effect_estimate().into();

    // Check projection mismatch
    let mismatch = posterior.projection_mismatch_q > calibration.projection_mismatch_thresh;
    effect.has_interpretation_caveat = mismatch;

    if mismatch {
        // Compute top contributing quantiles from residuals
        // The residual for each quantile k is: r_k = delta_post_k - (X * beta_proj)_k
        // We rank by |r_k| / sqrt(Sigma_n[k,k])
        let delta_post = &posterior.delta_post;
        let beta_proj = &posterior.beta_proj;

        // X matrix: column 0 is all 1s (shift), column 1 is tail weights
        let tail_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut residuals: [(f64, u8); 9] = [(0.0, 0); 9];

        for k in 0..9 {
            let fitted = beta_proj[0] + tail_weights[k] * beta_proj[1];
            let residual = (delta_post[k] - fitted).abs();
            residuals[k] = (residual, k as u8);
        }

        // Sort by residual magnitude (descending)
        residuals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));

        // Take top 3
        effect.top_quantiles = [residuals[0].1, residuals[1].1, residuals[2].1];
    }

    effect
}

/// Convert AdaptiveOutcome to ToResult.
fn build_result(
    outcome: timing_oracle_core::adaptive::AdaptiveOutcome,
    calibration: &timing_oracle_core::adaptive::Calibration,
    timer_resolution_ns: f64,
    batch_k: usize,
    stationarity_ratio: f64,
    stationarity_ok: bool,
) -> ToResult {
    use timing_oracle_core::adaptive::AdaptiveOutcome;

    let timer_name = unsafe { to_get_timer_name() };

    match outcome {
        AdaptiveOutcome::LeakDetected {
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            // Determine exploitability based on new thresholds from Timeless Timing Attacks
            let max_effect = posterior.beta_proj[0]
                .abs()
                .max(posterior.beta_proj[1].abs());
            let exploitability = if max_effect < 10.0 {
                ToExploitability::SharedHardwareOnly
            } else if max_effect < 100.0 {
                ToExploitability::Http2Multiplexing
            } else if max_effect < 10_000.0 {
                ToExploitability::StandardRemote
            } else {
                ToExploitability::ObviousLeak
            };

            let diagnostics = build_diagnostics(
                calibration,
                Some(&posterior),
                elapsed_secs,
                stationarity_ratio,
                stationarity_ok,
            );

            ToResult {
                outcome: ToOutcome::Fail,
                leak_probability: posterior.leak_probability,
                effect: build_effect_with_mismatch(&posterior, calibration),
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
                diagnostics,
                ..ToResult::default()
            }
        }

        AdaptiveOutcome::NoLeakDetected {
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            let diagnostics = build_diagnostics(
                calibration,
                Some(&posterior),
                elapsed_secs,
                stationarity_ratio,
                stationarity_ok,
            );

            ToResult {
                outcome: ToOutcome::Pass,
                leak_probability: posterior.leak_probability,
                effect: build_effect_with_mismatch(&posterior, calibration),
                quality: build_quality(&posterior),
                samples_used: samples_per_class,
                elapsed_secs,
                exploitability: ToExploitability::SharedHardwareOnly,
                inconclusive_reason: ToInconclusiveReason::DataTooNoisy, // Not used
                operation_ns: 0.0,
                timer_resolution_ns,
                recommendation: ptr::null(),
                timer_name,
                platform: c"".as_ptr(),
                adaptive_batching_used: batch_k > 1,
                diagnostics,
                ..ToResult::default()
            }
        }

        AdaptiveOutcome::Inconclusive {
            reason,
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            let (c_reason, recommendation) = convert_reason(&reason);

            let (leak_probability, effect, quality) = if let Some(ref p) = posterior {
                (
                    p.leak_probability,
                    build_effect_with_mismatch(p, calibration),
                    build_quality(p),
                )
            } else {
                (0.5, ToEffect::default(), ToQuality::TooNoisy)
            };

            let recommendation_ptr = make_recommendation_from_option(recommendation);
            let diagnostics = build_diagnostics(
                calibration,
                posterior.as_ref(),
                elapsed_secs,
                stationarity_ratio,
                stationarity_ok,
            );

            ToResult {
                outcome: ToOutcome::Inconclusive,
                leak_probability,
                effect,
                quality,
                samples_used: samples_per_class,
                elapsed_secs,
                exploitability: ToExploitability::SharedHardwareOnly,
                inconclusive_reason: c_reason,
                operation_ns: 0.0,
                timer_resolution_ns,
                recommendation: recommendation_ptr,
                timer_name,
                platform: c"".as_ptr(),
                adaptive_batching_used: batch_k > 1,
                diagnostics,
                ..ToResult::default()
            }
        }
    }
}

fn build_quality(posterior: &timing_oracle_core::adaptive::Posterior) -> ToQuality {
    // Use core's method and convert to FFI type
    posterior.measurement_quality().into()
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
        InconclusiveReason::WouldTakeTooLong { guidance, .. } => (
            ToInconclusiveReason::WouldTakeTooLong,
            Some(guidance.clone()),
        ),
        InconclusiveReason::TimeBudgetExceeded { .. } => {
            (ToInconclusiveReason::TimeBudgetExceeded, None)
        }
        InconclusiveReason::SampleBudgetExceeded { .. } => {
            (ToInconclusiveReason::SampleBudgetExceeded, None)
        }
        InconclusiveReason::ConditionsChanged {
            message, guidance, ..
        } => (
            ToInconclusiveReason::ConditionsChanged,
            Some(format!("{} {}", message, guidance)),
        ),
        InconclusiveReason::ThresholdUnachievable {
            message, guidance, ..
        } => (
            ToInconclusiveReason::ThresholdUnachievable,
            Some(format!("{} {}", message, guidance)),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ffi::CStr;

    // =========================================================================
    // Group 1: String Conversion Functions
    // =========================================================================

    #[test]
    fn test_outcome_str_all_variants() {
        // Test all ToOutcome variants return valid C strings
        let variants = [
            (ToOutcome::Pass, "Pass"),
            (ToOutcome::Fail, "Fail"),
            (ToOutcome::Inconclusive, "Inconclusive"),
            (ToOutcome::Unmeasurable, "Unmeasurable"),
        ];

        for (variant, expected) in variants {
            let ptr = to_outcome_str(variant);
            assert!(
                !ptr.is_null(),
                "to_outcome_str returned null for {:?}",
                variant
            );
            let cstr = unsafe { CStr::from_ptr(ptr) };
            assert_eq!(cstr.to_str().unwrap(), expected);
        }
    }

    #[test]
    fn test_effect_pattern_str_all() {
        let variants = [
            (ToEffectPattern::UniformShift, "UniformShift"),
            (ToEffectPattern::TailEffect, "TailEffect"),
            (ToEffectPattern::Mixed, "Mixed"),
            (ToEffectPattern::Indeterminate, "Indeterminate"),
        ];

        for (variant, expected) in variants {
            let ptr = to_effect_pattern_str(variant);
            assert!(
                !ptr.is_null(),
                "to_effect_pattern_str returned null for {:?}",
                variant
            );
            let cstr = unsafe { CStr::from_ptr(ptr) };
            assert_eq!(cstr.to_str().unwrap(), expected);
        }
    }

    #[test]
    fn test_exploitability_str_all() {
        let variants = [
            (ToExploitability::SharedHardwareOnly, "SharedHardwareOnly"),
            (ToExploitability::Http2Multiplexing, "Http2Multiplexing"),
            (ToExploitability::StandardRemote, "StandardRemote"),
            (ToExploitability::ObviousLeak, "ObviousLeak"),
        ];

        for (variant, expected) in variants {
            let ptr = to_exploitability_str(variant);
            assert!(
                !ptr.is_null(),
                "to_exploitability_str returned null for {:?}",
                variant
            );
            let cstr = unsafe { CStr::from_ptr(ptr) };
            assert_eq!(cstr.to_str().unwrap(), expected);
        }
    }

    #[test]
    fn test_quality_str_all() {
        let variants = [
            (ToQuality::Excellent, "Excellent"),
            (ToQuality::Good, "Good"),
            (ToQuality::Poor, "Poor"),
            (ToQuality::TooNoisy, "TooNoisy"),
        ];

        for (variant, expected) in variants {
            let ptr = to_quality_str(variant);
            assert!(
                !ptr.is_null(),
                "to_quality_str returned null for {:?}",
                variant
            );
            let cstr = unsafe { CStr::from_ptr(ptr) };
            assert_eq!(cstr.to_str().unwrap(), expected);
        }
    }

    #[test]
    fn test_inconclusive_reason_str_all() {
        let variants = [
            (ToInconclusiveReason::DataTooNoisy, "DataTooNoisy"),
            (ToInconclusiveReason::NotLearning, "NotLearning"),
            (ToInconclusiveReason::WouldTakeTooLong, "WouldTakeTooLong"),
            (
                ToInconclusiveReason::TimeBudgetExceeded,
                "TimeBudgetExceeded",
            ),
            (
                ToInconclusiveReason::SampleBudgetExceeded,
                "SampleBudgetExceeded",
            ),
            (ToInconclusiveReason::ConditionsChanged, "ConditionsChanged"),
            (
                ToInconclusiveReason::ThresholdUnachievable,
                "ThresholdUnachievable",
            ),
        ];

        for (variant, expected) in variants {
            let ptr = to_inconclusive_reason_str(variant);
            assert!(
                !ptr.is_null(),
                "to_inconclusive_reason_str returned null for {:?}",
                variant
            );
            let cstr = unsafe { CStr::from_ptr(ptr) };
            assert_eq!(cstr.to_str().unwrap(), expected);
        }
    }

    // =========================================================================
    // Group 2: Type Conversions (From implementations)
    // =========================================================================

    #[test]
    fn test_effect_pattern_from_core() {
        use timing_oracle_core::result::EffectPattern;

        let conversions = [
            (EffectPattern::UniformShift, ToEffectPattern::UniformShift),
            (EffectPattern::TailEffect, ToEffectPattern::TailEffect),
            (EffectPattern::Mixed, ToEffectPattern::Mixed),
            (EffectPattern::Indeterminate, ToEffectPattern::Indeterminate),
        ];

        for (core_val, expected) in conversions {
            let ffi_val: ToEffectPattern = core_val.into();
            assert_eq!(ffi_val, expected, "Conversion failed for {:?}", core_val);
        }
    }

    #[test]
    fn test_quality_from_core() {
        use timing_oracle_core::result::MeasurementQuality;

        let conversions = [
            (MeasurementQuality::Excellent, ToQuality::Excellent),
            (MeasurementQuality::Good, ToQuality::Good),
            (MeasurementQuality::Poor, ToQuality::Poor),
            (MeasurementQuality::TooNoisy, ToQuality::TooNoisy),
        ];

        for (core_val, expected) in conversions {
            let ffi_val: ToQuality = core_val.into();
            assert_eq!(ffi_val, expected, "Conversion failed for {:?}", core_val);
        }
    }

    #[test]
    fn test_effect_from_core() {
        use timing_oracle_core::result::{EffectEstimate, EffectPattern};

        let core_effect = EffectEstimate {
            shift_ns: 10.5,
            tail_ns: 5.2,
            credible_interval_ns: (2.0, 18.0),
            pattern: EffectPattern::Mixed,
            interpretation_caveat: None,
        };

        let ffi_effect: ToEffect = core_effect.into();
        assert!((ffi_effect.shift_ns - 10.5).abs() < 1e-10);
        assert!((ffi_effect.tail_ns - 5.2).abs() < 1e-10);
        assert!((ffi_effect.ci_low_ns - 2.0).abs() < 1e-10);
        assert!((ffi_effect.ci_high_ns - 18.0).abs() < 1e-10);
        assert_eq!(ffi_effect.pattern, ToEffectPattern::Mixed);
    }

    #[test]
    fn test_attacker_model_all_thresholds() {
        // Test all attacker model threshold values
        let models = [
            (ToAttackerModel::SharedHardware, 0.6),
            (ToAttackerModel::PostQuantum, 3.3),
            (ToAttackerModel::AdjacentNetwork, 100.0),
            (ToAttackerModel::RemoteNetwork, 50_000.0),
            (ToAttackerModel::Research, 0.0),
        ];

        for (model, expected) in models {
            let threshold = model.to_threshold_ns(0.0);
            assert!(
                (threshold - expected).abs() < 1e-10,
                "Threshold mismatch for {:?}: got {}, expected {}",
                model,
                threshold,
                expected
            );
        }

        // Test custom threshold
        assert!((ToAttackerModel::Custom.to_threshold_ns(123.45) - 123.45).abs() < 1e-10);
    }

    // =========================================================================
    // Group 3: Default Values & Config
    // =========================================================================

    #[test]
    fn test_config_default_all_models() {
        let models = [
            ToAttackerModel::SharedHardware,
            ToAttackerModel::PostQuantum,
            ToAttackerModel::AdjacentNetwork,
            ToAttackerModel::RemoteNetwork,
            ToAttackerModel::Research,
        ];

        for model in models {
            let config = to_config_default(model);
            assert_eq!(config.attacker_model, model);
            // Default thresholds should be 0.05 and 0.95
            assert!((config.pass_threshold - 0.05).abs() < 1e-10);
            assert!((config.fail_threshold - 0.95).abs() < 1e-10);
            // Default max_samples should be 0 (meaning use default 100,000)
            assert_eq!(config.max_samples, 0);
            // Default time_budget_secs is 0.0 (meaning use internal default of 30s)
            assert!((config.time_budget_secs - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_result_default() {
        let result = ToResult::default();
        assert_eq!(result.outcome, ToOutcome::Pass);
        assert!((result.leak_probability - 0.0).abs() < 1e-10);
        assert_eq!(result.samples_used, 0);
        assert!((result.elapsed_secs - 0.0).abs() < 1e-10);
        assert_eq!(result.exploitability, ToExploitability::SharedHardwareOnly);
        assert!(result.recommendation.is_null());
        assert!(!result.adaptive_batching_used);
    }

    #[test]
    fn test_config_values_reasonable() {
        let config = to_config_default(ToAttackerModel::AdjacentNetwork);

        // Thresholds should be in [0, 1]
        assert!(config.pass_threshold >= 0.0 && config.pass_threshold <= 1.0);
        assert!(config.fail_threshold >= 0.0 && config.fail_threshold <= 1.0);

        // Pass threshold should be less than fail threshold
        assert!(config.pass_threshold < config.fail_threshold);

        // Time budget can be 0.0 (meaning "use internal default")
        // or any non-negative value
        assert!(config.time_budget_secs >= 0.0);

        // Seed can be any value (0 is valid, means use random seed)
        let _seed = config.seed;
    }

    // =========================================================================
    // Group 4: Metadata Functions
    // =========================================================================

    #[test]
    fn test_version_format() {
        let ptr = to_version();
        assert!(!ptr.is_null(), "to_version returned null");

        let cstr = unsafe { CStr::from_ptr(ptr) };
        let version = cstr.to_str().expect("Invalid UTF-8 in version string");

        // Should be a valid semver format (x.y.z)
        let parts: Vec<&str> = version.split('.').collect();
        assert!(
            parts.len() >= 2,
            "Version should have at least major.minor: {}",
            version
        );

        // Each part should be a valid number
        for (i, part) in parts.iter().enumerate() {
            // Handle pre-release suffixes like "0.1.0-alpha"
            let num_part = part.split('-').next().unwrap();
            assert!(
                num_part.parse::<u32>().is_ok(),
                "Version part {} is not a number: {}",
                i,
                part
            );
        }
    }

    #[test]
    fn test_timer_name_non_null() {
        let ptr = to_timer_name();
        assert!(!ptr.is_null(), "to_timer_name returned null");

        let cstr = unsafe { CStr::from_ptr(ptr) };
        let name = cstr.to_str().expect("Invalid UTF-8 in timer name");

        // Should be a non-empty string
        assert!(!name.is_empty(), "Timer name should not be empty");

        // Should be one of the known timer names
        let known_timers = ["rdtsc", "cntvct_el0", "clock_gettime", "perf", "kperf"];
        assert!(
            known_timers.iter().any(|&t| name.contains(t)),
            "Unknown timer name: {}",
            name
        );
    }

    #[test]
    fn test_timer_frequency_reasonable() {
        let freq = to_timer_frequency();

        // Timer frequency varies by platform:
        // - x86_64 TSC: ~1-5 GHz (CPU frequency)
        // - ARM64 CNTVCT_EL0: ~24 MHz (system timer)
        // - clock_gettime fallback: 1 GHz (nanoseconds)
        // Minimum reasonable is 1 MHz
        assert!(
            freq >= 1_000_000,
            "Timer frequency too low: {} Hz (expected >= 1 MHz)",
            freq
        );

        // Should be less than 10 GHz (sanity check)
        assert!(
            freq <= 10_000_000_000,
            "Timer frequency unreasonably high: {} Hz",
            freq
        );
    }
}
