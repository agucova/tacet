//! C bindings for timing-oracle with support for C measurement loops.
//!
//! This crate provides a C-compatible API for timing side-channel detection.
//! It supports two usage patterns:
//!
//! 1. **C Measurement Loop**: Collect timing samples in C, call Rust only for analysis
//! 2. **One-Shot Analysis**: Analyze pre-collected timing data
//!
//! The C measurement loop approach minimizes FFI overhead during timing-critical code.
//!
//! # API Overview
//!
//! ## Low-Level Adaptive API
//!
//! For a C measurement loop with adaptive sampling:
//!
//! 1. Call `to_calibrate()` with initial samples to get calibration data
//! 2. Create state with `to_state_new()`
//! 3. In a loop: collect more samples in C, then call `to_step()`
//! 4. Stop when `to_step()` returns a decision
//!
//! ## High-Level API
//!
//! For one-shot analysis of pre-collected data:
//!
//! 1. Call `to_analyze()` with all your samples
//!
//! # Example (pseudocode)
//!
//! ```c
//! // Low-level adaptive loop
//! ToConfig config = to_config_adjacent_network();
//! ToError err;
//!
//! // Collect calibration samples
//! uint64_t baseline[5000], sample[5000];
//! collect_samples(baseline, sample, 5000);
//!
//! // Calibrate
//! ToCalibration* cal = to_calibrate(baseline, sample, 5000, &config, &err);
//! ToState* state = to_state_new();
//!
//! // Adaptive loop
//! ToStepResult step_result;
//! double start_time = get_time();
//! while (1) {
//!     // Collect a batch of new samples
//!     uint64_t new_baseline[1000], new_sample[1000];
//!     collect_samples(new_baseline, new_sample, 1000);
//!
//!     double elapsed = get_time() - start_time;
//!     err = to_step(cal, state, new_baseline, new_sample, 1000, &config, elapsed, &step_result);
//!
//!     if (step_result.has_decision) {
//!         printf("Decision: %d, P(leak)=%.2f%%\n",
//!                step_result.result.outcome,
//!                step_result.result.leak_probability * 100);
//!         break;
//!     }
//! }
//!
//! to_state_free(state);
//! to_calibration_free(cal);
//! ```

#![allow(clippy::missing_safety_doc)]

use std::ptr;
use std::slice;

use timing_oracle_core::adaptive::{
    calibrate_t_prior_scale, compute_c_floor_9d, AdaptiveState, AdaptiveStepConfig, Calibration,
    CalibrationSnapshot, StepResult,
};
use timing_oracle_core::analysis::compute_bayes_gibbs;
use timing_oracle_core::constants::DEFAULT_BOOTSTRAP_ITERATIONS;
use timing_oracle_core::result::{Exploitability, MeasurementQuality};
use timing_oracle_core::statistics::{
    bootstrap_difference_covariance_discrete, compute_deciles_inplace, optimal_block_length,
    StatsSnapshot,
};
use timing_oracle_core::types::AttackerModel;

mod types;

pub use types::*;

// ============================================================================
// Configuration
// ============================================================================

/// Create a default configuration for the given attacker model.
#[no_mangle]
pub extern "C" fn to_config_default(attacker_model: ToAttackerModel) -> ToConfig {
    ToConfig {
        attacker_model,
        ..ToConfig::default()
    }
}

/// Create a configuration for SharedHardware attacker model.
#[no_mangle]
pub extern "C" fn to_config_shared_hardware() -> ToConfig {
    to_config_default(ToAttackerModel::SharedHardware)
}

/// Create a configuration for AdjacentNetwork attacker model.
#[no_mangle]
pub extern "C" fn to_config_adjacent_network() -> ToConfig {
    to_config_default(ToAttackerModel::AdjacentNetwork)
}

/// Create a configuration for RemoteNetwork attacker model.
#[no_mangle]
pub extern "C" fn to_config_remote_network() -> ToConfig {
    to_config_default(ToAttackerModel::RemoteNetwork)
}

// ============================================================================
// Environment Variable Configuration
// ============================================================================

/// Merge configuration from TO_* environment variables.
///
/// Allows CI systems to override configuration without recompiling.
///
/// Supported environment variables:
/// - TO_TIME_BUDGET_SECS: Time budget in seconds (float)
/// - TO_MAX_SAMPLES: Maximum samples per class (integer)
/// - TO_PASS_THRESHOLD: Pass threshold for P(leak) (float, e.g., 0.05)
/// - TO_FAIL_THRESHOLD: Fail threshold for P(leak) (float, e.g., 0.95)
/// - TO_SEED: Random seed (integer, 0 = use default)
/// - TO_THRESHOLD_NS: Custom threshold in nanoseconds (float)
///
/// # Example
///
/// ```c
/// // In shell: export TO_TIME_BUDGET_SECS=60
/// ToConfig cfg = to_config_adjacent_network();
/// cfg.time_budget_secs = 30.0;
/// cfg.max_samples = 100000;
/// cfg = to_config_from_env(cfg);  // CI can override
/// ```
#[no_mangle]
pub extern "C" fn to_config_from_env(mut base: ToConfig) -> ToConfig {
    if let Ok(v) = std::env::var("TO_TIME_BUDGET_SECS") {
        if let Ok(secs) = v.parse::<f64>() {
            if secs > 0.0 {
                base.time_budget_secs = secs;
            }
        }
    }

    if let Ok(v) = std::env::var("TO_MAX_SAMPLES") {
        if let Ok(samples) = v.parse::<u64>() {
            if samples > 0 {
                base.max_samples = samples;
            }
        }
    }

    if let Ok(v) = std::env::var("TO_PASS_THRESHOLD") {
        if let Ok(thresh) = v.parse::<f64>() {
            if (0.0..=1.0).contains(&thresh) {
                base.pass_threshold = thresh;
            }
        }
    }

    if let Ok(v) = std::env::var("TO_FAIL_THRESHOLD") {
        if let Ok(thresh) = v.parse::<f64>() {
            if (0.0..=1.0).contains(&thresh) {
                base.fail_threshold = thresh;
            }
        }
    }

    if let Ok(v) = std::env::var("TO_SEED") {
        if let Ok(seed) = v.parse::<u64>() {
            base.seed = seed;
        }
    }

    if let Ok(v) = std::env::var("TO_THRESHOLD_NS") {
        if let Ok(thresh) = v.parse::<f64>() {
            if thresh > 0.0 {
                base.custom_threshold_ns = thresh;
            }
        }
    }

    base
}

// ============================================================================
// Low-Level API: C Measurement Loop
// ============================================================================

/// Create a new adaptive state for tracking the measurement loop.
///
/// Must be freed with `to_state_free()`.
///
/// # Returns
/// Pointer to new state, or NULL on allocation failure.
#[no_mangle]
pub extern "C" fn to_state_new() -> *mut ToState {
    Box::into_raw(Box::new(ToState {
        inner: AdaptiveState::new(),
    }))
}

/// Free an adaptive state.
///
/// # Safety
/// `state` must be a valid pointer returned by `to_state_new()`, or NULL.
#[no_mangle]
pub unsafe extern "C" fn to_state_free(state: *mut ToState) {
    if !state.is_null() {
        drop(Box::from_raw(state));
    }
}

/// Get the total number of samples collected (both classes combined).
///
/// # Safety
/// `state` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn to_state_total_samples(state: *const ToState) -> u64 {
    if state.is_null() {
        return 0;
    }
    (*state).inner.n_total() as u64
}

/// Get the current leak probability estimate.
///
/// Returns 0.5 if no posterior has been computed yet.
///
/// # Safety
/// `state` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn to_state_leak_probability(state: *const ToState) -> f64 {
    if state.is_null() {
        return 0.5;
    }
    (*state)
        .inner
        .current_posterior()
        .map(|p| p.leak_probability)
        .unwrap_or(0.5)
}

/// Run calibration on initial samples.
///
/// This should be called once at the start with calibration samples
/// (typically 5000 per class). The returned calibration handle is used
/// for subsequent `to_step()` calls.
///
/// # Parameters
/// - `baseline`: Array of baseline timing samples (in timer ticks)
/// - `sample`: Array of sample timing samples (in timer ticks)
/// - `count`: Number of samples in each array
/// - `config`: Configuration
/// - `error_out`: Optional pointer to receive error code
///
/// # Returns
/// Pointer to calibration data, or NULL on error.
/// Must be freed with `to_calibration_free()`.
///
/// # Safety
/// `baseline` and `sample` must be valid pointers to arrays of at least `count` elements.
#[no_mangle]
pub unsafe extern "C" fn to_calibrate(
    baseline: *const u64,
    sample: *const u64,
    count: usize,
    config: *const ToConfig,
    error_out: *mut ToError,
) -> *mut ToCalibration {
    let set_error = |e: ToError| {
        if !error_out.is_null() {
            *error_out = e;
        }
    };

    if baseline.is_null() || sample.is_null() || config.is_null() {
        set_error(ToError::NullPointer);
        return ptr::null_mut();
    }

    if count < 100 {
        set_error(ToError::NotEnoughSamples);
        return ptr::null_mut();
    }

    let baseline_slice = slice::from_raw_parts(baseline, count);
    let sample_slice = slice::from_raw_parts(sample, count);
    let cfg = &*config;

    // Get threshold from config
    let theta_ns = cfg.threshold_ns();

    // Convert to nanoseconds (assuming 1 tick = 1 ns if no frequency specified)
    let ns_per_tick = if cfg.timer_frequency_hz == 0 {
        1.0
    } else {
        1e9 / cfg.timer_frequency_hz as f64
    };

    let baseline_ns: Vec<f64> = baseline_slice
        .iter()
        .map(|&t| t as f64 * ns_per_tick)
        .collect();
    let sample_ns: Vec<f64> = sample_slice
        .iter()
        .map(|&t| t as f64 * ns_per_tick)
        .collect();

    // Check for discrete mode (< 10% unique values)
    let unique_baseline: std::collections::HashSet<u64> = baseline_slice.iter().copied().collect();
    let unique_sample: std::collections::HashSet<u64> = sample_slice.iter().copied().collect();
    let unique_fraction =
        (unique_baseline.len() + unique_sample.len()) as f64 / (2 * count) as f64;
    let discrete_mode = unique_fraction < 0.1;

    // Compute block length using Politis-White algorithm (use circular bootstrap length)
    let block_length_baseline = optimal_block_length(&baseline_ns).circular.ceil() as usize;
    let block_length_sample = optimal_block_length(&sample_ns).circular.ceil() as usize;
    let block_length = block_length_baseline.max(block_length_sample).max(1);

    // Bootstrap covariance estimation using discrete mode function (works for both)
    let seed = if cfg.seed == 0 { 42 } else { cfg.seed };
    let cov_estimate = bootstrap_difference_covariance_discrete(
        &baseline_ns,
        &sample_ns,
        DEFAULT_BOOTSTRAP_ITERATIONS,
        seed,
    );
    let sigma_rate = cov_estimate.matrix * count as f64; // Convert to rate: sigma_rate = sigma_cal * n_cal

    // Compute c_floor for theta_floor estimation
    let c_floor = compute_c_floor_9d(&sigma_rate, seed);

    // Compute initial theta_floor
    let n_eff = (count / block_length).max(1);
    let theta_floor_stat = c_floor / (n_eff as f64).sqrt();
    let theta_tick = ns_per_tick; // 1 tick resolution
    let theta_floor_initial = theta_floor_stat.max(theta_tick);
    let theta_eff = theta_ns.max(theta_floor_initial);

    // Calibrate Student's t prior scale
    let (sigma_t, l_r) =
        calibrate_t_prior_scale(&sigma_rate, theta_eff, count, discrete_mode, seed);

    // Compute calibration snapshot for drift detection
    let baseline_stats = compute_stats_snapshot(&baseline_ns);
    let sample_stats = compute_stats_snapshot(&sample_ns);
    let calibration_snapshot = CalibrationSnapshot::new(baseline_stats, sample_stats);

    // Placeholder MDE (would need proper computation)
    let mde_shift_ns = theta_floor_stat;
    let mde_tail_ns = theta_floor_stat * 1.5;

    // Projection mismatch threshold (chi-squared 99th percentile for 7 DoF)
    let projection_mismatch_thresh = 18.48;

    // Estimate samples per second (would be measured during calibration in real usage)
    let samples_per_second = 100_000.0; // Placeholder

    let calibration = Calibration::new(
        sigma_rate,
        block_length,
        sigma_t,
        l_r,
        theta_ns,
        count,
        discrete_mode,
        mde_shift_ns,
        mde_tail_ns,
        calibration_snapshot,
        ns_per_tick, // timer_resolution_ns
        samples_per_second,
        c_floor,
        projection_mismatch_thresh,
        theta_tick,
        theta_eff,
        theta_floor_initial,
        seed,
        1, // batch_k (no batching for C measurement loop)
    );

    set_error(ToError::Ok);
    Box::into_raw(Box::new(ToCalibration {
        inner: calibration,
        ns_per_tick,
    }))
}

/// Compute stats snapshot from samples.
fn compute_stats_snapshot(samples: &[f64]) -> StatsSnapshot {
    let count = samples.len();
    if count == 0 {
        return StatsSnapshot {
            mean: 0.0,
            variance: 0.0,
            autocorr_lag1: 0.0,
            count: 0,
        };
    }

    let mean = samples.iter().sum::<f64>() / count as f64;
    let variance = if count > 1 {
        samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64
    } else {
        0.0
    };

    // Simple lag-1 autocorrelation
    let autocorr_lag1 = if count > 1 && variance > 0.0 {
        let mut sum = 0.0;
        for i in 0..count - 1 {
            sum += (samples[i] - mean) * (samples[i + 1] - mean);
        }
        sum / ((count - 1) as f64 * variance)
    } else {
        0.0
    };

    StatsSnapshot {
        mean,
        variance,
        autocorr_lag1,
        count,
    }
}

/// Free calibration data.
///
/// # Safety
/// `calibration` must be a valid pointer returned by `to_calibrate()`, or NULL.
#[no_mangle]
pub unsafe extern "C" fn to_calibration_free(calibration: *mut ToCalibration) {
    if !calibration.is_null() {
        drop(Box::from_raw(calibration));
    }
}

/// Run one adaptive step with a batch of new samples.
///
/// Call this in a loop after `to_calibrate()`. Each call processes a batch
/// of new timing samples and updates the posterior probability.
///
/// # Parameters
/// - `calibration`: Calibration data from `to_calibrate()`
/// - `state`: Adaptive state from `to_state_new()`
/// - `baseline`: Array of new baseline timing samples (in timer ticks)
/// - `sample`: Array of new sample timing samples (in timer ticks)
/// - `count`: Number of samples in each array
/// - `config`: Configuration
/// - `elapsed_secs`: Total elapsed time since start
/// - `result_out`: Pointer to receive the step result
///
/// # Returns
/// Error code.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn to_step(
    calibration: *const ToCalibration,
    state: *mut ToState,
    baseline: *const u64,
    sample: *const u64,
    count: usize,
    config: *const ToConfig,
    elapsed_secs: f64,
    result_out: *mut ToStepResult,
) -> ToError {
    if calibration.is_null()
        || state.is_null()
        || baseline.is_null()
        || sample.is_null()
        || config.is_null()
        || result_out.is_null()
    {
        return ToError::NullPointer;
    }

    let baseline_slice = slice::from_raw_parts(baseline, count);
    let sample_slice = slice::from_raw_parts(sample, count);
    let cfg = &*config;
    let cal = &(*calibration).inner;
    let ns_per_tick = (*calibration).ns_per_tick;

    // Add batch to state
    (*state)
        .inner
        .add_batch(baseline_slice.to_vec(), sample_slice.to_vec());

    // Create step config
    let step_config = AdaptiveStepConfig {
        pass_threshold: cfg.pass_threshold,
        fail_threshold: cfg.fail_threshold,
        time_budget_secs: cfg.time_budget_secs,
        max_samples: if cfg.max_samples == 0 {
            100_000
        } else {
            cfg.max_samples as usize
        },
        theta_ns: cfg.threshold_ns(),
        seed: if cfg.seed == 0 { 42 } else { cfg.seed },
        ..Default::default()
    };

    // Run adaptive step
    let step_result = timing_oracle_core::adaptive::adaptive_step(
        cal,
        &mut (*state).inner,
        ns_per_tick,
        elapsed_secs,
        &step_config,
    );

    match step_result {
        StepResult::Continue {
            posterior,
            samples_per_class,
        } => {
            (*result_out).has_decision = false;
            (*result_out).leak_probability = posterior.leak_probability;
            (*result_out).samples_used = samples_per_class as u64;
        }
        StepResult::Decision(outcome) => {
            (*result_out).has_decision = true;
            (*result_out).samples_used = outcome.samples_per_class() as u64;
            (*result_out).elapsed_secs = outcome.elapsed_secs();

            // Convert outcome to C result
            let result = convert_adaptive_outcome(&outcome, cal, cfg);
            (*result_out).result = result;
            (*result_out).leak_probability = outcome.leak_probability().unwrap_or(0.5);
        }
    }

    ToError::Ok
}

/// Convert an AdaptiveOutcome to a ToResult using FFI summary types.
/// This uses the canonical effect pattern classification from core (spec §3.4.6).
fn convert_adaptive_outcome(
    outcome: &timing_oracle_core::adaptive::AdaptiveOutcome,
    cal: &Calibration,
    _cfg: &ToConfig,
) -> ToResult {
    use timing_oracle_core::ffi_summary::{InconclusiveReasonKind, OutcomeType};
    use timing_oracle_core::result::EffectPattern;

    let summary = outcome.to_summary(cal);

    // Map OutcomeType to C ToOutcome
    let to_outcome = match summary.outcome_type {
        OutcomeType::Pass => ToOutcome::Pass,
        OutcomeType::Fail => ToOutcome::Fail,
        OutcomeType::Inconclusive | OutcomeType::ThresholdElevated => ToOutcome::Inconclusive,
    };

    // Map InconclusiveReasonKind to C ToInconclusiveReason
    let inconclusive_reason = match summary.inconclusive_reason {
        InconclusiveReasonKind::None => ToInconclusiveReason::None,
        InconclusiveReasonKind::DataTooNoisy => ToInconclusiveReason::DataTooNoisy,
        InconclusiveReasonKind::NotLearning => ToInconclusiveReason::NotLearning,
        InconclusiveReasonKind::WouldTakeTooLong => ToInconclusiveReason::WouldTakeTooLong,
        InconclusiveReasonKind::TimeBudgetExceeded => ToInconclusiveReason::TimeBudgetExceeded,
        InconclusiveReasonKind::SampleBudgetExceeded => ToInconclusiveReason::SampleBudgetExceeded,
        InconclusiveReasonKind::ConditionsChanged => ToInconclusiveReason::ConditionsChanged,
        InconclusiveReasonKind::ThresholdElevated => ToInconclusiveReason::ThresholdElevated,
    };

    // Map EffectPattern to C ToEffectPattern
    let pattern = match summary.effect.pattern {
        EffectPattern::UniformShift => ToEffectPattern::UniformShift,
        EffectPattern::TailEffect => ToEffectPattern::TailEffect,
        EffectPattern::Mixed => ToEffectPattern::Mixed,
        EffectPattern::Indeterminate => ToEffectPattern::Indeterminate,
    };

    ToResult {
        outcome: to_outcome,
        leak_probability: summary.leak_probability,
        effect: ToEffect {
            shift_ns: summary.effect.shift_ns,
            tail_ns: summary.effect.tail_ns,
            ci_low_ns: summary.effect.ci_low_ns,
            ci_high_ns: summary.effect.ci_high_ns,
            pattern,
        },
        quality: summary.quality.into(),
        samples_used: summary.samples_per_class as u64,
        elapsed_secs: summary.elapsed_secs,
        exploitability: summary.exploitability.into(),
        inconclusive_reason,
        mde_shift_ns: summary.mde_shift_ns,
        mde_tail_ns: summary.mde_tail_ns,
        theta_user_ns: summary.theta_user,
        theta_eff_ns: summary.theta_eff,
        theta_floor_ns: summary.theta_floor,
        timer_resolution_ns: summary.diagnostics.timer_resolution_ns,
        decision_threshold_ns: summary.theta_eff,
        diagnostics: ToDiagnostics {
            dependence_length: summary.diagnostics.dependence_length as u64,
            effective_sample_size: summary.diagnostics.effective_sample_size as u64,
            stationarity_ratio: summary.diagnostics.stationarity_ratio,
            stationarity_ok: summary.diagnostics.stationarity_ok,
            projection_mismatch_q: summary.diagnostics.projection_mismatch_q,
            projection_mismatch_ok: summary.diagnostics.projection_mismatch_ok,
            discrete_mode: summary.diagnostics.discrete_mode,
            timer_resolution_ns: summary.diagnostics.timer_resolution_ns,
            lambda_mean: summary.diagnostics.lambda_mean,
            lambda_sd: 0.0, // Not in DiagnosticsSummary yet
            lambda_ess: 0.0, // Not in DiagnosticsSummary yet
            lambda_mixing_ok: summary.diagnostics.lambda_mixing_ok,
            kappa_mean: summary.diagnostics.kappa_mean,
            kappa_cv: summary.diagnostics.kappa_cv,
            kappa_ess: summary.diagnostics.kappa_ess,
            kappa_mixing_ok: summary.diagnostics.kappa_mixing_ok,
        },
    }
}

// ============================================================================
// High-Level API: Callback-Based Test
// ============================================================================

/// Default calibration samples per class for to_test().
const TO_TEST_CALIBRATION_SAMPLES: usize = 5000;

/// Default batch size for to_test() adaptive loop.
const TO_TEST_BATCH_SIZE: usize = 1000;

/// Run a complete timing test using a callback for sample collection.
///
/// This is the recommended API for testing timing side channels. It handles
/// the full adaptive sampling loop internally:
///
/// 1. Collects calibration samples via the callback
/// 2. Runs calibration phase
/// 3. Loops: collects batches via callback, runs adaptive step
/// 4. Stops when a decision is reached or budget is exceeded
///
/// The callback function is called multiple times to collect samples. Each
/// invocation should fill the provided arrays with fresh timing measurements.
///
/// # Parameters
/// - `config`: Configuration (use presets like `to_config_balanced()`)
/// - `collect_fn`: Callback function to collect timing samples
/// - `user_ctx`: User context pointer passed to the callback (can be NULL)
/// - `result_out`: Pointer to receive the final result
///
/// # Returns
/// Error code.
///
/// # Example
///
/// ```c
/// void my_collect(uint64_t* baseline, uint64_t* sample, size_t count, void* ctx) {
///     for (size_t i = 0; i < count; i++) {
///         baseline[i] = measure(baseline_input());
///         sample[i] = measure(sample_input());
///     }
/// }
///
/// int main(void) {
///     ToConfig cfg = to_config_balanced();
///     cfg = to_config_from_env(cfg);  // CI can override via TO_TIME_BUDGET_SECS
///
///     ToResult result;
///     to_test(&cfg, my_collect, NULL, &result);
///
///     if (result.outcome == Fail) {
///         printf("Leak: P=%.1f%%, effect=%.1fns\\n",
///                result.leak_probability * 100, result.effect.shift_ns);
///         return 1;
///     }
///     return 0;
/// }
/// ```
///
/// # Safety
/// - `config` must be a valid pointer
/// - `collect_fn` must be a valid function pointer
/// - `result_out` must be a valid pointer
/// - The callback must fill exactly `count` samples in each array
#[no_mangle]
pub unsafe extern "C" fn to_test(
    config: *const ToConfig,
    collect_fn: ToCollectFn,
    user_ctx: *mut std::ffi::c_void,
    result_out: *mut ToResult,
) -> ToError {
    // Validate inputs
    if config.is_null() || result_out.is_null() {
        return ToError::NullPointer;
    }

    let collect_fn = match collect_fn {
        Some(f) => f,
        None => return ToError::NullPointer,
    };

    let cfg = &*config;

    // Get time tracking
    let start_time = std::time::Instant::now();
    let time_budget = if cfg.time_budget_secs > 0.0 {
        std::time::Duration::from_secs_f64(cfg.time_budget_secs)
    } else {
        std::time::Duration::from_secs(30)
    };

    // Phase 1: Collect calibration samples
    let mut cal_baseline = vec![0u64; TO_TEST_CALIBRATION_SAMPLES];
    let mut cal_sample = vec![0u64; TO_TEST_CALIBRATION_SAMPLES];

    collect_fn(
        cal_baseline.as_mut_ptr(),
        cal_sample.as_mut_ptr(),
        TO_TEST_CALIBRATION_SAMPLES,
        user_ctx,
    );

    // Phase 2: Run calibration
    let mut error = ToError::Ok;
    let cal_ptr = to_calibrate(
        cal_baseline.as_ptr(),
        cal_sample.as_ptr(),
        TO_TEST_CALIBRATION_SAMPLES,
        config,
        &mut error,
    );

    if cal_ptr.is_null() || error != ToError::Ok {
        return if error != ToError::Ok {
            error
        } else {
            ToError::CalibrationFailed
        };
    }

    let cal = &(*cal_ptr).inner;
    let ns_per_tick = (*cal_ptr).ns_per_tick;

    // Create adaptive state
    let mut state = AdaptiveState::new();

    // Create step config
    let step_config = AdaptiveStepConfig {
        pass_threshold: cfg.pass_threshold,
        fail_threshold: cfg.fail_threshold,
        time_budget_secs: cfg.time_budget_secs,
        max_samples: if cfg.max_samples == 0 {
            100_000
        } else {
            cfg.max_samples as usize
        },
        theta_ns: cfg.threshold_ns(),
        seed: if cfg.seed == 0 { 42 } else { cfg.seed },
        ..Default::default()
    };

    // Phase 3: Adaptive sampling loop
    let mut batch_baseline = vec![0u64; TO_TEST_BATCH_SIZE];
    let mut batch_sample = vec![0u64; TO_TEST_BATCH_SIZE];

    loop {
        // Check time budget before collecting more samples
        let elapsed = start_time.elapsed();
        if elapsed >= time_budget {
            // Time budget exceeded - return inconclusive result
            *result_out = ToResult {
                outcome: ToOutcome::Inconclusive,
                leak_probability: state.current_posterior().map(|p| p.leak_probability).unwrap_or(0.5),
                inconclusive_reason: ToInconclusiveReason::TimeBudgetExceeded,
                elapsed_secs: elapsed.as_secs_f64(),
                samples_used: state.n_total() as u64 / 2,
                ..ToResult::default()
            };
            to_calibration_free(cal_ptr);
            return ToError::Ok;
        }

        // Collect a batch of samples
        collect_fn(
            batch_baseline.as_mut_ptr(),
            batch_sample.as_mut_ptr(),
            TO_TEST_BATCH_SIZE,
            user_ctx,
        );

        // Add batch to state
        state.add_batch(batch_baseline.clone(), batch_sample.clone());

        // Run adaptive step
        let elapsed_secs = start_time.elapsed().as_secs_f64();
        let step_result = timing_oracle_core::adaptive::adaptive_step(
            cal,
            &mut state,
            ns_per_tick,
            elapsed_secs,
            &step_config,
        );

        match step_result {
            StepResult::Continue { .. } => {
                // Continue loop
            }
            StepResult::Decision(outcome) => {
                // Convert outcome to C result
                *result_out = convert_adaptive_outcome(&outcome, cal, cfg);
                to_calibration_free(cal_ptr);
                return ToError::Ok;
            }
        }
    }
}

// ============================================================================
// High-Level API: One-Shot Analysis
// ============================================================================

/// Analyze pre-collected timing samples.
///
/// This is a convenience function for one-shot analysis when you already
/// have timing data collected. For a C measurement loop, use
/// `to_calibrate()` + `to_step()` instead.
///
/// # Parameters
/// - `baseline`: Array of baseline timing samples (in timer ticks)
/// - `sample`: Array of sample timing samples (in timer ticks)
/// - `count`: Number of samples in each array
/// - `config`: Configuration
/// - `result_out`: Pointer to receive the result
///
/// # Returns
/// Error code.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn to_analyze(
    baseline: *const u64,
    sample: *const u64,
    count: usize,
    config: *const ToConfig,
    result_out: *mut ToResult,
) -> ToError {
    if baseline.is_null() || sample.is_null() || config.is_null() || result_out.is_null() {
        return ToError::NullPointer;
    }

    if count < 100 {
        return ToError::NotEnoughSamples;
    }

    let baseline_slice = slice::from_raw_parts(baseline, count);
    let sample_slice = slice::from_raw_parts(sample, count);
    let cfg = &*config;

    // Get threshold and conversion factor
    let theta_ns = cfg.threshold_ns();
    let ns_per_tick = if cfg.timer_frequency_hz == 0 {
        1.0
    } else {
        1e9 / cfg.timer_frequency_hz as f64
    };

    // Convert to nanoseconds
    let baseline_ns: Vec<f64> = baseline_slice
        .iter()
        .map(|&t| t as f64 * ns_per_tick)
        .collect();
    let sample_ns: Vec<f64> = sample_slice
        .iter()
        .map(|&t| t as f64 * ns_per_tick)
        .collect();

    // Compute quantile differences
    let mut baseline_sorted = baseline_ns.clone();
    let mut sample_sorted = sample_ns.clone();
    let q_baseline = compute_deciles_inplace(&mut baseline_sorted);
    let q_sample = compute_deciles_inplace(&mut sample_sorted);
    let observed_diff = q_baseline - q_sample;

    // Bootstrap covariance
    let seed = if cfg.seed == 0 { 42 } else { cfg.seed };
    let cov_estimate = bootstrap_difference_covariance_discrete(
        &baseline_ns,
        &sample_ns,
        DEFAULT_BOOTSTRAP_ITERATIONS,
        seed,
    );
    let sigma_rate = cov_estimate.matrix * count as f64;

    // Compute theta_eff
    let c_floor = compute_c_floor_9d(&sigma_rate, seed);
    let block_length_baseline = optimal_block_length(&baseline_ns).circular.ceil() as usize;
    let block_length_sample = optimal_block_length(&sample_ns).circular.ceil() as usize;
    let block_length = block_length_baseline.max(block_length_sample).max(1);
    let n_eff = (count / block_length).max(1);
    let theta_floor = (c_floor / (n_eff as f64).sqrt()).max(ns_per_tick);
    let theta_eff = theta_ns.max(theta_floor);

    // Calibrate prior
    let discrete_mode = false; // Simple assumption for one-shot
    let (sigma_t, l_r) =
        calibrate_t_prior_scale(&sigma_rate, theta_eff, count, discrete_mode, seed);

    // Scale covariance
    let sigma_n = sigma_rate / n_eff as f64;

    // Run Bayesian inference
    let bayes_result = compute_bayes_gibbs(
        &observed_diff,
        &sigma_n,
        sigma_t,
        &l_r,
        theta_eff,
        Some(seed),
    );

    // Extract effect estimate
    let shift_ns = bayes_result.beta_proj[0];
    let tail_ns = bayes_result.beta_proj[1];

    // Compute CIs from posterior covariance (simplified)
    let shift_sd = bayes_result.beta_proj_cov[(0, 0)].sqrt();
    let tail_sd = bayes_result.beta_proj_cov[(1, 1)].sqrt();
    let total_effect = (shift_ns.powi(2) + tail_ns.powi(2)).sqrt();
    let total_sd = (shift_sd.powi(2) + tail_sd.powi(2)).sqrt();

    // Simple 95% CI approximation
    let ci_low = (total_effect - 1.96 * total_sd).max(0.0);
    let ci_high = total_effect + 1.96 * total_sd;

    // Classify pattern using canonical classification from core
    // For one-shot analysis without posterior draws, we use a simplified heuristic
    // based on effect magnitudes relative to their standard errors (2×SE threshold)
    let pattern = {
        let shift_significant = shift_ns.abs() > 2.0 * shift_sd;
        let tail_significant = tail_ns.abs() > 2.0 * tail_sd;
        match (shift_significant, tail_significant) {
            (true, false) => ToEffectPattern::UniformShift,
            (false, true) => ToEffectPattern::TailEffect,
            (true, true) => ToEffectPattern::Mixed,
            (false, false) => ToEffectPattern::Indeterminate,
        }
    };

    // Determine outcome
    let leak_prob = bayes_result.leak_probability;
    let (outcome, inconclusive_reason) = if leak_prob < cfg.pass_threshold {
        (ToOutcome::Pass, ToInconclusiveReason::None)
    } else if leak_prob > cfg.fail_threshold {
        (ToOutcome::Fail, ToInconclusiveReason::None)
    } else {
        (ToOutcome::Inconclusive, ToInconclusiveReason::None)
    };

    // Compute exploitability (delegate to core)
    let exploitability: ToExploitability = Exploitability::from_effect_ns(total_effect).into();

    // Determine quality (delegate to core)
    let mde_ns = theta_floor;
    let quality: ToMeasurementQuality = MeasurementQuality::from_mde_ns(mde_ns).into();

    // Build diagnostics from bayes_result
    let diagnostics = ToDiagnostics {
        dependence_length: block_length as u64,
        effective_sample_size: n_eff as u64,
        stationarity_ratio: 1.0,
        stationarity_ok: true,
        projection_mismatch_q: bayes_result.projection_mismatch_q,
        projection_mismatch_ok: true, // Would need threshold from calibration
        discrete_mode,
        timer_resolution_ns: ns_per_tick,
        lambda_mean: bayes_result.lambda_mean,
        lambda_sd: bayes_result.lambda_sd,
        lambda_ess: bayes_result.lambda_ess,
        lambda_mixing_ok: bayes_result.lambda_mixing_ok,
        kappa_mean: bayes_result.kappa_mean,
        kappa_cv: bayes_result.kappa_cv,
        kappa_ess: bayes_result.kappa_ess,
        kappa_mixing_ok: bayes_result.kappa_mixing_ok,
    };

    // Build result
    *result_out = ToResult {
        outcome,
        leak_probability: leak_prob,
        effect: ToEffect {
            shift_ns,
            tail_ns,
            ci_low_ns: ci_low,
            ci_high_ns: ci_high,
            pattern,
        },
        quality,
        samples_used: count as u64,
        elapsed_secs: 0.0,
        exploitability,
        inconclusive_reason,
        mde_shift_ns: mde_ns,
        mde_tail_ns: mde_ns * 1.5,
        theta_user_ns: theta_ns,
        theta_eff_ns: theta_eff,
        theta_floor_ns: theta_floor,
        timer_resolution_ns: ns_per_tick,
        decision_threshold_ns: theta_eff,
        diagnostics,
    };

    ToError::Ok
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get the library version string.
///
/// # Returns
/// Pointer to a static null-terminated string.
#[no_mangle]
pub extern "C" fn to_version() -> *const std::ffi::c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const std::ffi::c_char
}

/// Get attacker model threshold in nanoseconds.
#[no_mangle]
pub extern "C" fn to_attacker_threshold_ns(model: ToAttackerModel) -> f64 {
    let rust_model: AttackerModel = model.into();
    rust_model.to_threshold_ns()
}
