//! FFI bindings for Go to timing-oracle statistical analysis.
//!
//! This crate exposes the statistical analysis functions from timing-oracle-core
//! via a C FFI suitable for Go's CGo. It does NOT include measurement functionality -
//! the Go side is expected to collect timing samples and pass raw arrays to these
//! functions for analysis.
//!
//! # Design
//!
//! The measurement loop should be implemented in pure Go to avoid FFI overhead
//! during timing-critical code. Only the statistical analysis crosses the FFI
//! boundary, which is not timing-sensitive.
//!
//! # Usage
//!
//! 1. Go collects timing samples (baseline and sample class)
//! 2. Go calls `togo_calibrate()` with initial samples
//! 3. Go collects more samples in batches
//! 4. Go calls `togo_adaptive_step()` after each batch
//! 5. Repeat until decision is reached
//! 6. Go calls `togo_result_free()` to clean up

#![allow(clippy::missing_safety_doc)]

pub mod types;

use std::ffi::{c_char, c_void, CString};
use std::panic::catch_unwind;
use std::ptr;
use std::slice;

use timing_oracle::adaptive::{calibrate, CalibrationConfig};
use timing_oracle_core::adaptive::{
    adaptive_step, AdaptiveOutcome, AdaptiveState, AdaptiveStepConfig, Calibration,
    InconclusiveReason, StepResult,
};
use timing_oracle_core::math::sqrt;

use types::*;

/// Library version string.
#[allow(dead_code)]
const VERSION: &str = env!("CARGO_PKG_VERSION");

// =============================================================================
// Version and metadata
// =============================================================================

/// Get the library version string.
#[no_mangle]
pub extern "C" fn togo_version() -> *const c_char {
    // Use a static to ensure the pointer remains valid
    static VERSION_CSTR: &[u8] = b"0.1.0\0";
    VERSION_CSTR.as_ptr() as *const c_char
}

// =============================================================================
// Configuration
// =============================================================================

/// Create a default configuration for the given attacker model.
#[no_mangle]
pub extern "C" fn togo_config_default(model: ToGoAttackerModel) -> ToGoConfig {
    ToGoConfig {
        attacker_model: model,
        ..ToGoConfig::default()
    }
}

// =============================================================================
// Calibration
// =============================================================================

/// Run calibration phase on initial timing samples.
///
/// This function performs the calibration phase:
/// 1. Computes quantile differences between baseline and sample classes
/// 2. Estimates covariance matrix via block bootstrap
/// 3. Sets up prior distribution for Bayesian inference
/// 4. Returns calibration state for use in adaptive steps
///
/// # Arguments
///
/// * `baseline` - Pointer to baseline timing samples (raw ticks/cycles)
/// * `baseline_len` - Number of baseline samples
/// * `sample` - Pointer to sample class timing samples (raw ticks/cycles)
/// * `sample_len` - Number of sample class samples
/// * `config` - Analysis configuration
/// * `calibration` - Output: calibration state handle
///
/// # Returns
///
/// * 0 on success
/// * -1 on null pointer
/// * -2 on insufficient samples
/// * -3 on internal error
#[no_mangle]
pub unsafe extern "C" fn togo_calibrate(
    baseline: *const u64,
    baseline_len: usize,
    sample: *const u64,
    sample_len: usize,
    config: *const ToGoConfig,
    calibration: *mut ToGoCalibration,
) -> i32 {
    // Validate inputs
    if baseline.is_null() || sample.is_null() || config.is_null() || calibration.is_null() {
        return -1;
    }

    if baseline_len < 100 || sample_len < 100 {
        return -2; // Need minimum samples for calibration
    }

    let result = catch_unwind(|| {
        let config = &*config;
        let baseline_slice = slice::from_raw_parts(baseline, baseline_len);
        let sample_slice = slice::from_raw_parts(sample, sample_len);

        // Configuration
        let theta_ns = config.theta_ns();
        let ns_per_tick = config.ns_per_tick();
        let seed = if config.seed == 0 {
            // Use a simple seed based on sample data
            baseline_slice.iter().take(10).sum::<u64>() ^ 0x12345678
        } else {
            config.seed
        };

        // Create calibration config
        let cal_config = CalibrationConfig {
            calibration_samples: baseline_len.min(sample_len),
            bootstrap_iterations: 200, // Reasonable default
            timer_resolution_ns: ns_per_tick,
            theta_ns,
            alpha: 0.01,
            seed,
            skip_preflight: true, // Skip preflight for FFI (Go handles measurement)
        };

        // Run calibration
        match calibrate(baseline_slice, sample_slice, ns_per_tick, &cal_config) {
            Ok(cal) => {
                // Convert to core Calibration type for use with adaptive_step (v5.2 mixture prior)
                let core_cal = timing_oracle_core::adaptive::Calibration::new(
                    cal.sigma_rate,
                    cal.block_length,
                    cal.prior_cov_narrow,
                    cal.prior_cov_slab,
                    cal.sigma_narrow,
                    cal.sigma_slab,
                    cal.prior_weight,
                    cal.theta_ns,
                    cal.calibration_samples,
                    cal.discrete_mode,
                    cal.mde_shift_ns,
                    cal.mde_tail_ns,
                    cal.calibration_snapshot.clone(),
                    cal.timer_resolution_ns,
                    cal.samples_per_second,
                    cal.c_floor,
                    cal.projection_mismatch_thresh,
                    cal.theta_tick,
                    cal.theta_eff,
                    cal.theta_floor_initial,
                    cal.rng_seed,
                );

                let cal_box = Box::new(core_cal);
                Ok(Box::into_raw(cal_box) as *mut c_void)
            }
            Err(_e) => Err(-3),
        }
    });

    match result {
        Ok(Ok(ptr)) => {
            (*calibration).ptr = ptr;
            0
        }
        Ok(Err(code)) => code,
        Err(_) => -3,
    }
}

/// Free calibration state.
#[no_mangle]
pub unsafe extern "C" fn togo_calibration_free(calibration: *mut ToGoCalibration) {
    if calibration.is_null() {
        return;
    }

    let ptr = (*calibration).ptr;
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr as *mut Calibration);
        (*calibration).ptr = ptr::null_mut();
    }
}

// =============================================================================
// Adaptive Step
// =============================================================================

/// Create a new adaptive state for tracking samples across batches.
#[no_mangle]
pub extern "C" fn togo_adaptive_state_new() -> ToGoAdaptiveState {
    let state = Box::new(AdaptiveState::new());
    ToGoAdaptiveState {
        total_baseline: 0,
        total_sample: 0,
        current_probability: 0.5,
        ptr: Box::into_raw(state) as *mut c_void,
    }
}

/// Free adaptive state.
#[no_mangle]
pub unsafe extern "C" fn togo_adaptive_state_free(state: *mut ToGoAdaptiveState) {
    if state.is_null() {
        return;
    }

    let ptr = (*state).ptr;
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr as *mut AdaptiveState);
        (*state).ptr = ptr::null_mut();
    }
}

/// Run one adaptive step with a new batch of samples.
///
/// # Arguments
///
/// * `calibration` - Calibration state from togo_calibrate()
/// * `baseline` - Pointer to new baseline timing samples
/// * `baseline_len` - Number of new baseline samples
/// * `sample` - Pointer to new sample class timing samples
/// * `sample_len` - Number of new sample class samples
/// * `config` - Analysis configuration
/// * `elapsed_secs` - Elapsed time in seconds (measured by Go)
/// * `state` - In/out adaptive state
/// * `result` - Output result (only filled if decision reached)
///
/// # Returns
///
/// * 0: Continue sampling (no decision yet)
/// * 1: Decision reached (result is filled)
/// * -1: Null pointer
/// * -2: Invalid calibration
/// * -3: Internal error
#[no_mangle]
pub unsafe extern "C" fn togo_adaptive_step(
    calibration: *const ToGoCalibration,
    baseline: *const u64,
    baseline_len: usize,
    sample: *const u64,
    sample_len: usize,
    config: *const ToGoConfig,
    elapsed_secs: f64,
    state: *mut ToGoAdaptiveState,
    result: *mut ToGoResult,
) -> i32 {
    // Validate inputs
    if calibration.is_null()
        || baseline.is_null()
        || sample.is_null()
        || config.is_null()
        || state.is_null()
        || result.is_null()
    {
        return -1;
    }

    if (*calibration).ptr.is_null() || (*state).ptr.is_null() {
        return -2;
    }

    let step_result = catch_unwind(|| {
        let config = &*config;
        let cal = &*((*calibration).ptr as *const Calibration);
        let adaptive_state = &mut *((*state).ptr as *mut AdaptiveState);

        let baseline_slice = slice::from_raw_parts(baseline, baseline_len);
        let sample_slice = slice::from_raw_parts(sample, sample_len);

        // Add batch to state
        adaptive_state.add_batch(baseline_slice.to_vec(), sample_slice.to_vec());

        // Update external tracking
        (*state).total_baseline = adaptive_state.baseline_samples.len();
        (*state).total_sample = adaptive_state.sample_samples.len();

        // Create step config
        let step_config = AdaptiveStepConfig {
            pass_threshold: config.pass_threshold,
            fail_threshold: config.fail_threshold,
            time_budget_secs: config.time_budget_secs,
            max_samples: config.max_samples,
            theta_ns: config.theta_ns(),
            seed: config.seed,
            ..AdaptiveStepConfig::default()
        };

        // Run adaptive step
        let ns_per_tick = config.ns_per_tick();
        let step = adaptive_step(cal, adaptive_state, ns_per_tick, elapsed_secs, &step_config);

        // Update current probability
        if let Some(prob) = step.leak_probability() {
            (*state).current_probability = prob;
        }

        step
    });

    match step_result {
        Ok(step) => match step {
            StepResult::Continue { .. } => 0,
            StepResult::Decision(outcome) => {
                fill_result_from_outcome(&outcome, &*config, result);
                1
            }
        },
        Err(_) => -3,
    }
}

// =============================================================================
// One-shot Analysis
// =============================================================================

/// Run complete analysis on pre-collected timing data.
///
/// This is a convenience function that runs calibration and adaptive analysis
/// in a single call. Use the separate calibrate/adaptive_step functions for
/// incremental analysis.
///
/// # Arguments
///
/// * `baseline` - Pointer to all baseline timing samples
/// * `baseline_len` - Number of baseline samples
/// * `sample` - Pointer to all sample class timing samples
/// * `sample_len` - Number of sample class samples
/// * `config` - Analysis configuration
/// * `result` - Output result
///
/// # Returns
///
/// * 0: Success
/// * -1: Null pointer
/// * -2: Insufficient samples
/// * -3: Internal error
#[no_mangle]
pub unsafe extern "C" fn togo_analyze(
    baseline: *const u64,
    baseline_len: usize,
    sample: *const u64,
    sample_len: usize,
    config: *const ToGoConfig,
    result: *mut ToGoResult,
) -> i32 {
    // Validate inputs
    if baseline.is_null() || sample.is_null() || config.is_null() || result.is_null() {
        return -1;
    }

    if baseline_len < 100 || sample_len < 100 {
        return -2;
    }

    let analysis_result = catch_unwind(|| {
        let config = &*config;
        let baseline_slice = slice::from_raw_parts(baseline, baseline_len);
        let sample_slice = slice::from_raw_parts(sample, sample_len);

        let theta_ns = config.theta_ns();
        let ns_per_tick = config.ns_per_tick();
        let seed = if config.seed == 0 {
            baseline_slice.iter().take(10).sum::<u64>() ^ 0x12345678
        } else {
            config.seed
        };

        // Split into calibration and adaptive samples
        let cal_samples = 5000.min(baseline_len / 2);

        // Calibration config
        let cal_config = CalibrationConfig {
            calibration_samples: cal_samples,
            bootstrap_iterations: 200,
            timer_resolution_ns: ns_per_tick,
            theta_ns,
            alpha: 0.01,
            seed,
            skip_preflight: true,
        };

        // Run calibration on first portion
        let cal = match calibrate(
            &baseline_slice[..cal_samples],
            &sample_slice[..cal_samples],
            ns_per_tick,
            &cal_config,
        ) {
            Ok(c) => c,
            Err(_) => {
                return Err(-3);
            }
        };

        // Convert to core Calibration (v5.2 mixture prior)
        let core_cal = timing_oracle_core::adaptive::Calibration::new(
            cal.sigma_rate,
            cal.block_length,
            cal.prior_cov_narrow,
            cal.prior_cov_slab,
            cal.sigma_narrow,
            cal.sigma_slab,
            cal.prior_weight,
            cal.theta_ns,
            cal.calibration_samples,
            cal.discrete_mode,
            cal.mde_shift_ns,
            cal.mde_tail_ns,
            cal.calibration_snapshot.clone(),
            cal.timer_resolution_ns,
            cal.samples_per_second,
            cal.c_floor,
            cal.projection_mismatch_thresh,
            cal.theta_tick,
            cal.theta_eff,
            cal.theta_floor_initial,
            cal.rng_seed,
        );

        // Run adaptive loop on all samples
        let mut state = AdaptiveState::new();
        let step_config = AdaptiveStepConfig {
            pass_threshold: config.pass_threshold,
            fail_threshold: config.fail_threshold,
            time_budget_secs: config.time_budget_secs,
            max_samples: config.max_samples,
            theta_ns,
            seed,
            ..AdaptiveStepConfig::default()
        };

        // Add all samples in batches
        let batch_size = 1000;
        let mut elapsed_secs = 0.0;
        let time_per_batch = 0.01; // Estimate

        for (b_chunk, s_chunk) in baseline_slice
            .chunks(batch_size)
            .zip(sample_slice.chunks(batch_size))
        {
            state.add_batch(b_chunk.to_vec(), s_chunk.to_vec());
            elapsed_secs += time_per_batch;

            let step = adaptive_step(
                &core_cal,
                &mut state,
                ns_per_tick,
                elapsed_secs,
                &step_config,
            );

            if let StepResult::Decision(outcome) = step {
                return Ok((
                    outcome,
                    cal.mde_shift_ns,
                    cal.mde_tail_ns,
                    theta_ns,
                    cal.theta_eff,
                ));
            }
        }

        // If we exhausted samples without decision, return inconclusive
        let outcome = AdaptiveOutcome::Inconclusive {
            reason: InconclusiveReason::SampleBudgetExceeded {
                current_probability: state
                    .current_posterior()
                    .map(|p| p.leak_probability)
                    .unwrap_or(0.5),
                samples_collected: state.n_total(),
            },
            posterior: state.current_posterior().cloned(),
            samples_per_class: state.n_total(),
            elapsed_secs,
        };

        Ok((
            outcome,
            cal.mde_shift_ns,
            cal.mde_tail_ns,
            theta_ns,
            cal.theta_eff,
        ))
    });

    match analysis_result {
        Ok(Ok((outcome, mde_shift, mde_tail, theta_user, theta_eff))) => {
            fill_result_from_outcome(&outcome, &*config, result);
            (*result).mde_shift_ns = mde_shift;
            (*result).mde_tail_ns = mde_tail;
            (*result).theta_user_ns = theta_user;
            (*result).theta_eff_ns = theta_eff;
            0
        }
        Ok(Err(code)) => code,
        Err(_) => -3,
    }
}

// =============================================================================
// Result cleanup
// =============================================================================

/// Free a result's owned resources.
#[no_mangle]
pub unsafe extern "C" fn togo_result_free(result: *mut ToGoResult) {
    if result.is_null() {
        return;
    }

    let rec = (*result).recommendation;
    if !rec.is_null() {
        // Reconstruct and drop the CString
        let _ = CString::from_raw(rec as *mut c_char);
        (*result).recommendation = ptr::null();
    }
}

// =============================================================================
// String conversion utilities
// =============================================================================

/// Get string representation of outcome.
#[no_mangle]
pub extern "C" fn togo_outcome_str(outcome: ToGoOutcome) -> *const c_char {
    match outcome {
        ToGoOutcome::Pass => c"Pass".as_ptr(),
        ToGoOutcome::Fail => c"Fail".as_ptr(),
        ToGoOutcome::Inconclusive => c"Inconclusive".as_ptr(),
        ToGoOutcome::Unmeasurable => c"Unmeasurable".as_ptr(),
    }
}

/// Get string representation of effect pattern.
#[no_mangle]
pub extern "C" fn togo_effect_pattern_str(pattern: ToGoEffectPattern) -> *const c_char {
    match pattern {
        ToGoEffectPattern::UniformShift => c"UniformShift".as_ptr(),
        ToGoEffectPattern::TailEffect => c"TailEffect".as_ptr(),
        ToGoEffectPattern::Mixed => c"Mixed".as_ptr(),
        ToGoEffectPattern::Indeterminate => c"Indeterminate".as_ptr(),
    }
}

/// Get string representation of exploitability.
#[no_mangle]
pub extern "C" fn togo_exploitability_str(exploit: ToGoExploitability) -> *const c_char {
    match exploit {
        ToGoExploitability::SharedHardwareOnly => c"SharedHardwareOnly".as_ptr(),
        ToGoExploitability::Http2Multiplexing => c"Http2Multiplexing".as_ptr(),
        ToGoExploitability::StandardRemote => c"StandardRemote".as_ptr(),
        ToGoExploitability::ObviousLeak => c"ObviousLeak".as_ptr(),
    }
}

/// Get string representation of quality.
#[no_mangle]
pub extern "C" fn togo_quality_str(quality: ToGoQuality) -> *const c_char {
    match quality {
        ToGoQuality::Excellent => c"Excellent".as_ptr(),
        ToGoQuality::Good => c"Good".as_ptr(),
        ToGoQuality::Poor => c"Poor".as_ptr(),
        ToGoQuality::TooNoisy => c"TooNoisy".as_ptr(),
    }
}

/// Get string representation of inconclusive reason.
#[no_mangle]
pub extern "C" fn togo_inconclusive_reason_str(reason: ToGoInconclusiveReason) -> *const c_char {
    match reason {
        ToGoInconclusiveReason::None => c"None".as_ptr(),
        ToGoInconclusiveReason::DataTooNoisy => c"DataTooNoisy".as_ptr(),
        ToGoInconclusiveReason::NotLearning => c"NotLearning".as_ptr(),
        ToGoInconclusiveReason::WouldTakeTooLong => c"WouldTakeTooLong".as_ptr(),
        ToGoInconclusiveReason::TimeBudgetExceeded => c"TimeBudgetExceeded".as_ptr(),
        ToGoInconclusiveReason::SampleBudgetExceeded => c"SampleBudgetExceeded".as_ptr(),
        ToGoInconclusiveReason::ConditionsChanged => c"ConditionsChanged".as_ptr(),
        ToGoInconclusiveReason::ThresholdUnachievable => c"ThresholdUnachievable".as_ptr(),
    }
}

// =============================================================================
// Internal helpers
// =============================================================================

/// Fill result struct from adaptive outcome.
unsafe fn fill_result_from_outcome(
    outcome: &AdaptiveOutcome,
    config: &ToGoConfig,
    result: *mut ToGoResult,
) {
    match outcome {
        AdaptiveOutcome::LeakDetected {
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            (*result).outcome = ToGoOutcome::Fail;
            (*result).leak_probability = posterior.leak_probability;
            (*result).samples_used = *samples_per_class;
            (*result).elapsed_secs = *elapsed_secs;

            // Effect from posterior beta
            let shift_ns = posterior.beta_proj[0];
            let tail_ns = posterior.beta_proj[1];
            let total_effect = sqrt(shift_ns * shift_ns + tail_ns * tail_ns);

            (*result).effect.shift_ns = shift_ns;
            (*result).effect.tail_ns = tail_ns;
            (*result).effect.pattern = classify_effect_pattern(shift_ns, tail_ns);

            // CI from posterior covariance (simplified)
            let ci_width =
                1.96 * sqrt(posterior.beta_proj_cov[(0, 0)] + posterior.beta_proj_cov[(1, 1)]);
            (*result).effect.ci_low_ns = total_effect - ci_width;
            (*result).effect.ci_high_ns = total_effect + ci_width;

            (*result).exploitability = exploitability_from_effect_ns(total_effect);
            (*result).quality = quality_from_mde_ns(total_effect / 5.0); // Rough estimate
        }

        AdaptiveOutcome::NoLeakDetected {
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            (*result).outcome = ToGoOutcome::Pass;
            (*result).leak_probability = posterior.leak_probability;
            (*result).samples_used = *samples_per_class;
            (*result).elapsed_secs = *elapsed_secs;

            let shift_ns = posterior.beta_proj[0];
            let tail_ns = posterior.beta_proj[1];

            (*result).effect.shift_ns = shift_ns;
            (*result).effect.tail_ns = tail_ns;
            (*result).effect.pattern = classify_effect_pattern(shift_ns, tail_ns);

            let total_effect = sqrt(shift_ns * shift_ns + tail_ns * tail_ns);
            let ci_width =
                1.96 * sqrt(posterior.beta_proj_cov[(0, 0)] + posterior.beta_proj_cov[(1, 1)]);
            (*result).effect.ci_low_ns = total_effect - ci_width;
            (*result).effect.ci_high_ns = total_effect + ci_width;

            (*result).quality = ToGoQuality::Good; // Passed, so quality is at least good
        }

        AdaptiveOutcome::Inconclusive {
            reason,
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            (*result).outcome = ToGoOutcome::Inconclusive;
            (*result).samples_used = *samples_per_class;
            (*result).elapsed_secs = *elapsed_secs;

            if let Some(p) = posterior {
                (*result).leak_probability = p.leak_probability;
                (*result).effect.shift_ns = p.beta_proj[0];
                (*result).effect.tail_ns = p.beta_proj[1];
            }

            // Map inconclusive reason
            (*result).inconclusive_reason = match reason {
                InconclusiveReason::DataTooNoisy { .. } => ToGoInconclusiveReason::DataTooNoisy,
                InconclusiveReason::NotLearning { .. } => ToGoInconclusiveReason::NotLearning,
                InconclusiveReason::WouldTakeTooLong { .. } => {
                    ToGoInconclusiveReason::WouldTakeTooLong
                }
                InconclusiveReason::TimeBudgetExceeded { .. } => {
                    ToGoInconclusiveReason::TimeBudgetExceeded
                }
                InconclusiveReason::SampleBudgetExceeded { .. } => {
                    ToGoInconclusiveReason::SampleBudgetExceeded
                }
                InconclusiveReason::ConditionsChanged { .. } => {
                    ToGoInconclusiveReason::ConditionsChanged
                }
                InconclusiveReason::ThresholdUnachievable { .. } => {
                    ToGoInconclusiveReason::ThresholdUnachievable
                }
            };

            // Set recommendation string
            let guidance = match reason {
                InconclusiveReason::DataTooNoisy { guidance, .. } => guidance.clone(),
                InconclusiveReason::NotLearning { guidance, .. } => guidance.clone(),
                InconclusiveReason::WouldTakeTooLong { guidance, .. } => guidance.clone(),
                InconclusiveReason::TimeBudgetExceeded { .. } => {
                    String::from("Increase time budget or reduce threshold")
                }
                InconclusiveReason::SampleBudgetExceeded { .. } => {
                    String::from("Increase sample budget or reduce threshold")
                }
                InconclusiveReason::ConditionsChanged { guidance, .. } => guidance.clone(),
                InconclusiveReason::ThresholdUnachievable { guidance, .. } => guidance.clone(),
            };

            if let Ok(cstr) = CString::new(guidance) {
                (*result).recommendation = cstr.into_raw();
            }
        }
    }

    (*result).theta_user_ns = config.theta_ns();
}

/// Classify effect pattern from shift and tail components.
fn classify_effect_pattern(shift_ns: f64, tail_ns: f64) -> ToGoEffectPattern {
    let shift_abs = shift_ns.abs();
    let tail_abs = tail_ns.abs();

    if shift_abs < 1.0 && tail_abs < 1.0 {
        ToGoEffectPattern::Indeterminate
    } else if shift_abs > tail_abs * 2.0 {
        ToGoEffectPattern::UniformShift
    } else if tail_abs > shift_abs * 2.0 {
        ToGoEffectPattern::TailEffect
    } else {
        ToGoEffectPattern::Mixed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = togo_config_default(ToGoAttackerModel::AdjacentNetwork);
        assert!((config.theta_ns() - 100.0).abs() < 1e-10);
        assert!((config.pass_threshold - 0.05).abs() < 1e-10);
        assert!((config.fail_threshold - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_attacker_thresholds() {
        assert!((ToGoAttackerModel::SharedHardware.to_threshold_ns(0.0) - 0.6).abs() < 1e-10);
        assert!((ToGoAttackerModel::AdjacentNetwork.to_threshold_ns(0.0) - 100.0).abs() < 1e-10);
        assert!((ToGoAttackerModel::RemoteNetwork.to_threshold_ns(0.0) - 50_000.0).abs() < 1e-10);
        assert!((ToGoAttackerModel::Custom.to_threshold_ns(42.0) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_effect_pattern_classification() {
        assert_eq!(
            classify_effect_pattern(100.0, 10.0),
            ToGoEffectPattern::UniformShift
        );
        assert_eq!(
            classify_effect_pattern(10.0, 100.0),
            ToGoEffectPattern::TailEffect
        );
        assert_eq!(
            classify_effect_pattern(50.0, 50.0),
            ToGoEffectPattern::Mixed
        );
        assert_eq!(
            classify_effect_pattern(0.1, 0.1),
            ToGoEffectPattern::Indeterminate
        );
    }

    #[test]
    fn test_exploitability_thresholds() {
        // < 10 ns: SharedHardwareOnly
        assert_eq!(
            exploitability_from_effect_ns(5.0),
            ToGoExploitability::SharedHardwareOnly
        );
        // 10-100 ns: Http2Multiplexing
        assert_eq!(
            exploitability_from_effect_ns(50.0),
            ToGoExploitability::Http2Multiplexing
        );
        // 100 ns - 10 μs: StandardRemote
        assert_eq!(
            exploitability_from_effect_ns(1000.0),
            ToGoExploitability::StandardRemote
        );
        // > 10 μs: ObviousLeak
        assert_eq!(
            exploitability_from_effect_ns(50_000.0),
            ToGoExploitability::ObviousLeak
        );
    }
}
