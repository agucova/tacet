//! Analysis API for JavaScript/TypeScript.
//!
//! Exports calibration and analysis functions. The measurement loop
//! is implemented in TypeScript for zero FFI overhead during operation execution.

use std::sync::RwLock;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use timing_oracle::adaptive::{calibrate, CalibrationConfig};
use timing_oracle_core::adaptive::{
    adaptive_step, AdaptiveOutcome, AdaptiveState as CoreAdaptiveState,
    AdaptiveStepConfig, Calibration as CoreCalibration, InconclusiveReason as CoreInconclusiveReason,
    StepResult,
};
use timing_oracle_core::math::sqrt;

use crate::types::*;

/// Opaque calibration state handle.
#[napi]
pub struct Calibration {
    inner: CoreCalibration,
    ns_per_tick: f64,
}

/// Adaptive sampling state.
#[napi]
pub struct AdaptiveState {
    inner: RwLock<CoreAdaptiveState>,
}

#[napi]
impl AdaptiveState {
    /// Create a new adaptive state.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(CoreAdaptiveState::new()),
        }
    }

    /// Get total baseline samples collected.
    #[napi(getter)]
    pub fn total_baseline(&self) -> u32 {
        self.inner.read().unwrap().baseline_samples.len() as u32
    }

    /// Get total sample class samples collected.
    #[napi(getter)]
    pub fn total_sample(&self) -> u32 {
        self.inner.read().unwrap().sample_samples.len() as u32
    }

    /// Get current leak probability estimate.
    #[napi(getter)]
    pub fn current_probability(&self) -> f64 {
        self.inner
            .read()
            .unwrap()
            .current_posterior()
            .map(|p| p.leak_probability)
            .unwrap_or(0.5)
    }

    /// Get the number of batches collected so far.
    #[napi(getter)]
    pub fn batch_count(&self) -> u32 {
        self.inner.read().unwrap().batch_count as u32
    }
}

impl Default for AdaptiveState {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of an adaptive step.
#[napi(object)]
pub struct AdaptiveStepResult {
    /// Whether a decision was reached.
    pub is_decision: bool,
    /// Current leak probability estimate.
    pub current_probability: f64,
    /// Samples collected per class so far.
    pub samples_per_class: u32,
    /// The final result (only valid if is_decision is true).
    pub result: Option<AnalysisResult>,
}

/// Calibrate timing samples.
///
/// # Arguments
/// - `baseline` - Baseline timing samples (raw ticks)
/// - `sample` - Sample class timing samples (raw ticks)
/// - `config` - Analysis configuration
/// - `timer_frequency_hz` - Timer frequency in Hz
#[napi]
pub fn calibrate_samples(
    baseline: BigInt64Array,
    sample: BigInt64Array,
    config: Config,
    timer_frequency_hz: f64,
) -> Result<Calibration> {
    let baseline: Vec<u64> = baseline.iter().map(|x| *x as u64).collect();
    let sample: Vec<u64> = sample.iter().map(|x| *x as u64).collect();

    if baseline.len() < 100 || sample.len() < 100 {
        return Err(Error::from_reason(format!(
            "Insufficient samples: need at least 100, got {} baseline and {} sample",
            baseline.len(),
            sample.len()
        )));
    }

    let theta_ns = config.theta_ns();
    let ns_per_tick = 1_000_000_000.0 / timer_frequency_hz;
    let seed = config.seed.unwrap_or_else(|| {
        (baseline.iter().take(10).sum::<u64>() ^ 0x12345678) as u32
    }) as u64;

    let cal_config = CalibrationConfig {
        calibration_samples: baseline.len().min(sample.len()),
        bootstrap_iterations: 200,
        timer_resolution_ns: ns_per_tick,
        theta_ns,
        alpha: 0.01,
        seed,
        skip_preflight: true,
        force_discrete_mode: false,
    };

    match calibrate(&baseline, &sample, ns_per_tick, &cal_config) {
        Ok(cal) => {
            let core_cal = CoreCalibration::new(
                cal.sigma_rate,
                cal.block_length,
                cal.sigma_t,
                cal.l_r,
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
                cal.batch_k,
            );
            Ok(Calibration {
                inner: core_cal,
                ns_per_tick,
            })
        }
        Err(e) => Err(Error::from_reason(format!("Calibration failed: {:?}", e))),
    }
}

/// Run one adaptive step with a new batch of samples.
#[napi]
pub fn adaptive_step_batch(
    calibration: &Calibration,
    state: &AdaptiveState,
    baseline: BigInt64Array,
    sample: BigInt64Array,
    config: Config,
    elapsed_secs: f64,
) -> Result<AdaptiveStepResult> {
    let baseline: Vec<u64> = baseline.iter().map(|x| *x as u64).collect();
    let sample: Vec<u64> = sample.iter().map(|x| *x as u64).collect();

    let mut inner_state = state.inner.write().map_err(|_| {
        Error::from_reason("Failed to acquire state lock")
    })?;

    // Add batch to state
    inner_state.add_batch(baseline, sample);

    // Create step config
    let step_config = AdaptiveStepConfig {
        pass_threshold: config.pass_threshold,
        fail_threshold: config.fail_threshold,
        time_budget_secs: config.time_budget_secs(),
        max_samples: config.max_samples as usize,
        theta_ns: config.theta_ns(),
        seed: config.seed.unwrap_or(0) as u64,
        ..AdaptiveStepConfig::default()
    };

    // Run adaptive step
    let step = adaptive_step(
        &calibration.inner,
        &mut inner_state,
        calibration.ns_per_tick,
        elapsed_secs,
        &step_config,
    );

    match step {
        StepResult::Continue { posterior, samples_per_class } => {
            Ok(AdaptiveStepResult {
                is_decision: false,
                current_probability: posterior.leak_probability,
                samples_per_class: samples_per_class as u32,
                result: None,
            })
        }
        StepResult::Decision(outcome) => {
            let result = build_result_from_outcome(&outcome, &config, Some(&calibration.inner));
            Ok(AdaptiveStepResult {
                is_decision: true,
                current_probability: result.leak_probability,
                samples_per_class: result.samples_used,
                result: Some(result),
            })
        }
    }
}

/// Run complete analysis on pre-collected timing data.
#[napi]
pub fn analyze(
    baseline: BigInt64Array,
    sample: BigInt64Array,
    config: Config,
    timer_frequency_hz: f64,
) -> Result<AnalysisResult> {
    let baseline: Vec<u64> = baseline.iter().map(|x| *x as u64).collect();
    let sample: Vec<u64> = sample.iter().map(|x| *x as u64).collect();

    if baseline.len() < 100 || sample.len() < 100 {
        return Err(Error::from_reason(format!(
            "Insufficient samples: need at least 100, got {} baseline and {} sample",
            baseline.len(),
            sample.len()
        )));
    }

    let theta_ns = config.theta_ns();
    let ns_per_tick = 1_000_000_000.0 / timer_frequency_hz;
    let seed = config.seed.unwrap_or_else(|| {
        (baseline.iter().take(10).sum::<u64>() ^ 0x12345678) as u32
    }) as u64;

    // Split into calibration and adaptive samples
    let cal_samples = 5000.min(baseline.len() / 2);

    let cal_config = CalibrationConfig {
        calibration_samples: cal_samples,
        bootstrap_iterations: 200,
        timer_resolution_ns: ns_per_tick,
        theta_ns,
        alpha: 0.01,
        seed,
        skip_preflight: true,
        force_discrete_mode: false,
    };

    // Run calibration
    let cal = match calibrate(&baseline[..cal_samples], &sample[..cal_samples], ns_per_tick, &cal_config) {
        Ok(c) => c,
        Err(e) => {
            return Err(Error::from_reason(format!("Calibration failed: {:?}", e)));
        }
    };

    // Convert to core Calibration
    let core_cal = CoreCalibration::new(
        cal.sigma_rate,
        cal.block_length,
        cal.sigma_t,
        cal.l_r,
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
        cal.batch_k,
    );

    // Run adaptive loop
    let mut state = CoreAdaptiveState::new();
    let step_config = AdaptiveStepConfig {
        pass_threshold: config.pass_threshold,
        fail_threshold: config.fail_threshold,
        time_budget_secs: config.time_budget_secs(),
        max_samples: config.max_samples as usize,
        theta_ns,
        seed,
        ..AdaptiveStepConfig::default()
    };

    let batch_size = 1000;
    let mut elapsed_secs = 0.0;
    let time_per_batch = 0.01;

    for (b_chunk, s_chunk) in baseline.chunks(batch_size).zip(sample.chunks(batch_size)) {
        state.add_batch(b_chunk.to_vec(), s_chunk.to_vec());
        elapsed_secs += time_per_batch;

        let step = adaptive_step(&core_cal, &mut state, ns_per_tick, elapsed_secs, &step_config);

        if let StepResult::Decision(outcome) = step {
            let mut result = build_result_from_outcome(&outcome, &config, Some(&core_cal));
            result.mde_shift_ns = cal.mde_shift_ns;
            result.mde_tail_ns = cal.mde_tail_ns;
            result.theta_user_ns = theta_ns;
            result.theta_eff_ns = cal.theta_eff;
            return Ok(result);
        }
    }

    // Exhausted samples without decision
    let outcome = AdaptiveOutcome::Inconclusive {
        reason: CoreInconclusiveReason::SampleBudgetExceeded {
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

    let mut result = build_result_from_outcome(&outcome, &config, Some(&core_cal));
    result.mde_shift_ns = cal.mde_shift_ns;
    result.mde_tail_ns = cal.mde_tail_ns;
    result.theta_user_ns = theta_ns;
    result.theta_eff_ns = cal.theta_eff;
    Ok(result)
}

// Helper functions

fn classify_effect_pattern(shift_ns: f64, tail_ns: f64) -> EffectPattern {
    let shift_abs = shift_ns.abs();
    let tail_abs = tail_ns.abs();

    if shift_abs < 1.0 && tail_abs < 1.0 {
        EffectPattern::Indeterminate
    } else if shift_abs > tail_abs * 2.0 {
        EffectPattern::UniformShift
    } else if tail_abs > shift_abs * 2.0 {
        EffectPattern::TailEffect
    } else {
        EffectPattern::Mixed
    }
}

fn build_diagnostics_from_posterior(
    posterior: Option<&timing_oracle_core::adaptive::Posterior>,
    calibration: Option<&CoreCalibration>,
) -> Diagnostics {
    let mut diag = Diagnostics::default();

    if let Some(p) = posterior {
        diag.effective_sample_size = p.n as u32;
        diag.projection_mismatch_q = p.projection_mismatch_q;
        diag.projection_mismatch_ok = calibration
            .map(|c| p.projection_mismatch_q <= c.projection_mismatch_thresh)
            .unwrap_or(true);

        diag.lambda_mean = p.lambda_mean.unwrap_or(1.0);
        diag.lambda_mixing_ok = p.lambda_mixing_ok.unwrap_or(true);
    }

    if let Some(cal) = calibration {
        diag.dependence_length = cal.block_length as u32;
        diag.discrete_mode = cal.discrete_mode;
        diag.timer_resolution_ns = cal.timer_resolution_ns;
    }

    diag
}

fn build_result_from_outcome(
    outcome: &AdaptiveOutcome,
    config: &Config,
    calibration: Option<&CoreCalibration>,
) -> AnalysisResult {
    match outcome {
        AdaptiveOutcome::LeakDetected {
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            let shift_ns = posterior.beta_proj[0];
            let tail_ns = posterior.beta_proj[1];
            let total_effect = sqrt(shift_ns * shift_ns + tail_ns * tail_ns);
            let ci_width = 1.96 * sqrt(posterior.beta_proj_cov[(0, 0)] + posterior.beta_proj_cov[(1, 1)]);

            AnalysisResult {
                outcome: Outcome::Fail,
                leak_probability: posterior.leak_probability,
                effect: EffectEstimate {
                    shift_ns,
                    tail_ns,
                    credible_interval_low: total_effect - ci_width,
                    credible_interval_high: total_effect + ci_width,
                    pattern: classify_effect_pattern(shift_ns, tail_ns),
                    interpretation_caveat: None,
                },
                quality: MeasurementQuality::Good,
                samples_used: *samples_per_class as u32,
                elapsed_secs: *elapsed_secs,
                exploitability: Exploitability::from_effect_ns(total_effect),
                inconclusive_reason: InconclusiveReason::None,
                mde_shift_ns: 0.0,
                mde_tail_ns: 0.0,
                timer_resolution_ns: calibration.map(|c| c.timer_resolution_ns).unwrap_or(0.0),
                theta_user_ns: config.theta_ns(),
                theta_eff_ns: calibration.map(|c| c.theta_eff).unwrap_or(config.theta_ns()),
                recommendation: String::new(),
                diagnostics: build_diagnostics_from_posterior(Some(posterior), calibration),
            }
        }

        AdaptiveOutcome::NoLeakDetected {
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            let shift_ns = posterior.beta_proj[0];
            let tail_ns = posterior.beta_proj[1];
            let total_effect = sqrt(shift_ns * shift_ns + tail_ns * tail_ns);
            let ci_width = 1.96 * sqrt(posterior.beta_proj_cov[(0, 0)] + posterior.beta_proj_cov[(1, 1)]);

            AnalysisResult {
                outcome: Outcome::Pass,
                leak_probability: posterior.leak_probability,
                effect: EffectEstimate {
                    shift_ns,
                    tail_ns,
                    credible_interval_low: total_effect - ci_width,
                    credible_interval_high: total_effect + ci_width,
                    pattern: classify_effect_pattern(shift_ns, tail_ns),
                    interpretation_caveat: None,
                },
                quality: MeasurementQuality::Good,
                samples_used: *samples_per_class as u32,
                elapsed_secs: *elapsed_secs,
                exploitability: Exploitability::SharedHardwareOnly,
                inconclusive_reason: InconclusiveReason::None,
                mde_shift_ns: 0.0,
                mde_tail_ns: 0.0,
                timer_resolution_ns: calibration.map(|c| c.timer_resolution_ns).unwrap_or(0.0),
                theta_user_ns: config.theta_ns(),
                theta_eff_ns: calibration.map(|c| c.theta_eff).unwrap_or(config.theta_ns()),
                recommendation: String::new(),
                diagnostics: build_diagnostics_from_posterior(Some(posterior), calibration),
            }
        }

        AdaptiveOutcome::ThresholdElevated {
            posterior,
            theta_user,
            theta_eff,
            achievable_at_max,
            samples_per_class,
            elapsed_secs,
            ..
        } => {
            let shift_ns = posterior.beta_proj[0];
            let tail_ns = posterior.beta_proj[1];
            let total_effect = sqrt(shift_ns * shift_ns + tail_ns * tail_ns);
            let ci_width = 1.96 * sqrt(posterior.beta_proj_cov[(0, 0)] + posterior.beta_proj_cov[(1, 1)]);

            let recommendation = if *achievable_at_max {
                format!(
                    "Threshold elevated from {:.0}ns to {:.1}ns. More samples could achieve the requested threshold.",
                    theta_user, theta_eff
                )
            } else {
                format!(
                    "Threshold elevated from {:.0}ns to {:.1}ns. Use a cycle counter for better resolution.",
                    theta_user, theta_eff
                )
            };

            AnalysisResult {
                outcome: Outcome::Inconclusive,
                leak_probability: posterior.leak_probability,
                effect: EffectEstimate {
                    shift_ns,
                    tail_ns,
                    credible_interval_low: total_effect - ci_width,
                    credible_interval_high: total_effect + ci_width,
                    pattern: classify_effect_pattern(shift_ns, tail_ns),
                    interpretation_caveat: None,
                },
                quality: MeasurementQuality::Good,
                samples_used: *samples_per_class as u32,
                elapsed_secs: *elapsed_secs,
                exploitability: Exploitability::SharedHardwareOnly,
                inconclusive_reason: InconclusiveReason::ThresholdElevated,
                mde_shift_ns: 0.0,
                mde_tail_ns: 0.0,
                timer_resolution_ns: calibration.map(|c| c.timer_resolution_ns).unwrap_or(0.0),
                theta_user_ns: *theta_user,
                theta_eff_ns: *theta_eff,
                recommendation,
                diagnostics: build_diagnostics_from_posterior(Some(posterior), calibration),
            }
        }

        AdaptiveOutcome::Inconclusive {
            reason,
            posterior,
            samples_per_class,
            elapsed_secs,
        } => {
            let (shift_ns, tail_ns) = posterior
                .as_ref()
                .map(|p| (p.beta_proj[0], p.beta_proj[1]))
                .unwrap_or((0.0, 0.0));

            let (inconclusive_reason, recommendation) = convert_inconclusive_reason(reason);

            AnalysisResult {
                outcome: Outcome::Inconclusive,
                leak_probability: posterior.as_ref().map(|p| p.leak_probability).unwrap_or(0.5),
                effect: EffectEstimate {
                    shift_ns,
                    tail_ns,
                    credible_interval_low: 0.0,
                    credible_interval_high: 0.0,
                    pattern: classify_effect_pattern(shift_ns, tail_ns),
                    interpretation_caveat: None,
                },
                quality: MeasurementQuality::Poor,
                samples_used: *samples_per_class as u32,
                elapsed_secs: *elapsed_secs,
                exploitability: Exploitability::SharedHardwareOnly,
                inconclusive_reason,
                mde_shift_ns: 0.0,
                mde_tail_ns: 0.0,
                timer_resolution_ns: calibration.map(|c| c.timer_resolution_ns).unwrap_or(0.0),
                theta_user_ns: config.theta_ns(),
                theta_eff_ns: calibration.map(|c| c.theta_eff).unwrap_or(config.theta_ns()),
                recommendation,
                diagnostics: build_diagnostics_from_posterior(posterior.as_ref(), calibration),
            }
        }
    }
}

fn convert_inconclusive_reason(reason: &CoreInconclusiveReason) -> (InconclusiveReason, String) {
    match reason {
        CoreInconclusiveReason::DataTooNoisy { guidance, .. } => {
            (InconclusiveReason::DataTooNoisy, guidance.clone())
        }
        CoreInconclusiveReason::NotLearning { guidance, .. } => {
            (InconclusiveReason::NotLearning, guidance.clone())
        }
        CoreInconclusiveReason::WouldTakeTooLong { guidance, .. } => {
            (InconclusiveReason::WouldTakeTooLong, guidance.clone())
        }
        CoreInconclusiveReason::TimeBudgetExceeded { .. } => {
            (InconclusiveReason::TimeBudgetExceeded, "Increase time budget or reduce threshold".to_string())
        }
        CoreInconclusiveReason::SampleBudgetExceeded { .. } => {
            (InconclusiveReason::SampleBudgetExceeded, "Increase sample budget or reduce threshold".to_string())
        }
        CoreInconclusiveReason::ConditionsChanged { guidance, .. } => {
            (InconclusiveReason::ConditionsChanged, guidance.clone())
        }
        CoreInconclusiveReason::ThresholdElevated { guidance, .. } => {
            (InconclusiveReason::ThresholdElevated, guidance.clone())
        }
    }
}
