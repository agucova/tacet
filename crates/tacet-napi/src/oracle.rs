//! Analysis API for JavaScript/TypeScript.
//!
//! Exports calibration and analysis functions. The measurement loop
//! is implemented in TypeScript for zero FFI overhead during operation execution.

use std::sync::RwLock;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use tacet::adaptive::{calibrate, CalibrationConfig};
use tacet_core::adaptive::{
    adaptive_step, AdaptiveOutcome, AdaptiveState as CoreAdaptiveState, AdaptiveStepConfig,
    Calibration as CoreCalibration, InconclusiveReason as CoreInconclusiveReason, StepResult,
};

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

// Helper function: Convert AdaptiveOutcome to AnalysisResult using FFI summary types.
// This uses the canonical effect pattern classification from core (spec §3.4.6).
fn build_result_from_outcome(
    outcome: &AdaptiveOutcome,
    _config: &Config,
    calibration: Option<&CoreCalibration>,
) -> AnalysisResult {
    use tacet_core::ffi_summary::{InconclusiveReasonKind, OutcomeType};

    // Get the calibration - required for summary conversion
    let cal = calibration.expect("Calibration required for result conversion");
    let summary = outcome.to_summary(cal);

    // Map OutcomeType to NAPI Outcome
    let outcome_enum = match summary.outcome_type {
        OutcomeType::Pass => Outcome::Pass,
        OutcomeType::Fail => Outcome::Fail,
        OutcomeType::Inconclusive | OutcomeType::ThresholdElevated => Outcome::Inconclusive,
    };

    // Map InconclusiveReasonKind to NAPI InconclusiveReason
    let inconclusive_reason = match summary.inconclusive_reason {
        InconclusiveReasonKind::None => InconclusiveReason::None,
        InconclusiveReasonKind::DataTooNoisy => InconclusiveReason::DataTooNoisy,
        InconclusiveReasonKind::NotLearning => InconclusiveReason::NotLearning,
        InconclusiveReasonKind::WouldTakeTooLong => InconclusiveReason::WouldTakeTooLong,
        InconclusiveReasonKind::TimeBudgetExceeded => InconclusiveReason::TimeBudgetExceeded,
        InconclusiveReasonKind::SampleBudgetExceeded => InconclusiveReason::SampleBudgetExceeded,
        InconclusiveReasonKind::ConditionsChanged => InconclusiveReason::ConditionsChanged,
        InconclusiveReasonKind::ThresholdElevated => InconclusiveReason::ThresholdElevated,
    };

    AnalysisResult {
        outcome: outcome_enum,
        leak_probability: summary.leak_probability,
        effect: EffectEstimate {
            shift_ns: summary.effect.shift_ns,
            tail_ns: summary.effect.tail_ns,
            credible_interval_low: summary.effect.ci_low_ns,
            credible_interval_high: summary.effect.ci_high_ns,
            pattern: summary.effect.pattern.into(),
            interpretation_caveat: summary.effect.interpretation_caveat,
        },
        quality: summary.quality.into(),
        samples_used: summary.samples_per_class as u32,
        elapsed_secs: summary.elapsed_secs,
        exploitability: summary.exploitability.into(),
        inconclusive_reason,
        mde_shift_ns: summary.mde_shift_ns,
        mde_tail_ns: summary.mde_tail_ns,
        timer_resolution_ns: summary.diagnostics.timer_resolution_ns,
        theta_user_ns: summary.theta_user,
        theta_eff_ns: summary.theta_eff,
        recommendation: summary.recommendation,
        diagnostics: Diagnostics {
            dependence_length: summary.diagnostics.dependence_length as u32,
            effective_sample_size: summary.diagnostics.effective_sample_size as u32,
            stationarity_ratio: summary.diagnostics.stationarity_ratio,
            stationarity_ok: summary.diagnostics.stationarity_ok,
            projection_mismatch_q: summary.diagnostics.projection_mismatch_q,
            projection_mismatch_ok: summary.diagnostics.projection_mismatch_ok,
            discrete_mode: summary.diagnostics.discrete_mode,
            timer_resolution_ns: summary.diagnostics.timer_resolution_ns,
            lambda_mean: summary.diagnostics.lambda_mean,
            lambda_mixing_ok: summary.diagnostics.lambda_mixing_ok,
        },
    }
}
