//! Analysis API for WebAssembly.
//!
//! Exports calibration and analysis functions. The measurement loop
//! is implemented in TypeScript for zero FFI overhead during operation execution.

use std::sync::Mutex;

use js_sys::BigInt64Array;
use wasm_bindgen::prelude::*;

use tacet_core::adaptive::{
    adaptive_step, calibrate, AdaptiveOutcome, AdaptiveState as CoreAdaptiveState,
    AdaptiveStepConfig, Calibration as CoreCalibration, CalibrationConfig,
    InconclusiveReason as CoreInconclusiveReason, StepResult,
};

use crate::types::*;

/// Handle counter for calibration state.
static NEXT_CALIBRATION_ID: Mutex<u32> = Mutex::new(0);

/// Handle counter for adaptive state.
static NEXT_ADAPTIVE_ID: Mutex<u32> = Mutex::new(0);

/// Storage for calibration handles.
static CALIBRATIONS: Mutex<Vec<Option<(CoreCalibration, f64)>>> = Mutex::new(Vec::new());

/// Storage for adaptive state handles.
static ADAPTIVE_STATES: Mutex<Vec<Option<CoreAdaptiveState>>> = Mutex::new(Vec::new());

/// Opaque calibration handle returned to JS.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    id: u32,
}

#[wasm_bindgen]
impl Calibration {
    /// Get the handle ID (for debugging).
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> u32 {
        self.id
    }
}

/// Opaque adaptive state handle returned to JS.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveState {
    id: u32,
}

#[wasm_bindgen]
impl AdaptiveState {
    /// Create a new adaptive state.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut id_counter = NEXT_ADAPTIVE_ID.lock().unwrap();
        let id = *id_counter;
        *id_counter += 1;

        let mut states = ADAPTIVE_STATES.lock().unwrap();
        if id as usize >= states.len() {
            states.resize_with(id as usize + 1, || None);
        }
        states[id as usize] = Some(CoreAdaptiveState::new());

        Self { id }
    }

    /// Get total baseline samples collected.
    #[wasm_bindgen(getter, js_name = totalBaseline)]
    pub fn total_baseline(&self) -> u32 {
        let states = ADAPTIVE_STATES.lock().unwrap();
        states
            .get(self.id as usize)
            .and_then(|s| s.as_ref())
            .map(|s| s.baseline_samples.len() as u32)
            .unwrap_or(0)
    }

    /// Get total sample class samples collected.
    #[wasm_bindgen(getter, js_name = totalSample)]
    pub fn total_sample(&self) -> u32 {
        let states = ADAPTIVE_STATES.lock().unwrap();
        states
            .get(self.id as usize)
            .and_then(|s| s.as_ref())
            .map(|s| s.sample_samples.len() as u32)
            .unwrap_or(0)
    }

    /// Get current leak probability estimate.
    #[wasm_bindgen(getter, js_name = currentProbability)]
    pub fn current_probability(&self) -> f64 {
        let states = ADAPTIVE_STATES.lock().unwrap();
        states
            .get(self.id as usize)
            .and_then(|s| s.as_ref())
            .and_then(|s| s.current_posterior())
            .map(|p| p.leak_probability)
            .unwrap_or(0.5)
    }

    /// Get the number of batches collected so far.
    #[wasm_bindgen(getter, js_name = batchCount)]
    pub fn batch_count(&self) -> u32 {
        let states = ADAPTIVE_STATES.lock().unwrap();
        states
            .get(self.id as usize)
            .and_then(|s| s.as_ref())
            .map(|s| s.batch_count as u32)
            .unwrap_or(0)
    }

    /// Free the adaptive state resources.
    pub fn free(&self) {
        let mut states = ADAPTIVE_STATES.lock().unwrap();
        if let Some(slot) = states.get_mut(self.id as usize) {
            *slot = None;
        }
    }
}

impl Default for AdaptiveState {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize the WASM module (call once at startup).
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get the library version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Create a default configuration for a given attacker model.
#[wasm_bindgen(js_name = defaultConfig)]
pub fn default_config(attacker_model: AttackerModel) -> Config {
    Config {
        attacker_model,
        ..Config::default()
    }
}

/// Create a configuration for adjacent network attacker (100ns threshold).
#[wasm_bindgen(js_name = configAdjacentNetwork)]
pub fn config_adjacent_network() -> Config {
    default_config(AttackerModel::AdjacentNetwork)
}

/// Create a configuration for shared hardware attacker (0.6ns threshold).
#[wasm_bindgen(js_name = configSharedHardware)]
pub fn config_shared_hardware() -> Config {
    default_config(AttackerModel::SharedHardware)
}

/// Create a configuration for remote network attacker (50us threshold).
#[wasm_bindgen(js_name = configRemoteNetwork)]
pub fn config_remote_network() -> Config {
    default_config(AttackerModel::RemoteNetwork)
}

/// Calibrate timing samples.
///
/// # Arguments
/// - `baseline` - Baseline timing samples (raw ticks as BigInt64Array)
/// - `sample` - Sample class timing samples (raw ticks as BigInt64Array)
/// - `config` - Analysis configuration
/// - `timer_frequency_hz` - Timer frequency in Hz
#[wasm_bindgen(js_name = calibrateSamples)]
pub fn calibrate_samples(
    baseline: &BigInt64Array,
    sample: &BigInt64Array,
    config: Config,
    timer_frequency_hz: f64,
) -> Result<Calibration, JsError> {
    let baseline: Vec<u64> = baseline.to_vec().into_iter().map(|x| x as u64).collect();
    let sample: Vec<u64> = sample.to_vec().into_iter().map(|x| x as u64).collect();

    if baseline.len() < 100 || sample.len() < 100 {
        return Err(JsError::new(&format!(
            "Insufficient samples: need at least 100, got {} baseline and {} sample",
            baseline.len(),
            sample.len()
        )));
    }

    let theta_ns = config.theta_ns();
    let ns_per_tick = 1_000_000_000.0 / timer_frequency_hz;
    let seed = config
        .seed
        .unwrap_or_else(|| (baseline.iter().take(10).sum::<u64>() ^ 0x12345678) as u32)
        as u64;

    let n = baseline.len().min(sample.len());

    let cal_config = CalibrationConfig {
        calibration_samples: n,
        bootstrap_iterations: 200,
        timer_resolution_ns: ns_per_tick,
        theta_ns,
        alpha: 0.01,
        seed,
        skip_preflight: true,
        force_discrete_mode: false,
    };

    // Use a placeholder for samples_per_second since we can't use std::time::Instant in WASM.
    // This is only used for time estimation, not for statistical analysis.
    // TypeScript can update this value if needed via the calibration handle.
    let samples_per_second = 100_000.0; // Conservative estimate

    match calibrate(
        &baseline,
        &sample,
        ns_per_tick,
        &cal_config,
        samples_per_second,
    ) {
        Ok(core_cal) => {
            // Store calibration and get handle
            let mut id_counter = NEXT_CALIBRATION_ID.lock().unwrap();
            let id = *id_counter;
            *id_counter += 1;

            let mut calibrations = CALIBRATIONS.lock().unwrap();
            if id as usize >= calibrations.len() {
                calibrations.resize_with(id as usize + 1, || None);
            }
            calibrations[id as usize] = Some((core_cal, ns_per_tick));

            Ok(Calibration { id })
        }
        Err(e) => Err(JsError::new(&format!("Calibration failed: {}", e))),
    }
}

/// Free calibration resources.
#[wasm_bindgen(js_name = freeCalibration)]
pub fn free_calibration(calibration: &Calibration) {
    let mut calibrations = CALIBRATIONS.lock().unwrap();
    if let Some(slot) = calibrations.get_mut(calibration.id as usize) {
        *slot = None;
    }
}

/// Run one adaptive step with a new batch of samples.
#[wasm_bindgen(js_name = adaptiveStepBatch)]
pub fn adaptive_step_batch(
    calibration: &Calibration,
    state: &AdaptiveState,
    baseline: &BigInt64Array,
    sample: &BigInt64Array,
    config: Config,
    elapsed_secs: f64,
) -> Result<AdaptiveStepResult, JsError> {
    let baseline: Vec<u64> = baseline.to_vec().into_iter().map(|x| x as u64).collect();
    let sample: Vec<u64> = sample.to_vec().into_iter().map(|x| x as u64).collect();

    // Get calibration data
    let calibrations = CALIBRATIONS.lock().unwrap();
    let (core_cal, ns_per_tick) = calibrations
        .get(calibration.id as usize)
        .and_then(|c| c.as_ref())
        .ok_or_else(|| JsError::new("Invalid calibration handle"))?;
    let core_cal = core_cal.clone();
    let ns_per_tick = *ns_per_tick;
    drop(calibrations);

    // Get and update adaptive state
    let mut states = ADAPTIVE_STATES.lock().unwrap();
    let inner_state = states
        .get_mut(state.id as usize)
        .and_then(|s| s.as_mut())
        .ok_or_else(|| JsError::new("Invalid adaptive state handle"))?;

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
        &core_cal,
        inner_state,
        ns_per_tick,
        elapsed_secs,
        &step_config,
    );

    match step {
        StepResult::Continue {
            posterior,
            samples_per_class,
        } => Ok(AdaptiveStepResult {
            is_decision: false,
            current_probability: posterior.leak_probability,
            samples_per_class: samples_per_class as u32,
            result: None,
        }),
        StepResult::Decision(outcome) => {
            let result = build_result_from_outcome(&outcome, &config, Some(&core_cal));
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
#[wasm_bindgen]
pub fn analyze(
    baseline: &BigInt64Array,
    sample: &BigInt64Array,
    config: Config,
    timer_frequency_hz: f64,
) -> Result<AnalysisResult, JsError> {
    let baseline: Vec<u64> = baseline.to_vec().into_iter().map(|x| x as u64).collect();
    let sample: Vec<u64> = sample.to_vec().into_iter().map(|x| x as u64).collect();

    if baseline.len() < 100 || sample.len() < 100 {
        return Err(JsError::new(&format!(
            "Insufficient samples: need at least 100, got {} baseline and {} sample",
            baseline.len(),
            sample.len()
        )));
    }

    let theta_ns = config.theta_ns();
    let ns_per_tick = 1_000_000_000.0 / timer_frequency_hz;
    let seed = config
        .seed
        .unwrap_or_else(|| (baseline.iter().take(10).sum::<u64>() ^ 0x12345678) as u32)
        as u64;

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

    // Use a placeholder for samples_per_second since we can't use std::time::Instant in WASM.
    let samples_per_second = 100_000.0;

    // Run calibration - returns CoreCalibration directly
    let core_cal = calibrate(
        &baseline[..cal_samples],
        &sample[..cal_samples],
        ns_per_tick,
        &cal_config,
        samples_per_second,
    )
    .map_err(|e| JsError::new(&format!("Calibration failed: {}", e)))?;

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

        let step = adaptive_step(
            &core_cal,
            &mut state,
            ns_per_tick,
            elapsed_secs,
            &step_config,
        );

        if let StepResult::Decision(outcome) = step {
            let mut result = build_result_from_outcome(&outcome, &config, Some(&core_cal));
            result.mde_shift_ns = core_cal.mde_shift_ns;
            result.mde_tail_ns = core_cal.mde_tail_ns;
            result.theta_user_ns = theta_ns;
            result.theta_eff_ns = core_cal.theta_eff;
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
    result.mde_shift_ns = core_cal.mde_shift_ns;
    result.mde_tail_ns = core_cal.mde_tail_ns;
    result.theta_user_ns = theta_ns;
    result.theta_eff_ns = core_cal.theta_eff;
    Ok(result)
}

/// Helper function: Convert AdaptiveOutcome to AnalysisResult using FFI summary types.
fn build_result_from_outcome(
    outcome: &AdaptiveOutcome,
    _config: &Config,
    calibration: Option<&CoreCalibration>,
) -> AnalysisResult {
    use tacet_core::ffi_summary::{InconclusiveReasonKind, OutcomeType};

    // Get the calibration - required for summary conversion
    let cal = calibration.expect("Calibration required for result conversion");
    let summary = outcome.to_summary(cal);

    // Map OutcomeType to WASM Outcome
    let outcome_enum = match summary.outcome_type {
        OutcomeType::Pass => Outcome::Pass,
        OutcomeType::Fail => Outcome::Fail,
        OutcomeType::Inconclusive | OutcomeType::ThresholdElevated => Outcome::Inconclusive,
    };

    // Map InconclusiveReasonKind to WASM InconclusiveReason
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
