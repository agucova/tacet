//! Adaptive sampling logic for timing-oracle (no_std compatible).
//!
//! This module provides the core statistical machinery for adaptive sampling,
//! designed to work in no_std environments (embedded, WASM, SGX) with only
//! an allocator.
//!
//! The orchestration layer (time tracking, sample collection) lives in the
//! `timing-oracle` crate; this module provides stateless functions that take
//! samples and return statistical results.
//!
//! # Key Components
//!
//! - **AdaptiveState**: Sample storage and posterior tracking (no time tracking)
//! - **Posterior**: Bayesian posterior distribution for effect vector β = (μ, τ)
//! - **Quality gates**: Decision logic for when to stop sampling
//! - **KL divergence**: Tracking learning rate during adaptive loop
//! - **Drift detection**: Checking if measurement conditions changed

mod calibration;
mod drift;
mod kl_divergence;
mod posterior;
mod quality_gates;
mod state;
mod step;

pub use calibration::{
    calibrate_t_prior_scale, compute_c_floor_9d, compute_prior_cov_9d, Calibration,
    CalibrationConfig, NU,
};
pub use drift::{CalibrationSnapshot, ConditionDrift, DriftThresholds};
pub use kl_divergence::kl_divergence_gaussian;
pub use posterior::Posterior;
pub use quality_gates::{
    check_quality_gates, compute_achievable_at_max, is_threshold_elevated, InconclusiveReason,
    QualityGateCheckInputs, QualityGateConfig, QualityGateResult,
};
pub use state::AdaptiveState;
pub use step::{adaptive_step, AdaptiveOutcome, AdaptiveStepConfig, StepResult};
