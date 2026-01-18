//! Adaptive sampling module for timing-oracle.
//!
//! Implements a two-phase approach for efficient timing analysis:
//!
//! 1. **Calibration** (spec Section 2.3, Section 2.6): Collect initial samples to estimate covariance
//!    and set Bayesian priors. This establishes the "sigma rate" - covariance scaled by sample
//!    count - which allows efficient scaling as more samples are collected.
//!
//! 2. **Adaptive loop** (spec Section 2.5): Collect batches until decision thresholds are reached:
//!    - If P(leak > theta) > fail_threshold: Fail (timing leak detected)
//!    - If P(leak > theta) < pass_threshold: Pass (no significant leak)
//!    - If quality gates trigger: Inconclusive (measurement issues)
//!
//! ## Key Design Decisions
//!
//! - **Sigma rate scaling**: Instead of recomputing covariance for each batch, we estimate
//!   Sigma_rate = Sigma_cal * n_cal from calibration, then scale as Sigma_n = Sigma_rate / n.
//!   This assumes stationarity and avoids expensive bootstrap on each iteration.
//!
//! - **KL divergence tracking**: We track KL(posterior_new || posterior_old) to detect when
//!   learning has stalled. If recent KL divergences sum to < 0.001, data isn't informative.
//!
//! - **Quality gates**: Multiple stopping conditions prevent wasted computation:
//!   - Posterior too close to prior (variance ratio > 0.5)
//!   - Learning rate collapsed (KL sum < 0.001 over 5 batches)
//!   - Extrapolated time exceeds 10x budget
//!   - Time budget exceeded
//!   - Sample budget exceeded
//!   - Condition drift detected (calibration assumptions violated)

mod calibration;
mod loop_runner;
pub mod single_pass;
mod state;

// Re-export from timing-oracle-core
pub use timing_oracle_core::adaptive::{
    kl_divergence_gaussian, CalibrationSnapshot, ConditionDrift, DriftThresholds,
    InconclusiveReason, Posterior, QualityGateCheckInputs, QualityGateConfig, QualityGateResult,
};

// Local exports
pub use calibration::{calibrate, Calibration, CalibrationConfig, CalibrationError};
pub use loop_runner::{run_adaptive, AdaptiveConfig, AdaptiveOutcome};
pub use single_pass::{analyze_single_pass, SinglePassConfig, SinglePassResult};
pub use state::AdaptiveState;
