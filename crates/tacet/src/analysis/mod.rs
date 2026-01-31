//! Analysis module for timing leak detection.
//!
//! This module implements the adaptive Bayesian analysis pipeline:
//!
//! 1. **Bayesian Inference** ([`bayes`]): Posterior probability of timing leak with adaptive thresholds
//! 2. **Effect Estimation** ([`effect`]): Maximum effect and top quantile computation
//! 3. **MDE Estimation** ([`mde`]): Minimum detectable effect at current noise level
//! 4. **Diagnostics** ([`diagnostics`]): Reliability checks (stationarity, outlier asymmetry)

// Re-export analysis functions from core
pub use tacet_core::analysis::{
    analytical_mde, bayes, compute_bayes_gibbs, compute_effect_estimate,
    compute_max_effect_ci, effect, estimate_mde, mde, BayesResult,
    MaxEffectCI, MdeEstimate,
};

// Keep diagnostics locally (depends on main crate types)
mod diagnostics;
pub use diagnostics::{compute_diagnostics, DiagnosticsExtra};

// Stationarity tracking for drift detection
mod stationarity;
pub use stationarity::{StationarityResult, StationarityTracker};
