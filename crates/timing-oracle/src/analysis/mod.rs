//! Analysis module for timing leak detection.
//!
//! This module implements the adaptive Bayesian analysis pipeline:
//!
//! 1. **Bayesian Inference** ([`bayes`]): Posterior probability of timing leak with adaptive thresholds
//! 2. **Effect Decomposition** ([`effect`]): Separate uniform shift from tail effects
//! 3. **MDE Estimation** ([`mde`]): Minimum detectable effect at current noise level
//! 4. **Diagnostics** ([`diagnostics`]): Reliability checks (stationarity, model fit, outlier asymmetry)

// Re-export analysis functions from core
pub use timing_oracle_core::analysis::{
    analytical_mde, bayes, build_design_matrix, classify_pattern, compute_bayes_factor,
    compute_bayes_factor_mixture, compute_max_effect_ci, decompose_effect, effect, estimate_mde,
    mde, BayesResult, EffectDecomposition, MaxEffectCI, MdeEstimate,
};

// Keep diagnostics locally (depends on main crate types)
mod diagnostics;
pub use diagnostics::{compute_diagnostics, DiagnosticsExtra};

// Stationarity tracking for drift detection
mod stationarity;
pub use stationarity::{StationarityResult, StationarityTracker};
