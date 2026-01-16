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
    bayes, effect, mde,
    build_design_matrix, compute_bayes_factor, BayesResult,
    classify_pattern, decompose_effect, EffectDecomposition,
    analytical_mde, estimate_mde, MdeEstimate,
};

// Keep diagnostics locally (depends on main crate types)
mod diagnostics;
pub use diagnostics::{compute_diagnostics, DiagnosticsExtra};
