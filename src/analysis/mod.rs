//! Analysis module for timing leak detection.
//!
//! This module implements the adaptive Bayesian analysis pipeline:
//!
//! 1. **Bayesian Inference** ([`bayes`]): Posterior probability of timing leak with adaptive thresholds
//! 2. **Effect Decomposition** ([`effect`]): Separate uniform shift from tail effects
//! 3. **MDE Estimation** ([`mde`]): Minimum detectable effect at current noise level
//! 4. **Diagnostics** ([`diagnostics`]): Reliability checks (stationarity, model fit, outlier asymmetry)

pub mod bayes;
mod diagnostics;
pub mod effect;
pub mod mde;

pub use bayes::{compute_bayes_factor, BayesResult};
pub use diagnostics::{compute_diagnostics, DiagnosticsExtra};
pub use effect::{classify_pattern, decompose_effect, EffectDecomposition};
pub use mde::{analytical_mde, estimate_mde, estimate_mde_monte_carlo, MdeEstimate};
