//! Statistical analysis layers for timing leak detection.
//!
//! This module implements the two-layer analysis framework (spec §2):
//!
//! - **Bayesian inference** (`bayes`): Posterior probability of timing leak
//! - **Effect decomposition** (`effect`): Separates uniform shift from tail effects
//! - **MDE estimation** (`mde`): Minimum detectable effect for power analysis

pub mod bayes;
pub mod effect;
pub mod mde;

pub use bayes::{
    build_design_matrix, compute_2d_projection, compute_bayes_factor, compute_max_effect_ci,
    compute_quantile_exceedances, BayesResult, MaxEffectCI,
};
pub use effect::{classify_pattern, decompose_effect, EffectDecomposition, EffectEstimate};
pub use mde::{analytical_mde, estimate_mde, MdeEstimate};
