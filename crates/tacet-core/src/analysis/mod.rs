//! Statistical analysis layers for timing leak detection.
//!
//! This module implements the statistical analysis framework (spec §3-5):
//!
//! - **Bayesian inference** (`bayes`): Posterior probability of timing leak
//! - **Effect estimation** (`effect`): Max effect and top quantile computation
//! - **MDE estimation** (`mde`): Minimum detectable effect for power analysis

pub mod bayes;
pub mod effect;
pub mod gibbs;
pub mod mde;

pub use bayes::{compute_bayes_gibbs, compute_max_effect_ci, BayesResult, MaxEffectCI};
pub use effect::{
    compute_effect_estimate, compute_effect_estimate_analytical, compute_top_quantiles,
    regularize_covariance,
};
pub use gibbs::{run_gibbs_inference, GibbsResult, NU, N_BURN, N_GIBBS, N_KEEP};
pub use mde::{analytical_mde, estimate_mde, MdeEstimate};
