//! Analysis module for timing leak detection.
//!
//! This module implements the multi-layer analysis pipeline:
//!
//! 1. **CI Gate** ([`ci_gate`]): Fast, frequentist screening with bounded false positive rate
//! 2. **Bayesian Inference** ([`bayes`]): Posterior probability of timing leak
//! 3. **Effect Decomposition** ([`effect`]): Separate uniform shift from tail effects
//! 4. **MDE Estimation** ([`mde`]): Minimum detectable effect at current noise level

mod bayes;
mod ci_gate;
mod effect;
mod mde;

pub use bayes::{compute_bayes_factor, compute_posterior_probability, BayesResult};
pub use ci_gate::{run_ci_gate, CiGateInput};
pub use effect::{decompose_effect, EffectDecomposition};
pub use mde::{estimate_mde, MdeEstimate};
