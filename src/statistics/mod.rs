//! Statistical methods for timing analysis.
//!
//! This module provides the core statistical infrastructure for timing oracle:
//! - Quantile computation using efficient O(n) selection algorithms
//! - Block bootstrap for resampling with autocorrelation preservation
//! - Covariance estimation via bootstrap
//! - Autocorrelation function computation

mod autocorrelation;
mod bootstrap;
mod covariance;
mod quantile;

pub use autocorrelation::{lag1_autocorrelation, lag2_autocorrelation};
pub use bootstrap::{
    block_bootstrap_resample, block_bootstrap_resample_into, compute_block_size, counter_rng_seed,
};
pub use covariance::{bootstrap_covariance_matrix, CovarianceEstimate};
pub use quantile::{
    compute_deciles, compute_deciles_fast, compute_deciles_inplace, compute_deciles_sorted,
    compute_deciles_with_buffer, compute_quantile,
};
