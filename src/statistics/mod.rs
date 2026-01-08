//! Statistical methods for timing analysis.
//!
//! This module provides the core statistical infrastructure for timing oracle:
//! - Quantile computation using efficient O(n) selection algorithms
//! - Block bootstrap for resampling with autocorrelation preservation
//! - Covariance estimation via bootstrap
//! - Autocorrelation function computation
//! - Optimal block length estimation using Politis-White algorithm

mod autocorrelation;
mod block_length;
mod bootstrap;
mod covariance;
mod quantile;

pub use autocorrelation::{estimate_dependence_length, lag1_autocorrelation, lag2_autocorrelation};
pub use block_length::{optimal_block_length, paired_optimal_block_length, OptimalBlockLength};
pub use bootstrap::{
    block_bootstrap_resample, block_bootstrap_resample_into, block_bootstrap_resample_joint_into,
    compute_block_size, counter_rng_seed,
};
pub use covariance::{
    apply_variance_floor, bootstrap_covariance_matrix, bootstrap_difference_covariance,
    bootstrap_difference_covariance_discrete,
    scale_covariance_for_inference, CovarianceEstimate,
};
pub use quantile::{
    compute_deciles, compute_deciles_fast, compute_deciles_inplace, compute_deciles_sorted,
    compute_deciles_with_buffer, compute_midquantile, compute_midquantile_deciles,
    compute_midquantile_deciles_sorted, compute_quantile,
};
