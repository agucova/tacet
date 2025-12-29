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
pub use bootstrap::{block_bootstrap_resample, compute_block_size};
pub use covariance::{bootstrap_covariance_matrix, CovarianceEstimate};
pub use quantile::{compute_deciles, compute_quantile};
