//! Statistical calibration tests
//!
//! These tests validate the statistical properties of the timing oracle:
//! - FPR: False positive rate validation
//! - Power: Detection power across effect sizes
//! - Coverage: Credible interval coverage
//! - Bayesian: Posterior calibration

// Shared utilities - must be declared first and made public for other modules
#[path = "calibration/utils.rs"]
pub mod calibration_utils;

#[path = "calibration/fpr.rs"]
mod fpr;
#[path = "calibration/power.rs"]
mod power;
#[path = "calibration/power_curve.rs"]
mod power_curve;
#[path = "calibration/bayesian.rs"]
mod bayesian;
#[path = "calibration/coverage.rs"]
mod coverage;
#[path = "calibration/estimation.rs"]
mod estimation;
#[path = "calibration/autocorrelation.rs"]
mod autocorrelation;
#[path = "calibration/stress.rs"]
mod stress;
