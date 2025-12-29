//! Type aliases and common types.

use nalgebra::{SMatrix, SVector};

/// 9x9 covariance matrix for quantile differences.
pub type Matrix9 = SMatrix<f64, 9, 9>;

/// 9-dimensional vector for quantile differences.
pub type Vector9 = SVector<f64, 9>;

/// 9x2 design matrix [ones | b_tail] for effect decomposition.
pub type Matrix9x2 = SMatrix<f64, 9, 2>;

/// 2x2 matrix for effect covariance.
pub type Matrix2 = SMatrix<f64, 2, 2>;

/// 2-dimensional vector for effect parameters (shift, tail).
pub type Vector2 = SVector<f64, 2>;

/// Input class identifier for timing measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Class {
    /// Fixed input that might trigger timing variations.
    Fixed,
    /// Randomly sampled input for comparison.
    Random,
}
