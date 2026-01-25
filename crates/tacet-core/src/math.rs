//! Math functions for no_std compatibility.
//!
//! In no_std mode, f64 doesn't have transcendental methods like sqrt, ln, cos, etc.
//! This module provides these functions using libm.

/// Square root.
#[inline]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// Natural logarithm.
#[inline]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// Base-10 logarithm.
#[inline]
pub fn log10(x: f64) -> f64 {
    libm::log10(x)
}

/// Cosine.
#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// Exponential (e^x).
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// Ceiling (round up).
#[inline]
pub fn ceil(x: f64) -> f64 {
    libm::ceil(x)
}

/// Power (x^y).
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// Cube root.
#[inline]
pub fn cbrt(x: f64) -> f64 {
    libm::cbrt(x)
}

/// Floor (round down).
#[inline]
pub fn floor(x: f64) -> f64 {
    libm::floor(x)
}

/// Round to nearest integer.
#[inline]
pub fn round(x: f64) -> f64 {
    libm::round(x)
}

/// Square (x^2).
#[inline]
pub fn sq(x: f64) -> f64 {
    x * x
}

/// Standard normal CDF: Φ(x) = (1 + erf(x/√2)) / 2
#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x * core::f64::consts::FRAC_1_SQRT_2))
}

/// Absolute value.
#[inline]
pub fn abs(x: f64) -> f64 {
    libm::fabs(x)
}
