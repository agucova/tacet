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

/// Sine.
#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
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

/// Cube (x^3).
///
/// Evaluated as `(x * x) * x`, matching the expansion `f64::powi(3)` lowers to,
/// so results are bit-identical to the std method.
#[inline]
pub fn cube(x: f64) -> f64 {
    x * x * x
}

/// Standard normal CDF: Φ(x) = (1 + erf(x/√2)) / 2
#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x * core::f64::consts::FRAC_1_SQRT_2))
}

/// Standard normal quantile (inverse CDF): Φ⁻¹(p).
///
/// Uses the rational approximation by Peter Acklam, which provides
/// ~10⁻⁹ relative accuracy across the full range.
pub fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Coefficients for rational approximation
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383_577_518_672_69e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        // Rational approximation for lower region
        let q = sqrt(-2.0 * ln(p));
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Rational approximation for upper region
        let q = sqrt(-2.0 * ln(1.0 - p));
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Absolute value.
#[inline]
pub fn abs(x: f64) -> f64 {
    libm::fabs(x)
}
