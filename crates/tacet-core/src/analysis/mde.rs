//! Minimum Detectable Effect (MDE) estimation (spec §3.3).
//!
//! The MDE answers: "given the noise level in this measurement, what's the
//! smallest effect I could reliably detect?"
//!
//! This is critical for interpreting negative results. If MDE is 50ns and you're
//! concerned about 10ns effects, a passing test doesn't mean the code is safe—
//! it means the measurement wasn't sensitive enough.
//!
//! Formula with 50% power (spec §3.3):
//! ```text
//! MDE = z_{1-α/2} × max_k sqrt(Σ₀[k,k])
//! ```
//! where z_{1-α/2} is the (1-α/2) percentile of the standard normal.

extern crate alloc;

use crate::math;
use crate::result::MinDetectableEffect;
use crate::types::Matrix9;

/// Result from MDE estimation.
#[derive(Debug, Clone)]
pub struct MdeEstimate {
    /// Minimum detectable effect in nanoseconds.
    pub mde_ns: f64,
    /// Number of simulations used (0 for analytical method).
    pub n_simulations: usize,
}

impl From<MdeEstimate> for MinDetectableEffect {
    fn from(mde: MdeEstimate) -> Self {
        MinDetectableEffect { mde_ns: mde.mde_ns }
    }
}

/// Compute MDE analytically (spec §3.3).
///
/// For max|δ| detection, MDE is determined by the largest marginal variance:
/// ```text
/// MDE = z_{1-α/2} × max_k sqrt(Σ₀[k,k])
/// ```
///
/// # Arguments
///
/// * `covariance` - Pooled covariance matrix of quantile differences (Σ₀)
/// * `alpha` - Significance level (e.g., 0.05 for 95% confidence)
///
/// # Returns
///
/// The minimum detectable effect in nanoseconds.
pub fn analytical_mde(covariance: &Matrix9, alpha: f64) -> f64 {
    // Find maximum marginal variance
    let max_var = (0..9)
        .map(|i| covariance[(i, i)])
        .fold(0.0_f64, f64::max);

    // MDE with 50% power: z_{1-α/2} × sqrt(max variance)
    let z = probit(1.0 - alpha / 2.0);
    z * math::sqrt(max_var)
}

/// Inverse normal CDF (probit function).
///
/// Computes Φ⁻¹(p) using the Abramowitz & Stegun approximation (26.2.23).
/// Accurate to ~4.5×10⁻⁴ for p ∈ (0, 1).
fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Use symmetry: for p < 0.5, compute -probit(1-p)
    let (sign, q) = if p < 0.5 { (-1.0, 1.0 - p) } else { (1.0, p) };

    // Rational approximation constants (Abramowitz & Stegun 26.2.23)
    const C0: f64 = 2.515517;
    const C1: f64 = 0.802853;
    const C2: f64 = 0.010328;
    const D1: f64 = 1.432788;
    const D2: f64 = 0.189269;
    const D3: f64 = 0.001308;

    let t = math::sqrt(-2.0 * math::ln(1.0 - q));
    let z = t - (C0 + C1 * t + C2 * t * t) / (1.0 + D1 * t + D2 * t * t + D3 * t * t * t);

    sign * z
}

/// Estimate the minimum detectable effect (spec §3.3).
///
/// # Arguments
///
/// * `covariance` - Pooled covariance matrix of quantile differences (Σ₀)
/// * `alpha` - Significance level (e.g., 0.01 for 99% confidence)
///
/// # Returns
///
/// An `MdeEstimate` with MDE in nanoseconds.
pub fn estimate_mde(covariance: &Matrix9, alpha: f64) -> MdeEstimate {
    let mde_ns = analytical_mde(covariance, alpha);

    MdeEstimate {
        mde_ns,
        n_simulations: 0, // Analytical method doesn't use simulations
    }
}

/// Safe Cholesky decomposition with adaptive jitter for near-singular matrices.
///
/// Uses the same regularization strategy as covariance estimation.
#[allow(dead_code)]
fn safe_cholesky(matrix: &Matrix9) -> nalgebra::Cholesky<f64, nalgebra::Const<9>> {
    // Try decomposition first
    if let Some(chol) = nalgebra::Cholesky::new(*matrix) {
        return chol;
    }

    // Add adaptive jitter for near-singular matrices
    let trace = matrix.trace();
    let base_jitter = 1e-10;
    let adaptive_jitter = (trace / 9.0) * 1e-8;
    let jitter = base_jitter + adaptive_jitter;

    let mut regularized = *matrix;
    for i in 0..9 {
        regularized[(i, i)] += jitter;
    }

    nalgebra::Cholesky::new(regularized).expect("Cholesky failed even after regularization")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mde_positive() {
        // With identity covariance, MDE should be positive
        let cov = Matrix9::identity();
        let mde = estimate_mde(&cov, 0.05);

        assert!(mde.mde_ns > 0.0, "MDE should be positive");
    }

    #[test]
    fn test_probit_accuracy() {
        // Test against known values
        assert!((probit(0.5) - 0.0).abs() < 1e-3, "probit(0.5) should be 0");
        assert!(
            (probit(0.975) - 1.96).abs() < 1e-2,
            "probit(0.975) should be ~1.96"
        );
        assert!(
            (probit(0.995) - 2.576).abs() < 1e-2,
            "probit(0.995) should be ~2.576"
        );
        assert!(
            (probit(0.025) + 1.96).abs() < 1e-2,
            "probit(0.025) should be ~-1.96"
        );
    }

    #[test]
    fn test_analytical_mde_sanity_check() {
        // For i.i.d. quantiles with Σ₀ = σ² I (σ = 1) and α = 0.05:
        // - z_{0.975} ≈ 1.96
        // - max variance = 1.0
        // - MDE = 1.96 * sqrt(1.0) = 1.96

        let cov = Matrix9::identity();
        let mde = analytical_mde(&cov, 0.05);

        let expected = 1.96;
        assert!(
            (mde - expected).abs() < 0.05,
            "MDE should be ~{:.3}, got {:.3}",
            expected,
            mde
        );
    }

    #[test]
    fn test_analytical_mde_alpha_scaling() {
        // MDE should increase with stricter alpha (smaller α → larger z → larger MDE)
        let cov = Matrix9::identity();
        let mde_05 = analytical_mde(&cov, 0.05); // z ≈ 1.96
        let mde_01 = analytical_mde(&cov, 0.01); // z ≈ 2.58

        assert!(
            mde_01 > mde_05,
            "MDE at α=0.01 ({:.3}) should be larger than α=0.05 ({:.3})",
            mde_01,
            mde_05
        );
    }

    #[test]
    fn test_analytical_mde_diagonal_covariance() {
        // Diagonal covariance with varying variances
        let mut cov = Matrix9::zeros();
        for i in 0..9 {
            cov[(i, i)] = (i + 1) as f64;
        }

        let mde = analytical_mde(&cov, 0.05);

        // MDE should be based on max variance (9.0)
        let expected = 1.96 * math::sqrt(9.0);
        assert!(
            (mde - expected).abs() < 0.1,
            "MDE should be ~{:.3}, got {:.3}",
            expected,
            mde
        );
    }
}
