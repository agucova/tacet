//! Minimum Detectable Effect (MDE) estimation (spec §2.7).
//!
//! The MDE answers: "given the noise level in this measurement, what's the
//! smallest effect I could reliably detect?"
//!
//! This is critical for interpreting negative results. If MDE is 50ns and you're
//! concerned about 10ns effects, a passing test doesn't mean the code is safe—
//! it means the measurement wasn't sensitive enough.
//!
//! Formula (spec §2.7) with 50% power:
//! ```text
//! MDE_μ = z_{1-α/2} × sqrt((1ᵀ Σ₀⁻¹ 1)⁻¹)
//! MDE_τ = z_{1-α/2} × sqrt((bᵀ Σ₀⁻¹ b)⁻¹)
//! ```
//! where z_{1-α/2} is the (1-α/2) percentile of the standard normal.

use rand_distr::{Distribution, StandardNormal};

use crate::constants::{B_TAIL, ONES};
use crate::result::MinDetectableEffect;
use crate::types::{Matrix9, Vector9};

use super::effect::decompose_effect;

/// Result from MDE estimation.
#[derive(Debug, Clone)]
pub struct MdeEstimate {
    /// Minimum detectable uniform shift in nanoseconds.
    pub shift_ns: f64,
    /// Minimum detectable tail effect in nanoseconds.
    pub tail_ns: f64,
    /// Number of null simulations used.
    pub n_simulations: usize,
}

impl From<MdeEstimate> for MinDetectableEffect {
    fn from(mde: MdeEstimate) -> Self {
        MinDetectableEffect {
            shift_ns: mde.shift_ns,
            tail_ns: mde.tail_ns,
        }
    }
}

/// Compute MDE analytically using single-effect projection formulas (spec §2.7).
///
/// Formula:
/// ```text
/// MDE_μ = z_{1-α/2} × sqrt((1ᵀ Σ₀⁻¹ 1)⁻¹)
/// MDE_τ = z_{1-α/2} × sqrt((bᵀ Σ₀⁻¹ b)⁻¹)
/// ```
///
/// # Arguments
///
/// * `covariance` - Pooled covariance matrix of quantile differences (Σ₀)
/// * `alpha` - Significance level (e.g., 0.05 for 95% confidence)
///
/// # Returns
///
/// A tuple `(mde_shift, mde_tail)` with minimum detectable effects in nanoseconds.
pub fn analytical_mde(covariance: &Matrix9, alpha: f64) -> (f64, f64) {
    // Use Cholesky decomposition for numerically stable solve of Σ₀⁻¹ v
    let chol = safe_cholesky(covariance);

    // For shift MDE: Var(μ̂) = (1ᵀ Σ₀⁻¹ 1)⁻¹
    let ones_vec = Vector9::from_iterator(ONES.iter().cloned());
    let sigma_inv_ones = chol.solve(&ones_vec);
    let precision_shift = ones_vec.dot(&sigma_inv_ones);
    let var_shift = if precision_shift.abs() > 1e-12 {
        1.0 / precision_shift
    } else {
        // Near-singular: return huge MDE (conservative fallback)
        1e12
    };

    // For tail MDE: Var(τ̂) = (bᵀ Σ₀⁻¹ b)⁻¹
    let b_tail_vec = Vector9::from_iterator(B_TAIL.iter().cloned());
    let sigma_inv_b = chol.solve(&b_tail_vec);
    let precision_tail = b_tail_vec.dot(&sigma_inv_b);
    let var_tail = if precision_tail.abs() > 1e-12 {
        1.0 / precision_tail
    } else {
        1e12
    };

    // MDE with 50% power: z_{1-α/2} × SE (spec §2.7)
    let z = probit(1.0 - alpha / 2.0);
    let mde_shift = z * var_shift.sqrt();
    let mde_tail = z * var_tail.sqrt();

    (mde_shift, mde_tail)
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

    let t = (-2.0 * (1.0 - q).ln()).sqrt();
    let z = t - (C0 + C1 * t + C2 * t * t) / (1.0 + D1 * t + D2 * t * t + D3 * t * t * t);

    sign * z
}

/// Monte Carlo MDE estimation (kept for benchmarking).
///
/// This is the original implementation using Monte Carlo sampling.
/// Kept as a separate function for comparison and validation purposes.
#[allow(dead_code)]
pub fn estimate_mde_monte_carlo(
    covariance: &Matrix9,
    n_simulations: usize,
    prior_sigmas: (f64, f64),
) -> MdeEstimate {
    let mut rng = rand::rng();

    // Cache Cholesky decomposition (compute once, reuse for all samples)
    let chol = match nalgebra::Cholesky::new(*covariance) {
        Some(c) => c,
        None => {
            // Regularize if not positive definite
            let regularized = covariance + Matrix9::identity() * 1e-10;
            nalgebra::Cholesky::new(regularized)
                .expect("Regularized covariance should be positive definite")
        }
    };

    // Collect effect estimates from null samples
    let mut shift_effects = Vec::with_capacity(n_simulations);
    let mut tail_effects = Vec::with_capacity(n_simulations);

    for _ in 0..n_simulations {
        // Sample from null distribution using cached Cholesky
        let z: Vector9 = Vector9::from_fn(|_, _| StandardNormal.sample(&mut rng));
        let null_sample = chol.l() * z;

        // Fit effect model
        let decomp = decompose_effect(&null_sample, covariance, prior_sigmas);

        // Collect absolute effects
        shift_effects.push(decomp.posterior_mean[0].abs());
        tail_effects.push(decomp.posterior_mean[1].abs());
    }

    // Compute 95th percentiles
    let shift_mde = percentile(&mut shift_effects, 0.95);
    let tail_mde = percentile(&mut tail_effects, 0.95);

    MdeEstimate {
        shift_ns: shift_mde,
        tail_ns: tail_mde,
        n_simulations,
    }
}

/// Estimate the minimum detectable effect (spec §2.7).
///
/// # Arguments
///
/// * `covariance` - Pooled covariance matrix of quantile differences (Σ₀)
/// * `alpha` - Significance level (e.g., 0.01 for 99% confidence)
///
/// # Returns
///
/// An `MdeEstimate` with shift and tail MDE in nanoseconds.
pub fn estimate_mde(covariance: &Matrix9, alpha: f64) -> MdeEstimate {
    let (shift_ns, tail_ns) = analytical_mde(covariance, alpha);

    MdeEstimate {
        shift_ns,
        tail_ns,
        n_simulations: 0, // Analytical method doesn't use simulations
    }
}

/// Compute the p-th percentile of a vector.
///
/// Modifies the input vector by sorting it.
fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.sort_by(|a, b| a.total_cmp(b));

    let idx = (p * (values.len() - 1) as f64).round() as usize;
    let idx = idx.min(values.len() - 1);

    values[idx]
}

/// Safe Cholesky decomposition with adaptive jitter for near-singular matrices.
///
/// Uses the same regularization strategy as covariance estimation.
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

    nalgebra::Cholesky::new(regularized)
        .expect("Cholesky failed even after regularization")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_basic() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&mut values, 0.5) - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_95() {
        let mut values: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let p95 = percentile(&mut values, 0.95);
        assert!((p95 - 95.0).abs() < 1.0);
    }

    #[test]
    fn test_mde_positive() {
        // With identity covariance, MDE should be positive
        let cov = Matrix9::identity();
        let mde = estimate_mde(&cov, 0.05);

        assert!(mde.shift_ns > 0.0, "MDE shift should be positive");
        assert!(mde.tail_ns > 0.0, "MDE tail should be positive");
    }

    #[test]
    fn test_probit_accuracy() {
        // Test against known values
        assert!((probit(0.5) - 0.0).abs() < 1e-3, "probit(0.5) should be 0");
        assert!((probit(0.975) - 1.96).abs() < 1e-2, "probit(0.975) should be ~1.96");
        assert!((probit(0.995) - 2.576).abs() < 1e-2, "probit(0.995) should be ~2.576");
        assert!((probit(0.025) + 1.96).abs() < 1e-2, "probit(0.025) should be ~-1.96");
    }

    #[test]
    fn test_analytical_mde_iid_sanity_check() {
        // For i.i.d. quantiles with Σ₀ = σ² I (σ = 1) and α = 0.05, with 50% power:
        // - z_{0.975} ≈ 1.96
        // - 1ᵀ Σ⁻¹ 1 = 9, Var(μ̂) = 1/9
        // - MDE_μ = 1.96 / 3 ≈ 0.653
        //
        // - bᵀ Σ⁻¹ b = 0.9375, Var(τ̂) = 1/0.9375
        // - MDE_τ = 1.96 * sqrt(1/0.9375) ≈ 2.02

        let cov = Matrix9::identity();
        let (mde_shift, mde_tail) = analytical_mde(&cov, 0.05);

        let z = 1.96; // z_alpha for 50% power
        let expected_shift = z / 3.0;
        let expected_tail = z * (1.0 / 0.9375_f64).sqrt();

        assert!(
            (mde_shift - expected_shift).abs() < 0.05,
            "shift MDE should be ~{:.3}, got {:.3}",
            expected_shift,
            mde_shift
        );
        assert!(
            (mde_tail - expected_tail).abs() < 0.1,
            "tail MDE should be ~{:.3}, got {:.3}",
            expected_tail,
            mde_tail
        );
    }

    #[test]
    fn test_analytical_mde_alpha_scaling() {
        // MDE should increase with stricter alpha (smaller α → larger z → larger MDE)
        let cov = Matrix9::identity();
        let (mde_05, _) = analytical_mde(&cov, 0.05);  // z ≈ 1.96
        let (mde_01, _) = analytical_mde(&cov, 0.01);  // z ≈ 2.58

        assert!(
            mde_01 > mde_05,
            "MDE at α=0.01 ({:.3}) should be larger than α=0.05 ({:.3})",
            mde_01,
            mde_05
        );
    }

    #[test]
    fn test_analytical_mde_diagonal_covariance() {
        // Diagonal covariance (uncorrelated quantiles)
        let mut cov = Matrix9::zeros();
        for i in 0..9 {
            cov[(i, i)] = (i + 1) as f64;
        }

        let (mde_shift, mde_tail) = analytical_mde(&cov, 0.05);

        assert!(
            mde_shift.is_finite() && mde_shift > 0.0,
            "shift MDE not finite or positive: {}",
            mde_shift
        );
        assert!(
            mde_tail.is_finite() && mde_tail > 0.0,
            "tail MDE not finite or positive: {}",
            mde_tail
        );
    }

    #[test]
    fn test_analytical_mde_near_singular() {
        // Nearly rank-deficient covariance
        let mut cov = Matrix9::identity() * 1e-6;
        cov[(0, 0)] = 1.0;

        let (mde_shift, mde_tail) = analytical_mde(&cov, 0.05);

        assert!(mde_shift.is_finite(), "shift MDE not finite: {}", mde_shift);
        assert!(mde_tail.is_finite(), "tail MDE not finite: {}", mde_tail);
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)] // Run in release mode for accurate timing
    fn test_analytical_mde_performance() {
        let mut cov = Matrix9::identity();
        for i in 0..9 {
            for j in 0..9 {
                let dist = (i as f64 - j as f64).abs();
                cov[(i, j)] = (-dist / 2.0).exp();
            }
        }

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = analytical_mde(&cov, 0.05);
        }
        let analytical_time = start.elapsed();

        assert!(
            analytical_time.as_micros() < 1000,
            "analytical MDE too slow: {:.1}µs per call",
            analytical_time.as_micros() as f64 / 1000.0
        );
    }

    // Sample-size scaling is handled by the covariance estimate; no direct test here.
}
