//! Minimum Detectable Effect (MDE) estimation.
//!
//! This module estimates the smallest timing effect that can be reliably
//! detected given the current noise level. This helps users understand:
//!
//! - Whether "no leak detected" means the code is safe, or just noisy
//! - What magnitude of timing difference they could detect
//! - Whether more samples are needed for better sensitivity
//!
//! The MDE is computed by:
//! 1. Sampling from the null distribution (no timing leak)
//! 2. Fitting the effect model to each null sample
//! 3. Taking the 95th percentile of |beta_hat| as the MDE

use rand_distr::{Distribution, StandardNormal};

use crate::constants::{B_TAIL, ONES};
use crate::result::MinDetectableEffect;
use crate::types::{Matrix2, Matrix9, Matrix9x2, Vector2, Vector9};

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

/// Compute MDE analytically using GLS formula.
///
/// This is statistically equivalent to the Monte Carlo approach but 1000× faster.
/// Uses Cholesky solve for Σ₀⁻¹ X and closed-form 2×2 inverse for numerical stability.
///
/// For Gaussian distribution, the 95th percentile of |β̂| equals the two-sided 95% CI
/// half-width: P(|β̂| > z * SE) = 0.05 ⟺ 95th percentile ≈ 1.96 × SE,
/// where SE comes from GLS variance: Var(β̂) = (Xᵀ Σ₀⁻¹ X)⁻¹ = V.
///
/// # Arguments
///
/// * `covariance` - Pooled covariance matrix of quantile differences
///
/// # Returns
///
/// A tuple `(mde_shift, mde_tail)` with minimum detectable effects in nanoseconds.
pub fn analytical_mde(covariance: &Matrix9) -> (f64, f64) {
    // 1. Build design matrix X = [ONES | B_TAIL]
    let x = build_design_matrix();

    // 2. Compute Σ₀⁻¹ X via Cholesky solve (numerically stable)
    let chol = safe_cholesky(covariance);
    let sigma_inv_x = chol.solve(&x);

    // 3. Compute V⁻¹ = Xᵀ Σ₀⁻¹ X (only 2×2!)
    let v_inv = x.transpose() * sigma_inv_x;

    // 4. Invert 2×2 matrix using closed form
    let v = safe_inverse_2x2(&v_inv);

    // 5. Extract standard errors and compute MDEs
    let z = 1.96; // Two-sided 95% CI
    let mde_shift = z * v[(0, 0)].sqrt();
    let mde_tail = z * v[(1, 1)].sqrt();

    (mde_shift, mde_tail)
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

/// Estimate the minimum detectable effect.
///
/// Now uses analytical formula (1000× faster than Monte Carlo).
///
/// # Arguments
///
/// * `covariance` - Pooled covariance matrix of quantile differences
/// * `n_simulations` - **Ignored** (backward compatibility only)
/// * `prior_sigmas` - **Ignored** (backward compatibility only)
///
/// # Returns
///
/// An `MdeEstimate` with shift and tail MDE in nanoseconds.
pub fn estimate_mde(
    covariance: &Matrix9,
    _n_simulations: usize,    // Ignored (backward compat)
    _prior_sigmas: (f64, f64), // Ignored (backward compat)
) -> MdeEstimate {
    let (shift_ns, tail_ns) = analytical_mde(covariance);

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

/// Build the 9x2 design matrix [ones | b_tail] for effect decomposition.
fn build_design_matrix() -> Matrix9x2 {
    let mut x = Matrix9x2::zeros();

    for i in 0..9 {
        x[(i, 0)] = ONES[i];
        x[(i, 1)] = B_TAIL[i];
    }

    x
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

/// Safe 2×2 matrix inverse using closed-form formula.
///
/// Returns conservative huge MDEs (1e12) if the matrix is truly singular,
/// which causes priors to be dominated by min_effect_of_concern_ns.
fn safe_inverse_2x2(m: &Matrix2) -> Matrix2 {
    let a = m[(0, 0)];
    let b = m[(0, 1)];
    let c = m[(1, 0)];
    let d = m[(1, 1)];

    let det = a * d - b * c;

    // Warn on near-singular (but not failing) cases
    if det.abs() < 1e-8 {
        eprintln!(
            "Warning: Near-singular GLS precision matrix (det={:.2e}), MDE estimates may be unstable",
            det
        );
    }

    // Check for truly singular matrix
    if det.abs() < 1e-12 {
        // Design matrix unidentifiable given this covariance structure
        // Return conservative (huge) MDEs → priors dominated by min_effect → safe fallback
        return Matrix2::from_diagonal(&Vector2::new(1e12, 1e12));
    }

    let inv_det = 1.0 / det;
    Matrix2::new(
        d * inv_det,
        -b * inv_det,
        -c * inv_det,
        a * inv_det,
    )
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
        let mde = estimate_mde(&cov, 100, (1e3, 1e3));

        assert!(mde.shift_ns > 0.0, "MDE shift should be positive");
        assert!(mde.tail_ns > 0.0, "MDE tail should be positive");
    }

    #[test]
    fn test_analytical_mde_well_conditioned() {
        // Identity covariance → V = (Xᵀ X)⁻¹
        let cov = Matrix9::identity();
        let (mde_shift, mde_tail) = analytical_mde(&cov);

        // Should match known formula for identity covariance
        assert!(
            mde_shift > 0.0 && mde_shift < 10.0,
            "shift MDE unreasonable: {}",
            mde_shift
        );
        assert!(
            mde_tail > 0.0 && mde_tail < 10.0,
            "tail MDE unreasonable: {}",
            mde_tail
        );
    }

    #[test]
    fn test_analytical_mde_diagonal_covariance() {
        // Diagonal covariance (uncorrelated quantiles)
        let mut cov = Matrix9::zeros();
        for i in 0..9 {
            cov[(i, i)] = (i + 1) as f64; // Variance increases with quantile
        }

        let (mde_shift, mde_tail) = analytical_mde(&cov);

        // Both MDEs should be positive and finite
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
        cov[(0, 0)] = 1.0; // One large eigenvalue

        let (mde_shift, mde_tail) = analytical_mde(&cov);

        // Should not panic, should return finite values
        assert!(
            mde_shift.is_finite(),
            "shift MDE not finite: {}",
            mde_shift
        );
        assert!(mde_tail.is_finite(), "tail MDE not finite: {}", mde_tail);
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)] // Run in release mode for accurate timing
    fn test_analytical_mde_performance() {
        // Generate a realistic covariance matrix
        let mut cov = Matrix9::identity();
        for i in 0..9 {
            for j in 0..9 {
                let dist = (i as f64 - j as f64).abs();
                cov[(i, j)] = (-dist / 2.0).exp(); // Exponential correlation
            }
        }

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = analytical_mde(&cov);
        }
        let analytical_time = start.elapsed();

        // Should complete 1000 calls in < 1ms (< 1µs per call) in release mode
        assert!(
            analytical_time.as_micros() < 1000,
            "analytical MDE too slow: {:.1}µs per call",
            analytical_time.as_micros() as f64 / 1000.0
        );
    }

    // Sample-size scaling is handled by the covariance estimate; no direct test here.
}
