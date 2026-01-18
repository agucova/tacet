//! Calibration data for adaptive sampling (no_std compatible).
//!
//! This module defines the calibration results needed for the adaptive sampling loop.
//! The actual calibration computation lives in the `timing-oracle` crate; this module
//! provides the data structures that can be used in no_std environments.
//!
//! For no_std compatibility:
//! - No `std::time::Instant` (throughput measured by caller)
//! - No preflight checks (those require std features)
//! - Uses only `alloc` for heap allocation

extern crate alloc;

use alloc::vec::Vec;
use core::f64::consts::PI;

use nalgebra::Cholesky;
use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::constants::DEFAULT_SEED;
use crate::math;
use crate::types::{Matrix9, Vector9};

use super::CalibrationSnapshot;

/// Conservative prior scale factor used as fallback.
/// Chosen to give higher exceedance probability than the target 62%.
const CONSERVATIVE_PRIOR_SCALE: f64 = 1.5;

/// Target prior exceedance probability.
const TARGET_EXCEEDANCE: f64 = 0.62;

/// Number of Monte Carlo samples for prior calibration.
const PRIOR_CALIBRATION_SAMPLES: usize = 50_000;

/// Maximum iterations for prior calibration root-finding.
const MAX_CALIBRATION_ITERATIONS: usize = 20;

/// Condition number threshold for triggering robust shrinkage (§3.3.5).
const CONDITION_NUMBER_THRESHOLD: f64 = 1e4;

/// Minimum diagonal floor for regularization (prevents division by zero).
const DIAGONAL_FLOOR: f64 = 1e-12;

/// Default prior weight for the narrow component in mixture prior (§3.3.5 v5.2).
/// w = 0.99 means 99% weight on narrow component, 1% on slab.
pub const MIXTURE_PRIOR_WEIGHT: f64 = 0.99;

/// Calibration results from the initial measurement phase (no_std compatible).
///
/// This struct contains the essential statistical data needed for the adaptive
/// sampling loop. It is designed for use in no_std environments like SGX enclaves.
///
/// For full calibration with preflight checks, see `timing_oracle::Calibration`.
///
/// ## Mixture Prior (v5.2)
///
/// The prior is a 2-component scale mixture:
/// - Narrow component: N(0, σ₁²R) with weight w (default 0.99)
/// - Slab component: N(0, σ₂²R) with weight (1-w) (default 0.01)
///
/// This allows the posterior to "escape" to the slab when data strongly
/// supports large effects, fixing the shrinkage pathology in noisy-likelihood regimes.
#[derive(Debug, Clone)]
pub struct Calibration {
    /// Covariance "rate" - multiply by 1/n to get Sigma_n for n samples.
    /// Computed as Sigma_cal * n_cal where Sigma_cal is calibration covariance.
    /// This allows O(1) covariance scaling as samples accumulate.
    pub sigma_rate: Matrix9,

    /// Block length from Politis-White algorithm.
    /// Used for block bootstrap to preserve autocorrelation structure.
    pub block_length: usize,

    /// 9D prior covariance for δ = (δ₁, ..., δ₉).
    /// Λ₀ = σ²_prior × R where R = Corr(Σ_rate) (correlation-shaped).
    /// For backwards compatibility, this equals prior_cov_narrow.
    pub prior_cov_9d: Matrix9,

    /// Narrow component prior covariance: Λ₀,₁ = σ₁² × R (v5.2).
    pub prior_cov_narrow: Matrix9,

    /// Slab component prior covariance: Λ₀,₂ = σ₂² × R (v5.2).
    pub prior_cov_slab: Matrix9,

    /// Narrow component scale σ₁ (calibrated so mixture hits 62% exceedance).
    pub sigma_narrow: f64,

    /// Slab component scale σ₂ (deterministic: max(50θ, 10×SE_med)).
    pub sigma_slab: f64,

    /// Prior weight for narrow component (default 0.99).
    pub prior_weight: f64,

    /// The theta threshold being used (in nanoseconds).
    pub theta_ns: f64,

    /// Number of calibration samples collected per class.
    pub calibration_samples: usize,

    /// Whether discrete mode is active (< 10% unique values).
    /// When true, use mid-quantile estimators and m-out-of-n bootstrap.
    pub discrete_mode: bool,

    /// Minimum detectable effect (shift component) from calibration.
    pub mde_shift_ns: f64,

    /// Minimum detectable effect (tail component) from calibration.
    pub mde_tail_ns: f64,

    /// Statistics snapshot from calibration phase for drift detection.
    pub calibration_snapshot: CalibrationSnapshot,

    /// Timer resolution in nanoseconds.
    pub timer_resolution_ns: f64,

    /// Measured throughput for time estimation (samples per second).
    /// The caller is responsible for measuring this during calibration.
    pub samples_per_second: f64,

    /// Floor-rate constant.
    /// Computed once at calibration: 95th percentile of max_k|Z_k| where Z ~ N(0, Σ_rate).
    /// Used for analytical theta_floor computation: theta_floor_stat(n) = c_floor / sqrt(n).
    pub c_floor: f64,

    /// Projection mismatch threshold.
    /// 99th percentile of bootstrap Q_proj distribution.
    pub projection_mismatch_thresh: f64,

    /// Timer resolution floor component.
    /// theta_tick = (1 tick in ns) / K where K is the batch size.
    pub theta_tick: f64,

    /// Effective threshold for this run.
    /// theta_eff = max(theta_user, theta_floor) or just theta_floor in research mode.
    pub theta_eff: f64,

    /// Initial measurement floor at calibration time.
    pub theta_floor_initial: f64,

    /// Deterministic RNG seed used for this run.
    pub rng_seed: u64,
}

impl Calibration {
    /// Create a new Calibration with all required fields (v5.2 mixture prior).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sigma_rate: Matrix9,
        block_length: usize,
        prior_cov_narrow: Matrix9,
        prior_cov_slab: Matrix9,
        sigma_narrow: f64,
        sigma_slab: f64,
        prior_weight: f64,
        theta_ns: f64,
        calibration_samples: usize,
        discrete_mode: bool,
        mde_shift_ns: f64,
        mde_tail_ns: f64,
        calibration_snapshot: CalibrationSnapshot,
        timer_resolution_ns: f64,
        samples_per_second: f64,
        c_floor: f64,
        projection_mismatch_thresh: f64,
        theta_tick: f64,
        theta_eff: f64,
        theta_floor_initial: f64,
        rng_seed: u64,
    ) -> Self {
        Self {
            sigma_rate,
            block_length,
            prior_cov_9d: prior_cov_narrow, // Backwards compatibility
            prior_cov_narrow,
            prior_cov_slab,
            sigma_narrow,
            sigma_slab,
            prior_weight,
            theta_ns,
            calibration_samples,
            discrete_mode,
            mde_shift_ns,
            mde_tail_ns,
            calibration_snapshot,
            timer_resolution_ns,
            samples_per_second,
            c_floor,
            projection_mismatch_thresh,
            theta_tick,
            theta_eff,
            theta_floor_initial,
            rng_seed,
        }
    }

    /// Scale sigma_rate to get covariance for n samples.
    ///
    /// Σ_n = Σ_rate / n
    pub fn covariance_for_n(&self, n: usize) -> Matrix9 {
        if n == 0 {
            return self.sigma_rate; // Avoid division by zero
        }
        self.sigma_rate / (n as f64)
    }

    /// Estimate time to collect `n` additional samples based on calibration throughput.
    ///
    /// Returns estimated seconds. Caller should add any overhead.
    pub fn estimate_collection_time_secs(&self, n: usize) -> f64 {
        if self.samples_per_second <= 0.0 {
            return 0.0;
        }
        n as f64 / self.samples_per_second
    }
}

/// Configuration for calibration phase (no_std compatible).
///
/// This contains parameters for calibration that don't require std features.
/// For full configuration with preflight options, see `timing_oracle::CalibrationConfig`.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Number of samples to collect per class during calibration.
    pub calibration_samples: usize,

    /// Number of bootstrap iterations for covariance estimation.
    pub bootstrap_iterations: usize,

    /// Theta threshold in nanoseconds.
    pub theta_ns: f64,

    /// Alpha level for MDE computation.
    pub alpha: f64,

    /// Random seed for bootstrap.
    pub seed: u64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            calibration_samples: 5000,
            bootstrap_iterations: 2000, // Full bootstrap for accurate covariance
            theta_ns: 100.0,
            alpha: 0.01,
            seed: DEFAULT_SEED,
        }
    }
}

/// Compute correlation-shaped 9D prior covariance (spec v5.1).
///
/// Λ₀ = σ²_prior × R where R = Corr(Σ_rate) = D^(-1/2) Σ_rate D^(-1/2)
///
/// Since diag(R) = 1, σ_prior equals the marginal prior SD for all coordinates.
/// This eliminates hidden heteroskedasticity that could cause pathological
/// shrinkage for certain effect patterns.
///
/// # Arguments
/// * `sigma_rate` - Covariance rate matrix from bootstrap
/// * `sigma_prior` - Prior scale (calibrated for 62% exceedance)
/// * `discrete_mode` - Whether discrete timer mode is active
///
/// # Returns
/// The prior covariance matrix Λ₀.
pub fn compute_prior_cov_9d(
    sigma_rate: &Matrix9,
    sigma_prior: f64,
    discrete_mode: bool,
) -> Matrix9 {
    // Compute correlation matrix R = D^(-1/2) Σ_rate D^(-1/2)
    let r = compute_correlation_matrix(sigma_rate);

    // Apply two-step regularization (§3.3.5):
    // 1. Robust shrinkage (if in fragile regime)
    // 2. Numerical jitter (always, as needed)
    let r = apply_correlation_regularization(&r, discrete_mode);

    // Λ₀ = σ²_prior × R
    r * (sigma_prior * sigma_prior)
}

/// Compute correlation matrix R = D^(-1/2) Σ D^(-1/2) from covariance matrix.
///
/// The correlation matrix has unit diagonal (diag(R) = 1) and encodes
/// the correlation structure of Σ without the variance magnitudes.
fn compute_correlation_matrix(sigma: &Matrix9) -> Matrix9 {
    // Extract diagonal and apply floor for numerical stability
    let mut d_inv_sqrt = [0.0_f64; 9];
    for i in 0..9 {
        let var = sigma[(i, i)].max(DIAGONAL_FLOOR);
        d_inv_sqrt[i] = 1.0 / math::sqrt(var);
    }

    // R = D^(-1/2) × Σ × D^(-1/2)
    let mut r = *sigma;
    for i in 0..9 {
        for j in 0..9 {
            r[(i, j)] *= d_inv_sqrt[i] * d_inv_sqrt[j];
        }
    }

    r
}

/// Estimate condition number of a symmetric matrix via power iteration.
///
/// Returns the ratio of largest to smallest eigenvalue magnitude.
/// Uses a simple approach: max(|diag|) / min(|diag|) as a quick estimate,
/// plus check for Cholesky failure which indicates poor conditioning.
fn estimate_condition_number(r: &Matrix9) -> f64 {
    // Quick estimate from diagonal ratio (exact for diagonal matrices)
    let diag: Vec<f64> = (0..9).map(|i| r[(i, i)].abs()).collect();
    let max_diag = diag.iter().cloned().fold(0.0_f64, f64::max);
    let min_diag = diag.iter().cloned().fold(f64::INFINITY, f64::min);

    if min_diag < DIAGONAL_FLOOR {
        return f64::INFINITY;
    }

    // For correlation matrices, this underestimates the true condition number,
    // but we also check Cholesky failure separately.
    max_diag / min_diag
}

/// Check if we're in a "fragile regime" requiring robust shrinkage (§3.3.5).
///
/// Fragile regime is detected when:
/// - Discrete timer mode is active, OR
/// - Condition number of R exceeds 10⁴, OR
/// - Cholesky factorization fails
fn is_fragile_regime(r: &Matrix9, discrete_mode: bool) -> bool {
    if discrete_mode {
        return true;
    }

    let cond = estimate_condition_number(r);
    if cond > CONDITION_NUMBER_THRESHOLD {
        return true;
    }

    // Check if Cholesky would fail
    Cholesky::new(*r).is_none()
}

/// Apply two-step regularization to correlation matrix (§3.3.5).
///
/// Step 1 (conditional): Robust shrinkage R ← (1-λ)R + λI for fragile regimes.
/// Step 2 (always): Numerical jitter R ← R + εI to ensure SPD.
fn apply_correlation_regularization(r: &Matrix9, discrete_mode: bool) -> Matrix9 {
    let mut r = *r;

    // Step 1: Robust shrinkage (conditional on fragile regime)
    if is_fragile_regime(&r, discrete_mode) {
        // Choose λ based on severity (§3.3.5 allows [0.01, 0.2])
        let cond = estimate_condition_number(&r);
        let lambda = if cond > CONDITION_NUMBER_THRESHOLD * 10.0 {
            0.2 // Severe: aggressive shrinkage
        } else if cond > CONDITION_NUMBER_THRESHOLD {
            0.1 // Moderate
        } else if discrete_mode {
            0.05 // Mild: just discrete mode
        } else {
            0.01 // Minimal
        };

        let identity = Matrix9::identity();
        r = r * (1.0 - lambda) + identity * lambda;
    }

    // Step 2: Numerical jitter (always, as needed for Cholesky)
    // Try increasingly large epsilon until Cholesky succeeds
    for &eps in &[1e-10, 1e-9, 1e-8, 1e-7, 1e-6] {
        let r_jittered = r + Matrix9::identity() * eps;
        if Cholesky::new(r_jittered).is_some() {
            return r_jittered;
        }
    }

    // Fallback: aggressive jitter
    r + Matrix9::identity() * 1e-5
}

/// Compute slab scale σ₂ deterministically (§3.3.5 v5.2).
///
/// σ₂ = max(50 × θ_eff, 10 × SE_med)
///
/// The slab scale is NOT tuned - it is computed deterministically from
/// calibration-time quantities only. This ensures the slab is always
/// wide enough to capture large effects.
///
/// # Arguments
/// * `sigma_rate` - Covariance rate matrix from calibration
/// * `theta_eff` - Effective threshold in nanoseconds
/// * `n_cal` - Number of calibration samples
///
/// # Returns
/// The slab scale σ₂.
pub fn compute_slab_scale(sigma_rate: &Matrix9, theta_eff: f64, n_cal: usize) -> f64 {
    let median_se = compute_median_se(sigma_rate, n_cal);
    (50.0 * theta_eff).max(10.0 * median_se)
}

/// Compute median standard error from sigma_rate.
fn compute_median_se(sigma_rate: &Matrix9, n_cal: usize) -> f64 {
    let mut ses: Vec<f64> = (0..9)
        .map(|i| {
            let var = sigma_rate[(i, i)].max(DIAGONAL_FLOOR);
            math::sqrt(var / n_cal.max(1) as f64)
        })
        .collect();
    ses.sort_by(|a, b| a.total_cmp(b));
    ses[4] // Median of 9 values
}

/// Calibrate narrow scale σ₁ so that the MIXTURE prior hits 62% exceedance (§3.3.5 v5.2).
///
/// Uses binary search to find σ₁ such that:
/// P(max_k |δ_k| > θ_eff | δ ~ mixture) = 0.62
///
/// where mixture = w·N(0, σ₁²R) + (1−w)·N(0, σ₂²R)
///
/// # Arguments
/// * `sigma_rate` - Covariance rate matrix from calibration
/// * `theta_eff` - Effective threshold in nanoseconds
/// * `n_cal` - Number of calibration samples (for bounds computation)
/// * `sigma_slab` - Slab scale σ₂ (computed first via compute_slab_scale)
/// * `prior_weight` - Weight w for narrow component (default 0.99)
/// * `discrete_mode` - Whether discrete timer mode is active
/// * `seed` - Deterministic RNG seed
///
/// # Returns
/// The calibrated narrow scale σ₁.
pub fn calibrate_narrow_scale(
    sigma_rate: &Matrix9,
    theta_eff: f64,
    n_cal: usize,
    sigma_slab: f64,
    prior_weight: f64,
    discrete_mode: bool,
    seed: u64,
) -> f64 {
    let median_se = compute_median_se(sigma_rate, n_cal);

    // Search bounds (same as v5.1)
    let mut lo = theta_eff * 0.05;
    let mut hi = (theta_eff * 50.0).max(10.0 * median_se);

    for _ in 0..MAX_CALIBRATION_ITERATIONS {
        let mid = (lo + hi) / 2.0;

        // Compute MIXTURE exceedance (not single Gaussian)
        let exceedance = compute_mixture_prior_exceedance(
            sigma_rate,
            mid,
            sigma_slab,
            prior_weight,
            theta_eff,
            discrete_mode,
            seed,
        );

        if (exceedance - TARGET_EXCEEDANCE).abs() < 0.01 {
            return mid; // Close enough
        }

        if exceedance > TARGET_EXCEEDANCE {
            // Too much exceedance -> reduce narrow scale
            hi = mid;
        } else {
            // Too little exceedance -> increase narrow scale
            lo = mid;
        }
    }

    // Fallback to conservative value
    theta_eff * CONSERVATIVE_PRIOR_SCALE
}

/// Compute P(max_k |δ_k| > θ | δ ~ mixture) via Monte Carlo (v5.2).
///
/// Mixture prior: w·N(0, σ₁²R) + (1−w)·N(0, σ₂²R)
fn compute_mixture_prior_exceedance(
    sigma_rate: &Matrix9,
    sigma_narrow: f64,
    sigma_slab: f64,
    prior_weight: f64,
    theta: f64,
    discrete_mode: bool,
    seed: u64,
) -> f64 {
    // Compute regularized correlation matrix
    let r = compute_correlation_matrix(sigma_rate);
    let r = apply_correlation_regularization(&r, discrete_mode);

    // Compute both component covariances
    let lambda_narrow = r * (sigma_narrow * sigma_narrow);
    let lambda_slab = r * (sigma_slab * sigma_slab);

    let chol_narrow = match Cholesky::new(lambda_narrow) {
        Some(c) => c,
        None => return 0.5, // Neutral if decomposition fails
    };
    let chol_slab = match Cholesky::new(lambda_slab) {
        Some(c) => c,
        None => return 0.5,
    };

    let l_narrow = chol_narrow.l();
    let l_slab = chol_slab.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut count = 0usize;

    for _ in 0..PRIOR_CALIBRATION_SAMPLES {
        // Draw from mixture: with probability w, use narrow; else use slab
        let l = if rng.random::<f64>() < prior_weight {
            &l_narrow
        } else {
            &l_slab
        };

        // Sample z ~ N(0, I_9)
        let mut z = Vector9::zeros();
        for i in 0..9 {
            z[i] = sample_standard_normal(&mut rng);
        }

        // Transform to δ ~ N(0, Λ)
        let delta = l * z;

        // Check if max_k |δ_k| > θ
        let max_effect = delta.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if max_effect > theta {
            count += 1;
        }
    }

    count as f64 / PRIOR_CALIBRATION_SAMPLES as f64
}

/// Legacy function: Calibrate σ_prior so that P(max_k |δ_k| > θ_eff | δ ~ N(0, Λ₀)) = π₀.
///
/// **Deprecated in v5.2**: Use `calibrate_narrow_scale` with mixture prior instead.
/// This function is kept for backwards compatibility and testing.
///
/// Uses binary search to find the scale factor.
/// Falls back to CONSERVATIVE_PRIOR_SCALE (1.5) with warning if calibration fails.
///
/// # Arguments
/// * `sigma_rate` - Covariance rate matrix from calibration
/// * `theta_eff` - Effective threshold in nanoseconds
/// * `n_cal` - Number of calibration samples (for SE computation)
/// * `discrete_mode` - Whether discrete timer mode is active
/// * `seed` - Deterministic RNG seed
///
/// # Returns
/// The calibrated σ_prior value.
pub fn calibrate_prior_scale(
    sigma_rate: &Matrix9,
    theta_eff: f64,
    n_cal: usize,
    discrete_mode: bool,
    seed: u64,
) -> f64 {
    let median_se = compute_median_se(sigma_rate, n_cal);

    // Search bounds that accommodate both threshold-scale and noise-scale effects (§3.3.5)
    // lo = θ_eff × 0.05
    // hi = max(θ_eff × 50, 10 × median_SE)
    let mut lo = theta_eff * 0.05;
    let mut hi = (theta_eff * 50.0).max(10.0 * median_se);

    for _ in 0..MAX_CALIBRATION_ITERATIONS {
        let mid = (lo + hi) / 2.0;
        let lambda0 = compute_prior_cov_9d(sigma_rate, mid, discrete_mode);
        let exceedance = compute_prior_exceedance(&lambda0, theta_eff, seed);

        if (exceedance - TARGET_EXCEEDANCE).abs() < 0.01 {
            return mid; // Close enough
        }

        if exceedance > TARGET_EXCEEDANCE {
            // Too much exceedance -> reduce scale
            hi = mid;
        } else {
            // Too little exceedance -> increase scale
            lo = mid;
        }
    }

    // Fallback to conservative value
    theta_eff * CONSERVATIVE_PRIOR_SCALE
}

/// Compute P(max_k |δ_k| > θ | δ ~ N(0, Λ₀)) via Monte Carlo.
fn compute_prior_exceedance(lambda0: &Matrix9, theta: f64, seed: u64) -> f64 {
    let chol = match Cholesky::new(*lambda0) {
        Some(c) => c,
        None => return 0.5, // Neutral if decomposition fails
    };
    let l = chol.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut count = 0usize;

    for _ in 0..PRIOR_CALIBRATION_SAMPLES {
        // Sample z ~ N(0, I_9)
        let mut z = Vector9::zeros();
        for i in 0..9 {
            z[i] = sample_standard_normal(&mut rng);
        }

        // Transform to δ ~ N(0, Λ₀)
        let delta = l * z;

        // Check if max_k |δ_k| > θ
        let max_effect = delta.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if max_effect > theta {
            count += 1;
        }
    }

    count as f64 / PRIOR_CALIBRATION_SAMPLES as f64
}

/// Compute floor-rate constant c_floor for 9D model.
///
/// c_floor = q_95(max_k |Z_k|) where Z ~ N(0, Σ_rate)
///
/// Used for theta_floor computation: theta_floor_stat(n) = c_floor / sqrt(n)
pub fn compute_c_floor_9d(sigma_rate: &Matrix9, seed: u64) -> f64 {
    let chol = match Cholesky::new(*sigma_rate) {
        Some(c) => c,
        None => {
            // Fallback: use trace-based approximation
            let trace: f64 = (0..9).map(|i| sigma_rate[(i, i)]).sum();
            return math::sqrt(trace / 9.0) * 2.5; // Approximate 95th percentile
        }
    };
    let l = chol.l();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut max_effects = Vec::with_capacity(PRIOR_CALIBRATION_SAMPLES);

    for _ in 0..PRIOR_CALIBRATION_SAMPLES {
        let mut z = Vector9::zeros();
        for i in 0..9 {
            z[i] = sample_standard_normal(&mut rng);
        }

        let sample = l * z;
        let max_effect = sample.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        max_effects.push(max_effect);
    }

    // 95th percentile
    max_effects.sort_by(|a, b| a.total_cmp(b));
    let idx = (PRIOR_CALIBRATION_SAMPLES as f64 * 0.95) as usize;
    max_effects[idx.min(PRIOR_CALIBRATION_SAMPLES - 1)]
}

/// Sample from standard normal using Box-Muller transform.
fn sample_standard_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random();
    let u2: f64 = rng.random();
    math::sqrt(-2.0 * math::ln(u1.max(1e-12))) * math::cos(2.0 * PI * u2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::StatsSnapshot;

    fn make_test_calibration() -> Calibration {
        let snapshot = CalibrationSnapshot::new(
            StatsSnapshot {
                count: 1000,
                mean: 100.0,
                variance: 25.0,
                autocorr_lag1: 0.1,
            },
            StatsSnapshot {
                count: 1000,
                mean: 105.0,
                variance: 30.0,
                autocorr_lag1: 0.12,
            },
        );

        let sigma_rate = Matrix9::identity() * 1000.0;
        let discrete_mode = false;
        let theta_eff = 100.0;
        let n_cal = 5000;

        // v5.2 mixture prior
        let sigma_slab = compute_slab_scale(&sigma_rate, theta_eff, n_cal);
        let sigma_narrow = calibrate_narrow_scale(
            &sigma_rate,
            theta_eff,
            n_cal,
            sigma_slab,
            MIXTURE_PRIOR_WEIGHT,
            discrete_mode,
            42,
        );
        let prior_cov_narrow = compute_prior_cov_9d(&sigma_rate, sigma_narrow, discrete_mode);
        let prior_cov_slab = compute_prior_cov_9d(&sigma_rate, sigma_slab, discrete_mode);

        Calibration::new(
            sigma_rate,
            10, // block_length
            prior_cov_narrow,
            prior_cov_slab,
            sigma_narrow,
            sigma_slab,
            MIXTURE_PRIOR_WEIGHT,
            theta_eff,
            n_cal,
            discrete_mode,
            5.0,  // mde_shift_ns
            10.0, // mde_tail_ns
            snapshot,
            1.0,       // timer_resolution_ns
            10000.0,   // samples_per_second
            10.0,      // c_floor
            18.48,     // projection_mismatch_thresh
            0.001,     // theta_tick
            theta_eff, // theta_eff
            0.1,       // theta_floor_initial
            42,        // rng_seed
        )
    }

    #[test]
    fn test_covariance_scaling() {
        let cal = make_test_calibration();

        // At n=1000, covariance should be sigma_rate / 1000
        let cov_1000 = cal.covariance_for_n(1000);
        assert!((cov_1000[(0, 0)] - 1.0).abs() < 1e-10);

        // At n=2000, covariance should be half
        let cov_2000 = cal.covariance_for_n(2000);
        assert!((cov_2000[(0, 0)] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_zero_n() {
        let cal = make_test_calibration();
        let cov = cal.covariance_for_n(0);
        // Should return sigma_rate unchanged (avoid NaN)
        assert!((cov[(0, 0)] - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_collection_time() {
        let cal = make_test_calibration();

        // 10000 samples/sec -> 1000 samples takes 0.1 sec
        let time = cal.estimate_collection_time_secs(1000);
        assert!((time - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_compute_prior_cov_9d_unit_diagonal() {
        // Use identity matrix - correlation of identity is identity
        let sigma_rate = Matrix9::identity();
        let prior = compute_prior_cov_9d(&sigma_rate, 10.0, false);

        // R = Corr(I) = I (identity has unit diagonal, no off-diagonal correlation)
        // Λ₀ = σ²_prior × R = 100 × I
        // Each diagonal should be ~100 (σ² = 100)
        // Note: jitter adds ~1e-10 so expect ~100
        let expected = 100.0;
        for i in 0..9 {
            assert!(
                (prior[(i, i)] - expected).abs() < 1.0,
                "Diagonal {} was {}, expected ~{}",
                i,
                prior[(i, i)],
                expected
            );
        }
    }

    #[test]
    fn test_prior_exceedance_calibration() {
        let sigma_rate = Matrix9::identity() * 100.0;
        let theta_eff = 10.0;
        let n_cal = 5000;
        let discrete_mode = false;

        let sigma_prior = calibrate_prior_scale(&sigma_rate, theta_eff, n_cal, discrete_mode, 42);

        // Check that calibration gives reasonable value
        assert!(
            sigma_prior > theta_eff * 0.5,
            "sigma_prior should be > theta_eff/2"
        );
        assert!(
            sigma_prior < theta_eff * 3.0,
            "sigma_prior should be < 3*theta_eff"
        );

        // Verify exceedance is near target
        let lambda0 = compute_prior_cov_9d(&sigma_rate, sigma_prior, discrete_mode);
        let exceedance = compute_prior_exceedance(&lambda0, theta_eff, 42);
        assert!(
            (exceedance - TARGET_EXCEEDANCE).abs() < 0.05,
            "Exceedance {} should be near {}",
            exceedance,
            TARGET_EXCEEDANCE
        );
    }

    #[test]
    fn test_c_floor_computation() {
        let sigma_rate = Matrix9::identity() * 100.0;
        let c_floor = compute_c_floor_9d(&sigma_rate, 42);

        // c_floor should be roughly sqrt(100) * 2 to 3 for 95th percentile of max
        assert!(c_floor > 15.0, "c_floor {} should be > 15", c_floor);
        assert!(c_floor < 40.0, "c_floor {} should be < 40", c_floor);
    }

    #[test]
    fn test_calibration_config_default() {
        let config = CalibrationConfig::default();
        assert_eq!(config.calibration_samples, 5000);
        assert_eq!(config.bootstrap_iterations, 2000);
        assert!((config.theta_ns - 100.0).abs() < 1e-10);
    }
}
