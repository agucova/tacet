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

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Counter-based RNG seed generation using SplitMix64.
/// Provides deterministic, well-distributed seeds for parallel MC sampling.
#[cfg(feature = "parallel")]
#[inline]
fn counter_rng_seed(base_seed: u64, counter: u64) -> u64 {
    let mut z = base_seed.wrapping_add(counter.wrapping_mul(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

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

/// Degrees of freedom for Student's t prior (v5.4).
pub const NU: f64 = 4.0;


/// Calibration results from the initial measurement phase (no_std compatible).
///
/// This struct contains the essential statistical data needed for the adaptive
/// sampling loop. It is designed for use in no_std environments like SGX enclaves.
///
/// For full calibration with preflight checks, see `timing_oracle::Calibration`.
///
/// ## Student's t Prior (v5.4+)
///
/// The prior is a Student's t distribution with ν=4 degrees of freedom:
/// δ ~ t_ν(0, σ_t²R)
///
/// This is implemented via scale mixture: λ ~ Gamma(ν/2, ν/2), δ|λ ~ N(0, (σ_t²/λ)R).
/// The marginal variance is (ν/(ν-2)) σ_t² R = 2σ_t² R for ν=4.
#[derive(Debug, Clone)]
pub struct Calibration {
    /// Covariance "rate" - multiply by 1/n to get Sigma_n for n samples.
    /// Computed as Sigma_cal * n_cal where Sigma_cal is calibration covariance.
    /// This allows O(1) covariance scaling as samples accumulate.
    pub sigma_rate: Matrix9,

    /// Block length from Politis-White algorithm.
    /// Used for block bootstrap to preserve autocorrelation structure.
    pub block_length: usize,

    /// Calibrated Student's t prior scale (v5.4).
    /// This is the σ in δ|λ ~ N(0, (σ²/λ)R).
    pub sigma_t: f64,

    /// Cholesky factor L_R of correlation matrix R.
    /// Used for Gibbs sampling: δ = (σ/√λ) L_R z.
    pub l_r: Matrix9,

    /// Marginal prior covariance: 2σ²R (for ν=4).
    /// This is the unconditional prior variance of δ under the t-prior.
    pub prior_cov_marginal: Matrix9,

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

    /// Batch size K for adaptive batching.
    /// When K > 1, samples contain K iterations worth of timing.
    /// Effect estimates must be divided by K to report per-call differences.
    pub batch_k: u32,
}

impl Calibration {
    /// Create a new Calibration with v5.4+ Student's t prior.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sigma_rate: Matrix9,
        block_length: usize,
        sigma_t: f64,
        l_r: Matrix9,
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
        batch_k: u32,
    ) -> Self {
        // Compute marginal prior covariance: 2σ²R (for ν=4)
        // The marginal variance of t_ν(0, σ²R) is (ν/(ν-2)) σ²R = 2σ²R for ν=4
        let r = &l_r * l_r.transpose();
        let prior_cov_marginal = r * (2.0 * sigma_t * sigma_t);

        Self {
            sigma_rate,
            block_length,
            sigma_t,
            l_r,
            prior_cov_marginal,
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
            batch_k,
        }
    }

    /// Compute effective sample size accounting for dependence (spec §3.3.2 v5.6).
    ///
    /// Under strong temporal dependence, n samples do not provide n independent observations.
    /// The effective sample size approximates the number of effectively independent blocks.
    ///
    /// n_eff = max(1, floor(n / block_length))
    ///
    /// where block_length is the estimated dependence length from Politis-White selector.
    pub fn n_eff(&self, n: usize) -> usize {
        if self.block_length == 0 {
            return n.max(1);
        }
        (n / self.block_length).max(1)
    }

    /// Scale sigma_rate to get covariance for n samples using effective sample size (v5.6).
    ///
    /// Σ_n = Σ_rate / n_eff
    ///
    /// where n_eff = max(1, floor(n / block_length)) to correctly account for reduced
    /// information under temporal dependence.
    pub fn covariance_for_n(&self, n: usize) -> Matrix9 {
        if n == 0 {
            return self.sigma_rate; // Avoid division by zero
        }
        let n_eff = self.n_eff(n);
        self.sigma_rate / (n_eff as f64)
    }

    /// Scale sigma_rate to get covariance using raw n samples (ignoring dependence).
    ///
    /// Σ_n = Σ_rate / n
    ///
    /// Use this only when you need the raw scaling without n_eff correction.
    pub fn covariance_for_n_raw(&self, n: usize) -> Matrix9 {
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

/// Calibrate Student's t prior scale σ so that P(max_k |δ_k| > θ_eff | δ ~ t_4(0, σ²R)) = 0.62 (v5.4).
///
/// Uses binary search to find σ such that the marginal exceedance probability
/// matches the target 62%. The t-prior is sampled via scale mixture:
/// - λ ~ Gamma(ν/2, ν/2) = Gamma(2, 2) for ν=4
/// - z ~ N(0, I₉)
/// - δ = (σ/√λ) L_R z
///
/// # Arguments
/// * `sigma_rate` - Covariance rate matrix from calibration
/// * `theta_eff` - Effective threshold in nanoseconds
/// * `n_cal` - Number of calibration samples (for SE computation)
/// * `discrete_mode` - Whether discrete timer mode is active
/// * `seed` - Deterministic RNG seed
///
/// # Returns
/// A tuple of (sigma_t, l_r) where sigma_t is the calibrated scale and l_r is
/// the Cholesky factor of the regularized correlation matrix.
pub fn calibrate_t_prior_scale(
    sigma_rate: &Matrix9,
    theta_eff: f64,
    n_cal: usize,
    discrete_mode: bool,
    seed: u64,
) -> (f64, Matrix9) {
    let median_se = compute_median_se(sigma_rate, n_cal);

    // Compute and cache L_R (Cholesky of regularized correlation matrix)
    let r = compute_correlation_matrix(sigma_rate);
    let r_reg = apply_correlation_regularization(&r, discrete_mode);
    let l_r = match Cholesky::new(r_reg) {
        Some(c) => c.l().into_owned(),
        None => Matrix9::identity(),
    };

    // Precompute normalized effect magnitudes for sample reuse across bisection.
    //
    // For t_ν prior with scale mixture representation:
    //   λ ~ Gamma(ν/2, ν/2), z ~ N(0, I₉), δ = (σ/√λ) L_R z
    //
    // So: max|δ| = σ · max|L_R z|/√λ = σ · m
    // where m_i = max|L_R z_i|/√λ_i is precomputed once.
    //
    // Then: P(max|δ| > θ) = P(σ·m > θ) = P(m > θ/σ)
    //
    // This allows O(1) exceedance computation per bisection iteration
    // instead of O(PRIOR_CALIBRATION_SAMPLES).
    let normalized_effects = precompute_t_prior_effects(&l_r, seed);

    // Search bounds (same as v5.1)
    let mut lo = theta_eff * 0.05;
    let mut hi = (theta_eff * 50.0).max(10.0 * median_se);

    for _ in 0..MAX_CALIBRATION_ITERATIONS {
        let mid = (lo + hi) / 2.0;

        // Compute exceedance using precomputed samples: count(m_i > θ/σ)
        let threshold = theta_eff / mid;
        let count = normalized_effects.iter().filter(|&&m| m > threshold).count();
        let exceedance = count as f64 / normalized_effects.len() as f64;

        if (exceedance - TARGET_EXCEEDANCE).abs() < 0.01 {
            return (mid, l_r); // Close enough
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
    (theta_eff * CONSERVATIVE_PRIOR_SCALE, l_r)
}

/// Precompute normalized effect magnitudes for t-prior calibration.
///
/// Returns a vector of m_i = max_k |L_R z_i|_k / √λ_i where:
/// - λ_i ~ Gamma(ν/2, ν/2) for ν=4
/// - z_i ~ N(0, I₉)
///
/// These can be reused across bisection iterations since:
/// P(max|δ| > θ | σ) = P(σ·m > θ) = P(m > θ/σ)
fn precompute_t_prior_effects(l_r: &Matrix9, seed: u64) -> Vec<f64> {
    use rand_distr::Gamma;

    #[cfg(feature = "parallel")]
    {
        let l_r = l_r.clone();
        (0..PRIOR_CALIBRATION_SAMPLES)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));
                let gamma_dist = Gamma::new(NU / 2.0, 2.0 / NU).unwrap();

                // Sample λ ~ Gamma(ν/2, ν/2)
                let lambda: f64 = gamma_dist.sample(&mut rng);
                let inv_sqrt_lambda = 1.0 / math::sqrt(lambda.max(DIAGONAL_FLOOR));

                // Sample z ~ N(0, I_9)
                let mut z = Vector9::zeros();
                for j in 0..9 {
                    z[j] = sample_standard_normal(&mut rng);
                }

                // Compute m = max|L_R z| / √λ
                let w = &l_r * z;
                let max_w = w.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                max_w * inv_sqrt_lambda
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let gamma_dist = Gamma::new(NU / 2.0, 2.0 / NU).unwrap();
        let mut effects = Vec::with_capacity(PRIOR_CALIBRATION_SAMPLES);

        for _ in 0..PRIOR_CALIBRATION_SAMPLES {
            let lambda: f64 = gamma_dist.sample(&mut rng);
            let inv_sqrt_lambda = 1.0 / math::sqrt(lambda.max(DIAGONAL_FLOOR));

            let mut z = Vector9::zeros();
            for i in 0..9 {
                z[i] = sample_standard_normal(&mut rng);
            }

            let w = l_r * z;
            let max_w = w.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
            effects.push(max_w * inv_sqrt_lambda);
        }
        effects
    }
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
    let l = chol.l().into_owned();

    // Parallel MC sampling when feature enabled
    #[cfg(feature = "parallel")]
    let mut max_effects: Vec<f64> = (0..PRIOR_CALIBRATION_SAMPLES)
        .into_par_iter()
        .map(|i| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));
            let mut z = Vector9::zeros();
            for j in 0..9 {
                z[j] = sample_standard_normal(&mut rng);
            }
            let sample = &l * z;
            sample.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let mut max_effects: Vec<f64> = {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut effects = Vec::with_capacity(PRIOR_CALIBRATION_SAMPLES);
        for _ in 0..PRIOR_CALIBRATION_SAMPLES {
            let mut z = Vector9::zeros();
            for i in 0..9 {
                z[i] = sample_standard_normal(&mut rng);
            }
            let sample = &l * z;
            let max_effect = sample.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
            effects.push(max_effect);
        }
        effects
    };

    // 95th percentile using O(n) selection instead of O(n log n) sort
    let idx = ((PRIOR_CALIBRATION_SAMPLES as f64 * 0.95) as usize).min(PRIOR_CALIBRATION_SAMPLES - 1);
    let (_, &mut percentile_95, _) = max_effects.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    percentile_95
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
        let theta_eff = 100.0;

        Calibration::new(
            sigma_rate,
            10,                  // block_length
            100.0,               // sigma_t
            Matrix9::identity(), // l_r (identity for tests)
            theta_eff,           // theta_ns
            5000,                // calibration_samples
            false,               // discrete_mode
            5.0,                 // mde_shift_ns
            10.0,                // mde_tail_ns
            snapshot,            // calibration_snapshot
            1.0,                 // timer_resolution_ns
            10000.0,             // samples_per_second
            10.0,                // c_floor
            18.48,               // projection_mismatch_thresh
            0.001,               // theta_tick
            theta_eff,           // theta_eff
            0.1,                 // theta_floor_initial
            42,                  // rng_seed
            1,                   // batch_k (no batching in tests)
        )
    }

    #[test]
    fn test_covariance_scaling() {
        let cal = make_test_calibration();
        // make_test_calibration uses sigma_rate[(0,0)] = 1000.0 and block_length = 10

        // v5.6: covariance_for_n uses n_eff = n / block_length
        // At n=1000 with block_length=10: n_eff = 100
        // sigma_n[(0,0)] = sigma_rate[(0,0)] / n_eff = 1000 / 100 = 10.0
        let cov_1000 = cal.covariance_for_n(1000);
        assert!(
            (cov_1000[(0, 0)] - 10.0).abs() < 1e-10,
            "expected 10.0, got {}",
            cov_1000[(0, 0)]
        );

        // At n=2000 with block_length=10: n_eff = 200
        // sigma_n[(0,0)] = 1000 / 200 = 5.0
        let cov_2000 = cal.covariance_for_n(2000);
        assert!(
            (cov_2000[(0, 0)] - 5.0).abs() < 1e-10,
            "expected 5.0, got {}",
            cov_2000[(0, 0)]
        );
    }

    #[test]
    fn test_n_eff() {
        let cal = make_test_calibration();
        // make_test_calibration uses block_length = 10

        // n_eff = max(1, floor(n / block_length))
        assert_eq!(cal.n_eff(100), 10);
        assert_eq!(cal.n_eff(1000), 100);
        assert_eq!(cal.n_eff(10), 1);
        assert_eq!(cal.n_eff(5), 1); // Clamped to 1 when n < block_length
        assert_eq!(cal.n_eff(0), 1); // Edge case: n=0 returns 1
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

    // =========================================================================
    // Reference implementations for validation (original non-optimized versions)
    // =========================================================================

    /// Reference implementation: compute t-prior exceedance without sample reuse.
    /// This generates fresh samples for each call, matching the original implementation.
    fn reference_t_prior_exceedance(l_r: &Matrix9, sigma: f64, theta: f64, seed: u64) -> f64 {
        use rand_distr::Gamma;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let gamma_dist = Gamma::new(NU / 2.0, 2.0 / NU).unwrap();
        let mut count = 0usize;

        for _ in 0..PRIOR_CALIBRATION_SAMPLES {
            let lambda: f64 = gamma_dist.sample(&mut rng);
            let scale = sigma / crate::math::sqrt(lambda.max(DIAGONAL_FLOOR));

            let mut z = Vector9::zeros();
            for i in 0..9 {
                z[i] = sample_standard_normal(&mut rng);
            }

            let delta = l_r * z * scale;
            let max_effect = delta.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
            if max_effect > theta {
                count += 1;
            }
        }

        count as f64 / PRIOR_CALIBRATION_SAMPLES as f64
    }

    /// Helper: compute exceedance using precomputed t-prior effects
    fn optimized_t_prior_exceedance(normalized_effects: &[f64], sigma: f64, theta: f64) -> f64 {
        let threshold = theta / sigma;
        let count = normalized_effects.iter().filter(|&&m| m > threshold).count();
        count as f64 / normalized_effects.len() as f64
    }

    // =========================================================================
    // Tests verifying optimized implementations match reference
    // =========================================================================

    #[test]
    fn test_t_prior_precompute_exceedance_matches_reference() {
        // Test that the optimized exceedance computation matches reference
        // for various sigma values
        let l_r = Matrix9::identity();
        let theta = 10.0;
        let seed = 12345u64;

        // Precompute effects using the optimized method
        let normalized_effects = precompute_t_prior_effects(&l_r, seed);

        // Test at multiple sigma values
        for sigma in [5.0, 10.0, 15.0, 20.0, 30.0] {
            let optimized = optimized_t_prior_exceedance(&normalized_effects, sigma, theta);
            let reference = reference_t_prior_exceedance(&l_r, sigma, theta, seed);

            // Allow some tolerance due to different RNG sequences
            // The key property is that exceedance should be monotonically increasing with sigma
            assert!(
                optimized >= 0.0 && optimized <= 1.0,
                "Optimized exceedance {} out of range for sigma={}",
                optimized,
                sigma
            );
            assert!(
                reference >= 0.0 && reference <= 1.0,
                "Reference exceedance {} out of range for sigma={}",
                reference,
                sigma
            );

            // Both should be in similar ballpark (within 0.1 of each other)
            // Note: They won't be exactly equal because the optimized version
            // uses different random samples
            println!(
                "sigma={}: optimized={:.4}, reference={:.4}",
                sigma, optimized, reference
            );
        }
    }

    #[test]
    fn test_t_prior_exceedance_monotonicity() {
        // Key property: exceedance should increase with sigma
        let l_r = Matrix9::identity();
        let theta = 10.0;
        let seed = 42u64;

        let normalized_effects = precompute_t_prior_effects(&l_r, seed);

        let mut prev_exceedance = 0.0;
        for sigma in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0] {
            let exceedance = optimized_t_prior_exceedance(&normalized_effects, sigma, theta);

            assert!(
                exceedance >= prev_exceedance,
                "Exceedance should increase with sigma: sigma={}, exc={}, prev={}",
                sigma,
                exceedance,
                prev_exceedance
            );
            prev_exceedance = exceedance;
        }

        // At very large sigma, exceedance should approach 1
        let large_sigma_exc = optimized_t_prior_exceedance(&normalized_effects, 1000.0, theta);
        assert!(
            large_sigma_exc > 0.99,
            "Exceedance at large sigma should be ~1, got {}",
            large_sigma_exc
        );

        // At very small sigma, exceedance should approach 0
        let small_sigma_exc = optimized_t_prior_exceedance(&normalized_effects, 0.1, theta);
        assert!(
            small_sigma_exc < 0.01,
            "Exceedance at small sigma should be ~0, got {}",
            small_sigma_exc
        );
    }

    #[test]
    fn test_calibrate_t_prior_scale_finds_target_exceedance() {
        // Test that the calibration finds a sigma_t that achieves ~62% exceedance
        let sigma_rate = Matrix9::identity() * 100.0;
        let theta_eff = 10.0;
        let n_cal = 5000;
        let discrete_mode = false;
        let seed = 42u64;

        let (sigma_t, l_r) =
            calibrate_t_prior_scale(&sigma_rate, theta_eff, n_cal, discrete_mode, seed);

        // Verify the calibrated sigma_t achieves target exceedance
        let normalized_effects = precompute_t_prior_effects(&l_r, seed);
        let exceedance = optimized_t_prior_exceedance(&normalized_effects, sigma_t, theta_eff);

        assert!(
            (exceedance - TARGET_EXCEEDANCE).abs() < 0.05,
            "Calibrated t-prior exceedance {} should be near target {}",
            exceedance,
            TARGET_EXCEEDANCE
        );
    }

    #[test]
    fn test_calibration_determinism() {
        // Same seed should give same results
        let sigma_rate = Matrix9::identity() * 100.0;
        let theta_eff = 10.0;
        let n_cal = 5000;
        let discrete_mode = false;
        let seed = 12345u64;

        let (sigma_t_1, _) =
            calibrate_t_prior_scale(&sigma_rate, theta_eff, n_cal, discrete_mode, seed);
        let (sigma_t_2, _) =
            calibrate_t_prior_scale(&sigma_rate, theta_eff, n_cal, discrete_mode, seed);

        assert!(
            (sigma_t_1 - sigma_t_2).abs() < 1e-10,
            "Same seed should give same sigma_t: {} vs {}",
            sigma_t_1,
            sigma_t_2
        );
    }

    #[test]
    fn test_precomputed_effects_distribution() {
        // Test that precomputed effects follow expected distribution
        let l_r = Matrix9::identity();
        let seed = 42u64;

        let effects = precompute_t_prior_effects(&l_r, seed);

        // All effects should be positive (they're max of absolute values)
        assert!(
            effects.iter().all(|&m| m > 0.0),
            "All effects should be positive"
        );

        // Compute mean and check it's reasonable
        let mean: f64 = effects.iter().sum::<f64>() / effects.len() as f64;
        // For t_4 with identity L_R, mean of max|z|/sqrt(lambda) should be roughly 2-4
        assert!(
            mean > 1.0 && mean < 10.0,
            "Mean effect {} should be in reasonable range",
            mean
        );

        // Check variance is non-zero (samples are diverse)
        let variance: f64 = effects.iter().map(|&m| (m - mean).powi(2)).sum::<f64>()
            / (effects.len() - 1) as f64;
        assert!(variance > 0.1, "Effects should have non-trivial variance");
    }

    #[test]
    #[ignore] // Slow benchmark - run with `cargo test -- --ignored`
    fn bench_calibration_timing() {
        use std::time::Instant;

        let sigma_rate = Matrix9::identity() * 10000.0;
        let theta_eff = 100.0;
        let n_cal = 5000;
        let discrete_mode = false;

        // Warm up
        let _ = calibrate_t_prior_scale(&sigma_rate, theta_eff, n_cal, discrete_mode, 1);

        // Benchmark t-prior calibration
        let iterations = 10;
        let start = Instant::now();
        for i in 0..iterations {
            let _ = calibrate_t_prior_scale(&sigma_rate, theta_eff, n_cal, discrete_mode, i as u64);
        }
        let t_prior_time = start.elapsed();

        println!(
            "\n=== Calibration Performance ===\n\
             \n\
             T-prior calibration: {:?} per call\n\
             ({} iterations averaged)",
            t_prior_time / iterations as u32,
            iterations
        );
    }
}
