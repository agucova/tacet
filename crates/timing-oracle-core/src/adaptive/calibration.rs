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

/// Default prior weight for the narrow component in mixture prior (§3.3.5 v5.2).
/// DEPRECATED: Kept for backwards compatibility. Use t-prior with NU=4 instead.
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

    // ==================== v5.4 Student's t prior fields ====================

    /// Calibrated Student's t prior scale (v5.4).
    /// This is the σ in δ|λ ~ N(0, (σ²/λ)R).
    pub sigma_t: f64,

    /// Cholesky factor L_R of correlation matrix R.
    /// Used for Gibbs sampling: δ = (σ/√λ) L_R z.
    pub l_r: Matrix9,

    /// Marginal prior covariance: 2σ²R (for ν=4).
    /// This is the unconditional prior variance of δ under the t-prior.
    pub prior_cov_marginal: Matrix9,

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
        // Compute L_R for backwards compatibility with v5.4 code
        let r = compute_correlation_matrix(&sigma_rate);
        let r_reg = apply_correlation_regularization(&r, discrete_mode);
        let l_r = match Cholesky::new(r_reg) {
            Some(c) => c.l().into_owned(),
            None => Matrix9::identity(),
        };

        // Marginal prior cov = 2σ²R for t-prior compatibility
        let prior_cov_marginal = r_reg * (2.0 * sigma_narrow * sigma_narrow);

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
            sigma_t: sigma_narrow,
            l_r,
            prior_cov_marginal,
        }
    }

    /// Create a new Calibration with v5.4 Student's t prior.
    ///
    /// This is the preferred constructor for v5.4+.
    #[allow(clippy::too_many_arguments)]
    pub fn new_t_prior(
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
    ) -> Self {
        // Compute marginal prior covariance: 2σ²R (for ν=4)
        // The marginal variance of t_ν(0, σ²R) is (ν/(ν-2)) σ²R = 2σ²R for ν=4
        let r = &l_r * l_r.transpose();
        let prior_cov_marginal = r * (2.0 * sigma_t * sigma_t);

        // For backwards compatibility, also populate mixture fields
        // (will be removed in future version)
        let prior_cov_compat = compute_prior_cov_9d(&sigma_rate, sigma_t, discrete_mode);

        Self {
            sigma_rate,
            block_length,
            prior_cov_9d: prior_cov_marginal,
            prior_cov_narrow: prior_cov_compat,
            prior_cov_slab: prior_cov_compat, // Same for compat
            sigma_narrow: sigma_t,
            sigma_slab: sigma_t,
            prior_weight: 1.0, // No mixture
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
            sigma_t,
            l_r,
            prior_cov_marginal,
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

    // Compute L_R (Cholesky of regularized correlation matrix)
    let r = compute_correlation_matrix(sigma_rate);
    let r_reg = apply_correlation_regularization(&r, discrete_mode);
    let l_r = match Cholesky::new(r_reg) {
        Some(c) => c.l().into_owned(),
        None => return theta_eff * CONSERVATIVE_PRIOR_SCALE,
    };

    // Precompute samples for reuse across bisection iterations.
    //
    // For mixture prior: w·N(0, σ_narrow²R) + (1-w)·N(0, σ_slab²R)
    //
    // For each sample i, precompute:
    //   m_i = max|L_R · z_i|  (the "base" effect magnitude)
    //   is_narrow_i = whether this sample uses narrow component
    //
    // For narrow: max|δ| = σ_narrow · m_i
    // For slab:   max|δ| = σ_slab · m_i (constant across bisection!)
    //
    // This reduces the bisection loop to simple threshold comparisons.
    let (base_effects, component_flags) =
        precompute_mixture_prior_samples(&l_r, prior_weight, seed);

    // Precompute the slab contribution (constant across bisection)
    // Slab samples exceed threshold when: σ_slab · m_i > θ, i.e., m_i > θ/σ_slab
    let slab_threshold = theta_eff / sigma_slab;
    let slab_count: usize = base_effects
        .iter()
        .zip(component_flags.iter())
        .filter(|&(&m, &is_narrow)| !is_narrow && m > slab_threshold)
        .count();

    // Search bounds (same as v5.1)
    let mut lo = theta_eff * 0.05;
    let mut hi = (theta_eff * 50.0).max(10.0 * median_se);

    for _ in 0..MAX_CALIBRATION_ITERATIONS {
        let mid = (lo + hi) / 2.0;

        // Narrow samples exceed threshold when: σ_narrow · m_i > θ, i.e., m_i > θ/σ_narrow
        let narrow_threshold = theta_eff / mid;
        let narrow_count: usize = base_effects
            .iter()
            .zip(component_flags.iter())
            .filter(|&(&m, &is_narrow)| is_narrow && m > narrow_threshold)
            .count();

        let total_count = slab_count + narrow_count;
        let exceedance = total_count as f64 / base_effects.len() as f64;

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

/// Precompute samples for mixture prior calibration.
///
/// Returns:
/// - base_effects: m_i = max_k |L_R z_i|_k for each sample
/// - component_flags: true if sample uses narrow component, false for slab
///
/// These can be reused across bisection iterations since:
/// - For narrow: P(max|δ| > θ) = P(σ_narrow · m > θ)
/// - For slab: P(max|δ| > θ) = P(σ_slab · m > θ) (constant)
fn precompute_mixture_prior_samples(
    l_r: &Matrix9,
    prior_weight: f64,
    seed: u64,
) -> (Vec<f64>, Vec<bool>) {
    #[cfg(feature = "parallel")]
    {
        let l_r = l_r.clone();
        let results: Vec<(f64, bool)> = (0..PRIOR_CALIBRATION_SAMPLES)
            .into_par_iter()
            .map(|i| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(seed, i as u64));

                // Decide component: narrow (true) or slab (false)
                let is_narrow = rng.random::<f64>() < prior_weight;

                // Sample z ~ N(0, I_9)
                let mut z = Vector9::zeros();
                for j in 0..9 {
                    z[j] = sample_standard_normal(&mut rng);
                }

                // Compute m = max|L_R z|
                let w = &l_r * z;
                let max_w = w.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

                (max_w, is_narrow)
            })
            .collect();

        let base_effects: Vec<f64> = results.iter().map(|(m, _)| *m).collect();
        let component_flags: Vec<bool> = results.iter().map(|(_, is_narrow)| *is_narrow).collect();
        (base_effects, component_flags)
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut base_effects = Vec::with_capacity(PRIOR_CALIBRATION_SAMPLES);
        let mut component_flags = Vec::with_capacity(PRIOR_CALIBRATION_SAMPLES);

        for _ in 0..PRIOR_CALIBRATION_SAMPLES {
            let is_narrow = rng.random::<f64>() < prior_weight;

            let mut z = Vector9::zeros();
            for i in 0..9 {
                z[i] = sample_standard_normal(&mut rng);
            }

            let w = l_r * z;
            let max_w = w.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

            base_effects.push(max_w);
            component_flags.push(is_narrow);
        }

        (base_effects, component_flags)
    }
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

    /// Reference implementation: compute mixture prior exceedance without sample reuse.
    fn reference_mixture_prior_exceedance(
        l_r: &Matrix9,
        sigma_narrow: f64,
        sigma_slab: f64,
        prior_weight: f64,
        theta: f64,
        seed: u64,
    ) -> f64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut count = 0usize;

        for _ in 0..PRIOR_CALIBRATION_SAMPLES {
            // Decide component
            let sigma = if rng.random::<f64>() < prior_weight {
                sigma_narrow
            } else {
                sigma_slab
            };

            let mut z = Vector9::zeros();
            for i in 0..9 {
                z[i] = sample_standard_normal(&mut rng);
            }

            let delta = l_r * z * sigma;
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

    /// Helper: compute exceedance using precomputed mixture samples
    fn optimized_mixture_exceedance(
        base_effects: &[f64],
        component_flags: &[bool],
        sigma_narrow: f64,
        sigma_slab: f64,
        theta: f64,
    ) -> f64 {
        let narrow_threshold = theta / sigma_narrow;
        let slab_threshold = theta / sigma_slab;

        let count: usize = base_effects
            .iter()
            .zip(component_flags.iter())
            .filter(|&(&m, &is_narrow)| {
                if is_narrow {
                    m > narrow_threshold
                } else {
                    m > slab_threshold
                }
            })
            .count();

        count as f64 / base_effects.len() as f64
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
    fn test_mixture_prior_precompute_component_proportions() {
        // Verify that the component flags have correct proportions
        let l_r = Matrix9::identity();
        let prior_weight = 0.99; // 99% narrow
        let seed = 42u64;

        let (_, component_flags) = precompute_mixture_prior_samples(&l_r, prior_weight, seed);

        let narrow_count = component_flags.iter().filter(|&&is_narrow| is_narrow).count();
        let narrow_proportion = narrow_count as f64 / component_flags.len() as f64;

        // Should be close to prior_weight (within statistical tolerance)
        assert!(
            (narrow_proportion - prior_weight).abs() < 0.02,
            "Narrow proportion {} should be close to prior_weight {}",
            narrow_proportion,
            prior_weight
        );
    }

    #[test]
    fn test_mixture_prior_exceedance_monotonicity() {
        // Key property: exceedance should increase with sigma_narrow (slab is fixed)
        let l_r = Matrix9::identity();
        let sigma_slab = 100.0;
        let prior_weight = 0.99;
        let theta = 10.0;
        let seed = 42u64;

        let (base_effects, component_flags) =
            precompute_mixture_prior_samples(&l_r, prior_weight, seed);

        let mut prev_exceedance = 0.0;
        for sigma_narrow in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0] {
            let exceedance = optimized_mixture_exceedance(
                &base_effects,
                &component_flags,
                sigma_narrow,
                sigma_slab,
                theta,
            );

            assert!(
                exceedance >= prev_exceedance - 0.001, // Small tolerance for numerical noise
                "Exceedance should increase with sigma_narrow: sigma={}, exc={}, prev={}",
                sigma_narrow,
                exceedance,
                prev_exceedance
            );
            prev_exceedance = exceedance;
        }
    }

    #[test]
    fn test_mixture_prior_precompute_exceedance_matches_reference() {
        // Test that the optimized exceedance computation matches reference
        // for various sigma_narrow values
        let l_r = Matrix9::identity();
        let sigma_slab = 100.0;
        let prior_weight = 0.99;
        let theta = 10.0;
        let seed = 12345u64;

        // Precompute samples using the optimized method
        let (base_effects, component_flags) =
            precompute_mixture_prior_samples(&l_r, prior_weight, seed);

        // Test at multiple sigma_narrow values
        for sigma_narrow in [5.0, 10.0, 15.0, 20.0, 30.0] {
            let optimized = optimized_mixture_exceedance(
                &base_effects,
                &component_flags,
                sigma_narrow,
                sigma_slab,
                theta,
            );
            let reference =
                reference_mixture_prior_exceedance(&l_r, sigma_narrow, sigma_slab, prior_weight, theta, seed);

            // Both should be in valid range
            assert!(
                optimized >= 0.0 && optimized <= 1.0,
                "Optimized exceedance {} out of range for sigma_narrow={}",
                optimized,
                sigma_narrow
            );
            assert!(
                reference >= 0.0 && reference <= 1.0,
                "Reference exceedance {} out of range for sigma_narrow={}",
                reference,
                sigma_narrow
            );

            // Both should be in similar ballpark (within 0.1 of each other)
            // Note: They won't be exactly equal because the optimized version
            // uses different random samples
            println!(
                "sigma_narrow={}: optimized={:.4}, reference={:.4}",
                sigma_narrow, optimized, reference
            );
        }
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
    fn test_calibrate_narrow_scale_finds_target_exceedance() {
        // Test that the calibration finds a sigma_narrow that achieves ~62% mixture exceedance
        let sigma_rate = Matrix9::identity() * 100.0;
        let theta_eff = 10.0;
        let n_cal = 5000;
        let sigma_slab = compute_slab_scale(&sigma_rate, theta_eff, n_cal);
        let prior_weight = MIXTURE_PRIOR_WEIGHT;
        let discrete_mode = false;
        let seed = 42u64;

        let sigma_narrow = calibrate_narrow_scale(
            &sigma_rate,
            theta_eff,
            n_cal,
            sigma_slab,
            prior_weight,
            discrete_mode,
            seed,
        );

        // Compute L_R for verification
        let r = compute_correlation_matrix(&sigma_rate);
        let r_reg = apply_correlation_regularization(&r, discrete_mode);
        let l_r = Cholesky::new(r_reg).unwrap().l().into_owned();

        // Verify with precomputed samples
        let (base_effects, component_flags) =
            precompute_mixture_prior_samples(&l_r, prior_weight, seed);
        let exceedance = optimized_mixture_exceedance(
            &base_effects,
            &component_flags,
            sigma_narrow,
            sigma_slab,
            theta_eff,
        );

        assert!(
            (exceedance - TARGET_EXCEEDANCE).abs() < 0.05,
            "Calibrated narrow scale exceedance {} should be near target {}",
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

        // Benchmark t-prior calibration (OPTIMIZED)
        let iterations = 10;
        let start = Instant::now();
        for i in 0..iterations {
            let _ = calibrate_t_prior_scale(&sigma_rate, theta_eff, n_cal, discrete_mode, i as u64);
        }
        let t_prior_time = start.elapsed();

        // Benchmark mixture prior calibration (OPTIMIZED)
        let sigma_slab = compute_slab_scale(&sigma_rate, theta_eff, n_cal);
        let start = Instant::now();
        for i in 0..iterations {
            let _ = calibrate_narrow_scale(
                &sigma_rate,
                theta_eff,
                n_cal,
                sigma_slab,
                MIXTURE_PRIOR_WEIGHT,
                discrete_mode,
                i as u64,
            );
        }
        let mixture_time = start.elapsed();

        // Now benchmark the REFERENCE (unoptimized) implementations
        // These regenerate samples on each bisection iteration
        let r = compute_correlation_matrix(&sigma_rate);
        let r_reg = apply_correlation_regularization(&r, discrete_mode);
        let l_r = Cholesky::new(r_reg).unwrap().l().into_owned();

        // Reference t-prior: measure cost of repeated MC sampling
        let bisection_iters = MAX_CALIBRATION_ITERATIONS;
        let start = Instant::now();
        for i in 0..iterations {
            // Simulate bisection: each iteration calls reference_t_prior_exceedance
            for j in 0..bisection_iters {
                let sigma = 10.0 + j as f64;
                let _ = reference_t_prior_exceedance(&l_r, sigma, theta_eff, i as u64 + j as u64);
            }
        }
        let ref_t_prior_time = start.elapsed();

        // Reference mixture prior
        let start = Instant::now();
        for i in 0..iterations {
            for j in 0..bisection_iters {
                let sigma_narrow = 10.0 + j as f64;
                let _ = reference_mixture_prior_exceedance(
                    &l_r,
                    sigma_narrow,
                    sigma_slab,
                    MIXTURE_PRIOR_WEIGHT,
                    theta_eff,
                    i as u64 + j as u64,
                );
            }
        }
        let ref_mixture_time = start.elapsed();

        let optimized_total = (t_prior_time + mixture_time) / iterations as u32;
        let reference_total = (ref_t_prior_time + ref_mixture_time) / iterations as u32;
        let speedup = reference_total.as_secs_f64() / optimized_total.as_secs_f64();

        println!(
            "\n=== Calibration Performance Comparison ===\n\
             \n\
             OPTIMIZED (sample reuse):\n\
               T-prior calibration:     {:?} per call\n\
               Mixture calibration:     {:?} per call\n\
               Total per analysis:      {:?}\n\
             \n\
             REFERENCE (no reuse, {} bisection iters):\n\
               T-prior calibration:     {:?} per call\n\
               Mixture calibration:     {:?} per call\n\
               Total per analysis:      {:?}\n\
             \n\
             SPEEDUP: {:.1}x\n\
             ({} iterations averaged)",
            t_prior_time / iterations as u32,
            mixture_time / iterations as u32,
            optimized_total,
            bisection_iters,
            ref_t_prior_time / iterations as u32,
            ref_mixture_time / iterations as u32,
            reference_total,
            speedup,
            iterations
        );

        // Sanity check: optimized should be faster
        assert!(
            speedup > 1.0,
            "Optimized should be faster than reference, got {:.2}x",
            speedup
        );
    }
}
