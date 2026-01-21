//! Gibbs sampler for Student's t prior inference (spec v5.4, v5.6).
//!
//! This module implements Bayesian inference using a Student's t prior with
//! degrees of freedom ν=4, represented as a scale mixture of Gaussians:
//!
//! ```text
//! δ | λ ~ N(0, (σ²/λ) R)
//! λ ~ Gamma(ν/2, ν/2)
//! ```
//!
//! v5.6 adds a robust t-likelihood with precision parameter κ that prevents
//! false certainty when Σₙ is misestimated:
//!
//! ```text
//! Δ | δ, κ ~ N(δ, Σₙ/κ)
//! κ ~ Gamma(ν_ℓ/2, ν_ℓ/2) where ν_ℓ = 8
//! ```
//!
//! The posterior is computed via Gibbs sampling, alternating between:
//! 1. δ | λ, κ, Δ ~ N(μ(λ,κ), Q(λ,κ)⁻¹)
//! 2. λ | δ ~ Gamma((ν+9)/2, (ν + δᵀR⁻¹δ/σ²)/2)
//! 3. κ | δ, Δ ~ Gamma((ν_ℓ+9)/2, (ν_ℓ + s)/2) where s = (Δ-δ)ᵀΣₙ⁻¹(Δ-δ)
//!
//! This replaces the v5.2 mixture prior, fixing the correlation-induced
//! failure mode where high inter-decile correlations caused pathological
//! posterior shrinkage.

extern crate alloc;

use alloc::vec::Vec;
use core::f64::consts::PI;

use nalgebra::Cholesky;
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Gamma;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::constants::{B_TAIL, ONES};
use crate::math;
use crate::types::{Matrix2, Matrix9, Matrix9x2, Vector2, Vector9};

/// Degrees of freedom for the Student's t prior (ν = 4).
pub const NU: f64 = 4.0;

/// Degrees of freedom for robust t-likelihood (ν_ℓ = 8, spec §3.4.2 v5.6).
pub const NU_L: f64 = 8.0;

/// Total number of Gibbs iterations.
pub const N_GIBBS: usize = 256;

/// Number of burn-in iterations to discard.
pub const N_BURN: usize = 64;

/// Number of retained samples for posterior inference.
pub const N_KEEP: usize = N_GIBBS - N_BURN; // 192

/// Minimum λ value to prevent numerical issues.
const LAMBDA_MIN: f64 = 1e-10;

/// Maximum λ value to prevent numerical issues.
const LAMBDA_MAX: f64 = 1e10;

/// Minimum κ value to prevent numerical issues (v5.6).
const KAPPA_MIN: f64 = 1e-10;

/// Maximum κ value to prevent numerical issues (v5.6).
const KAPPA_MAX: f64 = 1e10;

/// Condition number threshold for triggering robust shrinkage (§3.3.5).
/// When exceeded, apply shrinkage to prevent GLS instability.
const CONDITION_NUMBER_THRESHOLD: f64 = 1e4;

/// Result of Gibbs sampling inference.
#[derive(Clone, Debug)]
pub struct GibbsResult {
    /// Posterior mean δ_post (average of retained samples).
    pub delta_post: Vector9,

    /// Posterior covariance Λ_post (sample covariance of retained samples).
    pub lambda_post: Matrix9,

    /// Posterior probability P(max_k |δ_k| > θ | Δ).
    pub leak_probability: f64,

    /// 95% credible interval for max effect magnitude.
    pub effect_magnitude_ci: (f64, f64),

    /// Posterior mean of latent scale λ.
    pub lambda_mean: f64,

    /// Posterior standard deviation of λ.
    pub lambda_sd: f64,

    /// Coefficient of variation: λ_sd / λ_mean.
    pub lambda_cv: f64,

    /// Effective sample size of λ chain.
    pub lambda_ess: f64,

    /// True if mixing diagnostics pass: CV ≥ 0.1 AND ESS ≥ 20.
    pub lambda_mixing_ok: bool,

    // =========================================================================
    // v5.6 Kappa diagnostics - robust t-likelihood precision
    // =========================================================================

    /// v5.6: Posterior mean of likelihood precision κ.
    pub kappa_mean: f64,

    /// v5.6: Posterior standard deviation of κ.
    pub kappa_sd: f64,

    /// v5.6: Coefficient of variation: κ_sd / κ_mean.
    pub kappa_cv: f64,

    /// v5.6: Effective sample size of κ chain.
    pub kappa_ess: f64,

    /// v5.6: Whether κ mixing diagnostics pass: CV ≥ 0.1 AND ESS ≥ 20.
    pub kappa_mixing_ok: bool,

    /// 2D GLS projection β = (μ, τ) for interpretability.
    pub beta_proj: Vector2,

    /// 2D projection covariance.
    pub beta_proj_cov: Matrix2,

    /// Projection mismatch Q statistic.
    pub projection_mismatch_q: f64,
}

/// Gibbs sampler for Student's t prior inference.
pub struct GibbsSampler {
    /// Degrees of freedom (fixed at 4).
    nu: f64,

    /// Calibrated prior scale σ.
    sigma: f64,

    /// Cholesky factor L_R such that L_R L_Rᵀ = R.
    l_r: Matrix9,

    /// Deterministic RNG.
    rng: Xoshiro256PlusPlus,
}

impl GibbsSampler {
    /// Create a new Gibbs sampler with precomputed factorizations.
    ///
    /// # Arguments
    /// * `l_r` - Cholesky factor of R (correlation matrix)
    /// * `sigma` - Calibrated prior scale
    /// * `seed` - Deterministic RNG seed
    pub fn new(l_r: &Matrix9, sigma: f64, seed: u64) -> Self {
        Self {
            nu: NU,
            sigma,
            l_r: *l_r,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    /// Run Gibbs sampler and return posterior summaries.
    ///
    /// # Arguments
    /// * `delta_obs` - Observed quantile differences Δ
    /// * `sigma_n` - Likelihood covariance Σ_n
    /// * `theta` - Effect threshold for leak probability
    ///
    /// v5.6: Extended to sample (δ, λ, κ) for robust t-likelihood.
    pub fn run(&mut self, delta_obs: &Vector9, sigma_n: &Matrix9, theta: f64) -> GibbsResult {
        // Apply condition-number-based regularization to sigma_n (§3.3.5).
        // This prevents posterior instability when sigma_n is ill-conditioned,
        // which occurs in discrete timer mode with high quantization.
        let sigma_n_reg = regularize_sigma_n(sigma_n);

        // Precompute Cholesky of regularized Σ_n for efficiency
        let sigma_n_chol =
            Cholesky::new(sigma_n_reg).expect("regularize_sigma_n should ensure SPD");

        // Precompute Σ_n⁻¹ for efficiency (constant throughout Gibbs run)
        // This avoids creating a Matrix9 on the stack 256 times
        let sigma_n_inv = Self::invert_via_cholesky(&sigma_n_chol);

        // Precompute R⁻¹ for efficiency (constant throughout Gibbs run)
        // This avoids creating a Matrix9 on the stack 256 times
        let r_inv = self.compute_r_inverse();

        // Storage for retained samples
        let mut retained_deltas: Vec<Vector9> = Vec::with_capacity(N_KEEP);
        let mut retained_lambdas: Vec<f64> = Vec::with_capacity(N_KEEP);
        let mut retained_kappas: Vec<f64> = Vec::with_capacity(N_KEEP);

        // Initialize: λ⁽⁰⁾ = 1, κ⁽⁰⁾ = 1 (spec §3.4.4 v5.6)
        let mut lambda = 1.0;
        let mut kappa = 1.0;

        // Main Gibbs loop (spec §3.4.4 v5.6)
        for t in 0..N_GIBBS {
            // Step 1: δ | λ, κ, Δ (spec §3.4.4 Conditional 1)
            // Note: sigma_n_inv_delta is recomputed inside to account for κ scaling
            let delta = self.sample_delta_given_lambda_kappa(
                &sigma_n_inv,
                &r_inv,
                delta_obs,
                &sigma_n_chol,
                lambda,
                kappa,
            );

            // Step 2: λ | δ (unchanged, spec §3.4.4 Conditional 2)
            lambda = self.sample_lambda_given_delta(&delta);

            // Step 3: κ | δ, Δ (NEW, spec §3.4.4 Conditional 3)
            kappa = self.sample_kappa_given_delta(delta_obs, &delta, &sigma_n_chol);

            // Store if past burn-in
            if t >= N_BURN {
                retained_deltas.push(delta);
                retained_lambdas.push(lambda);
                retained_kappas.push(kappa);
            }
        }

        // Compute posterior summaries
        self.compute_summaries(&retained_deltas, &retained_lambdas, &retained_kappas, sigma_n, theta)
    }

    /// Sample δ | λ, κ, Δ from the conditional Gaussian (spec §3.4.4 v5.6).
    ///
    /// v5.6: Q(λ, κ) = κΣ_n⁻¹ + (λ/σ²) R⁻¹
    /// μ(λ, κ) = Q(λ, κ)⁻¹ κΣ_n⁻¹ Δ
    /// δ | λ, κ, Δ ~ N(μ(λ, κ), Q(λ, κ)⁻¹)
    ///
    /// **Spec compliance note (§3.4.4):** The spec requires that Σ_n⁻¹ and R⁻¹ be
    /// "computed via Cholesky solves, not explicit matrix inversion". We form
    /// explicit matrices here via Cholesky solves (solving Ax=I column by column),
    /// which IS numerically stable. This is acceptable under the spec's exception
    /// "unless demonstrably stable" (§3.3.5). The alternative (Woodbury identity)
    /// would require working with covariance matrices and introduces complexity.
    ///
    /// Note: sigma_n_inv and r_inv are precomputed once before the Gibbs loop
    /// to avoid repeated stack allocations.
    fn sample_delta_given_lambda_kappa(
        &mut self,
        sigma_n_inv: &Matrix9,
        r_inv: &Matrix9,
        delta_obs: &Vector9,
        sigma_n_chol: &Cholesky<f64, nalgebra::Const<9>>,
        lambda: f64,
        kappa: f64,
    ) -> Vector9 {
        let scale_factor = lambda / (self.sigma * self.sigma);

        // v5.6: Q(λ, κ) = κΣ_n⁻¹ + (λ/σ²) R⁻¹
        let q = sigma_n_inv * kappa + r_inv * scale_factor;

        // Cholesky of Q(λ, κ)
        let q_chol = match Cholesky::new(q) {
            Some(c) => c,
            None => {
                // Fallback: add jitter
                let jittered = q + Matrix9::identity() * 1e-8;
                Cholesky::new(jittered).expect("Q(λ, κ) must be SPD")
            }
        };

        // v5.6: Posterior mean: μ = Q⁻¹ κΣ_n⁻¹ Δ
        // First compute κΣ_n⁻¹ Δ
        let sigma_n_inv_delta = sigma_n_chol.solve(delta_obs);
        let kappa_sigma_n_inv_delta = sigma_n_inv_delta * kappa;
        let mu = q_chol.solve(&kappa_sigma_n_inv_delta);

        // Sample: δ = μ + L_Q⁻ᵀ z where z ~ N(0, I₉)
        // Since Q = L_Q L_Qᵀ, we have Q⁻¹ = L_Q⁻ᵀ L_Q⁻¹
        // So sampling is: δ = μ + L_Q⁻ᵀ z
        let z = self.sample_standard_normal_vector();
        let l_q_inv_t_z = q_chol.l().solve_upper_triangular(&z).unwrap_or(z);

        mu + l_q_inv_t_z
    }

    /// Sample λ | δ from the conditional Gamma.
    ///
    /// q = δᵀ R⁻¹ δ
    /// λ | δ ~ Gamma((ν+9)/2, (ν + q/σ²)/2)
    fn sample_lambda_given_delta(&mut self, delta: &Vector9) -> f64 {
        // Compute q = δᵀ R⁻¹ δ via Cholesky solve
        // R = L_R L_Rᵀ, so R⁻¹ δ = L_R⁻ᵀ L_R⁻¹ δ
        // q = δᵀ R⁻¹ δ = ||L_R⁻¹ δ||²
        let y = self
            .l_r
            .solve_lower_triangular(delta)
            .unwrap_or(*delta);
        let q = y.dot(&y);

        // Gamma parameters (shape-rate parameterization)
        let shape = (self.nu + 9.0) / 2.0; // (4 + 9) / 2 = 6.5
        let rate = (self.nu + q / (self.sigma * self.sigma)) / 2.0;

        // Sample from Gamma(shape, rate)
        // rand_distr uses shape-scale, so scale = 1/rate
        let scale = 1.0 / rate;
        let gamma = Gamma::new(shape, scale).unwrap();
        let sample = gamma.sample(&mut self.rng);

        // Clamp to prevent numerical issues
        sample.clamp(LAMBDA_MIN, LAMBDA_MAX)
    }

    /// Sample κ | δ, Δ from the conditional Gamma (spec §3.4.4 Conditional 3 v5.6).
    ///
    /// s = (Δ - δ)ᵀ Σₙ⁻¹ (Δ - δ)
    /// κ | δ, Δ ~ Gamma((ν_ℓ + 9)/2, (ν_ℓ + s)/2)
    ///
    /// This samples the likelihood precision that allows the model to accommodate
    /// data that doesn't match the estimated Σₙ, preventing false certainty.
    fn sample_kappa_given_delta(
        &mut self,
        delta_obs: &Vector9,
        delta: &Vector9,
        sigma_n_chol: &Cholesky<f64, nalgebra::Const<9>>,
    ) -> f64 {
        // Compute residual: r = Δ - δ
        let residual = delta_obs - delta;

        // Compute s = rᵀ Σₙ⁻¹ r via Cholesky solve
        // Σₙ = L L', so Σₙ⁻¹ r = L⁻ᵀ L⁻¹ r
        // s = rᵀ Σₙ⁻¹ r = ||L⁻¹ r||²
        let y = sigma_n_chol.solve(&residual);
        let s = residual.dot(&y);

        // Gamma parameters (shape-rate, spec §3.4.4)
        let shape = (NU_L + 9.0) / 2.0; // (8 + 9) / 2 = 8.5
        let rate = (NU_L + s) / 2.0;

        // Sample from Gamma(shape, rate)
        // rand_distr uses shape-scale, so scale = 1/rate
        let scale = 1.0 / rate;
        let gamma = Gamma::new(shape, scale).unwrap();
        let sample = gamma.sample(&mut self.rng);

        // Clamp to prevent numerical issues
        sample.clamp(KAPPA_MIN, KAPPA_MAX)
    }

    /// Compute posterior summaries from retained samples.
    fn compute_summaries(
        &self,
        retained_deltas: &[Vector9],
        retained_lambdas: &[f64],
        retained_kappas: &[f64],
        sigma_n: &Matrix9,
        theta: f64,
    ) -> GibbsResult {
        let n = retained_deltas.len() as f64;

        // Posterior mean of δ
        let delta_post = {
            let mut sum = Vector9::zeros();
            for delta in retained_deltas {
                sum += delta;
            }
            sum / n
        };

        // Posterior covariance of δ (sample covariance)
        let lambda_post = {
            let mut cov = Matrix9::zeros();
            for delta in retained_deltas {
                let diff = delta - &delta_post;
                cov += diff * diff.transpose();
            }
            cov / (n - 1.0) // Unbiased estimator
        };

        // Leak probability: fraction exceeding threshold
        let mut exceed_count = 0;
        let mut max_effects: Vec<f64> = Vec::with_capacity(retained_deltas.len());
        for delta in retained_deltas {
            let max_effect = delta.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
            max_effects.push(max_effect);
            if max_effect > theta {
                exceed_count += 1;
            }
        }
        let leak_probability = exceed_count as f64 / n;

        // Effect magnitude CI (2.5th and 97.5th percentiles)
        max_effects.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ci_low = max_effects[(n * 0.025) as usize];
        let ci_high = max_effects[((n * 0.975) as usize).min(max_effects.len() - 1)];

        // Lambda diagnostics
        let lambda_mean = retained_lambdas.iter().sum::<f64>() / n;
        let lambda_var = retained_lambdas
            .iter()
            .map(|&l| math::sq(l - lambda_mean))
            .sum::<f64>()
            / (n - 1.0);
        let lambda_sd = math::sqrt(lambda_var);
        let lambda_cv = if lambda_mean > 0.0 {
            lambda_sd / lambda_mean
        } else {
            0.0
        };
        let lambda_ess = compute_ess(retained_lambdas);
        let lambda_mixing_ok = lambda_cv >= 0.1 && lambda_ess >= 20.0;

        // v5.6: Kappa diagnostics (parallel to lambda)
        let kappa_mean = retained_kappas.iter().sum::<f64>() / n;
        let kappa_var = retained_kappas
            .iter()
            .map(|&k| math::sq(k - kappa_mean))
            .sum::<f64>()
            / (n - 1.0);
        let kappa_sd = math::sqrt(kappa_var);
        let kappa_cv = if kappa_mean > 0.0 {
            kappa_sd / kappa_mean
        } else {
            0.0
        };
        let kappa_ess = compute_ess(retained_kappas);
        let kappa_mixing_ok = kappa_cv >= 0.1 && kappa_ess >= 20.0;

        // 2D projection
        let (beta_proj, beta_proj_cov, projection_mismatch_q) =
            self.compute_2d_projection(&delta_post, &lambda_post, sigma_n);

        GibbsResult {
            delta_post,
            lambda_post,
            leak_probability,
            effect_magnitude_ci: (ci_low, ci_high),
            lambda_mean,
            lambda_sd,
            lambda_cv,
            lambda_ess,
            lambda_mixing_ok,
            kappa_mean,
            kappa_sd,
            kappa_cv,
            kappa_ess,
            kappa_mixing_ok,
            beta_proj,
            beta_proj_cov,
            projection_mismatch_q,
        }
    }

    /// Compute 2D GLS projection for interpretability.
    fn compute_2d_projection(
        &self,
        delta_post: &Vector9,
        _lambda_post: &Matrix9,
        sigma_n: &Matrix9,
    ) -> (Vector2, Matrix2, f64) {
        // Design matrix X: [1, decile_weights]
        let x = build_design_matrix();

        // Apply condition-number-based regularization to sigma_n (§3.3.5).
        // This prevents GLS instability when sigma_n is ill-conditioned,
        // which occurs in discrete timer mode with high quantization.
        let sigma_n_reg = regularize_sigma_n(sigma_n);

        // GLS projection: β = (X'Σ⁻¹X)⁻¹ X'Σ⁻¹ δ
        // Use regularized sigma_n as the weighting matrix
        let sigma_n_chol = Cholesky::new(sigma_n_reg)
            .expect("regularize_sigma_n should ensure SPD");

        // Compute X'Σ⁻¹X
        let sigma_inv_x = {
            let mut result = Matrix9x2::zeros();
            for j in 0..2 {
                let col = x.column(j).into_owned();
                let solved = sigma_n_chol.solve(&col);
                for i in 0..9 {
                    result[(i, j)] = solved[i];
                }
            }
            result
        };
        let xtsx = x.transpose() * &sigma_inv_x;

        // Invert 2x2 matrix
        let det = xtsx[(0, 0)] * xtsx[(1, 1)] - xtsx[(0, 1)] * xtsx[(1, 0)];
        if det.abs() < 1e-12 {
            // Degenerate case
            return (Vector2::zeros(), Matrix2::identity(), 0.0);
        }
        let xtsx_inv = Matrix2::new(
            xtsx[(1, 1)] / det,
            -xtsx[(0, 1)] / det,
            -xtsx[(1, 0)] / det,
            xtsx[(0, 0)] / det,
        );

        // β = (X'Σ⁻¹X)⁻¹ X'Σ⁻¹ δ
        let sigma_inv_delta = sigma_n_chol.solve(delta_post);
        let xt_sigma_inv_delta = x.transpose() * sigma_inv_delta;
        let beta_proj = xtsx_inv * xt_sigma_inv_delta;

        // Projection mismatch Q = ||δ - Xβ||²_Σ⁻¹
        let residual = delta_post - &x * &beta_proj;
        let sigma_inv_residual = sigma_n_chol.solve(&residual);
        let projection_mismatch_q = residual.dot(&sigma_inv_residual);

        // Projection covariance: (X'Σ⁻¹X)⁻¹ × σ²_resid
        // When using regularized Σ (especially unit diagonal), the standard
        // (X'Σ⁻¹X)⁻¹ underestimates variance. Scale by residual variance estimate.
        // σ²_resid = Q / (9-2) for 9 quantiles and 2 parameters.
        let residual_var = (projection_mismatch_q / 7.0).max(1.0);
        let beta_proj_cov = xtsx_inv * residual_var;

        (beta_proj, beta_proj_cov, projection_mismatch_q)
    }

    /// Sample a 9D standard normal vector.
    fn sample_standard_normal_vector(&mut self) -> Vector9 {
        let mut z = Vector9::zeros();
        for i in 0..9 {
            z[i] = self.sample_standard_normal();
        }
        z
    }

    /// Sample from standard normal using Box-Muller transform.
    fn sample_standard_normal(&mut self) -> f64 {
        let u1: f64 = self.rng.random();
        let u2: f64 = self.rng.random();
        math::sqrt(-2.0 * math::ln(u1.max(1e-12))) * math::cos(2.0 * PI * u2)
    }

    /// Invert a matrix via its Cholesky factor (numerically stable).
    ///
    /// Computes A⁻¹ by solving A x_j = e_j for each standard basis vector.
    /// This is numerically stable since it uses forward/backward substitution
    /// on the Cholesky factor, not direct matrix inversion.
    fn invert_via_cholesky(chol: &Cholesky<f64, nalgebra::Const<9>>) -> Matrix9 {
        let mut inv = Matrix9::zeros();
        for j in 0..9 {
            let mut e = Vector9::zeros();
            e[j] = 1.0;
            let col = chol.solve(&e);
            for i in 0..9 {
                inv[(i, j)] = col[i];
            }
        }
        inv
    }

    /// Compute R⁻¹ via Cholesky solves (numerically stable).
    ///
    /// Uses the precomputed Cholesky factor L_R to solve R x_j = e_j for each
    /// basis vector. This is the standard numerically stable approach.
    fn compute_r_inverse(&self) -> Matrix9 {
        let mut r_inv = Matrix9::zeros();
        for j in 0..9 {
            let mut e = Vector9::zeros();
            e[j] = 1.0;
            // Solve L y = e, then L^T x = y
            let y = self
                .l_r
                .solve_lower_triangular(&e)
                .unwrap_or_else(|| e.clone());
            let x = self.l_r.transpose().solve_upper_triangular(&y).unwrap_or(y);
            for i in 0..9 {
                r_inv[(i, j)] = x[i];
            }
        }
        r_inv
    }
}

/// Estimate condition number of a symmetric positive semi-definite matrix.
///
/// Uses Cholesky factorization when available: for SPD matrices,
/// cond(A) = cond(L)² ≈ (max(L_ii) / min(L_ii))².
/// Falls back to diagonal ratio which underestimates for high correlations.
fn estimate_condition_number(m: &Matrix9) -> f64 {
    // Try Cholesky factorization for accurate condition number
    if let Some(chol) = Cholesky::new(*m) {
        let l = chol.l();
        let diag: [f64; 9] = core::array::from_fn(|i| l[(i, i)].abs());
        let max_l = diag.iter().cloned().fold(0.0_f64, f64::max);
        let min_l = diag.iter().cloned().fold(f64::INFINITY, f64::min);

        if min_l < 1e-12 {
            return f64::INFINITY;
        }

        // cond(A) = cond(L)² for Cholesky L such that A = LL'
        let cond_l = max_l / min_l;
        return cond_l * cond_l;
    }

    // Cholesky failed: matrix is definitely ill-conditioned
    f64::INFINITY
}

/// Regularize covariance matrix for GLS stability (§3.3.5).
///
/// Applies condition-number-based shrinkage when the matrix is ill-conditioned:
/// Σ ← (1-λ)Σ + λ·diag(Σ) where λ scales with condition severity.
/// For extremely ill-conditioned matrices, uses diagonal-only (OLS) as fallback.
fn regularize_sigma_n(sigma_n: &Matrix9) -> Matrix9 {
    let cond = estimate_condition_number(sigma_n);

    if cond <= CONDITION_NUMBER_THRESHOLD {
        // Well-conditioned: return as-is (add minimal jitter if needed)
        if Cholesky::new(*sigma_n).is_some() {
            return *sigma_n;
        }
    }

    // Extract diagonal for shrinkage target
    let diag_sigma = Matrix9::from_diagonal(&sigma_n.diagonal());

    // For extremely ill-conditioned matrices (cond > 10^6), use unit diagonal (OLS).
    // This gives up on modeling correlations but guarantees stable GLS.
    if cond > CONDITION_NUMBER_THRESHOLD * 1e2 || cond.is_infinite() {
        return Matrix9::identity();
    }

    // Moderately ill-conditioned: use aggressive shrinkage
    // Scale lambda with log of condition number excess
    let log_excess = (cond / CONDITION_NUMBER_THRESHOLD).ln().max(0.0);
    let lambda = (0.1 + 0.2 * log_excess).min(0.95); // Range: 10% to 95%

    // Shrink toward diagonal: (1-λ)Σ + λ·diag(Σ)
    let regularized = *sigma_n * (1.0 - lambda) + diag_sigma * lambda;

    // Ensure SPD with increasing jitter if needed
    for &eps in &[1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5] {
        let jittered = regularized + Matrix9::identity() * eps;
        if Cholesky::new(jittered).is_some() {
            return jittered;
        }
    }

    // Final fallback: use diagonal only
    diag_sigma + Matrix9::identity() * 1e-6
}

/// Build the design matrix X for 2D projection.
///
/// X[:,0] = 1 (shift component) - uniform effect across all quantiles
/// X[:,1] = b_tail (tail component) - centered tail weights per spec §3.4.6:
///          (−0.5, −0.375, −0.25, −0.125, 0, 0.125, 0.25, 0.375, 0.5)ᵀ
fn build_design_matrix() -> Matrix9x2 {
    let mut x = Matrix9x2::zeros();
    for i in 0..9 {
        x[(i, 0)] = ONES[i]; // Shift
        x[(i, 1)] = B_TAIL[i]; // Tail (spec §3.4.6)
    }
    x
}

/// Compute effective sample size of a chain accounting for autocorrelation.
///
/// ESS = N / (1 + 2 * Σ_k ρ_k)
/// where ρ_k is the lag-k autocorrelation.
fn compute_ess(chain: &[f64]) -> f64 {
    let n = chain.len();
    if n < 2 {
        return n as f64;
    }

    let mean: f64 = chain.iter().sum::<f64>() / n as f64;
    let var: f64 = chain.iter().map(|&x| math::sq(x - mean)).sum::<f64>() / n as f64;

    if var < 1e-12 {
        return n as f64; // No variance, treat as independent
    }

    let mut sum_rho = 0.0;
    for k in 1..=50.min(n / 2) {
        let rho_k = autocorrelation(chain, k, mean, var);
        if rho_k < 0.05 {
            break;
        }
        sum_rho += rho_k;
    }

    n as f64 / (1.0 + 2.0 * sum_rho)
}

/// Compute lag-k autocorrelation.
fn autocorrelation(chain: &[f64], k: usize, mean: f64, var: f64) -> f64 {
    let n = chain.len();
    if k >= n {
        return 0.0;
    }

    let cov: f64 = (0..(n - k))
        .map(|i| (chain[i] - mean) * (chain[i + k] - mean))
        .sum::<f64>()
        / (n - k) as f64;

    cov / var
}

/// Public interface: Run Gibbs inference on observed data.
///
/// # Arguments
/// * `delta` - Observed quantile differences Δ
/// * `sigma_n` - Likelihood covariance Σ_n = Σ_rate / n
/// * `sigma_t` - Calibrated Student's t prior scale
/// * `l_r` - Cholesky factor of correlation matrix R
/// * `theta` - Effect threshold
/// * `seed` - Deterministic RNG seed
pub fn run_gibbs_inference(
    delta: &Vector9,
    sigma_n: &Matrix9,
    sigma_t: f64,
    l_r: &Matrix9,
    theta: f64,
    seed: u64,
) -> GibbsResult {
    let mut sampler = GibbsSampler::new(l_r, sigma_t, seed);
    sampler.run(delta, sigma_n, theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gibbs_determinism() {
        let l_r = Matrix9::identity();
        let sigma_n = Matrix9::identity() * 100.0;
        let delta = Vector9::from_row_slice(&[10.0; 9]);
        let sigma_t = 50.0;
        let theta = 5.0;

        let result1 = run_gibbs_inference(&delta, &sigma_n, sigma_t, &l_r, theta, 42);
        let result2 = run_gibbs_inference(&delta, &sigma_n, sigma_t, &l_r, theta, 42);

        assert!(
            (result1.leak_probability - result2.leak_probability).abs() < 1e-10,
            "Same seed should give same result"
        );
    }

    #[test]
    fn test_lambda_diagnostics() {
        let l_r = Matrix9::identity();
        let sigma_n = Matrix9::identity() * 100.0;
        let delta = Vector9::from_row_slice(&[50.0; 9]);
        let sigma_t = 50.0;
        let theta = 10.0;

        let result = run_gibbs_inference(&delta, &sigma_n, sigma_t, &l_r, theta, 42);

        // Lambda mean should be positive
        assert!(result.lambda_mean > 0.0);

        // Lambda SD should be positive
        assert!(result.lambda_sd > 0.0);

        // ESS should be reasonable (between 1 and N_KEEP)
        assert!(result.lambda_ess >= 1.0);
        assert!(result.lambda_ess <= N_KEEP as f64);
    }

    #[test]
    fn test_large_effect_detection() {
        // With large effect, leak probability should be high
        let l_r = Matrix9::identity();
        let sigma_n = Matrix9::identity() * 100.0;
        let delta = Vector9::from_row_slice(&[500.0; 9]); // Large effect
        let sigma_t = 50.0;
        let theta = 10.0;

        let result = run_gibbs_inference(&delta, &sigma_n, sigma_t, &l_r, theta, 42);

        assert!(
            result.leak_probability > 0.95,
            "Large effect should give high leak probability, got {}",
            result.leak_probability
        );
    }

    #[test]
    fn test_no_effect_low_probability() {
        // With no effect, leak probability should be low
        let l_r = Matrix9::identity();
        let sigma_n = Matrix9::identity() * 100.0;
        let delta = Vector9::zeros(); // No effect
        let sigma_t = 50.0;
        let theta = 100.0;

        let result = run_gibbs_inference(&delta, &sigma_n, sigma_t, &l_r, theta, 42);

        assert!(
            result.leak_probability < 0.5,
            "No effect should give low leak probability, got {}",
            result.leak_probability
        );
    }

    #[test]
    fn test_ess_computation() {
        // Test ESS with known autocorrelated sequence
        let chain: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let ess = compute_ess(&chain);

        // ESS should be less than N due to autocorrelation
        assert!(ess < 100.0);
        assert!(ess > 0.0);
    }

    #[test]
    fn test_kappa_diagnostics() {
        // v5.6: Test kappa diagnostics similar to lambda diagnostics
        let l_r = Matrix9::identity();
        let sigma_n = Matrix9::identity() * 100.0;
        let delta = Vector9::from_row_slice(&[50.0; 9]);
        let sigma_t = 50.0;
        let theta = 10.0;

        let result = run_gibbs_inference(&delta, &sigma_n, sigma_t, &l_r, theta, 42);

        // Kappa mean should be positive
        assert!(result.kappa_mean > 0.0, "kappa_mean should be positive");

        // Kappa SD should be positive
        assert!(result.kappa_sd > 0.0, "kappa_sd should be positive");

        // Kappa ESS should be reasonable (between 1 and N_KEEP)
        assert!(
            result.kappa_ess >= 1.0,
            "kappa_ess should be >= 1, got {}",
            result.kappa_ess
        );
        assert!(
            result.kappa_ess <= N_KEEP as f64,
            "kappa_ess should be <= N_KEEP"
        );

        // CV should be positive when SD > 0
        assert!(result.kappa_cv >= 0.0, "kappa_cv should be non-negative");
    }

    #[test]
    fn test_kappa_responds_to_residual_magnitude() {
        // v5.6: κ should be < 1 when residuals are larger than expected under Σₙ.
        // When the observed data differs from the sampled δ significantly,
        // the Gamma posterior for κ should shift toward smaller values.

        let l_r = Matrix9::identity();
        let sigma_n = Matrix9::identity(); // Small Σₙ
        let sigma_t = 50.0;
        let theta = 10.0;

        // Small effect: residuals ≈ small relative to Σₙ → κ ≈ 1
        let delta_small = Vector9::from_row_slice(&[1.0; 9]);
        let result_small = run_gibbs_inference(&delta_small, &sigma_n, sigma_t, &l_r, theta, 42);

        // Large effect: same Σₙ but much larger observed Δ → κ < 1
        // (The observed data is far from what the model expects)
        let delta_large = Vector9::from_row_slice(&[100.0; 9]);
        let result_large = run_gibbs_inference(&delta_large, &sigma_n, sigma_t, &l_r, theta, 42);

        // When residuals are large relative to Σₙ, kappa should be smaller
        // to inflate the likelihood covariance for robustness
        assert!(
            result_large.kappa_mean < result_small.kappa_mean,
            "Large residuals should give smaller kappa: {} vs {}",
            result_large.kappa_mean,
            result_small.kappa_mean
        );
    }
}
