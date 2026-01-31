//! KL divergence computation for Gaussian distributions.
//!
//! Used to track the learning rate during adaptive sampling. When the posterior
//! stops updating (KL divergence becomes very small), we've either converged or
//! the data is uninformative.

use super::Posterior;

/// KL divergence KL(p || q) for 9D Gaussian distributions.
///
/// For Gaussians p ~ N(μ_p, Σ_p) and q ~ N(μ_q, Σ_q):
///
/// KL(p||q) = 0.5 × (tr(Σ_q⁻¹ Σ_p) + (μ_q - μ_p)ᵀ Σ_q⁻¹ (μ_q - μ_p) - k + ln(det(Σ_q)/det(Σ_p)))
///
/// where k = 9 (dimension).
///
/// This measures how much the posterior has changed from the previous iteration.
/// Small KL indicates the posterior is no longer updating despite new data.
///
/// Uses the 9D posterior mean and covariance for tracking.
///
/// # Arguments
///
/// * `p` - The new posterior (typically more peaked)
/// * `q` - The old posterior (reference distribution)
///
/// # Returns
///
/// KL divergence in nats. Returns `f64::INFINITY` if q has singular covariance.
pub fn kl_divergence_gaussian(p: &Posterior, q: &Posterior) -> f64 {
    use nalgebra::Cholesky;

    const K: f64 = 9.0;

    // Compute Cholesky of q's covariance for stable inversion
    let q_chol = match Cholesky::new(q.lambda_post) {
        Some(c) => c,
        None => return f64::INFINITY, // Singular covariance
    };

    // Mean difference
    let mu_diff = p.delta_post - q.delta_post;

    // tr(Σ_q⁻¹ Σ_p) via Cholesky solves: sum of diagonal of (L_q⁻¹ Σ_p L_q⁻ᵀ)
    let mut trace_term = 0.0;
    for j in 0..9 {
        let col = p.lambda_post.column(j).into_owned();
        let solved = q_chol.solve(&col);
        trace_term += solved[j];
    }

    // (μ_q - μ_p)ᵀ Σ_q⁻¹ (μ_q - μ_p) via Cholesky
    let solved_diff = q_chol.solve(&mu_diff);
    let mahalanobis = mu_diff.dot(&solved_diff);

    // ln(det(Σ_q)/det(Σ_p)) via log determinants
    let q_log_det: f64 = (0..9)
        .map(|i| libm::log(q_chol.l()[(i, i)].abs().max(1e-300)))
        .sum::<f64>()
        * 2.0;

    let p_chol = match Cholesky::new(p.lambda_post) {
        Some(c) => c,
        None => return f64::INFINITY,
    };
    let p_log_det: f64 = (0..9)
        .map(|i| libm::log(p_chol.l()[(i, i)].abs().max(1e-300)))
        .sum::<f64>()
        * 2.0;

    if !q_log_det.is_finite() || !p_log_det.is_finite() {
        return f64::INFINITY;
    }

    let log_det_ratio = q_log_det - p_log_det;

    0.5 * (trace_term + mahalanobis - K + log_det_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Matrix9, Vector9};

    fn make_test_posterior(delta_post: Vector9, variance: f64, leak_prob: f64) -> Posterior {
        Posterior::new(
            delta_post,
            Matrix9::identity() * variance,
            Vec::new(), // delta_draws
            leak_prob,
            1.0,
            100,
        )
    }

    #[test]
    fn test_kl_identical_distributions() {
        let p = make_test_posterior(
            Vector9::from_row_slice(&[5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1.0,
            0.5,
        );
        let q = p.clone();

        let kl = kl_divergence_gaussian(&p, &q);

        // KL divergence of identical distributions should be 0
        assert!(
            libm::fabs(kl) < 1e-10,
            "KL of identical distributions should be 0, got {}",
            kl
        );
    }

    #[test]
    fn test_kl_different_means() {
        let p = make_test_posterior(
            Vector9::from_row_slice(&[5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1.0,
            0.5,
        );
        let q = make_test_posterior(Vector9::zeros(), 1.0, 0.5);

        let kl = kl_divergence_gaussian(&p, &q);

        // For same covariance, KL = 0.5 × ||μ_p - μ_q||²
        // = 0.5 × (25 + 9) = 17.0
        assert!(
            libm::fabs(kl - 17.0) < 1e-10,
            "KL should be 17.0, got {}",
            kl
        );
    }

    #[test]
    fn test_kl_different_variances() {
        let p = make_test_posterior(Vector9::zeros(), 2.0, 0.5);
        let q = make_test_posterior(Vector9::zeros(), 1.0, 0.5);

        let kl = kl_divergence_gaussian(&p, &q);

        // KL should be positive for different variances
        assert!(kl > 0.0, "KL should be positive for different variances");
    }
}
