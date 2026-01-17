//! KL divergence computation for Gaussian distributions.
//!
//! Used to track the learning rate during adaptive sampling. When the posterior
//! stops updating (KL divergence becomes very small), we've either converged or
//! the data is uninformative.

use crate::types::Matrix2;

use super::Posterior;

/// KL divergence KL(p || q) for 2D Gaussian distributions.
///
/// For Gaussians p ~ N(μ_p, Σ_p) and q ~ N(μ_q, Σ_q):
///
/// KL(p||q) = 0.5 × (tr(Σ_q⁻¹ Σ_p) + (μ_q - μ_p)ᵀ Σ_q⁻¹ (μ_q - μ_p) - k + ln(det(Σ_q)/det(Σ_p)))
///
/// where k = 2 (dimension).
///
/// This measures how much the posterior has changed from the previous iteration.
/// Small KL indicates the posterior is no longer updating despite new data.
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
    let k = 2.0_f64;

    // Compute Σ_q⁻¹
    let q_cov_inv = match invert_2x2(&q.beta_cov) {
        Some(inv) => inv,
        None => return f64::INFINITY, // Singular covariance
    };

    let mu_diff = [
        p.beta_mean[0] - q.beta_mean[0],
        p.beta_mean[1] - q.beta_mean[1],
    ];

    // tr(Σ_q⁻¹ Σ_p)
    let trace_term = trace_product_2x2(&q_cov_inv, &p.beta_cov);

    // (μ_q - μ_p)ᵀ Σ_q⁻¹ (μ_q - μ_p)
    let mahalanobis = mahalanobis_2d(&mu_diff, &q_cov_inv);

    // ln(det(Σ_q)/det(Σ_p))
    let det_p = determinant_2x2(&p.beta_cov);
    let det_q = determinant_2x2(&q.beta_cov);
    let log_det_ratio = if det_p > 0.0 && det_q > 0.0 {
        libm::log(det_q / det_p)
    } else {
        f64::INFINITY
    };

    0.5 * (trace_term + mahalanobis - k + log_det_ratio)
}

/// Invert a 2×2 matrix.
///
/// For 2×2 matrix [[a,b],[c,d]], inverse is 1/det × [[d,-b],[-c,a]]
fn invert_2x2(m: &Matrix2) -> Option<Matrix2> {
    let det = determinant_2x2(m);
    if libm::fabs(det) < 1e-10 {
        return None;
    }
    Some(Matrix2::new(
        m[(1, 1)] / det,
        -m[(0, 1)] / det,
        -m[(1, 0)] / det,
        m[(0, 0)] / det,
    ))
}

/// Compute determinant of a 2×2 matrix.
#[inline]
fn determinant_2x2(m: &Matrix2) -> f64 {
    m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)]
}

/// Compute tr(AB) for 2×2 matrices.
///
/// tr(AB) = Σ_i (AB)_ii = Σ_i Σ_j a_ij × b_ji
#[inline]
fn trace_product_2x2(a: &Matrix2, b: &Matrix2) -> f64 {
    // (AB)_00 = a_00×b_00 + a_01×b_10
    // (AB)_11 = a_10×b_01 + a_11×b_11
    // tr(AB) = (AB)_00 + (AB)_11
    (a[(0, 0)] * b[(0, 0)] + a[(0, 1)] * b[(1, 0)])
        + (a[(1, 0)] * b[(0, 1)] + a[(1, 1)] * b[(1, 1)])
}

/// Compute x^T Σ⁻¹ x (Mahalanobis distance squared) for 2D.
#[inline]
fn mahalanobis_2d(x: &[f64; 2], sigma_inv: &Matrix2) -> f64 {
    // temp = Σ⁻¹ x
    let temp = [
        sigma_inv[(0, 0)] * x[0] + sigma_inv[(0, 1)] * x[1],
        sigma_inv[(1, 0)] * x[0] + sigma_inv[(1, 1)] * x[1],
    ];
    // x^T temp
    x[0] * temp[0] + x[1] * temp[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Vector2;

    #[test]
    fn test_kl_identical_distributions() {
        let p = Posterior::new(
            Vector2::new(5.0, 3.0),
            Matrix2::new(1.0, 0.0, 0.0, 1.0),
            0.5,
            100,
            5.0, // model_fit_q
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
        let p = Posterior::new(
            Vector2::new(5.0, 3.0),
            Matrix2::new(1.0, 0.0, 0.0, 1.0),
            0.5,
            100,
            5.0, // model_fit_q
        );
        let q = Posterior::new(
            Vector2::new(0.0, 0.0),
            Matrix2::new(1.0, 0.0, 0.0, 1.0),
            0.5,
            100,
            5.0, // model_fit_q
        );

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
        let p = Posterior::new(
            Vector2::new(0.0, 0.0),
            Matrix2::new(2.0, 0.0, 0.0, 2.0), // Wider
            0.5,
            100,
            5.0, // model_fit_q
        );
        let q = Posterior::new(
            Vector2::new(0.0, 0.0),
            Matrix2::new(1.0, 0.0, 0.0, 1.0),
            0.5,
            100,
            5.0, // model_fit_q
        );

        let kl = kl_divergence_gaussian(&p, &q);

        // KL should be positive for different variances
        assert!(kl > 0.0, "KL should be positive for different variances");
    }

    #[test]
    fn test_determinant_2x2() {
        let m = Matrix2::new(3.0, 1.0, 2.0, 4.0);
        let det = determinant_2x2(&m);
        assert!(
            libm::fabs(det - 10.0) < 1e-10,
            "det should be 3*4 - 1*2 = 10"
        );
    }

    #[test]
    fn test_invert_2x2() {
        let m = Matrix2::new(4.0, 0.0, 0.0, 2.0);
        let inv = invert_2x2(&m).expect("Should be invertible");

        // Check m × inv = I
        let product = m * inv;
        assert!(libm::fabs(product[(0, 0)] - 1.0) < 1e-10);
        assert!(libm::fabs(product[(1, 1)] - 1.0) < 1e-10);
        assert!(libm::fabs(product[(0, 1)]) < 1e-10);
        assert!(libm::fabs(product[(1, 0)]) < 1e-10);
    }

    #[test]
    fn test_invert_singular() {
        let m = Matrix2::new(1.0, 1.0, 1.0, 1.0); // Singular
        assert!(invert_2x2(&m).is_none());
    }
}
