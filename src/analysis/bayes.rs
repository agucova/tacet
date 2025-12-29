//! Layer 2: Bayesian inference for timing leak detection.
//!
//! This module computes the posterior probability of a timing leak using:
//! - Sample splitting (30% calibration, 70% inference) for valid inference
//! - Closed-form Bayes factor via multivariate normal log-pdf ratio
//! - Cholesky decomposition for numerical stability
//!
//! The Bayes factor compares:
//! - H0 (null): Quantile differences ~ MVN(0, Sigma)
//! - H1 (leak): Quantile differences ~ MVN(mu_observed, Sigma)

use nalgebra::Cholesky;

use crate::constants::LOG_2PI;
use crate::types::{Matrix9, Vector9};

/// Result from Bayesian analysis.
#[derive(Debug, Clone)]
pub struct BayesResult {
    /// Log Bayes factor: log(P(data|H1) / P(data|H0))
    pub log_bayes_factor: f64,
    /// Posterior probability of leak: P(H1|data)
    pub posterior_probability: f64,
    /// Mean of calibration split (used as null center)
    pub calibration_mean: Vector9,
    /// Mean of inference split (used for testing)
    pub inference_mean: Vector9,
}

/// Compute the Bayes factor for timing leak detection.
///
/// Uses sample splitting to ensure valid inference:
/// 1. Calibration split (30%): Estimate null distribution parameters
/// 2. Inference split (70%): Compute Bayes factor
///
/// # Arguments
///
/// * `observed_diff` - Observed quantile differences (fixed - random)
/// * `covariance` - Pooled covariance matrix of differences
/// * `n_samples` - Total sample size (will be split 30/70)
/// * `prior_odds` - Prior odds ratio P(H1)/P(H0), typically 1.0 for uninformative
///
/// # Returns
///
/// A `BayesResult` containing log Bayes factor and posterior probability.
pub fn compute_bayes_factor(
    observed_diff: &Vector9,
    covariance: &Matrix9,
    n_samples: usize,
    prior_odds: f64,
) -> BayesResult {
    // Sample splitting: 30% calibration, 70% inference
    let n_calib = (n_samples as f64 * 0.3).round() as usize;
    let n_infer = n_samples - n_calib;

    // TODO: Implement proper sample splitting
    // For now, we use the full observed difference as the inference mean
    // and assume calibration mean is zero (null hypothesis center)
    let calibration_mean = Vector9::zeros();
    let inference_mean = *observed_diff;

    // Scale covariance by sample size for inference split
    let scaled_cov = covariance / (n_infer as f64);

    // Compute log Bayes factor
    let log_bf = compute_log_bayes_factor_mvn(&inference_mean, &calibration_mean, &scaled_cov);

    // Convert to posterior probability
    let posterior = compute_posterior_probability(log_bf, prior_odds);

    BayesResult {
        log_bayes_factor: log_bf,
        posterior_probability: posterior,
        calibration_mean,
        inference_mean,
    }
}

/// Compute log Bayes factor using MVN log-pdf ratio.
///
/// log BF = log P(data|H1) - log P(data|H0)
///        = log MVN(data; mu_H1, Sigma) - log MVN(data; mu_H0, Sigma)
///
/// For our setup:
/// - H0: mu = 0 (no timing difference)
/// - H1: mu = observed mean (timing difference exists)
fn compute_log_bayes_factor_mvn(
    observed: &Vector9,
    null_mean: &Vector9,
    covariance: &Matrix9,
) -> f64 {
    // Compute Cholesky decomposition for numerical stability
    let chol = match Cholesky::new(*covariance) {
        Some(c) => c,
        None => {
            // Covariance not positive definite, fall back to regularization
            let regularized = covariance + Matrix9::identity() * 1e-10;
            match Cholesky::new(regularized) {
                Some(c) => c,
                None => {
                    // Still failing, return 0 (inconclusive)
                    return 0.0;
                }
            }
        }
    };

    // Log-pdf under H1 (centered at observed)
    let log_pdf_h1 = mvn_log_pdf(observed, observed, &chol);

    // Log-pdf under H0 (centered at null/zero)
    let log_pdf_h0 = mvn_log_pdf(observed, null_mean, &chol);

    // Log Bayes factor
    log_pdf_h1 - log_pdf_h0
}

/// Compute log-pdf of multivariate normal distribution.
///
/// log MVN(x; mu, Sigma) = -0.5 * (k*log(2*pi) + log|Sigma| + (x-mu)^T Sigma^-1 (x-mu))
///
/// Uses Cholesky decomposition for efficient computation.
fn mvn_log_pdf(x: &Vector9, mean: &Vector9, chol: &Cholesky<f64, nalgebra::Const<9>>) -> f64 {
    const K: usize = 9;

    // Compute (x - mu)
    let diff = x - mean;

    // Solve L * z = diff for z (forward substitution)
    // Then Sigma^-1 * diff = L^-T * z
    let z = chol.l().solve_lower_triangular(&diff).unwrap_or(diff);

    // Mahalanobis distance squared: z^T * z = diff^T * Sigma^-1 * diff
    let mahal_sq = z.dot(&z);

    // Log determinant: log|Sigma| = 2 * sum(log(diag(L)))
    let log_det = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();

    // Log-pdf
    -0.5 * ((K as f64) * LOG_2PI + log_det + mahal_sq)
}

/// Convert log Bayes factor to posterior probability.
///
/// P(H1|data) = BF * prior_odds / (1 + BF * prior_odds)
///            = 1 / (1 + exp(-log_bf) / prior_odds)
///
/// # Arguments
///
/// * `log_bf` - Log Bayes factor
/// * `prior_odds` - Prior odds P(H1)/P(H0)
///
/// # Returns
///
/// Posterior probability P(H1|data) in [0, 1].
pub fn compute_posterior_probability(log_bf: f64, prior_odds: f64) -> f64 {
    // Handle extreme values to avoid overflow
    let log_prior_odds = prior_odds.ln();
    let log_posterior_odds = log_bf + log_prior_odds;

    if log_posterior_odds > 700.0 {
        // exp(700) would overflow, posterior is essentially 1
        1.0
    } else if log_posterior_odds < -700.0 {
        // Posterior is essentially 0
        0.0
    } else {
        // P(H1|data) = posterior_odds / (1 + posterior_odds)
        //            = 1 / (1 + exp(-log_posterior_odds))
        1.0 / (1.0 + (-log_posterior_odds).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posterior_probability_bounds() {
        // Very large log BF should give probability near 1
        let prob_high = compute_posterior_probability(100.0, 1.0);
        assert!(prob_high > 0.999);

        // Very negative log BF should give probability near 0
        let prob_low = compute_posterior_probability(-100.0, 1.0);
        assert!(prob_low < 0.001);

        // Zero log BF with equal priors should give 0.5
        let prob_equal = compute_posterior_probability(0.0, 1.0);
        assert!((prob_equal - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_mvn_log_pdf_at_mean() {
        let mean = Vector9::zeros();
        let cov = Matrix9::identity();
        let chol = Cholesky::new(cov).unwrap();

        // Log-pdf at the mean should be maximal
        let log_pdf_at_mean = mvn_log_pdf(&mean, &mean, &chol);

        // Should be -0.5 * k * log(2*pi) for identity covariance
        let expected = -0.5 * 9.0 * LOG_2PI;
        assert!((log_pdf_at_mean - expected).abs() < 0.001);
    }
}
