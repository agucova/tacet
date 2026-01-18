//! Single-pass analysis for pre-collected timing samples.
//!
//! This module provides functionality to analyze fixed sets of timing samples
//! in a single pass, without the iterative adaptive sampling loop. This is
//! useful for:
//!
//! - Analyzing data from external tools (SILENT, dudect, etc.)
//! - Replaying historical measurements
//! - Testing with synthetic or simulated data
//!
//! The analysis follows the same statistical methodology as the adaptive loop,
//! but computes the posterior from all available samples at once.

use std::time::{Duration, Instant};

use timing_oracle_core::adaptive::{calibrate_prior_scale, compute_prior_cov_9d};
use timing_oracle_core::analysis::{classify_pattern, compute_bayes_factor};
use timing_oracle_core::result::{
    Diagnostics, EffectEstimate, EffectPattern, Exploitability, InconclusiveReason,
    MeasurementQuality, Outcome,
};
use timing_oracle_core::statistics::{
    bootstrap_difference_covariance_discrete, compute_deciles_inplace,
};
use timing_oracle_core::types::AttackerModel;
use timing_oracle_core::Vector9;

/// Configuration for single-pass analysis.
#[derive(Debug, Clone)]
pub struct SinglePassConfig {
    /// Minimum effect threshold in nanoseconds.
    pub theta_ns: f64,

    /// False positive rate threshold (default 0.05).
    pub pass_threshold: f64,

    /// False negative rate threshold (default 0.95).
    pub fail_threshold: f64,

    /// Number of bootstrap iterations for covariance estimation (default 2000).
    pub bootstrap_iterations: usize,

    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for SinglePassConfig {
    fn default() -> Self {
        Self {
            theta_ns: 100.0, // AdjacentNetwork default
            pass_threshold: 0.05,
            fail_threshold: 0.95,
            bootstrap_iterations: 2000,
            seed: 0xDEADBEEF,
        }
    }
}

impl SinglePassConfig {
    /// Create config from an attacker model.
    pub fn for_attacker(model: AttackerModel) -> Self {
        let theta_ns = match model {
            AttackerModel::SharedHardware => 0.6,
            AttackerModel::AdjacentNetwork => 100.0,
            AttackerModel::RemoteNetwork => 50_000.0,
            AttackerModel::Research => 0.001, // Near-zero but not exactly zero
            AttackerModel::Custom { threshold_ns } => threshold_ns,
            AttackerModel::PostQuantumSentinel => 10.0, // 10ns for PQ
        };
        Self {
            theta_ns,
            ..Default::default()
        }
    }
}

/// Result of single-pass analysis.
#[derive(Debug, Clone)]
pub struct SinglePassResult {
    /// The final outcome.
    pub outcome: Outcome,

    /// Posterior probability of leak > theta.
    pub leak_probability: f64,

    /// Estimated effect (shift and tail components).
    pub effect_estimate: EffectEstimate,

    /// Measurement quality assessment.
    pub quality: MeasurementQuality,

    /// Number of samples used per class.
    pub samples_used: usize,

    /// Time taken for analysis.
    pub analysis_time: Duration,
}

/// Analyze pre-collected timing samples in a single pass.
///
/// This function computes the posterior probability of a timing leak given
/// fixed sets of baseline and test samples. Unlike the adaptive loop, it
/// cannot collect additional samples - it works with what it has.
///
/// # Arguments
/// * `baseline_ns` - Baseline timing samples in nanoseconds
/// * `test_ns` - Test timing samples in nanoseconds
/// * `config` - Analysis configuration
///
/// # Returns
/// `SinglePassResult` containing the outcome and diagnostics.
pub fn analyze_single_pass(
    baseline_ns: &[f64],
    test_ns: &[f64],
    config: &SinglePassConfig,
) -> SinglePassResult {
    let start_time = Instant::now();
    let n = baseline_ns.len().min(test_ns.len());

    // Validate minimum samples
    const MIN_SAMPLES: usize = 100;
    if n < MIN_SAMPLES {
        let effect = EffectEstimate {
            shift_ns: 0.0,
            tail_ns: 0.0,
            credible_interval_ns: (0.0, 0.0),
            pattern: EffectPattern::Indeterminate,
            interpretation_caveat: None,
        };
        return SinglePassResult {
            outcome: Outcome::Inconclusive {
                reason: InconclusiveReason::DataTooNoisy {
                    message: format!(
                        "Insufficient samples: {} (need at least {})",
                        n, MIN_SAMPLES
                    ),
                    guidance: "Collect more timing measurements".to_string(),
                },
                leak_probability: 0.5,
                effect: effect.clone(),
                quality: MeasurementQuality::TooNoisy,
                diagnostics: Diagnostics::default(),
                samples_used: n,
                theta_user: config.theta_ns,
                theta_eff: config.theta_ns,
                theta_floor: f64::INFINITY,
            },
            leak_probability: 0.5,
            effect_estimate: effect,
            quality: MeasurementQuality::TooNoisy,
            samples_used: n,
            analysis_time: start_time.elapsed(),
        };
    }

    // Use same-length slices
    let baseline = &baseline_ns[..n];
    let test = &test_ns[..n];

    // Step 1: Compute quantile differences (the observed effect)
    let mut baseline_sorted = baseline.to_vec();
    let mut test_sorted = test.to_vec();
    let q_baseline = compute_deciles_inplace(&mut baseline_sorted);
    let q_test = compute_deciles_inplace(&mut test_sorted);
    let delta_hat: Vector9 = q_baseline - q_test;

    // Step 2: Bootstrap covariance estimation using discrete mode (separate arrays)
    let cov_estimate = bootstrap_difference_covariance_discrete(
        baseline,
        test,
        config.bootstrap_iterations,
        config.seed,
    );
    let sigma = cov_estimate.matrix;

    // Step 3: Compute prior covariance (MDE-based)
    // Convert covariance to rate: sigma_rate = sigma * n (for scaling purposes)
    let sigma_rate = sigma * (n as f64);

    // Detect discrete mode (< 10% unique values)
    let unique_baseline: std::collections::HashSet<i64> =
        baseline.iter().map(|&v| v as i64).collect();
    let discrete_mode = (unique_baseline.len() as f64 / n as f64) < 0.10;

    let prior_scale = calibrate_prior_scale(&sigma_rate, config.theta_ns, n, discrete_mode, config.seed);
    let prior_cov = compute_prior_cov_9d(&sigma_rate, prior_scale, discrete_mode);

    // Step 4: Compute Bayesian posterior
    let bayes_result =
        compute_bayes_factor(&delta_hat, &sigma, &prior_cov, config.theta_ns, Some(config.seed));
    let leak_probability = bayes_result.leak_probability;

    // Step 5: Compute effect estimate from Bayesian result
    let pattern = classify_pattern(&bayes_result.beta_proj, &bayes_result.beta_proj_cov);
    let effect_estimate = EffectEstimate {
        shift_ns: bayes_result.beta_proj[0],
        tail_ns: bayes_result.beta_proj[1],
        credible_interval_ns: bayes_result.effect_magnitude_ci,
        pattern,
        interpretation_caveat: if bayes_result.projection_mismatch_q > cov_estimate.q_thresh {
            Some("Model fit is poor; effect decomposition may be unreliable".to_string())
        } else {
            None
        },
    };

    // Step 6: Assess measurement quality based on MDE
    // MDE approximation: theta where we'd have 80% power
    let sigma_trace = sigma.trace();
    let mde_approx = (sigma_trace / n as f64).sqrt() * 2.8; // ~z_0.8 + z_0.05
    let quality = MeasurementQuality::from_mde_ns(mde_approx);

    // Step 7: Make decision
    let outcome = if leak_probability < config.pass_threshold {
        Outcome::Pass {
            leak_probability,
            effect: effect_estimate.clone(),
            quality,
            diagnostics: Diagnostics::default(),
            samples_used: n,
            theta_user: config.theta_ns,
            theta_eff: config.theta_ns,
            theta_floor: mde_approx,
        }
    } else if leak_probability > config.fail_threshold {
        let exploitability = Exploitability::from_effect_ns(effect_estimate.shift_ns.abs());
        Outcome::Fail {
            leak_probability,
            effect: effect_estimate.clone(),
            exploitability,
            quality,
            diagnostics: Diagnostics::default(),
            samples_used: n,
            theta_user: config.theta_ns,
            theta_eff: config.theta_ns,
            theta_floor: mde_approx,
        }
    } else {
        Outcome::Inconclusive {
            reason: InconclusiveReason::SampleBudgetExceeded {
                current_probability: leak_probability,
                samples_collected: n,
            },
            leak_probability,
            effect: effect_estimate.clone(),
            quality,
            diagnostics: Diagnostics::default(),
            samples_used: n,
            theta_user: config.theta_ns,
            theta_eff: config.theta_ns,
            theta_floor: mde_approx,
        }
    };

    SinglePassResult {
        outcome,
        leak_probability,
        effect_estimate,
        quality,
        samples_used: n,
        analysis_time: start_time.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn generate_samples(mean: f64, std: f64, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = Normal::new(mean, std).unwrap();
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }

    #[test]
    fn test_no_effect_passes() {
        let baseline = generate_samples(1000.0, 50.0, 1000, 42);
        let test = generate_samples(1000.0, 50.0, 1000, 43);

        let config = SinglePassConfig {
            theta_ns: 100.0,
            ..Default::default()
        };

        let result = analyze_single_pass(&baseline, &test, &config);

        // Should pass - no real effect
        assert!(
            result.leak_probability < 0.5,
            "Expected low leak probability for null effect, got {}",
            result.leak_probability
        );
    }

    #[test]
    fn test_large_effect_fails() {
        let baseline = generate_samples(1000.0, 50.0, 1000, 42);
        let test = generate_samples(1200.0, 50.0, 1000, 43); // 200ns difference

        let config = SinglePassConfig {
            theta_ns: 100.0,
            ..Default::default()
        };

        let result = analyze_single_pass(&baseline, &test, &config);

        // Should detect the 200ns effect (above 100ns threshold)
        assert!(
            result.leak_probability > 0.9,
            "Expected high leak probability for 200ns effect, got {}",
            result.leak_probability
        );
        assert!(matches!(result.outcome, Outcome::Fail { .. }));
    }

    #[test]
    fn test_effect_below_threshold_passes() {
        let baseline = generate_samples(1000.0, 50.0, 1000, 42);
        let test = generate_samples(1050.0, 50.0, 1000, 43); // 50ns difference

        let config = SinglePassConfig {
            theta_ns: 100.0, // Threshold is 100ns
            ..Default::default()
        };

        let result = analyze_single_pass(&baseline, &test, &config);

        // 50ns effect should be below 100ns threshold - should pass or be inconclusive
        // (Bayesian approach: small effect relative to threshold)
        assert!(
            result.leak_probability < 0.95,
            "Expected lower leak probability for sub-threshold effect, got {}",
            result.leak_probability
        );
    }

    #[test]
    fn test_insufficient_samples() {
        let baseline = vec![100.0; 50]; // Only 50 samples
        let test = vec![100.0; 50];

        let config = SinglePassConfig::default();
        let result = analyze_single_pass(&baseline, &test, &config);

        assert!(matches!(
            result.outcome,
            Outcome::Inconclusive {
                reason: InconclusiveReason::DataTooNoisy { .. },
                ..
            }
        ));
    }
}
