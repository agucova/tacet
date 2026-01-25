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
//!
//! ## Alignment with Main Pipeline
//!
//! This module replicates the statistical methodology of the main adaptive pipeline:
//! - Bootstrap covariance estimation with acquisition stream (spec §3.3.1)
//! - Block length computation for dependence-aware bootstrap (spec §3.3.2)
//! - Measurement floor with timer tick floor (spec §3.3.4)
//! - Student's t prior calibration (spec §3.3.5)
//! - Gibbs sampling inference (spec §3.4.4)
//! - Variance ratio quality check (spec §3.5.2)
//! - Proper MDE estimation for quality assessment

use std::time::{Duration, Instant};

use timing_oracle_core::adaptive::{
    calibrate_t_prior_scale, compute_achievable_at_max, compute_c_floor_9d, compute_prior_cov_9d,
    is_threshold_elevated,
};
use timing_oracle_core::analysis::{classify_pattern, compute_bayes_gibbs, estimate_mde};
use timing_oracle_core::constants::{
    DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_FAIL_THRESHOLD, DEFAULT_PASS_THRESHOLD,
};
use timing_oracle_core::result::{
    Diagnostics, EffectEstimate, EffectPattern, Exploitability, InconclusiveReason,
    MeasurementQuality, Outcome, IssueCode, QualityIssue,
};
use timing_oracle_core::statistics::{
    bootstrap_difference_covariance, bootstrap_difference_covariance_discrete,
    compute_deciles_inplace, AcquisitionStream,
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

    /// Timer resolution in nanoseconds (default 1.0 for pre-collected data).
    /// Set this to the actual timer resolution if known (e.g., 41.7 for Apple Silicon cntvct_el0).
    pub timer_resolution_ns: f64,

    /// Random seed for reproducibility.
    pub seed: u64,

    /// Maximum variance ratio for quality gate (default 0.95).
    /// If posterior variance > this fraction of prior variance, data is not informative.
    pub max_variance_ratio: f64,
}

impl Default for SinglePassConfig {
    fn default() -> Self {
        Self {
            theta_ns: 100.0, // AdjacentNetwork default
            pass_threshold: DEFAULT_PASS_THRESHOLD,
            fail_threshold: DEFAULT_FAIL_THRESHOLD,
            bootstrap_iterations: DEFAULT_BOOTSTRAP_ITERATIONS,
            timer_resolution_ns: 1.0, // Assume 1ns resolution for pre-collected data
            seed: 0xDEADBEEF,
            max_variance_ratio: 0.95,
        }
    }
}

impl SinglePassConfig {
    /// Create config from an attacker model.
    pub fn for_attacker(model: AttackerModel) -> Self {
        Self {
            theta_ns: model.to_threshold_ns(),
            ..Default::default()
        }
    }

    /// Set the timer resolution (useful when analyzing data from known timer sources).
    pub fn with_timer_resolution(mut self, resolution_ns: f64) -> Self {
        self.timer_resolution_ns = resolution_ns;
        self
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
/// The analysis follows the same statistical methodology as the main pipeline:
/// 1. Compute quantile differences (observed effect)
/// 2. Bootstrap covariance estimation with acquisition stream preservation (spec §3.3.1)
/// 3. Block length computation for dependence structure (spec §3.3.2)
/// 4. Measurement floor with timer tick floor (spec §3.3.4)
/// 5. Student's t prior calibration (spec §3.3.5)
/// 6. Gibbs sampling inference (spec §3.4.4)
/// 7. Variance ratio quality check (spec §3.5.2)
/// 8. Decision based on posterior probability
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
                diagnostics: make_default_diagnostics(config.timer_resolution_ns),
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

    // Step 2: Detect discrete mode (spec §3.3.2)
    let unique_baseline: std::collections::HashSet<i64> =
        baseline.iter().map(|&v| v as i64).collect();
    let unique_test: std::collections::HashSet<i64> =
        test.iter().map(|&v| v as i64).collect();
    let min_uniqueness = (unique_baseline.len() as f64 / n as f64)
        .min(unique_test.len() as f64 / n as f64);
    let discrete_mode = min_uniqueness < 0.10;

    // Step 3: Bootstrap covariance estimation (spec §3.3.1)
    // Use acquisition stream for non-discrete mode to preserve dependence structure
    let cov_estimate = if discrete_mode {
        // Discrete mode: use m-out-of-n bootstrap
        bootstrap_difference_covariance_discrete(
            baseline,
            test,
            config.bootstrap_iterations,
            config.seed,
        )
    } else {
        // Non-discrete mode: create interleaved acquisition stream per spec §3.3.1
        let mut acquisition_stream = AcquisitionStream::with_capacity(2 * n);
        acquisition_stream.push_batch_interleaved(baseline, test);
        let interleaved = acquisition_stream.to_timing_samples();

        bootstrap_difference_covariance(
            &interleaved,
            config.bootstrap_iterations,
            config.seed,
            false, // is_fragile - let block length handle it
        )
    };
    let sigma = cov_estimate.matrix;

    // Check covariance validity
    if !cov_estimate.is_stable() {
        let effect = EffectEstimate {
            shift_ns: 0.0,
            tail_ns: 0.0,
            credible_interval_ns: (0.0, 0.0),
            pattern: EffectPattern::Indeterminate,
            interpretation_caveat: Some("Covariance matrix is not positive definite".to_string()),
        };
        return SinglePassResult {
            outcome: Outcome::Inconclusive {
                reason: InconclusiveReason::DataTooNoisy {
                    message: "Covariance matrix estimation failed".to_string(),
                    guidance: "Data may have too little variance or numerical issues".to_string(),
                },
                leak_probability: 0.5,
                effect: effect.clone(),
                quality: MeasurementQuality::TooNoisy,
                diagnostics: make_default_diagnostics(config.timer_resolution_ns),
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

    // Step 4: Compute covariance rate and measurement floor (spec §3.3.4)
    // sigma_rate = sigma * n (covariance scales as 1/n)
    let sigma_rate = sigma * (n as f64);

    // Compute c_floor: 95th percentile of max_k |Z_k| where Z ~ N(0, Σ_rate)
    let c_floor = compute_c_floor_9d(&sigma_rate, config.seed);

    // Statistical floor: c_floor / √n
    let theta_floor_stat = c_floor / (n as f64).sqrt();

    // Timer tick floor (spec §3.3.4)
    let theta_tick = config.timer_resolution_ns;

    // Combined measurement floor: max(statistical, timer tick)
    let theta_floor = theta_floor_stat.max(theta_tick);

    // Compute theta_eff per spec §3.3.4:
    // theta_eff = max(theta_user, theta_floor) if theta_user > 0
    let theta_eff = if config.theta_ns > 0.0 {
        config.theta_ns.max(theta_floor)
    } else {
        // Research mode: use floor directly
        theta_floor
    };

    // Step 5: Student's t prior calibration (spec §3.3.5)
    let (sigma_t, l_r) = calibrate_t_prior_scale(&sigma_rate, theta_eff, n, discrete_mode, config.seed);

    // Compute marginal prior covariance for variance ratio check (spec §3.5.2)
    // For t_4: Var(δ) = (ν/(ν-2)) σ² R = 2σ² R
    let prior_cov_marginal = compute_prior_cov_9d(&sigma_rate, sigma_t * std::f64::consts::SQRT_2, discrete_mode);

    // Step 6: Compute MDE for quality assessment (spec §3.4.6)
    let mde = estimate_mde(&sigma, 0.05); // α = 0.05
    let quality = MeasurementQuality::from_mde_ns(mde.shift_ns);

    // Debug output for investigation
    if std::env::var("TIMING_ORACLE_DEBUG").is_ok() {
        eprintln!("[DEBUG] n = {}, discrete_mode = {}, block_length = {}", n, discrete_mode, cov_estimate.block_size);
        eprintln!("[DEBUG] theta_user = {:.2} ns, theta_floor_stat = {:.2} ns, theta_tick = {:.2} ns, theta_floor = {:.2} ns, theta_eff = {:.2} ns",
            config.theta_ns, theta_floor_stat, theta_tick, theta_floor, theta_eff);
        eprintln!("[DEBUG] c_floor = {:.2} ns·√n", c_floor);
        eprintln!("[DEBUG] MDE: shift = {:.2} ns, tail = {:.2} ns", mde.shift_ns, mde.tail_ns);
        eprintln!("[DEBUG] delta_hat = {:?}", delta_hat.as_slice());
        eprintln!(
            "[DEBUG] sigma diagonal = [{:.2e}, {:.2e}, {:.2e}, ..., {:.2e}]",
            sigma[(0, 0)],
            sigma[(1, 1)],
            sigma[(2, 2)],
            sigma[(8, 8)]
        );
        eprintln!("[DEBUG] sigma_t = {:.2e}", sigma_t);
    }

    // Step 7: Compute Bayesian posterior with Gibbs sampling (spec §3.4.4)
    let bayes_result = compute_bayes_gibbs(
        &delta_hat,
        &sigma,
        sigma_t,
        &l_r,
        theta_eff,
        Some(config.seed),
    );
    let leak_probability = bayes_result.leak_probability;

    if std::env::var("TIMING_ORACLE_DEBUG").is_ok() {
        eprintln!("[DEBUG] bayes_result.leak_probability = {}", leak_probability);
        eprintln!(
            "[DEBUG] lambda: mean={:.3}, sd={:.3}, cv={:.3}, ess={:.1}, mixing_ok={}",
            bayes_result.lambda_mean,
            bayes_result.lambda_sd,
            bayes_result.lambda_cv,
            bayes_result.lambda_ess,
            bayes_result.lambda_mixing_ok
        );
        eprintln!(
            "[DEBUG] delta_post = [{:.1}, {:.1}, {:.1}, ..., {:.1}]",
            bayes_result.delta_post[0],
            bayes_result.delta_post[1],
            bayes_result.delta_post[2],
            bayes_result.delta_post[8]
        );
    }

    // Step 8: Compute effect estimate from Bayesian result
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

    // Step 9: Check variance ratio quality gate (spec §3.5.2)
    // Compute variance ratio: posterior_var / prior_var
    let variance_ratio = compute_variance_ratio(&bayes_result.lambda_post, &prior_cov_marginal);
    let data_too_noisy = variance_ratio > config.max_variance_ratio;

    if std::env::var("TIMING_ORACLE_DEBUG").is_ok() {
        eprintln!("[DEBUG] variance_ratio = {:.3}, max = {:.3}, data_too_noisy = {}",
            variance_ratio, config.max_variance_ratio, data_too_noisy);
    }

    // Build quality issues
    let mut quality_issues = Vec::new();
    if theta_eff > config.theta_ns && config.theta_ns > 0.0 {
        quality_issues.push(QualityIssue {
            code: IssueCode::ThresholdElevated,
            message: format!(
                "Threshold elevated from {:.0} ns to {:.1} ns (measurement floor)",
                config.theta_ns, theta_eff
            ),
            guidance: "For better resolution, use more samples or a higher-resolution timer.".to_string(),
        });
    }

    // v5.6: Emit KappaMixingPoor when kappa chain mixing is bad
    if !bayes_result.kappa_mixing_ok {
        quality_issues.push(QualityIssue {
            code: IssueCode::KappaMixingPoor,
            message: format!(
                "κ chain mixing poor (CV={:.2}, ESS={:.0})",
                bayes_result.kappa_cv, bayes_result.kappa_ess
            ),
            guidance: "Posterior may be unreliable; consider longer time budget.".to_string(),
        });
    }

    // v5.6: Emit LikelihoodInflated when kappa_mean < 0.3
    if bayes_result.kappa_mean < 0.3 {
        quality_issues.push(QualityIssue {
            code: IssueCode::LikelihoodInflated,
            message: format!(
                "Likelihood covariance inflated ~{:.1}x due to data/model mismatch",
                1.0 / bayes_result.kappa_mean
            ),
            guidance: "Uncertainty was increased for robustness. Effect estimates remain valid.".to_string(),
        });
    }

    // Build diagnostics
    // Use block_size from covariance estimation (spec §3.3.2 compliant)
    let block_length = cov_estimate.block_size;
    let diagnostics = Diagnostics {
        dependence_length: block_length,
        effective_sample_size: n / block_length.max(1),
        stationarity_ratio: 1.0, // Assume stationary for pre-collected data
        stationarity_ok: true,
        projection_mismatch_q: bayes_result.projection_mismatch_q,
        projection_mismatch_threshold: cov_estimate.q_thresh,
        projection_mismatch_ok: bayes_result.projection_mismatch_q <= cov_estimate.q_thresh,
        top_quantiles: None,
        outlier_rate_baseline: 0.0, // Not computed for pre-collected data
        outlier_rate_sample: 0.0,
        outlier_asymmetry_ok: true,
        discrete_mode,
        timer_resolution_ns: config.timer_resolution_ns,
        duplicate_fraction: 1.0 - min_uniqueness,
        preflight_ok: true, // Not applicable for pre-collected data
        preflight_warnings: Vec::new(),
        calibration_samples: n,
        total_time_secs: start_time.elapsed().as_secs_f64(),
        warnings: Vec::new(),
        quality_issues,
        seed: Some(config.seed),
        attacker_model: None,
        threshold_ns: config.theta_ns,
        timer_name: "external".to_string(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        timer_fallback_reason: None, // N/A for pre-collected data
        // Gibbs sampler diagnostics
        gibbs_iters_total: 256,
        gibbs_burnin: 64,
        gibbs_retained: 192,
        lambda_mean: bayes_result.lambda_mean,
        lambda_sd: bayes_result.lambda_sd,
        lambda_cv: bayes_result.lambda_cv,
        lambda_ess: bayes_result.lambda_ess,
        lambda_mixing_ok: bayes_result.lambda_mixing_ok,
        // v5.6 kappa diagnostics
        kappa_mean: bayes_result.kappa_mean,
        kappa_sd: bayes_result.kappa_sd,
        kappa_cv: bayes_result.kappa_cv,
        kappa_ess: bayes_result.kappa_ess,
        kappa_mixing_ok: bayes_result.kappa_mixing_ok,
    };

    // Step 10: Make decision based on quality check and posterior (v5.5 threshold elevation rule)
    let outcome = if data_too_noisy {
        // Quality gate failed - return Inconclusive
        Outcome::Inconclusive {
            reason: InconclusiveReason::DataTooNoisy {
                message: format!(
                    "Posterior variance is {:.0}% of prior; data not informative",
                    variance_ratio * 100.0
                ),
                guidance: "Try: more samples, higher-resolution timer, reduce system load".to_string(),
            },
            leak_probability,
            effect: effect_estimate.clone(),
            quality,
            diagnostics,
            samples_used: n,
            theta_user: config.theta_ns,
            theta_eff,
            theta_floor,
        }
    } else if leak_probability > config.fail_threshold {
        // Fail propagates regardless of threshold elevation (v5.5)
        let exploitability = Exploitability::from_effect_ns(effect_estimate.shift_ns.abs());
        Outcome::Fail {
            leak_probability,
            effect: effect_estimate.clone(),
            exploitability,
            quality,
            diagnostics,
            samples_used: n,
            theta_user: config.theta_ns,
            theta_eff,
            theta_floor,
        }
    } else if leak_probability < config.pass_threshold {
        // Pass requires both P < pass_threshold AND θ_eff ≤ θ_user + ε (v5.5)
        if is_threshold_elevated(theta_eff, config.theta_ns, theta_tick) {
            // Threshold elevated: return Inconclusive(ThresholdElevated) instead of Pass
            // For single-pass, we use n as max_samples since we can't collect more
            let achievable_at_max = compute_achievable_at_max(
                c_floor,
                theta_tick,
                config.theta_ns,
                n, // Single-pass has fixed samples
                block_length, // v5.6: block_length for n_eff computation
            );

            Outcome::Inconclusive {
                reason: InconclusiveReason::ThresholdElevated {
                    theta_user: config.theta_ns,
                    theta_eff,
                    leak_probability_at_eff: leak_probability,
                    meets_pass_criterion_at_eff: true,
                    achievable_at_max,
                    message: format!(
                        "Threshold elevated from {:.0}ns to {:.1}ns; P={:.1}% at elevated threshold",
                        config.theta_ns, theta_eff, leak_probability * 100.0
                    ),
                    guidance: "Use more samples or a higher-resolution timer to achieve the requested threshold.".to_string(),
                },
                leak_probability,
                effect: effect_estimate.clone(),
                quality,
                diagnostics,
                samples_used: n,
                theta_user: config.theta_ns,
                theta_eff,
                theta_floor,
            }
        } else {
            // Threshold not elevated: true Pass
            Outcome::Pass {
                leak_probability,
                effect: effect_estimate.clone(),
                quality,
                diagnostics,
                samples_used: n,
                theta_user: config.theta_ns,
                theta_eff,
                theta_floor,
            }
        }
    } else {
        // Between thresholds - inconclusive
        Outcome::Inconclusive {
            reason: InconclusiveReason::SampleBudgetExceeded {
                current_probability: leak_probability,
                samples_collected: n,
            },
            leak_probability,
            effect: effect_estimate.clone(),
            quality,
            diagnostics,
            samples_used: n,
            theta_user: config.theta_ns,
            theta_eff,
            theta_floor,
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

/// Compute variance ratio (average diagonal posterior/prior ratio).
fn compute_variance_ratio(posterior_cov: &timing_oracle_core::Matrix9, prior_cov: &timing_oracle_core::Matrix9) -> f64 {
    let mut sum = 0.0;
    for i in 0..9 {
        let prior_var = prior_cov[(i, i)];
        let post_var = posterior_cov[(i, i)];
        if prior_var > 1e-12 {
            sum += post_var / prior_var;
        }
    }
    sum / 9.0
}

/// Create default diagnostics for error cases.
fn make_default_diagnostics(timer_resolution_ns: f64) -> Diagnostics {
    Diagnostics {
        dependence_length: 1,
        effective_sample_size: 0,
        stationarity_ratio: 1.0,
        stationarity_ok: true,
        projection_mismatch_q: 0.0,
        projection_mismatch_threshold: f64::INFINITY,
        projection_mismatch_ok: true,
        top_quantiles: None,
        outlier_rate_baseline: 0.0,
        outlier_rate_sample: 0.0,
        outlier_asymmetry_ok: true,
        discrete_mode: false,
        timer_resolution_ns,
        duplicate_fraction: 0.0,
        preflight_ok: true,
        preflight_warnings: Vec::new(),
        calibration_samples: 0,
        total_time_secs: 0.0,
        warnings: Vec::new(),
        quality_issues: Vec::new(),
        seed: Some(0),
        attacker_model: None,
        threshold_ns: 0.0,
        timer_name: "external".to_string(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        timer_fallback_reason: None,
        gibbs_iters_total: 0,
        gibbs_burnin: 0,
        gibbs_retained: 0,
        lambda_mean: 1.0,
        lambda_sd: 0.0,
        lambda_cv: 0.0,
        lambda_ess: 0.0,
        lambda_mixing_ok: false,
        // v5.6 kappa diagnostics
        kappa_mean: 1.0,
        kappa_sd: 0.0,
        kappa_cv: 0.0,
        kappa_ess: 0.0,
        kappa_mixing_ok: false,
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

    #[test]
    fn test_diagnostics_populated() {
        let baseline = generate_samples(1000.0, 50.0, 1000, 42);
        let test = generate_samples(1000.0, 50.0, 1000, 43);

        let config = SinglePassConfig::default();
        let result = analyze_single_pass(&baseline, &test, &config);

        // Check that diagnostics are populated
        match &result.outcome {
            Outcome::Pass { diagnostics, .. } |
            Outcome::Fail { diagnostics, .. } |
            Outcome::Inconclusive { diagnostics, .. } => {
                assert!(diagnostics.dependence_length > 0, "Block length should be > 0");
                assert!(diagnostics.effective_sample_size > 0, "ESS should be > 0");
            }
            _ => {}
        }
    }

    #[test]
    fn test_timer_resolution_config() {
        let baseline = generate_samples(1000.0, 50.0, 1000, 42);
        let test = generate_samples(1000.0, 50.0, 1000, 43);

        // Test with high timer resolution (should affect theta_floor)
        let config = SinglePassConfig::for_attacker(AttackerModel::SharedHardware)
            .with_timer_resolution(42.0); // Apple Silicon-like resolution

        let result = analyze_single_pass(&baseline, &test, &config);

        // Should complete without error
        assert!(result.samples_used == 1000);
    }
}
