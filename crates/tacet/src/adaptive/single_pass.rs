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
//! - KL divergence quality check (spec §3.5.2)
//! - Proper MDE estimation for quality assessment

use std::time::{Duration, Instant};

use tacet_core::adaptive::{
    calibrate_floor_from_null, calibrate_halft_prior_scale_1d, compute_achievable_at_max,
    is_threshold_elevated,
};
use tacet_core::analysis::{compute_effect_estimate, estimate_mde};
use tacet_core::constants::{
    DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_FAIL_THRESHOLD, DEFAULT_PASS_THRESHOLD,
};
use tacet_core::result::{
    Diagnostics, EffectEstimate, Exploitability, InconclusiveReason, IssueCode, MeasurementQuality,
    Outcome, QualityIssue,
};
use tacet_core::statistics::{bootstrap_w1_variance, compute_w1_distance, AcquisitionStream};
use tacet_core::types::AttackerModel;

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

    /// Minimum KL divergence (nats) for quality gate (default 0.7, per spec §3.5.2).
    /// If KL(posterior || prior) < this threshold, data is not informative.
    pub kl_min: f64,

    /// Bootstrap resampling method (Joint or Stratified).
    pub bootstrap_method: tacet_core::statistics::BootstrapMethod,
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
            kl_min: 0.7,
            bootstrap_method: tacet_core::statistics::BootstrapMethod::default(),
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
        let effect = EffectEstimate::default();
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

    // Step 1: Compute raw W₁ distance (the observed effect)
    // Use raw W₁, NOT debiased — consistent with the adaptive loop (loop_runner.rs).
    // Debiased W₁ subtracts the null floor and clamps to zero, which biases the
    // likelihood and breaks the statistical model. Debiased is display-only.
    let w1_obs = compute_w1_distance(baseline, test);

    // Step 2: Detect discrete mode (spec §3.3.2)
    let unique_baseline: std::collections::HashSet<i64> =
        baseline.iter().map(|&v| v as i64).collect();
    let unique_test: std::collections::HashSet<i64> = test.iter().map(|&v| v as i64).collect();
    let min_uniqueness =
        (unique_baseline.len() as f64 / n as f64).min(unique_test.len() as f64 / n as f64);
    let discrete_mode = min_uniqueness < 0.10;

    // Step 3: Bootstrap W₁ variance estimation (spec §3.3.1)
    // Create interleaved acquisition stream per spec §3.3.1
    let mut acquisition_stream = AcquisitionStream::with_capacity(2 * n);
    acquisition_stream.push_batch_interleaved(baseline, test);
    let interleaved = acquisition_stream.to_timing_samples();

    let var_estimate = bootstrap_w1_variance(
        &interleaved,
        config.bootstrap_iterations,
        config.seed,
        discrete_mode, // is_fragile - use discrete mode flag
        config.bootstrap_method,
    );
    let variance = var_estimate.variance;

    // Check variance validity
    if variance <= 1e-12 {
        let effect = EffectEstimate::default();
        return SinglePassResult {
            outcome: Outcome::Inconclusive {
                reason: InconclusiveReason::DataTooNoisy {
                    message: "Variance estimation failed or too small".to_string(),
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

    // Step 4: Compute variance rate and measurement floor (spec §3.3.4)
    // var_rate = variance * n (variance scales as 1/n)
    let var_rate = variance * (n as f64);
    let block_length = var_estimate.block_size;

    // Compute null calibration (floor + null variance for inference)
    let null_cal = calibrate_floor_from_null(
        &interleaved,
        block_length,
        config.bootstrap_iterations,
        config.seed,
    );
    let c_floor = null_cal.c_floor;

    // Statistical floor: c_floor / √n_blocks
    let n_blocks = n.checked_div(block_length).unwrap_or(n).max(1);
    let theta_floor_stat = c_floor / (n_blocks as f64).sqrt();

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

    // Step 5: half-t prior calibration for 1D (spec §3.3.5)
    let sigma_t = calibrate_halft_prior_scale_1d(var_rate, config.theta_ns, n, config.seed);

    // Compute marginal prior variance for variance ratio check (spec §3.5.2)
    // For half-t_4: Var(|δ|) ≈ 2σ_t² (using half-normal approximation)
    let prior_var = 2.0 * sigma_t * sigma_t;

    // Step 6: Compute MDE for quality assessment (spec §3.4.6)
    let mde = estimate_mde(variance, 0.05); // α = 0.05
    let quality = MeasurementQuality::from_mde_ns(mde.mde_ns);

    // Debug output for investigation (first part — before posterior)
    if std::env::var("TIMING_ORACLE_DEBUG").is_ok() {
        eprintln!(
            "[DEBUG] n = {}, discrete_mode = {}, block_length = {}",
            n, discrete_mode, var_estimate.block_size
        );
        eprintln!("[DEBUG] theta_user = {:.2} ns, theta_floor_stat = {:.2} ns, theta_tick = {:.2} ns, theta_floor = {:.2} ns, theta_eff = {:.2} ns",
            config.theta_ns, theta_floor_stat, theta_tick, theta_floor, theta_eff);
        eprintln!("[DEBUG] c_floor = {:.2} ns·√n", c_floor);
        eprintln!("[DEBUG] MDE: {:.2} ns", mde.mde_ns);
        eprintln!("[DEBUG] w1_obs = {:.2} ns", w1_obs);
        eprintln!("[DEBUG] variance = {:.2e}", variance);
        eprintln!("[DEBUG] sigma_t = {:.2e}", sigma_t);
    }

    // Step 7: Compute 1D Bayesian posterior (spec §3.4.4)
    // var_n = var_rate / n = (variance * n) / n = variance
    // The bootstrap already gives Var(W₁) at the current sample size n.
    let var_n = var_rate / (n as f64);

    // Null-calibrated variance floor: prevents posterior overconfidence under
    // dependence (AR(1) etc.) where bootstrap may underestimate W₁ variance.
    // null_var_rate is Var(scaled_null_W₁) in rate form; dividing by n_blocks
    // gives the null variance at the current sample size.
    let null_var_n = null_cal.null_var_rate / (n_blocks as f64);
    let var_eff = var_n.max(null_var_n);

    if std::env::var("TIMING_ORACLE_DEBUG").is_ok() {
        eprintln!(
            "[DEBUG] var_n = {:.2e}, null_var_n = {:.2e}, var_eff = {:.2e} (null floor active: {})",
            var_n,
            null_var_n,
            var_eff,
            null_var_n > var_n
        );
    }

    let bayes_result = tacet_core::analysis::compute_bayes_1d(
        w1_obs,
        var_eff,
        sigma_t,
        theta_eff,
        config.seed,
        8.0, // nu_likelihood: Student-t df (§A.1: ν_ℓ = 8)
    );
    let leak_probability = bayes_result.leak_probability;

    if std::env::var("TIMING_ORACLE_DEBUG").is_ok() {
        eprintln!(
            "[DEBUG] bayes_result.leak_probability = {}",
            leak_probability
        );
        eprintln!(
            "[DEBUG] w1_post = {:.3}, var_post = {:.3e}",
            bayes_result.w1_post, bayes_result.var_post
        );
    }

    // Step 8: Compute effect estimate from Bayesian result
    let effect_estimate = compute_effect_estimate(&bayes_result.w1_draws);

    // Step 9: Check KL divergence quality gate (spec §3.5.2)
    // KL(posterior || prior) for Gaussian surrogates
    let v_ratio = bayes_result.var_post / prior_var;
    let mu_sq_term = (bayes_result.w1_post * bayes_result.w1_post) / prior_var;
    let kl_divergence = if v_ratio <= 0.0 || !v_ratio.is_finite() {
        f64::INFINITY
    } else {
        0.5 * (v_ratio + mu_sq_term - 1.0 + (1.0 / v_ratio).ln())
    };
    let data_too_noisy = kl_divergence < config.kl_min;

    if std::env::var("TIMING_ORACLE_DEBUG").is_ok() {
        eprintln!(
            "[DEBUG] kl_divergence = {:.3} nats, kl_min = {:.3}, data_too_noisy = {}",
            kl_divergence, config.kl_min, data_too_noisy
        );
    }

    // Build quality issues
    let mut quality_issues = Vec::new();
    if theta_eff > config.theta_ns && config.theta_ns > 0.0 {
        quality_issues.push(QualityIssue {
            code: IssueCode::ThresholdIssue,
            message: format!(
                "Threshold elevated from {:.0} ns to {:.1} ns (measurement floor)",
                config.theta_ns, theta_eff
            ),
            guidance: "For better resolution, use more samples or a higher-resolution timer."
                .to_string(),
        });
    }

    // Build diagnostics
    let diagnostics = Diagnostics {
        dependence_length: block_length,
        effective_sample_size: n / block_length.max(1),
        stationarity_ratio: 1.0, // Assume stationary for pre-collected data
        stationarity_ok: true,
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
        // Gibbs sampler diagnostics (no longer applicable for 1D, use defaults)
        gibbs_iters_total: 5000,
        gibbs_burnin: 1000,
        gibbs_retained: 4000,
        lambda_mean: 1.0,
        lambda_sd: 0.0,
        lambda_cv: 0.0,
        lambda_ess: 0.0,
        lambda_mixing_ok: true,
        // v5.6 kappa diagnostics (not used in 1D W₁)
        kappa_mean: 1.0,
        kappa_sd: 0.0,
        kappa_cv: 0.0,
        kappa_ess: 0.0,
        kappa_mixing_ok: true,
    };

    // Step 10: Make decision based on quality check and posterior (v5.5 threshold elevation rule)
    let outcome = if data_too_noisy {
        // Quality gate failed - return Inconclusive
        Outcome::Inconclusive {
            reason: InconclusiveReason::DataTooNoisy {
                message: format!(
                    "KL divergence {:.2} nats < {:.1} nats threshold; data not informative",
                    kl_divergence, config.kl_min
                ),
                guidance: "Try: more samples, higher-resolution timer, reduce system load"
                    .to_string(),
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
        let exploitability = Exploitability::from_effect_ns(effect_estimate.max_effect_ns.abs());
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
                n,            // Single-pass has fixed samples
                block_length, // v5.6: block_length for n_blocks computation
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

/// Create default diagnostics for error cases.
fn make_default_diagnostics(timer_resolution_ns: f64) -> Diagnostics {
    Diagnostics {
        dependence_length: 1,
        effective_sample_size: 0,
        stationarity_ratio: 1.0,
        stationarity_ok: true,
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
            Outcome::Pass { diagnostics, .. }
            | Outcome::Fail { diagnostics, .. }
            | Outcome::Inconclusive { diagnostics, .. } => {
                assert!(
                    diagnostics.dependence_length > 0,
                    "Block length should be > 0"
                );
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
