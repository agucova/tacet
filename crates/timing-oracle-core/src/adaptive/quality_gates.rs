//! Quality gates for the adaptive sampling loop.
//!
//! Quality gates detect conditions where continuing to sample is unlikely to
//! yield a definitive answer. They prevent wasted computation and provide
//! actionable feedback to users.
//!
//! Gates are checked in order of priority:
//! 1. Posterior too close to prior (data uninformative)
//! 2. Learning rate collapsed (posterior stopped updating)
//! 3. Would take too long (extrapolated time exceeds budget)
//! 4. Time budget exceeded
//! 5. Sample budget exceeded
//! 6. Condition drift detected (calibration assumptions violated)

use alloc::string::String;

use super::drift::{CalibrationSnapshot, ConditionDrift, DriftThresholds};
use super::Posterior;
use crate::constants::{DEFAULT_FAIL_THRESHOLD, DEFAULT_MAX_SAMPLES, DEFAULT_PASS_THRESHOLD};
use crate::types::Matrix9;

/// Result of quality gate checks.
#[derive(Debug, Clone)]
pub enum QualityGateResult {
    /// All gates passed, continue sampling.
    Continue,

    /// A gate triggered, stop with inconclusive result.
    Stop(InconclusiveReason),
}

/// Reason why the adaptive loop stopped inconclusively.
#[derive(Debug, Clone)]
pub enum InconclusiveReason {
    /// Posterior is too close to prior - data isn't informative.
    DataTooNoisy {
        /// Human-readable message.
        message: String,
        /// Suggested remediation.
        guidance: String,
        /// Variance ratio (posterior/prior).
        variance_ratio: f64,
    },

    /// Posterior stopped updating despite new data.
    NotLearning {
        /// Human-readable message.
        message: String,
        /// Suggested remediation.
        guidance: String,
        /// Sum of recent KL divergences.
        recent_kl_sum: f64,
    },

    /// Estimated time to decision exceeds acceptable limit.
    WouldTakeTooLong {
        /// Estimated time in seconds.
        estimated_time_secs: f64,
        /// Estimated samples needed.
        samples_needed: usize,
        /// Suggested remediation.
        guidance: String,
    },

    /// Time budget exceeded without reaching decision.
    TimeBudgetExceeded {
        /// Current leak probability.
        current_probability: f64,
        /// Samples collected so far.
        samples_collected: usize,
        /// Time spent.
        elapsed_secs: f64,
    },

    /// Sample budget exceeded without reaching decision.
    SampleBudgetExceeded {
        /// Current leak probability.
        current_probability: f64,
        /// Samples collected.
        samples_collected: usize,
    },

    /// Measurement conditions changed during the test (Gate 6).
    ///
    /// Detected by comparing calibration statistics with post-test statistics.
    /// This can indicate environmental interference (CPU frequency scaling,
    /// concurrent processes, etc.) that invalidates the covariance estimate.
    ConditionsChanged {
        /// Human-readable description of what changed.
        message: String,
        /// Suggested remediation.
        guidance: String,
        /// The specific drift metrics that were detected.
        drift_description: String,
    },

    /// Threshold was elevated and pass criterion was met at effective threshold.
    ///
    /// The measurement floor exceeded the user's requested threshold, so inference
    /// was performed at an elevated effective threshold. The posterior probability
    /// dropped below pass_threshold at θ_eff, but since θ_eff > θ_user + ε, we
    /// cannot guarantee the user's original requirement is met.
    ///
    /// This is NOT a quality gate - it's checked at decision time in loop_runner.
    ThresholdElevated {
        /// User's requested threshold in nanoseconds (θ_user).
        theta_user: f64,
        /// Effective threshold used for inference (θ_eff = max(θ_user, θ_floor)).
        theta_eff: f64,
        /// Posterior probability at θ_eff (was < pass_threshold).
        leak_probability_at_eff: f64,
        /// True: P(leak > θ_eff) < pass_threshold.
        meets_pass_criterion_at_eff: bool,
        /// True: θ_floor at max_samples would be ≤ θ_user + ε.
        achievable_at_max: bool,
        /// Human-readable message.
        message: String,
        /// Suggested remediation.
        guidance: String,
    },
}

/// Configuration for quality gate thresholds.
#[derive(Debug, Clone)]
pub struct QualityGateConfig {
    /// Maximum variance ratio (posterior/prior) before declaring data uninformative.
    /// Default: 0.5 (posterior variance must be at most 50% of prior).
    pub max_variance_ratio: f64,

    /// Minimum sum of recent KL divergences before declaring learning stalled.
    /// Default: 0.001
    pub min_kl_sum: f64,

    /// Maximum extrapolated time as multiple of budget.
    /// Default: 10.0 (stop if estimated time > 10x budget).
    pub max_time_multiplier: f64,

    /// Time budget for adaptive sampling in seconds.
    pub time_budget_secs: f64,

    /// Maximum samples per class.
    pub max_samples: usize,

    /// Pass threshold for leak probability.
    pub pass_threshold: f64,

    /// Fail threshold for leak probability.
    pub fail_threshold: f64,

    /// Whether to enable condition drift detection (Gate 6).
    /// Default: true
    pub enable_drift_detection: bool,

    /// Thresholds for condition drift detection.
    pub drift_thresholds: DriftThresholds,
}

impl Default for QualityGateConfig {
    fn default() -> Self {
        Self {
            max_variance_ratio: 0.5,
            min_kl_sum: 0.001,
            max_time_multiplier: 10.0,
            time_budget_secs: 30.0,
            max_samples: DEFAULT_MAX_SAMPLES,
            pass_threshold: DEFAULT_PASS_THRESHOLD,
            fail_threshold: DEFAULT_FAIL_THRESHOLD,
            enable_drift_detection: true,
            drift_thresholds: DriftThresholds::default(),
        }
    }
}

/// Inputs required to check quality gates.
///
/// This is a stateless struct that contains all the information needed to
/// check quality gates, avoiding the need for mutable state or time tracking.
#[derive(Debug)]
pub struct QualityGateCheckInputs<'a> {
    /// Current posterior distribution.
    pub posterior: &'a Posterior,

    /// Marginal prior covariance matrix Λ₀^marginal = 2σ²R (for ν=4).
    /// This is the unconditional prior variance of δ under the t-prior.
    /// Used by Gate 1 for the KL divergence check (spec §3.5.2).
    pub prior_cov_marginal: &'a Matrix9,

    /// Effect threshold θ in nanoseconds (user's requested threshold).
    pub theta_ns: f64,

    /// Total samples per class collected so far.
    pub n_total: usize,

    /// Elapsed time in seconds since adaptive phase started.
    pub elapsed_secs: f64,

    /// Sum of recent KL divergences (last 5 batches).
    /// Pass `None` if fewer than 5 batches have been collected.
    pub recent_kl_sum: Option<f64>,

    /// Samples per second (throughput from calibration).
    pub samples_per_second: f64,

    /// Calibration snapshot for drift detection.
    /// Pass `None` to skip drift detection.
    pub calibration_snapshot: Option<&'a CalibrationSnapshot>,

    /// Current stats snapshot for drift detection.
    /// Pass `None` to skip drift detection.
    pub current_stats_snapshot: Option<&'a CalibrationSnapshot>,

    /// Floor-rate constant (c_floor) from calibration.
    /// Used to compute theta_floor(n) = c_floor / sqrt(n).
    pub c_floor: f64,

    /// Timer tick floor (theta_tick) from calibration.
    /// The floor below which timer quantization dominates.
    pub theta_tick: f64,

    /// Projection mismatch Q statistic (r^T Σ^{-1} r).
    /// Pass `None` if not yet computed.
    pub projection_mismatch_q: Option<f64>,

    /// Projection mismatch threshold from bootstrap calibration.
    pub projection_mismatch_thresh: f64,

    // ==================== v5.4 Gibbs sampler fields ====================

    /// Whether the Gibbs sampler's lambda chain mixed well (v5.4).
    /// `None` if not using Gibbs sampler (mixture mode).
    /// When `Some(false)`, indicates potential posterior unreliability.
    pub lambda_mixing_ok: Option<bool>,
}

/// Check all quality gates and return result.
///
/// Gates are checked in priority order. Returns `Continue` if all pass,
/// or `Stop` with the reason if any gate triggers.
///
/// **Gate order (spec Section 3.5.2, v5.5):**
/// 1. Posterior too close to prior (data not informative)
/// 2. Learning rate collapsed
/// 3. Would take too long
/// 4. Time budget exceeded
/// 5. Sample budget exceeded
/// 6. Condition drift detected
///
/// **Note**: Threshold elevation (v5.5) is NOT a quality gate. It's checked at
/// decision time in loop_runner.rs. The decision rule requires:
/// - Pass: P < pass_threshold AND θ_eff ≤ θ_user + ε
/// - Fail: P > fail_threshold (propagates regardless of elevation)
/// - ThresholdElevated: P < pass_threshold AND θ_eff > θ_user + ε
///
/// # Arguments
///
/// * `inputs` - All inputs needed for gate checks (stateless)
/// * `config` - Quality gate configuration
pub fn check_quality_gates(
    inputs: &QualityGateCheckInputs,
    config: &QualityGateConfig,
) -> QualityGateResult {
    // Gate 1: Posterior too close to prior (data not informative) - v5.6: uses KL divergence
    if let Some(reason) = check_kl_divergence(inputs, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 2: Learning rate collapsed
    if let Some(reason) = check_learning_rate(inputs, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 3: Would take too long
    if let Some(reason) = check_extrapolated_time(inputs, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 4: Time budget exceeded
    if let Some(reason) = check_time_budget(inputs, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 5: Sample budget exceeded
    if let Some(reason) = check_sample_budget(inputs, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 6: Condition drift detected
    if let Some(reason) = check_condition_drift(inputs, config) {
        return QualityGateResult::Stop(reason);
    }

    QualityGateResult::Continue
}

/// Check if the requested threshold is achievable at max_samples (helper for v5.5).
///
/// Returns `true` if theta_floor at max_samples would be ≤ theta_user + epsilon,
/// meaning more samples could eventually achieve the user's threshold.
///
/// This is NOT a verdict-blocking gate in v5.5. It's used by the decision logic
/// to populate the `achievable_at_max` field in ThresholdElevated outcomes.
///
/// v5.6: Uses n_eff = max(1, floor(max_samples / block_length)) for proper
/// scaling under temporal dependence.
pub fn compute_achievable_at_max(
    c_floor: f64,
    theta_tick: f64,
    theta_user: f64,
    max_samples: usize,
    block_length: usize,
) -> bool {
    // Research mode (theta_user = 0) is always "achievable" (no user target)
    if theta_user <= 0.0 {
        return true;
    }

    // v5.6: Use n_eff for achievability check (spec §3.3.4)
    let n_eff_max = if block_length > 0 {
        (max_samples / block_length).max(1)
    } else {
        max_samples.max(1)
    };

    // Compute theta_floor at max_samples using n_eff
    let theta_floor_at_max = libm::fmax(
        c_floor / libm::sqrt(n_eff_max as f64),
        theta_tick,
    );

    // Compute epsilon: max(theta_tick, 1e-6 * theta_user)
    let epsilon = libm::fmax(theta_tick, 1e-6 * theta_user);

    // Achievable if floor at max_samples would be within tolerance of user threshold
    theta_floor_at_max <= theta_user + epsilon
}

/// Check if the threshold is elevated beyond tolerance (v5.5).
///
/// Returns `true` if θ_eff > θ_user + ε, meaning the effective threshold
/// is elevated beyond the tolerance band around the user's requested threshold.
///
/// The epsilon tolerance is: ε = max(θ_tick, 1e-6 * θ_user)
///
/// This check is used at decision time: if P < pass_threshold but the threshold
/// is elevated, we return ThresholdElevated instead of Pass.
pub fn is_threshold_elevated(theta_eff: f64, theta_user: f64, theta_tick: f64) -> bool {
    // Research mode (theta_user <= 0) is never "elevated"
    if theta_user <= 0.0 {
        return false;
    }

    // Compute epsilon: max(theta_tick, 1e-6 * theta_user)
    let epsilon = libm::fmax(theta_tick, 1e-6 * theta_user);

    // Elevated if effective threshold exceeds user threshold + tolerance
    theta_eff > theta_user + epsilon
}

/// Minimum KL divergence threshold (nats) for Gate 1 (spec §3.5.2 v5.6).
const KL_MIN: f64 = 0.7;

/// Gate 1: Check if posterior learned from prior using KL divergence (spec §3.5.2 v5.6).
///
/// v5.6 CHANGE: Replaces the log-det variance ratio with full KL divergence.
///
/// KL = 0.5 * (tr(Λ₀⁻¹Λ_post) + μ_postᵀΛ₀⁻¹μ_post - 9 + ln|Λ₀|/|Λ_post|)
///
/// Gate triggers when KL < KL_min where KL_min := 0.7 nats.
///
/// v5.6 CHANGE: NO decisive probability bypass - the bypass logic is removed entirely.
/// This ensures consistent treatment regardless of observed leak probability.
///
/// **Cholesky fallback (spec §3.5.2)**: If Cholesky fails after jitter ladder,
/// uses trace ratio as conservative fallback.
fn check_kl_divergence(
    inputs: &QualityGateCheckInputs,
    _config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    // v5.6: NO decisive probability bypass - removed per spec
    // The gate checks KL divergence unconditionally

    // v5.4: Use the marginal prior covariance Λ₀^marginal = 2σ²R (spec §3.5.2)
    let prior_cov = inputs.prior_cov_marginal;
    let post_cov = &inputs.posterior.lambda_post;
    let post_mean = &inputs.posterior.delta_post;

    // Try to compute KL divergence via Cholesky
    let kl = match compute_kl_divergence(prior_cov, post_cov, post_mean) {
        Some(kl) => kl,
        None => {
            // Fall back to trace ratio if KL computation fails
            let trace_ratio = post_cov.trace() / prior_cov.trace();
            if trace_ratio > 0.5 {
                return Some(InconclusiveReason::DataTooNoisy {
                    message: alloc::format!(
                        "Posterior variance is {:.0}% of prior; data not informative (KL computation failed)",
                        trace_ratio * 100.0
                    ),
                    guidance: String::from("Try: cycle counter, reduce system load, increase batch size"),
                    variance_ratio: trace_ratio,
                });
            }
            return None;
        }
    };

    // Trigger when KL < KL_min (spec §3.5.2 v5.6)
    if kl < KL_MIN {
        return Some(InconclusiveReason::DataTooNoisy {
            message: alloc::format!(
                "KL divergence {:.2} nats < {:.1} threshold; posterior ≈ prior",
                kl, KL_MIN
            ),
            guidance: String::from("Try: cycle counter, reduce system load, increase batch size"),
            variance_ratio: kl / KL_MIN, // Report KL ratio for diagnostics
        });
    }

    None
}

/// Compute KL(N(μ_post, Λ_post) || N(0, Λ₀)) via Cholesky (spec §3.5.2 v5.6).
///
/// KL = 0.5 * (tr(Λ₀⁻¹Λ_post) + μ_postᵀΛ₀⁻¹μ_post - d + ln|Λ₀| - ln|Λ_post|)
///
/// Returns `None` if Cholesky fails even after jitter ladder.
fn compute_kl_divergence(
    prior_cov: &crate::types::Matrix9,
    post_cov: &crate::types::Matrix9,
    post_mean: &crate::types::Vector9,
) -> Option<f64> {
    // Try Cholesky with jitter ladder (spec §3.5.2)
    let prior_chol = try_cholesky_with_jitter(prior_cov)?;
    let post_chol = try_cholesky_with_jitter(post_cov)?;

    // Log determinants from Cholesky: log|A| = 2 * sum(log(L_ii))
    let prior_log_det: f64 = (0..9)
        .map(|i| libm::log(prior_chol.l()[(i, i)]))
        .sum::<f64>() * 2.0;
    let post_log_det: f64 = (0..9)
        .map(|i| libm::log(post_chol.l()[(i, i)]))
        .sum::<f64>() * 2.0;

    if !prior_log_det.is_finite() || !post_log_det.is_finite() {
        return None;
    }

    // tr(Λ₀⁻¹Λ_post) via Cholesky solves
    let mut trace_term = 0.0;
    for j in 0..9 {
        let col = post_cov.column(j).into_owned();
        let solved = prior_chol.solve(&col);
        trace_term += solved[j];
    }

    // μ_postᵀΛ₀⁻¹μ_post
    let solved_mean = prior_chol.solve(post_mean);
    let quad_term = post_mean.dot(&solved_mean);

    // KL = 0.5 * (tr + quad - d + ln|Λ₀| - ln|Λ_post|)
    let kl = 0.5 * (trace_term + quad_term - 9.0 + prior_log_det - post_log_det);

    // KL should be non-negative
    Some(kl.max(0.0))
}

/// Try Cholesky decomposition with jitter ladder (spec §3.5.2 v5.6).
///
/// Starts at 10⁻¹⁰ and increases by factors of 10 until success or 10⁻⁴.
fn try_cholesky_with_jitter(
    matrix: &crate::types::Matrix9,
) -> Option<nalgebra::Cholesky<f64, nalgebra::Const<9>>> {
    // Try without jitter first
    if let Some(chol) = nalgebra::Cholesky::new(*matrix) {
        return Some(chol);
    }

    // Jitter ladder: 10⁻¹⁰, 10⁻⁹, ..., 10⁻⁴
    for exp in -10..=-4 {
        let jitter = libm::pow(10.0, exp as f64);
        let jittered = matrix + crate::types::Matrix9::identity() * jitter;
        if let Some(chol) = nalgebra::Cholesky::new(jittered) {
            return Some(chol);
        }
    }

    None
}

/// Gate 2: Check if learning has stalled (posterior stopped updating).
fn check_learning_rate(
    inputs: &QualityGateCheckInputs,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    let recent_kl_sum = inputs.recent_kl_sum?;

    if recent_kl_sum < config.min_kl_sum {
        return Some(InconclusiveReason::NotLearning {
            message: String::from("Posterior stopped updating despite new data"),
            guidance: String::from(
                "Measurement may have systematic issues or effect is very close to boundary",
            ),
            recent_kl_sum,
        });
    }

    None
}

/// Gate 3: Check if reaching a decision would take too long.
fn check_extrapolated_time(
    inputs: &QualityGateCheckInputs,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    // Need at least some samples to extrapolate
    if inputs.n_total < 100 {
        return None;
    }

    let samples_needed = extrapolate_samples_to_decision(inputs, config);

    if samples_needed == usize::MAX {
        // Can't extrapolate
        return None;
    }

    let additional_samples = samples_needed.saturating_sub(inputs.n_total);
    let time_needed_secs = additional_samples as f64 / inputs.samples_per_second;

    if time_needed_secs > config.time_budget_secs * config.max_time_multiplier {
        return Some(InconclusiveReason::WouldTakeTooLong {
            estimated_time_secs: time_needed_secs,
            samples_needed,
            guidance: alloc::format!(
                "Effect may be very close to threshold; consider adjusting theta (current: {:.1}ns)",
                inputs.theta_ns
            ),
        });
    }

    None
}

/// Gate 4: Check if time budget is exceeded.
fn check_time_budget(
    inputs: &QualityGateCheckInputs,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    if inputs.elapsed_secs > config.time_budget_secs {
        return Some(InconclusiveReason::TimeBudgetExceeded {
            current_probability: inputs.posterior.leak_probability,
            samples_collected: inputs.n_total,
            elapsed_secs: inputs.elapsed_secs,
        });
    }

    None
}

/// Gate 5: Check if sample budget is exceeded.
fn check_sample_budget(
    inputs: &QualityGateCheckInputs,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    if inputs.n_total >= config.max_samples {
        return Some(InconclusiveReason::SampleBudgetExceeded {
            current_probability: inputs.posterior.leak_probability,
            samples_collected: inputs.n_total,
        });
    }

    None
}

/// Extrapolate how many samples are needed to reach a decision.
///
/// Uses the fact that posterior standard deviation decreases as sqrt(n).
/// If current std is much larger than the margin to threshold, we can
/// estimate how many more samples are needed.
fn extrapolate_samples_to_decision(
    inputs: &QualityGateCheckInputs,
    config: &QualityGateConfig,
) -> usize {
    let p = inputs.posterior.leak_probability;

    // Distance to nearest threshold
    let margin = libm::fmin(
        libm::fabs(p - config.pass_threshold),
        libm::fabs(config.fail_threshold - p),
    );

    if margin < 1e-9 {
        return usize::MAX; // Already at threshold
    }

    // Posterior std (use trace of 2D projection as proxy for overall uncertainty)
    let current_std = libm::sqrt(inputs.posterior.beta_proj_cov.trace());

    if current_std < 1e-9 {
        return inputs.n_total; // Already very certain
    }

    // Std scales as 1/sqrt(n), so to reduce std by factor k we need k^2 more samples
    // We need std to be comparable to margin for a clear decision
    let std_reduction_needed = current_std / margin;

    if std_reduction_needed <= 1.0 {
        // Current uncertainty is already small enough
        return inputs.n_total;
    }

    let sample_multiplier = std_reduction_needed * std_reduction_needed;

    // Cap at 100x current to avoid overflow
    let multiplier = libm::fmin(sample_multiplier, 100.0);

    libm::ceil(inputs.n_total as f64 * multiplier) as usize
}

/// Gate 6: Check if measurement conditions changed during the test.
///
/// Compares the calibration statistics snapshot with the current state's
/// online statistics to detect environmental interference (CPU frequency
/// scaling, concurrent processes, etc.) that would invalidate the covariance
/// estimate.
///
/// See spec Section 2.6, Gate 6.
fn check_condition_drift(
    inputs: &QualityGateCheckInputs,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    // Skip if drift detection is disabled
    if !config.enable_drift_detection {
        return None;
    }

    // Need both snapshots to detect drift
    let cal_snapshot = inputs.calibration_snapshot?;
    let post_snapshot = inputs.current_stats_snapshot?;

    // Compute drift between calibration and post-test
    let drift = ConditionDrift::compute(cal_snapshot, post_snapshot);

    // Check if drift exceeds thresholds
    if drift.is_significant(&config.drift_thresholds) {
        return Some(InconclusiveReason::ConditionsChanged {
            message: String::from("Measurement conditions changed during test"),
            guidance: String::from(
                "Ensure stable environment: disable CPU frequency scaling, \
                minimize concurrent processes, use performance CPU governor",
            ),
            drift_description: drift.description(&config.drift_thresholds),
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::StatsSnapshot;
    use crate::types::{Matrix2, Vector2, Vector9};

    fn make_posterior(leak_prob: f64, variance: f64) -> Posterior {
        Posterior::new(
            Vector9::zeros(),
            Matrix9::identity() * variance,
            Vector2::new(5.0, 3.0),
            Matrix2::new(variance, 0.0, 0.0, variance),
            Vec::new(), // beta_draws
            leak_prob,
            1.0, // projection_mismatch_q
            1000,
        )
    }

    fn make_prior_cov_marginal() -> Matrix9 {
        Matrix9::identity() * 100.0 // Prior variance = 100 per dimension
    }

    fn make_inputs<'a>(
        posterior: &'a Posterior,
        prior_cov_marginal: &'a Matrix9,
    ) -> QualityGateCheckInputs<'a> {
        QualityGateCheckInputs {
            posterior,
            prior_cov_marginal,
            theta_ns: 100.0,
            n_total: 5000,
            elapsed_secs: 5.0,
            recent_kl_sum: Some(0.05),
            samples_per_second: 100_000.0,
            calibration_snapshot: None,
            current_stats_snapshot: None,
            c_floor: 3535.5, // ~50 * sqrt(5000)
            theta_tick: 1.0,
            projection_mismatch_q: None,
            projection_mismatch_thresh: 18.48,
            lambda_mixing_ok: None,
        }
    }

    #[test]
    fn test_kl_divergence_gate_passes() {
        // v5.6: Use KL divergence instead of variance ratio
        // Small posterior variance = large covariance contraction = high KL
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_marginal = make_prior_cov_marginal();
        let inputs = make_inputs(&posterior, &prior_cov_marginal);
        let config = QualityGateConfig::default();

        let result = check_kl_divergence(&inputs, &config);
        assert!(result.is_none(), "Low posterior variance should give high KL (pass)");
    }

    #[test]
    fn test_kl_divergence_gate_fails() {
        // v5.6: Use KL divergence instead of variance ratio
        // High posterior variance ≈ prior = low KL
        let posterior = make_posterior(0.5, 95.0); // Close to prior variance of 100
        let prior_cov_marginal = make_prior_cov_marginal();
        let inputs = make_inputs(&posterior, &prior_cov_marginal);
        let config = QualityGateConfig::default();

        let result = check_kl_divergence(&inputs, &config);
        assert!(matches!(
            result,
            Some(InconclusiveReason::DataTooNoisy { .. })
        ), "Posterior ≈ prior should give low KL (fail)");
    }

    #[test]
    fn test_learning_rate_gate_passes() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_marginal = make_prior_cov_marginal();
        let mut inputs = make_inputs(&posterior, &prior_cov_marginal);
        inputs.recent_kl_sum = Some(0.05); // Sum > 0.001
        let config = QualityGateConfig::default();

        let result = check_learning_rate(&inputs, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_learning_rate_gate_fails() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_marginal = make_prior_cov_marginal();
        let mut inputs = make_inputs(&posterior, &prior_cov_marginal);
        inputs.recent_kl_sum = Some(0.0005); // Sum < 0.001
        let config = QualityGateConfig::default();

        let result = check_learning_rate(&inputs, &config);
        assert!(matches!(
            result,
            Some(InconclusiveReason::NotLearning { .. })
        ));
    }

    #[test]
    fn test_time_budget_gate() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_marginal = make_prior_cov_marginal();
        let mut inputs = make_inputs(&posterior, &prior_cov_marginal);
        inputs.elapsed_secs = 35.0; // Exceeds 30s budget
        let config = QualityGateConfig::default();

        let result = check_time_budget(&inputs, &config);
        assert!(matches!(
            result,
            Some(InconclusiveReason::TimeBudgetExceeded { .. })
        ));
    }

    #[test]
    fn test_sample_budget_gate() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_marginal = make_prior_cov_marginal();
        let mut inputs = make_inputs(&posterior, &prior_cov_marginal);
        inputs.n_total = 1_000_001; // Exceeds 1M budget
        let config = QualityGateConfig::default();

        let result = check_sample_budget(&inputs, &config);
        assert!(matches!(
            result,
            Some(InconclusiveReason::SampleBudgetExceeded { .. })
        ));
    }

    #[test]
    fn test_condition_drift_gate_no_snapshots() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_marginal = make_prior_cov_marginal();
        let inputs = make_inputs(&posterior, &prior_cov_marginal);
        // No snapshots provided
        let config = QualityGateConfig::default();

        let result = check_condition_drift(&inputs, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_condition_drift_gate_no_drift() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_marginal = make_prior_cov_marginal();

        let stats = StatsSnapshot {
            mean: 100.0,
            variance: 25.0,
            autocorr_lag1: 0.1,
            count: 5000,
        };
        let cal_snapshot = CalibrationSnapshot::new(stats, stats);
        let post_snapshot = CalibrationSnapshot::new(stats, stats);

        let mut inputs = make_inputs(&posterior, &prior_cov_marginal);
        inputs.calibration_snapshot = Some(&cal_snapshot);
        inputs.current_stats_snapshot = Some(&post_snapshot);

        let config = QualityGateConfig::default();

        let result = check_condition_drift(&inputs, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_condition_drift_gate_detects_variance_change() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_marginal = make_prior_cov_marginal();

        let cal_stats = StatsSnapshot {
            mean: 100.0,
            variance: 25.0,
            autocorr_lag1: 0.1,
            count: 5000,
        };
        let post_stats = StatsSnapshot {
            mean: 100.0,
            variance: 75.0, // 3x variance increase
            autocorr_lag1: 0.1,
            count: 5000,
        };
        let cal_snapshot = CalibrationSnapshot::new(cal_stats, cal_stats);
        let post_snapshot = CalibrationSnapshot::new(post_stats, post_stats);

        let mut inputs = make_inputs(&posterior, &prior_cov_marginal);
        inputs.calibration_snapshot = Some(&cal_snapshot);
        inputs.current_stats_snapshot = Some(&post_snapshot);

        let config = QualityGateConfig::default();

        let result = check_condition_drift(&inputs, &config);
        assert!(matches!(
            result,
            Some(InconclusiveReason::ConditionsChanged { .. })
        ));
    }

    #[test]
    fn test_full_quality_gates_pass() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_marginal = make_prior_cov_marginal();
        let inputs = make_inputs(&posterior, &prior_cov_marginal);
        let config = QualityGateConfig::default();

        let result = check_quality_gates(&inputs, &config);
        assert!(matches!(result, QualityGateResult::Continue));
    }
}
