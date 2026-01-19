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
            max_samples: 1_000_000,
            pass_threshold: 0.05,
            fail_threshold: 0.95,
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

    /// 9D prior covariance matrix (from calibration).
    /// For v5.2 mixture prior, this should be the narrow component prior.
    /// Use `prior_cov_slab` and `narrow_weight_post` for dominant component selection.
    pub prior_cov_9d: &'a Matrix9,

    /// v5.4: Marginal prior covariance matrix Λ₀^marginal = 2σ²R (for ν=4).
    /// This is the unconditional prior variance of δ under the t-prior.
    /// Used by Gate 1 for the variance ratio check (spec §3.5.2).
    pub prior_cov_marginal: &'a Matrix9,

    /// v5.2: Slab (wide) prior covariance matrix.
    /// Pass `None` if not using mixture prior.
    pub prior_cov_slab: Option<&'a Matrix9>,

    /// v5.2: Posterior weight of the narrow component (0.0-1.0).
    /// Pass `None` if not using mixture prior.
    /// When < 0.5, the slab component dominates and should be used for quality gates.
    pub narrow_weight_post: Option<f64>,

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
    // Gate 1: Posterior too close to prior (data not informative)
    if let Some(reason) = check_variance_ratio(inputs, config) {
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
pub fn compute_achievable_at_max(
    c_floor: f64,
    theta_tick: f64,
    theta_user: f64,
    max_samples: usize,
) -> bool {
    // Research mode (theta_user = 0) is always "achievable" (no user target)
    if theta_user <= 0.0 {
        return true;
    }

    // Compute theta_floor at max_samples
    let theta_floor_at_max = libm::fmax(
        c_floor / libm::sqrt(max_samples as f64),
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

/// Gate 1: Check if posterior variance is too close to prior variance (spec §3.5.2).
///
/// Uses 9D log-det ratio: ρ = log(|Λ_post| / |Λ₀^marginal|)
/// where Λ₀^marginal = 2σ²R is the marginal prior covariance under t-prior (ν=4).
///
/// Gate triggers when ρ > log(0.5) (i.e., |Λ_post|/|Λ₀^marginal| > 0.5).
///
/// **Cholesky fallback (spec §3.5.2)**: If Cholesky fails, uses trace ratio instead:
/// ρ_trace = tr(Λ_post) / tr(Λ₀^marginal), triggers if ρ_trace > 0.5.
///
/// **Exception**: If the leak probability is already decisive (>99.5% or <0.5%)
/// AND the projection mismatch Q is reasonable (< 1000x threshold),
/// we allow the verdict through. This handles cases like slow operations
/// (modexp) where variance doesn't shrink much but there's a clear timing leak.
fn check_variance_ratio(
    inputs: &QualityGateCheckInputs,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    // Allow confident verdicts through IF:
    // 1. leak_probability is very high (>99.5%) or very low (<0.5%)
    // 2. AND projection mismatch Q is not astronomically high (< 1000x threshold)
    //
    // Astronomical Q (e.g., 100000+) indicates pathological data patterns that don't
    // resemble normal timing leaks - likely measurement artifacts.
    // Moderate Q (e.g., 100-1000) can occur with real but unusual timing patterns.
    let p = inputs.posterior.leak_probability;
    let q = inputs.projection_mismatch_q.unwrap_or(0.0);
    let q_limit = inputs.projection_mismatch_thresh * 1000.0; // Filter only extreme pathological cases

    if !(0.005..=0.995).contains(&p) && q < q_limit {
        return None;
    }

    // v5.4: Use the marginal prior covariance Λ₀^marginal = 2σ²R (spec §3.5.2)
    let prior_cov = inputs.prior_cov_marginal;
    let post_cov = &inputs.posterior.lambda_post;

    // Try Cholesky-based log-det computation (spec §3.5.2)
    let variance_ratio = match (
        nalgebra::Cholesky::new(*post_cov),
        nalgebra::Cholesky::new(*prior_cov),
    ) {
        (Some(post_chol), Some(prior_chol)) => {
            // Compute log-det from Cholesky diagonals: log|A| = 2 * sum(log(L_ii))
            let post_log_det: f64 = (0..9)
                .map(|i| libm::log(post_chol.l()[(i, i)]))
                .sum::<f64>()
                * 2.0;
            let prior_log_det: f64 = (0..9)
                .map(|i| libm::log(prior_chol.l()[(i, i)]))
                .sum::<f64>()
                * 2.0;

            // Check for NaN/Inf (can happen with very small diagonal elements)
            if !post_log_det.is_finite() || !prior_log_det.is_finite() {
                // Fall back to trace ratio
                let trace_ratio = post_cov.trace() / prior_cov.trace();
                trace_ratio
            } else {
                // log-det ratio: ρ = log(|Λ_post| / |Λ₀^marginal|)
                let log_det_ratio = post_log_det - prior_log_det;
                // Convert to per-dimension ratio (geometric mean)
                let log_variance_ratio = log_det_ratio / 9.0;
                libm::exp(log_variance_ratio)
            }
        }
        _ => {
            // Cholesky failed - fall back to trace ratio (spec §3.5.2)
            let trace_ratio = post_cov.trace() / prior_cov.trace();
            trace_ratio
        }
    };

    // Trigger when variance ratio > threshold (default 0.5)
    if variance_ratio > config.max_variance_ratio {
        return Some(InconclusiveReason::DataTooNoisy {
            message: alloc::format!(
                "Posterior variance is {:.0}% of prior; data not informative",
                variance_ratio * 100.0
            ),
            guidance: String::from("Try: cycle counter, reduce system load, increase batch size"),
            variance_ratio,
        });
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
            leak_prob,
            1.0, // projection_mismatch_q
            1000,
        )
    }

    fn make_prior_cov_9d() -> Matrix9 {
        Matrix9::identity() * 100.0 // Prior variance = 100 per dimension
    }

    fn make_inputs<'a>(
        posterior: &'a Posterior,
        prior_cov_9d: &'a Matrix9,
    ) -> QualityGateCheckInputs<'a> {
        QualityGateCheckInputs {
            posterior,
            prior_cov_9d,
            prior_cov_marginal: prior_cov_9d, // v5.4: use same as prior_cov_9d for tests
            prior_cov_slab: None,       // v5.2: no mixture prior in tests
            narrow_weight_post: None,   // v5.2: no mixture prior in tests
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
            lambda_mixing_ok: None,     // v5.4: no Gibbs sampler in tests
        }
    }

    #[test]
    fn test_variance_ratio_gate_passes() {
        let posterior = make_posterior(0.5, 10.0); // variance = 10, ratio = 10/100 = 0.1 < 0.5
        let prior_cov_9d = make_prior_cov_9d();
        let inputs = make_inputs(&posterior, &prior_cov_9d);
        let config = QualityGateConfig::default();

        let result = check_variance_ratio(&inputs, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_variance_ratio_gate_fails() {
        let posterior = make_posterior(0.5, 80.0); // variance = 80, ratio = 80/100 = 0.8 > 0.5
        let prior_cov_9d = make_prior_cov_9d();
        let inputs = make_inputs(&posterior, &prior_cov_9d);
        let config = QualityGateConfig::default();

        let result = check_variance_ratio(&inputs, &config);
        assert!(matches!(
            result,
            Some(InconclusiveReason::DataTooNoisy { .. })
        ));
    }

    #[test]
    fn test_learning_rate_gate_passes() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_9d = make_prior_cov_9d();
        let mut inputs = make_inputs(&posterior, &prior_cov_9d);
        inputs.recent_kl_sum = Some(0.05); // Sum > 0.001
        let config = QualityGateConfig::default();

        let result = check_learning_rate(&inputs, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_learning_rate_gate_fails() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_9d = make_prior_cov_9d();
        let mut inputs = make_inputs(&posterior, &prior_cov_9d);
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
        let prior_cov_9d = make_prior_cov_9d();
        let mut inputs = make_inputs(&posterior, &prior_cov_9d);
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
        let prior_cov_9d = make_prior_cov_9d();
        let mut inputs = make_inputs(&posterior, &prior_cov_9d);
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
        let prior_cov_9d = make_prior_cov_9d();
        let inputs = make_inputs(&posterior, &prior_cov_9d);
        // No snapshots provided
        let config = QualityGateConfig::default();

        let result = check_condition_drift(&inputs, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_condition_drift_gate_no_drift() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_9d = make_prior_cov_9d();

        let stats = StatsSnapshot {
            mean: 100.0,
            variance: 25.0,
            autocorr_lag1: 0.1,
            count: 5000,
        };
        let cal_snapshot = CalibrationSnapshot::new(stats, stats);
        let post_snapshot = CalibrationSnapshot::new(stats, stats);

        let mut inputs = make_inputs(&posterior, &prior_cov_9d);
        inputs.calibration_snapshot = Some(&cal_snapshot);
        inputs.current_stats_snapshot = Some(&post_snapshot);

        let config = QualityGateConfig::default();

        let result = check_condition_drift(&inputs, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_condition_drift_gate_detects_variance_change() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov_9d = make_prior_cov_9d();

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

        let mut inputs = make_inputs(&posterior, &prior_cov_9d);
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
        let prior_cov_9d = make_prior_cov_9d();
        let inputs = make_inputs(&posterior, &prior_cov_9d);
        let config = QualityGateConfig::default();

        let result = check_quality_gates(&inputs, &config);
        assert!(matches!(result, QualityGateResult::Continue));
    }
}
