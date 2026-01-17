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
use crate::types::Matrix2;

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

    /// Requested threshold is unachievable given maximum sample budget (Gate 7).
    ///
    /// The measurement floor at max_samples exceeds the user's requested threshold.
    /// Even with the maximum allowed samples, we cannot resolve the requested precision.
    ThresholdUnachievable {
        /// User's requested threshold in nanoseconds.
        theta_user: f64,
        /// Best achievable threshold at max_samples.
        best_achievable: f64,
        /// Human-readable message.
        message: String,
        /// Suggested remediation.
        guidance: String,
    },

    /// Model mismatch detected - the 2D shift+tail model doesn't fit (Gate 8, verdict-blocking).
    ///
    /// The residual Q statistic exceeds the bootstrap-calibrated threshold,
    /// indicating the observed quantile differences don't match the expected
    /// shift + tail pattern. This is a verdict-blocking condition.
    ModelMismatch {
        /// The Q statistic (residual^T Σ^{-1} residual).
        q_statistic: f64,
        /// The threshold from bootstrap calibration.
        q_threshold: f64,
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

    /// Prior covariance matrix (from calibration).
    pub prior_cov: &'a Matrix2,

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

    // === v4.1 additions for new quality gates ===

    /// Floor-rate constant (c_floor) from calibration.
    /// Used to compute theta_floor(n) = c_floor / sqrt(n).
    pub c_floor: f64,

    /// Timer tick floor (theta_tick) from calibration.
    /// The floor below which timer quantization dominates.
    pub theta_tick: f64,

    /// Model fit Q statistic (residual^T Σ^{-1} residual).
    /// Pass `None` if not yet computed.
    pub q_fit: Option<f64>,

    /// Model mismatch threshold from bootstrap calibration.
    /// Q > q_thresh indicates model mismatch.
    pub q_thresh: f64,
}

/// Check all quality gates and return result.
///
/// Gates are checked in priority order. Returns `Continue` if all pass,
/// or `Stop` with the reason if any gate triggers.
///
/// **Gate order (spec Section 2.6):**
/// 1. Posterior too close to prior (data not informative)
/// 2. Learning rate collapsed
/// 3. Would take too long
/// 4. Threshold unachievable (v4.1)
/// 5. Time budget exceeded
/// 6. Sample budget exceeded
/// 7. Condition drift detected
/// 8. Model mismatch (v4.1, verdict-blocking)
///
/// Gates 1-7 are checked before any Pass/Fail decision.
/// Gate 8 (ModelMismatch) is verdict-blocking and prevents confident verdicts.
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

    // Gate 4 (v4.1): Threshold unachievable
    if let Some(reason) = check_threshold_unachievable(inputs, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 5: Time budget exceeded
    if let Some(reason) = check_time_budget(inputs, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 6: Sample budget exceeded
    if let Some(reason) = check_sample_budget(inputs, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 7: Condition drift detected
    if let Some(reason) = check_condition_drift(inputs, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 8 (v4.1): Model mismatch (verdict-blocking)
    // This gate prevents Pass/Fail verdicts when the model doesn't fit
    if let Some(reason) = check_model_mismatch(inputs, config) {
        return QualityGateResult::Stop(reason);
    }

    QualityGateResult::Continue
}

/// Gate 4 (v4.1): Check if the requested threshold is unachievable.
///
/// If theta_floor at max_samples exceeds theta_user, we can never resolve
/// the requested precision even with the maximum sample budget.
fn check_threshold_unachievable(
    inputs: &QualityGateCheckInputs,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    // Only check if user specified a threshold (not research mode)
    if inputs.theta_ns <= 0.0 {
        return None;
    }

    // Compute theta_floor at max_samples
    let theta_floor_at_max = libm::fmax(
        inputs.c_floor / libm::sqrt(config.max_samples as f64),
        inputs.theta_tick,
    );

    // If floor at max_samples exceeds user's threshold, it's unachievable
    if theta_floor_at_max > inputs.theta_ns {
        return Some(InconclusiveReason::ThresholdUnachievable {
            theta_user: inputs.theta_ns,
            best_achievable: theta_floor_at_max,
            message: alloc::format!(
                "Requested θ = {:.1} ns, but measurement floor is {:.1} ns",
                inputs.theta_ns,
                theta_floor_at_max
            ),
            guidance: String::from(
                "Use a cycle counter (PmuTimer/LinuxPerfTimer) for better resolution, \
                increase max_samples, or use a higher threshold",
            ),
        });
    }

    None
}

/// Gate 8 (v4.1): Check if the model doesn't fit the data (verdict-blocking).
///
/// If Q > q_thresh, the 2D (shift + tail) model doesn't explain the observed
/// quantile differences well. This is a verdict-blocking condition that
/// prevents confident Pass/Fail verdicts.
fn check_model_mismatch(
    inputs: &QualityGateCheckInputs,
    _config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    // Need Q statistic to check
    let q_fit = inputs.q_fit?;

    if q_fit > inputs.q_thresh {
        return Some(InconclusiveReason::ModelMismatch {
            q_statistic: q_fit,
            q_threshold: inputs.q_thresh,
            message: alloc::format!(
                "Model mismatch: Q = {:.2} > threshold {:.2}",
                q_fit,
                inputs.q_thresh
            ),
            guidance: String::from(
                "The observed timing pattern doesn't fit the shift+tail model. \
                This could indicate: non-standard timing patterns, measurement artifacts, \
                or genuine but unusual side-channel behavior. Interpret results with caution.",
            ),
        });
    }

    None
}

/// Gate 1: Check if posterior variance is too close to prior variance.
///
/// If det(posterior_cov) / det(prior_cov) > max_variance_ratio, the data
/// isn't reducing our uncertainty enough to be useful.
fn check_variance_ratio(
    inputs: &QualityGateCheckInputs,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    let prior_det = inputs.prior_cov.determinant();
    let post_det = inputs.posterior.beta_cov.determinant();

    if prior_det <= 0.0 || post_det <= 0.0 {
        // Can't compute ratio with non-positive determinants
        return None;
    }

    let variance_ratio = post_det / prior_det;

    if variance_ratio > config.max_variance_ratio {
        return Some(InconclusiveReason::DataTooNoisy {
            message: alloc::format!(
                "Posterior variance is {:.0}% of prior; data not informative",
                variance_ratio * 100.0
            ),
            guidance: String::from(
                "Try: cycle counter, reduce system load, increase batch size",
            ),
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

    // Posterior std (use trace as proxy for overall uncertainty)
    let current_std = libm::sqrt(inputs.posterior.beta_cov.trace());

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
    use crate::types::Vector2;

    fn make_posterior(leak_prob: f64, variance: f64) -> Posterior {
        Posterior::new(
            Vector2::new(5.0, 3.0),
            Matrix2::new(variance, 0.0, 0.0, variance),
            leak_prob,
            1000,
            5.0, // model_fit_q
        )
    }

    fn make_prior_cov() -> Matrix2 {
        Matrix2::new(100.0, 0.0, 0.0, 100.0) // Prior variance = 100
    }

    fn make_inputs<'a>(
        posterior: &'a Posterior,
        prior_cov: &'a Matrix2,
    ) -> QualityGateCheckInputs<'a> {
        QualityGateCheckInputs {
            posterior,
            prior_cov,
            theta_ns: 100.0,
            n_total: 5000,
            elapsed_secs: 5.0,
            recent_kl_sum: Some(0.05),
            samples_per_second: 100_000.0,
            calibration_snapshot: None,
            current_stats_snapshot: None,
            // v4.1 fields
            c_floor: 3535.5, // ~50 * sqrt(5000)
            theta_tick: 1.0,
            q_fit: None,     // No model fit check in basic tests
            q_thresh: 18.48, // chi-squared(7, 0.99) fallback
        }
    }

    #[test]
    fn test_variance_ratio_gate_passes() {
        let posterior = make_posterior(0.5, 10.0); // variance = 10, ratio = 10/100 = 0.1 < 0.5
        let prior_cov = make_prior_cov();
        let inputs = make_inputs(&posterior, &prior_cov);
        let config = QualityGateConfig::default();

        let result = check_variance_ratio(&inputs, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_variance_ratio_gate_fails() {
        let posterior = make_posterior(0.5, 80.0); // variance = 80, ratio = 80/100 = 0.8 > 0.5
        let prior_cov = make_prior_cov();
        let inputs = make_inputs(&posterior, &prior_cov);
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
        let prior_cov = make_prior_cov();
        let mut inputs = make_inputs(&posterior, &prior_cov);
        inputs.recent_kl_sum = Some(0.05); // Sum > 0.001
        let config = QualityGateConfig::default();

        let result = check_learning_rate(&inputs, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_learning_rate_gate_fails() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov = make_prior_cov();
        let mut inputs = make_inputs(&posterior, &prior_cov);
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
        let prior_cov = make_prior_cov();
        let mut inputs = make_inputs(&posterior, &prior_cov);
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
        let prior_cov = make_prior_cov();
        let mut inputs = make_inputs(&posterior, &prior_cov);
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
        let prior_cov = make_prior_cov();
        let inputs = make_inputs(&posterior, &prior_cov);
        // No snapshots provided
        let config = QualityGateConfig::default();

        let result = check_condition_drift(&inputs, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_condition_drift_gate_no_drift() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov = make_prior_cov();

        let stats = StatsSnapshot {
            mean: 100.0,
            variance: 25.0,
            autocorr_lag1: 0.1,
            count: 5000,
        };
        let cal_snapshot = CalibrationSnapshot::new(stats.clone(), stats.clone());
        let post_snapshot = CalibrationSnapshot::new(stats.clone(), stats.clone());

        let mut inputs = make_inputs(&posterior, &prior_cov);
        inputs.calibration_snapshot = Some(&cal_snapshot);
        inputs.current_stats_snapshot = Some(&post_snapshot);

        let config = QualityGateConfig::default();

        let result = check_condition_drift(&inputs, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_condition_drift_gate_detects_variance_change() {
        let posterior = make_posterior(0.5, 10.0);
        let prior_cov = make_prior_cov();

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
        let cal_snapshot = CalibrationSnapshot::new(cal_stats.clone(), cal_stats.clone());
        let post_snapshot = CalibrationSnapshot::new(post_stats.clone(), post_stats.clone());

        let mut inputs = make_inputs(&posterior, &prior_cov);
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
        let prior_cov = make_prior_cov();
        let inputs = make_inputs(&posterior, &prior_cov);
        let config = QualityGateConfig::default();

        let result = check_quality_gates(&inputs, &config);
        assert!(matches!(result, QualityGateResult::Continue));
    }
}
