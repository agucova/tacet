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
        /// KL divergence from prior to posterior (nats). Low values indicate uninformative data.
        kl_divergence: f64,
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
    /// Minimum KL divergence (nats) from prior to posterior for conclusive verdict.
    /// Default: 0.7 nats (spec §3.5.2).
    pub kl_min: f64,

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
            kl_min: 0.7,
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
    // Gate 1: Skipped - now using 1D variance gate (check_gate1_1d) directly in loop_runner
    // The 1D gate requires a Posterior1D which is constructed from W₁ inference

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
/// Uses √n_blocks scaling consistent with spec §3.3.3 and floor calibration.
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

    // Spec §3.3.3: theta_floor(n) = max(theta_tick, c_floor / sqrt(n_blocks(n)))
    let n_blocks = max_samples
        .checked_div(block_length)
        .unwrap_or(max_samples)
        .max(1);
    let theta_floor_at_max = libm::fmax(c_floor / libm::sqrt(n_blocks as f64), theta_tick);

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
/// The epsilon tolerance is: ε = max(θ_tick, 1e-6 × θ_user)
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

// Old 9D KL divergence functions removed - now using 1D variance gates

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

    // Posterior std: use 1D posterior variance
    let current_std = libm::sqrt(inputs.posterior.var_post);

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
/// See spec §3.5.4, Gate 4 (Condition Drift).
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

// ============================================================================
// Phase 3: 1D Variance Quality Gates (W₁ path)
// ============================================================================

/// Temporary 1D posterior structure for W₁ inference.
///
/// This is used during the Phase 3 migration to 1D variance inference.
/// Once the migration is complete, this will replace the full 9D `Posterior` struct.
///
/// # Fields
///
/// * `w1_post` - Posterior mean of W₁ (maximum quantile difference in ns)
/// * `var_post` - Posterior variance of W₁
/// * `leak_probability` - P(W₁ > θ | data)
/// * `n` - Total samples per class used for inference
/// * `theta` - Effect threshold in nanoseconds
#[derive(Debug, Clone)]
pub struct Posterior1D {
    /// Posterior mean of W₁ (maximum quantile difference in ns).
    pub w1_post: f64,

    /// Posterior variance of W₁ (ns²).
    pub var_post: f64,

    /// Posterior leak probability: P(W₁ > θ | data).
    pub leak_probability: f64,

    /// Total samples per class used for this posterior.
    pub n: usize,

    /// Effect threshold θ in nanoseconds.
    pub theta: f64,
}

impl Posterior1D {
    /// Create a new 1D posterior.
    pub fn new(w1_post: f64, var_post: f64, leak_probability: f64, n: usize, theta: f64) -> Self {
        Self {
            w1_post,
            var_post,
            leak_probability,
            n,
            theta,
        }
    }
}

/// Check Gate 1: Data too noisy (1D version).
///
/// Triggers if posterior variance is too close to prior variance,
/// indicating the data isn't informative enough to learn about W₁.
///
/// This is the 1D analog of the KL divergence check for the 9D case.
/// For a scalar normal distribution:
///
/// KL(N(μ_post, σ²_post) || N(0, σ²_prior)) = 0.5 * (σ²_post/σ²_prior + μ²_post/σ²_prior - 1 + ln(σ²_prior/σ²_post))
///
/// Computes the KL divergence between Gaussian surrogates of prior and
/// posterior (spec §3.5.2). The prior surrogate is N(0, V₀) and the
/// posterior surrogate is N(μ_post, V_post), giving:
///
///   KL = ½ (V_post/V₀ + μ_post²/V₀ - 1 + ln(V₀/V_post))
///
/// Low KL means the posterior barely moved from the prior.
///
/// # Arguments
///
/// * `posterior` - 1D posterior state for W₁
/// * `prior_var` - Marginal prior variance V₀ = 2σ_t² (spec §3.4.3)
/// * `config` - Quality gate configuration (provides kl_min threshold)
///
/// # Returns
///
/// `Some(InconclusiveReason)` if KL < kl_min, `None` otherwise.
///
/// # Spec Reference
///
/// See spec §3.5.2 (Gate 1: Insufficient Information Gain).
pub fn check_gate1_1d(
    posterior: &Posterior1D,
    prior_var: f64,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    // KL(posterior || prior) for Gaussian surrogates (spec §3.5.2)
    // KL = ½ (V_post/V₀ + μ_post²/V₀ - 1 + ln(V₀/V_post))
    let v_ratio = posterior.var_post / prior_var;
    let mu_sq_term = (posterior.w1_post * posterior.w1_post) / prior_var;

    // Guard against degenerate cases
    let kl = if v_ratio <= 0.0 || !v_ratio.is_finite() {
        f64::INFINITY // Degenerate posterior is maximally different from prior
    } else {
        0.5 * (v_ratio + mu_sq_term - 1.0 + libm::log(1.0 / v_ratio))
    };

    if kl < config.kl_min {
        Some(InconclusiveReason::DataTooNoisy {
            message: alloc::format!(
                "KL divergence {:.2} nats < {:.1} nats threshold; data not informative",
                kl,
                config.kl_min
            ),
            guidance: String::from(
                "Try: more samples, higher-resolution timer, reduce system load",
            ),
            kl_divergence: kl,
        })
    } else {
        None
    }
}

/// Check if measurement variance exceeds threshold feasibility.
///
/// If sqrt(Var_post) > K × θ_eff, we cannot reliably detect effects at threshold θ_eff.
/// This is a conservative check to ensure the measurement noise is small enough
/// relative to the effect threshold.
///
/// The factor K = 3.0 is conservative: it requires that the posterior standard
/// deviation be less than 3× the threshold. This ensures we have enough precision
/// to distinguish effects near the threshold from noise.
///
/// # Arguments
///
/// * `var_post` - Posterior variance of W₁ (ns²)
/// * `theta_eff` - Effective threshold θ_eff = max(θ_user, θ_floor) in nanoseconds
///
/// # Returns
///
/// `Some(InconclusiveReason)` if the variance floor is exceeded, `None` otherwise.
///
/// # Example
///
/// If θ_eff = 10ns and Var_post = 900ns², then SD = 30ns > 3×10ns = 30ns.
/// This triggers the gate because the measurement uncertainty is too large
/// relative to the threshold we're trying to detect.
pub fn check_variance_floor_exceeded(var_post: f64, theta_eff: f64) -> Option<InconclusiveReason> {
    const K: f64 = 3.0; // Conservative factor: require SD < 3× threshold
    let sd_estimate = libm::sqrt(var_post);

    if sd_estimate > K * theta_eff {
        Some(InconclusiveReason::DataTooNoisy {
            message: alloc::format!(
                "Estimate SD ({:.1}ns) exceeds {}× threshold ({:.1}ns)",
                sd_estimate,
                K,
                theta_eff
            ),
            guidance: String::from("Measurement noise too high for this threshold"),
            kl_divergence: 0.0, // Not applicable for variance floor check
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Old 9D tests removed - quality gates now work with 1D variance inference

    // ========================================================================
    // Phase 3: 1D Variance Quality Gate Tests
    // ========================================================================

    #[test]
    fn test_gate1_1d_passes_with_informative_data() {
        // Prior variance = 100 ns²
        let prior_var = 100.0;

        // Posterior mean shifted to 5ns with small variance → high KL
        // KL = 0.5 * (10/100 + 25/100 - 1 + ln(100/10))
        //    = 0.5 * (0.1 + 0.25 - 1 + 2.303) = 0.5 * 1.653 = 0.826 > 0.7
        let posterior = Posterior1D::new(
            5.0,   // w1_post
            10.0,  // var_post
            0.3,   // leak_probability
            5000,  // n
            100.0, // theta
        );

        let config = QualityGateConfig::default();

        let result = check_gate1_1d(&posterior, prior_var, &config);
        assert!(
            result.is_none(),
            "Informative data (KL ≈ 0.83) should pass Gate 1"
        );
    }

    #[test]
    fn test_gate1_1d_fails_with_uninformative_data() {
        // Prior variance = 100 ns²
        let prior_var = 100.0;
        // Posterior barely moved: mean ≈ 0, variance ≈ prior
        // KL = 0.5 * (95/100 + 0/100 - 1 + ln(100/95))
        //    = 0.5 * (0.95 + 0 - 1 + 0.0513) = 0.5 * 0.0013 ≈ 0.0007 < 0.7
        let posterior = Posterior1D::new(
            0.0,   // w1_post (no shift from prior mean of 0)
            95.0,  // var_post (barely moved from prior_var=100)
            0.5,   // leak_probability
            5000,  // n
            100.0, // theta
        );

        let config = QualityGateConfig::default();

        let result = check_gate1_1d(&posterior, prior_var, &config);
        assert!(
            matches!(result, Some(InconclusiveReason::DataTooNoisy { .. })),
            "Uninformative data (KL ≈ 0.0007) should trigger Gate 1"
        );

        if let Some(InconclusiveReason::DataTooNoisy { kl_divergence, .. }) = result {
            assert!(
                kl_divergence < 0.7,
                "KL divergence should be below threshold, got {:.4}",
                kl_divergence
            );
        }
    }

    #[test]
    fn test_gate1_1d_large_mean_shift_passes() {
        // Prior variance = 0.5 ns² (calibrated for θ=0.4ns)
        let prior_var = 0.5;

        // Posterior mean = 100ns (20σ effect), variance ≈ prior (λ adapted)
        // KL = 0.5 * (0.55/0.5 + 100²/0.5 - 1 + ln(0.5/0.55))
        //    = 0.5 * (1.1 + 20000 - 1 + (-0.095)) ≈ 10000 >> 0.7
        let posterior = Posterior1D::new(
            100.0, // w1_post (large shift)
            0.55,  // var_post (110% of prior — would fail old variance ratio check)
            1.0,   // leak_probability
            5000,  // n
            0.4,   // theta
        );

        let config = QualityGateConfig::default();

        let result = check_gate1_1d(&posterior, prior_var, &config);
        assert!(
            result.is_none(),
            "Large mean shift (KL ≈ 10000) should pass Gate 1 even with high variance ratio"
        );
    }

    #[test]
    fn test_variance_floor_not_exceeded() {
        // var_post = 25 ns² → SD = 5 ns
        // theta_eff = 10 ns
        // K * theta_eff = 3 * 10 = 30 ns
        // 5 ns < 30 ns → should pass
        let var_post = 25.0;
        let theta_eff = 10.0;

        let result = check_variance_floor_exceeded(var_post, theta_eff);
        assert!(
            result.is_none(),
            "SD (5ns) < 3× threshold (30ns) should pass"
        );
    }

    #[test]
    fn test_variance_floor_exceeded() {
        // var_post = 900 ns² → SD = 30 ns
        // theta_eff = 10 ns
        // K * theta_eff = 3 * 10 = 30 ns
        // 30 ns == 30 ns → boundary, but should fail (SD > K*theta, not >=)
        //
        // Let's use a clear case: SD = 31 ns > 30 ns
        let var_post = 961.0; // SD = 31 ns
        let theta_eff = 10.0;

        let result = check_variance_floor_exceeded(var_post, theta_eff);
        assert!(
            matches!(result, Some(InconclusiveReason::DataTooNoisy { .. })),
            "SD (31ns) > 3× threshold (30ns) should trigger gate"
        );

        if let Some(InconclusiveReason::DataTooNoisy { kl_divergence, .. }) = result {
            assert!(
                kl_divergence == 0.0,
                "Variance floor check should set kl_divergence to 0.0 (N/A)"
            );
        }
    }

    #[test]
    fn test_variance_floor_boundary() {
        // var_post = 900 ns² → SD = 30 ns
        // theta_eff = 10 ns
        // K * theta_eff = 3 * 10 = 30 ns
        // 30 ns == 30 ns → should pass (> not >=)
        let var_post = 900.0;
        let theta_eff = 10.0;

        let result = check_variance_floor_exceeded(var_post, theta_eff);
        assert!(
            result.is_none(),
            "Boundary case (SD = 3× threshold) should pass"
        );
    }

    #[test]
    fn test_variance_floor_low_threshold() {
        // Test with very low threshold (SharedHardware scenario)
        // var_post = 4 ns² → SD = 2 ns
        // theta_eff = 0.4 ns (SharedHardware threshold)
        // K * theta_eff = 3 * 0.4 = 1.2 ns
        // 2 ns > 1.2 ns → should trigger
        let var_post = 4.0;
        let theta_eff = 0.4;

        let result = check_variance_floor_exceeded(var_post, theta_eff);
        assert!(
            matches!(result, Some(InconclusiveReason::DataTooNoisy { .. })),
            "High variance relative to low threshold should trigger gate"
        );
    }

    #[test]
    fn test_posterior1d_construction() {
        let posterior = Posterior1D::new(12.5, 25.0, 0.85, 10000, 10.0);

        assert!((posterior.w1_post - 12.5).abs() < 1e-9);
        assert!((posterior.var_post - 25.0).abs() < 1e-9);
        assert!((posterior.leak_probability - 0.85).abs() < 1e-9);
        assert_eq!(posterior.n, 10000);
        assert!((posterior.theta - 10.0).abs() < 1e-9);
    }
}
