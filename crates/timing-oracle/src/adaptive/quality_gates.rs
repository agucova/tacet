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

use std::time::Duration;

use crate::adaptive::{AdaptiveState, Calibration, Posterior};

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
}

impl std::fmt::Display for InconclusiveReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InconclusiveReason::DataTooNoisy { message, .. } => write!(f, "{}", message),
            InconclusiveReason::NotLearning { message, .. } => write!(f, "{}", message),
            InconclusiveReason::WouldTakeTooLong {
                estimated_time_secs,
                samples_needed,
                ..
            } => {
                write!(
                    f,
                    "Would take {:.1}s ({} samples) to reach decision",
                    estimated_time_secs, samples_needed
                )
            }
            InconclusiveReason::TimeBudgetExceeded {
                current_probability,
                samples_collected,
                elapsed_secs,
            } => {
                write!(
                    f,
                    "Time budget exceeded after {:.1}s with {} samples (P={:.1}%)",
                    elapsed_secs,
                    samples_collected,
                    current_probability * 100.0
                )
            }
            InconclusiveReason::SampleBudgetExceeded {
                current_probability,
                samples_collected,
            } => {
                write!(
                    f,
                    "Sample budget exceeded at {} samples (P={:.1}%)",
                    samples_collected,
                    current_probability * 100.0
                )
            }
        }
    }
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

    /// Time budget for adaptive sampling.
    pub time_budget: Duration,

    /// Maximum samples per class.
    pub max_samples: usize,

    /// Pass threshold for leak probability.
    pub pass_threshold: f64,

    /// Fail threshold for leak probability.
    pub fail_threshold: f64,
}

impl Default for QualityGateConfig {
    fn default() -> Self {
        Self {
            max_variance_ratio: 0.5,
            min_kl_sum: 0.001,
            max_time_multiplier: 10.0,
            time_budget: Duration::from_secs(30),
            max_samples: 1_000_000,
            pass_threshold: 0.05,
            fail_threshold: 0.95,
        }
    }
}

/// Check all quality gates and return result.
///
/// Gates are checked in priority order. Returns `Continue` if all pass,
/// or `Stop` with the reason if any gate triggers.
///
/// # Arguments
///
/// * `state` - Current adaptive sampling state
/// * `posterior` - Current posterior distribution
/// * `calibration` - Calibration results
/// * `config` - Quality gate configuration
pub fn check_quality_gates(
    state: &AdaptiveState,
    posterior: &Posterior,
    calibration: &Calibration,
    config: &QualityGateConfig,
) -> QualityGateResult {
    // Gate 1: Posterior too close to prior (data not informative)
    if let Some(reason) = check_variance_ratio(posterior, calibration, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 2: Learning rate collapsed
    if let Some(reason) = check_learning_rate(state, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 3: Would take too long
    if let Some(reason) = check_extrapolated_time(state, posterior, calibration, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 4: Time budget exceeded
    if let Some(reason) = check_time_budget(state, posterior, config) {
        return QualityGateResult::Stop(reason);
    }

    // Gate 5: Sample budget exceeded
    if let Some(reason) = check_sample_budget(state, posterior, config) {
        return QualityGateResult::Stop(reason);
    }

    QualityGateResult::Continue
}

/// Gate 1: Check if posterior variance is too close to prior variance.
///
/// If det(posterior_cov) / det(prior_cov) > max_variance_ratio, the data
/// isn't reducing our uncertainty enough to be useful.
fn check_variance_ratio(
    posterior: &Posterior,
    calibration: &Calibration,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    let prior_det = calibration.prior_cov.determinant();
    let post_det = posterior.beta_cov.determinant();

    if prior_det <= 0.0 || post_det <= 0.0 {
        // Can't compute ratio with non-positive determinants
        return None;
    }

    let variance_ratio = post_det / prior_det;

    if variance_ratio > config.max_variance_ratio {
        return Some(InconclusiveReason::DataTooNoisy {
            message: format!(
                "Posterior variance is {:.0}% of prior; data not informative",
                variance_ratio * 100.0
            ),
            guidance: "Try: cycle counter, reduce system load, increase batch size".to_string(),
            variance_ratio,
        });
    }

    None
}

/// Gate 2: Check if learning has stalled (posterior stopped updating).
fn check_learning_rate(
    state: &AdaptiveState,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    if !state.has_kl_history() {
        return None; // Need at least 5 batches of history
    }

    let recent_kl_sum = state.recent_kl_sum();

    if recent_kl_sum < config.min_kl_sum {
        return Some(InconclusiveReason::NotLearning {
            message: "Posterior stopped updating despite new data".to_string(),
            guidance: "Measurement may have systematic issues or effect is very close to boundary"
                .to_string(),
            recent_kl_sum,
        });
    }

    None
}

/// Gate 3: Check if reaching a decision would take too long.
fn check_extrapolated_time(
    state: &AdaptiveState,
    posterior: &Posterior,
    calibration: &Calibration,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    // Need at least some samples to extrapolate
    if state.n_total() < 100 {
        return None;
    }

    let samples_needed = extrapolate_samples_to_decision(state, posterior, config);

    if samples_needed == usize::MAX {
        // Can't extrapolate
        return None;
    }

    let additional_samples = samples_needed.saturating_sub(state.n_total());
    let time_needed_secs = additional_samples as f64 / calibration.samples_per_second;
    let budget_secs = config.time_budget.as_secs_f64();

    if time_needed_secs > budget_secs * config.max_time_multiplier {
        return Some(InconclusiveReason::WouldTakeTooLong {
            estimated_time_secs: time_needed_secs,
            samples_needed,
            guidance: format!(
                "Effect may be very close to threshold; consider adjusting theta (current: {:.1}ns)",
                calibration.theta_ns
            ),
        });
    }

    None
}

/// Gate 4: Check if time budget is exceeded.
fn check_time_budget(
    state: &AdaptiveState,
    posterior: &Posterior,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    let elapsed = state.elapsed();

    if elapsed > config.time_budget {
        return Some(InconclusiveReason::TimeBudgetExceeded {
            current_probability: posterior.leak_probability,
            samples_collected: state.n_total(),
            elapsed_secs: elapsed.as_secs_f64(),
        });
    }

    None
}

/// Gate 5: Check if sample budget is exceeded.
fn check_sample_budget(
    state: &AdaptiveState,
    posterior: &Posterior,
    config: &QualityGateConfig,
) -> Option<InconclusiveReason> {
    if state.n_total() >= config.max_samples {
        return Some(InconclusiveReason::SampleBudgetExceeded {
            current_probability: posterior.leak_probability,
            samples_collected: state.n_total(),
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
    state: &AdaptiveState,
    posterior: &Posterior,
    config: &QualityGateConfig,
) -> usize {
    let p = posterior.leak_probability;

    // Distance to nearest threshold
    let margin = f64::min(
        (p - config.pass_threshold).abs(),
        (config.fail_threshold - p).abs(),
    );

    if margin < 1e-9 {
        return usize::MAX; // Already at threshold
    }

    // Posterior std (use trace as proxy for overall uncertainty)
    let current_std = posterior.beta_cov.trace().sqrt();

    if current_std < 1e-9 {
        return state.n_total(); // Already very certain
    }

    // Std scales as 1/sqrt(n), so to reduce std by factor k we need k^2 more samples
    // We need std to be comparable to margin for a clear decision
    let std_reduction_needed = current_std / margin;

    if std_reduction_needed <= 1.0 {
        // Current uncertainty is already small enough
        return state.n_total();
    }

    let sample_multiplier = std_reduction_needed.powi(2);

    // Cap at 100x current to avoid overflow
    let multiplier = sample_multiplier.min(100.0);

    ((state.n_total() as f64) * multiplier).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Matrix2, Vector2};

    fn make_posterior(leak_prob: f64, variance: f64) -> Posterior {
        Posterior::new(
            Vector2::new(5.0, 3.0),
            Matrix2::new(variance, 0.0, 0.0, variance),
            leak_prob,
            1000,
        )
    }

    fn make_calibration() -> Calibration {
        Calibration {
            sigma_rate: crate::types::Matrix9::identity() * 1000.0,
            block_length: 10,
            prior_cov: Matrix2::new(100.0, 0.0, 0.0, 100.0), // Prior variance = 100
            timer_resolution_ns: 1.0,
            samples_per_second: 100_000.0,
            discrete_mode: false,
            theta_ns: 100.0,
            calibration_samples: 5000,
            mde_shift_ns: 5.0,
            mde_tail_ns: 10.0,
            preflight_result: crate::preflight::PreflightResult::new(),
        }
    }

    #[test]
    fn test_variance_ratio_gate_passes() {
        let posterior = make_posterior(0.5, 10.0); // variance = 10, ratio = 10/100 = 0.1 < 0.5
        let calibration = make_calibration();
        let config = QualityGateConfig::default();

        let result = check_variance_ratio(&posterior, &calibration, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_variance_ratio_gate_fails() {
        let posterior = make_posterior(0.5, 80.0); // variance = 80, ratio = 80/100 = 0.8 > 0.5
        let calibration = make_calibration();
        let config = QualityGateConfig::default();

        let result = check_variance_ratio(&posterior, &calibration, &config);
        assert!(matches!(
            result,
            Some(InconclusiveReason::DataTooNoisy { .. })
        ));
    }

    #[test]
    fn test_learning_rate_gate_passes() {
        let mut state = AdaptiveState::new();
        // Add KL divergences that sum to more than threshold
        for _ in 0..5 {
            state.update_kl(0.01); // Sum = 0.05 > 0.001
        }
        let config = QualityGateConfig::default();

        let result = check_learning_rate(&state, &config);
        assert!(result.is_none());
    }

    #[test]
    fn test_learning_rate_gate_fails() {
        let mut state = AdaptiveState::new();
        // Add KL divergences that sum to less than threshold
        for _ in 0..5 {
            state.update_kl(0.0001); // Sum = 0.0005 < 0.001
        }
        let config = QualityGateConfig::default();

        let result = check_learning_rate(&state, &config);
        assert!(matches!(
            result,
            Some(InconclusiveReason::NotLearning { .. })
        ));
    }

    #[test]
    fn test_time_budget_gate() {
        let mut state = AdaptiveState::new();
        state.add_batch(vec![100; 1000], vec![101; 1000]);

        let posterior = make_posterior(0.5, 10.0);
        let mut config = QualityGateConfig::default();
        config.time_budget = Duration::from_nanos(1); // Immediate timeout

        // Wait a tiny bit to ensure we exceed budget
        std::thread::sleep(Duration::from_micros(10));

        let result = check_time_budget(&state, &posterior, &config);
        assert!(matches!(
            result,
            Some(InconclusiveReason::TimeBudgetExceeded { .. })
        ));
    }

    #[test]
    fn test_sample_budget_gate() {
        let mut state = AdaptiveState::new();
        state.add_batch(vec![100; 1000], vec![101; 1000]);

        let posterior = make_posterior(0.5, 10.0);
        let mut config = QualityGateConfig::default();
        config.max_samples = 500; // Below what we have

        let result = check_sample_budget(&state, &posterior, &config);
        assert!(matches!(
            result,
            Some(InconclusiveReason::SampleBudgetExceeded { .. })
        ));
    }

    #[test]
    fn test_extrapolate_samples_at_threshold() {
        let state = AdaptiveState::with_capacity(1000);
        let mut state = state;
        state.add_batch(vec![100; 1000], vec![101; 1000]);

        let posterior = make_posterior(0.05, 10.0); // Right at pass threshold
        let config = QualityGateConfig::default();

        let needed = extrapolate_samples_to_decision(&state, &posterior, &config);
        // At threshold, can't extrapolate meaningfully
        assert_eq!(needed, usize::MAX);
    }

    #[test]
    fn test_extrapolate_samples_away_from_threshold() {
        let mut state = AdaptiveState::new();
        state.add_batch(vec![100; 1000], vec![101; 1000]);

        let posterior = make_posterior(0.5, 10.0); // In the middle
        let config = QualityGateConfig::default();

        let needed = extrapolate_samples_to_decision(&state, &posterior, &config);
        // Should need more samples to reach threshold
        assert!(needed > 1000);
    }

    #[test]
    fn test_inconclusive_reason_display() {
        let reason = InconclusiveReason::TimeBudgetExceeded {
            current_probability: 0.5,
            samples_collected: 10000,
            elapsed_secs: 30.0,
        };
        let display = format!("{}", reason);
        assert!(display.contains("30.0"));
        assert!(display.contains("10000"));
        assert!(display.contains("50.0%"));
    }
}
