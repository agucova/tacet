//! Adaptive sampling loop runner.
//!
//! This module implements the main adaptive sampling loop that collects
//! batches of measurements until a definitive decision is reached or
//! a quality gate triggers.
//!
//! The loop follows this pattern (spec Section 2.5):
//! 1. Collect a batch of samples
//! 2. Compute the posterior distribution
//! 3. Check if P(leak > theta) > fail_threshold -> Fail
//! 4. Check if P(leak > theta) < pass_threshold -> Pass
//! 5. Check quality gates -> Inconclusive if triggered
//! 6. Continue if undecided

use std::time::Duration;

use crate::adaptive::{
    AdaptiveState, Calibration, InconclusiveReason, Posterior, QualityGateCheckInputs,
    QualityGateConfig, QualityGateResult,
};
use crate::analysis::bayes::compute_bayes_factor;
use crate::constants::DEFAULT_SEED;
use crate::statistics::{compute_deciles_inplace, compute_midquantile_deciles};
use timing_oracle_core::adaptive::check_quality_gates;

/// Configuration for the adaptive sampling loop.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Number of samples per batch.
    pub batch_size: usize,

    /// Threshold below which we pass (no significant leak).
    pub pass_threshold: f64,

    /// Threshold above which we fail (leak detected).
    pub fail_threshold: f64,

    /// Time budget for adaptive sampling.
    pub time_budget: Duration,

    /// Maximum samples per class.
    pub max_samples: usize,

    /// Effect threshold (theta) in nanoseconds.
    pub theta_ns: f64,

    /// Random seed for Monte Carlo integration.
    pub seed: u64,

    /// Quality gate configuration.
    pub quality_gates: QualityGateConfig,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            pass_threshold: 0.05,
            fail_threshold: 0.95,
            time_budget: Duration::from_secs(30),
            max_samples: 1_000_000,
            theta_ns: 100.0,
            seed: DEFAULT_SEED,
            quality_gates: QualityGateConfig::default(),
        }
    }
}

impl AdaptiveConfig {
    /// Create a new config with the given theta threshold.
    pub fn with_theta(theta_ns: f64) -> Self {
        let mut config = Self {
            theta_ns,
            ..Self::default()
        };
        config.quality_gates.pass_threshold = config.pass_threshold;
        config.quality_gates.fail_threshold = config.fail_threshold;
        config
    }

    /// Builder method to set pass threshold.
    pub fn pass_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold;
        self.quality_gates.pass_threshold = threshold;
        self
    }

    /// Builder method to set fail threshold.
    pub fn fail_threshold(mut self, threshold: f64) -> Self {
        self.fail_threshold = threshold;
        self.quality_gates.fail_threshold = threshold;
        self
    }

    /// Builder method to set time budget.
    pub fn time_budget(mut self, budget: Duration) -> Self {
        self.time_budget = budget;
        self.quality_gates.time_budget_secs = budget.as_secs_f64();
        self
    }

    /// Builder method to set max samples.
    pub fn max_samples(mut self, max: usize) -> Self {
        self.max_samples = max;
        self.quality_gates.max_samples = max;
        self
    }
}

/// Outcome of the adaptive sampling loop.
#[derive(Debug, Clone)]
pub enum AdaptiveOutcome {
    /// Leak probability exceeded fail threshold - timing leak detected.
    LeakDetected {
        /// Final posterior distribution.
        posterior: Posterior,
        /// Number of samples collected per class.
        samples_per_class: usize,
        /// Time spent in adaptive loop.
        elapsed: Duration,
    },

    /// Leak probability dropped below pass threshold - no significant leak.
    NoLeakDetected {
        /// Final posterior distribution.
        posterior: Posterior,
        /// Number of samples collected per class.
        samples_per_class: usize,
        /// Time spent in adaptive loop.
        elapsed: Duration,
    },

    /// A quality gate triggered before reaching a decision.
    Inconclusive {
        /// Reason for stopping.
        reason: InconclusiveReason,
        /// Final posterior distribution (if available).
        posterior: Option<Posterior>,
        /// Number of samples collected per class.
        samples_per_class: usize,
        /// Time spent in adaptive loop.
        elapsed: Duration,
    },

    /// Quality gates passed but no decision reached yet.
    /// Caller should collect more samples and call again.
    Continue {
        /// Current posterior distribution.
        posterior: Posterior,
        /// Number of samples collected per class so far.
        samples_per_class: usize,
        /// Time spent so far.
        elapsed: Duration,
    },
}

impl AdaptiveOutcome {
    /// Get the final leak probability, if available.
    pub fn leak_probability(&self) -> Option<f64> {
        match self {
            AdaptiveOutcome::LeakDetected { posterior, .. } => Some(posterior.leak_probability),
            AdaptiveOutcome::NoLeakDetected { posterior, .. } => Some(posterior.leak_probability),
            AdaptiveOutcome::Continue { posterior, .. } => Some(posterior.leak_probability),
            AdaptiveOutcome::Inconclusive { posterior, .. } => {
                posterior.as_ref().map(|p| p.leak_probability)
            }
        }
    }

    /// Check if the outcome indicates a leak was detected.
    pub fn is_leak_detected(&self) -> bool {
        matches!(self, AdaptiveOutcome::LeakDetected { .. })
    }

    /// Check if the outcome is conclusive (either pass or fail).
    pub fn is_conclusive(&self) -> bool {
        matches!(
            self,
            AdaptiveOutcome::LeakDetected { .. } | AdaptiveOutcome::NoLeakDetected { .. }
        )
    }
}

/// Run the adaptive sampling loop until a decision is reached.
///
/// This function assumes calibration has already been performed and timing
/// samples are being collected externally. It manages the loop logic,
/// posterior computation, and decision-making.
///
/// # Arguments
///
/// * `calibration` - Results from the calibration phase
/// * `state` - Mutable state containing accumulated samples
/// * `ns_per_tick` - Conversion factor from native units to nanoseconds
/// * `config` - Adaptive loop configuration
///
/// # Returns
///
/// An `AdaptiveOutcome` indicating the result of the adaptive loop.
///
/// # Note
///
/// This function expects the caller to add batches to `state` before calling
/// or to use `run_adaptive_with_collector` which handles batch collection.
pub fn run_adaptive(
    calibration: &Calibration,
    state: &mut AdaptiveState,
    ns_per_tick: f64,
    config: &AdaptiveConfig,
) -> AdaptiveOutcome {
    // Compute posterior from current samples
    let posterior = match compute_posterior_from_state(state, calibration, ns_per_tick, config) {
        Some(p) => p,
        None => {
            return AdaptiveOutcome::Inconclusive {
                reason: InconclusiveReason::DataTooNoisy {
                    message: "Could not compute posterior from samples".to_string(),
                    guidance: "Check timer resolution and sample count".to_string(),
                    variance_ratio: 1.0,
                },
                posterior: None,
                samples_per_class: state.n_total(),
                elapsed: state.elapsed(),
            };
        }
    };

    // Track KL divergence
    let _kl = state.update_posterior(posterior.clone());

    // =========================================================================
    // CRITICAL: Check ALL quality gates BEFORE decision boundaries (spec §2.6)
    // =========================================================================
    // Quality gates are verdict-blocking: if any gate triggers, we cannot make
    // a confident Pass/Fail decision, even if the posterior would otherwise
    // cross the threshold.
    let current_stats = state.get_stats_snapshot();
    let gate_inputs = QualityGateCheckInputs {
        posterior: &posterior,
        prior_cov_9d: &calibration.prior_cov_9d,
        theta_ns: config.theta_ns,
        n_total: state.n_total(),
        elapsed_secs: state.elapsed().as_secs_f64(),
        recent_kl_sum: if state.has_kl_history() {
            Some(state.recent_kl_sum())
        } else {
            None
        },
        samples_per_second: calibration.samples_per_second,
        calibration_snapshot: Some(&calibration.calibration_snapshot),
        current_stats_snapshot: current_stats.as_ref(),
        c_floor: calibration.c_floor,
        theta_tick: calibration.theta_tick,
        projection_mismatch_q: if posterior.projection_mismatch_q.is_nan() {
            None
        } else {
            Some(posterior.projection_mismatch_q)
        },
        projection_mismatch_thresh: calibration.projection_mismatch_thresh,
    };

    match check_quality_gates(&gate_inputs, &config.quality_gates) {
        QualityGateResult::Stop(reason) => {
            // Quality gate triggered - cannot make confident verdict
            return AdaptiveOutcome::Inconclusive {
                reason,
                posterior: Some(posterior),
                samples_per_class: state.n_total(),
                elapsed: state.elapsed(),
            };
        }
        QualityGateResult::Continue => {
            // All gates passed - now check decision boundaries
        }
    }

    // =========================================================================
    // Decision boundaries (only reached if ALL quality gates passed)
    // =========================================================================
    if posterior.leak_probability > config.fail_threshold {
        return AdaptiveOutcome::LeakDetected {
            posterior,
            samples_per_class: state.n_total(),
            elapsed: state.elapsed(),
        };
    }

    if posterior.leak_probability < config.pass_threshold {
        return AdaptiveOutcome::NoLeakDetected {
            posterior,
            samples_per_class: state.n_total(),
            elapsed: state.elapsed(),
        };
    }

    // Not yet decisive - continue sampling
    AdaptiveOutcome::Continue {
        posterior,
        samples_per_class: state.n_total(),
        elapsed: state.elapsed(),
    }
}

/// Compute posterior distribution from current state.
///
/// Uses scaled covariance: Sigma_n = Sigma_rate / n
fn compute_posterior_from_state(
    state: &AdaptiveState,
    calibration: &Calibration,
    ns_per_tick: f64,
    config: &AdaptiveConfig,
) -> Option<Posterior> {
    let n = state.n_total();
    if n < 20 {
        return None; // Need minimum samples
    }

    // Convert samples to nanoseconds
    let baseline_ns = state.baseline_ns(ns_per_tick);
    let sample_ns = state.sample_ns(ns_per_tick);

    // Compute quantile differences: sample - baseline
    // Positive values mean sample is slower (timing leak detected)
    // Negative values mean sample is faster (no leak, or unusual)
    let observed_diff = if calibration.discrete_mode {
        let q_baseline = compute_midquantile_deciles(&baseline_ns);
        let q_sample = compute_midquantile_deciles(&sample_ns);
        q_sample - q_baseline
    } else {
        let mut baseline_sorted = baseline_ns;
        let mut sample_sorted = sample_ns;
        let q_baseline = compute_deciles_inplace(&mut baseline_sorted);
        let q_sample = compute_deciles_inplace(&mut sample_sorted);
        q_sample - q_baseline
    };

    // Scale covariance: Sigma_n = Sigma_rate / n
    let sigma_n = calibration.sigma_rate / (n as f64);

    // Run 9D Bayesian inference with prior covariance
    let bayes_result = compute_bayes_factor(
        &observed_diff,
        &sigma_n,
        &calibration.prior_cov_9d,
        config.theta_ns,
        Some(config.seed),
    );

    Some(Posterior::new(
        bayes_result.delta_post,
        bayes_result.lambda_post,
        bayes_result.beta_proj,
        bayes_result.beta_proj_cov,
        bayes_result.leak_probability,
        bayes_result.projection_mismatch_q,
        n,
    ))
}

/// Single-iteration adaptive step.
///
/// This is useful for external loop control where the caller manages
/// batch collection and wants fine-grained control over the loop.
///
/// # Arguments
///
/// * `calibration` - Results from calibration phase
/// * `state` - Current adaptive state with accumulated samples
/// * `ns_per_tick` - Conversion factor from native units to nanoseconds
/// * `config` - Adaptive loop configuration
///
/// # Returns
///
/// - `Ok(None)` - Continue collecting samples
/// - `Ok(Some(outcome))` - Decision reached or quality gate triggered
/// - `Err(reason)` - Error during computation
#[allow(dead_code)]
pub fn adaptive_step(
    calibration: &Calibration,
    state: &mut AdaptiveState,
    ns_per_tick: f64,
    config: &AdaptiveConfig,
) -> Result<Option<AdaptiveOutcome>, InconclusiveReason> {
    // Compute posterior
    let posterior = match compute_posterior_from_state(state, calibration, ns_per_tick, config) {
        Some(p) => p,
        None => {
            // Not enough samples yet
            return Ok(None);
        }
    };

    // Track KL divergence
    let _kl = state.update_posterior(posterior.clone());

    // =========================================================================
    // CRITICAL: Check ALL quality gates BEFORE decision boundaries (spec §2.6)
    // =========================================================================
    let current_stats = state.get_stats_snapshot();
    let gate_inputs = QualityGateCheckInputs {
        posterior: &posterior,
        prior_cov_9d: &calibration.prior_cov_9d,
        theta_ns: config.theta_ns,
        n_total: state.n_total(),
        elapsed_secs: state.elapsed().as_secs_f64(),
        recent_kl_sum: if state.has_kl_history() {
            Some(state.recent_kl_sum())
        } else {
            None
        },
        samples_per_second: calibration.samples_per_second,
        calibration_snapshot: Some(&calibration.calibration_snapshot),
        current_stats_snapshot: current_stats.as_ref(),
        c_floor: calibration.c_floor,
        theta_tick: calibration.theta_tick,
        projection_mismatch_q: if posterior.projection_mismatch_q.is_nan() {
            None
        } else {
            Some(posterior.projection_mismatch_q)
        },
        projection_mismatch_thresh: calibration.projection_mismatch_thresh,
    };

    match check_quality_gates(&gate_inputs, &config.quality_gates) {
        QualityGateResult::Stop(reason) => {
            return Ok(Some(AdaptiveOutcome::Inconclusive {
                reason,
                posterior: Some(posterior),
                samples_per_class: state.n_total(),
                elapsed: state.elapsed(),
            }));
        }
        QualityGateResult::Continue => {
            // All gates passed - proceed to decision boundaries
        }
    }

    // =========================================================================
    // Decision boundaries (only reached if ALL quality gates passed)
    // =========================================================================
    if posterior.leak_probability > config.fail_threshold {
        return Ok(Some(AdaptiveOutcome::LeakDetected {
            posterior,
            samples_per_class: state.n_total(),
            elapsed: state.elapsed(),
        }));
    }

    if posterior.leak_probability < config.pass_threshold {
        return Ok(Some(AdaptiveOutcome::NoLeakDetected {
            posterior,
            samples_per_class: state.n_total(),
            elapsed: state.elapsed(),
        }));
    }

    // Not yet decisive - continue sampling
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Matrix2, Matrix9, Vector2, Vector9};

    fn make_calibration() -> Calibration {
        use crate::adaptive::CalibrationSnapshot;
        use crate::statistics::StatsSnapshot;

        // Create a default calibration snapshot for tests
        let default_stats = StatsSnapshot {
            mean: 1000.0,
            variance: 25.0,
            autocorr_lag1: 0.1,
            count: 5000,
        };
        let calibration_snapshot = CalibrationSnapshot::new(default_stats, default_stats);

        Calibration {
            sigma_rate: Matrix9::identity() * 1000.0,
            block_length: 10,
            prior_cov_9d: Matrix9::identity() * 10000.0, // 9D prior covariance
            timer_resolution_ns: 1.0,
            samples_per_second: 100_000.0,
            discrete_mode: false,
            theta_ns: 100.0,
            calibration_samples: 5000,
            mde_shift_ns: 5.0,
            mde_tail_ns: 10.0,
            preflight_result: crate::preflight::PreflightResult::new(),
            calibration_snapshot,
            c_floor: 3535.5, // ~50 * sqrt(5000) - conservative floor-rate constant
            projection_mismatch_thresh: 18.48, // chi-squared(7, 0.99) fallback
            theta_floor_initial: 50.0, // c_floor / sqrt(5000) = 50
            theta_eff: 100.0, // max(theta_ns, theta_floor_initial)
            theta_tick: 1.0, // Timer resolution
            rng_seed: 42,    // Test seed
        }
    }

    #[test]
    fn test_adaptive_config_builder() {
        let config = AdaptiveConfig::with_theta(50.0)
            .pass_threshold(0.01)
            .fail_threshold(0.99)
            .time_budget(Duration::from_secs(60))
            .max_samples(500_000);

        assert_eq!(config.theta_ns, 50.0);
        assert_eq!(config.pass_threshold, 0.01);
        assert_eq!(config.fail_threshold, 0.99);
        assert_eq!(config.time_budget, Duration::from_secs(60));
        assert_eq!(config.max_samples, 500_000);
    }

    #[test]
    fn test_adaptive_outcome_accessors() {
        let posterior = Posterior::new(
            Vector9::zeros(),           // delta_post (dummy 9D)
            Matrix9::identity(),        // lambda_post (dummy 9D)
            Vector2::new(10.0, 5.0),    // beta_proj
            Matrix2::new(1.0, 0.0, 0.0, 1.0), // beta_proj_cov
            0.95,                       // leak_probability
            5.0,                        // projection_mismatch_q
            1000,                       // n
        );

        let outcome = AdaptiveOutcome::LeakDetected {
            posterior: posterior.clone(),
            samples_per_class: 1000,
            elapsed: Duration::from_secs(1),
        };

        assert!(outcome.is_leak_detected());
        assert!(outcome.is_conclusive());
        assert_eq!(outcome.leak_probability(), Some(0.95));

        let outcome = AdaptiveOutcome::NoLeakDetected {
            posterior,
            samples_per_class: 1000,
            elapsed: Duration::from_secs(1),
        };

        assert!(!outcome.is_leak_detected());
        assert!(outcome.is_conclusive());
    }

    #[test]
    fn test_adaptive_step_insufficient_samples() {
        let calibration = make_calibration();
        let mut state = AdaptiveState::new();
        state.add_batch(vec![100; 10], vec![101; 10]); // Only 10 samples

        let config = AdaptiveConfig::default();
        let result = adaptive_step(&calibration, &mut state, 1.0, &config);

        // Should return None (need more samples)
        assert!(matches!(result, Ok(None)));
    }

    #[test]
    fn test_compute_posterior_basic() {
        let calibration = make_calibration();
        let mut state = AdaptiveState::new();

        // Add samples with no timing difference
        let baseline: Vec<u64> = (0..1000).map(|i| 1000 + (i % 10)).collect();
        let sample: Vec<u64> = (0..1000).map(|i| 1000 + (i % 10)).collect();
        state.add_batch(baseline, sample);

        let config = AdaptiveConfig::with_theta(100.0);

        let posterior = compute_posterior_from_state(&state, &calibration, 1.0, &config);

        assert!(posterior.is_some());
        let p = posterior.unwrap();

        // With identical distributions, leak probability should be low
        assert!(
            p.leak_probability < 0.5,
            "Identical distributions should have low leak probability, got {}",
            p.leak_probability
        );
    }

    #[test]
    fn test_compute_posterior_with_difference() {
        let calibration = make_calibration();
        let mut state = AdaptiveState::new();

        // Add samples with clear timing difference (200ns)
        let baseline: Vec<u64> = (0..1000).map(|i| 1000 + (i % 10)).collect();
        let sample: Vec<u64> = (0..1000).map(|i| 1200 + (i % 10)).collect();
        state.add_batch(baseline, sample);

        let config = AdaptiveConfig::with_theta(100.0); // Effect is 200ns, threshold is 100ns

        let posterior = compute_posterior_from_state(&state, &calibration, 1.0, &config);

        assert!(posterior.is_some());
        let p = posterior.unwrap();

        // With 200ns difference vs 100ns threshold, leak probability should be high
        assert!(
            p.leak_probability > 0.5,
            "Clear difference should have high leak probability, got {}",
            p.leak_probability
        );
    }
}
