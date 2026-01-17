//! Adaptive sampling step logic (no_std compatible).
//!
//! This module provides the core adaptive step logic that can be used in
//! no_std environments. Time tracking is caller-provided rather than using
//! `std::time::Instant`.
//!
//! # Usage Pattern
//!
//! ```ignore
//! let mut state = AdaptiveState::new();
//! let calibration = /* ... */;
//! let config = AdaptiveStepConfig::default();
//!
//! loop {
//!     // Caller collects samples and tracks time
//!     state.add_batch(baseline, sample);
//!     let elapsed_secs = /* measured by caller */;
//!
//!     match adaptive_step(&calibration, &mut state, ns_per_tick, elapsed_secs, &config) {
//!         StepResult::Decision(outcome) => return outcome,
//!         StepResult::Continue { .. } => continue,
//!     }
//! }
//! ```

use alloc::string::String;

use crate::analysis::bayes::compute_bayes_factor;
use crate::constants::DEFAULT_SEED;
use crate::math::sqrt;
use crate::statistics::{compute_deciles_inplace, compute_midquantile_deciles};

use super::{
    check_quality_gates, AdaptiveState, Calibration, InconclusiveReason, Posterior,
    QualityGateCheckInputs, QualityGateConfig, QualityGateResult,
};

/// Configuration for the adaptive step (no_std compatible).
///
/// Uses `f64` for time values instead of `std::time::Duration`.
#[derive(Debug, Clone)]
pub struct AdaptiveStepConfig {
    /// Threshold below which we pass (no significant leak).
    pub pass_threshold: f64,

    /// Threshold above which we fail (leak detected).
    pub fail_threshold: f64,

    /// Time budget in seconds.
    pub time_budget_secs: f64,

    /// Maximum samples per class.
    pub max_samples: usize,

    /// Effect threshold (theta) in nanoseconds.
    pub theta_ns: f64,

    /// Random seed for Monte Carlo integration.
    pub seed: u64,

    /// Quality gate configuration.
    pub quality_gates: QualityGateConfig,
}

impl Default for AdaptiveStepConfig {
    fn default() -> Self {
        Self {
            pass_threshold: 0.05,
            fail_threshold: 0.95,
            time_budget_secs: 30.0,
            max_samples: 1_000_000,
            theta_ns: 100.0,
            seed: DEFAULT_SEED,
            quality_gates: QualityGateConfig::default(),
        }
    }
}

impl AdaptiveStepConfig {
    /// Create a new config with the given theta threshold.
    pub fn with_theta(theta_ns: f64) -> Self {
        let mut config = Self {
            theta_ns,
            ..Self::default()
        };
        config.quality_gates.pass_threshold = config.pass_threshold;
        config.quality_gates.fail_threshold = config.fail_threshold;
        config.quality_gates.time_budget_secs = config.time_budget_secs;
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

    /// Builder method to set time budget in seconds.
    pub fn time_budget_secs(mut self, secs: f64) -> Self {
        self.time_budget_secs = secs;
        self.quality_gates.time_budget_secs = secs;
        self
    }

    /// Builder method to set max samples.
    pub fn max_samples(mut self, max: usize) -> Self {
        self.max_samples = max;
        self.quality_gates.max_samples = max;
        self
    }
}

/// Outcome of an adaptive sampling decision (no_std compatible).
///
/// Uses `f64` for elapsed time instead of `std::time::Duration`.
#[derive(Debug, Clone)]
pub enum AdaptiveOutcome {
    /// Leak probability exceeded fail threshold - timing leak detected.
    LeakDetected {
        /// Final posterior distribution.
        posterior: Posterior,
        /// Number of samples collected per class.
        samples_per_class: usize,
        /// Time spent in seconds.
        elapsed_secs: f64,
    },

    /// Leak probability dropped below pass threshold - no significant leak.
    NoLeakDetected {
        /// Final posterior distribution.
        posterior: Posterior,
        /// Number of samples collected per class.
        samples_per_class: usize,
        /// Time spent in seconds.
        elapsed_secs: f64,
    },

    /// A quality gate triggered before reaching a decision.
    Inconclusive {
        /// Reason for stopping.
        reason: InconclusiveReason,
        /// Final posterior distribution (if available).
        posterior: Option<Posterior>,
        /// Number of samples collected per class.
        samples_per_class: usize,
        /// Time spent in seconds.
        elapsed_secs: f64,
    },
}

impl AdaptiveOutcome {
    /// Get the final leak probability, if available.
    pub fn leak_probability(&self) -> Option<f64> {
        match self {
            AdaptiveOutcome::LeakDetected { posterior, .. } => Some(posterior.leak_probability),
            AdaptiveOutcome::NoLeakDetected { posterior, .. } => Some(posterior.leak_probability),
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

    /// Get the number of samples per class.
    pub fn samples_per_class(&self) -> usize {
        match self {
            AdaptiveOutcome::LeakDetected {
                samples_per_class, ..
            } => *samples_per_class,
            AdaptiveOutcome::NoLeakDetected {
                samples_per_class, ..
            } => *samples_per_class,
            AdaptiveOutcome::Inconclusive {
                samples_per_class, ..
            } => *samples_per_class,
        }
    }

    /// Get elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        match self {
            AdaptiveOutcome::LeakDetected { elapsed_secs, .. } => *elapsed_secs,
            AdaptiveOutcome::NoLeakDetected { elapsed_secs, .. } => *elapsed_secs,
            AdaptiveOutcome::Inconclusive { elapsed_secs, .. } => *elapsed_secs,
        }
    }
}

/// Result of a single adaptive step.
#[derive(Debug, Clone)]
pub enum StepResult {
    /// A decision was reached (Pass, Fail, or Inconclusive).
    Decision(AdaptiveOutcome),

    /// Continue sampling - no decision yet.
    Continue {
        /// Current posterior distribution.
        posterior: Posterior,
        /// Number of samples collected per class so far.
        samples_per_class: usize,
    },
}

impl StepResult {
    /// Check if a decision was reached.
    pub fn is_decision(&self) -> bool {
        matches!(self, StepResult::Decision(_))
    }

    /// Get the outcome if a decision was reached.
    pub fn into_decision(self) -> Option<AdaptiveOutcome> {
        match self {
            StepResult::Decision(outcome) => Some(outcome),
            StepResult::Continue { .. } => None,
        }
    }

    /// Get the current leak probability.
    pub fn leak_probability(&self) -> Option<f64> {
        match self {
            StepResult::Decision(outcome) => outcome.leak_probability(),
            StepResult::Continue { posterior, .. } => Some(posterior.leak_probability),
        }
    }
}

/// Single-iteration adaptive step (no_std compatible).
///
/// This function performs one iteration of the adaptive sampling loop:
/// 1. Computes the posterior from current samples
/// 2. Checks if decision thresholds are met
/// 3. Checks quality gates
/// 4. Returns whether to continue or stop
///
/// # Arguments
///
/// * `calibration` - Results from calibration phase
/// * `state` - Current adaptive state with accumulated samples
/// * `ns_per_tick` - Conversion factor from native units to nanoseconds
/// * `elapsed_secs` - Elapsed time in seconds (provided by caller)
/// * `config` - Adaptive step configuration
///
/// # Returns
///
/// - `StepResult::Decision(outcome)` - Decision reached or quality gate triggered
/// - `StepResult::Continue { .. }` - Need more samples
pub fn adaptive_step(
    calibration: &Calibration,
    state: &mut AdaptiveState,
    ns_per_tick: f64,
    elapsed_secs: f64,
    config: &AdaptiveStepConfig,
) -> StepResult {
    // Compute posterior from current samples
    let posterior = match compute_posterior(state, calibration, ns_per_tick, config) {
        Some(p) => p,
        None => {
            // Not enough samples for reliable posterior
            if state.n_total() < 20 {
                // Need more samples - not a failure, just need to continue
                // We can't return Continue without a posterior, so return a dummy
                // Actually, let's just return Inconclusive with a clear message
                return StepResult::Decision(AdaptiveOutcome::Inconclusive {
                    reason: InconclusiveReason::DataTooNoisy {
                        message: String::from("Insufficient samples for posterior computation"),
                        guidance: String::from("Need at least 20 samples per class"),
                        variance_ratio: 1.0,
                    },
                    posterior: None,
                    samples_per_class: state.n_total(),
                    elapsed_secs,
                });
            }
            // Computation failed for other reasons
            return StepResult::Decision(AdaptiveOutcome::Inconclusive {
                reason: InconclusiveReason::DataTooNoisy {
                    message: String::from("Could not compute posterior from samples"),
                    guidance: String::from("Check timer resolution and sample count"),
                    variance_ratio: 1.0,
                },
                posterior: None,
                samples_per_class: state.n_total(),
                elapsed_secs,
            });
        }
    };

    // Track KL divergence for learning rate monitoring
    let _kl = state.update_posterior(posterior.clone());

    // Check decision boundaries
    if posterior.leak_probability > config.fail_threshold {
        return StepResult::Decision(AdaptiveOutcome::LeakDetected {
            posterior,
            samples_per_class: state.n_total(),
            elapsed_secs,
        });
    }

    if posterior.leak_probability < config.pass_threshold {
        return StepResult::Decision(AdaptiveOutcome::NoLeakDetected {
            posterior,
            samples_per_class: state.n_total(),
            elapsed_secs,
        });
    }

    // Check quality gates
    let current_stats = state.get_stats_snapshot();
    let gate_inputs = QualityGateCheckInputs {
        posterior: &posterior,
        prior_cov: &calibration.prior_cov,
        theta_ns: config.theta_ns,
        n_total: state.n_total(),
        elapsed_secs,
        recent_kl_sum: if state.has_kl_history() {
            Some(state.recent_kl_sum())
        } else {
            None
        },
        samples_per_second: calibration.samples_per_second,
        calibration_snapshot: Some(&calibration.calibration_snapshot),
        current_stats_snapshot: current_stats.as_ref(),
        // v4.1 fields for new quality gates
        c_floor: calibration.c_floor,
        theta_tick: calibration.theta_tick,
        q_fit: if posterior.model_fit_q.is_nan() {
            None
        } else {
            Some(posterior.model_fit_q)
        },
        q_thresh: calibration.q_thresh,
    };

    match check_quality_gates(&gate_inputs, &config.quality_gates) {
        QualityGateResult::Continue => StepResult::Continue {
            posterior,
            samples_per_class: state.n_total(),
        },
        QualityGateResult::Stop(reason) => StepResult::Decision(AdaptiveOutcome::Inconclusive {
            reason,
            posterior: Some(posterior),
            samples_per_class: state.n_total(),
            elapsed_secs,
        }),
    }
}

/// Compute posterior distribution from current state.
///
/// Uses scaled covariance: Sigma_n = Sigma_rate / n
fn compute_posterior(
    state: &AdaptiveState,
    calibration: &Calibration,
    ns_per_tick: f64,
    config: &AdaptiveStepConfig,
) -> Option<Posterior> {
    let n = state.n_total();
    if n < 20 {
        return None; // Need minimum samples for stable quantiles
    }

    // Convert samples to nanoseconds
    let baseline_ns = state.baseline_ns(ns_per_tick);
    let sample_ns = state.sample_ns(ns_per_tick);

    // Compute quantile differences
    let observed_diff = if calibration.discrete_mode {
        let q_baseline = compute_midquantile_deciles(&baseline_ns);
        let q_sample = compute_midquantile_deciles(&sample_ns);
        q_baseline - q_sample
    } else {
        let mut baseline_sorted = baseline_ns;
        let mut sample_sorted = sample_ns;
        let q_baseline = compute_deciles_inplace(&mut baseline_sorted);
        let q_sample = compute_deciles_inplace(&mut sample_sorted);
        q_baseline - q_sample
    };

    // Scale covariance: Sigma_n = Sigma_rate / n
    let sigma_n = calibration.covariance_for_n(n);

    // Compute prior sigmas from prior covariance
    let prior_sigma_shift = sqrt(calibration.prior_cov[(0, 0)]);
    let prior_sigma_tail = sqrt(calibration.prior_cov[(1, 1)]);

    // Run Bayesian inference
    let bayes_result = compute_bayes_factor(
        &observed_diff,
        &sigma_n,
        (prior_sigma_shift, prior_sigma_tail),
        config.theta_ns,
        Some(config.seed),
    );

    Some(Posterior::new(
        bayes_result.beta_mean,
        bayes_result.beta_cov,
        bayes_result.leak_probability,
        n,
        bayes_result.model_fit_q,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::StatsSnapshot;
    use crate::types::{Matrix2, Matrix9, Vector2};

    fn make_test_calibration() -> Calibration {
        use crate::adaptive::CalibrationSnapshot;

        let snapshot = CalibrationSnapshot::new(
            StatsSnapshot {
                count: 5000,
                mean: 1000.0,
                variance: 25.0,
                autocorr_lag1: 0.1,
            },
            StatsSnapshot {
                count: 5000,
                mean: 1000.0,
                variance: 25.0,
                autocorr_lag1: 0.1,
            },
        );

        Calibration::new(
            Matrix9::identity() * 1000.0,             // sigma_rate
            10,                                       // block_length
            Matrix2::new(10000.0, 0.0, 0.0, 10000.0), // prior_cov
            100.0,                                    // theta_ns
            5000,                                     // calibration_samples
            false,                                    // discrete_mode
            5.0,                                      // mde_shift_ns
            10.0,                                     // mde_tail_ns
            snapshot,                                 // calibration_snapshot
            1.0,                                      // timer_resolution_ns
            100_000.0,                                // samples_per_second
            // v4.1 fields
            10.0,  // c_floor
            18.48, // q_thresh (chi-squared(7, 0.99) fallback)
            0.001, // theta_tick
            100.0, // theta_eff
            0.1,   // theta_floor_initial
            42,    // rng_seed
        )
    }

    #[test]
    fn test_adaptive_step_config_default() {
        let config = AdaptiveStepConfig::default();
        assert!((config.pass_threshold - 0.05).abs() < 1e-10);
        assert!((config.fail_threshold - 0.95).abs() < 1e-10);
        assert!((config.time_budget_secs - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_step_config_builder() {
        let config = AdaptiveStepConfig::with_theta(50.0)
            .pass_threshold(0.01)
            .fail_threshold(0.99)
            .time_budget_secs(60.0)
            .max_samples(500_000);

        assert!((config.theta_ns - 50.0).abs() < 1e-10);
        assert!((config.pass_threshold - 0.01).abs() < 1e-10);
        assert!((config.fail_threshold - 0.99).abs() < 1e-10);
        assert!((config.time_budget_secs - 60.0).abs() < 1e-10);
        assert_eq!(config.max_samples, 500_000);
    }

    #[test]
    fn test_adaptive_outcome_accessors() {
        let posterior = Posterior::new(
            Vector2::new(10.0, 5.0),
            Matrix2::new(1.0, 0.0, 0.0, 1.0),
            0.95,
            1000,
            5.0, // model_fit_q
        );

        let outcome = AdaptiveOutcome::LeakDetected {
            posterior: posterior.clone(),
            samples_per_class: 1000,
            elapsed_secs: 1.5,
        };

        assert!(outcome.is_leak_detected());
        assert!(outcome.is_conclusive());
        assert_eq!(outcome.leak_probability(), Some(0.95));
        assert_eq!(outcome.samples_per_class(), 1000);
        assert!((outcome.elapsed_secs() - 1.5).abs() < 1e-10);

        let outcome = AdaptiveOutcome::NoLeakDetected {
            posterior,
            samples_per_class: 2000,
            elapsed_secs: 2.5,
        };

        assert!(!outcome.is_leak_detected());
        assert!(outcome.is_conclusive());
        assert_eq!(outcome.samples_per_class(), 2000);
    }

    #[test]
    fn test_step_result_accessors() {
        let posterior = Posterior::new(
            Vector2::new(10.0, 5.0),
            Matrix2::new(1.0, 0.0, 0.0, 1.0),
            0.5,
            1000,
            5.0, // model_fit_q
        );

        let result = StepResult::Continue {
            posterior: posterior.clone(),
            samples_per_class: 1000,
        };

        assert!(!result.is_decision());
        assert_eq!(result.leak_probability(), Some(0.5));

        let result = StepResult::Decision(AdaptiveOutcome::LeakDetected {
            posterior,
            samples_per_class: 1000,
            elapsed_secs: 1.0,
        });

        assert!(result.is_decision());
    }

    #[test]
    fn test_adaptive_step_insufficient_samples() {
        let calibration = make_test_calibration();
        let mut state = AdaptiveState::new();
        state.add_batch(vec![100; 10], vec![101; 10]); // Only 10 samples

        let config = AdaptiveStepConfig::default();
        let result = adaptive_step(&calibration, &mut state, 1.0, 0.1, &config);

        // Should return Inconclusive (need more samples)
        assert!(result.is_decision());
        if let StepResult::Decision(AdaptiveOutcome::Inconclusive { reason, .. }) = result {
            assert!(matches!(reason, InconclusiveReason::DataTooNoisy { .. }));
        } else {
            panic!("Expected Inconclusive with DataTooNoisy");
        }
    }

    #[test]
    fn test_compute_posterior_no_difference() {
        let calibration = make_test_calibration();
        let mut state = AdaptiveState::new();

        // Add samples with no timing difference
        let baseline: Vec<u64> = (0..1000).map(|i| 1000 + (i % 10)).collect();
        let sample: Vec<u64> = (0..1000).map(|i| 1000 + (i % 10)).collect();
        state.add_batch(baseline, sample);

        let config = AdaptiveStepConfig::with_theta(100.0);

        let posterior = compute_posterior(&state, &calibration, 1.0, &config);

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
        let calibration = make_test_calibration();
        let mut state = AdaptiveState::new();

        // Add samples with clear timing difference (200ns)
        let baseline: Vec<u64> = (0..1000).map(|i| 1000 + (i % 10)).collect();
        let sample: Vec<u64> = (0..1000).map(|i| 1200 + (i % 10)).collect();
        state.add_batch(baseline, sample);

        let config = AdaptiveStepConfig::with_theta(100.0); // Effect is 200ns, threshold is 100ns

        let posterior = compute_posterior(&state, &calibration, 1.0, &config);

        assert!(posterior.is_some());
        let p = posterior.unwrap();

        // With 200ns difference vs 100ns threshold, leak probability should be high
        assert!(
            p.leak_probability > 0.5,
            "Clear difference should have high leak probability, got {}",
            p.leak_probability
        );
    }

    #[test]
    fn test_adaptive_step_detects_leak() {
        let calibration = make_test_calibration();
        let mut state = AdaptiveState::new();

        // Add samples with large timing difference
        let baseline: Vec<u64> = (0..1000).map(|i| 1000 + (i % 10)).collect();
        let sample: Vec<u64> = (0..1000).map(|i| 1500 + (i % 10)).collect();
        state.add_batch(baseline, sample);

        let config = AdaptiveStepConfig::with_theta(100.0);
        let result = adaptive_step(&calibration, &mut state, 1.0, 1.0, &config);

        // Should detect a leak
        assert!(result.is_decision());
        if let StepResult::Decision(outcome) = result {
            assert!(outcome.is_leak_detected());
        } else {
            panic!("Expected Decision");
        }
    }
}
