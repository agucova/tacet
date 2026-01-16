//! State management for adaptive sampling loop.
//!
//! This module defines the state maintained during the adaptive sampling process,
//! including sample storage, posterior tracking, and KL divergence history.

use std::collections::VecDeque;
use std::time::Instant;

use crate::types::{Matrix2, Vector2};

/// Posterior distribution parameters for the effect vector beta = (mu, tau).
///
/// The posterior is Gaussian: beta | Delta ~ N(beta_mean, beta_cov)
/// where:
/// - mu is the uniform shift component
/// - tau is the tail effect component
///
/// See spec Section 2.5 for the Bayesian model details.
#[derive(Clone, Debug)]
pub struct Posterior {
    /// Posterior mean of beta = (mu, tau).
    pub beta_mean: Vector2,

    /// Posterior covariance of beta.
    pub beta_cov: Matrix2,

    /// Leak probability: P(max_k |(X*beta)_k| > theta | Delta).
    /// Computed via Monte Carlo integration over the posterior.
    pub leak_probability: f64,

    /// Number of samples used in this posterior computation.
    pub n: usize,
}

impl Posterior {
    /// Create a new posterior with given parameters.
    pub fn new(beta_mean: Vector2, beta_cov: Matrix2, leak_probability: f64, n: usize) -> Self {
        Self {
            beta_mean,
            beta_cov,
            leak_probability,
            n,
        }
    }

    /// Get the shift component (mu) from the posterior mean.
    pub fn shift_ns(&self) -> f64 {
        self.beta_mean[0]
    }

    /// Get the tail component (tau) from the posterior mean.
    pub fn tail_ns(&self) -> f64 {
        self.beta_mean[1]
    }

    /// Get the standard error of the shift component.
    pub fn shift_se(&self) -> f64 {
        self.beta_cov[(0, 0)].sqrt()
    }

    /// Get the standard error of the tail component.
    pub fn tail_se(&self) -> f64 {
        self.beta_cov[(1, 1)].sqrt()
    }
}

/// State maintained during adaptive sampling loop.
///
/// This struct accumulates timing samples and tracks the evolution of the
/// posterior distribution across batches, enabling quality gate checks like
/// KL divergence monitoring.
pub struct AdaptiveState {
    /// Baseline class timing samples (in cycles/ticks/native units).
    pub baseline_samples: Vec<u64>,

    /// Sample class timing samples (in cycles/ticks/native units).
    pub sample_samples: Vec<u64>,

    /// Previous posterior for KL divergence tracking.
    /// None until we have at least one posterior computed.
    pub previous_posterior: Option<Posterior>,

    /// Recent KL divergences (last 5 batches) for learning rate monitoring.
    /// If sum of recent KL < 0.001, learning has stalled.
    pub recent_kl_divergences: VecDeque<f64>,

    /// Start time of adaptive phase for timeout tracking.
    pub start_time: Instant,

    /// Number of batches collected so far.
    pub batch_count: usize,
}

impl AdaptiveState {
    /// Create a new empty adaptive state.
    pub fn new() -> Self {
        Self {
            baseline_samples: Vec::new(),
            sample_samples: Vec::new(),
            previous_posterior: None,
            recent_kl_divergences: VecDeque::with_capacity(5),
            start_time: Instant::now(),
            batch_count: 0,
        }
    }

    /// Create a new adaptive state with pre-allocated capacity.
    pub fn with_capacity(expected_samples: usize) -> Self {
        Self {
            baseline_samples: Vec::with_capacity(expected_samples),
            sample_samples: Vec::with_capacity(expected_samples),
            previous_posterior: None,
            recent_kl_divergences: VecDeque::with_capacity(5),
            start_time: Instant::now(),
            batch_count: 0,
        }
    }

    /// Get the total number of samples per class.
    pub fn n_total(&self) -> usize {
        self.baseline_samples.len()
    }

    /// Get elapsed time since adaptive phase started.
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Add a batch of samples to the state.
    ///
    /// Both baseline and sample vectors should have the same length.
    pub fn add_batch(&mut self, baseline: Vec<u64>, sample: Vec<u64>) {
        debug_assert_eq!(
            baseline.len(),
            sample.len(),
            "Baseline and sample batch sizes must match"
        );
        self.baseline_samples.extend(baseline);
        self.sample_samples.extend(sample);
        self.batch_count += 1;
    }

    /// Update KL divergence history with a new value.
    ///
    /// Maintains a sliding window of the last 5 KL divergences for
    /// learning rate monitoring.
    pub fn update_kl(&mut self, kl: f64) {
        self.recent_kl_divergences.push_back(kl);
        if self.recent_kl_divergences.len() > 5 {
            self.recent_kl_divergences.pop_front();
        }
    }

    /// Get the sum of recent KL divergences.
    ///
    /// Used to detect learning stall (sum < 0.001 indicates posterior
    /// has stopped updating despite new data).
    pub fn recent_kl_sum(&self) -> f64 {
        self.recent_kl_divergences.iter().sum()
    }

    /// Check if we have enough KL history for learning rate assessment.
    pub fn has_kl_history(&self) -> bool {
        self.recent_kl_divergences.len() >= 5
    }

    /// Update the posterior and track KL divergence.
    ///
    /// Returns the KL divergence from the previous posterior, or 0.0 if
    /// this is the first posterior.
    pub fn update_posterior(&mut self, new_posterior: Posterior) -> f64 {
        let kl = if let Some(ref prev) = self.previous_posterior {
            crate::adaptive::kl_divergence_gaussian(&new_posterior, prev)
        } else {
            0.0
        };

        self.previous_posterior = Some(new_posterior);

        if kl.is_finite() {
            self.update_kl(kl);
        }

        kl
    }

    /// Get the current posterior, if computed.
    pub fn current_posterior(&self) -> Option<&Posterior> {
        self.previous_posterior.as_ref()
    }

    /// Convert baseline samples to f64 nanoseconds.
    pub fn baseline_ns(&self, ns_per_tick: f64) -> Vec<f64> {
        self.baseline_samples
            .iter()
            .map(|&t| t as f64 * ns_per_tick)
            .collect()
    }

    /// Convert sample samples to f64 nanoseconds.
    pub fn sample_ns(&self, ns_per_tick: f64) -> Vec<f64> {
        self.sample_samples
            .iter()
            .map(|&t| t as f64 * ns_per_tick)
            .collect()
    }
}

impl Default for AdaptiveState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_state_new() {
        let state = AdaptiveState::new();
        assert_eq!(state.n_total(), 0);
        assert_eq!(state.batch_count, 0);
        assert!(state.previous_posterior.is_none());
        assert!(!state.has_kl_history());
    }

    #[test]
    fn test_add_batch() {
        let mut state = AdaptiveState::new();
        state.add_batch(vec![100, 101, 102], vec![200, 201, 202]);

        assert_eq!(state.n_total(), 3);
        assert_eq!(state.batch_count, 1);
        assert_eq!(state.baseline_samples, vec![100, 101, 102]);
        assert_eq!(state.sample_samples, vec![200, 201, 202]);
    }

    #[test]
    fn test_kl_history() {
        let mut state = AdaptiveState::new();

        for i in 0..5 {
            state.update_kl(0.1 * (i + 1) as f64);
        }

        assert!(state.has_kl_history());
        assert!((state.recent_kl_sum() - 1.5).abs() < 1e-10); // 0.1 + 0.2 + 0.3 + 0.4 + 0.5

        // Adding one more should evict the oldest
        state.update_kl(1.0);
        assert!((state.recent_kl_sum() - 2.4).abs() < 1e-10); // 0.2 + 0.3 + 0.4 + 0.5 + 1.0
    }

    #[test]
    fn test_posterior_accessors() {
        let posterior = Posterior::new(
            Vector2::new(10.0, 5.0),
            Matrix2::new(4.0, 0.0, 0.0, 1.0),
            0.75,
            1000,
        );

        assert_eq!(posterior.shift_ns(), 10.0);
        assert_eq!(posterior.tail_ns(), 5.0);
        assert_eq!(posterior.shift_se(), 2.0); // sqrt(4.0)
        assert_eq!(posterior.tail_se(), 1.0); // sqrt(1.0)
    }
}
