//! State management for adaptive sampling loop (std wrapper).
//!
//! This module provides a wrapper around `timing_oracle_core::adaptive::AdaptiveState`
//! that adds time tracking using `std::time::Instant`.

use std::time::{Duration, Instant};

use timing_oracle_core::adaptive::AdaptiveState as CoreAdaptiveState;
use timing_oracle_core::adaptive::{CalibrationSnapshot, Posterior};
use timing_oracle_core::statistics::StatsSnapshot;

/// State maintained during adaptive sampling loop.
///
/// This is a wrapper around the no_std `AdaptiveState` from `timing-oracle-core`
/// that adds time tracking using `std::time::Instant`.
pub struct AdaptiveState {
    /// The core state (no_std compatible).
    core: CoreAdaptiveState,

    /// Start time of adaptive phase for timeout tracking.
    start_time: Instant,
}

impl AdaptiveState {
    /// Create a new empty adaptive state.
    pub fn new() -> Self {
        Self {
            core: CoreAdaptiveState::new(),
            start_time: Instant::now(),
        }
    }

    /// Create a new adaptive state with pre-allocated capacity.
    pub fn with_capacity(expected_samples: usize) -> Self {
        Self {
            core: CoreAdaptiveState::with_capacity(expected_samples),
            start_time: Instant::now(),
        }
    }

    /// Get a reference to the core state.
    pub fn core(&self) -> &CoreAdaptiveState {
        &self.core
    }

    /// Get a mutable reference to the core state.
    pub fn core_mut(&mut self) -> &mut CoreAdaptiveState {
        &mut self.core
    }

    /// Get the total number of samples per class.
    pub fn n_total(&self) -> usize {
        self.core.n_total()
    }

    /// Get elapsed time since adaptive phase started.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Reset the start time (useful when restarting the adaptive phase).
    pub fn reset_start_time(&mut self) {
        self.start_time = Instant::now();
    }

    /// Get the batch count.
    pub fn batch_count(&self) -> usize {
        self.core.batch_count
    }

    /// Access baseline samples directly.
    pub fn baseline_samples(&self) -> &[u64] {
        &self.core.baseline_samples
    }

    /// Access sample samples directly.
    pub fn sample_samples(&self) -> &[u64] {
        &self.core.sample_samples
    }

    /// Add a batch of samples to the state.
    ///
    /// Both baseline and sample vectors should have the same length.
    /// Note: This method does not track online statistics since ns_per_tick is not known.
    /// Use `add_batch_with_conversion` if you need drift detection.
    pub fn add_batch(&mut self, baseline: Vec<u64>, sample: Vec<u64>) {
        self.core.add_batch(baseline, sample);
    }

    /// Add a batch of samples and track online statistics for drift detection.
    ///
    /// Both baseline and sample vectors should have the same length.
    ///
    /// # Arguments
    ///
    /// * `baseline` - Baseline class timing samples (in native units)
    /// * `sample` - Sample class timing samples (in native units)
    /// * `ns_per_tick` - Conversion factor from native units to nanoseconds
    pub fn add_batch_with_conversion(
        &mut self,
        baseline: Vec<u64>,
        sample: Vec<u64>,
        ns_per_tick: f64,
    ) {
        self.core
            .add_batch_with_conversion(baseline, sample, ns_per_tick);
    }

    /// Update KL divergence history with a new value.
    pub fn update_kl(&mut self, kl: f64) {
        self.core.update_kl(kl);
    }

    /// Get the sum of recent KL divergences.
    pub fn recent_kl_sum(&self) -> f64 {
        self.core.recent_kl_sum()
    }

    /// Check if we have enough KL history for learning rate assessment.
    pub fn has_kl_history(&self) -> bool {
        self.core.has_kl_history()
    }

    /// Update the posterior and track KL divergence.
    pub fn update_posterior(&mut self, new_posterior: Posterior) -> f64 {
        self.core.update_posterior(new_posterior)
    }

    /// Get the current posterior, if computed.
    pub fn current_posterior(&self) -> Option<&Posterior> {
        self.core.current_posterior()
    }

    /// Convert baseline samples to f64 nanoseconds.
    pub fn baseline_ns(&self, ns_per_tick: f64) -> Vec<f64> {
        self.core.baseline_ns(ns_per_tick)
    }

    /// Convert sample samples to f64 nanoseconds.
    pub fn sample_ns(&self, ns_per_tick: f64) -> Vec<f64> {
        self.core.sample_ns(ns_per_tick)
    }

    /// Get the current online statistics for the baseline class.
    pub fn baseline_stats(&self) -> Option<StatsSnapshot> {
        self.core.baseline_stats()
    }

    /// Get the current online statistics for the sample class.
    pub fn sample_stats(&self) -> Option<StatsSnapshot> {
        self.core.sample_stats()
    }

    /// Get a CalibrationSnapshot from the current online statistics.
    pub fn get_stats_snapshot(&self) -> Option<CalibrationSnapshot> {
        self.core.get_stats_snapshot()
    }

    /// Check if online statistics are being tracked.
    pub fn has_stats_tracking(&self) -> bool {
        self.core.has_stats_tracking()
    }

    /// Reset the state for a new test run.
    pub fn reset(&mut self) {
        self.core.reset();
        self.start_time = Instant::now();
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
    use crate::types::{Matrix2, Vector2};

    #[test]
    fn test_adaptive_state_new() {
        let state = AdaptiveState::new();
        assert_eq!(state.n_total(), 0);
        assert_eq!(state.batch_count(), 0);
        assert!(state.current_posterior().is_none());
        assert!(!state.has_kl_history());
    }

    #[test]
    fn test_add_batch() {
        let mut state = AdaptiveState::new();
        state.add_batch(vec![100, 101, 102], vec![200, 201, 202]);

        assert_eq!(state.n_total(), 3);
        assert_eq!(state.batch_count(), 1);
        assert_eq!(state.baseline_samples(), &[100, 101, 102]);
        assert_eq!(state.sample_samples(), &[200, 201, 202]);
    }

    #[test]
    fn test_elapsed() {
        let state = AdaptiveState::new();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = state.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_kl_history() {
        let mut state = AdaptiveState::new();

        for i in 0..5 {
            state.update_kl(0.1 * (i + 1) as f64);
        }

        assert!(state.has_kl_history());
        assert!((state.recent_kl_sum() - 1.5).abs() < 1e-10);

        state.update_kl(1.0);
        assert!((state.recent_kl_sum() - 2.4).abs() < 1e-10);
    }

    #[test]
    fn test_posterior_update() {
        let mut state = AdaptiveState::new();

        let posterior = Posterior::new(
            Vector2::new(10.0, 5.0),
            Matrix2::new(4.0, 0.0, 0.0, 1.0),
            0.75,
            1000,
            5.0, // model_fit_q
        );

        let kl = state.update_posterior(posterior.clone());
        assert_eq!(kl, 0.0); // First posterior has no previous

        assert!(state.current_posterior().is_some());
    }

    #[test]
    fn test_add_batch_with_conversion() {
        let mut state = AdaptiveState::new();

        state.add_batch_with_conversion(vec![100, 110, 120], vec![200, 210, 220], 2.0);

        assert_eq!(state.n_total(), 3);
        assert_eq!(state.batch_count(), 1);
        assert!(state.has_stats_tracking());
    }

    #[test]
    fn test_online_stats_tracking() {
        let mut state = AdaptiveState::new();

        let baseline: Vec<u64> = (0..100).map(|i| 1000 + (i % 10)).collect();
        let sample: Vec<u64> = (0..100).map(|i| 1100 + (i % 10)).collect();
        state.add_batch_with_conversion(baseline, sample, 1.0);

        let baseline_stats = state.baseline_stats().expect("Should have baseline stats");
        assert_eq!(baseline_stats.count, 100);
        assert!(
            (baseline_stats.mean - 1004.5).abs() < 1.0,
            "Baseline mean {} should be near 1004.5",
            baseline_stats.mean
        );

        let sample_stats = state.sample_stats().expect("Should have sample stats");
        assert_eq!(sample_stats.count, 100);
        assert!(
            (sample_stats.mean - 1104.5).abs() < 1.0,
            "Sample mean {} should be near 1104.5",
            sample_stats.mean
        );
    }

    #[test]
    fn test_reset() {
        let mut state = AdaptiveState::new();
        state.add_batch_with_conversion(vec![100, 110], vec![200, 210], 1.0);
        state.update_kl(0.5);

        let posterior = Posterior::new(
            Vector2::new(10.0, 5.0),
            Matrix2::new(4.0, 0.0, 0.0, 1.0),
            0.75,
            100,
            5.0, // model_fit_q
        );
        state.update_posterior(posterior);

        assert!(state.n_total() > 0);

        state.reset();

        assert_eq!(state.n_total(), 0);
        assert_eq!(state.batch_count(), 0);
        assert!(state.current_posterior().is_none());
        assert!(!state.has_kl_history());
        assert!(!state.has_stats_tracking());
    }

    #[test]
    fn test_stats_not_tracked_without_conversion() {
        let mut state = AdaptiveState::new();

        state.add_batch(vec![100, 110, 120], vec![200, 210, 220]);

        assert!(!state.has_stats_tracking());
        assert!(state.baseline_stats().is_none());
        assert!(state.sample_stats().is_none());
    }
}
