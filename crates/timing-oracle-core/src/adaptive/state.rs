//! State management for adaptive sampling loop (no_std compatible).
//!
//! This module defines the state maintained during the adaptive sampling process,
//! including sample storage, posterior tracking, and KL divergence history.
//!
//! Time tracking is handled by the caller - this module is stateless with respect
//! to wall-clock time, making it suitable for no_std environments like SGX enclaves.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::statistics::{OnlineStats, StatsSnapshot};

use super::{kl_divergence_gaussian, CalibrationSnapshot, Posterior};

/// State maintained during adaptive sampling loop.
///
/// This struct accumulates timing samples and tracks the evolution of the
/// posterior distribution across batches, enabling quality gate checks like
/// KL divergence monitoring.
///
/// Also maintains online statistics (mean, variance, lag-1 autocorrelation)
/// for condition drift detection between calibration and post-test phases.
///
/// # Time Tracking
///
/// This struct does NOT track elapsed time internally. The caller must provide
/// `elapsed_secs` to functions that need it (e.g., quality gate checks). This
/// allows use in no_std environments where `std::time::Instant` is unavailable.
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

    /// Number of batches collected so far.
    pub batch_count: usize,

    /// Online statistics tracker for baseline class (for drift detection).
    baseline_stats: OnlineStats,

    /// Online statistics tracker for sample class (for drift detection).
    sample_stats: OnlineStats,

    /// Conversion factor from native units to nanoseconds.
    /// Set when first batch is added with ns_per_tick context.
    ns_per_tick: Option<f64>,
}

impl AdaptiveState {
    /// Create a new empty adaptive state.
    pub fn new() -> Self {
        Self {
            baseline_samples: Vec::new(),
            sample_samples: Vec::new(),
            previous_posterior: None,
            recent_kl_divergences: VecDeque::with_capacity(5),
            batch_count: 0,
            baseline_stats: OnlineStats::new(),
            sample_stats: OnlineStats::new(),
            ns_per_tick: None,
        }
    }

    /// Create a new adaptive state with pre-allocated capacity.
    pub fn with_capacity(expected_samples: usize) -> Self {
        Self {
            baseline_samples: Vec::with_capacity(expected_samples),
            sample_samples: Vec::with_capacity(expected_samples),
            previous_posterior: None,
            recent_kl_divergences: VecDeque::with_capacity(5),
            batch_count: 0,
            baseline_stats: OnlineStats::new(),
            sample_stats: OnlineStats::new(),
            ns_per_tick: None,
        }
    }

    /// Get the total number of samples per class.
    pub fn n_total(&self) -> usize {
        self.baseline_samples.len()
    }

    /// Add a batch of samples to the state.
    ///
    /// Both baseline and sample vectors should have the same length.
    /// Note: This method does not track online statistics since ns_per_tick is not known.
    /// Use `add_batch_with_conversion` if you need drift detection.
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
        debug_assert_eq!(
            baseline.len(),
            sample.len(),
            "Baseline and sample batch sizes must match"
        );

        // Store the conversion factor for later use
        self.ns_per_tick = Some(ns_per_tick);

        // Update online statistics with nanosecond-converted values
        for &t in &baseline {
            self.baseline_stats.update(t as f64 * ns_per_tick);
        }
        for &t in &sample {
            self.sample_stats.update(t as f64 * ns_per_tick);
        }

        // Store raw samples
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
            kl_divergence_gaussian(&new_posterior, prev)
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

    /// Get the current online statistics for the baseline class.
    ///
    /// Returns `None` if no samples have been added with conversion tracking.
    pub fn baseline_stats(&self) -> Option<StatsSnapshot> {
        if self.baseline_stats.count() < 2 {
            return None;
        }
        Some(self.baseline_stats.finalize())
    }

    /// Get the current online statistics for the sample class.
    ///
    /// Returns `None` if no samples have been added with conversion tracking.
    pub fn sample_stats(&self) -> Option<StatsSnapshot> {
        if self.sample_stats.count() < 2 {
            return None;
        }
        Some(self.sample_stats.finalize())
    }

    /// Get a CalibrationSnapshot from the current online statistics.
    ///
    /// Returns `None` if insufficient samples have been tracked.
    pub fn get_stats_snapshot(&self) -> Option<CalibrationSnapshot> {
        let baseline = self.baseline_stats()?;
        let sample = self.sample_stats()?;
        Some(CalibrationSnapshot::new(baseline, sample))
    }

    /// Check if online statistics are being tracked.
    ///
    /// Returns `true` if `add_batch_with_conversion` has been used.
    pub fn has_stats_tracking(&self) -> bool {
        self.ns_per_tick.is_some() && self.baseline_stats.count() > 0
    }

    /// Reset the state for a new test run.
    ///
    /// Clears all samples, posteriors, and statistics while preserving capacity.
    pub fn reset(&mut self) {
        self.baseline_samples.clear();
        self.sample_samples.clear();
        self.previous_posterior = None;
        self.recent_kl_divergences.clear();
        self.batch_count = 0;
        self.baseline_stats = OnlineStats::new();
        self.sample_stats = OnlineStats::new();
        self.ns_per_tick = None;
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
    use crate::types::{Matrix2, Matrix9, Vector2, Vector9};

    fn make_test_posterior(
        beta_proj: Vector2,
        beta_proj_cov: Matrix2,
        leak_prob: f64,
        n: usize,
    ) -> Posterior {
        Posterior::new(
            Vector9::zeros(),
            Matrix9::identity(),
            beta_proj,
            beta_proj_cov,
            leak_prob,
            1.0, // projection_mismatch_q
            n,
        )
    }

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
    fn test_posterior_update() {
        let mut state = AdaptiveState::new();

        let posterior1 = make_test_posterior(
            Vector2::new(10.0, 5.0),
            Matrix2::new(4.0, 0.0, 0.0, 1.0),
            0.75,
            1000,
        );

        // First update - no previous posterior
        let kl1 = state.update_posterior(posterior1.clone());
        assert_eq!(kl1, 0.0);
        assert!(state.current_posterior().is_some());

        // Second update - should compute KL
        let posterior2 = make_test_posterior(
            Vector2::new(11.0, 5.5),
            Matrix2::new(3.5, 0.0, 0.0, 0.9),
            0.80,
            2000,
        );
        let kl2 = state.update_posterior(posterior2);
        assert!(kl2 > 0.0); // Should have some divergence
    }

    #[test]
    fn test_add_batch_with_conversion() {
        let mut state = AdaptiveState::new();

        // Add samples with ns_per_tick = 2.0 (2ns per tick)
        state.add_batch_with_conversion(vec![100, 110, 120], vec![200, 210, 220], 2.0);

        assert_eq!(state.n_total(), 3);
        assert_eq!(state.batch_count, 1);
        assert!(state.has_stats_tracking());

        // Check that samples were stored correctly
        assert_eq!(state.baseline_samples, vec![100, 110, 120]);
        assert_eq!(state.sample_samples, vec![200, 210, 220]);
    }

    #[test]
    fn test_online_stats_tracking() {
        let mut state = AdaptiveState::new();

        // Add enough samples for meaningful statistics
        let baseline: Vec<u64> = (0..100).map(|i| 1000 + (i % 10)).collect();
        let sample: Vec<u64> = (0..100).map(|i| 1100 + (i % 10)).collect();
        state.add_batch_with_conversion(baseline, sample, 1.0);

        // Check baseline stats
        let baseline_stats = state.baseline_stats().expect("Should have baseline stats");
        assert_eq!(baseline_stats.count, 100);
        // Mean should be around 1004.5 (0..9 has mean 4.5, plus base 1000)
        assert!(
            (baseline_stats.mean - 1004.5).abs() < 1.0,
            "Baseline mean {} should be near 1004.5",
            baseline_stats.mean
        );

        // Check sample stats
        let sample_stats = state.sample_stats().expect("Should have sample stats");
        assert_eq!(sample_stats.count, 100);
        // Mean should be around 1104.5
        assert!(
            (sample_stats.mean - 1104.5).abs() < 1.0,
            "Sample mean {} should be near 1104.5",
            sample_stats.mean
        );
    }

    #[test]
    fn test_reset() {
        let mut state = AdaptiveState::new();

        // Add some data
        state.add_batch_with_conversion(vec![100, 110], vec![200, 210], 1.0);
        state.update_kl(0.5);
        let posterior = make_test_posterior(
            Vector2::new(10.0, 5.0),
            Matrix2::new(4.0, 0.0, 0.0, 1.0),
            0.75,
            100,
        );
        state.update_posterior(posterior);

        assert!(state.n_total() > 0);

        // Reset
        state.reset();

        assert_eq!(state.n_total(), 0);
        assert_eq!(state.batch_count, 0);
        assert!(state.previous_posterior.is_none());
        assert!(!state.has_kl_history());
        assert!(!state.has_stats_tracking());
    }

    #[test]
    fn test_stats_not_tracked_without_conversion() {
        let mut state = AdaptiveState::new();

        // Use regular add_batch without conversion
        state.add_batch(vec![100, 110, 120], vec![200, 210, 220]);

        // Should not have stats tracking
        assert!(!state.has_stats_tracking());
        assert!(state.baseline_stats().is_none());
        assert!(state.sample_stats().is_none());
    }
}
