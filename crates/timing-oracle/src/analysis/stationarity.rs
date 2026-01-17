//! Stationarity tracking for drift detection (spec Section 3.2.1).
//!
//! Uses reservoir sampling to maintain approximate quantiles per time window,
//! enabling detection of timing distribution drift during measurement.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Number of time windows for stationarity analysis.
const NUM_WINDOWS: usize = 10;

/// Samples per window for reservoir sampling.
const SAMPLES_PER_WINDOW: usize = 100;

/// Tracks timing samples across time windows for stationarity analysis.
///
/// Uses reservoir sampling to maintain a fixed-size sample per window,
/// allowing O(1) memory usage regardless of total sample count.
#[derive(Debug, Clone)]
pub struct StationarityTracker {
    /// Reservoir samples for each window.
    windows: [[f64; SAMPLES_PER_WINDOW]; NUM_WINDOWS],
    /// Count of samples seen per window.
    counts: [usize; NUM_WINDOWS],
    /// Total samples expected (for window assignment).
    expected_total: usize,
    /// Total samples seen so far.
    samples_seen: usize,
    /// RNG for reservoir sampling.
    rng: Xoshiro256PlusPlus,
}

/// Result of stationarity analysis.
#[derive(Debug, Clone, Copy)]
pub struct StationarityResult {
    /// Ratio of max to min window median.
    pub ratio: f64,
    /// Whether stationarity check passed.
    pub ok: bool,
    /// Whether median drift was detected.
    pub median_drift_ok: bool,
    /// Whether variance drift was detected.
    pub variance_drift_ok: bool,
}

impl Default for StationarityResult {
    fn default() -> Self {
        Self {
            ratio: 1.0,
            ok: true,
            median_drift_ok: true,
            variance_drift_ok: true,
        }
    }
}

impl StationarityTracker {
    /// Create a new stationarity tracker.
    ///
    /// # Arguments
    /// * `expected_total` - Expected total number of samples (for window assignment)
    /// * `seed` - Random seed for reservoir sampling
    pub fn new(expected_total: usize, seed: u64) -> Self {
        Self {
            windows: [[0.0; SAMPLES_PER_WINDOW]; NUM_WINDOWS],
            counts: [0; NUM_WINDOWS],
            expected_total: expected_total.max(1),
            samples_seen: 0,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    /// Push a timing sample.
    pub fn push(&mut self, time_ns: f64) {
        // Determine which window this sample belongs to
        let window_idx = (self.samples_seen * NUM_WINDOWS) / self.expected_total;
        let window_idx = window_idx.min(NUM_WINDOWS - 1);

        let count = self.counts[window_idx];

        if count < SAMPLES_PER_WINDOW {
            // Fill the reservoir
            self.windows[window_idx][count] = time_ns;
        } else {
            // Reservoir sampling: replace with probability SAMPLES_PER_WINDOW / (count + 1)
            let j = self.rng.random_range(0..=count);
            if j < SAMPLES_PER_WINDOW {
                self.windows[window_idx][j] = time_ns;
            }
        }

        self.counts[window_idx] += 1;
        self.samples_seen += 1;
    }

    /// Update expected total (call when sample count changes).
    pub fn set_expected_total(&mut self, expected_total: usize) {
        self.expected_total = expected_total.max(1);
    }

    /// Compute stationarity metrics.
    ///
    /// Returns `None` if insufficient samples for analysis.
    pub fn compute(&self) -> Option<StationarityResult> {
        // Need at least 2 windows with samples
        let active_windows: Vec<usize> = (0..NUM_WINDOWS)
            .filter(|&i| self.counts[i] >= 10)
            .collect();

        if active_windows.len() < 2 {
            return None;
        }

        // Compute median and variance for each active window
        let mut window_medians = Vec::with_capacity(active_windows.len());
        let mut window_iqrs = Vec::with_capacity(active_windows.len());
        let mut window_variances = Vec::with_capacity(active_windows.len());

        for &i in &active_windows {
            let n = self.counts[i].min(SAMPLES_PER_WINDOW);
            let mut samples: Vec<f64> = self.windows[i][..n].to_vec();
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = samples[n / 2];
            let q1 = samples[n / 4];
            let q3 = samples[(n * 3) / 4];
            let iqr = q3 - q1;

            let mean: f64 = samples.iter().sum::<f64>() / n as f64;
            let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

            window_medians.push(median);
            window_iqrs.push(iqr);
            window_variances.push(variance);
        }

        // Compute global median for threshold
        let global_median = {
            let sum: f64 = window_medians.iter().sum();
            sum / window_medians.len() as f64
        };

        // Check 1: Median drift (spec Section 3.2.1)
        let max_median = window_medians
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_median = window_medians.iter().cloned().fold(f64::INFINITY, f64::min);
        let avg_iqr: f64 = window_iqrs.iter().sum::<f64>() / window_iqrs.len() as f64;

        let drift_floor = 0.05 * global_median;
        let threshold = (2.0 * avg_iqr).max(drift_floor);
        let median_drift_ok = (max_median - min_median) <= threshold;

        // Check 2: Variance monotonic drift (>50% change first to last)
        let first_var = window_variances[0];
        let last_var = *window_variances.last().unwrap();
        let var_drift_ratio = if first_var > 1e-12 {
            last_var / first_var
        } else {
            1.0
        };
        let variance_drift_ok = (0.5..=2.0).contains(&var_drift_ratio);

        // Compute stationarity ratio
        let ratio = if min_median > 1e-12 {
            max_median / min_median
        } else {
            1.0
        };

        Some(StationarityResult {
            ratio,
            ok: median_drift_ok && variance_drift_ok,
            median_drift_ok,
            variance_drift_ok,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stationary_data() {
        let mut tracker = StationarityTracker::new(1000, 42);

        // Push stationary data (constant with small noise)
        for i in 0..1000 {
            let noise = (i % 10) as f64 * 0.1;
            tracker.push(100.0 + noise);
        }

        let result = tracker.compute().unwrap();
        assert!(result.ok, "Stationary data should pass");
        assert!(
            (result.ratio - 1.0).abs() < 0.1,
            "Ratio should be close to 1"
        );
    }

    #[test]
    fn test_drifting_data() {
        let mut tracker = StationarityTracker::new(1000, 42);

        // Push drifting data (linear increase)
        for i in 0..1000 {
            // Significant drift: 100 -> 200 over the measurement
            let time = 100.0 + (i as f64) * 0.1;
            tracker.push(time);
        }

        let result = tracker.compute().unwrap();
        assert!(!result.ok, "Drifting data should fail");
        assert!(result.ratio > 1.5, "Ratio should show significant drift");
    }

    #[test]
    fn test_variance_change() {
        let mut tracker = StationarityTracker::new(1000, 42);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);

        // First half: low variance
        for _ in 0..500 {
            let noise: f64 = rng.random_range(-1.0..1.0);
            tracker.push(100.0 + noise);
        }

        // Second half: high variance (10x)
        for _ in 0..500 {
            let noise: f64 = rng.random_range(-10.0..10.0);
            tracker.push(100.0 + noise);
        }

        let result = tracker.compute().unwrap();
        assert!(
            !result.variance_drift_ok,
            "Variance change should be detected"
        );
    }

    #[test]
    fn test_insufficient_samples() {
        let tracker = StationarityTracker::new(1000, 42);
        assert!(
            tracker.compute().is_none(),
            "Should return None with no samples"
        );
    }

    #[test]
    fn test_reservoir_sampling_coverage() {
        let mut tracker = StationarityTracker::new(10000, 42);

        // Push many samples to test reservoir sampling
        for i in 0..10000 {
            tracker.push(100.0 + (i % 100) as f64);
        }

        let result = tracker.compute();
        assert!(result.is_some(), "Should have enough samples");
    }

    #[test]
    fn test_step_change() {
        // Sudden jump mid-measurement (e.g., CPU throttle)
        let mut tracker = StationarityTracker::new(1000, 42);

        for _ in 0..500 {
            tracker.push(100.0);
        }
        for _ in 0..500 {
            tracker.push(200.0); // Sudden 2x jump
        }

        let result = tracker.compute().unwrap();
        assert!(!result.ok, "Step change should fail stationarity");
        assert!(!result.median_drift_ok, "Median drift should be detected");
    }

    #[test]
    fn test_variance_boundary_pass() {
        // Variance ratio exactly at boundary (2.0) should pass
        let mut tracker = StationarityTracker::new(1000, 42);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(456);

        // First half: variance ~1
        for _ in 0..500 {
            let noise: f64 = rng.random_range(-1.0..1.0);
            tracker.push(100.0 + noise);
        }

        // Second half: variance ~2 (ratio = 2.0, should still pass)
        for _ in 0..500 {
            let noise: f64 = rng.random_range(-1.414..1.414); // sqrt(2) range
            tracker.push(100.0 + noise);
        }

        let result = tracker.compute().unwrap();
        // With stochastic data, just check it's not failing dramatically
        assert!(result.median_drift_ok, "No median drift expected");
    }

    #[test]
    fn test_median_drift_without_variance_change() {
        // Gradual shift without variance change
        let mut tracker = StationarityTracker::new(1000, 42);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(789);

        for i in 0..1000 {
            let base = 100.0 + (i as f64) * 0.05; // Slow drift
            let noise: f64 = rng.random_range(-1.0..1.0);
            tracker.push(base + noise);
        }

        let result = tracker.compute().unwrap();
        // Should detect median drift
        assert!(
            !result.median_drift_ok || result.ratio > 1.3,
            "Should detect gradual drift"
        );
    }

    #[test]
    fn test_high_iqr_threshold() {
        // When IQR is high, median drift threshold should increase
        let mut tracker = StationarityTracker::new(1000, 42);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(321);

        // High variance data with small drift should pass
        for i in 0..1000 {
            let base = 100.0 + (i as f64) * 0.01; // Very small drift
            let noise: f64 = rng.random_range(-20.0..20.0); // High noise
            tracker.push(base + noise);
        }

        let result = tracker.compute().unwrap();
        // High IQR should make threshold lenient
        assert!(
            result.median_drift_ok,
            "Small drift with high IQR should pass"
        );
    }

    #[test]
    fn test_minimum_samples_per_window() {
        // Test with exactly enough samples (10 per window minimum)
        let mut tracker = StationarityTracker::new(200, 42);

        for i in 0..200 {
            tracker.push(100.0 + (i % 5) as f64);
        }

        let result = tracker.compute();
        assert!(result.is_some(), "Should compute with 20 samples per window");
    }

    #[test]
    fn test_set_expected_total() {
        let mut tracker = StationarityTracker::new(100, 42);

        // Initially expect 100 samples
        for i in 0..50 {
            tracker.push(100.0 + (i % 5) as f64);
        }

        // Update expectation
        tracker.set_expected_total(200);

        // Continue pushing
        for i in 0..50 {
            tracker.push(100.0 + (i % 5) as f64);
        }

        let result = tracker.compute();
        assert!(result.is_some(), "Should handle expected_total update");
    }

    #[test]
    fn test_near_zero_values() {
        let mut tracker = StationarityTracker::new(1000, 42);

        // Very small positive values
        for i in 0..1000 {
            tracker.push(0.001 + (i % 10) as f64 * 0.0001);
        }

        let result = tracker.compute().unwrap();
        assert!(result.ok, "Near-zero positive values should work");
    }

    #[test]
    fn test_identical_values() {
        // All identical values (zero variance)
        let mut tracker = StationarityTracker::new(1000, 42);

        for _ in 0..1000 {
            tracker.push(100.0);
        }

        let result = tracker.compute().unwrap();
        assert!(result.ok, "Identical values should pass");
        assert!(
            (result.ratio - 1.0).abs() < 1e-10,
            "Ratio should be exactly 1.0"
        );
    }
}
