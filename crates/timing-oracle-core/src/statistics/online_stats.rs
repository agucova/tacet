//! Online (streaming) statistics computation using Welford's algorithm.
//!
//! This module provides incremental computation of mean, variance, and lag-1
//! autocorrelation with O(1) memory and O(1) per-sample overhead.
//!
//! Used for condition drift detection (spec Section 2.6, Gate 6) to compare
//! measurement statistics from calibration vs the full test run.

/// Online statistics accumulator using Welford's algorithm.
///
/// Tracks mean, variance, and lag-1 autocorrelation incrementally with
/// O(1) overhead per sample. This is used to detect condition drift
/// between calibration and the adaptive loop.
///
/// # Example
///
/// ```
/// use timing_oracle_core::statistics::OnlineStats;
///
/// let mut stats = OnlineStats::new();
/// for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
///     stats.update(x);
/// }
/// let snapshot = stats.finalize();
/// assert!((snapshot.mean - 3.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct OnlineStats {
    /// Number of samples seen.
    count: usize,
    /// Running mean.
    mean: f64,
    /// Welford's M2: sum of squared deviations from current mean.
    /// Variance = m2 / (count - 1) for sample variance.
    m2: f64,
    /// Previous value (for lag-1 autocorrelation).
    prev_value: f64,
    /// Previous mean (for lag-1 autocorrelation with current mean estimate).
    prev_mean: f64,
    /// Sum of (x_i - mean)(x_{i-1} - mean) products for autocorrelation.
    autocorr_sum: f64,
    /// Number of pairs for autocorrelation (count - 1).
    autocorr_count: usize,
}

impl Default for OnlineStats {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineStats {
    /// Create a new empty statistics accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            prev_value: 0.0,
            prev_mean: 0.0,
            autocorr_sum: 0.0,
            autocorr_count: 0,
        }
    }

    /// Update statistics with a new sample.
    ///
    /// Uses Welford's online algorithm for numerically stable variance computation.
    /// Also tracks lag-1 autocorrelation incrementally.
    ///
    /// # Arguments
    ///
    /// * `x` - The new sample value
    pub fn update(&mut self, x: f64) {
        // Welford's online algorithm for mean and variance
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        // Lag-1 autocorrelation: accumulate (x_i - μ)(x_{i-1} - μ)
        // We use the current mean estimate for both terms, which introduces
        // a small bias but is acceptable for drift detection purposes.
        if self.count > 1 {
            // Use deviation from current mean for both current and previous value
            let dev_curr = x - self.mean;
            let dev_prev = self.prev_value - self.mean;
            self.autocorr_sum += dev_curr * dev_prev;
            self.autocorr_count += 1;
        }

        self.prev_value = x;
        self.prev_mean = self.mean;
    }

    /// Finalize and return the computed statistics.
    ///
    /// # Returns
    ///
    /// A `StatsSnapshot` containing mean, variance, and lag-1 autocorrelation.
    /// Returns default values (0.0) if insufficient samples.
    pub fn finalize(&self) -> StatsSnapshot {
        if self.count < 2 {
            return StatsSnapshot {
                mean: self.mean,
                variance: 0.0,
                autocorr_lag1: 0.0,
                count: self.count,
            };
        }

        let variance = self.m2 / (self.count - 1) as f64;

        // Autocorrelation coefficient: cov(x_i, x_{i-1}) / var(x)
        let autocorr_lag1 = if self.autocorr_count > 0 && variance > 1e-15 {
            let autocovariance = self.autocorr_sum / self.autocorr_count as f64;
            (autocovariance / variance).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        StatsSnapshot {
            mean: self.mean,
            variance,
            autocorr_lag1,
            count: self.count,
        }
    }

    /// Get the current sample count.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the current mean estimate.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the current variance estimate (returns 0 if count < 2).
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }
}

/// Snapshot of computed statistics at a point in time.
///
/// This is returned by `OnlineStats::finalize()` and used for
/// comparing calibration vs post-test statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StatsSnapshot {
    /// Sample mean.
    pub mean: f64,
    /// Sample variance (using n-1 denominator).
    pub variance: f64,
    /// Lag-1 autocorrelation coefficient, in range [-1, 1].
    pub autocorr_lag1: f64,
    /// Number of samples.
    pub count: usize,
}

impl StatsSnapshot {
    /// Get the standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_stats_basic() {
        let mut stats = OnlineStats::new();
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];

        for &x in &data {
            stats.update(x);
        }

        let snapshot = stats.finalize();

        // Mean should be 3.0
        assert!(
            (snapshot.mean - 3.0).abs() < 1e-10,
            "Expected mean=3.0, got {}",
            snapshot.mean
        );

        // Variance should be 2.5 (sample variance of [1,2,3,4,5])
        assert!(
            (snapshot.variance - 2.5).abs() < 1e-10,
            "Expected variance=2.5, got {}",
            snapshot.variance
        );

        assert_eq!(snapshot.count, 5);
    }

    #[test]
    fn test_online_stats_single_value() {
        let mut stats = OnlineStats::new();
        stats.update(42.0);

        let snapshot = stats.finalize();

        assert!((snapshot.mean - 42.0).abs() < 1e-10);
        assert!((snapshot.variance - 0.0).abs() < 1e-10);
        assert_eq!(snapshot.count, 1);
    }

    #[test]
    fn test_online_stats_empty() {
        let stats = OnlineStats::new();
        let snapshot = stats.finalize();

        assert!((snapshot.mean - 0.0).abs() < 1e-10);
        assert!((snapshot.variance - 0.0).abs() < 1e-10);
        assert_eq!(snapshot.count, 0);
    }

    #[test]
    fn test_online_stats_constant_values() {
        let mut stats = OnlineStats::new();
        for _ in 0..100 {
            stats.update(5.0);
        }

        let snapshot = stats.finalize();

        assert!((snapshot.mean - 5.0).abs() < 1e-10);
        assert!(snapshot.variance < 1e-10, "Constant values should have ~0 variance");
        assert_eq!(snapshot.count, 100);
    }

    #[test]
    fn test_online_stats_matches_batch() {
        // Compare online computation with batch computation
        let data: Vec<f64> = (0..1000).map(|i| (i as f64).sin() * 100.0).collect();

        // Online computation
        let mut stats = OnlineStats::new();
        for &x in &data {
            stats.update(x);
        }
        let online = stats.finalize();

        // Batch computation
        let n = data.len() as f64;
        let batch_mean: f64 = data.iter().sum::<f64>() / n;
        let batch_variance: f64 =
            data.iter().map(|x| (x - batch_mean).powi(2)).sum::<f64>() / (n - 1.0);

        assert!(
            (online.mean - batch_mean).abs() < 1e-10,
            "Mean mismatch: online={}, batch={}",
            online.mean,
            batch_mean
        );
        assert!(
            (online.variance - batch_variance).abs() < 1e-6,
            "Variance mismatch: online={}, batch={}",
            online.variance,
            batch_variance
        );
    }

    #[test]
    fn test_online_stats_autocorr_positive() {
        // Highly autocorrelated data: each value close to previous
        let mut stats = OnlineStats::new();
        let mut x = 0.0;
        for _ in 0..1000 {
            x += 0.1; // Monotonically increasing
            stats.update(x);
        }

        let snapshot = stats.finalize();

        // Strong positive autocorrelation expected
        assert!(
            snapshot.autocorr_lag1 > 0.9,
            "Expected high positive autocorrelation, got {}",
            snapshot.autocorr_lag1
        );
    }

    #[test]
    fn test_online_stats_autocorr_negative() {
        // Alternating values: strong negative autocorrelation
        let mut stats = OnlineStats::new();
        for i in 0..1000 {
            let x = if i % 2 == 0 { 100.0 } else { -100.0 };
            stats.update(x);
        }

        let snapshot = stats.finalize();

        // Strong negative autocorrelation expected
        assert!(
            snapshot.autocorr_lag1 < -0.9,
            "Expected high negative autocorrelation, got {}",
            snapshot.autocorr_lag1
        );
    }

    #[test]
    fn test_online_stats_autocorr_near_zero() {
        // Random-ish data: low autocorrelation
        let mut stats = OnlineStats::new();
        let mut state: u64 = 12345;
        for _ in 0..1000 {
            // Pseudo-random using simple LCG with wrapping arithmetic
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let x = (state % 1000) as f64;
            stats.update(x);
        }

        let snapshot = stats.finalize();

        // Near-zero autocorrelation expected for pseudo-random data
        assert!(
            snapshot.autocorr_lag1.abs() < 0.1,
            "Expected near-zero autocorrelation, got {}",
            snapshot.autocorr_lag1
        );
    }

    #[test]
    fn test_stats_snapshot_std_dev() {
        let snapshot = StatsSnapshot {
            mean: 5.0,
            variance: 4.0,
            autocorr_lag1: 0.0,
            count: 100,
        };

        assert!((snapshot.std_dev() - 2.0).abs() < 1e-10);
    }
}
