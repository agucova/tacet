//! Trace preprocessing utilities.
//!
//! This module provides preprocessing operations for power traces,
//! including outlier removal (winsorization) and normalization.

use super::config::PreprocessingConfig;

/// Compute the percentile of a sorted slice.
fn percentile_sorted(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let p = p.clamp(0.0, 100.0);
    let n = sorted.len() as f64;
    let idx = (p / 100.0) * (n - 1.0);
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if lower >= sorted.len() {
        sorted[sorted.len() - 1]
    } else if upper >= sorted.len() {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac as f32) + sorted[upper] * frac as f32
    }
}

/// Compute the percentile of a slice.
///
/// This creates a sorted copy of the input.
pub fn percentile(values: &[f32], p: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_sorted(&sorted, p)
}

/// Compute the median of a slice.
pub fn median(values: &[f32]) -> f32 {
    percentile(values, 50.0)
}

/// Apply winsorization to clamp outliers.
///
/// Values below the `lower_percentile` are clamped to that percentile's value.
/// Values above the `upper_percentile` are clamped to that percentile's value.
///
/// # Arguments
///
/// * `values` - Mutable slice of values to winsorize
/// * `lower_percentile` - Lower percentile (0-100)
/// * `upper_percentile` - Upper percentile (0-100)
pub fn winsorize(values: &mut [f32], lower_percentile: f64, upper_percentile: f64) {
    if values.is_empty() {
        return;
    }

    // Compute percentile bounds from original data
    let lower_bound = percentile(values, lower_percentile);
    let upper_bound = percentile(values, upper_percentile);

    // Clamp values
    for v in values.iter_mut() {
        if *v < lower_bound {
            *v = lower_bound;
        } else if *v > upper_bound {
            *v = upper_bound;
        }
    }
}

/// Normalize a slice to zero mean.
///
/// Subtracts the mean from all values.
pub fn normalize_mean(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }

    let sum: f64 = values.iter().map(|v| *v as f64).sum();
    let mean = (sum / values.len() as f64) as f32;

    for v in values.iter_mut() {
        *v -= mean;
    }
}

/// Normalize a slice to unit variance.
///
/// Divides by the standard deviation. Assumes zero mean or uses actual mean.
pub fn normalize_variance(values: &mut [f32]) {
    if values.len() < 2 {
        return;
    }

    let n = values.len() as f64;
    let sum: f64 = values.iter().map(|v| *v as f64).sum();
    let mean = sum / n;

    let variance: f64 = values.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt() as f32;

    if std_dev > 1e-10 {
        for v in values.iter_mut() {
            *v /= std_dev;
        }
    }
}

/// Apply all preprocessing steps according to config.
///
/// # Arguments
///
/// * `samples` - Mutable slice of samples to preprocess
/// * `config` - Preprocessing configuration
pub fn preprocess(samples: &mut [f32], config: &PreprocessingConfig) {
    // 1. Winsorization (outlier removal)
    if config.winsorize_lower > 0.0 || config.winsorize_upper < 100.0 {
        winsorize(samples, config.winsorize_lower, config.winsorize_upper);
    }

    // 2. Mean normalization
    if config.normalize_mean {
        normalize_mean(samples);
    }

    // 3. Variance normalization
    if config.normalize_variance {
        normalize_variance(samples);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&values, 0.0) - 1.0).abs() < 1e-6);
        assert!((percentile(&values, 50.0) - 3.0).abs() < 1e-6);
        assert!((percentile(&values, 100.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_median() {
        assert!((median(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-6);
        assert!((median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_winsorize() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // 100 is an outlier
        winsorize(&mut values, 0.0, 80.0);
        // The 80th percentile should clamp the outlier
        assert!(values[4] < 100.0);
        assert!(values[4] >= 4.0);
    }

    #[test]
    fn test_normalize_mean() {
        let mut values = vec![10.0, 20.0, 30.0];
        normalize_mean(&mut values);

        let sum: f32 = values.iter().sum();
        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn test_normalize_variance() {
        let mut values = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        normalize_mean(&mut values);
        normalize_variance(&mut values);

        let n = values.len() as f64;
        let variance: f64 = values.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / (n - 1.0);
        assert!((variance - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_preprocess_full() {
        let mut values = vec![0.0, 10.0, 20.0, 30.0, 1000.0]; // 1000 is outlier
        let config = PreprocessingConfig {
            winsorize_lower: 0.0,
            winsorize_upper: 90.0,
            normalize_mean: true,
            normalize_variance: true,
        };

        preprocess(&mut values, &config);

        // Check mean is ~0
        let sum: f32 = values.iter().sum();
        assert!(sum.abs() < 1e-3);

        // Check variance is ~1
        let n = values.len() as f64;
        let variance: f64 = values.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / (n - 1.0);
        assert!((variance - 1.0).abs() < 0.2);
    }
}
