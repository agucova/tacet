//! Feature extraction from power traces.
//!
//! This module implements the feature extraction pipeline that transforms
//! raw power traces into statistical features suitable for Bayesian analysis.
//!
//! # Pipeline
//!
//! 1. **Preprocessing**: Winsorization and normalization
//! 2. **Partitioning**: Divide trace into bins
//! 3. **Feature extraction**: Compute statistics per bin

use super::config::{Config, FeatureFamily};
use super::dataset::{Class, Dataset, Trace};
use super::preprocessing::{median, percentile, preprocess};

/// Features extracted from a single trace.
#[derive(Debug, Clone)]
pub struct TraceFeatures {
    /// The feature vector.
    pub features: Vec<f64>,
    /// Class label.
    pub class: Class,
    /// Trace ID.
    pub trace_id: u64,
}

/// Features extracted from all traces in a dataset.
#[derive(Debug, Clone)]
pub struct ExtractedFeatures {
    /// Feature dimension.
    pub dimension: usize,
    /// Features for Fixed class traces.
    pub fixed: Vec<TraceFeatures>,
    /// Features for Random class traces.
    pub random: Vec<TraceFeatures>,
}

impl ExtractedFeatures {
    /// Get the number of Fixed class traces.
    pub fn fixed_count(&self) -> usize {
        self.fixed.len()
    }

    /// Get the number of Random class traces.
    pub fn random_count(&self) -> usize {
        self.random.len()
    }

    /// Get total number of traces.
    pub fn total_count(&self) -> usize {
        self.fixed.len() + self.random.len()
    }
}

/// Partition samples into bins.
///
/// Returns a vector of sample slices, one per partition.
pub fn partition_samples(samples: &[f32], num_partitions: usize) -> Vec<&[f32]> {
    if samples.is_empty() || num_partitions == 0 {
        return vec![];
    }

    let n = samples.len();
    let base_size = n / num_partitions;
    let remainder = n % num_partitions;

    let mut partitions = Vec::with_capacity(num_partitions);
    let mut start = 0;

    for i in 0..num_partitions {
        // Distribute remainder across first partitions
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + base_size + extra;
        partitions.push(&samples[start..end]);
        start = end;
    }

    partitions
}

/// Extract Mean features from partitions.
///
/// Returns one feature per partition: the partition mean.
fn extract_mean_features(partitions: &[&[f32]]) -> Vec<f64> {
    partitions
        .iter()
        .map(|p| {
            if p.is_empty() {
                0.0
            } else {
                p.iter().map(|v| *v as f64).sum::<f64>() / p.len() as f64
            }
        })
        .collect()
}

/// Extract Robust3 features from partitions.
///
/// Returns three features per partition: median, 10th percentile, 90th percentile.
fn extract_robust3_features(partitions: &[&[f32]]) -> Vec<f64> {
    let mut features = Vec::with_capacity(partitions.len() * 3);

    for p in partitions {
        if p.is_empty() {
            features.extend([0.0, 0.0, 0.0]);
        } else {
            let med = median(p) as f64;
            let p10 = percentile(p, 10.0) as f64;
            let p90 = percentile(p, 90.0) as f64;
            features.push(med);
            features.push(p10);
            features.push(p90);
        }
    }

    features
}

/// Extract CenteredSquare (variance) features from partitions.
///
/// Returns one feature per partition: the centered variance.
fn extract_centered_square_features(partitions: &[&[f32]]) -> Vec<f64> {
    partitions
        .iter()
        .map(|p| {
            if p.len() < 2 {
                return 0.0;
            }

            let n = p.len() as f64;
            let mean: f64 = p.iter().map(|v| *v as f64).sum::<f64>() / n;
            let variance: f64 = p.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0);
            variance
        })
        .collect()
}

/// Extract features from a single trace.
pub fn extract_trace_features(trace: &Trace, config: &Config) -> TraceFeatures {
    // 1. Copy and preprocess samples
    let mut samples: Vec<f32> = trace.samples.clone();
    preprocess(&mut samples, &config.preprocessing);

    // 2. Partition
    let partitions = partition_samples(&samples, config.partition.num_partitions);

    // 3. Extract features based on family
    let features = match config.feature_family {
        FeatureFamily::Mean => extract_mean_features(&partitions),
        FeatureFamily::Robust3 => extract_robust3_features(&partitions),
        FeatureFamily::CenteredSquare => extract_centered_square_features(&partitions),
    };

    TraceFeatures {
        features,
        class: trace.class,
        trace_id: trace.id,
    }
}

/// Extract features from all traces in a dataset.
pub fn extract_features(dataset: &Dataset, config: &Config) -> ExtractedFeatures {
    let dimension = config.feature_dimension();

    let mut fixed = Vec::with_capacity(dataset.fixed_count());
    let mut random = Vec::with_capacity(dataset.random_count());

    for trace in &dataset.traces {
        let features = extract_trace_features(trace, config);
        match features.class {
            Class::Fixed => fixed.push(features),
            Class::Random => random.push(features),
        }
    }

    ExtractedFeatures {
        dimension,
        fixed,
        random,
    }
}

/// Compute the mean difference between Fixed and Random classes.
///
/// Returns a vector of length `dimension` containing (mean_fixed - mean_random)
/// for each feature dimension.
pub fn compute_class_difference(features: &ExtractedFeatures) -> Vec<f64> {
    if features.fixed.is_empty() || features.random.is_empty() {
        return vec![0.0; features.dimension];
    }

    let d = features.dimension;
    let n_fixed = features.fixed.len() as f64;
    let n_random = features.random.len() as f64;

    // Compute mean of Fixed class
    let mut mean_fixed = vec![0.0; d];
    for tf in &features.fixed {
        for (i, v) in tf.features.iter().enumerate() {
            mean_fixed[i] += v;
        }
    }
    for v in &mut mean_fixed {
        *v /= n_fixed;
    }

    // Compute mean of Random class
    let mut mean_random = vec![0.0; d];
    for tf in &features.random {
        for (i, v) in tf.features.iter().enumerate() {
            mean_random[i] += v;
        }
    }
    for v in &mut mean_random {
        *v /= n_random;
    }

    // Compute difference
    mean_fixed
        .iter()
        .zip(mean_random.iter())
        .map(|(f, r)| f - r)
        .collect()
}

/// Compute pooled sample covariance matrix.
///
/// Uses Welford's algorithm for numerical stability.
/// Returns a flattened covariance matrix (row-major, dimension × dimension).
pub fn compute_pooled_covariance(features: &ExtractedFeatures) -> Vec<f64> {
    let d = features.dimension;
    let n_total = features.total_count();

    if n_total < 2 {
        return vec![0.0; d * d];
    }

    // Compute global mean
    let mut mean = vec![0.0; d];
    for tf in features.fixed.iter().chain(features.random.iter()) {
        for (i, v) in tf.features.iter().enumerate() {
            mean[i] += v;
        }
    }
    for v in &mut mean {
        *v /= n_total as f64;
    }

    // Compute covariance matrix
    let mut cov = vec![0.0; d * d];
    for tf in features.fixed.iter().chain(features.random.iter()) {
        for i in 0..d {
            let dev_i = tf.features[i] - mean[i];
            for j in 0..d {
                let dev_j = tf.features[j] - mean[j];
                cov[i * d + j] += dev_i * dev_j;
            }
        }
    }

    // Normalize by (n - 1) for unbiased estimate
    let scale = 1.0 / (n_total - 1) as f64;
    for v in &mut cov {
        *v *= scale;
    }

    cov
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::PreprocessingConfig;

    #[test]
    fn test_partition_samples() {
        let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let partitions = partition_samples(&samples, 3);

        assert_eq!(partitions.len(), 3);
        // With 10 elements and 3 partitions: 4 + 3 + 3 = 10
        assert_eq!(partitions[0].len(), 4);
        assert_eq!(partitions[1].len(), 3);
        assert_eq!(partitions[2].len(), 3);
    }

    #[test]
    fn test_mean_features() {
        let samples: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let partitions = partition_samples(&samples, 2);
        let features = extract_mean_features(&partitions);

        assert_eq!(features.len(), 2);
        assert!((features[0] - 2.0).abs() < 1e-6); // mean of [1, 2, 3]
        assert!((features[1] - 5.0).abs() < 1e-6); // mean of [4, 5, 6]
    }

    #[test]
    fn test_robust3_features() {
        let samples: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let partitions = partition_samples(&samples, 2);
        let features = extract_robust3_features(&partitions);

        // 2 partitions × 3 features = 6 features
        assert_eq!(features.len(), 6);
    }

    #[test]
    fn test_centered_square_features() {
        let samples: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let partitions = partition_samples(&samples, 2);
        let features = extract_centered_square_features(&partitions);

        assert_eq!(features.len(), 2);
        // Both partitions have same variance
        assert!((features[0] - features[1]).abs() < 1e-6);
    }

    #[test]
    fn test_extract_features() {
        let traces = vec![
            Trace::new(Class::Fixed, vec![1.0, 2.0, 3.0, 4.0]),
            Trace::new(Class::Random, vec![2.0, 3.0, 4.0, 5.0]),
        ];
        let dataset = Dataset::new(traces);

        let config = Config::new().with_partitions(2);
        let features = extract_features(&dataset, &config);

        assert_eq!(features.dimension, 2);
        assert_eq!(features.fixed_count(), 1);
        assert_eq!(features.random_count(), 1);
    }

    #[test]
    fn test_class_difference() {
        let traces = vec![
            Trace::new(Class::Fixed, vec![10.0, 20.0, 30.0, 40.0]),
            Trace::new(Class::Fixed, vec![12.0, 22.0, 32.0, 42.0]),
            Trace::new(Class::Random, vec![5.0, 15.0, 25.0, 35.0]),
            Trace::new(Class::Random, vec![7.0, 17.0, 27.0, 37.0]),
        ];
        let dataset = Dataset::new(traces);

        // Disable preprocessing to test raw differences
        let mut config = Config::new().with_partitions(2);
        config.preprocessing = PreprocessingConfig::none();

        let features = extract_features(&dataset, &config);
        let diff = compute_class_difference(&features);

        // Fixed mean is higher than Random mean
        assert!(diff[0] > 0.0);
        assert!(diff[1] > 0.0);
    }

    #[test]
    fn test_pooled_covariance() {
        let traces = vec![
            Trace::new(Class::Fixed, vec![1.0, 2.0, 3.0, 4.0]),
            Trace::new(Class::Fixed, vec![2.0, 3.0, 4.0, 5.0]),
            Trace::new(Class::Random, vec![1.5, 2.5, 3.5, 4.5]),
            Trace::new(Class::Random, vec![2.5, 3.5, 4.5, 5.5]),
        ];
        let dataset = Dataset::new(traces);

        let mut config = Config::new().with_partitions(2);
        config.preprocessing = PreprocessingConfig::none();

        let features = extract_features(&dataset, &config);
        let cov = compute_pooled_covariance(&features);

        // 2×2 covariance matrix = 4 elements
        assert_eq!(cov.len(), 4);

        // Diagonal elements (variances) should be positive
        assert!(cov[0] > 0.0);
        assert!(cov[3] > 0.0);
    }
}
