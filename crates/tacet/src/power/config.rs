//! Configuration types for power analysis.

use serde::{Deserialize, Serialize};

/// Feature extraction family.
///
/// Determines which statistical features are extracted from each partition
/// of the power traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FeatureFamily {
    /// Per-partition mean (standard TVLA).
    ///
    /// Extracts `d = num_partitions` features per trace.
    /// This is the simplest and most commonly used approach.
    #[default]
    Mean,

    /// Robust statistics: median, 10th, and 90th percentiles.
    ///
    /// Extracts `d = 3 × num_partitions` features per trace.
    /// More robust to outliers than mean-based analysis.
    Robust3,

    /// Centered variance per partition.
    ///
    /// Extracts `d = num_partitions` features per trace.
    /// Detects second-order (variance) leakage that mean-based
    /// analysis would miss.
    CenteredSquare,
}

impl std::fmt::Display for FeatureFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureFamily::Mean => write!(f, "Mean"),
            FeatureFamily::Robust3 => write!(f, "Robust3 (median, P10, P90)"),
            FeatureFamily::CenteredSquare => write!(f, "CenteredSquare (variance)"),
        }
    }
}

/// Configuration for trace partitioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Number of partitions (bins) to divide each trace into.
    ///
    /// Default: 32
    ///
    /// Higher values provide finer granularity for localization but
    /// increase the feature dimension and may reduce statistical power.
    pub num_partitions: usize,

    /// Whether partitions should overlap.
    ///
    /// Default: false
    ///
    /// Overlapping partitions can improve sensitivity at boundaries
    /// but increase feature dimension and correlation.
    pub overlap: bool,

    /// Overlap fraction (0.0 to 0.5).
    ///
    /// Only used if `overlap` is true. A value of 0.25 means each
    /// partition overlaps 25% with its neighbors.
    pub overlap_fraction: f64,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            num_partitions: 32,
            overlap: false,
            overlap_fraction: 0.25,
        }
    }
}

impl PartitionConfig {
    /// Create a new partition config with the given number of partitions.
    pub fn new(num_partitions: usize) -> Self {
        Self {
            num_partitions,
            ..Default::default()
        }
    }

    /// Enable overlapping partitions with the given overlap fraction.
    pub fn with_overlap(mut self, fraction: f64) -> Self {
        self.overlap = true;
        self.overlap_fraction = fraction.clamp(0.0, 0.5);
        self
    }
}

/// Configuration for trace preprocessing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Lower percentile for winsorization (outlier removal).
    ///
    /// Default: 0.01 (0.01st percentile)
    ///
    /// Values below this percentile are clamped to the percentile value.
    pub winsorize_lower: f64,

    /// Upper percentile for winsorization (outlier removal).
    ///
    /// Default: 99.99 (99.99th percentile)
    ///
    /// Values above this percentile are clamped to the percentile value.
    pub winsorize_upper: f64,

    /// Normalize traces to zero mean.
    ///
    /// Default: true
    ///
    /// Removes DC offset, which is typically not informative.
    pub normalize_mean: bool,

    /// Normalize traces to unit variance.
    ///
    /// Default: false
    ///
    /// Can help with traces from different measurement setups but may
    /// obscure magnitude information.
    pub normalize_variance: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            winsorize_lower: 0.01,
            winsorize_upper: 99.99,
            normalize_mean: true,
            normalize_variance: false,
        }
    }
}

impl PreprocessingConfig {
    /// Disable all preprocessing (raw samples).
    pub fn none() -> Self {
        Self {
            winsorize_lower: 0.0,
            winsorize_upper: 100.0,
            normalize_mean: false,
            normalize_variance: false,
        }
    }

    /// Standard preprocessing with mean normalization.
    pub fn standard() -> Self {
        Self::default()
    }

    /// Full normalization (zero mean, unit variance).
    pub fn full() -> Self {
        Self {
            normalize_variance: true,
            ..Default::default()
        }
    }
}

/// Main configuration for power analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Feature extraction family.
    pub feature_family: FeatureFamily,

    /// Partition configuration.
    pub partition: PartitionConfig,

    /// Preprocessing configuration.
    pub preprocessing: PreprocessingConfig,

    /// Multiplier for noise floor threshold.
    ///
    /// Default: 5.0
    ///
    /// The effective threshold is `floor_multiplier × θ_floor` where
    /// θ_floor is estimated from calibration noise.
    pub floor_multiplier: f64,

    /// Whether to analyze stages independently.
    ///
    /// Default: true
    ///
    /// If traces have markers, each stage is analyzed separately.
    /// If false, the entire trace is analyzed as one segment.
    pub stage_wise: bool,

    /// Random seed for reproducibility.
    ///
    /// Default: None (random)
    pub seed: Option<u64>,

    /// Number of bootstrap iterations for covariance estimation.
    ///
    /// Default: 2000
    pub bootstrap_iterations: usize,

    /// Pass threshold for posterior probability.
    ///
    /// Default: 0.05
    ///
    /// If P(leak > θ | data) < pass_threshold, the test passes.
    pub pass_threshold: f64,

    /// Fail threshold for posterior probability.
    ///
    /// Default: 0.95
    ///
    /// If P(leak > θ | data) > fail_threshold, the test fails.
    pub fail_threshold: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            feature_family: FeatureFamily::default(),
            partition: PartitionConfig::default(),
            preprocessing: PreprocessingConfig::default(),
            floor_multiplier: 5.0,
            stage_wise: true,
            seed: None,
            bootstrap_iterations: 2000,
            pass_threshold: 0.05,
            fail_threshold: 0.95,
        }
    }
}

impl Config {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the feature family.
    pub fn with_feature_family(mut self, family: FeatureFamily) -> Self {
        self.feature_family = family;
        self
    }

    /// Set the number of partitions.
    pub fn with_partitions(mut self, num_partitions: usize) -> Self {
        self.partition.num_partitions = num_partitions;
        self
    }

    /// Set the floor multiplier.
    pub fn with_floor_multiplier(mut self, multiplier: f64) -> Self {
        self.floor_multiplier = multiplier;
        self
    }

    /// Disable stage-wise analysis.
    pub fn without_stages(mut self) -> Self {
        self.stage_wise = false;
        self
    }

    /// Set a random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get the feature dimension for this configuration.
    ///
    /// Returns the number of features extracted per trace.
    pub fn feature_dimension(&self) -> usize {
        match self.feature_family {
            FeatureFamily::Mean => self.partition.num_partitions,
            FeatureFamily::Robust3 => 3 * self.partition.num_partitions,
            FeatureFamily::CenteredSquare => self.partition.num_partitions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_dimension() {
        let mut config = Config::default();
        config.partition.num_partitions = 32;

        config.feature_family = FeatureFamily::Mean;
        assert_eq!(config.feature_dimension(), 32);

        config.feature_family = FeatureFamily::Robust3;
        assert_eq!(config.feature_dimension(), 96);

        config.feature_family = FeatureFamily::CenteredSquare;
        assert_eq!(config.feature_dimension(), 32);
    }

    #[test]
    fn test_config_builder() {
        let config = Config::new()
            .with_feature_family(FeatureFamily::Robust3)
            .with_partitions(64)
            .with_floor_multiplier(3.0)
            .with_seed(42);

        assert_eq!(config.feature_family, FeatureFamily::Robust3);
        assert_eq!(config.partition.num_partitions, 64);
        assert_eq!(config.floor_multiplier, 3.0);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_preprocessing_presets() {
        let none = PreprocessingConfig::none();
        assert!(!none.normalize_mean);
        assert!(!none.normalize_variance);

        let standard = PreprocessingConfig::standard();
        assert!(standard.normalize_mean);
        assert!(!standard.normalize_variance);

        let full = PreprocessingConfig::full();
        assert!(full.normalize_mean);
        assert!(full.normalize_variance);
    }
}
