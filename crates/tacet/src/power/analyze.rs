//! Main power analysis function.
//!
//! This module provides the entry point for power side-channel analysis,
//! connecting the feature extraction pipeline to the Bayesian inference engine.

use super::config::Config;
use super::dataset::Dataset;
use super::features::{compute_class_difference, compute_pooled_covariance, extract_features};
use super::report::{DimensionInfo, PowerDiagnostics, PowerOutcome, Report};

/// Analyze a dataset for power side-channel leakage.
///
/// This is the main entry point for power analysis. It:
/// 1. Validates the dataset
/// 2. Extracts features from traces
/// 3. Computes class difference and covariance
/// 4. Runs Bayesian inference
/// 5. Returns a detailed report
///
/// # Arguments
///
/// * `dataset` - The dataset to analyze
/// * `config` - Analysis configuration
///
/// # Returns
///
/// A `Report` containing the analysis results, including leak probability,
/// effect estimates, and localization information.
///
/// # Example
///
/// ```ignore
/// use tacet::power::{Dataset, Trace, Class, Config, analyze};
///
/// let traces = vec![
///     Trace::new(Class::Fixed, vec![1.0, 2.0, 3.0, 4.0]),
///     Trace::new(Class::Random, vec![1.1, 2.1, 3.1, 4.1]),
/// ];
/// let dataset = Dataset::new(traces);
/// let config = Config::default();
/// let report = analyze(&dataset, &config);
///
/// println!("Leak probability: {:.1}%", report.leak_probability * 100.0);
/// ```
pub fn analyze(dataset: &Dataset, config: &Config) -> Report {
    // 1. Validate dataset
    if let Err(e) = dataset.validate() {
        let dim = DimensionInfo::new(config.feature_dimension(), 0.0);
        return Report::new(
            PowerOutcome::Inconclusive {
                reason: format!("Dataset validation failed: {}", e),
                leak_probability: 0.5,
            },
            dim,
        );
    }

    // 2. Extract features
    let features = extract_features(dataset, config);

    // 3. Compute statistics
    let difference = compute_class_difference(&features);
    let _covariance = compute_pooled_covariance(&features);

    // 4. Compute effective sample size
    // For now, use simple pooled n_eff (will integrate IACT later)
    let n_fixed = features.fixed_count();
    let n_random = features.random_count();
    let n_total = n_fixed + n_random;
    let n_eff = (n_fixed as f64 * n_random as f64) / n_total as f64;

    // 5. Dimension info
    let d = features.dimension;
    let dim_info = DimensionInfo::new(d, n_eff);

    // 6. Compute max effect (simple approach for now)
    let max_effect = difference
        .iter()
        .map(|v| v.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    // 7. Estimate noise floor (θ_floor)
    // Use standard error of the difference as a proxy
    let se_per_dim: Vec<f64> = difference
        .iter()
        .enumerate()
        .map(|(i, _)| {
            // Compute standard error for this dimension
            let mut var_fixed = 0.0;
            let mut var_random = 0.0;

            for tf in &features.fixed {
                let diff = tf.features[i] - difference[i];
                var_fixed += diff * diff;
            }
            for tf in &features.random {
                let diff = tf.features[i];
                var_random += diff * diff;
            }

            if n_fixed > 1 {
                var_fixed /= (n_fixed - 1) as f64;
            }
            if n_random > 1 {
                var_random /= (n_random - 1) as f64;
            }

            let pooled_var = var_fixed / n_fixed as f64 + var_random / n_random as f64;
            pooled_var.sqrt()
        })
        .collect();

    let theta_floor = se_per_dim
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(0.0);

    let theta_eff = theta_floor * config.floor_multiplier;

    // 8. Simple decision logic (will be replaced with full Bayesian inference)
    // For now, use a t-test-like approach
    let max_t_stat = if theta_floor > 1e-10 {
        max_effect / theta_floor
    } else {
        0.0
    };

    // Rough approximation of leak probability based on t-statistic
    // This will be replaced with proper Bayesian posterior computation
    let leak_probability = if max_t_stat > 0.0 {
        // Use normal CDF approximation via statrs
        use statrs::distribution::{ContinuousCDF, Normal};
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.cdf(max_t_stat)
    } else {
        0.5
    };

    // 9. Determine outcome
    let outcome = if leak_probability < config.pass_threshold {
        PowerOutcome::Pass {
            leak_probability,
            max_effect,
        }
    } else if leak_probability > config.fail_threshold {
        // Simple CI estimate (will be replaced with posterior CI)
        let ci_width = 1.96 * theta_floor;
        PowerOutcome::Fail {
            leak_probability,
            max_effect,
            max_effect_ci95: (max_effect - ci_width, max_effect + ci_width),
        }
    } else {
        PowerOutcome::Inconclusive {
            reason: "Posterior probability between thresholds".to_string(),
            leak_probability,
        }
    };

    // 10. Build diagnostics
    let diagnostics = PowerDiagnostics {
        n_fixed,
        n_random,
        n_total,
        n_eff,
        iact_fixed: 1.0,
        iact_random: 1.0,
        iact_combined: 1.0,
        theta_floor,
        block_length: 1,
        gibbs_samples: 0,
        gibbs_burnin: 0,
        convergence: None,
        warnings: Vec::new(),
    };

    // 11. Build report
    let mut report = Report::new(outcome, dim_info);
    report.max_effect = max_effect;
    report.theta_floor = theta_floor;
    report.floor_multiplier = config.floor_multiplier;
    report.theta_eff = theta_eff;
    report.units = dataset.units.clone();
    report.feature_family = config.feature_family;
    report.diagnostics = diagnostics;

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::PreprocessingConfig;
    use super::super::dataset::{Class, Trace};

    #[test]
    fn test_analyze_empty_dataset() {
        let dataset = Dataset::new(vec![]);
        let config = Config::default();
        let report = analyze(&dataset, &config);

        assert!(!report.outcome.is_conclusive());
    }

    #[test]
    fn test_analyze_identical_classes() {
        // Create traces with nearly identical values between classes
        let traces: Vec<Trace> = (0..50)
            .map(|i| {
                let class = if i % 2 == 0 { Class::Fixed } else { Class::Random };
                Trace::new(class, vec![1.0, 2.0, 3.0, 4.0])
            })
            .collect();

        let dataset = Dataset::new(traces);
        let mut config = Config::new().with_partitions(2);
        // Disable preprocessing to test raw data comparison
        config.preprocessing = PreprocessingConfig::none();

        let report = analyze(&dataset, &config);

        // With identical data, max_effect should be ~0
        assert!(report.max_effect.abs() < 1e-6, "max_effect should be ~0 for identical data, got {}", report.max_effect);
    }

    #[test]
    fn test_analyze_different_classes() {
        // Create traces with clear difference between classes
        let mut traces = Vec::new();

        // Fixed class: higher values
        for i in 0..25 {
            traces.push(Trace::with_id(
                Class::Fixed,
                vec![10.0, 20.0, 30.0, 40.0],
                i as u64,
            ));
        }

        // Random class: lower values
        for i in 25..50 {
            traces.push(Trace::with_id(
                Class::Random,
                vec![0.0, 1.0, 2.0, 3.0],
                i as u64,
            ));
        }

        let dataset = Dataset::new(traces);
        let mut config = Config::new().with_partitions(2);
        config.preprocessing = PreprocessingConfig::none();

        let report = analyze(&dataset, &config);

        // With clear difference, should have high leak probability
        assert!(report.leak_probability > 0.9);
        assert!(report.max_effect > 5.0);
    }

    #[test]
    fn test_analyze_report_fields() {
        let traces = vec![
            Trace::new(Class::Fixed, vec![1.0, 2.0, 3.0, 4.0]),
            Trace::new(Class::Fixed, vec![1.1, 2.1, 3.1, 4.1]),
            Trace::new(Class::Random, vec![1.0, 2.0, 3.0, 4.0]),
            Trace::new(Class::Random, vec![0.9, 1.9, 2.9, 3.9]),
        ];

        let dataset = Dataset::new(traces);
        let config = Config::new()
            .with_partitions(2)
            .with_floor_multiplier(3.0);

        let report = analyze(&dataset, &config);

        // Check that fields are populated
        assert_eq!(report.diagnostics.n_fixed, 2);
        assert_eq!(report.diagnostics.n_random, 2);
        assert_eq!(report.dimension.d, 2);
        assert!(report.diagnostics.n_eff > 0.0);
        assert_eq!(report.floor_multiplier, 3.0);
    }
}
