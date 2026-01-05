//! Tests for configuration validation.
//!
//! These tests verify that invalid configuration values are rejected
//! by the builder methods with appropriate panic messages.

use timing_oracle::TimingOracle;

// =============================================================================
// SAMPLE COUNT VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "samples must be > 0")]
fn samples_zero_panics() {
    let _ = TimingOracle::new().samples(0);
}

#[test]
fn samples_one_valid() {
    // Edge case: 1 sample is technically valid at builder level
    // (downstream analysis will fail, but builder accepts it)
    let oracle = TimingOracle::new().samples(1);
    assert_eq!(oracle.config().samples, 1);
}

#[test]
fn samples_minimum_valid() {
    // 50 is the minimum for meaningful statistics
    let oracle = TimingOracle::new().samples(50);
    assert_eq!(oracle.config().samples, 50);
}

#[test]
fn samples_large_valid() {
    let oracle = TimingOracle::new().samples(1_000_000);
    assert_eq!(oracle.config().samples, 1_000_000);
}

// =============================================================================
// ALPHA (CI FALSE POSITIVE RATE) VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "alpha must be in (0, 1)")]
fn alpha_zero_panics() {
    let _ = TimingOracle::new().alpha(0.0);
}

#[test]
#[should_panic(expected = "alpha must be in (0, 1)")]
fn alpha_one_panics() {
    let _ = TimingOracle::new().alpha(1.0);
}

#[test]
#[should_panic(expected = "alpha must be in (0, 1)")]
fn alpha_negative_panics() {
    let _ = TimingOracle::new().alpha(-0.01);
}

#[test]
#[should_panic(expected = "alpha must be in (0, 1)")]
fn alpha_greater_than_one_panics() {
    let _ = TimingOracle::new().alpha(1.5);
}

#[test]
#[should_panic(expected = "alpha must be in (0, 1)")]
fn alpha_nan_panics() {
    let _ = TimingOracle::new().alpha(f64::NAN);
}

#[test]
fn alpha_tiny_valid() {
    let oracle = TimingOracle::new().alpha(0.001);
    assert_eq!(oracle.config().ci_alpha, 0.001);
}

#[test]
fn alpha_default_valid() {
    let oracle = TimingOracle::new().alpha(0.01);
    assert_eq!(oracle.config().ci_alpha, 0.01);
}

#[test]
fn alpha_common_valid() {
    let oracle = TimingOracle::new().alpha(0.05);
    assert_eq!(oracle.config().ci_alpha, 0.05);
}

#[test]
fn alpha_large_valid() {
    let oracle = TimingOracle::new().alpha(0.1);
    assert_eq!(oracle.config().ci_alpha, 0.1);
}

#[test]
fn alpha_near_zero_valid() {
    let oracle = TimingOracle::new().alpha(0.0001);
    assert_eq!(oracle.config().ci_alpha, 0.0001);
}

#[test]
fn alpha_near_one_valid() {
    let oracle = TimingOracle::new().alpha(0.9999);
    assert_eq!(oracle.config().ci_alpha, 0.9999);
}

// =============================================================================
// OUTLIER PERCENTILE VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "outlier_percentile must be in (0, 1]")]
fn outlier_percentile_zero_panics() {
    let _ = TimingOracle::new().outlier_percentile(0.0);
}

#[test]
#[should_panic(expected = "outlier_percentile must be in (0, 1]")]
fn outlier_percentile_negative_panics() {
    let _ = TimingOracle::new().outlier_percentile(-0.1);
}

#[test]
#[should_panic(expected = "outlier_percentile must be in (0, 1]")]
fn outlier_percentile_greater_than_one_panics() {
    let _ = TimingOracle::new().outlier_percentile(1.1);
}

#[test]
#[should_panic(expected = "outlier_percentile must be in (0, 1]")]
fn outlier_percentile_nan_panics() {
    let _ = TimingOracle::new().outlier_percentile(f64::NAN);
}

#[test]
fn outlier_percentile_one_valid() {
    // 1.0 disables filtering
    let oracle = TimingOracle::new().outlier_percentile(1.0);
    assert_eq!(oracle.config().outlier_percentile, 1.0);
}

#[test]
fn outlier_percentile_default_valid() {
    let oracle = TimingOracle::new().outlier_percentile(0.999);
    assert_eq!(oracle.config().outlier_percentile, 0.999);
}

#[test]
fn outlier_percentile_aggressive_valid() {
    let oracle = TimingOracle::new().outlier_percentile(0.95);
    assert_eq!(oracle.config().outlier_percentile, 0.95);
}

#[test]
fn outlier_percentile_tiny_valid() {
    let oracle = TimingOracle::new().outlier_percentile(0.001);
    assert_eq!(oracle.config().outlier_percentile, 0.001);
}

// =============================================================================
// PRIOR NO LEAK VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "prior_no_leak must be in (0, 1)")]
fn prior_no_leak_zero_panics() {
    let _ = TimingOracle::new().prior_no_leak(0.0);
}

#[test]
#[should_panic(expected = "prior_no_leak must be in (0, 1)")]
fn prior_no_leak_one_panics() {
    let _ = TimingOracle::new().prior_no_leak(1.0);
}

#[test]
#[should_panic(expected = "prior_no_leak must be in (0, 1)")]
fn prior_no_leak_negative_panics() {
    let _ = TimingOracle::new().prior_no_leak(-0.5);
}

#[test]
#[should_panic(expected = "prior_no_leak must be in (0, 1)")]
fn prior_no_leak_greater_than_one_panics() {
    let _ = TimingOracle::new().prior_no_leak(1.5);
}

#[test]
#[should_panic(expected = "prior_no_leak must be in (0, 1)")]
fn prior_no_leak_nan_panics() {
    let _ = TimingOracle::new().prior_no_leak(f64::NAN);
}

#[test]
fn prior_no_leak_default_valid() {
    let oracle = TimingOracle::new().prior_no_leak(0.75);
    assert_eq!(oracle.config().prior_no_leak, 0.75);
}

#[test]
fn prior_no_leak_half_valid() {
    let oracle = TimingOracle::new().prior_no_leak(0.5);
    assert_eq!(oracle.config().prior_no_leak, 0.5);
}

#[test]
fn prior_no_leak_near_zero_valid() {
    let oracle = TimingOracle::new().prior_no_leak(0.001);
    assert_eq!(oracle.config().prior_no_leak, 0.001);
}

#[test]
fn prior_no_leak_near_one_valid() {
    let oracle = TimingOracle::new().prior_no_leak(0.999);
    assert_eq!(oracle.config().prior_no_leak, 0.999);
}

// =============================================================================
// CALIBRATION FRACTION VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "calibration_fraction must be in (0, 1)")]
fn calibration_fraction_zero_panics() {
    let _ = TimingOracle::new().calibration_fraction(0.0);
}

#[test]
#[should_panic(expected = "calibration_fraction must be in (0, 1)")]
fn calibration_fraction_one_panics() {
    let _ = TimingOracle::new().calibration_fraction(1.0);
}

#[test]
#[should_panic(expected = "calibration_fraction must be in (0, 1)")]
fn calibration_fraction_negative_panics() {
    let _ = TimingOracle::new().calibration_fraction(-0.1);
}

#[test]
#[should_panic(expected = "calibration_fraction must be in (0, 1)")]
fn calibration_fraction_greater_than_one_panics() {
    let _ = TimingOracle::new().calibration_fraction(1.5);
}

#[test]
#[should_panic(expected = "calibration_fraction must be in (0, 1)")]
fn calibration_fraction_nan_panics() {
    let _ = TimingOracle::new().calibration_fraction(f32::NAN);
}

#[test]
fn calibration_fraction_default_valid() {
    let oracle = TimingOracle::new().calibration_fraction(0.3);
    assert_eq!(oracle.config().calibration_fraction, 0.3);
}

#[test]
fn calibration_fraction_half_valid() {
    let oracle = TimingOracle::new().calibration_fraction(0.5);
    assert_eq!(oracle.config().calibration_fraction, 0.5);
}

#[test]
fn calibration_fraction_small_valid() {
    let oracle = TimingOracle::new().calibration_fraction(0.01);
    assert_eq!(oracle.config().calibration_fraction, 0.01);
}

#[test]
fn calibration_fraction_large_valid() {
    let oracle = TimingOracle::new().calibration_fraction(0.99);
    assert_eq!(oracle.config().calibration_fraction, 0.99);
}

// =============================================================================
// BOOTSTRAP ITERATIONS VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "ci_bootstrap_iterations must be > 0")]
fn ci_bootstrap_zero_panics() {
    let _ = TimingOracle::new().ci_bootstrap_iterations(0);
}

#[test]
fn ci_bootstrap_one_valid() {
    let oracle = TimingOracle::new().ci_bootstrap_iterations(1);
    assert_eq!(oracle.config().ci_bootstrap_iterations, 1);
}

#[test]
fn ci_bootstrap_reasonable_valid() {
    let oracle = TimingOracle::new().ci_bootstrap_iterations(100);
    assert_eq!(oracle.config().ci_bootstrap_iterations, 100);
}

#[test]
fn ci_bootstrap_large_valid() {
    let oracle = TimingOracle::new().ci_bootstrap_iterations(10_000);
    assert_eq!(oracle.config().ci_bootstrap_iterations, 10_000);
}

#[test]
#[should_panic(expected = "cov_bootstrap_iterations must be > 0")]
fn cov_bootstrap_zero_panics() {
    let _ = TimingOracle::new().cov_bootstrap_iterations(0);
}

#[test]
fn cov_bootstrap_one_valid() {
    let oracle = TimingOracle::new().cov_bootstrap_iterations(1);
    assert_eq!(oracle.config().cov_bootstrap_iterations, 1);
}

#[test]
fn cov_bootstrap_reasonable_valid() {
    let oracle = TimingOracle::new().cov_bootstrap_iterations(50);
    assert_eq!(oracle.config().cov_bootstrap_iterations, 50);
}

#[test]
fn cov_bootstrap_large_valid() {
    let oracle = TimingOracle::new().cov_bootstrap_iterations(2_000);
    assert_eq!(oracle.config().cov_bootstrap_iterations, 2_000);
}

// =============================================================================
// MIN EFFECT SIZE VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "min_effect_ns must be >= 0")]
fn min_effect_negative_panics() {
    let _ = TimingOracle::new().min_effect_ns(-1.0);
}

#[test]
#[should_panic(expected = "min_effect_ns must be >= 0")]
fn min_effect_nan_panics() {
    let _ = TimingOracle::new().min_effect_ns(f64::NAN);
}

#[test]
fn min_effect_zero_valid() {
    // Zero is valid (treat all effects as significant)
    let oracle = TimingOracle::new().min_effect_ns(0.0);
    assert_eq!(oracle.config().min_effect_of_concern_ns, 0.0);
}

#[test]
fn min_effect_default_valid() {
    let oracle = TimingOracle::new().min_effect_ns(10.0);
    assert_eq!(oracle.config().min_effect_of_concern_ns, 10.0);
}

#[test]
fn min_effect_large_valid() {
    let oracle = TimingOracle::new().min_effect_ns(1000.0);
    assert_eq!(oracle.config().min_effect_of_concern_ns, 1000.0);
}

#[test]
fn min_effect_infinity_valid() {
    // Infinity is technically valid (nothing is significant)
    let oracle = TimingOracle::new().min_effect_ns(f64::INFINITY);
    assert!(oracle.config().min_effect_of_concern_ns.is_infinite());
}

// =============================================================================
// EFFECT THRESHOLD VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "effect_threshold_ns must be > 0")]
fn effect_threshold_zero_panics() {
    let _ = TimingOracle::new().effect_threshold_ns(0.0);
}

#[test]
#[should_panic(expected = "effect_threshold_ns must be > 0")]
fn effect_threshold_negative_panics() {
    let _ = TimingOracle::new().effect_threshold_ns(-1.0);
}

#[test]
#[should_panic(expected = "effect_threshold_ns must be > 0")]
fn effect_threshold_nan_panics() {
    let _ = TimingOracle::new().effect_threshold_ns(f64::NAN);
}

#[test]
fn effect_threshold_small_valid() {
    let oracle = TimingOracle::new().effect_threshold_ns(0.001);
    assert_eq!(oracle.config().effect_threshold_ns, Some(0.001));
}

#[test]
fn effect_threshold_reasonable_valid() {
    let oracle = TimingOracle::new().effect_threshold_ns(100.0);
    assert_eq!(oracle.config().effect_threshold_ns, Some(100.0));
}

#[test]
fn effect_threshold_infinity_valid() {
    let oracle = TimingOracle::new().effect_threshold_ns(f64::INFINITY);
    assert!(oracle.config().effect_threshold_ns.unwrap().is_infinite());
}

// =============================================================================
// WARMUP VALIDATION (no validation currently - document behavior)
// =============================================================================

#[test]
fn warmup_zero_valid() {
    // Zero warmup is allowed (no warmup phase)
    let oracle = TimingOracle::new().warmup(0);
    assert_eq!(oracle.config().warmup, 0);
}

#[test]
fn warmup_large_valid() {
    let oracle = TimingOracle::new().warmup(100_000);
    assert_eq!(oracle.config().warmup, 100_000);
}

// =============================================================================
// ENVIRONMENT VARIABLE PARSING
// =============================================================================

mod env_tests {
    use super::*;
    use std::env;

    // Helper to run test with env var set, then clean up
    fn with_env_var<F: FnOnce()>(key: &str, value: &str, test: F) {
        env::set_var(key, value);
        test();
        env::remove_var(key);
    }

    #[test]
    fn from_env_samples_valid() {
        with_env_var("TO_SAMPLES", "50000", || {
            let oracle = TimingOracle::new().from_env();
            assert_eq!(oracle.config().samples, 50_000);
        });
    }

    #[test]
    fn from_env_alpha_valid() {
        with_env_var("TO_ALPHA", "0.05", || {
            let oracle = TimingOracle::new().from_env();
            assert_eq!(oracle.config().ci_alpha, 0.05);
        });
    }

    #[test]
    fn from_env_samples_invalid_ignored() {
        with_env_var("TO_SAMPLES", "not_a_number", || {
            let oracle = TimingOracle::new().from_env();
            // Should use default, not panic
            assert_eq!(oracle.config().samples, 100_000);
        });
    }

    #[test]
    fn from_env_alpha_invalid_format_ignored() {
        with_env_var("TO_ALPHA", "abc", || {
            let oracle = TimingOracle::new().from_env();
            // Should use default, not panic
            assert_eq!(oracle.config().ci_alpha, 0.01);
        });
    }

    #[test]
    fn from_env_missing_uses_default() {
        // Ensure env var is not set
        env::remove_var("TO_SAMPLES");
        env::remove_var("TO_ALPHA");

        let oracle = TimingOracle::new().from_env();
        assert_eq!(oracle.config().samples, 100_000);
        assert_eq!(oracle.config().ci_alpha, 0.01);
    }

    #[test]
    fn from_env_calibration_frac_valid() {
        with_env_var("TO_CALIBRATION_FRAC", "0.5", || {
            let oracle = TimingOracle::new().from_env();
            assert_eq!(oracle.config().calibration_fraction, 0.5);
        });
    }

    #[test]
    fn from_env_seed_valid() {
        with_env_var("TO_SEED", "42", || {
            let oracle = TimingOracle::new().from_env();
            assert_eq!(oracle.config().measurement_seed, Some(42));
        });
    }

    #[test]
    fn from_env_max_duration_valid() {
        with_env_var("TO_MAX_DURATION_MS", "5000", || {
            let oracle = TimingOracle::new().from_env();
            assert_eq!(oracle.config().max_duration_ms, Some(5000));
        });
    }

    #[test]
    fn from_env_effect_threshold_valid() {
        with_env_var("TO_EFFECT_THRESHOLD_NS", "50.0", || {
            let oracle = TimingOracle::new().from_env();
            assert_eq!(oracle.config().effect_threshold_ns, Some(50.0));
        });
    }

    #[test]
    fn from_env_min_effect_valid() {
        with_env_var("TO_MIN_EFFECT_NS", "5.0", || {
            let oracle = TimingOracle::new().from_env();
            assert_eq!(oracle.config().min_effect_of_concern_ns, 5.0);
        });
    }
}

// =============================================================================
// PRESET CONFIGURATIONS
// =============================================================================

#[test]
fn preset_new_has_valid_config() {
    let oracle = TimingOracle::new();
    assert!(oracle.config().samples > 0);
    assert!(oracle.config().ci_alpha > 0.0 && oracle.config().ci_alpha < 1.0);
    assert!(oracle.config().outlier_percentile > 0.0 && oracle.config().outlier_percentile <= 1.0);
    assert!(oracle.config().prior_no_leak > 0.0 && oracle.config().prior_no_leak < 1.0);
    assert!(oracle.config().calibration_fraction > 0.0 && oracle.config().calibration_fraction < 1.0);
    assert!(oracle.config().ci_bootstrap_iterations > 0);
    assert!(oracle.config().cov_bootstrap_iterations > 0);
}

#[test]
fn preset_quick_has_valid_config() {
    let oracle = TimingOracle::quick();
    assert!(oracle.config().samples > 0);
    assert!(oracle.config().ci_alpha > 0.0 && oracle.config().ci_alpha < 1.0);
    assert!(oracle.config().outlier_percentile > 0.0 && oracle.config().outlier_percentile <= 1.0);
}

#[test]
fn preset_balanced_has_valid_config() {
    let oracle = TimingOracle::balanced();
    assert!(oracle.config().samples > 0);
    assert!(oracle.config().ci_alpha > 0.0 && oracle.config().ci_alpha < 1.0);
    assert!(oracle.config().outlier_percentile > 0.0 && oracle.config().outlier_percentile <= 1.0);
}

#[test]
fn preset_calibration_has_valid_config() {
    let oracle = TimingOracle::calibration();
    assert!(oracle.config().samples > 0);
    assert!(oracle.config().ci_alpha > 0.0 && oracle.config().ci_alpha < 1.0);
    assert!(oracle.config().outlier_percentile > 0.0 && oracle.config().outlier_percentile <= 1.0);
}

// =============================================================================
// BUILDER CHAINING
// =============================================================================

#[test]
fn builder_chaining_all_valid() {
    let oracle = TimingOracle::new()
        .samples(10_000)
        .warmup(100)
        .alpha(0.05)
        .min_effect_ns(5.0)
        .effect_threshold_ns(100.0)
        .ci_bootstrap_iterations(50)
        .cov_bootstrap_iterations(50)
        .outlier_percentile(0.99)
        .prior_no_leak(0.8)
        .calibration_fraction(0.25);

    assert_eq!(oracle.config().samples, 10_000);
    assert_eq!(oracle.config().warmup, 100);
    assert_eq!(oracle.config().ci_alpha, 0.05);
    assert_eq!(oracle.config().min_effect_of_concern_ns, 5.0);
    assert_eq!(oracle.config().effect_threshold_ns, Some(100.0));
    assert_eq!(oracle.config().ci_bootstrap_iterations, 50);
    assert_eq!(oracle.config().cov_bootstrap_iterations, 50);
    assert_eq!(oracle.config().outlier_percentile, 0.99);
    assert_eq!(oracle.config().prior_no_leak, 0.8);
    assert_eq!(oracle.config().calibration_fraction, 0.25);
}

#[test]
fn builder_override_preset() {
    // Start with quick preset, override samples
    let oracle = TimingOracle::quick().samples(50_000);
    assert_eq!(oracle.config().samples, 50_000);
    // Other quick settings preserved
    assert_eq!(oracle.config().warmup, 50);
}
