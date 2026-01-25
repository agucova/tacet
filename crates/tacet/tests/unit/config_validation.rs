//! Tests for configuration validation.
//!
//! These tests verify that invalid configuration values are rejected
//! by the builder methods with appropriate panic messages.

use std::time::Duration;
use tacet::{AttackerModel, TimingOracle};

// =============================================================================
// MAX SAMPLES VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "max_samples must be > 0")]
fn max_samples_zero_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).max_samples(0);
}

#[test]
fn max_samples_one_valid() {
    // Edge case: 1 sample is technically valid at builder level
    // (downstream analysis will fail, but builder accepts it)
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).max_samples(1);
    assert_eq!(oracle.config().max_samples, 1);
}

#[test]
fn max_samples_minimum_valid() {
    // 50 is a reasonable minimum for meaningful statistics
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).max_samples(50);
    assert_eq!(oracle.config().max_samples, 50);
}

#[test]
fn max_samples_large_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).max_samples(1_000_000);
    assert_eq!(oracle.config().max_samples, 1_000_000);
}

// =============================================================================
// PASS/FAIL THRESHOLD VALIDATION (replaces alpha)
// =============================================================================

#[test]
#[should_panic(expected = "pass_threshold must be in (0, 1)")]
fn pass_threshold_zero_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).pass_threshold(0.0);
}

#[test]
#[should_panic(expected = "pass_threshold must be in (0, 1)")]
fn pass_threshold_one_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).pass_threshold(1.0);
}

#[test]
#[should_panic(expected = "pass_threshold must be in (0, 1)")]
fn pass_threshold_negative_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).pass_threshold(-0.01);
}

#[test]
#[should_panic(expected = "pass_threshold must be in (0, 1)")]
fn pass_threshold_greater_than_one_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).pass_threshold(1.5);
}

#[test]
fn pass_threshold_tiny_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).pass_threshold(0.001);
    assert_eq!(oracle.config().pass_threshold, 0.001);
}

#[test]
fn pass_threshold_default_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).pass_threshold(0.05);
    assert_eq!(oracle.config().pass_threshold, 0.05);
}

#[test]
fn pass_threshold_common_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).pass_threshold(0.01);
    assert_eq!(oracle.config().pass_threshold, 0.01);
}

#[test]
#[should_panic(expected = "fail_threshold must be in (0, 1)")]
fn fail_threshold_zero_panics() {
    // First set pass_threshold low, then try invalid fail_threshold
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.0);
}

#[test]
#[should_panic(expected = "fail_threshold must be in (0, 1)")]
fn fail_threshold_one_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).fail_threshold(1.0);
}

#[test]
#[should_panic(expected = "fail_threshold must be > pass_threshold")]
fn fail_threshold_less_than_pass_panics() {
    // Default pass_threshold is 0.05
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).fail_threshold(0.01);
}

#[test]
fn fail_threshold_default_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).fail_threshold(0.95);
    assert_eq!(oracle.config().fail_threshold, 0.95);
}

#[test]
fn fail_threshold_high_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).fail_threshold(0.99);
    assert_eq!(oracle.config().fail_threshold, 0.99);
}

// =============================================================================
// OUTLIER PERCENTILE VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "outlier_percentile must be in (0, 1]")]
fn outlier_percentile_zero_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).outlier_percentile(0.0);
}

#[test]
#[should_panic(expected = "outlier_percentile must be in (0, 1]")]
fn outlier_percentile_negative_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).outlier_percentile(-0.1);
}

#[test]
#[should_panic(expected = "outlier_percentile must be in (0, 1]")]
fn outlier_percentile_greater_than_one_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).outlier_percentile(1.1);
}

#[test]
fn outlier_percentile_one_valid() {
    // 1.0 disables filtering
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).outlier_percentile(1.0);
    assert_eq!(oracle.config().outlier_percentile, 1.0);
}

#[test]
fn outlier_percentile_default_valid() {
    let oracle =
        TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).outlier_percentile(0.999);
    assert_eq!(oracle.config().outlier_percentile, 0.999);
}

#[test]
fn outlier_percentile_aggressive_valid() {
    let oracle =
        TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).outlier_percentile(0.95);
    assert_eq!(oracle.config().outlier_percentile, 0.95);
}

#[test]
fn outlier_percentile_tiny_valid() {
    let oracle =
        TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).outlier_percentile(0.001);
    assert_eq!(oracle.config().outlier_percentile, 0.001);
}

// =============================================================================
// PRIOR NO LEAK VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "prior_no_leak must be in (0, 1)")]
fn prior_no_leak_zero_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).prior_no_leak(0.0);
}

#[test]
#[should_panic(expected = "prior_no_leak must be in (0, 1)")]
fn prior_no_leak_one_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).prior_no_leak(1.0);
}

#[test]
#[should_panic(expected = "prior_no_leak must be in (0, 1)")]
fn prior_no_leak_negative_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).prior_no_leak(-0.5);
}

#[test]
#[should_panic(expected = "prior_no_leak must be in (0, 1)")]
fn prior_no_leak_greater_than_one_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).prior_no_leak(1.5);
}

#[test]
fn prior_no_leak_default_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).prior_no_leak(0.75);
    assert_eq!(oracle.config().prior_no_leak, 0.75);
}

#[test]
fn prior_no_leak_half_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).prior_no_leak(0.5);
    assert_eq!(oracle.config().prior_no_leak, 0.5);
}

#[test]
fn prior_no_leak_near_zero_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).prior_no_leak(0.001);
    assert_eq!(oracle.config().prior_no_leak, 0.001);
}

#[test]
fn prior_no_leak_near_one_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).prior_no_leak(0.999);
    assert_eq!(oracle.config().prior_no_leak, 0.999);
}

// =============================================================================
// CALIBRATION SAMPLES VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "calibration_samples must be > 0")]
fn calibration_samples_zero_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).calibration_samples(0);
}

#[test]
fn calibration_samples_one_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).calibration_samples(1);
    assert_eq!(oracle.config().calibration_samples, 1);
}

#[test]
fn calibration_samples_reasonable_valid() {
    let oracle =
        TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).calibration_samples(5000);
    assert_eq!(oracle.config().calibration_samples, 5000);
}

#[test]
fn calibration_samples_large_valid() {
    let oracle =
        TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).calibration_samples(50_000);
    assert_eq!(oracle.config().calibration_samples, 50_000);
}

// =============================================================================
// COV BOOTSTRAP ITERATIONS VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "cov_bootstrap_iterations must be > 0")]
fn cov_bootstrap_zero_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).cov_bootstrap_iterations(0);
}

#[test]
fn cov_bootstrap_one_valid() {
    let oracle =
        TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).cov_bootstrap_iterations(1);
    assert_eq!(oracle.config().cov_bootstrap_iterations, 1);
}

#[test]
fn cov_bootstrap_reasonable_valid() {
    let oracle =
        TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).cov_bootstrap_iterations(50);
    assert_eq!(oracle.config().cov_bootstrap_iterations, 50);
}

#[test]
fn cov_bootstrap_large_valid() {
    let oracle =
        TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).cov_bootstrap_iterations(2_000);
    assert_eq!(oracle.config().cov_bootstrap_iterations, 2_000);
}

// =============================================================================
// WARMUP VALIDATION (no validation currently - document behavior)
// =============================================================================

#[test]
fn warmup_zero_valid() {
    // Zero warmup is allowed (no warmup phase)
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).warmup(0);
    assert_eq!(oracle.config().warmup, 0);
}

#[test]
fn warmup_large_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).warmup(100_000);
    assert_eq!(oracle.config().warmup, 100_000);
}

// =============================================================================
// TIME BUDGET VALIDATION
// =============================================================================

#[test]
fn time_budget_short_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(5));
    assert_eq!(oracle.config().time_budget, Duration::from_secs(5));
}

#[test]
fn time_budget_long_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(300));
    assert_eq!(oracle.config().time_budget, Duration::from_secs(300));
}

#[test]
fn time_budget_secs_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget_secs(60);
    assert_eq!(oracle.config().time_budget, Duration::from_secs(60));
}

// =============================================================================
// BATCH SIZE VALIDATION
// =============================================================================

#[test]
#[should_panic(expected = "batch_size must be > 0")]
fn batch_size_zero_panics() {
    let _ = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).batch_size(0);
}

#[test]
fn batch_size_one_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).batch_size(1);
    assert_eq!(oracle.config().batch_size, 1);
}

#[test]
fn batch_size_reasonable_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).batch_size(500);
    assert_eq!(oracle.config().batch_size, 500);
}

#[test]
fn batch_size_large_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).batch_size(10_000);
    assert_eq!(oracle.config().batch_size, 10_000);
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
    fn from_env_max_samples_valid() {
        with_env_var("TO_MAX_SAMPLES", "50000", || {
            let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).from_env();
            assert_eq!(oracle.config().max_samples, 50_000);
        });
    }

    #[test]
    fn from_env_pass_threshold_valid() {
        with_env_var("TO_PASS_THRESHOLD", "0.01", || {
            let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).from_env();
            assert_eq!(oracle.config().pass_threshold, 0.01);
        });
    }

    #[test]
    fn from_env_fail_threshold_valid() {
        with_env_var("TO_FAIL_THRESHOLD", "0.99", || {
            let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).from_env();
            assert_eq!(oracle.config().fail_threshold, 0.99);
        });
    }

    #[test]
    fn from_env_max_samples_invalid_ignored() {
        with_env_var("TO_MAX_SAMPLES", "not_a_number", || {
            let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).from_env();
            // Should use default, not panic
            assert_eq!(oracle.config().max_samples, 1_000_000);
        });
    }

    #[test]
    fn from_env_pass_threshold_invalid_format_ignored() {
        with_env_var("TO_PASS_THRESHOLD", "abc", || {
            let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).from_env();
            // Should use default, not panic
            assert_eq!(oracle.config().pass_threshold, 0.05);
        });
    }

    #[test]
    fn from_env_missing_uses_default() {
        // Ensure env var is not set
        env::remove_var("TO_MAX_SAMPLES");
        env::remove_var("TO_PASS_THRESHOLD");

        let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).from_env();
        assert_eq!(oracle.config().max_samples, 1_000_000);
        assert_eq!(oracle.config().pass_threshold, 0.05);
    }

    #[test]
    fn from_env_seed_valid() {
        with_env_var("TO_SEED", "42", || {
            let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).from_env();
            assert_eq!(oracle.config().measurement_seed, Some(42));
        });
    }

    #[test]
    fn from_env_time_budget_valid() {
        with_env_var("TO_TIME_BUDGET_SECS", "120", || {
            let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).from_env();
            assert_eq!(oracle.config().time_budget, Duration::from_secs(120));
        });
    }

    #[test]
    fn from_env_min_effect_valid() {
        with_env_var("TO_MIN_EFFECT_NS", "5.0", || {
            let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).from_env();
            assert_eq!(oracle.config().min_effect_of_concern_ns, 5.0);
        });
    }
}

// =============================================================================
// ATTACKER MODEL PRESETS
// =============================================================================

#[test]
fn attacker_model_adjacent_network() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork);
    assert!(oracle.config().attacker_model.is_some());
}

#[test]
fn attacker_model_remote_network() {
    let oracle = TimingOracle::for_attacker(AttackerModel::RemoteNetwork);
    assert!(oracle.config().attacker_model.is_some());
}

#[test]
fn attacker_model_shared_hardware() {
    let oracle = TimingOracle::for_attacker(AttackerModel::SharedHardware);
    assert!(oracle.config().attacker_model.is_some());
}

#[test]
fn attacker_model_research() {
    let oracle = TimingOracle::for_attacker(AttackerModel::Research);
    assert!(oracle.config().attacker_model.is_some());
}

// =============================================================================
// PRESET CONFIGURATIONS (via time_budget)
// =============================================================================

#[test]
fn preset_default_has_valid_config() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork);
    assert!(oracle.config().max_samples > 0);
    assert!(oracle.config().pass_threshold > 0.0 && oracle.config().pass_threshold < 1.0);
    assert!(oracle.config().fail_threshold > 0.0 && oracle.config().fail_threshold < 1.0);
    assert!(oracle.config().outlier_percentile > 0.0 && oracle.config().outlier_percentile <= 1.0);
    assert!(oracle.config().prior_no_leak > 0.0 && oracle.config().prior_no_leak < 1.0);
    assert!(oracle.config().cov_bootstrap_iterations > 0);
}

#[test]
fn preset_quick_via_time_budget() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(10));
    assert_eq!(oracle.config().time_budget, Duration::from_secs(10));
    assert!(oracle.config().max_samples > 0);
    assert!(oracle.config().pass_threshold > 0.0 && oracle.config().pass_threshold < 1.0);
}

#[test]
fn preset_balanced_via_time_budget() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30));
    assert_eq!(oracle.config().time_budget, Duration::from_secs(30));
    assert!(oracle.config().max_samples > 0);
    assert!(oracle.config().pass_threshold > 0.0 && oracle.config().pass_threshold < 1.0);
}

#[test]
fn preset_calibration_via_time_budget() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(5));
    assert_eq!(oracle.config().time_budget, Duration::from_secs(5));
    assert!(oracle.config().max_samples > 0);
    assert!(oracle.config().pass_threshold > 0.0 && oracle.config().pass_threshold < 1.0);
}

// =============================================================================
// BUILDER CHAINING
// =============================================================================

#[test]
fn builder_chaining_all_valid() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(10_000)
        .warmup(100)
        .pass_threshold(0.01)
        .fail_threshold(0.99)
        .cov_bootstrap_iterations(50)
        .outlier_percentile(0.99)
        .prior_no_leak(0.8)
        .time_budget(Duration::from_secs(30))
        .batch_size(500)
        .calibration_samples(2000);

    assert_eq!(oracle.config().max_samples, 10_000);
    assert_eq!(oracle.config().warmup, 100);
    assert_eq!(oracle.config().pass_threshold, 0.01);
    assert_eq!(oracle.config().fail_threshold, 0.99);
    assert_eq!(oracle.config().cov_bootstrap_iterations, 50);
    assert_eq!(oracle.config().outlier_percentile, 0.99);
    assert_eq!(oracle.config().prior_no_leak, 0.8);
    assert_eq!(oracle.config().time_budget, Duration::from_secs(30));
    assert_eq!(oracle.config().batch_size, 500);
    assert_eq!(oracle.config().calibration_samples, 2000);
}

#[test]
fn builder_override_defaults() {
    // Start with attacker model, override max_samples
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(10))
        .max_samples(50_000);
    assert_eq!(oracle.config().max_samples, 50_000);
    assert_eq!(oracle.config().time_budget, Duration::from_secs(10));
}
