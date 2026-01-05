//! Unit tests for types in result.rs and types.rs
//!
//! Tests boundary conditions, edge cases, and method correctness for:
//! - Exploitability classification
//! - MeasurementQuality classification
//! - Outcome reliability checks
//! - UnreliablePolicy env parsing
//! - Diagnostics checks
//! - TestResult methods
//! - Serialization round-trips

use timing_oracle::{
    CiGate, Diagnostics, Effect, EffectPattern, Exploitability, MeasurementQuality,
    MinDetectableEffect, Outcome, TestResult, UnreliablePolicy,
};

// ============================================================================
// Exploitability::from_effect_ns() tests
// ============================================================================

#[test]
fn exploitability_negligible_zero() {
    assert_eq!(Exploitability::from_effect_ns(0.0), Exploitability::Negligible);
}

#[test]
fn exploitability_negligible_small() {
    assert_eq!(Exploitability::from_effect_ns(50.0), Exploitability::Negligible);
    assert_eq!(Exploitability::from_effect_ns(99.0), Exploitability::Negligible);
    assert_eq!(Exploitability::from_effect_ns(99.9), Exploitability::Negligible);
}

#[test]
fn exploitability_boundary_100ns() {
    // < 100 is Negligible, >= 100 is PossibleLAN
    assert_eq!(Exploitability::from_effect_ns(99.99), Exploitability::Negligible);
    assert_eq!(Exploitability::from_effect_ns(100.0), Exploitability::PossibleLAN);
    assert_eq!(Exploitability::from_effect_ns(100.01), Exploitability::PossibleLAN);
}

#[test]
fn exploitability_possible_lan() {
    assert_eq!(Exploitability::from_effect_ns(200.0), Exploitability::PossibleLAN);
    assert_eq!(Exploitability::from_effect_ns(499.0), Exploitability::PossibleLAN);
}

#[test]
fn exploitability_boundary_500ns() {
    // < 500 is PossibleLAN, >= 500 is LikelyLAN
    assert_eq!(Exploitability::from_effect_ns(499.99), Exploitability::PossibleLAN);
    assert_eq!(Exploitability::from_effect_ns(500.0), Exploitability::LikelyLAN);
    assert_eq!(Exploitability::from_effect_ns(500.01), Exploitability::LikelyLAN);
}

#[test]
fn exploitability_likely_lan() {
    assert_eq!(Exploitability::from_effect_ns(1_000.0), Exploitability::LikelyLAN);
    assert_eq!(Exploitability::from_effect_ns(10_000.0), Exploitability::LikelyLAN);
    assert_eq!(Exploitability::from_effect_ns(19_999.0), Exploitability::LikelyLAN);
}

#[test]
fn exploitability_boundary_20us() {
    // < 20000 is LikelyLAN, >= 20000 is PossibleRemote
    assert_eq!(Exploitability::from_effect_ns(19_999.99), Exploitability::LikelyLAN);
    assert_eq!(Exploitability::from_effect_ns(20_000.0), Exploitability::PossibleRemote);
    assert_eq!(Exploitability::from_effect_ns(20_000.01), Exploitability::PossibleRemote);
}

#[test]
fn exploitability_possible_remote() {
    assert_eq!(Exploitability::from_effect_ns(50_000.0), Exploitability::PossibleRemote);
    assert_eq!(Exploitability::from_effect_ns(1_000_000.0), Exploitability::PossibleRemote);
}

#[test]
fn exploitability_negative_uses_abs() {
    // Negative values should use absolute value
    assert_eq!(Exploitability::from_effect_ns(-50.0), Exploitability::Negligible);
    assert_eq!(Exploitability::from_effect_ns(-100.0), Exploitability::PossibleLAN);
    assert_eq!(Exploitability::from_effect_ns(-500.0), Exploitability::LikelyLAN);
    assert_eq!(Exploitability::from_effect_ns(-20_000.0), Exploitability::PossibleRemote);
}

#[test]
fn exploitability_special_values() {
    // NaN abs() is NaN, which is not < 100, not < 500, not < 20000
    // So NaN should fall through to PossibleRemote
    assert_eq!(Exploitability::from_effect_ns(f64::NAN), Exploitability::PossibleRemote);

    // Infinity is > 20000
    assert_eq!(Exploitability::from_effect_ns(f64::INFINITY), Exploitability::PossibleRemote);
    assert_eq!(Exploitability::from_effect_ns(f64::NEG_INFINITY), Exploitability::PossibleRemote);
}

// ============================================================================
// MeasurementQuality::from_mde_ns() tests
// ============================================================================

#[test]
fn quality_excellent() {
    assert_eq!(MeasurementQuality::from_mde_ns(0.5), MeasurementQuality::Excellent);
    assert_eq!(MeasurementQuality::from_mde_ns(1.0), MeasurementQuality::Excellent);
    assert_eq!(MeasurementQuality::from_mde_ns(4.9), MeasurementQuality::Excellent);
}

#[test]
fn quality_boundary_5ns() {
    // < 5 is Excellent, >= 5 is Good
    assert_eq!(MeasurementQuality::from_mde_ns(4.99), MeasurementQuality::Excellent);
    assert_eq!(MeasurementQuality::from_mde_ns(5.0), MeasurementQuality::Good);
    assert_eq!(MeasurementQuality::from_mde_ns(5.01), MeasurementQuality::Good);
}

#[test]
fn quality_good() {
    assert_eq!(MeasurementQuality::from_mde_ns(10.0), MeasurementQuality::Good);
    assert_eq!(MeasurementQuality::from_mde_ns(19.9), MeasurementQuality::Good);
}

#[test]
fn quality_boundary_20ns() {
    // < 20 is Good, >= 20 is Poor
    assert_eq!(MeasurementQuality::from_mde_ns(19.99), MeasurementQuality::Good);
    assert_eq!(MeasurementQuality::from_mde_ns(20.0), MeasurementQuality::Poor);
    assert_eq!(MeasurementQuality::from_mde_ns(20.01), MeasurementQuality::Poor);
}

#[test]
fn quality_poor() {
    assert_eq!(MeasurementQuality::from_mde_ns(50.0), MeasurementQuality::Poor);
    assert_eq!(MeasurementQuality::from_mde_ns(99.9), MeasurementQuality::Poor);
}

#[test]
fn quality_boundary_100ns() {
    // < 100 is Poor, >= 100 is TooNoisy
    assert_eq!(MeasurementQuality::from_mde_ns(99.99), MeasurementQuality::Poor);
    assert_eq!(MeasurementQuality::from_mde_ns(100.0), MeasurementQuality::TooNoisy);
    assert_eq!(MeasurementQuality::from_mde_ns(100.01), MeasurementQuality::TooNoisy);
}

#[test]
fn quality_too_noisy() {
    assert_eq!(MeasurementQuality::from_mde_ns(500.0), MeasurementQuality::TooNoisy);
    assert_eq!(MeasurementQuality::from_mde_ns(1000.0), MeasurementQuality::TooNoisy);
}

#[test]
fn quality_near_zero_is_too_noisy() {
    // MDE <= 0.01 indicates timer resolution failure
    assert_eq!(MeasurementQuality::from_mde_ns(0.01), MeasurementQuality::TooNoisy);
    assert_eq!(MeasurementQuality::from_mde_ns(0.009), MeasurementQuality::TooNoisy);
    assert_eq!(MeasurementQuality::from_mde_ns(0.0), MeasurementQuality::TooNoisy);
    assert_eq!(MeasurementQuality::from_mde_ns(-1.0), MeasurementQuality::TooNoisy);
}

#[test]
fn quality_just_above_threshold() {
    // Just above 0.01 should work normally
    assert_eq!(MeasurementQuality::from_mde_ns(0.011), MeasurementQuality::Excellent);
    assert_eq!(MeasurementQuality::from_mde_ns(0.02), MeasurementQuality::Excellent);
}

#[test]
fn quality_special_values() {
    // Non-finite values are TooNoisy
    assert_eq!(MeasurementQuality::from_mde_ns(f64::NAN), MeasurementQuality::TooNoisy);
    assert_eq!(MeasurementQuality::from_mde_ns(f64::INFINITY), MeasurementQuality::TooNoisy);
    assert_eq!(MeasurementQuality::from_mde_ns(f64::NEG_INFINITY), MeasurementQuality::TooNoisy);
}

// ============================================================================
// Diagnostics tests
// ============================================================================

#[test]
fn diagnostics_all_ok_constructor() {
    let diag = Diagnostics::all_ok();
    assert!(diag.stationarity_ok);
    assert!(diag.model_fit_ok);
    assert!(diag.outlier_asymmetry_ok);
    assert!(diag.warnings.is_empty());
    assert!(diag.all_checks_passed());
}

#[test]
fn diagnostics_all_checks_passed_all_true() {
    let diag = Diagnostics {
        stationarity_ratio: 1.0,
        stationarity_ok: true,
        model_fit_chi2: 5.0,
        model_fit_ok: true,
        outlier_rate_fixed: 0.001,
        outlier_rate_random: 0.001,
        outlier_asymmetry_ok: true,
        warnings: vec![],
    };
    assert!(diag.all_checks_passed());
}

#[test]
fn diagnostics_all_checks_passed_one_false() {
    // stationarity_ok = false
    let diag1 = Diagnostics {
        stationarity_ratio: 10.0,
        stationarity_ok: false,
        model_fit_chi2: 5.0,
        model_fit_ok: true,
        outlier_rate_fixed: 0.001,
        outlier_rate_random: 0.001,
        outlier_asymmetry_ok: true,
        warnings: vec![],
    };
    assert!(!diag1.all_checks_passed());

    // model_fit_ok = false
    let diag2 = Diagnostics {
        stationarity_ratio: 1.0,
        stationarity_ok: true,
        model_fit_chi2: 50.0,
        model_fit_ok: false,
        outlier_rate_fixed: 0.001,
        outlier_rate_random: 0.001,
        outlier_asymmetry_ok: true,
        warnings: vec![],
    };
    assert!(!diag2.all_checks_passed());

    // outlier_asymmetry_ok = false
    let diag3 = Diagnostics {
        stationarity_ratio: 1.0,
        stationarity_ok: true,
        model_fit_chi2: 5.0,
        model_fit_ok: true,
        outlier_rate_fixed: 0.1,
        outlier_rate_random: 0.001,
        outlier_asymmetry_ok: false,
        warnings: vec![],
    };
    assert!(!diag3.all_checks_passed());
}

#[test]
fn diagnostics_all_checks_passed_all_false() {
    let diag = Diagnostics {
        stationarity_ratio: 10.0,
        stationarity_ok: false,
        model_fit_chi2: 50.0,
        model_fit_ok: false,
        outlier_rate_fixed: 0.1,
        outlier_rate_random: 0.001,
        outlier_asymmetry_ok: false,
        warnings: vec!["warning".to_string()],
    };
    assert!(!diag.all_checks_passed());
}

// ============================================================================
// Outcome::is_reliable() tests
// ============================================================================

fn make_test_result(leak_prob: f64, mde_shift: f64, quality: MeasurementQuality) -> TestResult {
    TestResult {
        leak_probability: leak_prob,
        bayes_factor: 1.0,
        effect: None,
        exploitability: Exploitability::Negligible,
        min_detectable_effect: MinDetectableEffect {
            shift_ns: mde_shift,
            tail_ns: mde_shift,
        },
        ci_gate: CiGate {
            alpha: 0.01,
            passed: true,
            threshold: 10.0,
            max_observed: 5.0,
            observed: [0.0; 9],
        },
        quality,
        outlier_fraction: 0.001,
        diagnostics: Diagnostics::all_ok(),
        metadata: timing_oracle::Metadata {
            samples_per_class: 1000,
            cycles_per_ns: 3.0,
            timer: "test".to_string(),
            timer_resolution_ns: 1.0,
            batching: timing_oracle::BatchingInfo {
                enabled: false,
                k: 1,
                ticks_per_batch: 100.0,
                rationale: "test".to_string(),
                unmeasurable: None,
            },
            runtime_secs: 1.0,
        },
    }
}

#[test]
fn outcome_is_reliable_unmeasurable() {
    let outcome = Outcome::Unmeasurable {
        operation_ns: 10.0,
        threshold_ns: 100.0,
        platform: "test".to_string(),
        recommendation: "increase complexity".to_string(),
    };
    assert!(!outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_good_quality() {
    let result = make_test_result(0.5, 10.0, MeasurementQuality::Good);
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_excellent_quality() {
    let result = make_test_result(0.5, 3.0, MeasurementQuality::Excellent);
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_poor_quality_inconclusive() {
    // Poor quality but inconclusive (0.5) - still reliable (Poor != TooNoisy)
    let result = make_test_result(0.5, 50.0, MeasurementQuality::Poor);
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_too_noisy_conclusive_high() {
    // TooNoisy but conclusive (> 0.9) - reliable because signal overcame noise
    let result = make_test_result(0.95, 150.0, MeasurementQuality::TooNoisy);
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_too_noisy_conclusive_low() {
    // TooNoisy but conclusive (< 0.1) - reliable because signal overcame noise
    let result = make_test_result(0.05, 150.0, MeasurementQuality::TooNoisy);
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_too_noisy_inconclusive() {
    // TooNoisy AND inconclusive - NOT reliable
    let result = make_test_result(0.5, 150.0, MeasurementQuality::TooNoisy);
    let outcome = Outcome::Completed(result);
    assert!(!outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_mde_too_small() {
    // MDE <= 0.01 indicates timer resolution failure - NOT reliable even with good quality
    let result = make_test_result(0.5, 0.01, MeasurementQuality::Excellent);
    let outcome = Outcome::Completed(result);
    assert!(!outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_mde_zero() {
    let result = make_test_result(0.5, 0.0, MeasurementQuality::Excellent);
    let outcome = Outcome::Completed(result);
    assert!(!outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_mde_nan() {
    let result = make_test_result(0.5, f64::NAN, MeasurementQuality::Excellent);
    let outcome = Outcome::Completed(result);
    assert!(!outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_mde_infinity() {
    let result = make_test_result(0.5, f64::INFINITY, MeasurementQuality::Excellent);
    let outcome = Outcome::Completed(result);
    assert!(!outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_mde_just_above_threshold() {
    // MDE just above 0.01 should be valid
    let result = make_test_result(0.5, 0.011, MeasurementQuality::Excellent);
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

// ============================================================================
// Outcome::unwrap_completed() tests
// ============================================================================

#[test]
fn outcome_unwrap_completed_success() {
    let result = make_test_result(0.5, 10.0, MeasurementQuality::Good);
    let outcome = Outcome::Completed(result.clone());
    let unwrapped = outcome.unwrap_completed();
    assert_eq!(unwrapped.leak_probability, 0.5);
}

#[test]
#[should_panic(expected = "Test was unmeasurable")]
fn outcome_unwrap_completed_unmeasurable_panics() {
    let outcome = Outcome::Unmeasurable {
        operation_ns: 10.0,
        threshold_ns: 100.0,
        platform: "test".to_string(),
        recommendation: "test".to_string(),
    };
    let _ = outcome.unwrap_completed();
}

// ============================================================================
// Outcome::completed() tests
// ============================================================================

#[test]
fn outcome_completed_returns_some() {
    let result = make_test_result(0.5, 10.0, MeasurementQuality::Good);
    let outcome = Outcome::Completed(result);
    assert!(outcome.completed().is_some());
}

#[test]
fn outcome_completed_returns_none() {
    let outcome = Outcome::Unmeasurable {
        operation_ns: 10.0,
        threshold_ns: 100.0,
        platform: "test".to_string(),
        recommendation: "test".to_string(),
    };
    assert!(outcome.completed().is_none());
}

// ============================================================================
// UnreliablePolicy::from_env_or() tests
// ============================================================================

#[test]
fn unreliable_policy_default_is_fail_open() {
    assert_eq!(UnreliablePolicy::default(), UnreliablePolicy::FailOpen);
}

#[test]
fn unreliable_policy_from_env_missing() {
    // Clear env var if set
    std::env::remove_var("TIMING_ORACLE_UNRELIABLE_POLICY");

    let policy = UnreliablePolicy::from_env_or(UnreliablePolicy::FailOpen);
    assert_eq!(policy, UnreliablePolicy::FailOpen);

    let policy = UnreliablePolicy::from_env_or(UnreliablePolicy::FailClosed);
    assert_eq!(policy, UnreliablePolicy::FailClosed);
}

#[test]
fn unreliable_policy_from_env_fail_open() {
    std::env::set_var("TIMING_ORACLE_UNRELIABLE_POLICY", "fail_open");
    let policy = UnreliablePolicy::from_env_or(UnreliablePolicy::FailClosed);
    assert_eq!(policy, UnreliablePolicy::FailOpen);
    std::env::remove_var("TIMING_ORACLE_UNRELIABLE_POLICY");
}

#[test]
fn unreliable_policy_from_env_fail_closed() {
    std::env::set_var("TIMING_ORACLE_UNRELIABLE_POLICY", "fail_closed");
    let policy = UnreliablePolicy::from_env_or(UnreliablePolicy::FailOpen);
    assert_eq!(policy, UnreliablePolicy::FailClosed);
    std::env::remove_var("TIMING_ORACLE_UNRELIABLE_POLICY");
}

#[test]
fn unreliable_policy_from_env_invalid() {
    std::env::set_var("TIMING_ORACLE_UNRELIABLE_POLICY", "invalid_value");
    let policy = UnreliablePolicy::from_env_or(UnreliablePolicy::FailOpen);
    assert_eq!(policy, UnreliablePolicy::FailOpen);
    std::env::remove_var("TIMING_ORACLE_UNRELIABLE_POLICY");
}

#[test]
fn unreliable_policy_from_env_empty() {
    std::env::set_var("TIMING_ORACLE_UNRELIABLE_POLICY", "");
    let policy = UnreliablePolicy::from_env_or(UnreliablePolicy::FailClosed);
    assert_eq!(policy, UnreliablePolicy::FailClosed);
    std::env::remove_var("TIMING_ORACLE_UNRELIABLE_POLICY");
}

// ============================================================================
// TestResult::can_detect() tests
// ============================================================================

#[test]
fn test_result_can_detect_above_mde() {
    let result = make_test_result(0.5, 10.0, MeasurementQuality::Good);
    assert!(result.can_detect(15.0));
    assert!(result.can_detect(100.0));
}

#[test]
fn test_result_can_detect_below_mde() {
    let result = make_test_result(0.5, 10.0, MeasurementQuality::Good);
    assert!(!result.can_detect(5.0));
    assert!(!result.can_detect(1.0));
}

#[test]
fn test_result_can_detect_at_boundary() {
    let result = make_test_result(0.5, 10.0, MeasurementQuality::Good);
    // At exactly MDE, can_detect returns true (MDE <= effect_ns)
    assert!(result.can_detect(10.0));
}

#[test]
fn test_result_can_detect_just_below() {
    let result = make_test_result(0.5, 10.0, MeasurementQuality::Good);
    assert!(!result.can_detect(9.99));
}

// ============================================================================
// Serialization round-trip tests
// ============================================================================

#[test]
fn test_result_json_roundtrip() {
    let result = make_test_result(0.75, 8.0, MeasurementQuality::Good);
    let json = serde_json::to_string(&result).unwrap();
    let deserialized: TestResult = serde_json::from_str(&json).unwrap();

    assert_eq!(result.leak_probability, deserialized.leak_probability);
    assert_eq!(result.quality, deserialized.quality);
    assert_eq!(result.ci_gate.passed, deserialized.ci_gate.passed);
}

#[test]
fn effect_json_roundtrip() {
    let effect = Effect {
        shift_ns: 15.5,
        tail_ns: 3.2,
        credible_interval_ns: (10.0, 20.0),
        pattern: EffectPattern::UniformShift,
    };
    let json = serde_json::to_string(&effect).unwrap();
    let deserialized: Effect = serde_json::from_str(&json).unwrap();

    assert_eq!(effect.shift_ns, deserialized.shift_ns);
    assert_eq!(effect.tail_ns, deserialized.tail_ns);
    assert_eq!(effect.pattern, deserialized.pattern);
}

#[test]
fn ci_gate_json_roundtrip() {
    let gate = CiGate {
        alpha: 0.01,
        passed: true,
        threshold: 12.5,
        max_observed: 8.3,
        observed: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.3],
    };
    let json = serde_json::to_string(&gate).unwrap();
    let deserialized: CiGate = serde_json::from_str(&json).unwrap();

    assert_eq!(gate.alpha, deserialized.alpha);
    assert_eq!(gate.passed, deserialized.passed);
    assert_eq!(gate.threshold, deserialized.threshold);
    assert_eq!(gate.observed, deserialized.observed);
}

#[test]
fn exploitability_json_roundtrip() {
    for variant in [
        Exploitability::Negligible,
        Exploitability::PossibleLAN,
        Exploitability::LikelyLAN,
        Exploitability::PossibleRemote,
    ] {
        let json = serde_json::to_string(&variant).unwrap();
        let deserialized: Exploitability = serde_json::from_str(&json).unwrap();
        assert_eq!(variant, deserialized);
    }
}

#[test]
fn measurement_quality_json_roundtrip() {
    for variant in [
        MeasurementQuality::Excellent,
        MeasurementQuality::Good,
        MeasurementQuality::Poor,
        MeasurementQuality::TooNoisy,
    ] {
        let json = serde_json::to_string(&variant).unwrap();
        let deserialized: MeasurementQuality = serde_json::from_str(&json).unwrap();
        assert_eq!(variant, deserialized);
    }
}

#[test]
fn effect_pattern_json_roundtrip() {
    for variant in [
        EffectPattern::UniformShift,
        EffectPattern::TailEffect,
        EffectPattern::Mixed,
        EffectPattern::Indeterminate,
    ] {
        let json = serde_json::to_string(&variant).unwrap();
        let deserialized: EffectPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(variant, deserialized);
    }
}

#[test]
fn outcome_completed_json_roundtrip() {
    let result = make_test_result(0.8, 5.0, MeasurementQuality::Excellent);
    let outcome = Outcome::Completed(result);
    let json = serde_json::to_string(&outcome).unwrap();
    let deserialized: Outcome = serde_json::from_str(&json).unwrap();

    match deserialized {
        Outcome::Completed(r) => {
            assert_eq!(r.leak_probability, 0.8);
        }
        _ => panic!("Expected Completed variant"),
    }
}

#[test]
fn outcome_unmeasurable_json_roundtrip() {
    let outcome = Outcome::Unmeasurable {
        operation_ns: 5.5,
        threshold_ns: 100.0,
        platform: "test platform".to_string(),
        recommendation: "do something".to_string(),
    };
    let json = serde_json::to_string(&outcome).unwrap();
    let deserialized: Outcome = serde_json::from_str(&json).unwrap();

    match deserialized {
        Outcome::Unmeasurable { operation_ns, platform, .. } => {
            assert_eq!(operation_ns, 5.5);
            assert_eq!(platform, "test platform");
        }
        _ => panic!("Expected Unmeasurable variant"),
    }
}
