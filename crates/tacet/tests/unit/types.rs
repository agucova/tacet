//! Unit tests for types in result.rs and types.rs
//!
//! Tests boundary conditions, edge cases, and method correctness for:
//! - Exploitability classification
//! - MeasurementQuality classification
//! - Outcome reliability checks
//! - UnreliablePolicy env parsing
//! - Diagnostics checks
//! - EffectEstimate methods
//! - Serialization round-trips

use tacet::{
    Diagnostics, EffectEstimate, EffectPattern, Exploitability, InconclusiveReason, IssueCode,
    MeasurementQuality, Outcome, QualityIssue, UnreliablePolicy,
};

// ============================================================================
// Exploitability::from_effect_ns() tests
//
// New thresholds based on:
// - < 10 ns: SharedHardwareOnly (KyberSlash, Flush+Reload)
// - 10-100 ns: Http2Multiplexing (Timeless Timing Attacks, USENIX 2020)
// - 100 ns - 10 μs: StandardRemote (Crosby et al. 2009)
// - > 10 μs: ObviousLeak
// ============================================================================

#[test]
fn exploitability_shared_hardware_zero() {
    assert_eq!(
        Exploitability::from_effect_ns(0.0),
        Exploitability::SharedHardwareOnly
    );
}

#[test]
fn exploitability_shared_hardware_small() {
    assert_eq!(
        Exploitability::from_effect_ns(5.0),
        Exploitability::SharedHardwareOnly
    );
    assert_eq!(
        Exploitability::from_effect_ns(9.0),
        Exploitability::SharedHardwareOnly
    );
    assert_eq!(
        Exploitability::from_effect_ns(9.9),
        Exploitability::SharedHardwareOnly
    );
}

#[test]
fn exploitability_boundary_10ns() {
    // < 10 is SharedHardwareOnly, >= 10 is Http2Multiplexing
    assert_eq!(
        Exploitability::from_effect_ns(9.99),
        Exploitability::SharedHardwareOnly
    );
    assert_eq!(
        Exploitability::from_effect_ns(10.0),
        Exploitability::Http2Multiplexing
    );
    assert_eq!(
        Exploitability::from_effect_ns(10.01),
        Exploitability::Http2Multiplexing
    );
}

#[test]
fn exploitability_http2_multiplexing() {
    assert_eq!(
        Exploitability::from_effect_ns(50.0),
        Exploitability::Http2Multiplexing
    );
    assert_eq!(
        Exploitability::from_effect_ns(99.0),
        Exploitability::Http2Multiplexing
    );
}

#[test]
fn exploitability_boundary_100ns() {
    // < 100 is Http2Multiplexing, >= 100 is StandardRemote
    assert_eq!(
        Exploitability::from_effect_ns(99.99),
        Exploitability::Http2Multiplexing
    );
    assert_eq!(
        Exploitability::from_effect_ns(100.0),
        Exploitability::StandardRemote
    );
    assert_eq!(
        Exploitability::from_effect_ns(100.01),
        Exploitability::StandardRemote
    );
}

#[test]
fn exploitability_standard_remote() {
    assert_eq!(
        Exploitability::from_effect_ns(500.0),
        Exploitability::StandardRemote
    );
    assert_eq!(
        Exploitability::from_effect_ns(1_000.0),
        Exploitability::StandardRemote
    );
    assert_eq!(
        Exploitability::from_effect_ns(9_999.0),
        Exploitability::StandardRemote
    );
}

#[test]
fn exploitability_boundary_10us() {
    // < 10000 is StandardRemote, >= 10000 is ObviousLeak
    assert_eq!(
        Exploitability::from_effect_ns(9_999.99),
        Exploitability::StandardRemote
    );
    assert_eq!(
        Exploitability::from_effect_ns(10_000.0),
        Exploitability::ObviousLeak
    );
    assert_eq!(
        Exploitability::from_effect_ns(10_000.01),
        Exploitability::ObviousLeak
    );
}

#[test]
fn exploitability_obvious_leak() {
    assert_eq!(
        Exploitability::from_effect_ns(20_000.0),
        Exploitability::ObviousLeak
    );
    assert_eq!(
        Exploitability::from_effect_ns(50_000.0),
        Exploitability::ObviousLeak
    );
    assert_eq!(
        Exploitability::from_effect_ns(1_000_000.0),
        Exploitability::ObviousLeak
    );
}

#[test]
fn exploitability_negative_uses_abs() {
    // Negative values should use absolute value
    assert_eq!(
        Exploitability::from_effect_ns(-5.0),
        Exploitability::SharedHardwareOnly
    );
    assert_eq!(
        Exploitability::from_effect_ns(-50.0),
        Exploitability::Http2Multiplexing
    );
    assert_eq!(
        Exploitability::from_effect_ns(-500.0),
        Exploitability::StandardRemote
    );
    assert_eq!(
        Exploitability::from_effect_ns(-10_000.0),
        Exploitability::ObviousLeak
    );
}

#[test]
fn exploitability_special_values() {
    // NaN abs() is NaN, which fails all < comparisons
    // So NaN should fall through to ObviousLeak
    assert_eq!(
        Exploitability::from_effect_ns(f64::NAN),
        Exploitability::ObviousLeak
    );

    // Infinity is > 10000
    assert_eq!(
        Exploitability::from_effect_ns(f64::INFINITY),
        Exploitability::ObviousLeak
    );
    assert_eq!(
        Exploitability::from_effect_ns(f64::NEG_INFINITY),
        Exploitability::ObviousLeak
    );
}

// ============================================================================
// MeasurementQuality::from_mde_ns() tests
// ============================================================================

#[test]
fn quality_excellent() {
    assert_eq!(
        MeasurementQuality::from_mde_ns(0.5),
        MeasurementQuality::Excellent
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(1.0),
        MeasurementQuality::Excellent
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(4.9),
        MeasurementQuality::Excellent
    );
}

#[test]
fn quality_boundary_5ns() {
    // < 5 is Excellent, >= 5 is Good
    assert_eq!(
        MeasurementQuality::from_mde_ns(4.99),
        MeasurementQuality::Excellent
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(5.0),
        MeasurementQuality::Good
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(5.01),
        MeasurementQuality::Good
    );
}

#[test]
fn quality_good() {
    assert_eq!(
        MeasurementQuality::from_mde_ns(10.0),
        MeasurementQuality::Good
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(19.9),
        MeasurementQuality::Good
    );
}

#[test]
fn quality_boundary_20ns() {
    // < 20 is Good, >= 20 is Poor
    assert_eq!(
        MeasurementQuality::from_mde_ns(19.99),
        MeasurementQuality::Good
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(20.0),
        MeasurementQuality::Poor
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(20.01),
        MeasurementQuality::Poor
    );
}

#[test]
fn quality_poor() {
    assert_eq!(
        MeasurementQuality::from_mde_ns(50.0),
        MeasurementQuality::Poor
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(99.9),
        MeasurementQuality::Poor
    );
}

#[test]
fn quality_boundary_100ns() {
    // < 100 is Poor, >= 100 is TooNoisy
    assert_eq!(
        MeasurementQuality::from_mde_ns(99.99),
        MeasurementQuality::Poor
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(100.0),
        MeasurementQuality::TooNoisy
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(100.01),
        MeasurementQuality::TooNoisy
    );
}

#[test]
fn quality_too_noisy() {
    assert_eq!(
        MeasurementQuality::from_mde_ns(500.0),
        MeasurementQuality::TooNoisy
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(1000.0),
        MeasurementQuality::TooNoisy
    );
}

#[test]
fn quality_near_zero_is_too_noisy() {
    // MDE <= 0.01 indicates timer resolution failure
    assert_eq!(
        MeasurementQuality::from_mde_ns(0.01),
        MeasurementQuality::TooNoisy
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(0.009),
        MeasurementQuality::TooNoisy
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(0.0),
        MeasurementQuality::TooNoisy
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(-1.0),
        MeasurementQuality::TooNoisy
    );
}

#[test]
fn quality_just_above_threshold() {
    // Just above 0.01 should work normally
    assert_eq!(
        MeasurementQuality::from_mde_ns(0.011),
        MeasurementQuality::Excellent
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(0.02),
        MeasurementQuality::Excellent
    );
}

#[test]
fn quality_special_values() {
    // Non-finite values are TooNoisy
    assert_eq!(
        MeasurementQuality::from_mde_ns(f64::NAN),
        MeasurementQuality::TooNoisy
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(f64::INFINITY),
        MeasurementQuality::TooNoisy
    );
    assert_eq!(
        MeasurementQuality::from_mde_ns(f64::NEG_INFINITY),
        MeasurementQuality::TooNoisy
    );
}

// ============================================================================
// Diagnostics tests
// ============================================================================

#[test]
fn diagnostics_all_ok_constructor() {
    let diag = Diagnostics::all_ok();
    assert!(diag.stationarity_ok);
    assert!(diag.projection_mismatch_ok);
    assert!(diag.outlier_asymmetry_ok);
    assert!(diag.preflight_ok);
    assert!(diag.warnings.is_empty());
    assert!(diag.all_checks_passed());
}

#[test]
fn diagnostics_all_checks_passed_all_true() {
    let diag = Diagnostics {
        dependence_length: 1,
        effective_sample_size: 100,
        stationarity_ratio: 1.0,
        stationarity_ok: true,
        projection_mismatch_q: 5.0,
        projection_mismatch_ok: true,
        outlier_rate_baseline: 0.001,
        outlier_rate_sample: 0.001,
        outlier_asymmetry_ok: true,
        discrete_mode: false,
        timer_resolution_ns: 1.0,
        duplicate_fraction: 0.0,
        preflight_ok: true,
        calibration_samples: 5000,
        total_time_secs: 1.0,
        warnings: vec![],
        quality_issues: vec![],
        preflight_warnings: vec![],
        ..Diagnostics::all_ok()
    };
    assert!(diag.all_checks_passed());
}

#[test]
fn diagnostics_all_checks_passed_one_false() {
    // stationarity_ok = false
    let diag1 = Diagnostics {
        dependence_length: 1,
        effective_sample_size: 100,
        stationarity_ratio: 10.0,
        stationarity_ok: false,
        projection_mismatch_q: 5.0,
        projection_mismatch_ok: true,
        outlier_rate_baseline: 0.001,
        outlier_rate_sample: 0.001,
        outlier_asymmetry_ok: true,
        discrete_mode: false,
        timer_resolution_ns: 1.0,
        duplicate_fraction: 0.0,
        preflight_ok: true,
        calibration_samples: 5000,
        total_time_secs: 1.0,
        warnings: vec![],
        quality_issues: vec![],
        preflight_warnings: vec![],
        ..Diagnostics::all_ok()
    };
    assert!(!diag1.all_checks_passed());

    // projection_mismatch_ok = false
    let diag2 = Diagnostics {
        dependence_length: 1,
        effective_sample_size: 100,
        stationarity_ratio: 1.0,
        stationarity_ok: true,
        projection_mismatch_q: 50.0,
        projection_mismatch_ok: false,
        outlier_rate_baseline: 0.001,
        outlier_rate_sample: 0.001,
        outlier_asymmetry_ok: true,
        discrete_mode: false,
        timer_resolution_ns: 1.0,
        duplicate_fraction: 0.0,
        preflight_ok: true,
        calibration_samples: 5000,
        total_time_secs: 1.0,
        warnings: vec![],
        quality_issues: vec![],
        preflight_warnings: vec![],
        ..Diagnostics::all_ok()
    };
    assert!(!diag2.all_checks_passed());

    // outlier_asymmetry_ok = false
    let diag3 = Diagnostics {
        dependence_length: 1,
        effective_sample_size: 100,
        stationarity_ratio: 1.0,
        stationarity_ok: true,
        projection_mismatch_q: 5.0,
        projection_mismatch_ok: true,
        outlier_rate_baseline: 0.1,
        outlier_rate_sample: 0.001,
        outlier_asymmetry_ok: false,
        discrete_mode: false,
        timer_resolution_ns: 1.0,
        duplicate_fraction: 0.0,
        preflight_ok: true,
        calibration_samples: 5000,
        total_time_secs: 1.0,
        warnings: vec![],
        quality_issues: vec![],
        preflight_warnings: vec![],
        ..Diagnostics::all_ok()
    };
    assert!(!diag3.all_checks_passed());
}

#[test]
fn diagnostics_all_checks_passed_all_false() {
    let diag = Diagnostics {
        dependence_length: 1,
        effective_sample_size: 100,
        stationarity_ratio: 10.0,
        stationarity_ok: false,
        projection_mismatch_q: 50.0,
        projection_mismatch_ok: false,
        outlier_rate_baseline: 0.1,
        outlier_rate_sample: 0.001,
        outlier_asymmetry_ok: false,
        discrete_mode: false,
        timer_resolution_ns: 1.0,
        duplicate_fraction: 0.0,
        preflight_ok: false,
        calibration_samples: 5000,
        total_time_secs: 1.0,
        warnings: vec!["warning".to_string()],
        quality_issues: vec![],
        preflight_warnings: vec![],
        ..Diagnostics::all_ok()
    };
    assert!(!diag.all_checks_passed());
}

// ============================================================================
// EffectEstimate tests
// ============================================================================

#[test]
fn effect_estimate_total_effect() {
    let effect = EffectEstimate {
        shift_ns: 3.0,
        tail_ns: 4.0,
        credible_interval_ns: (0.0, 10.0),
        pattern: EffectPattern::Mixed,
        interpretation_caveat: None,
    };
    // sqrt(3^2 + 4^2) = 5
    assert!((effect.total_effect_ns() - 5.0).abs() < 0.001);
}

#[test]
fn effect_estimate_is_negligible() {
    let effect = EffectEstimate {
        shift_ns: 5.0,
        tail_ns: 5.0,
        credible_interval_ns: (0.0, 15.0),
        pattern: EffectPattern::Mixed,
        interpretation_caveat: None,
    };

    assert!(!effect.is_negligible(4.0)); // Both components > 4
    assert!(effect.is_negligible(10.0)); // Both components < 10
}

#[test]
fn effect_estimate_default() {
    let effect = EffectEstimate::default();
    assert_eq!(effect.shift_ns, 0.0);
    assert_eq!(effect.tail_ns, 0.0);
    assert_eq!(effect.pattern, EffectPattern::Indeterminate);
}

// ============================================================================
// Outcome::is_reliable() tests
// ============================================================================

fn make_pass(leak_prob: f64, quality: MeasurementQuality) -> Outcome {
    Outcome::Pass {
        leak_probability: leak_prob,
        effect: EffectEstimate::default(),
        samples_used: 10000,
        quality,
        diagnostics: Diagnostics::all_ok(),
        theta_user: 100.0,
        theta_eff: 100.0,
        theta_floor: 0.0,
    }
}

fn make_fail(leak_prob: f64, quality: MeasurementQuality) -> Outcome {
    Outcome::Fail {
        leak_probability: leak_prob,
        effect: EffectEstimate::default(),
        exploitability: Exploitability::SharedHardwareOnly,
        samples_used: 10000,
        quality,
        diagnostics: Diagnostics::all_ok(),
        theta_user: 100.0,
        theta_eff: 100.0,
        theta_floor: 0.0,
    }
}

fn make_inconclusive(leak_prob: f64, quality: MeasurementQuality) -> Outcome {
    Outcome::Inconclusive {
        reason: InconclusiveReason::DataTooNoisy {
            message: "test".to_string(),
            guidance: "test".to_string(),
        },
        leak_probability: leak_prob,
        effect: EffectEstimate::default(),
        samples_used: 5000,
        quality,
        diagnostics: Diagnostics::all_ok(),
        theta_user: 100.0,
        theta_eff: 100.0,
        theta_floor: 0.0,
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
fn outcome_is_reliable_inconclusive() {
    // Inconclusive is always unreliable
    let outcome = make_inconclusive(0.5, MeasurementQuality::Good);
    assert!(!outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_good_quality() {
    let pass = make_pass(0.02, MeasurementQuality::Good);
    assert!(pass.is_reliable());

    let fail = make_fail(0.98, MeasurementQuality::Good);
    assert!(fail.is_reliable());
}

#[test]
fn outcome_is_reliable_excellent_quality() {
    let outcome = make_pass(0.03, MeasurementQuality::Excellent);
    assert!(outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_poor_quality() {
    // Poor quality is not TooNoisy, so it's still reliable
    let outcome = make_fail(0.97, MeasurementQuality::Poor);
    assert!(outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_too_noisy_pass_conclusive() {
    // TooNoisy but conclusive (< 0.01) - reliable because signal overcame noise
    let outcome = make_pass(0.005, MeasurementQuality::TooNoisy);
    assert!(outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_too_noisy_fail_conclusive() {
    // TooNoisy but conclusive (> 0.99) - reliable because signal overcame noise
    let outcome = make_fail(0.995, MeasurementQuality::TooNoisy);
    assert!(outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_too_noisy_pass_not_conclusive() {
    // TooNoisy AND not very conclusive - NOT reliable
    let outcome = make_pass(0.04, MeasurementQuality::TooNoisy);
    assert!(!outcome.is_reliable());
}

#[test]
fn outcome_is_reliable_too_noisy_fail_not_conclusive() {
    // TooNoisy AND not very conclusive - NOT reliable
    let outcome = make_fail(0.96, MeasurementQuality::TooNoisy);
    assert!(!outcome.is_reliable());
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
// Serialization round-trip tests
// ============================================================================

#[test]
fn effect_estimate_json_roundtrip() {
    let effect = EffectEstimate {
        shift_ns: 15.5,
        tail_ns: 3.2,
        credible_interval_ns: (10.0, 20.0),
        pattern: EffectPattern::UniformShift,
        interpretation_caveat: None,
    };
    let json = serde_json::to_string(&effect).unwrap();
    let deserialized: EffectEstimate = serde_json::from_str(&json).unwrap();

    assert_eq!(effect.shift_ns, deserialized.shift_ns);
    assert_eq!(effect.tail_ns, deserialized.tail_ns);
    assert_eq!(effect.pattern, deserialized.pattern);
}

#[test]
fn exploitability_json_roundtrip() {
    for variant in [
        Exploitability::SharedHardwareOnly,
        Exploitability::Http2Multiplexing,
        Exploitability::StandardRemote,
        Exploitability::ObviousLeak,
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
fn outcome_pass_json_roundtrip() {
    let outcome = make_pass(0.02, MeasurementQuality::Excellent);
    let json = serde_json::to_string(&outcome).unwrap();
    let deserialized: Outcome = serde_json::from_str(&json).unwrap();

    match deserialized {
        Outcome::Pass {
            leak_probability,
            quality,
            ..
        } => {
            assert!((leak_probability - 0.02).abs() < 0.001);
            assert_eq!(quality, MeasurementQuality::Excellent);
        }
        _ => panic!("Expected Pass variant"),
    }
}

#[test]
fn outcome_fail_json_roundtrip() {
    let outcome = make_fail(0.98, MeasurementQuality::Good);
    let json = serde_json::to_string(&outcome).unwrap();
    let deserialized: Outcome = serde_json::from_str(&json).unwrap();

    match deserialized {
        Outcome::Fail {
            leak_probability,
            quality,
            ..
        } => {
            assert!((leak_probability - 0.98).abs() < 0.001);
            assert_eq!(quality, MeasurementQuality::Good);
        }
        _ => panic!("Expected Fail variant"),
    }
}

#[test]
fn outcome_inconclusive_json_roundtrip() {
    let outcome = make_inconclusive(0.5, MeasurementQuality::Poor);
    let json = serde_json::to_string(&outcome).unwrap();
    let deserialized: Outcome = serde_json::from_str(&json).unwrap();

    match deserialized {
        Outcome::Inconclusive {
            leak_probability,
            quality,
            ..
        } => {
            assert!((leak_probability - 0.5).abs() < 0.001);
            assert_eq!(quality, MeasurementQuality::Poor);
        }
        _ => panic!("Expected Inconclusive variant"),
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
        Outcome::Unmeasurable {
            operation_ns,
            platform,
            ..
        } => {
            assert_eq!(operation_ns, 5.5);
            assert_eq!(platform, "test platform");
        }
        _ => panic!("Expected Unmeasurable variant"),
    }
}

// ============================================================================
// AttackerModel tests
// ============================================================================

use tacet::AttackerModel;

#[test]
fn attacker_model_default_is_adjacent_network() {
    assert_eq!(AttackerModel::default(), AttackerModel::AdjacentNetwork);
}

#[test]
fn attacker_model_to_threshold_ns_presets() {
    // SharedHardware: 0.4ns (~2 cycles @ 5 GHz)
    assert!((AttackerModel::SharedHardware.to_threshold_ns() - 0.4).abs() < 0.01);

    // AdjacentNetwork: 100ns
    assert_eq!(AttackerModel::AdjacentNetwork.to_threshold_ns(), 100.0);

    // RemoteNetwork: 50μs
    assert_eq!(AttackerModel::RemoteNetwork.to_threshold_ns(), 50_000.0);

    // Research: 0 (detect any difference)
    assert_eq!(AttackerModel::Research.to_threshold_ns(), 0.0);
}

#[test]
fn attacker_model_custom() {
    let model = AttackerModel::Custom {
        threshold_ns: 250.0,
    };
    assert_eq!(model.to_threshold_ns(), 250.0);
}

#[test]
fn attacker_model_description() {
    // Just verify descriptions are non-empty and descriptive
    assert!(AttackerModel::SharedHardware
        .description()
        .contains("hardware"));
    assert!(
        AttackerModel::AdjacentNetwork
            .description()
            .contains("network")
            || AttackerModel::AdjacentNetwork.description().contains("LAN")
    );
    assert!(
        AttackerModel::RemoteNetwork
            .description()
            .contains("remote")
            || AttackerModel::RemoteNetwork
                .description()
                .contains("internet")
    );
    assert!(
        AttackerModel::Research.description().contains("research")
            || AttackerModel::Research.description().contains("any")
    );
}

// ============================================================================
// IssueCode tests (v5.6)
// ============================================================================

#[test]
fn issue_code_kappa_mixing_poor_exists() {
    // v5.6: Verify the new IssueCode variant exists and serializes correctly
    let issue = QualityIssue {
        code: IssueCode::KappaMixingPoor,
        message: "κ chain mixing poor (CV=0.05, ESS=10)".to_string(),
        guidance: "Posterior may be unreliable; consider longer time budget.".to_string(),
    };

    let json = serde_json::to_string(&issue).unwrap();
    assert!(json.contains("KappaMixingPoor"));

    let deserialized: QualityIssue = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.code, IssueCode::KappaMixingPoor);
}

#[test]
fn issue_code_likelihood_inflated_exists() {
    // v5.6: Verify the new IssueCode variant exists and serializes correctly
    let issue = QualityIssue {
        code: IssueCode::LikelihoodInflated,
        message: "Likelihood covariance inflated ~5.0x due to data/model mismatch".to_string(),
        guidance: "Uncertainty was increased for robustness. Effect estimates remain valid."
            .to_string(),
    };

    let json = serde_json::to_string(&issue).unwrap();
    assert!(json.contains("LikelihoodInflated"));

    let deserialized: QualityIssue = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.code, IssueCode::LikelihoodInflated);
}

#[test]
fn issue_code_json_roundtrip_all_variants() {
    // Test all IssueCode variants serialize and deserialize correctly
    let variants = vec![
        IssueCode::ThresholdElevated,
        IssueCode::LambdaMixingPoor,
        IssueCode::KappaMixingPoor,
        IssueCode::LikelihoodInflated,
    ];

    for code in variants {
        let issue = QualityIssue {
            code,
            message: "test message".to_string(),
            guidance: "test guidance".to_string(),
        };

        let json = serde_json::to_string(&issue).unwrap();
        let deserialized: QualityIssue = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.code, issue.code);
        assert_eq!(deserialized.message, "test message");
    }
}
