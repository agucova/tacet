//! Tests for the reliability handling API (Outcome, UnreliablePolicy, skip_if_unreliable!).

use timing_oracle::{
    BatchingInfo, CiGate, CiGateResult, Diagnostics, Effect, EffectPattern, Exploitability, MeasurementQuality,
    Metadata, MinDetectableEffect, Outcome, TestResult, UnreliablePolicy,
};

/// Create a test result with the given parameters.
fn make_result(leak_probability: f64, quality: MeasurementQuality) -> TestResult {
    TestResult {
        leak_probability,
        effect: Some(Effect {
            shift_ns: 100.0,
            tail_ns: 50.0,
            shift_ci_ns: (80.0, 120.0),
            tail_ci_ns: (30.0, 70.0),
            credible_interval_ns: (80.0, 120.0),
            pattern: EffectPattern::Mixed,
        }),
        exploitability: Exploitability::PossibleLAN,
        min_detectable_effect: MinDetectableEffect {
            shift_ns: 5.0,
            tail_ns: 10.0,
        },
        ci_gate: CiGate {
            alpha: 0.01,
            result: CiGateResult::Fail,
            threshold: 0.0,
            max_observed: 0.0,
            observed: [0.0; 9],
            p_value: 0.01,
        },
        quality,
        outlier_fraction: 0.01,
        diagnostics: Diagnostics::all_ok(),
        metadata: Metadata {
            samples_per_class: 10000,
            cycles_per_ns: 3.0,
            timer: "rdtsc".to_string(),
            timer_resolution_ns: 0.33,
            batching: BatchingInfo {
                enabled: false,
                k: 1,
                ticks_per_batch: 100.0,
                rationale: "no batching needed".to_string(),
                unmeasurable: None,
            },
            runtime_secs: 1.0,
        },
    }
}

// ============================================================================
// Outcome::is_reliable() tests
// ============================================================================

#[test]
fn is_reliable_unmeasurable_returns_false() {
    let outcome = Outcome::Unmeasurable {
        operation_ns: 15.0,
        threshold_ns: 200.0,
        platform: "Apple Silicon".to_string(),
        recommendation: "Use x86_64".to_string(),
    };
    assert!(!outcome.is_reliable());
}

#[test]
fn is_reliable_too_noisy_inconclusive_returns_false() {
    // TooNoisy quality with inconclusive posterior (0.5) should be unreliable
    let result = make_result(0.5, MeasurementQuality::TooNoisy);
    let outcome = Outcome::Completed(result);
    assert!(!outcome.is_reliable());
}

#[test]
fn is_reliable_too_noisy_but_conclusive_high_returns_true() {
    // TooNoisy but posterior > 0.9 is still reliable (signal overcame noise)
    let result = make_result(0.95, MeasurementQuality::TooNoisy);
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

#[test]
fn is_reliable_too_noisy_but_conclusive_low_returns_true() {
    // TooNoisy but posterior < 0.1 is still reliable (confidently no leak)
    let result = make_result(0.05, MeasurementQuality::TooNoisy);
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

#[test]
fn is_reliable_good_quality_returns_true() {
    // Good quality, any posterior, should be reliable
    let result = make_result(0.5, MeasurementQuality::Good);
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

#[test]
fn is_reliable_excellent_quality_returns_true() {
    let result = make_result(0.3, MeasurementQuality::Excellent);
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

#[test]
fn is_reliable_poor_quality_returns_true() {
    // Poor quality is not TooNoisy, so it's still considered reliable
    let result = make_result(0.7, MeasurementQuality::Poor);
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

#[test]
fn is_reliable_zero_mde_returns_false_even_if_conclusive() {
    // MDE of 0.0 indicates timer resolution failure - even conclusive posteriors are garbage
    let mut result = make_result(0.99, MeasurementQuality::TooNoisy);
    result.min_detectable_effect.shift_ns = 0.0;
    let outcome = Outcome::Completed(result);
    assert!(!outcome.is_reliable());
}

#[test]
fn is_reliable_nan_mde_returns_false() {
    // NaN MDE indicates measurement failure
    let mut result = make_result(0.99, MeasurementQuality::Good);
    result.min_detectable_effect.shift_ns = f64::NAN;
    let outcome = Outcome::Completed(result);
    assert!(!outcome.is_reliable());
}

#[test]
fn is_reliable_infinite_mde_returns_false() {
    // Infinite MDE indicates measurement failure
    let mut result = make_result(0.01, MeasurementQuality::Good);
    result.min_detectable_effect.shift_ns = f64::INFINITY;
    let outcome = Outcome::Completed(result);
    assert!(!outcome.is_reliable());
}

#[test]
fn is_reliable_valid_mde_with_conclusive_posterior_returns_true() {
    // Valid MDE (> 0.01ns) with conclusive posterior is reliable
    let mut result = make_result(0.99, MeasurementQuality::TooNoisy);
    result.min_detectable_effect.shift_ns = 1.0; // Valid MDE
    let outcome = Outcome::Completed(result);
    assert!(outcome.is_reliable());
}

// ============================================================================
// Outcome::handle_unreliable() tests
// ============================================================================

#[test]
fn handle_unreliable_reliable_returns_some() {
    let result = make_result(0.95, MeasurementQuality::Good);
    let outcome = Outcome::Completed(result);

    let handled = outcome.handle_unreliable("test", UnreliablePolicy::FailOpen);
    assert!(handled.is_some());
    assert_eq!(handled.unwrap().leak_probability, 0.95);
}

#[test]
fn handle_unreliable_fail_open_returns_none() {
    let result = make_result(0.5, MeasurementQuality::TooNoisy);
    let outcome = Outcome::Completed(result);

    let handled = outcome.handle_unreliable("test", UnreliablePolicy::FailOpen);
    assert!(handled.is_none());
}

#[test]
#[should_panic(expected = "[UNRELIABLE]")]
fn handle_unreliable_fail_closed_panics() {
    let result = make_result(0.5, MeasurementQuality::TooNoisy);
    let outcome = Outcome::Completed(result);

    let _ = outcome.handle_unreliable("test", UnreliablePolicy::FailClosed);
}

#[test]
fn handle_unreliable_unmeasurable_fail_open_returns_none() {
    let outcome = Outcome::Unmeasurable {
        operation_ns: 15.0,
        threshold_ns: 200.0,
        platform: "Apple Silicon".to_string(),
        recommendation: "Use x86_64".to_string(),
    };

    let handled = outcome.handle_unreliable("test", UnreliablePolicy::FailOpen);
    assert!(handled.is_none());
}

#[test]
#[should_panic(expected = "[UNRELIABLE]")]
fn handle_unreliable_unmeasurable_fail_closed_panics() {
    let outcome = Outcome::Unmeasurable {
        operation_ns: 15.0,
        threshold_ns: 200.0,
        platform: "Apple Silicon".to_string(),
        recommendation: "Use x86_64".to_string(),
    };

    let _ = outcome.handle_unreliable("test", UnreliablePolicy::FailClosed);
}

// ============================================================================
// Outcome::unwrap_completed() and completed() tests
// ============================================================================

#[test]
fn unwrap_completed_returns_result() {
    let result = make_result(0.8, MeasurementQuality::Good);
    let outcome = Outcome::Completed(result);

    let unwrapped = outcome.unwrap_completed();
    assert_eq!(unwrapped.leak_probability, 0.8);
}

#[test]
#[should_panic(expected = "unmeasurable")]
fn unwrap_completed_panics_on_unmeasurable() {
    let outcome = Outcome::Unmeasurable {
        operation_ns: 15.0,
        threshold_ns: 200.0,
        platform: "Test Platform".to_string(),
        recommendation: "N/A".to_string(),
    };

    let _ = outcome.unwrap_completed();
}

#[test]
fn completed_returns_some_for_completed() {
    let result = make_result(0.8, MeasurementQuality::Good);
    let outcome = Outcome::Completed(result);

    let completed = outcome.completed();
    assert!(completed.is_some());
}

#[test]
fn completed_returns_none_for_unmeasurable() {
    let outcome = Outcome::Unmeasurable {
        operation_ns: 15.0,
        threshold_ns: 200.0,
        platform: "Test".to_string(),
        recommendation: "N/A".to_string(),
    };

    let completed = outcome.completed();
    assert!(completed.is_none());
}

// ============================================================================
// UnreliablePolicy tests
// ============================================================================

#[test]
fn unreliable_policy_default_is_fail_open() {
    assert_eq!(UnreliablePolicy::default(), UnreliablePolicy::FailOpen);
}

#[test]
fn unreliable_policy_from_env_unset_returns_default() {
    // Temporarily unset the env var if it exists
    let original = std::env::var("TIMING_ORACLE_UNRELIABLE_POLICY").ok();
    std::env::remove_var("TIMING_ORACLE_UNRELIABLE_POLICY");

    let policy = UnreliablePolicy::from_env_or(UnreliablePolicy::FailClosed);
    assert_eq!(policy, UnreliablePolicy::FailClosed);

    // Restore original
    if let Some(val) = original {
        std::env::set_var("TIMING_ORACLE_UNRELIABLE_POLICY", val);
    }
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

// ============================================================================
// TestResult::can_detect() tests
// ============================================================================

#[test]
fn can_detect_true_when_mde_below_threshold() {
    let result = make_result(0.5, MeasurementQuality::Good);
    // MDE shift is 5.0, so can detect effects >= 5.0
    assert!(result.can_detect(5.0));
    assert!(result.can_detect(10.0));
    assert!(result.can_detect(100.0));
}

#[test]
fn can_detect_false_when_mde_above_threshold() {
    let result = make_result(0.5, MeasurementQuality::Good);
    // MDE shift is 5.0, so cannot detect effects < 5.0
    assert!(!result.can_detect(4.0));
    assert!(!result.can_detect(1.0));
}

// ============================================================================
// Macro tests (basic functionality)
// ============================================================================

#[test]
fn skip_if_unreliable_macro_skips_unreliable() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static REACHED_END: AtomicBool = AtomicBool::new(false);

    fn test_fn() {
        let result = make_result(0.5, MeasurementQuality::TooNoisy);
        let outcome = Outcome::Completed(result);
        let _result = timing_oracle::skip_if_unreliable!(outcome, "test");
        // If we get here, macro didn't skip
        REACHED_END.store(true, Ordering::SeqCst);
    }

    REACHED_END.store(false, Ordering::SeqCst);
    test_fn();
    // The function should have returned early (skipped), so REACHED_END should be false
    assert!(
        !REACHED_END.load(Ordering::SeqCst),
        "macro should have skipped unreliable test"
    );
}

#[test]
fn skip_if_unreliable_macro_returns_result_when_reliable() {
    let result = make_result(0.95, MeasurementQuality::Good);
    let outcome = Outcome::Completed(result);

    // This should not skip
    let returned = timing_oracle::skip_if_unreliable!(outcome, "test");
    assert_eq!(returned.leak_probability, 0.95);
}

#[test]
fn require_reliable_macro_returns_result_when_reliable() {
    let result = make_result(0.95, MeasurementQuality::Good);
    let outcome = Outcome::Completed(result);

    // This should return the result
    let returned = timing_oracle::require_reliable!(outcome, "test");
    assert_eq!(returned.leak_probability, 0.95);
}

#[test]
#[should_panic(expected = "[UNRELIABLE]")]
fn require_reliable_macro_panics_when_unreliable() {
    let result = make_result(0.5, MeasurementQuality::TooNoisy);
    let outcome = Outcome::Completed(result);

    // This should panic
    let _ = timing_oracle::require_reliable!(outcome, "test");
}
