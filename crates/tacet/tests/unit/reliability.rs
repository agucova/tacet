//! Tests for the reliability handling API (Outcome, UnreliablePolicy, skip_if_unreliable!).

use tacet::{
    Diagnostics, EffectEstimate, Exploitability, InconclusiveReason, MeasurementQuality, Outcome,
    UnreliablePolicy,
};

/// Create a Pass outcome with the given parameters.
fn make_pass(leak_probability: f64, quality: MeasurementQuality) -> Outcome {
    Outcome::Pass {
        leak_probability,
        effect: EffectEstimate {
            max_effect_ns: 7.0,
            credible_interval_ns: (0.0, 10.0),
            top_quantiles: Vec::new(),
        },
        samples_used: 10000,
        quality,
        diagnostics: Diagnostics::all_ok(),
        theta_user: 100.0,
        theta_eff: 100.0,
        theta_floor: 0.0,
    }
}

/// Create a Fail outcome with the given parameters.
fn make_fail(leak_probability: f64, quality: MeasurementQuality) -> Outcome {
    Outcome::Fail {
        leak_probability,
        effect: EffectEstimate {
            max_effect_ns: 150.0,
            credible_interval_ns: (80.0, 120.0),
            top_quantiles: Vec::new(),
        },
        exploitability: Exploitability::Http2Multiplexing,
        samples_used: 10000,
        quality,
        diagnostics: Diagnostics::all_ok(),
        theta_user: 100.0,
        theta_eff: 100.0,
        theta_floor: 0.0,
    }
}

/// Create an Inconclusive outcome.
fn make_inconclusive(leak_probability: f64, quality: MeasurementQuality) -> Outcome {
    Outcome::Inconclusive {
        reason: InconclusiveReason::DataTooNoisy {
            message: "Test message".to_string(),
            guidance: "Test guidance".to_string(),
        },
        leak_probability,
        effect: EffectEstimate::default(),
        samples_used: 5000,
        quality,
        diagnostics: Diagnostics::all_ok(),
        theta_user: 100.0,
        theta_eff: 100.0,
        theta_floor: 0.0,
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
fn is_reliable_inconclusive_returns_false() {
    let outcome = make_inconclusive(0.5, MeasurementQuality::Good);
    assert!(!outcome.is_reliable());
}

#[test]
fn is_reliable_too_noisy_pass_inconclusive_returns_false() {
    // TooNoisy quality with non-conclusive posterior (not < 0.01) should be unreliable
    let outcome = make_pass(0.04, MeasurementQuality::TooNoisy);
    assert!(!outcome.is_reliable());
}

#[test]
fn is_reliable_too_noisy_fail_inconclusive_returns_false() {
    // TooNoisy quality with non-conclusive posterior (not > 0.99) should be unreliable
    let outcome = make_fail(0.96, MeasurementQuality::TooNoisy);
    assert!(!outcome.is_reliable());
}

#[test]
fn is_reliable_too_noisy_pass_but_conclusive_returns_true() {
    // TooNoisy but posterior < 0.01 is still reliable (confidently no leak)
    let outcome = make_pass(0.005, MeasurementQuality::TooNoisy);
    assert!(outcome.is_reliable());
}

#[test]
fn is_reliable_too_noisy_fail_but_conclusive_returns_true() {
    // TooNoisy but posterior > 0.99 is still reliable (signal overcame noise)
    let outcome = make_fail(0.995, MeasurementQuality::TooNoisy);
    assert!(outcome.is_reliable());
}

#[test]
fn is_reliable_good_quality_pass_returns_true() {
    // Good quality, any posterior, should be reliable
    let outcome = make_pass(0.03, MeasurementQuality::Good);
    assert!(outcome.is_reliable());
}

#[test]
fn is_reliable_good_quality_fail_returns_true() {
    let outcome = make_fail(0.97, MeasurementQuality::Good);
    assert!(outcome.is_reliable());
}

#[test]
fn is_reliable_excellent_quality_returns_true() {
    let outcome = make_pass(0.02, MeasurementQuality::Excellent);
    assert!(outcome.is_reliable());
}

#[test]
fn is_reliable_poor_quality_returns_true() {
    // Poor quality is not TooNoisy, so it's still considered reliable
    let outcome = make_fail(0.98, MeasurementQuality::Poor);
    assert!(outcome.is_reliable());
}

// ============================================================================
// Outcome::handle_unreliable() tests
// ============================================================================

#[test]
fn handle_unreliable_reliable_pass_returns_some() {
    let outcome = make_pass(0.02, MeasurementQuality::Good);

    let handled = outcome.handle_unreliable("test", UnreliablePolicy::FailOpen);
    assert!(handled.is_some());
    assert!(handled.unwrap().passed());
}

#[test]
fn handle_unreliable_reliable_fail_returns_some() {
    let outcome = make_fail(0.98, MeasurementQuality::Good);

    let handled = outcome.handle_unreliable("test", UnreliablePolicy::FailOpen);
    assert!(handled.is_some());
    assert!(handled.unwrap().failed());
}

#[test]
fn handle_unreliable_fail_open_returns_none() {
    let outcome = make_inconclusive(0.5, MeasurementQuality::Good);

    let handled = outcome.handle_unreliable("test", UnreliablePolicy::FailOpen);
    assert!(handled.is_none());
}

#[test]
#[should_panic(expected = "[FAILED]")]
fn handle_unreliable_fail_closed_panics() {
    let outcome = make_inconclusive(0.5, MeasurementQuality::Good);

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
#[should_panic(expected = "[FAILED]")]
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
// Outcome helper method tests
// ============================================================================

#[test]
fn outcome_passed_returns_true_for_pass() {
    let outcome = make_pass(0.02, MeasurementQuality::Good);
    assert!(outcome.passed());
    assert!(!outcome.failed());
}

#[test]
fn outcome_failed_returns_true_for_fail() {
    let outcome = make_fail(0.98, MeasurementQuality::Good);
    assert!(outcome.failed());
    assert!(!outcome.passed());
}

#[test]
fn outcome_is_conclusive() {
    let pass = make_pass(0.02, MeasurementQuality::Good);
    assert!(pass.is_conclusive());

    let fail = make_fail(0.98, MeasurementQuality::Good);
    assert!(fail.is_conclusive());

    let inconclusive = make_inconclusive(0.5, MeasurementQuality::Good);
    assert!(!inconclusive.is_conclusive());

    let unmeasurable = Outcome::Unmeasurable {
        operation_ns: 10.0,
        threshold_ns: 100.0,
        platform: "test".to_string(),
        recommendation: "test".to_string(),
    };
    assert!(!unmeasurable.is_conclusive());
}

#[test]
fn outcome_is_measurable() {
    let pass = make_pass(0.02, MeasurementQuality::Good);
    assert!(pass.is_measurable());

    let fail = make_fail(0.98, MeasurementQuality::Good);
    assert!(fail.is_measurable());

    let inconclusive = make_inconclusive(0.5, MeasurementQuality::Good);
    assert!(inconclusive.is_measurable());

    let unmeasurable = Outcome::Unmeasurable {
        operation_ns: 10.0,
        threshold_ns: 100.0,
        platform: "test".to_string(),
        recommendation: "test".to_string(),
    };
    assert!(!unmeasurable.is_measurable());
}

#[test]
fn outcome_leak_probability_returns_value() {
    let pass = make_pass(0.02, MeasurementQuality::Good);
    assert_eq!(pass.leak_probability(), Some(0.02));

    let fail = make_fail(0.98, MeasurementQuality::Good);
    assert_eq!(fail.leak_probability(), Some(0.98));

    let inconclusive = make_inconclusive(0.5, MeasurementQuality::Good);
    assert_eq!(inconclusive.leak_probability(), Some(0.5));

    let unmeasurable = Outcome::Unmeasurable {
        operation_ns: 10.0,
        threshold_ns: 100.0,
        platform: "test".to_string(),
        recommendation: "test".to_string(),
    };
    assert_eq!(unmeasurable.leak_probability(), None);
}

#[test]
fn outcome_quality_returns_value() {
    let pass = make_pass(0.02, MeasurementQuality::Excellent);
    assert_eq!(pass.quality(), Some(MeasurementQuality::Excellent));

    let fail = make_fail(0.98, MeasurementQuality::Poor);
    assert_eq!(fail.quality(), Some(MeasurementQuality::Poor));

    let unmeasurable = Outcome::Unmeasurable {
        operation_ns: 10.0,
        threshold_ns: 100.0,
        platform: "test".to_string(),
        recommendation: "test".to_string(),
    };
    assert_eq!(unmeasurable.quality(), None);
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
// Macro tests (basic functionality)
// ============================================================================

#[test]
fn skip_if_unreliable_macro_skips_unreliable() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static REACHED_END: AtomicBool = AtomicBool::new(false);

    fn test_fn() {
        let outcome = make_inconclusive(0.5, MeasurementQuality::Good);
        let _result = tacet::skip_if_unreliable!(outcome, "test");
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
fn skip_if_unreliable_macro_returns_outcome_when_reliable() {
    let outcome = make_pass(0.02, MeasurementQuality::Good);

    // This should not skip
    let returned = tacet::skip_if_unreliable!(outcome, "test");
    assert!(returned.passed());
}

#[test]
fn require_reliable_macro_returns_outcome_when_reliable() {
    let outcome = make_fail(0.98, MeasurementQuality::Good);

    // This should return the outcome
    let returned = tacet::require_reliable!(outcome, "test");
    assert!(returned.failed());
}

#[test]
#[should_panic(expected = "[FAILED]")]
fn require_reliable_macro_panics_when_unreliable() {
    let outcome = make_inconclusive(0.5, MeasurementQuality::Good);

    // This should panic
    let _ = tacet::require_reliable!(outcome, "test");
}
