//! Integration tests for timing-oracle-c FFI API.
//!
//! These tests verify the C API works correctly end-to-end.

use std::ffi::c_void;
use std::ptr;

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

// Import the C FFI functions
use timing_oracle_c::*;

// =============================================================================
// Test Helpers: C Callback Functions
// =============================================================================

/// Context for test generators that need RNG state
struct TestContext {
    rng: StdRng,
}

/// Generator that produces zeros for baseline, random for sample
unsafe extern "C" fn random_generator(
    ctx: *mut c_void,
    is_baseline: bool,
    output: *mut u8,
    output_size: usize,
) {
    let slice = std::slice::from_raw_parts_mut(output, output_size);
    if is_baseline {
        slice.fill(0);
    } else {
        let test_ctx = &mut *(ctx as *mut TestContext);
        test_ctx.rng.fill_bytes(slice);
    }
}

/// Generator that produces all zeros (both classes identical)
unsafe extern "C" fn zeros_generator(
    _ctx: *mut c_void,
    _is_baseline: bool,
    output: *mut u8,
    output_size: usize,
) {
    let slice = std::slice::from_raw_parts_mut(output, output_size);
    slice.fill(0);
}

/// Constant-time XOR operation (should pass)
unsafe extern "C" fn xor_operation(_ctx: *mut c_void, input: *const u8, input_size: usize) {
    let data = std::slice::from_raw_parts(input, input_size);
    let mut acc = 0u8;
    for &byte in data {
        acc ^= byte;
    }
    std::hint::black_box(acc);
}

/// Constant-time bitwise accumulator (should pass)
unsafe extern "C" fn bitwise_accumulator(_ctx: *mut c_void, input: *const u8, input_size: usize) {
    let data = std::slice::from_raw_parts(input, input_size);
    let mut acc = 0u8;
    for &byte in data {
        acc |= byte;
        acc &= !byte | acc;
    }
    std::hint::black_box(acc);
}

/// Early-exit comparison (LEAKY - should fail)
unsafe extern "C" fn early_exit_operation(_ctx: *mut c_void, input: *const u8, input_size: usize) {
    let data = std::slice::from_raw_parts(input, input_size);
    // Early exit on first non-zero byte - this is a timing leak!
    for &byte in data {
        if byte != 0 {
            return;
        }
    }
}

/// Branch-based timing (LEAKY - should fail)
unsafe extern "C" fn branch_operation(_ctx: *mut c_void, input: *const u8, input_size: usize) {
    let data = std::slice::from_raw_parts(input, input_size);
    let mut sum = 0u64;
    for &byte in data {
        // Data-dependent branch - this is a timing leak!
        if byte > 127 {
            sum += byte as u64;
        } else {
            sum += 1;
        }
    }
    std::hint::black_box(sum);
}

/// No-op operation (very fast, likely unmeasurable)
unsafe extern "C" fn noop_operation(_ctx: *mut c_void, _input: *const u8, _input_size: usize) {
    std::hint::black_box(());
}

// =============================================================================
// Group 5: Basic API Tests
// =============================================================================

#[test]
fn test_to_test_basic() {
    let config = to_config_default(ToAttackerModel::AdjacentNetwork);
    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(12345),
    };

    // Use a short time budget for testing
    let mut config = config;
    config.time_budget_secs = 5.0;
    config.max_samples = 5000;

    let result = unsafe {
        to_test(
            &config,
            64, // input_size
            random_generator,
            xor_operation,
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // Verify result fields are populated
    assert!(result.samples_used > 0, "Should have used some samples");
    assert!(result.elapsed_secs > 0.0, "Should have taken some time");
    assert!(
        result.leak_probability >= 0.0 && result.leak_probability <= 1.0,
        "Leak probability should be in [0, 1]"
    );
    // Note: In Rust tests, we don't call to_result_free - it's for C callers.
    // The recommendation string may leak, but that's acceptable in tests.
}

#[test]
fn test_to_test_with_time() {
    let config = to_config_default(ToAttackerModel::AdjacentNetwork);
    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(12345),
    };

    let mut config = config;
    config.time_budget_secs = 3.0;
    config.max_samples = 3000;

    // Caller-managed time tracking
    let start = std::time::Instant::now();

    let result = unsafe {
        // Update elapsed time before calling
        let elapsed_secs = start.elapsed().as_secs_f64();
        to_test_with_time(
            &config,
            64,
            random_generator,
            xor_operation,
            &mut ctx as *mut TestContext as *mut c_void,
            &elapsed_secs,
        )
    };

    // Verify it completed
    assert!(result.samples_used > 0 || result.outcome == ToOutcome::Unmeasurable);
}

#[test]
fn test_result_free_safe() {
    // Test that to_result_free handles various cases safely

    // 1. Null pointer should be handled gracefully
    unsafe { to_result_free(ptr::null_mut()) }; // Should not crash

    // 2. Normal result with null recommendation
    let mut result = ToResult::default();
    unsafe { to_result_free(&mut result as *mut ToResult) }; // Should not crash

    // 3. Result from actual test (may have recommendation)
    let config = to_config_default(ToAttackerModel::AdjacentNetwork);
    let mut config = config;
    config.time_budget_secs = 1.0;
    config.max_samples = 1000;

    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(99999),
    };

    let mut result = unsafe {
        to_test(
            &config,
            32,
            random_generator,
            xor_operation,
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // Free should handle recommendation string if present
    unsafe { to_result_free(&mut result as *mut ToResult) };
}

#[test]
fn test_null_config_uses_defaults() {
    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(54321),
    };

    // Pass null config - should use defaults
    let result = unsafe {
        to_test(
            ptr::null(),
            64,
            random_generator,
            xor_operation,
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // Should complete (uses default 30s budget, so will run for a while)
    // Just verify it returns something reasonable
    assert!(
        result.leak_probability >= 0.0 && result.leak_probability <= 1.0,
        "Leak probability should be valid even with null config"
    );

    // Note: to_result_free is for C callers; in Rust tests we skip it
}

// =============================================================================
// Group 6: Known-Safe Tests
// =============================================================================

#[test]
fn test_xor_constant_time() {
    let mut config = to_config_default(ToAttackerModel::AdjacentNetwork);
    config.time_budget_secs = 10.0;
    config.max_samples = 10000;
    // Relaxed thresholds to reduce false positives
    config.pass_threshold = 0.15;
    config.fail_threshold = 0.99;

    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(11111),
    };

    let result = unsafe {
        to_test(
            &config,
            256, // Larger input for better measurability
            random_generator,
            xor_operation,
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // XOR is constant-time, should NOT fail
    // Accept Pass, Inconclusive, or Unmeasurable - just not Fail
    assert_ne!(
        result.outcome,
        ToOutcome::Fail,
        "XOR operation should not be detected as leaky (P(leak)={:.3})",
        result.leak_probability
    );

    // Note: to_result_free is for C callers; in Rust tests we skip it
}

#[test]
fn test_bitwise_accumulator_constant_time() {
    let mut config = to_config_default(ToAttackerModel::AdjacentNetwork);
    config.time_budget_secs = 10.0;
    config.max_samples = 10000;
    config.pass_threshold = 0.15;
    config.fail_threshold = 0.99;

    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(22222),
    };

    let result = unsafe {
        to_test(
            &config,
            256,
            random_generator,
            bitwise_accumulator,
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // Bitwise ops are constant-time, should NOT fail
    assert_ne!(
        result.outcome,
        ToOutcome::Fail,
        "Bitwise accumulator should not be detected as leaky (P(leak)={:.3})",
        result.leak_probability
    );

    // Note: to_result_free is for C callers; in Rust tests we skip it
}

// =============================================================================
// Group 7: Known-Leaky Tests
// =============================================================================

#[test]
fn test_early_exit_detected() {
    let mut config = to_config_default(ToAttackerModel::AdjacentNetwork);
    config.time_budget_secs = 15.0;
    config.max_samples = 20000;
    // Strict thresholds to ensure detection
    config.pass_threshold = 0.01;
    config.fail_threshold = 0.85;

    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(33333),
    };

    let result = unsafe {
        to_test(
            &config,
            512, // Large input to maximize leak visibility
            random_generator,
            early_exit_operation,
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // Early-exit is a timing leak, should be detected
    // Accept Fail or Inconclusive (may not have enough samples)
    // Should NOT pass
    if result.outcome == ToOutcome::Pass {
        panic!(
            "Early-exit leak should be detected, but got Pass with P(leak)={:.3}",
            result.leak_probability
        );
    }

    // If we got Fail, that's the expected result
    if result.outcome == ToOutcome::Fail {
        eprintln!(
            "[test_early_exit_detected] Correctly detected leak with P(leak)={:.3}",
            result.leak_probability
        );
    }

    // Note: to_result_free is for C callers; in Rust tests we skip it
}

#[test]
fn test_branch_timing_detected() {
    let mut config = to_config_default(ToAttackerModel::AdjacentNetwork);
    config.time_budget_secs = 15.0;
    config.max_samples = 20000;
    config.pass_threshold = 0.01;
    config.fail_threshold = 0.85;

    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(44444),
    };

    let result = unsafe {
        to_test(
            &config,
            512,
            random_generator,
            branch_operation,
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // Branch on data is a timing leak
    if result.outcome == ToOutcome::Pass {
        panic!(
            "Branch timing leak should be detected, but got Pass with P(leak)={:.3}",
            result.leak_probability
        );
    }

    if result.outcome == ToOutcome::Fail {
        eprintln!(
            "[test_branch_timing_detected] Correctly detected leak with P(leak)={:.3}",
            result.leak_probability
        );
    }

    // Note: to_result_free is for C callers; in Rust tests we skip it
}

// =============================================================================
// Group 8: Edge Cases & Robustness
// =============================================================================

#[test]
fn test_unmeasurable_fast_op() {
    let mut config = to_config_default(ToAttackerModel::AdjacentNetwork);
    config.time_budget_secs = 5.0;
    config.max_samples = 5000;

    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(55555),
    };

    let result = unsafe {
        to_test(
            &config,
            1, // Very small input
            zeros_generator,
            noop_operation,
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // No-op with tiny input is likely unmeasurable
    // But on some platforms it might still be measurable
    // Just verify we get a valid result
    assert!(
        result.outcome == ToOutcome::Pass
            || result.outcome == ToOutcome::Inconclusive
            || result.outcome == ToOutcome::Unmeasurable,
        "Expected Pass, Inconclusive, or Unmeasurable for fast no-op"
    );

    if result.outcome == ToOutcome::Unmeasurable {
        assert!(
            result.operation_ns >= 0.0,
            "Unmeasurable should report operation time"
        );
    }

    // Note: to_result_free is for C callers; in Rust tests we skip it
}

#[test]
fn test_short_time_budget() {
    let mut config = to_config_default(ToAttackerModel::AdjacentNetwork);
    config.time_budget_secs = 1.0; // Very short
    config.max_samples = 1000;

    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(66666),
    };

    let result = unsafe {
        to_test(
            &config,
            64,
            random_generator,
            xor_operation,
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // With 1 second budget, likely to be Inconclusive or hit budget
    // Just verify we got a valid result (calibration adds overhead beyond budget)
    assert!(
        result.leak_probability >= 0.0 && result.leak_probability <= 1.0,
        "Should get valid leak probability"
    );
    // Budget is best-effort; calibration phase adds overhead
    eprintln!(
        "[test_short_time_budget] Completed in {:.1}s with {} samples",
        result.elapsed_secs, result.samples_used
    );

    // Note: to_result_free is for C callers; in Rust tests we skip it
}

#[test]
fn test_custom_threshold() {
    let mut config = to_config_default(ToAttackerModel::Custom);
    config.custom_threshold_ns = 500.0; // Custom 500ns threshold
    config.time_budget_secs = 5.0;
    config.max_samples = 5000;

    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(77777),
    };

    let result = unsafe {
        to_test(
            &config,
            64,
            random_generator,
            xor_operation,
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // Verify custom threshold was used (test completes successfully)
    assert!(
        result.leak_probability >= 0.0 && result.leak_probability <= 1.0,
        "Should get valid leak probability with custom threshold"
    );

    // Note: to_result_free is for C callers; in Rust tests we skip it
}

#[test]
fn test_research_mode() {
    let mut config = to_config_default(ToAttackerModel::Research);
    config.time_budget_secs = 5.0;
    config.max_samples = 5000;

    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(88888),
    };

    let result = unsafe {
        to_test(
            &config,
            64,
            random_generator,
            xor_operation,
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // Research mode should return Research outcome (not Pass/Fail/Inconclusive)
    assert_eq!(
        result.outcome,
        ToOutcome::Research,
        "Research mode should return Research outcome, got {:?}",
        result.outcome
    );

    // Verify research-specific fields are populated
    assert!(
        result.leak_probability >= 0.0 && result.leak_probability <= 1.0,
        "Research mode should return valid probability"
    );

    // Check research_status is a valid variant
    assert!(
        matches!(
            result.research_status,
            ToResearchStatus::EffectDetected
                | ToResearchStatus::NoEffectDetected
                | ToResearchStatus::ResolutionLimitReached
                | ToResearchStatus::QualityIssue
                | ToResearchStatus::BudgetExhausted
        ),
        "Research status should be a valid variant"
    );

    // max_effect fields should be populated (even if 0.0)
    assert!(
        result.max_effect_ns.is_finite(),
        "max_effect_ns should be finite"
    );
    assert!(
        result.max_effect_ci_low.is_finite(),
        "max_effect_ci_low should be finite"
    );
    assert!(
        result.max_effect_ci_high.is_finite(),
        "max_effect_ci_high should be finite"
    );

    // CI should be ordered correctly
    assert!(
        result.max_effect_ci_low <= result.max_effect_ns,
        "CI low ({}) should be <= mean ({})",
        result.max_effect_ci_low,
        result.max_effect_ns
    );
    assert!(
        result.max_effect_ns <= result.max_effect_ci_high,
        "mean ({}) should be <= CI high ({})",
        result.max_effect_ns,
        result.max_effect_ci_high
    );

    // Diagnostics should include theta values
    assert!(
        result.diagnostics.theta_user == 0.0,
        "Research mode theta_user should be 0.0"
    );
    assert!(
        result.diagnostics.theta_floor >= 0.0,
        "theta_floor should be non-negative"
    );

    eprintln!(
        "[test_research_mode] status={:?}, max_effect={:.2}ns [{:.2}, {:.2}], detectable={}",
        result.research_status,
        result.max_effect_ns,
        result.max_effect_ci_low,
        result.max_effect_ci_high,
        result.research_detectable
    );

    // Note: to_result_free is for C callers; in Rust tests we skip it
}

/// Test research mode with a known-leaky operation to verify effect detection
#[test]
fn test_research_mode_with_leak() {
    let mut config = to_config_default(ToAttackerModel::Research);
    config.time_budget_secs = 15.0;
    config.max_samples = 15000;

    let mut ctx = TestContext {
        rng: StdRng::seed_from_u64(99999),
    };

    let result = unsafe {
        to_test(
            &config,
            512, // Larger input for more pronounced leak
            random_generator,
            early_exit_operation, // Known leaky operation
            &mut ctx as *mut TestContext as *mut c_void,
        )
    };

    // Should return Research outcome
    assert_eq!(
        result.outcome,
        ToOutcome::Research,
        "Research mode should return Research outcome, got {:?}",
        result.outcome
    );

    // max_effect fields should be populated with valid values
    assert!(
        result.max_effect_ns.is_finite(),
        "max_effect_ns should be finite"
    );
    assert!(
        result.max_effect_ci_low.is_finite(),
        "max_effect_ci_low should be finite"
    );
    assert!(
        result.max_effect_ci_high.is_finite(),
        "max_effect_ci_high should be finite"
    );

    // With a leaky operation, we expect to see an effect
    // EffectDetected or BudgetExhausted (if we didn't reach threshold)
    // QualityIssue is also possible if noise is high
    eprintln!(
        "[test_research_mode_with_leak] status={:?}, max_effect={:.2}ns [{:.2}, {:.2}], detectable={}, theta_floor={:.2}ns",
        result.research_status,
        result.max_effect_ns,
        result.max_effect_ci_low,
        result.max_effect_ci_high,
        result.research_detectable,
        result.diagnostics.theta_floor
    );

    // If we got EffectDetected, verify that research_detectable is true
    if result.research_status == ToResearchStatus::EffectDetected {
        assert!(
            result.research_detectable,
            "research_detectable should be true when EffectDetected"
        );
        assert!(
            result.max_effect_ci_low > 0.0,
            "max_effect_ci_low should be > 0 when effect detected"
        );
    }

    // Note: to_result_free is for C callers; in Rust tests we skip it
}
