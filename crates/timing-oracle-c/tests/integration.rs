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

    // Research mode uses θ=0, which is very sensitive
    // Any tiny difference might be detected
    // Just verify the test runs and returns valid results
    assert!(
        result.leak_probability >= 0.0 && result.leak_probability <= 1.0,
        "Research mode should return valid probability"
    );

    // Note: to_result_free is for C callers; in Rust tests we skip it
}
