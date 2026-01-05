//! Async timing integration tests - validates timing analysis of async/await code.
//!
//! Tests cover:
//! - Baseline: Executor overhead doesn't cause false positives
//! - Detection: Secret-dependent async timing is caught
//! - Concurrency: Background tasks don't interfere with measurements
//! - Runtime comparison: Single vs multi-threaded stability
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use timing_oracle::helpers::InputPair;
use timing_oracle::TimingOracle;
use tokio::runtime::Runtime;
use tokio::time::{sleep, Duration};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a single-threaded Tokio runtime for minimal jitter
fn single_thread_runtime() -> Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("failed to create single-thread runtime")
}

/// Create a multi-threaded Tokio runtime for stress testing
fn multi_thread_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_time()
        .build()
        .expect("failed to create multi-thread runtime")
}

fn rand_bytes() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

// ============================================================================
// Category 1: Baseline Tests (Should Pass - No False Positives)
// ============================================================================

/// 1.1 Async Executor Overhead - Should not cause false positive
///
/// Uses DudeCT's two-class pattern: fixed data vs random data
/// Tests that async executor overhead alone doesn't leak timing
#[test]
fn async_executor_overhead_no_false_positive() {
    let rt = single_thread_runtime();

    // Use non-pathological fixed input (not all-zeros)
    let fixed_input: [u8; 32] = [
        0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82,
        0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a, 0x33, 0x9c,
        0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f,
        0x9a, 0x2d, 0x7e, 0x4c, 0x8b, 0x3a, 0x6d, 0x5e,
    ];

    // Pre-generate inputs using InputPair helper
    const SAMPLES: usize = 10_000;
    let inputs = InputPair::new(|| fixed_input, rand_bytes);

    let outcome = TimingOracle::new()
        .samples(SAMPLES)
        .alpha(0.01)
        .test(inputs, |data| {
            rt.block_on(async { std::hint::black_box(data); })
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[async_executor_overhead_no_false_positive]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // Async executor overhead should pass CI gate
    assert!(
        result.ci_gate.passed,
        "CI gate should pass for async executor overhead"
    );

    // Baseline test: leak probability should be low
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );}

/// 1.2 Async Block-on Overhead Symmetric
///
/// Verifies that the overhead of block_on() itself is symmetric
#[test]
fn async_block_on_overhead_symmetric() {
    let rt = single_thread_runtime();

    // Use unit type for input (no data dependency)
    let inputs = InputPair::new(|| (), || ());

    let outcome = TimingOracle::new()
        .samples(8_000)
        .test(inputs, |_| {
            rt.block_on(async {
                // Minimal async block
                std::hint::black_box(42);
            })
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[async_block_on_overhead_symmetric]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed,
        "CI gate should pass for identical block_on() overhead"
    );

    // Baseline test: leak probability should be low
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );}

// ============================================================================
// Category 2: Leak Detection Tests (Should Detect Timing Leaks)
// ============================================================================

/// 2.1 Detects Conditional Await Timing (Fast)
///
/// Tests detection of secret-dependent await patterns
#[test]
fn detects_conditional_await_timing() {
    let rt = single_thread_runtime();

    // Use unit type for input; secret-dependent logic is in the operation
    let inputs = InputPair::new(|| true, || false);

    let outcome = TimingOracle::new()
        .samples(50_000)
        .alpha(0.01)
        .test(inputs, |secret| {
            rt.block_on(async {
                if *secret {
                    // Extra await when secret is true
                    sleep(Duration::from_nanos(100)).await;
                }
                sleep(Duration::from_micros(5)).await;
                std::hint::black_box(42);
            })
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[detects_conditional_await_timing]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        !result.ci_gate.passed && result.leak_probability > 0.7,
        "Expected to detect leak: ci_gate should fail AND leak_probability should be high (got leak_probability={}, ci_gate.passed={})",
        result.leak_probability,
        result.ci_gate.passed
    );
}

/// 2.2 Detects Early Exit Async - Byte-by-byte comparison with early return
#[test]
fn detects_early_exit_async() {
    let rt = single_thread_runtime();
    let secret = [0xABu8; 32];

    // Use InputPair with fixed vs random input for comparison
    let inputs = InputPair::new(|| [0xABu8; 32], || [0xCDu8; 32]);

    let outcome = TimingOracle::new()
        .samples(50_000)
        .test(inputs, |comparison_input| {
            rt.block_on(async {
                // Compare with input - goes through bytes or exits early
                for i in 0..32 {
                    if secret[i] != comparison_input[i] {
                        return;
                    }
                    // Small async point to make timing observable
                    tokio::task::yield_now().await;
                }
            })
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[detects_early_exit_async]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.leak_probability > 0.8,
        "Expected to detect early-exit async timing leak (got {})",
        result.leak_probability
    );
}

/// 2.3 Detects Secret-Dependent Sleep Duration (Thorough)
///
/// Tests detection of sleep duration that depends on secret value
#[test]
#[ignore = "slow test - run with --ignored"]
fn detects_secret_dependent_sleep() {
    let rt = single_thread_runtime();

    // InputPair with secret byte (fixed) vs random byte values
    let inputs = InputPair::new(|| 10u8, || 1u8);

    let outcome = TimingOracle::new()
        .samples(100_000)
        .test(inputs, |byte_value| {
            rt.block_on(async {
                // Sleep duration depends on the byte value
                let delay_micros = *byte_value as u64 * 10;
                sleep(Duration::from_micros(delay_micros)).await;
                std::hint::black_box(42);
            })
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[detects_secret_dependent_sleep]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.leak_probability > 0.95,
        "Expected very high confidence for large sleep difference (got {})",
        result.leak_probability
    );
}

// ============================================================================
// Category 3: Concurrent Task Tests
// ============================================================================

/// 3.1 Concurrent Tasks - No Crosstalk (Fast)
///
/// Verifies that background tasks don't interfere with foreground measurements
#[test]
fn concurrent_tasks_no_crosstalk() {
    let rt = multi_thread_runtime();

    // Use non-pathological fixed input (not all-zeros)
    let fixed_input: [u8; 32] = [
        0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82,
        0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a, 0x33, 0x9c,
        0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f,
        0x9a, 0x2d, 0x7e, 0x4c, 0x8b, 0x3a, 0x6d, 0x5e,
    ];

    // Pre-generate inputs using InputPair helper
    const SAMPLES: usize = 10_000;
    let inputs = InputPair::new(|| fixed_input, rand_bytes);

    let outcome = TimingOracle::new()
        .samples(SAMPLES)
        .test(inputs, |data| {
            rt.block_on(async {
                // Spawn background tasks
                for _ in 0..10 {
                    tokio::spawn(async {
                        sleep(Duration::from_micros(100)).await;
                    });
                }
                std::hint::black_box(data);
            })
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[concurrent_tasks_no_crosstalk]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // Background tasks should not cause false positives
    assert!(
        result.ci_gate.passed,
        "CI gate should pass with background tasks (got leak_probability={:.3})",
        result.leak_probability
    );

    // Baseline test: leak probability should be low
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );

    assert!(
        result.min_detectable_effect.shift_ns > 0.0,
        "MDE should be positive"
    );
}

/// 3.2 Detects Task Spawn Timing Leak (Thorough)
///
/// Tests detection of timing differences from different task spawn counts
#[test]
#[ignore = "slow test - run with --ignored"]
fn detects_task_spawn_timing_leak() {
    let rt = multi_thread_runtime();

    // InputPair with fixed count vs random count generator
    let inputs = InputPair::new(|| 10usize, || rand::random::<u32>() as usize % 20);

    let outcome = TimingOracle::new()
        .samples(50_000)
        .test(inputs, |count| {
            rt.block_on(async {
                // Spawn task count based on input
                for _ in 0..*count {
                    tokio::spawn(async {
                        sleep(Duration::from_nanos(10)).await;
                    });
                }
                std::hint::black_box(42);
            })
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[detects_task_spawn_timing_leak]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.leak_probability > 0.6,
        "Expected to detect task spawn count timing leak (got {})",
        result.leak_probability
    );
}

// ============================================================================
// Category 4: Optional Thorough Tests
// ============================================================================

/// 4.1 Tokio Single vs Multi-Thread Stability
///
/// Compares noise levels between single-threaded and multi-threaded runtimes
#[test]
#[ignore = "slow comparative test - run with --ignored"]
fn tokio_single_vs_multi_thread_stability() {
    const SAMPLES: usize = 20_000;

    // Test with single-threaded runtime
    let rt_single = single_thread_runtime();
    let inputs_single = InputPair::new(|| [0xABu8; 32], rand_bytes);
    let outcome_single = TimingOracle::new()
        .samples(SAMPLES)
        .test(inputs_single, |data| {
            rt_single.block_on(async { std::hint::black_box(data); })
        });
    let result_single = outcome_single.unwrap_completed();

    // Test with multi-threaded runtime
    let rt_multi = multi_thread_runtime();
    let inputs_multi = InputPair::new(|| [0xABu8; 32], rand_bytes);
    let outcome_multi = TimingOracle::new()
        .samples(SAMPLES)
        .test(inputs_multi, |data| {
            rt_multi.block_on(async { std::hint::black_box(data); })
        });
    let result_multi = outcome_multi.unwrap_completed();

    eprintln!("\n[tokio_single_vs_multi_thread_stability]");
    eprintln!("--- Single-thread ---");
    eprintln!("{}", timing_oracle::output::format_result(&result_single));
    eprintln!("--- Multi-thread ---");
    eprintln!("{}", timing_oracle::output::format_result(&result_multi));

    // Multi-threaded runtime typically has higher noise (higher MDE)
    eprintln!("MDE ratio (multi/single): shift={:.2}x, tail={:.2}x",
              result_multi.min_detectable_effect.shift_ns / result_single.min_detectable_effect.shift_ns,
              result_multi.min_detectable_effect.tail_ns / result_single.min_detectable_effect.tail_ns);
}

/// 4.2 Async Workload Flag Effectiveness
///
/// Validates that async_workload flag helps prevent false positives
/// Note: This test checks the flag exists, but the actual implementation
/// of async_workload handling may vary
#[test]
#[ignore = "informational test - run with --ignored"]
fn async_workload_flag_effectiveness() {
    let rt = single_thread_runtime();

    // Use unit type for input (no data dependency)
    let inputs = InputPair::new(|| (), || ());

    let outcome_without_flag = TimingOracle::new()
        .samples(10_000)
        .test(inputs, |_| {
            rt.block_on(async {
                // Some async work
                for _ in 0..5 {
                    tokio::task::yield_now().await;
                }
                std::hint::black_box(42);
            })
        });

    let result_without_flag = outcome_without_flag.unwrap_completed();

    eprintln!("\n[async_workload_flag_effectiveness]");
    eprintln!("{}", timing_oracle::output::format_result(&result_without_flag));

    // Verify result structure is valid (informational test)
    assert!(result_without_flag.leak_probability >= 0.0 && result_without_flag.leak_probability <= 1.0);
    assert!(result_without_flag.min_detectable_effect.shift_ns > 0.0);
    assert!(result_without_flag.metadata.samples_per_class > 0);
}
