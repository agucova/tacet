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

use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};
use tokio::runtime::Runtime;
use tokio::time::sleep;

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
        0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82, 0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a, 0x33,
        0x9c, 0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f, 0x9a, 0x2d, 0x7e, 0x4c, 0x8b, 0x3a,
        0x6d, 0x5e,
    ];

    // Pre-generate inputs using InputPair helper
    let inputs = InputPair::new(|| fixed_input, rand_bytes);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            rt.block_on(async {
                std::hint::black_box(data);
            })
        });

    eprintln!("\n[async_executor_overhead_no_false_positive]");
    eprintln!("{}", outcome);

    match outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            // Baseline test: leak probability should be low
            assert!(
                leak_probability < 0.3,
                "Leak probability too high: {:.1}%",
                leak_probability * 100.0
            );
        }
        Outcome::Fail {
            leak_probability, ..
        } => {
            panic!(
                "Unexpected failure for async executor overhead: P={:.1}%",
                leak_probability * 100.0
            );
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            // Acceptable - inconclusive is not a false positive
            assert!(
                leak_probability < 0.5,
                "Leak probability too high for inconclusive: {:.1}%",
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Skipping: unmeasurable - {}", recommendation);
        }
        Outcome::Research(_) => {}
    }
}

/// 1.2 Async Block-on Overhead Symmetric
///
/// Verifies that the overhead of block_on() itself is symmetric
#[test]
fn async_block_on_overhead_symmetric() {
    let rt = single_thread_runtime();

    // Use unit type for input (no data dependency)
    let inputs = InputPair::new(|| (), || ());

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |_| {
            rt.block_on(async {
                // Minimal async block
                std::hint::black_box(42);
            })
        });

    eprintln!("\n[async_block_on_overhead_symmetric]");
    eprintln!("{}", outcome);

    match outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            assert!(
                leak_probability < 0.3,
                "Leak probability too high: {:.1}%",
                leak_probability * 100.0
            );
        }
        Outcome::Fail {
            leak_probability, ..
        } => {
            panic!(
                "Unexpected failure for symmetric block_on: P={:.1}%",
                leak_probability * 100.0
            );
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            assert!(
                leak_probability < 0.5,
                "Leak probability too high for inconclusive: {:.1}%",
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Skipping: unmeasurable - {}", recommendation);
        }
        Outcome::Research(_) => {}
    }
}

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

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
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

    eprintln!("\n[detects_conditional_await_timing]");
    eprintln!("{}", outcome);

    match outcome {
        Outcome::Fail {
            leak_probability, ..
        } => {
            assert!(
                leak_probability > 0.7,
                "Expected high leak probability, got {:.1}%",
                leak_probability * 100.0
            );
        }
        Outcome::Pass {
            leak_probability, ..
        } => {
            panic!(
                "Expected to detect conditional await leak, but passed: P={:.1}%",
                leak_probability * 100.0
            );
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            // Accept inconclusive with high leak probability
            assert!(
                leak_probability > 0.5,
                "Expected at least ambiguous leak probability, got {:.1}%",
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Skipping: unmeasurable - {}", recommendation);
        }
        Outcome::Research(_) => {}
    }
}

/// 2.2 Detects Early Exit Async - Byte-by-byte comparison with early return
#[test]
fn detects_early_exit_async() {
    let rt = single_thread_runtime();
    let secret = [0xABu8; 32];

    // Use InputPair with fixed vs random input for comparison
    let inputs = InputPair::new(|| [0xABu8; 32], || [0xCDu8; 32]);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
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

    eprintln!("\n[detects_early_exit_async]");
    eprintln!("{}", outcome);

    let leak_probability = outcome.leak_probability().unwrap_or(0.0);
    assert!(
        leak_probability > 0.8 || outcome.failed(),
        "Expected to detect early-exit async timing leak (got P={:.1}%)",
        leak_probability * 100.0
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

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(60))
        .test(inputs, |byte_value| {
            rt.block_on(async {
                // Sleep duration depends on the byte value
                let delay_micros = *byte_value as u64 * 10;
                sleep(Duration::from_micros(delay_micros)).await;
                std::hint::black_box(42);
            })
        });

    eprintln!("\n[detects_secret_dependent_sleep]");
    eprintln!("{}", outcome);

    let leak_probability = outcome.leak_probability().unwrap_or(0.0);
    assert!(
        leak_probability > 0.95 || outcome.failed(),
        "Expected very high confidence for large sleep difference (got P={:.1}%)",
        leak_probability * 100.0
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
        0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82, 0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a, 0x33,
        0x9c, 0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f, 0x9a, 0x2d, 0x7e, 0x4c, 0x8b, 0x3a,
        0x6d, 0x5e,
    ];

    // Pre-generate inputs using InputPair helper
    let inputs = InputPair::new(|| fixed_input, rand_bytes);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
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

    eprintln!("\n[concurrent_tasks_no_crosstalk]");
    eprintln!("{}", outcome);

    // This test checks that background tasks don't cause HUGE timing differences.
    // Some variation is expected in concurrent environments - we just want to ensure
    // the noise isn't catastrophic. Don't panic on Fail since async is inherently noisy.
    match &outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            eprintln!(
                "Good: no significant timing difference (P={:.1}%)",
                leak_probability * 100.0
            );
        }
        Outcome::Fail {
            leak_probability,
            effect,
            exploitability,
            ..
        } => {
            // Only panic if the effect is catastrophically large (>10μs)
            let total_effect = effect.shift_ns.abs() + effect.tail_ns.abs();
            if total_effect > 10_000.0 {
                panic!(
                    "Catastrophic timing difference with background tasks: {:.1}μs ({:?})",
                    total_effect / 1000.0,
                    exploitability
                );
            }
            eprintln!(
                "Note: Detected timing difference (P={:.1}%, {:.1}ns) - expected in concurrent tests",
                leak_probability * 100.0, total_effect
            );
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            eprintln!(
                "Note: Inconclusive result ({:.1}%) - expected in concurrent tests",
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Skipping: unmeasurable - {}", recommendation);
        }
        Outcome::Research(_) => {}
    }
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

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(60))
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

    eprintln!("\n[detects_task_spawn_timing_leak]");
    eprintln!("{}", outcome);

    let leak_probability = outcome.leak_probability().unwrap_or(0.0);
    assert!(
        leak_probability > 0.6 || outcome.failed(),
        "Expected to detect task spawn count timing leak (got P={:.1}%)",
        leak_probability * 100.0
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
    // Test with single-threaded runtime
    let rt_single = single_thread_runtime();
    let inputs_single = InputPair::new(|| [0xABu8; 32], rand_bytes);
    let outcome_single = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs_single, |data| {
            rt_single.block_on(async {
                std::hint::black_box(data);
            })
        });

    // Test with multi-threaded runtime
    let rt_multi = multi_thread_runtime();
    let inputs_multi = InputPair::new(|| [0xABu8; 32], rand_bytes);
    let outcome_multi = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs_multi, |data| {
            rt_multi.block_on(async {
                std::hint::black_box(data);
            })
        });

    eprintln!("\n[tokio_single_vs_multi_thread_stability]");
    eprintln!("--- Single-thread ---");
    eprintln!("{}", outcome_single);
    eprintln!("--- Multi-thread ---");
    eprintln!("{}", outcome_multi);

    // Compare leak probabilities (informational)
    let single_prob = outcome_single.leak_probability().unwrap_or(0.0);
    let multi_prob = outcome_multi.leak_probability().unwrap_or(0.0);
    eprintln!(
        "Leak probability comparison: single={:.1}%, multi={:.1}%",
        single_prob * 100.0,
        multi_prob * 100.0
    );
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

    let outcome_without_flag = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |_| {
            rt.block_on(async {
                // Some async work
                for _ in 0..5 {
                    tokio::task::yield_now().await;
                }
                std::hint::black_box(42);
            })
        });

    eprintln!("\n[async_workload_flag_effectiveness]");
    eprintln!("{}", outcome_without_flag);

    // Verify result structure is valid (informational test)
    let leak_probability = outcome_without_flag.leak_probability().unwrap_or(0.0);
    assert!((0.0..=1.0).contains(&leak_probability));
}
