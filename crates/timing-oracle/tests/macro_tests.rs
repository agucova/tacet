//! Unit tests for the timing_test_checked! macro.
//!
//! This file tests various invocation patterns of the macro to ensure
//! it correctly handles all syntax variations.

#![allow(clippy::redundant_closure)]

use std::time::Duration;
use timing_oracle::{timing_test, timing_test_checked, AttackerModel, Outcome, TimingOracle};

// ===========================================================================
// Basic Invocation Tests
// ===========================================================================

/// Test minimal macro invocation with just required fields.
#[test]
fn macro_minimal_syntax() {
    let result = timing_test_checked! {
        baseline: || 42u64,
        sample: || rand::random::<u64>(),
        measure: |input| {
            std::hint::black_box(input);
        },
    };

    assert!(
        result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }),
        "Macro should produce valid Outcome"
    );
}

/// Test macro with custom oracle configuration.
#[test]
fn macro_with_oracle() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || 0u8,
        sample: || rand::random::<u8>(),
        measure: |input| {
            std::hint::black_box(input);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with pre-measurement setup done before the macro.
/// Variables are captured from outer scope for shared state.
#[test]
fn macro_with_pre_setup() {
    // Pre-measurement work (e.g., cache warming) is done before the macro
    let key = [0u8; 16];
    for _ in 0..10 {
        std::hint::black_box(&key);
    }

    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || [0u8; 16],
        sample: || rand::random::<[u8; 16]>(),
        measure: |input| {
            // key is captured from outer scope
            std::hint::black_box((&key, input));
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with captured variables from outer scope.
#[test]
fn macro_with_captures() {
    // Variables defined BEFORE macro are capturable
    let multiplier = 42u64;

    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || 1u64,
        sample: || rand::random::<u64>(),
        measure: |input| {
            // Use variable from outer scope
            let _ = std::hint::black_box(input.wrapping_mul(multiplier));
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with oracle and captures from outer scope.
#[test]
fn macro_all_fields() {
    // Capture from outer scope
    let secret = [0xFFu8; 32];

    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)).max_samples(1_000),
        baseline: || [0u8; 32],
        sample: || rand::random::<[u8; 32]>(),
        measure: |input| {
            // Simulate XOR comparison using captured secret
            let mut acc = 0u8;
            for i in 0..32 {
                acc |= secret[i] ^ input[i];
            }
            std::hint::black_box(acc);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

// ===========================================================================
// Field Order Tests
// ===========================================================================

/// Test that field order doesn't matter (oracle at end).
#[test]
fn macro_field_order_oracle_last() {
    let result = timing_test_checked! {
        baseline: || 0u8,
        sample: || rand::random::<u8>(),
        measure: |input| {
            std::hint::black_box(input);
        },
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test that field order doesn't matter (measure before baseline/sample).
#[test]
fn macro_field_order_measure_first() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        measure: |input| {
            std::hint::black_box(input);
        },
        baseline: || 0u8,
        sample: || rand::random::<u8>(),
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

// ===========================================================================
// Input Type Tests
// ===========================================================================

/// Test macro with tuple inputs.
#[test]
#[ignore] // Slow (31s) - just tests tuple type, run with --ignored
fn macro_tuple_input() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || ([0u8; 12], [0u8; 64]),
        sample: || (rand::random::<[u8; 12]>(), rand::random::<[u8; 64]>()),
        measure: |(nonce, plaintext)| {
            std::hint::black_box((nonce, plaintext));
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with Vec inputs.
#[test]
#[ignore] // Slow (28s) - just tests Vec type, run with --ignored
fn macro_vec_input() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || vec![0u8; 64],
        sample: || (0..64).map(|_| rand::random::<u8>()).collect::<Vec<_>>(),
        measure: |input| {
            std::hint::black_box(&input[..]);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with String input.
#[test]
fn macro_string_input() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || String::from("constant_password"),
        sample: || format!("random_{:016x}", rand::random::<u64>()),
        measure: |input| {
            std::hint::black_box(input.len());
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with primitive u8.
#[test]
fn macro_primitive_u8() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || 0u8,
        sample: || rand::random::<u8>(),
        measure: |input| {
            std::hint::black_box(*input);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with larger arrays.
#[test]
#[ignore] // Slow (87s) - just tests array size, redundant with crypto tests
fn macro_large_array() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || [0u8; 256],
        sample: || {
            let mut arr = [0u8; 256];
            for b in &mut arr {
                *b = rand::random();
            }
            arr
        },
        measure: |input| {
            std::hint::black_box(&input[..]);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

// ===========================================================================
// Oracle Configuration Tests
// ===========================================================================

/// Test macro with balanced oracle preset.
#[test]
fn macro_oracle_balanced() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(30)),
        baseline: || 0u64,
        sample: || rand::random::<u64>(),
        measure: |input| {
            std::hint::black_box(input);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with calibration oracle preset.
#[test]
fn macro_oracle_calibration() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::Research).time_budget(Duration::from_secs(5)),
        baseline: || 0u64,
        sample: || rand::random::<u64>(),
        measure: |input| {
            std::hint::black_box(input);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with chained oracle configuration.
#[test]
fn macro_oracle_chained() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)).max_samples(2_000).warmup(100),
        baseline: || 0u64,
        sample: || rand::random::<u64>(),
        measure: |input| {
            std::hint::black_box(input);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

// ===========================================================================
// Complex Test Bodies
// ===========================================================================

/// Test macro with multi-statement measure body.
#[test]
fn macro_complex_measure_body() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || [0u8; 32],
        sample: || rand::random::<[u8; 32]>(),
        measure: |input| {
            // Multi-statement body
            let mut sum = 0u64;
            for byte in input.iter() {
                sum = sum.wrapping_add(*byte as u64);
            }
            let result = sum % 256;
            std::hint::black_box(result);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with early return in measure body.
#[test]
fn macro_measure_with_early_return() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || 0u64,
        sample: || rand::random::<u64>(),
        measure: |input| {
            if *input == 0 {
                return;
            }
            std::hint::black_box(input);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with nested closures in measure body.
#[test]
fn macro_nested_closures() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || [0u8; 16],
        sample: || rand::random::<[u8; 16]>(),
        measure: |input| {
            let process = |data: &[u8]| -> u64 {
                data.iter().map(|&b| b as u64).sum()
            };
            std::hint::black_box(process(input));
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

// ===========================================================================
// Result Access Tests
// ===========================================================================

/// Test that we can access outcome fields.
#[test]
fn macro_result_access() {
    let outcome = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || 0u64,
        sample: || rand::random::<u64>(),
        measure: |input| {
            std::hint::black_box(input);
        },
    };

    match outcome {
        Outcome::Pass {
            leak_probability,
            effect,
            ..
        }
        | Outcome::Fail {
            leak_probability,
            effect,
            ..
        }
        | Outcome::Inconclusive {
            leak_probability,
            effect,
            ..
        } => {
            // Verify fields are accessible
            let _leak_prob = leak_probability;
            let _effect = &effect;

            // Leak probability should be valid
            assert!(
                (0.0..=1.0).contains(&leak_probability),
                "Leak probability should be between 0 and 1"
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            // Unmeasurable is also valid, just verify we can access fields
            assert!(!recommendation.is_empty());
        }
        Outcome::Research(_) => {}
    }
}

// ===========================================================================
// No Trailing Comma Tests
// ===========================================================================

/// Test macro without trailing comma.
#[test]
fn macro_no_trailing_comma() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || 0u8,
        sample: || rand::random::<u8>(),
        measure: |input| {
            std::hint::black_box(input);
        }
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

// ===========================================================================
// Closure Syntax Variations
// ===========================================================================

/// Test macro with move closure in sample.
#[test]
fn macro_move_closure() {
    let seed = 42u64;

    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || 0u64,
        sample: move || rand::random::<u64>().wrapping_add(seed),
        measure: |input| {
            std::hint::black_box(input);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with complex sample generator.
#[test]
fn macro_complex_sample_generator() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || [0u8; 32],
        sample: || {
            // Complex multi-line generator
            let mut arr = [0u8; 32];
            for (i, byte) in arr.iter_mut().enumerate() {
                *byte = ((i as u8).wrapping_mul(17)).wrapping_add(rand::random::<u8>());
            }
            arr
        },
        measure: |input| {
            std::hint::black_box(input);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with mutable references in measure body.
#[test]
fn macro_mutable_state_captured() {
    use std::cell::Cell;

    // Use Cell for interior mutability since closures are FnMut
    let counter = Cell::new(0u64);

    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || 1u64,
        sample: || rand::random::<u64>(),
        measure: |input| {
            // Use wrapping_add to avoid overflow panic
            counter.set(counter.get().wrapping_add(*input));
            std::hint::black_box(counter.get());
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

// ===========================================================================
// Edge Case Tests
// ===========================================================================

/// Test macro with zero-sized type.
#[test]
fn macro_unit_type() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || (),
        sample: || (),
        measure: |_input| {
            std::hint::black_box(42);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

/// Test macro with bool type.
#[test]
fn macro_bool_type() {
    let result = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || false,
        sample: || rand::random::<bool>(),
        measure: |input| {
            std::hint::black_box(*input);
        },
    };

    assert!(result.is_measurable() || matches!(result, Outcome::Unmeasurable { .. }));
}

// ===========================================================================
// Coverage Tests: Both timing_test! and timing_test_checked!
// ===========================================================================

/// Test that timing_test! returns Outcome directly.
#[test]
#[ignore] // Slow (48s) - just tests return type, run with --ignored
fn timing_test_returns_outcome() {
    let outcome = timing_test! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || [0u8; 128],
        sample: || {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        },
        measure: |arr| {
            // More complex operation to ensure measurability
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        },
    };

    // Should be able to access Outcome methods
    if let Some(leak_prob) = outcome.leak_probability() {
        assert!((0.0..=1.0).contains(&leak_prob));
    }
    if let Some(samples) = outcome.samples_used() {
        assert!(samples > 0);
    }
}

/// Test that timing_test! can be used with helper methods.
#[test]
fn timing_test_no_pattern_matching_needed() {
    // This demonstrates the convenience of timing_test! helper methods
    let outcome = timing_test! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || [0u8; 64],
        sample: || {
            let mut arr = [0u8; 64];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        },
        measure: |arr| {
            // Hash-like operation
            let mut val = 0u8;
            for byte in arr {
                val = val.wrapping_add(*byte).wrapping_mul(31);
            }
            std::hint::black_box(val);
        },
    };

    // Use helper methods without full pattern matching
    if let Some(leak_prob) = outcome.leak_probability() {
        println!("Leak probability: {:.2}%", leak_prob * 100.0);
    }
    // Verify we got some result
    assert!(outcome.is_measurable() || matches!(outcome, Outcome::Unmeasurable { .. }));
}

/// Test that timing_test_checked! returns Outcome for explicit handling.
#[test]
fn timing_test_checked_returns_outcome() {
    let outcome: Outcome = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || 1u64,
        sample: || rand::random::<u64>(),
        measure: |x| {
            std::hint::black_box(*x);
        },
    };

    // Must pattern match or use helper methods on Outcome
    match outcome {
        Outcome::Pass {
            leak_probability, ..
        }
        | Outcome::Fail {
            leak_probability, ..
        }
        | Outcome::Inconclusive {
            leak_probability, ..
        } => {
            assert!(leak_probability >= 0.0);
        }
        Outcome::Unmeasurable { .. } => {
            // Test passes - this is a valid outcome
        }
        Outcome::Research(_) => {}
    }
}

/// Test timing_test_checked! with explicit unmeasurable handling.
#[test]
fn timing_test_checked_explicit_unmeasurable_handling() {
    let outcome = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || (),
        sample: || (),
        measure: |_| {
            // Extremely fast operation - might be unmeasurable
        },
    };

    // This pattern is useful when you want to handle unmeasurable gracefully
    match outcome {
        Outcome::Pass {
            leak_probability, ..
        }
        | Outcome::Fail {
            leak_probability, ..
        }
        | Outcome::Inconclusive {
            leak_probability, ..
        } => {
            println!("Measurable: {:.3}", leak_probability);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            println!("Unmeasurable: {}", recommendation);
        }
        Outcome::Research(_) => {}
    };
}

/// Test side-by-side comparison of both macros with identical operations.
#[test]
#[ignore] // Slow (54s) - runs 2 full oracle tests, run with --ignored
fn both_macros_with_identical_config() {
    // timing_test! - returns Outcome
    let outcome_direct = timing_test! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || [100u8; 64],
        sample: || {
            let mut arr = [0u8; 64];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        },
        measure: |arr| {
            // More complex operation
            let mut val = 0u32;
            for byte in arr {
                val = val.wrapping_mul(2).wrapping_add(*byte as u32);
            }
            std::hint::black_box(val);
        },
    };

    // timing_test_checked! - returns Outcome
    let outcome_checked = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || [100u8; 64],
        sample: || {
            let mut arr = [0u8; 64];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        },
        measure: |arr| {
            // Same operation
            let mut val = 0u32;
            for byte in arr {
                val = val.wrapping_mul(2).wrapping_add(*byte as u32);
            }
            std::hint::black_box(val);
        },
    };

    // Both should produce valid results
    if let Some(prob) = outcome_direct.leak_probability() {
        assert!(prob >= 0.0);
    }

    if let Some(prob) = outcome_checked.leak_probability() {
        assert!(prob >= 0.0);
    }
}

/// Test that timing_test! works with all optional fields.
#[test]
fn timing_test_all_optional_fields() {
    let multiplier = 7u32;
    // Pre-measurement work done before macro
    std::hint::black_box(multiplier);

    let outcome = timing_test! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)).max_samples(1000),
        baseline: || [10u8; 64],
        sample: || {
            let mut arr = [0u8; 64];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        },
        measure: |arr| {
            // Make it measurable
            let mut val = multiplier;
            for byte in arr {
                val = val.wrapping_mul(*byte as u32);
            }
            std::hint::black_box(val);
        },
    };

    // Note: actual samples may be adjusted based on batching
    if let Some(samples) = outcome.samples_used() {
        assert!(samples > 0);
    }
}

/// Test that timing_test_checked! works with all optional fields.
#[test]
fn timing_test_checked_all_optional_fields() {
    let divisor = 3u64;
    // Pre-measurement work done before macro
    std::hint::black_box(divisor);

    let outcome = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)).max_samples(500),
        baseline: || 99u64,
        sample: || rand::random::<u64>() | 1, // Ensure non-zero
        measure: |x| {
            std::hint::black_box(x.wrapping_div(*x.max(&1)));
        },
    };

    assert!(outcome.is_measurable() || matches!(outcome, Outcome::Unmeasurable { .. }));
}

/// Test timing_test! with complex types (arrays).
#[test]
fn timing_test_complex_array_type() {
    let outcome = timing_test! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || [0u8; 64],  // Larger array
        sample: || {
            let mut arr = [0u8; 64];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        },
        measure: |arr| {
            // More complex operation to make it measurable
            let mut acc = 0u8;
            for byte in arr {
                acc ^= byte;
                acc = acc.wrapping_mul(7).wrapping_add(13);
            }
            std::hint::black_box(acc);
        },
    };

    if let Some(prob) = outcome.leak_probability() {
        assert!(prob <= 1.0);
    }
}

/// Test timing_test_checked! with complex types (Vec).
#[test]
fn timing_test_checked_complex_vec_type() {
    let outcome = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(10)),
        baseline: || vec![0u32; 10],
        sample: || {
            (0..10).map(|_| rand::random::<u32>()).collect()
        },
        measure: |v| {
            let sum: u32 = v.iter().fold(0u32, |acc, x| acc.wrapping_add(*x));
            std::hint::black_box(sum);
        },
    };

    assert!(outcome.is_reliable() || !outcome.is_reliable()); // Always true, just testing API
}
