//! Tests for timing_oracle::helpers utilities

use timing_oracle::helpers::{byte_arrays_32, byte_vecs, InputPair};
use timing_oracle::{Outcome, TimingOracle};

#[test]
fn test_input_pair_basic() {
    let inputs = InputPair::new(|| 42u64, rand::random::<u64>);

    // baseline() should always return the same value
    for _ in 0..100 {
        assert_eq!(inputs.baseline(), 42);
        let _ = inputs.generate_sample(); // Random values vary
    }
}

#[test]
fn test_input_pair_generator_called() {
    // The generator is called each time generate_sample() is called
    let counter = std::cell::Cell::new(0u32);
    let inputs = InputPair::new(|| 0u32, || {
        let val = counter.get();
        counter.set(val + 1);
        val
    });

    // baseline() always returns 0
    assert_eq!(inputs.baseline(), 0);
    assert_eq!(inputs.baseline(), 0);

    // generate_sample() calls generator each time
    assert_eq!(inputs.generate_sample(), 0);
    assert_eq!(inputs.generate_sample(), 1);
    assert_eq!(inputs.generate_sample(), 2);
}

#[test]
fn test_byte_arrays_helper() {
    let inputs = byte_arrays_32();

    // Fixed should be all zeros
    assert_eq!(inputs.baseline(), [0u8; 32]);

    // Random should vary
    let r1 = inputs.generate_sample();
    let r2 = inputs.generate_sample();
    assert_ne!(r1, r2); // Very unlikely to be equal
}

#[test]
fn test_byte_vecs_helper() {
    let inputs = byte_vecs(128);

    // Fixed should be all zeros
    assert_eq!(inputs.baseline(), vec![0u8; 128]);

    // Random should vary and have correct length
    let r = inputs.generate_sample();
    assert_eq!(r.len(), 128);
}

#[test]
fn test_no_false_positive_with_helpers() {
    // Use helpers for constant-time XOR - should not false positive
    let inputs = byte_arrays_32();

    // Use balanced() for reasonable speed while still being accurate
    let outcome = TimingOracle::balanced().test(inputs, |data| {
        // XOR with zeros is identity - constant time
        let mut result = [0u8; 32];
        for i in 0..32 {
            result[i] = data[i] ^ 0;
        }
        std::hint::black_box(result);
    });

    if let Outcome::Completed(result) = outcome {
        assert!(result.ci_gate.passed, "Should not detect leak with helpers");
        assert!(result.leak_probability < 0.5);
    }
}

#[test]
fn test_anomaly_detection_constant_value() {
    // Simulate the common mistake: captured pre-evaluated value
    let constant_value = 42u64;
    let inputs = InputPair::new(|| 0u64, || constant_value);

    // Generate enough samples to trigger detection
    for _ in 0..200 {
        let val = inputs.generate_sample();
        inputs.track_value(&val);
    }

    let anomaly = inputs.check_anomaly();
    assert!(anomaly.is_some(), "Should detect constant value anomaly");
    assert!(anomaly.unwrap().contains("ANOMALY"));
}

#[test]
fn test_anomaly_detection_good_entropy() {
    let inputs = InputPair::new(|| 0u64, rand::random::<u64>);

    for _ in 0..200 {
        let val = inputs.generate_sample();
        inputs.track_value(&val);
    }

    assert!(
        inputs.check_anomaly().is_none(),
        "Should not warn for good entropy"
    );
}

#[test]
fn test_anomaly_detection_low_entropy() {
    let counter = std::cell::Cell::new(0u64);
    let inputs = InputPair::new(|| 0u64, || {
        let val = counter.get() % 10; // Only 10 unique values
        counter.set(counter.get() + 1);
        val
    });

    for _ in 0..200 {
        let val = inputs.generate_sample();
        inputs.track_value(&val);
    }

    let anomaly = inputs.check_anomaly();
    assert!(anomaly.is_some(), "Should detect low entropy");
    assert!(anomaly.unwrap().contains("WARNING"));
}

fn xor_bytes(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for i in 0..32 {
        result[i] = a[i] ^ b[i];
    }
    result
}
