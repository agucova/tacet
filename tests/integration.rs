//! End-to-end integration tests.

use timing_oracle::{helpers::InputPair, timing_test_checked, Outcome, TimingOracle};

/// Basic smoke test that the API works.
#[test]
fn smoke_test() {
    let inputs = InputPair::new(|| 1u32, || 2u32);
    let outcome = TimingOracle::new()
        .samples(100) // Minimal for speed
        .warmup(10)
        .test(inputs, |x| {
            std::hint::black_box(x + 1);
        });

    // Just verify we get a result without panicking
    let result = match outcome {
        Outcome::Completed(r) => r,
        Outcome::Unmeasurable { .. } => return, // Skip if unmeasurable
    };
    assert!(result.leak_probability >= 0.0 && result.leak_probability <= 1.0);
    assert!(result.ci_gate.passed, "Should not detect leak in simple operation");
    assert!(result.min_detectable_effect.shift_ns > 0.0);
    assert!(result.metadata.samples_per_class > 0);
}

/// Test builder API.
#[test]
fn builder_api() {
    let oracle = TimingOracle::new()
        .samples(1000)
        .warmup(100)
        .alpha(0.05)
        .min_effect_ns(5.0)
        .outlier_percentile(0.99);

    let config = oracle.config();
    assert_eq!(config.samples, 1000);
    assert_eq!(config.warmup, 100);
    assert!((config.ci_alpha - 0.05).abs() < 1e-10);
    assert!((config.min_effect_of_concern_ns - 5.0).abs() < 1e-10);
}

/// Test convenience function.
#[test]
fn convenience_function() {
    let inputs = InputPair::new(|| 42u32, || 42u32);
    let outcome = TimingOracle::new().samples(200).test(inputs, |x| {
        std::hint::black_box(*x);
    });

    let result = match outcome {
        Outcome::Completed(r) => r,
        Outcome::Unmeasurable { .. } => return,
    };
    assert!(result.leak_probability >= 0.0 && result.leak_probability <= 1.0);
    assert!(result.ci_gate.passed, "Should not detect leak in constant-time operation");
    assert!(result.min_detectable_effect.shift_ns > 0.0);
    assert!(result.metadata.samples_per_class > 0);
}

/// Test macro API.
#[test]
fn macro_api() {
    let outcome = timing_test_checked! {
        oracle: TimingOracle::new().samples(100),
        baseline: || 42u32,
        sample: || rand::random::<u32>(),
        measure: |x| {
            std::hint::black_box(*x);
        },
    };

    let result = match outcome {
        Outcome::Completed(r) => r,
        Outcome::Unmeasurable { .. } => return,
    };
    assert!(result.leak_probability >= 0.0 && result.leak_probability <= 1.0);
    assert!(result.ci_gate.passed, "Should not detect leak in constant-time operation");
    assert!(result.min_detectable_effect.shift_ns > 0.0);
    assert!(result.metadata.samples_per_class > 0);
}

/// Test result serialization.
#[test]
fn result_serialization() {
    let inputs = InputPair::new(|| (), || ());
    let outcome = TimingOracle::new().samples(100).test(inputs, |_| {});

    let result = match outcome {
        Outcome::Completed(r) => r,
        Outcome::Unmeasurable { .. } => return,
    };

    // Verify it can be serialized to JSON
    let json = serde_json::to_string(&result).expect("Should serialize");
    assert!(json.contains("leak_probability"));
    assert!(json.contains("ci_gate"));
}
