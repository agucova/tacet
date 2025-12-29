//! End-to-end integration tests.

use timing_oracle::{test, Config, TimingOracle};

/// Basic smoke test that the API works.
#[test]
fn smoke_test() {
    let result = TimingOracle::new()
        .samples(100) // Minimal for speed
        .warmup(10)
        .test(|| std::hint::black_box(1 + 1), || std::hint::black_box(2 + 2));

    // Just verify we get a result without panicking
    assert!(result.leak_probability >= 0.0);
    assert!(result.leak_probability <= 1.0);
}

/// Test builder API.
#[test]
fn builder_api() {
    let oracle = TimingOracle::new()
        .samples(1000)
        .warmup(100)
        .ci_alpha(0.05)
        .min_effect_of_concern(5.0)
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
    let result = test(
        || std::hint::black_box(42),
        || std::hint::black_box(42),
    );

    assert!(result.leak_probability >= 0.0);
    assert!(result.leak_probability <= 1.0);
}

/// Test result serialization.
#[test]
fn result_serialization() {
    let result = TimingOracle::new()
        .samples(100)
        .test(|| (), || ());

    // Verify it can be serialized to JSON
    let json = serde_json::to_string(&result).expect("Should serialize");
    assert!(json.contains("leak_probability"));
    assert!(json.contains("ci_gate"));
}
