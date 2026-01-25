//! End-to-end integration tests.

use std::time::Duration;
use tacet::{
    helpers::InputPair, timing_test_checked, AttackerModel, Outcome, TimingOracle,
};

/// Basic smoke test that the API works.
#[test]
fn smoke_test() {
    let inputs = InputPair::new(|| 1u32, || 2u32);
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(5))
        .max_samples(100) // Minimal for speed
        .warmup(10)
        .test(inputs, |x| {
            std::hint::black_box(x + 1);
        });

    // Just verify we get a result without panicking
    match outcome {
        Outcome::Pass {
            leak_probability,
            samples_used,
            ..
        } => {
            assert!((0.0..=1.0).contains(&leak_probability));
            assert!(samples_used > 0);
        }
        Outcome::Fail {
            leak_probability,
            samples_used,
            ..
        } => {
            // Unexpected but valid
            assert!((0.0..=1.0).contains(&leak_probability));
            assert!(samples_used > 0);
        }
        Outcome::Inconclusive {
            leak_probability,
            samples_used,
            ..
        } => {
            assert!((0.0..=1.0).contains(&leak_probability));
            assert!(samples_used > 0);
        }
        Outcome::Unmeasurable { .. } => (),
        Outcome::Research(_) => (), // Skip if unmeasurable
    }
}

/// Test builder API.
#[test]
fn builder_api() {
    let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .max_samples(1000)
        .warmup(100)
        .outlier_percentile(0.99);

    let config = oracle.config();
    assert_eq!(config.max_samples, 1000);
    assert_eq!(config.warmup, 100);
}

/// Test convenience function.
#[test]
fn convenience_function() {
    let inputs = InputPair::new(|| 42u32, || 42u32);
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(5))
        .max_samples(200)
        .test(inputs, |x| {
            std::hint::black_box(*x);
        });

    match outcome {
        Outcome::Pass {
            leak_probability,
            samples_used,
            ..
        } => {
            assert!((0.0..=1.0).contains(&leak_probability));
            assert!(samples_used > 0);
        }
        Outcome::Fail {
            leak_probability, ..
        } => {
            // Constant-time operation should not fail, but handle gracefully
            panic!(
                "Unexpected fail for constant-time operation: P={:.1}%",
                leak_probability * 100.0
            );
        }
        Outcome::Inconclusive {
            leak_probability,
            samples_used,
            ..
        } => {
            assert!((0.0..=1.0).contains(&leak_probability));
            assert!(samples_used > 0);
        }
        Outcome::Unmeasurable { .. } => (),
        Outcome::Research(_) => (),
    }
}

/// Test macro API.
#[test]
#[allow(clippy::redundant_closure)]
fn macro_api() {
    let outcome = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork).time_budget(Duration::from_secs(5)).max_samples(100),
        baseline: || 42u32,
        sample: || rand::random::<u32>(),
        measure: |x| {
            std::hint::black_box(*x);
        },
    };

    match outcome {
        Outcome::Pass {
            leak_probability,
            samples_used,
            ..
        } => {
            assert!((0.0..=1.0).contains(&leak_probability));
            assert!(samples_used > 0);
        }
        Outcome::Fail {
            leak_probability,
            samples_used,
            ..
        } => {
            // Shouldn't happen for constant-time operation
            assert!((0.0..=1.0).contains(&leak_probability));
            assert!(samples_used > 0);
        }
        Outcome::Inconclusive {
            leak_probability,
            samples_used,
            ..
        } => {
            assert!((0.0..=1.0).contains(&leak_probability));
            assert!(samples_used > 0);
        }
        Outcome::Unmeasurable { .. } => (),
        Outcome::Research(_) => (),
    }
}

/// Test result serialization.
#[test]
fn result_serialization() {
    let inputs = InputPair::new(|| (), || ());
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(5))
        .max_samples(100)
        .test(inputs, |_| {});

    // Verify outcome can be serialized to JSON
    let json = serde_json::to_string(&outcome).expect("Should serialize");
    assert!(json.contains("leak_probability") || json.contains("Unmeasurable"));
}
