//! Calibration tests to verify statistical properties.

use timing_oracle::TimingOracle;

/// Verify CI gate false positive rate is bounded.
///
/// Run many trials on pure noise data and check rejection rate <= 2*alpha.
#[test]
#[ignore = "requires full implementation and is slow"]
fn ci_gate_fpr_calibration() {
    const TRIALS: usize = 100;
    const ALPHA: f64 = 0.01;

    let mut rejections = 0;

    for _ in 0..TRIALS {
        // Pure noise: both classes are random
        let result = TimingOracle::new()
            .samples(10_000)
            .ci_alpha(ALPHA)
            .test(|| rand_bytes(), || rand_bytes());

        if !result.ci_gate.passed {
            rejections += 1;
        }
    }

    let rejection_rate = rejections as f64 / TRIALS as f64;
    assert!(
        rejection_rate <= 2.0 * ALPHA,
        "FPR {} exceeds 2*alpha={}",
        rejection_rate,
        2.0 * ALPHA
    );
}

/// Verify Bayesian layer doesn't over-concentrate on high probabilities for null data.
#[test]
#[ignore = "requires full implementation and is slow"]
fn bayesian_calibration() {
    const TRIALS: usize = 100;

    let mut high_prob_count = 0;

    for _ in 0..TRIALS {
        let result = TimingOracle::new()
            .samples(10_000)
            .test(|| rand_bytes(), || rand_bytes());

        if result.leak_probability > 0.9 {
            high_prob_count += 1;
        }
    }

    let high_prob_rate = high_prob_count as f64 / TRIALS as f64;
    assert!(
        high_prob_rate < 0.1,
        "Too many high probabilities on null data: {}",
        high_prob_rate
    );
}

fn rand_bytes() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
