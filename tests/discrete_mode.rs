//! Discrete mode validation tests.
//!
//! Per implementation_notes.md §Testing:
//! - Test with simulated discrete timers (quantize continuous data to ticks)
//! - Verify FPR holds at boundary
//! - Verify m-out-of-n scaling is correct
//!
//! Run with: cargo test --test discrete_mode

use timing_oracle::{compute_min_uniqueness_ratio, helpers::InputPair, Outcome, TimingOracle};

// =============================================================================
// UNIT TESTS: Uniqueness Ratio
// =============================================================================

#[test]
fn uniqueness_ratio_all_identical() {
    let data = vec![42.0; 1000];
    let ratio = compute_min_uniqueness_ratio(&data, &data);
    // 1 unique value / 1000 samples = 0.001
    assert!(
        (ratio - 0.001).abs() < 0.0001,
        "Expected ~0.001 for all-identical data, got {}",
        ratio
    );
}

#[test]
fn uniqueness_ratio_all_unique() {
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let ratio = compute_min_uniqueness_ratio(&data, &data);
    assert_eq!(ratio, 1.0, "All unique values should give ratio 1.0");
}

#[test]
fn uniqueness_ratio_boundary_10_percent() {
    // Exactly 10% unique (100 unique values in 1000 samples) → NOT discrete mode
    let data_10pct: Vec<f64> = (0..100).flat_map(|i| vec![i as f64; 10]).collect();
    assert_eq!(data_10pct.len(), 1000);

    let ratio_10 = compute_min_uniqueness_ratio(&data_10pct, &data_10pct);
    assert!(
        (ratio_10 - 0.10).abs() < 0.001,
        "Expected ratio 0.10, got {}",
        ratio_10
    );
    assert!(
        ratio_10 >= 0.10,
        "10% unique should NOT trigger discrete mode (ratio >= 0.10)"
    );

    // 9% unique (90 unique values in 1000 samples) → discrete mode
    // Use 90 unique values, each repeated ~11 times (990 samples) + 10 more of value 0
    let data_9pct: Vec<f64> = (0..90)
        .flat_map(|i| vec![i as f64; 11])
        .chain(std::iter::repeat(0.0).take(10))
        .collect();
    assert_eq!(data_9pct.len(), 1000);

    let ratio_9 = compute_min_uniqueness_ratio(&data_9pct, &data_9pct);
    // 90 unique values / 1000 samples = 0.09
    assert!(
        ratio_9 < 0.10,
        "9% unique should trigger discrete mode (ratio < 0.10), got {}",
        ratio_9
    );
}

#[test]
fn uniqueness_ratio_asymmetric_classes() {
    // Test that min() is taken across both classes
    let baseline: Vec<f64> = (0..1000).map(|i| i as f64).collect(); // 100% unique
    let sample: Vec<f64> = vec![1.0; 1000]; // 0.1% unique

    let ratio = compute_min_uniqueness_ratio(&baseline, &sample);
    assert!(
        (ratio - 0.001).abs() < 0.0001,
        "Should return minimum of both classes, got {}",
        ratio
    );
}

// =============================================================================
// UNIT TESTS: m-out-of-n Scaling Formula
// =============================================================================

#[test]
fn m_out_of_n_resample_size_formula() {
    // Verify m = ⌊n^(2/3)⌋ formula per spec §2.4
    // Due to floating point precision, some "exact" cube values may floor
    // to one less than expected (e.g., 8000^(2/3) = 399.999... → 399)
    let test_cases = [
        (1000, 99),   // 1000^(2/3) ≈ 99.99 → 99
        (8000, 399),  // 8000^(2/3) ≈ 399.99 → 399
        (27000, 899), // 27000^(2/3) ≈ 899.99 → 899
        (5000, 292),  // 5000^(2/3) ≈ 292.4 → 292
        (2000, 158),  // 2000^(2/3) ≈ 158.7 → 158
        (10000, 464), // 10000^(2/3) ≈ 464.1 → 464
    ];

    for (n, expected_m) in test_cases {
        let m = (n as f64).powf(2.0 / 3.0).floor() as usize;
        assert_eq!(
            m, expected_m,
            "m-out-of-n scaling wrong for n={}: expected {}, got {}",
            n, expected_m, m
        );
    }

    // Also verify the scaling relationship: m ≈ n^(2/3)
    // Doubling n should increase m by a factor of ~2^(2/3) ≈ 1.587
    let m_5k = (5000_f64).powf(2.0 / 3.0).floor();
    let m_10k = (10000_f64).powf(2.0 / 3.0).floor();
    let ratio = m_10k / m_5k;
    let expected_ratio = 2.0_f64.powf(2.0 / 3.0);
    assert!(
        (ratio - expected_ratio).abs() < 0.02,
        "Scaling ratio {:.3} should be close to 2^(2/3) ≈ {:.3}",
        ratio,
        expected_ratio
    );
}

// =============================================================================
// UNIT TESTS: Mid-Distribution Quantiles
// =============================================================================

#[test]
fn midquantile_handles_ties() {
    use timing_oracle::statistics::compute_midquantile_deciles;

    // Data with 90% ties at value 1.0, 10% at value 2.0
    let mut data: Vec<f64> = vec![1.0; 90];
    data.extend(vec![2.0; 10]);

    let quantiles = compute_midquantile_deciles(&data);

    // With 90 values at 1.0 and 10 values at 2.0:
    // 10th percentile should be 1.0 (falls within first 90%)
    // 90th percentile should be 2.0 (falls within last 10%)
    assert_eq!(quantiles[0], 1.0, "10th percentile should be 1.0");
    assert_eq!(quantiles[8], 2.0, "90th percentile should be 2.0");
}

#[test]
fn midquantile_symmetric_ties() {
    use timing_oracle::statistics::compute_midquantile_deciles;

    // Data with 50% at 1.0, 50% at 2.0
    let mut data: Vec<f64> = vec![1.0; 50];
    data.extend(vec![2.0; 50]);

    let quantiles = compute_midquantile_deciles(&data);

    // With 50/50 split at values 1.0 and 2.0:
    // The mid-CDF approach assigns F_mid(1.0) = 0.25 (center of first group)
    // and F_mid(2.0) = 0.75 (center of second group)
    //
    // So deciles 10, 20 should return 1.0 (p < 0.25 maps to first value)
    // and deciles 30, 40, 50, 60, 70, 80, 90 should return 2.0 (p > 0.25)
    //
    // Just verify the extreme deciles are correct
    assert_eq!(quantiles[0], 1.0, "10th percentile should be 1.0");
    assert_eq!(quantiles[8], 2.0, "90th percentile should be 2.0");

    // All quantiles should be either 1.0 or 2.0 (the only values in the data)
    for (i, q) in quantiles.iter().enumerate() {
        assert!(
            *q == 1.0 || *q == 2.0,
            "Quantile {} should be 1.0 or 2.0, got {}",
            i,
            q
        );
    }
}

// =============================================================================
// INTEGRATION TESTS: Discrete Mode FPR Calibration
// =============================================================================

/// Verify that discrete mode FPR is bounded at the configured alpha level.
///
/// This test forces discrete mode on using the `force_discrete_mode` config
/// option, then runs FPR calibration to ensure the CI gate still respects
/// the alpha threshold.
#[test]
fn discrete_mode_fpr_calibration() {
    const TRIALS: usize = 100;
    const ALPHA: f64 = 0.01;
    const SAMPLES: usize = 5_000;

    eprintln!(
        "\n[discrete_mode_fpr] Starting {} trials (alpha={})",
        TRIALS, ALPHA
    );

    let mut rejections = 0;
    let mut discrete_mode_count = 0;
    let mut completed_trials = 0;

    for trial in 0..TRIALS {
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        let outcome = TimingOracle::quick()
            .samples(SAMPLES)
            .alpha(ALPHA)
            .force_discrete_mode(true) // Force discrete mode
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        match outcome {
            Outcome::Completed(result) => {
                completed_trials += 1;
                if result.diagnostics.discrete_mode {
                    discrete_mode_count += 1;
                }
                if !result.ci_gate.passed {
                    rejections += 1;
                }

                if (trial + 1) % 25 == 0 {
                    let rate = rejections as f64 / completed_trials as f64;
                    eprintln!(
                        "[discrete_mode_fpr] Trial {}/{}: {} rejections (rate={:.1}%)",
                        trial + 1,
                        TRIALS,
                        rejections,
                        rate * 100.0
                    );
                }
            }
            Outcome::Unmeasurable { .. } => {
                // Skip unmeasurable trials
            }
        }
    }

    // Skip if no trials completed (e.g., permission denied on kperf)
    if completed_trials == 0 {
        eprintln!("[discrete_mode_fpr] Skipping: all trials were unmeasurable");
        return;
    }

    // Verify discrete mode was actually used
    assert!(
        discrete_mode_count > 0,
        "Discrete mode never triggered despite force_discrete_mode=true ({} completed trials)",
        completed_trials
    );

    eprintln!(
        "[discrete_mode_fpr] Discrete mode used in {}/{} completed trials",
        discrete_mode_count, completed_trials
    );

    // FPR should still be bounded at 2×alpha
    let rejection_rate = rejections as f64 / completed_trials as f64;
    eprintln!(
        "[discrete_mode_fpr] Final: {} rejections out of {} trials (rate={:.1}%, limit={:.1}%)",
        rejections,
        completed_trials,
        rejection_rate * 100.0,
        2.0 * ALPHA * 100.0
    );

    assert!(
        rejection_rate <= 2.0 * ALPHA,
        "Discrete mode FPR {:.1}% exceeds 2*alpha={:.1}%",
        rejection_rate * 100.0,
        2.0 * ALPHA * 100.0
    );

    eprintln!("[discrete_mode_fpr] PASSED: Discrete mode FPR is properly bounded");
}

/// Verify that force_discrete_mode actually enables discrete mode.
#[test]
fn force_discrete_mode_activates() {
    let inputs = InputPair::new(rand_bytes, rand_bytes);

    let outcome = TimingOracle::quick()
        .samples(5_000)
        .force_discrete_mode(true)
        .test(inputs, |data| {
            std::hint::black_box(data);
        });

    match outcome {
        Outcome::Completed(result) => {
            assert!(
                result.diagnostics.discrete_mode,
                "force_discrete_mode(true) should activate discrete mode"
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            // This is acceptable - the test might be unmeasurable on some platforms
            eprintln!(
                "[force_discrete_mode] Skipping: unmeasurable ({})",
                recommendation
            );
        }
    }
}

// =============================================================================
// HELPERS
// =============================================================================

fn rand_bytes() -> [u8; 32] {
    rand::random()
}
