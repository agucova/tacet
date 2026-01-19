//! DudeCT statistical testing implementation.
//!
//! This module is adapted from [dudect-bencher](https://github.com/rozbb/dudect-bencher)
//! by Michael Rosenberg, with modifications for analyzing pre-collected samples.
//!
//! The key feature is percentile-based cropping: rather than just running a single
//! t-test on all data, it runs 101 t-tests (one on full data, 100 at different
//! percentile cutoffs) to catch timing effects that only appear in certain parts
//! of the distribution (e.g., tail effects).

// Copyright 2017-2024 Michael Rosenberg
// Original source: https://github.com/rozbb/dudect-bencher
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option.

use std::cmp;

/// Summary of constant-time test results.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct CtSummary {
    /// Maximum t-statistic across all percentile tests.
    pub max_t: f64,
    /// Maximum tau (t normalized by sqrt(n)) across all tests.
    pub max_tau: f64,
    /// Total sample size used.
    pub sample_size: usize,
}

impl CtSummary {
    /// Format the summary for display.
    pub fn fmt(&self) -> String {
        let &CtSummary {
            max_t,
            max_tau,
            sample_size,
        } = self;
        format!(
            "n == {:+0.3}M, max t = {:+0.5}, max tau = {:+0.5}, (5/tau)^2 = {}",
            (sample_size as f64) / 1_000_000f64,
            max_t,
            max_tau,
            (5f64 / max_tau).powi(2) as usize
        )
    }
}

/// Internal test state tracking means and variances for both samples.
#[derive(Copy, Clone, Debug, Default)]
struct CtTest {
    means: (f64, f64),
    sq_diffs: (f64, f64),
    sizes: (usize, usize),
}

/// Context for running constant-time tests across percentiles.
#[derive(Default)]
pub struct CtCtx {
    tests: Vec<CtTest>,
    percentiles: Vec<f64>,
}

// NaNs are smaller than everything
fn local_cmp(x: f64, y: f64) -> cmp::Ordering {
    use std::cmp::Ordering::{Equal, Greater, Less};
    if y.is_nan() {
        Greater
    } else if x.is_nan() || x < y {
        Less
    } else if x == y {
        Equal
    } else {
        Greater
    }
}

/// Helper function: extract a value representing the `pct` percentile of a sorted sample-set,
/// using linear interpolation. If samples are not sorted, return nonsensical value.
fn percentile_of_sorted(sorted_samples: &[f64], pct: f64) -> f64 {
    assert!(!sorted_samples.is_empty());
    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }
    let zero = 0f64;
    assert!(zero <= pct);
    let hundred = 100f64;
    assert!(pct <= hundred);
    let length = (sorted_samples.len() - 1) as f64;
    let rank = (pct / hundred) * length;
    let lrank = rank.floor();
    let d = rank - lrank;
    let n = lrank as usize;
    let lo = sorted_samples[n];
    let hi = sorted_samples[n + 1];
    lo + (hi - lo) * d
}

/// Return the percentiles at f(1), f(2), ..., f(100) of the runtime distribution, where
/// `f(k) = 1 - 0.5^(10k / 100)`
pub fn prepare_percentiles(durations: &[u64]) -> Vec<f64> {
    let sorted: Vec<f64> = {
        let mut v = durations.to_vec();
        v.sort();
        v.into_iter().map(|d| d as f64).collect()
    };

    // Collect all the percentile values
    (0..100)
        .map(|i| {
            let pct = {
                let exp = f64::from(10 * (i + 1)) / 100f64;
                1f64 - 0.5f64.powf(exp)
            };
            percentile_of_sorted(&sorted, 100f64 * pct)
        })
        .collect()
}

/// Run constant-time statistical tests on two sample sets.
///
/// This is the main entry point for analyzing pre-collected timing measurements.
/// It runs 101 t-tests: one on all data, and 100 on data cropped at different
/// percentile thresholds.
///
/// # Arguments
/// * `ctx` - Optional context from a previous run (for streaming analysis)
/// * `samples` - Tuple of (left_samples, right_samples) as Vec<u64>
///
/// # Returns
/// * `CtSummary` - Summary with max_t, max_tau, and sample_size
/// * `CtCtx` - Context for subsequent streaming updates
///
/// # Example
/// ```ignore
/// let baseline: Vec<u64> = collect_baseline();
/// let test: Vec<u64> = collect_test();
/// let (summary, _ctx) = update_ct_stats(None, &(baseline, test));
/// if summary.max_t.abs() > 4.5 {
///     println!("Timing leak detected!");
/// }
/// ```
pub fn update_ct_stats(
    ctx: Option<CtCtx>,
    (left_samples, right_samples): &(Vec<u64>, Vec<u64>),
) -> (CtSummary, CtCtx) {
    // Only construct the context (that is, percentiles and test structs) on the first run
    let (mut tests, percentiles) = match ctx {
        Some(c) => (c.tests, c.percentiles),
        None => {
            let all_samples = {
                let mut v = left_samples.clone();
                v.extend_from_slice(right_samples);
                v
            };
            let pcts = prepare_percentiles(&all_samples);
            let tests = vec![CtTest::default(); 101];

            (tests, pcts)
        }
    };

    let left_samples: Vec<f64> = left_samples.iter().map(|&n| n as f64).collect();
    let right_samples: Vec<f64> = right_samples.iter().map(|&n| n as f64).collect();

    for &left_sample in left_samples.iter() {
        update_test_left(&mut tests[0], left_sample);
    }
    for &right_sample in right_samples.iter() {
        update_test_right(&mut tests[0], right_sample);
    }

    for (test, &pct) in tests.iter_mut().skip(1).zip(percentiles.iter()) {
        let left_cropped = left_samples.iter().filter(|&&x| x < pct);
        let right_cropped = right_samples.iter().filter(|&&x| x < pct);

        for &left_sample in left_cropped {
            update_test_left(test, left_sample);
        }
        for &right_sample in right_cropped {
            update_test_right(test, right_sample);
        }
    }

    let (max_t, max_tau, sample_size) = {
        // Get the test with the maximum t
        let max_test = tests
            .iter()
            .max_by(|&x, &y| local_cmp(compute_t(x).abs(), compute_t(y).abs()))
            .unwrap();
        let sample_size = max_test.sizes.0 + max_test.sizes.1;
        let max_t = compute_t(max_test);
        let max_tau = max_t / (sample_size as f64).sqrt();

        (max_t, max_tau, sample_size)
    };

    let new_ctx = CtCtx { tests, percentiles };
    let summ = CtSummary {
        max_t,
        max_tau,
        sample_size,
    };

    (summ, new_ctx)
}

/// Compute Welch's t-statistic for a test.
fn compute_t(test: &CtTest) -> f64 {
    let &CtTest {
        means,
        sq_diffs,
        sizes,
    } = test;
    let num = means.0 - means.1;
    let n0 = sizes.0 as f64;
    let n1 = sizes.1 as f64;
    let var0 = sq_diffs.0 / (n0 - 1f64);
    let var1 = sq_diffs.1 / (n1 - 1f64);
    let den = (var0 / n0 + var1 / n1).sqrt();

    num / den
}

/// Update left (baseline) sample statistics using Welford's online algorithm.
fn update_test_left(test: &mut CtTest, datum: f64) {
    test.sizes.0 += 1;
    let diff = datum - test.means.0;
    test.means.0 += diff / (test.sizes.0 as f64);
    test.sq_diffs.0 += diff * (datum - test.means.0);
}

/// Update right (test) sample statistics using Welford's online algorithm.
fn update_test_right(test: &mut CtTest, datum: f64) {
    test.sizes.1 += 1;
    let diff = datum - test.means.1;
    test.means.1 += diff / (test.sizes.1 as f64);
    test.sq_diffs.1 += diff * (datum - test.means.1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_distributions() {
        // Two identical distributions should have t ≈ 0
        let left: Vec<u64> = (0..1000).map(|i| 1000 + (i % 100)).collect();
        let right: Vec<u64> = (0..1000).map(|i| 1000 + (i % 100)).collect();

        let (summary, _) = update_ct_stats(None, &(left, right));

        assert!(
            summary.max_t.abs() < 1.0,
            "Identical distributions should have |t| < 1, got {}",
            summary.max_t
        );
    }

    #[test]
    fn test_different_distributions() {
        // Clearly different distributions should have large |t|
        let left: Vec<u64> = (0..1000).map(|_| 1000).collect();
        let right: Vec<u64> = (0..1000).map(|_| 2000).collect();

        let (summary, _) = update_ct_stats(None, &(left, right));

        assert!(
            summary.max_t.abs() > 10.0,
            "Different distributions should have large |t|, got {}",
            summary.max_t
        );
    }

    #[test]
    fn test_sample_size_tracking() {
        let left: Vec<u64> = vec![100; 500];
        let right: Vec<u64> = vec![100; 300];

        let (summary, _) = update_ct_stats(None, &(left, right));

        // Sample size should be from the test with max |t|
        // For identical data, it should use all samples
        assert!(summary.sample_size > 0);
    }

    #[test]
    fn test_percentile_cropping_catches_tail() {
        // Create distributions that only differ in the tail
        let mut left: Vec<u64> = (0..900).map(|_| 1000).collect();
        left.extend((0..100).map(|_| 5000)); // 10% outliers

        let right: Vec<u64> = (0..1000).map(|_| 1000).collect(); // No outliers

        let (summary, _) = update_ct_stats(None, &(left, right));

        // Should detect the difference due to percentile cropping
        assert!(
            summary.max_t.abs() > 2.0,
            "Should detect tail difference, got |t|={}",
            summary.max_t.abs()
        );
    }
}
