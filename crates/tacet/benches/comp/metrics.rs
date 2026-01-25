//! Metrics collection for comparing timing analysis tools.

#![allow(dead_code)]

use super::adapters::dudect_adapter::DudectDetector;
use super::adapters::tacet_adapter::TimingOracleDetector;
use super::adapters::Detector;
use super::test_cases::TestCase;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[cfg(feature = "progress-bars")]
use indicatif::{ProgressBar, ProgressStyle};

/// Results of detection rate measurement
#[derive(Debug, Clone)]
pub struct DetectionRateResult {
    pub detections: usize,
    pub total_trials: usize,
    pub detection_rate: f64,
    pub avg_confidence: f64,
    pub avg_duration: Duration,
    pub avg_samples_used: usize,
    pub avg_time_per_sample: Duration,
}

/// Results of sample efficiency measurement (single test)
#[derive(Debug, Clone)]
pub struct SampleEfficiencyResult {
    pub min_samples_to_detect: Option<usize>,
    pub tested_sample_sizes: Vec<usize>,
    pub detection_results: Vec<bool>,
}

/// Sample efficiency statistics across multiple trials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleEfficiencyStats {
    pub median_samples: usize,
    pub ci_lower: usize, // 2.5th percentile
    pub ci_upper: usize, // 97.5th percentile
    pub min_samples: usize,
    pub max_samples: usize,
    pub success_rate: f64, // Fraction of trials that correctly detected
}

/// Measure detection rate (true positive rate) on a test case
pub fn measure_detection_rate(
    detector: &dyn Detector,
    test_case: &dyn TestCase,
    samples: usize,
    trials: usize,
) -> DetectionRateResult {
    let mut detections = 0;
    let mut total_confidence = 0.0;
    let mut total_duration = Duration::ZERO;
    let mut total_samples = 0;

    #[cfg(feature = "progress-bars")]
    let progress = ProgressBar::new(trials as u64)
        .with_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        )
        .with_message(format!("{} - {}", detector.name(), test_case.name()));

    for _trial in 0..trials {
        #[cfg(feature = "progress-bars")]
        progress.set_message(format!(
            "{} - {} - trial {}",
            detector.name(),
            test_case.name(),
            _trial + 1
        ));

        // Prepare the test case before calling detect()
        if let Some(dudect) = detector.as_any().downcast_ref::<DudectDetector>() {
            dudect.prepare_test_case(test_case);
        }
        if let Some(tacet) = detector.as_any().downcast_ref::<TimingOracleDetector>() {
            tacet.prepare_test_case(test_case);
        }

        let fixed_op = test_case.fixed_operation();
        let random_op = test_case.random_operation();

        let result = detector.detect(&|| fixed_op(), &|| random_op(), samples);

        if result.detected_leak {
            detections += 1;
        }
        total_confidence += result.confidence_metric;
        total_duration += result.duration;
        total_samples += result.samples_used;

        #[cfg(feature = "progress-bars")]
        progress.inc(1);
    }

    #[cfg(feature = "progress-bars")]
    progress.finish_with_message(format!(
        "{} - {} complete",
        detector.name(),
        test_case.name()
    ));

    let avg_samples_used = total_samples / trials;
    let avg_duration = total_duration / trials as u32;
    let avg_time_per_sample = if avg_samples_used > 0 {
        Duration::from_secs_f64(avg_duration.as_secs_f64() / avg_samples_used as f64)
    } else {
        Duration::ZERO
    };

    DetectionRateResult {
        detections,
        total_trials: trials,
        detection_rate: detections as f64 / trials as f64,
        avg_confidence: total_confidence / trials as f64,
        avg_duration,
        avg_samples_used,
        avg_time_per_sample,
    }
}

/// Measure false positive rate on a known-safe test case
pub fn measure_false_positive_rate(
    detector: &dyn Detector,
    test_case: &dyn TestCase,
    samples: usize,
    trials: usize,
) -> DetectionRateResult {
    // Same as detection rate, but on safe cases
    measure_detection_rate(detector, test_case, samples, trials)
}

/// Measure sample efficiency - find minimum samples needed to detect leak
pub fn measure_sample_efficiency(
    detector: &dyn Detector,
    test_case: &dyn TestCase,
) -> SampleEfficiencyResult {
    let sample_sizes = vec![1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000];
    let mut detection_results = Vec::new();
    let mut min_samples = None;

    for &samples in &sample_sizes {
        eprintln!(
            "[{}] Sample efficiency test with {} samples for {}",
            detector.name(),
            samples,
            test_case.name()
        );

        // Prepare the test case before calling detect()
        if let Some(dudect) = detector.as_any().downcast_ref::<DudectDetector>() {
            dudect.prepare_test_case(test_case);
        }
        if let Some(tacet) = detector.as_any().downcast_ref::<TimingOracleDetector>() {
            tacet.prepare_test_case(test_case);
        }

        let fixed_op = test_case.fixed_operation();
        let random_op = test_case.random_operation();

        let result = detector.detect(&|| fixed_op(), &|| random_op(), samples);

        let detected = result.detected_leak;
        detection_results.push(detected);

        if detected && min_samples.is_none() {
            min_samples = Some(samples);
        }
    }

    SampleEfficiencyResult {
        min_samples_to_detect: min_samples,
        tested_sample_sizes: sample_sizes,
        detection_results,
    }
}

/// Measure sample efficiency with multiple trials and confidence intervals
///
/// For each test case, runs multiple trials to determine how many samples
/// are needed to reliably detect the leak (or confirm no leak). Returns
/// statistics including median, 95% CI, min, max, and success rate.
pub fn measure_sample_efficiency_stats(
    detector: &dyn Detector,
    test_case: &dyn TestCase,
    sample_sizes: &[usize],
    trials_per_size: usize,
) -> SampleEfficiencyStats {
    let mut all_samples_used = Vec::new();
    let mut successful_detections = 0;
    let expected_leaky = test_case.expected_leaky();

    eprintln!(
        "[{}] Sample efficiency analysis for {} ({} trials per size)",
        detector.name(),
        test_case.name(),
        trials_per_size
    );

    // For DudeCT, sample_sizes are ignored - it uses adaptive sampling
    // So we just run trials_per_size trials and record what it actually uses
    let is_dudect = detector.as_any().downcast_ref::<DudectDetector>().is_some();

    if is_dudect {
        // DudeCT: Run fixed number of trials, record actual samples used
        for trial in 0..trials_per_size {
            eprintln!(
                "    Trial {}/{} (adaptive sampling)",
                trial + 1,
                trials_per_size
            );

            if let Some(dudect) = detector.as_any().downcast_ref::<DudectDetector>() {
                dudect.prepare_test_case(test_case);
            }
            if let Some(tacet) = detector.as_any().downcast_ref::<TimingOracleDetector>() {
                tacet.prepare_test_case(test_case);
            }

            let fixed_op = test_case.fixed_operation();
            let random_op = test_case.random_operation();

            let result = detector.detect(
                &|| fixed_op(),
                &|| random_op(),
                0, // Ignored by DudeCT
            );

            all_samples_used.push(result.samples_used);

            // Check if detection matches expectation
            if result.detected_leak == expected_leaky {
                successful_detections += 1;
            }
        }
    } else {
        // tacet: Test each sample size with multiple trials
        for &samples in sample_sizes {
            for trial in 0..trials_per_size {
                eprintln!(
                    "    Testing {} samples (trial {}/{})",
                    samples,
                    trial + 1,
                    trials_per_size
                );

                // Prepare the test case before calling detect()
                if let Some(tacet) = detector.as_any().downcast_ref::<TimingOracleDetector>() {
                    tacet.prepare_test_case(test_case);
                }

                let fixed_op = test_case.fixed_operation();
                let random_op = test_case.random_operation();

                let result = detector.detect(&|| fixed_op(), &|| random_op(), samples);

                all_samples_used.push(result.samples_used);

                // Check if detection matches expectation
                if result.detected_leak == expected_leaky {
                    successful_detections += 1;
                }
            }
        }
    }

    // Compute statistics
    if all_samples_used.is_empty() {
        return SampleEfficiencyStats {
            median_samples: 0,
            ci_lower: 0,
            ci_upper: 0,
            min_samples: 0,
            max_samples: 0,
            success_rate: 0.0,
        };
    }

    all_samples_used.sort_unstable();

    let n = all_samples_used.len();
    let median_samples = all_samples_used[n / 2];
    let ci_lower = percentile(&all_samples_used, 0.025);
    let ci_upper = percentile(&all_samples_used, 0.975);
    let min_samples = all_samples_used[0];
    let max_samples = all_samples_used[n - 1];
    let total_trials = if is_dudect {
        trials_per_size
    } else {
        sample_sizes.len() * trials_per_size
    };
    let success_rate = successful_detections as f64 / total_trials as f64;

    SampleEfficiencyStats {
        median_samples,
        ci_lower,
        ci_upper,
        min_samples,
        max_samples,
        success_rate,
    }
}

/// Helper function to compute percentile
fn percentile(sorted_data: &[usize], p: f64) -> usize {
    let n = sorted_data.len();
    if n == 0 {
        return 0;
    }
    let index = ((n as f64) * p).floor() as usize;
    sorted_data[index.min(n - 1)]
}

/// Single point on ROC curve with sample counts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocPoint {
    pub threshold: f64,
    pub fpr: f64,
    pub tpr: f64,
    pub avg_samples_leaky: usize,
    pub avg_samples_safe: usize,
}

/// Results of ROC curve analysis for a single detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocCurveResult {
    /// Tool name
    pub detector_name: String,
    /// ROC curve points with sample counts
    pub roc_points: Vec<RocPoint>,
    /// Area under the curve
    pub auc: f64,
}

/// Generate ROC curve by varying detection threshold
pub fn generate_roc_curve(
    detector: &dyn Detector,
    leaky_cases: &[&dyn TestCase],
    safe_cases: &[&dyn TestCase],
    thresholds: &[f64],
    samples: usize,
    trials_per_case: usize,
) -> RocCurveResult {
    let mut roc_points = Vec::new();

    for &threshold in thresholds {
        eprintln!(
            "[{}] ROC analysis at threshold {}",
            detector.name(),
            threshold
        );

        // Measure TPR on leaky cases
        let mut true_positives = 0;
        let mut total_positives = 0;
        let mut total_samples_leaky = 0;

        for test_case in leaky_cases {
            for _ in 0..trials_per_case {
                // Prepare the test case before calling detect()
                if let Some(dudect) = detector.as_any().downcast_ref::<DudectDetector>() {
                    dudect.prepare_test_case(*test_case);
                }
                if let Some(tacet) = detector.as_any().downcast_ref::<TimingOracleDetector>() {
                    tacet.prepare_test_case(*test_case);
                }

                let fixed_op = test_case.fixed_operation();
                let random_op = test_case.random_operation();

                let result = detector.detect(&|| fixed_op(), &|| random_op(), samples);

                total_positives += 1;
                total_samples_leaky += result.samples_used;
                if detector.exceeds_threshold(result.confidence_metric, threshold) {
                    true_positives += 1;
                }
            }
        }

        let tpr = if total_positives > 0 {
            true_positives as f64 / total_positives as f64
        } else {
            0.0
        };
        let avg_samples_leaky = if total_positives > 0 {
            total_samples_leaky / total_positives
        } else {
            0
        };

        // Measure FPR on safe cases
        let mut false_positives = 0;
        let mut total_negatives = 0;
        let mut total_samples_safe = 0;

        for test_case in safe_cases {
            for _ in 0..trials_per_case {
                // Prepare the test case before calling detect()
                if let Some(dudect) = detector.as_any().downcast_ref::<DudectDetector>() {
                    dudect.prepare_test_case(*test_case);
                }
                if let Some(tacet) = detector.as_any().downcast_ref::<TimingOracleDetector>() {
                    tacet.prepare_test_case(*test_case);
                }

                let fixed_op = test_case.fixed_operation();
                let random_op = test_case.random_operation();

                let result = detector.detect(&|| fixed_op(), &|| random_op(), samples);

                total_negatives += 1;
                total_samples_safe += result.samples_used;
                if detector.exceeds_threshold(result.confidence_metric, threshold) {
                    false_positives += 1;
                }
            }
        }

        let fpr = if total_negatives > 0 {
            false_positives as f64 / total_negatives as f64
        } else {
            0.0
        };
        let avg_samples_safe = if total_negatives > 0 {
            total_samples_safe / total_negatives
        } else {
            0
        };

        roc_points.push(RocPoint {
            threshold,
            fpr,
            tpr,
            avg_samples_leaky,
            avg_samples_safe,
        });
    }

    // Calculate AUC using trapezoidal rule
    let auc = calculate_auc(&roc_points);

    RocCurveResult {
        detector_name: detector.name().to_string(),
        roc_points,
        auc,
    }
}

/// Calculate area under ROC curve using trapezoidal rule
fn calculate_auc(roc_points: &[RocPoint]) -> f64 {
    if roc_points.len() < 2 {
        return 0.0;
    }

    // Sort by FPR (x-axis)
    let mut sorted_points = roc_points.to_vec();
    sorted_points.sort_by(|a, b| a.fpr.partial_cmp(&b.fpr).unwrap());

    let mut auc = 0.0;
    for i in 1..sorted_points.len() {
        let x0 = sorted_points[i - 1].fpr;
        let y0 = sorted_points[i - 1].tpr;
        let x1 = sorted_points[i].fpr;
        let y1 = sorted_points[i].tpr;
        // Trapezoidal area
        auc += (x1 - x0) * (y0 + y1) / 2.0;
    }

    auc
}
