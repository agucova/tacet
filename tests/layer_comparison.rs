//! Layer comparison benchmark: CI Gate (1 - p_value) vs Bayesian (leak_probability).
//!
//! This test compares the two analysis layers to help decide which approach is more useful:
//! - CI Gate (Layer 1): Frequentist, outputs p-value
//! - Bayesian (Layer 2): Outputs posterior probability of leak
//!
//! Run with: cargo test layer_power_comparison -- --nocapture
//! Expected runtime: ~3-5 minutes

use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{Outcome, TimingOracle};

/// Base work that both classes do - enough to be measurable.
#[inline(never)]
fn do_base_work(data: &[u8; 32]) -> u64 {
    let mut acc = 0u64;
    // Do enough work to be measurable (~50-100ns)
    for _ in 0..4 {
        for &b in data {
            acc = acc.wrapping_add(b as u64);
        }
    }
    std::hint::black_box(acc)
}

/// Add minimal extra work - just a few operations.
/// k=1 adds ~2-5ns, k=5 adds ~10-25ns, etc.
#[inline(never)]
fn do_extra_ops(data: &[u8; 32], k: usize) -> u64 {
    let mut acc = 0u64;
    for i in 0..k {
        // Each iteration: one add, one array access
        acc = acc.wrapping_add(data[i % 32] as u64);
    }
    std::hint::black_box(acc)
}

/// Configuration for the comparison benchmark.
/// Using smaller effects to stress-test detection near the noise floor.
const TRIALS_PER_CONFIG: usize = 50;
const NULL_TRIALS: usize = 100;
// Target effects in the challenging detection range (15-35ns).
// Applied with 50% probability to create tail effects.
const EFFECT_SIZES_NS: &[f64] = &[12.0, 18.0, 24.0, 30.0];
const SAMPLE_SIZES: &[usize] = &[5_000, 10_000, 20_000];

/// Results for a single configuration.
#[derive(Debug, Clone)]
struct ConfigResult {
    effect_ns: f64,
    samples: usize,
    ci_scores: Vec<f64>,      // 1 - p_value for each trial
    bayes_scores: Vec<f64>,   // leak_probability for each trial
}

impl ConfigResult {
    fn ci_mean(&self) -> f64 {
        self.ci_scores.iter().sum::<f64>() / self.ci_scores.len() as f64
    }

    fn bayes_mean(&self) -> f64 {
        self.bayes_scores.iter().sum::<f64>() / self.bayes_scores.len() as f64
    }

    fn ci_detection_rate(&self) -> f64 {
        self.ci_scores.iter().filter(|&&s| s > 0.5).count() as f64 / self.ci_scores.len() as f64
    }

    fn bayes_detection_rate(&self) -> f64 {
        self.bayes_scores.iter().filter(|&&s| s > 0.5).count() as f64 / self.bayes_scores.len() as f64
    }

    fn agreement_rate(&self) -> f64 {
        let agreements = self.ci_scores.iter().zip(&self.bayes_scores)
            .filter(|(&ci, &bayes)| (ci > 0.5) == (bayes > 0.5))
            .count();
        agreements as f64 / self.ci_scores.len() as f64
    }

    fn correlation(&self) -> f64 {
        let n = self.ci_scores.len() as f64;
        let ci_mean = self.ci_mean();
        let bayes_mean = self.bayes_mean();

        let mut cov = 0.0;
        let mut ci_var = 0.0;
        let mut bayes_var = 0.0;

        for (&ci, &bayes) in self.ci_scores.iter().zip(&self.bayes_scores) {
            cov += (ci - ci_mean) * (bayes - bayes_mean);
            ci_var += (ci - ci_mean).powi(2);
            bayes_var += (bayes - bayes_mean).powi(2);
        }

        if ci_var < 1e-10 || bayes_var < 1e-10 {
            return 0.0;
        }

        cov / (ci_var.sqrt() * bayes_var.sqrt())
    }
}

/// Theta boundary for "minimum effect of concern" (default in Config).
const THETA_NS: f64 = 10.0;

/// Main comparison test.
#[test]
fn layer_power_comparison() {
    eprintln!("\n");
    eprintln!("════════════════════════════════════════════════════════════════");
    eprintln!("  Layer Comparison: CI Gate (1-pval) vs Bayesian (P(leak))");
    eprintln!("════════════════════════════════════════════════════════════════\n");

    // Phase 1: False Positive Rate (null hypothesis)
    eprintln!("Phase 1: FALSE POSITIVE RATE (0ns effect, 10k samples, {} trials)", NULL_TRIALS);
    eprintln!("─────────────────────────────────────────────────────────────────");

    let fpr_result = run_trials(0.0, 10_000, NULL_TRIALS);

    let ci_fpr = fpr_result.ci_detection_rate();
    let bayes_fpr = fpr_result.bayes_detection_rate();
    let (ci_fpr_lo, ci_fpr_hi) = wilson_ci(ci_fpr, NULL_TRIALS);
    let (bayes_fpr_lo, bayes_fpr_hi) = wilson_ci(bayes_fpr, NULL_TRIALS);

    eprintln!("  CI Gate FPR:   {:5.1}% [95% CI: {:4.1}%-{:4.1}%]  (target: ≤5%)",
              ci_fpr * 100.0, ci_fpr_lo * 100.0, ci_fpr_hi * 100.0);
    eprintln!("  Bayesian FPR:  {:5.1}% [95% CI: {:4.1}%-{:4.1}%]",
              bayes_fpr * 100.0, bayes_fpr_lo * 100.0, bayes_fpr_hi * 100.0);
    eprintln!("  Avg scores:    CI={:.3}, Bayes={:.3}", fpr_result.ci_mean(), fpr_result.bayes_mean());
    eprintln!();

    // Phase 1b: FPR at theta boundary
    eprintln!("Phase 1b: FPR AT THETA BOUNDARY ({}ns effect = θ, 10k samples, {} trials)", THETA_NS, NULL_TRIALS);
    eprintln!("─────────────────────────────────────────────────────────────────");
    eprintln!("  (Tests Type I error at the boundary of practical significance)");

    let theta_result = run_trials(THETA_NS, 10_000, NULL_TRIALS);

    let ci_theta_detect = theta_result.ci_detection_rate();
    let bayes_theta_detect = theta_result.bayes_detection_rate();
    let (ci_theta_lo, ci_theta_hi) = wilson_ci(ci_theta_detect, NULL_TRIALS);
    let (bayes_theta_lo, bayes_theta_hi) = wilson_ci(bayes_theta_detect, NULL_TRIALS);

    eprintln!("  CI Gate detection:   {:5.1}% [95% CI: {:4.1}%-{:4.1}%]",
              ci_theta_detect * 100.0, ci_theta_lo * 100.0, ci_theta_hi * 100.0);
    eprintln!("  Bayesian detection:  {:5.1}% [95% CI: {:4.1}%-{:4.1}%]",
              bayes_theta_detect * 100.0, bayes_theta_lo * 100.0, bayes_theta_hi * 100.0);
    eprintln!("  Avg scores:    CI={:.3}, Bayes={:.3}", theta_result.ci_mean(), theta_result.bayes_mean());
    eprintln!("  → At θ boundary, ~50% detection is expected (neither FP nor TP)");
    eprintln!();

    // Phase 2: Detection Power
    eprintln!("Phase 2: DETECTION POWER ({} trials per config)", TRIALS_PER_CONFIG);
    eprintln!("─────────────────────────────────────────────────────────────────\n");

    let mut all_results: Vec<ConfigResult> = Vec::new();

    for &effect_ns in EFFECT_SIZES_NS {
        eprintln!("Effect: {}ns", effect_ns);
        eprintln!("─────────────────────────────────────────────────────────────────");
        eprintln!("Samples │ CI (1-pval) │ Bayes P(leak) │ Agree │ CI>Bayes │ Bayes>CI");
        eprintln!("────────┼─────────────┼───────────────┼───────┼──────────┼─────────");

        for &samples in SAMPLE_SIZES {
            let result = run_trials(effect_ns, samples, TRIALS_PER_CONFIG);

            let ci_detect = result.ci_detection_rate();
            let bayes_detect = result.bayes_detection_rate();
            let agree = result.agreement_rate();

            // Count disagreements
            let ci_only = result.ci_scores.iter().zip(&result.bayes_scores)
                .filter(|(&ci, &bayes)| ci > 0.5 && bayes <= 0.5)
                .count() as f64 / result.ci_scores.len() as f64;
            let bayes_only = result.ci_scores.iter().zip(&result.bayes_scores)
                .filter(|(&ci, &bayes)| ci <= 0.5 && bayes > 0.5)
                .count() as f64 / result.ci_scores.len() as f64;

            eprintln!("{:>7} │  {:5.1}%     │    {:5.1}%      │ {:4.0}% │   {:4.1}%   │  {:4.1}%",
                      samples, ci_detect * 100.0, bayes_detect * 100.0,
                      agree * 100.0, ci_only * 100.0, bayes_only * 100.0);

            all_results.push(result);
        }
        eprintln!();
    }

    // Phase 3: Summary
    eprintln!("════════════════════════════════════════════════════════════════");
    eprintln!("  SUMMARY");
    eprintln!("════════════════════════════════════════════════════════════════\n");

    let overall_ci_detect: f64 = all_results.iter().map(|r| r.ci_detection_rate()).sum::<f64>()
        / all_results.len() as f64;
    let overall_bayes_detect: f64 = all_results.iter().map(|r| r.bayes_detection_rate()).sum::<f64>()
        / all_results.len() as f64;
    let overall_agree: f64 = all_results.iter().map(|r| r.agreement_rate()).sum::<f64>()
        / all_results.len() as f64;

    // Compute overall correlation across all trials
    let all_ci: Vec<f64> = all_results.iter().flat_map(|r| r.ci_scores.clone()).collect();
    let all_bayes: Vec<f64> = all_results.iter().flat_map(|r| r.bayes_scores.clone()).collect();
    let overall_corr = pearson_correlation(&all_ci, &all_bayes);

    eprintln!("False Positive Rate (at 0ns):");
    eprintln!("  CI Gate:   {:5.1}%  (design target: ≤5%)", ci_fpr * 100.0);
    eprintln!("  Bayesian:  {:5.1}%", bayes_fpr * 100.0);
    eprintln!();
    eprintln!("Detection at theta boundary ({}ns):", THETA_NS);
    eprintln!("  CI Gate:   {:5.1}%  (expected: ~50% at boundary)", ci_theta_detect * 100.0);
    eprintln!("  Bayesian:  {:5.1}%", bayes_theta_detect * 100.0);
    eprintln!();
    eprintln!("Average Detection Power (across all effect sizes):");
    eprintln!("  CI Gate:   {:5.1}%", overall_ci_detect * 100.0);
    eprintln!("  Bayesian:  {:5.1}%", overall_bayes_detect * 100.0);
    eprintln!();
    eprintln!("Agreement & Correlation:");
    eprintln!("  Agreement rate: {:5.1}%", overall_agree * 100.0);
    eprintln!("  Correlation:    {:.3}", overall_corr);
    eprintln!();

    // Decision guidance
    eprintln!("════════════════════════════════════════════════════════════════");
    eprintln!("  INTERPRETATION");
    eprintln!("════════════════════════════════════════════════════════════════\n");

    if (overall_ci_detect - overall_bayes_detect).abs() < 0.05 {
        eprintln!("  → Powers are similar (within 5%).");
        if ci_fpr < bayes_fpr * 0.7 {
            eprintln!("  → CI Gate has significantly lower FPR.");
            eprintln!("  → RECOMMENDATION: CI Gate alone may suffice.");
        } else {
            eprintln!("  → FPRs are comparable.");
            eprintln!("  → RECOMMENDATION: Either layer works; choose based on interpretation preference.");
        }
    } else if overall_bayes_detect > overall_ci_detect {
        let power_diff = overall_bayes_detect - overall_ci_detect;
        eprintln!("  → Bayesian has {:.1}% higher power.", power_diff * 100.0);
        if bayes_fpr <= ci_fpr * 1.5 {
            eprintln!("  → FPR difference is acceptable.");
            eprintln!("  → RECOMMENDATION: Keep Bayesian layer for better sensitivity.");
        } else {
            eprintln!("  → But Bayesian has higher FPR ({:.1}% vs {:.1}%).",
                      bayes_fpr * 100.0, ci_fpr * 100.0);
            eprintln!("  → RECOMMENDATION: Trade-off depends on your FPR tolerance.");
        }
    } else {
        let power_diff = overall_ci_detect - overall_bayes_detect;
        eprintln!("  → CI Gate has {:.1}% higher power (unexpected).", power_diff * 100.0);
        eprintln!("  → RECOMMENDATION: Investigate - this is unusual.");
    }

    if overall_corr > 0.9 {
        eprintln!("\n  → High correlation ({:.2}) suggests the layers measure the same thing.", overall_corr);
        eprintln!("  → One layer may be redundant.");
    }

    eprintln!();
}

/// Run trials for a given effect size and sample count.
fn run_trials(effect_ns: f64, samples: usize, trials: usize) -> ConfigResult {
    let mut ci_scores = Vec::with_capacity(trials);
    let mut bayes_scores = Vec::with_capacity(trials);
    let mut unmeasurable_count = 0;

    // Convert effect_ns to number of extra operations.
    // Each operation adds roughly 2-5ns on modern CPUs.
    // k=0: no extra work, k=1: ~2-5ns, k=5: ~10-25ns, etc.
    let extra_ops = if effect_ns < 100.0 {
        // For small effects, use operation count
        (effect_ns / 3.0).round() as usize
    } else {
        0  // Will use spin delay instead
    };
    let use_spin = effect_ns >= 100.0;

    for trial in 0..trials {
        // Use (bool, [u8; 32]) pairs to explicitly track class
        // This avoids any data-dependent timing from checking the data
        let inputs = InputPair::new(
            || (false, [0u8; 32]),  // Baseline: is_sample=false
            || (true, rand::random::<[u8; 32]>()),  // Sample: is_sample=true
        );

        let outcome = TimingOracle::quick()
            .samples(samples)
            .test(inputs, |(is_sample, data)| {
                // Always do the same base work for both classes
                let _ = do_base_work(data);

                // Add timing difference ONLY for sample class, ONLY if effect > 0
                // Apply with 50% probability to create a TAIL EFFECT (harder to detect)
                // Use first byte of data as pseudo-random source
                let should_delay = *is_sample && effect_ns > 0.0 && (data[0] & 1) == 1;

                if should_delay {
                    if use_spin {
                        spin_delay_ns(effect_ns as u64);
                    } else if extra_ops > 0 {
                        let _ = do_extra_ops(data, extra_ops);
                    }
                }
            });

        match outcome {
            Outcome::Completed(result) => {
                // CI Gate score: 1 - p_value (higher = more evidence of leak)
                let ci_score = 1.0 - result.ci_gate.p_value;
                // Bayesian score: leak_probability (higher = more evidence of leak)
                let bayes_score = result.leak_probability;

                ci_scores.push(ci_score);
                bayes_scores.push(bayes_score);
            }
            Outcome::Unmeasurable { .. } => {
                unmeasurable_count += 1;
                // Skip this trial but continue
            }
        }

        // Progress indicator every 25 trials
        if trials >= 50 && (trial + 1) % 25 == 0 {
            eprint!(".");
        }
    }
    if trials >= 50 {
        eprintln!();
    }

    if unmeasurable_count > 0 {
        eprintln!("  [Warning: {} of {} trials were unmeasurable]", unmeasurable_count, trials);
    }

    // If too many trials failed, warn but continue with what we have
    if ci_scores.is_empty() {
        eprintln!("  [ERROR: All trials were unmeasurable - skipping this config]");
        // Return dummy results
        return ConfigResult {
            effect_ns,
            samples,
            ci_scores: vec![0.5],  // Neutral scores
            bayes_scores: vec![0.5],
        };
    }

    ConfigResult {
        effect_ns,
        samples,
        ci_scores,
        bayes_scores,
    }
}

/// Spin-wait for approximately the given number of nanoseconds.
#[inline(never)]
fn spin_delay_ns(ns: u64) {
    let start = std::time::Instant::now();
    let target = Duration::from_nanos(ns);
    while start.elapsed() < target {
        std::hint::spin_loop();
    }
}

/// Wilson score confidence interval for a proportion.
fn wilson_ci(p: f64, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }

    let n_f = n as f64;
    let z = 1.96; // 95% CI
    let z2 = z * z;

    let denom = 1.0 + z2 / n_f;
    let center = (p + z2 / (2.0 * n_f)) / denom;
    let margin = z * ((p * (1.0 - p) + z2 / (4.0 * n_f)) / n_f).sqrt() / denom;

    ((center - margin).max(0.0), (center + margin).min(1.0))
}

/// Pearson correlation coefficient.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let x_mean: f64 = x.iter().sum::<f64>() / n;
    let y_mean: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;

    for (&xi, &yi) in x.iter().zip(y) {
        cov += (xi - x_mean) * (yi - y_mean);
        x_var += (xi - x_mean).powi(2);
        y_var += (yi - y_mean).powi(2);
    }

    if x_var < 1e-10 || y_var < 1e-10 {
        return 0.0;
    }

    cov / (x_var.sqrt() * y_var.sqrt())
}
