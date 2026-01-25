//! Realistic timing data generation using tacet's measurement infrastructure.
//!
//! Unlike the synthetic module (which samples from mathematical distributions),
//! this module generates timing data by executing real operations measured with
//! tacet's precise cycle counters and effect injection.
//!
//! # When to Use
//!
//! - **Synthetic (`synthetic.rs`)**: Fast generation, tests statistical methodology,
//!   but doesn't capture real measurement noise or platform artifacts.
//! - **Realistic (`realistic.rs`)**: Slower but captures actual timing characteristics
//!   including timer resolution, platform noise, and measurement overhead.
//!
//! # Example
//!
//! ```ignore
//! use tacet_bench::realistic::{RealisticConfig, collect_realistic_dataset};
//! use tacet::BenchmarkEffect;
//!
//! let config = RealisticConfig {
//!     samples_per_class: 5000,
//!     effect: BenchmarkEffect::FixedDelay { delay_ns: 100 },
//!     ..Default::default()
//! };
//!
//! let dataset = collect_realistic_dataset(&config);
//! ```

use rand::prelude::*;
use std::hint::black_box;
use std::time::Instant;
use tacet::measurement::{BoxedTimer, TimerSpec};
use tacet::{busy_wait_ns, BenchmarkEffect, Class};

/// Configuration for realistic dataset collection.
#[derive(Debug, Clone)]
pub struct RealisticConfig {
    /// Number of samples per class (baseline and test each get this many).
    pub samples_per_class: usize,
    /// Effect to inject into the test class.
    pub effect: BenchmarkEffect,
    /// Random seed for reproducibility of interleaving order.
    pub seed: u64,
    /// Base operation time in nanoseconds. Default is 1000ns (1μs).
    /// This simulates a crypto operation; the effect is added on top for test class.
    pub base_operation_ns: u64,
    /// Warmup iterations before measurement. Default is 1000.
    pub warmup_iterations: usize,
}

impl Default for RealisticConfig {
    fn default() -> Self {
        Self {
            samples_per_class: 5000,
            effect: BenchmarkEffect::Null,
            seed: 42,
            base_operation_ns: 1000, // 1μs base operation
            warmup_iterations: 1000,
        }
    }
}

/// Generated realistic dataset with both interleaved and blocked formats.
#[derive(Debug, Clone)]
pub struct RealisticDataset {
    /// Interleaved samples: (class, value) pairs in random order of acquisition.
    pub interleaved: Vec<(Class, u64)>,
    /// Blocked samples: baseline and test separated.
    pub blocked: RealisticBlockedData,
    /// Actual collection time.
    pub collection_time_ms: u64,
    /// Timer resolution in nanoseconds (for diagnostics).
    pub timer_resolution_ns: f64,
    /// Cycles per nanosecond (for diagnostics).
    pub cycles_per_ns: f64,
}

/// Blocked format for realistic data.
#[derive(Debug, Clone)]
pub struct RealisticBlockedData {
    /// Baseline (control) timing samples in nanoseconds.
    pub baseline: Vec<u64>,
    /// Test (treatment) timing samples in nanoseconds.
    pub test: Vec<u64>,
}

/// Collect a realistic timing dataset using tacet's measurement infrastructure.
///
/// The collection follows these steps:
/// 1. Initialize Timer with cycle counter calibration
/// 2. Warmup phase to stabilize caches and branch predictors
/// 3. Interleaved collection: randomly alternate between baseline and test
/// 4. For each sample, measure using precise cycle counter
///
/// # Arguments
/// * `config` - Configuration specifying effect type, sample count, etc.
///
/// # Returns
/// A `RealisticDataset` with both interleaved (acquisition order) and blocked views.
pub fn collect_realistic_dataset(config: &RealisticConfig) -> RealisticDataset {
    let wall_start = Instant::now();

    // Initialize cycle-accurate timer - panics if unavailable (requires sudo on ARM64)
    // This ensures we get cycle-accurate measurements (~0.3ns resolution)
    // instead of the coarse cntvct_el0 timer (~42ns on Apple Silicon)
    let (mut timer, _fallback_reason) = TimerSpec::RequireCycleAccurate.create_timer();
    let cycles_per_ns = timer.cycles_per_ns();
    let resolution_ns = timer.resolution_ns();

    let mut rng = StdRng::seed_from_u64(config.seed);

    // Create execution plan: random interleaving of baseline/test
    let total_samples = config.samples_per_class * 2;
    let mut plan: Vec<Class> = Vec::with_capacity(total_samples);
    plan.extend(std::iter::repeat_n(Class::Baseline, config.samples_per_class));
    plan.extend(std::iter::repeat_n(Class::Sample, config.samples_per_class));
    plan.shuffle(&mut rng);

    // Warmup phase - run both baseline and sample operations
    for i in 0..config.warmup_iterations {
        let is_sample = i % 2 == 1;
        black_box(measure_operation(&mut timer, config.base_operation_ns, is_sample, &config.effect));
    }

    // Collect samples in interleaved order using precise timer
    let mut interleaved: Vec<(Class, u64)> = Vec::with_capacity(total_samples);
    let mut baseline_samples: Vec<u64> = Vec::with_capacity(config.samples_per_class);
    let mut test_samples: Vec<u64> = Vec::with_capacity(config.samples_per_class);

    for class in &plan {
        let is_sample = matches!(class, Class::Sample);
        let timing_ns = measure_operation(&mut timer, config.base_operation_ns, is_sample, &config.effect);

        interleaved.push((*class, timing_ns));
        match class {
            Class::Baseline => baseline_samples.push(timing_ns),
            Class::Sample => test_samples.push(timing_ns),
        }
    }

    let collection_time_ms = wall_start.elapsed().as_millis() as u64;

    RealisticDataset {
        interleaved,
        blocked: RealisticBlockedData {
            baseline: baseline_samples,
            test: test_samples,
        },
        collection_time_ms,
        timer_resolution_ns: resolution_ns,
        cycles_per_ns,
    }
}

/// Compute effect delay BEFORE measurement to ensure constant-time code paths.
///
/// This is called outside the timed region so RNG overhead doesn't affect measurements.
/// Returns 0 for null effects or when apply_effect is false.
///
/// IMPORTANT: This function ensures identical code paths for baseline and test when
/// effect is Null. Both execute the same match statement and get 0, with the result
/// masked by apply_effect using branchless multiplication. This avoids introducing
/// harness-induced timing differences that could cause false positives in sensitive
/// statistical tests.
#[inline(never)]
fn compute_effect_delay(apply_effect: bool, effect: &BenchmarkEffect) -> u64 {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    // Thread-local RNG for stochastic effects
    thread_local! {
        static RNG: std::cell::RefCell<rand::rngs::ThreadRng> = std::cell::RefCell::new(rand::thread_rng());
    }

    // ALWAYS compute the delay (same code path for baseline and test)
    // This ensures no branch prediction or instruction cache differences
    let delay = match effect {
        BenchmarkEffect::Null => 0,
        BenchmarkEffect::FixedDelay { delay_ns } => *delay_ns,
        BenchmarkEffect::ThetaMultiple { theta_ns, multiplier } => {
            (*theta_ns * *multiplier) as u64
        }
        BenchmarkEffect::EarlyExit { .. } => 0,
        BenchmarkEffect::HammingWeight { .. } => 0,
        BenchmarkEffect::Bimodal { slow_prob, slow_delay_ns } => {
            if *slow_delay_ns == 0 {
                0
            } else {
                RNG.with(|rng| {
                    if rng.borrow_mut().gen::<f64>() < *slow_prob {
                        *slow_delay_ns
                    } else {
                        0
                    }
                })
            }
        }
        BenchmarkEffect::VariableDelay { mean_ns, std_ns } => {
            if *mean_ns == 0 {
                0
            } else {
                RNG.with(|rng| {
                    let normal = Normal::new(*mean_ns as f64, *std_ns as f64)
                        .unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
                    let delay: f64 = normal.sample(&mut *rng.borrow_mut());
                    delay.max(0.0) as u64
                })
            }
        }
        BenchmarkEffect::TailEffect { base_delay_ns, tail_prob, tail_mult } => {
            if *base_delay_ns == 0 {
                0
            } else {
                RNG.with(|rng| {
                    let is_tail = rng.borrow_mut().gen::<f64>() < *tail_prob;
                    if is_tail {
                        (*base_delay_ns as f64 * *tail_mult) as u64
                    } else {
                        *base_delay_ns
                    }
                })
            }
        }
    };

    // Branchless masking: multiply by 1 if apply_effect, 0 otherwise
    // This ensures identical instructions executed regardless of apply_effect
    delay.wrapping_mul(apply_effect as u64)
}

/// Measure a single operation with optional effect injection using precise timer.
///
/// IMPORTANT: For Null effects, this function guarantees identical code paths for
/// baseline and test by bypassing compute_effect_delay entirely. This ensures
/// FPR measurements are not contaminated by harness-induced timing differences.
#[inline(never)]
fn measure_operation(timer: &mut BoxedTimer, base_ns: u64, apply_effect: bool, effect: &BenchmarkEffect) -> u64 {
    // For Null effects, use a dedicated constant-time path that's identical
    // for both baseline and test. This avoids any code path differences from
    // the complex compute_effect_delay function.
    let effect_delay = if matches!(effect, BenchmarkEffect::Null) {
        // Null effect: both baseline and test execute this exact path
        // No branching on apply_effect, no complex match statements
        //
        // black_box prevents the compiler from optimizing based on apply_effect,
        // ensuring identical code generation regardless of its value
        let _ = std::hint::black_box(apply_effect);
        0u64
    } else {
        // Non-null effects: compute the actual delay
        // Code path differences here are acceptable since we're injecting real effects
        compute_effect_delay(apply_effect, effect)
    };

    // Use BoxedTimer::measure_cycles and convert to nanoseconds
    let cycles = timer.measure_cycles(|| {
        busy_wait_ns(base_ns + effect_delay);
    });

    // Convert cycles to nanoseconds
    timer.cycles_to_ns(cycles).round() as u64
}

/// Standard realistic benchmark configurations.
///
/// These mirror the synthetic configs but use actual timing measurements.
pub fn standard_realistic_configs() -> Vec<(&'static str, RealisticConfig)> {
    let base = RealisticConfig::default();

    vec![
        // Null hypothesis tests (FPR validation)
        (
            "realistic-5k-null",
            RealisticConfig {
                samples_per_class: 5000,
                effect: BenchmarkEffect::Null,
                ..base.clone()
            },
        ),
        (
            "realistic-10k-null",
            RealisticConfig {
                samples_per_class: 10000,
                effect: BenchmarkEffect::Null,
                ..base.clone()
            },
        ),
        // Fixed delay effects (power tests)
        (
            "realistic-5k-100ns",
            RealisticConfig {
                samples_per_class: 5000,
                effect: BenchmarkEffect::FixedDelay { delay_ns: 100 },
                ..base.clone()
            },
        ),
        (
            "realistic-5k-500ns",
            RealisticConfig {
                samples_per_class: 5000,
                effect: BenchmarkEffect::FixedDelay { delay_ns: 500 },
                ..base.clone()
            },
        ),
        (
            "realistic-5k-1us",
            RealisticConfig {
                samples_per_class: 5000,
                effect: BenchmarkEffect::FixedDelay { delay_ns: 1000 },
                ..base.clone()
            },
        ),
        // Theta-relative effects (test threshold sensitivity)
        (
            "realistic-5k-2x-theta-100ns",
            RealisticConfig {
                samples_per_class: 5000,
                effect: BenchmarkEffect::ThetaMultiple {
                    theta_ns: 100.0,
                    multiplier: 2.0,
                },
                ..base.clone()
            },
        ),
        (
            "realistic-5k-0.5x-theta-100ns",
            RealisticConfig {
                samples_per_class: 5000,
                effect: BenchmarkEffect::ThetaMultiple {
                    theta_ns: 100.0,
                    multiplier: 0.5,
                },
                ..base
            },
        ),
    ]
}

/// Convert a realistic dataset to the format expected by tool adapters.
pub fn realistic_to_generated(dataset: &RealisticDataset) -> crate::GeneratedDataset {
    crate::GeneratedDataset {
        interleaved: dataset.interleaved.clone(),
        blocked: crate::BlockedData {
            baseline: dataset.blocked.baseline.clone(),
            test: dataset.blocked.test.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Skip test if cycle-accurate timing is not available (requires sudo on ARM64)
    fn skip_if_no_cycle_accurate() -> bool {
        if !TimerSpec::cycle_accurate_available() {
            eprintln!("Skipping test: cycle-accurate timing not available (requires sudo on ARM64)");
            return true;
        }
        false
    }

    #[test]
    fn test_collect_null_dataset() {
        if skip_if_no_cycle_accurate() {
            return;
        }

        let config = RealisticConfig {
            samples_per_class: 100,
            effect: BenchmarkEffect::Null,
            warmup_iterations: 10,
            ..Default::default()
        };

        let dataset = collect_realistic_dataset(&config);

        assert_eq!(dataset.blocked.baseline.len(), 100);
        assert_eq!(dataset.blocked.test.len(), 100);
        assert_eq!(dataset.interleaved.len(), 200);

        // All timings should be > 0
        assert!(dataset.blocked.baseline.iter().all(|&t| t > 0));
        assert!(dataset.blocked.test.iter().all(|&t| t > 0));

        // Timer diagnostics should be reasonable
        assert!(dataset.cycles_per_ns > 0.0);
        assert!(dataset.timer_resolution_ns > 0.0);
    }

    #[test]
    fn test_collect_fixed_delay_dataset() {
        if skip_if_no_cycle_accurate() {
            return;
        }

        let config = RealisticConfig {
            samples_per_class: 100,
            effect: BenchmarkEffect::FixedDelay { delay_ns: 10_000 }, // 10μs delay
            base_operation_ns: 1000,                                   // 1μs base
            warmup_iterations: 10,
            ..Default::default()
        };

        let dataset = collect_realistic_dataset(&config);

        // Test samples should be noticeably slower on average
        let baseline_mean: f64 =
            dataset.blocked.baseline.iter().map(|&x| x as f64).sum::<f64>() / 100.0;
        let test_mean: f64 =
            dataset.blocked.test.iter().map(|&x| x as f64).sum::<f64>() / 100.0;

        // Test should be at least 5μs slower on average (we injected 10μs)
        assert!(
            test_mean > baseline_mean + 5000.0,
            "Test mean {} should be > baseline mean {} + 5000ns",
            test_mean,
            baseline_mean
        );
    }

    #[test]
    fn test_interleaved_ordering() {
        if skip_if_no_cycle_accurate() {
            return;
        }

        let config = RealisticConfig {
            samples_per_class: 50,
            effect: BenchmarkEffect::Null,
            warmup_iterations: 10,
            ..Default::default()
        };

        let dataset = collect_realistic_dataset(&config);

        // Interleaved should have mixed classes (not all baseline first)
        let first_10: Vec<_> = dataset.interleaved.iter().take(10).map(|(c, _)| c).collect();
        let has_baseline = first_10.iter().any(|c| matches!(c, Class::Baseline));
        let has_test = first_10.iter().any(|c| matches!(c, Class::Sample));

        // With 50/50 split and random shuffle, extremely unlikely to have only one class
        // in first 10 samples (probability < 0.002)
        assert!(
            has_baseline && has_test,
            "First 10 samples should have both classes: {:?}",
            first_10
        );
    }

    #[test]
    fn test_deterministic_ordering() {
        if skip_if_no_cycle_accurate() {
            return;
        }

        // Same seed should produce same interleaving order
        let config = RealisticConfig {
            samples_per_class: 20,
            effect: BenchmarkEffect::Null,
            seed: 12345,
            warmup_iterations: 5,
            base_operation_ns: 100,
            ..Default::default()
        };

        let dataset1 = collect_realistic_dataset(&config);
        let dataset2 = collect_realistic_dataset(&config);

        // Class ordering should be identical
        let classes1: Vec<_> = dataset1.interleaved.iter().map(|(c, _)| *c).collect();
        let classes2: Vec<_> = dataset2.interleaved.iter().map(|(c, _)| *c).collect();
        assert_eq!(classes1, classes2);

        // Timing values will differ due to real measurement variability
    }

    #[test]
    fn test_timer_uses_precise_counter() {
        if skip_if_no_cycle_accurate() {
            return;
        }

        let config = RealisticConfig {
            samples_per_class: 10,
            effect: BenchmarkEffect::Null,
            warmup_iterations: 5,
            base_operation_ns: 100,
            ..Default::default()
        };

        let dataset = collect_realistic_dataset(&config);

        // On x86_64, resolution should be ~1ns or less
        // On aarch64 (Apple Silicon), resolution is ~42ns
        // Either way, it should be < 100ns for any modern system
        assert!(
            dataset.timer_resolution_ns < 100.0,
            "Timer resolution {} ns is too coarse",
            dataset.timer_resolution_ns
        );
    }

    // ========================================================================
    // Unit tests for compute_effect_delay
    // ========================================================================

    #[test]
    fn test_compute_effect_delay_null() {
        // Null effect should always return 0
        assert_eq!(compute_effect_delay(true, &BenchmarkEffect::Null), 0);
        assert_eq!(compute_effect_delay(false, &BenchmarkEffect::Null), 0);
    }

    #[test]
    fn test_compute_effect_delay_fixed() {
        let effect = BenchmarkEffect::FixedDelay { delay_ns: 1000 };

        // apply_effect=true should return the delay
        assert_eq!(compute_effect_delay(true, &effect), 1000);

        // apply_effect=false should return 0 (branchless masking)
        assert_eq!(compute_effect_delay(false, &effect), 0);
    }

    #[test]
    fn test_compute_effect_delay_theta_multiple() {
        let effect = BenchmarkEffect::ThetaMultiple {
            theta_ns: 100.0,
            multiplier: 2.5,
        };

        // 100 * 2.5 = 250
        assert_eq!(compute_effect_delay(true, &effect), 250);
        assert_eq!(compute_effect_delay(false, &effect), 0);
    }

    #[test]
    fn test_compute_effect_delay_bimodal() {
        let effect = BenchmarkEffect::Bimodal {
            slow_prob: 1.0, // Always slow for deterministic test
            slow_delay_ns: 5000,
        };

        // With slow_prob=1.0, should always return the slow delay
        assert_eq!(compute_effect_delay(true, &effect), 5000);
        assert_eq!(compute_effect_delay(false, &effect), 0);

        // With slow_prob=0.0, should always return 0
        let effect_never_slow = BenchmarkEffect::Bimodal {
            slow_prob: 0.0,
            slow_delay_ns: 5000,
        };
        assert_eq!(compute_effect_delay(true, &effect_never_slow), 0);
    }

    #[test]
    fn test_compute_effect_delay_bimodal_zero_delay() {
        // Edge case: slow_delay_ns=0 should return 0 regardless of probability
        let effect = BenchmarkEffect::Bimodal {
            slow_prob: 1.0,
            slow_delay_ns: 0,
        };
        assert_eq!(compute_effect_delay(true, &effect), 0);
    }

    #[test]
    fn test_compute_effect_delay_tail_effect() {
        // With tail_prob=1.0, always returns tail delay
        let effect = BenchmarkEffect::TailEffect {
            base_delay_ns: 1000,
            tail_prob: 1.0,
            tail_mult: 3.0,
        };
        assert_eq!(compute_effect_delay(true, &effect), 3000); // 1000 * 3.0
        assert_eq!(compute_effect_delay(false, &effect), 0);

        // With tail_prob=0.0, always returns base delay
        let effect_no_tail = BenchmarkEffect::TailEffect {
            base_delay_ns: 1000,
            tail_prob: 0.0,
            tail_mult: 3.0,
        };
        assert_eq!(compute_effect_delay(true, &effect_no_tail), 1000);
    }

    #[test]
    fn test_compute_effect_delay_tail_effect_zero_base() {
        // Edge case: base_delay_ns=0 should return 0
        let effect = BenchmarkEffect::TailEffect {
            base_delay_ns: 0,
            tail_prob: 1.0,
            tail_mult: 3.0,
        };
        assert_eq!(compute_effect_delay(true, &effect), 0);
    }

    #[test]
    fn test_compute_effect_delay_variable() {
        // With std=0, should return exactly the mean
        let effect = BenchmarkEffect::VariableDelay {
            mean_ns: 500,
            std_ns: 0,
        };
        assert_eq!(compute_effect_delay(true, &effect), 500);
        assert_eq!(compute_effect_delay(false, &effect), 0);
    }

    #[test]
    fn test_compute_effect_delay_variable_zero_mean() {
        // Edge case: mean_ns=0 should return 0
        let effect = BenchmarkEffect::VariableDelay {
            mean_ns: 0,
            std_ns: 100,
        };
        assert_eq!(compute_effect_delay(true, &effect), 0);
    }

    #[test]
    fn test_compute_effect_delay_early_exit_and_hamming() {
        // These effects return 0 (they affect code paths, not delays)
        let early_exit = BenchmarkEffect::EarlyExit { max_delay_ns: 128 };
        let hamming = BenchmarkEffect::HammingWeight { ns_per_bit: 10.0 };

        assert_eq!(compute_effect_delay(true, &early_exit), 0);
        assert_eq!(compute_effect_delay(true, &hamming), 0);
    }

    #[test]
    fn test_branchless_masking_consistency() {
        // Verify that apply_effect=false ALWAYS returns 0 for all effect types
        let effects: Vec<BenchmarkEffect> = vec![
            BenchmarkEffect::Null,
            BenchmarkEffect::FixedDelay { delay_ns: 999 },
            BenchmarkEffect::ThetaMultiple { theta_ns: 100.0, multiplier: 5.0 },
            BenchmarkEffect::Bimodal { slow_prob: 1.0, slow_delay_ns: 888 },
            BenchmarkEffect::VariableDelay { mean_ns: 777, std_ns: 0 },
            BenchmarkEffect::TailEffect { base_delay_ns: 666, tail_prob: 1.0, tail_mult: 2.0 },
        ];

        for effect in &effects {
            assert_eq!(
                compute_effect_delay(false, effect),
                0,
                "apply_effect=false should return 0 for {:?}",
                effect
            );
        }
    }
}
