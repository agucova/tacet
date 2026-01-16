//! Sample collection with randomized interleaved design.
//!
//! Implements a measurement strategy that alternates between Fixed and Random
//! inputs in a randomized order to minimize systematic biases from:
//! - CPU frequency scaling
//! - Cache warming/cooling
//! - Branch predictor state
//! - Other temporal effects

use crate::result::{BatchingInfo, UnmeasurableInfo};
use crate::types::Class;
use rand::seq::SliceRandom;

use super::timer::{black_box, rdtsc, Timer};

/// Minimum ticks per single call for reliable measurement.
/// Below this, we cannot distinguish timing differences from quantization noise.
pub const MIN_TICKS_SINGLE_CALL: f64 = 5.0;

/// Target ticks per batch for stable quantile-based inference.
/// Below ~50 ticks, the empirical distribution collapses to a sparse PMF.
pub const TARGET_TICKS_PER_BATCH: f64 = 50.0;

/// Maximum batch size to limit microarchitectural state accumulation.
/// Higher values cause false positives from cache/predictor effects.
/// With K ≤ 20, these artifacts are limited.
pub const MAX_BATCH_SIZE: u32 = 20;

/// Number of pilot samples for adaptive K selection.
pub const PILOT_SAMPLES: usize = 100;

/// A single timing measurement sample.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// The input class (Fixed or Random).
    pub class: Class,
    /// The measured execution time in cycles (batch total if batching enabled).
    pub cycles: u64,
}

impl Sample {
    /// Create a new sample.
    pub fn new(class: Class, cycles: u64) -> Self {
        Self { class, cycles }
    }
}

/// Collector for gathering timing measurements with interleaved design.
///
/// The collector alternates between measuring Fixed and Random inputs
/// in a randomized order to minimize systematic biases.
#[derive(Debug)]
pub struct Collector {
    /// The timer used for measurements.
    timer: Timer,
    /// Number of warmup iterations to run before measuring.
    warmup_iterations: usize,
    /// Maximum batch size (set to 1 to disable batching entirely).
    max_batch_size: u32,
    /// Target ticks per batch for adaptive K selection.
    target_ticks_per_batch: f64,
}

impl Collector {
    /// Create a new collector with the given warmup iterations.
    pub fn new(warmup_iterations: usize) -> Self {
        let timer = Timer::new();
        Self {
            timer,
            warmup_iterations,
            max_batch_size: MAX_BATCH_SIZE,
            target_ticks_per_batch: TARGET_TICKS_PER_BATCH,
        }
    }

    /// Create a collector with a pre-calibrated timer.
    pub fn with_timer(timer: Timer, warmup_iterations: usize) -> Self {
        Self {
            timer,
            warmup_iterations,
            max_batch_size: MAX_BATCH_SIZE,
            target_ticks_per_batch: TARGET_TICKS_PER_BATCH,
        }
    }

    /// Create a collector with explicit max batch size (set to 1 to disable batching).
    pub fn with_max_batch_size(timer: Timer, warmup_iterations: usize, max_batch_size: u32) -> Self {
        Self {
            timer,
            warmup_iterations,
            max_batch_size: max_batch_size.max(1),
            target_ticks_per_batch: TARGET_TICKS_PER_BATCH,
        }
    }

    /// Get a reference to the internal timer.
    pub fn timer(&self) -> &Timer {
        &self.timer
    }

    /// Run pilot phase to measure operation duration and select K.
    ///
    /// During warmup, measures ~100 individual operations to determine
    /// median operation time, then selects K to achieve target tick density.
    ///
    /// # Returns
    ///
    /// BatchingInfo including K, ticks_per_batch, rationale, and unmeasurable status.
    fn pilot_and_warmup<F, R, T>(
        &self,
        mut fixed: F,
        mut random: R,
    ) -> BatchingInfo
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        let pilot_count = PILOT_SAMPLES.min(self.warmup_iterations);
        let warmup_only = self.warmup_iterations.saturating_sub(pilot_count);

        // Run initial warmup (without measurement)
        for _ in 0..warmup_only {
            black_box(fixed());
            black_box(random());
        }

        // Pilot phase: measure individual operations
        let mut pilot_cycles = Vec::with_capacity(pilot_count * 2);

        for _ in 0..pilot_count {
            let start = rdtsc();
            black_box(fixed());
            let end = rdtsc();
            pilot_cycles.push(end.saturating_sub(start));

            let start = rdtsc();
            black_box(random());
            let end = rdtsc();
            pilot_cycles.push(end.saturating_sub(start));
        }

        // Compute median operation time
        pilot_cycles.sort_unstable();
        let median_cycles = pilot_cycles[pilot_cycles.len() / 2];
        let median_ns = self.timer.cycles_to_ns(median_cycles);
        let resolution_ns = self.timer.resolution_ns();

        // Calculate ticks per call (how many timer ticks per single operation)
        let ticks_per_call = median_ns / resolution_ns;
        let threshold_ns = resolution_ns * TARGET_TICKS_PER_BATCH / MAX_BATCH_SIZE as f64;

        // Select K to achieve target tick density
        let (k, enabled, unmeasurable, rationale) = if ticks_per_call >= self.target_ticks_per_batch {
            // No batching needed - individual measurements have enough resolution
            (
                1,
                false,
                None,
                format!(
                    "no batching needed ({:.1} ticks/call >= {:.0} target)",
                    ticks_per_call, self.target_ticks_per_batch
                ),
            )
        } else {
            // Need batching to achieve target tick density
            let k_raw = (self.target_ticks_per_batch / ticks_per_call).ceil() as u32;
            let k_attempt = k_raw.clamp(1, self.max_batch_size);
            let actual_ticks = ticks_per_call * k_attempt as f64;

            // Check if even with max batching we're still below measurability threshold
            // Spec §8.2.4: If K*Top < 50*Tres even at K=20, it's Unmeasurable.
            if actual_ticks < self.target_ticks_per_batch {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                let suggestion = ". On macOS, run with sudo to enable kperf cycle counting (~1ns resolution)";
                #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
                let suggestion = ". Run with sudo and --features perf for cycle-accurate timing";
                #[cfg(not(target_arch = "aarch64"))]
                let suggestion = "";

                let rationale = format!(
                    "UNMEASURABLE: {:.1} ticks/batch < {:.0} minimum even at K={} (op ~{:.2}ns, threshold ~{:.2}ns){}",
                    actual_ticks,
                    self.target_ticks_per_batch,
                    k_attempt,
                    median_ns,
                    resolution_ns * self.target_ticks_per_batch / k_attempt as f64,
                    suggestion
                );

                (
                    1, // Revert to k=1 for reporting unmeasurable
                    false,
                    Some(UnmeasurableInfo {
                        operation_ns: median_ns,
                        threshold_ns: resolution_ns * self.target_ticks_per_batch / k_attempt as f64,
                        ticks_per_call,
                    }),
                    rationale,
                )
            } else {
                let partial = actual_ticks < self.target_ticks_per_batch;
                let rationale = if partial {
                    format!(
                        "K={} ({:.1} ticks/batch < {:.0} target, capped at MAX_BATCH_SIZE={})",
                        k_attempt, actual_ticks, self.target_ticks_per_batch, self.max_batch_size
                    )
                } else {
                    format!(
                        "K={} ({:.1} ticks/batch, {:.2} ticks/call, timer res {:.1}ns)",
                        k_attempt, actual_ticks, ticks_per_call, resolution_ns
                    )
                };
                (k_attempt, k_attempt > 1, None, rationale)
            }
        };

        BatchingInfo {
            enabled,
            k,
            ticks_per_batch: ticks_per_call * k as f64,
            rationale,
            unmeasurable,
        }
    }

    /// Collect timing samples using randomized interleaved design.
    ///
    /// This method:
    /// 1. Runs pilot phase to measure operation duration and select K
    /// 2. Runs remaining warmup iterations
    /// 3. Creates a randomized schedule alternating Fixed/Random
    /// 4. Measures each execution and records the timing
    ///
    /// # Adaptive Batching
    ///
    /// When timer resolution is coarse relative to operation duration, batching
    /// multiple iterations recovers effective resolution by measuring aggregate time.
    ///
    /// **Target condition:** `batch_ticks = K × operation_time / timer_resolution > 50`
    ///
    /// Below ~50 ticks, the empirical distribution collapses to a sparse PMF,
    /// making quantile-based inference unstable.
    ///
    /// # What Batching Tests
    ///
    /// Batching changes the estimand: you're testing **amortized cost over K executions**
    /// rather than individual operation timing. This is a valid threat model - real
    /// attackers often average repeated measurements.
    ///
    /// When batching is enabled:
    /// - Samples contain **batch totals** (not divided by K)
    /// - Effect sizes should be divided by K when reporting per-call differences
    ///
    /// # Arguments
    ///
    /// * `samples_per_class` - Number of samples to collect for each class
    /// * `fixed` - Closure that executes with the fixed input
    /// * `random` - Closure that executes with random inputs
    ///
    /// # Returns
    ///
    /// A tuple of (samples, batching_info).
    pub fn collect_with_info<F, R, T>(
        &self,
        samples_per_class: usize,
        mut fixed: F,
        mut random: R,
    ) -> (Vec<Sample>, BatchingInfo)
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        // Run pilot and warmup, get batching configuration
        let batching_info = self.pilot_and_warmup(&mut fixed, &mut random);
        let k = batching_info.k;

        // Create measurement schedule
        let schedule = self.create_schedule(samples_per_class);

        // Collect measurements
        let mut samples = Vec::with_capacity(samples_per_class * 2);

        if k == 1 {
            // Single-iteration measurement
            for class in schedule {
                let cycles = match class {
                    Class::Baseline => self.timer.measure_cycles(&mut fixed),
                    Class::Sample => self.timer.measure_cycles(&mut random),
                };
                samples.push(Sample::new(class, cycles));
            }
        } else {
            // Batched measurement - return batch totals (not divided)
            for class in schedule {
                let cycles = match class {
                    Class::Baseline => self.measure_batch_total(&mut fixed, k),
                    Class::Sample => self.measure_batch_total(&mut random, k),
                };
                samples.push(Sample::new(class, cycles));
            }
        }

        (samples, batching_info)
    }

    /// Collect timing samples (convenience method without batching info).
    ///
    /// For backward compatibility. Use `collect_with_info` to get batching details.
    pub fn collect<F, R, T>(
        &self,
        samples_per_class: usize,
        fixed: F,
        random: R,
    ) -> Vec<Sample>
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        let (samples, _) = self.collect_with_info(samples_per_class, fixed, random);
        samples
    }

    /// Measure a batch of K iterations and return the total cycles (not divided).
    #[inline]
    fn measure_batch_total<F, T>(&self, f: &mut F, k: u32) -> u64
    where
        F: FnMut() -> T,
    {
        let start = rdtsc();
        for _ in 0..k {
            black_box(f());
        }
        let end = rdtsc();
        end.saturating_sub(start)
    }

    /// Create a randomized interleaved measurement schedule.
    ///
    /// The schedule ensures equal numbers of Fixed and Random measurements
    /// while randomizing the order to prevent systematic biases.
    fn create_schedule(&self, samples_per_class: usize) -> Vec<Class> {
        let mut rng = rand::rng();

        // Create balanced schedule
        let mut schedule: Vec<Class> = Vec::with_capacity(samples_per_class * 2);
        schedule.extend(std::iter::repeat(Class::Baseline).take(samples_per_class));
        schedule.extend(std::iter::repeat(Class::Sample).take(samples_per_class));

        // Shuffle for randomized interleaving
        schedule.shuffle(&mut rng);

        schedule
    }

    /// Collect samples and separate by class, with batching info.
    ///
    /// # Returns
    ///
    /// A tuple of (fixed_samples, random_samples, batching_info).
    pub fn collect_separated<F, R, T>(
        &self,
        samples_per_class: usize,
        fixed: F,
        random: R,
    ) -> (Vec<u64>, Vec<u64>, BatchingInfo)
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        let (samples, batching_info) = self.collect_with_info(samples_per_class, fixed, random);

        let mut fixed_samples = Vec::with_capacity(samples_per_class);
        let mut random_samples = Vec::with_capacity(samples_per_class);

        for sample in samples {
            match sample.class {
                Class::Baseline => fixed_samples.push(sample.cycles),
                Class::Sample => random_samples.push(sample.cycles),
            }
        }

        (fixed_samples, random_samples, batching_info)
    }
}

impl Default for Collector {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_creation() {
        let sample = Sample::new(Class::Baseline, 1000);
        assert_eq!(sample.class, Class::Baseline);
        assert_eq!(sample.cycles, 1000);
    }

    #[test]
    fn test_schedule_balanced() {
        let collector = Collector::new(0);
        let schedule = collector.create_schedule(100);

        let baseline_count = schedule.iter().filter(|c| **c == Class::Baseline).count();
        let sample_count = schedule.iter().filter(|c| **c == Class::Sample).count();

        assert_eq!(baseline_count, 100);
        assert_eq!(sample_count, 100);
    }

    #[test]
    fn test_collector_basic() {
        let collector = Collector::new(10);

        let counter = std::sync::atomic::AtomicU64::new(0);
        let (fixed, random, _batching) = collector.collect_separated(
            100,
            || counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            || counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        );

        assert_eq!(fixed.len(), 100);
        assert_eq!(random.len(), 100);
    }

    #[test]
    fn test_unmeasurable_detection() {
        // Test that extremely fast operations are detected as unmeasurable
        let collector = Collector::new(10);

        // A trivial operation that completes in near-zero time
        let (_, batching) = collector.collect_with_info(
            100,
            || 42u8,  // Trivial constant return
            || 42u8,
        );

        // On ARM (41ns resolution), this should be unmeasurable
        // On x86 (0.3ns resolution), it might still measure
        #[cfg(target_arch = "aarch64")]
        {
            // ARM timer has coarse resolution, trivial ops should be unmeasurable
            if batching.ticks_per_batch < MIN_TICKS_SINGLE_CALL {
                assert!(
                    batching.unmeasurable.is_some(),
                    "Expected unmeasurable for trivial op on ARM, got: {:?}",
                    batching
                );
                let info = batching.unmeasurable.as_ref().unwrap();
                assert!(info.ticks_per_call < MIN_TICKS_SINGLE_CALL);
                assert!(batching.rationale.contains("UNMEASURABLE"));
            }
        }

        // On any architecture, if unmeasurable is set, fields should be consistent
        if let Some(ref info) = batching.unmeasurable {
            assert!(info.ticks_per_call < MIN_TICKS_SINGLE_CALL);
            assert!(info.threshold_ns > 0.0);
            assert!(info.operation_ns >= 0.0);
            assert!(!batching.enabled);
            assert_eq!(batching.k, 1);
        }
    }

    #[test]
    fn test_batching_k_selection() {
        // Test K selection logic with a controlled slow operation
        let collector = Collector::new(10);

        // Create an operation slow enough to be measurable but potentially need batching
        let (_, batching) = collector.collect_with_info(
            100,
            || {
                // Busy work to create measurable timing
                let mut x = 0u64;
                for i in 0..1000 {
                    x = x.wrapping_add(black_box(i));
                }
                black_box(x)
            },
            || {
                let mut x = 0u64;
                for i in 0..1000 {
                    x = x.wrapping_add(black_box(i));
                }
                black_box(x)
            },
        );

        // If measurable, verify K selection logic
        if batching.unmeasurable.is_none() {
            // K should be at least 1
            assert!(batching.k >= 1, "K should be at least 1");

            // If batching enabled, K > 1
            if batching.enabled {
                assert!(batching.k > 1, "Batching enabled but K <= 1");
            }

            // K should never exceed MAX_BATCH_SIZE
            assert!(
                batching.k <= MAX_BATCH_SIZE,
                "K {} exceeds MAX_BATCH_SIZE {}",
                batching.k,
                MAX_BATCH_SIZE
            );

            // ticks_per_batch should be reasonable
            assert!(batching.ticks_per_batch > 0.0);
        }
    }

    #[test]
    fn test_max_batch_size_cap() {
        // Create a collector with very low max_batch_size
        let timer = Timer::new();
        let collector = Collector::with_max_batch_size(timer, 10, 5);

        let (_, batching) = collector.collect_with_info(
            100,
            || black_box(42u8),
            || black_box(42u8),
        );

        // K should never exceed the configured max_batch_size
        assert!(
            batching.k <= 5,
            "K {} exceeded configured max_batch_size 5",
            batching.k
        );
    }

    #[test]
    fn test_batching_disabled_for_slow_ops() {
        // Test that slow operations don't get batched
        let collector = Collector::new(10);

        let (_, batching) = collector.collect_with_info(
            50,
            || {
                // Slow operation: enough work to exceed TARGET_TICKS_PER_BATCH
                let mut x = 0u64;
                for i in 0..100_000 {
                    x = x.wrapping_add(black_box(i));
                }
                std::hint::black_box(x)
            },
            || {
                let mut x = 0u64;
                for i in 0..100_000 {
                    x = x.wrapping_add(black_box(i));
                }
                std::hint::black_box(x)
            },
        );

        // A slow enough operation should not need batching
        if batching.unmeasurable.is_none() && batching.ticks_per_batch >= TARGET_TICKS_PER_BATCH {
            assert!(
                !batching.enabled || batching.k == 1,
                "Slow op should not need batching: {:?}",
                batching
            );
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_kperf_suggestion_on_macos_arm64() {
        // On macOS ARM64, unmeasurable operations should suggest kperf
        let collector = Collector::new(10);

        let (_, batching) = collector.collect_with_info(
            100,
            || 42u8,  // Trivial op
            || 42u8,
        );

        // If unmeasurable on macOS ARM64, should mention kperf
        if batching.unmeasurable.is_some() {
            assert!(
                batching.rationale.contains("kperf") || batching.rationale.contains("macOS"),
                "macOS ARM64 unmeasurable should mention kperf, got: {}",
                batching.rationale
            );
        }
    }

    #[test]
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    fn test_suggestion_on_linux_arm64() {
        // On Linux ARM64, unmeasurable operations should suggest --features perf
        let collector = Collector::new(10);

        let (_, batching) = collector.collect_with_info(
            100,
            || 42u8,  // Trivial op
            || 42u8,
        );

        // If unmeasurable on Linux ARM64, should mention perf feature
        if batching.unmeasurable.is_some() {
            assert!(
                batching.rationale.contains("--features perf"),
                "Linux ARM64 unmeasurable should mention --features perf, got: {}",
                batching.rationale
            );
        }
    }

    #[test]
    fn test_batching_info_consistency() {
        // Verify BatchingInfo fields are internally consistent
        let collector = Collector::new(10);

        let (_, batching) = collector.collect_with_info(
            100,
            || {
                let mut x = 0u64;
                for i in 0..500 {
                    x = x.wrapping_add(black_box(i));
                }
                black_box(x)
            },
            || {
                let mut x = 0u64;
                for i in 0..500 {
                    x = x.wrapping_add(black_box(i));
                }
                black_box(x)
            },
        );

        // enabled should match k > 1
        assert_eq!(
            batching.enabled,
            batching.k > 1,
            "enabled={} should match k > 1 (k={})",
            batching.enabled,
            batching.k
        );

        // rationale should not be empty
        assert!(!batching.rationale.is_empty(), "rationale should not be empty");

        // If unmeasurable, k should be 1 and enabled should be false
        if batching.unmeasurable.is_some() {
            assert_eq!(batching.k, 1, "unmeasurable should have k=1");
            assert!(!batching.enabled, "unmeasurable should not be enabled");
        }
    }
}
