//! Sample collection with randomized interleaved design.
//!
//! Implements a measurement strategy that alternates between Fixed and Random
//! inputs in a randomized order to minimize systematic biases from:
//! - CPU frequency scaling
//! - Cache warming/cooling
//! - Branch predictor state
//! - Other temporal effects

use crate::types::Class;
use rand::seq::SliceRandom;

use super::timer::{black_box, rdtsc, Timer};

/// A single timing measurement sample.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// The input class (Fixed or Random).
    pub class: Class,
    /// The measured execution time in cycles.
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
    /// Suggested iterations per sample (may be overridden by adaptive detection).
    suggested_iterations: usize,
}

impl Collector {
    /// Create a new collector with the given warmup iterations.
    ///
    /// Uses auto-detected iterations per sample based on timer resolution.
    pub fn new(warmup_iterations: usize) -> Self {
        let timer = Timer::new();
        let suggested_iterations = timer.suggested_iterations(10.0);
        Self {
            timer,
            warmup_iterations,
            suggested_iterations,
        }
    }

    /// Create a collector with a pre-calibrated timer.
    ///
    /// Uses auto-detected iterations per sample based on timer resolution.
    pub fn with_timer(timer: Timer, warmup_iterations: usize) -> Self {
        let suggested_iterations = timer.suggested_iterations(10.0);
        Self {
            timer,
            warmup_iterations,
            suggested_iterations,
        }
    }

    /// Create a collector with explicit iterations per sample.
    ///
    /// Use this to override auto-detection when you know the expected
    /// operation duration or want to force batching.
    pub fn with_iterations(timer: Timer, warmup_iterations: usize, iterations_per_sample: usize) -> Self {
        Self {
            timer,
            warmup_iterations,
            suggested_iterations: iterations_per_sample.max(1),
        }
    }

    /// Get a reference to the internal timer.
    pub fn timer(&self) -> &Timer {
        &self.timer
    }

    /// Get the suggested number of iterations per sample.
    ///
    /// Note: Actual iterations may differ if adaptive detection determines
    /// that operations are slow enough to measure individually.
    pub fn iterations_per_sample(&self) -> usize {
        self.suggested_iterations
    }

    /// Run warmup iterations and probe operation duration.
    ///
    /// This helps stabilize CPU frequency, warm caches, and train branch predictors
    /// before actual measurements begin. It also measures a few operations to
    /// estimate their typical duration for adaptive batching decisions.
    ///
    /// # Returns
    ///
    /// The median operation duration in cycles (minimum of fixed and random).
    fn warmup_and_probe<F, R, T>(&self, mut fixed: F, mut random: R) -> u64
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        // Run most warmup iterations without measuring
        let probe_count = 20.min(self.warmup_iterations);
        let warmup_only = self.warmup_iterations.saturating_sub(probe_count);

        for _ in 0..warmup_only {
            black_box(fixed());
            black_box(random());
        }

        // Probe operation duration during final warmup iterations
        let mut fixed_samples = Vec::with_capacity(probe_count);
        let mut random_samples = Vec::with_capacity(probe_count);

        for _ in 0..probe_count {
            let start = rdtsc();
            black_box(fixed());
            let end = rdtsc();
            fixed_samples.push(end.saturating_sub(start));

            let start = rdtsc();
            black_box(random());
            let end = rdtsc();
            random_samples.push(end.saturating_sub(start));
        }

        // Return 90th percentile of both classes combined
        // We use high percentile because if EITHER class has occasional slow
        // operations (tail effects), we want to detect them and avoid batching
        // to preserve variance information
        let percentile_90 = |samples: &mut [u64]| -> u64 {
            if samples.is_empty() {
                return 0;
            }
            samples.sort_unstable();
            let idx = (samples.len() as f64 * 0.9) as usize;
            samples[idx.min(samples.len() - 1)]
        };

        let fixed_p90 = percentile_90(&mut fixed_samples);
        let random_p90 = percentile_90(&mut random_samples);

        fixed_p90.max(random_p90)
    }

    /// Collect timing samples using randomized interleaved design.
    ///
    /// This method:
    /// 1. Runs warmup iterations and probes operation duration
    /// 2. Adaptively decides whether to batch based on operation duration
    /// 3. Creates a randomized schedule alternating Fixed/Random
    /// 4. Measures each execution and records the timing
    ///
    /// Adaptive batching logic:
    /// - If operations are fast (< 5× timer resolution), batch to improve precision
    /// - If operations are slow enough, measure individually to preserve variance
    ///   (important for detecting tail effects)
    ///
    /// # Arguments
    ///
    /// * `samples_per_class` - Number of samples to collect for each class
    /// * `fixed` - Closure that executes with the fixed input
    /// * `random` - Closure that executes with random inputs
    ///
    /// # Returns
    ///
    /// A vector of `Sample` structs containing the measurements.
    pub fn collect<F, R, T>(&self, samples_per_class: usize, mut fixed: F, mut random: R) -> Vec<Sample>
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        // Run warmup and probe operation duration
        let probed_cycles = self.warmup_and_probe(&mut fixed, &mut random);
        let probed_ns = self.timer.cycles_to_ns(probed_cycles);

        // Adaptive batching: if operations are slow enough, don't batch
        // This preserves variance information (important for tail effects)
        // Threshold: operations > 5× timer resolution can be measured individually
        let use_batching = probed_ns < self.timer.resolution_ns() * 5.0;
        let iterations = if use_batching {
            self.suggested_iterations
        } else {
            1
        };

        // Create measurement schedule
        let schedule = self.create_schedule(samples_per_class);

        // Collect measurements
        let mut samples = Vec::with_capacity(samples_per_class * 2);

        if iterations <= 1 {
            // Single-iteration measurement
            for class in schedule {
                let cycles = match class {
                    Class::Fixed => self.timer.measure_cycles(&mut fixed),
                    Class::Random => self.timer.measure_cycles(&mut random),
                };
                samples.push(Sample::new(class, cycles));
            }
        } else {
            // Batched measurement
            for class in schedule {
                let cycles = match class {
                    Class::Fixed => self.measure_batched(&mut fixed, iterations),
                    Class::Random => self.measure_batched(&mut random, iterations),
                };
                samples.push(Sample::new(class, cycles));
            }
        }

        samples
    }

    /// Measure a batch of iterations and return per-iteration cycles.
    #[inline]
    fn measure_batched<F, T>(&self, f: &mut F, iterations: usize) -> u64
    where
        F: FnMut() -> T,
    {
        let start = rdtsc();
        for _ in 0..iterations {
            black_box(f());
        }
        let end = rdtsc();
        end.saturating_sub(start) / iterations as u64
    }

    /// Create a randomized interleaved measurement schedule.
    ///
    /// The schedule ensures equal numbers of Fixed and Random measurements
    /// while randomizing the order to prevent systematic biases.
    fn create_schedule(&self, samples_per_class: usize) -> Vec<Class> {
        let mut rng = rand::rng();

        // Create balanced schedule
        let mut schedule: Vec<Class> = Vec::with_capacity(samples_per_class * 2);
        schedule.extend(std::iter::repeat(Class::Fixed).take(samples_per_class));
        schedule.extend(std::iter::repeat(Class::Random).take(samples_per_class));

        // Shuffle for randomized interleaving
        schedule.shuffle(&mut rng);

        schedule
    }

    /// Collect samples and separate by class.
    ///
    /// Convenience method that collects samples and returns them
    /// separated into Fixed and Random vectors.
    ///
    /// # Returns
    ///
    /// A tuple of (fixed_samples, random_samples) as cycle counts.
    pub fn collect_separated<F, R, T>(
        &self,
        samples_per_class: usize,
        fixed: F,
        random: R,
    ) -> (Vec<u64>, Vec<u64>)
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        let samples = self.collect(samples_per_class, fixed, random);

        let mut fixed_samples = Vec::with_capacity(samples_per_class);
        let mut random_samples = Vec::with_capacity(samples_per_class);

        for sample in samples {
            match sample.class {
                Class::Fixed => fixed_samples.push(sample.cycles),
                Class::Random => random_samples.push(sample.cycles),
            }
        }

        (fixed_samples, random_samples)
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
        let sample = Sample::new(Class::Fixed, 1000);
        assert_eq!(sample.class, Class::Fixed);
        assert_eq!(sample.cycles, 1000);
    }

    #[test]
    fn test_schedule_balanced() {
        let collector = Collector::new(0);
        let schedule = collector.create_schedule(100);

        let fixed_count = schedule.iter().filter(|c| **c == Class::Fixed).count();
        let random_count = schedule.iter().filter(|c| **c == Class::Random).count();

        assert_eq!(fixed_count, 100);
        assert_eq!(random_count, 100);
    }

    #[test]
    fn test_collector_basic() {
        let collector = Collector::new(10);

        let counter = std::sync::atomic::AtomicU64::new(0);
        let (fixed, random) = collector.collect_separated(
            100,
            || counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            || counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        );

        assert_eq!(fixed.len(), 100);
        assert_eq!(random.len(), 100);
    }
}
