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
use rand::Rng;

use super::timer::{black_box, Timer};

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
}

impl Collector {
    /// Create a new collector with the given warmup iterations.
    pub fn new(warmup_iterations: usize) -> Self {
        Self {
            timer: Timer::new(),
            warmup_iterations,
        }
    }

    /// Create a collector with a pre-calibrated timer.
    pub fn with_timer(timer: Timer, warmup_iterations: usize) -> Self {
        Self {
            timer,
            warmup_iterations,
        }
    }

    /// Get a reference to the internal timer.
    pub fn timer(&self) -> &Timer {
        &self.timer
    }

    /// Run warmup iterations for both functions.
    ///
    /// This helps stabilize CPU frequency, warm caches, and train branch predictors
    /// before actual measurements begin.
    fn warmup<F, R, T>(&self, mut fixed: F, mut random: R)
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        for _ in 0..self.warmup_iterations {
            black_box(fixed());
            black_box(random());
        }
    }

    /// Collect timing samples using randomized interleaved design.
    ///
    /// This method:
    /// 1. Runs warmup iterations
    /// 2. Creates a randomized schedule alternating Fixed/Random
    /// 3. Measures each execution and records the timing
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
        // Run warmup
        self.warmup(&mut fixed, &mut random);

        // Create measurement schedule
        let schedule = self.create_schedule(samples_per_class);

        // Collect measurements
        let mut samples = Vec::with_capacity(samples_per_class * 2);

        for class in schedule {
            let cycles = match class {
                Class::Fixed => self.timer.measure_cycles(&mut fixed),
                Class::Random => self.timer.measure_cycles(&mut random),
            };
            samples.push(Sample::new(class, cycles));
        }

        samples
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

/// Create a randomized interleaved schedule without collecting samples.
///
/// Useful for testing or custom measurement loops.
pub fn create_interleaved_schedule<R: Rng>(rng: &mut R, samples_per_class: usize) -> Vec<Class> {
    let mut schedule: Vec<Class> = Vec::with_capacity(samples_per_class * 2);
    schedule.extend(std::iter::repeat(Class::Fixed).take(samples_per_class));
    schedule.extend(std::iter::repeat(Class::Random).take(samples_per_class));
    schedule.shuffle(rng);
    schedule
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
