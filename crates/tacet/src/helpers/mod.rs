//! Utilities for correct input handling in timing tests.
//!
//! This module also provides effect injection utilities for testing and benchmarking.
//! See [`effect`] for controlled timing delay injection.
//!
//! The most common mistake when using tacet is calling RNG functions
//! or allocating memory inside the measured closures. This creates timing
//! overhead that drowns out the actual signal.
//!
//! [`InputPair`] separates input generation from measurement, ensuring both
//! closures execute identical code paths (only the data differs).
//!
//! # Example
//!
//! ```ignore
//! use tacet::helpers::InputPair;
//!
//! let inputs = InputPair::new(
//!     || [0u8; 32],                 // Baseline: closure (called per sample)
//!     || rand::random::<[u8; 32]>() // Sample: closure (called per sample)
//! );
//!
//! let result = tacet::test(inputs, |input| {
//!     my_function(input);
//! });
//!
//! // Check for common mistakes after measurement
//! if let Some(warning) = inputs.check_anomaly() {
//!     eprintln!("[tacet] {}", warning);
//! }
//! ```
//!
//! # Pre-generation Contract
//!
//! The `baseline` and `sample` closures are:
//! - Called in batch to pre-generate all inputs before timing begins
//! - Never invoked inside the timed region
//! - Never interleaved with measurements
//!
//! Only `measure` runs inside the timed region.
//!
//! # Anomaly Detection
//!
//! `InputPair` tracks the first 1,000 sample values to detect common mistakes:
//!
//! ```ignore
//! // WRONG: Captured pre-evaluated value
//! let value = rand::random::<[u8; 32]>();
//! let inputs = InputPair::new(|| [0u8; 32], || value);  // Always returns same!
//!
//! // After measurement, check_anomaly() will warn about this
//! ```

pub mod effect;

pub use effect::{busy_wait_ns, init_effect_injection, min_injectable_effect_ns, timer_resolution_ns, counter_frequency_hz};

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Number of sample values to track for anomaly detection.
/// Only the first N samples are hashed to limit memory/CPU overhead.
pub const ANOMALY_DETECTION_WINDOW: usize = 1000;

/// Minimum samples before anomaly check is meaningful.
/// With fewer samples, low uniqueness could be coincidence.
pub const ANOMALY_DETECTION_MIN_SAMPLES: usize = 100;

/// Uniqueness threshold for warning.
/// If unique_samples / total_samples < this, emit a warning.
/// 0.5 means: if fewer than 50% of samples are unique, something is wrong.
pub const ANOMALY_DETECTION_THRESHOLD: f64 = 0.5;

/// Pre-generated inputs for timing tests.
///
/// Both `baseline()` and `sample()` call their respective closures to generate
/// values. This symmetric design ensures both classes have identical calling patterns.
///
/// # Type Parameters
///
/// - `T`: The input type (e.g., `[u8; 32]`, `Vec<u8>`)
/// - `F1`: The baseline closure type
/// - `F2`: The sample closure type
///
/// # Example
///
/// ```ignore
/// use tacet::helpers::InputPair;
///
/// // Both are closures (symmetric)
/// let inputs = InputPair::new(
///     || [0u8; 32],          // baseline closure
///     || rand::random(),     // sample closure
/// );
///
/// // Use in test - both call their closures
/// let baseline_val = inputs.baseline();
/// let sample_val = inputs.sample();
/// ```
pub struct InputPair<T, F1, F2> {
    baseline_fn: RefCell<F1>,
    sample_fn: RefCell<F2>,
    // Runtime anomaly detection state (tracks sample values only)
    samples_seen: Cell<usize>,
    unique_samples: RefCell<HashSet<u64>>,
    _phantom: std::marker::PhantomData<T>,
}

// Main implementation for types that are Clone + Hash (with anomaly detection)
impl<T, F1, F2> InputPair<T, F1, F2>
where
    T: Clone + Hash,
    F1: FnMut() -> T,
    F2: FnMut() -> T,
{
    /// Create a new input pair with anomaly detection.
    ///
    /// Both arguments are closures that generate input values.
    ///
    /// # Arguments
    ///
    /// - `baseline`: Closure that generates the baseline value (typically constant)
    /// - `sample`: Closure that generates varied sample values
    ///
    /// # Type Requirements
    ///
    /// - `T: Clone` for internal operations
    /// - `T: Hash` for anomaly detection (use [`new_untracked`](Self::new_untracked) for non-Hash types)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let inputs = InputPair::new(
    ///     || [0u8; 32],                // baseline: returns constant value
    ///     || rand::random::<[u8; 32]>() // sample: generates varied values
    /// );
    /// ```
    pub fn new(baseline: F1, sample: F2) -> Self {
        Self {
            baseline_fn: RefCell::new(baseline),
            sample_fn: RefCell::new(sample),
            samples_seen: Cell::new(0),
            unique_samples: RefCell::new(HashSet::new()),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new input pair without anomaly detection.
    ///
    /// Use this when you intentionally use deterministic inputs (e.g., fixed nonces,
    /// pre-generated signatures) and want to suppress the "identical values" warning.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Testing with fixed nonces - intentionally deterministic
    /// let nonces = InputPair::new_unchecked(
    ///     || [0x00u8; 12],  // Fixed nonce A
    ///     || [0xFFu8; 12],  // Fixed nonce B
    /// );
    /// ```
    pub fn new_unchecked(baseline: F1, sample: F2) -> Self {
        Self {
            baseline_fn: RefCell::new(baseline),
            sample_fn: RefCell::new(sample),
            samples_seen: Cell::new(usize::MAX), // Disable tracking
            unique_samples: RefCell::new(HashSet::new()),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Generate a baseline value by calling the baseline closure.
    ///
    /// Returns `T` by calling the baseline closure each time.
    ///
    /// ```ignore
    /// let inputs = InputPair::new(|| [0u8; 32], || rand::random());
    /// let val = inputs.baseline();  // Calls || [0u8; 32]
    /// ```
    #[inline]
    pub fn baseline(&self) -> T {
        (self.baseline_fn.borrow_mut())()
    }

    /// Generate a sample value with anomaly tracking.
    ///
    /// Calls the sample closure and tracks the hash for anomaly detection
    /// (only the first [`ANOMALY_DETECTION_WINDOW`] samples are tracked).
    ///
    /// **Note:** This method includes tracking overhead. For timing-critical code,
    /// use `generate_sample()` during pre-generation and check anomalies after.
    #[inline]
    pub fn sample(&self) -> T {
        let value = self.generate_sample();

        // Track for anomaly detection (only first N samples)
        let count = self.samples_seen.get();
        if count < ANOMALY_DETECTION_WINDOW {
            self.samples_seen.set(count + 1);
            // Hash-based uniqueness tracking (avoids storing full values)
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            self.unique_samples.borrow_mut().insert(hasher.finish());
        }

        value
    }

    /// Generate a sample value without anomaly tracking.
    ///
    /// Use this for pre-generating inputs before measurement. After pre-generation,
    /// call `track_value()` on each value to enable anomaly detection.
    ///
    /// This is used internally by `TimingOracle::test()` to avoid overhead
    /// during the measurement loop.
    #[inline]
    pub fn generate_sample(&self) -> T {
        (self.sample_fn.borrow_mut())()
    }

    /// Generate a baseline value without any overhead.
    ///
    /// Identical to `baseline()` but named for symmetry with `generate_sample()`.
    #[inline]
    pub fn generate_baseline(&self) -> T {
        (self.baseline_fn.borrow_mut())()
    }

    /// Track a sample value for anomaly detection.
    ///
    /// Call this on pre-generated sample values to enable anomaly detection without
    /// adding overhead during measurement.
    #[inline]
    pub fn track_value(&self, value: &T) {
        let count = self.samples_seen.get();
        if count < ANOMALY_DETECTION_WINDOW {
            self.samples_seen.set(count + 1);
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            self.unique_samples.borrow_mut().insert(hasher.finish());
        }
    }

    /// Check if the sample generator appears to be producing constant values.
    ///
    /// Returns a warning message if anomaly detected, `None` otherwise.
    /// Should be called after measurement completes.
    ///
    /// # Detected Anomalies
    ///
    /// | Condition | Severity | Message |
    /// |-----------|----------|---------|
    /// | All samples identical | Error | Likely captured pre-evaluated value |
    /// | <50% unique samples | Warning | Low entropy, possible mistake |
    /// | Normal entropy | OK | None |
    ///
    /// # Example
    ///
    /// ```ignore
    /// // After running the test
    /// if let Some(warning) = inputs.check_anomaly() {
    ///     eprintln!("[tacet] {}", warning);
    /// }
    /// ```
    pub fn check_anomaly(&self) -> Option<String> {
        let count = self.samples_seen.get();

        // Check if tracking was disabled (new_unchecked)
        if count == usize::MAX {
            return None; // Anomaly detection explicitly disabled
        }

        if count < ANOMALY_DETECTION_MIN_SAMPLES {
            return None; // Not enough samples to judge
        }

        let unique = self.unique_samples.borrow().len();
        let unique_ratio = unique as f64 / count as f64;

        if unique == 1 {
            Some(format!(
                "ANOMALY: sample() returned identical values for all {} samples.\n\
                 \n\
                 Common causes:\n\
                 1. CLOSURE CAPTURE BUG: You may have captured a pre-evaluated value.\n\
                    ❌ Bad:  let val = random(); InputPair::new(|| baseline, || val)\n\
                    ✓  Good: InputPair::new(|| baseline, || random())\n\
                 \n\
                 2. INTENTIONAL (testing with fixed inputs): This is OK if you're testing\n\
                    deterministic operations (e.g., fixed nonces, pre-generated signatures).\n\
                    You can safely ignore this warning in that case.\n\
                 \n\
                 To suppress this warning, use InputPair::new_unchecked() instead.",
                count
            ))
        } else if unique_ratio < ANOMALY_DETECTION_THRESHOLD {
            Some(format!(
                "WARNING: sample() produced only {} unique values out of {} samples \
                 ({:.1}% unique). Expected high entropy for sample inputs.",
                unique,
                count,
                unique_ratio * 100.0
            ))
        } else {
            None
        }
    }
}

// Implementation for types that are Clone but not Hash (untracked version)
impl<T, F1, F2> InputPair<T, F1, F2>
where
    T: Clone,
    F1: FnMut() -> T,
    F2: FnMut() -> T,
{
    /// Create without anomaly detection (for non-Hash types).
    ///
    /// Use this when `T` doesn't implement `Hash` (e.g., cryptographic scalars,
    /// field elements, big integers).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Scalar type that doesn't implement Hash
    /// let inputs = InputPair::new_untracked(
    ///     || Scalar::zero(),
    ///     || Scalar::random(&mut rng)
    /// );
    /// ```
    pub fn new_untracked(baseline: F1, sample: F2) -> Self {
        Self {
            baseline_fn: RefCell::new(baseline),
            sample_fn: RefCell::new(sample),
            samples_seen: Cell::new(0),
            unique_samples: RefCell::new(HashSet::new()),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Generate a baseline value (untracked version).
    #[inline]
    pub fn baseline_untracked(&self) -> T {
        (self.baseline_fn.borrow_mut())()
    }

    /// Generate a sample value without tracking (for non-Hash types).
    #[inline]
    pub fn sample_untracked(&self) -> T {
        (self.sample_fn.borrow_mut())()
    }

    /// Check anomaly - always returns None for untracked InputPairs.
    ///
    /// This method exists so code can call `check_anomaly()` uniformly
    /// without knowing whether tracking is enabled.
    pub fn check_anomaly_untracked(&self) -> Option<String> {
        None
    }
}

// Convenience constructors for common types

/// Helper for 32-byte arrays (common in cryptography).
///
/// Creates an `InputPair` with:
/// - Baseline: all zeros `[0u8; 32]`
/// - Sample: `rand::random()`
///
/// # Example
///
/// ```ignore
/// use tacet::helpers::byte_arrays_32;
///
/// let inputs = byte_arrays_32();
/// tacet::test(inputs, |input| {
///     encrypt(input);
/// });
/// ```
pub fn byte_arrays_32() -> InputPair<[u8; 32], impl FnMut() -> [u8; 32], impl FnMut() -> [u8; 32]> {
    InputPair::new(|| [0u8; 32], rand::random)
}

/// Helper for byte vectors of specific length.
///
/// Creates an `InputPair` with:
/// - Baseline: all zeros `vec![0u8; len]`
/// - Sample: random bytes of length `len`
///
/// # Example
///
/// ```ignore
/// use tacet::helpers::byte_vecs;
///
/// let inputs = byte_vecs(1024);
/// tacet::test(inputs, |input| {
///     encrypt(input);
/// });
/// ```
pub fn byte_vecs(
    len: usize,
) -> InputPair<Vec<u8>, impl FnMut() -> Vec<u8>, impl FnMut() -> Vec<u8>> {
    InputPair::new(
        move || vec![0u8; len],
        move || (0..len).map(|_| rand::random()).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_pair_basic() {
        let inputs = InputPair::new(|| 42u64, || 100u64);

        assert_eq!(inputs.baseline(), 42);
        assert_eq!(inputs.sample(), 100);
        assert_eq!(inputs.baseline(), 42); // Still 42 (closure returns constant)
    }

    #[test]
    fn test_input_pair_generator_called_each_time() {
        let counter = Cell::new(0u64);
        let inputs = InputPair::new(
            || 0u64,
            || {
                let val = counter.get();
                counter.set(val + 1);
                val
            },
        );

        assert_eq!(inputs.sample(), 0);
        assert_eq!(inputs.sample(), 1);
        assert_eq!(inputs.sample(), 2);
    }

    #[test]
    fn test_anomaly_detection_constant() {
        let constant_value = 42u64;
        let inputs = InputPair::new(|| 0u64, || constant_value);

        // Generate enough samples to trigger detection
        for _ in 0..200 {
            let _ = inputs.sample();
        }

        let anomaly = inputs.check_anomaly();
        assert!(anomaly.is_some());
        assert!(anomaly.unwrap().contains("ANOMALY"));
    }

    #[test]
    fn test_anomaly_detection_good_entropy() {
        let counter = Cell::new(0u64);
        let inputs = InputPair::new(
            || 0u64,
            || {
                let val = counter.get();
                counter.set(val + 1);
                val // Each value is unique
            },
        );

        for _ in 0..200 {
            let _ = inputs.sample();
        }

        assert!(inputs.check_anomaly().is_none());
    }

    #[test]
    fn test_anomaly_detection_low_entropy() {
        let counter = Cell::new(0u64);
        let inputs = InputPair::new(
            || 0u64,
            || {
                let val = counter.get() % 10; // Only 10 unique values
                counter.set(counter.get() + 1);
                val
            },
        );

        for _ in 0..200 {
            let _ = inputs.sample();
        }

        let anomaly = inputs.check_anomaly();
        assert!(anomaly.is_some());
        assert!(anomaly.unwrap().contains("WARNING"));
    }

    #[test]
    fn test_anomaly_detection_insufficient_samples() {
        let inputs = InputPair::new(|| 0u64, || 42u64);

        // Only 50 samples - below minimum
        for _ in 0..50 {
            let _ = inputs.sample();
        }

        // Should return None (not enough samples)
        assert!(inputs.check_anomaly().is_none());
    }

    #[test]
    fn test_untracked_version() {
        let inputs = InputPair::new_untracked(|| 0u64, || 42u64);

        assert_eq!(inputs.baseline_untracked(), 0);
        assert_eq!(inputs.sample_untracked(), 42);
        assert!(inputs.check_anomaly_untracked().is_none());
    }

    #[test]
    fn test_byte_arrays_32() {
        let inputs = byte_arrays_32();
        assert_eq!(inputs.baseline(), [0u8; 32]);
        // sample() should return different values (with high probability)
        let r1 = inputs.sample();
        let r2 = inputs.sample();
        // Very unlikely to be equal if truly random
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_byte_vecs() {
        let inputs = byte_vecs(64);
        assert_eq!(inputs.baseline(), vec![0u8; 64]);
        assert_eq!(inputs.baseline().len(), 64);
        assert_eq!(inputs.sample().len(), 64);
    }
}
