//! Acquisition stream model for timing measurements.
//!
//! This module defines the data structure for storing timing measurements in their
//! acquisition order, preserving the temporal dependence structure needed for
//! correct bootstrap resampling.
//!
//! See spec Section 2.3.1 (Acquisition Stream Model).

extern crate alloc;

use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Class label for a timing sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SampleClass {
    /// Fixed (baseline) class - typically all-zeros or a specific input.
    Fixed,
    /// Random class - randomly sampled inputs.
    Random,
}

/// An interleaved acquisition stream of timing measurements.
///
/// Measurement produces an interleaved stream indexed by acquisition time:
/// `{(c_t, y_t)}` where `c_t` is the class label and `y_t` is the timing.
///
/// This structure is critical for correct dependence estimation. The underlying
/// stochastic process operates in continuous time—drift, frequency scaling, and
/// cache state evolution affect nearby samples regardless of class. Bootstrap
/// resampling must preserve adjacency in acquisition order, not per-class position.
///
/// See spec Section 2.3.1 for the full rationale.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AcquisitionStream {
    /// Interleaved (class, timing) pairs in acquisition order.
    samples: Vec<(SampleClass, f64)>,
}

impl AcquisitionStream {
    /// Create a new empty acquisition stream.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Create a new acquisition stream with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
        }
    }

    /// Push a new sample onto the stream.
    #[inline]
    pub fn push(&mut self, class: SampleClass, timing: f64) {
        self.samples.push((class, timing));
    }

    /// Push a batch of samples, interleaving Fixed and Random classes.
    ///
    /// Samples are pushed in interleaved order: F, R, F, R, ...
    /// Both vectors must have the same length.
    pub fn push_batch_interleaved(&mut self, fixed: &[f64], random: &[f64]) {
        debug_assert_eq!(
            fixed.len(),
            random.len(),
            "Fixed and random batches must have same length"
        );

        self.samples.reserve(fixed.len() + random.len());
        for (f, r) in fixed.iter().zip(random.iter()) {
            self.samples.push((SampleClass::Fixed, *f));
            self.samples.push((SampleClass::Random, *r));
        }
    }

    /// Get the total number of samples in the stream.
    #[inline]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if the stream is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get the number of samples per class (assumes balanced classes).
    #[inline]
    pub fn n_per_class(&self) -> usize {
        self.samples.len() / 2
    }

    /// Split the stream into per-class vectors.
    ///
    /// Returns (fixed_timings, random_timings).
    pub fn split_by_class(&self) -> (Vec<f64>, Vec<f64>) {
        let mut fixed = Vec::with_capacity(self.samples.len() / 2);
        let mut random = Vec::with_capacity(self.samples.len() / 2);

        for &(class, timing) in &self.samples {
            match class {
                SampleClass::Fixed => fixed.push(timing),
                SampleClass::Random => random.push(timing),
            }
        }

        (fixed, random)
    }

    /// Get an iterator over all timings (ignoring class labels).
    ///
    /// Used for ACF computation on the pooled stream.
    pub fn timings(&self) -> impl Iterator<Item = f64> + '_ {
        self.samples.iter().map(|&(_, t)| t)
    }

    /// Get a slice of the raw samples.
    pub fn as_slice(&self) -> &[(SampleClass, f64)] {
        &self.samples
    }

    /// Get mutable access to the raw samples.
    pub fn as_mut_slice(&mut self) -> &mut [(SampleClass, f64)] {
        &mut self.samples
    }

    /// Get an iterator over (class, timing) pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(SampleClass, f64)> {
        self.samples.iter()
    }

    /// Clear the stream, removing all samples but keeping capacity.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Truncate the stream to the given length.
    pub fn truncate(&mut self, len: usize) {
        self.samples.truncate(len);
    }

    /// Get the sample at the given index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&(SampleClass, f64)> {
        self.samples.get(index)
    }

    /// Convert raw u64 measurements to nanoseconds and store as stream.
    ///
    /// Interleaves the measurements: baseline[0], sample[0], baseline[1], sample[1], ...
    pub fn from_raw_interleaved(baseline: &[u64], sample: &[u64], ns_per_tick: f64) -> Self {
        debug_assert_eq!(
            baseline.len(),
            sample.len(),
            "Baseline and sample must have same length"
        );

        let mut stream = Self::with_capacity(baseline.len() + sample.len());
        for (b, s) in baseline.iter().zip(sample.iter()) {
            stream.push(SampleClass::Fixed, *b as f64 * ns_per_tick);
            stream.push(SampleClass::Random, *s as f64 * ns_per_tick);
        }
        stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_stream() {
        let stream = AcquisitionStream::new();
        assert!(stream.is_empty());
        assert_eq!(stream.len(), 0);
        assert_eq!(stream.n_per_class(), 0);
    }

    #[test]
    fn test_push_and_split() {
        let mut stream = AcquisitionStream::new();

        // Push interleaved samples
        stream.push(SampleClass::Fixed, 100.0);
        stream.push(SampleClass::Random, 105.0);
        stream.push(SampleClass::Fixed, 101.0);
        stream.push(SampleClass::Random, 106.0);

        assert_eq!(stream.len(), 4);
        assert_eq!(stream.n_per_class(), 2);

        let (fixed, random) = stream.split_by_class();
        assert_eq!(fixed, vec![100.0, 101.0]);
        assert_eq!(random, vec![105.0, 106.0]);
    }

    #[test]
    fn test_push_batch_interleaved() {
        let mut stream = AcquisitionStream::new();

        let fixed = vec![100.0, 101.0, 102.0];
        let random = vec![200.0, 201.0, 202.0];
        stream.push_batch_interleaved(&fixed, &random);

        assert_eq!(stream.len(), 6);

        // Check interleaving order
        assert_eq!(stream.samples[0], (SampleClass::Fixed, 100.0));
        assert_eq!(stream.samples[1], (SampleClass::Random, 200.0));
        assert_eq!(stream.samples[2], (SampleClass::Fixed, 101.0));
        assert_eq!(stream.samples[3], (SampleClass::Random, 201.0));
    }

    #[test]
    fn test_timings_iterator() {
        let mut stream = AcquisitionStream::new();
        stream.push(SampleClass::Fixed, 1.0);
        stream.push(SampleClass::Random, 2.0);
        stream.push(SampleClass::Fixed, 3.0);

        let timings: Vec<f64> = stream.timings().collect();
        assert_eq!(timings, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_raw_interleaved() {
        let baseline = vec![100u64, 110, 120];
        let sample = vec![200u64, 210, 220];
        let ns_per_tick = 2.0;

        let stream = AcquisitionStream::from_raw_interleaved(&baseline, &sample, ns_per_tick);

        assert_eq!(stream.len(), 6);
        assert_eq!(stream.n_per_class(), 3);

        let (fixed, random) = stream.split_by_class();
        assert_eq!(fixed, vec![200.0, 220.0, 240.0]); // 100*2, 110*2, 120*2
        assert_eq!(random, vec![400.0, 420.0, 440.0]); // 200*2, 210*2, 220*2
    }

    #[test]
    fn test_clear_and_truncate() {
        let mut stream = AcquisitionStream::new();
        stream.push(SampleClass::Fixed, 1.0);
        stream.push(SampleClass::Random, 2.0);
        stream.push(SampleClass::Fixed, 3.0);
        stream.push(SampleClass::Random, 4.0);

        stream.truncate(2);
        assert_eq!(stream.len(), 2);

        stream.clear();
        assert!(stream.is_empty());
    }
}
