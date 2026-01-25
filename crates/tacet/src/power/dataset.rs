//! Power trace data model.
//!
//! This module defines the core data types for representing power traces
//! and datasets in TVLA-style fixed-vs-random analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Class label for TVLA-style analysis.
///
/// Power analysis uses a fixed-vs-random methodology where:
/// - **Fixed**: All traces use the same (fixed) secret input
/// - **Random**: Each trace uses a different random input
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Class {
    /// Fixed input class (e.g., all-zeros key).
    Fixed,
    /// Random input class (different random values per trace).
    Random,
}

impl std::fmt::Display for Class {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Class::Fixed => write!(f, "Fixed"),
            Class::Random => write!(f, "Random"),
        }
    }
}

/// Identifier for a stage/segment within a trace.
///
/// Stages can be defined by markers (e.g., from trigger signals) or
/// by algorithmic segmentation. Each stage is analyzed independently.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StageId(pub String);

impl StageId {
    /// Create a new stage ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Create a numbered stage ID (e.g., "stage_0", "stage_1").
    pub fn numbered(n: usize) -> Self {
        Self(format!("stage_{}", n))
    }
}

impl std::fmt::Display for StageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A marker indicating a stage boundary within a trace.
///
/// Markers are typically derived from trigger signals captured during
/// measurement, indicating the start and end of cryptographic operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    /// Stage identifier.
    pub stage: StageId,
    /// Start sample index (inclusive).
    pub start: usize,
    /// End sample index (exclusive).
    pub end: usize,
}

impl Marker {
    /// Create a new marker.
    pub fn new(stage: StageId, start: usize, end: usize) -> Self {
        debug_assert!(start < end, "Marker start must be before end");
        Self { stage, start, end }
    }

    /// Get the length of the marked segment.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if the marker is empty.
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// Units for power measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PowerUnits {
    /// Raw ADC counts (integer values).
    ADC,
    /// Volts.
    Volts,
    /// Millivolts.
    Millivolts,
    /// Arbitrary/unknown units with a description.
    Arbitrary(String),
}

impl Default for PowerUnits {
    fn default() -> Self {
        Self::Arbitrary("unknown".to_string())
    }
}

impl std::fmt::Display for PowerUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PowerUnits::ADC => write!(f, "ADC"),
            PowerUnits::Volts => write!(f, "V"),
            PowerUnits::Millivolts => write!(f, "mV"),
            PowerUnits::Arbitrary(s) => write!(f, "{}", s),
        }
    }
}

/// A single power trace.
///
/// A trace consists of power samples captured during execution of
/// a cryptographic operation, along with class label and optional markers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    /// Class label (Fixed or Random).
    pub class: Class,
    /// Power samples as floating-point values.
    pub samples: Vec<f32>,
    /// Optional stage markers within the trace.
    pub markers: Option<Vec<Marker>>,
    /// Unique trace identifier.
    pub id: u64,
}

impl Trace {
    /// Create a new trace with the given class and samples.
    pub fn new(class: Class, samples: Vec<f32>) -> Self {
        Self {
            class,
            samples,
            markers: None,
            id: 0,
        }
    }

    /// Create a new trace with ID.
    pub fn with_id(class: Class, samples: Vec<f32>, id: u64) -> Self {
        Self {
            class,
            samples,
            markers: None,
            id,
        }
    }

    /// Set markers for this trace.
    pub fn with_markers(mut self, markers: Vec<Marker>) -> Self {
        self.markers = Some(markers);
        self
    }

    /// Get the number of samples in this trace.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get samples for a specific stage.
    pub fn stage_samples(&self, stage: &StageId) -> Option<&[f32]> {
        self.markers.as_ref().and_then(|markers| {
            markers
                .iter()
                .find(|m| &m.stage == stage)
                .map(|m| &self.samples[m.start..m.end])
        })
    }

    /// Get all stage IDs present in this trace.
    pub fn stage_ids(&self) -> Vec<&StageId> {
        self.markers
            .as_ref()
            .map(|markers| markers.iter().map(|m| &m.stage).collect())
            .unwrap_or_default()
    }
}

/// Metadata for a dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Meta {
    /// Optional description of the dataset.
    pub description: Option<String>,
    /// Optional device/target name.
    pub device: Option<String>,
    /// Optional algorithm being analyzed.
    pub algorithm: Option<String>,
    /// Arbitrary key-value metadata.
    #[serde(default)]
    pub extra: HashMap<String, String>,
}

impl Meta {
    /// Create empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set device name.
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = Some(device.into());
        self
    }

    /// Set algorithm name.
    pub fn with_algorithm(mut self, algo: impl Into<String>) -> Self {
        self.algorithm = Some(algo.into());
        self
    }

    /// Add extra metadata.
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// A dataset of power traces for analysis.
///
/// Datasets contain traces from both Fixed and Random classes,
/// along with measurement parameters and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// All traces in the dataset.
    pub traces: Vec<Trace>,
    /// Units of the power measurements.
    pub units: PowerUnits,
    /// Sample rate in Hz (if known).
    pub sample_rate_hz: Option<f64>,
    /// Dataset metadata.
    pub meta: Meta,
}

impl Dataset {
    /// Create a new dataset from traces.
    pub fn new(traces: Vec<Trace>) -> Self {
        Self {
            traces,
            units: PowerUnits::default(),
            sample_rate_hz: None,
            meta: Meta::default(),
        }
    }

    /// Set the power units.
    pub fn with_units(mut self, units: PowerUnits) -> Self {
        self.units = units;
        self
    }

    /// Set the sample rate.
    pub fn with_sample_rate(mut self, rate_hz: f64) -> Self {
        self.sample_rate_hz = Some(rate_hz);
        self
    }

    /// Set metadata.
    pub fn with_meta(mut self, meta: Meta) -> Self {
        self.meta = meta;
        self
    }

    /// Get the number of traces.
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Get the number of Fixed class traces.
    pub fn fixed_count(&self) -> usize {
        self.traces.iter().filter(|t| t.class == Class::Fixed).count()
    }

    /// Get the number of Random class traces.
    pub fn random_count(&self) -> usize {
        self.traces.iter().filter(|t| t.class == Class::Random).count()
    }

    /// Iterate over Fixed class traces.
    pub fn fixed_traces(&self) -> impl Iterator<Item = &Trace> {
        self.traces.iter().filter(|t| t.class == Class::Fixed)
    }

    /// Iterate over Random class traces.
    pub fn random_traces(&self) -> impl Iterator<Item = &Trace> {
        self.traces.iter().filter(|t| t.class == Class::Random)
    }

    /// Get the trace length (assumes all traces have the same length).
    pub fn trace_length(&self) -> Option<usize> {
        self.traces.first().map(|t| t.len())
    }

    /// Check if all traces have the same length.
    pub fn is_aligned(&self) -> bool {
        if let Some(first_len) = self.trace_length() {
            self.traces.iter().all(|t| t.len() == first_len)
        } else {
            true // Empty dataset is considered aligned
        }
    }

    /// Get all unique stage IDs across all traces.
    pub fn stage_ids(&self) -> Vec<StageId> {
        let mut ids: Vec<StageId> = self
            .traces
            .iter()
            .flat_map(|t| t.stage_ids().into_iter().cloned())
            .collect();
        ids.sort_by(|a, b| a.0.cmp(&b.0));
        ids.dedup();
        ids
    }

    /// Validate the dataset for analysis.
    ///
    /// Returns an error if the dataset is invalid.
    pub fn validate(&self) -> Result<(), DatasetError> {
        if self.traces.is_empty() {
            return Err(DatasetError::Empty);
        }

        if self.fixed_count() == 0 {
            return Err(DatasetError::NoFixedTraces);
        }

        if self.random_count() == 0 {
            return Err(DatasetError::NoRandomTraces);
        }

        if !self.is_aligned() {
            return Err(DatasetError::UnequalTraceLengths);
        }

        if self.traces.iter().any(|t| t.is_empty()) {
            return Err(DatasetError::EmptyTraces);
        }

        Ok(())
    }
}

/// Errors that can occur with datasets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetError {
    /// Dataset has no traces.
    Empty,
    /// Dataset has no Fixed class traces.
    NoFixedTraces,
    /// Dataset has no Random class traces.
    NoRandomTraces,
    /// Traces have different lengths.
    UnequalTraceLengths,
    /// Some traces are empty.
    EmptyTraces,
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::Empty => write!(f, "Dataset is empty"),
            DatasetError::NoFixedTraces => write!(f, "No Fixed class traces"),
            DatasetError::NoRandomTraces => write!(f, "No Random class traces"),
            DatasetError::UnequalTraceLengths => write!(f, "Traces have different lengths"),
            DatasetError::EmptyTraces => write!(f, "Some traces are empty"),
        }
    }
}

impl std::error::Error for DatasetError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_creation() {
        let trace = Trace::new(Class::Fixed, vec![1.0, 2.0, 3.0]);
        assert_eq!(trace.class, Class::Fixed);
        assert_eq!(trace.len(), 3);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_trace_with_markers() {
        let stage = StageId::new("round_1");
        let marker = Marker::new(stage.clone(), 0, 10);
        let trace = Trace::new(Class::Random, vec![0.0; 20]).with_markers(vec![marker]);

        assert!(trace.markers.is_some());
        assert_eq!(trace.stage_ids().len(), 1);
    }

    #[test]
    fn test_dataset_class_counts() {
        let traces = vec![
            Trace::new(Class::Fixed, vec![1.0, 2.0]),
            Trace::new(Class::Fixed, vec![1.1, 2.1]),
            Trace::new(Class::Random, vec![0.5, 1.5]),
        ];
        let dataset = Dataset::new(traces);

        assert_eq!(dataset.fixed_count(), 2);
        assert_eq!(dataset.random_count(), 1);
        assert!(dataset.is_aligned());
    }

    #[test]
    fn test_dataset_validation() {
        // Valid dataset
        let valid = Dataset::new(vec![
            Trace::new(Class::Fixed, vec![1.0]),
            Trace::new(Class::Random, vec![2.0]),
        ]);
        assert!(valid.validate().is_ok());

        // Empty dataset
        let empty = Dataset::new(vec![]);
        assert_eq!(empty.validate(), Err(DatasetError::Empty));

        // No fixed traces
        let no_fixed = Dataset::new(vec![Trace::new(Class::Random, vec![1.0])]);
        assert_eq!(no_fixed.validate(), Err(DatasetError::NoFixedTraces));

        // No random traces
        let no_random = Dataset::new(vec![Trace::new(Class::Fixed, vec![1.0])]);
        assert_eq!(no_random.validate(), Err(DatasetError::NoRandomTraces));

        // Unequal lengths
        let unequal = Dataset::new(vec![
            Trace::new(Class::Fixed, vec![1.0, 2.0]),
            Trace::new(Class::Random, vec![1.0]),
        ]);
        assert_eq!(unequal.validate(), Err(DatasetError::UnequalTraceLengths));
    }
}
