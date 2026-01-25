//! Data loading utilities for pre-collected timing measurements.
//!
//! This module provides utilities for loading timing data from external sources,
//! enabling analysis of measurements collected by other tools (SILENT, dudect, etc.)
//! or historical data.
//!
//! # Supported Formats
//!
//! - **SILENT format**: CSV with `V1,V2` header, group labels (X/Y) in first column
//! - **Generic two-column**: Any CSV with group label and timing value columns
//! - **Separate files**: Two files, one per group
//!
//! # Example
//!
//! ```ignore
//! use tacet::data::{load_silent_csv, TimeUnit};
//! use std::path::Path;
//!
//! // Load SILENT-format data
//! let data = load_silent_csv(Path::new("measurements.csv"))?;
//! println!("Loaded {} baseline, {} test samples",
//!          data.baseline_samples.len(),
//!          data.test_samples.len());
//! ```

mod csv;
mod units;

pub use csv::{load_separate_files, load_silent_csv, load_two_column_csv};
pub use units::{to_nanoseconds, TimeUnit};

use std::fmt;

/// Errors that can occur during data loading.
#[derive(Debug)]
pub enum DataError {
    /// IO error reading file.
    Io(std::io::Error),

    /// CSV parse error at a specific line.
    Parse {
        /// Line number where the error occurred (1-indexed).
        line: usize,
        /// Description of the parse error.
        message: String,
    },

    /// Missing required group in data.
    MissingGroup {
        /// The group label that was expected but not found.
        expected: String,
        /// The group labels that were actually found in the data.
        found: Vec<String>,
    },

    /// Insufficient samples for analysis.
    InsufficientSamples {
        /// Name of the group with insufficient samples.
        group: String,
        /// Number of samples found.
        got: usize,
        /// Minimum number of samples required.
        min: usize,
    },

    /// Invalid time value.
    InvalidValue {
        /// Line number where the invalid value was found (1-indexed).
        line: usize,
        /// The invalid value string.
        value: String,
    },
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::Io(e) => write!(f, "IO error: {}", e),
            DataError::Parse { line, message } => {
                write!(f, "Parse error at line {}: {}", line, message)
            }
            DataError::MissingGroup { expected, found } => {
                write!(
                    f,
                    "Missing group '{}' in data. Found groups: {:?}",
                    expected, found
                )
            }
            DataError::InsufficientSamples { group, got, min } => {
                write!(
                    f,
                    "Insufficient samples for group '{}': got {}, need at least {}",
                    group, got, min
                )
            }
            DataError::InvalidValue { line, value } => {
                write!(f, "Invalid timing value at line {}: '{}'", line, value)
            }
        }
    }
}

impl std::error::Error for DataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DataError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DataError {
    fn from(e: std::io::Error) -> Self {
        DataError::Io(e)
    }
}

/// Loaded timing data with two sample groups.
///
/// Represents timing measurements split into two groups for comparison:
/// - `baseline_samples`: Control/reference measurements (e.g., "X" group in SILENT)
/// - `test_samples`: Test/treatment measurements (e.g., "Y" group in SILENT)
#[derive(Debug, Clone)]
pub struct TimingData {
    /// Samples for the baseline/control group.
    pub baseline_samples: Vec<u64>,

    /// Samples for the test/treatment group.
    pub test_samples: Vec<u64>,

    /// Time unit of the samples.
    pub unit: TimeUnit,

    /// Optional metadata about the data source.
    pub metadata: Option<DataMetadata>,
}

impl TimingData {
    /// Create new timing data from two sample vectors.
    pub fn new(baseline: Vec<u64>, test: Vec<u64>, unit: TimeUnit) -> Self {
        Self {
            baseline_samples: baseline,
            test_samples: test,
            unit,
            metadata: None,
        }
    }

    /// Create timing data with metadata.
    pub fn with_metadata(
        baseline: Vec<u64>,
        test: Vec<u64>,
        unit: TimeUnit,
        metadata: DataMetadata,
    ) -> Self {
        Self {
            baseline_samples: baseline,
            test_samples: test,
            unit,
            metadata: Some(metadata),
        }
    }

    /// Get the number of samples in the smaller group.
    pub fn min_samples(&self) -> usize {
        self.baseline_samples.len().min(self.test_samples.len())
    }

    /// Get total number of samples across both groups.
    pub fn total_samples(&self) -> usize {
        self.baseline_samples.len() + self.test_samples.len()
    }

    /// Check if there are enough samples for analysis.
    ///
    /// Returns `Ok(())` if both groups have at least `min_samples`,
    /// or an appropriate `DataError` otherwise.
    pub fn validate(&self, min_samples: usize) -> Result<(), DataError> {
        if self.baseline_samples.len() < min_samples {
            return Err(DataError::InsufficientSamples {
                group: "baseline".to_string(),
                got: self.baseline_samples.len(),
                min: min_samples,
            });
        }
        if self.test_samples.len() < min_samples {
            return Err(DataError::InsufficientSamples {
                group: "test".to_string(),
                got: self.test_samples.len(),
                min: min_samples,
            });
        }
        Ok(())
    }

    /// Convert samples to nanoseconds using the specified conversion factor.
    ///
    /// # Arguments
    /// * `ns_per_unit` - Nanoseconds per sample unit (e.g., 0.33 for cycles at 3GHz)
    ///
    /// # Returns
    /// Tuple of (baseline_ns, test_ns) as f64 vectors.
    pub fn to_nanoseconds(&self, ns_per_unit: f64) -> (Vec<f64>, Vec<f64>) {
        let baseline_ns: Vec<f64> = self
            .baseline_samples
            .iter()
            .map(|&s| s as f64 * ns_per_unit)
            .collect();
        let test_ns: Vec<f64> = self
            .test_samples
            .iter()
            .map(|&s| s as f64 * ns_per_unit)
            .collect();
        (baseline_ns, test_ns)
    }
}

/// Metadata about the data source.
#[derive(Debug, Clone, Default)]
pub struct DataMetadata {
    /// Original filename or identifier.
    pub source: Option<String>,

    /// Labels used for the two groups in the source file.
    pub group_labels: Option<(String, String)>,

    /// Any additional context (e.g., from SILENT summary JSON).
    pub context: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_data_validation() {
        let data = TimingData::new(vec![1, 2, 3], vec![4, 5], TimeUnit::Cycles);

        assert!(data.validate(2).is_ok());
        assert!(data.validate(3).is_err()); // test group only has 2
    }

    #[test]
    fn test_timing_data_to_nanoseconds() {
        let data = TimingData::new(vec![100, 200], vec![150, 250], TimeUnit::Cycles);

        let (baseline_ns, test_ns) = data.to_nanoseconds(0.5);

        assert_eq!(baseline_ns, vec![50.0, 100.0]);
        assert_eq!(test_ns, vec![75.0, 125.0]);
    }
}
