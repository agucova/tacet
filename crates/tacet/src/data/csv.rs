//! CSV file parsing for timing data.
//!
//! Supports multiple CSV formats commonly used in timing side-channel analysis.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::{DataError, DataMetadata, TimeUnit, TimingData};

/// Load timing data from a SILENT-format CSV file.
///
/// SILENT format has:
/// - Header: `V1,V2`
/// - Column 1: Group label (`X` for baseline, `Y` for test)
/// - Column 2: Timing value (integer)
///
/// # Example file content
/// ```csv
/// V1,V2
/// X,1067424
/// Y,531296
/// X,1051010
/// Y,530842
/// ```
///
/// # Arguments
/// * `path` - Path to the CSV file
///
/// # Returns
/// `TimingData` with X samples as baseline, Y samples as test.
///
/// # Errors
/// Returns `DataError` if the file cannot be read, parsed, or lacks required groups.
pub fn load_silent_csv(path: &Path) -> Result<TimingData, DataError> {
    load_two_column_csv(path, true, "X", "Y")
}

/// Load timing data from a generic two-column CSV file.
///
/// # Arguments
/// * `path` - Path to the CSV file
/// * `has_header` - Whether the first line is a header (skip it)
/// * `baseline_label` - Label for the baseline group (e.g., "X", "control", "0")
/// * `test_label` - Label for the test group (e.g., "Y", "treatment", "1")
///
/// # Returns
/// `TimingData` with samples split by group label.
pub fn load_two_column_csv(
    path: &Path,
    has_header: bool,
    baseline_label: &str,
    test_label: &str,
) -> Result<TimingData, DataError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut baseline_samples = Vec::new();
    let mut test_samples = Vec::new();
    let mut found_labels: HashMap<String, usize> = HashMap::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Skip header if specified
        if has_header && line_num == 0 {
            continue;
        }

        // Parse line
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            return Err(DataError::Parse {
                line: line_num + 1,
                message: format!("Expected 2 columns, got {}", parts.len()),
            });
        }

        let label = parts[0].trim();
        let value_str = parts[1].trim();

        // Track found labels for error reporting
        *found_labels.entry(label.to_string()).or_insert(0) += 1;

        // Parse the timing value (handle both integer and float formats)
        let value: u64 = if value_str.contains('.') {
            // Parse as float first, then convert to u64
            let float_val: f64 = value_str.parse().map_err(|_| DataError::InvalidValue {
                line: line_num + 1,
                value: value_str.to_string(),
            })?;
            float_val as u64
        } else {
            value_str.parse().map_err(|_| DataError::InvalidValue {
                line: line_num + 1,
                value: value_str.to_string(),
            })?
        };

        // Assign to appropriate group
        if label == baseline_label {
            baseline_samples.push(value);
        } else if label == test_label {
            test_samples.push(value);
        }
        // Ignore other labels (if any)
    }

    // Validate that we found both groups
    if baseline_samples.is_empty() {
        return Err(DataError::MissingGroup {
            expected: baseline_label.to_string(),
            found: found_labels.keys().cloned().collect(),
        });
    }
    if test_samples.is_empty() {
        return Err(DataError::MissingGroup {
            expected: test_label.to_string(),
            found: found_labels.keys().cloned().collect(),
        });
    }

    let metadata = DataMetadata {
        source: path.to_string_lossy().to_string().into(),
        group_labels: Some((baseline_label.to_string(), test_label.to_string())),
        context: None,
    };

    Ok(TimingData::with_metadata(
        baseline_samples,
        test_samples,
        TimeUnit::Cycles, // Assume cycles by default
        metadata,
    ))
}

/// Load timing data from two separate files (one per group).
///
/// Each file should contain one timing value per line.
///
/// # Arguments
/// * `baseline_path` - Path to file containing baseline samples
/// * `test_path` - Path to file containing test samples
/// * `unit` - Time unit of the values
///
/// # Returns
/// `TimingData` with samples from each file.
pub fn load_separate_files(
    baseline_path: &Path,
    test_path: &Path,
    unit: TimeUnit,
) -> Result<TimingData, DataError> {
    let baseline_samples = load_single_column_file(baseline_path)?;
    let test_samples = load_single_column_file(test_path)?;

    let metadata = DataMetadata {
        source: Some(format!(
            "{} + {}",
            baseline_path.display(),
            test_path.display()
        )),
        group_labels: None,
        context: None,
    };

    Ok(TimingData::with_metadata(
        baseline_samples,
        test_samples,
        unit,
        metadata,
    ))
}

/// Load a single-column file of timing values.
fn load_single_column_file(path: &Path) -> Result<Vec<u64>, DataError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut samples = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Skip comments
        if line.starts_with('#') {
            continue;
        }

        // Parse value
        let value: u64 = line.parse().map_err(|_| DataError::InvalidValue {
            line: line_num + 1,
            value: line.to_string(),
        })?;

        samples.push(value);
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_silent_csv() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "V1,V2").unwrap();
        writeln!(file, "X,1000").unwrap();
        writeln!(file, "Y,2000").unwrap();
        writeln!(file, "X,1100").unwrap();
        writeln!(file, "Y,2100").unwrap();
        writeln!(file, "X,1200").unwrap();
        file.flush().unwrap();

        let data = load_silent_csv(file.path()).unwrap();

        assert_eq!(data.baseline_samples, vec![1000, 1100, 1200]);
        assert_eq!(data.test_samples, vec![2000, 2100]);
    }

    #[test]
    fn test_load_silent_csv_missing_group() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "V1,V2").unwrap();
        writeln!(file, "X,1000").unwrap();
        writeln!(file, "X,1100").unwrap();
        file.flush().unwrap();

        let result = load_silent_csv(file.path());
        assert!(result.is_err());

        if let Err(DataError::MissingGroup { expected, .. }) = result {
            assert_eq!(expected, "Y");
        } else {
            panic!("Expected MissingGroup error");
        }
    }

    #[test]
    fn test_load_two_column_csv_custom_labels() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "group,time").unwrap();
        writeln!(file, "control,500").unwrap();
        writeln!(file, "treatment,600").unwrap();
        writeln!(file, "control,510").unwrap();
        writeln!(file, "treatment,610").unwrap();
        file.flush().unwrap();

        let data = load_two_column_csv(file.path(), true, "control", "treatment").unwrap();

        assert_eq!(data.baseline_samples, vec![500, 510]);
        assert_eq!(data.test_samples, vec![600, 610]);
    }

    #[test]
    fn test_invalid_value() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "V1,V2").unwrap();
        writeln!(file, "X,1000").unwrap();
        writeln!(file, "Y,not_a_number").unwrap();
        file.flush().unwrap();

        let result = load_silent_csv(file.path());
        assert!(result.is_err());

        if let Err(DataError::InvalidValue { line, value }) = result {
            assert_eq!(line, 3);
            assert_eq!(value, "not_a_number");
        } else {
            panic!("Expected InvalidValue error");
        }
    }
}
