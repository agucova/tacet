//! Checkpoint and incremental CSV writing for benchmark resumability.
//!
//! This module provides:
//! - `WorkItemKey`: Unique identifier for a benchmark work item
//! - `IncrementalCsvWriter`: Thread-safe incremental CSV writer with checkpoint support
//!
//! # Example
//!
//! ```ignore
//! use timing_oracle_bench::checkpoint::IncrementalCsvWriter;
//! use std::path::Path;
//!
//! // Create writer (resume=true loads existing results)
//! let writer = IncrementalCsvWriter::new(Path::new("results.csv"), true)?;
//!
//! // Check if work already done
//! if !writer.is_completed(&key) {
//!     let result = run_benchmark();
//!     writer.write_result(&result)?;
//! }
//! ```

use crate::sweep::BenchmarkResult;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

/// CSV header for benchmark results.
pub const CSV_HEADER: &str = "tool,preset,effect_pattern,effect_sigma_mult,noise_model,dataset_id,samples_per_class,detected,statistic,p_value,time_ms,samples_used";

/// Unique identifier for a benchmark work item.
///
/// Used to track which work items have been completed for resumability.
/// The key uses string representation of f64 to avoid floating-point comparison issues.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkItemKey {
    pub tool: String,
    pub effect_pattern: String,
    /// String representation of effect_sigma_mult (e.g., "0.050000")
    pub effect_sigma_mult_str: String,
    pub noise_model: String,
    pub dataset_id: usize,
}

impl WorkItemKey {
    /// Create a key from a benchmark result.
    pub fn from_result(r: &BenchmarkResult) -> Self {
        Self {
            tool: r.tool.clone(),
            effect_pattern: r.effect_pattern.clone(),
            effect_sigma_mult_str: format!("{:.6}", r.effect_sigma_mult),
            noise_model: r.noise_model.clone(),
            dataset_id: r.dataset_id,
        }
    }

    /// Create a key from individual components.
    pub fn new(
        tool: &str,
        effect_pattern: &str,
        effect_sigma_mult: f64,
        noise_model: &str,
        dataset_id: usize,
    ) -> Self {
        Self {
            tool: tool.to_string(),
            effect_pattern: effect_pattern.to_string(),
            effect_sigma_mult_str: format!("{:.6}", effect_sigma_mult),
            noise_model: noise_model.to_string(),
            dataset_id,
        }
    }
}

/// Thread-safe incremental CSV writer with checkpoint support.
///
/// Writes benchmark results to CSV as they complete, enabling resumability
/// after interruption. Uses mutexes for thread-safe access from parallel
/// rayon tasks.
pub struct IncrementalCsvWriter {
    /// Buffered file writer protected by mutex for thread-safe writes.
    file: Mutex<BufWriter<File>>,
    /// Set of completed work items for resume filtering.
    completed: Mutex<HashSet<WorkItemKey>>,
    /// Number of results loaded from checkpoint (for progress reporting).
    pub resumed_count: usize,
}

impl IncrementalCsvWriter {
    /// Create a new incremental CSV writer.
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    /// * `resume` - If true, load existing results from the file
    ///
    /// # Returns
    /// A new writer instance, or an error if file operations fail.
    ///
    /// # Behavior
    /// - If `resume=false`: Creates a new file (truncating if exists), writes header
    /// - If `resume=true` and file exists: Loads completed items, appends new results
    /// - If `resume=true` and file doesn't exist: Creates new file with header
    pub fn new(path: &Path, resume: bool) -> io::Result<Self> {
        let (file, completed, resumed_count) = if resume && path.exists() {
            // Load existing results
            let completed = Self::load_completed(path)?;
            let resumed_count = completed.len();

            // Open for appending
            let file = OpenOptions::new().append(true).open(path)?;

            (file, completed, resumed_count)
        } else {
            // Create new file with header
            let mut file = File::create(path)?;
            writeln!(file, "{}", CSV_HEADER)?;
            file.flush()?;

            (file, HashSet::new(), 0)
        };

        Ok(Self {
            file: Mutex::new(BufWriter::new(file)),
            completed: Mutex::new(completed),
            resumed_count,
        })
    }

    /// Check if a work item has already been completed.
    ///
    /// Used to filter work items when resuming an interrupted benchmark.
    pub fn is_completed(&self, key: &WorkItemKey) -> bool {
        let completed = self.completed.lock().unwrap();
        completed.contains(key)
    }

    /// Get the number of completed work items.
    pub fn completed_count(&self) -> usize {
        let completed = self.completed.lock().unwrap();
        completed.len()
    }

    /// Write a single benchmark result to the CSV file.
    ///
    /// Thread-safe: can be called from multiple rayon tasks concurrently.
    /// Flushes after each write to ensure data reaches disk.
    pub fn write_result(&self, result: &BenchmarkResult) -> io::Result<()> {
        let key = WorkItemKey::from_result(result);

        // Write to file
        {
            let mut file = self.file.lock().unwrap();
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{},{},{}",
                result.tool,
                result.preset,
                result.effect_pattern,
                result.effect_sigma_mult,
                result.noise_model,
                result.dataset_id,
                result.samples_per_class,
                result.detected,
                result
                    .statistic
                    .map(|s| format!("{:.6}", s))
                    .unwrap_or_default(),
                result
                    .p_value
                    .map(|p| format!("{:.6}", p))
                    .unwrap_or_default(),
                result.time_ms,
                result
                    .samples_used
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
            )?;
            file.flush()?;
        }

        // Update completed set
        {
            let mut completed = self.completed.lock().unwrap();
            completed.insert(key);
        }

        Ok(())
    }

    /// Load completed work items from an existing CSV file.
    fn load_completed(path: &Path) -> io::Result<HashSet<WorkItemKey>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut completed = HashSet::new();

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = match line_result {
                Ok(l) => l,
                Err(e) => {
                    eprintln!(
                        "Warning: Skipping malformed line {} in checkpoint: {}",
                        line_num + 1,
                        e
                    );
                    continue;
                }
            };

            // Skip header
            if line_num == 0 && line.starts_with("tool,") {
                continue;
            }

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            if let Some(key) = Self::parse_csv_row(&line) {
                completed.insert(key);
            } else {
                eprintln!(
                    "Warning: Could not parse line {} in checkpoint: {}",
                    line_num + 1,
                    &line[..line.len().min(80)]
                );
            }
        }

        Ok(completed)
    }

    /// Parse a CSV row into a WorkItemKey.
    ///
    /// Returns None if the row cannot be parsed (malformed or incomplete).
    fn parse_csv_row(line: &str) -> Option<WorkItemKey> {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 6 {
            // Normalize effect_sigma_mult to consistent format
            let effect_mult: f64 = parts[3].parse().ok()?;
            Some(WorkItemKey {
                tool: parts[0].to_string(),
                effect_pattern: parts[2].to_string(),
                effect_sigma_mult_str: format!("{:.6}", effect_mult),
                noise_model: parts[4].to_string(),
                dataset_id: parts[5].parse().ok()?,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn mock_result(tool: &str, dataset_id: usize, effect_mult: f64) -> BenchmarkResult {
        BenchmarkResult {
            tool: tool.to_string(),
            preset: "quick".to_string(),
            effect_pattern: "shift".to_string(),
            effect_sigma_mult: effect_mult,
            noise_model: "iid".to_string(),
            dataset_id,
            samples_per_class: 5000,
            detected: false,
            statistic: Some(1.5),
            p_value: Some(0.15),
            time_ms: 100,
            samples_used: Some(5000),
        }
    }

    #[test]
    fn test_work_item_key_equality() {
        let key1 = WorkItemKey::new("tool-a", "shift", 0.05, "iid", 0);
        let key2 = WorkItemKey::new("tool-a", "shift", 0.05, "iid", 0);
        let key3 = WorkItemKey::new("tool-a", "shift", 0.05, "iid", 1);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_work_item_key_from_result() {
        let result = mock_result("test-tool", 5, 0.123456);
        let key = WorkItemKey::from_result(&result);

        assert_eq!(key.tool, "test-tool");
        assert_eq!(key.effect_pattern, "shift");
        assert_eq!(key.effect_sigma_mult_str, "0.123456");
        assert_eq!(key.noise_model, "iid");
        assert_eq!(key.dataset_id, 5);
    }

    #[test]
    fn test_incremental_write_new_file() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.csv");

        let writer = IncrementalCsvWriter::new(&path, false).unwrap();
        assert_eq!(writer.completed_count(), 0);
        assert_eq!(writer.resumed_count, 0);

        let result = mock_result("test-tool", 0, 0.05);
        writer.write_result(&result).unwrap();

        assert_eq!(writer.completed_count(), 1);
        assert!(writer.is_completed(&WorkItemKey::from_result(&result)));

        // Verify file content
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("tool,preset,effect_pattern"));
        assert!(content.contains("test-tool"));
    }

    #[test]
    fn test_resume_from_existing() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.csv");

        // Write some results
        {
            let writer = IncrementalCsvWriter::new(&path, false).unwrap();
            writer.write_result(&mock_result("tool-a", 0, 0.0)).unwrap();
            writer.write_result(&mock_result("tool-a", 1, 0.0)).unwrap();
            writer
                .write_result(&mock_result("tool-b", 0, 0.05))
                .unwrap();
        }

        // Resume
        let writer = IncrementalCsvWriter::new(&path, true).unwrap();
        assert_eq!(writer.resumed_count, 3);
        assert_eq!(writer.completed_count(), 3);

        // Check existing items are marked completed
        assert!(writer.is_completed(&WorkItemKey::new("tool-a", "shift", 0.0, "iid", 0)));
        assert!(writer.is_completed(&WorkItemKey::new("tool-a", "shift", 0.0, "iid", 1)));
        assert!(writer.is_completed(&WorkItemKey::new("tool-b", "shift", 0.05, "iid", 0)));

        // Check non-existing item
        assert!(!writer.is_completed(&WorkItemKey::new("tool-c", "shift", 0.0, "iid", 0)));

        // Write new result
        writer.write_result(&mock_result("tool-c", 0, 0.0)).unwrap();
        assert_eq!(writer.completed_count(), 4);
    }

    #[test]
    fn test_resume_nonexistent_file() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("nonexistent.csv");

        // Resume on non-existent file should create it
        let writer = IncrementalCsvWriter::new(&path, true).unwrap();
        assert_eq!(writer.resumed_count, 0);
        assert_eq!(writer.completed_count(), 0);
        assert!(path.exists());
    }

    #[test]
    fn test_parallel_writes() {
        use std::sync::Arc;

        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("parallel.csv");
        let writer = Arc::new(IncrementalCsvWriter::new(&path, false).unwrap());

        // Write from multiple threads
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let writer = Arc::clone(&writer);
                std::thread::spawn(move || {
                    for j in 0..10 {
                        let result = mock_result("parallel-tool", i * 10 + j, 0.0);
                        writer.write_result(&result).unwrap();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(writer.completed_count(), 100);

        // Verify all were written by loading again
        let writer2 = IncrementalCsvWriter::new(&path, true).unwrap();
        assert_eq!(writer2.resumed_count, 100);
    }

    #[test]
    fn test_malformed_line_handling() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("malformed.csv");

        // Write file with some malformed lines
        std::fs::write(
            &path,
            "tool,preset,effect_pattern,effect_sigma_mult,noise_model,dataset_id,samples_per_class,detected,statistic,p_value,time_ms,samples_used\n\
             tool-a,quick,shift,0.0,iid,0,5000,false,1.5,0.15,100,5000\n\
             malformed line\n\
             tool-b,quick,shift,0.05,iid,1,5000,true,2.0,0.01,150,5000\n\
             \n\
             tool-c,quick\n"
        ).unwrap();

        // Should load 2 valid entries, skip malformed
        let writer = IncrementalCsvWriter::new(&path, true).unwrap();
        assert_eq!(writer.resumed_count, 2);
    }
}
