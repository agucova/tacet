//! Vertical histogram rendering for timing distributions.
//!
//! This module provides compact terminal-based visualization of timing distributions,
//! showing baseline vs sample side-by-side using Unicode block characters.

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write;

/// Unicode block characters for vertical bars (increasing height).
const BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Configuration for histogram rendering.
#[derive(Debug, Clone)]
pub struct HistogramConfig {
    /// Number of bins to use.
    pub bins: usize,
    /// Height of the histogram in character rows.
    pub height: usize,
    /// Width of each column in characters.
    pub column_width: usize,
    /// Whether to show percentile markers.
    pub show_percentiles: bool,
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self {
            bins: 20,
            height: 10,
            column_width: 3,
            show_percentiles: true,
        }
    }
}

/// Render a side-by-side histogram comparing baseline and sample distributions.
///
/// # Arguments
///
/// * `baseline` - Timing samples for baseline class (in nanoseconds)
/// * `sample` - Timing samples for sample class (in nanoseconds)
/// * `config` - Histogram configuration
///
/// # Returns
///
/// A formatted string containing the vertical histogram visualization.
///
/// # Example output
///
/// ```text
/// Baseline       │ Sample          W₁ = 12.3ns
///       █        │
///       █        │   █
///     █ █        │   █
///     █ █        │ █ █
///   █ █ █        │ █ █ █
///   █ █ █ █      │ █ █ █ █
/// ──────────────┼──────────────
/// p50      p90  │ p50      p90
/// 123ns   156ns │ 135ns   178ns
/// ```
pub fn render_histogram(baseline: &[f64], sample: &[f64], config: &HistogramConfig) -> String {
    render_histogram_with_w1(baseline, sample, config, None)
}

/// Render a side-by-side histogram with optional W₁ distance display.
///
/// # Arguments
///
/// * `baseline` - Timing samples for baseline class (in nanoseconds)
/// * `sample` - Timing samples for sample class (in nanoseconds)
/// * `config` - Histogram configuration
/// * `w1_distance` - Optional Wasserstein-1 distance to display in header
pub fn render_histogram_with_w1(
    baseline: &[f64],
    sample: &[f64],
    config: &HistogramConfig,
    w1_distance: Option<f64>,
) -> String {
    if baseline.is_empty() || sample.is_empty() {
        return String::from("(empty distributions)");
    }

    // Compute shared bin edges across both distributions
    let (min_val, max_val) = compute_range(baseline, sample);
    if max_val <= min_val {
        return String::from("(degenerate distributions)");
    }

    let bin_edges = compute_bin_edges(min_val, max_val, config.bins);

    // Build histograms
    let baseline_hist = build_histogram(baseline, &bin_edges);
    let sample_hist = build_histogram(sample, &bin_edges);

    // Normalize to height
    let baseline_norm = normalize_histogram(&baseline_hist, config.height);
    let sample_norm = normalize_histogram(&sample_hist, config.height);

    // Render
    let mut out = String::new();

    // Header
    let left_width = config.bins * config.column_width;
    if let Some(w1) = w1_distance {
        writeln!(
            out,
            "{:width$} │ Sample          W₁ = {:.1}ns",
            "Baseline",
            w1,
            width = left_width
        )
        .unwrap();
    } else {
        writeln!(out, "{:width$} │ Sample", "Baseline", width = left_width).unwrap();
    }

    // Bars (top to bottom)
    for row in (0..config.height).rev() {
        // Baseline side
        for &count in &baseline_norm {
            let block = if count > row {
                BLOCKS[8] // Full block
            } else if count == row {
                // Partial block based on fractional part
                BLOCKS[4] // Medium block for simplicity
            } else {
                BLOCKS[0] // Empty
            };
            for _ in 0..config.column_width {
                write!(out, "{}", block).unwrap();
            }
        }

        write!(out, " │ ").unwrap();

        // Sample side
        for &count in &sample_norm {
            let block = if count > row {
                BLOCKS[8]
            } else if count == row {
                BLOCKS[4]
            } else {
                BLOCKS[0]
            };
            for _ in 0..config.column_width {
                write!(out, "{}", block).unwrap();
            }
        }

        writeln!(out).unwrap();
    }

    // Separator
    let sep_left = "─".repeat(left_width);
    let sep_right = "─".repeat(config.bins * config.column_width);
    writeln!(out, "{}┼{}", sep_left, sep_right).unwrap();

    // Percentile markers (if enabled)
    if config.show_percentiles {
        let baseline_p50 = percentile(baseline, 0.5);
        let baseline_p90 = percentile(baseline, 0.9);
        let sample_p50 = percentile(sample, 0.5);
        let sample_p90 = percentile(sample, 0.9);

        writeln!(
            out,
            "{:width$} │ p50      p90",
            "p50      p90",
            width = left_width
        )
        .unwrap();

        writeln!(
            out,
            "{:.1}ns  {:.1}ns{:width$} │ {:.1}ns  {:.1}ns",
            baseline_p50,
            baseline_p90,
            "",
            sample_p50,
            sample_p90,
            width = left_width.saturating_sub(20)
        )
        .unwrap();
    }

    out
}

/// Compute the overall range across both distributions.
fn compute_range(baseline: &[f64], sample: &[f64]) -> (f64, f64) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for &val in baseline {
        if val.is_finite() {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    for &val in sample {
        if val.is_finite() {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    (min_val, max_val)
}

/// Generate bin edges for the histogram.
fn compute_bin_edges(min_val: f64, max_val: f64, bins: usize) -> Vec<f64> {
    let mut edges = Vec::with_capacity(bins + 1);
    let step = (max_val - min_val) / bins as f64;

    for i in 0..=bins {
        edges.push(min_val + i as f64 * step);
    }

    edges
}

/// Build histogram counts from samples.
fn build_histogram(samples: &[f64], bin_edges: &[f64]) -> Vec<usize> {
    let mut counts = vec![0; bin_edges.len().saturating_sub(1)];

    for &val in samples {
        if !val.is_finite() {
            continue;
        }

        // Find bin index
        for i in 0..counts.len() {
            if val >= bin_edges[i] && val < bin_edges[i + 1] {
                counts[i] += 1;
                break;
            } else if i == counts.len() - 1 && val >= bin_edges[i] && val <= bin_edges[i + 1] {
                // Include right edge in last bin
                counts[i] += 1;
                break;
            }
        }
    }

    counts
}

/// Normalize histogram counts to a given height.
fn normalize_histogram(counts: &[usize], height: usize) -> Vec<usize> {
    if counts.is_empty() {
        return Vec::new();
    }

    let max_count = *counts.iter().max().unwrap();
    if max_count == 0 {
        return vec![0; counts.len()];
    }

    counts.iter().map(|&c| (c * height) / max_count).collect()
}

/// Compute percentile of a distribution.
fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    if sorted.is_empty() {
        return 0.0;
    }

    let idx = ((sorted.len() - 1) as f64 * p) as usize;
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_histogram_basic() {
        let baseline = vec![100.0, 105.0, 110.0, 115.0, 120.0];
        let sample = vec![110.0, 115.0, 120.0, 125.0, 130.0];

        let config = HistogramConfig {
            bins: 10,
            height: 5,
            column_width: 2,
            show_percentiles: true,
        };

        let output = render_histogram(&baseline, &sample, &config);
        assert!(output.contains("Baseline"));
        assert!(output.contains("Sample"));
        assert!(output.contains("p50"));
        assert!(output.contains("p90"));
    }

    #[test]
    fn test_render_histogram_empty() {
        let baseline: Vec<f64> = vec![];
        let sample = vec![100.0];

        let config = HistogramConfig::default();
        let output = render_histogram(&baseline, &sample, &config);
        assert!(output.contains("empty"));
    }

    #[test]
    fn test_compute_range() {
        let baseline = vec![10.0, 20.0, 30.0];
        let sample = vec![15.0, 25.0, 35.0];

        let (min, max) = compute_range(&baseline, &sample);
        assert_eq!(min, 10.0);
        assert_eq!(max, 35.0);
    }

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let p50 = percentile(&values, 0.5);
        // For 10 values, p50 is at index (10-1)*0.5 = 4.5, which rounds to index 4 (value 5.0)
        assert!((p50 - 5.0).abs() < 0.1);

        let p90 = percentile(&values, 0.9);
        // For 10 values, p90 is at index (10-1)*0.9 = 8.1, which rounds to index 8 (value 9.0)
        assert!((p90 - 9.0).abs() < 0.1);
    }

    #[test]
    fn test_build_histogram() {
        let samples = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let bin_edges = vec![0.0, 1.0, 2.0, 3.0, 4.0];

        let hist = build_histogram(&samples, &bin_edges);
        // With 5 bin_edges, there are 4 bins:
        // Bin 0: [0.0, 1.0) - empty
        // Bin 1: [1.0, 2.0) - has 1.0, 1.5
        // Bin 2: [2.0, 3.0) - has 2.0, 2.5
        // Bin 3: [3.0, 4.0] - has 3.0
        assert_eq!(hist.len(), 4);
        assert_eq!(hist, vec![0, 2, 2, 1]);
    }

    #[test]
    fn test_normalize_histogram() {
        let counts = vec![1, 2, 4, 2, 1];
        let normalized = normalize_histogram(&counts, 10);

        assert_eq!(normalized[2], 10); // Max count should map to height
        assert!(normalized[0] < normalized[2]); // Smaller counts should be proportionally lower
    }
}
