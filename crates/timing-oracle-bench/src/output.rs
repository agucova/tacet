//! Output formatters for benchmark results.
//!
//! This module provides:
//! - CSV export for raw results
//! - Markdown report generation
//! - Summary tables for FPR and power analysis

use crate::sweep::{PointSummary, SweepResults};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Write benchmark results to CSV.
pub fn write_csv(results: &SweepResults, path: &Path) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Header
    writeln!(
        writer,
        "tool,preset,effect_pattern,effect_sigma_mult,noise_model,dataset_id,samples_per_class,detected,statistic,p_value,time_ms,samples_used"
    )?;

    // Data rows
    for r in &results.results {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            r.tool,
            r.preset,
            r.effect_pattern,
            r.effect_sigma_mult,
            r.noise_model,
            r.dataset_id,
            r.samples_per_class,
            r.detected,
            r.statistic.map(|s| format!("{:.6}", s)).unwrap_or_default(),
            r.p_value.map(|p| format!("{:.6}", p)).unwrap_or_default(),
            r.time_ms,
            r.samples_used.map(|s| s.to_string()).unwrap_or_default(),
        )?;
    }

    writer.flush()
}

/// Write summary statistics to CSV.
pub fn write_summary_csv(results: &SweepResults, path: &Path) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Header
    writeln!(
        writer,
        "tool,effect_pattern,effect_sigma_mult,noise_model,n_datasets,detection_rate,ci_low,ci_high,median_time_ms,median_samples"
    )?;

    // Data rows
    for s in results.summarize() {
        writeln!(
            writer,
            "{},{},{},{},{},{:.4},{:.4},{:.4},{},{}",
            s.tool,
            s.effect_pattern,
            s.effect_sigma_mult,
            s.noise_model,
            s.n_datasets,
            s.detection_rate,
            s.ci_low,
            s.ci_high,
            s.median_time_ms,
            s.median_samples.map(|s| s.to_string()).unwrap_or_default(),
        )?;
    }

    writer.flush()
}

/// Generate markdown report from results.
pub fn to_markdown(results: &SweepResults) -> String {
    let mut md = String::new();

    // Title
    md.push_str(&format!(
        "# Benchmark Results ({})\n\n",
        results.config.preset.name()
    ));

    // Summary
    md.push_str("## Configuration\n\n");
    md.push_str(&format!(
        "- **Preset**: {}\n",
        results.config.preset.name()
    ));
    md.push_str(&format!(
        "- **Samples per class**: {}\n",
        results.config.samples_per_class
    ));
    md.push_str(&format!(
        "- **Datasets per point**: {}\n",
        results.config.datasets_per_point
    ));
    md.push_str(&format!(
        "- **Effect multipliers**: {:?}\n",
        results.config.effect_multipliers
    ));
    md.push_str(&format!(
        "- **Total execution time**: {:.1}s\n\n",
        results.total_time.as_secs_f64()
    ));

    // FPR Table (effect = 0)
    md.push_str("## False Positive Rate (effect = 0)\n\n");
    md.push_str(&fpr_table(results));
    md.push('\n');

    // Power Tables (effect > 0)
    md.push_str("## Power (Detection Rate)\n\n");
    md.push_str(&power_table(results));
    md.push('\n');

    // Per-tool detailed results
    md.push_str("## Detailed Results by Tool\n\n");
    for tool in results.tools() {
        md.push_str(&format!("### {}\n\n", tool));
        md.push_str(&tool_table(results, &tool));
        md.push('\n');
    }

    md
}

/// Generate FPR table (effect_sigma_mult = 0).
fn fpr_table(results: &SweepResults) -> String {
    let summaries = results.summarize();

    // Get unique tools and noise models (with realistic suffix if applicable)
    let tools = results.tools();
    let noise_models: Vec<String> = results
        .config
        .noise_models
        .iter()
        .map(|n| {
            if results.config.use_realistic {
                format!("{}-realistic", n.name())
            } else {
                n.name()
            }
        })
        .collect();

    // Filter to null effect
    let fpr_summaries: Vec<&PointSummary> = summaries
        .iter()
        .filter(|s| s.effect_sigma_mult == 0.0 && s.effect_pattern == "null")
        .collect();

    let mut table = String::new();

    // Header
    table.push_str("| Tool |");
    for noise in &noise_models {
        table.push_str(&format!(" {} |", noise));
    }
    table.push_str("\n|------|");
    for _ in &noise_models {
        table.push_str("--------|");
    }
    table.push('\n');

    // Rows
    for tool in &tools {
        table.push_str(&format!("| {} |", tool));
        for noise in &noise_models {
            if let Some(s) = fpr_summaries
                .iter()
                .find(|s| s.tool == *tool && s.noise_model == *noise)
            {
                table.push_str(&format!(
                    " {:.1}% ± {:.1}% |",
                    s.detection_rate * 100.0,
                    (s.ci_high - s.ci_low) / 2.0 * 100.0
                ));
            } else {
                table.push_str(" - |");
            }
        }
        table.push('\n');
    }

    table
}

/// Generate power table for shift pattern.
fn power_table(results: &SweepResults) -> String {
    let summaries = results.summarize();

    // Get unique tools and effect multipliers (excluding 0)
    let tools = results.tools();
    let effect_mults: Vec<f64> = results
        .config
        .effect_multipliers
        .iter()
        .filter(|&&m| m > 0.0)
        .copied()
        .collect();

    // Determine noise model name (with realistic suffix if applicable)
    let iid_noise = if results.config.use_realistic {
        "iid-realistic".to_string()
    } else {
        "iid".to_string()
    };

    // Filter to shift pattern with IID noise (or IID-realistic)
    let power_summaries: Vec<&PointSummary> = summaries
        .iter()
        .filter(|s| s.effect_pattern == "shift" && s.noise_model == iid_noise)
        .collect();

    let mut table = String::new();

    // Header
    table.push_str("| Tool |");
    for mult in &effect_mults {
        table.push_str(&format!(" {}σ |", mult));
    }
    table.push_str("\n|------|");
    for _ in &effect_mults {
        table.push_str("------|");
    }
    table.push('\n');

    // Rows
    for tool in &tools {
        table.push_str(&format!("| {} |", tool));
        for mult in &effect_mults {
            if let Some(s) = power_summaries
                .iter()
                .find(|s| s.tool == *tool && (s.effect_sigma_mult - mult).abs() < 0.001)
            {
                table.push_str(&format!(" {:.0}% |", s.detection_rate * 100.0));
            } else {
                table.push_str(" - |");
            }
        }
        table.push('\n');
    }

    table
}

/// Generate detailed table for a specific tool.
fn tool_table(results: &SweepResults, tool: &str) -> String {
    let summaries: Vec<PointSummary> = results
        .summarize()
        .into_iter()
        .filter(|s| s.tool == tool)
        .collect();

    if summaries.is_empty() {
        return "No results\n".to_string();
    }

    let mut table = String::new();

    // Header
    table.push_str("| Pattern | Effect | Noise | Rate | 95% CI | Time (ms) |\n");
    table.push_str("|---------|--------|-------|------|--------|----------|\n");

    // Sort by pattern, effect, noise
    let mut sorted = summaries;
    sorted.sort_by(|a, b| {
        a.effect_pattern
            .cmp(&b.effect_pattern)
            .then(a.effect_sigma_mult.partial_cmp(&b.effect_sigma_mult).unwrap())
            .then(a.noise_model.cmp(&b.noise_model))
    });

    for s in sorted {
        table.push_str(&format!(
            "| {} | {}σ | {} | {:.1}% | [{:.1}%, {:.1}%] | {} |\n",
            s.effect_pattern,
            s.effect_sigma_mult,
            s.noise_model,
            s.detection_rate * 100.0,
            s.ci_low * 100.0,
            s.ci_high * 100.0,
            s.median_time_ms,
        ));
    }

    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sweep::{BenchmarkResult, SweepConfig};
    use std::time::Duration;

    fn mock_results() -> SweepResults {
        let config = SweepConfig::quick();
        let mut results = SweepResults::new(config);
        results.total_time = Duration::from_secs(10);

        // Add some mock results
        results.push(BenchmarkResult {
            tool: "test-tool".to_string(),
            preset: "quick".to_string(),
            effect_pattern: "null".to_string(),
            effect_sigma_mult: 0.0,
            noise_model: "iid".to_string(),
            dataset_id: 0,
            samples_per_class: 5000,
            detected: false,
            statistic: Some(1.5),
            p_value: Some(0.15),
            time_ms: 100,
            samples_used: Some(5000),
        });

        results.push(BenchmarkResult {
            tool: "test-tool".to_string(),
            preset: "quick".to_string(),
            effect_pattern: "shift".to_string(),
            effect_sigma_mult: 1.0,
            noise_model: "iid".to_string(),
            dataset_id: 0,
            samples_per_class: 5000,
            detected: true,
            statistic: Some(5.5),
            p_value: Some(0.001),
            time_ms: 120,
            samples_used: Some(5000),
        });

        results
    }

    #[test]
    fn test_csv_output() {
        let results = mock_results();
        let temp_dir = tempfile::TempDir::new().unwrap();
        let path = temp_dir.path().join("test.csv");

        write_csv(&results, &path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("tool,preset,effect_pattern"));
        assert!(content.contains("test-tool"));
        assert!(content.contains("shift"));
    }

    #[test]
    fn test_markdown_output() {
        let results = mock_results();
        let md = to_markdown(&results);

        assert!(md.contains("# Benchmark Results"));
        assert!(md.contains("False Positive Rate"));
        assert!(md.contains("Power"));
    }
}
