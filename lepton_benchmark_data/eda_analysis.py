#!/usr/bin/env python3
"""
Exploratory Data Analysis for timing-oracle fine-threshold benchmark results.
Compares timing-oracle against DudeCT, TVLA, KS-test, AD-test, MONA, RTLF, and SILENT.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette for tools
TOOL_COLORS = {
    'timing-oracle': '#2ecc71',  # Green
    'dudect': '#3498db',         # Blue
    'timing-tvla': '#9b59b6',    # Purple
    'ks-test': '#e74c3c',        # Red
    'ad-test': '#f39c12',        # Orange
    'mona': '#1abc9c',           # Teal
    'rtlf-native': '#e91e63',    # Pink
    'silent-native': '#607d8b',  # Gray-blue
}

TOOL_MARKERS = {
    'timing-oracle': 'o',
    'dudect': 's',
    'timing-tvla': '^',
    'ks-test': 'D',
    'ad-test': 'v',
    'mona': 'P',
    'rtlf-native': 'X',
    'silent-native': '*',
}

def load_data():
    """Load benchmark results and summary CSVs."""
    results = pd.read_csv('benchmark_results.csv')
    summary = pd.read_csv('benchmark_summary.csv')
    return results, summary

def compute_fpr_by_tool(results):
    """Compute False Positive Rate (detection rate when effect=0) for each tool."""
    null_effect = results[results['effect_sigma_mult'] == 0].copy()

    fpr_data = null_effect.groupby(['tool', 'noise_model']).agg({
        'detected': ['mean', 'count', 'std']
    }).reset_index()
    fpr_data.columns = ['tool', 'noise_model', 'fpr', 'n_trials', 'std']

    # Wilson confidence interval
    z = 1.96
    fpr_data['ci_low'] = fpr_data.apply(
        lambda r: max(0, (r['fpr'] + z**2/(2*r['n_trials']) - z*np.sqrt((r['fpr']*(1-r['fpr']) + z**2/(4*r['n_trials']))/r['n_trials'])) / (1 + z**2/r['n_trials'])),
        axis=1
    )
    fpr_data['ci_high'] = fpr_data.apply(
        lambda r: min(1, (r['fpr'] + z**2/(2*r['n_trials']) + z*np.sqrt((r['fpr']*(1-r['fpr']) + z**2/(4*r['n_trials']))/r['n_trials'])) / (1 + z**2/r['n_trials'])),
        axis=1
    )

    return fpr_data

def compute_power_curves(summary):
    """Extract power curves from summary data."""
    # Filter to shift pattern and aggregate across noise models
    shift_data = summary[summary['effect_pattern'] == 'shift'].copy()

    power_curves = shift_data.groupby(['tool', 'effect_sigma_mult']).agg({
        'detection_rate': 'mean',
        'ci_low': 'mean',
        'ci_high': 'mean',
        'n_datasets': 'sum',
        'median_time_ms': 'median'
    }).reset_index()

    return power_curves

def compute_execution_time(results):
    """Compute execution time statistics by tool."""
    time_data = results.groupby('tool').agg({
        'time_ms': ['mean', 'median', 'std', 'min', 'max']
    }).reset_index()
    time_data.columns = ['tool', 'mean_ms', 'median_ms', 'std_ms', 'min_ms', 'max_ms']
    return time_data

def plot_fpr_comparison(fpr_data, output_dir):
    """Plot False Positive Rate comparison across tools and noise models."""
    fig, ax = plt.subplots(figsize=(14, 8))

    tools = sorted(fpr_data['tool'].unique())
    noise_models = sorted(fpr_data['noise_model'].unique())

    x = np.arange(len(noise_models))
    width = 0.1
    multiplier = 0

    for tool in tools:
        tool_data = fpr_data[fpr_data['tool'] == tool].set_index('noise_model')
        fprs = [tool_data.loc[nm, 'fpr'] if nm in tool_data.index else 0 for nm in noise_models]
        ci_lows = [tool_data.loc[nm, 'ci_low'] if nm in tool_data.index else 0 for nm in noise_models]
        ci_highs = [tool_data.loc[nm, 'ci_high'] if nm in tool_data.index else 0 for nm in noise_models]

        errors = [[f - l for f, l in zip(fprs, ci_lows)],
                  [h - f for f, h in zip(fprs, ci_highs)]]

        offset = width * multiplier
        bars = ax.bar(x + offset, fprs, width, label=tool, color=TOOL_COLORS.get(tool, '#888'),
                     yerr=errors, capsize=2, error_kw={'linewidth': 0.8})
        multiplier += 1

    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='α = 0.05 target')
    ax.set_xlabel('Noise Model')
    ax.set_ylabel('False Positive Rate')
    ax.set_title('False Positive Rate Comparison (effect = 0)\nLower is better, target ≤ 0.05')
    ax.set_xticks(x + width * (len(tools) - 1) / 2)
    ax.set_xticklabels(noise_models, rotation=45, ha='right')
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'fpr_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    return fig

def plot_power_curves(power_curves, output_dir):
    """Plot statistical power curves for each tool."""
    fig, ax = plt.subplots(figsize=(12, 8))

    tools = sorted(power_curves['tool'].unique())

    for tool in tools:
        tool_data = power_curves[power_curves['tool'] == tool].sort_values('effect_sigma_mult')

        ax.plot(tool_data['effect_sigma_mult'], tool_data['detection_rate'],
                marker=TOOL_MARKERS.get(tool, 'o'), markersize=6,
                color=TOOL_COLORS.get(tool, '#888'), label=tool, linewidth=2)

        ax.fill_between(tool_data['effect_sigma_mult'],
                       tool_data['ci_low'], tool_data['ci_high'],
                       color=TOOL_COLORS.get(tool, '#888'), alpha=0.15)

    ax.axhline(y=0.80, color='green', linestyle=':', linewidth=1.5, label='80% power target')
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='α = 0.05 (FPR)')

    ax.set_xlabel('Effect Size (σ multiplier)')
    ax.set_ylabel('Detection Rate (Power)')
    ax.set_title('Statistical Power Curves\nHigher detection at smaller effects is better')
    ax.legend(loc='lower right', ncol=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=-0.0001)

    plt.tight_layout()
    plt.savefig(output_dir / 'power_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    return fig

def plot_power_curves_zoomed(power_curves, output_dir):
    """Plot power curves zoomed into small effect region."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter to small effects
    small_effects = power_curves[power_curves['effect_sigma_mult'] <= 0.001].copy()
    tools = sorted(small_effects['tool'].unique())

    for tool in tools:
        tool_data = small_effects[small_effects['tool'] == tool].sort_values('effect_sigma_mult')

        ax.plot(tool_data['effect_sigma_mult'], tool_data['detection_rate'],
                marker=TOOL_MARKERS.get(tool, 'o'), markersize=8,
                color=TOOL_COLORS.get(tool, '#888'), label=tool, linewidth=2.5)

        ax.fill_between(tool_data['effect_sigma_mult'],
                       tool_data['ci_low'], tool_data['ci_high'],
                       color=TOOL_COLORS.get(tool, '#888'), alpha=0.15)

    ax.axhline(y=0.80, color='green', linestyle=':', linewidth=1.5, label='80% power target')
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='α = 0.05 (FPR)')

    ax.set_xlabel('Effect Size (σ multiplier)')
    ax.set_ylabel('Detection Rate (Power)')
    ax.set_title('Power Curves — Small Effect Region (σ ≤ 0.001)\nCritical region for detecting subtle timing leaks')
    ax.legend(loc='lower right', ncol=2)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'power_curves_zoomed.png', dpi=150, bbox_inches='tight')
    plt.close()

    return fig

def plot_execution_time(time_data, output_dir):
    """Plot execution time comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    time_data_sorted = time_data.sort_values('median_ms')

    colors = [TOOL_COLORS.get(t, '#888') for t in time_data_sorted['tool']]
    bars = ax.barh(time_data_sorted['tool'], time_data_sorted['median_ms'], color=colors)

    # Add error bars
    ax.errorbar(time_data_sorted['median_ms'], time_data_sorted['tool'],
               xerr=time_data_sorted['std_ms'], fmt='none', color='black', capsize=3)

    ax.set_xlabel('Median Execution Time (ms)')
    ax.set_title('Execution Time Comparison\nLower is better')
    ax.set_xscale('log')

    # Add value labels
    for bar, val in zip(bars, time_data_sorted['median_ms']):
        ax.text(val + val*0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}ms', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'execution_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    return fig

def plot_fpr_heatmap(fpr_data, output_dir):
    """Plot FPR as a heatmap across tools and noise models."""
    pivot = fpr_data.pivot(index='tool', columns='noise_model', values='fpr')

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(pivot.values, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('False Positive Rate', rotation=-90, va='bottom')

    ax.set_title('False Positive Rate Heatmap\nGreen = low FPR (good), Red = high FPR (bad)')

    plt.tight_layout()
    plt.savefig(output_dir / 'fpr_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    return fig

def plot_noise_robustness(summary, output_dir):
    """Plot detection rate variance across noise models at fixed effect size."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    effect_sizes = [0.0004, 0.0006, 0.0008, 0.001]

    for ax, effect in zip(axes.flat, effect_sizes):
        effect_data = summary[(summary['effect_sigma_mult'] == effect) &
                              (summary['effect_pattern'] == 'shift')].copy()

        tools = sorted(effect_data['tool'].unique())
        noise_models = sorted(effect_data['noise_model'].unique())

        x = np.arange(len(noise_models))
        width = 0.1
        multiplier = 0

        for tool in tools:
            tool_data = effect_data[effect_data['tool'] == tool].set_index('noise_model')
            rates = [tool_data.loc[nm, 'detection_rate'] if nm in tool_data.index else 0
                    for nm in noise_models]

            offset = width * multiplier
            ax.bar(x + offset, rates, width, label=tool,
                  color=TOOL_COLORS.get(tool, '#888'))
            multiplier += 1

        ax.axhline(y=0.80, color='green', linestyle=':', linewidth=1, alpha=0.7)
        ax.set_xlabel('Noise Model')
        ax.set_ylabel('Detection Rate')
        ax.set_title(f'Effect σ = {effect}')
        ax.set_xticks(x + width * (len(tools) - 1) / 2)
        ax.set_xticklabels(noise_models, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Detection Rate Robustness Across Noise Models', y=1.05, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'noise_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()

    return fig

def generate_summary_stats(results, summary, fpr_data, output_dir):
    """Generate summary statistics table."""
    stats = []

    tools = sorted(results['tool'].unique())

    for tool in tools:
        tool_results = results[results['tool'] == tool]
        tool_summary = summary[summary['tool'] == tool]
        tool_fpr = fpr_data[fpr_data['tool'] == tool]

        # FPR (average across noise models at effect=0)
        avg_fpr = tool_fpr['fpr'].mean()

        # Power at small effect (σ=0.0006)
        small_effect = tool_summary[tool_summary['effect_sigma_mult'] == 0.0006]
        power_small = small_effect['detection_rate'].mean() if len(small_effect) > 0 else np.nan

        # Power at medium effect (σ=0.001)
        med_effect = tool_summary[tool_summary['effect_sigma_mult'] == 0.001]
        power_med = med_effect['detection_rate'].mean() if len(med_effect) > 0 else np.nan

        # Execution time
        median_time = tool_results['time_ms'].median()

        # Effect size for 80% power (interpolated)
        shift_data = tool_summary[tool_summary['effect_pattern'] == 'shift'].groupby('effect_sigma_mult')['detection_rate'].mean()
        effect_80 = np.nan
        for eff, rate in shift_data.items():
            if rate >= 0.80:
                effect_80 = eff
                break

        stats.append({
            'Tool': tool,
            'Avg FPR': f'{avg_fpr:.3f}',
            'FPR OK?': '✓' if avg_fpr <= 0.10 else '✗',
            'Power@0.0006σ': f'{power_small:.2f}' if not np.isnan(power_small) else 'N/A',
            'Power@0.001σ': f'{power_med:.2f}' if not np.isnan(power_med) else 'N/A',
            'Effect for 80%': f'{effect_80:.4f}' if not np.isnan(effect_80) else '>0.003',
            'Median Time (ms)': f'{median_time:.0f}'
        })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / 'summary_stats.csv', index=False)

    return stats_df

def main():
    output_dir = Path('eda_output')
    output_dir.mkdir(exist_ok=True)

    print("Loading data...")
    results, summary = load_data()

    print(f"Results: {len(results)} rows")
    print(f"Summary: {len(summary)} rows")
    print(f"Tools: {results['tool'].unique()}")
    print(f"Effect sizes: {sorted(results['effect_sigma_mult'].unique())}")
    print(f"Noise models: {results['noise_model'].unique()}")
    print()

    print("Computing FPR by tool...")
    fpr_data = compute_fpr_by_tool(results)
    print(fpr_data.groupby('tool')['fpr'].mean().sort_values())
    print()

    print("Computing power curves...")
    power_curves = compute_power_curves(summary)

    print("Computing execution times...")
    time_data = compute_execution_time(results)
    print(time_data.sort_values('median_ms'))
    print()

    print("Generating plots...")
    plot_fpr_comparison(fpr_data, output_dir)
    print("  - fpr_comparison.png")

    plot_fpr_heatmap(fpr_data, output_dir)
    print("  - fpr_heatmap.png")

    plot_power_curves(power_curves, output_dir)
    print("  - power_curves.png")

    plot_power_curves_zoomed(power_curves, output_dir)
    print("  - power_curves_zoomed.png")

    plot_execution_time(time_data, output_dir)
    print("  - execution_time.png")

    plot_noise_robustness(summary, output_dir)
    print("  - noise_robustness.png")

    print("\nGenerating summary statistics...")
    stats_df = generate_summary_stats(results, summary, fpr_data, output_dir)
    print(stats_df.to_string(index=False))

    # Generate markdown report
    report = f"""# Fine-Threshold Benchmark EDA Report

## Overview

This analysis examines the performance of timing side-channel detection tools on fine-threshold synthetic benchmarks with realistic noise models.

**Dataset:**
- {len(results):,} individual trial results
- {len(summary)} aggregated conditions
- Tools compared: {', '.join(sorted(results['tool'].unique()))}
- Effect sizes: {min(results['effect_sigma_mult'])} to {max(results['effect_sigma_mult'])} σ
- Noise models: {', '.join(sorted(results['noise_model'].unique()))}

## Key Findings

### 1. False Positive Rate (Type I Error)

| Tool | Average FPR | Status |
|------|-------------|--------|
"""

    for _, row in fpr_data.groupby('tool')['fpr'].mean().sort_values().reset_index().iterrows():
        status = "✓ Good" if row['fpr'] <= 0.10 else "✗ High"
        report += f"| {row['tool']} | {row['fpr']:.3f} | {status} |\n"

    report += f"""
**Interpretation:** Tools with FPR > 0.05 are producing false alarms above the nominal α=0.05 level.
The AD-test and KS-test show very high FPR (~0.90), making them unreliable for CI use.

### 2. Statistical Power

{stats_df.to_markdown(index=False)}

### 3. Speed vs Accuracy Trade-off

| Tool | Median Time | Speed Rank |
|------|-------------|------------|
"""

    for rank, (_, row) in enumerate(time_data.sort_values('median_ms').iterrows(), 1):
        report += f"| {row['tool']} | {row['median_ms']:.0f}ms | #{rank} |\n"

    report += """
## Conclusions

1. **timing-oracle** achieves excellent FPR control while maintaining competitive power
2. **AD-test and KS-test** have severely inflated FPR (~90%) and should not be used for decision-making
3. **RTLF and SILENT** maintain good FPR but at the cost of slower execution
4. **DudeCT** shows moderate FPR but fast execution

## Plots

- `fpr_comparison.png` - FPR across tools and noise models
- `fpr_heatmap.png` - FPR heatmap visualization
- `power_curves.png` - Statistical power vs effect size
- `power_curves_zoomed.png` - Power curves in critical small-effect region
- `execution_time.png` - Execution time comparison
- `noise_robustness.png` - Robustness across noise models
"""

    with open(output_dir / 'EDA_REPORT.md', 'w') as f:
        f.write(report)

    print(f"\nReport saved to {output_dir / 'EDA_REPORT.md'}")
    print(f"All outputs in {output_dir}/")

if __name__ == '__main__':
    main()
