"""
SILENT-style visualization functions for benchmark data.

Replicates the heatmap style from the SILENT paper (arXiv:2504.19821):
- Dual colormap: H0 region (μ < Δ) uses green→red, H1 region (μ ≥ Δ) uses red→green
- Dashed vertical line at threshold Δ
- Axes labeled "Side channel (μ)" and "Dependence (Φ)"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from pathlib import Path

from benchmark_utils import (
    load_fine_threshold_data,
    load_overnight_data,
    get_tool_display_name,
)


# SILENT paper colormaps
# H0: Low rejection is good (green), high is bad (red)
CMAP_H0 = LinearSegmentedColormap.from_list(
    'h0_cmap',
    [(0.95, 0.95, 0.95),  # Very light (0%)
     (0.8, 0.9, 0.8),     # Light green
     (0.6, 0.8, 0.4),     # Yellow-green
     (0.9, 0.8, 0.2),     # Yellow
     (0.95, 0.6, 0.2),    # Orange
     (0.9, 0.2, 0.2)],    # Red (100%)
    N=256
)

# H1: High detection is good (green), low is bad (red)
CMAP_H1 = LinearSegmentedColormap.from_list(
    'h1_cmap',
    [(0.9, 0.2, 0.2),     # Red (0%)
     (0.95, 0.6, 0.2),    # Orange
     (0.9, 0.8, 0.2),     # Yellow
     (0.6, 0.8, 0.4),     # Yellow-green
     (0.3, 0.7, 0.3),     # Green
     (0.1, 0.5, 0.1)],    # Dark green (100%)
    N=256
)


def create_detection_matrix(df: pd.DataFrame, tool: str) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Create a matrix of detection rates for heatmap plotting.

    Returns:
        (matrix, effect_values, autocorr_values)
        matrix[i, j] = detection rate at autocorr_values[i], effect_values[j]
    """
    tool_df = df[df['tool'] == tool].copy()

    # Aggregate by (effect_sigma_mult, autocorr)
    agg_df = tool_df.groupby(['effect_sigma_mult', 'autocorr'])['detected'].mean().reset_index()

    effect_values = sorted(agg_df['effect_sigma_mult'].unique())
    autocorr_values = sorted(agg_df['autocorr'].unique())

    matrix = np.full((len(autocorr_values), len(effect_values)), np.nan)

    for _, row in agg_df.iterrows():
        i = autocorr_values.index(row['autocorr'])
        j = effect_values.index(row['effect_sigma_mult'])
        matrix[i, j] = row['detected']

    return matrix, effect_values, autocorr_values


def plot_silent_style_heatmap(
    df: pd.DataFrame,
    tool: str,
    threshold_idx: int = 1,  # Index of first H1 column (effect >= threshold)
    ax: plt.Axes = None,
    title: str = None,
    show_values: bool = False,
) -> plt.Axes:
    """
    Plot a SILENT-style heatmap with dual colormap.

    Args:
        df: Benchmark DataFrame with 'tool', 'effect_sigma_mult', 'autocorr', 'detected' columns
        tool: Tool name to filter
        threshold_idx: Index of the first column that's in H1 region (μ ≥ Δ)
        ax: Matplotlib axes (created if None)
        title: Plot title (uses tool name if None)
        show_values: Whether to show numeric values in cells
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    matrix, effect_values, autocorr_values = create_detection_matrix(df, tool)

    n_autocorr = len(autocorr_values)
    n_effect = len(effect_values)

    # Create the dual-colored heatmap
    # We'll plot H0 and H1 regions separately

    # H0 region (columns < threshold_idx)
    if threshold_idx > 0:
        h0_matrix = matrix[:, :threshold_idx]
        h0_extent = [-0.5, threshold_idx - 0.5, -0.5, n_autocorr - 0.5]
        ax.imshow(h0_matrix, cmap=CMAP_H0, aspect='auto', vmin=0, vmax=1,
                  extent=h0_extent, origin='lower')

    # H1 region (columns >= threshold_idx)
    if threshold_idx < n_effect:
        h1_matrix = matrix[:, threshold_idx:]
        h1_extent = [threshold_idx - 0.5, n_effect - 0.5, -0.5, n_autocorr - 0.5]
        ax.imshow(h1_matrix, cmap=CMAP_H1, aspect='auto', vmin=0, vmax=1,
                  extent=h1_extent, origin='lower')

    # Add dashed vertical line at threshold
    ax.axvline(x=threshold_idx - 0.5, color='black', linestyle='--', linewidth=2)

    # Add cell values if requested
    if show_values:
        for i in range(n_autocorr):
            for j in range(n_effect):
                val = matrix[i, j]
                if not np.isnan(val):
                    # Choose text color based on background
                    if j < threshold_idx:
                        text_color = 'white' if val > 0.5 else 'black'
                    else:
                        text_color = 'white' if val < 0.5 else 'black'
                    ax.text(j, i, f'{val*100:.0f}', ha='center', va='center',
                           fontsize=7, color=text_color, fontweight='bold')

    # Axis labels and ticks
    ax.set_xticks(range(n_effect))
    ax.set_xticklabels([f'{e:.4f}' if e < 0.001 else f'{e:.3f}' for e in effect_values],
                       rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_autocorr))
    ax.set_yticklabels([f'{a:.1f}' for a in autocorr_values], fontsize=10)

    ax.set_xlabel('Side channel (μ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dependence (Φ)', fontsize=12, fontweight='bold')

    if title is None:
        title = get_tool_display_name(tool)
    ax.set_title(title, fontsize=14, fontweight='bold')

    return ax


def create_silent_legend(ax: plt.Axes = None, orientation: str = 'vertical') -> plt.Figure:
    """
    Create a SILENT-style legend with dual colorbars for H0 and H1.
    """
    if orientation == 'vertical':
        fig, axes = plt.subplots(1, 2, figsize=(3, 5))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(5, 2))

    # H0 colorbar
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    axes[0].imshow(gradient, cmap=CMAP_H0, aspect='auto', origin='lower')
    axes[0].set_xticks([])
    axes[0].set_yticks([0, 64, 128, 192, 255])
    axes[0].set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
    axes[0].set_ylabel('H₀', fontsize=14, fontweight='bold', rotation=0, labelpad=15)
    axes[0].yaxis.set_label_position('right')

    # H1 colorbar
    axes[1].imshow(gradient, cmap=CMAP_H1, aspect='auto', origin='lower')
    axes[1].set_xticks([])
    axes[1].set_yticks([0, 64, 128, 192, 255])
    axes[1].set_yticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
    axes[1].set_ylabel('H₁', fontsize=14, fontweight='bold', rotation=0, labelpad=15)
    axes[1].yaxis.set_label_position('right')

    plt.tight_layout()
    return fig


def generate_all_silent_heatmaps(output_dir: Path = None):
    """
    Generate SILENT-style heatmaps for all tools in the fine-threshold dataset.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    df = load_fine_threshold_data()

    # Tools to plot (exclude tlsfuzzer which has issues)
    tools = ['timing-oracle', 'silent', 'dudect', 'timing-tvla', 'ks-test', 'ad-test']

    # For fine-threshold data, threshold is at effect=0 (first non-zero column is H1)
    # effect_sigma_mult values: [0, 0.0001, 0.0002, ...]
    # threshold_idx = 1 means column 0 is H0, columns 1+ are H1
    threshold_idx = 1

    # Generate individual heatmaps
    for tool in tools:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_silent_style_heatmap(df, tool, threshold_idx=threshold_idx, ax=ax, show_values=True)
        plt.tight_layout()
        fig.savefig(output_dir / f'silent_style_{tool.replace("-", "_")}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_dir / f'silent_style_{tool.replace('-', '_')}.png'}")

    # Generate comparison grid (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, tool in enumerate(tools):
        plot_silent_style_heatmap(df, tool, threshold_idx=threshold_idx, ax=axes[i], show_values=False)

    fig.suptitle('Detection Rate Heatmaps (SILENT-style)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'silent_style_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / 'silent_style_comparison.png'}")

    # Generate legend
    legend_fig = create_silent_legend()
    legend_fig.savefig(output_dir / 'silent_style_legend.png', dpi=150, bbox_inches='tight')
    plt.close(legend_fig)
    print(f"Saved: {output_dir / 'silent_style_legend.png'}")


if __name__ == '__main__':
    generate_all_silent_heatmaps()
