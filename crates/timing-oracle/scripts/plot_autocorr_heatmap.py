#!/usr/bin/env python3
"""
Plot autocorrelation robustness heatmaps from calibration test data.

This replicates SILENT paper's Figure 1 style visualization, showing
rejection rates across a (μ, Φ) grid where:
- μ = effect size as multiple of threshold θ
- Φ = autocorrelation coefficient

Usage:
    python plot_autocorr_heatmap.py [data_dir] [output_file]

Reads CSV files matching 'autocorr_heatmap_*.csv' from data_dir.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def load_heatmap_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all autocorrelation heatmap CSVs from data_dir."""
    heatmaps = {}
    for csv_path in data_dir.glob("autocorr_heatmap_*.csv"):
        name = csv_path.stem.replace("autocorr_heatmap_", "")
        try:
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                heatmaps[name] = df
                print(f"Loaded {csv_path.name}: {len(df)} cells")
        except Exception as e:
            print(f"Warning: Could not load {csv_path}: {e}")
    return heatmaps


def create_heatmap_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list, list]:
    """Convert DataFrame to matrix for heatmap plotting."""
    # Get unique values
    mu_vals = sorted(df["mu_mult"].unique())
    phi_vals = sorted(df["phi"].unique())

    # Create matrix
    matrix = np.zeros((len(mu_vals), len(phi_vals)))

    for _, row in df.iterrows():
        i = mu_vals.index(row["mu_mult"])
        j = phi_vals.index(row["phi"])
        matrix[i, j] = row["rejection_rate"] * 100

    return matrix, mu_vals, phi_vals


def plot_single_heatmap(ax, df: pd.DataFrame, title: str, show_cbar: bool = True):
    """Plot a single heatmap on the given axes."""
    matrix, mu_vals, phi_vals = create_heatmap_matrix(df)

    # Custom colormap: green (low) -> yellow -> red (high)
    # For type-1 error testing, we want low values (green) when μ < θ
    cmap = LinearSegmentedColormap.from_list(
        "rejection",
        [(0.2, 0.7, 0.2), (0.9, 0.9, 0.2), (0.8, 0.2, 0.2)],  # green  # yellow  # red
    )

    im = ax.imshow(
        matrix,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=100,
        origin="lower",
    )

    # Add cell text annotations
    for i in range(len(mu_vals)):
        for j in range(len(phi_vals)):
            value = matrix[i, j]
            # Use white text for dark cells, black for light cells
            text_color = "white" if value > 50 else "black"
            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    # Axis labels
    ax.set_xticks(range(len(phi_vals)))
    ax.set_xticklabels([f"{p:.1f}" for p in phi_vals])
    ax.set_yticks(range(len(mu_vals)))
    ax.set_yticklabels([f"{m:.2f}" for m in mu_vals])
    ax.set_xlabel("Autocorrelation (Φ)", fontsize=11)
    ax.set_ylabel("Effect Size (μ/θ)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add horizontal line at μ = θ (boundary between H0 and H1)
    if 1.0 in mu_vals:
        boundary_idx = mu_vals.index(1.0)
        ax.axhline(y=boundary_idx - 0.5, color="white", linestyle="--", linewidth=2)

    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Rejection Rate (%)", fontsize=10)

    return im


def plot_heatmaps(heatmaps: dict[str, pd.DataFrame], output_path: Path):
    """Generate heatmap plot(s)."""
    n_heatmaps = len(heatmaps)

    if n_heatmaps == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        name, df = list(heatmaps.items())[0]
        plot_single_heatmap(ax, df, f"timing-oracle: {name}")
    else:
        # Multiple heatmaps in a grid
        cols = min(n_heatmaps, 3)
        rows = (n_heatmaps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = np.array(axes).flatten() if n_heatmaps > 1 else [axes]

        for ax, (name, df) in zip(axes, heatmaps.items()):
            plot_single_heatmap(ax, df, name, show_cbar=True)

        # Hide unused axes
        for ax in axes[n_heatmaps:]:
            ax.set_visible(False)

    # Add interpretation note
    fig.text(
        0.5,
        0.02,
        "Interpretation: For μ < θ (below dashed line), rejection rate should be ≤ α (10%).\n"
        "High values (red) above this line indicate good power. High values below indicate type-1 error inflation.",
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")

    if sys.stdout.isatty():
        plt.show()


def plot_comparison_with_silent(df: pd.DataFrame, output_path: Path):
    """
    Generate a comparison plot similar to SILENT paper's Figure 1.

    Shows our tool alongside what classical tools would show.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Our tool
    plot_single_heatmap(axes[0], df, "timing-oracle (this work)")

    # Simulated classical tool behavior (illustrative)
    # Classical tools have inflated type-1 error for Φ > 0
    matrix, mu_vals, phi_vals = create_heatmap_matrix(df)
    classical_matrix = matrix.copy()

    for i, mu in enumerate(mu_vals):
        for j, phi in enumerate(phi_vals):
            if phi > 0 and mu < 1.0:
                # Classical tools have inflated FPR with positive autocorrelation
                classical_matrix[i, j] = min(100, matrix[i, j] * (1 + 2 * phi))
            elif phi < 0 and mu >= 1.0:
                # Classical tools have reduced power with negative autocorrelation
                classical_matrix[i, j] = max(0, matrix[i, j] * (1 + phi))

    # Create a synthetic DataFrame for the classical simulation
    classical_df = df.copy()
    for idx, row in classical_df.iterrows():
        i = mu_vals.index(row["mu_mult"])
        j = phi_vals.index(row["phi"])
        classical_df.at[idx, "rejection_rate"] = classical_matrix[i, j] / 100

    plot_single_heatmap(
        axes[1], classical_df, "Classical Tools (simulated)\n(e.g., dudect, TVLA)"
    )

    fig.suptitle(
        "Autocorrelation Robustness: timing-oracle vs. Classical Tools",
        fontsize=14,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.02,
        "Note: Classical tool behavior is simulated based on SILENT paper findings.\n"
        "Classical tools fail to control type-1 error under positive autocorrelation (Φ > 0).",
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    comparison_path = output_path.with_stem(output_path.stem + "_comparison")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {comparison_path}")


def print_summary_table(heatmaps: dict[str, pd.DataFrame]):
    """Print a summary of type-1 error control."""
    print("\n" + "=" * 70)
    print("Autocorrelation Robustness Summary")
    print("=" * 70)

    for name, df in heatmaps.items():
        print(f"\n{name}:")

        # Check type-1 error (μ < θ)
        h0_cells = df[df["mu_mult"] < 1.0]
        if len(h0_cells) > 0:
            max_fpr = h0_cells["rejection_rate"].max() * 100
            mean_fpr = h0_cells["rejection_rate"].mean() * 100
            worst_phi = h0_cells.loc[h0_cells["rejection_rate"].idxmax(), "phi"]

            print(f"  Type-1 Error (μ < θ):")
            print(f"    Max FPR: {max_fpr:.1f}% at Φ={worst_phi:.1f}")
            print(f"    Mean FPR: {mean_fpr:.1f}%")

            if max_fpr > 15:
                print(f"    WARNING: Type-1 error exceeds 15%!")
            else:
                print(f"    OK: Type-1 error controlled")

        # Check power (μ ≥ θ)
        h1_cells = df[df["mu_mult"] >= 1.0]
        if len(h1_cells) > 0:
            min_power = h1_cells["rejection_rate"].min() * 100
            mean_power = h1_cells["rejection_rate"].mean() * 100

            print(f"  Power (μ ≥ θ):")
            print(f"    Min Power: {min_power:.1f}%")
            print(f"    Mean Power: {mean_power:.1f}%")


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("calibration_data")
    output_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "autocorr_heatmap.png"
    )

    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        print(
            "Run calibration tests first: CALIBRATION_TIER=validation cargo test --test calibration_autocorrelation -- --ignored"
        )
        sys.exit(1)

    heatmaps = load_heatmap_data(data_dir)

    if not heatmaps:
        print(f"No heatmap data found in {data_dir}")
        print("Looking for files matching: autocorr_heatmap_*.csv")
        sys.exit(1)

    print_summary_table(heatmaps)
    plot_heatmaps(heatmaps, output_path)

    # If we have data, also generate comparison plot
    if len(heatmaps) > 0:
        first_df = list(heatmaps.values())[0]
        plot_comparison_with_silent(first_df, output_path)


if __name__ == "__main__":
    main()
