#!/usr/bin/env python3
"""
Plot power curves from calibration test data.

Usage:
    python plot_power_curve.py [data_dir] [output_file]

Reads CSV files matching 'power_curve_*.csv' from data_dir (default: calibration_data)
and generates a power curve plot showing detection rate vs. effect size.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_power_curve_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all power curve CSVs from data_dir."""
    curves = {}
    for csv_path in data_dir.glob("power_curve_*.csv"):
        name = csv_path.stem.replace("power_curve_", "")
        try:
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                curves[name] = df
                print(f"Loaded {csv_path.name}: {len(df)} points")
        except Exception as e:
            print(f"Warning: Could not load {csv_path}: {e}")
    return curves


def plot_power_curves(curves: dict[str, pd.DataFrame], output_path: Path):
    """Generate power curve plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette for different curves
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(curves), 1)))

    for (name, df), color in zip(curves.items(), colors):
        x = df["effect_mult"]
        y = df["detection_rate"] * 100
        ci_low = df["ci_low"] * 100
        ci_high = df["ci_high"] * 100

        # Plot line with CI band
        ax.plot(x, y, "o-", color=color, label=name, linewidth=2, markersize=6)
        ax.fill_between(x, ci_low, ci_high, color=color, alpha=0.2)

    # Reference lines
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="α = 10% (FPR target)")
    ax.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="90% power")
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5, label="θ (threshold)")

    # Formatting
    ax.set_xlabel("Effect Size (×θ)", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title("Power Curve: Detection Rate vs. Effect Size", fontsize=14)
    ax.set_xlim(-0.1, max(df["effect_mult"].max() for df in curves.values()) + 0.5)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    # Add annotation box
    textstr = "\n".join(
        [
            "Interpretation:",
            "• At 0×θ: Should be ≤10% (FPR)",
            "• At 1×θ: Transitional (~50%)",
            "• At 2×θ: Should be ≥70%",
            "• At 5×θ: Should be ≥90%",
        ]
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved power curve plot to {output_path}")

    # Also show if running interactively
    if sys.stdout.isatty():
        plt.show()


def print_summary_table(curves: dict[str, pd.DataFrame]):
    """Print a summary table of key metrics."""
    print("\n" + "=" * 70)
    print("Power Curve Summary")
    print("=" * 70)

    for name, df in curves.items():
        print(f"\n{name}:")
        print(f"  {'Effect':>8} | {'Rate':>8} | {'95% CI':>16} | {'Samples':>8}")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*16}-+-{'-'*8}")

        for _, row in df.iterrows():
            print(
                f"  {row['effect_mult']:>6.2f}×θ | {row['detection_rate']*100:>6.1f}% | "
                f"[{row['ci_low']*100:>5.1f}%, {row['ci_high']*100:>5.1f}%] | {row['median_samples']:>8}"
            )


def main():
    # Parse arguments
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("calibration_data")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "power_curve.png"

    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        print(
            "Run calibration tests first: CALIBRATION_TIER=validation cargo test --test calibration_power_curve -- --ignored"
        )
        sys.exit(1)

    # Load data
    curves = load_power_curve_data(data_dir)

    if not curves:
        print(f"No power curve data found in {data_dir}")
        print("Looking for files matching: power_curve_*.csv")
        sys.exit(1)

    # Print summary
    print_summary_table(curves)

    # Plot
    plot_power_curves(curves, output_path)


if __name__ == "__main__":
    main()
