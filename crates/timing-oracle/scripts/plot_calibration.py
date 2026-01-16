# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "plotnine>=0.13",
#   "pandas>=2.0",
#   "numpy>=1.24",
# ]
# ///
"""
Calibration test visualization for timing-oracle.

Generates publication-quality plots from calibration test CSV data.

Usage:
    uv run plot_calibration.py <data_dir> [--output <output_dir>]

Example:
    CALIBRATION_DATA_DIR=./calibration_data cargo test --release --test calibration_power
    uv run scripts/plot_calibration.py ./calibration_data --output ./plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_hline,
    geom_vline,
    geom_errorbar,
    geom_bar,
    geom_density,
    geom_segment,
    geom_text,
    geom_abline,
    labs,
    theme_minimal,
    theme,
    element_text,
    element_line,
    element_rect,
    element_blank,
    scale_x_continuous,
    scale_x_log10,
    scale_y_continuous,
    scale_color_manual,
    scale_fill_manual,
    coord_cartesian,
    annotate,
    facet_wrap,
    position_dodge,
    after_stat,
)

# =============================================================================
# COLOR PALETTE (Colorblind-safe)
# =============================================================================

COLORS = {
    "primary": "#2563eb",      # Blue 600
    "secondary": "#0d9488",    # Teal 600
    "accent": "#f97316",       # Orange 500
    "error": "#ef4444",        # Red 500
    "text": "#374151",         # Gray 700
    "muted": "#9ca3af",        # Gray 400
    "light": "#e5e7eb",        # Gray 200
    "background": "#ffffff",   # White
}

# =============================================================================
# CUSTOM THEME
# =============================================================================

def theme_timing_oracle():
    """Clean, minimal theme for timing-oracle plots."""
    return (
        theme_minimal() +
        theme(
            # Text
            text=element_text(family="sans-serif", color=COLORS["text"]),
            plot_title=element_text(size=14, weight="bold", margin={"b": 12}),
            plot_subtitle=element_text(size=10, color=COLORS["muted"], margin={"b": 8}),
            axis_title=element_text(size=10, margin={"t": 8, "r": 8}),
            axis_text=element_text(size=9, color=COLORS["text"]),
            legend_title=element_text(size=9, weight="bold"),
            legend_text=element_text(size=8),

            # Panel
            panel_grid_major=element_line(color=COLORS["light"], size=0.5),
            panel_grid_minor=element_blank(),
            panel_background=element_rect(fill=COLORS["background"]),
            plot_background=element_rect(fill=COLORS["background"]),

            # Legend
            legend_position="bottom",
            legend_background=element_rect(fill=COLORS["background"], color=None),

            # Spacing
            figure_size=(8, 5),
            dpi=150,
        )
    )

# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for binomial proportion."""
    if trials == 0:
        return (0.0, 1.0)

    n = trials
    p_hat = successes / n

    # z-score for confidence level
    z = 1.96 if abs(confidence - 0.95) < 0.001 else 1.645

    if successes == 0:
        upper = 1.0 - ((1.0 - confidence) / 2.0) ** (1.0 / n)
        return (0.0, upper)

    if successes == trials:
        lower = ((1.0 - confidence) / 2.0) ** (1.0 / n)
        return (lower, 1.0)

    z2 = z * z
    denom = 1.0 + z2 / n

    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = z * np.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return (lower, upper)

# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

def infer_effect_from_test_name(test_name: str) -> float:
    """Infer injected effect size from test name patterns."""
    import re

    # Pattern: power_*_Nx_theta_* where N is the multiplier
    # AdjacentNetwork has theta=100ns
    # Research has theta=50ns (nominal)
    if "adjacent_network" in test_name.lower() or "adjacentnetwork" in test_name.lower():
        theta = 100.0
    elif "research" in test_name.lower():
        theta = 50.0
    elif "remote_network" in test_name.lower() or "remotenetwork" in test_name.lower():
        theta = 50000.0
    else:
        theta = 100.0  # Default

    # Try to extract multiplier from name
    match = re.search(r'(\d+)x_theta', test_name.lower())
    if match:
        multiplier = float(match.group(1))
        return theta * multiplier

    # Fall back to looking for effect size patterns
    match = re.search(r'(\d+)ns', test_name.lower())
    if match:
        return float(match.group(1))

    return 0.0


def plot_power_curve(df: pd.DataFrame, output_path: Path):
    """
    Plot power curve: detection rate vs effect size.

    X-axis: Injected effect size (ns)
    Y-axis: Detection rate (power) %
    """
    # Filter to power test data
    power_df = df[df["test_type"] == "power"].copy()

    if power_df.empty:
        print("  No power test data found, skipping power curve")
        return

    # If injected_effect_ns is all zeros, try to infer from test names
    if (power_df["injected_effect_ns"] == 0).all():
        power_df["injected_effect_ns"] = power_df["test_name"].apply(infer_effect_from_test_name)
        print("  Inferred effect sizes from test names")

    # Filter out rows where we couldn't determine effect size
    power_df = power_df[power_df["injected_effect_ns"] > 0]

    if power_df.empty:
        print("  Could not determine effect sizes, skipping power curve")
        return

    # Aggregate by effect size
    agg = power_df.groupby("injected_effect_ns").agg(
        detected=("decision", lambda x: (x == "fail").sum()),
        total=("decision", "count"),
    ).reset_index()

    agg["power"] = agg["detected"] / agg["total"]
    agg["ci_low"] = agg.apply(lambda r: wilson_ci(int(r["detected"]), int(r["total"]))[0], axis=1)
    agg["ci_high"] = agg.apply(lambda r: wilson_ci(int(r["detected"]), int(r["total"]))[1], axis=1)

    # Create plot
    p = (
        ggplot(agg, aes(x="injected_effect_ns", y="power"))
        + geom_ribbon(aes(ymin="ci_low", ymax="ci_high"), fill=COLORS["primary"], alpha=0.2)
        + geom_line(color=COLORS["primary"], size=1.2)
        + geom_point(color=COLORS["primary"], size=3)

        # Reference lines
        + geom_hline(yintercept=0.70, linetype="dashed", color=COLORS["muted"], size=0.7)
        + geom_hline(yintercept=0.90, linetype="dotted", color=COLORS["muted"], size=0.7)

        # Labels
        + labs(
            title="Power Curve: Detection Rate vs Effect Size",
            subtitle="Shaded region shows 95% Wilson confidence interval",
            x="Injected Effect Size (ns)",
            y="Detection Rate",
        )
        + scale_y_continuous(labels=lambda l: [f"{v:.0%}" for v in l], limits=(0, 1.05))
        + scale_x_log10()
        + theme_timing_oracle()

        # Annotations
        + annotate("text", x=agg["injected_effect_ns"].max() * 0.7, y=0.72,
                   label="70% min @ 2\u03b8", size=8, color=COLORS["muted"])
        + annotate("text", x=agg["injected_effect_ns"].max() * 0.7, y=0.92,
                   label="90% min @ 5\u03b8", size=8, color=COLORS["muted"])
    )

    p.save(output_path / "power_curve.png", width=8, height=5, dpi=200)
    print(f"  Saved: {output_path / 'power_curve.png'}")


def plot_fpr_calibration(df: pd.DataFrame, output_path: Path):
    """
    Plot FPR calibration: observed FPR with confidence intervals.

    X-axis: Test configuration
    Y-axis: False positive rate %
    """
    # Filter to FPR test data
    fpr_df = df[df["test_type"] == "fpr"].copy()

    if fpr_df.empty:
        print("  No FPR test data found, skipping FPR calibration plot")
        return

    # Aggregate by test name
    agg = fpr_df.groupby("test_name").agg(
        false_positives=("decision", lambda x: (x == "fail").sum()),
        total=("decision", "count"),
    ).reset_index()

    agg["fpr"] = agg["false_positives"] / agg["total"]
    agg["ci_low"] = agg.apply(lambda r: wilson_ci(int(r["false_positives"]), int(r["total"]))[0], axis=1)
    agg["ci_high"] = agg.apply(lambda r: wilson_ci(int(r["false_positives"]), int(r["total"]))[1], axis=1)

    # Shorten test names for display
    agg["label"] = agg["test_name"].str.replace("fpr_quick_", "").str.replace("_", " ")

    # Create plot
    p = (
        ggplot(agg, aes(x="label", y="fpr"))
        + geom_errorbar(aes(ymin="ci_low", ymax="ci_high"), width=0.2, color=COLORS["text"], size=0.8)
        + geom_point(color=COLORS["text"], size=4)

        # Reference lines
        + geom_hline(yintercept=0.05, linetype="dashed", color=COLORS["primary"], size=0.8)
        + geom_hline(yintercept=0.10, linetype="dotted", color=COLORS["accent"], size=0.7)

        # Labels
        + labs(
            title="FPR Calibration: False Positive Rate Under Null",
            subtitle="Error bars show 95% Wilson confidence interval",
            x="Test Configuration",
            y="False Positive Rate",
        )
        + scale_y_continuous(labels=lambda l: [f"{v:.0%}" for v in l], limits=(0, 0.20))
        + theme_timing_oracle()
        + theme(axis_text_x=element_text(angle=15, hjust=1))

        # Annotations
        + annotate("text", x=0.5, y=0.055, label="\u03b1 = 5%", size=8, color=COLORS["primary"], ha="left")
        + annotate("text", x=0.5, y=0.105, label="max = 10%", size=8, color=COLORS["accent"], ha="left")
    )

    p.save(output_path / "fpr_calibration.png", width=7, height=5, dpi=200)
    print(f"  Saved: {output_path / 'fpr_calibration.png'}")


def plot_coverage_calibration(df: pd.DataFrame, output_path: Path):
    """
    Plot coverage calibration: CI coverage rate by effect size.

    X-axis: Injected effect size
    Y-axis: Coverage rate %
    """
    # Filter to coverage test data
    cov_df = df[df["test_type"] == "coverage"].copy()

    if cov_df.empty:
        print("  No coverage test data found, skipping coverage plot")
        return

    # For coverage, we need to check if CI contains true value
    # The data should have ci_low_ns, ci_high_ns, and injected_effect_ns
    cov_df = cov_df.dropna(subset=["ci_low_ns", "ci_high_ns"])

    if cov_df.empty:
        print("  No valid coverage data with CIs, skipping coverage plot")
        return

    # Check if CI covers the true injected value
    cov_df["covered"] = (
        (cov_df["ci_low_ns"] <= cov_df["injected_effect_ns"]) &
        (cov_df["injected_effect_ns"] <= cov_df["ci_high_ns"])
    )

    # Aggregate by effect size
    agg = cov_df.groupby("injected_effect_ns").agg(
        covered_count=("covered", "sum"),
        total=("covered", "count"),
    ).reset_index()

    agg["coverage"] = agg["covered_count"] / agg["total"]
    agg["ci_low"] = agg.apply(lambda r: wilson_ci(int(r["covered_count"]), int(r["total"]))[0], axis=1)
    agg["ci_high"] = agg.apply(lambda r: wilson_ci(int(r["covered_count"]), int(r["total"]))[1], axis=1)

    # Create plot
    p = (
        ggplot(agg, aes(x="injected_effect_ns", y="coverage"))
        + geom_bar(stat="identity", fill=COLORS["secondary"], alpha=0.8, width=40)
        + geom_errorbar(aes(ymin="ci_low", ymax="ci_high"), width=20, color=COLORS["text"], size=0.7)

        # Reference lines
        + geom_hline(yintercept=0.95, linetype="dashed", color=COLORS["primary"], size=0.8)
        + geom_hline(yintercept=0.85, linetype="dotted", color=COLORS["muted"], size=0.7)

        # Labels
        + labs(
            title="Coverage Calibration: 95% CI Contains True Value",
            subtitle="Nominal coverage = 95%, minimum acceptable = 85%",
            x="Injected Effect Size (ns)",
            y="Coverage Rate",
        )
        + scale_y_continuous(labels=lambda l: [f"{v:.0%}" for v in l], limits=(0.75, 1.02))
        + theme_timing_oracle()

        # Annotations
        + annotate("text", x=agg["injected_effect_ns"].min() * 0.9, y=0.96,
                   label="95% nominal", size=8, color=COLORS["primary"], ha="left")
    )

    p.save(output_path / "coverage_calibration.png", width=7, height=5, dpi=200)
    print(f"  Saved: {output_path / 'coverage_calibration.png'}")


def plot_effect_estimation(df: pd.DataFrame, output_path: Path):
    """
    Plot effect estimation accuracy: estimated vs true effect.

    X-axis: True injected effect (ns)
    Y-axis: Estimated effect (ns)
    """
    # Filter to data with effect estimates
    est_df = df[df["shift_ns"].notna() & (df["injected_effect_ns"] > 0)].copy()

    if est_df.empty:
        print("  No effect estimation data found, skipping scatter plot")
        return

    # Check if CI contains true value (for coloring)
    est_df["covered"] = (
        (est_df["ci_low_ns"] <= est_df["injected_effect_ns"]) &
        (est_df["injected_effect_ns"] <= est_df["ci_high_ns"])
    )
    est_df["covered_label"] = est_df["covered"].map({True: "CI covers true", False: "CI misses true"})

    max_val = max(est_df["injected_effect_ns"].max(), est_df["shift_ns"].max()) * 1.1

    # Create plot
    p = (
        ggplot(est_df, aes(x="injected_effect_ns", y="shift_ns", color="covered_label"))

        # Identity line
        + geom_abline(intercept=0, slope=1, linetype="dashed", color=COLORS["muted"], size=0.8)

        # Error bars and points
        + geom_errorbar(aes(ymin="ci_low_ns", ymax="ci_high_ns"), width=0, alpha=0.4, size=0.5)
        + geom_point(size=2, alpha=0.7)

        # Colors
        + scale_color_manual(values={
            "CI covers true": COLORS["primary"],
            "CI misses true": COLORS["error"],
        })

        # Labels
        + labs(
            title="Effect Estimation Accuracy",
            subtitle="Dashed line shows perfect estimation (y = x)",
            x="True Injected Effect (ns)",
            y="Estimated Effect (ns)",
            color="",
        )
        + coord_cartesian(xlim=(0, max_val), ylim=(0, max_val))
        + theme_timing_oracle()
        + theme(legend_position="bottom")
    )

    p.save(output_path / "effect_estimation.png", width=6, height=6, dpi=200)
    print(f"  Saved: {output_path / 'effect_estimation.png'}")


def plot_summary_dashboard(df: pd.DataFrame, output_path: Path):
    """
    Create a summary showing key metrics.
    """
    # Calculate summary stats
    total_trials = len(df)
    completed = len(df[df["decision"] != "unmeasurable"])
    unmeasurable = len(df[df["decision"] == "unmeasurable"])

    power_df = df[df["test_type"] == "power"]
    fpr_df = df[df["test_type"] == "fpr"]

    if not power_df.empty:
        power_at_max = power_df[power_df["injected_effect_ns"] == power_df["injected_effect_ns"].max()]
        power_rate = (power_at_max["decision"] == "fail").mean() if not power_at_max.empty else 0
    else:
        power_rate = 0

    if not fpr_df.empty:
        fpr_rate = (fpr_df["decision"] == "fail").mean()
    else:
        fpr_rate = 0

    # Create a simple summary text plot
    summary_text = f"""
    Calibration Summary
    ===================

    Total Trials: {total_trials}
    Completed: {completed} ({completed/total_trials*100:.1f}%)
    Unmeasurable: {unmeasurable} ({unmeasurable/total_trials*100:.1f}%)

    FPR (null hypothesis): {fpr_rate*100:.1f}%
    Power (max effect): {power_rate*100:.1f}%
    """

    # Save as text file instead of plot
    with open(output_path / "summary.txt", "w") as f:
        f.write(summary_text)
    print(f"  Saved: {output_path / 'summary.txt'}")


# =============================================================================
# MAIN
# =============================================================================

def load_data(data_dir: Path) -> pd.DataFrame:
    """Load all CSV files from data directory."""
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  Loaded: {f.name} ({len(df)} records)")
        except Exception as e:
            print(f"  Warning: Failed to load {f.name}: {e}")

    if not dfs:
        raise ValueError("No valid CSV files could be loaded")

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Generate calibration plots for timing-oracle")
    parser.add_argument("data_dir", type=Path, help="Directory containing CSV data files")
    parser.add_argument("--output", "-o", type=Path, default=Path("./plots"), help="Output directory for plots")
    args = parser.parse_args()

    # Validate input
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data from {args.data_dir}...")
    df = load_data(args.data_dir)
    print(f"Total records: {len(df)}")

    print(f"\nGenerating plots to {args.output}...")

    # Generate all plots
    plot_power_curve(df, args.output)
    plot_fpr_calibration(df, args.output)
    plot_coverage_calibration(df, args.output)
    plot_effect_estimation(df, args.output)
    plot_summary_dashboard(df, args.output)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
