# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "plotnine>=0.13",
#   "pandas>=2.0",
#   "numpy>=1.24",
#   "matplotlib>=3.7",
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


def plot_bayesian_calibration(df: pd.DataFrame, output_path: Path):
    """
    Plot Bayesian calibration curve: stated probability vs empirical frequency.

    X-axis: Stated P(leak) binned into deciles
    Y-axis: Empirical frequency of true positives in that bin
    Diagonal line: Perfect calibration (y = x)
    """
    # Need leak_probability column
    calib_df = df[df["leak_probability"].notna()].copy()

    if calib_df.empty:
        print("  No Bayesian calibration data found, skipping calibration curve")
        return

    # Determine true positives: trials where there was an actual effect
    # Use injected_effect_ns > 0 as ground truth for true positives
    calib_df["is_true_positive"] = calib_df["injected_effect_ns"] > 0

    # Bin by stated probability (deciles: 0-10%, 10-20%, ..., 90-100%)
    calib_df["prob_bin"] = pd.cut(
        calib_df["leak_probability"],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        labels=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        include_lowest=True
    )

    # Aggregate by bin
    agg = calib_df.groupby("prob_bin", observed=True).agg(
        true_positives=("is_true_positive", "sum"),
        total=("is_true_positive", "count"),
    ).reset_index()

    agg["prob_bin"] = agg["prob_bin"].astype(float)
    agg["empirical_rate"] = agg["true_positives"] / agg["total"]
    agg["ci_low"] = agg.apply(lambda r: wilson_ci(int(r["true_positives"]), int(r["total"]))[0], axis=1)
    agg["ci_high"] = agg.apply(lambda r: wilson_ci(int(r["true_positives"]), int(r["total"]))[1], axis=1)

    # Create plot
    p = (
        ggplot(agg, aes(x="prob_bin", y="empirical_rate"))

        # Identity line (perfect calibration)
        + geom_abline(intercept=0, slope=1, linetype="dashed", color=COLORS["muted"], size=0.8)

        # Calibration regions
        + geom_ribbon(aes(ymin="ci_low", ymax="ci_high"), fill=COLORS["primary"], alpha=0.2)

        # Points and lines
        + geom_line(color=COLORS["primary"], size=1)
        + geom_point(color=COLORS["primary"], size=3)

        # Labels
        + labs(
            title="Bayesian Calibration Curve",
            subtitle="Stated P(leak) vs empirical true positive rate (dashed = perfect calibration)",
            x="Stated P(leak)",
            y="Empirical True Positive Rate",
        )
        + scale_x_continuous(
            labels=lambda l: [f"{v:.0%}" for v in l],
            breaks=[0.05, 0.25, 0.50, 0.75, 0.95],
            limits=(0, 1)
        )
        + scale_y_continuous(
            labels=lambda l: [f"{v:.0%}" for v in l],
            limits=(0, 1)
        )
        + coord_cartesian(xlim=(0, 1), ylim=(0, 1))
        + theme_timing_oracle()
    )

    p.save(output_path / "bayesian_calibration.png", width=6, height=6, dpi=200)
    print(f"  Saved: {output_path / 'bayesian_calibration.png'}")


def plot_power_curves_faceted(df: pd.DataFrame, output_path: Path):
    """
    Plot power curves faceted by attacker model.

    One panel per attacker model showing power vs effect size.
    """
    power_df = df[df["test_type"] == "power"].copy()

    if power_df.empty:
        print("  No power test data found, skipping faceted power curves")
        return

    # Extract attacker model from test name
    def extract_model(name: str) -> str:
        name_lower = name.lower()
        if "adjacent" in name_lower:
            return "AdjacentNetwork"
        elif "remote" in name_lower:
            return "RemoteNetwork"
        elif "research" in name_lower:
            return "Research"
        elif "shared" in name_lower:
            return "SharedHardware"
        elif "pq" in name_lower or "quantum" in name_lower:
            return "PostQuantumSentinel"
        return "Unknown"

    power_df["attacker_model"] = power_df["test_name"].apply(extract_model)

    # If injected_effect_ns is all zeros, try to infer from test names
    if (power_df["injected_effect_ns"] == 0).all():
        power_df["injected_effect_ns"] = power_df["test_name"].apply(infer_effect_from_test_name)

    # Filter out rows where we couldn't determine effect size
    power_df = power_df[power_df["injected_effect_ns"] > 0]

    if power_df.empty:
        print("  Could not determine effect sizes, skipping faceted power curves")
        return

    # Get unique models
    models = power_df["attacker_model"].unique()
    if len(models) <= 1:
        print("  Only one attacker model found, skipping faceted plot")
        return

    # Aggregate by model and effect size
    agg = power_df.groupby(["attacker_model", "injected_effect_ns"]).agg(
        detected=("decision", lambda x: (x == "fail").sum()),
        total=("decision", "count"),
    ).reset_index()

    agg["power"] = agg["detected"] / agg["total"]
    agg["ci_low"] = agg.apply(lambda r: wilson_ci(int(r["detected"]), int(r["total"]))[0], axis=1)
    agg["ci_high"] = agg.apply(lambda r: wilson_ci(int(r["detected"]), int(r["total"]))[1], axis=1)

    # Create faceted plot
    p = (
        ggplot(agg, aes(x="injected_effect_ns", y="power"))
        + geom_ribbon(aes(ymin="ci_low", ymax="ci_high"), fill=COLORS["primary"], alpha=0.2)
        + geom_line(color=COLORS["primary"], size=1)
        + geom_point(color=COLORS["primary"], size=2.5)

        # Reference lines
        + geom_hline(yintercept=0.70, linetype="dashed", color=COLORS["muted"], size=0.5)
        + geom_hline(yintercept=0.90, linetype="dotted", color=COLORS["muted"], size=0.5)

        # Facet by attacker model
        + facet_wrap("~ attacker_model", scales="free_x", ncol=2)

        # Labels
        + labs(
            title="Power Curves by Attacker Model",
            subtitle="Detection rate vs effect size (horizontal lines: 70%, 90% targets)",
            x="Injected Effect Size (ns)",
            y="Detection Rate",
        )
        + scale_y_continuous(labels=lambda l: [f"{v:.0%}" for v in l], limits=(0, 1.05))
        + scale_x_log10()
        + theme_timing_oracle()
        + theme(figure_size=(10, 8))
    )

    p.save(output_path / "power_curves_faceted.png", width=10, height=8, dpi=200)
    print(f"  Saved: {output_path / 'power_curves_faceted.png'}")


def plot_estimation_bias(df: pd.DataFrame, output_path: Path):
    """
    Plot estimation bias: bias and RMSE by effect size.

    X-axis: True effect size
    Y-axis: Bias (estimated - true) as percentage
    """
    est_df = df[df["shift_ns"].notna() & (df["injected_effect_ns"] > 0)].copy()

    if est_df.empty:
        print("  No effect estimation data found, skipping bias plot")
        return

    # Calculate bias for each trial
    est_df["bias_ns"] = est_df["shift_ns"] - est_df["injected_effect_ns"]
    est_df["bias_pct"] = est_df["bias_ns"] / est_df["injected_effect_ns"] * 100

    # Aggregate by effect size
    agg = est_df.groupby("injected_effect_ns").agg(
        mean_bias_pct=("bias_pct", "mean"),
        std_bias_pct=("bias_pct", "std"),
        mean_estimate=("shift_ns", "mean"),
        count=("shift_ns", "count"),
    ).reset_index()

    # Compute RMSE - use groupby key from index since it's excluded from group
    def compute_rmse(group):
        # The injected_effect_ns is the group key, access via group.name
        # shift_ns is positive when sample is slower (timing leak detected)
        true_effect = group.name
        return np.sqrt(((group["shift_ns"] - true_effect) ** 2).mean())

    rmse_df = est_df.groupby("injected_effect_ns").apply(compute_rmse, include_groups=False).reset_index()
    rmse_df.columns = ["injected_effect_ns", "rmse"]
    agg = agg.merge(rmse_df, on="injected_effect_ns")

    agg["rmse_pct"] = agg["rmse"] / agg["injected_effect_ns"] * 100
    agg["ci_low"] = agg["mean_bias_pct"] - 1.96 * agg["std_bias_pct"] / np.sqrt(agg["count"])
    agg["ci_high"] = agg["mean_bias_pct"] + 1.96 * agg["std_bias_pct"] / np.sqrt(agg["count"])

    # Create plot
    p = (
        ggplot(agg, aes(x="injected_effect_ns", y="mean_bias_pct"))
        + geom_hline(yintercept=0, linetype="solid", color=COLORS["muted"], size=0.8)
        + geom_hline(yintercept=20, linetype="dashed", color=COLORS["accent"], size=0.5)
        + geom_hline(yintercept=-20, linetype="dashed", color=COLORS["accent"], size=0.5)

        + geom_errorbar(aes(ymin="ci_low", ymax="ci_high"), width=0.05, color=COLORS["text"], size=0.7)
        + geom_point(color=COLORS["primary"], size=4)

        # Labels
        + labs(
            title="Estimation Bias by Effect Size",
            subtitle="Dashed lines show \u00b120% bias threshold",
            x="True Effect Size (ns)",
            y="Bias ((estimate - true) / true \u00d7 100%)",
        )
        + scale_x_log10()
        + scale_y_continuous(labels=lambda l: [f"{v:.0f}%" for v in l])
        + theme_timing_oracle()

        # Annotate with RMSE
        + geom_text(aes(label="rmse_pct"), format_string="{:.0f}% RMSE", nudge_y=10, size=7, color=COLORS["muted"])
    )

    p.save(output_path / "estimation_bias.png", width=8, height=5, dpi=200)
    print(f"  Saved: {output_path / 'estimation_bias.png'}")


def plot_compact_dashboard(df: pd.DataFrame, output_path: Path):
    """
    Create a compact, information-dense dashboard combining all metrics.

    Single plot with:
    - Top: Key metrics table with pass/fail status
    - Middle: Effect estimation scatter with per-point data
    - Bottom: FPR and power summary bars
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 10), facecolor='white')
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 2, 1.5], hspace=0.3, wspace=0.3)

    # Color scheme
    PASS_COLOR = '#22c55e'  # Green
    FAIL_COLOR = '#ef4444'  # Red
    PRIMARY = '#2563eb'     # Blue
    MUTED = '#9ca3af'       # Gray

    # =========================================================================
    # TOP LEFT: FPR Results
    # =========================================================================
    ax_fpr = fig.add_subplot(gs[0, 0])

    fpr_df = df[df["test_type"] == "fpr"].copy()
    if not fpr_df.empty:
        # Group by test configuration
        fpr_groups = fpr_df.groupby("test_name").agg(
            failures=("decision", lambda x: (x == "fail").sum()),
            total=("decision", "count"),
        ).reset_index()
        fpr_groups["fpr"] = fpr_groups["failures"] / fpr_groups["total"]
        fpr_groups["label"] = fpr_groups["test_name"].str.extract(r'(fixed|random)[^_]*_vs_([^_]+)', expand=False).apply(
            lambda x: f"{x[0]} vs {x[1]}" if isinstance(x, tuple) and len(x) == 2 else "unknown", axis=1
        )

        # Simple extraction
        fpr_groups["label"] = fpr_groups["test_name"].apply(
            lambda x: "fixed vs fixed" if "fixed_vs_fixed" in x else ("random vs random" if "random" in x else x[:20])
        )

        y_pos = range(len(fpr_groups))
        colors = [PASS_COLOR if fpr <= 0.10 else FAIL_COLOR for fpr in fpr_groups["fpr"]]

        bars = ax_fpr.barh(y_pos, fpr_groups["fpr"] * 100, color=colors, height=0.6, alpha=0.8)
        ax_fpr.set_yticks(y_pos)
        ax_fpr.set_yticklabels(fpr_groups["label"])
        ax_fpr.set_xlabel("FPR (%)")
        ax_fpr.set_title("False Positive Rate (target ≤10%)", fontweight='bold', fontsize=11)
        ax_fpr.axvline(x=10, color=FAIL_COLOR, linestyle='--', alpha=0.5, label='max=10%')
        ax_fpr.axvline(x=5, color=MUTED, linestyle=':', alpha=0.5, label='α=5%')
        ax_fpr.set_xlim(0, max(15, fpr_groups["fpr"].max() * 100 + 5))

        # Add value labels
        for i, (bar, row) in enumerate(zip(bars, fpr_groups.itertuples())):
            width = bar.get_width()
            ci_low, ci_high = wilson_ci(int(row.failures), int(row.total))
            ax_fpr.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{row.fpr*100:.1f}% [{ci_low*100:.0f}-{ci_high*100:.0f}%] n={row.total}',
                       va='center', fontsize=9)
    else:
        ax_fpr.text(0.5, 0.5, "No FPR data", ha='center', va='center', transform=ax_fpr.transAxes)
        ax_fpr.set_title("False Positive Rate", fontweight='bold', fontsize=11)

    ax_fpr.spines['top'].set_visible(False)
    ax_fpr.spines['right'].set_visible(False)

    # =========================================================================
    # TOP RIGHT: Power Results
    # =========================================================================
    ax_power = fig.add_subplot(gs[0, 1])

    power_df = df[df["test_type"] == "power"].copy()
    if not power_df.empty:
        # Infer effect sizes if needed
        if (power_df["injected_effect_ns"] == 0).all():
            power_df["injected_effect_ns"] = power_df["test_name"].apply(infer_effect_from_test_name)

        power_df = power_df[power_df["injected_effect_ns"] > 0]

        if not power_df.empty:
            power_groups = power_df.groupby("injected_effect_ns").agg(
                detections=("decision", lambda x: (x == "fail").sum()),
                total=("decision", "count"),
            ).reset_index()
            power_groups["power"] = power_groups["detections"] / power_groups["total"]

            y_pos = range(len(power_groups))
            colors = [PASS_COLOR if p >= 0.90 else (MUTED if p >= 0.70 else FAIL_COLOR) for p in power_groups["power"]]

            bars = ax_power.barh(y_pos, power_groups["power"] * 100, color=colors, height=0.6, alpha=0.8)
            ax_power.set_yticks(y_pos)
            ax_power.set_yticklabels([f'{int(e)}ns' for e in power_groups["injected_effect_ns"]])
            ax_power.set_xlabel("Detection Rate (%)")
            ax_power.set_title("Statistical Power (target ≥90%)", fontweight='bold', fontsize=11)
            ax_power.axvline(x=90, color=PASS_COLOR, linestyle='--', alpha=0.5)
            ax_power.axvline(x=70, color=MUTED, linestyle=':', alpha=0.5)
            ax_power.set_xlim(0, 105)

            # Add value labels
            for i, (bar, row) in enumerate(zip(bars, power_groups.itertuples())):
                width = bar.get_width()
                ci_low, ci_high = wilson_ci(int(row.detections), int(row.total))
                ax_power.text(width + 1, bar.get_y() + bar.get_height()/2,
                            f'{row.power*100:.0f}% n={row.total}',
                            va='center', fontsize=9)
    else:
        ax_power.text(0.5, 0.5, "No power data", ha='center', va='center', transform=ax_power.transAxes)
        ax_power.set_title("Statistical Power", fontweight='bold', fontsize=11)

    ax_power.spines['top'].set_visible(False)
    ax_power.spines['right'].set_visible(False)

    # =========================================================================
    # MIDDLE: Effect Estimation Scatter
    # =========================================================================
    ax_est = fig.add_subplot(gs[1, :])

    est_df = df[df["test_type"] == "estimation"].copy()
    if est_df.empty:
        # Try to find estimation data from test names
        est_df = df[df["test_name"].str.contains("estimation", case=False, na=False)].copy()

    if not est_df.empty and "shift_ns" in est_df.columns:
        est_df = est_df[est_df["shift_ns"].notna() & (est_df["injected_effect_ns"] > 0)].copy()

        if not est_df.empty:
            # shift_ns is positive when sample is slower (timing leak detected)
            est_df["estimated"] = est_df["shift_ns"]
            est_df["true"] = est_df["injected_effect_ns"]
            est_df["bias"] = est_df["estimated"] - est_df["true"]
            est_df["bias_pct"] = est_df["bias"] / est_df["true"] * 100

            # Check if CI covers true value
            if "ci_low_ns" in est_df.columns and "ci_high_ns" in est_df.columns:
                est_df["covers"] = (est_df["ci_low_ns"] <= est_df["true"]) & (est_df["true"] <= est_df["ci_high_ns"])
            else:
                est_df["covers"] = False

            # Plot identity line
            max_val = max(est_df["true"].max(), est_df["estimated"].max()) * 1.1
            ax_est.plot([0, max_val], [0, max_val], '--', color=MUTED, alpha=0.7, label='Perfect (y=x)')

            # Plot ±20% bounds
            x_range = np.linspace(0, max_val, 100)
            ax_est.fill_between(x_range, x_range * 0.8, x_range * 1.2, alpha=0.1, color=PASS_COLOR, label='±20% bounds')

            # Scatter with CI error bars
            for true_val in est_df["true"].unique():
                subset = est_df[est_df["true"] == true_val]

                # Plot error bars if available
                if "ci_low_ns" in subset.columns:
                    for _, row in subset.iterrows():
                        color = PRIMARY if row["covers"] else FAIL_COLOR
                        ax_est.plot([row["true"], row["true"]], [row["ci_low_ns"], row["ci_high_ns"]],
                                   color=color, alpha=0.3, linewidth=1)

                # Plot points
                colors = [PRIMARY if c else FAIL_COLOR for c in subset["covers"]]
                ax_est.scatter(subset["true"], subset["estimated"], c=colors, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

                # Add stats annotation
                mean_est = subset["estimated"].mean()
                mean_bias = subset["bias_pct"].mean()
                n = len(subset)
                coverage = subset["covers"].mean() * 100

                # Position annotation to the right of points
                ax_est.annotate(
                    f'μ={mean_est:.0f}ns\nbias={mean_bias:+.1f}%\nCI cov={coverage:.0f}%\nn={n}',
                    xy=(true_val, mean_est),
                    xytext=(true_val + max_val*0.08, mean_est),
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=MUTED, alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color=MUTED, alpha=0.5)
                )

            ax_est.set_xlabel("True Injected Effect (ns)", fontsize=10)
            ax_est.set_ylabel("Estimated Effect (ns)", fontsize=10)
            ax_est.set_title("Effect Estimation Accuracy (each point = one trial)", fontweight='bold', fontsize=11)
            ax_est.set_xlim(0, max_val)
            ax_est.set_ylim(0, max_val)

            # Custom legend
            handles = [
                mpatches.Patch(color=PRIMARY, alpha=0.6, label='CI covers true'),
                mpatches.Patch(color=FAIL_COLOR, alpha=0.6, label='CI misses true'),
                plt.Line2D([0], [0], linestyle='--', color=MUTED, label='Perfect estimation'),
            ]
            ax_est.legend(handles=handles, loc='upper left', fontsize=9)
    else:
        ax_est.text(0.5, 0.5, "No estimation data", ha='center', va='center', transform=ax_est.transAxes)
        ax_est.set_title("Effect Estimation Accuracy", fontweight='bold', fontsize=11)

    ax_est.spines['top'].set_visible(False)
    ax_est.spines['right'].set_visible(False)

    # =========================================================================
    # BOTTOM: Summary Statistics Table
    # =========================================================================
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    # Compute summary stats
    total_trials = len(df)
    completed = len(df[df["decision"].isin(["pass", "fail", "inconclusive"])])
    unmeasurable = len(df[df["decision"] == "unmeasurable"])

    # FPR stats
    if not fpr_df.empty:
        fpr_failures = (fpr_df["decision"] == "fail").sum()
        fpr_total = len(fpr_df)
        fpr_rate = fpr_failures / fpr_total
        fpr_low, fpr_high = wilson_ci(fpr_failures, fpr_total)
        fpr_status = "✓ PASS" if fpr_high <= 0.10 else "✗ FAIL"
    else:
        fpr_rate, fpr_total, fpr_status = 0, 0, "N/A"

    # Power stats
    if not power_df.empty and len(power_df) > 0:
        power_detections = (power_df["decision"] == "fail").sum()
        power_total = len(power_df)
        power_rate = power_detections / power_total
        power_status = "✓ PASS" if power_rate >= 0.90 else "✗ FAIL"
    else:
        power_rate, power_total, power_status = 0, 0, "N/A"

    # Estimation stats
    if not est_df.empty and "bias_pct" in est_df.columns:
        mean_bias = est_df["bias_pct"].abs().mean()
        est_status = "✓ PASS" if mean_bias <= 25 else "✗ FAIL"
    else:
        mean_bias, est_status = 0, "N/A"

    # Create table data
    table_data = [
        ["Metric", "Value", "Target", "Status"],
        ["─" * 20, "─" * 25, "─" * 15, "─" * 10],
        ["Total Trials", f"{total_trials}", "", ""],
        ["Completed", f"{completed} ({completed/max(1,total_trials)*100:.0f}%)", "", ""],
        ["", "", "", ""],
        ["False Positive Rate", f"{fpr_rate*100:.1f}% (n={fpr_total})", "≤ 10%", fpr_status],
        ["Statistical Power", f"{power_rate*100:.0f}% (n={power_total})", "≥ 90%", power_status],
        ["Mean |Bias|", f"{mean_bias:.1f}%", "≤ 25%", est_status],
    ]

    # Draw table as text
    y_start = 0.85
    row_height = 0.11
    col_widths = [0.25, 0.35, 0.2, 0.15]
    col_starts = [0.02, 0.27, 0.62, 0.82]

    for i, row in enumerate(table_data):
        y = y_start - i * row_height
        for j, (cell, col_start) in enumerate(zip(row, col_starts)):
            weight = 'bold' if i == 0 else 'normal'
            color = PASS_COLOR if '✓' in cell else (FAIL_COLOR if '✗' in cell else 'black')
            ax_table.text(col_start, y, cell, transform=ax_table.transAxes, fontsize=11,
                         fontweight=weight, color=color, family='monospace')

    ax_table.set_title("Summary Statistics", fontweight='bold', fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(output_path / "dashboard.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path / 'dashboard.png'}")


def plot_summary_dashboard(df: pd.DataFrame, output_path: Path):
    """
    Create a comprehensive summary showing key metrics.
    """
    # Calculate summary stats
    total_trials = len(df)
    completed = len(df[df["decision"] != "unmeasurable"])
    unmeasurable = len(df[df["decision"] == "unmeasurable"])

    power_df = df[df["test_type"] == "power"]
    fpr_df = df[df["test_type"] == "fpr"]
    coverage_df = df[df["test_type"] == "coverage"]
    bayesian_df = df[df["leak_probability"].notna()]

    # FPR stats
    if not fpr_df.empty:
        fpr_rate = (fpr_df["decision"] == "fail").mean()
        fpr_trials = len(fpr_df)
        fpr_low, fpr_high = wilson_ci(int((fpr_df["decision"] == "fail").sum()), fpr_trials)
    else:
        fpr_rate, fpr_trials, fpr_low, fpr_high = 0, 0, 0, 0

    # Power stats at various effect sizes
    power_stats = []
    if not power_df.empty:
        for effect in power_df["injected_effect_ns"].unique():
            if effect > 0:
                subset = power_df[power_df["injected_effect_ns"] == effect]
                detected = (subset["decision"] == "fail").sum()
                total = len(subset)
                rate = detected / total if total > 0 else 0
                low, high = wilson_ci(detected, total)
                power_stats.append((effect, rate, low, high, total))

    # Coverage stats
    if not coverage_df.empty:
        coverage_df = coverage_df.dropna(subset=["ci_low_ns", "ci_high_ns"])
        if not coverage_df.empty:
            covered = (
                (coverage_df["ci_low_ns"] <= coverage_df["injected_effect_ns"]) &
                (coverage_df["injected_effect_ns"] <= coverage_df["ci_high_ns"])
            ).sum()
            coverage_total = len(coverage_df)
            coverage_rate = covered / coverage_total if coverage_total > 0 else 0
            coverage_low, coverage_high = wilson_ci(covered, coverage_total)
        else:
            coverage_rate, coverage_total, coverage_low, coverage_high = 0, 0, 0, 0
    else:
        coverage_rate, coverage_total, coverage_low, coverage_high = 0, 0, 0, 0

    # Bayesian calibration stats
    if not bayesian_df.empty:
        calib_df = bayesian_df.copy()
        calib_df["is_true_positive"] = calib_df["injected_effect_ns"] > 0

        # Compute calibration error (mean absolute deviation from identity)
        calib_df["prob_bin"] = (calib_df["leak_probability"] * 10).astype(int) / 10 + 0.05
        bin_agg = calib_df.groupby("prob_bin").agg(
            tp=("is_true_positive", "sum"),
            total=("is_true_positive", "count"),
        )
        bin_agg["empirical"] = bin_agg["tp"] / bin_agg["total"]
        bin_agg["deviation"] = (bin_agg.index - bin_agg["empirical"]).abs()
        mean_calibration_error = bin_agg["deviation"].mean()
        max_calibration_error = bin_agg["deviation"].max()
    else:
        mean_calibration_error, max_calibration_error = 0, 0

    # Create summary text
    summary_text = f"""
================================================================================
                    CALIBRATION SUMMARY REPORT
================================================================================

OVERVIEW
--------
Total Trials:     {total_trials:>8}
Completed:        {completed:>8} ({completed/total_trials*100:.1f}%)
Unmeasurable:     {unmeasurable:>8} ({unmeasurable/total_trials*100:.1f}%)

FALSE POSITIVE RATE (FPR)
-------------------------
FPR:              {fpr_rate*100:>7.1f}%  [95% CI: {fpr_low*100:.1f}% - {fpr_high*100:.1f}%]
Trials:           {fpr_trials:>8}
Target:           ≤ 5% (α = 0.05), acceptable ≤ 10%
Status:           {'✓ PASS' if fpr_high <= 0.10 else '✗ FAIL'}

STATISTICAL POWER
-----------------"""

    for effect, rate, low, high, n in sorted(power_stats):
        target = "70%" if effect <= 200 else ("90%" if effect <= 500 else "95%")
        status = "✓" if rate >= 0.70 else "✗"
        summary_text += f"""
{effect:>6.0f}ns effect: {rate*100:>5.1f}%  [95% CI: {low*100:.1f}% - {high*100:.1f}%] (n={n}) {status}"""

    summary_text += f"""

CI COVERAGE
-----------
Coverage:         {coverage_rate*100:>7.1f}%  [95% CI: {coverage_low*100:.1f}% - {coverage_high*100:.1f}%]
Trials:           {coverage_total:>8}
Target:           ≥ 85% (nominal 95%)
Status:           {'✓ PASS' if coverage_rate >= 0.85 else '✗ FAIL'}

BAYESIAN CALIBRATION
--------------------
Mean Calibration Error: {mean_calibration_error*100:>5.1f}%
Max Deviation:          {max_calibration_error*100:>5.1f}%
Target:                 ≤ 15% mean, ≤ 25% max
Status:                 {'✓ PASS' if mean_calibration_error <= 0.15 and max_calibration_error <= 0.25 else '✗ FAIL'}

================================================================================
"""

    # Save as text file
    with open(output_path / "summary.txt", "w") as f:
        f.write(summary_text)
    print(f"  Saved: {output_path / 'summary.txt'}")

    # Also print to console
    print(summary_text)


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
    plot_compact_dashboard(df, args.output)  # Main dashboard - most useful
    plot_power_curve(df, args.output)
    plot_power_curves_faceted(df, args.output)
    plot_fpr_calibration(df, args.output)
    plot_bayesian_calibration(df, args.output)
    plot_coverage_calibration(df, args.output)
    plot_effect_estimation(df, args.output)
    plot_estimation_bias(df, args.output)
    plot_summary_dashboard(df, args.output)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
