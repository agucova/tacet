"""
Benchmark data loading and cleaning utilities for EDA.

Provides functions for loading overnight and fine-threshold benchmark CSV files,
parsing noise model strings into autocorrelation coefficients, and computing
detection rates with confidence intervals.
"""

from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from scipy import stats


# Colorblind-safe palette (IBM Design Library / Wong palette)
COLORBLIND_PALETTE = {
    "timing-oracle": "#0072B2",  # Blue
    "silent": "#009E73",         # Green
    "dudect": "#D55E00",         # Vermillion
    "timing-tvla": "#CC79A7",    # Reddish purple
    "ks-test": "#F0E442",        # Yellow
    "ad-test": "#56B4E9",        # Sky blue
    "mona": "#E69F00",           # Orange
    "rtlf-native": "#999999",    # Gray
    "tlsfuzzer": "#000000",      # Black
}

# Tool display names for plots
TOOL_DISPLAY_NAMES = {
    "timing-oracle": "timing-oracle",
    "silent": "SILENT",
    "dudect": "dudect",
    "timing-tvla": "TVLA",
    "ks-test": "KS-test",
    "ad-test": "AD-test",
    "mona": "MONA",
    "rtlf-native": "RTLF",
    "tlsfuzzer": "tlsfuzzer",
}

# Primary tools for main comparisons (excluding tlsfuzzer due to configuration issues)
PRIMARY_TOOLS = ["timing-oracle", "silent", "dudect", "timing-tvla", "ks-test", "ad-test", "mona"]


def parse_autocorr(noise_model: str) -> float:
    """
    Parse noise model string into autocorrelation coefficient.

    Examples:
        "ar1-0.3-realistic" -> 0.3
        "ar1-n0.4-realistic" -> -0.4
        "iid-realistic" -> 0.0
    """
    if noise_model.startswith("iid"):
        return 0.0
    elif noise_model.startswith("ar1-n"):
        # Negative autocorrelation: ar1-n0.4-realistic -> -0.4
        parts = noise_model.split("-")
        return -float(parts[1][1:])  # Skip 'n' prefix
    elif noise_model.startswith("ar1-"):
        # Positive autocorrelation: ar1-0.3-realistic -> 0.3
        parts = noise_model.split("-")
        return float(parts[1])
    else:
        return np.nan


def load_benchmark_data(path: str | Path) -> pd.DataFrame:
    """
    Load benchmark CSV and parse noise model into autocorrelation coefficient.

    Args:
        path: Path to benchmark_results.csv

    Returns:
        DataFrame with added 'autocorr' column
    """
    # keep_default_na=False prevents "null" string from being read as NaN
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    df["autocorr"] = df["noise_model"].apply(parse_autocorr)
    return df


def _get_repo_root() -> Path:
    """Get repository root directory (parent of analysis/)."""
    return Path(__file__).parent.parent


def load_overnight_data() -> pd.DataFrame:
    """Load overnight benchmark data."""
    path = _get_repo_root() / "bench_results/overnight_20260120/benchmark_results.csv"
    return load_benchmark_data(path)


def load_fine_threshold_data() -> pd.DataFrame:
    """Load fine-threshold benchmark data."""
    path = _get_repo_root() / "bench_results/fine_threshold_20260121/benchmark_results.csv"
    return load_benchmark_data(path)


def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Compute Wilson score confidence interval for a proportion.

    More accurate than normal approximation for extreme proportions (near 0 or 1).

    Args:
        successes: Number of successes
        trials: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    if trials == 0:
        return (0.0, 1.0)

    z = stats.norm.ppf((1 + confidence) / 2)
    p_hat = successes / trials

    denom = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denom

    return (max(0, center - spread), min(1, center + spread))


def compute_detection_rate(df: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    """
    Compute detection rate with Wilson 95% CI for grouped data.

    Args:
        df: DataFrame with 'detected' boolean column
        groupby_cols: Columns to group by

    Returns:
        DataFrame with detection_rate, ci_low, ci_high, n_trials columns
    """
    def agg_func(group):
        n = len(group)
        successes = group["detected"].sum()
        rate = successes / n if n > 0 else 0
        ci_low, ci_high = wilson_ci(successes, n)
        return pd.Series({
            "detection_rate": rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_trials": n,
        })

    return df.groupby(groupby_cols, as_index=False).apply(agg_func, include_groups=False)


def compute_fpr(df: pd.DataFrame, by_tool: bool = True) -> pd.DataFrame:
    """
    Compute False Positive Rate (detection at effect_sigma_mult=0).

    Args:
        df: Benchmark DataFrame
        by_tool: If True, compute FPR per tool

    Returns:
        DataFrame with FPR statistics
    """
    null_df = df[df["effect_sigma_mult"] == 0]

    if by_tool:
        return compute_detection_rate(null_df, ["tool"])
    else:
        return compute_detection_rate(null_df, [])


def compute_power_by_effect(df: pd.DataFrame, tool: str | None = None) -> pd.DataFrame:
    """
    Compute detection power across effect sizes.

    Args:
        df: Benchmark DataFrame
        tool: If specified, filter to this tool only

    Returns:
        DataFrame with detection rate by effect size
    """
    if tool is not None:
        df = df[df["tool"] == tool]

    return compute_detection_rate(df, ["tool", "effect_sigma_mult"])


def filter_primary_tools(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to primary tools for main comparisons."""
    return df[df["tool"].isin(PRIMARY_TOOLS)]


def get_tool_color(tool: str) -> str:
    """Get colorblind-safe color for a tool."""
    return COLORBLIND_PALETTE.get(tool, "#333333")


def get_tool_display_name(tool: str) -> str:
    """Get display name for a tool."""
    return TOOL_DISPLAY_NAMES.get(tool, tool)


def format_percent(value: float, decimals: int = 1) -> str:
    """Format a proportion as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def create_effect_autocorr_matrix(
    df: pd.DataFrame,
    tool: str,
    value_col: Literal["detection_rate", "detected"] = "detection_rate"
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Create a matrix of effect size vs autocorrelation for heatmaps.

    Args:
        df: Benchmark DataFrame
        tool: Tool to filter to
        value_col: Column to aggregate (detection_rate if already aggregated, detected if raw)

    Returns:
        (matrix, effect_values, autocorr_values) tuple
    """
    tool_df = df[df["tool"] == tool].copy()

    if value_col == "detected":
        # Aggregate raw data
        agg_df = tool_df.groupby(["effect_sigma_mult", "autocorr"])["detected"].mean().reset_index()
        agg_df.columns = ["effect_sigma_mult", "autocorr", "detection_rate"]
    else:
        agg_df = tool_df

    effect_values = sorted(agg_df["effect_sigma_mult"].unique())
    autocorr_values = sorted(agg_df["autocorr"].unique())

    matrix = np.full((len(autocorr_values), len(effect_values)), np.nan)

    for _, row in agg_df.iterrows():
        i = autocorr_values.index(row["autocorr"])
        j = effect_values.index(row["effect_sigma_mult"])
        matrix[i, j] = row["detection_rate"]

    return matrix, effect_values, autocorr_values
