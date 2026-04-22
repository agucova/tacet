#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///
"""Analyze MARVIN budget-sweep results.

Reads results.csv produced by `scripts/marvin_budget_sweep.sh` + the
`marvin_budget_sweep` Rust binary, and emits:

1. Per-budget table (markdown) with:
   - N (rows)
   - median P(leak>θ), IQR
   - % Fail (with Wilson 95% CI)
   - median effect, median CI width
   - median ESS, median block length
2. A learning-curve plot: P(leak>θ) vs samples_used on log scale,
   one trace per seed, median overlaid, 0.95 decision line.
3. A `headline.json` with the numbers for the rebuttal paragraph:
   - samples @ each budget_label
   - %Fail @ 1x, 3x, 5x
   - median_P @ 1x, 3x, 5x
   - decision: "A" | "B" | "C" per the plan's variant selector.

Usage:
    uv run scripts/analyze_marvin_budget.py [results_csv] [output_dir]

Defaults:
    results_csv = $HOME/marvin-sweep/results.csv
    output_dir  = $HOME/marvin-sweep/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # Φ⁻¹(1 - α/2) for α=0.05
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Group by budget_label, compute summary statistics."""
    ordered = (
        df[["budget_label", "samples_requested"]]
        .drop_duplicates()
        .sort_values("samples_requested")
    )
    out_rows: list[dict] = []
    for _, row in ordered.iterrows():
        label = row["budget_label"]
        sub = df[df["budget_label"] == label]
        n = len(sub)
        p_med = sub["leak_probability"].median()
        p_lo, p_hi = sub["leak_probability"].quantile([0.25, 0.75])
        k_fail = int((sub["verdict"] == "fail").sum())
        k_inc = int((sub["verdict"] == "inconclusive").sum())
        k_pass = int((sub["verdict"] == "pass").sum())
        fail_pct = k_fail / n if n else 0.0
        wlo, whi = wilson_ci(k_fail, n)
        effect_med = sub["effect_ns"].median()
        ci_width = (sub["ci_hi_ns"] - sub["ci_lo_ns"]).median()
        ess_med = sub["effective_sample_size"].median()
        block_med = sub["dependence_length"].median()
        analysis_s_med = sub["analysis_ms"].median() / 1000.0
        out_rows.append(
            {
                "budget_label": label,
                "samples_per_class": int(row["samples_requested"]),
                "n_runs": n,
                "pct_fail": fail_pct,
                "fail_ci_lo": wlo,
                "fail_ci_hi": whi,
                "n_pass": k_pass,
                "n_inc": k_inc,
                "n_fail": k_fail,
                "P_median": p_med,
                "P_iqr_lo": p_lo,
                "P_iqr_hi": p_hi,
                "effect_median_ns": effect_med,
                "ci_width_median_ns": ci_width,
                "ess_median": ess_med,
                "block_len_median": block_med,
                "analysis_s_median": analysis_s_med,
            }
        )
    return pd.DataFrame(out_rows)


def emit_markdown_table(summary: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("| budget | N/class | N | %Fail (Wilson 95%) | median P | IQR P | median effect (ns) | median CI width (ns) | median ESS | block |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['budget_label']} | {int(r['samples_per_class']):,} | {int(r['n_runs'])} "
            f"| {r['pct_fail']*100:.1f}% ({r['fail_ci_lo']*100:.0f}–{r['fail_ci_hi']*100:.0f}%) "
            f"| {r['P_median']:.3f} | [{r['P_iqr_lo']:.3f}, {r['P_iqr_hi']:.3f}] "
            f"| {r['effect_median_ns']:.1f} | {r['ci_width_median_ns']:.1f} "
            f"| {int(r['ess_median'])} | {int(r['block_len_median'])} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def plot_learning_curve(df: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=130)

    # Per-seed traces: one line per seed, sorted by samples_per_class.
    for seed, g in df.groupby("seed"):
        g2 = g.sort_values("samples_requested")
        ax.plot(
            g2["samples_requested"],
            g2["leak_probability"],
            color="#888",
            alpha=0.25,
            linewidth=0.9,
        )

    # Median trace.
    ax.plot(
        summary["samples_per_class"],
        summary["P_median"],
        color="#0077aa",
        linewidth=2.3,
        label="median P(leak>θ)",
        marker="o",
    )

    ax.axhline(0.95, color="#cc3311", linestyle="--", linewidth=1.3, label="Fail threshold (0.95)")
    ax.axhline(0.05, color="#118833", linestyle="--", linewidth=1.0, label="Pass threshold (0.05)")

    ax.set_xscale("log")
    ax.set_xlabel("samples per class (log scale)")
    ax.set_ylabel(r"posterior $P(\mathrm{leak}>\theta\mid \mathrm{data})$")
    ax.set_title("MARVIN posterior vs. sample budget (20 seeds × 7 budgets)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def pick_variant(summary: pd.DataFrame) -> tuple[str, dict]:
    """Decision tree from the plan."""
    by_label = {r["budget_label"]: r for _, r in summary.iterrows()}
    fail_3x = by_label.get("3.0x", {}).get("pct_fail", 0.0) if "3.0x" in by_label else 0.0
    fail_5x = by_label.get("5.0x", {}).get("pct_fail", 0.0) if "5.0x" in by_label else 0.0

    if fail_3x >= 0.85 or fail_5x >= 0.95:
        v = "A"  # budget-converges
    elif fail_5x < 0.50:
        v = "C"  # autocorrelation-bound
    else:
        v = "B"  # correctly cautious
    return v, {
        "pct_fail_3x": fail_3x,
        "pct_fail_5x": fail_5x,
    }


def main() -> int:
    results_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "marvin-sweep" / "results.csv"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_csv.parent

    if not results_csv.exists():
        print(f"ERROR: results csv missing: {results_csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(results_csv)
    required = {
        "budget_label",
        "samples_requested",
        "leak_probability",
        "effect_ns",
        "ci_lo_ns",
        "ci_hi_ns",
        "verdict",
        "dependence_length",
        "effective_sample_size",
        "seed",
        "analysis_ms",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing columns: {missing}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))

    emit_markdown_table(summary, out_dir / "summary.md")
    plot_learning_curve(df, summary, out_dir / "learning_curve")

    variant, stats = pick_variant(summary)
    headline: dict = {
        "variant": variant,
        "n_total_rows": int(len(df)),
        "by_budget": summary.to_dict(orient="records"),
        **stats,
    }
    # Convenience extraction for easy paragraph drop-in.
    for label in ("0.5x", "1.0x", "1.5x", "2.0x", "2.5x", "3.0x", "5.0x"):
        if label not in summary["budget_label"].values:
            continue
        r = summary[summary["budget_label"] == label].iloc[0]
        headline[f"P_median_{label}"] = float(r["P_median"])
        headline[f"pct_fail_{label}"] = float(r["pct_fail"])
        headline[f"n_fail_{label}"] = int(r["n_fail"])
        headline[f"n_runs_{label}"] = int(r["n_runs"])
        headline[f"effect_median_{label}_ns"] = float(r["effect_median_ns"])

    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2, default=str))
    print(f"\n[variant] → {variant}")
    print(f"[headline] written to {out_dir / 'headline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
