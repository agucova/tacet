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
"""Analyze MARVIN pool-size sweep results (Reviewer B Q2).

Reads results.csv produced by `scripts/marvin_pool_sweep.sh` + the
`marvin_budget_sweep` Rust binary (with `--pool-size`), and emits:

1. Per-pool_size table (markdown) with:
   - N (rows)
   - median P(leak>θ), IQR
   - % Fail, % Inconclusive, % Pass (with Wilson 95% CI on Fail)
   - median effect, median CI width
   - median ESS, median block length
2. A pool-curve plot: effect_ns vs pool_size on log-x, per-seed + median,
   plus a second axis with %Fail.
3. A `headline.json` with rebuttal-ready per-pool numbers.

Usage:
    uv run scripts/analyze_marvin_pool.py [results_csv] [output_dir]

Defaults:
    results_csv = $HOME/marvin-pool-sweep/results.csv
    output_dir  = $HOME/marvin-pool-sweep/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Group by pool_size; compute summary statistics."""
    ordered_pools = sorted(df["pool_size"].unique())
    out_rows: list[dict] = []
    for ps in ordered_pools:
        sub = df[df["pool_size"] == ps]
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
                "pool_size": int(ps),
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
    lines.append(
        "| pool_size | N | verdicts (F/I/P) | %Fail (Wilson 95%) | median P | IQR P | median effect (ns) | median CI width (ns) | median ESS | block |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['pool_size'])} | {int(r['n_runs'])} "
            f"| {int(r['n_fail'])}/{int(r['n_inc'])}/{int(r['n_pass'])} "
            f"| {r['pct_fail']*100:.1f}% ({r['fail_ci_lo']*100:.0f}–{r['fail_ci_hi']*100:.0f}%) "
            f"| {r['P_median']:.3f} | [{r['P_iqr_lo']:.3f}, {r['P_iqr_hi']:.3f}] "
            f"| {r['effect_median_ns']:.1f} | {r['ci_width_median_ns']:.1f} "
            f"| {int(r['ess_median'])} | {int(r['block_len_median'])} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def plot_pool_curve(df: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=130)

    # Per-seed effect dots (jittered vertically by seed for readability).
    for _, row in df.iterrows():
        ax1.scatter(
            row["pool_size"],
            row["effect_ns"],
            color="#888",
            alpha=0.4,
            s=14,
        )

    # Median effect trace.
    ax1.plot(
        summary["pool_size"],
        summary["effect_median_ns"],
        color="#0077aa",
        linewidth=2.3,
        label="median effect (ns)",
        marker="o",
    )

    ax1.axhline(100.0, color="#cc3311", linestyle="--", linewidth=1.2, label=r"$\theta$ = 100 ns (AdjacentNetwork)")
    ax1.set_xscale("log")
    ax1.set_xlabel("sample-class pool size (log scale)")
    ax1.set_ylabel("effect estimate (ns)")
    ax1.grid(True, alpha=0.3, which="both")

    # Secondary axis: %Fail.
    ax2 = ax1.twinx()
    ax2.plot(
        summary["pool_size"],
        summary["pct_fail"] * 100.0,
        color="#118833",
        linewidth=2.0,
        label="%Fail",
        marker="s",
    )
    ax2.set_ylabel("% Fail (of 20 seeds)")
    ax2.set_ylim(-2, 102)

    # Combined legend.
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="lower right", frameon=False)

    ax1.set_title("MARVIN pool-size sweep (fixed 62k samples/class, 20 seeds/pool)")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"))
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    results_csv = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "marvin-pool-sweep" / "results.csv"
    )
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_csv.parent

    if not results_csv.exists():
        print(f"ERROR: results csv missing: {results_csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(results_csv)
    required = {
        "pool_size",
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
    plot_pool_curve(df, summary, out_dir / "pool_curve")

    # Monotonicity check: is %Fail monotone non-decreasing in pool_size?
    pct = summary.sort_values("pool_size")["pct_fail"].tolist()
    monotone = all(b >= a - 0.1 for a, b in zip(pct, pct[1:]))  # allow 10pt noise
    saturating = pct[-1] > 0 and (pct[-1] - pct[-2]) < 0.1 if len(pct) >= 2 else False

    headline: dict = {
        "n_total_rows": int(len(df)),
        "pool_sizes": [int(ps) for ps in summary["pool_size"]],
        "by_pool": summary.to_dict(orient="records"),
        "monotone_fail_rate": bool(monotone),
        "saturating": bool(saturating),
    }
    for _, r in summary.iterrows():
        ps = int(r["pool_size"])
        headline[f"pct_fail_pool{ps}"] = float(r["pct_fail"])
        headline[f"n_fail_pool{ps}"] = int(r["n_fail"])
        headline[f"n_inc_pool{ps}"] = int(r["n_inc"])
        headline[f"n_pass_pool{ps}"] = int(r["n_pass"])
        headline[f"P_median_pool{ps}"] = float(r["P_median"])
        headline[f"effect_median_pool{ps}_ns"] = float(r["effect_median_ns"])
        headline[f"n_runs_pool{ps}"] = int(r["n_runs"])

    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2, default=str))
    print(f"\n[monotone] fail rate non-decreasing: {monotone}")
    print(f"[saturating] last-step delta < 10pt: {saturating}")
    print(f"[headline] written to {out_dir / 'headline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
