#!/usr/bin/env python3
"""Analyze CVE TPR CSV into a three-way verdict table.

Input: CSV from scripts/measure_cve_tpr.sh.
Output: Markdown table matching Table 2 in the USENIX rebuttal plan:

    CVE / target    | Detect | Inconclusive | Miss

Plus Wilson 95% CIs for total Detect proportion (denominator excludes SKIP).

Usage:
    uv run python scripts/analyze_cve_tpr.py <csv_path>
    uv run python scripts/analyze_cve_tpr.py <csv_path> --format markdown
    uv run python scripts/analyze_cve_tpr.py <csv_path> --format csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


VERDICT_MAP = {
    "PASS": "Detect",
    "FAIL": "Miss",
    "INCONCLUSIVE": "Inconclusive",
    "SKIP": "Skip",
    "UNKNOWN": "Skip",
}


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a Bernoulli proportion.

    Returns (point_estimate, lo, hi) as fractions in [0, 1]. Bounds are
    clipped to [0, 1]. Uses denominator n = number of non-skip trials.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def aggregate(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    """Group rows by cve_id and tally verdicts."""
    by_cve: dict[str, dict[str, int]] = defaultdict(
        lambda: {"Detect": 0, "Inconclusive": 0, "Miss": 0, "Skip": 0}
    )
    for row in rows:
        cve = row["cve_id"]
        verdict = VERDICT_MAP.get(row.get("outcome", "UNKNOWN"), "Skip")
        by_cve[cve][verdict] += 1
    return by_cve


def render_markdown(by_cve: dict[str, dict[str, int]]) -> str:
    lines = [
        "| CVE / target                           | Detect | Inconclusive | Miss | Skip | N (non-skip) | Detect rate [Wilson 95%] |",
        "|----------------------------------------|-------:|-------------:|-----:|-----:|-------------:|--------------------------|",
    ]
    totals = {"Detect": 0, "Inconclusive": 0, "Miss": 0, "Skip": 0}
    for cve, counts in sorted(by_cve.items()):
        n = counts["Detect"] + counts["Inconclusive"] + counts["Miss"]
        p, lo, hi = wilson_ci(counts["Detect"], n)
        lines.append(
            f"| {cve:<38} | {counts['Detect']:>6} | {counts['Inconclusive']:>12} |"
            f" {counts['Miss']:>4} | {counts['Skip']:>4} | {n:>12} |"
            f" {100*p:>5.1f}% [{100*lo:>4.1f}, {100*hi:>5.1f}]         |"
        )
        for k in totals:
            totals[k] += counts[k]
    n = totals["Detect"] + totals["Inconclusive"] + totals["Miss"]
    p, lo, hi = wilson_ci(totals["Detect"], n)
    lines.append(
        f"| {'**Totals**':<38} | **{totals['Detect']:>4}** |"
        f" **{totals['Inconclusive']:>10}** | **{totals['Miss']:>2}** |"
        f" **{totals['Skip']:>2}** | **{n:>10}** |"
        f" **{100*p:.1f}% [{100*lo:.1f}, {100*hi:.1f}]**    |"
    )
    return "\n".join(lines)


def render_csv(by_cve: dict[str, dict[str, int]]) -> str:
    rows = ["cve_id,detect,inconclusive,miss,skip,n,detect_rate,wilson_lo,wilson_hi"]
    for cve, counts in sorted(by_cve.items()):
        n = counts["Detect"] + counts["Inconclusive"] + counts["Miss"]
        p, lo, hi = wilson_ci(counts["Detect"], n)
        rows.append(
            f"{cve},{counts['Detect']},{counts['Inconclusive']},{counts['Miss']},"
            f"{counts['Skip']},{n},{p:.6f},{lo:.6f},{hi:.6f}"
        )
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    args = ap.parse_args()

    if not args.csv_path.exists():
        print(f"error: {args.csv_path} not found", file=sys.stderr)
        return 2

    rows = load_rows(args.csv_path)
    by_cve = aggregate(rows)

    if not by_cve:
        print("error: no rows found", file=sys.stderr)
        return 1

    if args.format == "markdown":
        print(render_markdown(by_cve))
    else:
        print(render_csv(by_cve))
    return 0


if __name__ == "__main__":
    sys.exit(main())
