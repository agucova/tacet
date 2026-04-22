#!/usr/bin/env python3
"""Compute Tier 1 FPR (Wilson 95%) + MARVIN three-way verdicts.

Usage:
    uv run --no-project paper/author-response/crypto-cross-tool/analyze.py \
        paper/author-response/crypto-cross-tool/results.csv
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict


def wilson(s: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = s / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    print(f"Total rows: {len(rows)}")

    t1 = defaultdict(lambda: {"fp": 0, "n": 0, "err": 0})
    for r in rows:
        if r["expected"] != "constant_time":
            continue
        k = (r["tool"], r["dither_ns"])
        if r["outcome"] == "error":
            t1[k]["err"] += 1
            continue
        t1[k]["n"] += 1
        if r["outcome"] == "fail":
            t1[k]["fp"] += 1

    print("\n=== Tier 1 FPR (N=140 per cell; SILENT raw omits 22 tied-sample NaN) ===")
    print(
        f"{'Tool':<10} {'Dither':<8} {'FP/N':<12} {'Rate':<8} {'Wilson 95%':<22} {'Err'}"
    )
    for k in sorted(t1.keys()):
        d = t1[k]
        p, lo, hi = wilson(d["fp"], d["n"])
        print(
            f"{k[0]:<10} {k[1]:<8} {d['fp']:>3}/{d['n']:<5}     "
            f"{100 * p:5.1f}%  [{100 * lo:4.1f}, {100 * hi:5.1f}]%     {d['err']}"
        )

    marvin = defaultdict(lambda: {"fail": 0, "inc": 0, "pass": 0, "err": 0})
    for r in rows:
        if "marvin" not in r["test_id"]:
            continue
        k = (r["tool"], r["dither_ns"])
        v = {"fail": "fail", "inconclusive": "inc", "pass": "pass", "error": "err"}.get(
            r["outcome"], "err"
        )
        marvin[k][v] += 1

    print("\n=== MARVIN (N=20 per cell; 10k samples/class) ===")
    print(
        f"{'Tool':<10} {'Dither':<8} {'Det':<5} {'Inc':<5} {'Miss':<5} {'Err':<5} {'Det rate Wilson'}"
    )
    for k in sorted(marvin.keys()):
        d = marvin[k]
        n = d["fail"] + d["inc"] + d["pass"]
        p, lo, hi = wilson(d["fail"], n)
        print(
            f"{k[0]:<10} {k[1]:<8} {d['fail']:<5} {d['inc']:<5} {d['pass']:<5} "
            f"{d['err']:<5} {100 * p:4.0f}% [{100 * lo:3.0f}, {100 * hi:3.0f}]%"
        )

    per = defaultdict(lambda: defaultdict(lambda: {"fp": 0, "n": 0}))
    for r in rows:
        if r["expected"] != "constant_time" or r["dither_ns"] != "0.000":
            continue
        if r["outcome"] == "error":
            continue
        per[r["test_id"]][r["tool"]]["n"] += 1
        if r["outcome"] == "fail":
            per[r["test_id"]][r["tool"]]["fp"] += 1

    print("\n=== Per-primitive FPR (dither=0.0 raw) ===")
    tools = ["tacet", "dudect", "tvla", "silent", "rtlf"]
    print(f"{'Test':<48} " + " ".join(f"{t:>8}" for t in tools))
    for test in sorted(per.keys()):
        line = f"{test:<48} "
        for t in tools:
            d = per[test][t]
            line += f"{d['fp']:>2}/{d['n']:<2}    "
        print(line)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
