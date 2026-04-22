#!/usr/bin/env python3
"""
Analyze True Positive Rate (TPR) results from measure_tpr.sh.

Inverted-semantics companion to analyze_fpr.py: for known-leaky crypto tests,
CSV outcome `PASS` means the oracle detected the leak (true positive) and
CSV outcome `FAIL` means the oracle missed the leak (false negative).

Computes:
- Overall TPR with Wilson 95% CIs (conditional on definitive verdict).
- Per-family (test-module) detection curve.
- Per-test breakdown.
- Identifies specific false negatives.
- If tests share the `injected_shift_<N>ns` naming pattern, produces a
  detection-rate-vs-delay curve that is ready to paste into the response.

Usage:
    ./scripts/analyze_tpr.py <results.csv>
"""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Pattern-match on test names like "injected::injected_shift_100ns" to extract
# the injected delay magnitude for the power curve. An optional "_k<N>" suffix
# marks operation-loop amplification (cf. ShowTime, Rokicki et al.): the sample
# class repeats the per-op busy_wait N times per query, so effective per-query
# delay = delay_ns × k. k defaults to 1 when the suffix is absent.
_DELAY_RE = re.compile(r"injected_shift_(\d+)ns(?:_k(\d+))?")


def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if trials == 0:
        return (0.0, 1.0)
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt((p * (1 - p) / trials) + (z * z / (4 * trials * trials))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_results(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _classify(row: dict) -> str:
    """Map CSV outcome to leaky-test semantic class."""
    outcome = row["outcome"]
    if outcome == "PASS":
        return "tp"         # oracle detected leak
    if outcome == "FAIL":
        return "fn"         # oracle missed leak
    if outcome == "INCONCLUSIVE":
        return "inconclusive"
    if outcome in ("SKIP", "UNKNOWN"):
        return "skip"
    return "skip"


def analyze(results: list[dict]) -> None:
    total = len(results)
    if total == 0:
        print("No results found in CSV file.")
        return

    classes = Counter(_classify(r) for r in results)
    tp = classes.get("tp", 0)
    fn = classes.get("fn", 0)
    inc = classes.get("inconclusive", 0)
    skip = classes.get("skip", 0)

    print("=" * 70)
    print("TRUE POSITIVE RATE ANALYSIS (leaky-test interpretation)")
    print("=" * 70)
    print()
    print(f"Total test runs: {total}")
    print(f"  True positive (detected):  {tp:5d} ({100 * tp / total:5.1f}%)")
    print(f"  False negative (missed):   {fn:5d} ({100 * fn / total:5.1f}%)")
    print(f"  Inconclusive:              {inc:5d} ({100 * inc / total:5.1f}%)")
    if skip:
        print(f"  Skipped / unmeasurable:    {skip:5d} ({100 * skip / total:5.1f}%)")

    # TPR conditional on a definitive verdict (not Inconclusive, not Skip).
    definitive = tp + fn
    if definitive > 0:
        tpr = tp / definitive
        lo, hi = wilson_ci(tp, definitive)
        print()
        print("-" * 70)
        print("TPR CONDITIONAL ON DEFINITIVE VERDICT (Pass or Fail)")
        print("-" * 70)
        print(f"  {tp}/{definitive} = {100 * tpr:.2f}%")
        print(f"  Wilson 95% CI: [{100 * lo:.2f}%, {100 * hi:.2f}%]")
    else:
        print("\n(No definitive verdicts — all trials were Inconclusive or skipped.)")

    # Also report "resolved-rate": fraction where the oracle made ANY call,
    # not just Inconclusive. This is the companion metric for the calibration
    # claim: Inconclusive rate tells you how often the oracle declined to
    # commit rather than issuing an unreliable verdict.
    resolvable = tp + fn + inc
    if resolvable > 0:
        inc_rate = inc / resolvable
        print()
        print(f"Inconclusive rate among non-skipped: {100 * inc_rate:.2f}%  ({inc}/{resolvable})")

    # ----- Per-family breakdown ---------------------------------------------
    print()
    print("-" * 70)
    print("PER-FAMILY BREAKDOWN")
    print("-" * 70)

    families: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "tp": 0, "fn": 0, "inc": 0, "skip": 0}
    )
    for r in results:
        fam = r.get("library", "") or r.get("test_name", "").split("::")[0]
        cls = _classify(r)
        families[fam]["total"] += 1
        families[fam][cls if cls != "inconclusive" else "inc"] += 1

    for fam in sorted(families):
        s = families[fam]
        tot = s["total"]
        defn = s["tp"] + s["fn"]
        print()
        print(f"{fam}:")
        print(f"  Trials: {tot}  |  TP: {s['tp']}  |  FN: {s['fn']}  |  Inc: {s['inc']}  |  Skip: {s['skip']}")
        if defn > 0:
            lo, hi = wilson_ci(s["tp"], defn)
            print(f"  TPR (conditional): {100 * s['tp'] / defn:.2f}%  "
                  f"(95% CI: [{100 * lo:.2f}%, {100 * hi:.2f}%])")

    # ----- Per-test breakdown -----------------------------------------------
    print()
    print("-" * 70)
    print("PER-TEST BREAKDOWN")
    print("-" * 70)

    by_test: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "tp": 0, "fn": 0, "inc": 0, "skip": 0}
    )
    for r in results:
        name = r["test_name"]
        by_test[name]["total"] += 1
        by_test[name][_classify(r) if _classify(r) != "inconclusive" else "inc"] += 1

    # Print in a fixed, predictable order: shortest name first, then lex.
    for name in sorted(by_test, key=lambda n: (len(n), n)):
        s = by_test[name]
        defn = s["tp"] + s["fn"]
        tpr_str = "--"
        if defn > 0:
            lo, hi = wilson_ci(s["tp"], defn)
            tpr_str = f"{100 * s['tp'] / defn:5.1f}%  [{100 * lo:4.1f}%, {100 * hi:5.1f}%]"
        print(f"  {name:<40s}  n={s['total']:3d}  TP={s['tp']:3d}  FN={s['fn']:3d}  Inc={s['inc']:3d}  Skip={s['skip']:3d}  TPR={tpr_str}")

    # ----- Detection curve for injected_shift_<N>ns(_k<K>)? -----------------
    # Keyed by (delay_ns, k). k defaults to 1 for un-amplified tests.
    shift_tests: dict[tuple[int, int], dict] = {}
    for name, s in by_test.items():
        m = _DELAY_RE.search(name)
        if m:
            delay = int(m.group(1))
            k = int(m.group(2)) if m.group(2) else 1
            shift_tests[(delay, k)] = s

    if shift_tests:
        any_amplified = any(k > 1 for (_, k) in shift_tests)
        print()
        print("-" * 70)
        if any_amplified:
            # Extended overlay table: sort by effective_ns so amplified points
            # sit next to the baseline single-op points at the same effective
            # delay. This is the overlay test for the scaling-law claim.
            print("INJECTED-SHIFT DETECTION CURVE (with amplification overlay)")
            print("-" * 70)
            print(
                f"{'delay_ns':>9s} {'k':>5s} {'eff_ns':>8s}  "
                f"{'n':>4s}  {'TP':>4s}  {'FN':>4s}  {'Inc':>4s}  "
                f"{'TPR':>8s}  {'95% CI':>20s}"
            )
            # Sort by (effective_ns, k) — baseline (k=1) sorts first within each tier.
            ordered = sorted(shift_tests.items(), key=lambda kv: (kv[0][0] * kv[0][1], kv[0][1]))
            for (delay, k), s in ordered:
                effective = delay * k
                defn = s["tp"] + s["fn"]
                if defn == 0:
                    tpr_str, ci_str = "--", "--"
                else:
                    lo, hi = wilson_ci(s["tp"], defn)
                    tpr_str = f"{100 * s['tp'] / defn:7.1f}%"
                    ci_str = f"[{100 * lo:5.1f}%, {100 * hi:5.1f}%]"
                print(
                    f"{delay:>9d} {k:>5d} {effective:>8d}  "
                    f"{s['total']:>4d}  {s['tp']:>4d}  {s['fn']:>4d}  {s['inc']:>4d}  "
                    f"{tpr_str:>8s}  {ci_str:>20s}"
                )
        else:
            print("INJECTED-SHIFT DETECTION CURVE")
            print("-" * 70)
            print(
                f"{'delay_ns':>10s}  {'n':>4s}  {'TP':>4s}  {'FN':>4s}  {'Inc':>4s}  "
                f"{'TPR':>8s}  {'95% CI':>20s}"
            )
            for (delay, _), s in sorted(shift_tests.items()):
                defn = s["tp"] + s["fn"]
                if defn == 0:
                    tpr_str, ci_str = "--", "--"
                else:
                    lo, hi = wilson_ci(s["tp"], defn)
                    tpr_str = f"{100 * s['tp'] / defn:7.1f}%"
                    ci_str = f"[{100 * lo:5.1f}%, {100 * hi:5.1f}%]"
                print(
                    f"{delay:>10d}  {s['total']:>4d}  {s['tp']:>4d}  "
                    f"{s['fn']:>4d}  {s['inc']:>4d}  {tpr_str:>8s}  {ci_str:>20s}"
                )

    # ----- List specific false negatives ------------------------------------
    if fn > 0:
        print()
        print("-" * 70)
        print("FALSE-NEGATIVE DETAILS (oracle said Pass on known-leaky code)")
        print("-" * 70)
        fns = [r for r in results if _classify(r) == "fn"]
        by_fn_test = defaultdict(list)
        for r in fns:
            by_fn_test[(r.get("library", ""), r["test_name"])].append(r)
        for (lib, test), rows in sorted(by_fn_test.items()):
            test_total = by_test[test]["total"]
            print(f"\n  {lib} / {test}")
            print(f"    Missed {len(rows)}/{test_total} iterations")
            for r in rows[:3]:
                print(f"      iter={r['iteration']}  P(leak)={r['leak_probability']}%  samples={r['samples']}  t={r['elapsed_sec']}s")
            if len(rows) > 3:
                print(f"      ... and {len(rows) - 3} more")

    # ----- Paper-ready summary ---------------------------------------------
    print()
    print("-" * 70)
    print("PAPER-READY SUMMARY")
    print("-" * 70)
    unique_tests = len(by_test)
    print(f"\n{unique_tests} known-leaky tests × average "
          f"{total / max(unique_tests, 1):.1f} iterations = {total} trials.")
    if definitive > 0:
        tpr = tp / definitive
        lo, hi = wilson_ci(tp, definitive)
        print(f"tacet detected leaks in {tp}/{definitive} = {100 * tpr:.2f}% of definitive verdicts")
        print(f"(Wilson 95% CI: [{100 * lo:.2f}%, {100 * hi:.2f}%]).")
    if resolvable > 0:
        print(f"Inconclusive rate: {100 * inc / resolvable:.2f}% ({inc}/{resolvable} non-skipped trials).")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: analyze_tpr.py <results.csv>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    analyze(load_results(path))


if __name__ == "__main__":
    main()
