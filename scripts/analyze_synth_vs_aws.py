#!/usr/bin/env python3
"""Compare statistical properties of synthetic AR(1) streams vs real AWS crypto timings.

Reads two directories of raw-sample CSVs (schema: `class,timing_ns`) and emits:
  - paper/author-response/synth_vs_aws.csv  (per-stream metrics, long form)
  - paper/author-response/synth_vs_aws.md   (aggregate summary table)

Metrics per stream:
  - lag-k autocorrelation (k = 1, 5, 10)
  - upper-tail CDF: P(z > 2), P(z > 3), P(z > 4), p99.9 quantile in σ units
  - Geyer initial-monotone-sequence IACT τ̂
  - Politis-White optimal block length (stationary + circular bootstrap)

Uses: numpy only. Algorithms cross-referenced against
`crates/tacet-core/src/statistics/{block_length,iact,autocorrelation}.rs`.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_raw_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (baseline, test) float arrays from a `class,timing_ns` CSV."""
    baseline: list[float] = []
    test: list[float] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header == ["class", "timing_ns"], f"bad header in {path}: {header}"
        for row in reader:
            klass, val = row[0], float(row[1])
            if klass == "baseline":
                baseline.append(val)
            elif klass == "test":
                test.append(val)
            else:
                raise ValueError(f"unknown class {klass} in {path}")
    return np.asarray(baseline, dtype=np.float64), np.asarray(test, dtype=np.float64)


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased ACF estimator ρ̂(k) for k = 0..max_lag, ρ̂(0) = 1."""
    x = x - x.mean()
    var = np.dot(x, x) / len(x)
    if var <= 0.0:
        return np.zeros(max_lag + 1)
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        num = np.dot(x[:-k], x[k:]) / len(x)
        out[k] = num / var
    return out


# ---------------------------------------------------------------------------
# Tail
# ---------------------------------------------------------------------------


def tail_profile(x: np.ndarray) -> dict[str, float]:
    """Upper-tail exceedance in *robust* (MAD) σ units.

    MAD is 50% efficient under Gaussian but resistant to outliers — timing
    streams on loaded hardware have occasional 100× spikes that completely
    distort the plain-std σ, giving misleadingly short tails. Using
    robust_sigma = 1.4826 × MAD(x) matches the standard convention.
    """
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 0.0:
        return {"p_gt_3sigma": 0.0, "p_gt_5sigma": 0.0, "p999_sigmas": 0.0, "robust_sigma": 0.0}
    z = (x - med) / robust_sigma
    return {
        "p_gt_3sigma": float((z > 3.0).mean()),
        "p_gt_5sigma": float((z > 5.0).mean()),
        "p999_sigmas": float(np.quantile(z, 0.999)),
        "robust_sigma": robust_sigma,
    }


# ---------------------------------------------------------------------------
# Geyer IMS IACT
# ---------------------------------------------------------------------------


def geyer_ims_iact(x: np.ndarray, max_lag: int | None = None) -> float:
    """Geyer (1992) initial-monotone-sequence IACT τ̂.

    Pair consecutive autocovariances γ(2k) + γ(2k+1) until the sum goes
    non-positive; then enforce monotone non-increasing on the accepted pairs.
    τ̂ = -1 + 2 * Σ (accepted pairs) / γ(0).

    Cross-reference: `tacet-core/src/statistics/iact.rs::geyer_ims_iact`.
    """
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 4, 2000)
    # raw autocovariance (biased)
    y = x - x.mean()
    gamma = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        gamma[k] = np.dot(y[: n - k], y[k:]) / n
    if gamma[0] <= 0.0:
        return 1.0

    # Pair sums Γ_k = γ(2k) + γ(2k+1).
    num_pairs = (max_lag - 1) // 2
    big_gamma = np.empty(num_pairs + 1)
    for k in range(num_pairs + 1):
        big_gamma[k] = gamma[2 * k] + gamma[2 * k + 1]

    # Initial positive: truncate at first non-positive pair.
    cutoff = num_pairs + 1
    for k in range(num_pairs + 1):
        if big_gamma[k] <= 0.0:
            cutoff = k
            break
    if cutoff == 0:
        return 1.0
    accepted = big_gamma[:cutoff].copy()
    # Initial monotone: enforce non-increasing.
    for k in range(1, cutoff):
        if accepted[k] > accepted[k - 1]:
            accepted[k] = accepted[k - 1]

    tau = -1.0 + 2.0 * accepted.sum() / gamma[0]
    return max(tau, 1.0)


# ---------------------------------------------------------------------------
# Politis-White block length
# ---------------------------------------------------------------------------


def _politis_white_mhat(rho: np.ndarray, n: int) -> int:
    """Find smallest m such that |ρ(m+j)| < 2√(log10 N / N) for j = 1..K_N.

    K_N = max(5, log10(N)), as in Politis-White (2004) §4.
    """
    K_N = max(5, int(math.log10(n) + 0.5))
    threshold = 2.0 * math.sqrt(math.log10(n) / n)
    max_k = len(rho) - 1 - K_N
    for m in range(1, max_k + 1):
        if all(abs(rho[m + j]) < threshold for j in range(1, K_N + 1)):
            return m
    # Fell through: use M as cap.
    return max_k if max_k > 0 else 1


def _flat_top_kernel(s: float) -> float:
    """Politis-Romano flat-top trapezoidal kernel."""
    t = abs(s)
    if t <= 0.5:
        return 1.0
    if t <= 1.0:
        return 2.0 * (1.0 - t)
    return 0.0


@dataclass
class PolitisWhiteResult:
    m_hat: int
    block_length_stationary: float
    block_length_circular: float


def politis_white(x: np.ndarray) -> PolitisWhiteResult:
    """Politis-White (2004) optimal block-length estimator.

    Cross-reference: `tacet-core/src/statistics/block_length.rs::optimal_block_length`.
    """
    n = len(x)
    max_lag = min(int(3 * math.sqrt(n)), n // 3)
    rho = autocorr(x, max_lag)
    m_hat = _politis_white_mhat(rho, n)
    M_hat = min(2 * m_hat, max_lag)

    # γ̂(0) from biased ACF (ρ(0) = 1), use sample variance as γ̂(0).
    y = x - x.mean()
    gamma0 = float(np.dot(y, y) / n)
    # γ̂(k) = ρ̂(k) * γ̂(0)
    gamma = rho[: M_hat + 1] * gamma0

    # G = Σ |k| λ(k/M̂) γ(k) over k = -M̂..M̂ (use symmetry → 2 × one-sided).
    G = 0.0
    # σ²(0) from kernel-smoothed sum: γ(0) + 2 Σ λ(k/M̂) γ(k)
    sigma2_0 = gamma0
    for k in range(1, M_hat + 1):
        w = _flat_top_kernel(k / M_hat)
        G += 2.0 * k * w * gamma[k]
        sigma2_0 += 2.0 * w * gamma[k]

    if sigma2_0 <= 0.0:
        return PolitisWhiteResult(m_hat=m_hat, block_length_stationary=1.0, block_length_circular=1.0)

    # Stationary bootstrap: D_SB = 2 σ²(0)²
    D_SB = 2.0 * sigma2_0 * sigma2_0
    # Circular bootstrap: D_CB = (4/3) σ²(0)²
    D_CB = (4.0 / 3.0) * sigma2_0 * sigma2_0

    b_SB = math.pow(2.0 * G * G / D_SB, 1.0 / 3.0) * math.pow(n, 1.0 / 3.0)
    b_CB = math.pow(2.0 * G * G / D_CB, 1.0 / 3.0) * math.pow(n, 1.0 / 3.0)
    return PolitisWhiteResult(m_hat=m_hat, block_length_stationary=b_SB, block_length_circular=b_CB)


# ---------------------------------------------------------------------------
# Per-stream pipeline
# ---------------------------------------------------------------------------


@dataclass
class StreamMetrics:
    source: str
    regime: str
    label: str
    class_name: str
    n: int
    n_unique: int
    quantization_limited: bool
    rho1: float
    rho5: float
    rho10: float
    p_gt_3sigma: float
    p_gt_5sigma: float
    p999_sigmas: float
    robust_sigma: float
    iact: float
    pw_mhat: int
    pw_block_stationary: float
    pw_block_circular: float


QUANTIZATION_UNIQUE_THRESHOLD = 100


def analyze_stream(
    x: np.ndarray, source: str, regime: str, label: str, class_name: str
) -> StreamMetrics:
    n_unique = int(np.unique(x).size)
    quantization_limited = n_unique < QUANTIZATION_UNIQUE_THRESHOLD
    rho = autocorr(x, max_lag=12)
    tail = tail_profile(x)
    pw = politis_white(x)
    return StreamMetrics(
        source=source,
        regime=regime,
        label=label,
        class_name=class_name,
        n=len(x),
        n_unique=n_unique,
        quantization_limited=quantization_limited,
        rho1=float(rho[1]),
        rho5=float(rho[5]),
        rho10=float(rho[10]),
        p_gt_3sigma=tail["p_gt_3sigma"],
        p_gt_5sigma=tail["p_gt_5sigma"],
        p999_sigmas=tail["p999_sigmas"],
        robust_sigma=tail["robust_sigma"],
        iact=geyer_ims_iact(x),
        pw_mhat=pw.m_hat,
        pw_block_stationary=pw.block_length_stationary,
        pw_block_circular=pw.block_length_circular,
    )


def walk_dir(directory: Path, source: str, regime: str) -> Iterator[StreamMetrics]:
    if not directory.exists():
        return
    for csv_path in sorted(directory.glob("*.csv")):
        baseline, test = load_raw_csv(csv_path)
        yield analyze_stream(baseline, source, regime, csv_path.stem, "baseline")
        yield analyze_stream(test, source, regime, csv_path.stem, "test")


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------


def summarise(
    rows: list[StreamMetrics], group_fn
) -> list[tuple[str, int, dict[str, tuple[float, float, float]]]]:
    """Group rows and report (median, q25, q75) per metric."""
    groups: dict[str, list[StreamMetrics]] = {}
    for r in rows:
        key = group_fn(r)
        if key:
            groups.setdefault(key, []).append(r)
    out = []
    metric_names = [
        "rho1",
        "rho5",
        "rho10",
        "p_gt_3sigma",
        "p_gt_5sigma",
        "p999_sigmas",
        "iact",
        "pw_mhat",
        "pw_block_stationary",
        "pw_block_circular",
    ]
    for key, items in sorted(groups.items()):
        stats = {}
        for m in metric_names:
            vals = np.asarray([getattr(r, m) for r in items], dtype=np.float64)
            stats[m] = (
                float(np.median(vals)),
                float(np.quantile(vals, 0.25)),
                float(np.quantile(vals, 0.75)),
            )
        out.append((key, len(items), stats))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=Path, default=Path("paper/author-response/synth-dump"))
    ap.add_argument("--idle-dir", type=Path, default=Path("paper/author-response/raw-aws/idle"))
    ap.add_argument("--loaded-dir", type=Path, default=Path("paper/author-response/raw-aws/loaded"))
    ap.add_argument("--out-csv", type=Path, default=Path("paper/author-response/synth_vs_aws.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("paper/author-response/synth_vs_aws.md"))
    args = ap.parse_args()

    rows: list[StreamMetrics] = []
    rows.extend(walk_dir(args.synth_dir, "synthetic", "N/A"))
    rows.extend(walk_dir(args.idle_dir, "aws-c8a", "idle"))
    rows.extend(walk_dir(args.loaded_dir, "aws-c8a", "loaded"))

    # Long-form CSV.
    fieldnames = [
        "source",
        "regime",
        "label",
        "class_name",
        "n",
        "n_unique",
        "quantization_limited",
        "rho1",
        "rho5",
        "rho10",
        "p_gt_3sigma",
        "p_gt_5sigma",
        "p999_sigmas",
        "robust_sigma",
        "iact",
        "pw_mhat",
        "pw_block_stationary",
        "pw_block_circular",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({f: getattr(r, f) for f in fieldnames})

    # Markdown summary.
    def synth_group(r: StreamMetrics) -> str:
        if r.source != "synthetic":
            return ""
        # label looks like: synth-null-rho0p0-seed0
        parts = r.label.split("-")
        pattern = parts[1]
        rho = parts[2].replace("rho", "").replace("p", ".")
        return f"synth {pattern} φ={rho}"

    # Group AWS by primitive so quantization-limited ones can be flagged separately.
    def aws_primitive(r: StreamMetrics) -> str:
        # label: "<lib>-<primitive>-iter###", e.g. "dalek-x25519-scalar_mult-iter001"
        parts = r.label.split("-")
        # Drop trailing "iter###"
        if parts and parts[-1].startswith("iter"):
            parts = parts[:-1]
        return "-".join(parts)

    def aws_group(r: StreamMetrics) -> str:
        if r.source != "aws-c8a":
            return ""
        qflag = " [quant]" if r.quantization_limited else ""
        return f"aws-c8a {r.regime} {aws_primitive(r)}{qflag}"

    def aws_aggregate_group(r: StreamMetrics) -> str:
        if r.source != "aws-c8a" or r.quantization_limited:
            return ""
        return f"aws-c8a {r.regime} (continuous streams)"

    synth_summary = summarise([r for r in rows if r.source == "synthetic"], synth_group)
    aws_per_primitive = summarise([r for r in rows if r.source == "aws-c8a"], aws_group)
    aws_aggregate = summarise([r for r in rows if r.source == "aws-c8a"], aws_aggregate_group)

    def fmt_stats(stats: dict) -> dict[str, str]:
        def f(m: str, prec: int = 3) -> str:
            med, q25, q75 = stats[m]
            return f"{med:.{prec}f}"

        return {
            "rho1": f("rho1"),
            "rho5": f("rho5"),
            "rho10": f("rho10"),
            "p_gt_3sigma": f("p_gt_3sigma", 4),
            "p999": f("p999_sigmas", 2),
            "iact": f("iact", 2),
            "pw_sb": f("pw_block_stationary", 1),
        }

    def write_table(target: list[str], title: str, summary) -> None:
        target.append(f"\n### {title}\n")
        target.append("| Group | n | ρ₁ | ρ₅ | ρ₁₀ | P(z>3σ)_robust | p99.9 (σ_MAD) | IACT τ̂ | PW block (SB) |")
        target.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key, count, stats in summary:
            s = fmt_stats(stats)
            target.append(
                f"| {key} | {count} | {s['rho1']} | {s['rho5']} | {s['rho10']} | {s['p_gt_3sigma']} | {s['p999']} | {s['iact']} | {s['pw_sb']} |"
            )

    lines: list[str] = []
    lines.append("# Synthetic vs. real-AWS timing characterisation\n")
    lines.append(
        "Raw streams: synthetic AR(1) × LogNormal from `crates/tacet-bench/src/synthetic.rs` "
        "and real crypto timings collected on `c8a.4xlarge` (16-vCPU AMD EPYC 9R45, rdtsc "
        "resolution 0.385 ns). Row counts = baseline + test × primitives × iterations × seeds. "
        "Reported statistic is the median across rows in each group. Tail reported in *robust* "
        "σ units (MAD × 1.4826) to prevent outlier inflation. "
        f"Streams with <{QUANTIZATION_UNIQUE_THRESHOLD} unique timing values are flagged `[quant]` "
        "and excluded from the aggregate (these are sub-100-cycle ops at the rdtsc resolution floor — "
        "the paper's `Unmeasurable` category).\n"
    )
    write_table(lines, "Synthetic (nominal AR(1) coefficient φ)", synth_summary)
    write_table(lines, "AWS c8a by primitive", aws_per_primitive)
    write_table(lines, "AWS c8a aggregate (continuous streams only)", aws_aggregate)

    lines.append("\n## Sanity checks\n")
    lines.append("- AR(1) ρ=0.6 theoretical on the underlying noise: ρ₁=0.6, ρ₅≈0.078, IACT≈(1+ρ)/(1−ρ)=4.0.")
    lines.append("- AR(1) ρ=0.8 theoretical on the underlying noise: ρ₁=0.8, ρ₅≈0.328, IACT=9.0.")
    lines.append("- The synthetic generator adds AR(1) noise *multiplicatively in log-space* with a scale factor of 0.10, so measured ρ₁ on the final stream is diluted relative to nominal φ. This is expected — what matters is whether measured AWS ρ₁ lands inside the *measured* synthetic range, not the nominal one.")
    lines.append("- IID theoretical: ρ₁≈0, IACT≈1, Gaussian p99.9≈3.09σ.")
    lines.append("")
    args.out_md.write_text("\n".join(lines))
    print(f"wrote {args.out_csv} ({len(rows)} rows) and {args.out_md}")


if __name__ == "__main__":
    main()
