#!/usr/bin/env python3
"""Aggregate the hyperparameter-sensitivity ablation for the USENIX Sec'26 rebuttal.

Reads each config subdirectory produced by aws-ablation-sweep.sh, computes
per-config FPR, detection rates, and Inconclusive rates with Wilson 95% CIs
for BOTH attacker models (AdjacentNetwork θ=100 ns, SharedHardware θ≈0.4 ns),
and writes `ablation_summary.md` plus `ablation_summary.csv`.

Usage:
    uv run scripts/analyze_ablation.py <OUTPUT_ROOT>
    # default OUTPUT_ROOT = ~/bench-results/ablation

Expected directory layout:
    <OUTPUT_ROOT>/<label>/benchmark_results.csv  for label in CONFIG_ORDER.

Filtering (from sweep.rs::iter_configs):
- `medium` preset iterates BOTH AttackerModels and both Effect patterns. We split
  the aggregate per `attacker_threshold_ns` and produce one column block per
  attacker model.
- `medium` preset iterates `synthetic_sigma_ns_values = [2,5,10,20,50]` but ONLY
  at effect=0 + IID (Heatmap-3 condition). All other rows sit at the default
  sigma=5 ns. We filter to `synthetic_sigma_ns == 5.0` to keep null-IID
  denominators balanced with null-AR(1) across the sweep.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# The benchmark_results.csv schema produced by tacet-bench (sweep.rs:611-643):
#   tool, preset, effect_pattern, effect_sigma_mult, noise_model,
#   synthetic_sigma_ns, attacker_threshold_ns, dataset_id, samples_per_class,
#   detected, statistic, p_value, time_ms, samples_used, status, outcome
#
# For tacet, `statistic` holds the posterior P(leak > θ).
# `outcome` ∈ {pass, fail, inconclusive, unmeasurable}.


CONFIG_ORDER = [
    "baseline",
    "pi0_low",
    "pi0_high",
    "pi0_extreme",
    "alpha_tight",
    "alpha_loose",
    "kl_loose",
    "kl_strict",
    "nu_low",
    "nu_high",
    "nu_ell_extreme",
    "nu_ell_cauchy",
    "nu_prior_low",
    "nu_prior_high",
    "combo_stress_heavy",
    "combo_stress_light",
]

# Human-readable label + group + value for each config.
CONFIG_META: dict[str, tuple[str, str, str]] = {
    "baseline":           ("Baseline (defaults)",       "—",          "π₀=0.62 α=.05 β=.95 kl=0.7 ν_ℓ=8 ν=4"),
    "pi0_low":            ("π₀ = 0.50",                 "π₀",         "0.50"),
    "pi0_high":           ("π₀ = 0.75",                 "π₀",         "0.75"),
    "pi0_extreme":        ("π₀ = 0.85 (stress)",        "π₀",         "0.85"),
    "alpha_tight":        ("α/1-β = 0.01/0.99",         "α / 1-β",    "0.01 / 0.99"),
    "alpha_loose":        ("α/1-β = 0.10/0.90",         "α / 1-β",    "0.10 / 0.90"),
    "kl_loose":           ("kl_min = 0.3",              "kl_min",     "0.3"),
    "kl_strict":          ("kl_min = 1.5",              "kl_min",     "1.5"),
    "nu_low":             ("ν_ℓ = 4",                   "ν_ℓ",        "4"),
    "nu_high":            ("ν_ℓ = 16",                  "ν_ℓ",        "16"),
    "nu_ell_extreme":     ("ν_ℓ = 2.5 (stress)",        "ν_ℓ",        "2.5"),
    "nu_ell_cauchy":      ("ν_ℓ = 2.01 (near-Cauchy)",  "ν_ℓ",        "2.01"),
    "nu_prior_low":       ("ν (prior) = 2.5",           "ν (prior)",  "2.5"),
    "nu_prior_high":      ("ν (prior) = 16",            "ν (prior)",  "16"),
    "combo_stress_heavy": ("Combo stress (aggressive)", "combo",      "π₀=.85 α=.01 kl=1.5 ν_ℓ=2.5 ν=2.5"),
    "combo_stress_light": ("Combo stress (conservative)", "combo",    "π₀=.50 α=.10 kl=0.3 ν_ℓ=16 ν=16"),
}

# AttackerModel thresholds (sweep.rs exports these verbatim; float equality is
# safe because no arithmetic mutates them between write and read).
ATTACKER_THRESHOLDS: dict[str, float] = {
    "AdjacentNetwork": 100.0,
    "SharedHardware": 0.4,
}

# Canonical synthetic noise σ for reporting; the sigma sweep at null+IID is
# reserved for Heatmap-3 and is not part of the ablation story.
DEFAULT_SIGMA_NS: float = 5.0


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    n = float(total)
    p = successes / n
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z / (1 + z2 / n)) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class PerAttackerSummary:
    """Metrics for one (config, attacker_model) cell."""
    n_null: int
    n_fail_null: int
    n_incon_null: int
    n_pos_1sig_shift: int
    n_detect_1sig_shift: int
    n_pos_1sig_tail: int
    n_detect_1sig_tail: int
    n_pos_2sig_shift: int
    n_detect_2sig_shift: int

    def fpr(self) -> tuple[float, float, float]:
        p = self.n_fail_null / self.n_null if self.n_null else 0.0
        lo, hi = wilson_ci(self.n_fail_null, self.n_null)
        return p, lo, hi

    def incon_null(self) -> tuple[float, float, float]:
        p = self.n_incon_null / self.n_null if self.n_null else 0.0
        lo, hi = wilson_ci(self.n_incon_null, self.n_null)
        return p, lo, hi

    def detect_1sig_shift(self) -> tuple[float, float, float]:
        p = self.n_detect_1sig_shift / self.n_pos_1sig_shift if self.n_pos_1sig_shift else 0.0
        lo, hi = wilson_ci(self.n_detect_1sig_shift, self.n_pos_1sig_shift)
        return p, lo, hi

    def detect_1sig_tail(self) -> tuple[float, float, float]:
        p = self.n_detect_1sig_tail / self.n_pos_1sig_tail if self.n_pos_1sig_tail else 0.0
        lo, hi = wilson_ci(self.n_detect_1sig_tail, self.n_pos_1sig_tail)
        return p, lo, hi

    def detect_2sig_shift(self) -> tuple[float, float, float]:
        p = self.n_detect_2sig_shift / self.n_pos_2sig_shift if self.n_pos_2sig_shift else 0.0
        lo, hi = wilson_ci(self.n_detect_2sig_shift, self.n_pos_2sig_shift)
        return p, lo, hi


@dataclass
class ConfigSummary:
    label: str
    group: str
    value: str
    attackers: dict[str, PerAttackerSummary]  # keyed by attacker name


def summarize_attacker(df: pd.DataFrame) -> PerAttackerSummary:
    """Compute headline rates for one (config, attacker) slice."""
    null_rows = df[df["effect_sigma_mult"] == 0.0]
    pos_1sig_shift = df[(df["effect_sigma_mult"] == 1.0) & (df["effect_pattern"] == "shift")]
    pos_1sig_tail = df[(df["effect_sigma_mult"] == 1.0) & (df["effect_pattern"] == "tail")]
    pos_2sig_shift = df[(df["effect_sigma_mult"] == 2.0) & (df["effect_pattern"] == "shift")]

    return PerAttackerSummary(
        n_null=len(null_rows),
        n_fail_null=int((null_rows["outcome"] == "fail").sum()),
        n_incon_null=int((null_rows["outcome"] == "inconclusive").sum()),
        n_pos_1sig_shift=len(pos_1sig_shift),
        n_detect_1sig_shift=int((pos_1sig_shift["outcome"] == "fail").sum()),
        n_pos_1sig_tail=len(pos_1sig_tail),
        n_detect_1sig_tail=int((pos_1sig_tail["outcome"] == "fail").sum()),
        n_pos_2sig_shift=len(pos_2sig_shift),
        n_detect_2sig_shift=int((pos_2sig_shift["outcome"] == "fail").sum()),
    )


def summarize_config(label: str, df: pd.DataFrame) -> ConfigSummary:
    """Split CSV by attacker_threshold_ns and summarize each attacker cell."""
    meta_label, group, value = CONFIG_META[label]
    # Keep only rows at the canonical default sigma.
    df = df[df["synthetic_sigma_ns"] == DEFAULT_SIGMA_NS]

    attackers: dict[str, PerAttackerSummary] = {}
    for name, threshold in ATTACKER_THRESHOLDS.items():
        slice_df = df[df["attacker_threshold_ns"] == threshold]
        attackers[name] = summarize_attacker(slice_df)

    return ConfigSummary(label=meta_label, group=group, value=value, attackers=attackers)


def fmt_rate(rate: tuple[float, float, float]) -> str:
    p, lo, hi = rate
    return f"{p*100:5.2f}% [{lo*100:4.1f}, {hi*100:5.1f}]"


def render_markdown(summaries: list[ConfigSummary]) -> str:
    """Render the rebuttal-ready table, one section per attacker."""
    header = (
        "# Hyperparameter sensitivity ablation\n\n"
        "Medium-grid sweep (6 effect sizes × 2 patterns × 4 AR(1) noise × 30 datasets "
        f"at σ={DEFAULT_SIGMA_NS:g} ns, 9 configs). Rates are percentages with "
        "Wilson 95% CIs.\n\n"
    )

    sections = []
    for attacker_name in ATTACKER_THRESHOLDS:
        threshold = ATTACKER_THRESHOLDS[attacker_name]
        sec = [
            f"## {attacker_name} (θ = {threshold:g} ns)\n",
            (
                "| Config | FPR (null) | Detect @1σ Shift | Detect @1σ Tail | "
                "Detect @2σ Shift | Incon @ null |\n"
                "|--------|-----------|------------------|-----------------|"
                "------------------|--------------|"
            ),
        ]
        for s in summaries:
            a = s.attackers[attacker_name]
            sec.append(
                f"| {s.label} | {fmt_rate(a.fpr())} | "
                f"{fmt_rate(a.detect_1sig_shift())} | "
                f"{fmt_rate(a.detect_1sig_tail())} | "
                f"{fmt_rate(a.detect_2sig_shift())} | "
                f"{fmt_rate(a.incon_null())} |"
            )
        sections.append("\n".join(sec))

    # Caption-ready summary for the rebuttal prose.
    summary_lines = ["\n## Summary (rebuttal-ready)\n"]
    for attacker_name in ATTACKER_THRESHOLDS:
        fprs = [s.attackers[attacker_name].fpr()[0] for s in summaries]
        d1_shift = [s.attackers[attacker_name].detect_1sig_shift()[0] for s in summaries]
        d2_shift = [s.attackers[attacker_name].detect_2sig_shift()[0] for s in summaries]
        d1_tail = [s.attackers[attacker_name].detect_1sig_tail()[0] for s in summaries]
        incons = [s.attackers[attacker_name].incon_null()[0] for s in summaries]
        if not fprs:
            continue
        summary_lines.append(
            f"- **{attacker_name}**: FPR ∈ [{min(fprs)*100:.2f}%, {max(fprs)*100:.2f}%]; "
            f"Detect@1σ-Shift ∈ [{min(d1_shift)*100:.1f}%, {max(d1_shift)*100:.1f}%]; "
            f"Detect@1σ-Tail ∈ [{min(d1_tail)*100:.1f}%, {max(d1_tail)*100:.1f}%]; "
            f"Detect@2σ-Shift ∈ [{min(d2_shift)*100:.1f}%, {max(d2_shift)*100:.1f}%]; "
            f"Incon@null ∈ [{min(incons)*100:.1f}%, {max(incons)*100:.1f}%] "
            f"across {len(summaries)} configs."
        )

    footer_lines = [f"\n\nTrials per config (sigma filtered to {DEFAULT_SIGMA_NS:g} ns):"]
    for attacker_name in ATTACKER_THRESHOLDS:
        a = summaries[0].attackers[attacker_name]
        footer_lines.append(
            f"  - {attacker_name}: null={a.n_null}, "
            f"@1σ-Shift={a.n_pos_1sig_shift}, @1σ-Tail={a.n_pos_1sig_tail}, "
            f"@2σ-Shift={a.n_pos_2sig_shift}"
        )
    footer = "\n".join(footer_lines) + "\n"
    return header + "\n\n".join(sections) + "\n" + "\n".join(summary_lines) + footer


def render_csv(summaries: list[ConfigSummary]) -> pd.DataFrame:
    """Flat CSV for further analysis — one row per (config, attacker)."""
    rows = []
    for s in summaries:
        for attacker_name, a in s.attackers.items():
            fpr_p, fpr_lo, fpr_hi = a.fpr()
            incon_p, incon_lo, incon_hi = a.incon_null()
            d1s_p, d1s_lo, d1s_hi = a.detect_1sig_shift()
            d1t_p, d1t_lo, d1t_hi = a.detect_1sig_tail()
            d2s_p, d2s_lo, d2s_hi = a.detect_2sig_shift()
            rows.append({
                "config": s.label,
                "group": s.group,
                "value": s.value,
                "attacker": attacker_name,
                "attacker_threshold_ns": ATTACKER_THRESHOLDS[attacker_name],
                "n_null": a.n_null,
                "fpr": fpr_p,
                "fpr_ci_lo": fpr_lo,
                "fpr_ci_hi": fpr_hi,
                "incon_null": incon_p,
                "incon_null_ci_lo": incon_lo,
                "incon_null_ci_hi": incon_hi,
                "n_pos_1sig_shift": a.n_pos_1sig_shift,
                "detect_1sig_shift": d1s_p,
                "detect_1sig_shift_ci_lo": d1s_lo,
                "detect_1sig_shift_ci_hi": d1s_hi,
                "n_pos_1sig_tail": a.n_pos_1sig_tail,
                "detect_1sig_tail": d1t_p,
                "detect_1sig_tail_ci_lo": d1t_lo,
                "detect_1sig_tail_ci_hi": d1t_hi,
                "n_pos_2sig_shift": a.n_pos_2sig_shift,
                "detect_2sig_shift": d2s_p,
                "detect_2sig_shift_ci_lo": d2s_lo,
                "detect_2sig_shift_ci_hi": d2s_hi,
            })
    return pd.DataFrame(rows)


def load_config(root: Path, label: str) -> pd.DataFrame | None:
    csv = root / label / "benchmark_results.csv"
    if not csv.exists():
        print(f"[warn] missing: {csv}", file=sys.stderr)
        return None
    df = pd.read_csv(csv)
    df["effect_sigma_mult"] = df["effect_sigma_mult"].astype(float)
    df["synthetic_sigma_ns"] = df["synthetic_sigma_ns"].astype(float)
    df["attacker_threshold_ns"] = df["attacker_threshold_ns"].astype(float)
    df["outcome"] = df["outcome"].str.strip().str.lower()
    df["effect_pattern"] = df["effect_pattern"].str.strip().str.lower()
    return df


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "bench-results" / "ablation")
    if not root.exists():
        print(f"[error] output root does not exist: {root}", file=sys.stderr)
        return 1

    summaries: list[ConfigSummary] = []
    for label in CONFIG_ORDER:
        df = load_config(root, label)
        if df is None:
            continue
        summaries.append(summarize_config(label, df))

    if not summaries:
        print("[error] no configs loaded", file=sys.stderr)
        return 1

    md = render_markdown(summaries)
    csv = render_csv(summaries)

    md_path = root / "ablation_summary.md"
    csv_path = root / "ablation_summary.csv"
    md_path.write_text(md)
    csv.to_csv(csv_path, index=False)

    print(md)
    print(f"\n[wrote] {md_path}")
    print(f"[wrote] {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
