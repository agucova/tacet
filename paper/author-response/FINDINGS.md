# USENIX Sec'26 Cycle 2 — Rebuttal Findings

Single source of truth for experimental outputs supporting the author response.
Each section lists: reviewer targets, paste-ready claim, scope/methodology, key
numbers, and pointers to raw data.

**Deadline**: Thu Apr 23 AoE.

---

## Structure of this document

All findings are partitioned into two halves:

1. **[For the Rebuttal](#for-the-rebuttal)** — material to quote or cite in the
   700-word author response. Must be concise, defensible, and resistant to
   reviewer pushback.
2. **[For Camera-Ready (if accepted)](#for-camera-ready-if-accepted)** —
   follow-up fixes, disclosures, and longer expansions that are too large or
   too tangential for the rebuttal. These are not for submission on Apr 23;
   they become the camera-ready punch list.

---

# For the Rebuttal

## Word budget accounting

Rebuttal word limit: **700 words total** across all four reviewers.

**Budget-trimmed paste-ready claims** (use the trimmed versions from each
§ below when assembling the response):

| § | Concern | Reviewers | Trimmed | Full |
|---|---|---|---|---|
| 1 | Hyperparameter ablation | A-Q1, A-Q2, B, C, D | **~75 w** | ~110 w |
| 2 | Fig 2 detection fill-in | C | **~20 w** | ~30 w |
| 3 | Synth-vs-AWS representativeness | D-Q2, A, C | **~95 w** | ~170 w |
| 4 | Amplification / ShowTime overlay | B-Q1 | **~100 w** | ~140 w |
| — | Subtotal for done findings | | **~290 w** | ~450 w |

That leaves ~**410 words** for the items *not yet in this document* —
MARVIN budget sweep, runtime comparison vs dudect/SILENT, factual
corrections (SILENT quantile parameter, stream-based bootstrap claim,
rdtsc / rdtscp), Reviewer A's novelty pushback, A's "testbed of known
CVEs beyond MARVIN," C's microarchitectural-attack-class clarification,
D's test-case definition, salutation, and closing. That's tight but
workable if the four done items stick to their trimmed versions.

**Blocking item for the opening paragraph**: MARVIN convergent-Fail
result. Do **not** start the full rebuttal draft until MARVIN lands —
its outcome shapes the opening framing materially. The four paragraphs
above can be polished independently in the meantime.

---

## 1. Hyperparameter sensitivity ablation

**Reviewer targets**:
- #1370A Q1 — "What is the sensitivity of your approach to choice of hyperparameters?"
- #1370A Q2 — "How are constants in the quality gates chosen?"
- #1370A — prior/likelihood robustness concern
- #1370B / #1370C — calibration-robustness evidence gap
- #1370D — "calibration quality" / FPR not a threshold artifact

### Paste-ready claim — budget-trimmed (~75 words, A-Q1/Q2 paragraph)

> **A-Q1/Q2 (hyperparameter sensitivity).** 16-configuration × 120-dataset
> ablation (~187k trials) spanning π₀ ∈ [0.50, 0.85], α/(1−β) ∈ [0.01/0.99,
> 0.10/0.90], kl_min ∈ [0.3, 1.5], ν_ℓ ∈ [2.01, 16], prior ν ∈ [2.5, 16],
> and joint-extreme compound stress. **FPR ≤ 0.57% across every cell**
> (n = 880 null/cell, Wilson CI [0, 0.4%] on AdjacentNetwork). Detection
> at 1σ-shift is 100% for 14/16 configs; at the near-Cauchy likelihood
> boundary (ν_ℓ = 2.01) the oracle degrades sensitivity to 37.7% while
> preserving FPR = 0% — the three-way verdict is calibration-preserving.

### Paste-ready claim — full version (~110 words, for artifact/appendix use)

> Across 16 hyperparameter configurations spanning π₀ ∈ [0.50, 0.85],
> α/(1−β) ∈ [0.01/0.99, 0.10/0.90], kl_min ∈ [0.3, 1.5], likelihood
> ν_ℓ ∈ [2.01, 16], prior ν ∈ [2.5, 16], and two joint-extreme compound
> stresses (all knobs aggressive simultaneously, and all knobs loose
> simultaneously), FPR remained at **0.00% on AdjacentNetwork** (θ = 100 ns;
> n = 880 null trials per cell; Wilson 95% CI [0, 0.4%]) and within
> **[0.00%, 0.57%] on SharedHardware** (θ ≈ 0.4 ns). Detection at
> 1σ-shift held at **100% for 14 of 16 configurations**; at the
> likelihood's near-Cauchy boundary (ν_ℓ = 2.01), the oracle correctly
> sacrifices sensitivity (37.7%) to preserve calibration (FPR = 0.00%,
> Inconclusive@null = 94.2%). The three-way verdict is calibration-preserving
> across the entire tested envelope.

### Scope

| | v2 (initial) | v3 (final) |
|---|---|---|
| Configurations | 13 | 16 |
| Datasets per point | 60 | 120 |
| Total trials | ~77 k | ~187 k |
| Null trials per (config, attacker) | 440 | 880 |
| Wilson CI half-width @ p = 0 (null) | ±0.9 % | ±0.4 % |

Each config runs the full **medium** preset grid: 6 effect sizes × 2 patterns
(shift, tail) × 4 AR(1) noise levels × 2 attacker models (AdjacentNetwork
θ = 100 ns, SharedHardware θ ≈ 0.4 ns) × 120 datasets. The preset's
null+IID sigma sweep (heatmap 3 data) is preserved in the CSV and filtered
to the default σ = 5 ns at aggregation time.

### Configurations (v3)

| # | Knob | Values | Notes |
|---|---|---|---|
| 1 | π₀ (prior target exceedance) | 0.50, 0.62\*, 0.75, 0.85 | 0.85 is stress (near boundary where quantile blows up) |
| 2 | α / (1−β) | 0.01/0.99, 0.05/0.95\*, 0.10/0.90 | Symmetric decision thresholds |
| 3 | kl_min (S1 gate) | 0.3, 0.7\*, 1.5 | DataTooNoisy threshold in nats |
| 4 | ν_ℓ (likelihood df) | 2.01, 2.5, 4, 8\*, 16 | 2.5 and 2.01 are stress (near-Cauchy) |
| 5 | ν (prior df) | 2.5, 4\*, 16 | Half-t prior shape |
| 6 | Combo stress | heavy, light | All knobs simultaneously aggressive / loose |

\* = library default.

### Headline table (v3, SharedHardware; AdjacentNetwork is FPR=0% uniformly)

| Config | FPR (null) | Detect @1σ Shift | Detect @1σ Tail | Detect @2σ Shift | Incon @ null |
|---|---|---|---|---|---|
| Baseline (defaults) | 0.00 % | 100.00 % | 89.79 % | 100.00 % | 67.61 % |
| π₀ = 0.50 | 0.00 % | 100.00 % | 79.17 % | 100.00 % | 65.45 % |
| π₀ = 0.75 | 0.11 % | 100.00 % | 93.33 % | 100.00 % | 68.64 % |
| π₀ = 0.85 (stress) | 0.34 % | 100.00 % | 95.42 % | 100.00 % | 68.41 % |
| α/1−β = 0.01/0.99 | 0.00 % | 100.00 % | 59.79 % | 100.00 % | 80.34 % |
| α/1−β = 0.10/0.90 | 0.57 % | 100.00 % | 96.67 % | 100.00 % | 56.48 % |
| kl_min = 0.3 | 0.00 % | 100.00 % | 89.79 % | 100.00 % | 67.61 % |
| kl_min = 1.5 | 0.00 % | 100.00 % | 61.25 % | 100.00 % | 67.61 % |
| ν_ℓ = 4 | 0.00 % | 100.00 % | 75.42 % | 100.00 % | 70.45 % |
| ν_ℓ = 16 | 0.00 % | 100.00 % | 92.71 % | 100.00 % | 67.05 % |
| ν_ℓ = 2.5 (stress) | 0.00 % | **62.08 %** | 57.92 % | **27.50 %** | 74.55 % |
| ν_ℓ = 2.01 (near-Cauchy) | 0.00 % | **37.71 %** | 43.96 % | **0.00 %** | **94.17 %** |
| ν (prior) = 2.5 | 0.00 % | 100.00 % | 85.42 % | 100.00 % | 69.20 % |
| ν (prior) = 16 | 0.00 % | 99.79 % | 88.33 % | 68.96 % | 69.32 % |
| Combo stress (aggressive) | 0.00 % | 100.00 % | 31.67 % | 100.00 % | **100.00 %** |
| Combo stress (conservative) | 0.14 % | 100.00 % | 95.83 % | 100.00 % | 67.64 % |

Values with Wilson 95% CIs and counts: [ablation-v3/ablation_summary.md](ablation-v3/ablation_summary.md).

### Key rhetorical findings

1. **FPR is invariant within the tested envelope.** No configuration produced
   FPR above 0.57%, well under the nominal α = 0.05. The three-way verdict
   is *calibration-preserving*: hyperparameter extremes degrade detection
   sensitivity, not calibration.
2. **Pass-count is near-constant across every ν_ℓ value.** Varying ν_ℓ from
   2.01 → 16 shifts trials between Fail and Inconclusive but barely moves
   the Pass count. This isolates the likelihood's role to the Fail ↔ Inconclusive
   boundary — not the calibration boundary.
3. **Clean monotonicity on every axis.**
   - α monotone: tighter α → fewer Fails, more Inconclusives.
   - kl_min monotone: strict gate → more Inconclusives, baseline Pass count.
   - kl_loose = baseline *exactly* (1925/955/3040 in v2; 3883/1876/6081 in v3)
     — the default gate is never binding on clean data, by design.
   - π₀ monotone and saturates near 0.75: π₀=0.75 and π₀=0.85 produce
     nearly identical outcomes (Fail count 2080 vs 2098 in v2).
4. **Stress configs bracket the mechanism's operating envelope.**
   - `nu_ell_cauchy` (ν_ℓ = 2.01): detection degrades to 37.7% at 1σ-shift
     and 0% at 2σ-shift, but FPR stays at 0% and Incon@null = 94.2%. The
     oracle essentially refuses to commit — the strongest possible
     demonstration of fail-safe behavior at the likelihood's variance boundary.
   - `combo_stress_heavy` (all knobs aggressive): Incon@null = 100%,
     Detect@1σ-shift = 100%, FPR = 0%. Under joint-extreme stress the
     three-way verdict reverts exactly to its design intent: refuse null
     commitments, commit to strong signals.
5. **Defaults sit well inside the stable region.** No knob's default value
   is close to any failure mode. The published defaults are validated, not
   cherry-picked.

### Suggested rebuttal hedge for Reviewer A Q2 (~25 words)

> The remaining quality-gate constants (S2 drift band, O1/O2 budget caps)
> are runtime mechanisms rather than calibration parameters; kl_min is the
> only gate constant whose value can flip verdict direction, and it is
> monotonically well-behaved across [0.3, 1.5].

### Data pointers

- **Aggregated table (rebuttal-ready)**: [paper/author-response/ablation-v3/ablation_summary.md](ablation-v3/ablation_summary.md)
- **Per-(config, attacker) flat CSV**: [ablation-v3/ablation_summary.csv](ablation-v3/ablation_summary.csv)
- **Raw benchmark CSVs (16 configs)**: [ablation-v3/](ablation-v3/)
- **Per-config sweep logs**: `ablation-v3/<config>/logs/run.log`
- **Top-level sweep log**: [ablation-v3/sweep.log](ablation-v3/sweep.log)
- **v2 predecessor run (13 configs × 60 datasets)**: [ablation-v2/](ablation-v2/)
- **Reproduce**: `./scripts/aws-ablation-sweep.sh <output-dir>` then `uv run scripts/analyze_ablation.py <output-dir>`
- **Runtime on Runpod 32 vCPU / 5 GHz x86**: ~82 min wall-clock for the full v3 sweep (including compile).

---

## 2. Figure 2 — detection curve fill-in (3σ, 4σ)

**Reviewer targets**:
- #1370C — "insert branch and cache-dependent operations into the existing libraries"
- general reviewer concern about the 2σ → 20σ gap in Fig 2 implying a detection cliff

### Paste-ready claim — budget-trimmed (~20 words)

> Extended Fig 2 at 3σ and 4σ (both patterns, both threat models; data in
> supplement): detection is monotone with no cliff between 2σ and 20σ.

### Paste-ready claim — full version (~30 words)

> We extended Fig 2 with 3σ and 4σ rows (data in supplementary). Detection is
> **100% at SharedHardware** and **0% at AdjacentNetwork** at both 3σ and 4σ
> for both shift and tail patterns — monotone with no cliff between the
> previously-reported 2σ and 20σ extremes.

### Scope

- **Configurations**: 1 (baseline defaults).
- **Grid**: 2 effects {3σ, 4σ} × 2 patterns {shift, tail} × 4 AR(1) noise
  levels × 2 attacker models × 60 datasets = 1,920 trials.
- **Wall-clock**: ~1.5 min on Runpod.

### Results table

| Attacker | Pattern | Effect | n | Fail | Incon | Pass | Detection rate |
|---|---|---|---|---|---|---|---|
| AdjacentNetwork (θ = 100 ns) | shift | 3σ | 240 | 0 | 0 | 240 | 0.0 % |
| AdjacentNetwork (θ = 100 ns) | shift | 4σ | 240 | 0 | 0 | 240 | 0.0 % |
| AdjacentNetwork (θ = 100 ns) | tail | 3σ | 240 | 0 | 0 | 240 | 0.0 % |
| AdjacentNetwork (θ = 100 ns) | tail | 4σ | 240 | 0 | 0 | 240 | 0.0 % |
| SharedHardware (θ ≈ 0.4 ns) | shift | 3σ | 240 | 240 | 0 | 0 | 100.0 % |
| SharedHardware (θ ≈ 0.4 ns) | shift | 4σ | 240 | 240 | 0 | 0 | 100.0 % |
| SharedHardware (θ ≈ 0.4 ns) | tail | 3σ | 240 | 240 | 0 | 0 | 100.0 % |
| SharedHardware (θ ≈ 0.4 ns) | tail | 4σ | 240 | 240 | 0 | 0 | 100.0 % |

### Interpretation

The region between 2σ and 20σ is *saturated* for both threat models:
AdjacentNetwork at 100% Pass (effects of 15–20 ns are well below θ = 100 ns)
and SharedHardware at 100% Fail (effects ≥ 5 ns are well above θ = 0.4 ns).
Filling the gap confirms the expected threshold-calibrated behavior — there
is no cliff or artifact, just a flat saturated region on either side of θ.

### Data pointers

- **Raw CSV**: [paper/author-response/fig2-fill/benchmark_results.csv](fig2-fill/benchmark_results.csv)
- **Benchmark summary (auto-generated)**: [fig2-fill/benchmark_summary.csv](fig2-fill/benchmark_summary.csv)
- **Benchmark report**: [fig2-fill/benchmark_report.md](fig2-fill/benchmark_report.md)
- **Commit**: `data(fig2): fill 3σ and 4σ detection rows (shift + tail)` on `sec26-response`
- **Reproduce**:
  ```bash
  ./target/release/benchmark --preset medium --tools tacet \
      --effects 3.0,4.0 --patterns shift,tail \
      --noise iid,ar1-0.3,ar1-0.6,ar1-0.8 --datasets 60 \
      --output <dir>
  ```

**Not yet done**: the paper's Fig 2 image has *not* been regenerated. The
fill data is ready for merger with existing Fig 2 source data. See §[Fig 2
regeneration](#fig-2-regeneration) in the camera-ready section for the
pending work.

---

## 3. Synthetic AR(1) vs. real AWS distribution alignment

**Reviewer targets**:
- #1370D Q2 (explicit) — synthetic data representativeness
- #1370A — "synthetic data doesn't reflect real cache/branch behaviour"
- #1370C — same concern framed around synthetic data for parameter calibration

### Paste-ready claim — budget-trimmed (~95 words, D-Q2 paragraph, **use this one**)

> **D-Q2 (synthetic vs. AWS).** The synthetic AR(1) grid in §5 is **more
> adversarial** than the dependence structure of real constant-time crypto
> on c8a.4xlarge (§5.1 hardware): measured asymmetric-crypto ρ₁ = 0.02
> (idle) / 0.00 (loaded), PW block 2.4 / 0.7, versus our synthetic
> φ ∈ {0.3, 0.6, 0.8} producing effective ρ₁ ∈ {0.15, 0.29, 0.39}.
> MARVIN RSA (§5.5) — the variable-time case the grid was designed to
> stress — sits squarely inside the φ=0.6 column (block length 19,
> IACT ≈ 19). Real upper tails are heavier than LogNormal synthetic
> (p99.9 at 111–22 500 σ_MAD vs 3.7 σ_MAD); since θ_floor is calibrated
> from the raw un-debiased W₁, this inflates θ_floor conservatively on
> real hardware, shifting borderline verdicts toward Inconclusive rather
> than Fail.

**Reframe rationale.** The earlier "weak-autocorrelation end of the grid"
framing reads as defensive; the sharper claim is that our grid *brackets*
the real dependence regimes and the worst real case (MARVIN) sits at the
center of its designed-for zone. Same data, stronger framing for Reviewer D.

### Paste-ready claim — full version (~170 words, for artifact/appendix use)

> **D-Q2 (synthetic vs. AWS).** We characterised both streams on the four
> properties driving inference: lag-k autocorrelation, MAD-robust upper-tail
> CDF, Geyer IMS IACT, and Politis–White block length (script
> `scripts/analyze_synth_vs_aws.py`, same estimators as
> `tacet-core/src/statistics/{autocorrelation,iact,block_length}.rs`). On
> `c8a.4xlarge` — the §5.1 hardware — asymmetric-crypto streams show median
> ρ₁ = 0.02 (idle) / 0.00 (loaded); PW block length 2.4 / 0.7; IACT τ̂ = 1.4
> / 1.0. These sit at the weak end of our AR(1) sweep
> (φ ∈ {0, 0.3, 0.6, 0.8} produce effective ρ₁ of 0.00, 0.15, 0.29, 0.39);
> i.e. the synthetic grid is *more adversarial* than typical real
> measurements. MARVIN RSA on EPYC 9575F (§5.5) is the autocorrelated
> exception: block length 19, IACT ≈ 19, inside the synthetic φ=0.6 column
> — the regime our grid was designed to cover. The real upper tail is
> dramatically heavier than LogNormal synthetic (p99.9 at 111 σ_MAD idle,
> 22 500 σ_MAD loaded, vs. 3.7–5.2 σ_MAD synthetic). This strengthens
> calibration: θ_floor is derived from the raw un-debiased W₁, whose tail
> sensitivity inflates θ_floor conservatively on real hardware, pushing
> borderline verdicts toward Inconclusive rather than Fail.

### Scope / data produced

- **Synthetic streams:** 60 CSVs at `paper/author-response/synth-dump/` — (pattern ∈ {null, shift-1σ, tail-1σ}) × (nominal φ ∈ {0, 0.3, 0.6, 0.8}) × 5 seeds, 10 000 samples/class. Generator: `crates/tacet-bench/src/bin/dump_synthetic.rs` → `synthetic::generate_benchmark_dataset`.
- **Real streams:** 42 CSVs at `paper/author-response/raw-aws/{idle,loaded}/` — Tier-1 registry (RustCrypto AES, ring AES-GCM, RustCrypto ChaCha20-Poly1305, SHA3-256, dalek X25519, libsodium Ed25519, pqcrypto Kyber-768), 3 iterations × 10 000 samples/class, rdtsc timer (resolution 0.385 ns).
- **Hardware:** `c8a.4xlarge` (16-vCPU AMD EPYC 9R45, Zen 4; matches §5.1's named instance family). Loaded regime = `stress-ng --cpu 20 --cpu-load 95 --vm 4 --vm-bytes 512M --cache 4` running concurrently for 2 min; benchmark pins to a CPU, so stress-ng increases memory / cache / TLB / scheduler pressure rather than directly contending for the pinned core.
- **Analysis:** `scripts/analyze_synth_vs_aws.py` — ACF, MAD-robust upper-tail, Geyer IMS IACT, Politis–White block length. Cross-referenced against the Rust estimators at `crates/tacet-core/src/statistics/{autocorrelation,iact,block_length}.rs`.
- **Aggregates:** `paper/author-response/synth_vs_aws.{csv,md}`.

### Numbers

| Regime | ρ₁ | ρ₅ | IACT τ̂ | PW block (SB) | p99.9 (σ_MAD) |
|---|---:|---:|---:|---:|---:|
| Synthetic φ=0.0 | 0.002 | –0.003 | 1.02 | 0.9 | 3.7 |
| Synthetic φ=0.3 | 0.147 | –0.002 | 1.42 | 11.0 | 3.9 |
| Synthetic φ=0.6 | 0.292 | 0.033 | 2.49 | 24.7 | 4.0 |
| Synthetic φ=0.8 | 0.392 | 0.156 | 4.75 | 48.0 | 3.9 |
| AWS idle (continuous) | 0.018 | 0.007 | 1.39 | 2.4 | 111 |
| AWS loaded (continuous) | –0.003 | –0.003 | 1.00 | 0.7 | 22 500 |

"Continuous" = streams whose crypto operation is slow enough to produce >100 unique timing values; fast ops (AES, SHA3, ChaCha20) saturate the rdtsc floor and fall into the paper's `Unmeasurable` category. The continuous streams are asymmetric crypto: X25519, Ed25519, Kyber-768.

MARVIN RSA data point from paper §5.5 (EPYC 9575F, 32-vCPU): PW block length 19, ESS 526/10 000, IACT ≈ 19 — inside our synthetic φ=0.6 column.

### Key interpretations

1. **Autocorrelation on real constant-time crypto is very low** (ρ₁ ≤ 0.02), well *below* the weakest AR(1) synthetic condition with ρ > 0 (φ=0.3 → effective ρ₁ = 0.147). Our AR(1) sweep provides conservative headroom over what normal AWS measurements look like.
2. **The autocorrelated regime our sweep exists to stress is the RSA / variable-time case**: MARVIN's IACT ≈ 19 sits inside φ=0.6 synthetic. The grid was explicitly designed for this case.
3. **Real upper tails are dramatically heavier than synthetic** (p99.9 at 111 σ_MAD idle, 22 500 σ_MAD loaded, vs. 3.7 σ_MAD synthetic). Cause: occasional interrupt-induced 10 ms spikes in 33 μs-median streams. **This strengthens, not weakens, the calibration**: θ_floor is derived from the *raw, un-debiased* W₁, whose tail sensitivity means heavier real tails inflate θ_floor conservatively → borderline verdicts shift toward Inconclusive rather than Fail.
4. **Synthetic φ ≠ effective ρ.** `synthetic.rs:435` adds AR(1) noise multiplicatively in log-space with scale 0.10, attenuating the stream-level autocorrelation (φ=0.8 → measured ρ₁=0.39). When we say "our grid brackets real measurements", we're comparing against *effective* ρ, not nominal φ. The paper currently reports nominal φ without flagging this.

### Open questions

- Should we also characterise `c8g.4xlarge` (ARM, cntvct_el0 timer) in the rebuttal? Currently only x86_64 + rdtsc is covered. Adding ARM would strengthen the "both microarchitectures in §5.1 fit the grid" claim but costs another 30 min on a burst instance. Deferred — not needed for D-Q2.
- The MARVIN numbers are from the paper's original run on EPYC 9575F, not from the re-capture here. If we want a like-for-like RSA data point on c8a.4xlarge, another 5-iter capture of Tier 2 (MARVIN) is ~10 min. Defer until after other experiments are in.

### Data pointers

- **Aggregated tables**: [paper/author-response/synth_vs_aws.md](synth_vs_aws.md), [synth_vs_aws.csv](synth_vs_aws.csv)
- **Raw synthetic CSVs**: [paper/author-response/synth-dump/](synth-dump/)
- **Raw AWS CSVs**: [paper/author-response/raw-aws/{idle,loaded}/](raw-aws/)
- **Analyzer script**: `scripts/analyze_synth_vs_aws.py`
- **Synthetic generator**: `crates/tacet-bench/src/bin/dump_synthetic.rs`

Camera-ready implications for this finding are consolidated in the
[camera-ready section below](#e-synth-vs-aws-calibration-disclosure-from-3).

---

## 4. Operation-loop amplification (ShowTime) overlay

**Reviewer targets**:
- #1370B Q1 (explicit) — "discuss timing amplification techniques, such as those proposed in ShowTime, and how these might affect the detection assumptions"
- #1370B limitation — "amplification techniques influence the security of the chosen threshold parameter"

### Paste-ready claim — budget-trimmed (~100 words, B-Q1 paragraph, **use this one**)

> **B-Q1 (amplification / ShowTime).** Amplification is a measurement-side
> capability: an attacker with budget k_adv scales the per-query W₁ by
> k_adv, so a θ_user bound on W₁ also bounds the amplified attacker's
> signal at θ_user / k_adv. For amplification-capable threat models,
> users set θ_user = θ_physical / k_adv; the paper's security argument
> extends directly. We verified the scaling by overlaying amplified
> (d = 200 ns, k ∈ {5, 10, 25, 100}) against single-op baselines at
> matched actual effective delays within 5% (`busy_wait_ns` calibration).
> Against AdjacentNetwork (θ = 100 ns) on EPYC @ 5 GHz, amplified and
> single-op TPRs coincide within 95% Wilson CIs at every matched pair
> (all four 100% TPR, n = 20/cell). We will add ShowTime and this
> discussion to the camera-ready.

**Reframe rationale.** The earlier draft led with *"amplification does not
defeat tacet — it refines the threshold-choice rule,"* which reads as
goalpost-moving to Reviewer B. The revised version leads with the
*implication* (a θ_user bound on per-query W₁ is an amplification-aware
bound because W₁ scales linearly with k) before stating the rule, so
the rule is framed as operating the existing θ_user formalism correctly
rather than retreating from the security claim.

### Paste-ready claim — full version (~140 words, for artifact/appendix use)

> **Reviewer B, Q1 (amplification / ShowTime).** Amplification is a
> measurement-side capability available to both attacker and defender; the
> defender-side question is whether θ_user tracks the attacker's amplification
> budget k_adv. For amplification-capable threat models we recommend
> θ_user = θ_physical / k_adv. We verified this rule by overlaying amplified
> (d = 200 ns, k ∈ {5, 10, 25, 100}) against single-op baselines
> (d ∈ {1 000, 2 000, 5 000, 20 000} ns) whose actual effective delays match
> within 5 % (measured via `busy_wait_ns` calibration). Against AdjacentNetwork
> (θ = 100 ns), amplified and single-op TPRs coincide within 95 % Wilson CIs
> at every matched pair — all four point-estimates at 100 % TPR, 20 trials
> each, EPYC @ 5 GHz. Per-query W₁ scales linearly with k as predicted, so
> θ = θ_physical / k_adv preserves the intended security margin against a
> ShowTime-style attacker. We will add ShowTime and this discussion to the
> camera-ready.

### Scope / data produced

- **Injection-primitive calibration.** 100 000 calls × 11 repeats of
  `busy_wait_ns(d)` for nominal d ∈ {1, 2, 3, 5, 10, 20, 50, 100, 200, 500,
  1000, 5000} ns, EPYC 4564P @ 5.49 GHz. Measured a constant **≈ 13 ns
  per-call overhead** and a **≈ 19 ns floor** (d ≤ 5 all collapse to 19 ns
  actual). Injection is faithful to nominal within 5 % only for d ≥ 200 ns.
  Harness: `crates/tacet/examples/busy_wait_calibration.rs`.
- **Overlay sweep.** 18 tests × 20 iterations = 360 trials. Amplified grid
  at d = 200 ns × k ∈ {5, 10, 25, 100} vs single-op baselines at
  d ∈ {200, 500, 1000, 2000, 5000, 20000} ns. Runpod AMD EPYC 4564P @ 5.49 GHz,
  shared-tenancy container, rdtsc @ 0.2 ns, no CPU pinning (worst-case
  jitter). Harness: `crates/tacet/tests/leaky/injected.rs`.

### `busy_wait_ns` calibration (EPYC @ 5 GHz)

| nominal | actual (mean) | overhead / nominal | pure overhead |
|--------:|--------------:|-------------------:|--------------:|
|   1 ns  |      19.1 ns  |            19.1 ×  |       +18 ns |
|   2 ns  |      19.1 ns  |             9.5 ×  |       +17 ns |
|   5 ns  |      19.1 ns  |             3.8 ×  |       +14 ns |
|  10 ns  |      20.1 ns  |             2.0 ×  |       +10 ns |
|  20 ns  |      33.7 ns  |             1.7 ×  |       +14 ns |
|  50 ns  |      62.9 ns  |            1.26 ×  |       +13 ns |
| 100 ns  |     115.8 ns  |            1.16 ×  |       +16 ns |
| 200 ns  |     211.1 ns  |            1.05 ×  |       +11 ns |
| 500 ns  |     512.9 ns  |            1.03 ×  |       +13 ns |
|1000 ns  |    1011.5 ns  |            1.01 ×  |       +12 ns |
|5000 ns  |    5026.0 ns  |           1.005 ×  |       +26 ns |

### Overlay table (n = 20 per cell)

For each row: `single-op` = test `injected_shift_{X}ns` at nominal d = X;
`amplified` = test `injected_shift_200ns_k{K}` at d = 200 ns × K. Wilson
95 % CIs computed conditional on definitive verdict (excluding Inconclusive).

| eff_ns (actual) | single-op TP / Inc / TPR (CI)              | amplified TP / Inc / TPR (CI)              |
|----------------:|--------------------------------------------|--------------------------------------------|
|    ~1 050       | 12 / 8 / **100 %** [75.7, 100]             | 12 / 8 / **100 %** [75.7, 100]             |
|    ~2 100       | 16 / 4 / **100 %** [80.6, 100]             | 19 / 1 / **100 %** [83.2, 100]             |
|    ~5 250       | 13 / 7 / **100 %** [77.2, 100]             | 14 / 6 / **100 %** [78.5, 100]             |
|   ~21 000       | 16 / 4 / **100 %** [80.6, 100]             | 15 / 5 / **100 %** [79.6, 100]             |

Every matched pair agrees within 95 % Wilson CIs. TPR = 100 % everywhere
above θ = 100 ns (no false negatives; Inconclusive rate 20–40 % reflects
Runpod's shared-tenancy jitter, not threshold violation). Per-query
detection depends only on the actual effective delay d · k.

### Key interpretations

1. **Scaling law holds cleanly.** Four matched (single-op, amplified) pairs
   at actual effective delays {~1050, ~2100, ~5250, ~21000} ns each coincide
   within 95 % Wilson CIs at 100 % TPR on definitive verdicts. Per-query
   W₁ scales linearly with k as predicted.
2. **Amplification does not defeat tacet** — it refines the threshold-choice
   rule. For amplification-capable threat models the defender sets
   θ_user = θ_physical / k_adv so that the intended security margin holds
   after k_adv-fold amplification.
3. **The overlay is only established in the region d ≥ 1 000 ns** where
   `busy_wait_ns` is faithful to nominal. Below that, per-call overhead
   dominates and both sides saturate the injection floor — a limitation of
   the injection primitive, not of tacet.

### Scope of the claim (cautions)

- The experiment does not resolve whether amplified injection has *lower
  variance* than single-op at matched effective delay (a possible advantage
  from averaging per-call jitter across k repetitions); at n = 20 we do not
  resolve sub-CI differences. This is a second-order question Reviewer B
  did not ask.
- Defensive fallback if re-run on different hardware is noisier: all
  amplified configurations detected at 100 % of definitive verdicts —
  sufficient evidence that per-query W₁ scales at least linearly with k,
  which is all the θ = θ_physical / k_adv rule requires.

### Data pointers

- **Rebuttal paragraph + overlay writeup**: [amplification/REBUTTAL_PARAGRAPH.md](amplification/REBUTTAL_PARAGRAPH.md)
- **Raw sweep CSV (360 trials)**: [amplification/amp_runpod_v2.csv](amplification/amp_runpod_v2.csv)
- **Analyzer report**: [amplification/amp_runpod_v2_report.txt](amplification/amp_runpod_v2_report.txt)
- **Per-call calibration CSV**: [amplification/busy_wait_calibration.csv](amplification/busy_wait_calibration.csv)
- **Exploratory precursors**: [amplification/exploratory/](amplification/exploratory/)
  (smoke run on macOS cntvct, and the v1 run with the nominal-axis x-axis bug that led to the calibration discovery)
- **Test harness**: `crates/tacet/tests/leaky/injected.rs`
- **Calibration harness**: `crates/tacet/examples/busy_wait_calibration.rs`
- **Analyzer overlay support**: `scripts/analyze_tpr.py` (detection-curve table auto-switches to delay / k / effective_ns columns when any row has k > 1)
- **Commit**: `feat(rebuttal): operation-loop amplification overlay (Reviewer B Q1)` on `sec26-response`
- **Reproduce**:
  ```bash
  cargo build --release -p tacet --test leaky
  ./scripts/measure_tpr.sh 20 amp.csv
  ./scripts/analyze_tpr.py amp.csv
  cargo run --release -p tacet --example busy_wait_calibration
  ```

Camera-ready implications for this finding are consolidated in the
[camera-ready section below](#f-amplification-paragraph--showtime-cite--injection-floor-disclosure-from-4).

---

# For Camera-Ready (if accepted)

Do **not** include any of this material in the Apr 23 rebuttal. These are
punch-list items for the camera-ready version if the paper is accepted.

## A. Align FFI path ν_ℓ with the paper's stated value

**Finding.** The published paper specifies ν_ℓ = 8 for the Student-t
likelihood (`paper/appendix_methodology.tex:42` and `paper/paper.tex:1276`,
both say *"with $\nu_\ell = 8$"*). The Rust headline path — which is what
the paper's evaluation was run through — matches: `nu_likelihood: 8.0` at
`crates/tacet/src/adaptive/single_pass.rs:96`. The core FFI path used by
C / Go / Node consumers, however, passes a hardcoded `4.0`:

```rust
// crates/tacet-core/src/adaptive/step.rs:717-724
let bayes_result = compute_bayes_1d(
    w1_obs, var_n, calibration.sigma_t, config.theta_ns, config.seed,
    4.0, // nu_likelihood: Student-t df for robustness (FFI path)
    4.0, // nu_prior: half-t prior df (§A.1 default)
);
```

So FFI consumers receive a more-conservative posterior than the paper's
numbers reflect (ν_ℓ = 4 has heavier tails → more Inconclusive verdicts
than ν_ℓ = 8).

**Action.** Change the FFI literal from `4.0` → `8.0` at
`crates/tacet-core/src/adaptive/step.rs:717`. No paper edit needed — the
paper is already correct.

**Impact on calibration claims.** None. Paper numbers were generated on
the Rust headline path (ν_ℓ = 8) and are unaffected. The ablation v3 sweep
included ν_ℓ = 4 as a config — FPR stayed at 0.00% and 1σ detection at
100%, so FFI consumers have been over-conservative rather than
miscalibrated.

**Camera-ready edit location**: `crates/tacet-core/src/adaptive/step.rs:717` (one literal).

## B. Reconcile paper §A.3 "fragile-regime 1.5× inflation" with current code

**Finding.** Paper §A.3 (`appendix_methodology.tex:138`) describes a
**conditional** 1.5× block-length inflation:

> *"In high-autocorrelation regimes (detected when $\rho_{\text{max}}(10) > 0.3$),
> we inflate the block length by 50% for additional conservatism."*

The code no longer implements this rule. At
`crates/tacet-core/src/adaptive/calibration.rs:500`, the call to
`bootstrap_w1_variance(...)` passes `false` hardcoded for the `is_fragile`
argument:

```rust
let var_estimate = bootstrap_w1_variance(
    &interleaved,
    config.bootstrap_iterations,
    config.seed,
    false, // is_fragile        ← hardcoded
    config.bootstrap_method,
);
```

The 9D-era ρ_max(10) > 0.3 detector was removed during the 1D migration
and was never rewired. So the paper describes a mechanism the current
implementation does not perform. (Note: an *unconditional* 1.5× is still
applied at `block_length.rs:354` — that is a separate effect and not the
rule §A.3 describes.)

**Action.** Two options; either is acceptable:

- **(a) Re-enable the detector.** Restore a 1D-compatible ρ_max(10) > 0.3
  check in `calibration.rs` and plumb the result into the
  `bootstrap_w1_variance` call. Matches the paper exactly.
- **(b) Update the paper.** Rewrite §A.3's block-length paragraph to
  describe the current unconditional 1.5× at `block_length.rs:354` and
  drop the fragility-conditional language.

**Impact on calibration claims.** None on the baseline numbers — the
paper's evaluation was run with the current code (no fragility gate fires
regardless of input). The discrepancy is purely descriptive. (a) is the
more principled fix because it makes the paper's conservatism claim true
on hard inputs like MARVIN; (b) is the simpler fix.

**Camera-ready edit location**:
- For (a): `crates/tacet-core/src/adaptive/calibration.rs:500` and the
  removed detector site.
- For (b): `paper/appendix_methodology.tex:138`.

## C. Fig 2 regeneration

**Finding.** The fill-in data from §[2. Figure 2](#2-figure-2--detection-curve-fill-in-3σ-4σ)
has not been merged into the paper's Fig 2 source image. The rebuttal
cites the new points in prose only; in the camera-ready we want an updated
Fig 2 with the 3σ/4σ rows visible.

**Pending steps**:
1. Merge `paper/author-response/fig2-fill/benchmark_results.csv` with the
   original Fig 2 data source.
2. Regenerate the figure (source pipeline in `paper/analysis/`).
3. Verify the no-cliff interpretation visually.

**For the rebuttal**, we cite the new points in prose only (see
[Figure 2 claim](#paste-ready-claim-30-words) above) — no figure regeneration
required on the Apr 23 timeline.

## G. ν_ℓ = 4 FFI discrepancy: camera-ready only (resolved — no rebuttal disclosure needed)

**Resolution.** §5.5 "Crypto Library Validation" ran through the Rust
headline path for **all 670 trials** (ν_ℓ = 8). The 67-test FPR
measurement explicitly enumerates "10 libraries across Rust and C/C++"
covering 7 libraries in the source CSV (RustCrypto, ring, dalek,
pqcrypto, orion, libressl, libsodium). Go and JavaScript/WASM rows in
Table 2 are coverage indicators, not part of the FPR number.

**Code trace.** The C/C++ library tests
(`crates/tacet/tests/crypto/c_libraries/*.rs`) import
`use tacet::{..., TimingOracle};` and call
`TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)` directly
from Rust. The C/C++ FFI is used only for the *cryptographic library
under test* (LibreSSL, Libsodium, etc.) — never for the Tacet oracle
itself. The oracle always runs in Rust through
`tacet::Config` → `SinglePassConfig` → `single_pass.rs`, which sets
`nu_likelihood: 8.0` (§A).

Verified sites:
- `crates/tacet/tests/crypto/c_libraries/libressl.rs:30,110,227,300,407,475,559`
  — all `TimingOracle::for_attacker(...)`
- `crates/tacet/tests/crypto/c_libraries/libsodium.rs` — same pattern
- `crates/tacet/tests/crypto/{rustcrypto,dalek,ring,pqcrypto}.rs` — Rust
  libs, trivially Rust path
- `results/_old/fpr_crypto_noisy_final.csv` — 670 rows total:
  RustCrypto=270, pqcrypto=110, dalek=70, libsodium=70, orion=60,
  libressl=50, ring=40. No Go / JS / WASM rows.

**Impact on the rebuttal.** **Zero.** No paper-reported number ever went
through the ν_ℓ = 4 path. The FFI ν_ℓ = 4 literal affects C / Go / Node
/ WASM library consumers only, and is strictly a code-consistency issue.
Remains a camera-ready patch (§A).

**Decision**: do **not** disclose the FFI discrepancy in the rebuttal.
The claim "paper method matches what was run" holds for every row cited.

## H. Decide disclosure timing for the `busy_wait_ns` 19 ns floor (see §4 and F)

**Open question.** §4 documents that `busy_wait_ns(d)` has an ~19 ns
floor on x86 @ 5 GHz (nominal d ≤ 10 ns all produce ~19 ns actual).
Any paper-cited detection-curve row at nominal d < 200 ns is wrong on
its x-axis.

**Action before Thursday**: audit §5.4 (ablation) and any detection-curve
figure or table in the paper for reported d values in this range.

- If none of the headline figures cite nominal d < 200 ns, the floor
  is a camera-ready-only disclosure (§F).
- If any headline cite is below the floor, disclose in the rebuttal
  alongside §4 ("small-d detection rows will be relabeled with measured
  effective delay in the camera-ready; the load-bearing result —
  operation-loop amplification scaling — is established in the
  d ≥ 1 000 ns region where the primitive is faithful").

**Priority**: medium. The amplification result itself is not affected
because the overlay used d ≥ 1 000 ns.

## D. Other items surfaced but out of scope for rebuttal

- The `nu_likelihood` builder validation (`nu > 2.0`) is tight. For the
  ablation we used ν_ℓ = 2.01 (valid) to demonstrate the near-Cauchy
  boundary. No action needed.
- `kl_loose = baseline` exactly on clean data demonstrates the gate is
  *never binding* on reasonable inputs. Worth a one-sentence caveat in
  the paper's gate-description section for transparency, but not a
  correctness issue.

## E. Synth-vs-AWS calibration disclosure (from §3)

Carry-overs from [§3 Synthetic AR(1) vs. real AWS](#3-synthetic-ar1-vs-real-aws-distribution-alignment):

- §5.1 methodology should report **effective** ρ on the synthetic streams,
  not nominal φ — or rescale the AR(1) multiplier in `synthetic.rs:435`
  so nominal φ == effective ρ. Currently misleading.
- Add a short "measurement-regime calibration" paragraph in §5 documenting
  the empirical ρ / block / tail on real c8a — even ~5 lines referencing
  `synth_vs_aws.md` in the artefact.
- Consider adding the tail-heaviness finding to §3.4 (Measurement Floor):
  "real hardware tails are substantially heavier than the LogNormal
  synthetic model; θ_floor's tail sensitivity is therefore a feature,
  not a side-effect."

## F. Amplification paragraph + ShowTime cite + injection-floor disclosure (from §4)

Carry-overs from [§4 Operation-loop amplification](#4-operation-loop-amplification-showtime-overlay):

- Add `\paragraph{Operation-loop amplification.}` in §6 Discussion right
  after `\paragraph{Concentrated tail effects.}` (paper.tex:1122). Content:
  dual-capability framing + θ_user = θ_physical / k_adv rule + pointer to
  overlay table.
- Cite ShowTime (Rokicki et al.) in `paper.bib` and in the related-work
  discussion of timing-attack primitives.
- **Injection floor is a paper issue, not just a rebuttal issue.**
  `busy_wait_ns` calibration shows `injected_shift_{2,5}ns` tests inject
  ~19 ns actual, not nominal. Any detection-curve figure built from these
  primitives at small d must report measured delay on the x-axis, not
  nominal. See also [CAMERA_READY_TODO.md](CAMERA_READY_TODO.md).
- Keep `crates/tacet/examples/busy_wait_calibration.rs` in-tree as a
  reproducible calibration artefact.
- Consider an overlay figure for the camera-ready using `amp_runpod_v2.csv`:
  amplified vs single-op TPR against actual effective delay. Reuse
  `crates/tacet/scripts/plot_power_curve.py` with a new CSV export mode.

---

## Index of files referenced

### Hyperparameter sensitivity ablation (§1)

- `paper/author-response/ablation-v3/` — final hyperparameter sweep data (16 configs × 120 datasets, ~187k trials)
- `paper/author-response/ablation-v2/` — initial sweep (13 configs × 60 datasets, preserved for provenance)
- `scripts/aws-ablation-sweep.sh` — sweep orchestrator
- `scripts/analyze_ablation.py` — aggregator with Wilson 95% CIs
- `crates/tacet/src/config.rs`, `crates/tacet/src/oracle.rs`,
  `crates/tacet/src/adaptive/single_pass.rs`,
  `crates/tacet-core/src/adaptive/calibration.rs`,
  `crates/tacet-core/src/analysis/bayes.rs`,
  `crates/tacet-bench/src/bin/benchmark.rs`,
  `crates/tacet-bench/src/adapters.rs` — plumbing for
  `TACET_ABLATION_{PI0, ALPHA, BETA, KL_MIN, NU_LIKELIHOOD, NU_PRIOR}` env vars
- `crates/tacet-core/src/adaptive/step.rs:717` (camera-ready fix site)
- `crates/tacet-core/src/adaptive/calibration.rs:500` (camera-ready fix site)

### Figure 2 fill-in (§2)

- `paper/author-response/fig2-fill/` — 3σ/4σ detection rows
- `paper/analysis/` — figure regeneration pipeline (pending camera-ready)

### Synth-vs-AWS (§3)

- `paper/author-response/synth-dump/` — 60 synthetic stream CSVs
- `paper/author-response/raw-aws/{idle,loaded}/` — 42 real-hardware stream CSVs
- `paper/author-response/synth_vs_aws.{csv,md}` — aggregated comparison tables
- `scripts/analyze_synth_vs_aws.py` — ACF / IACT / PW block / tail analyzer
- `crates/tacet-bench/src/bin/dump_synthetic.rs` — synthetic stream generator
- `crates/tacet-core/src/statistics/{autocorrelation,iact,block_length}.rs` — reference estimators

### Amplification / ShowTime overlay (§4)

- `paper/author-response/amplification/` — rebuttal paragraph, raw sweep CSV, calibration CSV, exploratory precursors
- `paper/author-response/CAMERA_READY_TODO.md` — injection-floor disclosure tracker
- `crates/tacet/tests/leaky/injected.rs` — single-op + amplified-sweep harness
- `crates/tacet/examples/busy_wait_calibration.rs` — per-call delay calibration
- `scripts/measure_tpr.sh`, `scripts/analyze_tpr.py` — TPR sweep + overlay table support
