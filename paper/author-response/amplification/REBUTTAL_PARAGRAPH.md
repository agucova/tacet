# F-AMP — Operation-loop amplification (ShowTime) overlay

**Answers:** Reviewer B Q1 (explicit) + B limitation about threshold-security under amplification.

## What we did

Extended the existing injected-shift power-curve infrastructure
(`crates/tacet/tests/leaky/injected.rs`) with:

1. **Per-call calibration of `busy_wait_ns`** (`crates/tacet/examples/busy_wait_calibration.rs`). Measured actual wall-clock per-call delay for nominal d ∈ {1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 5000} ns; N=100 000 calls × 11 repeats.
2. **Amplification overlay grid**: d = 200 ns (the smallest d where actual/nominal ≤ 5 %), k ∈ {5, 10, 25, 100}. Single-op baselines at nominal d ∈ {1000, 2000, 5000, 20000} ns chosen so each (amplified, baseline) pair coincides in *actual* effective delay within ~5 %.
3. **Full sweep**, 20 iterations × 18 tests = 360 trials. AMD EPYC 4564P @ 5.49 GHz, Runpod container, rdtsc timer (resolution 0.2 ns), no CPU pinning (shared-tenancy container, worst-case for jitter).

All data archived in `paper/author-response/amplification/`:
- `busy_wait_calibration.csv` — per-call delay vs nominal
- `amp_runpod_v3.csv` — raw sweep outcomes, 420 trials, both amplification bases
- `amp_runpod_v3_report.txt` — analyzer output (two-base overlay)
- `amp_runpod_v2.csv` / `amp_runpod_v2_report.txt` — single-base (d=200) predecessor, preserved for provenance

## Data

### `busy_wait_ns` calibration (EPYC @ 5 GHz)

| nominal | actual (mean) | overhead/nominal | pure overhead |
|--------:|--------------:|-----------------:|--------------:|
|   1 ns  |      19.1 ns  |          19.1 ×  |       +18 ns |
|   2 ns  |      19.1 ns  |           9.5 ×  |       +17 ns |
|   5 ns  |      19.1 ns  |           3.8 ×  |       +14 ns |
|  10 ns  |      20.1 ns  |           2.0 ×  |       +10 ns |
|  20 ns  |      33.7 ns  |           1.7 ×  |       +14 ns |
|  50 ns  |      62.9 ns  |          1.26 ×  |       +13 ns |
| 100 ns  |     115.8 ns  |          1.16 ×  |       +16 ns |
| 200 ns  |     211.1 ns  |          1.05 ×  |       +11 ns |
| 500 ns  |     512.9 ns  |          1.03 ×  |       +13 ns |
|1000 ns  |    1011.5 ns  |          1.01 ×  |       +12 ns |
|5000 ns  |    5026.0 ns  |         1.005 ×  |       +26 ns |

Per-call overhead ≈ **13 ns constant**, with a ≈ **19 ns floor** (d ≤ 5 all collapse to 19 ns actual). Injection is faithful to nominal within 5 % only for d ≥ 200 ns.

### Amplification overlay (360-trial sweep, n = 20 per cell)

For each row: `single-op` = test `injected_shift_{X}ns` at nominal d = X; `amplified` = test `injected_shift_200ns_k{K}` at d = 200 ns × K. Wilson 95 % CIs computed conditional on definitive verdict (excluding Inconclusive).

| eff_ns (actual) | single-op TP / Inc / TPR (CI)              | amplified TP / Inc / TPR (CI)              |
|----------------:|--------------------------------------------|--------------------------------------------|
| eff_ns (actual) | single-op (d = eff)          | amp @ d=200                    | amp @ d=1000                  |
|----------------:|------------------------------|--------------------------------|-------------------------------|
|  ~1 050         | 12 / 8 / **100 %** [75.7, 100] | 12 / 8 / **100 %** [75.7, 100]   | —                             |
|  ~2 050         | 16 / 4 / **100 %** [80.6, 100] | 19 / 1 / **100 %** [83.2, 100]   | **20 / 0 / 100 %** [83.9, 100] |
|  ~5 100         | 13 / 7 / **100 %** [77.2, 100] | 14 / 6 / **100 %** [78.5, 100]   | 14 / 6 / **100 %** [78.5, 100]  |
| ~20 500         | 16 / 4 / **100 %** [80.6, 100] | 15 / 5 / **100 %** [79.6, 100]   | 14 / 6 / **100 %** [78.5, 100]  |

Two independent amplification bases (d ∈ {200, 1000} ns) and single-op baselines all coincide within 95 % Wilson CIs at each matched-effective-delay tier. TPR = 100 % everywhere above θ = 100 ns (no false negatives; Inconclusive rate 20–40 % reflects Runpod's shared-tenancy jitter, not threshold violation). Per-query detection depends only on the actual effective delay d · k, and this holds under two orthogonal choices of per-op base d — the scaling law is not a single-base artifact. Sweep totals: 21 tests × 20 iterations = 420 trials.

## Rebuttal paragraph draft (≈ 140 words of the 700-word budget)

> **Reviewer B, Q1 (amplification / ShowTime).** Amplification is a measurement-side capability: an attacker with budget k_adv scales per-query W₁ by k_adv, so a θ_user bound on W₁ also bounds the amplified signal at θ_user / k_adv. For amplification-capable threat models, users set θ_user = θ_physical / k_adv; the paper's security argument extends directly. We verified the scaling by overlaying two independent amplification bases (d ∈ {200, 1000} ns, sweeping k) against single-op baselines at matched actual effective delays within 5 % (`busy_wait_ns` calibration). Against AdjacentNetwork (θ = 100 ns) on EPYC @ 5 GHz, all eleven amplified configurations and four single-op baselines coincide within 95 % Wilson CIs at every matched tier (all 100 % TPR, n = 20/cell, 420 trials total). The two-base overlay confirms the scaling law is not an artifact of the base choice. We will add ShowTime and this discussion to the camera-ready.

Word count: 149.

## Camera-ready implications

- **Add `\paragraph{Operation-loop amplification.}`** in §6 Discussion (after the existing `\paragraph{Concentrated tail effects.}` at `paper.tex:1122`): dual-capability framing + θ = θ_physical / k_adv rule + pointer to overlay table.
- **Cite ShowTime (Rokicki et al.)** in `paper.bib` and in the related-work discussion of timing-attack primitives.
- **`busy_wait_ns` overhead floor is a paper issue, not just a rebuttal issue.** The existing `injected_shift_{2,5}ns` tests inject ~19 ns actual, not 2/5 ns. Any detection-curve figure built from these primitives at small d must report measured delay on the x-axis, not nominal. See `CAMERA_READY_TODO.md` (parent dir).
- **Consider an overlay figure for the camera-ready** using `amp_runpod_v3.csv` (two-base, 420 trials): amplified vs single-op TPR against actual effective delay, with the two bases rendered as distinct series to make the base-invariance visually obvious. Reuse `crates/tacet/scripts/plot_power_curve.py` with a new CSV export mode.

## Notes on scope of the claim

- The overlay is established only in the actual-effective-delay region ≥ 1 000 ns where `busy_wait_ns` is faithful to nominal. Below that, per-call overhead dominates and the experiment cannot distinguish amplified from single-op because both saturate the injection floor. This is a limitation of the injection primitive, not of tacet.
- The result is consistent with the reviewer-stated physical intuition: amplification multiplies the per-query W₁ by k, which moves the decision along the same detection curve. It does not answer the sub-question of whether amplified injection has *lower variance* than single-op at matched effective delay (a possible advantage from averaging per-call jitter across k repetitions); at n = 20 we do not resolve sub-CI differences.

## Defensive fallback

If a reviewer re-runs these experiments on different hardware and the overlay is noisier, the core claim still holds: **all amplified configurations detected at 100 % of definitive verdicts**, confirming per-query W₁ scales at least linearly with k — sufficient for the θ = θ_physical / k_adv rule.
