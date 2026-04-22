# MARVIN pool-size sweep: runtime conditions and findings

We swept sample-class pool size N ∈ {1, 10, 100, 1000} on the §5.6
cache-warming MARVIN variant at fixed budget = 62 000 samples/class,
20 seeds/pool (80 runs total). This answers **Reviewer B Q2** on input
generation sensitivity directly, rather than in prose.

## Target host

| Item | Value |
|------|-------|
| Instance type | RunPod container (shared host) |
| CPU | AMD EPYC 4564P (Genoa, 16C/32T) |
| Cores × threads | 16 physical, **SMT on** (2 threads/core) |
| L1d / L1i / L2 / L3 | 512 KiB × 16 / 512 KiB × 16 / 16 MiB / 64 MiB |
| NUMA | single node |
| Kernel | 6.8.0-51-generic (Ubuntu 22.04) |
| Rust | 1.90.0 (2025-09-14) |
| Timer | invariant TSC (`rdtsc`); measured resolution **≈ 0.223 ns** |
| `perf_event_paranoid` | 4 |
| Governor | `powersave` (RunPod containers do not expose governor control) |
| Core pinning | `taskset` per worker; 8-core groups (0–7, 8–15, 16–23, 24–31) |

This is the **same box** used for §5's cross-tool comparison and §8's
50k-adaptive MARVIN addendum. SMT-on + powersave governor mean
measurement-floor conditions are noisier than §8's c8a.8xlarge run
(SMT off, performance-pinned); block length ≈ 1,056 on all 80 runs
matches §8's 62k-on-c8a value precisely, confirming the autocorrelation
regime is dominated by the RSA decryption workload, not the host.

## Which MARVIN test

Same `--marvin-mode cache` variant as §8:

- Baseline: one fixed valid PKCS#1 v1.5 ciphertext (encrypted from
  `[0x42; 32]`), repeated for every measurement.
- Sample: pool of **N distinct valid ciphertexts** (deterministic under
  the per-iteration seed), cycled modulo pool size.

At **N = 1**, the sample class is a single random valid ciphertext
repeated. Both classes are then cache/branch-predictor warm on a
single ciphertext each, so the residual class-mean difference is
dominated by the **pure MARVIN variable-time RSA-CRT signal** for that
(fixed-ct, sample-ct) pair. At **N ≫ 1**, the sample class cycles
through many ciphertexts, mixing in cache/branch-predictor state
variance across the pool — this is §5.6's original configuration
(N = 200).

## Sweep parameters

| Parameter | Value |
|---|---|
| Budget (samples/class) | 62 000 (§5.6's baseline, unchanged) |
| Pool sizes | **1, 10, 100, 1000** |
| Seeds per pool | 20 |
| Seed derivation | `md5("20260422\|marvin-pool\|{pool_size}\|{iter}")`, low 60 bits |
| Parallelism | 4 concurrent runs; each pinned to its own 8-core group |
| Attacker model | `AdjacentNetwork` (θ = 100 ns), matches §5.6 / §8 |
| Calibration samples | tacet default (10 000) |
| Analysis codepath | `TimingOracle::analyze_raw_samples_with_resolution` (single-pass) |

## Results

All 80 runs completed.

| pool_size | N | verdicts (F/I/P) | %Fail (Wilson 95%) | median P | IQR P | median effect (ns) | median CI width (ns) | median ESS | block |
|---|---|---|---|---|---|---|---|---|---|
| 1     | 20 | 10/8/2 | **50%** (30–70%) | 0.919 | [0.30, 1.00] | **228** | **78** | 58 | 1,056 |
| 10    | 20 | 6/14/0 | 30% (15–52%) | 0.593 | [0.42, 0.98] | 189 | 300 | 58 | 1,056 |
| 100   | 20 | 6/14/0 | 30% (15–52%) | 0.509 | [0.12, 0.98] | 189 | 355 | 58 | 1,056 |
| 1000  | 20 | 5/13/2 | 25% (11–47%) | 0.509 | [0.19, 0.94] | 197 | 300 | 58 | 1,056 |

### Finding 1: Effect estimate is stable across pool size

Median effect stays within **189–228 ns** across a 1000× sweep in pool
size. The underlying leak magnitude the oracle recovers is
input-generation-invariant on this case.

### Finding 2: Detection rate is *highest* at N = 1, saturates at N ≥ 10

%Fail: **50% → 30% → 30% → 25%** as pool size grows. The naive
hypothesis "more pool diversity → more detection power" is wrong on
MARVIN. Detection rate effectively saturates by N ≈ 10 (Wilson CIs
overlap for N ∈ {10, 100, 1000}).

### Finding 3: Larger pools inflate posterior uncertainty, not signal

Median CI width: **78 ns at N = 1** vs **300–355 ns at N ≥ 10** —
a **~4× inflation**. At N = 1 both classes are cache-warm on a single
ciphertext each; within-class variance is noise-limited. At N ≫ 1,
sample-class variance includes variation across the variable-time
distribution of many ciphertexts, widening class-mean CIs without
shifting the point estimate. The posterior *correctly* reports more
uncertainty under more-diverse input generation.

### Finding 4: N = 1 is brittle to specific-ciphertext choice

2/20 seeds at N = 1 landed Pass with per-key effects < θ = 100 ns
(47, 61 ns) — those random ciphertexts happened to fall in the lower
tail of the MARVIN variable-time distribution. At N ≥ 10, averaging
over the pool produces a representative effect (~190 ns, above θ)
at the cost of wider CIs. **N = 1 gives the highest detection rate
but is less robust**; larger N is more representative but yields
more Inconclusives at 62k budget.

### Finding 5: DudeCT two-class assumption is load-bearing

Tacet inherits DudeCT's fixed-vs-random two-class input model. All
four pool-size regimes are valid within that model; the sweep shows
detection is **not monotone in pool diversity** but rather reflects
the trade-off between per-pair signal sharpness (small N) and
distributional coverage (large N). Structure-aware input generators
(e.g., Bleichenbacher-oracle-style ciphertext templating, Marvin's
ASN.1 malformations) lie outside tacet's DudeCT interface and are
deferred to future work.

## Deviations from §8

- Same MARVIN cache variant and analysis codepath; budget fixed at 1×.
- **Different hardware** (RunPod EPYC 4564P, SMT on, powersave)
  vs §8's c8a.8xlarge (SMT off, BIOS-pinned frequency). Block-length
  estimate (1,056) matches §8's 62k-on-c8a exactly, indicating the
  autocorrelation regime is workload-dominated; absolute detection
  rates may differ, but pool-size *relative* behavior transfers.
- 80 runs (vs §8's 140), fixed budget (vs §8's 7 budgets). The
  pool-size axis is independent of §8's budget axis.

## Limitations

- Sweep is on a single CVE (MARVIN). Generalization to other leak
  families (cache table lookups, AES T-tables) remains untested.
  We expect the opposite sign on signal-dominant-by-diversity leaks
  (where larger pools *do* improve detection), but cannot demonstrate
  it within the rebuttal timeline.
- N = 1 result is seed-sensitive (2/20 Pass); the headline 50%
  Fail rate has a wider Wilson CI than N ≥ 10 cells due to the
  bimodal effect distribution.
- This sweep does not evaluate *structure-aware* inputs (e.g.,
  oracle-targeting ASN.1 malformations); it sweeps only the number
  of ciphertexts within the existing DudeCT-style valid-valid
  configuration.
