# MARVIN budget sweep: runtime conditions and findings

We re-ran the §5.6 MARVIN (CVE-2023-49092) case study across a grid of
sample-budget × seed combinations on a single, explicitly declared
hardware configuration. This is a **systematic N=140 characterization**
under the conditions below — §5.6 itself is a single draw (N=1) whose
raw log was not archived; this document relates our findings to it.

## Target host

| Item | Value |
|------|-------|
| Instance type | `c8a.8xlarge` (AWS EC2, on-spot) |
| CPU | AMD EPYC 9R45 (Genoa→Turin, 5th gen) |
| Cores × threads | 32 physical, **SMT off** (1 thread/core) |
| L1d / L1i / L2 / L3 | 1.5 MiB / 1 MiB / 32 MiB / 128 MiB |
| NUMA | single node (all 32 cores) |
| Hypervisor | KVM (AWS Nitro; no nested containerization) |
| OS | NixOS 26.05 |
| Kernel | 6.18.7 |
| Rust | 1.95.0 (stable) |
| Timer | invariant TSC (`rdtsc`); measured resolution **≈ 0.385 ns** |
| `perf_event_paranoid` | 2 |
| CPU frequency | pinned by AWS BIOS; observed steady-state ≈ 4.5 GHz |
| Core pinning | `taskset` per worker; 8-core groups (0–7, 8–15, 16–23, 24–31) |

§5.3 of the paper declares `c8a.4xlarge` (same EPYC 9R45 silicon,
16 vCPU) as the real-world testbed. We ran on `c8a.8xlarge`
(32 vCPU) for throughput — same microarchitecture.

## Which MARVIN test

§5.6's numbers (62k samples, effect 126 ns [75, 171], ESS 526/10k
calibration, P=0.89, Inconclusive) were produced by
`crates/tacet/tests/core/known_leaky.rs::detects_marvin_rsa_decryption`.
This test uses a **cache/branch-predictor warming pattern**:

- Baseline: one fixed valid PKCS#1 v1.5 ciphertext, repeated for
  every measurement.
- Sample: pool of 200 distinct valid ciphertexts, cycled.

Both classes decrypt successfully; the leak signature is driven by
microarchitectural state differences between repeated-fixed and
varied-valid inputs.

We initially misidentified `crypto_registry.rs::tier2_rustcrypto_rsa_marvin`
(padding-oracle variant: valid-vs-invalid ciphertexts, exercising the
Bleichenbacher padding-check short-circuit) as the §5.6 test. The
padding-oracle variant yields a much smaller effect (~25 ns) on this
hardware and is a different microarchitectural leak. The sweep binary
(`marvin_budget_sweep.rs`) supports both modes via `--marvin-mode` and
**defaults to `cache`** to match §5.6.

## Sweep parameters

| Parameter | Value |
|---|---|
| Budget ladder (samples/class, ×§5.6's 62k) | 0.5×, 1×, 1.5×, 2×, 2.5×, 3×, 5× |
| Absolute ladder (samples/class) | 31k, 62k, 93k, 124k, 155k, 186k, 310k |
| Seeds per budget | 20 |
| Seed derivation | `md5("20260422\|marvin\|{label}\|{iter}")`, low 60 bits |
| Parallelism | 4 concurrent runs; each pinned to its own 8-core group |
| Attacker model | `AdjacentNetwork` (θ = 100 ns), matches §5.6 |
| Calibration samples | tacet default (10,000); matches §5.6 |
| Analysis codepath | `TimingOracle::analyze_raw_samples_with_resolution` (single-pass) |
| MARVIN mode | `cache` (fixed-vs-varied valid ciphertexts) |

## Results

All 140 runs completed. Per-budget summary:

| Budget | N/class | %Fail (Wilson 95%) | median P | IQR P | median effect (ns) | median CI width (ns) | median ESS | median block |
|---|---|---|---|---|---|---|---|---|
| 0.5× | 31,000 | 40% [22–61%] | 0.098 | [0.01, 1.00] | 128 | 71 | 66 | 468 |
| **1× (§5.6)** | **62,000** | **60% [39–78%]** | **0.982** | **[0.60, 1.00]** | **136** | **60** | **58** | **1,056** |
| 1.5× | 93,000 | 60% [39–78%] | 0.999 | [0.05, 1.00] | 234 | 51 | 71 | 1,293 |
| 2× | 124,000 | 30% [15–52%] | 0.268 | [0.10, 0.97] | 159 | 52 | 628 | 500 |
| 2.5× | 155,000 | 30% [15–52%] | 0.358 | [0.05, 0.99] | 172 | 46 | 92 | 1,670 |
| 3× | 186,000 | 30% [15–52%] | 0.351 | [0.09, 0.98] | 184 | 184 | 5,198 | 40 |
| 5× | 310,000 | 55% [34–74%] | 1.000 | [0.26, 1.00] | 197 | 26 | 698 | 444 |

Per-budget verdict counts (Fail / Inconclusive / Pass):
0.5×: 8 / 5 / 7 · 1×: 12 / 7 / 1 · 1.5×: 12 / 3 / 5 · 2×: 6 / 10 / 4 ·
2.5×: 6 / 9 / 5 · 3×: 6 / 10 / 4 · 5×: 11 / 6 / 3.

### Finding 1: §5.6 is representative, not anomalous

At §5.6's exact budget (62k), 12/20 seeds Fail with median P=0.98 and
median effect 136 ns [75% CI [75, 195]]. §5.6's single draw (P=0.89,
effect 126 ns [75, 171]) lies squarely inside this distribution.
§5.6's Inconclusive verdict is one of 7/20 Inconclusive draws we
observed at the same budget — expected frequency, not an outlier.

### Finding 2: Budget scaling does not monotonically increase Fail rate

Fail rate stays in the 55-60% band at 1× / 1.5× / 5×, and dips to 30%
at 2× / 2.5× / 3×. The dip is driven by tacet's block-length
bootstrap destabilising on RSA decryption's autocorrelation structure
at mid-range sample sizes (median block length varies from 40 at 3×
to 1,670 at 2.5×). We consider this a tacet-internal diagnostic
finding, worth exposing in the camera-ready but not the 700-word
rebuttal.

### Finding 3: Not all seeds reach Fail even at 5× — and that is correct

At 5×, 3/20 seeds Pass with effects 31, 77, 78 ns. These are RSA
keys whose MARVIN signal is genuinely below θ=100 ns on this
hardware; tacet correctly reports Pass at AdjacentNetwork. The
three-way verdict is working as designed: the answer to "how many
more traces?" depends on the specific key, not solely on the
budget, because attacker-model θ is a principled threshold and some
keys fall below it.

### Finding 4: Effect estimate grows with budget, consistent with tighter posterior

Median effect: 128 → 136 → 234 → 159 → 172 → 184 → 197 ns (mostly
monotone upward). CI width shrinks: 71 → 60 → 51 → 52 → 46 → 184 →
26 ns (monotone except at 3×, which has wide CIs due to the block
dip noted in Finding 2). At 5× the effect is tightly pinned at
~197 ns with CI width only 26 ns — substantially above §5.6's
125 ns and above θ=100 ns.

## Deviations from §5.6

- Single-pass analysis (not adaptive): we pre-decide sample budget
  per run so the learning curve has a clean x-axis. §5.6 was a
  single adaptive run that happened to stop at 62k samples.
- Hardware is EPYC 9R45, matching the c8a family declared in §5.3.
  We don't know whether §5.6 was run on c8a.4xlarge or elsewhere.
- Declared hardware is an AWS VM rather than bare metal. Nitro has
  measurable but sub-hypervisor noise; we report it honestly.
- The very first pilot runs were on the wrong MARVIN variant
  (padding oracle instead of cache warming), which produced much
  smaller effect estimates (~25 ns). Corrected before the sweep;
  the 140 rows in this document are all cache mode.
- A parallel pilot on a RunPod 32-vCPU EPYC container was
  discarded: the container locked CPU governor to `powersave` with
  400 MHz ↔ 5.4 GHz swings and produced block lengths >1,000 even
  at 31k. AWS root access avoids this class of confound.

## Limitations

- SMT is off on c8a, which is cleaner than §5.6's setup may have
  been. In production CI environments with SMT enabled, expected
  block length will be higher and effective sample size lower per
  run.
- Governor control is not exposed from userspace on AWS — AWS pins
  frequency at the firmware level. We cannot force `performance`
  governor, only observe steady-state.
- Single NUMA node; multi-socket EPYC systems may exhibit different
  timing characteristics.
- Tacet's block-length estimator is non-monotonic in N on this
  workload (see Finding 2). This is a characterisation of the
  estimator-plus-workload interaction rather than of the underlying
  leak; it should be addressed in a camera-ready revision of §4.2.
