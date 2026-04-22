# Camera-ready TODO list

Carry-overs from the USENIX Sec'26 author-response experiments and related
discoveries. These are not rebuttal items — they belong in the revision.

## Injection primitive calibration (discovered 2026-04-22)

- `tacet::helpers::effect::busy_wait_ns` has a **~13 ns per-call overhead**
  and a **~19 ns floor** on x86_64 AMD EPYC @ 5 GHz (measured via
  `crates/tacet/examples/busy_wait_calibration.rs`, n=100,000 × 11 reps).
- Consequence: nominal `d ∈ {1, 2, 3, 5}` all inject **~19 ns actual per
  call**; nominal d=10 injects ~20 ns. Overhead relative to nominal falls
  below 10 % only for d ≥ 100 ns and below 5 % for d ≥ 200 ns.
- `crates/tacet/tests/leaky/injected.rs` contains `injected_shift_{2,5}ns`
  tests that are misleadingly named — they inject ~19 ns, not 2 / 5 ns.
  Options: (a) rename to reflect measured effective delay, (b) add a module
  doc block + a `#[test] fn busy_wait_floor_sanity` that asserts the floor
  so the limitation is visible in the source.
- Any paper figure or table built from these primitives at small d must
  report **measured delay on the x-axis**, not nominal. The `busy_wait_ns`
  calibration CSV should travel with the paper artifact.
- The `crates/tacet/examples/busy_wait_calibration.rs` harness should stay
  in-repo as a reproducible calibration artefact.

## USENIX Sec'26 reviewer-response additions (conditional on acceptance)

- **Related work:** add ShowTime (Rokicki et al.) to `paper.bib`, cite
  once alongside the existing timing-attack primitive list.
- **Discussion §6, after "Concentrated tail effects":** add
  `\paragraph{Operation-loop amplification.}` — dual-capability framing +
  θ = θ_physical / k_adv rule + pointer to overlay table.
- **Possible figure:** overlay plot from `amp_runpod_v2.csv`, amplified
  vs single-op TPR against actual effective delay on shared x-axis. Reuse
  `crates/tacet/scripts/plot_power_curve.py` with a new CSV export mode.

## §5.6 MARVIN: clarify measurement pattern (cache-warming, not padding-oracle)

§5.6's claim is "tacet flagged CVE-2023-49092 during routine testing." The
test code producing those numbers is
`crates/tacet/tests/core/known_leaky.rs::detects_marvin_rsa_decryption`,
which uses a **cache/branch-predictor warming pattern** (baseline = 1 fixed
valid ciphertext repeated; sample = 200 varied valid ciphertexts). **Both
classes decrypt successfully**; the timing difference comes from
microarchitectural state differences between repeated-fixed and
varied-valid inputs.

CVE-2023-49092 / MARVIN (Kario 2023) is specifically about the PKCS#1 v1.5
**padding oracle** — padding-valid vs padding-invalid timing differences.
The cross-tool registry entry at
`crates/tacet-bench/src/crypto_registry.rs::tier2_rustcrypto_rsa_marvin`
implements this padding-oracle pattern (valid vs random-invalid
ciphertexts). On c8a.8xlarge the padding-oracle variant yields a ~25 ns
effect (below θ=100 ns AdjacentNetwork); the cache-warming variant yields
~140 ns, matching §5.6.

Both are real timing-leak signals, but they exercise different
microarchitectural channels and have different exploitability properties.
The §5.6 paragraph should clarify which variant it measured and acknowledge
that a full MARVIN-oracle test on the same hardware yields a different
effect magnitude. Two options:

1. **Rename §5.6's leak** to "RSA-1024 PKCS#1 v1.5 decryption timing
   anomaly" (generic) rather than conflating it with MARVIN's specific
   padding-oracle mechanism. Still triggers a correct CVE investigation
   — the case study's narrative stands.
2. **Add a paragraph** distinguishing "cache-warming leak" from
   "padding-oracle leak" in §5.6, citing both the `known_leaky.rs` test
   and the `crypto_registry.rs` registry entry. Report both variants'
   effect magnitudes and tacet's verdict on each.

**Do NOT raise this distinction in the Apr 23 rebuttal.** It is not a
reviewer-raised concern. §8's sweep data answers the reviewers' questions
regardless of which MARVIN-adjacent channel is being measured.

## Other reviewer asks to address in revision (not all in the 700 words)

- A: hyperparameter sensitivity study (ablation table of the 10+ calibrated
  knobs, not just W1 vs 9-decile).
- A: testbed of known timing-vulnerable CVEs beyond MARVIN; compare tools
  side by side on them.
- A: clarify the SILENT stream-bootstrap characterization (§7.1 of ref [6]
  apparently handles dependent data).
- A: clarify Table 1 — RTLF and SILENT use CSV input, not R-language;
  add Tacet limitations row.
- C: execution-time comparison vs prior tools on crypto libraries.
- C: clarify which microarchitectural attack classes are covered (branch,
  cache, port-contention, etc.).
- D: clarify "cryptographic test case" definition; include one existing-
  tool baseline in §5.2 rather than Fig 1 alone.
- Universal: calibration data representativeness — transferability of
  AR(1) synthetic benchmarks vs measured timing distributions.
