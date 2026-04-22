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
