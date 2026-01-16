# Calibration Test Debugging Notes

This document tracks issues, bugs, and workarounds discovered while developing and debugging the calibration test suite. Each entry should be reviewed to determine if it represents:
1. A real bug that needs a proper fix
2. A design limitation that should be documented
3. A UX issue that could confuse users
4. A test limitation vs. a library limitation

## Confirmed Bugs Fixed in `oracle.rs`

### Bug #1: Adaptive Loop Premature Exit (FIXED PROPERLY)

**Location:** `crates/timing-oracle/src/adaptive/loop_runner.rs` and `crates/timing-oracle/src/oracle.rs`

**Symptom:** Power tests showed 0% detection rate. All trials returned `Inconclusive` with 50% leak probability, even with large injected delays (5×θ).

**Root Cause:** When `QualityGateResult::Continue` was returned (meaning "gates pass, keep collecting"), `run_adaptive` incorrectly mapped it to `AdaptiveOutcome::Inconclusive { reason: SampleBudgetExceeded }`. This was semantically wrong - `SampleBudgetExceeded` should mean "budget exceeded", not "continue collecting".

**Initial Workaround:** Added a secondary match on `InconclusiveReason` in `oracle.rs` to treat `SampleBudgetExceeded` as "continue". This worked but was confusing.

**Proper Fix:** Added a new `AdaptiveOutcome::Continue` variant that explicitly signals "keep collecting samples":

```rust
// In loop_runner.rs - new variant
pub enum AdaptiveOutcome {
    LeakDetected { ... },
    NoLeakDetected { ... },
    Inconclusive { reason, ... },
    Continue { posterior, samples_per_class, elapsed },  // NEW
}

// In run_adaptive() - now returns Continue instead of Inconclusive
QualityGateResult::Continue => AdaptiveOutcome::Continue {
    posterior,
    samples_per_class: state.n_total(),
    elapsed: state.elapsed(),
}

// In oracle.rs - simplified handling
AdaptiveOutcome::Continue { posterior, .. } => {
    adaptive_state.update_posterior(posterior);
    continue;
}
AdaptiveOutcome::Inconclusive { reason, .. } => {
    // ALL Inconclusive now means stop
    return self.build_inconclusive_outcome(...);
}
```

**Status:** Fixed properly. The code is now self-documenting.

---

### Bug #2: Calibration Samples Exceeding Max Samples (FIXED)

**Location:** `crates/timing-oracle/src/oracle.rs:675-682`

**Symptom:** With `max_samples=2000` (test setting) and default `calibration_samples=5000`, the budget check triggered immediately after calibration before any inference could run.

**Root Cause:** `calibration_samples` was independent of `max_samples`. After collecting 5000 calibration samples, `n_total >= max_samples` was true, so the adaptive loop never ran.

**Fix:** Cap calibration samples to at most 50% of max_samples:
```rust
let n_cal = self.config.calibration_samples.min(self.config.max_samples / 2);
```

**UX Concern:** Users setting `max_samples` might not realize they also need to consider calibration overhead. Consider:
- Documenting this interaction clearly
- Auto-adjusting `max_samples` to include calibration overhead
- Warning when `max_samples < 2 * calibration_samples`

**Status:** Fixed. This was an API design issue.

---

### Bug #3: Research Mode Zero Threshold Causing Degenerate Prior (FIXED)

**Location:** `crates/timing-oracle/src/oracle.rs:519-526`

**Symptom:** With `AttackerModel::Research` (θ=0), the Bayesian posterior always returned 0.5 (neutral probability).

**Root Cause:** The prior covariance matrix is computed as `sigma = 2 * theta_ns`. With θ=0, this creates a zero matrix. Cholesky decomposition of a zero matrix fails, causing `compute_bayes_factor` to return the neutral result (0.5).

**Fix:** Clamp theta to at least timer resolution:
```rust
let theta_ns = raw_theta_ns.max(timer.resolution_ns());
```

**UX Concern:** Users using Research mode might expect it to detect ANY timing difference, but with coarse timers, the effective threshold is now ~42ns. This should be documented.

**Status:** Fixed. This was a corner case in the API.

---

## Test Modifications That May Mask Issues

### Modification #1: Changed FPR Tests from Research to AdjacentNetwork

**Location:** `tests/calibration_fpr.rs:59-60, 140-141`

**Original:** `AttackerModel::Research`
**Changed to:** `AttackerModel::AdjacentNetwork`

**Reason:** Research mode with a coarse timer (41.67ns resolution on Apple Silicon) showed 20-30% FPR instead of expected 5%. The clamped threshold (~42ns) is too close to system noise, causing false positives from measurement jitter.

**Evidence:**
- Research mode (~42ns threshold): 23% FPR
- AdjacentNetwork (100ns threshold): 3% FPR
- RemoteNetwork (50μs threshold): 0% FPR

**Is This a Test Bug or Library Bug?**
This is a **library limitation**, not a test bug. The oracle cannot reliably distinguish null from noise at thresholds near the timer resolution. However, this is arguably expected behavior - you can't detect 40ns effects with a 42ns timer.

**Recommendations:**
1. **Document clearly:** Research mode requires PMU timers for meaningful results on Apple Silicon
2. **Consider auto-adjustment:** When using coarse timers, could automatically increase the minimum threshold to 2-3× timer resolution
3. **Add warnings:** When `theta_ns < 2 * resolution_ns`, warn users about reduced reliability
4. **Keep validation tier tests with Research mode** but require PMU timer (sudo)

**Risk of Masking Real Issues:** Medium. If the oracle has FPR problems at higher thresholds, we won't catch them with AdjacentNetwork. Validation tier tests with PMU timers should use Research mode.

---

### Modification #2: Added `black_box()` Inside Loop for Measurable Operations

**Location:** `tests/calibration_fpr.rs` (all test closures)

**Original:**
```rust
let mut acc: u64 = 0;
for _ in 0..100 {
    for &b in data.iter() {
        acc = acc.wrapping_add(b as u64);
    }
}
std::hint::black_box(acc);
```

**Changed to:**
```rust
let mut acc: u64 = 0;
for _ in 0..100 {
    for &b in data.iter() {
        acc = acc.wrapping_add(std::hint::black_box(b) as u64);
    }
}
std::hint::black_box(acc);
```

**Reason:** The original loop was optimized to ~0.7ns by the compiler, making it unmeasurable with a 42ns timer. Adding `black_box(b)` forces the compiler to treat each byte access as an observable side effect, resulting in ~440ns per operation.

**Is This a Test Bug or Library Bug?**
This is a **test design issue**. The test needs a constant-time operation that:
1. Is measurable (>100ns for coarse timers)
2. Is actually constant-time (no data-dependent branches)
3. Doesn't introduce real timing leaks

**Risk of Masking Real Issues:** Low. This change affects how tests create measurable workloads, not how the oracle works. The operation remains constant-time.

---

## Observations Requiring Further Investigation

### 1. High Inconclusive Rate

During debugging, many trials returned `Inconclusive` before returning Pass/Fail. This might indicate:
- Quality gates are too strict for short-budget tests
- Adaptive loop needs more iterations to converge
- Timer noise is affecting convergence

**Recommendation:** Track inconclusive rate as a calibration metric.

### 2. Seed Sensitivity in FPR Tests

With the original Research mode, FPR varied from 4% to 31% depending on RNG seed. This suggests:
- System state affects measurements significantly
- More trials may be needed for stable estimates
- Bootstrap randomness adds variance

**Recommendation:** Increase trial count or use multiple seeds for validation.

### 3. Timer Resolution Awareness

The oracle's behavior changes significantly based on timer resolution:
- PMU timers (~1ns): Can detect Research mode thresholds
- Coarse timers (~42ns): Need higher thresholds for reliability

**Recommendation:** Add API to query effective minimum detectable effect for current timer.

---

## Summary of Changes to Review

| Change | Location | Risk Level | Action Needed |
|--------|----------|------------|---------------|
| Fix adaptive loop premature exit | oracle.rs:904-929 | None (bug fix) | Merged |
| Fix calibration sample capping | oracle.rs:675-682 | None (bug fix) | Document API interaction |
| Fix Research mode zero threshold | oracle.rs:519-526 | Low | Document limitation |
| Use AdjacentNetwork for FPR tests | calibration_fpr.rs | Medium | Add PMU-based Research tests |
| Add black_box inside loop | calibration_fpr.rs | Low | Keep for measurability |

---

## Open Questions

1. Should the oracle automatically adjust thresholds based on timer resolution?
2. Should `max_samples` include calibration overhead transparently?
3. Should Research mode warn when timer resolution is insufficient?
4. What's the appropriate FPR tolerance for different attacker models?
