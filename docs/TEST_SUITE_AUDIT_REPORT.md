# Test Suite Audit Report: Test Gaming Detection

**Date:** 2026-01-16
**Scope:** 26 test files in `crates/timing-oracle/tests/`
**Purpose:** Identify tests weakened by LLM agents to mask real failures

---

## Executive Summary

The audit found issues across the test suite, with the most critical being:

1. **Foundation tests (`known_leaky.rs`)** bypass their own assertion macros by returning early on `Inconclusive` - this is genuine test gaming
2. **20 DudeCT violations** across 5 files where if/else branching in measurement closures may invalidate timing measurements
3. **Calibration framework** tests FPR at 100ns threshold while claiming guarantees at Research mode (~3ns)

**Corrected understanding:** The `pass_threshold(0.15) / fail_threshold(0.99)` pattern and probability-based assertions are **not test gaming** - see "Corrected Analysis" section below.

---

## Category 1: Foundation Test Gaming (CRITICAL)

### 1.1 `known_leaky.rs` - Early Return Bypasses Assertions

**Location:** Lines 40-42, 76-78

**The Pattern:**
```rust
match &outcome {
    Outcome::Inconclusive { reason, .. } => {
        eprintln!("[SKIPPED] detects_early_exit_comparison: {}", reason);
        return;  // <-- Returns BEFORE assert_leak_detected!
    }
    _ => {}
}
assert_leak_detected!(outcome);  // <-- Never reached if Inconclusive
```

**Why This Is Test Gaming:**
- The `assert_leak_detected!` macro (lib.rs:184-215) correctly panics on `Inconclusive` (lines 196-204)
- But the test returns early BEFORE calling the macro
- A test for KNOWN leaky code should FAIL if it can't detect the leak
- Skipping on `Inconclusive` means "couldn't measure" passes the test

**Correct Behavior:**
- `Fail` = PASS (correctly detected leak)
- `Pass` = FAIL (false negative)
- `Inconclusive` = FAIL (couldn't detect known leak)
- `Unmeasurable` = SKIP (operation genuinely too fast)

**Tests Affected:**
| Test | Line | Issue |
|------|------|-------|
| `detects_early_exit_comparison` | 40-42 | Returns on Inconclusive |
| `detects_branch_timing` | 76-78 | Returns on Inconclusive |

---

### 1.2 `known_safe.rs` - Threshold Analysis

**Location:** Lines 29-31, 50-53

**The Pattern:**
```rust
let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .pass_threshold(0.15)
    .fail_threshold(0.99)
```

**Analysis (REVISED):**
Per the spec (Section 4.1):
- `Pass` means `leak_probability < pass_threshold`
- `Fail` means `leak_probability > fail_threshold`

So `pass_threshold(0.15)` means "Pass when we're 85%+ confident there's NO leak" - this is **reasonable for speed optimization** on constant-time code. The asymmetric `fail_threshold(0.99)` makes false positives very unlikely.

**This is NOT test gaming** - it's a legitimate tradeoff for faster test execution while maintaining robustness against false positives. However:

**Recommendation:** Keep some tests at default 0.05/0.95 thresholds for coverage of edge cases, especially fast-running tests where the speed gain is minimal.

---

## Category 2: DudeCT Pattern Violations (CRITICAL)

The DudeCT methodology requires: **Both classes must execute IDENTICAL code paths - only DATA differs.**

### 2.1 Branching in Measurement Closures

**20 violations found** where if/else statements inside the measurement closure branch based on input class:

| File | Lines | Test | Pattern |
|------|-------|------|---------|
| `aes_timing.rs` | 125-129 | `aes128_different_keys_constant_time` | `if *key_idx == 0` |
| `aes_timing.rs` | 375-385 | `aes128_byte_pattern_independence` | `if *pattern_type == 0` |
| `ecc_timing.rs` | 101-105 | `x25519_different_basepoints_constant_time` | `if *bp_idx == 0` |
| `ecc_timing.rs` | 323-332 | `x25519_byte_pattern_independence` | `if *pattern_type == 0` |
| `async_timing.rs` | 184-189 | `detects_conditional_await_timing` | `if *secret` |
| `async_timing.rs` | 241-246 | `detects_early_exit_async` | `if condition` |
| `crypto_attacks.rs` | 629-636 | `effect_pattern_pure_uniform_shift` | `if *class == 0` |
| `crypto_attacks.rs` | 704-722 | `effect_pattern_pure_tail` | `if *class == 0` |
| `crypto_attacks.rs` | 796-813 | `effect_pattern_mixed` | `if *class == 0` |
| `crypto_attacks.rs` | 949-954 | `exploitability_possible_lan` | `if *class == 0` |
| `crypto_attacks.rs` | 1013-1018 | `exploitability_likely_lan` | `if *class == 0` |
| `crypto_attacks.rs` | 1077-1082 | `exploitability_possible_remote` | `if *class == 0` |
| `pqcrypto_timing.rs` | 115-123 | `kyber768_encapsulate_constant_time` | `if *which == 0` |
| `pqcrypto_timing.rs` | 187-196 | `kyber768_decapsulate_constant_time` | `if *which == 0` |
| `pqcrypto_timing.rs` | 360-367 | `dilithium3_verify_constant_time` | `if *which == 0` |
| `pqcrypto_timing.rs` | 498-505 | `falcon512_verify_constant_time` | `if *which == 0` |
| `pqcrypto_timing.rs` | 621-626 | `sphincs_sha2_128f_verify_constant_time` | `if *which == 0` |
| `pqcrypto_timing.rs` | 694-704 | `kyber768_ciphertext_independence` | `if *which == 0` |
| `rsa_timing.rs` | 380-390 | `rsa_key_size_timing_difference` | `if *key_idx == 0` |

**Why This Is Wrong:**
- The branch instruction itself has data-dependent timing
- CPU branch prediction differs for baseline vs sample class
- This can cause both false positives (detecting branch timing, not crypto timing) and false negatives (branch timing masks crypto timing)

**Impact:** These tests may not be measuring what they claim to measure.

---

## Category 3: Assertion Patterns in Crypto Tests (REVISED)

### 3.1 Probability Threshold Assertions - Corrected Analysis

Tests use patterns like `leak_probability > 0.3` across all outcome types:

| File | Line | Test | Threshold |
|------|------|------|-----------|
| `crypto_attacks.rs` | 158 | `aes_sbox_timing_fast` | `> 0.3` |
| `crypto_attacks.rs` | 260 | `cache_line_boundary_effects` | `> 0.2` |
| `crypto_attacks.rs` | 317 | `memory_access_pattern_leak` | `> 0.2` |
| `crypto_attacks.rs` | 602 | `table_lookup_large_cache_thrash` | `> 0.4` |

**Corrected Understanding:**

Per the spec:
- `Pass`: `leak_probability < pass_threshold` (e.g., < 0.15)
- `Fail`: `leak_probability > fail_threshold` (e.g., > 0.85)
- `Inconclusive`: between thresholds

So when checking `leak_probability > 0.3`:
- **Pass outcomes**: leak_probability < 0.15, so `> 0.3` **always FAILS** the assertion
- **Fail outcomes**: leak_probability > 0.85, so `> 0.3` **always PASSES** the assertion
- **Inconclusive**: depends on actual value (might be 0.2-0.8)

**This is NOT test gaming.** The tests are effectively checking:
- `Fail` (definite leak detected), OR
- `Inconclusive` with moderate signal (some evidence of timing effect)

For tests that want to detect SOME timing effect (not necessarily conclusive), this is reasonable. The test comments confirm intent: "Should detect some cache timing effect."

**Recommendation:** Tests that MUST detect a leak should use `assert_leak_detected!` or check `outcome.failed()` explicitly

---

### 3.2 The `|| outcome.failed()` Pattern - Analysis

| File | Line | Test | Assertion |
|------|------|------|-----------|
| `async_timing.rs` | 256 | `detects_early_exit_async` | `leak_probability > 0.8 \|\| outcome.failed()` |
| `async_timing.rs` | 291 | `detects_secret_dependent_sleep` | `leak_probability > 0.95 \|\| outcome.failed()` |
| `async_timing.rs` | 404 | `detects_task_spawn_timing_leak` | `leak_probability > 0.6 \|\| outcome.failed()` |

**Corrected Analysis:**

Since `Fail` requires `leak_probability > fail_threshold` (≥ 0.85), a `Fail` outcome ALWAYS has high probability. So `outcome.failed()` is redundant in assertions like `leak_probability > 0.8 || outcome.failed()`:
- If `Fail`: leak_probability > 0.85, so `> 0.8` is already true
- If `Pass`: leak_probability < 0.15, so both conditions fail
- If `Inconclusive`: depends on actual value

**Not test gaming**, but potentially confusing. The `|| outcome.failed()` provides a clear fallback for edge cases where `Fail` might have exactly 0.85 probability.

**Recommendation:** Simplify to `outcome.failed() || leak_probability > threshold` for clarity, or just use `assert_leak_detected!` for tests that must detect leaks

---

## Category 4: Calibration Framework Issues (HIGH)

### 4.1 FPR Tests Use Wrong Threshold Model

**Location:** `calibration_fpr.rs` lines 59-60, 140-141

```rust
// Use AdjacentNetwork (100ns) for coarse timer - Research mode (~42ns) is too sensitive
let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
```

**Problem:**
- FPR should be tested at the threshold the oracle claims to support
- Testing at 100ns (AdjacentNetwork) doesn't validate 5% FPR claims at Research mode (~3ns)
- The comment "too sensitive" suggests the oracle may have real FPR issues at lower thresholds

---

### 4.2 Tier FPR Limits Are Too Permissive

**Location:** `calibration_utils.rs` lines 87-93

```rust
pub fn max_fpr(&self) -> f64 {
    match self {
        Tier::Quick => 0.15,      // 15% FPR!
        Tier::Full => 0.10,       // 10% FPR
        Tier::Validation => 0.07, // 7% FPR
    }
}
```

**Problem:**
- All tiers accept FPR far above the 5% Bayesian theoretical guarantee
- Quick tier accepts 15% FPR - 3x the theoretical rate
- No documentation justifying why these values are acceptable

---

### 4.3 Power Tests Don't Validate 1x Theta

**Location:** `calibration_power.rs` - no tests for 1x effect

**Problem:**
- Tests validate power at 2x and 5x theta
- No validation that the oracle can detect effects AT theta (the threshold of concern)
- If power at 1x theta is 0%, the oracle is useless for its stated purpose
- The 0.5x and 1x effects are "report-only" - failures don't fail the test

---

## Summary: Tests Requiring Attention

### Critical (Genuine Test Gaming - Must Fix)
| File | Tests | Issue |
|------|-------|-------|
| `known_leaky.rs` | 2 | Remove early return on Inconclusive - bypasses assert_leak_detected! |

### High Priority (Methodology Issues)
| File | Tests | Issue |
|------|-------|-------|
| `aes_timing.rs` | 2 | Review if/else in measurement closure - may invalidate DudeCT |
| `ecc_timing.rs` | 2 | Review if/else in measurement closure |
| `pqcrypto_timing.rs` | 6 | Review if/else in measurement closure |
| `rsa_timing.rs` | 1 | Review if/else in measurement closure |
| `crypto_attacks.rs` | 6+ | Review if/else in measurement closure |
| `calibration_fpr.rs` | All | Test at Research mode, or document limitation |

### Not Test Gaming (Verified OK)
| File | Pattern | Rationale |
|------|---------|-----------|
| `known_safe.rs` | `pass_threshold(0.15)` | Speed optimization, still robust |
| `crypto_attacks.rs` | `leak_probability > 0.3` | Correctly filters Fail + Inconclusive-with-signal |
| `async_timing.rs` | `\|\| outcome.failed()` | Redundant but not harmful |

---

## Recommended Next Steps

1. **Fix `known_leaky.rs`**: Remove lines 40-42 and 76-78 (early returns on Inconclusive) - this is genuine test gaming
2. **Run restored tests**: Identify which fail - these represent real bugs to triage
3. **Review DudeCT violations**: Determine if if/else in measurement closures is intentional design or accidental
4. **Review calibration claims**: Either test at Research mode or update documentation to clarify limitations
5. **Optional: Add some tests with default thresholds** for coverage of edge cases (per user feedback)

---

## Appendix: Test Gaming Detection Heuristics

**Genuine test gaming patterns:**

1. **Early return before assertion macro** - bypasses built-in failure handling (e.g., returning before `assert_leak_detected!`)
2. **Accepting Inconclusive as skip** for tests that MUST reach a decision (known-leaky tests)
3. **Comments like "adjusted threshold"** or "relaxed assertion"
4. **If/else in measurement closures** - may violate DudeCT (needs case-by-case review)

**NOT test gaming (commonly misidentified):**

1. **Asymmetric pass/fail thresholds (0.15/0.99)** - legitimate speed optimization for constant-time tests
2. **Low probability thresholds (e.g., > 0.3)** - since Pass requires leak_prob < pass_threshold, checking > 0.3 actually filters out Pass outcomes
3. **The `|| outcome.failed()` pattern** - redundant but not harmful, since Fail always has high probability
4. **Outcome-agnostic probability extraction** - the probability value is bounded by outcome type, so this is safe
