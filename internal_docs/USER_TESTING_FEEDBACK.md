# User Testing Feedback: DudeCT Examples Implementation

This document captures feedback from a simulated "new user" experience implementing a test suite based on [dudect examples](https://github.com/oreparaz/dudect/tree/master/examples), using only the public documentation (not source code).

**Test file created:** `crates/tacet/tests/dudect_examples.rs`

---

## Summary

Overall, the library is well-designed and the core abstractions (attacker models, four-way outcomes, effect decomposition) are genuinely useful. The API is intuitive for basic usage.

**Key strengths:**
- Attacker model abstraction maps threat scenarios to meaningful thresholds
- Honest uncertainty handling (Inconclusive when precision insufficient)
- Rich diagnostic output (effect decomposition, credible intervals, exploitability)

**Main friction points:**
- SharedHardware threshold (0.6ns) is often impractical — measurement floor is ~9ns on Apple Silicon
- Measurement floor concept and threshold elevation need better documentation
- Two-class pattern requires careful thought about what creates timing asymmetry (fixed in docs)
- macOS kperf requires `--test-threads=1` (not documented)
- `Outcome::Research` variant undocumented

---

## Documentation Gaps

### 1. Undocumented `Outcome::Research` Variant

**Issue:** The `Outcome` enum has a `Research` variant that is not documented in `api-rust.md` or `guide.md`. When writing match statements following the documented four-variant pattern, compilation fails:

```rust
// This pattern from the docs doesn't compile:
match outcome {
    Outcome::Pass { .. } => { ... }
    Outcome::Fail { .. } => { ... }
    Outcome::Inconclusive { .. } => { ... }
    Outcome::Unmeasurable { .. } => { ... }
    // ERROR: non-exhaustive patterns: `Outcome::Research(_)` not covered
}
```

**Impact:** Users must add a wildcard `_ => {}` arm without understanding what `Research` means or when it occurs.

**Recommendation:** Document `Outcome::Research` in the API reference, or make the enum `#[non_exhaustive]` if new variants are expected.

---

### 2. Attacker Model Default Recommendation

**Issue:** The documentation examples consistently use `AttackerModel::AdjacentNetwork` (100ns threshold), which may be too lenient for cryptographic code testing.

From `guide.md`:
```rust
// The default example uses AdjacentNetwork
TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
```

**Concern:** For cryptographic implementations (AES, ECC, RSA), `SharedHardware` (~0.6ns / ~2 cycles) is more appropriate since:
- Crypto code often runs in environments with co-resident attackers (containers, VMs, SGX)
- Cycle-level leaks in crypto are well-documented attack vectors
- The dudect examples this test suite replicates are specifically for crypto timing validation

**Recommendation:** Consider recommending `SharedHardware` as the default for crypto testing in the documentation, with `AdjacentNetwork` for API/network service testing.

**Example guidance:**
```rust
// For cryptographic code (AES, ECC, RSA, etc.)
TimingOracle::for_attacker(AttackerModel::SharedHardware)

// For network APIs and services
TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
```

---

## API Friction

### 3. x25519-dalek `StaticSecret` Requires Feature Flag

**Issue:** The `x25519-dalek` crate requires the `static_secrets` feature to expose `StaticSecret`, but the dev-dependencies don't enable it:

```toml
# Current in Cargo.toml
x25519-dalek = "2.0"

# Would need
x25519-dalek = { version = "2.0", features = ["static_secrets"] }
```

**Workaround used:** Used `PublicKey::from([u8; 32])` instead, which performs scalar multiplication with the basepoint (equivalent to what dudect's donna example does).

**Impact:** Minor — the workaround is actually more accurate to the dudect example. But users trying to test full ECDH flows would hit this.

---

## Unmeasurable Message Clarity

### 4. Current Message Assessment

**Actual output:**
```
SKIPPED: Operation too fast to measure
  Recommendation: Run with sudo to enable kperf cycle counting (~0.3ns resolution),
                  or increase operation complexity
```

**Rating: 6/10** — Functional but could be more educational.

#### What works well:
- Clear that the operation was too fast
- Actionable recommendation ("Run with sudo")
- Mentions the resolution improvement (~0.3ns)

#### Issues identified:

| Issue | Description | Suggestion |
|-------|-------------|------------|
| **Jargon** | "kperf" is Apple-specific and unfamiliar | Use "hardware cycle counter" or "PMU" |
| **Vague guidance** | "increase operation complexity" is not actionable | Suggest specific approaches (batch iterations, larger inputs) |
| **Missing context** | Doesn't show actual vs threshold timing | Include: "Operation: ~15ns, Minimum measurable: ~42ns" |
| **No explanation** | Doesn't explain why sudo is needed | Add: "PMU access requires elevated privileges" |
| **Platform-specific** | Message assumes macOS | Use platform-agnostic language with platform-specific details |

#### Suggested improved message:

```
UNMEASURABLE: Operation too fast for current timer resolution

  Measured:    ~15 ns per operation
  Required:    >42 ns (current timer resolution)

  Options:
  1. Enable hardware cycle counter (requires sudo/root):
     - macOS: Enables kperf PMU (~0.3ns resolution)
     - Linux: Enables perf_event (~0.3ns resolution)

  2. Increase operation complexity:
     - Use larger input sizes
     - Batch multiple operations together
```

---

## Positive Observations

### What the library does well:

1. **Four-way outcome pattern** — Handling Pass/Fail/Inconclusive/Unmeasurable explicitly is much better than boolean pass/fail. Forces users to think about edge cases.

2. **Effect decomposition** — Breaking down leaks into shift + tail components with credible intervals is genuinely useful:
   ```
   Effect: 690.4 ns shift, 82.9 ns tail
   ```

3. **Exploitability classification** — Mapping effect sizes to attack feasibility (SharedHardwareOnly, Http2Multiplexing, StandardRemote, ObviousLeak) helps users understand real-world impact.

4. **Helpful Fail output** — When a leak is detected, the output includes everything needed to assess severity:
   ```
   EXPECTED FAIL: Detected timing leak in naive scalar mult
     Leak probability: 100.0%
     Exploitability: StandardRemote
     Effect: 690.4 ns shift, 82.9 ns tail
   ```

5. **DudeCT two-class pattern** — The `InputPair::new(|| zeros, || random)` pattern is well-documented and matches the established methodology.

6. **Builder pattern** — Configuration is clear and discoverable:
   ```rust
   TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
       .time_budget(Duration::from_secs(30))
       .test(inputs, |data| { ... })
   ```

---

## Test Results Summary

| Test | DudeCT Equivalent | Expected | Actual | Notes |
|------|-------------------|----------|--------|-------|
| `test_memcmp_constant_time` | `simple` (fixed) | Pass | **Pass** | `subtle::ct_eq` works |
| `test_aes128_constant_time` | `aesbitsliced` | Pass | **Pass** | AES-NI constant-time |
| `test_x25519_constant_time` | `donna` | Pass | **Unmeasurable** | Too fast without PMU |
| `test_memcmp_leaky` | `simple` | Fail | **Unmeasurable** | Too fast without PMU |
| `test_table_lookup_leaky` | `aes32` | Fail | **Unmeasurable** | Too fast without PMU |
| `test_naive_scalar_mult_leaky` | `donnabad` | Fail | **Fail** | 690ns leak detected |

**Note:** Tests were run without `sudo`, so PMU timing was unavailable on Apple Silicon. The `naive_scalar_mult` test was slow enough to measure with the standard timer.

### With `sudo -E` (PMU should be available)

| Test | Expected | Actual | Notes |
|------|----------|--------|-------|
| `test_memcmp_constant_time` | Pass | **Inconclusive** | ThresholdElevated (100ns → 172ns) |
| `test_aes128_constant_time` | Pass | **Pass** | Works correctly |
| `test_x25519_constant_time` | Pass | **Unmeasurable** | Still says "run with sudo" |
| `test_memcmp_leaky` | Fail | **Unmeasurable** | Still says "run with sudo" |
| `test_table_lookup_leaky` | Fail | **Unmeasurable** | Still says "run with sudo" |
| `test_naive_scalar_mult_leaky` | Fail | **Fail** | 690ns leak detected |

**Issue discovered:** Running with `sudo -E` does NOT automatically enable kperf. Three tests still report "Unmeasurable" with the recommendation to "Run with sudo" — but we ARE running with sudo. This is confusing.

The system should either:
1. Automatically detect and use kperf when privileges are available, or
2. Provide a different message explaining why kperf isn't activating despite elevated privileges

---

## Investigation Findings

### 5. Rust `==` Compiles to `memcmp` (Constant-Time Behavior)

**Investigation:** Why did `test_memcmp_leaky` pass when using Rust's `==` operator on slices?

**Finding:** The Rust compiler optimizes slice equality to a call to libc `memcmp`:

```asm
; Assembly for naive_byte_compare(a, b) which does `a == b`
bl _memcmp           ; Calls libc memcmp
cmp w0, #0           ; Compare result to 0
cset w0, eq          ; Set return value
```

On Apple Silicon, `memcmp` is highly optimized and may use SIMD comparisons that exhibit near-constant-time behavior for the sizes tested (512 bytes). This is platform-dependent.

**Solution:** Created `explicit_early_exit_compare` with a hand-written loop that definitely has early-exit behavior. This function correctly detected as leaky.

### 6. Test Design Flaw in Early-Exit Comparison

**Original bug:** The test used `secret = [0x42u8; 512]` (all bytes are 0x42):

```rust
let secret = [0x42u8; 512];  // All bytes are 0x42
let inputs = InputPair::new(
    || [0u8; 512],      // Baseline: all zeros
    || rand::random(),  // Sample: random bytes
);
```

When comparing against secret `[0x42; 512]`:
- **Baseline** `[0x00; 512]`: First byte is `0x00 ≠ 0x42` → exits on byte 0
- **Sample** (random): First random byte is almost certainly `≠ 0x42` → also exits on byte 0

**Both inputs exit on the first byte!** No timing difference to detect.

**Fix:** Changed secret to match baseline so there's asymmetric early-exit behavior:

```rust
let secret = [0u8; 512];  // All zeros - matches baseline entirely
let inputs = InputPair::new(
    || [0u8; 512],      // Baseline: matches → checks all 512 bytes (SLOW)
    || rand::random(),  // Sample: first mismatch → immediate exit (FAST)
);
```

**Result:** After fix, the test correctly detects a **2.5 μs timing leak** with 100% probability.

### 7. S-Box Cache Timing Not Detectable (Expected Behavior)

**Investigation:** Why did `test_table_lookup_leaky` pass when S-box lookups should have cache timing?

**Finding:** The 256-byte AES S-box fits entirely in L1 cache on modern CPUs:
- Apple M1/M2 L1 data cache: 128–192 KB
- AES S-box: 256 bytes
- After a few warmup iterations, all 256 entries are cached in L1
- All lookups hit L1 regardless of index → no timing difference

**Follow-up experiment with larger table:**

| Table Size | Attacker Model | Threshold | Result | Notes |
|------------|----------------|-----------|--------|-------|
| 256 bytes (S-box) | AdjacentNetwork | 100 ns | **Pass** | No detectable effect (0.1–1.9 ns) |
| 64 KB | AdjacentNetwork | 100 ns | **Pass** | ~8 ns effect, below threshold |
| 64 KB | SharedHardware | 0.6 ns → 9.3 ns | **Inconclusive** | Measurement floor too high |
| 64 KB | Custom 5 ns | 5 ns → 7.9 ns | **Fail (97.4%)** | Detects ~8 ns cache effect |

The 64KB table shows ~8ns cache timing effect. Whether this is flagged as a leak depends on threat model:
- **AdjacentNetwork (100ns)**: Pass — not exploitable over network
- **SharedHardware (0.6ns)**: Inconclusive — measurement floor (~9ns) exceeds threshold
- **Custom 5ns**: Fail — detects the effect as exploitable for shared-hardware scenarios

**Conclusion:** This is **expected behavior** and demonstrates the library working correctly. The same code can be "safe" or "leaky" depending on your threat model. The attacker model isn't just about sensitivity — it's a statement about what timing differences matter for your deployment context.

---

## Updated Test Results (with PMU)

Using `sudo -E cargo test --test dudect_examples -- --nocapture --test-threads=1`:

| Test | DudeCT Equivalent | Expected | Actual | Notes |
|------|-------------------|----------|--------|-------|
| `test_memcmp_constant_time` | `simple` (fixed) | Pass | **Pass** | `subtle::ct_eq` works |
| `test_aes128_constant_time` | `aesbitsliced` | Pass | **Pass** | AES-NI constant-time |
| `test_x25519_constant_time` | `donna` | Pass | **Pass** | x25519-dalek constant-time |
| `test_memcmp_leaky` | `simple` | Fail | **Pass** | memcmp SIMD-optimized |
| `test_explicit_early_exit_leaky` | `simple` | Fail | **Fail** | 2.5 μs leak detected |
| `test_table_lookup_leaky` | `aes32` | Fail | **Pass** | 256-byte S-box fits in L1 |
| `test_large_table_lookup_leaky` | `aes32` | Fail | **Fail (97.4%)** | 8 ns effect (Custom 5ns threshold) |
| `test_naive_scalar_mult_leaky` | `donnabad` | Fail | **Fail** | 690 ns leak detected |

**Key insight:** PMU requires `--test-threads=1` due to concurrent access limitations on kperf.

---

## Reflections on the Full Experience

### What Worked Well

1. **The attacker model abstraction is genuinely useful.** The cache timing experiment showed the same ~8ns effect being Pass/Inconclusive/Fail depending on threat model. This isn't just sensitivity tuning — it's a meaningful statement about deployment context.

2. **Honest uncertainty handling.** When SharedHardware's 0.6ns threshold couldn't be achieved, the library returned Inconclusive rather than a false positive. This is the right behavior.

3. **Research mode is invaluable for exploration.** Using `AttackerModel::Research` to characterize an effect before choosing a threshold was the right workflow, even though I discovered it by accident.

4. **The output is information-rich.** Effect decomposition (shift + tail), credible intervals, exploitability heuristics, and quality diagnostics all helped me understand what was happening.

### New Friction Points Discovered

#### 8. SharedHardware Threshold Is Often Impractical

**Issue:** `SharedHardware` (0.6ns / ~2 cycles) sounds like the right choice for crypto code, but on Apple Silicon even with kperf, the measurement floor is ~9ns. The threshold gets elevated 15x, and most tests return Inconclusive.

**User expectation:** "I'll use SharedHardware for my SGX/container crypto code."
**Reality:** SharedHardware is only achievable on specific hardware with very low noise floors.

**Recommendation:**
- Document when SharedHardware is actually achievable (specific platforms, conditions)
- Consider adding a preset like `SharedHardwareRealistic` at ~10ns for users who want cycle-level sensitivity but can't achieve 0.6ns
- Or guide users toward Custom thresholds with a workflow for choosing them

#### 9. Measurement Floor Concept Needs Documentation

**Issue:** I didn't understand "measurement floor" until I saw threshold elevation in action. The library has a minimum detectable effect based on timer resolution and sample variance — this fundamentally limits what thresholds are achievable.

**What I learned:**
- Threshold elevation isn't a bug, it's honest reporting of measurement limits
- You can't just pick any threshold — it must be above the measurement floor
- The measurement floor varies by platform, timer, and even the specific operation

**Recommendation:** Add a section to the guide explaining:
- What the measurement floor is and why it exists
- How to discover your measurement floor (run with Research mode, look at the CI)
- How to choose a realistic Custom threshold based on your floor

#### 10. Workflow for Choosing Thresholds

**Issue:** There's no documented workflow for users who find that preset attacker models don't fit their needs.

**Discovered workflow:**
1. Run with `AttackerModel::Research` to characterize the effect and see the measurement floor
2. Look at the 95% CI and "measurement floor" in the output
3. Choose a `Custom { threshold_ns }` that's:
   - Above your measurement floor (so you get conclusive results)
   - Appropriate for your threat model (what timing differences actually matter?)

**Recommendation:** Document this workflow explicitly. Many users will need Custom thresholds.

#### 11. Threshold Elevation Warning Could Be More Prominent

**Issue:** When SharedHardware (0.6ns) gets elevated to 9.3ns, a small warning appears in "Quality Issues." But this 15x increase fundamentally changes what you're testing — it deserves more prominence.

**Current output:**
```
⚠ Quality Issues
  • ThresholdElevated: Threshold elevated from 1 ns to 9.3 ns (measurement floor)
```

**Suggestion:** When elevation is >2x, consider:
- Making this a top-level warning, not buried in Quality Issues
- Showing both requested and effective threshold prominently
- Suggesting: "Consider using Custom { threshold_ns: 10.0 } to test at the achievable precision"

#### 12. Inconclusive Has Multiple Meanings

**Issue:** "Inconclusive" can mean:
- "The data is ambiguous" (posterior probability between pass/fail thresholds)
- "The measurement precision is insufficient" (threshold elevated beyond what's testable)
- "Budget exceeded before reaching conclusion"

These are different situations with different remedies.

**Suggestion:** Consider whether `Inconclusive { reason: ThresholdElevated, ... }` should surface the "can't achieve this precision" case more distinctly, or provide different guidance than other Inconclusive reasons.

#### 13. Document `--test-threads=1` Requirement for macOS

**Issue:** kperf only works with single-threaded test execution due to PMU resource contention. Without this, tests report "PMU unavailable (concurrent access)" and fall back to the coarse timer — but the Unmeasurable message still says "run with sudo."

**Impact:** Major friction for macOS users. I spent time debugging why sudo wasn't working.

**Recommendation:** Add to the guide and README:
```bash
# macOS: PMU timing requires single-threaded execution
sudo -E cargo test --test my_tests -- --test-threads=1
```

---

## Recommendations Summary

### Documentation

1. **Document `Outcome::Research`** or mark enum as `#[non_exhaustive]`
2. ~~**Recommend `SharedHardware`** as default for crypto code~~ → **Reconsidered:** SharedHardware is often impractical. Instead, document when it's achievable and guide users toward Custom thresholds.
3. **Improve `Unmeasurable` messages** with more context and platform-agnostic language
4. **Document `--test-threads=1` requirement** for macOS kperf usage prominently
5. **Add "Choosing a Threshold" section** explaining measurement floor and the Research→Custom workflow
6. ~~**Document test design patterns**~~ → **Done:** Updated `guide.md` and `testing-patterns.md`

### API/UX

7. ~~**Add `Display` impl for `Outcome`**~~ → **Done**
8. **Make threshold elevation more prominent** when it's >2x the requested threshold
9. **Consider a `SharedHardwareRealistic` preset** at ~10ns for users who want high sensitivity but can't achieve cycle-level precision
10. **Distinguish "insufficient precision" Inconclusive** from "ambiguous data" Inconclusive in messaging or reason variants

### Minor

11. **Consider enabling `static_secrets`** feature for x25519-dalek in dev-dependencies (optional)
