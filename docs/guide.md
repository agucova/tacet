# timing-oracle User Guide

This guide provides a comprehensive understanding of timing side channel detection, the statistical methodology behind timing-oracle, and advanced usage patterns.

For quick usage examples, see the [README](../README.md). For complete API documentation, see [api-reference.md](api-reference.md).

## Table of Contents

- [The Problem](#the-problem)
- [DudeCT Two-Class Pattern](#dudect-two-class-pattern)
- [How It Works](#how-it-works)
  - [Measurement Protocol](#measurement-protocol)
  - [Two-Layer Analysis](#two-layer-analysis)
  - [Quantile-Based Statistics](#quantile-based-statistics)
- [Timer Selection and Adaptive Batching](#timer-selection-and-adaptive-batching)
  - [Platform Timers](#platform-timers)
  - [Adaptive Batching](#adaptive-batching)
- [Advanced Examples](#advanced-examples)
  - [Testing with State](#testing-with-state)
  - [Quick Smoke Tests](#quick-smoke-tests)
  - [Reusing Calibrated Timer](#reusing-calibrated-timer)
  - [Interpreting Results](#interpreting-results)
  - [Testing Async/Await Code](#testing-asyncawait-code)
- [Statistical Details](#statistical-details)
  - [Covariance Estimation](#covariance-estimation)
  - [Sample Splitting](#sample-splitting)
  - [Bayes Factor Computation](#bayes-factor-computation)
- [Limitations](#limitations)
- [References](#references)

---

## The Problem

Timing side channels leak secrets through execution time variations. Classic examples:

```rust
// VULNERABLE: Early-exit comparison
fn compare(a: &[u8], b: &[u8]) -> bool {
    for i in 0..a.len() {
        if a[i] != b[i] {
            return false;  // Exits early on mismatch!
        }
    }
    true
}
```

When comparing a secret against attacker-controlled input, the early exit reveals how many bytes match. An attacker can guess the secret byte-by-byte.

Similar issues affect:
- RSA implementations with non-constant-time modular exponentiation
- AES with table lookups that hit cache differently
- HMAC verification with short-circuit comparison
- Password hashing that exits early on format validation

Existing tools like [dudect](https://github.com/oreparaz/dudect) detect such leaks but have limitations:
- Output t-statistics instead of interpretable probabilities
- No bounded false positive rate for CI integration
- Miss non-uniform timing effects (e.g., cache-related tail behavior)
- 3.6-8x slower per-sample measurement overhead (52 ns/sample vs 14-18 ns/sample for timing-oracle)

---

## DudeCT Two-Class Pattern

This library follows **DudeCT's two-class testing pattern** for detecting data-dependent timing:

- **Class 0**: All-zero data (0x00 repeated)
- **Class 1**: Random data

This pattern tests whether operations have **data-dependent timing** rather than comparing specific fixed values:

```rust
use timing_oracle::{TimingOracle, AttackerModel};

// Choose your threat model - this determines what effect size is considered "significant"
let result = TimingOracle::for_attacker(AttackerModel::LANConservative)
    .samples(50_000)
    .test(
        || {
            // Class 0: All zeros
            let data = [0u8; 32];
            my_crypto_operation(&data)
        },
        || {
            // Class 1: Random
            let data = rand_bytes();
            my_crypto_operation(&data)
        },
    );

// Modern crypto should have Negligible exploitability even if timing differences detected
if !result.ci_gate.passed {
    assert!(matches!(result.exploitability, Exploitability::Negligible));
}
```

**Attacker Model Presets:**

| Preset | Threshold | When to use |
|--------|-----------|-------------|
| `WANConservative` | 50 μs | Public APIs, internet-facing services |
| `LANConservative` | 100 ns | Internal microservices (Crosby-style) |
| `LANStrict` | 2 cycles | High-security internal services (Kario-style) |
| `LocalCycles` | 2 cycles | SGX enclaves, shared hosting |
| `KyberSlashSentinel` | 10 cycles | Post-quantum crypto libraries |
| `Research` | 0 | Academic analysis, detect any difference |

**Why this pattern works:**
- Simpler than fixed-vs-variable input patterns
- Tests for data dependencies in crypto operations
- Matches DudeCT's proven methodology
- See `tests/aes_timing.rs` and `tests/ecc_timing.rs` for real-world examples

---

## How It Works

### Measurement Protocol

1. **Warmup**: Run both operations to stabilize CPU state
2. **Interleaved sampling**: Alternate fixed/random in randomized order to prevent drift bias
3. **High-precision timing**: Use `rdtsc` (x86) or `cntvct_el0` (ARM) with serialization barriers
4. **Outlier filtering**: Apply symmetric percentile threshold to both classes

### Two-Layer Analysis

The key insight is that CI gates and interpretable statistics need different methodologies:

```
                    Timing Samples
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
   ┌───────────────┐           ┌──────────────────┐
   │  Layer 1:     │           │  Layer 2:        │
   │  CI Gate      │           │  Bayesian        │
   ├───────────────┤           ├──────────────────┤
   │ RTLF-style    │           │ Closed-form      │
   │ bootstrap     │           │ conjugate Bayes  │
   │               │           │                  │
   │ Bounded FPR   │           │ Probability +    │
   │ Pass/Fail     │           │ Effect size      │
   └───────────────┘           └──────────────────┘
```

**Layer 1 (CI Gate)** answers: "Should this block my build?"
- Uses RTLF-style bootstrap thresholds ([Dunsche et al., 2024](https://www.usenix.org/conference/usenixsecurity24/presentation/dunsche))
- Conservative max-based per-quantile bounds
- Guarantees false positive rate ≤ α (default 1%)

**Layer 2 (Bayesian)** answers: "What's the probability and magnitude?"
- Computes Bayes factor between H₀ (no leak) and H₁ (leak)
- Decomposes effect into uniform shift and tail components
- Provides 95% credible intervals

### Quantile-Based Statistics

Rather than comparing means (which miss distributional differences), we compare nine deciles:

```
Fixed class:   [q₁₀, q₂₀, q₃₀, q₄₀, q₅₀, q₆₀, q₇₀, q₈₀, q₉₀]
Random class:  [q₁₀, q₂₀, q₃₀, q₄₀, q₅₀, q₆₀, q₇₀, q₈₀, q₉₀]
Difference:    Δ ∈ ℝ⁹
```

This captures:
- **Uniform shifts**: Code path takes different branch (affects all quantiles equally)
- **Tail effects**: Cache misses on specific inputs (affects upper quantiles more)

The effect is decomposed using an orthogonalized basis:
```
Δ = μ·[1,1,...,1] + τ·[-0.5, -0.375, ..., 0.375, 0.5] + noise
```

Where μ is the uniform shift and τ is the tail effect.

---

## Timer Selection and Adaptive Batching

### Platform Timers

The library automatically selects the best available timer for your platform:

**x86_64:**
- Uses `rdtsc` instruction (~1ns resolution)
- Cycle-accurate timing without requiring privileges

**macOS ARM64 (Apple Silicon):**
- **Standard Timer** (default): `cntvct_el0` virtual timer (~42ns resolution for M1/M2/M3 at 24 MHz)
- **PmuTimer** (opt-in, requires `sudo`): PMU-based cycle counting (~1ns resolution) - see "Advanced: Cycle-Accurate Timing" in the README

**Linux:**
- **x86_64**: `rdtsc` instruction (~1ns, no privileges needed)
- **ARM64 Standard Timer** (default): `cntvct_el0` virtual timer (resolution varies by SoC)
  - ARMv8.6+ (Graviton4): ~1ns (1 GHz)
  - Ampere Altra: ~40ns (25 MHz)
  - Raspberry Pi 4: ~18ns (54 MHz)
- **LinuxPerfTimer** (opt-in, requires `sudo`/`CAP_PERFMON`): perf_event cycle counting (~1ns) - see "Advanced: Cycle-Accurate Timing" in the README

### Adaptive Batching

On platforms with coarse timer resolution (>5 ticks per operation), the library automatically enables **adaptive batching**:

1. **Pilot phase**: Measures ~100 warmup iterations to determine median operation time
2. **K selection**: Chooses batch size K to achieve 50+ timer ticks per batch
3. **Batch measurement**: Measures K iterations together and analyzes batch totals (reported with K)
4. **Bounded batching**: Never exceeds K=20 to prevent microarchitectural artifacts

**Example:** On Apple Silicon (42ns resolution) measuring a 100ns operation:
- Single call: ~2.4 ticks → unreliable (quantization noise)
- Batch of K=21: ~50 ticks → stable distribution for statistical inference

Batching is **automatically disabled** when:
- Timer resolution is fine enough (>5 ticks per call)
- Using `PmuTimer` (macOS) or `LinuxPerfTimer` (Linux) for cycle-accurate timing
- Operation is slow enough (>210ns on Apple Silicon with standard timer)
- Running on x86_64 (rdtsc provides cycle-accurate timing)

Batching is automatic in the public API; to avoid it entirely, use `PmuTimer` (macOS) or `LinuxPerfTimer` (Linux) for cycle-accurate timing.

---

## Advanced Examples

### Testing with State

For tests that need shared state or complex setup:

```rust
use timing_oracle::TimingOracle;

// Example: Testing database query timing
fn test_db_query_timing() {
    let result = TimingOracle::new()
        .samples(10_000)
        .test_with_state(
            || {
                // Setup: Initialize database connection
                let db = Database::connect("test.db");
                db.create_table();
                db
            },
            |db| {
                // Fixed input: Known user ID (might trigger caching)
                db.prepare_query("SELECT * FROM users WHERE id = ?", 1)
            },
            |db, rng| {
                // Random input: Random user ID
                let user_id = rng.gen_range(1..1000);
                db.prepare_query("SELECT * FROM users WHERE id = ?", user_id)
            },
            |db, query| {
                // Execute the query
                db.execute(query);
            },
        );

    assert!(result.ci_gate.passed,
            "Database query should not leak user ID through timing");
}
```

### Quick Smoke Tests

For faster iteration during development:

```rust
use timing_oracle::{TimingOracle, AttackerModel};

// Fast test with reduced samples (completes in ~0.5s)
let result = TimingOracle::quick()
    .attacker_model(AttackerModel::LANConservative)
    .test(|| my_function(&FIXED), || my_function(&random_input()));

println!("Quick smoke test: {:.0}% probability",
         result.leak_probability * 100.0);
```

### Interpreting Results

```rust
use timing_oracle::{TimingOracle, AttackerModel, Exploitability};

let result = TimingOracle::for_attacker(AttackerModel::LANConservative)
    .test(fixed_fn, random_fn);

// 1. Check CI gate for pass/fail decision
if !result.ci_gate.passed {
    println!("❌ CI gate failed - timing leak detected");
}

// 2. Examine leak probability
match result.leak_probability {
    p if p > 0.99 => println!("🔴 Virtually certain leak (>99%)"),
    p if p > 0.9  => println!("🟠 Very likely leak (>90%)"),
    p if p > 0.5  => println!("🟡 Probable leak (>50%)"),
    _             => println!("🟢 No significant leak detected"),
}

// 3. If leak detected, examine effect decomposition
if let Some(effect) = result.effect {
    println!("Effect breakdown:");
    println!("  Shift: {:.1} ns ({:?})", effect.shift_ns, effect.pattern);
    println!("  Tail:  {:.1} ns", effect.tail_ns);
    println!("  Total magnitude: {:.1}-{:.1} ns (95% CI)",
             effect.credible_interval_ns.0,
             effect.credible_interval_ns.1);
}

// 4. Check exploitability
match result.exploitability {
    Exploitability::Negligible =>
        println!("Not practically exploitable"),
    Exploitability::PossibleLAN =>
        println!("⚠️  Might be exploitable on LAN (~100k queries)"),
    Exploitability::LikelyLAN =>
        println!("⚠️  Likely exploitable on LAN (~10k queries)"),
    Exploitability::PossibleRemote =>
        println!("⚠️  Possibly exploitable remotely"),
}

// 5. Assess measurement quality
println!("Measurement quality: {:?}", result.quality);
println!("Min detectable effect: {:.1} ns (shift), {:.1} ns (tail)",
         result.min_detectable_effect.shift_ns,
         result.min_detectable_effect.tail_ns);
```

### Testing Async/Await Code

For async functions, use `Runtime::block_on()` to bridge async → sync for measurement:

```rust
use timing_oracle::{TimingOracle, AttackerModel};
use tokio::runtime::Runtime;
use tokio::time::{sleep, Duration};

// Create a runtime once per test
let rt = tokio::runtime::Builder::new_current_thread()
    .enable_time()
    .build()
    .unwrap();

let result = TimingOracle::for_attacker(AttackerModel::LANConservative)
    .samples(10_000)
    .test(
        || {
            rt.block_on(async {
                // Fixed async operation
                sleep(Duration::from_micros(10)).await;
                std::hint::black_box(42)
            })
        },
        || {
            rt.block_on(async {
                // Random async operation
                sleep(Duration::from_micros(10)).await;
                std::hint::black_box(43)
            })
        },
    );

// Verify no timing leak from async executor overhead
assert!(result.ci_gate.passed, "Async executor overhead should be symmetric");
```

**Key considerations for async testing:**
- Use **single-threaded** runtime for lower noise (`new_current_thread()`)
- Create the runtime **once** outside the measured closures
- Use `block_on()` to execute async code synchronously
- Both closures should perform **identical async operations** for baseline tests
- See `tests/async_timing.rs` for comprehensive async examples

---

## Statistical Details

### Covariance Estimation

Quantile differences are correlated (neighboring quantiles tend to move together). We estimate the 9x9 covariance matrix Σ₀ via block bootstrap, then use it in both the CI gate thresholds and Bayesian inference.

### Sample Splitting

To avoid "double-dipping" (using data to both set priors and compute posteriors), we split samples:
- **Calibration set (30%)**: Estimate Σ₀, compute minimum detectable effect, set prior hyperparameters
- **Inference set (70%)**: Compute Δ and Bayes factor with fixed parameters

### Bayes Factor Computation

Under both hypotheses, Δ follows a multivariate normal:
- H₀: Δ ~ N(0, Σ₀)
- H₁: Δ ~ N(0, Σ₀ + X·Λ₀·Xᵀ)

The log Bayes factor is computed in closed form via Cholesky decomposition.

---

## Limitations

- **Not a formal proof**: Statistical evidence, not cryptographic verification
- **Noise sensitivity**: High-noise environments may produce unreliable results
- **JIT and optimization**: Ensure test conditions match production
- **Platform-dependent**: Timing characteristics vary across CPUs
- **Exploitability is heuristic**: Actual exploitability depends on network conditions and attacker capabilities

For mission-critical code, combine with formal verification tools like [ct-verif](https://github.com/imdea-software/verifying-constant-time).

---

## References

1. Reparaz, O., Balasch, J., & Verbauwhede, I. (2016). "Dude, is my code constant time?" DATE. — Original dudect methodology
2. Dunsche, M., et al. (2024). "With Great Power Come Great Side Channels: Statistical Timing Side-Channel Analyses with Bounded Type-1 Errors." USENIX Security. — RTLF methodology for bounded FPR
3. Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009). "Opportunities and limits of remote timing attacks." ACM TISSEC. — Exploitability thresholds
