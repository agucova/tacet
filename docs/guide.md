# timing-oracle User Guide

This guide provides a comprehensive understanding of timing side channel detection, the statistical methodology behind timing-oracle, and advanced usage patterns.

For quick usage examples, see the [README](../README.md). For complete API documentation, see [api-reference.md](api-reference.md).

## Table of Contents

- [The Problem](#the-problem)
- [DudeCT Two-Class Pattern](#dudect-two-class-pattern)
- [How It Works](#how-it-works)
  - [Measurement Protocol](#measurement-protocol)
  - [Adaptive Bayesian Analysis](#adaptive-bayesian-analysis)
  - [Quantile-Based Statistics](#quantile-based-statistics)
- [Timer Selection and Adaptive Batching](#timer-selection-and-adaptive-batching)
  - [Platform Timers](#platform-timers)
  - [Adaptive Batching](#adaptive-batching)
- [Advanced Examples](#advanced-examples)
  - [Choosing an Attacker Model](#choosing-an-attacker-model)
  - [Configuration Presets](#configuration-presets)
  - [Interpreting Results](#interpreting-results)
  - [Testing Async/Await Code](#testing-asyncawait-code)
- [Statistical Details](#statistical-details)
  - [Covariance Estimation](#covariance-estimation)
  - [Adaptive Sampling](#adaptive-sampling)
  - [Bayesian Posterior Computation](#bayesian-posterior-computation)
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
- Results depend on sample count (more samples → more false positives)
- Miss non-uniform timing effects (e.g., cache-related tail behavior)
- 3.6-8x slower per-sample measurement overhead (52 ns/sample vs 14-18 ns/sample for timing-oracle)

---

## DudeCT Two-Class Pattern

This library follows **DudeCT's two-class testing pattern** for detecting data-dependent timing:

- **Class 0 (Baseline)**: All-zero data (0x00 repeated)
- **Class 1 (Sample)**: Random data

This pattern tests whether operations have **data-dependent timing** rather than comparing specific fixed values:

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};

// Create inputs with the two-class pattern
let inputs = InputPair::new(
    || [0u8; 32],          // Baseline: all zeros
    || rand::random(),     // Sample: random data
);

// Choose your threat model - this determines the minimum effect size to care about
let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .test(inputs, |data| {
        my_crypto_operation(data);
    });

// Handle result
match outcome {
    Outcome::Pass { .. } => println!("No leak detected"),
    Outcome::Fail { exploitability, .. } => {
        panic!("Timing leak: {:?}", exploitability);
    }
    Outcome::Inconclusive { reason, .. } => {
        println!("Inconclusive: {:?}", reason);
    }
    Outcome::Unmeasurable { recommendation, .. } => {
        println!("Skipped: {}", recommendation);
    }
}
```

**Attacker Model Presets:**

| Preset | Threshold | When to use |
|--------|-----------|-------------|
| `SharedHardware` | ~0.6 ns (~2 cycles) | SGX enclaves, cross-VM, containers |
| `PostQuantumSentinel` | ~3.3 ns (~10 cycles) | Post-quantum crypto (ML-KEM, ML-DSA) |
| `AdjacentNetwork` | 100 ns | LAN, HTTP/2 APIs (default) |
| `RemoteNetwork` | 50 us | Public APIs, general internet |
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
2. **Interleaved sampling**: Alternate baseline/sample in randomized order to prevent drift bias
3. **High-precision timing**: Use `rdtsc` (x86) or `cntvct_el0` (ARM) with serialization barriers
4. **Outlier winsorization**: Cap extreme outliers at percentile threshold

### Adaptive Bayesian Analysis

The oracle uses a two-phase adaptive approach:

```
                    Timing Samples
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
   ┌───────────────┐           ┌──────────────────┐
   │  Phase 1:     │           │  Phase 2:        │
   │  Calibration  │           │  Adaptive Loop   │
   ├───────────────┤           ├──────────────────┤
   │ 5,000 samples │           │ Batch collection │
   │ Covariance    │           │ Posterior update │
   │ estimation    │           │ Decision check   │
   │ Prior setup   │   ────>   │ Quality gates    │
   │               │           │                  │
   │ Fixed cost    │           │ Stops when       │
   │               │           │ P < 0.05 or      │
   │               │           │ P > 0.95         │
   └───────────────┘           └──────────────────┘
```

**Phase 1 (Calibration)** establishes statistical parameters:
- Estimates noise covariance matrix via block bootstrap
- Computes minimum detectable effect (MDE)
- Sets Bayesian priors from calibration data

**Phase 2 (Adaptive Loop)** collects samples until a decision:
- Collects samples in batches (default 1,000)
- Updates posterior probability after each batch
- Stops when P(leak) < 0.05 (Pass) or P(leak) > 0.95 (Fail)
- Applies quality gates to detect when more data won't help

This adaptive approach:
- **Stops early** on clear pass/fail cases (saves time)
- **Gathers more evidence** when the signal is near the threshold
- **Returns Inconclusive** when hitting resource limits or quality gates

### Quantile-Based Statistics

Rather than comparing means (which miss distributional differences), we compare nine deciles:

```
Baseline class:  [q₁₀, q₂₀, q₃₀, q₄₀, q₅₀, q₆₀, q₇₀, q₈₀, q₉₀]
Sample class:    [q₁₀, q₂₀, q₃₀, q₄₀, q₅₀, q₆₀, q₇₀, q₈₀, q₉₀]
Difference:      Δ ∈ ℝ⁹
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
- **PmuTimer** (opt-in, requires `sudo`): PMU-based cycle counting (~1ns resolution)

**Linux:**
- **x86_64**: `rdtsc` instruction (~1ns, no privileges needed)
- **ARM64 Standard Timer** (default): `cntvct_el0` virtual timer (resolution varies by SoC)
  - ARMv8.6+ (Graviton4): ~1ns (1 GHz)
  - Ampere Altra: ~40ns (25 MHz)
  - Raspberry Pi 4: ~18ns (54 MHz)
- **LinuxPerfTimer** (opt-in, requires `sudo`/`CAP_PERFMON`): perf_event cycle counting (~1ns)

### Adaptive Batching

On platforms with coarse timer resolution (>5 ticks per operation), the library automatically enables **adaptive batching**:

1. **Pilot phase**: Measures ~100 warmup iterations to determine median operation time
2. **K selection**: Chooses batch size K to achieve 50+ timer ticks per batch
3. **Batch measurement**: Measures K iterations together and analyzes batch totals
4. **Bounded batching**: Never exceeds K=20 to prevent microarchitectural artifacts

**Example:** On Apple Silicon (42ns resolution) measuring a 100ns operation:
- Single call: ~2.4 ticks → unreliable (quantization noise)
- Batch of K=21: ~50 ticks → stable distribution for statistical inference

Batching is **automatically disabled** when:
- Timer resolution is fine enough (>5 ticks per call)
- Using `PmuTimer` (macOS) or `LinuxPerfTimer` (Linux) for cycle-accurate timing
- Operation is slow enough (>210ns on Apple Silicon with standard timer)
- Running on x86_64 (rdtsc provides cycle-accurate timing)

---

## Advanced Examples

### Choosing an Attacker Model

```rust
use timing_oracle::{TimingOracle, AttackerModel};

// Internet-facing API: attacker measures over general internet
TimingOracle::for_attacker(AttackerModel::RemoteNetwork)  // θ = 50μs

// Internal microservice or HTTP/2 API: attacker on LAN or using request multiplexing
TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)  // θ = 100ns

// SGX enclave, containers, or shared hosting: co-resident attacker
TimingOracle::for_attacker(AttackerModel::SharedHardware)  // θ = 0.6ns (~2 cycles)

// Post-quantum crypto (catch KyberSlash-class bugs)
TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)  // θ = 10 cycles

// Research/debugging: detect any statistical difference
TimingOracle::for_attacker(AttackerModel::Research)  // θ → 0

// Custom threshold in nanoseconds
TimingOracle::for_attacker(AttackerModel::Custom { threshold_ns: 500.0 })
```

### Interpreting Results

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome, Exploitability, helpers::InputPair};

let inputs = InputPair::new(|| [0u8; 32], || rand::random());
let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .test(inputs, |data| my_function(data));

match outcome {
    Outcome::Pass { leak_probability, quality, .. } => {
        println!("No leak: P(leak)={:.1}%", leak_probability * 100.0);
        println!("Measurement quality: {:?}", quality);
    }

    Outcome::Fail { leak_probability, effect, exploitability, .. } => {
        println!("Leak detected: P(leak)={:.1}%", leak_probability * 100.0);
        println!("Effect breakdown:");
        println!("  Shift: {:.1} ns ({:?})", effect.shift_ns, effect.pattern);
        println!("  Tail:  {:.1} ns", effect.tail_ns);
        println!("  Total: {:.1}-{:.1} ns (95% CI)",
                 effect.credible_interval_ns.0,
                 effect.credible_interval_ns.1);

        // Check exploitability
        match exploitability {
            Exploitability::Negligible =>
                println!("Not practically exploitable"),
            Exploitability::PossibleLAN =>
                println!("Might be exploitable on LAN (~100k queries)"),
            Exploitability::LikelyLAN =>
                println!("Likely exploitable on LAN (~10k queries)"),
            Exploitability::PossibleRemote =>
                println!("Possibly exploitable remotely"),
        }
    }

    Outcome::Inconclusive { reason, leak_probability, .. } => {
        println!("Inconclusive: {:?}", reason);
        println!("Current P(leak): {:.1}%", leak_probability * 100.0);
    }

    Outcome::Unmeasurable { recommendation, .. } => {
        println!("Operation too fast: {}", recommendation);
    }
}
```

### Testing Async/Await Code

For async functions, use `Runtime::block_on()` to bridge async → sync for measurement:

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};
use tokio::runtime::Runtime;

// Create a runtime once per test
let rt = tokio::runtime::Builder::new_current_thread()
    .enable_time()
    .build()
    .unwrap();

let inputs = InputPair::new(
    || [0u8; 32],
    || rand::random(),
);

let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .test(inputs, |data| {
        rt.block_on(async {
            // Your async operation
            my_async_crypto_function(data).await;
        })
    });

// Verify no timing leak from async executor overhead
match outcome {
    Outcome::Pass { .. } => println!("Async code is constant-time"),
    Outcome::Fail { .. } => panic!("Timing leak in async code"),
    _ => {}
}
```

**Key considerations for async testing:**
- Use **single-threaded** runtime for lower noise (`new_current_thread()`)
- Create the runtime **once** outside the measured closures
- Use `block_on()` to execute async code synchronously
- Both inputs should perform **identical async operations** for baseline tests
- See `tests/async_timing.rs` for comprehensive async examples

---

## Statistical Details

### Covariance Estimation

Quantile differences are correlated (neighboring quantiles tend to move together). We estimate the 9x9 covariance matrix Σ₀ via block bootstrap during calibration, accounting for autocorrelation in the timing samples.

### Adaptive Sampling

The adaptive loop uses Bayesian inference to decide when to stop:

1. **Posterior Update**: After each batch, compute P(effect > θ | data)
2. **Decision Check**: Pass if P < 0.05, Fail if P > 0.95
3. **Quality Gates**: Stop early if:
   - Data too noisy (posterior ≈ prior after calibration)
   - Not learning (KL divergence collapsed for 5+ batches)
   - Would take too long (projected time exceeds budget)

### Bayesian Posterior Computation

The posterior probability of a timing leak is computed using conjugate Bayesian inference:

Under both hypotheses, Δ follows a multivariate normal:
- H₀: Δ ~ N(0, Σ₀/n) — no effect, just measurement noise
- H₁: Δ ~ N(β, Σ₀/n + Σ_prior) — effect plus noise

The posterior is:
```
P(leak | data) = P(data | leak) × P(leak) / P(data)
```

Where P(leak) is the prior (default 0.25) and the likelihood ratio is computed via the multivariate normal densities.

---

## Limitations

- **Not a formal proof**: Statistical evidence, not cryptographic verification
- **Noise sensitivity**: High-noise environments may produce unreliable results (check `MeasurementQuality`)
- **JIT and optimization**: Ensure test conditions match production
- **Platform-dependent**: Timing characteristics vary across CPUs
- **Exploitability is heuristic**: Actual exploitability depends on network conditions and attacker capabilities

For mission-critical code, combine with formal verification tools like [ct-verif](https://github.com/imdea-software/verifying-constant-time).

---

## References

1. Reparaz, O., Balasch, J., & Verbauwhede, I. (2016). "Dude, is my code constant time?" DATE. — Original dudect methodology
2. Van Goethem, T., et al. (2020). "Timeless Timing Attacks." USENIX Security. — HTTP/2 timing attacks over internet
3. Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009). "Opportunities and limits of remote timing attacks." ACM TISSEC. — Exploitability thresholds
