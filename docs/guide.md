# timing-oracle User Guide

This guide provides a comprehensive understanding of timing side channel detection, the statistical methodology behind timing-oracle, and advanced usage patterns.

For quick usage examples, see the [README](../README.md). For complete API documentation, see [api-rust.md](api-rust.md), [api-c.md](api-c.md), or [api-go.md](api-go.md).

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
- [Interpreting Results in Depth](#interpreting-results-in-depth)
  - [Understanding Leak Probability](#understanding-leak-probability)
  - [Understanding θ_user vs θ_eff vs θ_floor](#understanding-θ_user-vs-θ_eff-vs-θ_floor)
  - [Effect Size and Pattern](#effect-size-and-pattern)
  - [Exploitability Assessment](#exploitability-assessment)
  - [Handling Inconclusive Results](#handling-inconclusive-results)
  - [Projection Mismatch](#projection-mismatch)
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

        // Check exploitability (based on Timeless Timing Attacks research)
        match exploitability {
            Exploitability::SharedHardwareOnly =>
                println!("Only exploitable with shared hardware (SGX, containers)"),
            Exploitability::Http2Multiplexing =>
                println!("Exploitable via HTTP/2 request multiplexing (~100k queries)"),
            Exploitability::StandardRemote =>
                println!("Exploitable with standard remote timing (~1k-10k queries)"),
            Exploitability::ObviousLeak =>
                println!("Obvious leak, trivially exploitable (<100 queries)"),
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

## Interpreting Results in Depth

This section provides detailed guidance on understanding timing-oracle output.

### Understanding Leak Probability

The `leak_probability` is $P(\max_k |\delta_k| > \theta_{\text{eff}} \mid \Delta)$—the posterior probability that at least one quantile's timing difference exceeds your effective threshold.

| Probability | Interpretation | Outcome |
|-------------|----------------|---------|
| < 5% | Confident no leak | **Pass** |
| 5–50% | Probably safe, uncertain | Inconclusive (needs more data) |
| 50–95% | Probably leaking, uncertain | Inconclusive (needs more data) |
| > 95% | Confident leak | **Fail** |

**Important**: This is a Bayesian posterior probability, not a frequentist p-value. When we report "72% probability of a leak," we mean: given the data and our model, 72% of the posterior mass corresponds to effects exceeding the threshold.

### Understanding θ_user vs θ_eff vs θ_floor

The library distinguishes between three threshold values:

| Value | Meaning |
|-------|---------|
| θ_user | What you requested (via attacker model) |
| θ_floor | What the measurement could actually resolve |
| θ_eff | What was actually tested: max(θ_user, θ_floor) |

**When θ_eff = θ_user:** Results directly answer "is there a leak > θ_user?"

**When θ_eff > θ_user:** Results answer "is there a leak > θ_eff?" A `Pass` means "no leak above θ_eff," but there could still be a leak between θ_user and θ_eff that wasn't detectable.

If θ_eff > θ_user, the result will include a quality warning explaining that the requested threshold couldn't be achieved. Consider using a PMU timer or accepting the elevated threshold.

### Effect Size and Pattern

The `EffectEstimate` decomposes the timing difference into interpretable components:

**2D Projection (when it fits well):**
- **Shift (μ)**: Uniform timing difference affecting all quantiles equally. Typical cause: different code path, branch on secret bit.
- **Tail (τ)**: Upper quantiles affected more than lower. Typical cause: cache misses, secret-dependent memory access patterns.

**Effect patterns:**

| Pattern | Signature | Typical Cause |
|---------|-----------|---------------|
| UniformShift | μ significant, τ ≈ 0 | Branch on secret bit, different code path |
| TailEffect | μ ≈ 0, τ significant | Cache-timing leak, memory access patterns |
| Mixed | Both μ and τ significant | Multiple leak sources |
| Complex | Projection mismatch high | Asymmetric or multi-modal pattern |
| Indeterminate | Neither significant | No detectable leak |

**Top Quantiles (when projection doesn't fit):**

When the 2D shift+tail model doesn't fit well (projection mismatch is high), the result includes `top_quantiles` showing which specific quantiles are most affected. For example, if the 90th percentile shows a large effect but other quantiles don't, this suggests a tail-heavy leak that the 2D model can't fully capture.

### Exploitability Assessment

Risk assessment based on effect size, informed by academic research on timing attack feasibility:

| Level | Effect Size | Attack Vector |
|-------|-------------|---------------|
| SharedHardwareOnly | < 10 ns | Requires ~1k queries on same physical core (SGX, containers, hyperthreading) |
| Http2Multiplexing | 10–100 ns | Requires ~100k concurrent HTTP/2 requests using request multiplexing |
| StandardRemote | 100 ns – 10 μs | Requires ~1k-10k queries with standard network timing |
| ObviousLeak | > 10 μs | Trivially observable with <100 queries |

**Note**: Actual exploitability depends on network conditions, attacker capabilities, and whether the timing difference is consistent. These categories are heuristics based on published research.

### Handling Inconclusive Results

Inconclusive means "couldn't reach 95% confidence either way." This is not a failure—it's honest reporting that the data doesn't support a strong claim.

| Reason | Meaning | Suggested Action |
|--------|---------|------------------|
| DataTooNoisy | Posterior ≈ prior after sampling | Use a better timer (PMU), reduce system noise, or increase sample budget |
| NotLearning | KL divergence collapsed for 5+ batches | Check for systematic measurement issues, consider different inputs |
| WouldTakeTooLong | Projected time exceeds budget | Accept uncertainty, adjust θ, or increase time budget |
| ThresholdUnachievable | θ_user < θ_floor even at max budget | Use a cycle counter (PMU timer) or accept the elevated θ |
| TimeBudgetExceeded | Ran out of time | Increase `time_budget` configuration |
| SampleBudgetExceeded | Ran out of samples | Increase `max_samples` configuration |
| ConditionsChanged | Calibration assumptions violated | Run in isolation, reduce system load, ensure stable conditions |

**What to do with Inconclusive:**
1. Check the `leak_probability`—it shows the current best estimate
2. Look at quality issues in the diagnostics
3. Consider whether more resources (time, samples) would help
4. For CI, you may choose to treat Inconclusive as a soft failure requiring investigation

### Projection Mismatch

When `diagnostics.projection_mismatch_ok = false`:

- The 2D shift+tail summary doesn't fit the observed 9D quantile pattern well
- **This does NOT affect the verdict**—the 9D posterior handles arbitrary patterns
- Check `effect.top_quantiles` for which specific quantiles show timing differences
- The `interpretation_caveat` field explains why the 2D summary may be inaccurate

Common causes of projection mismatch:
- Bi-modal timing distributions (operation sometimes takes a fast path, sometimes slow)
- Asymmetric effects (only certain quantiles affected)
- Complex cache behavior with multiple distinct timing modes

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
