# timing-oracle

**Detect timing side channels in Rust code with statistically rigorous methods.**

```
$ cargo test --test aes_timing

timing-oracle · aes_encrypt (16 bytes)
────────────────────────────────────────
  Leak probability:  2.3%
  Effect size:       < 1 ns (Negligible)
  CI gate:           PASS (α = 1%)
  
  ✓ No significant timing leak detected
```

## Installation

```sh
cargo add timing-oracle --dev
```

## Quick Start

```rust
use timing_oracle::timing_test;

#[test]
fn constant_time_compare() {
    let secret = [0u8; 32];
    
    let result = timing_test! {
        baseline: || [0u8; 32],
        sample: || rand::random::<[u8; 32]>(),
        measure: |input| {
            constant_time_eq(&secret, &input);
        },
    };
    
    assert!(result.ci_gate.passed, "Timing leak detected!");
}
```

The macro handles measurement and statistical analysis, returning a `TestResult` with pass/fail plus detailed diagnostics.

---

## What It Catches

```rust
// ✗ This looks constant-time but isn't (early-exit on mismatch)
fn naive_compare(a: &[u8], b: &[u8]) -> bool {
    a == b  // ← timing-oracle detects this in ~1 second
}

// ✓ This is actually constant-time
fn ct_compare(a: &[u8], b: &[u8]) -> bool {
    subtle::ConstantTimeEq::ct_eq(a, b).into()
}
```

## Why Another Tool?

Empirical timing tools like [DudeCT](https://github.com/oreparaz/dudect) are hard to use and yield results that are difficult to interpret.

`timing-oracle` gives you what you actually want: **the probability your code has a timing leak**, plus how exploitable it would be.

| | DudeCT | timing-oracle |
|---|--------|---------------|
| **Output** | t-statistic + p-value | Probability of leak (0–100%) |
| **False positives** | Unbounded (more samples → more FPs) | Controlled at ≤1% |
| **Effect size** | Not provided | Estimated in nanoseconds |
| **Exploitability** | Manual interpretation | Automatic classification |
| **CI-friendly** | Flaky without tuning | Works out of the box |

---

## Macro API

### Configuration

Configure via the `oracle:` field with an **attacker model** that defines your threat scenario:

```rust
use timing_oracle::{timing_test, TimingOracle, AttackerModel};

timing_test! {
    // Choose your threat model:
    // - WANConservative (50μs): Public APIs, internet-facing
    // - LANConservative (100ns): Internal services
    // - LocalCycles (2 cycles): SGX, shared hosting
    oracle: TimingOracle::for_attacker(AttackerModel::LANConservative),
    baseline: || my_fixed_input,
    sample: || generate_random(),
    measure: |input| operation(&input),
}
```

**Attacker Model Presets:**

| Preset | Threshold | Use case |
|--------|-----------|----------|
| `WANConservative` | 50 μs | Public APIs, general internet |
| `WANOptimistic` | 15 μs | Low-jitter cloud paths |
| `LANConservative` | 100 ns | Internal services (Crosby-style) |
| `LANStrict` | 2 cycles | High-security LAN (Kario-style) |
| `LocalCycles` | 2 cycles | SGX enclaves, shared hosting |
| `KyberSlashSentinel` | 10 cycles | Post-quantum crypto |
| `Research` | 0 | Detect any difference (not for CI) |

For sample count tuning, combine presets:

```rust
timing_test! {
    oracle: TimingOracle::for_attacker(AttackerModel::LANConservative)
        .samples(50_000)   // More samples for higher power
        .alpha(0.05),      // 5% FPR instead of 1%
    baseline: || [0u8; 32],
    sample: || rand::random(),
    measure: |input| operation(&input),
}
```

### Shared State

For expensive initialization, define variables before the macro and capture them:

```rust
// Expensive setup done before the macro
let key = Aes128::new(&KEY);

timing_test! {
    baseline: || [0u8; 16],
    sample: || rand::random(),
    measure: |input| {
        // `key` is captured from outer scope
        key.encrypt(&input);
    },
}
```

### Multiple Inputs

Use tuple destructuring:

```rust
timing_test! {
    baseline: || ([0u8; 12], [0u8; 64]),
    sample: || (rand::random(), rand::random()),
    measure: |(nonce, plaintext)| {
        cipher.encrypt(&nonce, &plaintext);
    },
}
```

### Handling Unmeasurable Operations

`timing_test!` panics if the operation is too fast to measure. For explicit handling, use `timing_test_checked!`:

```rust
use timing_oracle::{timing_test_checked, TimingOracle, AttackerModel, Outcome};

let outcome = timing_test_checked! {
    oracle: TimingOracle::for_attacker(AttackerModel::LANConservative),
    baseline: || [0u8; 32],
    sample: || rand::random(),
    measure: |input| very_fast_operation(&input),
};

match outcome {
    Outcome::Completed(result) => assert!(result.ci_gate.passed),
    Outcome::Unmeasurable { .. } => println!("Skipped: operation too fast"),
}
```

## Builder API

For programmatic access without macros:

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};

let inputs = InputPair::new(|| [0u8; 32], || rand::random());

// Use attacker model to define threat scenario
let outcome = TimingOracle::for_attacker(AttackerModel::LANConservative)
    .samples(20_000)  // Optional: tune sample count
    .test(inputs, |data| constant_time_eq(&secret, data));

if let Outcome::Completed(result) = outcome {
    println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
    println!("Exploitability: {:?}", result.exploitability);
}
```

See [API documentation](https://docs.rs/timing-oracle) for full details.

---

## Interpreting Results

### TestResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `ci_gate.passed` | `bool` | Use this for CI pass/fail decisions |
| `leak_probability` | `f64` | Bayesian posterior probability (0.0–1.0) |
| `effect` | `Option<Effect>` | Magnitude in ns, with shift/tail decomposition |
| `exploitability` | `Exploitability` | How practical is exploitation? |
| `quality` | `MeasurementQuality` | Confidence in results |
| `min_detectable_effect` | `MinDetectableEffect` | Smallest effect we could have detected |

### Exploitability Levels

Based on [Crosby et al. (2009)](https://www.cs.rice.edu/~dwallach/pub/crosby-timing2009.pdf):

| Level | Effect Size | Meaning |
|-------|-------------|---------|
| `Negligible` | < 100 ns | Requires ~10k+ queries to exploit over LAN |
| `PossibleLAN` | 100–500 ns | Exploitable on LAN with statistical methods |
| `LikelyLAN` | 500 ns – 20 µs | Readily exploitable on local network |
| `PossibleRemote` | > 20 µs | Potentially exploitable over internet |

### Effect Patterns

When a leak is detected, the effect is decomposed:

| Pattern | Meaning | Typical cause |
|---------|---------|---------------|
| `UniformShift` | All quantiles shifted equally | Different code path, branch |
| `TailEffect` | Upper quantiles shifted more | Cache misses, memory patterns |
| `Mixed` | Both components significant | Multiple interacting factors |

---

## How It Works

### Quantile-Based Statistics

Instead of comparing means (which miss distributional differences), we compare nine
deciles between fixed and random inputs:

```
Fixed:   [q₁₀, q₂₀, q₃₀, ... q₉₀]  ─┐
                                     ├─► Δ ∈ ℝ⁹
Random:  [q₁₀, q₂₀, q₃₀, ... q₉₀]  ─┘
```

This captures both uniform shifts (branch timing) and tail effects (cache misses that only affect upper quantiles).

### Practical Significance Testing

The key question isn't "is there any timing difference?" but "is the difference exploitable under my threat model?" The attacker model (θ) defines your threshold:

```
H₀: max|Δ| ≤ θ  (effect within acceptable bounds)
H₁: max|Δ| > θ  (exploitable timing leak)
```

This follows the [SILENT methodology](https://arxiv.org/abs/2504.19821), which properly controls false positives at the boundary—you won't get spurious failures on code that just barely meets your threshold.

### Two-Layer Analysis

1. **CI Gate** (SILENT bootstrap): Studentized max-statistic test with bounded FPR ≤ α. Uses paired block bootstrap to handle autocorrelation. Rejects only when the effect significantly exceeds θ.

2. **Bayesian Layer**: Computes posterior probability P(leak > θ) and decomposes effects into uniform shift (different code paths) vs tail effects (cache misses).

Both layers use the same θ threshold and typically agree; divergence indicates edge cases worth investigating.

### Sample Splitting

To avoid double-dipping (using data to both set priors and test), samples are split temporally:
- **First 30%**: Calibration (covariance estimation, prior setting)
- **Last 70%**: Inference (actual hypothesis test)

Temporal split preserves autocorrelation structure and prevents overfitting.

For full methodology: [docs/spec.md](docs/spec.md)

## Platform Support

| Platform | Timer | Resolution | Notes |
|----------|-------|------------|-------|
| x86_64 | rdtsc | ~0.3 ns | Works out of the box |
| Apple Silicon | kperf | ~1 ns | Auto-enabled with `sudo` |
| Apple Silicon | cntvct | ~41 ns | Fallback with adaptive batching |
| Linux ARM | perf_event | ~1 ns | Requires `sudo` or `CAP_PERFMON` |

---

## References

- Dunsche et al. (2025): [SILENT](https://arxiv.org/abs/2504.19821) — Practical significance testing for timing side channels
- Dunsche et al. (2024): [RTLF bounded FPR](https://www.usenix.org/conference/usenixsecurity24/presentation/dunsche)
- Reparaz et al. (2016): [DudeCT](https://eprint.iacr.org/2016/1123)
- Crosby et al. (2009): [Timing attack feasibility](https://www.cs.rice.edu/~dwallach/pub/crosby-timing2009.pdf)

## License

MPL-2.0
