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
        fixed: [0u8; 32],
        random: || rand::random::<[u8; 32]>(),
        test: |input| {
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

Configure via the `oracle:` field:

```rust
timing_test! {
    oracle: TimingOracle::thorough(),  // 100k samples (~5s)
    fixed: my_fixed_input,
    random: || generate_random(),
    test: |input| operation(&input),
}
```

Presets: `TimingOracle::quick()` (~0.3s), `::balanced()` (~1s, default), `::thorough()` (~5s).

For fine-grained control:

```rust
timing_test! {
    oracle: TimingOracle::balanced()
        .samples(50_000)
        .ci_alpha(0.05),  // 5% FPR instead of 1%
    fixed: [0u8; 32],
    random: || rand::random(),
    test: |input| operation(&input),
}
```

### Setup Block

Use the `setup:` field for expensive initialization that shouldn't be timed:

```rust
timing_test! {
    setup: {
        let key = Aes128::new(&KEY);
    },
    fixed: [0u8; 16],
    random: || rand::random(),
    test: |input| {
        key.encrypt(&input);
    },
}
```

### Multiple Inputs

Use tuple destructuring:

```rust
timing_test! {
    fixed: ([0u8; 12], [0u8; 64]),
    random: || (rand::random(), rand::random()),
    test: |(nonce, plaintext)| {
        cipher.encrypt(&nonce, &plaintext);
    },
}
```

### Handling Unmeasurable Operations

`timing_test!` panics if the operation is too fast to measure. For explicit handling, use `timing_test_checked!`:

```rust
use timing_oracle::{timing_test_checked, Outcome};

let outcome = timing_test_checked! {
    fixed: [0u8; 32],
    random: || rand::random(),
    test: |input| very_fast_operation(&input),
};

match outcome {
    Outcome::Completed(result) => assert!(result.ci_gate.passed),
    Outcome::Unmeasurable { .. } => println!("Skipped: operation too fast"),
}
```

## Builder API

For programmatic access without macros:

```rust
use timing_oracle::{TimingTest, TimingOracle, Outcome};

let outcome = TimingTest::new()
    .oracle(TimingOracle::thorough())
    .fixed([0u8; 32])
    .random(|| rand::random())
    .test(|data| constant_time_eq(&secret, &data))
    .run();

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

Instead of comparing means (which miss distributional differences), we compare nine 
deciles between fixed and random inputs:

```
Fixed:   [q₁₀, q₂₀, q₃₀, ... q₉₀]  ─┐
                                     ├─► Δ ∈ ℝ⁹ ─► Bayesian inference ─► P(leak)
Random:  [q₁₀, q₂₀, q₃₀, ... q₉₀]  ─┘
```

**Two-layer analysis:**

1. **CI Gate**: Bootstraps the max quantile difference under the null hypothesis (pooled samples) to set a threshold controlling false positive rate at ~α. Designed not to flake in CI.

2. **Bayesian Layer**: Computes posterior probability of a leak and decomposes effects into uniform shift (different code paths) vs tail effects (cache misses, memory patterns).

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

- Reparaz et al. (2016): [DudeCT](https://eprint.iacr.org/2016/1123)
- Dunsche et al. (2024): [RTLF bounded FPR](https://www.usenix.org/conference/usenixsecurity24/presentation/dunsche)
- Crosby et al. (2009): [Timing attack feasibility](https://www.cs.rice.edu/~dwallach/pub/crosby-timing2009.pdf)

## License

MPL-2.0
