# timing-oracle

**Detect timing side channels in Rust code with statistically rigorous methods.**

```
$ cargo test --test aes_timing -- --nocapture

[aes128_block_encrypt_constant_time]
timing-oracle
──────────────────────────────────────────────────────────────

  Samples: 6000 per class
  Quality: Good

  ✓ No timing leak detected

    Probability of leak: 0.0%
    Effect: 0.0 ns
      Shift: 0.0 ns
      Tail:  0.0 ns
      95% CI: 0.0–12.5 ns

──────────────────────────────────────────────────────────────
```

## Installation

```sh
cargo add timing-oracle --dev
```

## Quick Start

```rust
use timing_oracle::{timing_test_checked, TimingOracle, AttackerModel, Outcome};

#[test]
fn constant_time_compare() {
    let secret = [0u8; 32];

    let outcome = timing_test_checked! {
        oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork),
        baseline: || [0u8; 32],
        sample: || rand::random::<[u8; 32]>(),
        measure: |input| {
            constant_time_eq(&secret, &input);
        },
    };

    match outcome {
        Outcome::Pass { .. } => { /* No leak */ }
        Outcome::Fail { exploitability, .. } => panic!("Timing leak: {:?}", exploitability),
        Outcome::Inconclusive { .. } => { /* Could not determine */ }
        Outcome::Unmeasurable { .. } => { /* Operation too fast */ }
    }
}
```

The macro handles measurement and statistical analysis, returning an `Outcome` with pass/fail/inconclusive plus detailed diagnostics.

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

## Real-World Validation

While testing the library, I incidentally rediscovered [CVE-2023-49092](https://rustsec.org/advisories/RUSTSEC-2023-0071.html) (Marvin Attack) in the RustCrypto `rsa` crate — a ~500ns timing leak in RSA decryption. I wasn't looking for it; the library just flagged it. See [the full investigation](docs/investigation-rsa-timing-anomaly.md).

## Why Another Tool?

Empirical timing tools like [DudeCT](https://github.com/oreparaz/dudect) are hard to use and yield results that are difficult to interpret.

`timing-oracle` gives you what you actually want: **the probability your code has a timing leak**, plus how exploitable it would be.

| | DudeCT | timing-oracle |
|---|--------|---------------|
| **Output** | t-statistic + p-value | Probability of leak (0–100%) |
| **False positives** | Unbounded (more samples → more FPs) | Converges to correct answer |
| **Effect size** | Not provided | Estimated in nanoseconds |
| **Exploitability** | Manual interpretation | Automatic classification |
| **CI-friendly** | Flaky without tuning | Works out of the box |

---

## Macro API

### Configuration

Configure via the `oracle:` field with an **attacker model** that defines your threat scenario:

```rust
use timing_oracle::{timing_test_checked, TimingOracle, AttackerModel, Outcome};

let outcome = timing_test_checked! {
    // Choose your threat model:
    // - RemoteNetwork (50μs): Public APIs, internet-facing
    // - AdjacentNetwork (100ns): Internal services, HTTP/2 APIs
    // - SharedHardware (~2 cycles): SGX, containers, shared hosting
    oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork),
    baseline: || my_fixed_input,
    sample: || generate_random(),
    measure: |input| operation(&input),
};
```

**Attacker Model Presets:**

| Preset | Threshold | Use case |
|--------|-----------|----------|
| `SharedHardware` | 0.6 ns (~2 cycles) | SGX, cross-VM, containers, hyperthreading |
| `AdjacentNetwork` | 100 ns | LAN, HTTP/2 (Timeless Timing Attacks) |
| `RemoteNetwork` | 50 μs | Public APIs, general internet |
| `Research` | 0 | Detect any difference (not for CI) |
| `Custom { threshold_ns }` | user-defined | Custom threshold |

For stricter testing, adjust decision thresholds:

```rust
timing_test_checked! {
    oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)   // More confident pass (default 0.05)
        .fail_threshold(0.99),  // More confident fail (default 0.95)
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

### Handling All Outcomes

`timing_test_checked!` returns an `Outcome` with four variants:

```rust
use timing_oracle::{timing_test_checked, TimingOracle, AttackerModel, Outcome};

let outcome = timing_test_checked! {
    oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork),
    baseline: || [0u8; 32],
    sample: || rand::random(),
    measure: |input| very_fast_operation(&input),
};

match outcome {
    Outcome::Pass { leak_probability, .. } => {
        println!("No leak: P(leak)={:.1}%", leak_probability * 100.0);
    }
    Outcome::Fail { exploitability, .. } => {
        panic!("Timing leak detected: {:?}", exploitability);
    }
    Outcome::Inconclusive { reason, .. } => {
        println!("Inconclusive: {:?}", reason);
    }
    Outcome::Unmeasurable { recommendation, .. } => {
        println!("Skipped: {}", recommendation);
    }
}
```

## Builder API

For programmatic access without macros:

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};

let inputs = InputPair::new(|| [0u8; 32], || rand::random());

// Use attacker model to define threat scenario
let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .test(inputs, |data| constant_time_eq(&secret, data));

match outcome {
    Outcome::Pass { leak_probability, .. } => {
        println!("No leak: P(leak)={:.1}%", leak_probability * 100.0);
    }
    Outcome::Fail { leak_probability, exploitability, .. } => {
        println!("Leak: P(leak)={:.1}%, {:?}", leak_probability * 100.0, exploitability);
    }
    _ => {}
}
```

See [API documentation](https://docs.rs/timing-oracle) for full details.

---

## Interpreting Results

### Outcome Variants

| Variant | Meaning | Key Fields |
|---------|---------|------------|
| `Pass` | No timing leak (P(leak) < 5%) | `leak_probability`, `effect`, `quality` |
| `Fail` | Timing leak confirmed (P(leak) > 95%) | `leak_probability`, `effect`, `exploitability` |
| `Inconclusive` | Cannot determine (5% < P(leak) < 95%) | `reason`, `leak_probability`, `effect` |
| `Unmeasurable` | Operation too fast to measure | `recommendation`, `platform` |

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

### Adaptive Bayesian Analysis

The oracle adaptively collects samples until it can reach a conclusion:

1. **Posterior Probability**: Computes P(leak > θ | data) using Bayesian inference
2. **Effect Decomposition**: Separates uniform shift (branch timing) from tail effects (cache misses)
3. **Early Stopping**: Stops when posterior probability crosses pass/fail thresholds (default 0.05/0.95)
4. **Inconclusive Handling**: Reports when conclusion cannot be reached within budget

### Adaptive Sampling

The oracle uses a two-phase approach:
1. **Calibration phase** (5,000 samples by default): Estimates covariance matrix and sets Bayesian priors
2. **Adaptive phase**: Collects samples in batches until decision thresholds are reached

This adaptive approach stops early when confident, saving time on clear pass/fail cases while gathering more evidence when needed.

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

- Van Goethem et al. (2020): [Timeless Timing Attacks](https://www.usenix.org/conference/usenixsecurity20/presentation/van-goethem) — HTTP/2 timing attacks over internet
- Reparaz et al. (2016): [DudeCT](https://eprint.iacr.org/2016/1123)
- Crosby et al. (2009): [Timing attack feasibility](https://www.cs.rice.edu/~dwallach/pub/crosby-timing2009.pdf)

## License

MPL-2.0
