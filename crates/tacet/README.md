# tacet

**Detect timing side channels in Rust code with statistically rigorous methods.**

```
$ cargo test --test aes_timing -- --nocapture

[aes128_block_encrypt_constant_time]
tacet
──────────────────────────────────────────────────────────────

  Samples: 6000 per class
  Quality: Good

  ✓ No timing leak detected

    Probability of leak: 0.0%
    95% CI: 0.0–12.5 ns

──────────────────────────────────────────────────────────────
```

## Installation

```sh
cargo add tacet --dev
```

## Quick Start

```rust
use tacet::{timing_test_checked, TimingOracle, AttackerModel, Outcome};

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

## Why Another Tool?

Existing tools like DudeCT output t-statistics and p-values that are hard to interpret. `tacet` gives you what you actually want: **the probability your code has a timing leak**, plus how exploitable it would be.

| | DudeCT | tacet |
|---|--------|---------------|
| **Output** | t-statistic + p-value | Probability of leak (0-100%) |
| **False positives** | Unbounded (more samples = more FPs) | Converges to correct answer |
| **Effect size** | Not provided | Estimated in nanoseconds |
| **Exploitability** | Manual interpretation | Automatic classification |
| **CI-friendly** | Flaky without tuning | Works out of the box |

## Attacker Model Presets

Choose your threat model to define what timing differences matter:

| Preset | Threshold | Use case |
|--------|-----------|----------|
| `SharedHardware` | 0.6 ns (~2 cycles) | SGX, cross-VM, containers |
| `AdjacentNetwork` | 100 ns | LAN, HTTP/2 APIs |
| `RemoteNetwork` | 50 μs | Public internet APIs |
| `Research` | 0 | Detect any difference |

## Documentation

- [API Documentation](https://docs.rs/tacet)
- [GitHub Repository](https://github.com/agucova/tacet)

## License

MPL-2.0
