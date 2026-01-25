<p align="center">
  <img src="website/public/logo-dark-bg.svg" alt="tacet" width="340" />
</p>

<p align="center">
  <strong>Detect side channels with statistically rigorous methods.</strong><br>
  Available for Rust, JavaScript/TypeScript, C/C++, and Go.
</p>

<p align="center">
  <a href="https://crates.io/crates/tacet"><img src="https://img.shields.io/crates/v/tacet" alt="crates.io"></a>
  <a href="https://www.npmjs.com/package/@tacet/js"><img src="https://img.shields.io/npm/v/@tacet/js" alt="npm"></a>
  <a href="https://jsr.io/@tacet/js"><img src="https://jsr.io/badges/@tacet/js" alt="JSR"></a>
  <a href="https://github.com/agucova/tacet/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MPL--2.0-blue" alt="License"></a>
</p>

---

| Language | Package | Install |
|----------|---------|---------|
| Rust | `tacet` | `cargo add tacet --dev` |
| JavaScript | `@tacet/js` | `bun add @tacet/js` |
| C/C++ | source | [Build instructions](https://tacet.sh/getting-started/installation) |
| Go | `tacet-go` | `go get github.com/agucova/tacet/bindings/go` |

**[Documentation →](https://tacet.sh)**

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

## Quick Start

```rust
use tacet::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};

#[test]
fn constant_time_compare() {
    let secret = [0u8; 32];

    let inputs = InputPair::new(
        || [0u8; 32],
        || rand::random::<[u8; 32]>(),
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .test(inputs, |input| {
            constant_time_eq(&secret, &input);
        });

    match outcome {
        Outcome::Pass { .. } => { /* No leak */ }
        Outcome::Fail { exploitability, .. } => panic!("Timing leak: {:?}", exploitability),
        Outcome::Inconclusive { .. } => { /* Could not determine */ }
        Outcome::Unmeasurable { .. } => { /* Operation too fast */ }
    }
}
```

> **Important:** The baseline input must be chosen to create timing asymmetry with the sample input. For comparison functions, baseline should **match** the secret so it runs the full comparison (slow) while random samples exit early (fast). See [Choosing Input Classes](https://tacet.sh/core-concepts/two-class-pattern) for details.

---

## What It Catches

```rust
// ✗ This looks constant-time but isn't (early-exit on mismatch)
fn naive_compare(a: &[u8], b: &[u8]) -> bool {
    a == b  // ← tacet detects this in ~1 second
}

// ✓ This is actually constant-time
fn ct_compare(a: &[u8], b: &[u8]) -> bool {
    subtle::ConstantTimeEq::ct_eq(a, b).into()
}
```

## Why Tacet?

Empirical timing tools like [DudeCT](https://github.com/oreparaz/dudect) are hard to use and yield results that are difficult to interpret.

Tacet gives you what you actually want: **the probability your code has a timing leak**, plus how exploitable it would be.

| | DudeCT | Tacet |
|---|--------|-------|
| **Output** | t-statistic + p-value | Probability of leak (0–100%) |
| **False positives** | Unbounded (more samples → more FPs) | Converges to correct answer |
| **Effect size** | Not provided | Estimated in nanoseconds |
| **Exploitability** | Manual interpretation | Automatic classification |
| **CI-friendly** | Flaky without tuning | Works out of the box |

## Real-World Validation

While testing the library, I incidentally rediscovered [CVE-2023-49092](https://rustsec.org/advisories/RUSTSEC-2023-0071.html) (Marvin Attack) in the RustCrypto `rsa` crate—a ~500ns timing leak in RSA decryption. I wasn't looking for it; the library just flagged it. See the [full investigation](https://tacet.sh/case-studies/rsa-timing-anomaly).

---

## Attacker Model Presets

Choose based on your threat model:

| Preset | Threshold | Use case |
|--------|-----------|----------|
| `SharedHardware` | 0.6 ns (~2 cycles) | SGX, cross-VM, containers, hyperthreading |
| `AdjacentNetwork` | 100 ns | LAN, HTTP/2 (Timeless Timing Attacks) |
| `RemoteNetwork` | 50 μs | Public APIs, general internet |
| `Research` | 0 | Detect any difference (not for CI) |
| `Custom { threshold_ns }` | user-defined | Custom threshold |

```rust
// Internet-facing API
TimingOracle::for_attacker(AttackerModel::RemoteNetwork)

// Internal microservice or HTTP/2 API
TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)

// SGX enclave or container
TimingOracle::for_attacker(AttackerModel::SharedHardware)
```

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

| Level | Effect Size | Meaning |
|-------|-------------|---------|
| `SharedHardwareOnly` | < 10 ns | Requires shared physical core |
| `Http2Multiplexing` | 10–100 ns | Exploitable via HTTP/2 |
| `StandardRemote` | 100 ns – 10 μs | Exploitable with network timing |
| `ObviousLeak` | > 10 μs | Trivially exploitable |

---

## Running Tests

```bash
# Run timing tests (single-threaded is recommended)
cargo test --test timing_tests -- --test-threads=1 --nocapture

# With higher-precision PMU timer (macOS ARM64)
sudo -E cargo test --test timing_tests -- --test-threads=1

# With higher-precision PMU timer (Linux)
sudo cargo test --test timing_tests -- --test-threads=1
```

See [CI Integration](https://tacet.sh/guides/ci-integration) for more details.

## Platform Support

| Platform | Timer | Resolution | Notes |
|----------|-------|------------|-------|
| x86_64 | rdtsc | ~0.3 ns | Best precision |
| Apple Silicon | kperf | ~1 ns | Requires `sudo` + `--test-threads=1` |
| Apple Silicon | cntvct | ~42 ns | Default, uses adaptive batching |
| Linux ARM64 | perf_event | ~1 ns | Requires `sudo` or `CAP_PERFMON` |

**Beyond timing:** Experimental support for power and EM side-channel analysis is available in the Rust crate via the `power` feature. See the [Power Analysis Guide](https://tacet.sh/guides/power-analysis).

---

## Documentation

- [Installation](https://tacet.sh/getting-started/installation)
- [Quick Start](https://tacet.sh/getting-started/quick-start)
- [Two-Class Pattern](https://tacet.sh/core-concepts/two-class-pattern) — Choosing input classes
- [Attacker Models](https://tacet.sh/core-concepts/attacker-models) — Choosing thresholds
- [CI Integration](https://tacet.sh/guides/ci-integration)
- [API Reference (Rust)](https://docs.rs/tacet)

## References

- Van Goethem et al. (2020): [Timeless Timing Attacks](https://www.usenix.org/conference/usenixsecurity20/presentation/van-goethem) — HTTP/2 timing attacks over internet
- Reparaz et al. (2016): [DudeCT](https://eprint.iacr.org/2016/1123) — Original two-class methodology
- Crosby et al. (2009): [Timing attack feasibility](https://www.cs.rice.edu/~dwallach/pub/crosby-timing2009.pdf) — Effect size to exploitability

## License

MPL-2.0
