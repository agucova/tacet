# timing-oracle Examples

This directory contains examples demonstrating various uses of timing-oracle.

## Getting Started

Run any example with:
```bash
cargo run --example <name>
```

## Examples by Category

### Core Usage

| Example | Description | Run Command |
|---------|-------------|-------------|
| `simple` | Basic usage with InputPair and both simple/builder APIs | `cargo run --example simple` |
| `compare` | Side-by-side comparison of leaky vs constant-time code | `cargo run --example compare` |
| `test_xor` | Verify XOR is constant-time (should not detect leak) | `cargo run --example test_xor` |

### Real-World Crypto

| Example | Description | Run Command |
|---------|-------------|-------------|
| `aes` | Test AES-256-GCM encryption for timing leaks | `cargo run --example aes` |

### Development and Debugging

| Example | Description | Run Command |
|---------|-------------|-------------|
| `profile_oracle` | Profile oracle performance | `cargo run --example profile_oracle --release` |
| `test_no_batch` | Test behavior without adaptive batching | `cargo run --example test_no_batch` |
| `test_exact_copy` | Test exact copy operations | `cargo run --example test_exact_copy` |

### Benchmarking (Internal)

| Example | Description | Run Command |
|---------|-------------|-------------|
| `bench_bootstrap` | Benchmark bootstrap performance | `cargo run --example bench_bootstrap --release` |
| `benchmark_baseline` | Establish baseline measurements | `cargo run --example benchmark_baseline --release` |
| `compare_mde_methods` | Compare MDE calculation methods | `cargo run --example compare_mde_methods --release` |

## Suggested Reading Order

**If you're new to timing-oracle:**
1. `simple` - Understand basic API and InputPair usage
2. `compare` - See how leaky vs safe code differs
3. `test_xor` - Verify constant-time operations don't false-positive
4. `aes` - Real-world crypto testing

**If you're debugging performance:**
1. `profile_oracle` - Identify bottlenecks
2. `benchmark_baseline` - Establish baseline

## Key Patterns

### Choosing an Attacker Model

Select a threat model that matches your deployment scenario:

```rust
use timing_oracle::{TimingOracle, AttackerModel};
use std::time::Duration;

// Internet-facing API
let oracle = TimingOracle::for_attacker(AttackerModel::RemoteNetwork)
    .time_budget(Duration::from_secs(30));

// Internal LAN service or HTTP/2 API
let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .time_budget(Duration::from_secs(30));

// SGX, containers, or shared hosting (strictest)
let oracle = TimingOracle::for_attacker(AttackerModel::SharedHardware)
    .time_budget(Duration::from_secs(60));
```

### Handling All Outcomes

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};
use std::time::Duration;

let inputs = InputPair::new(|| [0u8; 32], || rand::random());

let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .time_budget(Duration::from_secs(30))
    .test(inputs, |data| operation(data));

match outcome {
    Outcome::Pass { leak_probability, .. } => {
        println!("No leak: P(leak)={:.1}%", leak_probability * 100.0);
    }
    Outcome::Fail { leak_probability, exploitability, .. } => {
        println!("Leak: P(leak)={:.1}%, {:?}", leak_probability * 100.0, exploitability);
    }
    Outcome::Inconclusive { reason, leak_probability, .. } => {
        println!("Inconclusive: {:?}", reason);
    }
    Outcome::Unmeasurable { recommendation, .. } => {
        println!("Skipping: {}", recommendation);
    }
}
```

### Common Mistakes to Avoid

```rust
// WRONG - RNG inside closure
test(|| op(&FIXED), || op(&rand::random()));

// WRONG - Different code paths
test(|| op_a(), || op_b());

// CORRECT - Use InputPair
let inputs = InputPair::new(|| FIXED, || rand::random());
oracle.test(inputs, |data| op(data));
```

See the [README](../README.md#common-mistakes) for more details.
