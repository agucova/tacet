# tacet Examples

This directory contains examples demonstrating various uses of tacet.

## Getting Started

Run any example with:
```bash
cargo run --example <name>
```

## Examples

| Example | Description | Run Command |
|---------|-------------|-------------|
| `simple` | Basic usage with InputPair and both simple/builder APIs | `cargo run --example simple` |
| `compare` | Side-by-side comparison of leaky vs constant-time code | `cargo run --example compare` |
| `test_xor` | Verify XOR is constant-time (should not detect leak) | `cargo run --example test_xor` |
| `aes` | Test AES-256-GCM encryption for timing leaks | `cargo run --example aes` |

## Suggested Reading Order

**If you're new to tacet:**
1. `simple` - Understand basic API and InputPair usage
2. `compare` - See how leaky vs safe code differs
3. `test_xor` - Verify constant-time operations don't false-positive
4. `aes` - Real-world crypto testing

## Key Patterns

### Choosing an Attacker Model

Select a threat model that matches your deployment scenario:

```rust
use tacet::{TimingOracle, AttackerModel};
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
use tacet::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};
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
