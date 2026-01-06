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

### CI Integration

| Example | Description | Run Command |
|---------|-------------|-------------|
| `ci_integration` | Using CiTestBuilder with FailCriterion | `cargo run --example ci_integration` |

### Development & Debugging

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

**If you're integrating with CI:**
1. `ci_integration` - CiTestBuilder pattern
2. See [troubleshooting.md](../docs/troubleshooting.md) for handling flakiness

**If you're debugging performance:**
1. `profile_oracle` - Identify bottlenecks
2. `benchmark_baseline` - Establish baseline

## Key Patterns

### Choosing an Attacker Model

Select a threat model that matches your deployment scenario:

```rust
use timing_oracle::{TimingOracle, AttackerModel};

// Internet-facing API
let oracle = TimingOracle::for_attacker(AttackerModel::WANConservative);

// Internal LAN service
let oracle = TimingOracle::for_attacker(AttackerModel::LANConservative);

// SGX or shared hosting (strictest)
let oracle = TimingOracle::for_attacker(AttackerModel::LocalCycles);
```

### Correct Input Generation

All examples use InputPair to pre-generate inputs:

```rust
use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair};

// Pre-generate BEFORE measurement
let inputs = InputPair::new(
    [0u8; 32],              // Fixed value
    || rand::random(),      // Random generator
);

// Use attacker model + InputPair
let result = TimingOracle::for_attacker(AttackerModel::LANConservative)
    .test(inputs, |data| operation(data));
```

### Common Mistakes to Avoid

```rust
// WRONG - RNG inside closure
test(|| op(&FIXED), || op(&rand::random()));

// WRONG - Different code paths
test(|| op_a(), || op_b());

// CORRECT - Use InputPair
let inputs = InputPair::new(FIXED, || rand::random());
test(|| op(inputs.fixed()), || op(inputs.random()));
```

See the [README](../README.md#common-mistakes) for more details.
