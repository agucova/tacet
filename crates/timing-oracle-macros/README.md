# timing-oracle-macros

**Proc macros for timing side-channel tests.**

This crate provides the `timing_test!` and `timing_test_checked!` macros for [`timing-oracle`](https://lib.rs/crates/timing-oracle). You typically don't need to depend on this crate directly.

## Installation

The macros are included when you enable the `macros` feature on `timing-oracle` (enabled by default):

```toml
[dev-dependencies]
timing-oracle = "0.1"
```

## Usage

### `timing_test!`

Returns `Outcome` for pattern matching on all four variants:

```rust
use timing_oracle::{timing_test, Outcome};

let outcome = timing_test! {
    baseline: || [0u8; 32],
    sample: || rand::random::<[u8; 32]>(),
    measure: |input| {
        my_crypto_function(&input);
    },
};

match outcome {
    Outcome::Pass { leak_probability, .. } => {
        println!("No leak: {:.1}%", leak_probability * 100.0);
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

### `timing_test_checked!`

Same as `timing_test!` but panics on `Fail`:

```rust
use timing_oracle::{timing_test_checked, Outcome};

// Panics if timing leak detected
let outcome = timing_test_checked! {
    baseline: || [0u8; 32],
    sample: || rand::random::<[u8; 32]>(),
    measure: |input| {
        constant_time_eq(&secret, &input);
    },
};
```

### Custom Oracle Configuration

```rust
use timing_oracle::{timing_test, TimingOracle, AttackerModel};

let outcome = timing_test! {
    oracle: TimingOracle::for_attacker(AttackerModel::SharedHardware)
        .pass_threshold(0.01)
        .fail_threshold(0.99),
    baseline: || [0u8; 32],
    sample: || rand::random::<[u8; 32]>(),
    measure: |input| operation(&input),
};
```

## Syntax

```rust
timing_test! {
    // Optional: oracle configuration (defaults to AdjacentNetwork)
    oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork),

    // Required: baseline input generator
    baseline: || fixed_input,

    // Required: sample input generator
    sample: || random_input(),

    // Required: operation to measure
    measure: |input| operation(&input),
}
```

## Documentation

- [API Documentation](https://docs.rs/timing-oracle-macros)
- [Main crate](https://lib.rs/crates/timing-oracle)
- [GitHub Repository](https://github.com/agucova/timing-oracle)

## License

MPL-2.0
