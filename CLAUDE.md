# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`timing-oracle` is a Rust library for detecting timing side channels in cryptographic code. It uses statistical methodology to compare timing distributions between fixed and random inputs, outputting leak probability, effect sizes, and exploitability assessments.

## Build Commands

```bash
cargo build                    # Build the library
cargo build --features parallel # Build with parallel/rayon support
cargo test                     # Run all tests (many are ignored pending implementation)
cargo test -- --ignored        # Run ignored tests (requires full implementation)
cargo test <test_name>         # Run a specific test
cargo check                    # Type-check without building
cargo run --example simple     # Run the simple example
cargo run --example compare    # Run the compare example
```

## Architecture

### Core Pipeline

The timing oracle follows a multi-layer analysis pipeline:
1. **Preflight checks** (`src/preflight/`) - Validates measurement setup before analysis
2. **Measurement** (`src/measurement/`) - High-resolution cycle counting with interleaved sampling
3. **Outlier filtering** - Symmetric percentile-based trimming
4. **Quantile computation** - Decile differences between fixed/random classes
5. **CI Gate** (`src/analysis/ci_gate.rs`) - Fast frequentist screening with bounded false positive rate
6. **Bayesian inference** (`src/analysis/bayes.rs`) - Posterior probability of timing leak
7. **Effect decomposition** (`src/analysis/effect.rs`) - Separates uniform shift from tail effects

### Module Structure

- `TimingOracle` (`src/oracle.rs`) - Builder-pattern entry point
- `Config` (`src/config.rs`) - All tunable parameters (samples, alpha, thresholds)
- `TestResult` (`src/result.rs`) - Output struct with leak_probability, effect, ci_gate, exploitability
- `statistics/` - Quantile computation, bootstrap resampling, covariance estimation
- `output/` - Terminal and JSON formatters

### Key Types

- `Class::Fixed` / `Class::Random` - Input class identifiers
- `Matrix9`, `Vector9` - nalgebra types for 9 decile quantile differences
- `Exploitability` - Negligible (<100ns), PossibleLAN, LikelyLAN, PossibleRemote (>20μs)

### Test Organization

- `tests/known_leaky.rs` - Tests that MUST detect timing leaks (early-exit comparison, branches)
- `tests/known_safe.rs` - Tests that MUST NOT false-positive (XOR, constant-time comparison)
- Most tests are `#[ignore]` pending full implementation of the measurement pipeline

## API Usage Pattern

```rust
// Simple API
let result = timing_oracle::test(
    || my_function(&fixed_input),
    || my_function(&random_input()),
);

// Builder API with configuration
let result = TimingOracle::new()
    .samples(100_000)
    .ci_alpha(0.01)
    .min_effect_of_concern(10.0)
    .test(fixed_closure, random_closure);
```

## Feature Flags

- `default` - No extra features
- `parallel` - Enables rayon for parallel bootstrap computation
