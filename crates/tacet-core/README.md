# tacet-core

**Core statistical analysis for timing side-channel detection.**

This crate provides the fundamental statistical algorithms used by [`tacet`](https://lib.rs/crates/tacet), designed to work in `no_std` environments (embedded, WASM, SGX enclaves) with only an allocator.

## When to Use This Crate

Most users should use the main [`tacet`](https://lib.rs/crates/tacet) crate, which provides:
- Measurement collection and timing infrastructure
- Adaptive sampling orchestration
- Pretty terminal output and JSON formatting
- The `timing_test!` macro

Use `tacet-core` directly only if you need:
- `no_std` compatibility (embedded, WASM, SGX)
- Custom measurement collection
- Direct access to statistical primitives

## Features

- **`std`** (default): Standard library support
- **`parallel`**: Parallel bootstrap using rayon (requires `std`, 4-8x speedup)

For `no_std`, disable default features:

```toml
[dependencies]
tacet-core = { version = "0.1", default-features = false }
```

## What's Included

### Statistical Analysis
- Quantile-based test statistics (9 deciles)
- Block bootstrap for covariance estimation
- Bayesian posterior probability computation

### Result Types
- `Outcome` - Pass/Fail/Inconclusive/Unmeasurable
- `EffectEstimate` - Decomposed timing effect (shift + tail)
- `Exploitability` - Negligible/PossibleLAN/LikelyLAN/PossibleRemote
- `MeasurementQuality` - Excellent/Good/Poor/TooNoisy

### Primitives
- `Vector9` / `Matrix9` - Fixed-size linear algebra types
- `TimingSample` - Tagged timing measurement
- `AttackerModel` - Threat model presets

## Example

```rust
use tacet_core::{
    statistics::{bootstrap_covariance_matrix, compute_deciles},
    analysis::bayes::compute_posterior_probability,
    types::{Vector9, AttackerModel},
};

// Compute deciles from timing samples
let baseline_deciles: Vector9 = compute_deciles(&baseline_times);
let sample_deciles: Vector9 = compute_deciles(&sample_times);
let delta = sample_deciles - baseline_deciles;

// Bootstrap covariance matrix
let cov = bootstrap_covariance_matrix(&baseline_times, &sample_times, 2000);

// Compute posterior probability of timing leak
let threshold = AttackerModel::AdjacentNetwork.threshold_ns();
let p_leak = compute_posterior_probability(&delta, &cov, threshold);
```

## Documentation

- [API Documentation](https://docs.rs/tacet-core)
- [Main crate](https://lib.rs/crates/tacet)
- [GitHub Repository](https://github.com/agucova/tacet)

## License

MPL-2.0
