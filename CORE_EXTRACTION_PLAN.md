# timing-oracle-core Extraction Plan

This document outlines the plan for extracting `timing-oracle-core`, a `no_std + alloc` crate containing the pure statistical analysis code, from the existing `timing-oracle` crate.

## Goals

1. **Create `timing-oracle-core`**: A `no_std + alloc` crate with pure statistical analysis
2. **Refactor `timing-oracle`**: Depend on core, keep measurement/orchestration
3. **Enable FFI bindings**: Core's C ABI becomes the foundation for Go, Node.js, etc.
4. **Preserve backwards compatibility**: Existing users see no API changes

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         timing-oracle (std)                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Public API (unchanged)                                          │   │
│  │  - TimingOracle builder                                          │   │
│  │  - test() method with InputPair                                  │   │
│  │  - Macros (assert_constant_time!, etc.)                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Measurement Layer (std-only)                                    │   │
│  │  - measurement/ (timers, collectors, outlier filtering)          │   │
│  │  - preflight/ (system checks, generator validation)              │   │
│  │  - adaptive/ (loop runner, calibration, state)                   │   │
│  │  - output/ (terminal, JSON formatting)                           │   │
│  │  - helpers.rs (InputPair, warning callbacks)                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              │ depends on                               │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Re-exported from timing-oracle-core                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               │ pub dependency
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    timing-oracle-core (no_std + alloc)                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  analysis/                                                       │   │
│  │  - bayes.rs      (Bayesian inference, Monte Carlo)               │   │
│  │  - effect.rs     (Effect decomposition)                          │   │
│  │  - mde.rs        (Minimum detectable effect)                     │   │
│  │  - diagnostics.rs (Model fit, stationarity checks)               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  statistics/                                                     │   │
│  │  - bootstrap.rs       (Block bootstrap resampling)               │   │
│  │  - covariance.rs      (Covariance matrix estimation)             │   │
│  │  - quantile.rs        (Decile computation)                       │   │
│  │  - block_length.rs    (Politis-White optimal block length)       │   │
│  │  - autocorrelation.rs (Lag-k autocorrelation)                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  types.rs    (Matrix9, Vector9, AttackerModel, Class, etc.)      │   │
│  │  result.rs   (Outcome, EffectEstimate, Exploitability, etc.)     │   │
│  │  constants.rs (DECILES, B_TAIL, ONES, LOG_2PI)                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               │ C ABI (future)
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    timing-oracle-ffi (future)                           │
│  - extern "C" functions                                                 │
│  - #[repr(C)] result types                                              │
│  - cbindgen header generation                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Create Crate Structure (Day 1)

### 1.1 Create timing-oracle-core directory

```bash
mkdir -p timing-oracle-core/src
```

### 1.2 Create timing-oracle-core/Cargo.toml

```toml
[package]
name = "timing-oracle-core"
version = "0.1.0"
edition = "2021"
rust-version = "1.80"
license = "MPL-2.0"
description = "Core statistical analysis for timing side-channel detection (no_std)"
keywords = ["security", "cryptography", "timing", "side-channel", "no_std"]
categories = ["cryptography", "no-std"]
repository = "https://github.com/agucova/timing-oracle"
documentation = "https://docs.rs/timing-oracle-core"

[lib]
name = "timing_oracle_core"

[dependencies]
# Linear algebra - supports no_std with libm
nalgebra = { version = "0.34", default-features = false, features = ["libm"] }

# Random number generation - supports no_std with alloc
rand = { version = "0.9", default-features = false, features = ["alloc"] }
rand_distr = { version = "0.5", default-features = false }
rand_xoshiro = { version = "0.7", default-features = false }

# Serialization - supports no_std with alloc
serde = { version = "1.0", default-features = false, features = ["derive", "alloc"] }

[features]
default = ["std"]

# Enable std for convenience (Display impls, env var reading, etc.)
std = ["nalgebra/std", "rand/std", "serde/std"]

# Enable parallel bootstrap (requires std)
parallel = ["std", "dep:rayon"]

[dependencies.rayon]
version = "1.11"
optional = true

[dev-dependencies]
# For testing, we can use std
```

### 1.3 Update workspace Cargo.toml

```toml
[workspace]
members = [".", "timing-oracle-core", "timing-oracle-macros"]
```

---

## Phase 2: Move Core Modules (Days 1-2)

### 2.1 Files to MOVE to timing-oracle-core/src/

| Source | Destination | Changes Required |
|--------|-------------|------------------|
| `src/analysis/` | `core/src/analysis/` | Minor: remove any std imports |
| `src/statistics/` | `core/src/statistics/` | Minor: remove any std imports |
| `src/types.rs` | `core/src/types.rs` | None |
| `src/constants.rs` | `core/src/constants.rs` | None |
| `src/result.rs` | `core/src/result.rs` | Feature-gate `std::env`, `std::fmt::Display` |

### 2.2 Files to KEEP in timing-oracle/src/

| File/Module | Reason |
|-------------|--------|
| `src/oracle.rs` | Uses `std::time::Instant`, orchestrates everything |
| `src/config.rs` | Uses `std::time::Duration` |
| `src/measurement/` | Platform-specific timers, `std::time`, `std::fs` |
| `src/preflight/` | System checks using `std::fs`, `std::time` |
| `src/adaptive/` | Uses `std::time::Instant`, `std::collections::VecDeque` |
| `src/output/` | Uses `colored`, terminal formatting |
| `src/helpers.rs` | Uses `std::collections::HashSet`, callbacks |
| `src/thread_pool.rs` | Uses `std::sync::OnceLock`, rayon |
| `src/lib.rs` | Re-exports, macros |

---

## Phase 3: Handle no_std Compatibility (Days 2-3)

### 3.1 Core lib.rs setup

```rust
// timing-oracle-core/src/lib.rs

#![no_std]
#![warn(missing_docs)]
#![warn(clippy::all)]

extern crate alloc;

pub mod analysis;
pub mod statistics;
mod constants;
mod result;
mod types;

// Re-exports
pub use constants::{B_TAIL, DECILES, LOG_2PI, ONES};
pub use result::{
    Diagnostics, EffectEstimate, EffectPattern, Exploitability,
    InconclusiveReason, IssueCode, MeasurementQuality, MinDetectableEffect,
    Outcome, QualityIssue,
};
pub use types::{AttackerModel, Class, Matrix2, Matrix9, Matrix9x2, TimingSample, Vector2, Vector9};

// Re-export analysis and statistics
pub use analysis::{
    bayes::{compute_bayes_factor, BayesResult},
    diagnostics::{compute_diagnostics, DiagnosticsExtra},
    effect::{classify_pattern, decompose_effect, EffectDecomposition},
    mde::{analytical_mde, estimate_mde, MdeEstimate},
};
pub use statistics::{
    bootstrap_covariance_matrix, bootstrap_difference_covariance,
    compute_deciles, optimal_block_length,
};
```

### 3.2 Changes to result.rs for no_std

```rust
// timing-oracle-core/src/result.rs

// Replace std imports with alloc
use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;

// Remove UnreliablePolicy::from_env_or() entirely - users pass policy explicitly
// The std::env::var call was unnecessary complexity

// Display impls use core::fmt::Display (same trait, works in no_std)
impl core::fmt::Display for Outcome { ... }
```

### 3.3 Changes to statistics modules

Most changes are mechanical:

```rust
// Before (statistics/bootstrap.rs)
use std::collections::HashMap;

// After
use alloc::vec::Vec;
// Note: HashMap isn't used in bootstrap.rs, just Vec
```

The statistics modules primarily use:
- `Vec<f64>` → `alloc::vec::Vec<f64>`
- `nalgebra` types → unchanged (already no_std compatible)
- `rand` types → unchanged (with correct features)

### 3.4 Changes to analysis modules

Similar mechanical changes:

```rust
// Before (analysis/bayes.rs)
// (no std imports actually - already clean!)

// After - add alloc import at crate level
extern crate alloc;
```

The analysis modules are already quite clean - they use:
- `nalgebra` matrices/vectors
- `rand` for Monte Carlo sampling
- No std types directly

---

## Phase 4: Update timing-oracle to Use Core (Day 3)

### 4.1 Update timing-oracle/Cargo.toml

```toml
[dependencies]
# Core statistical analysis
timing-oracle-core = { path = "../timing-oracle-core", features = ["std"] }

# Remove duplicated dependencies that are now in core:
# - nalgebra (still needed for some orchestration, or re-export from core)
# - rand (still needed for measurement, or re-export from core)
# - statrs (check if still needed)

# Keep std-only dependencies:
colored = "3.0.0"
serde_json = "1.0"
rayon = { version = "1.11", optional = true }
# ... platform-specific deps unchanged
```

### 4.2 Update timing-oracle/src/lib.rs

```rust
// Re-export everything from core for backwards compatibility
pub use timing_oracle_core::{
    // Types
    AttackerModel, Class, TimingSample,
    Matrix2, Matrix9, Matrix9x2, Vector2, Vector9,

    // Results
    Outcome, EffectEstimate, EffectPattern, Exploitability,
    Diagnostics, MeasurementQuality, MinDetectableEffect,
    InconclusiveReason, IssueCode, QualityIssue,

    // Constants
    B_TAIL, DECILES, LOG_2PI, ONES,

    // Analysis (if users need low-level access)
    analysis, statistics,
};

// Keep local modules
pub mod adaptive;
mod config;
mod oracle;
// ... etc
```

### 4.3 Update internal imports

Throughout timing-oracle, change:
```rust
// Before
use crate::types::{Matrix9, Vector9};
use crate::result::Outcome;

// After
use timing_oracle_core::{Matrix9, Vector9, Outcome};
// Or use the re-exports:
use crate::{Matrix9, Vector9, Outcome};
```

---

## Phase 5: Testing Strategy (Day 4)

### 5.1 Core unit tests

Move relevant unit tests to timing-oracle-core:

```rust
// timing-oracle-core/src/statistics/bootstrap.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_bootstrap_basic() {
        // ... existing test
    }
}
```

### 5.2 Integration tests stay in timing-oracle

All integration tests (`tests/*.rs`) stay in the main crate since they test the full pipeline including measurement.

### 5.3 Add no_std compilation test

```rust
// timing-oracle-core/tests/no_std_build.rs
#![no_std]
extern crate alloc;

use timing_oracle_core::{Outcome, AttackerModel};

#[test]
fn compiles_without_std() {
    let _ = AttackerModel::AdjacentNetwork;
}
```

### 5.4 CI configuration

```yaml
# .github/workflows/ci.yml
jobs:
  test-core-no-std:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install no_std target
        run: rustup target add thumbv7em-none-eabihf
      - name: Check no_std build
        run: cargo check -p timing-oracle-core --target thumbv7em-none-eabihf --no-default-features
```

---

## Phase 6: Documentation Updates (Day 4-5)

### 6.1 Update AGENTS.md / CLAUDE.md

Add timing-oracle-core to the architecture description.

### 6.2 Core crate documentation

```rust
//! # timing-oracle-core
//!
//! Core statistical analysis for timing side-channel detection.
//!
//! This crate provides the pure computation layer for timing analysis:
//! - Bayesian inference for leak probability
//! - Effect decomposition (shift vs tail)
//! - Bootstrap covariance estimation
//! - Minimum detectable effect calculation
//!
//! ## no_std Support
//!
//! This crate is `no_std` compatible with the `alloc` crate.
//! Disable the `std` feature for embedded/WASM targets:
//!
//! ```toml
//! [dependencies]
//! timing-oracle-core = { version = "0.1", default-features = false }
//! ```
//!
//! ## For FFI Users
//!
//! This crate is designed to be called via FFI. The main entry point is:
//!
//! ```rust,ignore
//! use timing_oracle_core::{analysis::bayes::compute_bayes_factor, statistics::*};
//!
//! // Collect measurements in your language, pass as slices:
//! let baseline_ns: &[f64] = &[...];
//! let sample_ns: &[f64] = &[...];
//!
//! // Compute covariance, then Bayesian analysis
//! let cov = bootstrap_difference_covariance(baseline_ns, sample_ns, ...);
//! let result = compute_bayes_factor(delta, cov, threshold_ns, ...);
//! ```
```

### 6.3 README update

Add section explaining the crate split and when to use which crate.

---

## Phase 7: Verification Checklist (Day 5)

### 7.1 Backwards compatibility

- [ ] All existing tests pass
- [ ] Public API unchanged (same imports work)
- [ ] Examples compile and run
- [ ] Benchmarks still work

### 7.2 no_std verification

- [ ] `cargo check -p timing-oracle-core --no-default-features` succeeds
- [ ] `cargo check -p timing-oracle-core --target thumbv7em-none-eabihf --no-default-features` succeeds
- [ ] All core unit tests pass with `--no-default-features`

### 7.3 Integration verification

- [ ] timing-oracle compiles with timing-oracle-core dependency
- [ ] All timing-oracle tests pass
- [ ] `cargo doc` generates correct documentation

---

## Detailed File Changes

### Files to CREATE:

```
timing-oracle-core/
├── Cargo.toml                    # New crate manifest
├── src/
│   ├── lib.rs                    # Crate root with no_std setup
│   ├── analysis/
│   │   ├── mod.rs                # Copy from timing-oracle
│   │   ├── bayes.rs              # Copy + minor edits
│   │   ├── effect.rs             # Copy + minor edits
│   │   ├── mde.rs                # Copy + minor edits
│   │   └── diagnostics.rs        # Copy + minor edits
│   ├── statistics/
│   │   ├── mod.rs                # Copy from timing-oracle
│   │   ├── bootstrap.rs          # Copy + alloc imports
│   │   ├── covariance.rs         # Copy + alloc imports
│   │   ├── quantile.rs           # Copy + alloc imports
│   │   ├── block_length.rs       # Copy + alloc imports
│   │   └── autocorrelation.rs    # Copy + alloc imports
│   ├── types.rs                  # Copy (already clean)
│   ├── result.rs                 # Copy + feature-gate std parts
│   └── constants.rs              # Copy (already clean)
```

### Files to MODIFY in timing-oracle:

```
timing-oracle/
├── Cargo.toml                    # Add timing-oracle-core dep, update workspace
├── src/
│   ├── lib.rs                    # Re-export from core, remove moved modules
│   ├── oracle.rs                 # Update imports to use core types
│   ├── config.rs                 # Update imports
│   ├── adaptive/
│   │   ├── loop_runner.rs        # Update imports
│   │   ├── calibration.rs        # Update imports
│   │   └── ...                   # Update imports
│   └── ...                       # Update imports throughout
```

### Files to DELETE from timing-oracle/src:

```
src/analysis/          # Moved to core
src/statistics/        # Moved to core
src/types.rs           # Moved to core (keep re-export)
src/result.rs          # Moved to core (keep re-export)
src/constants.rs       # Moved to core (keep re-export)
```

---

## Risk Mitigation

### Risk: Breaking existing users

**Mitigation**: Re-export everything from core in timing-oracle's lib.rs. Users see identical API.

### Risk: Dependency conflicts

**Mitigation**: Use identical versions of nalgebra, rand, serde in both crates. Workspace ensures consistency.

### Risk: no_std edge cases

**Mitigation**: Test on actual no_std target (thumbv7em-none-eabihf) in CI.

### Risk: Performance regression

**Mitigation**: Run benchmarks before/after. The code is identical, just reorganized.

---

## Timeline Summary

| Day | Tasks |
|-----|-------|
| 1 | Create crate structure, move files, initial Cargo.toml |
| 2 | Handle no_std compatibility (alloc imports, feature gates) |
| 3 | Update timing-oracle to depend on core, fix imports |
| 4 | Testing, CI setup, verify no_std builds |
| 5 | Documentation, final verification, cleanup |

**Total: ~5 working days** for a careful, tested extraction.

---

## Future: FFI Layer (Phase 2)

Once core is extracted, the FFI layer becomes straightforward:

```rust
// timing-oracle-ffi/src/lib.rs

use timing_oracle_core::*;

#[repr(C)]
pub struct OutcomeC {
    tag: u32,
    leak_probability: f64,
    shift_ns: f64,
    tail_ns: f64,
    // ...
}

#[no_mangle]
pub extern "C" fn timing_oracle_analyze(
    baseline_ns: *const f64,
    sample_ns: *const f64,
    count: usize,
    threshold_ns: f64,
    result_out: *mut OutcomeC,
) -> i32 {
    // Call core analysis functions
    // Convert result to C-compatible struct
}
```

This is a separate effort (~2-3 weeks) that builds on the core extraction.
