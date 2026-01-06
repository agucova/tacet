# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`timing-oracle` is a Rust library for detecting timing side channels in cryptographic code. It uses statistical methodology to compare timing distributions between baseline and sample inputs, outputting leak probability, effect sizes, and exploitability assessments.

## Build Commands

```bash
cargo build                    # Build with all features (parallel + kperf/perf)
cargo build --no-default-features  # Minimal build
cargo test                     # Run all tests (many are ignored pending implementation)
cargo test -- --ignored        # Run ignored tests (requires full implementation)
cargo test <test_name>         # Run a specific test
cargo check                    # Type-check without building
sudo cargo test                # Run all tests including kperf/perf tests (requires elevated privileges)
cargo run --example simple     # Run the simple example
cargo run --example compare    # Run the compare example
cargo run --example test_xor   # Run XOR constant-time test
cargo bench --bench comparison # Run DudeCT comparison benchmarks
```

### Default Features

By default, the library includes:
- **`parallel`** - Rayon-based parallel bootstrap (4-8x speedup)
- **`kperf`** (macOS ARM64 only) - PMU-based cycle counting (opt-in via `PmuTimer`, requires `sudo`)
- **`perf`** (Linux only) - perf_event-based cycle counting (opt-in via `LinuxPerfTimer`, requires `sudo`)

The `kperf`/`perf` features compile in advanced timer support that users can opt into by explicitly using `PmuTimer` or `LinuxPerfTimer`.

## Architecture

### Core Pipeline

The timing oracle follows a multi-layer analysis pipeline:
1. **Preflight checks** (`src/preflight/`) - Validates measurement setup before analysis
2. **Measurement** (`src/measurement/`) - High-resolution cycle counting with interleaved sampling
3. **Outlier filtering** - Symmetric percentile-based trimming (99.9th percentile)
4. **Sample splitting** - Temporal 30/70 split: calibration set (first 30%) for covariance estimation, inference set (last 70%) for hypothesis testing
5. **Quantile computation** - 9 decile differences (10th-90th percentile) between baseline/sample classes
6. **Covariance estimation** - Block bootstrap on calibration set (2,000 iterations, block length ~n^(1/3))
7. **CI Gate** (`src/analysis/ci_gate.rs`) - Layer 1: Max-statistic bootstrap threshold with bounded FPR
8. **Bayesian inference** (`src/analysis/bayes.rs`) - Layer 2: Posterior probability of timing leak via Bayes factor
9. **Effect decomposition** (`src/analysis/effect.rs`) - Separates uniform shift from tail effects

### Module Structure

- `TimingOracle` (`src/oracle.rs`) - Builder-pattern entry point with presets and configuration
- `Config` (`src/config.rs`) - All tunable parameters (samples, alpha, thresholds)
- `TestResult`, `Outcome` (`src/result.rs`) - Output types with leak_probability, effect, ci_gate, exploitability
- `helpers` (`src/helpers.rs`) - Utilities for generating test inputs (`InputPair`, `byte_arrays_32`, `byte_vecs`)
- `preflight/` - Pre-flight validation checks
  - `sanity.rs` - Timer sanity and harness checks
  - `generator.rs` - Generator overhead detection
  - `autocorr.rs` - Autocorrelation checks
  - `system.rs` - System configuration checks (CPU governor)
  - `resolution.rs` - Timer resolution checks
- `measurement/` - Platform-specific timing and sample collection
  - `timer.rs` - Standard timer (rdtsc/cntvct_el0)
  - `kperf.rs` - macOS PMU-based cycle counting (requires sudo)
  - `perf.rs` - Linux perf_event cycle counting (requires sudo)
  - `collector.rs` - Sample collection with adaptive batching
  - `outlier.rs` - Outlier filtering
- `analysis/` - Statistical analysis
  - `ci_gate.rs` - CI gate (Layer 1 frequentist test)
  - `bayes.rs` - Bayesian inference (Layer 2)
  - `effect.rs` - Effect decomposition
  - `mde.rs` - Minimum detectable effect estimation
- `statistics/` - Quantile computation, bootstrap resampling, covariance estimation
- `output/` - Terminal and JSON formatters

### Key Types

- `Outcome` - Top-level result: `Completed(TestResult)` or `Unmeasurable`
- `TestResult` - Full analysis result with leak_probability, effect, ci_gate, exploitability, quality
- `Effect` - Decomposed effect: shift_ns, tail_ns, credible_interval_ns, pattern
- `EffectPattern` - UniformShift, TailEffect, or Mixed
- `CiGate` - Pass/fail decision with threshold and observed max statistic
- `Exploitability` - Negligible (<100ns), PossibleLAN (100-500ns), LikelyLAN (500ns-20μs), PossibleRemote (>20μs)
- `MeasurementQuality` - Excellent (<5ns MDE), Good (5-20ns), Poor (20-100ns), TooNoisy (>100ns)
- `MinDetectableEffect` - Minimum detectable shift_ns and tail_ns
- `AttackerModel` - Threat model presets for minimum effect thresholds (see below)
- `Class::Baseline` / `Class::Sample` - Input class identifiers
- `InputPair<T>` - Helper for generating baseline/sample test inputs via closures

### Attacker Model Presets

**There is no single correct threshold (θ).** Your choice of attacker model is a statement about your threat model. The library provides presets based on academic research:

| Preset | θ | Use case |
|--------|---|----------|
| `LocalCycles` | 2 cycles | SGX enclaves, shared hosting, local privilege escalation |
| `LocalCoarseTimer` | 1 tick | Sandboxed environments with coarse timers |
| `LANStrict` | 2 cycles | High-security LAN (Kario-style, most strict) |
| `LANConservative` | 100 ns | Internal services, microservices (Crosby-style) |
| `WANOptimistic` | 15 μs | Same-region cloud, low-jitter internet paths |
| `WANConservative` | 50 μs | Public APIs, general internet exposure |
| `KyberSlashSentinel` | 10 cycles | Post-quantum crypto, catch ~20-cycle leaks |
| `Research` | 0 | Academic analysis, detect any difference (not for CI) |

**The LAN dispute:** `LANStrict` vs `LANConservative` represents a genuine dispute in the literature:
- **Kario** argues timing differences as small as one clock cycle can be detectable over LAN
- **Crosby et al. (2009)** reports ~100ns accuracy over local networks

Pick based on your threat model, not convenience. For cycle-based presets on coarse timers, the library warns when the threshold would be smaller than the timer resolution.

### Adaptive Batching

On platforms with coarse timer resolution (e.g., Apple Silicon's 42ns cntvct_el0), the library automatically batches operations:
- **Pilot phase**: Measures ~100 warmup iterations to determine operation time
- **K selection**: Chooses batch size K (1-20) to achieve 50+ timer ticks per measurement
- **Automatic disable**: Batching disabled on x86_64, when using PmuTimer/LinuxPerfTimer, or for slow operations

This compensates for timer quantization while avoiding microarchitectural artifacts from excessive batching.

### Test Organization

**Core Validation Tests:**
- `tests/known_leaky.rs` - Tests that MUST detect timing leaks (early-exit comparison, branches) [2 tests]
- `tests/known_safe.rs` - Tests that MUST NOT false-positive (XOR, constant-time comparison) [2 tests]
- `tests/calibration.rs` - Statistical validation (CI gate FPR, Bayesian calibration) [2 tests, 100 trials each]

**Comprehensive Integration Tests:**

All integration tests use **DudeCT's two-class pattern**:
- **Baseline class**: All-zero data (0x00 repeated)
- **Sample class**: Random data

This pattern tests for data-dependent timing rather than specific value comparisons.

- `tests/crypto_attacks.rs` - Real-world crypto timing attacks (AES S-box, modular exponentiation, cache effects, effect patterns, exploitability thresholds) [20 tests total]
  - 4 cache-based tests (AES, cache lines, memory patterns)
  - 2 modular exponentiation tests (square-and-multiply, bit patterns)
  - 3 table lookup tests (L1/L2/L3 cache)
  - 3 effect pattern validation tests (UniformShift, TailEffect, Mixed)
  - 4 exploitability threshold tests (Negligible, PossibleLAN, LikelyLAN, PossibleRemote)
  - 4 tests marked `#[ignore]` for thorough validation
- `tests/async_timing.rs` - Async/await and concurrent task timing [9 tests total]
  - 2 baseline tests (executor overhead, block_on symmetry)
  - 3 leak detection tests (conditional await, early exit, sleep duration)
  - 2 concurrent task tests (crosstalk, spawn count)
  - 2 optional thorough tests (runtime comparison, flag effectiveness)
- `tests/aes_timing.rs` - AES-128 encryption timing tests inspired by DudeCT [7 tests total]
  - Block encryption with zeros vs random plaintexts
  - Different keys with fixed plaintext
  - Multiple blocks cumulative timing
  - Key initialization timing
  - Hamming weight independence (0x00 vs 0xFF)
  - Byte pattern independence (sequential vs reverse)
- `tests/ecc_timing.rs` - Curve25519/X25519 elliptic curve timing tests [8 tests total]
  - Scalar multiplication with zeros vs random scalars
  - Different basepoints timing
  - Multiple operations cumulative timing
  - Scalar clamping timing
  - Hamming weight independence (0x00 vs 0xFF)
  - Byte pattern independence (sequential vs reverse)
  - Full ECDH key exchange timing
- `tests/hash_timing.rs` - SHA-3 and BLAKE2 hash timing tests [10 tests total]
  - SHA3-256, SHA3-384, SHA3-512 constant-time tests
  - BLAKE2b-512, BLAKE2s-256 constant-time tests
  - Incremental hashing tests (update + finalize)
  - Hamming weight independence for each hash variant
- `tests/aead_timing.rs` - AEAD cipher timing tests [8 tests total]
  - ChaCha20-Poly1305 encryption/decryption (RustCrypto)
  - AES-256-GCM encryption/decryption (ring)
  - Nonce independence tests
  - Hamming weight independence tests
- `tests/rsa_timing.rs` - RSA-2048 timing tests [6 tests total]
  - RSA encryption/decryption with PKCS#1 v1.5
  - RSA signing/verification (SHA-256)
  - Hamming weight independence (all-zeros vs all-ones)
  - Key size comparison (2048 vs 4096, informational)
- `tests/pqcrypto_timing.rs` - Post-quantum crypto timing tests [12 tests total]
  - ML-KEM (Kyber-768): encapsulation/decapsulation
  - ML-DSA (Dilithium3): signing/verification
  - Falcon-512: signing/verification
  - SPHINCS+-SHA2-128s: signing/verification
  - Tests use pqcrypto crate (PQClean C bindings)

**Test Execution:**
```bash
# Fast suite (non-ignored tests)
cargo test --test crypto_attacks  # ~3-5 minutes
cargo test --test async_timing    # ~20-40 seconds
cargo test --test aes_timing      # ~2-3 minutes
cargo test --test ecc_timing      # ~2-3 minutes
cargo test --test hash_timing     # ~3-5 minutes
cargo test --test aead_timing     # ~2-3 minutes
cargo test --test rsa_timing      # ~5-10 minutes (RSA is slow)
cargo test --test pqcrypto_timing # ~5-8 minutes (PQ crypto)

# Full suite (includes ignored tests)
cargo test --test crypto_attacks -- --ignored  # ~15-20 minutes
cargo test --test async_timing -- --ignored    # ~5-10 minutes
cargo test --test rsa_timing -- --ignored      # ~15-20 minutes

# All integration tests
cargo test --test known_leaky --test known_safe --test calibration --test crypto_attacks --test async_timing --test aes_timing --test ecc_timing --test hash_timing --test aead_timing --test rsa_timing --test pqcrypto_timing
```

See `TESTING.md` for detailed documentation of each test category.

## API Usage Pattern

```rust
// Recommended: Use attacker model presets to define your threat model
use timing_oracle::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};

let inputs = InputPair::new(
    || [0u8; 32],                 // Baseline: closure returning all zeros
    || rand::random::<[u8; 32]>() // Sample: closure returning random data
);

// Choose attacker model based on your threat scenario:
// - Public API? Use WANConservative (50μs threshold)
// - Internal LAN service? Use LANConservative (100ns threshold)
// - SGX/shared hosting? Use LocalCycles (2 cycles threshold)
// - Post-quantum crypto? Use KyberSlashSentinel (10 cycles threshold)

let outcome = TimingOracle::for_attacker(AttackerModel::LANConservative)
    .samples(20_000)              // Optional: adjust sample count
    .test(inputs, |data| {
        my_crypto_function(data);
    });

// Handle result
match outcome {
    Outcome::Completed(result) => {
        assert!(result.ci_gate.passed, "Timing leak detected!");
    }
    Outcome::Unmeasurable { recommendation, .. } => {
        eprintln!("Skipping: {}", recommendation);
    }
}
```

### Choosing an Attacker Model

```rust
use timing_oracle::{TimingOracle, AttackerModel};

// Internet-facing API: attacker measures over general internet
TimingOracle::for_attacker(AttackerModel::WANConservative)  // θ = 50μs

// Internal microservice: attacker on local network
TimingOracle::for_attacker(AttackerModel::LANConservative)  // θ = 100ns

// High-security LAN (strict interpretation)
TimingOracle::for_attacker(AttackerModel::LANStrict)        // θ = 2 cycles

// SGX enclave or shared hosting: co-resident attacker
TimingOracle::for_attacker(AttackerModel::LocalCycles)      // θ = 2 cycles

// Post-quantum crypto (catch KyberSlash-class bugs)
TimingOracle::for_attacker(AttackerModel::KyberSlashSentinel) // θ = 10 cycles

// Research/debugging: detect any statistical difference
TimingOracle::for_attacker(AttackerModel::Research)         // θ → 0

// Custom threshold
TimingOracle::for_attacker(AttackerModel::CustomNs { threshold_ns: 500.0 })
```

### Combining Presets

```rust
// Attacker model + sample count preset
TimingOracle::for_attacker(AttackerModel::LANConservative)
    .samples(20_000)  // Faster than default 100k

// Or start with sample preset, add attacker model
TimingOracle::balanced()
    .attacker_model(AttackerModel::WANConservative)

// CI integration with environment variable overrides
TimingOracle::for_attacker(AttackerModel::LANConservative)
    .from_env()  // Override with TO_SAMPLES, TO_ALPHA env vars
    .test(inputs, |data| my_function(data));
```

### Legacy: Manual Threshold (not recommended)

For backwards compatibility, you can still set thresholds manually:

```rust
// Not recommended - use attacker models instead
TimingOracle::balanced()
    .min_effect_ns(100.0)  // Manual threshold in nanoseconds
```

### Macro API (with `macros` feature)

```rust
use timing_oracle::{timing_test, timing_test_checked, TimingOracle, AttackerModel};

// timing_test! returns TestResult directly (panics if unmeasurable)
let result = timing_test! {
    oracle: TimingOracle::for_attacker(AttackerModel::LANConservative),
    baseline: || [0u8; 32],
    sample: || rand::random::<[u8; 32]>(),
    measure: |input| my_function(&input),
};
assert!(result.ci_gate.passed);

// timing_test_checked! returns Outcome for explicit unmeasurable handling
let outcome = timing_test_checked! {
    oracle: TimingOracle::for_attacker(AttackerModel::WANConservative),
    baseline: || [0u8; 32],
    sample: || rand::random::<[u8; 32]>(),
    measure: |input| my_function(&input),
};
```

## Performance Optimization

### Configuration Presets

**Sample count presets** control speed vs accuracy. Combine with attacker models:

```rust
use timing_oracle::{TimingOracle, AttackerModel};

// Default - Most accurate, slowest (~5-10 seconds per test)
// 100k samples, 10,000 CI bootstrap, 2,000 covariance bootstrap
TimingOracle::for_attacker(AttackerModel::LANConservative)

// Balanced - Recommended for production (~1-2 seconds per test)
// 20k samples, 100 CI bootstrap, 50 covariance bootstrap
TimingOracle::for_attacker(AttackerModel::LANConservative)
    .samples(20_000)

// Quick - Fast iteration during development (~0.2-0.5 seconds per test)
// 5k samples, 50 CI bootstrap, 50 covariance bootstrap
TimingOracle::quick()
    .attacker_model(AttackerModel::LANConservative)

// Calibration - For running many trials (100+) (~0.1-0.2 seconds per test)
// 2k samples, 30 CI bootstrap, 20 covariance bootstrap
TimingOracle::calibration()
    .attacker_model(AttackerModel::Research)
```

### Parallel Processing

The `parallel` feature provides 4-8x speedup on multi-core systems by parallelizing bootstrap iterations across CPU cores using rayon. It's enabled by default.

To disable (e.g., for debugging):

```toml
[dependencies]
timing-oracle = { version = "0.1", default-features = false }
```

### Performance Tips

1. **Start with `.balanced()`** for most use cases - 5x faster than default with minimal accuracy loss
2. **Use `.calibration()` for test suites** that run 100+ trials
3. **Enable `parallel` feature** for maximum performance on multi-core systems
4. **Use `.quick()` during development** for rapid iteration

### Performance Comparison

With parallel feature enabled on an 8-core machine:

| Preset | Samples | Runtime | Speedup vs Default |
|--------|---------|---------|-------------------|
| Default | 100k | ~5s | 1x |
| Balanced | 20k | ~1s | 5x |
| Quick | 5k | ~0.3s | 17x |
| Calibration | 2k | ~0.15s | 33x |

## Feature Flags

Default features:
- `parallel` - Rayon-based parallel bootstrap (4-8x speedup)
- `kperf` - PMU-based cycle counting on macOS ARM64 (opt-in via `PmuTimer`, requires sudo)
- `perf` - perf_event-based cycle counting on Linux (opt-in via `LinuxPerfTimer`, requires sudo/CAP_PERFMON)

Optional features:
- `macros` - Proc macros (`timing_test!`, `timing_test_checked!`) for ergonomic test syntax

Minimal build:
```bash
cargo build --no-default-features
```

## Reliability Handling

Results may be unreliable on noisy systems or with coarse timers. Use the reliability macros:

```rust
use timing_oracle::{skip_if_unreliable, require_reliable, TimingOracle, AttackerModel, Outcome};

// Fail-open: Skip test if unreliable (returns early)
let outcome = TimingOracle::for_attacker(AttackerModel::LANConservative)
    .test(inputs, |d| op(d));
let result = skip_if_unreliable!(outcome, "test_name");

// Fail-closed: Panic if unreliable
let result = require_reliable!(outcome, "test_name");

// Check reliability programmatically
if outcome.is_reliable() {
    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}
```

Environment variable `TIMING_ORACLE_UNRELIABLE_POLICY` can override: set to `fail_closed` for stricter CI.
