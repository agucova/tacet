# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`timing-oracle` is a Rust library for detecting timing side channels in cryptographic code. It uses statistical methodology to compare timing distributions between baseline and sample inputs, outputting leak probability, effect sizes, and exploitability assessments.

## Documentation Structure

| Document | Purpose |
|----------|---------|
| `docs/spec.md` | Core specification (v5.0) — language-agnostic statistical methodology, abstract types, normative requirements (RFC 2119) |
| `docs/guide.md` | User guide — getting started, threat model selection, interpreting results, common pitfalls |
| `docs/implementation-guide.md` | Implementation guide — platform-specific timers, pre-flight checks, measurement protocol, optimization |
| `docs/api-rust.md` | Rust API reference |
| `docs/api-c.md` | C/C++ API reference |
| `docs/api-go.md` | Go API reference |

**Spec section references in code:** The codebase contains 150+ references to spec sections (§2.3, §3.1, etc.). When updating the spec, ensure section numbers are updated in code comments.

## Build Commands

## Profiling (LLM-friendly)

Use `just profile bench` or `just profile tests` to produce folded stack output at `PROFILE_DIR` (default `/var/tmp/timing-oracle-profile`). Then run `just profile summary target=bench` (or `target=tests`) for a quick top-leaf summary; the folded stacks are plain text and safe to `grep`/`sort`/`awk`.


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

The timing oracle follows a two-phase adaptive Bayesian pipeline:

**Phase 1: Calibration** (5,000 samples)
1. **Preflight checks** (`crates/timing-oracle/src/preflight/`) - Validates measurement setup before analysis
2. **Measurement** (`crates/timing-oracle/src/measurement/`) - High-resolution cycle counting with interleaved sampling
3. **Outlier filtering** - Symmetric percentile-based trimming (99.9th percentile)
4. **Covariance estimation** - Block bootstrap (2,000 iterations, block length ~n^(1/3))
5. **Prior setup** - MDE-based priors from calibration data

**Phase 2: Adaptive Sampling** (until decision or budget)
1. **Batch collection** - Collect samples in batches (default 1,000)
2. **Posterior update** - Scale covariance (Σ_rate / n), compute posterior P(leak > θ | data)
3. **Decision check** - Pass if P < 0.05, Fail if P > 0.95
4. **Quality gates** - Stop early if data too noisy, not learning, or budget exceeded
5. **Effect decomposition** (`crates/timing-oracle/src/analysis/effect.rs`) - Separates uniform shift from tail effects

### Module Structure

- `TimingOracle` (`crates/timing-oracle/src/oracle.rs`) - Builder-pattern entry point via `for_attacker(AttackerModel)`
- `Config` (`crates/timing-oracle/src/config.rs`) - All tunable parameters (time_budget, max_samples, thresholds)
- `Outcome` (`crates/timing-oracle/src/result.rs`) - Output enum: Pass, Fail, Inconclusive, Unmeasurable
- `helpers` (`crates/timing-oracle/src/helpers.rs`) - Utilities for generating test inputs (`InputPair`, `byte_arrays_32`, `byte_vecs`)
- `adaptive/` - Adaptive sampling loop
  - `calibration.rs` - Calibration phase: covariance estimation, prior setup
  - `loop_runner.rs` - Adaptive sampling loop with posterior updates
  - `quality_gates.rs` - Five quality gate checks for early stopping
  - `state.rs` - AdaptiveState struct for tracking samples and posteriors
  - `kl_divergence.rs` - KL divergence for tracking learning rate
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
  - `bayes.rs` - Bayesian inference: posterior probability of leak
  - `effect.rs` - Effect decomposition (shift + tail)
  - `mde.rs` - Minimum detectable effect estimation
  - `diagnostics.rs` - Diagnostic checks for result reliability
- `statistics/` - Quantile computation, bootstrap resampling, covariance estimation
- `output/` - Terminal and JSON formatters

### Key Types

- `Outcome` - Four-variant enum:
  - `Pass { leak_probability, effect, quality, diagnostics, samples_used }` - No timing leak detected
  - `Fail { leak_probability, effect, exploitability, quality, diagnostics, samples_used }` - Timing leak confirmed
  - `Inconclusive { reason, leak_probability, effect, quality, diagnostics, samples_used }` - Could not reach decision
  - `Unmeasurable { operation_ns, threshold_ns, platform, recommendation }` - Operation too fast to measure
- `InconclusiveReason` - Why the test was inconclusive:
  - `DataTooNoisy` - Posterior ≈ prior after calibration
  - `NotLearning` - KL divergence collapsed for 5+ batches
  - `WouldTakeTooLong` - Projected time exceeds budget
  - `TimeBudgetExceeded` - Time budget exhausted
  - `SampleBudgetExceeded` - Sample limit reached
- `EffectEstimate` - Decomposed effect: shift_ns, tail_ns, credible_interval_ns, pattern
- `EffectPattern` - UniformShift, TailEffect, or Mixed
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
| `SharedHardware` | 0.6 ns (~2 cycles @ 3GHz) | SGX, cross-VM, containers, hyperthreading |
| `AdjacentNetwork` | 100 ns | LAN, HTTP/2 (Timeless Timing Attacks) |
| `RemoteNetwork` | 50 μs | Internet, legacy services |
| `Research` | 0 | Profiling, debugging (not for CI) |
| `Custom { threshold_ns }` | user-defined | Custom threshold in nanoseconds |

**Choosing a model:**
- **SharedHardware**: Co-resident attacker with cycle-level timing (SGX, cross-VM, containers)
- **AdjacentNetwork**: LAN attacker or HTTP/2 APIs (Timeless Timing Attacks enable LAN-like precision remotely)
- **RemoteNetwork**: General internet exposure, legacy HTTP/1.1 services
- **Research**: Detect any difference (not for production CI)

### Adaptive Batching

On platforms with coarse timer resolution (e.g., Apple Silicon's 42ns cntvct_el0), the library automatically batches operations:
- **Pilot phase**: Measures ~100 warmup iterations to determine operation time
- **K selection**: Chooses batch size K (1-20) to achieve 50+ timer ticks per measurement
- **Automatic disable**: Batching disabled on x86_64, when using PmuTimer/LinuxPerfTimer, or for slow operations

This compensates for timer quantization while avoiding microarchitectural artifacts from excessive batching.

### Test Organization

**Core Validation Tests:**
- `crates/timing-oracle/tests/known_leaky.rs` - Tests that MUST detect timing leaks (early-exit comparison, branches) [2 tests]
- `crates/timing-oracle/tests/known_safe.rs` - Tests that MUST NOT false-positive (XOR, constant-time comparison) [2 tests]
- `crates/timing-oracle/tests/calibration.rs` - Statistical validation (false positive rate, Bayesian calibration) [2 tests, 100 trials each]

**Comprehensive Integration Tests:**

All integration tests use **DudeCT's two-class pattern**:
- **Baseline class**: All-zero data (0x00 repeated)
- **Sample class**: Random data

This pattern tests for data-dependent timing rather than specific value comparisons.

- `crates/timing-oracle/tests/crypto_attacks.rs` - Real-world crypto timing attacks (AES S-box, modular exponentiation, cache effects, effect patterns, exploitability thresholds) [20 tests total]
  - 4 cache-based tests (AES, cache lines, memory patterns)
  - 2 modular exponentiation tests (square-and-multiply, bit patterns)
  - 3 table lookup tests (L1/L2/L3 cache)
  - 3 effect pattern validation tests (UniformShift, TailEffect, Mixed)
  - 4 exploitability threshold tests (Negligible, PossibleLAN, LikelyLAN, PossibleRemote)
  - 4 tests marked `#[ignore]` for thorough validation
- `crates/timing-oracle/tests/async_timing.rs` - Async/await and concurrent task timing [9 tests total]
  - 2 baseline tests (executor overhead, block_on symmetry)
  - 3 leak detection tests (conditional await, early exit, sleep duration)
  - 2 concurrent task tests (crosstalk, spawn count)
  - 2 optional thorough tests (runtime comparison, flag effectiveness)
- `crates/timing-oracle/tests/aes_timing.rs` - AES-128 encryption timing tests inspired by DudeCT [7 tests total]
  - Block encryption with zeros vs random plaintexts
  - Different keys with fixed plaintext
  - Multiple blocks cumulative timing
  - Key initialization timing
  - Hamming weight independence (0x00 vs 0xFF)
  - Byte pattern independence (sequential vs reverse)
- `crates/timing-oracle/tests/ecc_timing.rs` - Curve25519/X25519 elliptic curve timing tests [8 tests total]
  - Scalar multiplication with zeros vs random scalars
  - Different basepoints timing
  - Multiple operations cumulative timing
  - Scalar clamping timing
  - Hamming weight independence (0x00 vs 0xFF)
  - Byte pattern independence (sequential vs reverse)
  - Full ECDH key exchange timing
- `crates/timing-oracle/tests/hash_timing.rs` - SHA-3 and BLAKE2 hash timing tests [10 tests total]
  - SHA3-256, SHA3-384, SHA3-512 constant-time tests
  - BLAKE2b-512, BLAKE2s-256 constant-time tests
  - Incremental hashing tests (update + finalize)
  - Hamming weight independence for each hash variant
- `crates/timing-oracle/tests/aead_timing.rs` - AEAD cipher timing tests [8 tests total]
  - ChaCha20-Poly1305 encryption/decryption (RustCrypto)
  - AES-256-GCM encryption/decryption (ring)
  - Nonce independence tests
  - Hamming weight independence tests
- `crates/timing-oracle/tests/rsa_timing.rs` - RSA-2048 timing tests [6 tests total]
  - RSA encryption/decryption with PKCS#1 v1.5
  - RSA signing/verification (SHA-256)
  - Hamming weight independence (all-zeros vs all-ones)
  - Key size comparison (2048 vs 4096, informational)
- `crates/timing-oracle/tests/pqcrypto_timing.rs` - Post-quantum crypto timing tests [12 tests total]
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
use std::time::Duration;

let inputs = InputPair::new(
    || [0u8; 32],                 // Baseline: closure returning all zeros
    || rand::random::<[u8; 32]>() // Sample: closure returning random data
);

// Choose attacker model based on your threat scenario:
// - Public API? Use RemoteNetwork (50μs threshold)
// - Internal LAN service or HTTP/2? Use AdjacentNetwork (100ns threshold)
// - SGX/containers/shared hosting? Use SharedHardware (~2 cycles threshold)

let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .time_budget(Duration::from_secs(30))  // Optional: set time limit
    .max_samples(100_000)                   // Optional: set sample limit
    .test(inputs, |data| {
        my_crypto_function(data);
    });

// Handle result - four possible outcomes
match outcome {
    Outcome::Pass { leak_probability, .. } => {
        println!("No leak: P(leak)={:.1}%", leak_probability * 100.0);
    }
    Outcome::Fail { leak_probability, exploitability, .. } => {
        panic!("Timing leak detected: P(leak)={:.1}%, {:?}",
               leak_probability * 100.0, exploitability);
    }
    Outcome::Inconclusive { reason, leak_probability, .. } => {
        eprintln!("Inconclusive: P(leak)={:.1}%", leak_probability * 100.0);
    }
    Outcome::Unmeasurable { recommendation, .. } => {
        eprintln!("Skipping: {}", recommendation);
    }
}
```

### Choosing an Attacker Model

```rust
use timing_oracle::{TimingOracle, AttackerModel};
use std::time::Duration;

// Internet-facing API: attacker measures over general internet
TimingOracle::for_attacker(AttackerModel::RemoteNetwork)  // θ = 50μs

// Internal microservice or HTTP/2 API: attacker on LAN or using request multiplexing
TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)  // θ = 100ns

// SGX enclave, containers, or shared hosting: co-resident attacker
TimingOracle::for_attacker(AttackerModel::SharedHardware)  // θ = 0.6ns (~2 cycles)

// Research/debugging: detect any statistical difference
TimingOracle::for_attacker(AttackerModel::Research)  // θ → 0

// Custom threshold
TimingOracle::for_attacker(AttackerModel::Custom { threshold_ns: 500.0 })
```

### Configuration Options

```rust
use timing_oracle::{TimingOracle, AttackerModel};
use std::time::Duration;

// Quick check during development
TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .time_budget(Duration::from_secs(10))

// Thorough CI check
TimingOracle::for_attacker(AttackerModel::SharedHardware)
    .time_budget(Duration::from_secs(60))
    .max_samples(100_000)

// Custom decision thresholds
TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .pass_threshold(0.01)   // More confident pass (default 0.05)
    .fail_threshold(0.99)   // More confident fail (default 0.95)
```

### Macro API (with `macros` feature)

```rust
use timing_oracle::{timing_test_checked, TimingOracle, AttackerModel, Outcome};
use std::time::Duration;

// timing_test_checked! returns Outcome with all four variants
let outcome = timing_test_checked! {
    oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30)),
    baseline: || [0u8; 32],
    sample: || rand::random::<[u8; 32]>(),
    measure: |input| my_function(&input),
};

match outcome {
    Outcome::Pass { leak_probability, .. } => {
        println!("No leak: P(leak)={:.1}%", leak_probability * 100.0);
    }
    Outcome::Fail { leak_probability, exploitability, .. } => {
        panic!("Leak detected: {:?}", exploitability);
    }
    Outcome::Inconclusive { reason, .. } => {
        eprintln!("Inconclusive");
    }
    Outcome::Unmeasurable { recommendation, .. } => {
        eprintln!("Skipping: {}", recommendation);
    }
}
```

## Performance Optimization

### Time Budget Configuration

The adaptive oracle uses a time budget to balance speed vs accuracy:

```rust
use timing_oracle::{TimingOracle, AttackerModel};
use std::time::Duration;

// Quick check during development (~10 seconds)
TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .time_budget(Duration::from_secs(10))

// Standard CI check (~30 seconds, default)
TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .time_budget(Duration::from_secs(30))

// Thorough check for security-critical code (~60 seconds)
TimingOracle::for_attacker(AttackerModel::SharedHardware)
    .time_budget(Duration::from_secs(60))
    .max_samples(100_000)
```

### Parallel Processing

The `parallel` feature provides 4-8x speedup on multi-core systems by parallelizing bootstrap iterations across CPU cores using rayon. It's enabled by default.

To disable (e.g., for debugging):

```toml
[dependencies]
timing-oracle = { version = "0.1", default-features = false }
```

### Performance Tips

1. **Use shorter time budgets during development** (10 seconds is usually sufficient for iteration)
2. **Use longer budgets for CI** (30-60 seconds for thorough checks)
3. **Enable `parallel` feature** for maximum performance on multi-core systems
4. **Set `max_samples` for noisy environments** to bound worst-case runtime

### Performance Comparison

With parallel feature enabled on an 8-core machine:

| Time Budget | Typical Runtime | Use Case |
|-------------|-----------------|----------|
| 10 seconds | ~5-10s | Development iteration |
| 30 seconds | ~10-30s | Standard CI check |
| 60 seconds | ~30-60s | Security-critical validation |

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

The adaptive oracle handles reliability through its outcome types:

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome};
use std::time::Duration;

let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .time_budget(Duration::from_secs(30))
    .test(inputs, |d| op(d));

// The four outcomes handle different reliability scenarios:
match outcome {
    Outcome::Pass { quality, .. } => {
        // Reliable pass - quality indicates measurement confidence
        println!("Quality: {:?}", quality);
    }
    Outcome::Fail { exploitability, .. } => {
        // Reliable fail - leak confirmed
        panic!("Timing leak: {:?}", exploitability);
    }
    Outcome::Inconclusive { reason, leak_probability, .. } => {
        // Cannot reach conclusion - check reason for guidance
        eprintln!("Inconclusive: {:?}", reason);
    }
    Outcome::Unmeasurable { recommendation, .. } => {
        // Operation too fast - skip or use different timer
        eprintln!("Skipping: {}", recommendation);
    }
}

// Check if result is conclusive
if outcome.is_conclusive() {
    // Either Pass or Fail - can make decision
}

// Check if measurement was possible
if outcome.is_measurable() {
    // Not Unmeasurable - got some result
}
```

Environment variable `TIMING_ORACLE_UNRELIABLE_POLICY` can override: set to `fail_closed` for stricter CI.
