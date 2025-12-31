# timing-oracle

**Detect timing side channels in Rust code with statistically rigorous methods.**

`timing-oracle` measures whether your code's execution time depends on secret data. It compares timing between a "fixed" input (e.g., all zeros, valid padding) and random inputs, then uses Bayesian inference to compute the probability of a timing leak and estimate its magnitude.

Unlike simple t-tests, this crate provides:
- **Leak probability** (0-100%): A Bayesian posterior, not a p-value
- **Effect size in nanoseconds**: How big is the leak?
- **CI gate with bounded false positives**: Reliable pass/fail for continuous integration
- **Exploitability assessment**: Is this leak practically exploitable?

## Highlights

**Performance**: 3.6-8× faster per-sample than DudeCT (14-18 ns/sample vs 52 ns/sample on Apple Silicon M3 Max)

**Detection accuracy**: 100% true positive rate, 0% false positive rate on known-leaky and known-safe test cases

**Sample efficiency**: Median 5-7k samples to reliably detect early-exit timing leaks (95% CI: 0.7k-35k samples)

**Statistical rigor**: Two-layer analysis combining frequentist CI gate (bounded ≤1% FPR) with Bayesian inference for interpretable probabilities

**Production-ready**: Successfully validates constant-time implementations in AES-128 encryption and Curve25519/X25519 ECDH

See `cargo bench --bench comparison` for detailed performance comparisons against DudeCT.

## Quick Start

```rust
use std::cell::Cell;
use timing_oracle::test;

let secret = [0u8; 32];

// CRITICAL: Both closures must execute IDENTICAL code paths.
// Pre-generate inputs for BOTH classes - only the DATA differs.
let fixed_inputs: Vec<[u8; 32]> = vec![[0u8; 32]; 100_000];    // All same
let random_inputs: Vec<[u8; 32]> = (0..100_000)
    .map(|_| rand_bytes())
    .collect();

// Shared index - both closures do identical Cell + Vec operations
let idx = Cell::new(0usize);

let result = test(
    || {
        let i = idx.get();
        idx.set(i.wrapping_add(1));
        compare(&secret, &fixed_inputs[i % fixed_inputs.len()])
    },
    || {
        let i = idx.get();
        idx.set(i.wrapping_add(1));
        compare(&secret, &random_inputs[i % random_inputs.len()])
    },
);

if result.leak_probability > 0.9 {
    panic!("Timing leak detected with {:.0}% confidence",
           result.leak_probability * 100.0);
}
```

## Common Mistakes

### ❌ Don't Generate Random Data Inside Closures

The most common mistake is calling RNG functions or allocating memory inside the measured closures:

```rust
// ❌ WRONG - RNG and allocation inside closure
let result = test(
    || encrypt(&KEY, &[0u8; 32]),
    || {
        let random_data: Vec<u8> = (0..32).map(|_| rand::random()).collect();
        encrypt(&KEY, &random_data)  // Measures RNG + allocation + encrypt!
    },
);
```

**Why this fails:**
- The random closure includes RNG overhead (~500+ cycles) + allocation
- The fixed closure is just the encrypt operation
- You're measuring RNG noise, not the encrypt timing difference
- Weak timing leaks (<100ns) are drowned out by this noise

**✅ Correct - Pre-generate all inputs:**

```rust
use timing_oracle::helpers::InputPair;

// Pre-generate outside the closures
let inputs = InputPair::new(
    [0u8; 32],                    // Fixed: all zeros
    || rand::random::<[u8; 32]>() // Random: generated once per sample
);

let result = test(
    || encrypt(&KEY, &inputs.fixed()),
    || encrypt(&KEY, &inputs.random()),
);
```

Both closures now execute **identical code paths** - only the data differs.

The `helpers` module provides convenient utilities for common patterns:

```rust
use timing_oracle::helpers;

// Byte arrays
let inputs = helpers::byte_arrays_32();

// Byte vectors of specific length
let inputs = helpers::byte_vecs(1024);

// Custom types with generators
let inputs = InputPair::from_fn(
    || vec![0u8; 1024],           // Fixed generator
    || random_vec(1024)           // Random generator
);
```

### Side-Effects to Avoid

Inside measured closures, never:
- Call `rand::random()` or any RNG functions
- Allocate with `vec![]`, `Vec::new()`, `String::new()`, etc.
- Perform I/O (file reads, network calls)
- Use `println!()` or other logging
- Access thread-locals or global state inconsistently

All setup must happen **before** the closures are defined.

## DudeCT Two-Class Pattern

This library follows **DudeCT's two-class testing pattern** for detecting data-dependent timing:

- **Class 0**: All-zero data (0x00 repeated)
- **Class 1**: Random data

This pattern tests whether operations have **data-dependent timing** rather than comparing specific fixed values:

```rust
use timing_oracle::TimingOracle;

let result = TimingOracle::new()
    .samples(50_000)
    .test(
        || {
            // Class 0: All zeros
            let data = [0u8; 32];
            my_crypto_operation(&data)
        },
        || {
            // Class 1: Random
            let data = rand_bytes();
            my_crypto_operation(&data)
        },
    );

// Modern crypto should have Negligible exploitability even if timing differences detected
if !result.ci_gate.passed {
    assert!(matches!(result.exploitability, Exploitability::Negligible));
}
```

**Why this pattern works:**
- Simpler than fixed-vs-variable input patterns
- Tests for data dependencies in crypto operations
- Matches DudeCT's proven methodology
- See `tests/aes_timing.rs` and `tests/ecc_timing.rs` for real-world examples

## Installation

```toml
[dev-dependencies]
timing-oracle = "0.1"
```

### Feature Flags

By default, the library includes:
- **`parallel`** - Rayon-based parallel bootstrap (4-8x speedup on multi-core systems)
- **`kperf`** (macOS ARM64 only) - Compiles PMU-based cycle counting support (opt-in, requires `sudo`)
- **`perf`** (Linux only) - Compiles perf_event-based cycle counting support (opt-in, requires `sudo` or `CAP_PERFMON`)

The kperf/perf features compile in advanced timer support but don't enable it by default - see "Advanced: Cycle-Accurate Timing" below for how to use them.

**To disable default features (e.g., for minimal builds):**
```toml
[dev-dependencies]
timing-oracle = { version = "0.1", default-features = false }
```

**Advanced: Cycle-Accurate Timing (Optional)**

For maximum timing precision on Apple Silicon or Linux, you can explicitly use PMU-based timers:

```rust
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
use timing_oracle::measurement::kperf::PmuTimer;

#[cfg(all(target_os = "linux", feature = "perf"))]
use timing_oracle::measurement::perf::LinuxPerfTimer;

// macOS: Use PmuTimer (requires sudo)
let pmu_timer = PmuTimer::new().expect("Run with sudo for PMU access");
let cycles = pmu_timer.measure_cycles(|| my_operation());

// Linux: Use LinuxPerfTimer (requires sudo or CAP_PERFMON)
let perf_timer = LinuxPerfTimer::new().expect("Run with sudo or grant CAP_PERFMON");
let cycles = perf_timer.measure_cycles(|| my_operation());
```

These advanced timers provide ~1ns resolution on Apple Silicon (vs ~42ns default) and on ARM64 Linux systems with coarse timers.

## Build & Test

```bash
cargo build                     # Build with all features (parallel + kperf/perf)
cargo test                      # Run all tests (unit + integration)
cargo test --lib                # Run only unit tests

# Integration tests
cargo test --test known_leaky   # Run leak detection tests
cargo test --test known_safe    # Run false-positive tests
cargo test --test calibration   # Run statistical calibration tests
cargo test --test aes_timing    # Run AES-128 encryption tests (DudeCT pattern)
cargo test --test ecc_timing    # Run Curve25519 ECDH tests (DudeCT pattern)

# Fast test execution
cargo nextest run               # Faster test runner (parallel enabled by default)
cargo nextest run --profile timing # Single-threaded timing profile

# Run tests (includes kperf/perf test suites if on macOS/Linux with sudo)
sudo cargo test                 # Tests PmuTimer/LinuxPerfTimer functionality

# Benchmarks
cargo bench                     # Microbenchmarks (Criterion, small sample sizes)
cargo bench --bench comparison  # Compare with DudeCT (comprehensive evaluation, ~10 min)
                                # Sections: Detection rate, ROC curves, sample efficiency

# Examples
cargo run --example simple      # Basic usage demonstration
cargo run --example compare     # Vulnerable vs constant-time comparison
cargo run --example aes         # AES timing analysis
cargo run --example test_xor    # XOR operation (should be constant-time)
```

**Note:** Integration tests use 50k-100k samples and may take 2-5 minutes each. Use `TimingOracle::quick()` during development for faster iteration.

If you're on Nix, `devenv shell` will bring in Rust toolchain + cargo-nextest/cargo-llvm-cov.

## The Problem

Timing side channels leak secrets through execution time variations. Classic examples:

```rust
// VULNERABLE: Early-exit comparison
fn compare(a: &[u8], b: &[u8]) -> bool {
    for i in 0..a.len() {
        if a[i] != b[i] {
            return false;  // Exits early on mismatch!
        }
    }
    true
}
```

When comparing a secret against attacker-controlled input, the early exit reveals how many bytes match. An attacker can guess the secret byte-by-byte.

Similar issues affect:
- RSA implementations with non-constant-time modular exponentiation
- AES with table lookups that hit cache differently
- HMAC verification with short-circuit comparison
- Password hashing that exits early on format validation

Existing tools like [dudect](https://github.com/oreparaz/dudect) detect such leaks but have limitations:
- Output t-statistics instead of interpretable probabilities
- No bounded false positive rate for CI integration
- Miss non-uniform timing effects (e.g., cache-related tail behavior)
- 3.6-8× slower per-sample measurement overhead (52 ns/sample vs 14-18 ns/sample for timing-oracle)

## Timer Selection and Adaptive Batching

The library automatically selects the best available timer for your platform:

### Platform Timers

**x86_64:**
- Uses `rdtsc` instruction (~1ns resolution)
- Cycle-accurate timing without requiring privileges

**macOS ARM64 (Apple Silicon):**
- **Standard Timer** (default): `cntvct_el0` virtual timer (~42ns resolution for M1/M2/M3 at 24 MHz)
- **PmuTimer** (opt-in, requires `sudo`): PMU-based cycle counting (~1ns resolution) - see "Advanced: Cycle-Accurate Timing"

**Linux:**
- **x86_64**: `rdtsc` instruction (~1ns, no privileges needed)
- **ARM64 Standard Timer** (default): `cntvct_el0` virtual timer (resolution varies by SoC)
  - ARMv8.6+ (Graviton4): ~1ns (1 GHz)
  - Ampere Altra: ~40ns (25 MHz)
  - Raspberry Pi 4: ~18ns (54 MHz)
- **LinuxPerfTimer** (opt-in, requires `sudo`/`CAP_PERFMON`): perf_event cycle counting (~1ns) - see "Advanced: Cycle-Accurate Timing"

### Adaptive Batching

On platforms with coarse timer resolution (>5 ticks per operation), the library automatically enables **adaptive batching**:

1. **Pilot phase**: Measures ~100 warmup iterations to determine median operation time
2. **K selection**: Chooses batch size K to achieve 50+ timer ticks per batch
3. **Batch measurement**: Measures K iterations together and analyzes batch totals (reported with K)
4. **Bounded batching**: Never exceeds K=20 to prevent microarchitectural artifacts

**Example:** On Apple Silicon (42ns resolution) measuring a 100ns operation:
- Single call: ~2.4 ticks → unreliable (quantization noise)
- Batch of K=21: ~50 ticks → stable distribution for statistical inference

Batching is **automatically disabled** when:
- Timer resolution is fine enough (>5 ticks per call)
- Using `PmuTimer` (macOS) or `LinuxPerfTimer` (Linux) for cycle-accurate timing
- Operation is slow enough (>210ns on Apple Silicon with standard timer)
- Running on x86_64 (rdtsc provides cycle-accurate timing)

Batching is automatic in the public API; to avoid it entirely, use `PmuTimer` (macOS) or `LinuxPerfTimer` (Linux) for cycle-accurate timing.

## Architecture

For a detailed explanation of the internal architecture, statistical methodology, and implementation decisions, see [`docs/architecture.md`](docs/architecture.md).

## How It Works

### Measurement Protocol

1. **Warmup**: Run both operations to stabilize CPU state
2. **Interleaved sampling**: Alternate fixed/random in randomized order to prevent drift bias
3. **High-precision timing**: Use `rdtsc` (x86) or `cntvct_el0` (ARM) with serialization barriers
4. **Outlier filtering**: Apply symmetric percentile threshold to both classes

### Two-Layer Analysis

The key insight is that CI gates and interpretable statistics need different methodologies:

```
                    Timing Samples
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
   ┌───────────────┐           ┌──────────────────┐
   │  Layer 1:     │           │  Layer 2:        │
   │  CI Gate      │           │  Bayesian        │
   ├───────────────┤           ├──────────────────┤
   │ RTLF-style    │           │ Closed-form      │
   │ bootstrap     │           │ conjugate Bayes  │
   │               │           │                  │
   │ Bounded FPR   │           │ Probability +    │
   │ Pass/Fail     │           │ Effect size      │
   └───────────────┘           └──────────────────┘
```

**Layer 1 (CI Gate)** answers: "Should this block my build?"
- Uses RTLF-style bootstrap thresholds ([Dunsche et al., 2024](https://www.usenix.org/conference/usenixsecurity24/presentation/dunsche))
- Conservative max-based per-quantile bounds
- Guarantees false positive rate ≤ α (default 1%)

**Layer 2 (Bayesian)** answers: "What's the probability and magnitude?"
- Computes Bayes factor between H₀ (no leak) and H₁ (leak)
- Decomposes effect into uniform shift and tail components
- Provides 95% credible intervals

### Quantile-Based Statistics

Rather than comparing means (which miss distributional differences), we compare nine deciles:

```
Fixed class:   [q₁₀, q₂₀, q₃₀, q₄₀, q₅₀, q₆₀, q₇₀, q₈₀, q₉₀]
Random class:  [q₁₀, q₂₀, q₃₀, q₄₀, q₅₀, q₆₀, q₇₀, q₈₀, q₉₀]
Difference:    Δ ∈ ℝ⁹
```

This captures:
- **Uniform shifts**: Code path takes different branch (affects all quantiles equally)
- **Tail effects**: Cache misses on specific inputs (affects upper quantiles more)

The effect is decomposed using an orthogonalized basis:
```
Δ = μ·[1,1,...,1] + τ·[-0.5, -0.375, ..., 0.375, 0.5] + noise
```

Where μ is the uniform shift and τ is the tail effect.

## API Reference

### Builder Pattern

```rust
use timing_oracle::TimingOracle;

let result = TimingOracle::new()
    .samples(100_000)           // Samples per class
    .warmup(1_000)              // Warmup iterations
    .ci_alpha(0.01)             // CI false positive rate
    .effect_prior_ns(10.0)      // Prior scale for effects (not a hard cutoff)
    .test(fixed_fn, random_fn);
```

### CI-focused builder

```rust
#[test]
fn compare_is_constant_time() {
    use timing_oracle::{Mode, TimingOracle};

    TimingOracle::ci_test()
        .from_env() // TO_MODE/TO_SAMPLES/TO_ALPHA/TO_REPORT/TO_SEED/...
        .mode(Mode::Smoke)
        .fail_on(timing_oracle::FailCriterion::CiGate)
        .run(
            || fixed_compare(&FIXED),
            || random_compare(&rand_input()),
        )
        .unwrap_or_report();
}
```

For async workloads, call `.async_workload(true)` or set `TO_ASYNC_WORKLOAD=1` to inflate priors/thresholds for higher noise and log the async flag.

### Test Result

```rust
pub struct TestResult {
    /// Bayesian posterior probability of timing leak (0.0 to 1.0)
    pub leak_probability: f64,

    /// Effect size (present if leak_probability > 0.5)
    pub effect: Option<Effect>,

    /// Exploitability assessment (heuristic)
    pub exploitability: Exploitability,

    /// CI gate result for pass/fail decisions
    pub ci_gate: CiGate,

    /// Measurement quality (Excellent/Good/Poor/TooNoisy)
    pub quality: MeasurementQuality,

    // ... other fields
}
```

### Effect Decomposition

```rust
pub struct Effect {
    /// Uniform shift in nanoseconds (positive = fixed is slower)
    pub shift_ns: f64,

    /// Tail effect in nanoseconds (positive = fixed has heavier tail)
    pub tail_ns: f64,

    /// 95% credible interval for total effect
    pub credible_interval_ns: (f64, f64),

    /// Dominant pattern
    pub pattern: EffectPattern,  // UniformShift, TailEffect, or Mixed
}
```

### Exploitability Thresholds

Based on [Crosby et al. (2009)](https://dl.acm.org/doi/10.1145/1455770.1455794):

| Effect Size | Assessment | Implications |
|------------|------------|--------------|
| < 100 ns | `Negligible` | Requires impractical measurement count |
| 100-500 ns | `PossibleLAN` | Exploitable on LAN with ~100k queries |
| 500 ns - 20 μs | `LikelyLAN` | Likely exploitable on LAN |
| > 20 μs | `PossibleRemote` | Possibly exploitable over internet |

## CI Integration

### Recommended Thresholds

```rust
#[test]
fn test_constant_time_compare() {
    let result = TimingOracle::new()
        .samples(100_000)
        .ci_alpha(0.001)  // Tighter for CI
        .test(/* ... */);

    // Two-tier decision:
    // - ci_gate.passed: Bounded FPR at alpha level
    // - leak_probability: Bayesian interpretation

    assert!(result.ci_gate.passed,
            "Timing leak detected (CI gate failed)");
    assert!(result.leak_probability < 0.5,
            "Elevated leak probability: {:.0}%",
            result.leak_probability * 100.0);
}
```

### Handling Multiple Tests

With α = 0.01 per test, running N tests gives P(≥1 false positive) ≈ 1 - (1-α)^N.

For large test suites:
- Use stricter α (e.g., 0.001)
- Treat first failure as warning, require confirmation on re-run
- Use hierarchical gating: `leak_probability > 0.99` = fail, `> 0.9` = warn

## Example: Constant-Time Comparison

```rust
use timing_oracle::TimingOracle;

fn main() {
    let secret = [0xABu8; 32];

    // Test variable-time (leaky)
    let result = TimingOracle::new()
        .samples(50_000)
        .test(
            || variable_time_compare(&secret, &[0xAB; 32]),
            || variable_time_compare(&secret, &rand_bytes()),
        );

    println!("Variable-time: {:.0}% leak probability",
             result.leak_probability * 100.0);
    // Output: Variable-time: 97% leak probability

    // Test constant-time (safe)
    let result = TimingOracle::new()
        .samples(50_000)
        .test(
            || constant_time_compare(&secret, &[0xAB; 32]),
            || constant_time_compare(&secret, &rand_bytes()),
        );

    println!("Constant-time: {:.0}% leak probability",
             result.leak_probability * 100.0);
    // Output: Constant-time: 12% leak probability
}

// VULNERABLE: Early-exit comparison
fn variable_time_compare(a: &[u8], b: &[u8]) -> bool {
    for i in 0..a.len() {
        if a[i] != b[i] { return false; }
    }
    true
}

// SAFE: Constant-time comparison
fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    let mut acc = 0u8;
    for i in 0..a.len() {
        acc |= a[i] ^ b[i];
    }
    acc == 0
}
```

## Advanced Examples

### Testing with State

For tests that need shared state or complex setup:

```rust
use timing_oracle::TimingOracle;

// Example: Testing database query timing
fn test_db_query_timing() {
    let result = TimingOracle::new()
        .samples(10_000)
        .test_with_state(
            || {
                // Setup: Initialize database connection
                let db = Database::connect("test.db");
                db.create_table();
                db
            },
            |db| {
                // Fixed input: Known user ID (might trigger caching)
                db.prepare_query("SELECT * FROM users WHERE id = ?", 1)
            },
            |db, rng| {
                // Random input: Random user ID
                let user_id = rng.gen_range(1..1000);
                db.prepare_query("SELECT * FROM users WHERE id = ?", user_id)
            },
            |db, query| {
                // Execute the query
                db.execute(query);
            },
        );

    assert!(result.ci_gate.passed,
            "Database query should not leak user ID through timing");
}
```

### Quick Smoke Tests

For faster iteration during development:

```rust
use timing_oracle::TimingOracle;

// Fast test with reduced samples (completes in ~0.5s)
let result = TimingOracle::quick()  // 5k samples, reduced bootstrap
    .test(|| my_function(&FIXED), || my_function(&random_input()));

println!("Quick smoke test: {:.0}% probability",
         result.leak_probability * 100.0);
```

### Reusing Calibrated Timer

When running many tests (e.g., in calibration studies), reuse a single timer to avoid repeated calibration overhead (~50ms per test):

```rust
use timing_oracle::{Timer, TimingOracle};

let timer = Timer::new();  // Calibrate once

for _ in 0..100 {
    let result = TimingOracle::quick()
        .with_timer(timer.clone())  // Reuse calibration
        .test(|| operation1(), || operation2());

    // Process results...
}
```

### Interpreting Results

```rust
let result = TimingOracle::new().test(fixed_fn, random_fn);

// 1. Check CI gate for pass/fail decision
if !result.ci_gate.passed {
    println!("❌ CI gate failed - timing leak detected");
}

// 2. Examine leak probability
match result.leak_probability {
    p if p > 0.99 => println!("🔴 Virtually certain leak (>99%)"),
    p if p > 0.9  => println!("🟠 Very likely leak (>90%)"),
    p if p > 0.5  => println!("🟡 Probable leak (>50%)"),
    _             => println!("🟢 No significant leak detected"),
}

// 3. If leak detected, examine effect decomposition
if let Some(effect) = result.effect {
    println!("Effect breakdown:");
    println!("  Shift: {:.1} ns ({:?})", effect.shift_ns, effect.pattern);
    println!("  Tail:  {:.1} ns", effect.tail_ns);
    println!("  Total magnitude: {:.1}-{:.1} ns (95% CI)",
             effect.credible_interval_ns.0,
             effect.credible_interval_ns.1);
}

// 4. Check exploitability
match result.exploitability {
    Exploitability::Negligible =>
        println!("Not practically exploitable"),
    Exploitability::PossibleLAN =>
        println!("⚠️  Might be exploitable on LAN (~100k queries)"),
    Exploitability::LikelyLAN =>
        println!("⚠️  Likely exploitable on LAN (~10k queries)"),
    Exploitability::PossibleRemote =>
        println!("⚠️  Possibly exploitable remotely"),
}

// 5. Assess measurement quality
println!("Measurement quality: {:?}", result.quality);
println!("Min detectable effect: {:.1} ns (shift), {:.1} ns (tail)",
         result.min_detectable_effect.shift_ns,
         result.min_detectable_effect.tail_ns);
```

### Testing Async/Await Code

For async functions, use `Runtime::block_on()` to bridge async → sync for measurement:

```rust
use timing_oracle::TimingOracle;
use tokio::runtime::Runtime;
use tokio::time::{sleep, Duration};

// Create a runtime once per test
let rt = tokio::runtime::Builder::new_current_thread()
    .enable_time()
    .build()
    .unwrap();

let result = TimingOracle::new()
    .samples(10_000)
    .test(
        || {
            rt.block_on(async {
                // Fixed async operation
                sleep(Duration::from_micros(10)).await;
                std::hint::black_box(42)
            })
        },
        || {
            rt.block_on(async {
                // Random async operation
                sleep(Duration::from_micros(10)).await;
                std::hint::black_box(43)
            })
        },
    );

// Verify no timing leak from async executor overhead
assert!(result.ci_gate.passed, "Async executor overhead should be symmetric");
```

**Key considerations for async testing:**
- Use **single-threaded** runtime for lower noise (`new_current_thread()`)
- Create the runtime **once** outside the measured closures
- Use `block_on()` to execute async code synchronously
- Both closures should perform **identical async operations** for baseline tests
- See `tests/async_timing.rs` for comprehensive async examples

## Troubleshooting

### High Noise / "TooNoisy" Quality

**Symptoms:**
- `quality: TooNoisy` or `Poor`
- High `min_detectable_effect` (>100 ns)
- Unreliable results

**Solutions:**
1. **Increase sample count:**
   ```rust
   TimingOracle::new().samples(500_000)  // vs default 100k
   ```

2. **Check CPU governor (Linux):**
   ```bash
   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
   # Should be "performance", not "powersave"
   sudo cpupower frequency-set -g performance
   ```

3. **Disable turbo boost:**
   ```bash
   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
   ```

4. **Run on isolated CPU cores:**
   ```bash
   cargo test --test timing_tests -- --test-threads=1
   ```

5. **Check for background processes:**
   - Close unnecessary applications
   - Verify low system load with `top` or `htop`

### False Positives on Noise

**Symptoms:**
- Constant-time code flagged as leaky
- `leak_probability` high on XOR operations

**Solutions:**
1. **Tighten CI alpha:**
   ```rust
   TimingOracle::new().ci_alpha(0.001)  // vs default 0.01
   ```

2. **Run sanity check manually:**
   ```rust
   // Split fixed samples in half - should not detect leak
   let result = TimingOracle::new()
       .test(|| my_fixed_op(), || my_fixed_op());

   assert!(result.ci_gate.passed, "Sanity check failed!");
   ```

3. **Check preflight warnings** - they'll indicate environmental issues

### Unable to Detect Known Leaks

**Symptoms:**
- Early-exit comparison shows `leak_probability < 0.5`
- CI gate passes on leaky code

**Solutions:**
1. **Increase sample count:**
   ```rust
   TimingOracle::new().samples(500_000)
   ```

2. **Ensure sufficient timing difference:**
   - Small leaks (<10ns) may require very large sample sizes
   - Verify leak is visible in raw timing data:
     ```rust
     // Manual inspection
     let mut fixed_times = vec![];
     let mut random_times = vec![];
     for _ in 0..1000 {
         let timer = Timer::new();
         fixed_times.push(timer.measure_cycles(|| fixed_op()));
         random_times.push(timer.measure_cycles(|| random_op()));
     }
     println!("Fixed median: {}", median(&fixed_times));
     println!("Random median: {}", median(&random_times));
     ```

3. **Disable compiler optimizations in test:**
   ```rust
   use std::hint::black_box;

   let result = TimingOracle::new().test(
       || black_box(my_function(black_box(&fixed_input))),
       || black_box(my_function(black_box(&random_input()))),
   );
   ```

### Virtual Machine Issues

**Symptoms:**
- Extremely high noise
- Unreliable timing
- Preflight warnings about VM

**Solutions:**
- Run tests on bare metal when possible
- If VM required, ensure:
  - Dedicated CPU cores (no overcommit)
  - Nested virtualization disabled
  - High-resolution timer support enabled

### Handling Test Flakiness

If tests occasionally fail/pass inconsistently:

```rust
use timing_oracle::{Mode, FailCriterion};

// Use hierarchical thresholds
TimingOracle::ci_test()
    .mode(Mode::Full)
    .fail_on(FailCriterion::Either {
        probability: 0.99,  // Very high bar
    })
    .run(fixed_fn, random_fn)
    .unwrap_or_report();

// Or require both gate AND probability
TimingOracle::ci_test()
    .fail_on(FailCriterion::Both {
        probability: 0.9,
    })
    .run(fixed_fn, random_fn)
    .unwrap_or_report();
```

## Performance Tips

### Typical Runtime

For default configuration (100k samples):
- **Measurement time:** Depends on operation cost
  - Fast operation (10ns): ~2 seconds
  - Medium operation (100ns): ~20 seconds
  - Slow operation (1μs): ~200 seconds
- **Analysis overhead:** <1 second (bootstrap + Bayesian inference)

### Optimization Strategies

1. **Start with quick mode during development:**
   ```rust
   TimingOracle::quick()  // 5k samples, ~10x faster
   ```

2. **Use parallel feature for bootstrap:**
   ```toml
   [dependencies]
   timing-oracle = { version = "0.1", features = ["parallel"] }
   ```
   Speedup: ~2-3× on 8-core machines

3. **Reduce bootstrap iterations for faster CI:**
   ```rust
   TimingOracle::new()
       .ci_bootstrap_iterations(2_000)    // vs 10k default
       .cov_bootstrap_iterations(500)     // vs 2k default
   ```

4. **Adjust calibration fraction:**
   ```rust
   TimingOracle::new()
       .calibration_fraction(0.2)  // vs 0.3 default
       // More data for inference, less for calibration
   ```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `samples` | 100,000 | Samples per class |
| `warmup` | 1,000 | Warmup iterations (not measured) |
| `ci_alpha` | 0.01 | CI gate false positive rate |
| `effect_prior_ns` | 10.0 | Prior scale for effects (σ_μ), not a pass/fail threshold |
| `effect_threshold_ns` | _unset_ | Optional hard threshold for reporting/panic |
| `outlier_percentile` | 0.999 | Percentile for outlier filtering (1.0 = disabled) |
| `prior_no_leak` | 0.75 | Prior probability of no leak |
| `calibration_fraction` | 0.3 | Fraction of samples used for calibration/preflight |
| `max_duration_ms` | _unset_ | Optional guardrail to abort long runs |
| `measurement_seed` | _unset_ | Deterministic seed for measurement randomness |

## Statistical Details

### Covariance Estimation

Quantile differences are correlated (neighboring quantiles tend to move together). We estimate the 9×9 covariance matrix Σ₀ via block bootstrap, then use it in both the CI gate thresholds and Bayesian inference.

### Sample Splitting

To avoid "double-dipping" (using data to both set priors and compute posteriors), we split samples:
- **Calibration set (30%)**: Estimate Σ₀, compute minimum detectable effect, set prior hyperparameters
- **Inference set (70%)**: Compute Δ and Bayes factor with fixed parameters

### Bayes Factor Computation

Under both hypotheses, Δ follows a multivariate normal:
- H₀: Δ ~ N(0, Σ₀)
- H₁: Δ ~ N(0, Σ₀ + X·Λ₀·Xᵀ)

The log Bayes factor is computed in closed form via Cholesky decomposition.

## Limitations

- **Not a formal proof**: Statistical evidence, not cryptographic verification
- **Noise sensitivity**: High-noise environments may produce unreliable results
- **JIT and optimization**: Ensure test conditions match production
- **Platform-dependent**: Timing characteristics vary across CPUs
- **Exploitability is heuristic**: Actual exploitability depends on network conditions and attacker capabilities

For mission-critical code, combine with formal verification tools like [ct-verif](https://github.com/imdea-software/verifying-constant-time).

## References

1. Reparaz, O., Balasch, J., & Verbauwhede, I. (2016). "Dude, is my code constant time?" DATE. — Original dudect methodology
2. Dunsche, M., et al. (2024). "With Great Power Come Great Side Channels: Statistical Timing Side-Channel Analyses with Bounded Type-1 Errors." USENIX Security. — RTLF methodology for bounded FPR
3. Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009). "Opportunities and limits of remote timing attacks." ACM TISSEC. — Exploitability thresholds

## License

MIT OR Apache-2.0
