# timing-oracle API Reference

Complete API documentation for timing-oracle. For conceptual overview, see [guide.md](guide.md). For troubleshooting, see [troubleshooting.md](troubleshooting.md).

## Table of Contents

- [Quick Reference](#quick-reference)
- [TimingOracle Builder](#timingoracle-builder)
  - [Presets](#presets)
  - [Configuration Methods](#configuration-methods)
  - [Test Methods](#test-methods)
- [CI Test Builder](#ci-test-builder)
- [Outcome and TestResult](#outcome-and-testresult)
  - [Outcome Enum](#outcome-enum)
  - [TestResult Struct](#testresult-struct)
  - [Effect Struct](#effect-struct)
  - [CiGate Struct](#cigate-struct)
- [Enums](#enums)
  - [Exploitability](#exploitability)
  - [MeasurementQuality](#measurementquality)
  - [EffectPattern](#effectpattern)
  - [Mode](#mode)
  - [FailCriterion](#failcriterion)
  - [UnreliablePolicy](#unreliablepolicy)
- [Helper Types](#helper-types)
  - [InputPair](#inputpair)
  - [Timer](#timer)
- [Macros](#macros)
- [Configuration Table](#configuration-table)

---

## Quick Reference

```rust
// Simple API
let result = timing_oracle::test(fixed_closure, random_closure);

// Builder API
let result = TimingOracle::new()
    .samples(100_000)
    .test(fixed_closure, random_closure);

// CI-focused API
TimingOracle::ci_test()
    .fail_on(FailCriterion::CiGate)
    .run(fixed_closure, random_closure)
    .unwrap_or_report();
```

---

## TimingOracle Builder

### Presets

```rust
use timing_oracle::TimingOracle;

// Default - Most accurate (~5-10 seconds per test)
// 100k samples, 100 CI bootstrap, 50 covariance bootstrap
TimingOracle::new()

// Balanced - Recommended for production (~1-2 seconds per test)
// 20k samples, 100 CI bootstrap, 50 covariance bootstrap
TimingOracle::balanced()

// Quick - Fast iteration during development (~0.2-0.5 seconds per test)
// 5k samples, 50 CI bootstrap, 50 covariance bootstrap
TimingOracle::quick()

// Calibration - For running many trials (100+) (~0.1-0.2 seconds per test)
// 2k samples, 30 CI bootstrap, 20 covariance bootstrap
TimingOracle::calibration()
```

### Configuration Methods

```rust
let oracle = TimingOracle::new()
    // Sample configuration
    .samples(100_000)                    // Samples per class
    .warmup(1_000)                       // Warmup iterations (not measured)

    // Statistical parameters
    .alpha(0.01)                      // CI gate false positive rate
    .min_effect_ns(10.0)               // Prior scale for effects (σ_μ)
    .effect_threshold_ns(100.0)          // Optional hard threshold
    .prior_no_leak(0.75)                 // Prior probability of no leak
    .outlier_percentile(0.999)           // Percentile for outlier filtering

    // Bootstrap configuration
    .ci_bootstrap_iterations(10_000)     // Bootstrap iterations for CI gate
    .cov_bootstrap_iterations(2_000)     // Bootstrap iterations for covariance

    // Resource limits
    .calibration_fraction(0.3)           // Fraction for calibration vs inference
    .max_duration_ms(60_000)             // Optional timeout

    // Reproducibility
    .measurement_seed(42);               // Deterministic seed
```

### Test Methods

```rust
// Basic test - returns TestResult directly
let result: TestResult = oracle.test(
    || fixed_operation(),
    || random_operation(),
);

// Test with state - for complex setup
let result = oracle.test_with_state(
    || setup_state(),           // State initializer
    |state| prepare_fixed(state),  // Fixed input generator
    |state, rng| prepare_random(state, rng),  // Random input generator
    |state, input| execute(state, input),     // Operation to time
);
```

---

## CI Test Builder

For CI integration with environment variable support:

```rust
use timing_oracle::{Mode, FailCriterion, TimingOracle};

TimingOracle::ci_test()
    .from_env()                          // Load TO_* environment variables
    .mode(Mode::Smoke)                   // Smoke, Quick, or Full
    .fail_on(FailCriterion::CiGate)      // When to fail
    .async_workload(true)                // Adjust for async noise
    .run(fixed_fn, random_fn)
    .unwrap_or_report();                 // Print report on failure
```

**Environment Variables:**
- `TO_MODE` - `smoke`, `quick`, or `full`
- `TO_SAMPLES` - Override sample count
- `TO_ALPHA` - Override CI alpha
- `TO_REPORT` - `always`, `on_fail`, or `never`
- `TO_SEED` - Deterministic seed
- `TO_ASYNC_WORKLOAD` - `1` to enable async mode

---

## Outcome and TestResult

### Outcome Enum

Top-level result type for handling unmeasurable operations:

```rust
pub enum Outcome {
    /// Analysis completed successfully
    Completed(TestResult),

    /// Operation too fast to measure reliably on this platform
    Unmeasurable {
        /// Estimated operation duration in nanoseconds
        operation_ns: f64,
        /// Minimum measurable duration on this platform
        threshold_ns: f64,
        /// Platform description (e.g., "Apple Silicon (cntvct_el0)")
        platform: String,
        /// Suggested actions to resolve
        recommendation: String,
    },
}

impl Outcome {
    /// Check if the measurement is reliable
    pub fn is_reliable(&self) -> bool;

    /// Handle unreliable measurements with a policy
    pub fn handle_unreliable(&self, name: &str, policy: UnreliablePolicy) -> Option<&TestResult>;

    /// Get the completed result, panicking if unmeasurable
    pub fn unwrap_completed(self) -> TestResult;

    /// Get the completed result as an Option
    pub fn completed(&self) -> Option<&TestResult>;
}
```

### TestResult Struct

```rust
pub struct TestResult {
    /// Bayesian posterior probability of timing leak (0.0 to 1.0)
    pub leak_probability: f64,

    /// Effect size decomposition (present if leak_probability > 0.5)
    pub effect: Option<Effect>,

    /// Exploitability assessment based on effect magnitude
    pub exploitability: Exploitability,

    /// Minimum detectable effect given measurement noise
    pub min_detectable_effect: MinDetectableEffect,

    /// CI gate result for pass/fail decisions
    pub ci_gate: CiGate,

    /// Measurement quality assessment
    pub quality: MeasurementQuality,

    /// Fraction of samples identified as outliers
    pub outlier_fraction: f64,

    /// Measurement metadata (samples, timer, runtime)
    pub metadata: Metadata,
}

impl TestResult {
    /// Check if this measurement can detect an effect of the given size
    pub fn can_detect(&self, effect_ns: f64) -> bool;
}
```

### Effect Struct

Decomposition of timing difference into interpretable components:

```rust
pub struct Effect {
    /// Uniform shift in nanoseconds (positive = fixed is slower)
    /// Indicates a consistent timing difference across all quantiles
    pub shift_ns: f64,

    /// Tail effect in nanoseconds (positive = fixed has heavier tail)
    /// Indicates timing differences concentrated in upper quantiles (e.g., cache misses)
    pub tail_ns: f64,

    /// 95% credible interval for total effect magnitude
    pub credible_interval_ns: (f64, f64),

    /// Dominant effect pattern
    pub pattern: EffectPattern,
}
```

### CiGate Struct

CI gate result for reliable pass/fail decisions:

```rust
pub struct CiGate {
    /// Significance level used
    pub alpha: f64,

    /// Whether the gate passed (no leak detected at alpha level)
    pub passed: bool,

    /// Bootstrap thresholds for each of 9 quantiles
    pub thresholds: [f64; 9],

    /// Observed quantile differences
    pub observed: [f64; 9],
}
```

---

## Enums

### Exploitability

Based on [Crosby et al. (2009)](https://dl.acm.org/doi/10.1145/1455770.1455794):

```rust
pub enum Exploitability {
    /// < 100 ns - Requires impractical measurement count
    Negligible,

    /// 100-500 ns - Exploitable on LAN with ~100k queries
    PossibleLAN,

    /// 500 ns - 20 μs - Likely exploitable on LAN
    LikelyLAN,

    /// > 20 μs - Possibly exploitable over internet
    PossibleRemote,
}
```

| Effect Size | Assessment | Implications |
|------------|------------|--------------|
| < 100 ns | `Negligible` | Requires impractical measurement count |
| 100-500 ns | `PossibleLAN` | Exploitable on LAN with ~100k queries |
| 500 ns - 20 μs | `LikelyLAN` | Likely exploitable on LAN |
| > 20 μs | `PossibleRemote` | Possibly exploitable over internet |

### MeasurementQuality

```rust
pub enum MeasurementQuality {
    /// Excellent signal-to-noise ratio
    Excellent,

    /// Good quality, reliable results
    Good,

    /// Noisy but usable
    Poor,

    /// Very noisy, results may be unreliable
    TooNoisy,
}
```

### EffectPattern

```rust
pub enum EffectPattern {
    /// Uniform shift across all quantiles (e.g., different code path)
    UniformShift,

    /// Effect concentrated in upper quantiles (e.g., cache misses)
    TailEffect,

    /// Both shift and tail components present
    Mixed,
}
```

### Mode

```rust
pub enum Mode {
    /// Minimal samples for quick validation (~1k samples)
    Smoke,

    /// Reduced samples for development (~5k samples)
    Quick,

    /// Full analysis (~100k samples)
    Full,
}
```

### FailCriterion

```rust
pub enum FailCriterion {
    /// Fail if CI gate fails
    CiGate,

    /// Fail if leak probability exceeds threshold
    Probability(f64),

    /// Fail if either criterion fails
    Either { probability: f64 },

    /// Fail only if both criteria fail
    Both { probability: f64 },
}
```

### UnreliablePolicy

```rust
pub enum UnreliablePolicy {
    /// Skip unreliable tests (return None)
    FailOpen,

    /// Panic on unreliable tests
    FailClosed,
}

impl UnreliablePolicy {
    /// Load from TIMING_ORACLE_UNRELIABLE_POLICY env var
    pub fn from_env_or(default: Self) -> Self;
}
```

---

## Helper Types

### InputPair

Pre-generate inputs for both classes to ensure identical code paths:

```rust
use timing_oracle::helpers::InputPair;

// From a fixed value and random generator
let inputs = InputPair::new(
    [0u8; 32],                    // Fixed value
    || rand::random::<[u8; 32]>() // Random generator
);

// With explicit sample count
let inputs = InputPair::with_samples(100_000, [0u8; 32], rand_generator);

// From two generator functions
let inputs = InputPair::from_fn(
    || fixed_value(),
    || random_value(),
);

// Access inputs
let fixed: &[u8; 32] = inputs.fixed();
let random: &[u8; 32] = inputs.random();
```

**Convenience functions:**

```rust
use timing_oracle::helpers;

// 32-byte arrays
let inputs = helpers::byte_arrays_32();

// Byte vectors of specific length
let inputs = helpers::byte_vecs(1024);
```

### Timer

```rust
use timing_oracle::Timer;

// Create and calibrate a timer
let timer = Timer::new();

// Measure cycles for an operation
let cycles = timer.measure_cycles(|| my_operation());
```

---

## Macros

### skip_if_unreliable!

Skip assertions if measurement is unreliable (fail-open policy):

```rust
use timing_oracle::{Outcome, skip_if_unreliable};

#[test]
fn test_cache_timing() {
    let result = TimingOracle::new().test(fixed, random);
    let outcome = Outcome::Completed(result);

    // Prints "[SKIPPED] test_cache_timing: ..." and returns early if unreliable
    let result = skip_if_unreliable!(outcome, "test_cache_timing");

    assert!(result.leak_probability > 0.5);
}
```

### require_reliable!

Panic if measurement is unreliable (fail-closed policy):

```rust
use timing_oracle::{Outcome, require_reliable};

#[test]
fn test_critical_crypto() {
    let result = TimingOracle::new().test(fixed, random);
    let outcome = Outcome::Completed(result);

    // Panics with "[UNRELIABLE] test_critical_crypto: ..." if unreliable
    let result = require_reliable!(outcome, "test_critical_crypto");

    assert!(result.leak_probability < 0.1);
}
```

---

## Configuration Table

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
| `ci_bootstrap_iterations` | 10,000 | Bootstrap iterations for CI gate thresholds |
| `cov_bootstrap_iterations` | 2,000 | Bootstrap iterations for covariance estimation |
| `max_duration_ms` | _unset_ | Optional guardrail to abort long runs |
| `measurement_seed` | _unset_ | Deterministic seed for measurement randomness |
