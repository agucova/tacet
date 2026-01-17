# timing-oracle Rust API Reference

Complete API documentation for the Rust implementation of timing-oracle. For conceptual overview, see [guide.md](guide.md). For troubleshooting, see [troubleshooting.md](troubleshooting.md). For other language bindings, see [api-c.md](api-c.md) or [api-go.md](api-go.md).

## Table of Contents

- [Quick Reference](#quick-reference)
- [TimingOracle Builder](#timingoracle-builder)
  - [Attacker Model Presets](#attacker-model-presets)
  - [Configuration Methods](#configuration-methods)
  - [Test Methods](#test-methods)
- [Outcome Enum](#outcome-enum)
  - [Pass](#pass)
  - [Fail](#fail)
  - [Inconclusive](#inconclusive)
  - [Unmeasurable](#unmeasurable)
- [Supporting Types](#supporting-types)
  - [EffectEstimate](#effectestimate)
  - [Exploitability](#exploitability)
  - [MeasurementQuality](#measurementquality)
  - [EffectPattern](#effectpattern)
  - [InconclusiveReason](#inclusivereason)
  - [Diagnostics](#diagnostics)
- [Helper Types](#helper-types)
  - [InputPair](#inputpair)
  - [Timer](#timer)
- [Assertion Macros](#assertion-macros)
- [Configuration Table](#configuration-table)

---

## Quick Reference

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};

// Create inputs using closures
let inputs = InputPair::new(
    || [0u8; 32],          // baseline: returns constant value
    || rand::random(),     // sample: generates varied values
);

// Run test with attacker model
let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .test(inputs, |data| my_function(data));

// Handle all four possible outcomes
match outcome {
    Outcome::Pass { leak_probability, .. } => {
        println!("No leak: P(leak)={:.1}%", leak_probability * 100.0);
    }
    Outcome::Fail { leak_probability, exploitability, .. } => {
        panic!("Timing leak: P(leak)={:.1}%, {:?}", leak_probability * 100.0, exploitability);
    }
    Outcome::Inconclusive { reason, .. } => {
        println!("Inconclusive: {:?}", reason);
    }
    Outcome::Unmeasurable { recommendation, .. } => {
        println!("Skipped: {}", recommendation);
    }
}
```

---

## TimingOracle Builder

### Attacker Model Presets

Choose your threat model to define the minimum effect size worth detecting:

```rust
use timing_oracle::{TimingOracle, AttackerModel};

// Shared hardware: SGX, cross-VM, containers, hyperthreading (~0.6ns / ~2 cycles)
TimingOracle::for_attacker(AttackerModel::SharedHardware)

// Post-quantum sentinel: Catch KyberSlash-class leaks (~3.3ns / ~10 cycles)
TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)

// Adjacent network: LAN or HTTP/2 endpoints (100ns) - DEFAULT
TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)

// Remote network: Public APIs, general internet (50us)
TimingOracle::for_attacker(AttackerModel::RemoteNetwork)

// Research: Detect any difference (not for CI)
TimingOracle::for_attacker(AttackerModel::Research)

// Custom threshold in nanoseconds
TimingOracle::for_attacker(AttackerModel::Custom { threshold_ns: 500.0 })
```

| Preset | Threshold | Use case |
|--------|-----------|----------|
| `SharedHardware` | ~0.6 ns (~2 cycles) | SGX enclaves, cross-VM, containers, hyperthreading |
| `PostQuantumSentinel` | ~3.3 ns (~10 cycles) | Post-quantum crypto (ML-KEM, ML-DSA) |
| `AdjacentNetwork` | 100 ns | LAN, HTTP/2 endpoints (Timeless Timing Attacks) |
| `RemoteNetwork` | 50 us | Public APIs, general internet |
| `Research` | 0 (clamped to timer resolution) | Academic analysis, profiling |
| `Custom { threshold_ns }` | user-defined | Custom threshold |

**Sources:**
- Crosby et al. (2009): ~100ns LAN accuracy, 15-100us internet
- Van Goethem et al. (2020): "Timeless Timing Attacks" - 100ns over internet via HTTP/2
- Flush+Reload, Prime+Probe literature: cycle-level attacks on shared hardware

### Configuration Methods

All configuration is optional. The defaults work well for most use cases.

```rust
use timing_oracle::{TimingOracle, AttackerModel, TimerSpec};
use std::time::Duration;

let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    // Decision thresholds
    .pass_threshold(0.05)                    // Below this = Pass (default: 0.05)
    .fail_threshold(0.95)                    // Above this = Fail (default: 0.95)

    // Resource limits (for CI or constrained environments)
    .time_budget(Duration::from_secs(30))    // Max time for analysis (default: 60s)
    .max_samples(100_000)                    // Hard cap on samples (default: 1M)

    // Advanced tuning (rarely needed)
    .batch_size(1_000)                       // Samples per batch (default: 1,000)
    .calibration_samples(5_000)              // Calibration samples (default: 5,000)
    .prior_no_leak(0.75)                     // Prior P(no leak) (default: 0.75)
    .cov_bootstrap_iterations(2_000)         // Bootstrap iterations (default: 2,000)
    .warmup(1_000)                           // Warmup iterations (default: 1,000)
    .outlier_percentile(0.9999)              // Outlier threshold (default: 0.9999)

    // Timer configuration
    .timer_spec(TimerSpec::Auto)             // Auto-detect best timer (default)
    .standard_timer()                        // Force standard timer (no PMU)
    .prefer_pmu()                            // Prefer PMU with fallback

    // Reproducibility
    .seed(42);                               // Deterministic seed
```

### Test Methods

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};

let inputs = InputPair::new(
    || [0u8; 32],          // baseline closure
    || rand::random(),     // sample closure
);

// Main test method
let outcome: Outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .test(inputs, |data| {
        my_crypto_function(data);
    });
```

---

## Outcome Enum

The `Outcome` enum represents all possible results from a timing test:

```rust
pub enum Outcome {
    Pass { leak_probability, effect, samples_used, quality, diagnostics },
    Fail { leak_probability, effect, exploitability, samples_used, quality, diagnostics },
    Inconclusive { reason, leak_probability, effect, samples_used, quality, diagnostics },
    Unmeasurable { operation_ns, threshold_ns, platform, recommendation },
}
```

### Pass

No timing leak detected. The posterior probability is below the pass threshold (default 5%).

```rust
Outcome::Pass {
    leak_probability: f64,      // P(leak) < 0.05
    effect: EffectEstimate,     // Effect size (likely small)
    samples_used: usize,        // Total samples collected
    quality: MeasurementQuality,
    diagnostics: Diagnostics,
}
```

### Fail

Timing leak confirmed. The posterior probability exceeds the fail threshold (default 95%).

```rust
Outcome::Fail {
    leak_probability: f64,        // P(leak) > 0.95
    effect: EffectEstimate,       // Effect size decomposition
    exploitability: Exploitability, // Risk assessment
    samples_used: usize,
    quality: MeasurementQuality,
    diagnostics: Diagnostics,
}
```

### Inconclusive

Cannot reach a definitive conclusion within the given constraints.

```rust
Outcome::Inconclusive {
    reason: InconclusiveReason,   // Why inconclusive
    leak_probability: f64,        // Current posterior (between thresholds)
    effect: EffectEstimate,
    samples_used: usize,
    quality: MeasurementQuality,
    diagnostics: Diagnostics,
}
```

### Unmeasurable

Operation too fast to measure reliably on this platform.

```rust
Outcome::Unmeasurable {
    operation_ns: f64,      // Estimated operation time
    threshold_ns: f64,      // Minimum measurable
    platform: String,       // Platform description
    recommendation: String, // How to resolve
}
```

### Outcome Methods

```rust
impl Outcome {
    /// Check if test passed
    fn passed(&self) -> bool;

    /// Check if test failed
    fn failed(&self) -> bool;

    /// Check if result is conclusive (Pass or Fail)
    fn is_conclusive(&self) -> bool;

    /// Check if operation was measurable
    fn is_measurable(&self) -> bool;

    /// Check if measurement is reliable enough for assertions
    fn is_reliable(&self) -> bool;

    /// Get leak probability if available
    fn leak_probability(&self) -> Option<f64>;

    /// Get effect estimate if available
    fn effect(&self) -> Option<&EffectEstimate>;

    /// Get measurement quality if available
    fn quality(&self) -> Option<MeasurementQuality>;

    /// Get diagnostics if available
    fn diagnostics(&self) -> Option<&Diagnostics>;

    /// Handle unreliable results according to policy
    fn handle_unreliable(self, test_name: &str, policy: UnreliablePolicy) -> Option<Self>;
}
```

---

## Supporting Types

### EffectEstimate

Decomposition of timing difference into interpretable components:

```rust
pub struct EffectEstimate {
    /// Uniform shift in nanoseconds (positive = baseline is slower)
    pub shift_ns: f64,

    /// Tail effect in nanoseconds (positive = baseline has heavier tail)
    pub tail_ns: f64,

    /// 95% credible interval for total effect magnitude
    pub credible_interval_ns: (f64, f64),

    /// Dominant effect pattern
    pub pattern: EffectPattern,
}

impl EffectEstimate {
    /// Compute total effect magnitude (L2 norm)
    fn total_effect_ns(&self) -> f64;

    /// Check if effect is negligible
    fn is_negligible(&self, threshold_ns: f64) -> bool;
}
```

### Exploitability

Risk assessment based on effect magnitude (Timeless Timing Attacks, Van Goethem et al. 2020):

```rust
pub enum Exploitability {
    /// < 10 ns - Requires shared hardware (SGX, containers)
    SharedHardwareOnly,

    /// 10-100 ns - Exploitable via HTTP/2 request multiplexing
    Http2Multiplexing,

    /// 100 ns - 10 μs - Exploitable with standard remote timing
    StandardRemote,

    /// > 10 μs - Obvious leak, trivially exploitable
    ObviousLeak,
}
```

| Effect Size | Assessment | Attack Vector |
|------------|------------|---------------|
| < 10 ns | `SharedHardwareOnly` | ~1k queries on same core (SGX, containers) |
| 10-100 ns | `Http2Multiplexing` | ~100k concurrent HTTP/2 requests |
| 100 ns - 10 μs | `StandardRemote` | ~1k-10k queries with standard timing |
| > 10 μs | `ObviousLeak` | <100 queries, trivially observable |

### MeasurementQuality

Assessment based on minimum detectable effect (MDE):

```rust
pub enum MeasurementQuality {
    /// MDE < 5 ns - Excellent signal-to-noise
    Excellent,

    /// MDE 5-20 ns - Good quality
    Good,

    /// MDE 20-100 ns - Noisy but usable
    Poor,

    /// MDE > 100 ns - Results may be unreliable
    TooNoisy,
}
```

### EffectPattern

Classification of timing difference type:

```rust
pub enum EffectPattern {
    /// Uniform shift across all quantiles (e.g., different code path)
    UniformShift,

    /// Effect concentrated in upper quantiles (e.g., cache misses)
    TailEffect,

    /// Both shift and tail components present
    Mixed,

    /// Effect too small to classify
    Indeterminate,
}
```

### InconclusiveReason

Why a test couldn't reach a conclusion:

```rust
pub enum InconclusiveReason {
    /// Measurement noise too high
    DataTooNoisy { message: String, guidance: String },

    /// Posterior not converging
    NotLearning { message: String, guidance: String },

    /// Would exceed time budget to reach conclusion
    WouldTakeTooLong { estimated_time_secs: f64, samples_needed: usize, guidance: String },

    /// Time budget exhausted
    TimeBudgetExceeded { current_probability: f64, samples_collected: usize },

    /// Sample budget exhausted
    SampleBudgetExceeded { current_probability: f64, samples_collected: usize },
}
```

### Diagnostics

Detailed diagnostic information for debugging:

```rust
pub struct Diagnostics {
    pub dependence_length: usize,        // Block bootstrap length
    pub effective_sample_size: usize,    // ESS after accounting for autocorrelation
    pub stationarity_ratio: f64,         // Variance ratio (0.5-2.0 normal)
    pub stationarity_ok: bool,
    pub model_fit_chi2: f64,             // Chi-squared for residuals
    pub model_fit_ok: bool,
    pub outlier_rate_baseline: f64,
    pub outlier_rate_sample: f64,
    pub outlier_asymmetry_ok: bool,
    pub discrete_mode: bool,             // Low timer resolution mode
    pub timer_resolution_ns: f64,
    pub duplicate_fraction: f64,
    pub preflight_ok: bool,
    pub calibration_samples: usize,
    pub total_time_secs: f64,
    pub warnings: Vec<String>,
    pub quality_issues: Vec<QualityIssue>,
    pub preflight_warnings: Vec<PreflightWarningInfo>,
    pub seed: Option<u64>,
    pub attacker_model: Option<String>,
    pub threshold_ns: f64,
    pub timer_name: String,
    pub platform: String,
}
```

---

## Helper Types

### InputPair

Pre-generates inputs for both classes to ensure identical code paths:

```rust
use timing_oracle::helpers::InputPair;

// Create with closures (recommended)
let inputs = InputPair::new(
    || [0u8; 32],                    // Baseline closure
    || rand::random::<[u8; 32]>(),   // Sample closure
);

// Access values (calls closures)
let baseline_val = inputs.baseline();
let sample_val = inputs.sample();

// Check for common mistakes after measurement
if let Some(warning) = inputs.check_anomaly() {
    eprintln!("[timing-oracle] {}", warning);
}
```

**Convenience functions:**

```rust
use timing_oracle::helpers;

// 32-byte arrays (zeros vs random)
let inputs = helpers::byte_arrays_32();

// Byte vectors of specific length
let inputs = helpers::byte_vecs(1024);
```

**Variants:**

```rust
// Without anomaly detection (for intentionally deterministic inputs)
let inputs = InputPair::new_unchecked(
    || [0x00u8; 12],  // Fixed nonce A
    || [0xFFu8; 12],  // Fixed nonce B
);

// For non-Hash types
let inputs = InputPair::new_untracked(
    || Scalar::zero(),
    || Scalar::random(&mut rng),
);
```

### Timer

```rust
use timing_oracle::Timer;

// Create and calibrate a timer
let timer = Timer::new();

// Measure cycles for an operation
let cycles = timer.measure_cycles(|| my_operation());

// Timer info
let resolution_ns = timer.resolution_ns();
let cycles_per_ns = timer.cycles_per_ns();
let name = timer.name();
```

---

## Assertion Macros

### assert_constant_time!

Assert that code is constant-time (panics on Fail or Inconclusive):

```rust
use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair, assert_constant_time};

let inputs = InputPair::new(|| [0u8; 32], || rand::random());
let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .test(inputs, |data| my_crypto_function(data));

assert_constant_time!(outcome);  // Panics with diagnostics if not constant-time
```

### assert_no_timing_leak!

Lenient assertion (only panics on Fail, allows Inconclusive):

```rust
use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair, assert_no_timing_leak};

let inputs = InputPair::new(|| [0u8; 32], || rand::random());
let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .test(inputs, |data| my_function(data));

assert_no_timing_leak!(outcome);  // Only panics if leak detected
```

### assert_leak_detected!

Assert that a leak WAS detected (for testing known-leaky code):

```rust
use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair, assert_leak_detected};

let inputs = InputPair::new(|| [0u8; 32], || rand::random());
let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .test(inputs, |data| leaky_function(data));

assert_leak_detected!(outcome);  // Panics if no leak found
```

### skip_if_unreliable!

Skip test if measurement is unreliable (fail-open policy):

```rust
use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair, skip_if_unreliable};

#[test]
fn test_crypto() {
    let inputs = InputPair::new(|| [0u8; 32], || rand::random());
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .test(inputs, |data| encrypt(data));

    let outcome = skip_if_unreliable!(outcome, "test_crypto");
    // Only reaches here if reliable
    assert!(outcome.passed());
}
```

### require_reliable!

Panic if measurement is unreliable (fail-closed policy):

```rust
use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair, require_reliable};

#[test]
fn test_critical_crypto() {
    let inputs = InputPair::new(|| [0u8; 32], || rand::random());
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .test(inputs, |data| encrypt(data));

    let outcome = require_reliable!(outcome, "test_critical_crypto");
    assert!(outcome.passed());
}
```

---

## Configuration Table

| Parameter | Default | Description |
|-----------|---------|-------------|
| `time_budget` | 60 seconds | Maximum time for analysis |
| `max_samples` | 1,000,000 | Maximum samples per class |
| `batch_size` | 1,000 | Samples collected per batch |
| `calibration_samples` | 5,000 | Samples for calibration phase |
| `pass_threshold` | 0.05 | P(leak) below this = Pass |
| `fail_threshold` | 0.95 | P(leak) above this = Fail |
| `prior_no_leak` | 0.75 | Prior probability of no leak |
| `cov_bootstrap_iterations` | 2,000 | Bootstrap iterations for covariance |
| `warmup` | 1,000 | Warmup iterations before measurement |
| `outlier_percentile` | 0.9999 | Percentile for outlier winsorization |

### Environment Variables

Override settings via environment variables:

| Variable | Description |
|----------|-------------|
| `TO_TIME_BUDGET_SECS` | Time budget in seconds |
| `TO_MAX_SAMPLES` | Maximum samples per class |
| `TO_BATCH_SIZE` | Batch size |
| `TO_CALIBRATION_SAMPLES` | Calibration samples |
| `TO_PASS_THRESHOLD` | Pass threshold (e.g., "0.05") |
| `TO_FAIL_THRESHOLD` | Fail threshold (e.g., "0.95") |
| `TO_MIN_EFFECT_NS` | Minimum effect threshold in ns |
| `TO_SEED` | Deterministic seed |
| `TIMING_ORACLE_UNRELIABLE_POLICY` | "fail_open" or "fail_closed" |
| `TIMING_ORACLE_SKIP_PREFLIGHT` | Set to skip preflight checks |

```rust
// Load configuration from environment
let oracle = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .from_env();
```
