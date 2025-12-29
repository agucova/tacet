# timing-oracle

**Detect timing side channels in Rust code with statistically rigorous methods.**

`timing-oracle` measures whether your code's execution time depends on secret data. It compares timing between a "fixed" input (e.g., all zeros, valid padding) and random inputs, then uses Bayesian inference to compute the probability of a timing leak and estimate its magnitude.

Unlike simple t-tests, this crate provides:
- **Leak probability** (0-100%): A Bayesian posterior, not a p-value
- **Effect size in nanoseconds**: How big is the leak?
- **CI gate with bounded false positives**: Reliable pass/fail for continuous integration
- **Exploitability assessment**: Is this leak practically exploitable?

## Quick Start

```rust
use timing_oracle::test;

let secret = [0u8; 32];

let result = test(
    || compare(&secret, &[0u8; 32]),    // Fixed input (e.g., correct password)
    || compare(&secret, &rand_bytes()), // Random inputs
);

if result.leak_probability > 0.9 {
    panic!("Timing leak detected with {:.0}% confidence",
           result.leak_probability * 100.0);
}
```

## Installation

```toml
[dev-dependencies]
timing-oracle = "0.1"
```

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
    .min_effect_of_concern(10.0) // Ignore effects < 10ns
    .test(fixed_fn, random_fn);
```

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

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `samples` | 100,000 | Samples per class |
| `warmup` | 1,000 | Warmup iterations (not measured) |
| `ci_alpha` | 0.01 | CI gate false positive rate |
| `min_effect_of_concern_ns` | 10.0 | Effects below this are considered negligible |
| `outlier_percentile` | 0.999 | Percentile for outlier filtering (1.0 = disabled) |
| `prior_no_leak` | 0.75 | Prior probability of no leak |

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
