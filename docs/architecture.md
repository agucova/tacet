# Architecture

This document describes the internal architecture of `timing-oracle`, detailing how the statistical methodology translates into code.

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEASUREMENT PHASE                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Pre-flight checks (sanity, generator cost, autocorrelation) │
│  2. Warmup iterations (not measured)                            │
│  3. Interleaved randomized measurement                          │
│  4. Outlier filtering (pooled symmetric threshold)              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTILE COMPUTATION                          │
├─────────────────────────────────────────────────────────────────┤
│  Compute deciles (10%, 20%, ..., 90%) for each class            │
│  Δ = q̂(Fixed) - q̂(Random) ∈ ℝ⁹                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│   LAYER 1: CI GATE       │      │   LAYER 2: BAYESIAN          │
├──────────────────────────┤      ├──────────────────────────────┤
│ Max-bootstrap pooled     │      │ Sample splitting:            │
│ Single unified threshold │      │  - 30% calibration set       │
│ Max-statistic over Δ     │      │  - 70% inference set         │
│ No multiple testing adj  │      │                              │
│                          │      │ Closed-form conjugate        │
│ Output: passed/failed    │      │ Bayes factor                 │
│         at α level       │      │                              │
│                          │      │ Bootstrap covariance Σ₀      │
│                          │      │ (single quantile vectors)    │
│                          │      │                              │
│                          │      │ Output: P(leak), effect size │
└──────────────────────────┘      └──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         TEST RESULT                              │
│  • leak_probability: Posterior P(H₁|data)                       │
│  • effect: Decomposed into shift_ns and tail_ns                 │
│  • exploitability: Heuristic classification                      │
│  • min_detectable_effect: Sensitivity of current setup          │
│  • ci_gate: Pass/fail decision for CI                           │
│  • quality: Measurement noise assessment                        │
└─────────────────────────────────────────────────────────────────┘
```

## Module Organization

### Core Entry Points (`src/`)

```
oracle.rs           ← TimingOracle builder & main pipeline
├─ config.rs        ← Configuration parameters
├─ result.rs        ← Output types (TestResult, Effect, etc.)
├─ types.rs         ← Type aliases (Matrix9, Vector9, Class)
└─ constants.rs     ← Mathematical constants (DECILES, B_TAIL)
```

### Measurement Layer (`src/measurement/`)

**Purpose:** Collect high-precision timing samples

```
timer.rs            ← Platform-specific cycle counters
│                     - x86_64: rdtsc with lfence
│                     - aarch64: cntvct_el0 with isb
│                     - fallback: std::time::Instant
│
collector.rs        ← Interleaved randomized sampling
│                     - Randomizes Fixed/Random order
│                     - Prevents drift from affecting one class
│
outlier.rs          ← Pooled symmetric threshold filtering
                      - Same threshold for both classes
                      - Preserves distributional symmetry
```

### Statistical Core (`src/statistics/`)

**Purpose:** Compute quantiles, bootstrap, covariance

```
quantile.rs         ← O(n) decile computation via select_nth_unstable
bootstrap.rs        ← Block bootstrap with √n block size
covariance.rs       ← Bootstrap covariance estimation
autocorrelation.rs  ← ACF for detecting periodic interference
```

### Analysis Layer (`src/analysis/`)

**Purpose:** Two-layer statistical inference

```
ci_gate.rs          ← Layer 1: Max-bootstrap frequentist gate
│                     - Pools samples from both classes under H₀
│                     - Single (1-α) quantile threshold on M* distribution
│                     - Max-statistic: M = max_p |Δ_p| (more powerful than Bonferroni)
│
bayes.rs            ← Layer 2: Bayesian interpretation
│                     - Closed-form Bayes factor
│                     - MVN log-pdf via Cholesky
│                     - Posterior probability P(H₁|data)
│
effect.rs           ← Effect decomposition
│                     - Bayesian linear regression
│                     - X = [ones | b_tail] design matrix
│                     - Separates shift from tail effects
│
mde.rs              ← Minimum Detectable Effect estimation
                      - Sample from null distribution
                      - 95th percentile of |β̂|
```

### Preflight Checks (`src/preflight/`)

**Purpose:** Validate measurement setup before analysis

```
sanity.rs           ← Fixed-vs-Fixed comparison
│                     - Splits fixed samples in half
│                     - Detects broken measurement harness
│
generator.rs        ← Generator cost comparison
│                     - Ensures fixed/random generators have similar overhead
│
autocorr.rs         ← Autocorrelation check
│                     - Detects periodic interference (lag-1, lag-2 > 0.3)
│
system.rs           ← Platform checks
                      - CPU governor (Linux)
                      - Turbo boost detection
                      - VM detection
```

### Output & Presentation (`src/output/`)

```
terminal.rs         ← Colored human-readable output
json.rs             ← Machine-readable JSON export
```

### CI Integration (`src/ci.rs`)

**Purpose:** Ergonomic CI testing with failure modes

```
CiTestBuilder       ← Builder for CI-focused tests
│ ├─ Mode::Smoke    ← Fast smoke test (5k samples)
│ ├─ Mode::Full     ← Standard test (100k samples)
│ └─ Mode::Exhaustive ← Thorough test (500k samples)
│
└─ FailCriterion    ← Configurable failure conditions
    ├─ Gate         ← Fail if CI gate fails
    ├─ Probability  ← Fail if leak_probability > threshold
    ├─ Either       ← Fail if gate fails OR probability exceeds
    └─ Both         ← Fail only if BOTH conditions met
```

## Data Flow

### 1. Measurement Phase

```rust
// User provides closures
fixed: || my_function(&fixed_input)
random: || my_function(&random_input())

           ↓ (Collector::collect)

// Interleaved sequence with randomization
[F₁, R₁, R₂, F₂, R₃, F₃, ...] with Class tags

           ↓ (split by Class)

fixed_cycles: Vec<u64>
random_cycles: Vec<u64>

           ↓ (filter_outliers)

// After pooled symmetric filtering
filtered_fixed: Vec<u64>
filtered_random: Vec<u64>
```

### 2. Sample Splitting for Bayesian Inference

```rust
// 30% calibration set (estimate Σ₀, MDE, set priors)
calib_fixed, calib_random

// 70% inference set (compute Δ, Bayes factor)
infer_fixed, infer_random
```

This prevents "double-dipping" where the same data informs both prior and likelihood.

### 3. Quantile Computation

```rust
// Convert to nanoseconds
fixed_ns = timer.cycles_to_ns(filtered_fixed)
random_ns = timer.cycles_to_ns(filtered_random)

// Compute deciles
q_fixed = compute_deciles(&fixed_ns)   // Vector9
q_random = compute_deciles(&random_ns) // Vector9

// Difference vector
delta = q_fixed - q_random  // Vector9
```

### 4. Parallel Analysis Layers

**CI Gate (on full dataset):**
```rust
// Bootstrap within each class
for _ in 0..B_bootstrap {
    F₁*, F₂* = block_resample(fixed_ns)
    R₁*, R₂* = block_resample(random_ns)

    delta_F* = quantiles(F₁*) - quantiles(F₂*)
    delta_R* = quantiles(R₁*) - quantiles(R₂*)
}

// Per-quantile thresholds
threshold[i] = max(
    quantile(|delta_F*[i]|, 1 - α/9),
    quantile(|delta_R*[i]|, 1 - α/9)
)

// Decision
passed = all(|delta[i]| ≤ threshold[i])
```

**Bayesian Layer (on inference set):**
```rust
// Estimate covariance from calibration set
Σ_F = bootstrap_cov(calib_fixed)
Σ_R = bootstrap_cov(calib_random)
Σ₀ = Σ_F + Σ_R

// Set prior scale proportional to threshold for calibrated priors
// Using 2*theta ensures P(|β| > θ | prior) ≈ 62%
prior_σ = 2 * theta

// Compute Δ from inference set
delta_infer = quantiles(infer_fixed) - quantiles(infer_random)

// Bayes factor
Σ₁ = Σ₀ + X Λ₀ Xᵀ
log_BF = log N(delta; 0, Σ₁) - log N(delta; 0, Σ₀)

// Posterior
P(leak|data) = 1 / (1 + exp(-log_BF) / prior_odds)
```

### 5. Effect Decomposition (if leak detected)

```rust
// Bayesian linear regression
delta = X β + ε
where X = [ones | b_tail], β = (shift, tail)

// Posterior
Λ_post⁻¹ = Xᵀ Σ₀⁻¹ X + Λ₀⁻¹
β_post = Λ_post Xᵀ Σ₀⁻¹ delta

// Sample for CI
samples ~ N(β_post, Λ_post)
magnitude = √(shift² + tail²)
CI = percentiles(magnitude, [0.025, 0.975])
```

## Why Two Layers?

The architecture separates **decision-making** from **interpretation**:

### Layer 1 (CI Gate): Binary Decision
- **Purpose:** "Should this block my CI build?"
- **Method:** RTLF-style bootstrap with bounded FPR
- **Output:** Binary pass/fail
- **Null:** H₀ = no difference between classes (pooled samples)
- **Method:** Max-statistic bootstrap with single unified threshold

### Layer 2 (Bayesian): Quantitative Assessment
- **Purpose:** "What's the probability and magnitude of the leak?"
- **Method:** Closed-form Bayesian inference with proper prior
- **Output:** Continuous probability + effect size
- **Null:** H₀ = delta ~ N(0, Σ₀)
- **Calibrated:** Sample splitting prevents overfitting

These use **different null constructions** because they serve different goals:
- CI gate: Conservative rejection requires comparing randomized class splits
- Bayesian: Interpretable probabilities require proper generative model

## Key Design Decisions

### 1. Sample Splitting (30/70)
- **Why:** Prevents using data to both inform prior and compute likelihood
- **Calibration (30%):** Estimate Σ₀, compute MDE, set hyperparameters
- **Inference (70%):** Compute Δ and Bayes factor using fixed Σ₀
- **Trade-off:** Slightly lower power, but posterior is a calibrated probability

### 2. Block Bootstrap
- **Why:** Preserves autocorrelation in timing data
- **Block size:** √n (e.g., ~316 for n=100k)
- **Alternative:** IID resampling assumes independence (unrealistic for timing)

### 3. Pooled Symmetric Outlier Filtering
- **Why:** Using per-class thresholds would bias distributional shape
- **Method:** Compute single threshold on pooled data, apply to both classes
- **Preserves:** Symmetry of quantile differences under null

### 4. Cholesky for Numerical Stability
- **Why:** Direct matrix inversion is numerically unstable
- **Benefits:**
  - log|Σ| = 2 Σ log(Lᵢᵢ) (diagonal of L)
  - Σ⁻¹x via forward/backward substitution
  - Positive definiteness guaranteed
- **Fallback:** Add small jitter (1e-9 × max_diag) if decomposition fails

### 5. Design Matrix Orthogonalization
```rust
ones = [1, 1, 1, 1, 1, 1, 1, 1, 1]
b_tail = [-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5]
```
- **Why centered:** Makes shift (μ) and tail (τ) approximately orthogonal
- **Interpretation:**
  - μ > 0: Fixed class consistently slower
  - τ > 0: Fixed class has heavier upper tail (e.g., cache misses)

## Performance Characteristics

### Typical Runtime (100k samples)
- Measurement: 0.5-2s (depends on operation cost)
- CI Gate bootstrap (10k iter): 0.2-0.5s
- Covariance bootstrap (2k iter): 0.1-0.2s
- Bayesian inference: <0.01s (closed-form)
- **Total:** <1s overhead beyond measurement time

### Memory Usage
- Samples: 2 × 100k × 8 bytes = 1.6 MB
- Bootstrap: ~9 × 10k × 8 bytes = 0.7 MB
- Covariance: 9×9 × 8 bytes = 648 bytes
- **Peak:** ~3 MB for default config

### Parallelization
- Bootstrap iterations can run in parallel (via `rayon` feature)
- Speedup: ~2-3× on 8-core machine for bootstrap-heavy configs
- **Not enabled by default** (adds dependency)

## Testing Strategy

### Unit Tests
- Each module has `#[cfg(test)]` section
- Test numerical properties (e.g., CI symmetry, MVN log-pdf)
- Fast execution (<0.1s per test)

### Integration Tests

**Known Leaky (`tests/known_leaky.rs`):**
- Early-exit comparison
- Branch-based timing
- **Expectation:** leak_probability > 0.9, ci_gate.passed = false

**Known Safe (`tests/known_safe.rs`):**
- Constant-time XOR comparison
- Simple XOR operation
- **Expectation:** leak_probability < 0.5, ci_gate.passed = true

**Calibration (`tests/calibration.rs`):**
- CI gate FPR: Run 100 trials on noise, verify rejections ≤ 2α
- Bayesian: Run 100 trials on noise, verify <10% have probability > 0.9

### Benchmarks
- Microbenchmarks in `benches/` using Criterion
- Test individual components (quantile computation, bootstrap, etc.)
- Track performance regressions

## Extension Points

### Adding New Timer Platforms
```rust
// In src/measurement/timer.rs
#[cfg(target_arch = "your_arch")]
pub fn read_cycles() -> u64 {
    // Your platform-specific cycle counter
}
```

### Custom Preflight Checks
```rust
// In src/preflight/mod.rs
pub fn run_all_checks(...) {
    // Add your check here
    if let Some(warning) = your_custom_check() {
        result.add_custom_warning(warning);
    }
}
```

### Custom Output Formats
```rust
// In src/output/
pub fn format_custom(result: &TestResult) -> YourFormat {
    // Transform TestResult to your format
}
```

## References

- **RTLF Paper:** Dunsche et al. (2024), "With Great Power Come Great Side Channels"
- **dudect:** Original Reparaz et al. (2016) methodology
- **Block Bootstrap:** Künsch (1989), "The Jackknife and the Bootstrap"
- **Exploitability:** Crosby et al. (2009), "Opportunities and limits of remote timing attacks"
