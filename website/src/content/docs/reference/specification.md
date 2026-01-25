---
title: Specification (v5.6)
description: Authoritative specification for timing-oracle's statistical methodology and requirements
sidebar:
  order: 1
---

This document is the authoritative specification for timing-oracle, a Bayesian timing side-channel detection system. It defines the statistical methodology, abstract types, and requirements that implementations MUST follow to be conformant.

For implementation guidance, see the [Implementation Guide](/guides/implementation). For language-specific APIs, see the [Rust API](/api/rust), [C API](/api/c), or [Go API](/api/go). For interpreting results, see the [User Guide](/guides/user-guide).

***

## Terminology (RFC 2119)

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

In summary:
- **MUST** / **REQUIRED** / **SHALL**: Absolute requirement
- **MUST NOT** / **SHALL NOT**: Absolute prohibition
- **SHOULD** / **RECOMMENDED**: Strong recommendation (valid reasons to deviate may exist)
- **SHOULD NOT**: Strong discouragement (valid reasons to deviate may exist)
- **MAY** / **OPTIONAL**: Truly optional

***

## 1. Overview

### 1.1 Problem Statement

Timing side-channel attacks exploit data-dependent execution time in cryptographic implementations. Existing detection tools have significant limitations:

- **T-test approaches** (DudeCT) compare means, missing distributional differences such as cache effects that only affect upper quantiles
- **P-value misinterpretation**: Statistical significance does not equal practical significance; with enough samples, negligible effects become "significant"
- **CI flakiness**: Fixed sample sizes cause tests to pass locally but fail in CI (or vice versa) due to environmental noise
- **Binary output**: No distinction between "no leak detected" and "couldn't measure reliably"

### 1.2 Solution

timing-oracle addresses these issues with:

1. **Quantile-based statistics**: Compare nine deciles to capture both uniform shifts and tail effects
2. **Adaptive Bayesian inference**: Collect samples until confident, with natural early stopping
3. **Three-way decisions**: Pass / Fail / Inconclusive, distinguishing "safe" from "unmeasurable"
4. **Interpretable output**: Posterior probability (0-100%) instead of p-values
5. **Fail-safe design**: Prefer Inconclusive over confidently wrong

### 1.3 Design Goals

- **Interpretable**: Output is a probability, not a t-statistic
- **Adaptive**: Collects more samples when uncertain, stops early when confident
- **CI-friendly**: Three-way output prevents flaky tests
- **Portable**: Handles different timer resolutions via adaptive batching
- **Honest**: Never silently clamps thresholds; reports what it can actually resolve
- **Fail-safe**: CI verdicts SHOULD almost never be confidently wrong
- **Reproducible**: Deterministic results given identical samples and configuration

***

## 2. Abstract Types and Semantics

This section defines the types that all implementations MUST provide. Types are specified using language-agnostic pseudocode.

### 2.1 Outcome

The primary result type returned by the oracle:

```
Outcome =
  | Pass {
      leak_probability: Float,        // P(m(δ) > θ_eff | Δ), always at θ_eff
      effect: EffectEstimate,
      theta_user: Float,              // User-requested threshold (ns)
      theta_eff: Float,               // Effective threshold used (ns)
      theta_floor: Float,             // Measurement floor at decision time (ns)
      decision_threshold_ns: Float,   // θ_eff at which decision was made
      samples_used: Int,
      quality: MeasurementQuality,
      diagnostics: Diagnostics
    }
  | Fail {
      leak_probability: Float,        // P(m(δ) > θ_eff | Δ)
      effect: EffectEstimate,
      exploitability: Exploitability,
      theta_user: Float,
      theta_eff: Float,               // MAY exceed θ_user (leak detected above floor)
      theta_floor: Float,
      decision_threshold_ns: Float,   // θ_eff at which decision was made
      samples_used: Int,
      quality: MeasurementQuality,
      diagnostics: Diagnostics
    }
  | Inconclusive {
      reason: InconclusiveReason,
      leak_probability: Float,        // P(m(δ) > θ_eff | Δ)
      effect: EffectEstimate,
      theta_user: Float,
      theta_eff: Float,
      theta_floor: Float,
      samples_used: Int,
      quality: MeasurementQuality,
      diagnostics: Diagnostics
    }
  | Unmeasurable {
      operation_ns: Float,            // Estimated operation time
      threshold_ns: Float,            // Timer resolution
      platform: String,
      recommendation: String
    }
  | Research {
      status: ResearchStatus,
      max_effect_ns: Float,           // Posterior mean of max|δ|
      max_effect_ci: (Float, Float),  // 95% credible interval
      theta_floor: Float,
      detectable: Bool,
      projection_mismatch: Bool,
      effect: EffectEstimate,
      samples_used: Int,
      quality: MeasurementQuality,
      diagnostics: Diagnostics
    }
```

**Semantics:**

The field `leak_probability` MUST be computed as $P(m(\delta) > \theta_{\text{eff}} \mid \Delta)$, where $\theta_{\text{eff}}$ is the effective threshold used for inference at decision time.

When $\theta_{\text{eff}}$ > $\theta_{\text{user}}$, the oracle cannot support a **Pass claim at $\theta_{\text{user}}$**, because effects in the range ($\theta_{\text{user}}$, $\theta_{\text{eff}}$] are not distinguishable from noise under the measured conditions.

Implementations MUST NOT substitute $\theta_{\text{user}}$ into `leak_probability` when $\theta_{\text{eff}}$ > $\theta_{\text{user}}$.

- **Pass**: MUST be returned when ALL of the following hold:
  1. `leak_probability < pass_threshold` (default 0.05)
  2. $\theta_{\text{eff}} \le \theta_{\text{user}} + \varepsilon_{\theta}$ where $\varepsilon_{\theta} = \max(\theta_{\text{tick}}, 10^{-6} \cdot \theta_{\text{user}})$
  3. All verdict-blocking quality gates pass
  4. $\theta_{\text{user}}$ > 0 (non-research mode)

- **Fail**: MUST be returned when ALL of the following hold:
  1. `leak_probability > fail_threshold` (default 0.95)
  2. All verdict-blocking quality gates pass

  Note: Fail MAY be returned when $\theta_{\text{eff}}$ > $\theta_{\text{user}}$. Detecting $m(\delta)$ > $\theta_{\text{eff}}$ implies $m(\delta)$ > $\theta_{\text{user}}$ (since $\theta_{\text{eff}}$ ≥ $\theta_{\text{user}}$ by construction).

- **Inconclusive**: MUST be returned when ANY of the following hold:
  1. A verdict-blocking quality gate fails
  2. Resource budgets exhausted without reaching a decision threshold
  3. `leak_probability < pass_threshold` but $\theta_{\text{eff}} > \theta_{\text{user}} + \varepsilon_{\theta}$ (threshold elevated)

- **Unmeasurable**: MUST be returned when the operation is too fast to measure reliably (see §4.5)

- **Research**: MUST be returned only when $\theta_{\text{user}}$ = 0. In research mode, Pass/Fail semantics do not apply; the oracle reports effect estimates at the measurement floor.

### 2.2 AttackerModel

Threat model presets defining the minimum effect size worth detecting:

```
AttackerModel =
  | SharedHardware      // θ = 0.6 ns (~2 cycles @ 3GHz)
  | PostQuantumSentinel // θ = 3.3 ns (~10 cycles @ 3GHz)
  | AdjacentNetwork     // θ = 100 ns
  | RemoteNetwork       // θ = 50,000 ns (50 μs)
  | Research            // θ = 0 (detect any difference)
  | Custom { threshold_ns: Float }
```

| Model | Threshold | Use Case |
|-------|-----------|----------|
| SharedHardware | 0.6 ns | SGX enclaves, cross-VM, containers, hyperthreading |
| PostQuantumSentinel | 3.3 ns | Post-quantum crypto (ML-KEM, ML-DSA) |
| AdjacentNetwork | 100 ns | LAN services, HTTP/2 APIs |
| RemoteNetwork | 50 $\mu\text{s}$ | Internet-exposed services |
| Research | 0 | Profiling, debugging (not for CI) |

**There is no single correct threshold.** The choice of attacker model is a statement about your threat model.

### 2.3 EffectEstimate

Decomposition of the timing effect:

```
EffectEstimate = {
  shift_ns: Float,                    // Uniform shift component
  tail_ns: Float,                     // Tail effect component
  credible_interval_ns: (Float, Float), // 95% CI for max|δ|
  pattern: EffectPattern,
  interpretation_caveat: Option<String>, // Present when projection mismatch
  top_quantiles: Option<List<TopQuantile>> // Present when projection mismatch
}

EffectPattern =
  | UniformShift   // Shift dominates: P(|μ| ≥ 5|τ|) ≥ 0.80
  | TailEffect     // Tail dominates: P(|τ| ≥ 5|μ|) ≥ 0.80
  | Mixed          // Neither dominates, both significant
  | Complex        // Projection mismatch high
  | Indeterminate  // Neither component significant

TopQuantile = {
  quantile_p: Float,     // e.g., 0.9 for 90th percentile
  mean_ns: Float,        // Posterior mean at this quantile
  ci95_ns: (Float, Float), // 95% marginal CI
  exceed_prob: Float     // P(|δ_k| > θ_eff | data)
}
```

### 2.4 Exploitability

Risk assessment based on effect size:

```
Exploitability =
  | SharedHardwareOnly  // < 10 ns
  | Http2Multiplexing   // 10–100 ns
  | StandardRemote      // 100 ns – 10 μs
  | ObviousLeak         // > 10 μs
```

| Level | Effect Size | Attack Vector |
|-------|-------------|---------------|
| SharedHardwareOnly | < 10 ns | ~1k queries on same physical core |
| Http2Multiplexing | 10–100 ns | ~100k concurrent HTTP/2 requests |
| StandardRemote | 100 ns – 10 $\mu\text{s}$ | ~1k-10k queries with standard timing |
| ObviousLeak | > 10 $\mu\text{s}$ | &lt;100 queries, trivially observable |

### 2.5 MeasurementQuality

Assessment of measurement precision:

```
MeasurementQuality =
  | Excellent  // MDE < 5 ns
  | Good       // MDE 5–20 ns
  | Poor       // MDE 20–100 ns
  | TooNoisy   // MDE > 100 ns
```

### 2.6 InconclusiveReason

```
InconclusiveReason =
  | DataTooNoisy { message: String, guidance: String }
  | NotLearning { message: String, guidance: String }
  | WouldTakeTooLong { estimated_time_secs: Float, samples_needed: Int, guidance: String }
  | ThresholdElevated {
      theta_user: Float,                    // What user requested
      theta_eff: Float,                     // What we measured at
      leak_probability_at_eff: Float,       // P(m(δ) > θ_eff | Δ)
      meets_pass_criterion_at_eff: Bool,    // True if P < pass_threshold at θ_eff
      achievable_at_max: Bool,              // Could θ_user be reached with max budget?
      message: String,
      guidance: String
    }
  | TimeBudgetExceeded { current_probability: Float, samples_collected: Int }
  | SampleBudgetExceeded { current_probability: Float, samples_collected: Int }
  | ConditionsChanged { drift: ConditionDrift }
```

The `meets_pass_criterion_at_eff` field indicates whether $P(m(\delta) > \theta_{\text{eff}} \mid \Delta)$ < pass_threshold. This allows CI systems to implement policies like "treat pass-criterion-met-at-floor as acceptable" without changing inference semantics.

The `achievable_at_max` field distinguishes:
- `true`: $\theta_{\text{floor}}$ > $\theta_{\text{user}}$ now, but $\theta_{\text{floor}}(n_{\max}) \le \theta_{\text{user}}$ (more sampling may help)
- `false`: $\theta_{\text{floor}}(n_{\max}) > \theta_{\text{user}}$ (cannot reach $\theta_{\text{user}}$ on this platform/configuration)

### 2.7 ResearchStatus

```
ResearchStatus =
  | EffectDetected       // CI clearly above θ_floor
  | NoEffectDetected     // CI clearly below θ_floor
  | ResolutionLimitReached // θ_floor ≈ θ_tick
  | QualityIssue { reason: InconclusiveReason }
  | BudgetExhausted
```

### 2.8 Diagnostics

```
Diagnostics = {
  dependence_length: Int,
  effective_sample_size: Int,
  stationarity_ratio: Float,
  stationarity_ok: Bool,
  outlier_rate_baseline: Float,
  outlier_rate_sample: Float,
  outlier_asymmetry_ok: Bool,
  projection_mismatch_q: Float,
  projection_mismatch_threshold: Float,
  projection_mismatch_ok: Bool,
  discrete_mode: Bool,
  timer_resolution_ns: Float,
  duplicate_fraction: Float,
  preflight_ok: Bool,
  calibration_samples: Int,
  total_time_secs: Float,
  warnings: List<String>,
  quality_issues: List<QualityIssue>,
  seed: Option<Int>,
  threshold_ns: Float,
  timer_name: String,
  platform: String,
  
  // Gibbs sampler diagnostics
  gibbs_iters_total: Int,           // N_gibbs used
  gibbs_burnin: Int,                // N_burn used
  gibbs_retained: Int,              // N_keep used
  lambda_mean: Float,               // Posterior mean of λ
  lambda_sd: Float,                 // Posterior SD of λ
  lambda_cv: Float,                 // Coefficient of variation: λ_sd / λ_mean
  lambda_ess: Float,                // Effective sample size of λ chain
  lambda_mixing_ok: Bool,           // See §3.4.4

  // Robust likelihood diagnostics
  kappa_mean: Float,                // Posterior mean of κ
  kappa_sd: Float,                  // Posterior SD of κ
  kappa_cv: Float,                  // Coefficient of variation: κ_sd / κ_mean
  kappa_ess: Float,                 // Effective sample size of κ chain
  kappa_mixing_ok: Bool             // See §3.4.4
}

QualityIssue = {
  code: IssueCode,
  message: String,
  guidance: String
}

IssueCode =
  | HighDependence | LowEffectiveSamples | StationaritySuspect
  | DiscreteTimer | SmallSampleDiscrete | HighGeneratorCost
  | LowUniqueInputs | QuantilesFiltered | ThresholdElevated
  | ThresholdClamped | HighWinsorRate | ProjectionMismatch
  | LambdaMixingPoor | KappaMixingPoor | LikelihoodInflated
```

***

## 3. Statistical Methodology

This section describes the mathematical foundation of timing-oracle. All formulas in this section are normative; implementations MUST produce equivalent results.

### 3.1 Test Statistic: Quantile Differences

We collect timing samples from two classes:
- **Fixed class (F)**: A specific input (e.g., all zeros)
- **Random class (R)**: Randomly sampled inputs

Rather than comparing means, we compare the distributions via their deciles. For each class, compute the 10th, 20th, ..., 90th percentiles, yielding two vectors in $\mathbb{R}^9$. The test statistic is their difference:

$$
\Delta = \hat{q}(\text{Fixed}) - \hat{q}(\text{Random}) \in \mathbb{R}^9
$$

where $\hat{q}_p$ denotes the empirical $p$-th quantile.

**Why quantiles instead of means?**

Timing leaks manifest in different ways:
- **Uniform shift**: A different code path adds constant overhead → all quantiles shift equally
- **Tail effect**: Cache misses occur probabilistically → upper quantiles shift more than lower

A t-test (comparing means) would detect uniform shifts but completely miss tail effects. Quantile differences capture both patterns in a single test statistic.

**Why nine deciles?**

This is a bias-variance tradeoff:
- Fewer quantiles would miss distributional structure
- More quantiles would require estimating a larger covariance matrix, introducing severe estimation noise

Nine deciles capture enough structure to distinguish shift from tail patterns while keeping the covariance matrix (9 × 9 = 81 parameters) tractable with typical sample sizes (10k–100k).

**Quantile computation:** Implementations SHOULD use **type 2** quantiles (inverse empirical CDF with averaging).¹ For sorted sample $x$ of size $n$ at probability $p$:

$$
h = n \cdot p + 0.5
$$

$$
\hat{q}_p = \frac{x_{\lfloor h \rfloor} + x_{\lceil h \rceil}}{2}
$$

For discrete timers with many tied values, use **mid-distribution quantiles** (see §3.7) which correctly handle atoms.

¹ Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages." The American Statistician 50(4):361–365.

### 3.2 Two-Phase Architecture

The system operates in two phases:

```
┌─────────────────────────────────────────────────────────────────┐
│                           Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Calibration │───▶│   Adaptive   │───▶│   Decision   │       │
│  │    Phase     │    │     Loop     │    │    Output    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│   • Estimate Σ_rate    • Collect batch    • Pass (P&lt;5%)         │
│   • Compute θ_floor    • Update Δ         • Fail (P>95%)        │
│   • Compute Q_thresh   • Scale Σ by 1/n   • Inconclusive        │
│   • Set prior scale    • Update θ_floor   • Research outcome    │
│   • Warmup caches      • Compute P(>θ)                          │
│   • Pre-flight checks  • Check quality                          │
│                        • Check stopping                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Phase 1: Calibration** (runs once)
- Collect initial samples to characterize measurement noise
- Estimate covariance structure via stream-based block bootstrap
- Compute "covariance rate" $\Sigma_{\text{rate}}$ that scales as $\Sigma = \Sigma_{\text{rate}} / n$
- Compute initial measurement floor $\theta_{\text{floor}}$ and floor-rate constant c_floor
- Compute projection mismatch threshold Q_proj,thresh from bootstrap distribution
- Compute effective threshold $\theta_{\text{eff}}$ and calibrate prior scale $\sigma$
- Run pre-flight checks (timer sanity, harness sanity, stationarity)

**Phase 2: Adaptive Loop** (iterates until decision)
- Collect batches of samples
- Update quantile estimates from all data collected so far
- Scale covariance: $\Sigma_n = \Sigma_{\text{rate}}/n$
- Update $\theta_{\text{floor}}(n)$ using floor-rate constant
- Run Gibbs sampler to approximate posterior and compute $P(\text{effect} > \theta_{\text{eff}})$
- Check quality gates (posterior ≈ prior → Inconclusive)
- Check decision boundaries (P > 95% → Fail, P < 5% → Pass)
- Check time/sample budgets

**Why this structure?**

The key insight is that covariance scales as 1/n for quantile estimators. By estimating the covariance *rate* once during calibration, we can cheaply update the posterior as more data arrives; no re-bootstrapping needed. This makes adaptive sampling computationally tractable.

### 3.3 Calibration Phase

The calibration phase runs once at startup to characterize measurement noise.

**Sample collection:**

Implementations SHOULD collect n_cal samples per class (default: 5,000). This is enough to estimate covariance structure reliably while keeping calibration fast.

**Fragile regime (base definition):**

A **fragile regime** is a measurement condition where standard statistical assumptions may not hold, requiring more conservative estimation. The base fragile regime is triggered when either:

- **Discrete timer mode**: The timer has coarse resolution (see §3.7)
- **Low uniqueness**: The minimum uniqueness ratio across classes is below 10%:
  $$\min\left(\frac{|\text{unique}(F)|}{n_F}, \frac{|\text{unique}(R)|}{n_R}\right) < 0.10$$

Subsequent sections extend this base definition with context-specific conditions. When a fragile regime is detected, implementations apply more conservative procedures (larger block lengths, regularized covariance) to maintain calibration.

#### 3.3.1 Acquisition Stream Model

Measurement produces an interleaved acquisition stream indexed by time:

$$
\{(c_t, y_t)\}_{t=1}^{T}, \quad c_t \in \{F, R\}, \; T \approx 2n
$$

where $y_t$ is the measured runtime (or ticks) at acquisition index $t$, and F/R denote Fixed and Random classes.

Per-class samples are obtained by filtering the stream:

$$
F := \{y_t : c_t = F\}, \quad R := \{y_t : c_t = R\}
$$

**Critical principle:** The acquisition stream is the data-generating process. All bootstrap and dependence estimation MUST preserve adjacency in acquisition order, not per-class position. The underlying stochastic process (including drift, frequency scaling, and cache state evolution) operates in continuous time indexed by $t$. Splitting by label and treating each class as an independent time series is statistically incorrect.

#### 3.3.2 Covariance Estimation via Stream-Based Block Bootstrap

Timing measurements exhibit autocorrelation: nearby samples are more similar than distant ones due to cache state, frequency scaling, etc. Standard bootstrap assumes i.i.d. samples, underestimating variance. Implementations MUST use block bootstrap on the acquisition stream to preserve the true dependence structure.

**Politis-White automatic block length selection:**

Implementations SHOULD use the Politis-White algorithm² to select the optimal block length. The key challenge is that block length selection must measure the autocorrelation relevant to quantile difference estimation, while respecting acquisition-stream timing.

**Step 1: Compute class-conditional acquisition-lag ACF**

The naive approach of computing ACF on the pooled stream $y_t$ directly is **anti-conservative**: class alternation in interleaved sampling masks within-class autocorrelation, leading to underestimated block lengths and inflated false positive rates.

Instead, compute autocorrelation at acquisition-stream lag $k$ using only **same-class pairs**:

$$
\hat{\rho}^{(F)}_k = \text{corr}(y_t, y_{t+k} \mid c_t = c_{t+k} = F)
$$

$$
\hat{\rho}^{(R)}_k = \text{corr}(y_t, y_{t+k} \mid c_t = c_{t+k} = R)
$$

Then combine conservatively:

$$
|\hat{\rho}^{(\max)}_k| = \max(|\hat{\rho}^{(F)}_k|, |\hat{\rho}^{(R)}_k|)
$$

This measures within-class dependence at acquisition-stream lags (the quantity that actually drives $\mathrm{Var}(\Delta)$) without being masked by class alternation.

**Why class-conditional ACF?** The variance of quantile estimators $\hat{q}(F)$ and $\hat{q}(R)$ depends on within-class autocorrelation. When computing ACF on the pooled interleaved stream, adjacent samples often belong to different classes, artificially reducing apparent autocorrelation. Class-conditional ACF preserves acquisition-stream timing (lag $k$ means $k$ acquisition steps apart) while measuring the relevant dependence structure.

Let $k_n = \max(5, \lfloor \log_{10}(T) \rfloor)$ and $m_{\max} = \lceil \sqrt{T} \rceil + k_n$.

Compute $|\hat{\rho}^{(\max)}_k|$ for $k = 0, \ldots, m_{\max}$. Find the first lag $m^*$ where $k_n$ consecutive values fall within the conservative band $\pm 2\sqrt{\log_{10}(T)/T}$. Set $m = \min(2 \cdot \max(m^*, 1), m_{\max})$.

**Step 2: Compute spectral quantities**

Using a flat-top (trapezoidal) kernel $h(x) = \min(1, 2(1 - |x|))$:

$$
\hat{\sigma}^2 = \sum_{k=-m}^{m} h\left(\frac{k}{m}\right) \hat{\gamma}^{(\max)}_k, \quad g = \sum_{k=-m}^{m} h\left(\frac{k}{m}\right) |k| \, \hat{\gamma}^{(\max)}_k
$$

where $\hat{\gamma}^{(\max)}_k$ is the autocovariance corresponding to $|\hat{\rho}^{(\max)}_k|$.

**Step 3: Compute optimal block length**

$$
\hat{b} = \left\lceil \left( \frac{g^2}{(\hat{\sigma}^2)^2} \right)^{1/3} T^{1/3} \right\rceil
$$

Capped at $b_{\max} = \min(3\sqrt{T}, T/3)$ to prevent degenerate blocks.

**Step 4: Apply safety bounds**

Automatic block length selectors can underestimate in pathological regimes (discrete timers, periodic interference). Implementations MUST apply:

$$
\hat{b} \leftarrow \max(\hat{b}, b_{\min})
$$

where $b_{\min} = 10$ (or a platform-specific floor).

In fragile regimes (base conditions above, or detected high autocorrelation where $|\hat{\rho}^{(\max)}_k| > 0.3$ persists beyond $k > b_{\min}$), implementations SHOULD apply an inflation factor:

$$
\hat{b} \leftarrow \lceil \gamma \cdot \hat{b} \rceil, \quad \gamma \in [1.2, 2.0]
$$

² Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection for the Dependent Bootstrap." Econometric Reviews 23(1):53–70.

**Stream-based bootstrap procedure:**

For each of $B$ bootstrap iterations (default: 2,000):

1. **Resample acquisition stream**: Sample block start indices on $t \in \{1, \ldots, T\}$ using moving block bootstrap with block length $\hat{b}$. Form resampled index sequence $(t_1, \ldots, t_T)$.

2. **Construct resampled stream**: $\{(c_{t_j}, y_{t_j})\}_{j=1}^{T}$

3. **Split by class**: Filter to obtain $F^* = \{y_{t_j} : c_{t_j} = F\}$ and $R^* = \{y_{t_j} : c_{t_j} = R\}$

4. **Compute quantile differences**: $\Delta^* = \hat{q}(F^*) - \hat{q}(R^*)$

Compute sample covariance of the $\Delta^*$ vectors using Welford's online algorithm³ (numerically stable).

³ Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419–420.

**Why stream-based resampling?**

The block bootstrap is only valid when blocks correspond to contiguous time segments of the process being modeled. Resampling per-class positions would implicitly model two independent time series, which is not what was measured. Stream-based resampling correctly preserves:
- Common-mode drift (thermal, frequency scaling)
- Cross-class correlation induced by interleaving
- The true dependence structure of the acquisition process

**Covariance rate:**

The covariance of quantile estimators scales as 1/n. We compute the "rate":

$$
\Sigma_{\text{rate}} = \hat{\Sigma}_{\text{cal}} \cdot n_{\text{cal}}
$$

**Effective sample size (n_eff) (normative):**

Under strong temporal dependence, n samples do not provide n independent observations. Implementations MUST define the effective sample size using the selected block length (dependence length estimate):

$$
n_{\text{eff}} := \max\left(1,\ \left\lfloor \frac{n}{\hat{b}} \right\rfloor\right)
$$

where $\hat{b}$ is the block length from the Politis-White selector (or platform-specific floor). This approximates the number of effectively independent blocks and prevents anti-conservative 1/n variance scaling when dependence is high.

Implementations MUST report:
- `Diagnostics.dependence_length` = $\hat{b}$
- `Diagnostics.effective_sample_size` = $n_{\text{eff}}$

**Covariance scaling during adaptive loop:**

During the adaptive loop, with $n$ total samples per class and $n_{\text{eff}}$ as defined above:

$$
\Sigma_n = \Sigma_{\text{rate}} / n_{\text{eff}}
$$

This allows cheap posterior updates without re-bootstrapping, while correctly accounting for reduced information under dependence.

**Numerical stability:**

With discrete timers or few unique values, some quantiles may have near-zero variance, making $\Sigma$ ill-conditioned. Implementations MUST regularize by ensuring a minimum diagonal value:

$$
\sigma^2_i \leftarrow \max(\sigma^2_i, 0.01 \cdot \bar{\sigma}^2) + \varepsilon
$$

where $\bar{\sigma}^2 = \text{tr}(\Sigma)/9$ and $\varepsilon = 10^{-10} + \bar{\sigma}^2 \cdot 10^{-8}$.

#### 3.3.3 Projection Mismatch Threshold Calibration

During bootstrap, calibrate the threshold for the projection mismatch diagnostic (see §3.5.3). This determines when the 2D shift+tail summary is a faithful description of the inferred 9D quantile profile.

For each bootstrap replicate $\Delta^*$:
1. Compute 9D posterior mean: $\delta^*_{\text{post}}$ under the Bayesian update
2. Compute GLS projection: $\beta^*_{\text{proj}} = A \delta^*_{\text{post}}$
3. Compute projection residual: $r^*_{\text{proj}} = \delta^*_{\text{post}} - X\beta^*_{\text{proj}}$
4. Compute mismatch statistic: $Q^*_{\text{proj}} = (r^*_{\text{proj}})^\top \Sigma_n^{-1} (r^*_{\text{proj}})$

The projection mismatch threshold is the 99th percentile of the bootstrap distribution:

$$
Q_{\text{proj,thresh}} = q_{0.99}(\{Q^*_{\text{proj},b}\}_{b=1}^B)
$$

**Note:** This threshold is used for the non-blocking projection mismatch diagnostic, not for verdict gating. The threshold is computed using the prior defined in §3.3.5; if the prior changes, Q_proj,thresh must be recomputed. For the bootstrap computation, use the posterior mean from a single Gibbs run (or a point estimate approximation for efficiency).

#### 3.3.4 Measurement Floor and Effective Threshold

A critical design element is distinguishing between what the user *wants* to detect ($\theta_{\text{user}}$) and what the measurement *can* detect ($\theta_{\text{floor}}$).

**Floor-rate constant:**

Under the model's scaling assumption $\Sigma_n = \Sigma_{\text{rate}}/n_{\text{eff}}$, the measurement floor scales as $1/\sqrt{n_{\text{eff}}}$. Implementations MUST compute a floor-rate constant once at calibration based on the 9D decision functional $m(\delta) = \max_k |\delta_k|$.

Draw $Z_0 \sim \mathcal{N}(0, \Sigma_{\text{rate}})$ via Monte Carlo (SHOULD use 50,000 samples) and compute:

$$
c_{\text{floor}} := q_{0.95}\left( \max_k |Z_{0,k}| \right)
$$

This is the 95th percentile of the max absolute value under unit sample size.

**Measurement floor (dynamic):**

During the adaptive loop with $n$ samples per class and $n_{\text{eff}}$ effective samples (see §3.3.2):

$$
\theta_{\text{floor,stat}}(n) = \frac{c_{\text{floor}}}{\sqrt{n_{\text{eff}}}}
$$

**Rationale for using n_eff:** If dependence reduces information, the floor should not pretend you gained √n resolution. Using n_eff ensures the floor honestly reflects achievable precision.

The tick floor is fixed once batching is determined:

$$
\theta_{\text{tick}} = \frac{\text{1 tick (ns)}}{K}
$$

where $K$ is the batch size.

The combined floor:

$$
\theta_{\text{floor}}(n) = \max\bigl(\theta_{\text{floor,stat}}(n), \, \theta_{\text{tick}}\bigr)
$$

**Effective threshold ($\theta_{\text{eff}}$):**

The threshold actually used for inference:

$$
\theta_{\text{eff}} = \begin{cases}
\max(\theta_{\text{user}}, \theta_{\text{floor}}) & \text{if } \theta_{\text{user}} > 0 \\
\theta_{\text{floor}} & \text{if } \theta_{\text{user}} = 0 \text{ (Research mode)}
\end{cases}
$$

**Threshold Elevation Decision Rule:**

When the measurement floor exceeds the user's requested threshold, claims at $\theta_{\text{user}}$ are fundamentally limited. This section defines the normative decision rule.

*Definitions:*

- $\theta_{\text{user}}$: User-requested threshold (from AttackerModel or custom)
- $\theta_{\text{floor}}(n)$: Measurement floor at sample count n (see formula above)
- $\theta_{\text{eff}}$: Effective threshold used for inference = $\max(\theta_{\text{user}}, \theta_{\text{floor}})$
- $\varepsilon_{\theta}$: Tolerance for threshold comparison = $\max(\theta_{\text{tick}}, 10^{-6} \cdot \theta_{\text{user}})$

*Core principle:*

When $\theta_{\text{eff}}$ > $\theta_{\text{user}}$, the oracle cannot support a Pass claim at $\theta_{\text{user}}$, because effects in the range ($\theta_{\text{user}}$, $\theta_{\text{eff}}$] are not distinguishable from noise under the measured conditions.

1. **Fail propagates**: Detecting $m(\delta)$ > $\theta_{\text{eff}}$ implies $m(\delta)$ > $\theta_{\text{user}}$ (since $\theta_{\text{eff}}$ ≥ $\theta_{\text{user}}$). Implementations MAY return Fail when $\theta_{\text{eff}}$ > $\theta_{\text{user}}$.

2. **Pass does not propagate**: "No detectable effect above $\theta_{\text{eff}}$" is compatible with effects in ($\theta_{\text{user}}$, $\theta_{\text{eff}}$]. Implementations MUST NOT return Pass when $\theta_{\text{eff}}$ > $\theta_{\text{user}}$ + $\varepsilon_{\theta}$.

*Decision rule (normative):*

At decision time, let $P = P(m(\delta) > \theta_{\text{eff}} \mid \Delta)$:

| Condition | P value | $\theta_{\text{eff}}$ vs $\theta_{\text{user}}$ | Outcome |
|-----------|---------|-----------------|---------|
| Leak detected | P > fail_threshold | any | **Fail** |
| No leak, threshold met | P < pass_threshold | $\theta_{\text{eff}}$ ≤ $\theta_{\text{user}}$ + $\varepsilon_{\theta}$ | **Pass** |
| No leak, threshold elevated | P < pass_threshold | $\theta_{\text{eff}}$ > $\theta_{\text{user}}$ + $\varepsilon_{\theta}$ | **Inconclusive**(ThresholdElevated) |
| Uncertain | else | any | Continue sampling or **Inconclusive** |

**Dynamic floor updates:**

During the adaptive loop, $\theta_{\text{floor}}(n)$ decreases as n grows. Implementations MUST:

1. Recompute $\theta_{\text{eff}} = \max(\theta_{\text{user}}, \theta_{\text{floor}}(n))$ after each batch
2. If $\theta_{\text{floor}}(n)$ drops to $\theta_{\text{user}}$ or below, Pass becomes possible (subject to posterior)
3. Report the $\theta_{\text{eff}}$ used for the **final** decision, not the maximum observed during the run

**Early termination heuristic (SHOULD):**

At calibration, compute $\theta_{\text{floor,max}} = \max(c_{\text{floor}}/\sqrt{n_{\max}}, \theta_{\text{tick}})$. If $\theta_{\text{floor,max}}$ > $\theta_{\text{user}}$ + $\varepsilon_{\theta}$:

- Set `achievable_at_max = false`
- If P < pass_threshold and the posterior is stable, implementations SHOULD terminate early with Inconclusive(ThresholdElevated) rather than exhausting the budget

This is a budget-control optimization, not a verdict-blocking gate.

#### 3.3.5 Prior Scale Calibration (Student's *t*)

The prior on the 9D effect profile $\delta$ MUST be a **correlation-shaped multivariate Student's *t*** distribution:

$$
\delta \sim t_\nu(0, \sigma^2 R)
$$

where R is the correlation matrix derived from $\Sigma_{\text{rate}}$, $\nu$ is a fixed degrees-of-freedom parameter, and $\sigma$ is a global scale calibrated to an exceedance target.

**Prior correlation matrix (R):**

Let D = diag($\Sigma_{\text{rate}}$), computed **after** any diagonal-floor regularization (§3.3.2). Define:

$$
R := \text{Corr}(\Sigma_{\text{rate}}) = D^{-1/2} \Sigma_{\text{rate}} D^{-1/2}
$$

**Numerical conditioning (required):** Implementations MUST ensure R is strictly SPD:

$$
R \leftarrow R + \varepsilon I, \quad \varepsilon \in [10^{-10}, 10^{-6}]
$$

chosen adaptively (start at $10^{-10}$, increase on Cholesky failure).

**Robustness fallback (fragile regimes) (normative):** In fragile regimes (§3.3) or when $\mathrm{cond}(R) > 10^4$, implementations MUST apply shrinkage:

$$
R \leftarrow (1-\lambda_{\text{shrink}})R + \lambda_{\text{shrink}} I, \quad \lambda_{\text{shrink}} \in [0.01, 0.2]
$$

chosen deterministically from conditioning diagnostics (e.g., stepwise based on condition number buckets).

**Precomputation requirement:** Implementations MUST precompute and cache a factorization sufficient to apply $R^{-1}$ efficiently (e.g., a Cholesky factor $L_R$ such that $L_R L_R^\top = R$). Implementations MUST NOT form explicit matrix inverses in floating point unless demonstrably stable. The quadratic form is computed as:

$$
\delta^\top R^{-1} \delta = \|L_R^{-1} \delta\|_2^2
$$

via a triangular solve.

**Degrees of freedom ($\nu$) (normative):**

Implementations MUST use:

$$
\nu := 4
$$

**Rationale:** $\nu = 4$ provides:
- Heavy tails (polynomial decay ∝ $|x|^{-5}$) that don't crush large effects
- Finite variance (required for Gate 1): $\mathrm{Var}(\delta)$ exists for $\nu > 2$
- Stable Gibbs mixing: not so heavy that $\lambda$ becomes degenerate

**Scale-mixture representation (normative):**

The Student's *t* prior MUST be implemented via the scale-mixture representation:

$$
\lambda \sim \text{Gamma}\left(\frac{\nu}{2}, \frac{\nu}{2}\right)
$$

$$
\delta \mid \lambda \sim \mathcal{N}\left(0, \frac{\sigma^2}{\lambda} R\right)
$$

**Gamma parameterization:** This specification uses **shape–rate** parameterization throughout. Under this parameterization, $E[\lambda] = 1$ and $\mathrm{Var}(\lambda) = 2/\nu$.

**Calibrating $\sigma$ via exceedance target (normative):**

The scale $\sigma$ MUST be chosen so that the prior exceedance probability equals a fixed target $\pi_0$ (default 0.62):

$$
P\left(\max_k |\delta_k| > \theta_{\text{eff}} \;\middle|\; \delta \sim t_\nu(0, \sigma^2 R)\right) = \pi_0
$$

This MUST be solved by deterministic 1D root-finding (e.g., bisection or Brent's method) using Monte Carlo integration.

**Monte Carlo sampling procedure:**

For each of M draws (SHOULD use M = 50,000):

1. Draw $\lambda \sim \text{Gamma}(\text{shape} = \nu/2, \text{rate} = \nu/2)$
2. Draw $z \sim \mathcal{N}(0, I_9)$
3. Compute $\delta = (\sigma/\sqrt{\lambda}) \cdot L_R z$, where $L_R$ is the Cholesky factor of $R$ (i.e., $L_R L_R^\top = R$)
4. Record whether $\max_k |\delta_k| > \theta_{\text{eff}}$

The exceedance probability is the fraction of draws exceeding $\theta_{\text{eff}}$.

**Search bounds (normative):**

Let the calibration-time standard errors be:

$$
\text{SE}_k := \sqrt{(\Sigma_{\text{rate}} / n_{\text{cal}})_{kk}}
$$

and SE_med be the median of {SE_k}_{k=1}^9.

Binary search bounds MUST be:
- $\sigma_{\text{lo}} = 0.05 \cdot \theta_{\text{eff}}$
- $\sigma_{\text{hi}} = \max(50 \cdot \theta_{\text{eff}}, 10 \cdot \text{SE}_{\text{med}})$

**Calibration timing:** The calibration MUST be performed once per run, during the calibration phase. The resulting $\sigma$ MUST remain fixed for the entire run, even if $\theta_{\text{eff}}$ changes due to floor updates.

#### 3.3.6 Deterministic Seeding Policy

To ensure reproducible results, all random number generation MUST be deterministic by default.

**Normative requirement:**

> Given identical timing samples and configuration, the oracle MUST produce identical results (up to floating-point roundoff). Algorithmic randomness is not epistemic uncertainty about the leak; it is approximation error that MUST NOT influence the decision rule.

**Seeding policy:**

- The bootstrap RNG seed MUST be deterministically derived from:
  - A fixed library constant seed (default: 0x74696D696E67, "timing" in ASCII)
  - A stable hash of configuration parameters

- All Monte Carlo RNG seeds (leak probability, floor constant, prior scale) MUST be similarly deterministic

- The Gibbs sampler RNG seed MUST be deterministic

- The chosen seeds SHOULD be reported in diagnostics

**Rationale:** For CI gating, the decision function must be well-defined given the data:

$$
\text{Verdict} = f(\text{timing samples}, \text{config})
$$

Not:

$$
\text{Verdict} = f(\text{timing samples}, \text{config}, \text{RNG state})
$$

### 3.4 Bayesian Model

We use a Student's *t* prior over the full 9D quantile-difference profile, implemented via Gibbs sampling on the scale-mixture representation.

#### 3.4.1 Latent Parameter

The latent parameter is the true per-decile timing difference profile:

$$
\delta \in \mathbb{R}^9 \quad \text{(true decile-difference profile)}
$$

This is unconstrained: the model can represent any quantile-difference pattern, unlike a low-rank projection.

#### 3.4.2 Likelihood (Robust t-likelihood via scale mixture)

Timing summary statistics $\Delta$ may deviate from the Gaussian approximation when $\Sigma_n$ is underestimated (e.g., high dependence, interference bursts, discrete timers). To prevent pathological posterior certainty under covariance misestimation, implementations MUST use a robust likelihood with a single scalar precision factor $\kappa$.

$$
\Delta \mid \delta, \kappa \sim \mathcal{N}\!\left(\delta, \frac{\Sigma_n}{\kappa}\right)
$$

$$
\kappa \sim \text{Gamma}\!\left(\frac{\nu_\ell}{2}, \frac{\nu_\ell}{2}\right)
$$

Marginally, this gives a multivariate t-distribution:

$$
\Delta \mid \delta \sim t_{\nu_\ell}(\delta, \Sigma_n)
$$

where $\Sigma_n = \Sigma_{\text{rate}} / n_{\text{eff}}$ is the scaled covariance (see §3.3.2 for $n_{\text{eff}}$ definition).

**Gamma parameterization:** shape–rate. $E[\kappa] = 1$.

**Degrees of freedom for likelihood ($\nu_{\ell}$) (normative) default:**

Implementations MUST use $\nu_{\ell}$ := 8 unless overridden by a calibration-driven conservatism escalation policy (§3.8).

**Rationale:** When $\Sigma_n$ is underestimated (common when dependence is worse than calibration captured), $\kappa$ is pulled downward and inflates uncertainty automatically, preventing pathological certainty (P(leak) → 0 or 1) from artifacts.

#### 3.4.3 Prior (Student's *t* via scale mixture)

The prior on $\delta \in \mathbb{R}^9$ is:

$$
\delta \sim t_\nu(0, \sigma^2 R), \quad \nu = 4
$$

Implementations MUST implement inference using the equivalent hierarchical model:

$$
\lambda \sim \text{Gamma}\left(\frac{\nu}{2}, \frac{\nu}{2}\right)
$$

$$
\delta \mid \lambda \sim \mathcal{N}\left(0, \frac{\sigma^2}{\lambda} R\right)
$$

where Gamma uses **shape–rate** parameterization.

**Marginal prior covariance:**

For $\nu$ > 2, the marginal covariance of $\delta$ is:

$$
\text{Cov}(\delta) = \frac{\nu}{\nu - 2} \sigma^2 R
$$

For $\nu$ = 4:

$$
\Lambda_0^{\text{marginal}} := \text{Cov}(\delta) = 2\sigma^2 R
$$

This matrix is used as the prior covariance reference in Gate 1 (§3.5.2).

**Why Student's *t* instead of Gaussian?**

A Gaussian prior with scale calibrated to the exceedance target can cause catastrophic shrinkage when measurement noise is high relative to $\theta_{\text{eff}}$. The Student's *t* prior's heavy tails (polynomial decay rather than exponential) naturally accommodate large effects without requiring explicit mixture components.

**Why Student's *t* instead of Gaussian mixture?**

A 2-component Gaussian mixture (as in v5.2) suffers from discrete scale selection. Under correlated likelihoods, the Bayes factor computation for mixture weights is dominated by Occam penalties (log-det terms), causing the posterior to stick to the narrow component even when data clearly favors larger effects, but not as large as the slab.

The Student's *t* prior makes the effective scale $\lambda$ continuous. The posterior on $\lambda$ can settle at whatever scale the data supports, rather than being forced to choose between two fixed options.

#### 3.4.4 Posterior Inference (Deterministic Gibbs Sampling)

Because the prior is Student's *t* and the likelihood uses a robust t-distribution (§3.4.2), the marginal posterior $p(\delta \mid \Delta)$ is not available in closed form. Implementations MUST approximate the posterior using a **deterministic Gibbs sampler** over $(\delta, \lambda, \kappa)$.

**Gibbs conditionals:**

**Conditional 1: $\delta$ | $\lambda$, $\kappa$, $\Delta$ (Gaussian)**

Given $\lambda$ (prior precision multiplier) and $\kappa$ (likelihood precision multiplier), the prior covariance is:

$$
\Lambda_0(\lambda) = \frac{\sigma^2}{\lambda} R
$$

The effective likelihood covariance is $\Sigma_n/\kappa$. The posterior is conjugate Gaussian:

$$
\delta \mid \lambda, \kappa, \Delta \sim \mathcal{N}(\mu(\lambda, \kappa), \Lambda(\lambda, \kappa))
$$

where:

$$
\Lambda(\lambda, \kappa) = \left(\kappa \Sigma_n^{-1} + \Lambda_0(\lambda)^{-1}\right)^{-1} = \left(\kappa \Sigma_n^{-1} + \frac{\lambda}{\sigma^2} R^{-1}\right)^{-1}
$$

$$
\mu(\lambda, \kappa) = \Lambda(\lambda, \kappa) \cdot \kappa \Sigma_n^{-1} \Delta
$$

**Sampling without explicit inverses (normative):**

Let $Q(\lambda, \kappa) := \kappa \Sigma_n^{-1} + (\lambda/\sigma^2) R^{-1}$. Note that $\Sigma_n^{-1}$ and $R^{-1}$ here denote the result of applying the inverse operator, which MUST be computed via Cholesky solves, not explicit matrix inversion. Compute the Cholesky factorization:

$$
L L^\top = Q(\lambda, \kappa)
$$

The posterior mean is computed by solving:

$$
Q(\lambda, \kappa) \mu(\lambda, \kappa) = \kappa \Sigma_n^{-1} \Delta
$$

via forward-backward substitution (two triangular solves).

To sample $\delta$:

$$
\delta = \mu(\lambda, \kappa) + L^{-\top} z, \quad z \sim \mathcal{N}(0, I_9)
$$

where $L^{-\top} z$ MUST be computed via a single triangular solve (backward substitution), not by forming $L^{-1}$.

**Conditional 2: $\lambda$ | $\delta$ (Gamma)**

Let d = 9 (dimension) and define:

$$
q := \delta^\top R^{-1} \delta = \|L_R^{-1} \delta\|_2^2
$$

The conditional posterior is:

$$
\lambda \mid \delta \sim \text{Gamma}\left(\text{shape} = \frac{\nu + d}{2}, \; \text{rate} = \frac{\nu + \sigma^{-2} q}{2}\right)
$$

**Derivation:** Combining the Gamma($\nu/2, \nu/2$) prior on $\lambda$ with the Gaussian likelihood contribution $\exp\left(-(\lambda/2\sigma^2)\delta^\top R^{-1}\delta\right)$ yields a Gamma posterior with shape increased by $d/2$ and rate increased by $q/(2\sigma^2)$.

**Conditional 3: $\kappa$ | $\delta$, $\Delta$ (Gamma) (normative)**

The likelihood precision multiplier $\kappa$ is sampled conditional on $\delta$ and $\Delta$. Let d = 9 and define the Mahalanobis residual:

$$
s := (\Delta - \delta)^\top \Sigma_n^{-1} (\Delta - \delta)
$$

The conditional posterior is:

$$
\kappa \mid \delta, \Delta \sim \text{Gamma}\left(\text{shape} = \frac{\nu_\ell + d}{2}, \; \text{rate} = \frac{\nu_\ell + s}{2}\right)
$$

where $\nu_{\ell}$ = 8 is the likelihood degrees of freedom (see §3.4.2).

**Derivation:** Combining the Gamma($\nu_{\ell}/2, \nu_{\ell}/2$) prior on $\kappa$ with the Gaussian likelihood contribution $\exp\left(-(\kappa/2)(\Delta-\delta)^\top \Sigma_n^{-1} (\Delta-\delta)\right)$ yields a Gamma posterior with shape increased by $d/2$ and rate increased by $s/2$.

**Interpretation:** When the observed residual s is large relative to expectations under $\Sigma_n$ (indicating potential covariance misestimation), $\kappa$ is pulled below 1, effectively inflating the likelihood covariance and preventing false certainty.

**Gibbs schedule (normative):**

Implementations MUST use the following deterministic schedule:

| Parameter | Value | Description |
|-----------|-------|-------------|
| N_gibbs | 256 | Total iterations |
| N_burn | 64 | Burn-in (discarded) |
| N_keep | 192 | Retained samples |

**Initialization (normative):**

The Gibbs sampler MUST be initialized at:

$$
\lambda^{(0)} = 1, \quad \kappa^{(0)} = 1
$$

**Rationale:** Under the Gamma priors, $E[\lambda] = E[\kappa] = 1$. Starting at the prior means provides a neutral initialization.

**Iteration order:**

For t = 1, ..., N_gibbs:
1. Sample $\delta^{(t)} \sim p(\delta \mid \lambda^{(t-1)}, \kappa^{(t-1)}, \Delta)$
2. Sample $\lambda^{(t)} \sim p(\lambda \mid \delta^{(t)})$
3. Sample $\kappa^{(t)} \sim p(\kappa \mid \delta^{(t)}, \Delta)$

Retain $\{(\delta^{(t)}, \lambda^{(t)}, \kappa^{(t)})\}_{t=N_{\text{burn}}+1}^{N_{\text{gibbs}}}$.

**RNG seeding:**

The RNG seed MUST be deterministic per §3.3.6. Given identical inputs ($\Delta$, $\Sigma_n$, configuration), the Gibbs sampler MUST produce identical output.

**Precomputation requirements:**

To minimize per-iteration cost, implementations MUST precompute during calibration:

1. **L_R** (Cholesky factor of R, i.e., L_R L_Rᵀ = R)
2. **$L_{\Sigma}$** (Cholesky factor of $\Sigma_n$, updated when n changes)
3. **$\Sigma_n^{-1} \Delta$** (the "data precision-weighted observation", computed via Cholesky solve)

Per iteration, the dominant cost is the Cholesky factorization of $Q(\lambda)$, which is $O(9^3) \approx 729$ flops (negligible).

**$\lambda$ mixing diagnostics (normative):**

Implementations SHOULD compute $\lambda$ mixing diagnostics and SHOULD set `lambda_mixing_ok = false` when either:

1. **Low coefficient of variation:** lambda_cv < 0.1, indicating $\lambda$ barely moved from initialization
2. **Low effective sample size:** lambda_ess < 20, indicating high autocorrelation

**Effective sample size computation:**

$$
\text{ESS} = \frac{N_{\text{keep}}}{1 + 2\sum_{k=1}^{K} \hat{\rho}_k}
$$

where $\hat{\rho}_k$ is the lag-k autocorrelation of the $\lambda$ chain, summed until $\hat{\rho}_k < 0.05$ or $k > 50$.

**Warning emission:**

When `lambda_mixing_ok = false`, implementations SHOULD emit a quality issue with code `LambdaMixingPoor` and guidance suggesting the user increase Gibbs iterations (if configurable) or noting that the posterior may be unreliable.

**$\kappa$ mixing diagnostics (normative):**

Implementations SHOULD compute $\kappa$ mixing diagnostics and SHOULD set `kappa_mixing_ok = false` when either:

1. **Low coefficient of variation:** kappa_cv < 0.1, indicating $\kappa$ barely moved from initialization
2. **Low effective sample size:** kappa_ess < 20, indicating high autocorrelation

ESS computation follows the same formula as for $\lambda$.

**Likelihood inflation warning:**

Implementations SHOULD emit `QualityIssue::LikelihoodInflated` (non-blocking) when:

- `kappa_mean < 0.3`

This indicates that the run appears inconsistent with the estimated $\Sigma_n$ and that uncertainty was inflated for robustness. The guidance SHOULD note that the likelihood covariance was effectively scaled up by $\sim 1/\kappa_{\text{mean}}$ to maintain calibration.

When `kappa_mixing_ok = false`, implementations SHOULD emit a quality issue with code `KappaMixingPoor`.

**Verdict-blocking status:**

These diagnostics MUST NOT be verdict-blocking. Poor $\lambda$ or $\kappa$ mixing typically indicates unusual data rather than invalid inference; the Gibbs sampler may simply be exploring a complex posterior.

#### 3.4.5 Decision Functional and Leak Probability

The decision functional is:

$$
m(\delta) := \max_k |\delta_k|
$$

The leak probability is:

$$
P(\text{leak} > \theta_{\text{eff}} \mid \Delta) = P(m(\delta) > \theta_{\text{eff}} \mid \Delta)
$$

**Estimation via Gibbs samples (normative):**

Implementations MUST estimate the leak probability from the retained Gibbs samples:

$$
\widehat{P}(\text{leak}) = \frac{1}{N_{\text{keep}}} \sum_{s=1}^{N_{\text{keep}}} \mathbf{1}\left[m(\delta^{(s)}) > \theta_{\text{eff}}\right]
$$

**Posterior summaries (normative):**

Implementations MUST compute:

- **Posterior mean:** $\delta_{\text{post}} := \frac{1}{N_{\text{keep}}} \sum_s \delta^{(s)}$
- **Max-effect credible interval:** 95% CI for $m(\delta)$ from empirical quantiles of $\{m(\delta^{(s)})\}$

Implementations SHOULD compute:

- **Posterior standard deviation:** element-wise SD of $\{\delta^{(s)}\}$
- **$\lambda$ posterior mean:** $\bar{\lambda} := \frac{1}{N_{\text{keep}}} \sum_s \lambda^{(s)}$

**Thinning:**

Thinning is NOT required. The Gibbs conditionals for this model mix rapidly (both are conjugate with moderate dimensionality). Implementations MAY thin retained samples but MUST keep the normative iteration counts unchanged.

**Monte Carlo error:**

With N_keep = 192 samples, the standard error on a probability estimate p̂ is approximately:

$$
\text{SE}(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{N_{\text{keep}}}} \leq \frac{1}{2\sqrt{192}} \approx 0.036
$$

This is acceptable for decision thresholds at 0.05 and 0.95.

**Interpreting the probability:**

This is a **posterior probability**, not a p-value. When we report "72% probability of a leak," we mean: given the data and our model, 72% of the posterior mass corresponds to effects exceeding $\theta_{\text{eff}}$.

**Note on $\theta_{\text{eff}}$:**

All inference uses $\theta_{\text{eff}}$, not $\theta_{\text{user}}$. When these differ, the output MUST clearly report both values so the user understands what was actually tested.

#### 3.4.6 2D Projection for Reporting

The 9D posterior is used for decisions, but we provide a 2D summary for interpretability when the pattern is simple.

**Design matrix (for projection only):**

$$
X = \begin{bmatrix} \mathbf{1} & \mathbf{b}_{\text{tail}} \end{bmatrix} \in \mathbb{R}^{9 \times 2}
$$

where:
- **1** = (1, 1, 1, 1, 1, 1, 1, 1, 1)ᵀ: uniform shift
- **b_tail** = (−0.5, −0.375, −0.25, −0.125, 0, 0.125, 0.25, 0.375, 0.5)ᵀ: tail effect

**GLS projection operator:**

$$
A := (X^\top \Sigma_n^{-1} X)^{-1} X^\top \Sigma_n^{-1}
$$

**Conditioning stability for $\Sigma_n$ (normative):** In fragile regimes (§3.3), $\Sigma_n$ may be severely ill-conditioned due to high correlations between quantiles. Implementations MUST apply condition-number-based regularization to $\Sigma_n$ before use in the Gibbs sampler and GLS projection:

- If $\mathrm{cond}(\Sigma_n) > 10^4$ but $\le 10^6$: Apply shrinkage $\Sigma_n \leftarrow (1-\lambda)\Sigma_n + \lambda \cdot \mathrm{diag}(\Sigma_n)$ with $\lambda \in [0.1, 0.95]$ based on condition severity
- If $\mathrm{cond}(\Sigma_n) > 10^6$ or Cholesky fails: Fall back to diagonal matrix $\mathrm{diag}(\Sigma_n)$ (weighted OLS, preserves per-quantile variances)

**Posterior for projection summary (normative):**

Implementations MUST compute $\beta$ projection statistics empirically from Gibbs draws rather than analytically. This ensures robustness when $\Sigma_n$ is regularized.

For each retained $\delta^{(s)}$ draw from the Gibbs sampler, compute $\beta^{(s)} = A \delta^{(s)}$. Then:

- **Mean**: $\beta_{\text{proj}} = \frac{1}{S} \sum_s \beta^{(s)}$ (empirical mean of projected draws)
- **Covariance**: $\mathrm{Cov}(\beta_{\text{proj}})$ = sample_cov($\beta^{(s)}$) (empirical covariance of projected draws)

Retain all $\beta^{(s)}$ draws for pattern classification (see below).

The projection gives interpretable components:
- **Shift ($\mu$)**: Uniform timing difference affecting all quantiles equally
- **Tail ($\tau$)**: Upper quantiles affected more than lower (or vice versa)

**Important:** The 2D projection is for reporting only. Decisions MUST be based on the 9D posterior. When the projection doesn't fit well (see §3.5.3), implementations MUST add an interpretation caveat and provide alternative explanations.

**Pattern Classification via Posterior Draws (normative):**

Pattern classification MUST use dominance probabilities computed from retained Gibbs draws, not point estimates or significance tests. This approach is robust to covariance regularization and discrete mode quantization.

For each retained $\delta$ draw, compute $\beta = A \delta$ to get a projected $(\mu, \tau)$ sample. The classification uses the following probabilities computed over all draws:

- **P_shift_dominates**: Fraction of draws where $|\mu| \ge 5|\tau|$ (shift dominates tail)
- **P_tail_dominates**: Fraction of draws where $|\tau| \ge 5|\mu|$ (tail dominates shift)
- **P_shift_significant**: Fraction of draws where $|\mu| > 10\,\text{ns}$ (absolute threshold)
- **P_tail_significant**: Fraction of draws where $|\tau| > 10\,\text{ns}$ (absolute threshold)

Classification rules (applied in order):
1. **UniformShift**: P_shift_dominates ≥ 0.80
2. **TailEffect**: P_tail_dominates ≥ 0.80
3. **Mixed**: P_shift_significant ≥ 0.80 AND P_tail_significant ≥ 0.80
4. **Indeterminate**: Otherwise

The 5× dominance ratio and 80% probability threshold are robust defaults. The absolute 10ns significance threshold prevents noise from triggering "significant" classifications.

**Rationale:** Significance-based classification (|effect| > 2×SE) is brittle when covariance is regularized or in discrete mode, because standard errors may be poorly estimated. Dominance-based classification using draws directly reflects posterior uncertainty about which component is larger.

### 3.5 Adaptive Sampling Loop

The core innovation: collect samples until confident, with natural early stopping.

**Verdict-blocking semantics:**

> Pass/Fail verdicts MUST be emitted **only** if all measurement quality gates pass and condition drift checks pass. Otherwise the oracle MUST return Inconclusive, reporting the posterior probability as an "estimate under the model" but not asserting a strong claim.

This policy ensures: **CI verdicts should almost never be confidently wrong.**

#### 3.5.1 Stopping Criteria

The adaptive loop terminates when any of these conditions is met:

1. **Pass**: `leak_probability < pass_threshold` (default 0.05) AND all quality gates pass
2. **Fail**: `leak_probability > fail_threshold` (default 0.95) AND all quality gates pass
3. **Inconclusive**: Any quality gate fails OR budget exhausted without reaching decision threshold

**Why adaptive sampling works for Bayesian inference:**

Frequentist methods suffer from **optional stopping**: if you keep sampling until you get a significant result, you inflate your false positive rate.

Bayesian methods don't have this problem. The posterior probability is valid regardless of when you stop; this is the **likelihood principle**. Your inference depends only on the data observed, not your sampling plan.

#### 3.5.2 Quality Gates (Verdict-Blocking)

Quality gates detect when data is too poor to reach a confident decision. When any gate triggers, the outcome MUST be Inconclusive.

**Gate 1: Low Information Gain (Posterior ≈ Prior) (verdict-blocking)**

Gate 1 MUST trigger Inconclusive when the data provides insufficient information relative to the prior. Implementations MUST compute the KL divergence between Gaussian surrogates of the prior and posterior.

**Prior surrogate (Gaussian):**

Under the Student's *t* prior with $\nu = 4$, the marginal prior covariance is:

$$
\Lambda_0^{\text{marginal}} := \text{Cov}(\delta) = 2\sigma^2 R
$$

The prior surrogate is N(0, $\Lambda_0^{\text{marginal}}$).

**Posterior surrogate (Gaussian):**

Estimate $\mu_{\text{post}}$ and $\Lambda_{\text{post}}$ from the Gibbs samples:

$$
\mu_{\text{post}} = \delta_{\text{post}} = \frac{1}{N_{\text{keep}}} \sum_s \delta^{(s)}
$$

$$
\Lambda_{\text{post}} \approx \frac{1}{N_{\text{keep}}-1} \sum_s (\delta^{(s)} - \delta_{\text{post}})(\delta^{(s)} - \delta_{\text{post}})^\top
$$

The posterior surrogate is $\mathcal{N}(\mu_{\text{post}}, \Lambda_{\text{post}})$.

**KL divergence computation (normative):**

The KL divergence from the prior surrogate to the posterior surrogate is:

$$
\mathrm{KL} = \frac{1}{2}\left(
\mathrm{tr}\left((\Lambda_0^{\text{marginal}})^{-1}\Lambda_{\text{post}}\right)
+ \mu_{\text{post}}^\top(\Lambda_0^{\text{marginal}})^{-1}\mu_{\text{post}}
- d + \ln\frac{|\Lambda_0^{\text{marginal}}|}{|\Lambda_{\text{post}}|}
\right)
$$

where d = 9.

Implementations MUST compute KL deterministically using Cholesky solves (no explicit inverses). If $\Lambda_{\text{post}}$ is not SPD, implementations MUST apply a deterministic diagonal jitter ladder (starting at $10^{-10}$, increasing by factors of 10) until Cholesky succeeds; if still failing after jitter reaches $10^{-4}$, implementations MUST fall back to a diagonal approximation (conservative).

**Trigger condition (normative default):**

Trigger gate when KL < KL_min where KL_min := 0.7 nats.

When triggered, return Inconclusive with reason `DataTooNoisy`.

**Rationale:** KL correctly accounts for both mean shift (information learned about the location of $\delta$) and covariance contraction (information learned about uncertainty). Unlike a pure log-det ratio, KL captures cases where the posterior mean moved significantly even if covariance contracted only modestly.

Implementations SHOULD validate KL_min via the null calibration tests (§3.8) and MAY apply deterministic escalation if FPR_gated exceeds targets.

**Gate 2: Learning Rate Collapsed**

Track KL divergence between successive posteriors. For the Gaussian approximation to the posterior (using sample mean and covariance from Gibbs):

$$
\text{KL}(p_{\text{new}} \| p_{\text{old}}) = \frac{1}{2}\left( \text{tr}(\Lambda_{\text{old}}^{-1} \Lambda_{\text{new}}) + (\mu_{\text{old}} - \mu_{\text{new}})^\top \Lambda_{\text{old}}^{-1} (\mu_{\text{old}} - \mu_{\text{new}}) - k + \ln\frac{|\Lambda_{\text{old}}|}{|\Lambda_{\text{new}}|} \right)
$$

where k = 9 for the 9D model.

If the sum of recent KL divergences (e.g., over last 5 batches) falls below 0.001, trigger Inconclusive with reason `NotLearning`.

**Gate 3: Would Take Too Long**

Extrapolate time to decision based on current convergence rate. If projected time exceeds budget by a large margin (e.g., 10×), trigger Inconclusive with reason `WouldTakeTooLong`.

**Gate 4: Time Budget Exceeded**

If elapsed time exceeds configured time budget, trigger Inconclusive with reason `TimeBudgetExceeded`.

**Gate 5: Sample Budget Exceeded**

If total samples per class exceeds configured maximum, trigger Inconclusive with reason `SampleBudgetExceeded`.

**Gate 6: Condition Drift Detected**

The covariance estimate $\Sigma_{\text{rate}}$ is computed during calibration. If measurement conditions change during the adaptive loop, this estimate becomes invalid.

Detect condition drift by comparing measurement statistics from calibration against the full test run:

- Variance ratio: $\sigma_{\text{post}}^2 / \sigma_{\text{cal}}^2$
- Autocorrelation change: $|\rho_{\text{post}}(1) - \rho_{\text{cal}}(1)|$
- Mean drift: $|\mu_{\text{post}} - \mu_{\text{cal}}| / \sigma_{\text{cal}}$

If variance ratio is outside [0.5, 2.0], or autocorrelation change exceeds 0.3, or mean drift exceeds 3.0, trigger Inconclusive with reason `ConditionsChanged`.

**Note on threshold elevation:** The case where $\theta_{\text{floor}}$ > $\theta_{\text{user}}$ is handled by the threshold elevation decision rule (§3.3.4), not by a quality gate. This allows Fail to be returned when large leaks are detected, even if the user's exact threshold cannot be achieved. See §3.3.4 for the early termination heuristic when `achievable_at_max = false`.

#### 3.5.3 Non-Blocking Diagnostics

**Projection Mismatch Diagnostic**

Under the 9D model, there is no risk of "mean model cannot represent the observed shape": the 9D posterior can represent any quantile pattern. The projection mismatch diagnostic measures whether the 2D shift+tail summary faithfully describes the inferred 9D shape.

**Projection mismatch statistic:**

Using the posterior mean $\delta_{\text{post}}$:

$$
r_{\text{proj}} := \delta_{\text{post}} - X\beta_{\text{proj,post}}
$$

$$
Q_{\text{proj}} := r_{\text{proj}}^\top \Sigma_n^{-1} r_{\text{proj}}
$$

**Semantics (normative):**

- Projection mismatch MUST NOT be verdict-blocking
- If $Q_{\text{proj}} > Q_{\text{proj,thresh}}$, implementations MUST:
  - Set `effect.interpretation_caveat`
  - Report a "top quantiles" shape hint (see below)
  - Mark `Diagnostics.projection_mismatch_ok = false`

**Top quantiles shape hint:**

When projection mismatch is high, report the 2–3 most relevant quantiles ranked by per-quantile exceedance probability:

$$
p_k := P(|\delta_k| > \theta_{\text{eff}} \mid \Delta)
$$

Compute this from the Gibbs samples as the fraction where $|\delta_k^{(s)}| > \theta_{\text{eff}}$.

**Selection rule:** Select top 2–3 indices by p_k (descending). Optional stability filter: include only those with p_k ≥ 0.10; if fewer than 2 remain, relax to include the top 2.

### 3.6 Research Mode

Research mode answers: "Is there a detectable timing difference above the measurement's noise floor?" without requiring a user-specified threshold.

#### 3.6.1 Motivation

Research mode serves users who want to:
- Profile implementations for any timing variation
- Debug suspected leaks without committing to a threshold
- Perform academic analysis of timing behavior

These users want the posterior distribution, not a binary Pass/Fail decision.

#### 3.6.2 Design Principle

Research mode uses the same Bayesian machinery as normal mode, but:
1. Sets $\theta_{\text{user}}$ = 0, so $\theta_{\text{eff}}$ = $\theta_{\text{floor}}$
2. Reports posterior credible intervals instead of exceedance probabilities
3. Returns a `Research` outcome instead of Pass/Fail

The decision functional is the same:

$$
m(\delta) = \max_k |\delta_k|
$$

#### 3.6.3 Stopping Conditions

| Condition | Criterion | Status |
|-----------|-----------|--------|
| Effect detected | CI lower bound > 1.1 × $\theta_{\text{floor}}$ | `EffectDetected` |
| No effect detected | CI upper bound < 0.9 × $\theta_{\text{floor}}$ | `NoEffectDetected` |
| Resolution limit | $\theta_{\text{floor}}$ ≈ $\theta_{\text{tick}}$ | `ResolutionLimitReached` |
| Quality issue | Any quality gate triggers | `QualityIssue` |
| Budget exhausted | Time or samples exceeded | `BudgetExhausted` |

The 1.1× and 0.9× margins provide hysteresis to prevent oscillating near the boundary.

### 3.7 Discrete Timer Mode

When the timer has low resolution (e.g., Apple Silicon's 41ns cntvct_el0), quantile estimation behaves differently due to tied values.

**Trigger condition:**

Discrete timer mode triggers on the low uniqueness condition from the base fragile regime definition (§3.3): minimum uniqueness ratio below 10%. This condition is one of the two base fragile regime triggers, meaning discrete timer mode automatically implies fragile regime handling throughout the pipeline.

**Mid-distribution quantiles:**

Instead of standard quantile estimators, use mid-distribution quantiles which handle ties correctly:

$$
F_{\text{mid}}(x) = F(x) - \frac{1}{2}p(x), \quad \hat{q}^{\text{mid}}_k = F^{-1}_{\text{mid}}(k)
$$

where p(x) is the probability mass at x.

**Work in ticks internally:**

In discrete mode, implementations SHOULD perform computations in **ticks** (timer's native unit):
- $\Delta$, $\theta$, effect sizes all in ticks
- Convert to nanoseconds only for display

**Covariance estimation:**

Implementations SHOULD use m-out-of-n bootstrap for covariance estimation in discrete mode:

$$
m = \lfloor n^{2/3} \rfloor
$$

This provides consistent variance estimation when the standard CLT doesn't apply.

**Prior correlation fallback:**

In discrete mode, apply the robustness shrinkage described in §3.3.5:

$$
R \leftarrow (1-\lambda_{\text{shrink}})R + \lambda_{\text{shrink}} I, \quad \lambda_{\text{shrink}} \in [0.01, 0.2]
$$

**Gaussian approximation caveat:**

The Gaussian likelihood is a rougher approximation with discrete data. Implementations MUST report a quality issue about discrete timer mode and frame probabilities as "approximate under Gaussianized model."

### 3.8 Calibration Validation Requirements

The Bayesian approach requires empirical validation that posteriors are well-calibrated. This section defines **normative, pipeline-level** calibration requirements.

**Null calibration test (normative requirement):**

Implementations MUST provide (or run in internal test suite) a "fixed-vs-fixed" validation that measures **end-to-end false positive rates**:

- Run the full pipeline under null (same distribution both classes)
- Compute two distinct FPR metrics:
  - FPR_overall = P(Fail | H₀): unconditional false positive rate
  - FPR_gated = P(Fail | H₀, all verdict-blocking gates pass): FPR conditional on conclusive results
- Also track Inconclusive rate and reasons

**Trial count requirement:**

Implementations SHOULD run **at least 500 trials** for stable FPR estimates. With 100 trials, a true 5% FPR has 95% CI of approximately [2%, 11%], which is too wide to reliably detect miscalibration.

**Acceptance criteria (normative):**

| Metric | Target | Acceptable | Action if Exceeded |
|--------|--------|------------|-------------------|
| FPR_gated | 2-5% | ≤ 5% | MUST escalate conservatism |
| FPR_overall | 2-5% | ≤ 10% | SHOULD escalate conservatism |

If FPR_gated > 5%, implementations MUST apply remediation (see below).

**Anti-conservative remediation:**

When null calibration tests detect elevated FPR, implementations SHOULD apply deterministic escalation:

1. **Increase block length**: Multiply b̂ by 1.5 (or 2.0 if FPR > 8%)
2. **Increase bootstrap iterations**: Use B = 4000 instead of 2000
3. **Tighten decision thresholds**: Use α_pass = 0.01 instead of 0.05

Re-run calibration after each escalation step. This remediation can run:
- In library CI (recommended)
- In a `--self-test` mode
- Automatically at first run on a new platform

**Expected calibration results:**

| True Effect | Expected $P(\text{leak} > \theta)$ | Acceptable Range |
|-------------|---------------------|------------------|
| 0 | ~2-5% | 0-10% |
| $\theta/2$ | ~5-15% | 0-25% |
| $\theta$ | ~50% | 35-65% |
| $2\theta$ | ~95% | 85-100% |
| $3\theta$ | ~99% | 95-100% |

This is an end-to-end check that $c_{\text{floor}}$, $\Sigma$ scaling, block length selection, prior calibration, and posterior computation are not systematically anti-conservative.

**Large-effect detection tests (normative requirement):**

Implementations MUST include validation tests for the "large effect + noisy likelihood" regime. These tests validate the inference layer directly (the probability P(leak), not the verdict), ensuring the Student's *t* prior correctly detects obvious leaks even when measurement noise is high.

**Test case 1: Diagonal $\Sigma_n$ (independent coordinates)**

Using synthetic data with diagonal covariance:
- $\theta_{\text{eff}}$ = 100 ns
- True effect: $\Delta$ with entries ~10,000–15,000 ns (≈100× threshold)
- Data SEs: ~2,500–8,000 ns (≈25–80× threshold)
- $\Sigma_n = \mathrm{diag}(SE_1^2, \ldots, SE_9^2)$
- $R = I_9$

Reference values (SILENT web app dataset):
```
Δ = [10366, 13156, 13296, 12800, 11741, 12936, 13215, 11804, 18715] ns
SE = [2731, 2796, 2612, 2555, 2734, 3125, 3953, 5662, 8105] ns
```

Assert: P(leak) > 0.99

**Test case 2: Correlated $\Sigma_n$ (AR(1) structure)**

Using synthetic data with AR(1) correlation structure. Let C denote the AR(1) correlation matrix (distinct from the prior correlation R):

$$
C_{ij} = \rho^{|i-j|}, \quad \rho = 0.7
$$

Construct the test covariance as:
- Same $\Delta$ and diagonal SEs as Test case 1
- $\Sigma_n = D^{1/2} C D^{1/2}$ where $D = \mathrm{diag}(SE^2)$

Assert: P(leak) > 0.99

**Test case 3: Correlated $\Sigma_n$ (bootstrap-estimated)**

Using a real or realistic bootstrap-estimated $\Sigma_n$ with non-trivial off-diagonal structure:
- Same $\Delta$ magnitudes as Test case 1
- $\Sigma_n$ from an actual calibration run or a reference covariance matrix

Assert: P(leak) > 0.99

**Test implementation requirements:**

- Tests MUST use fixed $\Sigma_n$, R, and $\Delta$ values for determinism
- Tests MUST run at the inference layer (Gibbs sampler + leak probability), not the full pipeline
- Tests MUST use deterministic seeds per §3.3.6
- Tests MUST NOT invoke verdict-blocking quality gates

***

## 4. Measurement Requirements

This section defines abstract requirements for measurement. For implementation details, see [implementation-guide.md](implementation-guide.md).

### 4.1 Timer Requirements

Implementations MUST use a timer that:
- Is monotonic (never decreases)
- Has known resolution
- Reports results that can be converted to nanoseconds

Implementations SHOULD use the highest-resolution timer available on the platform.

### 4.2 Acquisition Stream Requirements

Measurements MUST be collected as an interleaved acquisition stream (see §3.3.1):
- Fixed and Random class measurements MUST be interleaved
- The interleaving order SHOULD be randomized
- The full acquisition stream (with class labels) MUST be preserved for bootstrap

### 4.3 Input Pre-generation

All inputs MUST be generated before the measurement loop begins. Generating inputs inside the timed region causes false positives.

### 4.4 Outlier Handling

Implementations MUST cap (winsorize), not drop, outliers:
1. Compute t_cap = 99.99th percentile from pooled data
2. Cap samples exceeding t_cap
3. Winsorization happens before quantile computation

**Quality thresholds:** >0.1% capped → warning; >1% → acceptable; >5% → `TooNoisy`.

### 4.5 Adaptive Batching

On platforms with coarse timer resolution, implementations SHOULD batch operations:

**When batching is needed:**

If pilot measurement shows fewer than 5 ticks per call, enable batching.

**Batch size selection:**

$$
K = \text{clamp}\left( \left\lceil \frac{50}{\text{ticks\_per\_call}} \right\rceil, 1, 20 \right)
$$

**Effect scaling:**

Reported effects MUST be divided by K to give per-operation estimates.

### 4.6 Measurability

If ticks per call < 5 even with maximum batching (K=20), implementations MUST return `Unmeasurable`.

### 4.7 Pre-flight Checks

Implementations SHOULD perform pre-flight checks:
- **Timer sanity**: Verify monotonicity and reasonable resolution
- **Harness sanity (fixed-vs-fixed)**: Detect test harness bugs
- **Stationarity**: Detect drift during measurement

***

## 5. API Design Principles

This section provides language-agnostic guidance for API design. These are recommendations (SHOULD) unless marked otherwise.

### 5.1 Input Specification

**Two-class pattern:**

Implementations SHOULD expose the DudeCT two-class pattern:
- **Baseline class**: Fixed input (typically all zeros)
- **Sample class**: Variable input (typically random)

This pattern tests for data-dependent timing, not specific value comparisons.

**Generator/operation separation:**

Implementations SHOULD separate input generation from the operation being tested. Generators MUST be called before the measurement loop, not inside it.

### 5.2 Configuration Ergonomics

**Attacker model presets as primary entry point:**

The primary configuration entry point SHOULD be attacker model selection, not raw threshold values. This makes the API self-documenting about threat models.

**Sane defaults:**

Default configuration SHOULD:
- Use `AdjacentNetwork` attacker model (or equivalent)
- Set time budget to 60 seconds
- Set sample budget to 1,000,000
- Set pass/fail thresholds to 0.05/0.95

### 5.3 Result Communication

**Leak probability prominence:**

The leak probability MUST be prominently displayed in results and human-readable output.

**Threshold transparency:**

When $\theta_{\text{eff}}$ > $\theta_{\text{user}}$, implementations MUST clearly indicate this to the user.

**Exploitability context:**

For `Fail` outcomes, implementations SHOULD provide exploitability assessment to help users understand practical risk.

**Inconclusive guidance:**

For `Inconclusive` outcomes, implementations MUST provide the reason and SHOULD provide actionable guidance.

### 5.4 Error Prevention

**Unmeasurable detection:**

Implementations MUST detect unmeasurable operations early and return clear guidance rather than producing unreliable results.

**Quality issue surfacing:**

Quality issues SHOULD be included in results even for `Pass`/`Fail` outcomes, allowing users to assess measurement reliability.

**Pre-flight warning severity:**

Pre-flight warnings SHOULD distinguish informational messages from result-undermining issues.

***

## 6. Configuration Parameters

### 6.1 Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| attacker_model OR threshold_ns | AttackerModel or Float | Defines the effect threshold |

### 6.2 Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| time_budget | 60 seconds | Maximum test duration |
| max_samples | 1,000,000 | Maximum samples per class |
| pass_threshold | 0.05 | P(leak) below this → Pass |
| fail_threshold | 0.95 | P(leak) above this → Fail |
| calibration_samples | 5,000 | Samples for calibration phase |
| batch_size | 1,000 | Samples per adaptive batch |
| bootstrap_iterations | 2,000 | Bootstrap iterations for covariance |

***

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| {(c_t, y_t)} | Acquisition stream: class labels and timing measurements |
| T | Acquisition stream length (≈ 2n) |
| F, R | Per-class sample sets (filtered from stream) |
| $\Delta$ | 9-vector of observed quantile differences |
| $\delta$ | 9-vector of true (latent) quantile differences |
| q̂_F(k), q̂_R(k) | Empirical quantiles for Fixed and Random classes |
| X | 9 × 2 projection basis [**1** ∣ **b**_tail] |
| $\beta_{\text{proj}}$ | 2D projection: ($\mu$, $\tau$)ᵀ |
| $\Sigma_n$ | Covariance matrix at sample size n: $\Sigma_n = \Sigma_{\text{rate}}/n_{\text{eff}}$ |
| $\Sigma_{\text{rate}}$ | Covariance rate |
| n_eff | Effective sample size: n_eff = floor(n / b̂) |
| R | Prior correlation matrix: $\mathrm{Corr}(\Sigma_{\text{rate}})$ |
| $\nu$ | Student's *t* degrees of freedom (fixed at 4) |
| $\sigma$ | Student's *t* prior scale (calibrated via exceedance target) |
| $\lambda$ | Latent prior precision multiplier in scale-mixture representation |
| $\kappa$ | Latent likelihood precision multiplier (robust t-likelihood) |
| $\nu_{\ell}$ | Likelihood degrees of freedom (fixed at 8) |
| $t_{\nu}(0, \Sigma)$ | Multivariate Student's *t* with $\nu$ df, location 0, scale matrix $\Sigma$ |
| $\Lambda_0^{\text{marginal}}$ | Marginal prior covariance: $(\nu/(\nu-2))\sigma^2 R = 2\sigma^2 R$ for $\nu = 4$ |
| $\Lambda_{\text{post}}$ | Posterior covariance of $\delta$ (estimated from Gibbs samples) |
| $Q(\lambda, \kappa)$ | Precision matrix in Gibbs: $\kappa \Sigma_n^{-1} + (\lambda/\sigma^2) R^{-1}$ |
| s | Mahalanobis residual for $\kappa$ conditional: $(\Delta-\delta)^\top \Sigma_n^{-1} (\Delta-\delta)$ |
| $\delta_{\text{post}}$ | Posterior mean of $\delta$ (from Gibbs samples) |
| L_R | Cholesky factor of R (L_R L_Rᵀ = R) |
| N_gibbs | Total Gibbs iterations |
| N_burn | Burn-in iterations (discarded) |
| N_keep | Retained Gibbs samples |
| C | AR(1) correlation matrix in test cases ($C_{ij} = \rho^{|i-j|}$) |
| $\theta_{\text{user}}$ | User-requested threshold |
| $\theta_{\text{floor}}$ | Measurement floor (smallest resolvable effect) |
| $c_{\text{floor}}$ | Floor-rate constant: $\theta_{\text{floor,stat}} = c_{\text{floor}}/\sqrt{n_{\text{eff}}}$ |
| $\theta_{\text{tick}}$ | Timer resolution component of floor |
| $\theta_{\text{eff}}$ | Effective threshold used for inference |
| $m(\delta)$ | Decision functional: $\max_k \|\delta_k\|$ |
| Q_proj | Projection mismatch statistic |
| Q_proj,thresh | Projection mismatch threshold (99th percentile) |
| b̂ | Block length (Politis-White, on acquisition stream) |
| MDE | Minimum detectable effect |
| n | Samples per class |
| B | Bootstrap iterations |
| ESS | Effective sample size (Gibbs chain) |
| SE_k | Standard error of k-th quantile difference |
| SE_med | Median of SE_k over k ∈ {1,...,9} |
| $\lambda_{\text{shrink}}$ | Shrinkage parameter for correlation matrix regularization |
| KL | KL divergence: KL(posterior ∥ prior) for Gate 1 |
| KL_min | Minimum KL threshold for conclusive verdict (default 0.7 nats) |

***

## Appendix B: Normative Constants

These constants define conformant implementations. Implementations MAY use different values only where noted.

| Constant | Default | Normative | Rationale |
|----------|---------|-----------|-----------|
| Deciles | {0.1, ..., 0.9} | MUST | Nine quantile positions |
| Prior family | Student's *t* | MUST | Continuous scale adaptation |
| Degrees of freedom ($\nu$) | 4 | MUST | Heavy tails + finite variance |
| Gamma parameterization | shape–rate | MUST | Avoid library ambiguity |
| Gibbs iterations (N_gibbs) | 256 | MUST | Sufficient mixing for 10D |
| Gibbs burn-in (N_burn) | 64 | MUST | Conservative |
| Gibbs retained (N_keep) | 192 | MUST | MC variance control |
| Gibbs initialization ($\lambda^{(0)}$, $\kappa^{(0)}$) | 1 | MUST | Prior means |
| Likelihood df ($\nu_{\ell}$) | 8 | MUST | Robustness to $\Sigma_n$ misestimation |
| Prior exceedance target ($\pi_0$) | 0.62 | SHOULD | Genuine uncertainty |
| Prior calibration MC draws | 50,000 | SHOULD | Stable $\sigma$ calibration |
| $\sigma$ search lower bound | 0.05 · $\theta_{\text{eff}}$ | MUST | Prevent degenerate narrow prior |
| $\sigma$ search upper bound | max(50·$\theta_{\text{eff}}$, 10·SE_med) | MUST | Cover high-noise regimes |
| Bootstrap iterations | 2,000 | SHOULD | Covariance estimation accuracy |
| Monte Carlo samples (c_floor) | 50,000 | SHOULD | Floor-rate constant estimation |
| Batch size | 1,000 | SHOULD | Adaptive iteration granularity |
| Calibration samples | 5,000 | SHOULD | Initial covariance estimation |
| Pass threshold | 0.05 | SHOULD | 95% confidence of no leak |
| Fail threshold | 0.95 | SHOULD | 95% confidence of leak |
| KL_min (nats) | 0.7 | MUST | Minimum information gain for conclusive verdict |
| Projection mismatch percentile | 99th | SHOULD | Q_proj_thresh from bootstrap |
| Block length cap | min(3√T, T/3) | SHOULD | Prevent degenerate blocks |
| Discrete threshold | 10% unique | SHOULD | Trigger discrete mode |
| Min ticks per call | 5 | SHOULD | Measurability floor |
| Max batch size | 20 | SHOULD | Limit microarch artifacts |
| Default time budget | 60 s | MAY | Maximum runtime |
| Default sample budget | 1,000,000 | MAY | Maximum samples |
| CI hysteresis (Research) | 10% | SHOULD | Margin for detection decisions |
| Default RNG seed | 0x74696D696E67 | SHOULD | "timing" in ASCII |
| R jitter ($\varepsilon$) | $10^{-10}$–$10^{-6}$ | MUST | Ensure SPD |
| R shrinkage ($\lambda_{\text{shrink}}$) | 0.01–0.2 | MUST | Robustness under fragile regimes |
| Condition number threshold (R) | $10^4$ | MUST | Trigger R shrinkage |
| Condition number threshold ($\Sigma_n$) | $10^4$ | MUST | Trigger $\Sigma_n$ shrinkage |
| $\Sigma_n$ OLS fallback threshold | $10^6$ | MUST | Fall back to diagonal matrix |
| $\lambda$ mixing CV threshold | 0.1 | SHOULD | Detect stuck sampler |
| $\lambda$ mixing ESS threshold | 20 | SHOULD | Detect slow mixing |
| $\kappa$ mixing CV threshold | 0.1 | SHOULD | Detect stuck sampler |
| $\kappa$ mixing ESS threshold | 20 | SHOULD | Detect slow mixing |
| Likelihood inflation threshold | 0.3 | SHOULD | $\kappa_{\text{mean}}$ triggering LikelihoodInflated |

***

## Appendix C: References

**Statistical methodology:**

1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning, Ch. 3. Springer. (Bayesian linear regression)

2. Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection for the Dependent Bootstrap." Econometric Reviews 23(1):53–70.

3. Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." Annals of Statistics. (Block bootstrap)

4. Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages." The American Statistician 50(4):361–365.

5. Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419–420.

6. Gelman, A. et al. (2013). Bayesian Data Analysis, 3rd ed., Ch. 11-12. CRC Press. (Gibbs sampling, scale mixtures)

7. Lange, K. L., Little, R. J. A., & Taylor, J. M. G. (1989). "Robust Statistical Modeling Using the t Distribution." JASA 84(408):881-896. (Student's t for robustness)

**Timing attacks:**

8. Reparaz, O., Balasch, J., & Verbauwhede, I. (2016). "Dude, is my code constant time?" DATE. (DudeCT methodology)

9. Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009). "Opportunities and Limits of Remote Timing Attacks." ACM TISSEC 12(3):17. (Exploitability thresholds)

10. Van Goethem, T., et al. (2020). "Timeless Timing Attacks." USENIX Security. (HTTP/2 timing attacks)

11. Bernstein, D. J. et al. (2024). "KyberSlash." (Timing vulnerability example)

12. Dunsche, M. et al. (2025). "SILENT: A New Lens on Statistics in Software Timing Side Channels." arXiv:2504.19821. (Relevant hypotheses framework)

**Existing tools:**

13. dudect (C): https://github.com/oreparaz/dudect
14. dudect-bencher (Rust): https://github.com/rozbb/dudect-bencher

***

## Appendix D: Changelog

### v5.6 (from v5.5)

**Robust t-likelihood with $\kappa$ (§3.4.2, §3.4.4):**

- **Added:** Robust t-likelihood via scale mixture: $\Delta \mid \delta, \kappa \sim \mathcal{N}(\delta, \Sigma_n/\kappa)$ with $\kappa \sim \text{Gamma}(\nu_{\ell}/2, \nu_{\ell}/2)$
- **Added:** Likelihood degrees of freedom $\nu_{\ell}$ = 8 (normative default)
- **Added:** $\kappa$ conditional in Gibbs sampler (Conditional 3)
- **Added:** Gibbs now samples over ($\delta$, $\lambda$, $\kappa$) instead of ($\delta$, $\lambda$)
- **Changed:** Q($\lambda$) → $Q(\lambda, \kappa) = \kappa \Sigma_n^{-1} + (\lambda/\sigma^2) R^{-1}$
- **Rationale:** When $\Sigma_n$ is underestimated (common under high dependence, interference, discrete timers), $\kappa$ is pulled below 1, automatically inflating uncertainty and preventing pathological posterior certainty.

**Effective sample size ($n_{\text{eff}}$) for $\Sigma$ scaling (§3.3.2, §3.3.4):**

- **Added:** $n_{\text{eff}} := \max(1, \lfloor n/\hat{b} \rfloor)$ where $\hat{b}$ is the block length
- **Changed:** $\Sigma_n = \Sigma_{\text{rate}}/n_{\text{eff}}$ (was $\Sigma_{\text{rate}}/n$)
- **Changed:** $\theta_{\text{floor,stat}} = c_{\text{floor}}/\sqrt{n_{\text{eff}}}$ (was $c_{\text{floor}}/\sqrt{n}$)
- **Rationale:** Under strong dependence, n samples do not provide n independent observations. Using n_eff honestly reflects achievable precision and prevents anti-conservative variance scaling.

**KL-based Gate 1 (§3.5.2):**

- **Changed:** Gate 1 now uses KL divergence between Gaussian surrogates instead of log-det ratio
- **Removed:** "Exception for decisive probabilities" bypass (was allowing verdicts when P > 0.995 or P < 0.005)
- **Added:** KL_min = 0.7 nats as normative threshold for minimum information gain
- **Rationale:** KL correctly accounts for both mean shift and covariance contraction. With $\kappa$ in the likelihood, "decisive but suspicious" cases should no longer occur, eliminating the need for the bypass.

**$\kappa$ diagnostics (§2.8, §3.4.4):**

- **Added:** kappa_mean, kappa_sd, kappa_cv, kappa_ess, kappa_mixing_ok to Diagnostics
- **Added:** KappaMixingPoor, LikelihoodInflated issue codes
- **Added:** LikelihoodInflated warning when kappa_mean < 0.3
- **Rationale:** Users should see why P(leak) isn't decisive when $\kappa$ inflates uncertainty.

**Notation updates (Appendix A, B):**

- **Added:** $\kappa$, $\nu_{\ell}$, n_eff, s, KL, KL_min to notation table
- **Changed:** Q($\lambda$) → $Q(\lambda, \kappa)$ in notation
- **Added:** $\kappa$-related constants to Appendix B
- **Replaced:** Variance ratio gate (0.5) with KL_min (0.7 nats)

### v5.5 (from v5.4)

**Threshold elevation decision rule (§2.1, §2.6, §3.3.4, §3.5.2):**

- **Changed:** Pass now requires $\theta_{\text{eff}}$ ≤ $\theta_{\text{user}}$ + $\varepsilon_{\theta}$ (cannot Pass when threshold is elevated)
- **Changed:** When $\theta_{\text{floor}}$ > $\theta_{\text{user}}$ and P < pass_threshold, outcome is Inconclusive(ThresholdElevated), not Pass
- **Added:** Fail MAY be returned when $\theta_{\text{eff}}$ > $\theta_{\text{user}}$ (large leaks are still detectable)
- **Added:** `decision_threshold_ns` field to Pass/Fail outcomes
- **Removed:** Gate 4 (ThresholdUnachievable) as verdict-blocking gate; subsumed by decision rule
- **Renamed:** `ThresholdUnachievable` → `ThresholdElevated` with additional fields
- **Added:** `meets_pass_criterion_at_eff` and `achievable_at_max` fields to ThresholdElevated
- **Added:** $\varepsilon_{\theta}$ tolerance = $\max(\theta_{\text{tick}}, 10^{-6} \cdot \theta_{\text{user}})$ for threshold comparison
- **Rationale:** A Pass at $\theta_{\text{eff}}$ does not certify absence of leaks at $\theta_{\text{user}}$ when $\theta_{\text{eff}}$ > $\theta_{\text{user}}$. Effects in the range ($\theta_{\text{user}}$, $\theta_{\text{eff}}$] are not distinguishable from noise. Fail can propagate because detecting $m(\delta)$ > $\theta_{\text{eff}}$ implies $m(\delta)$ > $\theta_{\text{user}}$.

**Gate renumbering (§3.5.2):**

- Gate 4 removed (was ThresholdUnachievable)
- Gate 5 → Gate 4 (Time Budget Exceeded)
- Gate 6 → Gate 5 (Sample Budget Exceeded)
- Gate 7 → Gate 6 (Condition Drift Detected)

**leak_probability semantics (§2.1):**

- **Clarified:** `leak_probability` MUST always be $P(m(\delta) > \theta_{\text{eff}} \mid \Delta)$, never $P(m(\delta) > \theta_{\text{user}} \mid \Delta)$
- **Added:** Implementations MUST NOT substitute $\theta_{\text{user}}$ into leak_probability when $\theta_{\text{eff}}$ > $\theta_{\text{user}}$
- **Rationale:** Probabilities computed at sub-floor thresholds are not calibrated.

### v5.4 (from v5.2)

**Student's *t* prior (§3.3.5, §3.4.3):**

- **Changed:** Prior is now multivariate Student's *t* with $\nu = 4$: $\delta \sim t_4(0, \sigma^2 R)$
- **Removed:** 2-component Gaussian mixture ($w$, $\sigma_1$, $\sigma_2$, $\Lambda_{0,1}$, $\Lambda_{0,2}$)
- **Rationale:** The discrete mixture suffered from Occam penalty pathology under correlated likelihoods; the slab's log-det penalty dominated even when data favored moderate (not extreme) scale increases. Student's *t* provides continuous scale adaptation via the latent $\lambda$.

**Gibbs sampling inference (§3.4.4, §3.4.5):**

- **Changed:** Posterior computed via deterministic Gibbs sampling (256 iter, 64 burn-in)
- **Added:** Explicit initialization ($\lambda^{(0)} = 1$) and precomputation requirements
- **Added:** Explicit guidance against forming matrix inverses; use Cholesky solves throughout
- **Rationale:** Student's *t* posterior has no closed form; Gibbs exploits conjugacy in the scale-mixture representation.

**Gate 1 prior covariance (§3.5.2):**

- **Changed:** Uses marginal prior covariance $2\sigma^2 R$ (for $\nu = 4$) instead of per-component covariance
- **Added:** Deterministic fallback pathway for log-det computation (Cholesky success → log-det; failure → trace ratio)
- **Rationale:** Single prior family, no mixture components; ensures consistent behavior across platforms.

**$\lambda$ mixing diagnostics (§2.8, §3.4.4):**

- **Added:** gibbs_* fields, lambda_mean, lambda_sd, lambda_cv, lambda_ess, lambda_mixing_ok
- **Added:** LambdaMixingPoor quality issue code
- **Removed:** slab_weight_post, SlabDominant
- **Rationale:** New diagnostics for Gibbs sampler health replace mixture-specific diagnostics.

**Notation clarification (§3.8, Appendix A):**

- **Added:** C for AR(1) correlation matrix in test cases, distinct from prior correlation R
- **Rationale:** Avoid naming collision where R was overloaded.

**Validation tests (§3.8):**

- **Added:** Test case 3 (bootstrap-estimated covariance)
- **Rationale:** Ensures realistic covariance structures are covered, not just synthetic AR(1).

### v5.2 (from v5.1)

**Mixture prior (§3.3.5, §3.4.3, §3.4.4, §3.4.5):**

- **Changed:** Prior is now a 2-component Gaussian scale mixture: $w \cdot \mathcal{N}(0, \sigma_1^2 R) + (1-w) \cdot \mathcal{N}(0, \sigma_2^2 R)$
- **Added:** Slab component with deterministic scale $\sigma_2 = \max(50\theta_{\text{eff}}, 10\,\text{SE}_{\text{med}})$
- **Added:** Narrow component $\sigma_1$ calibrated so mixture satisfies 62% exceedance
- **Added:** Posterior is 2-component mixture with weights updated via marginal likelihood
- **Rationale:** A single $\theta$-calibrated Gaussian causes catastrophic shrinkage when SE ≫ $\theta$.

### v5.1 (from v5.0)

**Prior shape matrix change (§3.3.5, §3.4.3):**

- **Changed:** Prior covariance now uses the correlation matrix $R = \mathrm{Corr}(\Sigma_{\text{rate}})$ instead of the trace-normalized shape matrix $S = \Sigma_{\text{rate}}/\mathrm{tr}(\Sigma_{\text{rate}})$
- **Rationale:** The trace-normalized shape matrix caused heteroskedastic marginal prior variances across coordinates.
