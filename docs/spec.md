# timing-oracle Specification (v5.1)

This document is the authoritative specification for timing-oracle, a Bayesian timing side-channel detection system. It defines the statistical methodology, abstract types, and requirements that implementations MUST follow to be conformant.

For implementation guidance, see [implementation-guide.md](implementation-guide.md). For language-specific APIs, see [api-rust.md](api-rust.md), [api-c.md](api-c.md), or [api-go.md](api-go.md). For interpreting results, see [guide.md](guide.md).

---

## Terminology (RFC 2119)

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

In summary:
- **MUST** / **REQUIRED** / **SHALL**: Absolute requirement
- **MUST NOT** / **SHALL NOT**: Absolute prohibition
- **SHOULD** / **RECOMMENDED**: Strong recommendation (valid reasons to deviate may exist)
- **SHOULD NOT**: Strong discouragement (valid reasons to deviate may exist)
- **MAY** / **OPTIONAL**: Truly optional

---

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
3. **Three-way decisions**: Pass / Fail / Inconclusive—distinguishing "safe" from "unmeasurable"
4. **Interpretable output**: Posterior probability (0–100%) instead of p-values
5. **Fail-safe design**: Prefer Inconclusive over confidently wrong

### 1.3 Design Goals

- **Interpretable**: Output is a probability, not a t-statistic
- **Adaptive**: Collects more samples when uncertain, stops early when confident
- **CI-friendly**: Three-way output prevents flaky tests
- **Portable**: Handles different timer resolutions via adaptive batching
- **Honest**: Never silently clamps thresholds; reports what it can actually resolve
- **Fail-safe**: CI verdicts SHOULD almost never be confidently wrong
- **Reproducible**: Deterministic results given identical samples and configuration

---

## 2. Abstract Types and Semantics

This section defines the types that all implementations MUST provide. Types are specified using language-agnostic pseudocode.

### 2.1 Outcome

The primary result type returned by the oracle:

```
Outcome =
  | Pass {
      leak_probability: Float,      // P(max|δ| > θ_eff | data)
      effect: EffectEstimate,
      theta_user: Float,            // User-requested threshold (ns)
      theta_eff: Float,             // Effective threshold used (ns)
      theta_floor: Float,           // Measurement floor (ns)
      samples_used: Int,
      quality: MeasurementQuality,
      diagnostics: Diagnostics
    }
  | Fail {
      leak_probability: Float,
      effect: EffectEstimate,
      exploitability: Exploitability,
      theta_user: Float,
      theta_eff: Float,
      theta_floor: Float,
      samples_used: Int,
      quality: MeasurementQuality,
      diagnostics: Diagnostics
    }
  | Inconclusive {
      reason: InconclusiveReason,
      leak_probability: Float,      // Current posterior estimate
      effect: EffectEstimate,
      theta_user: Float,
      theta_eff: Float,
      theta_floor: Float,
      samples_used: Int,
      quality: MeasurementQuality,
      diagnostics: Diagnostics
    }
  | Unmeasurable {
      operation_ns: Float,          // Estimated operation time
      threshold_ns: Float,          // Timer resolution
      platform: String,
      recommendation: String
    }
  | Research {
      status: ResearchStatus,
      max_effect_ns: Float,         // Posterior mean of max|δ|
      max_effect_ci: (Float, Float), // 95% credible interval
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

- **Pass**: MUST be returned when `leak_probability < pass_threshold` (default 0.05) AND all verdict-blocking quality gates pass
- **Fail**: MUST be returned when `leak_probability > fail_threshold` (default 0.95) AND all verdict-blocking quality gates pass
- **Inconclusive**: MUST be returned when a verdict-blocking quality gate fails OR resource budgets are exhausted without reaching a decision threshold
- **Unmeasurable**: MUST be returned when the operation is too fast to measure reliably (see §4.5)
- **Research**: MUST be returned only in research mode (θ_user = 0)

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
| RemoteNetwork | 50 μs | Internet-exposed services |
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
  | UniformShift   // μ significant, τ ≈ 0
  | TailEffect     // μ ≈ 0, τ significant
  | Mixed          // Both significant
  | Complex        // Projection mismatch high
  | Indeterminate  // Neither significant

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
| StandardRemote | 100 ns – 10 μs | ~1k-10k queries with standard timing |
| ObviousLeak | > 10 μs | <100 queries, trivially observable |

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
  | ThresholdUnachievable { theta_user: Float, best_achievable: Float, message: String, guidance: String }
  | TimeBudgetExceeded { current_probability: Float, samples_collected: Int }
  | SampleBudgetExceeded { current_probability: Float, samples_collected: Int }
  | ConditionsChanged { drift: ConditionDrift }
```

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
  platform: String
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
```

---

## 3. Statistical Methodology

This section describes the mathematical foundation of timing-oracle. All formulas in this section are normative—implementations MUST produce equivalent results.

### 3.1 Test Statistic: Quantile Differences

We collect timing samples from two classes:
- **Fixed class (F)**: A specific input (e.g., all zeros)
- **Random class (R)**: Randomly sampled inputs

Rather than comparing means, we compare the distributions via their deciles. For each class, compute the 10th, 20th, ..., 90th percentiles, yielding two vectors in ℝ⁹. The test statistic is their difference:

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
│   • Estimate Σ_rate    • Collect batch    • Pass (P<5%)         │
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
- Compute "covariance rate" Σ_rate that scales as Σ = Σ_rate / n
- Compute initial measurement floor θ_floor and floor-rate constant c_floor
- Compute projection mismatch threshold Q_proj,thresh from bootstrap distribution
- Compute effective threshold θ_eff and calibrate prior scale
- Run pre-flight checks (timer sanity, harness sanity, stationarity)

**Phase 2: Adaptive Loop** (iterates until decision)
- Collect batches of samples
- Update quantile estimates from all data collected so far
- Scale covariance: Σ_n = Σ_rate / n
- Update θ_floor(n) using floor-rate constant
- Compute posterior and P(effect > θ_eff)
- Check quality gates (posterior ≈ prior → Inconclusive)
- Check decision boundaries (P > 95% → Fail, P < 5% → Pass)
- Check time/sample budgets

**Why this structure?**

The key insight is that covariance scales as 1/n for quantile estimators. By estimating the covariance *rate* once during calibration, we can cheaply update the posterior as more data arrives—no re-bootstrapping needed. This makes adaptive sampling computationally tractable.

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

**Critical principle:** The acquisition stream is the data-generating process. All bootstrap and dependence estimation MUST preserve adjacency in acquisition order, not per-class position. The underlying stochastic process—including drift, frequency scaling, and cache state evolution—operates in continuous time indexed by $t$. Splitting by label and treating each class as an independent time series is statistically incorrect.

#### 3.3.2 Covariance Estimation via Stream-Based Block Bootstrap

Timing measurements exhibit autocorrelation—nearby samples are more similar than distant ones due to cache state, frequency scaling, etc. Standard bootstrap assumes i.i.d. samples, underestimating variance. Implementations MUST use block bootstrap on the acquisition stream to preserve the true dependence structure.

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

This measures within-class dependence at acquisition-stream lags—the quantity that actually drives Var(Δ)—without being masked by class alternation.

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

Compute sample covariance of the Δ* vectors using Welford's online algorithm³ (numerically stable).

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

During the adaptive loop, with $n$ total samples per class:

$$
\Sigma_n = \Sigma_{\text{rate}} / n
$$

This allows cheap posterior updates without re-bootstrapping.

**Numerical stability:**

With discrete timers or few unique values, some quantiles may have near-zero variance, making Σ ill-conditioned. Implementations MUST regularize by ensuring a minimum diagonal value:

$$
\sigma^2_i \leftarrow \max(\sigma^2_i, 0.01 \cdot \bar{\sigma}^2) + \varepsilon
$$

where $\bar{\sigma}^2 = \text{tr}(\Sigma)/9$ and $\varepsilon = 10^{-10} + \bar{\sigma}^2 \cdot 10^{-8}$.

#### 3.3.3 Projection Mismatch Threshold Calibration

During bootstrap, calibrate the threshold for the projection mismatch diagnostic (see §3.5.3). This determines when the 2D shift+tail summary is a faithful description of the inferred 9D quantile profile.

For each bootstrap replicate Δ*:
1. Compute 9D posterior mean: δ*_post under the Bayesian update
2. Compute GLS projection: β*_proj = A δ*_post
3. Compute projection residual: r*_proj = δ*_post − Xβ*_proj
4. Compute mismatch statistic: Q*_proj = (r*_proj)ᵀ Σ_n⁻¹ (r*_proj)

The projection mismatch threshold is the 99th percentile of the bootstrap distribution:

$$
Q_{\text{proj,thresh}} = q_{0.99}(\{Q^*_{\text{proj},b}\}_{b=1}^B)
$$

**Note:** This threshold is used for the non-blocking projection mismatch diagnostic, not for verdict gating. The threshold is computed using the prior Λ₀ defined in §3.3.5; if the prior changes, Q_proj,thresh must be recomputed.

#### 3.3.4 Measurement Floor and Effective Threshold

A critical design element is distinguishing between what the user *wants* to detect (θ_user) and what the measurement *can* detect (θ_floor).

**Floor-rate constant:**

Under the model's scaling assumption Σ_n = Σ_rate/n, the measurement floor scales as 1/√n. Implementations MUST compute a floor-rate constant once at calibration based on the 9D decision functional m(δ) = max_k |δ_k|.

Draw Z₀ ~ N(0, Σ_rate) via Monte Carlo (SHOULD use 50,000 samples) and compute:

$$
c_{\text{floor}} := q_{0.95}\left( \max_k |Z_{0,k}| \right)
$$

This is the 95th percentile of the max absolute value under unit sample size.

**Measurement floor (dynamic):**

During the adaptive loop with $n$ samples per class:

$$
\theta_{\text{floor,stat}}(n) = \frac{c_{\text{floor}}}{\sqrt{n}}
$$

The tick floor is fixed once batching is determined:

$$
\theta_{\text{tick}} = \frac{\text{1 tick (ns)}}{K}
$$

where $K$ is the batch size.

The combined floor:

$$
\theta_{\text{floor}}(n) = \max\bigl(\theta_{\text{floor,stat}}(n), \, \theta_{\text{tick}}\bigr)
$$

**Effective threshold (θ_eff):**

The threshold actually used for inference:

$$
\theta_{\text{eff}} = \begin{cases}
\max(\theta_{\text{user}}, \theta_{\text{floor}}) & \text{if } \theta_{\text{user}} > 0 \\
\theta_{\text{floor}} & \text{if } \theta_{\text{user}} = 0 \text{ (Research mode)}
\end{cases}
$$

**Threshold elevation warning:**

When θ_eff > θ_user, implementations MUST emit a quality issue indicating that the requested threshold could not be achieved and reporting the effective threshold used.

**Dynamic floor updates:**

During the adaptive loop, θ_floor(n) is recomputed analytically as $n$ grows (no Monte Carlo needed). The policy:

1. If θ_floor(n) drops below θ_user during sampling, update θ_eff = θ_user (the user's threshold becomes achievable)
2. Track whether you're on pace to reach θ_user; if extrapolation shows you won't make it within budget, consider early termination with `Inconclusive`

#### 3.3.5 Prior Scale Calibration

The prior on the 9D effect profile δ should represent genuine uncertainty about whether effects exceed θ_eff—neither near-certain leak nor near-certain safety.

**Prior structure (correlation-shaped):**

The prior covariance uses the correlation structure of the empirical covariance, ensuring uniform marginal prior variance across all coordinates:

$$
R := \text{Corr}(\Sigma_{\text{rate}}) = D^{-1/2} \Sigma_{\text{rate}} D^{-1/2}, \quad D = \text{diag}(\Sigma_{\text{rate}})
$$

where D is computed *after* any diagonal-floor regularization of Σ_rate (ensuring strictly positive entries).

Then:

$$
\Lambda_0 = \sigma_{\text{prior}}^2 \cdot R
$$

Since diag(R) = 1, the parameter σ_prior is the marginal prior SD for each coordinate. This eliminates hidden heteroskedasticity across quantiles that could cause pathological shrinkage for certain effect patterns.

**Robustness fallback:**

In fragile regimes (base conditions in §3.3, or condition number of R exceeding 10⁴, or Cholesky failure), apply shrinkage toward identity:

$$
R_\lambda \leftarrow (1-\lambda)R + \lambda I, \quad \lambda \in [0.01, 0.2]
$$

chosen deterministically from conditioning diagnostics. This regularizes unreliable off-diagonal structure while preserving the unit-diagonal property.

**Numerical conditioning:**

After robustness shrinkage (if applied), implementations MUST ensure R is numerically SPD by adding jitter:

$$
R \leftarrow R + \varepsilon I, \quad \varepsilon \in [10^{-10}, 10^{-6}]
$$

chosen adaptively (e.g., increase ε on Cholesky failure). This is a purely numerical fix, applied as the last step before use.

**Numerical calibration of σ_prior:**

Choose σ_prior so that the prior exceedance probability equals a fixed target π₀ (default 0.62):

$$
P\left(\max_k |\delta_k| > \theta_{\text{eff}} \;\middle|\; \delta \sim \mathcal{N}(0, \Lambda_0)\right) = \pi_0
$$

Given Λ₀ = σ²_prior R, this is a 1D root-find on σ_prior using deterministic Monte Carlo (SHOULD use 50k draws). The target π₀ = 0.62 represents genuine uncertainty about whether effects exceed the effective threshold.

**Calibration search bounds:**

The binary search for σ_prior MUST use bounds that accommodate both threshold-scale effects and measurement-noise-scale effects:

- `lo = θ_eff × 0.05`
- `hi = max(θ_eff × 50, 10 × median_k(SE_k))`

where:

$$
\text{SE}_k := \sqrt{(\Sigma_{\text{rate}} / n_{\text{cal}})_{kk}}
$$

is the standard error at calibration sample size, and the median is taken over k ∈ {1, ..., 9}.

This ensures the prior family can represent effects at the scale the data actually exhibits, preventing pathological shrinkage when θ_eff ≪ measurement noise.

**Normative requirements:**

- The RNG seed MUST be deterministic (per §3.3.6)
- The calibration MUST be performed once per run, during calibration
- The prior MUST remain fixed throughout the adaptive loop. If θ_eff changes as n grows, it affects only exceedance checks, not the prior.

#### 3.3.6 Deterministic Seeding Policy

To ensure reproducible results, all random number generation MUST be deterministic by default.

**Normative requirement:**

> Given identical timing samples and configuration, the oracle MUST produce identical results (up to floating-point roundoff). Algorithmic randomness is not epistemic uncertainty about the leak—it is approximation error that MUST NOT influence the decision rule.

**Seeding policy:**

- The bootstrap RNG seed MUST be deterministically derived from:
  - A fixed library constant seed (default: 0x74696D696E67, "timing" in ASCII)
  - A stable hash of configuration parameters

- All Monte Carlo RNG seeds (leak probability, floor constant, prior scale) MUST be similarly deterministic

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

We use a conjugate Gaussian model over the full 9D quantile-difference profile that admits closed-form posteriors—no MCMC required.

#### 3.4.1 Latent Parameter

The latent parameter is the true per-decile timing difference profile:

$$
\delta \in \mathbb{R}^9 \quad \text{(true decile-difference profile)}
$$

This is unconstrained—the model can represent any quantile-difference pattern, unlike a low-rank projection.

#### 3.4.2 Likelihood

$$
\Delta \mid \delta \sim \mathcal{N}(\delta, \Sigma_n)
$$

where Σ_n = Σ_rate / n is the scaled covariance. This is the minimal mean-structure assumption: the covariance is estimated; the mean is unconstrained.

#### 3.4.3 Prior

$$
\delta \sim \mathcal{N}(0, \Lambda_0), \quad \Lambda_0 = \sigma_{\text{prior}}^2 \cdot R
$$

where R = Corr(Σ_rate) is the correlation matrix and σ_prior is calibrated numerically (see §3.3.5).

Since diag(R) = 1, the marginal prior SD for each coordinate equals σ_prior. This ensures uniform shrinkage across quantiles, preventing pathological behavior when the effect pattern differs from the covariance structure.

#### 3.4.4 Posterior

With Gaussian likelihood and Gaussian prior, the posterior is also Gaussian⁴:

$$
\delta \mid \Delta \sim \mathcal{N}(\delta_{\text{post}}, \Lambda_{\text{post}})
$$

where:

$$
\Lambda_{\text{post}} = \left( \Lambda_0^{-1} + \Sigma_n^{-1} \right)^{-1}
$$

$$
\delta_{\text{post}} = \Lambda_{\text{post}} \Sigma_n^{-1} \Delta
$$

The posterior mean δ_post ∈ ℝ⁹ gives the estimated timing difference at each decile in nanoseconds.

**Implementation guidance (normative):**

Implementations MUST NOT form explicit matrix inverses in floating point. Use Cholesky solves on SPD matrices. A stable compute path:

```
// Form A := Σ_n + Λ_0 (SPD)
// Solve A x = Δ for x
// Then δ_post = Λ_0 x
// And Λ_post = Λ_0 - Λ_0 A⁻¹ Λ_0 (via solves)
```

This avoids Σ_n⁻¹ explicitly.

⁴ Bishop, C. M. (2006). Pattern Recognition and Machine Learning, §3.3. Springer.

#### 3.4.5 Decision Functional and Leak Probability

The decision is based on the maximum absolute effect across all quantiles:

$$
m(\delta) := \max_k |\delta_k|
$$

The leak probability is:

$$
P(\text{leak} > \theta_{\text{eff}} \mid \Delta) = P\bigl(m(\delta) > \theta_{\text{eff}} \;\big|\; \Delta\bigr)
$$

Since the posterior is Gaussian, implementations SHOULD compute this by Monte Carlo integration (SHOULD use 10,000 samples with deterministic seed per §3.3.6).

**Interpreting the probability:**

This is a **posterior probability**, not a p-value. When we report "72% probability of a leak," we mean: given the data and our model, 72% of the posterior mass corresponds to effects exceeding θ_eff.

**Note on θ_eff:**

All inference uses θ_eff, not θ_user. When these differ, the output MUST clearly report both values so the user understands what was actually tested.

#### 3.4.6 2D Projection for Reporting

The 9D posterior is used for decisions, but we provide a 2D summary for interpretability when the pattern is simple.

**Design matrix (for projection only):**

$$
X = \begin{bmatrix} \mathbf{1} & \mathbf{b}_{\text{tail}} \end{bmatrix} \in \mathbb{R}^{9 \times 2}
$$

where:
- **1** = (1, 1, 1, 1, 1, 1, 1, 1, 1)ᵀ — uniform shift
- **b_tail** = (−0.5, −0.375, −0.25, −0.125, 0, 0.125, 0.25, 0.375, 0.5)ᵀ — tail effect

**GLS projection operator:**

$$
A := (X^\top \Sigma_n^{-1} X)^{-1} X^\top \Sigma_n^{-1}
$$

**Posterior for projection summary:**

Since β_proj = Aδ is linear in δ, under the 9D posterior:

- Mean: β_proj,post = A δ_post
- Covariance: Cov(β_proj | Δ) = A Λ_post Aᵀ

The projection gives interpretable components:
- **Shift (μ)**: Uniform timing difference affecting all quantiles equally
- **Tail (τ)**: Upper quantiles affected more than lower (or vice versa)

**Important:** The 2D projection is for reporting only. Decisions MUST be based on the 9D posterior. When the projection doesn't fit well (see §3.5.3), implementations MUST add an interpretation caveat and provide alternative explanations.

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

Bayesian methods don't have this problem. The posterior probability is valid regardless of when you stop—this is the **likelihood principle**. Your inference depends only on the data observed, not your sampling plan.

#### 3.5.2 Quality Gates (Verdict-Blocking)

Quality gates detect when data is too poor to reach a confident decision. When any gate triggers, the outcome MUST be Inconclusive.

**Gate 1: Posterior ≈ Prior (Data Too Noisy)**

If measurement uncertainty is high, the posterior barely moves from the prior. For the 9D model, use log-det ratio:

$$
\rho := \log \frac{|\Lambda_{\text{post}}|}{|\Lambda_0|}
$$

If ρ > log(0.5) (posterior volume not reduced much), trigger Inconclusive with reason `DataTooNoisy`.

Alternative: use trace ratio tr(Λ_post)/tr(Λ₀) > 0.5 if log-det is numerically unstable.

**Exception for decisive probabilities**: If P(leak) > 0.995 or P(leak) < 0.005, AND the projection mismatch Q < 1000 × Q_thresh, implementations MAY bypass this gate and allow a verdict. This handles slow operations (e.g., modular exponentiation) or unusual timing patterns (e.g., branch-dependent timing) where high autocorrelation limits effective sample size, but the effect is so large that the verdict is unambiguous.

Moderate Q values (e.g., Q ~ 100–1000) can occur with real timing leaks that have unusual patterns. The Q check filters only truly pathological cases: astronomical Q values (e.g., Q > 10000 × Q_thresh) indicate measurement artifacts rather than genuine timing differences. Such cases MUST still return Inconclusive.

**Gate 2: Learning Rate Collapsed**

Track KL divergence between successive posteriors. For Gaussian posteriors:

$$
\text{KL}(p_{\text{new}} \| p_{\text{old}}) = \frac{1}{2}\left( \text{tr}(\Lambda_{\text{old}}^{-1} \Lambda_{\text{new}}) + (\mu_{\text{old}} - \mu_{\text{new}})^\top \Lambda_{\text{old}}^{-1} (\mu_{\text{old}} - \mu_{\text{new}}) - k + \ln\frac{|\Lambda_{\text{old}}|}{|\Lambda_{\text{new}}|} \right)
$$

where k = 9 for the 9D model.

If the sum of recent KL divergences (e.g., over last 5 batches) falls below 0.001, trigger Inconclusive with reason `NotLearning`.

**Gate 3: Would Take Too Long**

Extrapolate time to decision based on current convergence rate. If projected time exceeds budget by a large margin (e.g., 10×), trigger Inconclusive with reason `WouldTakeTooLong`.

**Gate 4: Threshold Unachievable**

If even at maximum budget, we cannot reach θ_user:

$$
\theta_{\text{floor,max}} = \max\left(\frac{c_{\text{floor}}}{\sqrt{n_{\max}}}, \theta_{\text{tick}}\right)
$$

If θ_floor,max > θ_user and θ_user > 0, trigger Inconclusive with reason `ThresholdUnachievable`.

**Gate 5: Time Budget Exceeded**

If elapsed time exceeds configured time budget, trigger Inconclusive with reason `TimeBudgetExceeded`.

**Gate 6: Sample Budget Exceeded**

If total samples per class exceeds configured maximum, trigger Inconclusive with reason `SampleBudgetExceeded`.

**Gate 7: Condition Drift Detected**

The covariance estimate Σ_rate is computed during calibration. If measurement conditions change during the adaptive loop, this estimate becomes invalid.

Detect condition drift by comparing measurement statistics from calibration against the full test run:

- Variance ratio: σ²_post / σ²_cal
- Autocorrelation change: |ρ_post(1) − ρ_cal(1)|
- Mean drift: |μ_post − μ_cal| / σ_cal

If variance ratio is outside [0.5, 2.0], or autocorrelation change exceeds 0.3, or mean drift exceeds 3.0, trigger Inconclusive with reason `ConditionsChanged`.

#### 3.5.3 Non-Blocking Diagnostics

**Projection Mismatch Diagnostic**

Under the 9D model, there is no risk of "mean model cannot represent the observed shape"—the 9D posterior can represent any quantile pattern. The projection mismatch diagnostic measures whether the 2D shift+tail summary faithfully describes the inferred 9D shape.

**Projection mismatch statistic:**

$$
r_{\text{proj}} := \delta_{\text{post}} - X\beta_{\text{proj,post}}
$$

$$
Q_{\text{proj}} := r_{\text{proj}}^\top \Sigma_n^{-1} r_{\text{proj}}
$$

**Semantics (normative):**

- Projection mismatch MUST NOT be verdict-blocking
- If Q_proj > Q_proj,thresh, implementations MUST:
  - Set `effect.interpretation_caveat`
  - Report a "top quantiles" shape hint (see below)
  - Mark `Diagnostics.projection_mismatch_ok = false`

**Top quantiles shape hint:**

When projection mismatch is high, report the 2–3 most relevant quantiles ranked by per-quantile exceedance probability:

$$
p_k := P(|\delta_k| > \theta_{\text{eff}} \mid \Delta)
$$

Since the posterior is Gaussian, the marginal δ_k | Δ ~ N(μ_k, s²_k) where:

$$
\mu_k = (\delta_{\text{post}})_k, \quad s_k^2 = (\Lambda_{\text{post}})_{kk}
$$

Compute:

$$
p_k = 1 - \left[\Phi\left(\frac{\theta_{\text{eff}} - \mu_k}{s_k}\right) - \Phi\left(\frac{-\theta_{\text{eff}} - \mu_k}{s_k}\right)\right]
$$

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
1. Sets θ_user = 0, so θ_eff = θ_floor
2. Reports posterior credible intervals instead of exceedance probabilities
3. Returns a `Research` outcome instead of Pass/Fail

The decision functional is the same:

$$
m(\delta) = \max_k |\delta_k|
$$

#### 3.6.3 Stopping Conditions

| Condition | Criterion | Status |
|-----------|-----------|--------|
| Effect detected | CI lower bound > 1.1 × θ_floor | `EffectDetected` |
| No effect detected | CI upper bound < 0.9 × θ_floor | `NoEffectDetected` |
| Resolution limit | θ_floor ≈ θ_tick | `ResolutionLimitReached` |
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
- Δ, θ, effect sizes all in ticks
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
R_\lambda \leftarrow (1-\lambda)R + \lambda I, \quad \lambda \in [0.01, 0.2]
$$

**Gaussian approximation caveat:**

The Gaussian likelihood is a rougher approximation with discrete data. Implementations MUST report a quality issue about discrete timer mode and frame probabilities as "approximate under Gaussianized model."

### 3.8 Calibration Validation Requirements

The Bayesian approach requires empirical validation that posteriors are well-calibrated. This section defines **normative, pipeline-level** calibration requirements.

**Null calibration test (normative requirement):**

Implementations MUST provide (or run in internal test suite) a "fixed-vs-fixed" validation that measures **end-to-end false positive rates**:

- Run the full pipeline under null (same distribution both classes)
- Compute two distinct FPR metrics:
  - FPR_overall = P(Fail | H₀) — unconditional false positive rate
  - FPR_gated = P(Fail | H₀, all verdict-blocking gates pass) — FPR conditional on conclusive results
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

| True Effect | Expected P(leak > θ) | Acceptable Range |
|-------------|---------------------|------------------|
| 0 | ~2-5% | 0-10% |
| θ/2 | ~5-15% | 0-25% |
| θ | ~50% | 35-65% |
| 2θ | ~95% | 85-100% |
| 3θ | ~99% | 95-100% |

This is an end-to-end check that c_floor, Σ scaling, block length selection, prior calibration, and posterior computation are not systematically anti-conservative.

**Large-effect detection test (normative requirement):**

Implementations MUST include a validation test for the "large effect + noisy likelihood" regime. Using synthetic or reference data with:
- True effect ≫ θ_eff (e.g., 100× threshold)
- Data SEs ≫ θ_eff (e.g., 30× threshold)

Assert that P(leak) > 0.99 at the reference θ_eff. This test validates the inference layer directly (the probability, not the verdict), preventing prior shape pathologies that suppress detection of obvious leaks regardless of quality gate behavior.

The test SHOULD use fixed Σ_n, Σ_rate, and Δ values consistent with the regime, ensuring determinism independent of bootstrap randomness.

---

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

---

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
- Set time budget to 30 seconds
- Set sample budget to 100,000
- Set pass/fail thresholds to 0.05/0.95

### 5.3 Result Communication

**Leak probability prominence:**

The leak probability MUST be prominently displayed in results and human-readable output.

**Threshold transparency:**

When θ_eff > θ_user, implementations MUST clearly indicate this to the user.

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

---

## 6. Configuration Parameters

### 6.1 Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| attacker_model OR threshold_ns | AttackerModel or Float | Defines the effect threshold |

### 6.2 Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| time_budget | 30 seconds | Maximum test duration |
| max_samples | 100,000 | Maximum samples per class |
| pass_threshold | 0.05 | P(leak) below this → Pass |
| fail_threshold | 0.95 | P(leak) above this → Fail |
| calibration_samples | 5,000 | Samples for calibration phase |
| batch_size | 1,000 | Samples per adaptive batch |
| bootstrap_iterations | 2,000 | Bootstrap iterations for covariance |

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| {(c_t, y_t)} | Acquisition stream: class labels and timing measurements |
| T | Acquisition stream length (≈ 2n) |
| F, R | Per-class sample sets (filtered from stream) |
| Δ | 9-vector of observed quantile differences |
| δ | 9-vector of true (latent) quantile differences |
| q̂_F(k), q̂_R(k) | Empirical quantiles for Fixed and Random classes |
| X | 9 × 2 projection basis [**1** ∣ **b**_tail] |
| β_proj | 2D projection: (μ, τ)ᵀ |
| Σ_n | Covariance matrix at sample size n |
| Σ_rate | Covariance rate: Σ_n = Σ_rate / n |
| Λ₀ | Prior covariance: σ²_prior · R |
| Λ_post | Posterior covariance for δ |
| R | Prior correlation matrix: Corr(Σ_rate) |
| θ_user | User-requested threshold |
| θ_floor | Measurement floor (smallest resolvable effect) |
| c_floor | Floor-rate constant: θ_floor,stat = c_floor / √n |
| θ_tick | Timer resolution component of floor |
| θ_eff | Effective threshold used for inference |
| m(δ) | Decision functional: max_k \|δ_k\| |
| Q_proj | Projection mismatch statistic |
| Q_proj,thresh | Projection mismatch threshold (99th percentile) |
| b̂ | Block length (Politis-White, on acquisition stream) |
| MDE | Minimum detectable effect |
| n | Samples per class |
| B | Bootstrap iterations |
| ESS | Effective sample size |

---

## Appendix B: Normative Constants

These constants define conformant implementations. Implementations MAY use different values only where noted.

| Constant | Default | Normative | Rationale |
|----------|---------|-----------|-----------|
| Deciles | {0.1, ..., 0.9} | MUST | Nine quantile positions |
| Bootstrap iterations | 2,000 | SHOULD | Covariance estimation accuracy |
| Monte Carlo samples (leak prob) | 10,000 | SHOULD | Leak probability integration |
| Monte Carlo samples (c_floor) | 50,000 | SHOULD | Floor-rate constant estimation |
| Batch size | 1,000 | SHOULD | Adaptive iteration granularity |
| Calibration samples | 5,000 | SHOULD | Initial covariance estimation |
| Pass threshold | 0.05 | SHOULD | 95% confidence of no leak |
| Fail threshold | 0.95 | SHOULD | 95% confidence of leak |
| Prior exceedance target | 0.62 | SHOULD | Genuine uncertainty |
| Variance ratio gate | 0.5 | SHOULD | Posterior ≈ prior detection |
| Projection mismatch percentile | 99th | SHOULD | Q_proj_thresh from bootstrap |
| Block length cap | min(3√T, T/3) | SHOULD | Prevent degenerate blocks |
| Discrete threshold | 10% unique | SHOULD | Trigger discrete mode |
| Min ticks per call | 5 | SHOULD | Measurability floor |
| Max batch size | 20 | SHOULD | Limit μarch artifacts |
| Default time budget | 30 s | MAY | Maximum runtime |
| Default sample budget | 100,000 | MAY | Maximum samples |
| CI hysteresis (Research) | 10% | SHOULD | Margin for detection decisions |
| Default RNG seed | 0x74696D696E67 | SHOULD | "timing" in ASCII |
| Prior robust shrinkage (λ) | 0.01–0.2 | SHOULD | Robustness for unreliable R |
| Prior numerical jitter (ε) | 10⁻¹⁰–10⁻⁶ | SHOULD | SPD conditioning |
| Condition number threshold | 10⁴ | SHOULD | Trigger robust shrinkage |

---

## Appendix C: References

**Statistical methodology:**

1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning, Ch. 3. Springer. — Bayesian linear regression

2. Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection for the Dependent Bootstrap." Econometric Reviews 23(1):53–70.

3. Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." Annals of Statistics. — Block bootstrap

4. Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages." The American Statistician 50(4):361–365.

5. Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419–420.

**Timing attacks:**

6. Reparaz, O., Balasch, J., & Verbauwhede, I. (2016). "Dude, is my code constant time?" DATE. — DudeCT methodology

7. Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009). "Opportunities and Limits of Remote Timing Attacks." ACM TISSEC 12(3):17. — Exploitability thresholds

8. Van Goethem, T., et al. (2020). "Timeless Timing Attacks." USENIX Security. — HTTP/2 timing attacks

9. Bernstein, D. J. et al. (2024). "KyberSlash." — Timing vulnerability example

10. Dunsche, M. et al. (2025). "SILENT: A New Lens on Statistics in Software Timing Side Channels." arXiv:2504.19821. — Relevant hypotheses framework

**Existing tools:**

11. dudect (C): https://github.com/oreparaz/dudect
12. dudect-bencher (Rust): https://github.com/rozbb/dudect-bencher

---

## Appendix D: Changelog

### v5.1 (from v5.0)

**Prior shape matrix change (§3.3.5, §3.4.3):**

- **Changed:** Prior covariance now uses the correlation matrix R = Corr(Σ_rate) instead of the trace-normalized shape matrix S = Σ_rate / tr(Σ_rate)
- **Rationale:** The trace-normalized shape matrix caused heteroskedastic marginal prior variances across coordinates. When the exceedance calibration target (62%) was satisfied by high-variance coordinates, low-variance coordinates could have pathologically tight priors, causing catastrophic shrinkage for uniform-shift effects in noisy-likelihood regimes.
- **Impact:** With R, diag(R) = 1, so σ_prior equals the marginal prior SD for all coordinates uniformly. This eliminates the failure mode where large effects (e.g., 13μs) were shrunk to near-zero when θ_eff was small (e.g., 100ns) and measurement noise was high.

**Prior regularization (§3.3.5):**

- **Changed:** Separated numerical jitter (ε ∈ [10⁻¹⁰, 10⁻⁶]) from robust shrinkage (λ ∈ [0.01, 0.2])
- **Rationale:** These serve different purposes—jitter ensures SPD for Cholesky, shrinkage handles unreliable covariance structure. The previous spec conflated them.
- **Order:** Apply robust shrinkage first (if in fragile regime), then numerical jitter.

**Calibration bounds (§3.3.5):**

- **Added:** Explicit search bounds for σ_prior calibration: lo = 0.05θ_eff, hi = max(50θ_eff, 10 × median(SE_k))
- **Rationale:** Prevents "prior family can't represent what data obviously contains" when θ_eff ≪ measurement noise.

**Large-effect validation test (§3.8):**

- **Added:** Normative requirement for a "large effect + noisy likelihood" regression test asserting P(leak) > 0.99
- **Rationale:** Prevents future regressions that reintroduce prior shape pathologies.

**Fragile regime unification (§3.3, §3.3.2, §3.3.5, §3.7):**

- **Added:** Base fragile regime definition in §3.3 with shared conditions (discrete timer mode, low uniqueness)
- **Changed:** §3.3.2 and §3.3.5 now reference the base definition and extend it with context-specific conditions
- **Changed:** §3.7 now references §3.3 instead of duplicating the uniqueness formula
- **Rationale:** The previous spec defined "fragile regime" independently in two places with overlapping but not identical conditions. Unifying the base definition clarifies the shared foundation while preserving context-specific extensions (high autocorrelation for block length, numerical conditioning for covariance).

**Notation updates (Appendix A, B):**

- **Removed:** S (shaped prior matrix)
- **Added:** R (prior correlation matrix)
- **Added:** Prior numerical jitter (ε), condition number threshold constants
