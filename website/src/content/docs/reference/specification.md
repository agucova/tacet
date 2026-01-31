---
title: Specification (v6.0)
description: Authoritative specification for tacet's statistical methodology and requirements
sidebar:
  order: 1
---

This document is the authoritative specification for tacet, a Bayesian timing side-channel detection system. It defines the statistical methodology, abstract types, and requirements that implementations MUST follow to be conformant.

For implementation details (algorithms, numerical procedures), see the [Implementation Guide](/reference/implementation-guide). For language-specific APIs, see the [Rust API](/api/rust), [C API](/api/c), or [Go API](/api/go). For interpreting results, see the [User Guide](/guides/user-guide).

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

tacet addresses these issues with:

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
```

**Semantics:**

The field `leak_probability` MUST be computed as $P(m(\delta) > \theta_{\text{eff}} \mid \Delta)$, where $\theta_{\text{eff}}$ is the effective threshold used for inference at decision time.

When $\theta_{\text{eff}}$ > $\theta_{\text{user}}$, the oracle cannot support a **Pass claim at $\theta_{\text{user}}$**, because effects in the range ($\theta_{\text{user}}$, $\theta_{\text{eff}}$] are not distinguishable from noise under the measured conditions.

Implementations MUST NOT substitute $\theta_{\text{user}}$ into `leak_probability` when $\theta_{\text{eff}}$ > $\theta_{\text{user}}$.

- **Pass**: MUST be returned when ALL of the following hold:
  1. `leak_probability < pass_threshold` (default 0.05)
  2. $\theta_{\text{eff}} \le \theta_{\text{user}} + \varepsilon_{\theta}$ where $\varepsilon_{\theta} = \max(\theta_{\text{tick}}, 10^{-6} \cdot \theta_{\text{user}})$
  3. All verdict-blocking quality gates pass
  4. $\theta_{\text{user}}$ > 0

- **Fail**: MUST be returned when ALL of the following hold:
  1. `leak_probability > fail_threshold` (default 0.95)
  2. All verdict-blocking quality gates pass

  Note: Fail MAY be returned when $\theta_{\text{eff}}$ > $\theta_{\text{user}}$. Detecting $m(\delta)$ > $\theta_{\text{eff}}$ implies $m(\delta)$ > $\theta_{\text{user}}$ (since $\theta_{\text{eff}}$ ≥ $\theta_{\text{user}}$ by construction).

- **Inconclusive**: MUST be returned when ANY of the following hold:
  1. A verdict-blocking quality gate fails
  2. Resource budgets exhausted without reaching a decision threshold
  3. `leak_probability < pass_threshold` but $\theta_{\text{eff}} > \theta_{\text{user}} + \varepsilon_{\theta}$ (threshold elevated)

- **Unmeasurable**: MUST be returned when the operation is too fast to measure reliably (see §4.5)

**Exploratory mode (θ_user = 0):** When $\theta_{\text{user}}$ = 0, Pass/Fail semantics do not apply. The oracle MUST return Inconclusive with the posterior estimates, allowing users to interpret the results themselves. This mode is useful for profiling and debugging but is not suitable for CI gating.

### 2.2 AttackerModel

Threat model presets defining the minimum effect size worth detecting:

```
AttackerModel =
  | SharedHardware      // θ = 0.4 ns (~2 cycles @ 5GHz)
  | PostQuantumSentinel // θ = 2.0 ns (~10 cycles @ 5GHz)
  | AdjacentNetwork     // θ = 100 ns
  | RemoteNetwork       // θ = 50,000 ns (50 μs)
  | Custom { threshold_ns: Float }
```

| Model | Threshold | Use Case |
|-------|-----------|----------|
| SharedHardware | 0.4 ns | SGX enclaves, cross-VM, containers, hyperthreading |
| PostQuantumSentinel | 2.0 ns | Post-quantum crypto (ML-KEM, ML-DSA) |
| AdjacentNetwork | 100 ns | LAN services, HTTP/2 APIs |
| RemoteNetwork | 50 μs | Internet-exposed services |

Cycle-based thresholds use a conservative 5 GHz reference frequency (assumes fast attacker hardware—smaller θ = more sensitive = safer).

**There is no single correct threshold.** The choice of attacker model is a statement about your threat model.

For exploratory analysis without a decision threshold, use `Custom { threshold_ns: 0.0 }`.

### 2.3 EffectEstimate

Summary of the detected timing effect:

```
EffectEstimate = {
  max_effect_ns: Float,                 // Posterior mean of max_k |δ_k|
  credible_interval_ns: (Float, Float), // 95% CI for max|δ|
  top_quantiles: List<TopQuantile>      // Top 2-3 quantiles by exceedance probability
}

TopQuantile = {
  quantile_p: Float,       // e.g., 0.9 for 90th percentile
  mean_ns: Float,          // Posterior mean δ_k
  ci95_ns: (Float, Float), // 95% marginal CI
  exceed_prob: Float       // P(|δ_k| > θ_eff | data)
}
```

The `top_quantiles` field identifies which parts of the distribution show the largest effects, helping users understand whether the leak is a uniform shift (all quantiles affected) or a tail effect (upper quantiles affected more).

### 2.4 MeasurementQuality

Assessment of measurement precision:

```
MeasurementQuality =
  | Excellent  // MDE < 5 ns
  | Good       // MDE 5–20 ns
  | Poor       // MDE 20–100 ns
  | TooNoisy   // MDE > 100 ns
```

### 2.5 InconclusiveReason

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

### 2.6 Diagnostics

```
Diagnostics = {
  // Dependence and effective samples
  dependence_length: Int,           // b̂ from Politis-White (bootstrap only)
  effective_sample_size: Int,       // n / τ̂
  iact_combined: Float,             // max(τ̂_F, τ̂_R)

  // Stationarity
  stationarity_ratio: Float,
  stationarity_ok: Bool,

  // Outlier handling
  outlier_rate_baseline: Float,
  outlier_rate_sample: Float,
  outlier_asymmetry_ok: Bool,

  // Timer and mode
  discrete_mode: Bool,
  timer_resolution_ns: Float,
  duplicate_fraction: Float,

  // Run information
  preflight_ok: Bool,
  calibration_samples: Int,
  total_time_secs: Float,
  seed: Option<Int>,
  threshold_ns: Float,
  timer_name: String,
  platform: String,

  // Gibbs sampler
  gibbs_iters_total: Int,
  gibbs_burnin: Int,
  gibbs_retained: Int,
  lambda_mean: Float,               // Posterior mean of λ (prior precision)
  lambda_mixing_ok: Bool,

  // Robust likelihood
  likelihood_inflated: Bool,        // True if κ_mean < 0.3

  // Warnings
  warnings: List<String>,
  quality_issues: List<QualityIssue>
}

QualityIssue = {
  code: IssueCode,
  message: String,
  guidance: String
}

IssueCode =
  | DependenceHigh       // High autocorrelation, reduced effective samples
  | PrecisionLow         // Limited measurement precision
  | DiscreteMode         // Coarse timer resolution
  | ThresholdIssue       // Cannot achieve requested threshold
  | FilteringApplied     // Outliers were capped
  | StationarityIssue    // Conditions may have changed
  | NumericalIssue       // Gibbs sampler convergence concern
  | LikelihoodInflated   // Uncertainty inflated for robustness
```

***

## 3. Statistical Methodology

This section describes the mathematical foundation of tacet. All formulas in this section are normative; implementations MUST produce equivalent results.

### 3.1 Test Statistic: Quantile Differences

We collect timing samples from two classes:
- **Fixed class (F)**: A specific input (e.g., all zeros)
- **Random class (R)**: Randomly sampled inputs

Rather than comparing means, we compare the distributions via their deciles. For each class, compute the 10th, 20th, ..., 90th percentiles, yielding two vectors in $\mathbb{R}^d$. The test statistic is their difference:

$$
\Delta = \hat{q}(\text{Fixed}) - \hat{q}(\text{Random}) \in \mathbb{R}^d
$$

where $\hat{q}_p$ denotes the empirical $p$-th quantile.

#### 3.1.1 Dimension Parameterization

The statistical engine generalizes to arbitrary dimension $d$. For **timing analysis**, $d = 9$ (decile differences).

The decision functional is:

$$
m(\delta) = \max_k |\delta_k|
$$

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

**Quantile computation:** Implementations SHOULD use **type 2** quantiles (inverse empirical CDF with averaging). For discrete timers with many tied values, use **mid-distribution quantiles**. See the [Implementation Guide](/reference/implementation-guide) for algorithms.

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
│   • Set prior scale    • Scale Σ by 1/n   • Inconclusive        │
│   • Warmup caches      • Update θ_floor                         │
│   • Pre-flight checks  • Compute P(>θ)                          │
│                        • Check quality                          │
│                        • Check stopping                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Phase 1: Calibration** (runs once)
- Collect initial samples to characterize measurement noise
- Estimate covariance structure via stream-based block bootstrap
- Compute "covariance rate" $\Sigma_{\text{rate}}$ that scales as $\Sigma = \Sigma_{\text{rate}} / n$
- Compute initial measurement floor $\theta_{\text{floor}}$ and floor-rate constant c_floor
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

**Fragile regime:**

A **fragile regime** is a measurement condition where standard statistical assumptions may not hold, requiring more conservative estimation. The fragile regime is triggered when either:

- **Discrete timer mode**: The timer has coarse resolution (see §3.6)
- **Low uniqueness**: The minimum uniqueness ratio across classes is below 10%:
  $$\min\left(\frac{|\text{unique}(F)|}{n_F}, \frac{|\text{unique}(R)|}{n_R}\right) < 0.10$$

When a fragile regime is detected, implementations apply more conservative procedures (larger block lengths, regularized covariance) to maintain calibration.

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

**Critical principle:** The acquisition stream is the data-generating process. All bootstrap and dependence estimation MUST preserve adjacency in acquisition order, not per-class position.

#### 3.3.2 Covariance Estimation via Stream-Based Block Bootstrap

Timing measurements exhibit autocorrelation: nearby samples are more similar than distant ones due to cache state, frequency scaling, etc. Standard bootstrap assumes i.i.d. samples, underestimating variance. Implementations MUST use block bootstrap on the acquisition stream to preserve the true dependence structure.

**Block length selection:** Implementations SHOULD use the Politis-White algorithm to select the optimal block length, with class-conditional ACF to avoid underestimation from interleaved sampling. See the [Implementation Guide](/reference/implementation-guide#22-politis-white-block-length-selection) for the algorithm.

**Covariance rate:**

The covariance of quantile estimators scales as 1/n. We compute the "rate":

$$
\Sigma_{\text{rate}} = \hat{\Sigma}_{\text{cal}} \cdot n_{\text{cal}}
$$

Define the **calibration long-run variance proxy**:

$$
V_{\text{cal}} := \Sigma_{\text{cal}} \cdot n_{\text{cal}}
$$

**Effective sample size via IACT:**

Under strong temporal dependence, n samples do not provide n independent observations. Implementations MUST define effective sample size using integrated autocorrelation time (IACT):

$$
n_{\text{eff}} := \frac{n}{\hat{\tau}}
$$

where $\hat{\tau}$ is estimated via the Geyer Initial Monotone Sequence (IMS) method. See the [Implementation Guide](/reference/implementation-guide#3-iact-computation) for the algorithm.

Implementations MUST report:
- `Diagnostics.dependence_length` = $\hat{b}$ (for bootstrap)
- `Diagnostics.iact_combined` = $\hat{\tau}$
- `Diagnostics.effective_sample_size` = $n_{\text{eff}}$

**Covariance scaling during adaptive loop:**

During inference with $n$ samples and IACT estimates $\hat{\tau}$ (current) and $\hat{\tau}_{\text{cal}}$ (at calibration):

$$
\Sigma_n = V_{\text{cal}} \cdot \frac{\hat{\tau}}{\hat{\tau}_{\text{cal}}} \cdot \frac{1}{n}
$$

**Numerical stability:**

With discrete timers or few unique values, some quantiles may have near-zero variance, making $\Sigma$ ill-conditioned. Implementations MUST apply diagonal floor regularization. See the [Implementation Guide](/reference/implementation-guide#44-diagonal-floor-regularization).

#### 3.3.3 Measurement Floor and Effective Threshold

A critical design element is distinguishing between what the user *wants* to detect ($\theta_{\text{user}}$) and what the measurement *can* detect ($\theta_{\text{floor}}$).

**Floor-rate constant:**

Implementations MUST compute a floor-rate constant once at calibration:

$$
c_{\text{floor}} := q_{0.95}\left( \max_k |Z_{0,k}| \right), \quad Z_0 \sim \mathcal{N}(0, \Sigma_{\text{rate}})
$$

via Monte Carlo (SHOULD use 50,000 samples).

**Measurement floor (dynamic):**

During the adaptive loop:

$$
\theta_{\text{floor,stat}}(n) = \frac{c_{\text{floor}}}{\sqrt{n_{\text{eff}}}}
$$

The tick floor is fixed once batching is determined:

$$
\theta_{\text{tick}} = \frac{\text{1 tick (ns)}}{K}
$$

where $K$ is the batch size. The combined floor:

$$
\theta_{\text{floor}}(n) = \max\bigl(\theta_{\text{floor,stat}}(n), \, \theta_{\text{tick}}\bigr)
$$

**Effective threshold ($\theta_{\text{eff}}$):**

$$
\theta_{\text{eff}} = \max(\theta_{\text{user}}, \theta_{\text{floor}})
$$

**Threshold elevation decision rule:**

When $\theta_{\text{eff}}$ > $\theta_{\text{user}}$:

1. **Fail propagates**: Detecting $m(\delta)$ > $\theta_{\text{eff}}$ implies $m(\delta)$ > $\theta_{\text{user}}$. Implementations MAY return Fail.

2. **Pass does not propagate**: "No detectable effect above $\theta_{\text{eff}}$" is compatible with effects in ($\theta_{\text{user}}$, $\theta_{\text{eff}}$]. Implementations MUST NOT return Pass when $\theta_{\text{eff}}$ > $\theta_{\text{user}}$ + $\varepsilon_{\theta}$.

**Dynamic floor updates:**

During the adaptive loop, $\theta_{\text{floor}}(n)$ decreases as n grows. Implementations MUST:

1. Recompute $\theta_{\text{eff}} = \max(\theta_{\text{user}}, \theta_{\text{floor}}(n))$ after each batch
2. If $\theta_{\text{floor}}(n)$ drops to $\theta_{\text{user}}$ or below, Pass becomes possible (subject to posterior)
3. Report the $\theta_{\text{eff}}$ used for the **final** decision

#### 3.3.4 Prior Scale Calibration (Student's *t*)

The prior on the d-dimensional effect profile $\delta$ MUST be a **correlation-shaped multivariate Student's *t*** distribution:

$$
\delta \sim t_\nu(0, \sigma^2 R)
$$

where R is the correlation matrix derived from $\Sigma_{\text{rate}}$, $\nu$ is a fixed degrees-of-freedom parameter, and $\sigma$ is a global scale calibrated to an exceedance target.

**Prior correlation matrix (R):**

$$
R := \text{Corr}(\Sigma_{\text{rate}}) = D^{-1/2} \Sigma_{\text{rate}} D^{-1/2}
$$

where D = diag($\Sigma_{\text{rate}}$). Implementations MUST ensure R is strictly SPD via jitter ladder (see [Implementation Guide](/reference/implementation-guide#42-jitter-ladder-for-spd-enforcement)).

**Degrees of freedom ($\nu$):**

Implementations MUST use $\nu := 4$.

**Calibrating $\sigma$ via exceedance target:**

The scale $\sigma$ MUST be chosen so that the prior exceedance probability equals a fixed target $\pi_0$ (default 0.62):

$$
P\left(\max_k |\delta_k| > \theta_{\text{eff}} \;\middle|\; \delta \sim t_\nu(0, \sigma^2 R)\right) = \pi_0
$$

This MUST be solved by deterministic 1D root-finding using Monte Carlo integration. See the [Implementation Guide](/reference/implementation-guide#7-prior-scale-calibration) for the algorithm.

#### 3.3.5 Deterministic Seeding Policy

To ensure reproducible results, all random number generation MUST be deterministic by default.

**Normative requirement:**

> Given identical timing samples and configuration, the oracle MUST produce identical results (up to floating-point roundoff).

**Seeding policy:**

- The bootstrap RNG seed MUST be deterministically derived from:
  - A fixed library constant seed (default: 0x74696D696E67, "timing" in ASCII)
  - A stable hash of configuration parameters
- All Monte Carlo RNG seeds (leak probability, floor constant, prior scale) MUST be similarly deterministic
- The Gibbs sampler RNG seed MUST be deterministic
- The chosen seeds SHOULD be reported in diagnostics

### 3.4 Bayesian Model

We use a Student's *t* prior over the full d-dimensional quantile-difference profile, implemented via Gibbs sampling on the scale-mixture representation.

#### 3.4.1 Latent Parameter

The latent parameter is the true per-decile timing difference profile:

$$
\delta \in \mathbb{R}^d \quad \text{(true decile-difference profile)}
$$

This is unconstrained: the model can represent any quantile-difference pattern.

#### 3.4.2 Likelihood (Robust t-likelihood via scale mixture)

Timing summary statistics $\Delta$ may deviate from the Gaussian approximation when $\Sigma_n$ is underestimated. To prevent pathological posterior certainty under covariance misestimation, implementations MUST use a robust likelihood with a single scalar precision factor $\kappa$.

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

**Gamma parameterization:** shape–rate. $E[\kappa] = 1$.

**Degrees of freedom for likelihood ($\nu_{\ell}$):**

Implementations MUST use $\nu_{\ell}$ := 8.

**Rationale:** When $\Sigma_n$ is underestimated (common when dependence is worse than calibration captured), $\kappa$ is pulled downward and inflates uncertainty automatically, preventing pathological certainty.

**Likelihood inflation warning:**

Implementations SHOULD set `Diagnostics.likelihood_inflated = true` when $\kappa_{\text{mean}} < 0.3$, indicating the likelihood covariance was effectively scaled up for robustness.

#### 3.4.3 Prior (Student's *t* via scale mixture)

The prior on $\delta \in \mathbb{R}^d$ is:

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

For $\nu$ = 4:

$$
\Lambda_0^{\text{marginal}} := \text{Cov}(\delta) = 2\sigma^2 R
$$

This matrix is used as the prior covariance reference in Gate 1 (§3.5.2).

#### 3.4.4 Posterior Inference (Deterministic Gibbs Sampling)

The posterior is approximated using a **deterministic Gibbs sampler** over $(\delta, \lambda, \kappa)$.

**Gibbs schedule (normative):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| N_gibbs | 256 | Total iterations |
| N_burn | 64 | Burn-in (discarded) |
| N_keep | 192 | Retained samples |

**Initialization:**

$$
\lambda^{(0)} = 1, \quad \kappa^{(0)} = 1
$$

**Iteration order:**

For t = 1, ..., N_gibbs:
1. Sample $\delta^{(t)} \sim p(\delta \mid \lambda^{(t-1)}, \kappa^{(t-1)}, \Delta)$
2. Sample $\lambda^{(t)} \sim p(\lambda \mid \delta^{(t)})$
3. Sample $\kappa^{(t)} \sim p(\kappa \mid \delta^{(t)}, \Delta)$

The Gibbs conditionals and numerical implementation are detailed in the [Implementation Guide](/reference/implementation-guide#5-gibbs-sampler-implementation).

**$\lambda$ mixing diagnostics:**

Implementations SHOULD set `Diagnostics.lambda_mixing_ok = false` when lambda_cv < 0.1 or lambda_ess < 20, indicating the sampler may not have converged.

#### 3.4.5 Decision Functional and Leak Probability

The decision functional is:

$$
m(\delta) := \max_k |\delta_k|
$$

The leak probability is:

$$
P(\text{leak} > \theta_{\text{eff}} \mid \Delta) = P(m(\delta) > \theta_{\text{eff}} \mid \Delta)
$$

**Estimation via Gibbs samples:**

$$
\widehat{P}(\text{leak}) = \frac{1}{N_{\text{keep}}} \sum_{s=1}^{N_{\text{keep}}} \mathbf{1}\left[m(\delta^{(s)}) > \theta_{\text{eff}}\right]
$$

**Posterior summaries:**

Implementations MUST compute:

- **Posterior mean:** $\delta_{\text{post}} := \frac{1}{N_{\text{keep}}} \sum_s \delta^{(s)}$
- **Max-effect credible interval:** 95% CI for $m(\delta)$ from empirical quantiles of $\{m(\delta^{(s)})\}$
- **Top quantiles:** Top 2-3 quantiles ranked by exceedance probability $P(|\delta_k| > \theta_{\text{eff}} \mid \Delta)$

**Interpreting the probability:**

This is a **posterior probability**, not a p-value. When we report "72% probability of a leak," we mean: given the data and our model, 72% of the posterior mass corresponds to effects exceeding $\theta_{\text{eff}}$.

### 3.5 Adaptive Sampling Loop

The core innovation: collect samples until confident, with natural early stopping.

**Verdict-blocking semantics:**

> Pass/Fail verdicts MUST be emitted **only** if all measurement quality gates pass. Otherwise the oracle MUST return Inconclusive.

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

**Gate 1: Insufficient Information Gain**

This gate detects when the data provides insufficient information relative to the prior, either because the posterior barely moved from the prior (data too noisy) or because the posterior stopped updating (not learning).

Implementations MUST compute the KL divergence between Gaussian surrogates of the prior and posterior:

$$
\mathrm{KL} = \frac{1}{2}\left(
\mathrm{tr}\left((\Lambda_0^{\text{marginal}})^{-1}\Lambda_{\text{post}}\right)
+ \mu_{\text{post}}^\top(\Lambda_0^{\text{marginal}})^{-1}\mu_{\text{post}}
- d + \ln\frac{|\Lambda_0^{\text{marginal}}|}{|\Lambda_{\text{post}}|}
\right)
$$

**Trigger conditions:**
- KL < KL_min (default 0.7 nats) → `DataTooNoisy`
- Sum of recent inter-batch KL divergences < 0.001 for 5+ batches → `NotLearning`

**Gate 2: Would Take Too Long**

Extrapolate time to decision based on current convergence rate. If projected time exceeds budget by a large margin (e.g., 10×), trigger Inconclusive with reason `WouldTakeTooLong`.

**Gate 3: Resource Budget Exceeded**

If elapsed time exceeds configured time budget → `TimeBudgetExceeded`
If total samples per class exceeds configured maximum → `SampleBudgetExceeded`

**Gate 4: Condition Drift Detected**

The covariance estimate $\Sigma_{\text{rate}}$ is computed during calibration. If measurement conditions change during the adaptive loop, this estimate becomes invalid.

Detect condition drift by comparing measurement statistics from calibration against the full test run:

- Variance ratio: $\sigma_{\text{post}}^2 / \sigma_{\text{cal}}^2$
- Autocorrelation change: $|\rho_{\text{post}}(1) - \rho_{\text{cal}}(1)|$
- Mean drift: $|\mu_{\text{post}} - \mu_{\text{cal}}| / \sigma_{\text{cal}}$

If variance ratio is outside [0.5, 2.0], or autocorrelation change exceeds 0.3, or mean drift exceeds 3.0, trigger Inconclusive with reason `ConditionsChanged`.

**Note on threshold elevation:** The case where $\theta_{\text{floor}}$ > $\theta_{\text{user}}$ is handled by the threshold elevation decision rule (§3.3.3), not by a quality gate.

### 3.6 Discrete Timer Mode

When the timer has low resolution (e.g., Apple Silicon's 41ns cntvct_el0), quantile estimation behaves differently due to tied values.

**Trigger condition:**

Discrete timer mode triggers when minimum uniqueness ratio is below 10%.

**Mid-distribution quantiles:**

Instead of standard quantile estimators, use mid-distribution quantiles which handle ties correctly. See the [Implementation Guide](/reference/implementation-guide#62-mid-distribution-quantiles).

**Work in ticks internally:**

In discrete mode, implementations SHOULD perform computations in **ticks** (timer's native unit) and convert to nanoseconds only for display.

**Gaussian approximation caveat:**

The Gaussian likelihood is a rougher approximation with discrete data. Implementations MUST report a quality issue about discrete timer mode.

### 3.7 Calibration Validation Requirements

The Bayesian approach requires empirical validation that posteriors are well-calibrated.

**Null calibration test (normative requirement):**

Implementations MUST provide a "fixed-vs-fixed" validation that measures end-to-end false positive rates:

- FPR_gated = P(Fail | H₀, all verdict-blocking gates pass)

**Acceptance criteria:**

| Metric | Target | Action if Exceeded |
|--------|--------|-------------------|
| FPR_gated | 2-5% | ≤ 5% | MUST escalate conservatism |

**Large-effect detection tests:**

Implementations MUST include validation tests ensuring the Student's *t* prior correctly detects obvious leaks even when measurement noise is high.

***

## 4. Measurement Requirements

This section defines abstract requirements for measurement. For implementation details, see the [Implementation Guide](/reference/implementation-guide).

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

### 5.2 Configuration Ergonomics

**Attacker model presets as primary entry point:**

The primary configuration entry point SHOULD be attacker model selection, not raw threshold values.

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

**Inconclusive guidance:**

For `Inconclusive` outcomes, implementations MUST provide the reason and SHOULD provide actionable guidance.

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
| d | Feature dimension (9 for timing) |
| {(c_t, y_t)} | Acquisition stream: class labels and timing measurements |
| T | Acquisition stream length (≈ 2n) |
| F, R | Per-class sample sets (filtered from stream) |
| $\Delta$ | d-vector of observed quantile differences |
| $\delta$ | d-vector of true (latent) quantile differences |
| $\Sigma_n$ | Covariance matrix at sample size n |
| $\Sigma_{\text{rate}}$ | Covariance rate |
| $V_{\text{cal}}$ | Calibration long-run variance proxy: $\Sigma_{\text{cal}} \cdot n_{\text{cal}}$ |
| $\hat{\tau}$ | Integrated autocorrelation time (via Geyer IMS) |
| n_eff | Effective sample size: $n_{\text{eff}} = n / \hat{\tau}$ |
| R | Prior correlation matrix: $\mathrm{Corr}(\Sigma_{\text{rate}})$ |
| $\nu$ | Student's *t* degrees of freedom (fixed at 4) |
| $\sigma$ | Student's *t* prior scale (calibrated via exceedance target) |
| $\lambda$ | Latent prior precision multiplier in scale-mixture representation |
| $\kappa$ | Latent likelihood precision multiplier (robust t-likelihood) |
| $\nu_{\ell}$ | Likelihood degrees of freedom (fixed at 8) |
| $t_{\nu}(0, \Sigma)$ | Multivariate Student's *t* with $\nu$ df, location 0, scale matrix $\Sigma$ |
| $\Lambda_0^{\text{marginal}}$ | Marginal prior covariance: $2\sigma^2 R$ for $\nu = 4$ |
| $\Lambda_{\text{post}}$ | Posterior covariance of $\delta$ (estimated from Gibbs samples) |
| $\delta_{\text{post}}$ | Posterior mean of $\delta$ (from Gibbs samples) |
| L_R | Cholesky factor of R (L_R L_Rᵀ = R) |
| N_gibbs | Total Gibbs iterations |
| N_burn | Burn-in iterations (discarded) |
| N_keep | Retained Gibbs samples |
| $\theta_{\text{user}}$ | User-requested threshold |
| $\theta_{\text{floor}}$ | Measurement floor (smallest resolvable effect) |
| $c_{\text{floor}}$ | Floor-rate constant: $\theta_{\text{floor,stat}} = c_{\text{floor}}/\sqrt{n_{\text{eff}}}$ |
| $\theta_{\text{tick}}$ | Timer resolution component of floor |
| $\theta_{\text{eff}}$ | Effective threshold used for inference |
| $m(\delta)$ | Decision functional: $\max_k |\delta_k|$ |
| b̂ | Block length (Politis-White, on acquisition stream) |
| MDE | Minimum detectable effect |
| n | Samples per class |
| B | Bootstrap iterations |
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
| Gibbs iterations (N_gibbs) | 256 | MUST | Sufficient mixing |
| Gibbs burn-in (N_burn) | 64 | MUST | Conservative |
| Gibbs retained (N_keep) | 192 | MUST | MC variance control |
| Gibbs initialization ($\lambda^{(0)}$, $\kappa^{(0)}$) | 1 | MUST | Prior means |
| Likelihood df ($\nu_{\ell}$) | 8 | MUST | Robustness to $\Sigma_n$ misestimation |
| Prior exceedance target ($\pi_0$) | 0.62 | SHOULD | Genuine uncertainty |
| Prior calibration MC draws | 50,000 | SHOULD | Stable $\sigma$ calibration |
| Bootstrap iterations | 2,000 | SHOULD | Covariance estimation accuracy |
| Monte Carlo samples (c_floor) | 50,000 | SHOULD | Floor-rate constant estimation |
| Batch size | 1,000 | SHOULD | Adaptive iteration granularity |
| Calibration samples | 5,000 | SHOULD | Initial covariance estimation |
| Pass threshold | 0.05 | SHOULD | 95% confidence of no leak |
| Fail threshold | 0.95 | SHOULD | 95% confidence of leak |
| KL_min (nats) | 0.7 | MUST | Minimum information gain for conclusive verdict |
| Block length cap | min(3√T, T/3) | SHOULD | Prevent degenerate blocks |
| Discrete threshold | 10% unique | SHOULD | Trigger discrete mode |
| Min ticks per call | 5 | SHOULD | Measurability floor |
| Max batch size | 20 | SHOULD | Limit microarch artifacts |
| Default time budget | 60 s | MAY | Maximum runtime |
| Default sample budget | 1,000,000 | MAY | Maximum samples |
| Default RNG seed | 0x74696D696E67 | SHOULD | "timing" in ASCII |
| R jitter ($\varepsilon$) | $10^{-10}$–$10^{-6}$ | MUST | Ensure SPD |
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

### v6.0 (from v5.7)

**Specification simplification and modularization:**

- **Removed:** Research mode as separate outcome type. When θ_user = 0, the oracle returns Inconclusive with posterior estimates. This removes ~50 lines and the ResearchStatus enum.

- **Removed:** 2D projection for reporting (§3.4.6 in v5.7). The shift/tail decomposition and EffectPattern enum have been removed. EffectEstimate now reports max|δ|, 95% CI, and top quantiles by exceedance probability.

- **Removed:** Exploitability classification (§2.4 in v5.7). This is now documented only in the user guide, not the statistical specification.

- **Simplified:** κ robust likelihood diagnostics. Only `likelihood_inflated: Bool` is exposed; detailed κ mixing diagnostics (kappa_mean, kappa_sd, kappa_cv, kappa_ess, kappa_mixing_ok) are internal implementation details.

- **Consolidated:** Quality gates from 6 to 4:
  - Gates 1+2 → Insufficient Information Gain
  - Gates 4+5 → Resource Budget Exceeded
  - Gate 3 → Would Take Too Long
  - Gate 6 → Conditions Changed

- **Consolidated:** Issue codes from 30+ to 8 categories: DependenceHigh, PrecisionLow, DiscreteMode, ThresholdIssue, FilteringApplied, StationarityIssue, NumericalIssue, LikelihoodInflated.

- **Created:** Separate Implementation Guide for algorithms (IACT, block bootstrap, numerical stability, Gibbs conditionals, quantile computation).

- **Created:** Separate Power Module Specification for power/EM analysis.

- **Removed:** Projection mismatch diagnostics and threshold calibration (no longer needed without 2D projection).

- **Removed:** Dimension-aware regularization details (moved to Implementation Guide; timing uses d=9 which rarely triggers stressed regimes).

**Rationale:** These changes reduce specification complexity by ~40% while preserving all normative statistical requirements. The removed features were either:
1. Reporting conveniences (2D projection, exploitability classification) that don't affect inference
2. Separate code paths for edge cases (Research mode) that can be handled by existing mechanisms
3. Implementation details (algorithms, numerical procedures) better suited to a separate guide
