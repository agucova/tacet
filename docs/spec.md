# timing-oracle Specification (v4.1)

## 1. Overview

### Problem Statement

Timing side-channel attacks exploit the fact that cryptographic implementations often take different amounts of time depending on secret data. Detecting these leaks is critical for security, but existing tools have significant limitations:

- **DudeCT** and similar t-test approaches compare means, missing distributional differences (e.g., cache effects that only affect upper quantiles)
- **P-values are misinterpreted**: A p-value of 0.01 doesn't mean "1% chance of leak"—it means "1% chance of this data given no leak." These are very different.
- **Unbounded false positives**: With enough samples, even negligible effects become "statistically significant"
- **CI flakiness**: Tests that pass locally fail in CI due to environmental noise, or vice versa
- **Fixed sample sizes**: Existing tools require pre-committing to a sample count, wasting time on clear-cut cases and potentially under-sampling ambiguous ones

### Solution

`timing-oracle` addresses these issues with:

1. **Quantile-based statistics**: Compare nine deciles instead of just means, capturing both uniform shifts (branch timing) and tail effects (cache misses)

2. **Adaptive Bayesian inference**: Collect samples until confident, with natural early stopping for both clear leaks and clear passes

3. **Three-way decisions**: Pass / Fail / Inconclusive—explicitly distinguishing "no leak detected" from "couldn't measure reliably"

4. **Interpretable output**: "72% probability of a timing leak with ~50ns effect" instead of "$t = 2.34$, $p = 0.019$"

5. **Research mode**: Exploratory analysis without a decision threshold, reporting posterior distributions directly

6. **Fail-safe design**: Strong verdicts only when measurement quality and model fit are verified; prefer Inconclusive over confidently wrong

### Design Goals

- **Interpretable**: Output is a probability (0–100%), not a t-statistic
- **Adaptive**: Automatically collects more samples when uncertain, stops early when confident
- **CI-friendly**: Three-way output prevents flaky tests; inconclusive results are explicit
- **Fast**: Early stopping means clear cases finish quickly
- **Portable**: Handles different timer resolutions via adaptive batching
- **Honest**: Never silently clamps thresholds; reports what it can actually resolve
- **Fail-safe**: CI verdicts should almost never be confidently wrong; when assumptions are violated, prefer Inconclusive
- **Reproducible**: Deterministic results given identical timing samples and configuration

---

## 2. Statistical Methodology

This section describes the mathematical foundation of timing-oracle: why we use quantile differences instead of means, how adaptive Bayesian inference works, and how we estimate uncertainty while keeping computational costs reasonable.

### 2.1 Test Statistic: Quantile Differences

We collect timing samples from two classes:
- **Fixed class**: A specific input (e.g., all zeros, matching plaintext)
- **Random class**: Randomly sampled inputs

Rather than comparing means, we compare the distributions via their deciles. For each class, compute the 10th, 20th, ..., 90th percentiles, yielding two vectors in $\mathbb{R}^9$. The test statistic is their difference:

$$
\Delta = \hat{q}(\text{Fixed}) - \hat{q}(\text{Random}) \in \mathbb{R}^9
$$

where $\hat{q}_p$ denotes the empirical $p$-th quantile.

**Why quantiles instead of means?**

Timing leaks manifest in different ways:
- **Uniform shift**: A different code path adds constant overhead → all quantiles shift equally
- **Tail effect**: Cache misses occur probabilistically → upper quantiles shift more than lower

A t-test (comparing means) would detect uniform shifts but completely miss tail effects. Consider a cache-timing attack where 90% of operations are fast, but 10% hit a slow path. The mean shifts slightly, but the 90th percentile shifts dramatically. Quantile differences capture both patterns in a single test statistic.

**Why nine deciles specifically?**

This is a bias-variance tradeoff:
- Fewer quantiles (e.g., just the median) would miss distributional structure
- More quantiles (e.g., percentiles) would require estimating a $99 \times 99$ covariance matrix from limited data, introducing severe estimation noise

Nine deciles capture enough structure to distinguish shift from tail patterns while keeping the covariance matrix ($9 \times 9 = 81$ parameters) tractable with typical sample sizes (10k–100k).

**Quantile computation:** We use **type 2** quantiles (inverse empirical CDF with averaging).¹ For sorted sample $x$ of size $n$ at probability $p$:

$$
h = n \cdot p + 0.5
$$

$$
\hat{q}_p = \frac{x_{\lfloor h \rfloor} + x_{\lceil h \rceil}}{2}
$$

Type 2 uses the inverse of the empirical distribution function with averaging at discontinuities.

For discrete timers with many tied values, use **mid-distribution quantiles** (see §2.10) which correctly handle atoms.

¹ Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages." The American Statistician 50(4):361–365.

### 2.2 Architecture: Calibration + Adaptive Loop

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
│                        • Check model fit                         │
│                        • Check stopping                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Phase 1: Calibration** (runs once)
- Collect initial samples to characterize measurement noise
- Estimate covariance structure via stream-based block bootstrap
- Compute "covariance rate" $\Sigma_{\text{rate}}$ that scales as $\Sigma = \Sigma_{\text{rate}} / n$
- Compute initial measurement floor $\theta_{\text{floor}}$ and floor-rate constant $c_{\text{floor}}$
- Compute model mismatch threshold $Q_{\text{thresh}}$ from bootstrap distribution
- Compute effective threshold $\theta_{\text{eff}}$ and set prior scale
- Run pre-flight checks (timer sanity, harness sanity, stationarity)

**Phase 2: Adaptive Loop** (iterates until decision)
- Collect batches of samples
- Update quantile estimates from all data collected so far
- Scale covariance: $\Sigma_n = \Sigma_{\text{rate}} / n$
- Update $\theta_{\text{floor}}(n)$ using floor-rate constant
- Compute posterior and $P(\text{effect} > \theta_{\text{eff}})$
- Check quality gates first (posterior ≈ prior → Inconclusive)
- Check model fit gate (large residuals → Inconclusive)
- Check decision boundaries (P > 95% → Fail, P < 5% → Pass)
- Check time/sample budgets

**Why this structure?**

The key insight is that covariance scales as $1/n$ for quantile estimators. By estimating the covariance *rate* once during calibration, we can cheaply update the posterior as more data arrives—no re-bootstrapping needed. This makes adaptive sampling computationally tractable.

### 2.3 Calibration Phase

The calibration phase runs once at startup to characterize measurement noise.

**Sample collection:**

Collect $n_{\text{cal}}$ samples per class (default: 5,000). This is enough to estimate covariance structure reliably while keeping calibration fast.

#### 2.3.1 Acquisition Stream Model

Measurement produces an interleaved acquisition stream indexed by time:

$$
\{(c_t, y_t)\}_{t=1}^{T}, \quad c_t \in \{F, R\}, \; T \approx 2n
$$

where $y_t$ is the measured runtime (or ticks) at acquisition index $t$, and $F$/$R$ denote Fixed and Random classes.

Per-class samples are obtained by filtering the stream:

$$
F := \{y_t : c_t = F\}, \quad R := \{y_t : c_t = R\}
$$

**Critical principle:** The acquisition stream is the data-generating process. All bootstrap and dependence estimation must preserve adjacency in acquisition order, not per-class position. The underlying stochastic process—including drift, frequency scaling, and cache state evolution—operates in continuous time indexed by $t$. Splitting by label and treating each class as an independent time series is statistically incorrect.

#### 2.3.2 Covariance Estimation via Stream-Based Block Bootstrap

Timing measurements exhibit autocorrelation—nearby samples are more similar than distant ones due to cache state, frequency scaling, etc. Standard bootstrap assumes i.i.d. samples, underestimating variance. We use block bootstrap on the acquisition stream to preserve the true dependence structure.

**Politis-White automatic block length selection:**

We use the Politis-White algorithm² to select the optimal block length, computed on the **acquisition stream** (not per-class subsequences):

**Step 1: Compute ACF on acquisition stream**

Use either:
- The pooled stream $y_t$ directly, or
- A residualized stream $y_t - \text{median}(\text{window}_t)$ to reduce slow drift and focus on short-range correlation

Let $k_n = \max(5, \lfloor \log_{10}(T) \rfloor)$ and $m_{\max} = \lceil \sqrt{T} \rceil + k_n$.

Compute autocorrelations $\hat{\rho}_k$ for $k = 0, \ldots, m_{\max}$. Find the first lag $m^*$ where $k_n$ consecutive autocorrelations fall within the conservative band $\pm 2\sqrt{\log_{10}(T)/T}$. Set $m = \min(2 \cdot \max(m^*, 1), m_{\max})$.

**Step 2: Compute spectral quantities**

Using a flat-top (trapezoidal) kernel $h(x) = \min(1, 2(1 - |x|))$:

$$
\hat{\sigma}^2 = \sum_{k=-m}^{m} h\left(\frac{k}{m}\right) \hat{\gamma}_k, \quad g = \sum_{k=-m}^{m} h\left(\frac{k}{m}\right) |k| \, \hat{\gamma}_k
$$

where $\hat{\gamma}_k$ is the sample autocovariance at lag $k$.

**Step 3: Compute optimal block length**

$$
\hat{b} = \left\lceil \left( \frac{g^2}{(\hat{\sigma}^2)^2} \right)^{1/3} T^{1/3} \right\rceil
$$

Capped at $b_{\max} = \min(3\sqrt{T}, T/3)$ to prevent degenerate blocks.

² Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection for the Dependent Bootstrap." Econometric Reviews 23(1):53–70.

**Stream-based bootstrap procedure:**

For each of $B = 2{,}000$ bootstrap iterations:

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

The covariance of quantile estimators scales as $1/n$. We compute the "rate":

$$
\Sigma_{\text{rate}} = \hat{\Sigma}_{\text{cal}} \cdot n_{\text{cal}}
$$

During the adaptive loop, with $n$ total samples per class:

$$
\Sigma_n = \Sigma_{\text{rate}} / n
$$

This allows cheap posterior updates without re-bootstrapping.

**Numerical stability:**

With discrete timers or few unique values, some quantiles may have near-zero variance, making $\Sigma$ ill-conditioned. We regularize by ensuring a minimum diagonal value:

$$
\sigma^2_i \leftarrow \max(\sigma^2_i, 0.01 \cdot \bar{\sigma}^2) + \varepsilon
$$

where $\bar{\sigma}^2 = \text{tr}(\Sigma)/9$ and $\varepsilon = 10^{-10} + \bar{\sigma}^2 \cdot 10^{-8}$.

#### 2.3.3 Model Mismatch Threshold Calibration

During bootstrap, we also calibrate the threshold for the model fit diagnostic (see §2.6 Gate 8):

For each bootstrap replicate $\Delta^*$:
1. Compute GLS fit: $\hat{\beta}^* = (X^\top \Sigma_n^{-1} X)^{-1} X^\top \Sigma_n^{-1} \Delta^*$
2. Compute residual: $r^* = \Delta^* - X\hat{\beta}^*$
3. Compute fit statistic: $Q^* = (r^*)^\top \Sigma_n^{-1} r^*$

The mismatch threshold is the 99th percentile of the bootstrap distribution:

$$
Q_{\text{thresh}} = q_{0.99}(\{Q^*_b\}_{b=1}^B)
$$

This bootstrap-calibrated threshold is preferred over the asymptotic $\chi^2_7$ reference because:
- The posterior mean involves shrinkage from the prior
- $\Sigma_n$ is estimated, not known
- The bootstrap automatically adapts to the actual covariance structure

**Fallback:** If bootstrap calibration fails, use $Q_{\text{thresh}} = \chi^2_{7, 0.99} \approx 18.48$ as a conservative diagnostic cutoff.

#### 2.3.4 Measurement Floor and Effective Threshold

A critical design element is distinguishing between what the user *wants* to detect ($\theta_{\text{user}}$) and what the measurement *can* detect ($\theta_{\text{floor}}$).

**Floor-rate constant:**

Under the model's scaling assumption $\Sigma_n = \Sigma_{\text{rate}}/n$, the measurement floor scales as $1/\sqrt{n}$. We compute a floor-rate constant once at calibration:

Define the prediction covariance in "rate form":

$$
\Sigma_{\text{pred,rate}} := X \left( X^\top \Sigma_{\text{rate}}^{-1} X \right)^{-1} X^\top
$$

Draw $Z_0 \sim \mathcal{N}(0, \Sigma_{\text{pred,rate}})$ via Monte Carlo (50,000 samples) and compute:

$$
c_{\text{floor}} := q_{0.95}\left( \max_k |Z_{0,k}| \right)
$$

This is the 95th percentile of the max absolute prediction under unit sample size.

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

**Effective threshold ($\theta_{\text{eff}}$):**

The threshold actually used for inference:

$$
\theta_{\text{eff}} = \begin{cases}
\max(\theta_{\text{user}}, \theta_{\text{floor}}) & \text{if } \theta_{\text{user}} > 0 \\
\theta_{\text{floor}} & \text{if } \theta_{\text{user}} = 0 \text{ (Research mode)}
\end{cases}
$$

**Threshold elevation warning:**

When $\theta_{\text{eff}} > \theta_{\text{user}}$, emit a quality issue:

```rust
QualityIssue {
    code: IssueCode::ThresholdElevated,
    message: "Requested θ = 2 ns, but measurement floor is 41 ns",
    guidance: "Reporting probabilities for θ_eff = 41 ns. Use a cycle counter for better resolution.",
    theta_user: 2.0,
    theta_eff: 41.0,
}
```

This ensures we never silently pretend to test a threshold we cannot resolve.

**Dynamic floor updates:**

During the adaptive loop, $\theta_{\text{floor}}(n)$ is recomputed analytically as $n$ grows (no Monte Carlo needed). The policy:

1. If $\theta_{\text{floor}}(n)$ drops below $\theta_{\text{user}}$ during sampling, update $\theta_{\text{eff}} = \theta_{\text{user}}$ (the user's threshold becomes achievable)
2. Track whether you're on pace to reach $\theta_{\text{user}}$; if extrapolation shows you won't make it within budget, consider early termination with `Inconclusive`

**Prior scale:**

The prior on effect parameters $\beta = (\mu, \tau)$ is set proportional to the effective threshold:

$$
\sigma_{\text{prior}} = 1.12 \cdot \theta_{\text{eff}}
$$

This ensures the prior leak probability $P(\max_k |(X\beta)_k| > \theta_{\text{eff}} \mid \text{prior}) \approx 62\%$, representing genuine uncertainty about whether effects exceed the effective threshold.

**Important:** The leak test computes $P(\max_k |(X\beta)_k| > \theta_{\text{eff}})$ using the transformed predictions $X\beta$, not the raw parameter vector $\beta$. The design matrix $X$ amplifies $\beta$ into 9 correlated predictions, and taking the max over these increases the exceedance probability. The factor 1.12 (rather than 2.0) accounts for this transformation and was determined via Monte Carlo calibration.

#### 2.3.5 Deterministic Seeding Policy

To ensure reproducible results, all random number generation is deterministic by default.

**Normative requirement:**

> Given identical timing samples and configuration, the oracle must produce identical results (up to floating-point roundoff). Algorithmic randomness is not epistemic uncertainty about the leak—it is approximation error that should not influence the decision rule.

**Seeding policy:**

- The bootstrap RNG seed is deterministically derived from:
  - A fixed library constant seed (default: 0x74696D696E67, "timing" in ASCII)
  - A stable hash of configuration parameters

- All Monte Carlo RNG seeds (leak probability, floor constant) are similarly deterministic

- The chosen seeds are reported in `Diagnostics.seed` (and `Calibration.rng_seed`)

**Rationale:** For CI gating, the decision function must be well-defined given the data:

$$
\text{Verdict} = f(\text{timing samples}, \text{config})
$$

Not:

$$
\text{Verdict} = f(\text{timing samples}, \text{config}, \text{RNG state})
$$

**Calibration output:**

```rust
struct Calibration {
    sigma_rate: Matrix9x9,      // Covariance rate: Σ = Σ_rate / n
    block_length: usize,        // Estimated from acquisition stream
    c_floor: f64,               // Floor-rate constant
    q_thresh: f64,              // Model mismatch threshold (99th percentile)
    theta_floor_initial: f64,   // Initial measurement floor
    theta_eff: f64,             // Effective threshold for this run
    prior_cov: Matrix2x2,       // Prior covariance for β
    timer_resolution_ns: f64,   // Detected tick size
    samples_per_second: f64,    // For time estimation
    rng_seed: u64,              // Deterministic seed used
}
```

### 2.4 Bayesian Model

We use a conjugate Gaussian model that admits closed-form posteriors—no MCMC required.

#### Design Matrix

We decompose the observed quantile differences into interpretable components:

- **Uniform shift** ($\mu$): All quantiles move by the same amount (e.g., a branch that adds constant overhead)
- **Tail effect** ($\tau$): Upper quantiles shift more than lower ones (e.g., cache misses that occur probabilistically)

The design matrix encodes this decomposition:

$$
X = \begin{bmatrix} \mathbf{1} & \mathbf{b}_{\text{tail}} \end{bmatrix} \in \mathbb{R}^{9 \times 2}
$$

where:
- $\mathbf{1} = (1, 1, 1, 1, 1, 1, 1, 1, 1)^\top$ — uniform shift moves all 9 quantiles equally
- $\mathbf{b}_{\text{tail}} = (-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5)^\top$ — tail effect moves upper quantiles up, lower quantiles down

The tail basis is centered (sums to zero) so that $\mu$ and $\tau$ are orthogonal. An effect of $\mu = 10, \tau = 0$ means a pure 10ns uniform shift; $\mu = 0, \tau = 20$ means upper quantiles are 10ns slower while lower quantiles are 10ns faster.

**Design limitation:** The antisymmetric tail basis assumes symmetric patterns. Real cache-timing leaks often show asymmetric upper-tail effects. The model *can* fit these (μ absorbs the mean shift, τ captures the spread), but interpretation of τ becomes less intuitive. When the model fit is poor (see §2.6 Gate 8), the $(\mu, \tau)$ decomposition should be interpreted with caution.

#### Likelihood

$$
\Delta \mid \beta \sim \mathcal{N}(X\beta, \Sigma_n)
$$

where $\Sigma_n = \Sigma_{\text{rate}} / n$ is the scaled covariance.

#### Prior

$$
\beta \sim \mathcal{N}(0, \Lambda_0), \quad \Lambda_0 = \text{diag}(\sigma_{\text{prior}}^2, \sigma_{\text{prior}}^2)
$$

where $\sigma_{\text{prior}} = 1.12 \cdot \theta_{\text{eff}}$ (see §2.3.4).

#### Posterior

With Gaussian likelihood and Gaussian prior, the posterior is also Gaussian⁴:

$$
\beta \mid \Delta \sim \mathcal{N}(\beta_{\text{post}}, \Lambda_{\text{post}})
$$

where:

$$
\Lambda_{\text{post}} = \left( X^\top \Sigma_n^{-1} X + \Lambda_0^{-1} \right)^{-1}
$$

$$
\beta_{\text{post}} = \Lambda_{\text{post}} X^\top \Sigma_n^{-1} \Delta
$$

The posterior mean $\beta_{\text{post}} = (\mu, \tau)^\top$ gives the estimated shift and tail effects in nanoseconds.

⁴ Bishop, C. M. (2006). Pattern Recognition and Machine Learning, §3.3. Springer.

#### Leak Probability

The key question is: "what's the probability of a leak exceeding $\theta_{\text{eff}}$?" We compute:

$$
P(\text{leak} > \theta_{\text{eff}} \mid \Delta) = P\bigl(\max_k |(X\beta)_k| > \theta_{\text{eff}} \;\big|\; \Delta\bigr)
$$

Since the posterior is Gaussian, we compute this by Monte Carlo integration:

```python
samples = draw_from_normal(beta_post, Lambda_post, n=10000, seed=deterministic_seed)
count = 0
for beta in samples:
    pred = X @ beta  # 9-vector of predicted quantile differences
    max_effect = max(abs(pred))
    if max_effect > theta_eff:
        count += 1
leak_probability = count / 10000
```

This is O(1)—we're drawing from a 2D Gaussian and doing simple arithmetic. The seed is deterministic (see §2.3.5).

**Interpreting the probability:**

This is a **posterior probability**, not a p-value. When we report "72% probability of a leak," we mean: given the data and our model, 72% of the posterior mass corresponds to effects exceeding $\theta_{\text{eff}}$.

This is calibrated by construction under the Gaussian model. Empirical validation (§2.11) confirms calibration holds in practice.

**Note on $\theta_{\text{eff}}$:**

All inference uses $\theta_{\text{eff}}$, not $\theta_{\text{user}}$. When these differ, the output clearly reports both values so the user understands what was actually tested.

### 2.5 Adaptive Sampling Loop

The core innovation: collect samples until confident, with natural early stopping.

**Verdict-blocking semantics:**

> Pass/Fail verdicts are emitted **only** if (i) all measurement quality gates pass, (ii) condition drift checks pass, and (iii) the model fit gate passes (no ModelMismatch). Otherwise the oracle must return Inconclusive, reporting the posterior probability as an "estimate under the model" but not asserting a strong claim.

This policy ensures: **CI verdicts should almost never be confidently wrong.**

```rust
fn run_adaptive(calibration: &Calibration, theta_user: f64, config: &Config) -> Outcome {
    let mut state = State::new(calibration);
    let mut theta_eff = calibration.theta_eff;

    loop {
        // 1. Collect a batch
        state.collect_batch(config.batch_size);

        // 2. Update θ_floor analytically (no MC needed)
        let theta_floor = f64::max(
            calibration.c_floor / (state.n_total as f64).sqrt(),
            calibration.theta_tick,
        );
        if theta_user > 0.0 && theta_floor < theta_user {
            // User's threshold is now achievable
            theta_eff = theta_user;
        }

        // 3. Update posterior (cheap - no bootstrap)
        let posterior = state.compute_posterior(theta_eff);

        // 4. Compute model fit statistic
        let residual = state.delta - X * posterior.beta;
        let q_fit = residual.dot(&state.sigma_n_inv * residual);
        let model_mismatch = q_fit > calibration.q_thresh;

        // 5. Check ALL quality gates FIRST (verdict-blocking)
        if let Some(reason) = check_quality_gates(&state, &posterior, calibration, config) {
            return Outcome::Inconclusive {
                reason,
                leak_probability: posterior.leak_probability,
                effect: posterior.effect_estimate(),
                theta_user,
                theta_eff,
                theta_floor,
                samples_used: state.n_total,
            };
        }

        // 6. Check model fit gate (verdict-blocking)
        if model_mismatch {
            return Outcome::Inconclusive {
                reason: InconclusiveReason::ModelMismatch {
                    q_statistic: q_fit,
                    q_threshold: calibration.q_thresh,
                    message: "Observed quantile pattern not well-explained by shift+tail model".into(),
                    guidance: "The timing difference may have a shape not captured by the 2D basis. \
                               Inspect raw quantile differences for asymmetric or multi-modal patterns.".into(),
                },
                leak_probability: posterior.leak_probability,
                effect: posterior.effect_estimate_with_caveat(),
                theta_user,
                theta_eff,
                theta_floor,
                samples_used: state.n_total,
            };
        }

        // 7. Check decision boundaries (only if all gates pass)
        if posterior.leak_probability > 0.95 {
            return Outcome::Fail {
                leak_probability: posterior.leak_probability,
                effect: posterior.effect_estimate(),
                theta_user,
                theta_eff,
                theta_floor,
                samples_used: state.n_total,
            };
        }
        if posterior.leak_probability < 0.05 {
            return Outcome::Pass {
                leak_probability: posterior.leak_probability,
                effect: posterior.effect_estimate(),
                theta_user,
                theta_eff,
                theta_floor,
                samples_used: state.n_total,
            };
        }

        // 8. Check budgets
        if state.budget_exhausted(config) {
            return Outcome::Inconclusive {
                reason: InconclusiveReason::BudgetExhausted { .. },
                // ...
            };
        }
    }
}
```

**Why adaptive sampling works for Bayesian inference:**

Frequentist methods suffer from **optional stopping**: if you keep sampling until you get a significant result, you inflate your false positive rate.

Bayesian methods don't have this problem. The posterior probability is valid regardless of when you stop—this is the **likelihood principle**. Your inference depends only on the data observed, not your sampling plan.

So we can:
- Stop early when confident (P > 95% or P < 5%)
- Keep sampling when uncertain
- Give up when data quality is too poor to ever reach confidence

**Per-iteration cost:**

Each iteration requires:
1. Collect batch: O(batch_size)
2. Recompute quantiles: O(n log n) — but incremental updates possible
3. Scale covariance: O(1)
4. GLS regression: O(1) — 2×2 and 9×9 matrix operations
5. Model fit check: O(1) — single quadratic form
6. Monte Carlo integration: O(10,000)
7. Update θ_floor: O(1) — analytical formula

Total: dominated by quantile computation, which can be optimized with streaming quantile algorithms.

**Batch size selection:**

Default batch size is 1,000 samples. This balances:
- Large enough for stable quantile estimates per batch
- Small enough for responsive early stopping
- Matches typical calibration sample count

### 2.6 Quality Gates

Quality gates detect when data is too poor to reach a confident decision. **All gates are verdict-blocking**: if any gate triggers, the oracle returns Inconclusive rather than Pass/Fail.

#### Gate 1: Posterior ≈ Prior (Data Too Noisy)

If measurement uncertainty is high, the posterior barely moves from the prior:

```rust
let variance_ratio = posterior.beta_cov.det() / calibration.prior_cov.det();

if variance_ratio > 0.5 {
    return Some(InconclusiveReason::DataTooNoisy {
        message: "Posterior variance is >50% of prior; data not informative",
        guidance: "Try: cycle counter, reduce system load, increase batch size",
    });
}
```

**Why this matters:** A wide posterior centered at zero can have substantial mass above θ—not because we're confident there's a leak, but because we're uncertain about everything. This gate catches that pathology.

**Critical for threshold elevation:** When $\theta_{\text{user}} \ll \theta_{\text{floor}}$, this gate prevents false decisions by catching the case where the posterior hasn't learned anything meaningful relative to the effective threshold.

#### Gate 2: Learning Rate Collapsed

Track KL divergence between successive posteriors:

```rust
// KL(new || old) measures how much we learned this batch
let kl = kl_divergence(&posterior, &previous_posterior);

if recent_kl_sum < 0.001 {  // Over last 5 batches
    return Some(InconclusiveReason::NotLearning {
        message: "Posterior stopped updating despite new data",
        guidance: "Measurement may have systematic issues",
    });
}
```

For Gaussian posteriors:

$$
\text{KL}(p_{\text{new}} \| p_{\text{old}}) = \frac{1}{2}\left( \text{tr}(\Lambda_{\text{old}}^{-1} \Lambda_{\text{new}}) + (\mu_{\text{old}} - \mu_{\text{new}})^\top \Lambda_{\text{old}}^{-1} (\mu_{\text{old}} - \mu_{\text{new}}) - k + \ln\frac{|\Lambda_{\text{old}}|}{|\Lambda_{\text{new}}|} \right)
$$

#### Gate 3: Would Take Too Long

Extrapolate time to decision:

```rust
let samples_needed = extrapolate_samples_to_decision(&state, &posterior);
let time_needed = samples_needed as f64 / state.samples_per_second;

if time_needed > config.time_budget * 10.0 {
    return Some(InconclusiveReason::WouldTakeTooLong {
        estimated_time: time_needed,
        samples_needed,
        guidance: "Effect may be very close to threshold; consider adjusting θ",
    });
}
```

**Extrapolation heuristic:**

Posterior standard deviation shrinks as $1/\sqrt{n}$. To reach a decision boundary:

```rust
fn extrapolate_samples_to_decision(state: &State, posterior: &Posterior) -> usize {
    let p = posterior.leak_probability;

    // Distance to nearest decision boundary
    let margin = f64::min((p - 0.05).abs(), (0.95 - p).abs());

    // Current posterior spread
    let current_std = posterior.beta_cov.trace().sqrt();

    // Rough estimate: need to shrink std proportionally to margin
    let std_reduction_needed = current_std / margin;
    let sample_multiplier = std_reduction_needed.powi(2);

    (state.n_total as f64 * sample_multiplier) as usize
}
```

#### Gate 4: Threshold Unachievable

If even at maximum budget, we cannot reach $\theta_{\text{user}}$:

```rust
let theta_floor_at_max = calibration.c_floor / (config.max_samples as f64).sqrt();
let theta_floor_at_max = f64::max(theta_floor_at_max, calibration.theta_tick);

if theta_floor_at_max > theta_user && theta_user > 0.0 {
    return Some(InconclusiveReason::ThresholdUnachievable {
        theta_user,
        best_achievable: theta_floor_at_max,
        message: format!(
            "Requested θ = {:.1} ns, but best achievable is {:.1} ns",
            theta_user, theta_floor_at_max
        ),
        guidance: "Use a cycle counter or accept the elevated threshold",
    });
}
```

#### Gate 5: Time Budget Exceeded

```rust
if state.elapsed() > config.time_budget {
    return Some(InconclusiveReason::TimeBudgetExceeded {
        current_probability: posterior.leak_probability,
        samples_collected: state.n_total,
    });
}
```

#### Gate 6: Sample Budget Exceeded

```rust
if state.n_total >= config.max_samples {
    return Some(InconclusiveReason::SampleBudgetExceeded {
        current_probability: posterior.leak_probability,
        samples_collected: state.n_total,
    });
}
```

#### Gate 7: Condition Drift Detected

The covariance estimate $\Sigma_{\text{rate}}$ is computed during calibration. If measurement conditions change during the adaptive loop (e.g., concurrent system load, thermal throttling, or other interference), this estimate becomes invalid and can cause false positives or negatives.

We detect condition drift by comparing measurement statistics from calibration against the full test run:

**Online statistics tracking:**

During sample collection, we track per-class statistics incrementally using Welford's algorithm:

```rust
struct OnlineStats {
    count: usize,
    mean: f64,
    m2: f64,           // For variance: Σ(x - mean)²
    prev_value: f64,   // For lag-1 autocorrelation
    autocorr_sum: f64, // Σ(x_i - μ)(x_{i-1} - μ)
}
```

This has O(1) overhead per sample—negligible compared to the measurement itself.

**Calibration snapshot:**

At the end of calibration, we save a snapshot of the statistics for each class:

$$
S_{\text{cal}} = \{ \bar{x}_{\text{cal}}, \sigma^2_{\text{cal}}, \rho_{\text{cal}}(1) \}
$$

where $\rho(1)$ is the lag-1 autocorrelation coefficient.

**Post-test comparison:**

After the adaptive loop completes, we compute the same statistics over all collected samples and check for significant drift:

```rust
struct ConditionDrift {
    variance_ratio: f64,    // σ²_post / σ²_cal
    autocorr_change: f64,   // |ρ_post(1) - ρ_cal(1)|
    mean_drift: f64,        // |μ_post - μ_cal| / σ_cal
}

// Trigger Inconclusive if any threshold exceeded
if drift.variance_ratio > 2.0 || drift.variance_ratio < 0.5 {
    // Variance changed by 2x+ → measurement noise regime changed
    return Some(InconclusiveReason::ConditionsChanged { drift });
}

if drift.autocorr_change > 0.3 {
    // Autocorrelation structure changed → different interference pattern
    return Some(InconclusiveReason::ConditionsChanged { drift });
}

if drift.mean_drift > 3.0 {
    // Mean shifted by 3+ standard deviations → systematic change
    return Some(InconclusiveReason::ConditionsChanged { drift });
}
```

**Why this matters:**

The Bayesian layer assumes the covariance estimate from calibration remains valid. When running in noisy environments (e.g., concurrent test execution), system load can create:

1. **Systematic timing biases** that affect one measurement class more than the other
2. **Non-stationary noise** where variance or autocorrelation changes over time
3. **Regime changes** between calibration and the adaptive loop

These conditions can cause confident but incorrect verdicts. Gate 7 detects when calibration assumptions are violated and returns `Inconclusive` rather than a potentially misleading `Pass` or `Fail`.

**Default thresholds:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `max_variance_ratio` | 2.0 | Variance can at most double |
| `min_variance_ratio` | 0.5 | Variance can at most halve |
| `max_autocorr_change` | 0.3 | Autocorrelation can shift by ±0.3 |
| `max_mean_drift_sigmas` | 3.0 | Mean can shift by 3 standard deviations |

#### Gate 8: Model Mismatch (Lack of Fit)

The 2D effect model constrains the mean structure to $\text{span}(X)$. Real timing leaks can induce quantile-difference shapes outside this span (e.g., upper-tail-only effects, asymmetric bumps). When the model fits poorly, the posterior on $\beta$ can be confident while predictions are wrong.

**Residual and fit statistic:**

At each iteration, compute:

$$
r := \Delta - X\hat{\beta}, \quad \hat{\beta} := \beta_{\text{post}}
$$

$$
Q := r^\top \Sigma_n^{-1} r
$$

Under the idealized model (Gaussian likelihood, correctly specified $X$, known $\Sigma$), $Q \sim \chi^2_7$ with $\nu = 9 - 2 = 7$ degrees of freedom. However, due to shrinkage from the prior and estimated covariance, we use the bootstrap-calibrated threshold $Q_{\text{thresh}}$ (see §2.3.3) rather than relying on the asymptotic distribution.

**Gate rule:**

```rust
if q_fit > calibration.q_thresh {
    return Some(InconclusiveReason::ModelMismatch {
        q_statistic: q_fit,
        q_threshold: calibration.q_thresh,
        message: "Observed quantile pattern not well-explained by shift+tail model",
        guidance: "The timing difference may have a shape not captured by the 2D basis. \
                   Inspect raw quantile differences for asymmetric or multi-modal patterns.",
    });
}
```

**Verdict-blocking semantics:**

When ModelMismatch triggers:
- Return `Outcome::Inconclusive` (never Pass/Fail)
- Still report $(\mu, \tau)$ as a *projection summary* for debugging
- Label `EffectPattern` as `Indeterminate`
- Include caveat that the 2D interpretation is unreliable

**Why this matters:**

This gate makes the library **fail-safe under model misspecification**. You don't need to abandon the 2D model to be rigorous—you just need to refuse strong verdicts when the model's fit is demonstrably poor. This is classical "lack of fit" diagnostics applied to security tooling.

### 2.7 Minimum Detectable Effect

The MDE answers: "what's the smallest effect I could reliably detect given the noise level?"

**Derivation:**

Under the GLS estimator $\hat{\beta} = (X^\top \Sigma_n^{-1} X)^{-1} X^\top \Sigma_n^{-1} \Delta$, the MDE with 50% power is:

$$
\text{MDE}_\mu = z_{0.975} \cdot \sqrt{\left( \mathbf{1}^\top \Sigma_n^{-1} \mathbf{1} \right)^{-1}}
$$

$$
\text{MDE}_\tau = z_{0.975} \cdot \sqrt{\left( \mathbf{b}_{\text{tail}}^\top \Sigma_n^{-1} \mathbf{b}_{\text{tail}} \right)^{-1}}
$$

**Relation to $\theta_{\text{floor}}$:**

The MDE values above are per-parameter. The measurement floor $\theta_{\text{floor}}$ is defined for the decision functional $\max_k |(X\beta)_k|$, which requires the familywise correction described in §2.3.4.

**Scaling:** MDE ∝ $1/\sqrt{n}$. With adaptive sampling, we can report MDE dynamically:

```
Current MDE: 15ns (need ~4x more samples to detect 8ns effects)
```

**Relation to quality gates:**

If MDE >> θ after significant sampling, the quality gate "WouldTakeTooLong" will trigger. The MDE provides interpretable context for why.

### 2.8 Research Mode

Research mode answers: "Is there a detectable timing difference above the measurement's noise floor?" without requiring a user-specified threshold.

#### Motivation

Research mode serves users who want to:
- Profile implementations for any timing variation
- Debug suspected leaks without committing to a threshold
- Perform academic analysis of timing behavior

These users want the posterior distribution, not a binary Pass/Fail decision.

#### Design Principle

Research mode uses the same Bayesian machinery as normal mode, but:
1. Sets $\theta_{\text{user}} = 0$, so $\theta_{\text{eff}} = \theta_{\text{floor}}$
2. Reports posterior credible intervals instead of exceedance probabilities
3. Returns a `Research` outcome instead of Pass/Fail
4. Does not hard-stop on model mismatch (exploratory goal), but downgrades interpretation

The decision functional is the same scalar used everywhere:

$$
m(\beta) = \max_k |(X\beta)_k|
$$

#### Algorithm

```rust
fn run_research_mode(calibration: &Calibration, config: &Config) -> ResearchOutcome {
    let mut state = State::new(calibration);

    loop {
        // 1. Collect batch
        state.collect_batch(config.batch_size);

        // 2. Update posterior on β
        let posterior = state.compute_posterior();

        // 3. Recompute θ_floor analytically
        let theta_floor = f64::max(
            calibration.c_floor / (state.n_total as f64).sqrt(),
            calibration.theta_tick,
        );

        // 4. Check model fit (does not terminate, but affects interpretation)
        let residual = state.delta - X * posterior.beta;
        let q_fit = residual.dot(&state.sigma_n_inv * residual);
        let model_mismatch = q_fit > calibration.q_thresh;

        // 5. Compute posterior distribution of m = max|Xβ|
        let m_samples = posterior.sample_max_effect(10_000);
        let m_mean = mean(&m_samples);
        let m_ci = (percentile(&m_samples, 2.5), percentile(&m_samples, 97.5));

        // 6. Check stopping conditions

        // 6a. Confident detection: CI clearly above floor
        if m_ci.0 > theta_floor * 1.1 {
            return ResearchOutcome {
                status: ResearchStatus::EffectDetected,
                max_effect_ns: m_mean,
                max_effect_ci: m_ci,
                theta_floor,
                detectable: true,
                model_mismatch,
                effect: if model_mismatch {
                    posterior.effect_estimate_with_caveat()
                } else {
                    posterior.effect_estimate()
                },
                samples_used: state.n_total,
                quality: compute_quality(&state),
                diagnostics: compute_diagnostics(&state),
            };
        }

        // 6b. Confident non-detection: CI clearly below floor
        if m_ci.1 < theta_floor * 0.9 {
            return ResearchOutcome {
                status: ResearchStatus::NoEffectDetected,
                max_effect_ns: m_mean,
                max_effect_ci: m_ci,
                theta_floor,
                detectable: false,
                model_mismatch,
                effect: if model_mismatch {
                    posterior.effect_estimate_with_caveat()
                } else {
                    posterior.effect_estimate()
                },
                samples_used: state.n_total,
                quality: compute_quality(&state),
                diagnostics: compute_diagnostics(&state),
            };
        }

        // 6c. θ_floor hit the tick floor (can't improve further)
        let tick_floor = calibration.theta_tick;
        if theta_floor <= tick_floor * 1.01 {
            return ResearchOutcome {
                status: ResearchStatus::ResolutionLimitReached,
                max_effect_ns: m_mean,
                max_effect_ci: m_ci,
                theta_floor,
                detectable: m_ci.0 > theta_floor,
                model_mismatch,
                effect: if model_mismatch {
                    posterior.effect_estimate_with_caveat()
                } else {
                    posterior.effect_estimate()
                },
                samples_used: state.n_total,
                quality: compute_quality(&state),
                diagnostics: compute_diagnostics(&state),
            };
        }

        // 6d. Quality gates (except model mismatch, which is tracked but not blocking)
        if let Some(issue) = check_quality_gates_research(&state, &posterior, calibration, config) {
            return ResearchOutcome {
                status: ResearchStatus::QualityIssue(issue),
                max_effect_ns: m_mean,
                max_effect_ci: m_ci,
                theta_floor,
                detectable: m_ci.0 > theta_floor,
                model_mismatch,
                effect: posterior.effect_estimate_with_caveat(),
                samples_used: state.n_total,
                quality: MeasurementQuality::Poor,
                diagnostics: compute_diagnostics(&state),
            };
        }

        // 6e. Budget exhausted
        if state.budget_exhausted(config) {
            return ResearchOutcome {
                status: ResearchStatus::BudgetExhausted,
                max_effect_ns: m_mean,
                max_effect_ci: m_ci,
                theta_floor,
                detectable: m_ci.0 > theta_floor,
                model_mismatch,
                effect: if model_mismatch {
                    posterior.effect_estimate_with_caveat()
                } else {
                    posterior.effect_estimate()
                },
                samples_used: state.n_total,
                quality: compute_quality(&state),
                diagnostics: compute_diagnostics(&state),
            };
        }
    }
}
```

#### Model Mismatch in Research Mode

Unlike normal mode, Research mode does not terminate on model mismatch (since the goal is exploratory). However:

1. **Downgrade interpretation**: Set `EffectPattern::Indeterminate` and add caveat to effect estimate
2. **Flag mismatch**: Set `model_mismatch: true` in output
3. **Caveat the CI**: The `max_effect_ci` is labeled as "under projection model" when mismatch occurs

**Output caveat when model_mismatch is true:**

```
⚠ Model fit is poor (Q = 24.3 > threshold 18.5)
  The (μ, τ) decomposition may not accurately describe the timing pattern.
  The max effect CI is computed under the projection model and should be
  interpreted with caution. Inspect raw quantile differences directly.
```

#### Stopping Conditions

| Condition | Criterion | Status |
|-----------|-----------|--------|
| Effect detected | CI lower bound > 1.1 × θ_floor | `EffectDetected` |
| No effect detected | CI upper bound < 0.9 × θ_floor | `NoEffectDetected` |
| Resolution limit | θ_floor ≈ θ_tick | `ResolutionLimitReached` |
| Quality issue | Any quality gate triggers (except mismatch) | `QualityIssue` |
| Budget exhausted | Time or samples exceeded | `BudgetExhausted` |

The 1.1× and 0.9× margins provide hysteresis to prevent oscillating near the boundary.

#### Interpretation

Research mode answers: "Is there a distributional difference above what we can resolve?"

- **Detectable = true**: The 95% CI for max effect is entirely above θ_floor. There is a timing difference.
- **Detectable = false**: The 95% CI is entirely below θ_floor. No timing difference above the noise floor.
- **Inconclusive**: The CI straddles θ_floor, or we couldn't measure reliably.

When `model_mismatch = true`, the $(\mu, \tau)$ decomposition is unreliable, but the `detectable` flag and `max_effect_ci` (as a projection) may still provide useful guidance.

### 2.9 Dependence Estimation

We estimate dependence length using autocorrelation on the **acquisition stream** (not per-class):

1. Compute ACF on the acquisition stream $\{y_t\}$ or a residualized version
2. Find smallest lag $h$ where $|\rho(h)| < 2/\sqrt{T}$
3. Use this for block length selection and ESS estimation

**Effective sample size:**

$$
\text{ESS} \approx \frac{T}{1 + 2\sum_{h=1}^{\hat{m}} \rho(h)}
$$

where $T$ is the acquisition stream length and $\hat{m}$ is the estimated dependence length.

When ESS << T, warn about high dependence:

```rust
QualityIssue {
    code: IssueCode::HighDependence,
    message: "Estimated dependence length: 47 samples (ESS: 2,100 of 100,000)",
    guidance: "High autocorrelation reduces effective sample size",
}
```

**Important:** Dependence estimation must be on the acquisition stream to match the block bootstrap. Per-class ACF may be retained as a secondary diagnostic but should not drive block length selection.

### 2.10 Discrete Timer Mode

When the timer has low resolution (e.g., Apple Silicon's 41ns cntvct_el0), quantile estimation behaves differently due to tied values.

**Trigger condition:**

Discrete mode triggers when the minimum uniqueness ratio across both classes is below 10%:

$$
\min\left(\frac{|\text{unique}(F)|}{n_F}, \frac{|\text{unique}(R)|}{n_R}\right) < 0.10
$$

**Mid-distribution quantiles:**

Instead of standard quantile estimators, use mid-distribution quantiles which handle ties correctly:

$$
F_{\text{mid}}(x) = F(x) - \frac{1}{2}p(x), \quad \hat{q}^{\text{mid}}_k = F^{-1}_{\text{mid}}(k)
$$

where $p(x)$ is the probability mass at $x$.

**Work in ticks internally:**

In discrete mode, perform computations in **ticks** (timer's native unit):
- $\Delta$, θ, effect sizes all in ticks
- Convert to nanoseconds only for display

**Covariance estimation:**

Use m-out-of-n bootstrap for covariance estimation in discrete mode:

$$
m = \lfloor n^{2/3} \rfloor
$$

This provides consistent variance estimation when the standard CLT doesn't apply.

**Gaussian approximation caveat:**

The Gaussian likelihood is a rougher approximation with discrete data. We report `QualityIssue::DiscreteTimer` on all outputs and frame probabilities as "approximate under Gaussianized model."

### 2.11 Calibration Validation

The Bayesian approach requires empirical validation that posteriors are well-calibrated.

**Calibration testing procedure:**

1. Generate synthetic data with known true effects (0, θ/2, θ, 2θ, 3θ)
2. Run timing-oracle and record `leak_probability`
3. Repeat 100+ times per effect level
4. Check: when we report P%, approximately P% should be true positives

**Expected results:**

| True Effect | Expected P(leak > θ) | Acceptable Range |
|-------------|---------------------|------------------|
| 0 | ~2-5% | 0-10% |
| θ/2 | ~5-15% | 0-25% |
| θ | ~50% | 35-65% |
| 2θ | ~95% | 85-100% |
| 3θ | ~99% | 95-100% |

**Calibration issues to watch for:**

- **Overconfidence**: Reporting high P when true effect < θ → prior too narrow or covariance underestimated
- **Underconfidence**: Reporting low P when true effect > θ → covariance overestimated or prior too wide
- **Discrete mode pathology**: Calibration degradation with coarse timers

---

## 3. Measurement Model

This section describes how timing samples are collected: timer selection, warmup, interleaving, outlier handling, and adaptive batching.

### 3.1 Timer Abstraction

Timing-oracle uses the highest-resolution timer available:

| Platform | Timer | Typical Resolution |
|----------|-------|-------------------|
| x86_64 | rdtsc | ~0.3 ns |
| x86_64 (with perf) | perf_event cycles | ~0.3 ns |
| Apple Silicon | cntvct_el0 | ~41 ns |
| Apple Silicon (with kperf) | PMU cycles | ~1 ns |
| Linux ARM64 | cntvct_el0 | ~40 ns |
| Linux ARM64 (with perf) | perf_event cycles | ~1 ns |

PMU-based timers (kperf, perf_event) require elevated privileges but provide much better resolution.

**Cycle-to-nanosecond conversion:**

Results are reported in nanoseconds for interpretability. The conversion factor is calibrated at startup by measuring a known delay.

### 3.2 Pre-flight Checks

Before measurement begins, several sanity checks detect common problems:

**Timer sanity**: Verify the timer is monotonic and has reasonable resolution. Abort if the timer appears broken.

**Harness sanity (fixed-vs-fixed)**: Split fixed samples in half and run the analysis. If a "leak" is detected between identical inputs, something is wrong with the test harness. This catches bugs that would otherwise produce false positives.

### 3.2.1 Stationarity Check

Non-stationarity breaks bootstrap assumptions. We implement a rolling-variance check:

1. Divide samples into $W$ windows (default: 10)
2. Compute median and IQR within each window
3. Flag `stationarity_suspect = true` if:
   - Max window median differs from min by > $\max(2 \times \text{IQR}, 0.05 \times \text{global\_median})$
   - Or window variance changes monotonically by > 50%

**When stationarity is suspect:**

```rust
QualityIssue {
    code: IssueCode::StationaritySuspect,
    message: "Timing distribution appears to drift during measurement",
    guidance: "Try: longer warmup, CPU frequency pinning, stable system load",
}
```

The test continues (quality downgraded), but results should be interpreted cautiously.

**Warmup**: Run the operation 1,000 times before measurement to warm caches and stabilize frequency scaling.

### 3.3 Measurement Protocol

**Input pre-generation:**

All inputs must be generated *before* the measurement loop:

```rust
// CORRECT: All inputs pre-generated
let fixed_inputs: Vec<_> = (0..n).map(|_| fixed.clone()).collect();
let random_inputs: Vec<_> = (0..n).map(|_| rng.gen()).collect();

for i in 0..n {
    let input = if schedule[i] == Fixed { &fixed_inputs[i] } else { &random_inputs[i] };
    let t0 = timer.now();
    operation(input);
    let t1 = timer.now();
    record(t1 - t0);
}
```

Input generation inside the timed region causes false positives.

**Interleaved randomized sampling:**

Interleave Fixed and Random in shuffled order to prevent systematic drift from biasing one class.

**Acquisition stream storage:**

Store the full acquisition stream $\{(c_t, y_t)\}$ for bootstrap resampling. This is essential for correct covariance estimation (see §2.3.2).

**Outlier handling (winsorization):**

We cap (not drop) outliers to preserve signal from tail-heavy leaks:

1. Compute $t_{\text{cap}}$ = 99.99th percentile from pooled data
2. Cap samples exceeding $t_{\text{cap}}$
3. Winsorization happens before quantile computation

**Quality thresholds:** >0.1% capped → warning; >1% → Acceptable; >5% → TooNoisy.

### 3.4 Adaptive Batching

On platforms with coarse timers, fast operations may complete in fewer ticks than needed for reliable measurement.

**When batching is needed:**

If pilot measurement shows fewer than 5 ticks per call, enable batching.

**Batch size selection:**

$$
K = \text{clamp}\left( \left\lceil \frac{50}{\text{ticks\_per\_call}} \right\rceil, 1, 20 \right)
$$

Maximum of 20 prevents microarchitectural artifacts from accumulating.

**Effect scaling:**

Reported effects are divided by $K$ to give per-operation estimates.

**Threshold scaling:**

When batching is enabled, scale θ accordingly:

$$
\theta_{\text{batch}} = K \cdot \theta
$$

### 3.5 Measurability Thresholds

Some operations are too fast to measure reliably.

**Threshold:**

If ticks per call < 5 even with maximum batching, return `Outcome::Unmeasurable`:

```rust
Outcome::Unmeasurable {
    operation_ns: 8.2,
    threshold_ns: 205.0,
    platform: "Apple Silicon (cntvct)",
    recommendation: "Run with sudo for kperf, or test a more complex operation",
}
```

### 3.6 Anomaly Detection

Detect common user mistakes:

```rust
let value = rand::random();  // Evaluated once!
timing_test! {
    fixed: [0u8; 32],
    random: || value,  // Always returns the same thing ← BUG
    test: |input| compare(&input),
}
```

**Detection:**

Track uniqueness of random inputs by hashing the first 1,000 values:

| Condition | Action |
|-----------|--------|
| All identical | Error to stderr |
| < 50% unique | Warning to stderr |
| Normal entropy | Silent |

---

## 4. Public API

### 4.1 Result Types

The oracle returns one of five outcomes:

```rust
pub enum Outcome {
    /// No timing leak detected. P(leak) < pass_threshold (default 0.05).
    Pass {
        leak_probability: f64,     // P(effect > θ_eff | data)
        effect: EffectEstimate,    // Shift/tail decomposition
        theta_user: f64,           // What the user requested
        theta_eff: f64,            // What was actually tested
        theta_floor: f64,          // Measurement floor at final n
        samples_used: usize,
        quality: MeasurementQuality,
        diagnostics: Diagnostics,
    },

    /// Timing leak confirmed. P(leak) > fail_threshold (default 0.95).
    Fail {
        leak_probability: f64,
        effect: EffectEstimate,
        exploitability: Exploitability,
        theta_user: f64,
        theta_eff: f64,
        theta_floor: f64,
        samples_used: usize,
        quality: MeasurementQuality,
        diagnostics: Diagnostics,
    },

    /// Cannot reach a definitive conclusion.
    Inconclusive {
        reason: InconclusiveReason,
        leak_probability: f64,     // Current posterior
        effect: EffectEstimate,
        theta_user: f64,
        theta_eff: f64,
        theta_floor: f64,
        samples_used: usize,
        quality: MeasurementQuality,
        diagnostics: Diagnostics,
    },

    /// Operation too fast to measure reliably.
    Unmeasurable {
        operation_ns: f64,
        threshold_ns: f64,
        platform: String,
        recommendation: String,
    },

    /// Research mode: posterior analysis without pass/fail decision.
    Research(ResearchOutcome),
}

pub struct ResearchOutcome {
    /// Why we stopped
    pub status: ResearchStatus,

    /// Posterior mean of max_k |(Xβ)_k|
    pub max_effect_ns: f64,

    /// 95% credible interval for max effect
    pub max_effect_ci: (f64, f64),

    /// Smallest effect we could resolve (at final n)
    pub theta_floor: f64,

    /// Is the effect confidently above the floor?
    pub detectable: bool,

    /// True if model fit check failed (Q > threshold)
    /// When true, (μ, τ) decomposition is unreliable
    pub model_mismatch: bool,

    /// Decomposition into shift + tail components
    /// Interpret with caution if model_mismatch is true
    pub effect: EffectEstimate,

    /// Measurement metadata
    pub samples_used: usize,
    pub quality: MeasurementQuality,
    pub diagnostics: Diagnostics,
}

pub enum ResearchStatus {
    /// CI clearly above θ_floor—timing difference detected
    EffectDetected,

    /// CI clearly below θ_floor—no timing difference above noise
    NoEffectDetected,

    /// Hit timer resolution limit; θ_floor is as good as it gets
    ResolutionLimitReached,

    /// Data quality issue
    QualityIssue(InconclusiveReason),

    /// Ran out of time/samples before reaching conclusion
    BudgetExhausted,
}

pub enum InconclusiveReason {
    /// Data is too noisy to reach a conclusion.
    DataTooNoisy { message: String, guidance: String },

    /// Posterior is not converging toward either threshold.
    NotLearning { message: String, guidance: String },

    /// Reaching a conclusion would take too long.
    WouldTakeTooLong { estimated_time_secs: f64, samples_needed: usize, guidance: String },

    /// Requested threshold cannot be achieved with available resources.
    ThresholdUnachievable { theta_user: f64, best_achievable: f64, message: String, guidance: String },

    /// Time budget exhausted.
    TimeBudgetExceeded { current_probability: f64, samples_collected: usize },

    /// Sample budget exhausted.
    SampleBudgetExceeded { current_probability: f64, samples_collected: usize },

    /// Measurement conditions changed between calibration and test.
    ConditionsChanged { drift: ConditionDrift },

    /// Model fit is poor; quantile pattern not well-explained by shift+tail basis.
    ModelMismatch {
        q_statistic: f64,
        q_threshold: f64,
        message: String,
        guidance: String,
    },
}

pub struct ConditionDrift {
    pub variance_ratio_baseline: f64,
    pub variance_ratio_sample: f64,
    pub autocorr_change_baseline: f64,
    pub autocorr_change_sample: f64,
    pub mean_drift_baseline: f64,
    pub mean_drift_sample: f64,
}

pub struct EffectEstimate {
    pub shift_ns: f64,                     // Uniform shift (positive = baseline slower)
    pub tail_ns: f64,                      // Tail effect (positive = baseline has heavier upper tail)
    pub credible_interval_ns: (f64, f64),  // 95% CI for total effect
    pub pattern: EffectPattern,
    /// When true, the (μ, τ) decomposition may be unreliable due to poor model fit
    pub interpretation_caveat: Option<String>,
}

pub enum EffectPattern {
    UniformShift,   // All quantiles shifted equally
    TailEffect,     // Upper quantiles shifted more
    Mixed,          // Both components significant
    Indeterminate,  // Neither significant, or model fit poor
}

pub enum Exploitability {
    Negligible,     // < 100 ns
    PossibleLAN,    // 100–500 ns
    LikelyLAN,      // 500 ns – 20 μs
    PossibleRemote, // > 20 μs
}

pub enum MeasurementQuality {
    Excellent,  // MDE < 5 ns
    Good,       // MDE 5-20 ns
    Poor,       // MDE 20-100 ns
    TooNoisy,   // MDE > 100 ns
}

pub struct Diagnostics {
    // Core statistical checks
    pub dependence_length: usize,
    pub effective_sample_size: usize,
    pub stationarity_ratio: f64,
    pub stationarity_ok: bool,
    pub model_fit_chi2: f64,       // Chi-squared statistic for model fit (Q = r'Σ⁻¹r)
    pub model_fit_threshold: f64,  // Bootstrap-calibrated threshold or χ²₇(0.99) = 18.48 fallback
    pub model_fit_ok: bool,
    pub outlier_rate_baseline: f64,
    pub outlier_rate_sample: f64,
    pub outlier_asymmetry_ok: bool,

    // Timer info
    pub discrete_mode: bool,
    pub timer_resolution_ns: f64,
    pub duplicate_fraction: f64,

    // Measurement summary
    pub preflight_ok: bool,
    pub calibration_samples: usize,
    pub total_time_secs: f64,
    pub warnings: Vec<String>,
    pub quality_issues: Vec<QualityIssue>,
    pub preflight_warnings: Vec<PreflightWarningInfo>,

    // Reproduction info (for verbose/debug output)
    pub seed: Option<u64>,           // RNG seed used for reproducibility
    pub attacker_model: Option<String>,
    pub threshold_ns: f64,
    pub timer_name: String,
    pub platform: String,
}

pub struct QualityIssue {
    pub code: IssueCode,
    pub message: String,
    pub guidance: String,
}

pub enum IssueCode {
    HighDependence, LowEffectiveSamples, StationaritySuspect, DiscreteTimer,
    SmallSampleDiscrete, HighGeneratorCost, LowUniqueInputs, QuantilesFiltered,
    ThresholdElevated, ThresholdClamped, HighWinsorRate, ModelMismatch,
}

/// Preflight warning with severity and category for structured handling.
pub struct PreflightWarningInfo {
    pub category: PreflightCategory,
    pub severity: PreflightSeverity,
    pub message: String,
    pub guidance: Option<String>,
}

/// Severity of preflight warnings.
pub enum PreflightSeverity {
    /// Sampling efficiency issue - results still valid but may need more samples.
    Informational,
    /// Statistical assumption violation - undermines result confidence.
    ResultUndermining,
}

/// Category of preflight check that generated the warning.
pub enum PreflightCategory {
    TimerSanity,    // Timer sanity checks
    Sanity,         // Fixed-vs-Fixed consistency
    Generator,      // Generator cost comparison
    Autocorrelation,// Serial dependence check
    System,         // System configuration (CPU governor, etc.)
    Resolution,     // Timer resolution check
}
```

### 4.2 Configuration

#### Attacker Model Presets

The most important configuration choice is your **attacker model**, which determines what size leak you consider "negligible."

**There is no single correct θ.** Your choice of preset is a statement about your threat model.

```rust
pub enum AttackerModel {
    /// Co-resident attacker with cycle-level timing.
    /// θ = 0.6 ns (~2 cycles @ 3GHz)
    ///
    /// Use for: SGX enclaves, cross-VM attacks, containers on shared hosts,
    /// hyperthreading siblings, local privilege escalation.
    SharedHardware,

    /// Post-quantum crypto sentinel.
    /// θ = 3.3 ns (~10 cycles @ 3GHz)
    ///
    /// Use for: ML-KEM, ML-DSA, and other post-quantum implementations.
    /// Catches KyberSlash-class (~20 cycle) vulnerabilities.
    PostQuantumSentinel,

    /// Adjacent network attacker (LAN, HTTP/2 multiplexing).
    /// θ = 100 ns
    ///
    /// Use for: Internal services, microservices, HTTP/2 APIs where
    /// Timeless Timing Attacks apply (request multiplexing eliminates jitter).
    AdjacentNetwork,

    /// Remote network attacker (general internet).
    /// θ = 50 μs
    ///
    /// Use for: Public APIs, web services, legacy HTTP/1.1 services.
    RemoteNetwork,

    /// Research mode: no threshold, reports posterior directly.
    ///
    /// Use for: Profiling, debugging, academic analysis.
    /// Returns ResearchOutcome instead of Pass/Fail.
    Research,

    /// Custom threshold in nanoseconds.
    Custom { threshold_ns: f64 },
}
```

**Preset summary:**

| Preset | θ | Use case |
|--------|---|----------|
| `SharedHardware` | 0.6 ns (~2 cycles @ 3GHz) | SGX, cross-VM, containers, hyperthreading |
| `PostQuantumSentinel` | 3.3 ns (~10 cycles @ 3GHz) | Post-quantum crypto (ML-KEM, ML-DSA) |
| `AdjacentNetwork` | 100 ns | LAN, HTTP/2 (Timeless Timing Attacks) |
| `RemoteNetwork` | 50 μs | Internet, legacy services |
| `Research` | θ_floor (dynamic) | Profiling, debugging (not for CI) |

**Sources:**

- **Crosby et al. (2009)**: "Opportunities and Limits of Remote Timing Attacks." ACM TISSEC.
- **Van Goethem et al. (2020)**: "Timeless Timing Attacks." USENIX Security. Shows HTTP/2 request multiplexing enables LAN-like precision over the internet.
- **Kario**: Argues timing differences as small as one clock cycle can be detectable when sharing hardware.

**Recommended usage:**

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};

let inputs = InputPair::new(
    || [0u8; 32],
    || rand::random::<[u8; 32]>(),
);

// Choose based on deployment scenario
let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .test(inputs, |data| operation(data));

match outcome {
    Outcome::Pass { leak_probability, effect, theta_user, theta_eff, .. } => {
        if theta_eff > theta_user {
            println!("No leak at θ_eff={:.1}ns (requested {:.1}ns)", theta_eff, theta_user);
        } else {
            println!("No leak: P(leak)={:.1}%", leak_probability * 100.0);
        }
    }
    Outcome::Fail { leak_probability, effect, exploitability, .. } => {
        println!("Leak detected: P(leak)={:.1}%, {:?}", leak_probability * 100.0, exploitability);
    }
    Outcome::Inconclusive { reason, leak_probability, .. } => {
        println!("Inconclusive: P(leak)={:.1}%", leak_probability * 100.0);
        match reason {
            InconclusiveReason::ModelMismatch { guidance, .. } => {
                println!("Model fit issue: {}", guidance);
            }
            _ => {}
        }
    }
    Outcome::Unmeasurable { recommendation, .. } => {
        println!("Skipping: {}", recommendation);
    }
    Outcome::Research(res) => {
        println!("Max effect: {:.1} ns [{:.1}, {:.1}]",
                 res.max_effect_ns, res.max_effect_ci.0, res.max_effect_ci.1);
        println!("Detectable: {}", res.detectable);
        if res.model_mismatch {
            println!("⚠ Model fit is poor; (μ,τ) decomposition unreliable");
        }
    }
}
```

#### TimingOracle Configuration

The main configuration type. Use `for_attacker()` as the entry point:

```rust
impl TimingOracle {
    /// Create oracle for a specific attacker model (recommended entry point).
    pub fn for_attacker(model: AttackerModel) -> Self;

    // Decision thresholds
    pub fn pass_threshold(self, p: f64) -> Self;  // P(leak) < p → Pass (default 0.05)
    pub fn fail_threshold(self, p: f64) -> Self;  // P(leak) > p → Fail (default 0.95)

    // Resource limits (optional, for CI or constrained environments)
    pub fn time_budget(self, duration: Duration) -> Self;
    pub fn max_samples(self, n: usize) -> Self;

    // Run the test
    pub fn test<T, F>(self, inputs: InputPair<T>, operation: F) -> Outcome
    where
        F: FnMut(&T);

    // Research mode entry point
    pub fn research() -> Self;
}
```

All configuration is optional. The defaults work well for most use cases.

| Option | Default | Description |
|--------|---------|-------------|
| `pass_threshold` | 0.05 | P(leak) below this → Pass |
| `fail_threshold` | 0.95 | P(leak) above this → Fail |
| `time_budget` | 30 seconds | Maximum time to spend |
| `max_samples` | 100,000 | Maximum samples per class |

### 4.3 Macro API

```rust
use timing_oracle::{timing_test_checked, TimingOracle, AttackerModel, Outcome};

let outcome = timing_test_checked! {
    oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork),
    baseline: || [0u8; 32],
    sample: || rand::random::<[u8; 32]>(),
    measure: |input| {
        my_crypto_function(&input);
    },
};

match outcome {
    Outcome::Pass { .. } => println!("No timing leak detected"),
    Outcome::Fail { leak_probability, exploitability, .. } => {
        panic!("Timing leak: P={:.1}%, {:?}", leak_probability * 100.0, exploitability);
    }
    Outcome::Inconclusive { reason, leak_probability, .. } => {
        println!("Inconclusive (P={:.1}%): {:?}", leak_probability * 100.0, reason);
    }
    Outcome::Unmeasurable { recommendation, .. } => {
        println!("Cannot measure: {}", recommendation);
    }
    Outcome::Research(_) => unreachable!("Not in research mode"),
}
```

**For CI assertions:**

```rust
// Strict: fail on Fail, pass on Pass, fail on Inconclusive
assert!(matches!(outcome, Outcome::Pass { .. }));

// Lenient: fail only on Fail
assert!(!matches!(outcome, Outcome::Fail { .. }));
```

### 4.4 Builder API

```rust
use timing_oracle::{TimingOracle, AttackerModel, helpers::InputPair};

let inputs = InputPair::new(
    || [0u8; 32],
    || rand::random::<[u8; 32]>(),
);

let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .test(inputs, |data| my_function(data));
```

### 4.5 Research Mode API

```rust
use timing_oracle::{TimingOracle, Outcome, ResearchStatus};

let outcome = TimingOracle::research()
    .time_budget(Duration::from_secs(60))
    .test(inputs, |data| my_function(data));

match outcome {
    Outcome::Research(res) => {
        println!("Max effect: {:.1} ns", res.max_effect_ns);
        println!("95% CI: [{:.1}, {:.1}] ns", res.max_effect_ci.0, res.max_effect_ci.1);
        println!("Measurement floor: {:.1} ns", res.theta_floor);

        if res.model_mismatch {
            println!("⚠ Model fit is poor (interpret μ,τ with caution)");
        }

        match res.status {
            ResearchStatus::EffectDetected => {
                println!("✓ Timing difference detected (CI > θ_floor)");
            }
            ResearchStatus::NoEffectDetected => {
                println!("✗ No timing difference above {:.1} ns", res.theta_floor);
            }
            ResearchStatus::ResolutionLimitReached => {
                println!("⚠ Hit timer resolution limit");
                if res.detectable {
                    println!("  Effect appears present but cannot be precisely bounded");
                }
            }
            ResearchStatus::QualityIssue(reason) => {
                println!("⚠ Quality issue: {:?}", reason);
            }
            ResearchStatus::BudgetExhausted => {
                println!("⚠ Budget exhausted");
                println!("  Current estimate: {:.1} ns [{:.1}, {:.1}]",
                         res.max_effect_ns, res.max_effect_ci.0, res.max_effect_ci.1);
            }
        }
    }
    _ => unreachable!("Research mode always returns Research variant"),
}
```

---

## 5. Interpreting Results

### 5.1 Leak Probability

The `leak_probability` is $P(\max_k |(X\beta)_k| > \theta_{\text{eff}} \mid \Delta)$—the posterior probability of an effect exceeding your effective threshold.

| Probability | Interpretation | Outcome |
|-------------|----------------|---------|
| < 5% | Confident no leak | **Pass** |
| 5–50% | Probably safe, uncertain | Inconclusive (needs more data) |
| 50–95% | Probably leaking, uncertain | Inconclusive (needs more data) |
| > 95% | Confident leak | **Fail** |

The 5%/95% boundaries are configurable via `pass_threshold` and `fail_threshold`.

**Important:** Pass/Fail are only emitted when all quality gates pass, including the model fit gate. If you receive Inconclusive with `ModelMismatch`, the leak probability is reported as "estimate under the model" but should not be interpreted as a confident claim.

### 5.2 Understanding θ_user vs θ_eff vs θ_floor

When interpreting results, pay attention to three threshold values:

| Value | Meaning |
|-------|---------|
| θ_user | What you requested |
| θ_floor | What the measurement could resolve |
| θ_eff | What was actually tested (max of the above) |

**When θ_eff = θ_user:** The measurement achieved your requested sensitivity. Results directly answer "is there a leak > θ_user?"

**When θ_eff > θ_user:** The measurement couldn't achieve your requested sensitivity. Results answer "is there a leak > θ_eff?" which is a weaker statement. A `Pass` means "no leak above θ_eff," but there could still be a leak between θ_user and θ_eff.

**Example:**

```
Requested: θ_user = 10 ns
Achieved:  θ_floor = 41 ns (timer-limited)
Tested:    θ_eff = 41 ns

Pass with P(leak) = 2% means:
  ✓ No leak > 41 ns
  ? Unknown for 10-41 ns range
```

### 5.3 Effect Size

- **Shift (μ)**: Uniform timing difference. Cause: different code path, branch on secret.
- **Tail (τ)**: Upper quantiles affected more. Cause: cache misses, secret-dependent memory access.

Both reported in nanoseconds with 95% credible intervals.

**Effect patterns:**

| Pattern | Signature | Typical Cause |
|---------|-----------|---------------|
| UniformShift | μ significant, τ ≈ 0 | Branch on secret bit |
| TailEffect | μ ≈ 0, τ significant | Cache-timing leak |
| Mixed | Both significant | Multiple leak sources |
| Indeterminate | Neither significant, or model fit poor | No detectable leak, or pattern not captured |

**Model mismatch caveat:** When `interpretation_caveat` is set on the effect estimate, the $(\mu, \tau)$ decomposition may not accurately describe the timing pattern. Inspect raw quantile differences for asymmetric or multi-modal effects.

### 5.4 Exploitability

Rough risk assessment based on effect size:

| Level | Effect Size | Practical Impact |
|-------|-------------|------------------|
| Negligible | < 100 ns | Requires ~10k+ queries to exploit over LAN |
| PossibleLAN | 100–500 ns | Exploitable on LAN with statistical methods |
| LikelyLAN | 500 ns – 20 μs | Readily exploitable on local network |
| PossibleRemote | > 20 μs | Potentially exploitable over internet |

**Caveat**: Based on Crosby et al. (2009). Modern attacks may achieve better resolution.

### 5.5 Inconclusive Results

Inconclusive means "couldn't reach 95% confidence either way." This is different from "don't know"—you still get a probability estimate.

| Reason | Meaning | Suggested Action |
|--------|---------|------------------|
| DataTooNoisy | Posterior ≈ prior | Use better timer, reduce system noise |
| NotLearning | Data not reducing uncertainty | Check for systematic measurement issues |
| WouldTakeTooLong | Need too many samples | Accept uncertainty or adjust θ |
| ThresholdUnachievable | θ_user < θ_floor even at max budget | Use cycle counter or accept elevated θ |
| TimeBudgetExceeded | Ran out of time | Increase time budget |
| SampleBudgetExceeded | Ran out of samples | Increase sample budget |
| ConditionsChanged | Calibration assumptions violated | Run in isolation, reduce system load |
| **ModelMismatch** | Quantile pattern not well-explained by 2D basis | Inspect raw quantile differences; consider asymmetric leak patterns |

**For CI:**

```rust
match outcome {
    Outcome::Inconclusive { leak_probability, reason, .. } => {
        match reason {
            InconclusiveReason::ModelMismatch { .. } => {
                // Model doesn't fit—could be a leak with unusual shape
                eprintln!("Warning: Possible leak with atypical pattern (P={:.0}%)", 
                         leak_probability * 100.0);
            }
            _ if leak_probability > 0.5 => {
                // Probably leaking but not confident—treat as failure
                panic!("Likely timing leak (P={:.0}%)", leak_probability * 100.0);
            }
            _ => {
                // Probably safe but not confident—treat as warning
                eprintln!("Warning: Could not confirm constant-time behavior");
            }
        }
    }
    _ => {}
}
```

### 5.6 Research Mode Results

Research mode results should be interpreted as exploratory analysis:

| Status | Meaning | Confidence |
|--------|---------|------------|
| EffectDetected | CI clearly above θ_floor | High—there is a timing difference |
| NoEffectDetected | CI clearly below θ_floor | High—no detectable difference |
| ResolutionLimitReached | Hit timer limit | Medium—limited by hardware |
| QualityIssue | Measurement problems | Low—interpret with caution |
| BudgetExhausted | Ran out of resources | Variable—check CI width |

**Key insight:** `detectable` is always relative to θ_floor. A `detectable = false` result means "no effect above the noise floor," which depends on timer quality and sample count.

**Model mismatch in Research mode:** When `model_mismatch = true`:
- The `detectable` flag and `max_effect_ci` may still provide useful guidance
- The $(\mu, \tau)$ decomposition is unreliable
- Consider inspecting raw quantile differences directly

### 5.7 Quality Assessment

Quality reflects how trustworthy the measurement is:

| Quality | Meaning | MDE |
|---------|---------|-----|
| Excellent | Optimal measurement conditions | < 5 ns |
| Good | Minor issues, results valid | 5-20 ns |
| Poor | Results may lack precision | 20-100 ns |
| TooNoisy | Results may be unreliable | > 100 ns |

Common issues and fixes:

| Issue | Symptom | Fix |
|-------|---------|-----|
| HighDependence | ESS << n | Isolate to dedicated core |
| StationaritySuspect | Drift over time | Pin CPU frequency, longer warmup |
| DiscreteTimer | < 10% unique values | Use cycle counter if available |
| ThresholdElevated | θ_eff > θ_user | Use cycle counter or accept elevated θ |
| HighWinsorRate | > 1% outliers | Reduce system interference |
| ModelMismatch | High residual Q | Leak may have unusual shape; inspect quantiles |

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $\{(c_t, y_t)\}$ | Acquisition stream: class labels and timing measurements |
| $T$ | Acquisition stream length ($\approx 2n$) |
| $F$, $R$ | Per-class sample sets (filtered from stream) |
| $\Delta$ | 9-vector of signed quantile differences |
| $\hat{q}_F(k)$, $\hat{q}_R(k)$ | Empirical quantiles for Fixed and Random classes |
| $X$ | $9 \times 2$ design matrix $[\mathbf{1} \mid \mathbf{b}_{\text{tail}}]$ |
| $\beta = (\mu, \tau)^\top$ | Effect parameters: shift and tail |
| $\Sigma_n$ | Covariance matrix at sample size $n$ |
| $\Sigma_{\text{rate}}$ | Covariance rate: $\Sigma_n = \Sigma_{\text{rate}} / n$ |
| $\Sigma_{\text{pred,rate}}$ | Prediction covariance rate: $X (X^\top \Sigma_{\text{rate}}^{-1} X)^{-1} X^\top$ |
| $\Lambda_0$ | Prior covariance for $\beta$ |
| $\Lambda_{\text{post}}$ | Posterior covariance for $\beta$ |
| $\theta_{\text{user}}$ | User-requested threshold |
| $\theta_{\text{floor}}$ | Measurement floor (smallest resolvable effect) |
| $c_{\text{floor}}$ | Floor-rate constant: $\theta_{\text{floor,stat}} = c_{\text{floor}} / \sqrt{n}$ |
| $\theta_{\text{tick}}$ | Timer resolution component of floor |
| $\theta_{\text{eff}}$ | Effective threshold used for inference |
| $m(\beta)$ | Decision functional: $\max_k |(X\beta)_k|$ |
| $Q$ | Model fit statistic: $r^\top \Sigma_n^{-1} r$ |
| $Q_{\text{thresh}}$ | Bootstrap-calibrated mismatch threshold (99th percentile) |
| $\hat{b}$ | Block length (Politis-White, on acquisition stream) |
| MDE | Minimum detectable effect |
| $n$ | Samples per class |
| $B$ | Bootstrap iterations |
| ESS | Effective sample size |

## Appendix B: Constants

| Constant | Default | Rationale |
|----------|---------|-----------|
| Deciles | {0.1, ..., 0.9} | Nine quantile positions |
| Bootstrap iterations | 2,000 | Covariance estimation accuracy |
| Monte Carlo samples (leak prob) | 10,000 | Leak probability integration |
| Monte Carlo samples (c_floor) | 50,000 | Floor-rate constant estimation |
| Batch size | 1,000 | Adaptive iteration granularity |
| Calibration samples | 5,000 | Initial covariance estimation |
| Pass threshold | 0.05 | 95% confidence of no leak |
| Fail threshold | 0.95 | 95% confidence of leak |
| Variance ratio gate | 0.5 | Posterior ≈ prior detection |
| Model mismatch percentile | 99th | Q_thresh from bootstrap distribution |
| Model mismatch fallback | χ²₇,₀.₉₉ ≈ 18.48 | Conservative lack-of-fit cutoff |
| Block length cap | min(3√T, T/3) | Prevent degenerate blocks |
| Discrete threshold | 10% unique | Trigger discrete mode |
| Min ticks per call | 5 | Measurability floor |
| Max batch size | 20 | Limit μarch artifacts |
| Default θ | 100 ns (AdjacentNetwork) | Practical significance |
| Default time budget | 30 s | Maximum runtime |
| Default sample budget | 100,000 | Maximum samples |
| CI hysteresis (Research) | 10% | Margin for detection decisions |
| Default RNG seed | 0x74696D696E67 | "timing" in ASCII |

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

**Existing tools:**

10. dudect (C): https://github.com/oreparaz/dudect
11. dudect-bencher (Rust): https://github.com/rozbb/dudect-bencher

---

## Appendix D: Changelog from v3

### Summary of Changes

| Aspect | v3 | v4 |
|--------|----|----|
| Bootstrap resampling | Per-class paired indices | Stream-based (acquisition order) |
| Block length estimation | Per-class ACF | Acquisition stream ACF |
| Prior scale | σ = 1.12θ | σ = 1.12 × θ_eff |
| θ = 0 handling | Clamp to timer resolution | Research mode with CI-based semantics |
| Sub-floor θ | Silent clamp | Explicit elevation + ThresholdElevated warning |
| θ_floor computation | Not defined | Floor-rate constant + analytical 1/√n scaling |
| θ_floor dynamics | N/A | Analytical update (no per-iteration MC) |
| Model fit check | Not present | Q statistic with bootstrap-calibrated threshold |
| Quality gate order | After decisions | Before decisions (verdict-blocking) |
| New quality gates | N/A | ThresholdUnachievable, ModelMismatch |
| Research mode | "θ → 0" (degenerate) | Proper posterior + CI reporting |
| Research mismatch | N/A | Tracked but non-blocking; downgrades interpretation |
| RNG policy | Unspecified | Deterministic seeding (reproducible results) |
| Output types | 4 variants | 5 variants (+ Research) |
| InconclusiveReason | 6 variants | 8 variants (+ ThresholdUnachievable, ModelMismatch) |

### Key Fixes

1. **Stream-based bootstrap (Amendment 1)**: v3's per-class paired resampling didn't preserve adjacency in the acquisition process, potentially underestimating variance and producing overconfident posteriors. v4 resamples the acquisition stream directly, then splits by class—correctly preserving common-mode drift and cross-class correlation.

2. **Block length on acquisition stream**: v3 computed block length per-class. v4 computes it on the acquisition stream to match the bootstrap resampling.

3. **Model mismatch gate (Amendment 2)**: v3 had no check for whether the 2D basis could represent the observed quantile pattern. v4 computes a whitened residual statistic Q and compares to a bootstrap-calibrated threshold, returning Inconclusive when fit is poor. This makes the library fail-safe under model misspecification.

4. **Verdict-blocking semantics (Amendment 3)**: v3's quality gates could be checked after decisions in some code paths. v4 explicitly requires all gates to pass before Pass/Fail, ensuring strong verdicts only when assumptions are verified.

5. **Deterministic RNG (Amendment 4)**: v3 had unspecified RNG behavior. v4 requires deterministic seeding so that identical timing samples + config produce identical results. This eliminates CI flakiness from algorithmic randomness.

6. **Floor-rate constant (Amendment 5)**: v3 would have required per-iteration Monte Carlo for θ_floor. v4 computes c_floor once at calibration and uses analytical $1/\sqrt{n}$ scaling, improving reproducibility and performance.

7. **θ = 0 degeneracy**: v3's Research mode would always return ~100% leak probability because P(|X| > 0) = 1 for any continuous distribution. v4 uses CI-based semantics with θ_floor as the reference.

8. **"Fail from prior" pathology**: v3 could return Fail when θ_user ≪ θ_floor because the prior had high mass above θ_user. v4 checks quality gates before decisions and uses θ_eff consistently.

9. **Per-coordinate vs familywise floor**: v3's implicit MDE used per-coordinate z_{0.975}. v4 uses proper familywise correction via the floor-rate constant computed from max|Z| distribution.

10. **Silent threshold clamping**: v3 would clamp θ to timer resolution without telling the user. v4 reports θ_user, θ_eff, and θ_floor explicitly, with ThresholdElevated warnings.

### Design Philosophy

The guiding principle for v4 is:

> **CI verdicts should almost never be confidently wrong.**

When assumptions are violated (dependence mis-modeled, drift, misspecified effect shape), the oracle prefers **Inconclusive with an actionable reason** rather than emitting a confident Pass/Fail. This makes timing-oracle suitable for security-critical applications where false confidence is worse than admitted uncertainty.
