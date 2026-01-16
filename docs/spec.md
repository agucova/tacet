# timing-oracle Specification (v3)

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

### Design Goals

- **Interpretable**: Output is a probability (0–100%), not a t-statistic
- **Adaptive**: Automatically collects more samples when uncertain, stops early when confident
- **CI-friendly**: Three-way output prevents flaky tests; inconclusive results are explicit
- **Fast**: Early stopping means clear cases finish quickly
- **Portable**: Handles different timer resolutions via adaptive batching

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

For discrete timers with many tied values, use **mid-distribution quantiles** (see §2.5) which correctly handle atoms.

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
│   • Set prior scale    • Update Δ         • Fail (P>95%)        │
│   • Warmup caches      • Scale Σ by 1/n   • Inconclusive        │
│   • Pre-flight checks  • Compute P(>θ)                          │
│                        • Check quality                           │
│                        • Check stopping                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Phase 1: Calibration** (runs once)
- Collect initial samples to characterize measurement noise
- Estimate covariance structure via block bootstrap
- Compute "covariance rate" $\Sigma_{\text{rate}}$ that scales as $\Sigma = \Sigma_{\text{rate}} / n$
- Set prior scale based on observed noise level
- Run pre-flight checks (timer sanity, harness sanity, stationarity)

**Phase 2: Adaptive Loop** (iterates until decision)
- Collect batches of samples
- Update quantile estimates from all data collected so far
- Scale covariance: $\Sigma_n = \Sigma_{\text{rate}} / n$
- Compute posterior and $P(\text{effect} > \theta)$
- Check decision boundaries (P > 95% → Fail, P < 5% → Pass)
- Check quality gates (posterior ≈ prior → Inconclusive)
- Check time/sample budgets

**Why this structure?**

The key insight is that covariance scales as $1/n$ for quantile estimators. By estimating the covariance *rate* once during calibration, we can cheaply update the posterior as more data arrives—no re-bootstrapping needed. This makes adaptive sampling computationally tractable.

### 2.3 Calibration Phase

The calibration phase runs once at startup to characterize measurement noise.

**Sample collection:**

Collect $n_{\text{cal}}$ samples per class (default: 5,000). This is enough to estimate covariance structure reliably while keeping calibration fast.

**Covariance estimation via block bootstrap:**

Timing measurements exhibit autocorrelation—nearby samples are more similar than distant ones due to cache state, frequency scaling, etc. Standard bootstrap assumes i.i.d. samples, underestimating variance. We use block bootstrap to preserve autocorrelation structure.

**Politis-White automatic block length selection:**

We use the Politis-White algorithm² to select the optimal block length, computing it separately for each class and taking the maximum:

$$
\hat{b} = \lceil \max(b_F^{\text{opt}}, b_R^{\text{opt}}) \rceil
$$

The algorithm estimates the optimal block length by analyzing the autocorrelation structure:

**Step 1: Determine truncation lag $m$**

Let $k_n = \max(5, \lfloor \log_{10}(n) \rfloor)$ and $m_{\max} = \lceil \sqrt{n} \rceil + k_n$.

Compute autocorrelations $\hat{\rho}_k$ for $k = 0, \ldots, m_{\max}$. Find the first lag $m^*$ where $k_n$ consecutive autocorrelations fall within the conservative band $\pm 2\sqrt{\log_{10}(n)/n}$. Set $m = \min(2 \cdot \max(m^*, 1), m_{\max})$.

**Step 2: Compute spectral quantities**

Using a flat-top (trapezoidal) kernel $h(x) = \min(1, 2(1 - |x|))$:

$$
\hat{\sigma}^2 = \sum_{k=-m}^{m} h\left(\frac{k}{m}\right) \hat{\gamma}_k, \quad g = \sum_{k=-m}^{m} h\left(\frac{k}{m}\right) |k| \, \hat{\gamma}_k
$$

where $\hat{\gamma}_k = n^{-1} \sum_{i=k+1}^{n} (x_i - \bar{x})(x_{i-k} - \bar{x})$ is the sample autocovariance.

**Step 3: Compute optimal block length**

$$
b^{\text{opt}} = \left( \frac{g^2}{(\hat{\sigma}^2)^2} \right)^{1/3} n^{1/3}
$$

Capped at $b_{\max} = \min(3\sqrt{n}, n/3)$ to prevent degenerate blocks.

² Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection for the Dependent Bootstrap." Econometric Reviews 23(1):53–70.

**Bootstrap procedure:**

For each of $B = 2{,}000$ bootstrap iterations:
1. Generate block indices; apply to both classes (paired resampling)
2. Compute quantile vectors $\hat{q}_F^*$, $\hat{q}_R^*$ from resampled data
3. Record $\Delta^* = \hat{q}_F^* - \hat{q}_R^*$

Compute sample covariance of the $\Delta^*$ vectors using Welford's online algorithm³ (numerically stable).

³ Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419–420.

**Covariance rate:**

The covariance of quantile estimators scales as $1/n$. We compute the "rate":

$$
\Sigma_{\text{rate}} = \hat{\Sigma}_{\text{cal}} \cdot n_{\text{cal}}
$$

During the adaptive loop, with $n$ total samples:

$$
\Sigma_n = \Sigma_{\text{rate}} / n
$$

This allows cheap posterior updates without re-bootstrapping.

**Paired resampling:**

We use **paired resampling**: generate one set of block indices, apply it to both per-class sequences:

```
indices = sample_block_indices(n, block_length)
F* = F[indices]
R* = R[indices]
```

This preserves cross-covariance from common-mode noise (thermal drift, frequency scaling) that affects both classes at similar experimental positions.

**Numerical stability:**

With discrete timers or few unique values, some quantiles may have near-zero variance, making $\Sigma$ ill-conditioned. We regularize by ensuring a minimum diagonal value:

$$
\sigma^2_i \leftarrow \max(\sigma^2_i, 0.01 \cdot \bar{\sigma}^2) + \varepsilon
$$

where $\bar{\sigma}^2 = \text{tr}(\Sigma)/9$ and $\varepsilon = 10^{-10} + \bar{\sigma}^2 \cdot 10^{-8}$.

**Prior scale:**

The prior on effect parameters $\beta = (\mu, \tau)$ is set proportional to the threshold:

$$
\sigma_{\text{prior}} = 2\theta
$$

where θ is `min_effect_of_concern`. This ensures $P(|\beta| > \theta \mid \text{prior}) \approx 62\%$, representing genuine uncertainty.

**Calibration output:**

```rust
struct Calibration {
    sigma_rate: Matrix9x9,      // Covariance rate: Σ = Σ_rate / n
    block_length: usize,        // Estimated autocorrelation length
    prior_cov: Matrix2x2,       // Prior covariance for β
    timer_resolution_ns: f64,   // Detected tick size
    samples_per_second: f64,    // For time estimation
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

**Design limitation:** The antisymmetric tail basis assumes symmetric patterns. Real cache-timing leaks often show asymmetric upper-tail effects. The model *can* fit these (μ absorbs the mean shift, τ captures the spread), but interpretation of τ becomes less intuitive. A future version could add a third basis for asymmetric effects.

#### Likelihood

$$
\Delta \mid \beta \sim \mathcal{N}(X\beta, \Sigma_n)
$$

where $\Sigma_n = \Sigma_{\text{rate}} / n$ is the scaled covariance.

#### Prior

$$
\beta \sim \mathcal{N}(0, \Lambda_0), \quad \Lambda_0 = \text{diag}(\sigma_{\text{prior}}^2, \sigma_{\text{prior}}^2)
$$

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

The key question is: "what's the probability of a leak exceeding θ?" We compute:

$$
P(\text{leak} > \theta \mid \Delta) = P\bigl(\max_k |(X\beta)_k| > \theta \;\big|\; \Delta\bigr)
$$

Since the posterior is Gaussian, we compute this by Monte Carlo integration:

```python
samples = draw_from_normal(beta_post, Lambda_post, n=1000)
count = 0
for beta in samples:
    pred = X @ beta  # 9-vector of predicted quantile differences
    max_effect = max(abs(pred))
    if max_effect > theta:
        count += 1
leak_probability = count / 1000
```

This is O(1)—we're drawing from a 2D Gaussian and doing simple arithmetic.

**Interpreting the probability:**

This is a **posterior probability**, not a p-value. When we report "72% probability of a leak," we mean: given the data and our model, 72% of the posterior mass corresponds to effects exceeding θ.

This is calibrated by construction under the Gaussian model. Empirical validation (§2.8) confirms calibration holds in practice.

### 2.5 Adaptive Sampling Loop

The core innovation: collect samples until confident, with natural early stopping.

```rust
fn run_adaptive(calibration: &Calibration, theta: f64, config: &Config) -> Outcome {
    let mut state = State::new(calibration);

    loop {
        // 1. Collect a batch
        state.collect_batch(config.batch_size);

        // 2. Update posterior (cheap - no bootstrap)
        let posterior = state.compute_posterior(theta);

        // 3. Check decision boundaries
        if posterior.leak_probability > 0.95 {
            return Outcome::Fail {
                leak_probability: posterior.leak_probability,
                effect: posterior.effect_estimate(),
                samples_used: state.n_total,
            };
        }
        if posterior.leak_probability < 0.05 {
            return Outcome::Pass {
                leak_probability: posterior.leak_probability,
                effect: posterior.effect_estimate(),
                samples_used: state.n_total,
            };
        }

        // 4. Check quality gates
        if let Some(reason) = check_quality_gates(&state, &posterior, config) {
            return Outcome::Inconclusive {
                reason,
                leak_probability: posterior.leak_probability,
                effect: posterior.effect_estimate(),
                samples_used: state.n_total,
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
5. Monte Carlo integration: O(1000)

Total: dominated by quantile computation, which can be optimized with streaming quantile algorithms.

**Batch size selection:**

Default batch size is 1,000 samples. This balances:
- Large enough for stable quantile estimates per batch
- Small enough for responsive early stopping
- Matches typical calibration sample count

### 2.6 Quality Gates

Quality gates detect when data is too poor to reach a confident decision.

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

#### Gate 4: Time Budget Exceeded

```rust
if state.elapsed() > config.time_budget {
    return Some(InconclusiveReason::TimeBudgetExceeded {
        current_probability: posterior.leak_probability,
        samples_collected: state.n_total,
    });
}
```

#### Gate 5: Sample Budget Exceeded

```rust
if state.n_total >= config.max_samples {
    return Some(InconclusiveReason::SampleBudgetExceeded {
        current_probability: posterior.leak_probability,
        samples_collected: state.n_total,
    });
}
```

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

**Scaling:** MDE ∝ $1/\sqrt{n}$. With adaptive sampling, we can report MDE dynamically:

```
Current MDE: 15ns (need ~4x more samples to detect 8ns effects)
```

**Relation to quality gates:**

If MDE >> θ after significant sampling, the quality gate "WouldTakeTooLong" will trigger. The MDE provides interpretable context for why.

### 2.8 Discrete Timer Mode

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

**Threshold clamping:**

If θ < 1 tick, clamp to 1 tick and warn:

```rust
QualityIssue {
    code: IssueCode::ThresholdClamped,
    message: "θ=2ns clamped to 1 tick (41ns). Timer cannot resolve smaller effects.",
    guidance: "Use a cycle counter (perf/kperf) or accept reduced sensitivity.",
}
```

**Covariance estimation:**

Use m-out-of-n bootstrap for covariance estimation in discrete mode:

$$
m = \lfloor n^{2/3} \rfloor
$$

This provides consistent variance estimation when the standard CLT doesn't apply.

**Gaussian approximation caveat:**

The Gaussian likelihood is a rougher approximation with discrete data. We report `QualityIssue::DiscreteTimer` on all outputs and frame probabilities as "approximate under Gaussianized model."

### 2.9 Calibration Validation

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

**Generator overhead**: Measure the random input generator in isolation. If fixed and random generators differ in cost by more than 10%, **abort with an error**. The measured difference would reflect generator cost rather than the operation under test.

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

### 3.2.2 Dependence Estimation

We estimate dependence length using autocorrelation:

1. Compute ACF **per class** (not on interleaved sequence)
2. Find smallest lag where $|\rho(h)| < 2/\sqrt{n}$
3. Use maximum across both classes

**Effective sample size:**

$$
\text{ESS} \approx \frac{n}{1 + 2\sum_{h=1}^{\hat{m}} \rho(h)}
$$

When ESS << n, warn about high dependence:

```rust
QualityIssue {
    code: IssueCode::HighDependence,
    message: "Estimated dependence length: 47 samples (ESS: 2,100 of 100,000)",
    guidance: "High autocorrelation reduces effective sample size",
}
```

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

The adaptive Bayesian oracle returns one of four outcomes:

```rust
pub enum Outcome {
    /// No timing leak detected. P(leak) < pass_threshold (default 0.05).
    Pass {
        leak_probability: f64,     // P(effect > θ | data)
        effect: EffectEstimate,    // Shift/tail decomposition
        samples_used: usize,
        quality: MeasurementQuality,
        diagnostics: Diagnostics,
    },

    /// Timing leak confirmed. P(leak) > fail_threshold (default 0.95).
    Fail {
        leak_probability: f64,
        effect: EffectEstimate,
        exploitability: Exploitability,
        samples_used: usize,
        quality: MeasurementQuality,
        diagnostics: Diagnostics,
    },

    /// Cannot reach a definitive conclusion.
    Inconclusive {
        reason: InconclusiveReason,
        leak_probability: f64,     // Current posterior
        effect: EffectEstimate,
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
}

pub enum InconclusiveReason {
    /// Data is too noisy to reach a conclusion.
    DataTooNoisy { message: String, guidance: String },

    /// Posterior is not converging toward either threshold.
    NotLearning { message: String, guidance: String },

    /// Reaching a conclusion would take too long.
    WouldTakeTooLong { estimated_time_secs: f64, samples_needed: usize, guidance: String },

    /// Time budget exhausted.
    TimeBudgetExceeded { current_probability: f64, samples_collected: usize },

    /// Sample budget exhausted.
    SampleBudgetExceeded { current_probability: f64, samples_collected: usize },
}

pub struct EffectEstimate {
    pub shift_ns: f64,                     // Uniform shift (positive = baseline slower)
    pub tail_ns: f64,                      // Tail effect (positive = baseline has heavier upper tail)
    pub credible_interval_ns: (f64, f64),  // 95% CI for total effect
    pub pattern: EffectPattern,
}

pub enum EffectPattern {
    UniformShift,   // All quantiles shifted equally
    TailEffect,     // Upper quantiles shifted more
    Mixed,          // Both components significant
    Indeterminate,  // Neither significant
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
    pub dependence_length: usize,
    pub effective_sample_size: usize,
    pub stationarity_ratio: f64,
    pub stationarity_ok: bool,
    pub model_fit_chi2: f64,
    pub model_fit_ok: bool,
    pub outlier_rate_baseline: f64,
    pub outlier_rate_sample: f64,
    pub outlier_asymmetry_ok: bool,
    pub discrete_mode: bool,
    pub timer_resolution_ns: f64,
    pub duplicate_fraction: f64,
    pub preflight_ok: bool,
    pub calibration_samples: usize,
    pub total_time_secs: f64,
    pub warnings: Vec<String>,
    pub quality_issues: Vec<QualityIssue>,
}

pub struct QualityIssue {
    pub code: IssueCode,
    pub message: String,
    pub guidance: String,
}

pub enum IssueCode {
    HighDependence, LowEffectiveSamples, StationaritySuspect, DiscreteTimer,
    SmallSampleDiscrete, HighGeneratorCost, LowUniqueInputs, QuantilesFiltered,
    ThresholdClamped, HighWinsorRate,
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

    /// Research mode: detect any statistical difference (θ → 0).
    ///
    /// Warning: Will flag tiny, unexploitable differences. Not for CI.
    /// Use for: Profiling, debugging, academic analysis.
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
| `Research` | 0 | Profiling, debugging (not for CI) |

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
    Outcome::Pass { leak_probability, effect, .. } => {
        println!("No leak: P(leak)={:.1}%", leak_probability * 100.0);
    }
    Outcome::Fail { leak_probability, effect, exploitability, .. } => {
        println!("Leak detected: P(leak)={:.1}%, {:?}", leak_probability * 100.0, exploitability);
    }
    Outcome::Inconclusive { reason, leak_probability, .. } => {
        println!("Inconclusive: P(leak)={:.1}%", leak_probability * 100.0);
    }
    Outcome::Unmeasurable { recommendation, .. } => {
        println!("Skipping: {}", recommendation);
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

---

## 5. Interpreting Results

### 5.1 Leak Probability

The `leak_probability` is $P(\max_k |(X\beta)_k| > \theta \mid \Delta)$—the posterior probability of an effect exceeding your threshold.

| Probability | Interpretation | Outcome |
|-------------|----------------|---------|
| < 5% | Confident no leak | **Pass** |
| 5–50% | Probably safe, uncertain | Inconclusive (needs more data) |
| 50–95% | Probably leaking, uncertain | Inconclusive (needs more data) |
| > 95% | Confident leak | **Fail** |

The 5%/95% boundaries are configurable via `pass_threshold` and `fail_threshold`.

### 5.2 Effect Size

- **Shift (μ)**: Uniform timing difference. Cause: different code path, branch on secret.
- **Tail (τ)**: Upper quantiles affected more. Cause: cache misses, secret-dependent memory access.

Both reported in nanoseconds with 95% credible intervals.

**Effect patterns:**

| Pattern | Signature | Typical Cause |
|---------|-----------|---------------|
| UniformShift | μ significant, τ ≈ 0 | Branch on secret bit |
| TailEffect | μ ≈ 0, τ significant | Cache-timing leak |
| Mixed | Both significant | Multiple leak sources |
| Indeterminate | Neither significant | No detectable leak |

### 5.3 Exploitability

Rough risk assessment based on effect size:

| Level | Effect Size | Practical Impact |
|-------|-------------|------------------|
| Negligible | < 100 ns | Requires ~10k+ queries to exploit over LAN |
| PossibleLAN | 100–500 ns | Exploitable on LAN with statistical methods |
| LikelyLAN | 500 ns – 20 μs | Readily exploitable on local network |
| PossibleRemote | > 20 μs | Potentially exploitable over internet |

**Caveat**: Based on Crosby et al. (2009). Modern attacks may achieve better resolution.

### 5.4 Inconclusive Results

Inconclusive means "couldn't reach 95% confidence either way." This is different from "don't know"—you still get a probability estimate.

| Reason | Meaning | Suggested Action |
|--------|---------|------------------|
| DataTooNoisy | Posterior ≈ prior | Use better timer, reduce system noise |
| NotLearning | Data not reducing uncertainty | Check for systematic measurement issues |
| WouldTakeTooLong | Need too many samples | Accept uncertainty or adjust θ |
| TimeBudgetExceeded | Ran out of time | Increase time budget |
| SampleBudgetExceeded | Ran out of samples | Increase sample budget |

**For CI:**

```rust
match outcome {
    Outcome::Inconclusive { leak_probability, .. } if leak_probability > 0.5 => {
        // Probably leaking but not confident—treat as failure
        panic!("Likely timing leak (P={:.0}%)", leak_probability * 100.0);
    }
    Outcome::Inconclusive { .. } => {
        // Probably safe but not confident—treat as warning
        eprintln!("Warning: Could not confirm constant-time behavior");
    }
    _ => {}
}
```

### 5.5 Quality Assessment

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
| ThresholdClamped | θ < timer resolution | Accept coarser threshold or better timer |
| HighWinsorRate | > 1% outliers | Reduce system interference |

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $\Delta$ | 9-vector of signed quantile differences |
| $\hat{q}_F(k)$, $\hat{q}_R(k)$ | Empirical quantiles for Fixed and Random classes |
| $X$ | $9 \times 2$ design matrix $[\mathbf{1} \mid \mathbf{b}_{\text{tail}}]$ |
| $\beta = (\mu, \tau)^\top$ | Effect parameters: shift and tail |
| $\Sigma_n$ | Covariance matrix at sample size $n$ |
| $\Sigma_{\text{rate}}$ | Covariance rate: $\Sigma_n = \Sigma_{\text{rate}} / n$ |
| $\Lambda_0$ | Prior covariance for $\beta$ |
| $\Lambda_{\text{post}}$ | Posterior covariance for $\beta$ |
| $\theta$ | Practical significance threshold |
| MDE | Minimum detectable effect |
| $n$ | Samples per class |
| $\hat{b}$ | Block length (Politis-White) |
| $B$ | Bootstrap iterations |
| ESS | Effective sample size |

## Appendix B: Constants

| Constant | Default | Rationale |
|----------|---------|-----------|
| Deciles | {0.1, ..., 0.9} | Nine quantile positions |
| Bootstrap iterations | 2,000 | Covariance estimation accuracy |
| Monte Carlo samples | 1,000 | Leak probability integration |
| Batch size | 1,000 | Adaptive iteration granularity |
| Calibration samples | 5,000 | Initial covariance estimation |
| Pass threshold | 0.05 | 95% confidence of no leak |
| Fail threshold | 0.95 | 95% confidence of leak |
| Variance ratio gate | 0.5 | Posterior ≈ prior detection |
| Block length cap | min(3√n, n/3) | Prevent degenerate blocks |
| Discrete threshold | 10% unique | Trigger discrete mode |
| Min ticks per call | 5 | Measurability floor |
| Max batch size | 20 | Limit μarch artifacts |
| Default θ | 100 ns (AdjacentNetwork) | Practical significance |
| Default time budget | 30 s | Maximum runtime |
| Default sample budget | 100,000 | Maximum samples |

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
