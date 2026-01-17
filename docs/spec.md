# timing-oracle Specification (v4.2)

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
│                        • Check stopping                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Phase 1: Calibration** (runs once)
- Collect initial samples to characterize measurement noise
- Estimate covariance structure via stream-based block bootstrap
- Compute "covariance rate" $\Sigma_{\text{rate}}$ that scales as $\Sigma = \Sigma_{\text{rate}} / n$
- Compute initial measurement floor $\theta_{\text{floor}}$ and floor-rate constant $c_{\text{floor}}$
- Compute projection mismatch threshold $Q_{\text{proj,thresh}}$ from bootstrap distribution
- Compute effective threshold $\theta_{\text{eff}}$ and calibrate prior scale
- Run pre-flight checks (timer sanity, harness sanity, stationarity)

**Phase 2: Adaptive Loop** (iterates until decision)
- Collect batches of samples
- Update quantile estimates from all data collected so far
- Scale covariance: $\Sigma_n = \Sigma_{\text{rate}} / n$
- Update $\theta_{\text{floor}}(n)$ using floor-rate constant
- Compute posterior and $P(\text{effect} > \theta_{\text{eff}})$
- Check quality gates (posterior ≈ prior → Inconclusive)
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

#### 2.3.3 Projection Mismatch Threshold Calibration

During bootstrap, we calibrate the threshold for the projection mismatch diagnostic (see §2.6 Diagnostics). This determines when the 2D shift+tail summary is a faithful description of the inferred 9D quantile profile.

For each bootstrap replicate $\Delta^*$:
1. Compute 9D posterior mean: $\delta^*_{\text{post}}$ under the Bayesian update
2. Compute GLS projection: $\beta^*_{\text{proj}} = A \delta^*_{\text{post}}$
3. Compute projection residual: $r^*_{\text{proj}} = \delta^*_{\text{post}} - X\beta^*_{\text{proj}}$
4. Compute mismatch statistic: $Q^*_{\text{proj}} = (r^*_{\text{proj}})^\top \Sigma_n^{-1} r^*_{\text{proj}}$

The projection mismatch threshold is the 99th percentile of the bootstrap distribution:

$$
Q_{\text{proj,thresh}} = q_{0.99}(\{Q^*_{\text{proj},b}\}_{b=1}^B)
$$

**Note:** This threshold is used for the non-blocking projection mismatch diagnostic, not for verdict gating. See §2.6 for details.

#### 2.3.4 Measurement Floor and Effective Threshold

A critical design element is distinguishing between what the user *wants* to detect ($\theta_{\text{user}}$) and what the measurement *can* detect ($\theta_{\text{floor}}$).

**Floor-rate constant:**

Under the model's scaling assumption $\Sigma_n = \Sigma_{\text{rate}}/n$, the measurement floor scales as $1/\sqrt{n}$. We compute a floor-rate constant once at calibration based on the 9D decision functional $m(\delta) = \max_k |\delta_k|$.

Draw $Z_0 \sim \mathcal{N}(0, \Sigma_{\text{rate}})$ via Monte Carlo (50,000 samples) and compute:

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

#### 2.3.5 Prior Scale Calibration

The prior on the 9D effect profile $\delta$ should represent genuine uncertainty about whether effects exceed $\theta_{\text{eff}}$—neither near-certain leak nor near-certain safety.

**Prior structure (shaped ridge):**

The prior covariance is "shaped" to match the empirical covariance structure while controlling scale:

$$
\Lambda_0 = \sigma_{\text{prior}}^2 \cdot S, \quad S := \frac{\Sigma_{\text{rate}}}{\text{tr}(\Sigma_{\text{rate}})}
$$

Since $\text{tr}(S) = 1$, the parameter $\sigma_{\text{prior}}$ remains an interpretable global scale. This shaping respects the correlation structure across quantiles and improves stability without adding parameters.

**Robustness fallback:** If $\Sigma_{\text{rate}}$ is ill-conditioned or discrete-timer mode is active, apply shrinkage:

$$
S \leftarrow (1-\lambda)S + \lambda \frac{I_9}{9}
$$

with a conservative default $\lambda \in [0.05, 0.2]$, chosen deterministically from conditioning diagnostics. If conditioning remains poor, set $S = I_9/9$.

**Numerical calibration of $\sigma_{\text{prior}}$:**

Choose $\sigma_{\text{prior}}$ so that the prior exceedance probability equals a fixed target $\pi_0$ (default 0.62):

$$
P\left(\max_k |\delta_k| > \theta_{\text{eff}} \;\middle|\; \delta \sim \mathcal{N}(0, \Lambda_0)\right) = \pi_0
$$

Given $\Lambda_0 = \sigma_{\text{prior}}^2 S$, this is a 1D root-find on $\sigma_{\text{prior}}$ using deterministic Monte Carlo (e.g., 50k draws). The target $\pi_0 = 0.62$ represents genuine uncertainty about whether effects exceed the effective threshold.

**Normative requirements:**

- The RNG seed MUST be deterministic (per §2.3.6)
- The calibration MUST be performed once per run, during calibration
- The prior MUST remain fixed throughout the adaptive loop. If $\theta_{\text{eff}}$ changes as $n$ grows, it affects only exceedance checks, not the prior.

#### 2.3.6 Deterministic Seeding Policy

To ensure reproducible results, all random number generation is deterministic by default.

**Normative requirement:**

> Given identical timing samples and configuration, the oracle must produce identical results (up to floating-point roundoff). Algorithmic randomness is not epistemic uncertainty about the leak—it is approximation error that should not influence the decision rule.

**Seeding policy:**

- The bootstrap RNG seed is deterministically derived from:
  - A fixed library constant seed (default: 0x74696D696E67, "timing" in ASCII)
  - A stable hash of configuration parameters

- All Monte Carlo RNG seeds (leak probability, floor constant, prior scale) are similarly deterministic

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
    sigma_rate: Matrix9x9,       // Covariance rate: Σ = Σ_rate / n
    block_length: usize,         // Estimated from acquisition stream
    c_floor: f64,                // Floor-rate constant
    q_proj_thresh: f64,          // Projection mismatch threshold (99th percentile)
    theta_floor_initial: f64,    // Initial measurement floor
    theta_eff: f64,              // Effective threshold for this run
    prior_shape: Matrix9x9,      // S matrix for shaped prior
    sigma_prior: f64,            // Calibrated prior scale
    timer_resolution_ns: f64,    // Detected tick size
    samples_per_second: f64,     // For time estimation
    rng_seed: u64,               // Deterministic seed used
}
```

### 2.4 Bayesian Model

We use a conjugate Gaussian model over the full 9D quantile-difference profile that admits closed-form posteriors—no MCMC required.

#### 2.4.1 Latent Parameter

The latent parameter is the true per-decile timing difference profile:

$$
\delta \in \mathbb{R}^9 \quad \text{(true decile-difference profile)}
$$

This is unconstrained—the model can represent any quantile-difference pattern, unlike a low-rank projection.

#### 2.4.2 Likelihood

$$
\Delta \mid \delta \sim \mathcal{N}(\delta, \Sigma_n)
$$

where $\Sigma_n = \Sigma_{\text{rate}} / n$ is the scaled covariance. This is the minimal mean-structure assumption: the covariance is estimated; the mean is unconstrained.

#### 2.4.3 Prior

$$
\delta \sim \mathcal{N}(0, \Lambda_0), \quad \Lambda_0 = \sigma_{\text{prior}}^2 \cdot S
$$

where $S$ is the shaped prior matrix and $\sigma_{\text{prior}}$ is calibrated numerically (see §2.3.5).

#### 2.4.4 Posterior

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

The posterior mean $\delta_{\text{post}} \in \mathbb{R}^9$ gives the estimated timing difference at each decile in nanoseconds.

**Implementation guidance (normative):**

The oracle MUST NOT form explicit matrix inverses in floating point. Use Cholesky solves on SPD matrices. A stable compute path:

```rust
// Form A := Σ_n + Λ_0 (SPD)
// Solve A x = Δ for x
// Then δ_post = Λ_0 x
// And Λ_post = Λ_0 - Λ_0 A⁻¹ Λ_0 (via solves)
```

This avoids $\Sigma_n^{-1}$ explicitly.

⁴ Bishop, C. M. (2006). Pattern Recognition and Machine Learning, §3.3. Springer.

#### 2.4.5 Decision Functional and Leak Probability

The decision is based on the maximum absolute effect across all quantiles:

$$
m(\delta) := \max_k |\delta_k|
$$

The leak probability is:

$$
P(\text{leak} > \theta_{\text{eff}} \mid \Delta) = P\bigl(m(\delta) > \theta_{\text{eff}} \;\big|\; \Delta\bigr)
$$

Since the posterior is Gaussian, we compute this by Monte Carlo integration:

```python
samples = draw_from_normal(delta_post, Lambda_post, n=10000, seed=deterministic_seed)
count = 0
for delta in samples:
    max_effect = max(abs(delta))
    if max_effect > theta_eff:
        count += 1
leak_probability = count / 10000
```

This is O(1)—we're drawing from a 9D Gaussian and doing simple arithmetic. The seed is deterministic (see §2.3.6).

**Interpreting the probability:**

This is a **posterior probability**, not a p-value. When we report "72% probability of a leak," we mean: given the data and our model, 72% of the posterior mass corresponds to effects exceeding $\theta_{\text{eff}}$.

This is calibrated by construction under the Gaussian model. Empirical validation (§2.11) confirms calibration holds in practice.

**Note on $\theta_{\text{eff}}$:**

All inference uses $\theta_{\text{eff}}$, not $\theta_{\text{user}}$. When these differ, the output clearly reports both values so the user understands what was actually tested.

#### 2.4.6 2D Projection for Reporting

The 9D posterior is used for decisions, but we provide a 2D summary for interpretability when the pattern is simple.

**Design matrix (for projection only):**

$$
X = \begin{bmatrix} \mathbf{1} & \mathbf{b}_{\text{tail}} \end{bmatrix} \in \mathbb{R}^{9 \times 2}
$$

where:
- $\mathbf{1} = (1, 1, 1, 1, 1, 1, 1, 1, 1)^\top$ — uniform shift
- $\mathbf{b}_{\text{tail}} = (-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5)^\top$ — tail effect

**GLS projection operator:**

$$
A := (X^\top \Sigma_n^{-1} X)^{-1} X^\top \Sigma_n^{-1}
$$

**Posterior for projection summary:**

Since $\beta_{\text{proj}} = A\delta$ is linear in $\delta$, under the 9D posterior:

- Mean: $\beta_{\text{proj,post}} = A \delta_{\text{post}}$
- Covariance: $\text{Cov}(\beta_{\text{proj}} \mid \Delta) = A \Lambda_{\text{post}} A^\top$

The projection gives interpretable components:
- **Shift ($\mu$)**: Uniform timing difference affecting all quantiles equally
- **Tail ($\tau$)**: Upper quantiles affected more than lower (or vice versa)

**Important:** The 2D projection is for reporting only. Decisions are based on the 9D posterior. When the projection doesn't fit well (see §2.6 Diagnostics), the oracle adds an interpretation caveat and provides alternative explanations.

### 2.5 Adaptive Sampling Loop

The core innovation: collect samples until confident, with natural early stopping.

**Verdict-blocking semantics:**

> Pass/Fail verdicts are emitted **only** if all measurement quality gates pass and condition drift checks pass. Otherwise the oracle must return Inconclusive, reporting the posterior probability as an "estimate under the model" but not asserting a strong claim.

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

        // 4. Compute projection mismatch (non-blocking diagnostic)
        let beta_proj = projection_operator * posterior.delta_post;
        let r_proj = posterior.delta_post - design_matrix * beta_proj;
        let q_proj = r_proj.dot(&state.sigma_n_inv * r_proj);
        let projection_mismatch = q_proj > calibration.q_proj_thresh;

        // 5. Check ALL quality gates FIRST (verdict-blocking)
        if let Some(reason) = check_quality_gates(&state, &posterior, calibration, config) {
            return Outcome::Inconclusive {
                reason,
                leak_probability: posterior.leak_probability,
                effect: posterior.effect_estimate(projection_mismatch),
                theta_user,
                theta_eff,
                theta_floor,
                samples_used: state.n_total,
            };
        }

        // 6. Check decision boundaries (projection mismatch does NOT block)
        if posterior.leak_probability > 0.95 {
            return Outcome::Fail {
                leak_probability: posterior.leak_probability,
                effect: posterior.effect_estimate(projection_mismatch),
                theta_user,
                theta_eff,
                theta_floor,
                samples_used: state.n_total,
            };
        }
        if posterior.leak_probability < 0.05 {
            return Outcome::Pass {
                leak_probability: posterior.leak_probability,
                effect: posterior.effect_estimate(projection_mismatch),
                theta_user,
                theta_eff,
                theta_floor,
                samples_used: state.n_total,
            };
        }

        // 7. Check budgets
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
4. Posterior update: O(1) — 9×9 matrix operations
5. Projection mismatch check: O(1) — single quadratic form
6. Monte Carlo integration: O(10,000 × 9)
7. Update θ_floor: O(1) — analytical formula

Total: dominated by quantile computation, which can be optimized with streaming quantile algorithms.

**Batch size selection:**

Default batch size is 1,000 samples. This balances:
- Large enough for stable quantile estimates per batch
- Small enough for responsive early stopping
- Matches typical calibration sample count

### 2.6 Quality Gates and Diagnostics

Quality gates detect when data is too poor to reach a confident decision. **Verdict-blocking gates** cause the oracle to return Inconclusive rather than Pass/Fail. **Non-blocking diagnostics** add caveats to the output but do not prevent verdicts.

#### Verdict-Blocking Gates

##### Gate 1: Posterior ≈ Prior (Data Too Noisy)

If measurement uncertainty is high, the posterior barely moves from the prior. For the 9D model, use log-det ratio:

$$
\rho := \log \frac{|\Lambda_{\text{post}}|}{|\Lambda_0|}
$$

```rust
if rho > log(0.5) {  // Posterior volume not reduced much
    return Some(InconclusiveReason::DataTooNoisy {
        message: "Posterior variance is >50% of prior; data not informative",
        guidance: "Try: cycle counter, reduce system load, increase batch size",
    });
}
```

Alternative: use trace ratio $\text{tr}(\Lambda_{\text{post}})/\text{tr}(\Lambda_0) > 0.5$ if log-det is numerically unstable.

**Why this matters:** A wide posterior centered at zero can have substantial mass above θ—not because we're confident there's a leak, but because we're uncertain about everything. This gate catches that pathology.

##### Gate 2: Learning Rate Collapsed

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

where $k = 9$ for the 9D model.

##### Gate 3: Would Take Too Long

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

##### Gate 4: Threshold Unachievable

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

##### Gate 5: Time Budget Exceeded

```rust
if state.elapsed() > config.time_budget {
    return Some(InconclusiveReason::TimeBudgetExceeded {
        current_probability: posterior.leak_probability,
        samples_collected: state.n_total,
    });
}
```

##### Gate 6: Sample Budget Exceeded

```rust
if state.n_total >= config.max_samples {
    return Some(InconclusiveReason::SampleBudgetExceeded {
        current_probability: posterior.leak_probability,
        samples_collected: state.n_total,
    });
}
```

##### Gate 7: Condition Drift Detected

The covariance estimate $\Sigma_{\text{rate}}$ is computed during calibration. If measurement conditions change during the adaptive loop, this estimate becomes invalid.

We detect condition drift by comparing measurement statistics from calibration against the full test run:

```rust
struct ConditionDrift {
    variance_ratio: f64,    // σ²_post / σ²_cal
    autocorr_change: f64,   // |ρ_post(1) - ρ_cal(1)|
    mean_drift: f64,        // |μ_post - μ_cal| / σ_cal
}

if drift.variance_ratio > 2.0 || drift.variance_ratio < 0.5 {
    return Some(InconclusiveReason::ConditionsChanged { drift });
}

if drift.autocorr_change > 0.3 {
    return Some(InconclusiveReason::ConditionsChanged { drift });
}

if drift.mean_drift > 3.0 {
    return Some(InconclusiveReason::ConditionsChanged { drift });
}
```

#### Non-Blocking Diagnostics

##### Projection Mismatch Diagnostic

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
- If $Q_{\text{proj}} > Q_{\text{proj,thresh}}$, the oracle MUST:
  - Set `effect.interpretation_caveat`
  - Report a "top quantiles" shape hint (see below)
  - Mark `Diagnostics.projection_mismatch_ok = false`

**Top quantiles shape hint:**

When projection mismatch is high, report the 2–3 most relevant quantiles ranked by per-quantile exceedance probability:

$$
p_k := P(|\delta_k| > \theta_{\text{eff}} \mid \Delta)
$$

Since the posterior is Gaussian, the marginal $\delta_k \mid \Delta \sim \mathcal{N}(\mu_k, s_k^2)$ where:

$$
\mu_k = (\delta_{\text{post}})_k, \quad s_k^2 = (\Lambda_{\text{post}})_{kk}
$$

Compute:

$$
p_k = 1 - \left[\Phi\left(\frac{\theta_{\text{eff}} - \mu_k}{s_k}\right) - \Phi\left(\frac{-\theta_{\text{eff}} - \mu_k}{s_k}\right)\right]
$$

**Selection rule:** Select top 2–3 indices by $p_k$ (descending). Optional stability filter: include only those with $p_k \geq 0.10$; if fewer than 2 remain, relax to include the top 2.

**Reporting:** For each selected quantile, report mean, 95% CI, and exceedance probability. This is O(9) and requires no extra Monte Carlo.

### 2.7 Minimum Detectable Effect

The MDE answers: "what's the smallest effect I could reliably detect given the noise level?"

For the 9D model, the MDE is defined in terms of the decision functional $m(\delta) = \max_k |\delta_k|$. The measurement floor $\theta_{\text{floor}}$ serves this purpose—it's the smallest effect that can be reliably distinguished from noise at the current sample size.

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

The decision functional is the same:

$$
m(\delta) = \max_k |\delta_k|
$$

#### Algorithm

```rust
fn run_research_mode(calibration: &Calibration, config: &Config) -> ResearchOutcome {
    let mut state = State::new(calibration);

    loop {
        // 1. Collect batch
        state.collect_batch(config.batch_size);

        // 2. Update posterior on δ
        let posterior = state.compute_posterior();

        // 3. Recompute θ_floor analytically
        let theta_floor = f64::max(
            calibration.c_floor / (state.n_total as f64).sqrt(),
            calibration.theta_tick,
        );

        // 4. Check projection mismatch (affects interpretation, not stopping)
        let q_proj = compute_projection_mismatch(&posterior, &state);
        let projection_mismatch = q_proj > calibration.q_proj_thresh;

        // 5. Compute posterior distribution of m = max|δ|
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
                projection_mismatch,
                effect: posterior.effect_estimate(projection_mismatch),
                samples_used: state.n_total,
                // ...
            };
        }

        // 6b. Confident non-detection: CI clearly below floor
        if m_ci.1 < theta_floor * 0.9 {
            return ResearchOutcome {
                status: ResearchStatus::NoEffectDetected,
                // ...
            };
        }

        // 6c. θ_floor hit the tick floor (can't improve further)
        if theta_floor <= calibration.theta_tick * 1.01 {
            return ResearchOutcome {
                status: ResearchStatus::ResolutionLimitReached,
                // ...
            };
        }

        // 6d. Quality gates
        if let Some(issue) = check_quality_gates_research(&state, &posterior, calibration, config) {
            return ResearchOutcome {
                status: ResearchStatus::QualityIssue(issue),
                // ...
            };
        }

        // 6e. Budget exhausted
        if state.budget_exhausted(config) {
            return ResearchOutcome {
                status: ResearchStatus::BudgetExhausted,
                // ...
            };
        }
    }
}
```

#### Stopping Conditions

| Condition | Criterion | Status |
|-----------|-----------|--------|
| Effect detected | CI lower bound > 1.1 × θ_floor | `EffectDetected` |
| No effect detected | CI upper bound < 0.9 × θ_floor | `NoEffectDetected` |
| Resolution limit | θ_floor ≈ θ_tick | `ResolutionLimitReached` |
| Quality issue | Any quality gate triggers | `QualityIssue` |
| Budget exhausted | Time or samples exceeded | `BudgetExhausted` |

The 1.1× and 0.9× margins provide hysteresis to prevent oscillating near the boundary.

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

**Prior shape fallback:**

In discrete mode, apply shrinkage to the prior shape matrix (see §2.3.5).

**Gaussian approximation caveat:**

The Gaussian likelihood is a rougher approximation with discrete data. We report `QualityIssue::DiscreteTimer` on all outputs and frame probabilities as "approximate under Gaussianized model."

### 2.11 Calibration Validation

The Bayesian approach requires empirical validation that posteriors are well-calibrated.

**Null calibration test (normative requirement):**

The oracle MUST provide (or run in internal test suite) a "fixed-vs-fixed" validation:

- Run the full pipeline under null (same distribution both classes)
- Empirically verify that:
  - The rate of `Fail` is near 0 and bounded by configured error tolerance
  - The rate of `Pass` at the pass threshold is consistent with expectations, conditional on quality gates passing

This is an end-to-end check that $c_{\text{floor}}$, $\Sigma$ scaling, prior calibration, and posterior computation are not systematically anti-conservative.

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

#### 3.2.1 Stationarity Check

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
        leak_probability: f64,     // P(max|δ| > θ_eff | data)
        effect: EffectEstimate,    // Projection summary + top quantiles if needed
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

    /// Posterior mean of max_k |δ_k|
    pub max_effect_ns: f64,

    /// 95% credible interval for max effect
    pub max_effect_ci: (f64, f64),

    /// Smallest effect we could resolve (at final n)
    pub theta_floor: f64,

    /// Is the effect confidently above the floor?
    pub detectable: bool,

    /// True if 2D projection doesn't fit well
    pub projection_mismatch: bool,

    /// Decomposition into shift + tail components (projection summary)
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
    /// Uniform shift from 2D projection (positive = fixed class slower)
    pub shift_ns: f64,
    /// Tail effect from 2D projection (positive = fixed class has heavier upper tail)
    pub tail_ns: f64,
    /// 95% CI for total effect (from 9D posterior)
    pub credible_interval_ns: (f64, f64),
    /// Pattern classification
    pub pattern: EffectPattern,
    /// When projection mismatch is high, explains why 2D summary may be inaccurate
    pub interpretation_caveat: Option<String>,
    /// Top quantiles by exceedance probability (populated when projection mismatch is high)
    pub top_quantiles: Option<Vec<TopQuantile>>,
}

pub struct TopQuantile {
    /// Which quantile (e.g., 0.9 for 90th percentile)
    pub quantile_p: f64,
    /// Posterior mean effect at this quantile (ns)
    pub mean_ns: f64,
    /// 95% marginal credible interval (ns)
    pub ci95_ns: (f64, f64),
    /// P(|δ_k| > θ_eff | data) for this quantile
    pub exceed_prob: f64,
}

pub enum EffectPattern {
    UniformShift,   // All quantiles shifted equally
    TailEffect,     // Upper quantiles shifted more
    Mixed,          // Both components significant
    Complex,        // Projection mismatch high; pattern doesn't fit 2D basis
    Indeterminate,  // Neither significant
}

pub enum Exploitability {
    SharedHardwareOnly, // < 10 ns
    Http2Multiplexing,  // 10–100 ns
    StandardRemote,     // 100 ns – 10 μs
    ObviousLeak,        // > 10 μs
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
    pub outlier_rate_baseline: f64,
    pub outlier_rate_sample: f64,
    pub outlier_asymmetry_ok: bool,

    // Projection mismatch (non-blocking)
    pub projection_mismatch_q: f64,
    pub projection_mismatch_threshold: f64,
    pub projection_mismatch_ok: bool,

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

    // Reproduction info
    pub seed: Option<u64>,
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
    ThresholdElevated, ThresholdClamped, HighWinsorRate, ProjectionMismatch,
}

pub struct PreflightWarningInfo {
    pub category: PreflightCategory,
    pub severity: PreflightSeverity,
    pub message: String,
    pub guidance: Option<String>,
}

pub enum PreflightSeverity {
    Informational,
    ResultUndermining,
}

pub enum PreflightCategory {
    TimerSanity,
    Sanity,
    Generator,
    Autocorrelation,
    System,
    Resolution,
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
    SharedHardware,

    /// Post-quantum crypto sentinel.
    /// θ = 3.3 ns (~10 cycles @ 3GHz)
    PostQuantumSentinel,

    /// Adjacent network attacker (LAN, HTTP/2 multiplexing).
    /// θ = 100 ns
    AdjacentNetwork,

    /// Remote network attacker (general internet).
    /// θ = 50 μs
    RemoteNetwork,

    /// Research mode: no threshold, reports posterior directly.
    Research,

    /// Custom threshold in nanoseconds.
    Custom { threshold_ns: f64 },
}
```

| Preset | θ | Use case |
|--------|---|----------|
| `SharedHardware` | 0.6 ns (~2 cycles @ 3GHz) | SGX, cross-VM, containers, hyperthreading |
| `PostQuantumSentinel` | 3.3 ns (~10 cycles @ 3GHz) | Post-quantum crypto (ML-KEM, ML-DSA) |
| `AdjacentNetwork` | 100 ns | LAN, HTTP/2 (Timeless Timing Attacks) |
| `RemoteNetwork` | 50 μs | Internet, legacy services |
| `Research` | θ_floor (dynamic) | Profiling, debugging (not for CI) |

#### TimingOracle Configuration

```rust
impl TimingOracle {
    /// Create oracle for a specific attacker model (recommended entry point).
    pub fn for_attacker(model: AttackerModel) -> Self;

    // Decision thresholds
    pub fn pass_threshold(self, p: f64) -> Self;  // P(leak) < p → Pass (default 0.05)
    pub fn fail_threshold(self, p: f64) -> Self;  // P(leak) > p → Fail (default 0.95)

    // Resource limits
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

        if res.projection_mismatch {
            println!("⚠ Complex pattern; 2D summary is approximate");
            if let Some(top) = &res.effect.top_quantiles {
                for q in top {
                    println!("  {:}th percentile: {:.1} ns (P={:.0}%)",
                             (q.quantile_p * 100.0) as u32,
                             q.mean_ns,
                             q.exceed_prob * 100.0);
                }
            }
        }

        match res.status {
            ResearchStatus::EffectDetected => {
                println!("✓ Timing difference detected");
            }
            ResearchStatus::NoEffectDetected => {
                println!("✗ No timing difference above {:.1} ns", res.theta_floor);
            }
            _ => {}
        }
    }
    _ => unreachable!("Research mode always returns Research variant"),
}
```

---

## 5. Interpreting Results

### 5.1 Leak Probability

The `leak_probability` is $P(\max_k |\delta_k| > \theta_{\text{eff}} \mid \Delta)$—the posterior probability of an effect exceeding your effective threshold at any quantile.

| Probability | Interpretation | Outcome |
|-------------|----------------|---------|
| < 5% | Confident no leak | **Pass** |
| 5–50% | Probably safe, uncertain | Inconclusive (needs more data) |
| 50–95% | Probably leaking, uncertain | Inconclusive (needs more data) |
| > 95% | Confident leak | **Fail** |

### 5.2 Understanding θ_user vs θ_eff vs θ_floor

| Value | Meaning |
|-------|---------|
| θ_user | What you requested |
| θ_floor | What the measurement could resolve |
| θ_eff | What was actually tested (max of the above) |

**When θ_eff = θ_user:** Results directly answer "is there a leak > θ_user?"

**When θ_eff > θ_user:** Results answer "is there a leak > θ_eff?" A `Pass` means "no leak above θ_eff," but there could still be a leak between θ_user and θ_eff.

### 5.3 Effect Size and Pattern

The `EffectEstimate` provides two levels of explanation:

**2D Projection (when it fits well):**
- **Shift (μ)**: Uniform timing difference. Cause: different code path, branch on secret.
- **Tail (τ)**: Upper quantiles affected more. Cause: cache misses, secret-dependent memory access.

**Top Quantiles (when projection doesn't fit):**

When `projection_mismatch` is true, the pattern is complex. Check `top_quantiles` for where the effect concentrates:

```rust
if let Some(top) = &effect.top_quantiles {
    for q in top {
        println!("{:}th percentile: {:.1} ns [{:.1}, {:.1}] (P={:.0}%)",
                 (q.quantile_p * 100.0) as u32,
                 q.mean_ns,
                 q.ci95_ns.0, q.ci95_ns.1,
                 q.exceed_prob * 100.0);
    }
}
```

**Effect patterns:**

| Pattern | Signature | Typical Cause |
|---------|-----------|---------------|
| UniformShift | μ significant, τ ≈ 0 | Branch on secret bit |
| TailEffect | μ ≈ 0, τ significant | Cache-timing leak |
| Mixed | Both significant | Multiple leak sources |
| Complex | Projection mismatch high | Asymmetric or multi-modal pattern |
| Indeterminate | Neither significant | No detectable leak |

### 5.4 Exploitability

Risk assessment based on effect size:

| Level | Effect Size | Attack Vector |
|-------|-------------|---------------|
| SharedHardwareOnly | < 10 ns | ~1k queries on same core |
| Http2Multiplexing | 10–100 ns | ~100k concurrent HTTP/2 requests |
| StandardRemote | 100 ns – 10 μs | ~1k-10k queries with standard timing |
| ObviousLeak | > 10 μs | <100 queries, trivially observable |

### 5.5 Inconclusive Results

Inconclusive means "couldn't reach 95% confidence either way."

| Reason | Meaning | Suggested Action |
|--------|---------|------------------|
| DataTooNoisy | Posterior ≈ prior | Use better timer, reduce system noise |
| NotLearning | Data not reducing uncertainty | Check for systematic measurement issues |
| WouldTakeTooLong | Need too many samples | Accept uncertainty or adjust θ |
| ThresholdUnachievable | θ_user < θ_floor even at max budget | Use cycle counter or accept elevated θ |
| TimeBudgetExceeded | Ran out of time | Increase time budget |
| SampleBudgetExceeded | Ran out of samples | Increase sample budget |
| ConditionsChanged | Calibration assumptions violated | Run in isolation, reduce system load |

### 5.6 Projection Mismatch

When `diagnostics.projection_mismatch_ok = false`:

- The 2D shift+tail summary doesn't fit the observed pattern well
- **This does NOT affect the verdict**—the 9D posterior handles arbitrary patterns
- Check `effect.top_quantiles` for where the timing difference concentrates
- The `interpretation_caveat` explains why the 2D summary may be inaccurate

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $\{(c_t, y_t)\}$ | Acquisition stream: class labels and timing measurements |
| $T$ | Acquisition stream length ($\approx 2n$) |
| $F$, $R$ | Per-class sample sets (filtered from stream) |
| $\Delta$ | 9-vector of observed quantile differences |
| $\delta$ | 9-vector of true (latent) quantile differences |
| $\hat{q}_F(k)$, $\hat{q}_R(k)$ | Empirical quantiles for Fixed and Random classes |
| $X$ | $9 \times 2$ projection basis $[\mathbf{1} \mid \mathbf{b}_{\text{tail}}]$ |
| $\beta_{\text{proj}}$ | 2D projection: $(\mu, \tau)^\top$ |
| $\Sigma_n$ | Covariance matrix at sample size $n$ |
| $\Sigma_{\text{rate}}$ | Covariance rate: $\Sigma_n = \Sigma_{\text{rate}} / n$ |
| $\Lambda_0$ | Prior covariance: $\sigma_{\text{prior}}^2 \cdot S$ |
| $\Lambda_{\text{post}}$ | Posterior covariance for $\delta$ |
| $S$ | Shaped prior matrix: $\Sigma_{\text{rate}} / \text{tr}(\Sigma_{\text{rate}})$ |
| $\theta_{\text{user}}$ | User-requested threshold |
| $\theta_{\text{floor}}$ | Measurement floor (smallest resolvable effect) |
| $c_{\text{floor}}$ | Floor-rate constant: $\theta_{\text{floor,stat}} = c_{\text{floor}} / \sqrt{n}$ |
| $\theta_{\text{tick}}$ | Timer resolution component of floor |
| $\theta_{\text{eff}}$ | Effective threshold used for inference |
| $m(\delta)$ | Decision functional: $\max_k |\delta_k|$ |
| $Q_{\text{proj}}$ | Projection mismatch statistic |
| $Q_{\text{proj,thresh}}$ | Projection mismatch threshold (99th percentile) |
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
| Prior exceedance target | 0.62 | Genuine uncertainty about exceedance |
| Variance ratio gate | 0.5 | Posterior ≈ prior detection (log-det or trace) |
| Projection mismatch percentile | 99th | Q_proj_thresh from bootstrap |
| Block length cap | min(3√T, T/3) | Prevent degenerate blocks |
| Discrete threshold | 10% unique | Trigger discrete mode |
| Min ticks per call | 5 | Measurability floor |
| Max batch size | 20 | Limit μarch artifacts |
| Default θ | 100 ns (AdjacentNetwork) | Practical significance |
| Default time budget | 30 s | Maximum runtime |
| Default sample budget | 100,000 | Maximum samples |
| CI hysteresis (Research) | 10% | Margin for detection decisions |
| Default RNG seed | 0x74696D696E67 | "timing" in ASCII |
| Prior shrinkage (λ) | 0.05–0.2 | Robustness for ill-conditioned Σ |

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

## Appendix D: Changelog from v4.1

### Summary of Changes

| Aspect | v4.1 | v4.2 |
|--------|------|------|
| Inference model | 2D (β ∈ ℝ²) constrained to span(X) | 9D (δ ∈ ℝ⁹) unconstrained |
| Prior structure | Isotropic on β | Shaped ridge on δ: Λ₀ = σ²S |
| Prior scale | κ = 1.12 (closed-form) | Numerical root-finding |
| Decision functional | max_k \|(Xβ)_k\| | max_k \|δ_k\| |
| c_floor definition | Based on prediction covariance | Based on Σ_rate directly |
| Model mismatch | Verdict-blocking gate | Non-blocking diagnostic |
| Projection summary | Primary output | Secondary (when it fits) |
| Top quantiles | Not present | Shown when projection mismatch high |
| EffectPattern | 4 variants | 5 variants (+ Complex) |

### Key Fixes

1. **9D inference (Amendment 6)**: The 2D basis constraint caused frequent ModelMismatch → Inconclusive for real-world leaks with asymmetric patterns. Moving to 9D removes this failure mode while keeping conjugacy.

2. **Shaped prior**: The isotropic prior ignored quantile correlation. The shaped prior Λ₀ = σ²S respects the covariance structure.

3. **Numerical prior scale**: The closed-form κ ≈ 1.12 assumed i.i.d. coordinates. Numerical root-finding handles correlation correctly.

4. **Projection mismatch is non-blocking**: Since 9D inference can represent any pattern, poor 2D fit is an explainability issue, not a validity threat.

5. **Top quantiles reporting**: When the 2D summary doesn't fit, users get actionable information about where the effect concentrates.

### Migration Notes

- The calibration phase still estimates Σ_rate via stream-based block bootstrap on Δ. No change required.
- The adaptive loop still uses Σ_n = Σ_rate/n scaling.
- Verdict-blocking quality gates are unchanged except ModelMismatch is removed.
- API changes are additive: new fields in Diagnostics and EffectEstimate; InconclusiveReason::ModelMismatch removed.

### Design Philosophy

The guiding principle remains:

> **CI verdicts should almost never be confidently wrong.**

The 9D model strengthens this by eliminating a class of false Inconclusives that occurred when real leaks had patterns outside the 2D basis. Now, unusual patterns still get detected and reported—just with a different explanation format.
