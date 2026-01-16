# timing-oracle Specification (v2)

## 1. Overview

### Problem Statement

Timing side-channel attacks exploit the fact that cryptographic implementations often take different amounts of time depending on secret data. Detecting these leaks is critical for security, but existing tools have significant limitations:

- **DudeCT** and similar t-test approaches compare means, missing distributional differences (e.g., cache effects that only affect upper quantiles)
- **P-values are misinterpreted**: A p-value of 0.01 doesn't mean "1% chance of leak"—it means "1% chance of this data given no leak." These are very different.
- **Unbounded false positives**: With enough samples, even negligible effects become "statistically significant"
- **CI flakiness**: Tests that pass locally fail in CI due to environmental noise, or vice versa

### Solution

`timing-oracle` addresses these issues with:

1. **Quantile-based statistics**: Compare nine deciles instead of just means, capturing both uniform shifts (branch timing) and tail effects (cache misses)

2. **Two-layer analysis**:
   - Layer 1 (CI Gate): Bounded false positive rate for reliable pass/fail decisions
   - Layer 2 (Bayesian): Posterior probability of a leak, plus effect size estimates

3. **Interpretable output**: "72% probability of a timing leak with ~50ns effect" instead of "$t = 2.34$, $p = 0.019$"

### Design Goals

- **Controlled FPR**: The CI gate controls false positive rate at approximately $\alpha$ (default 1%)
- **Interpretable**: Output is a probability (0–100%), not a t-statistic
- **CI-friendly**: Works reliably across environments without manual tuning
- **Fast**: Under 5 seconds for 100k samples on typical hardware
- **Portable**: Handles different timer resolutions via adaptive batching

---

## 2. Statistical Methodology

This section describes the mathematical foundation of timing-oracle: why we use quantile differences instead of means, how the two-layer architecture provides both reliability and interpretability, and how we estimate uncertainty while keeping computational costs reasonable.

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

**Quantile computation (continuous mode):** Following SILENT, we use **type 2** quantiles (inverse empirical CDF with averaging).² For sorted sample $x$ of size $n$ at probability $p$:

$$
h = n \cdot p + 0.5
$$

$$
\hat{q}_p = \frac{x_{\lfloor h \rfloor} + x_{\lceil h \rceil}}{2}
$$

Type 2 uses the inverse of the empirical distribution function with averaging at discontinuities, which is more appropriate for the bootstrap-based inference than interpolating estimators.

**Quantile computation (discrete mode):** Use **mid-distribution quantiles** (see §2.4) which correctly handle tied values.

² Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages." The American Statistician 50(4):361–365. Type 2 is their taxonomy.

### 2.2 Two-Layer Architecture

A key design decision is using two separate statistical tests rather than one:

| Layer | Question | Method | Output |
|-------|----------|--------|--------|
| CI Gate | Is there a leak above my threshold? | SILENT relevant-hypothesis bootstrap | Pass/Fail |
| Bayesian | What's the probability and magnitude? | Posterior integration | $P(\text{leak} > \theta)$, effect size |

Both layers use the same practical significance threshold: `min_effect_of_concern` (θ, default 10ns). This ensures they agree on what counts as a "leak worth caring about."

**Why two layers?**

Each provides something the other can't:

*Bootstrap CI gate* (Layer 1) provides controlled false positive rate using the SILENT framework [2]. If you set α = 0.01, the test will incorrectly reject at most 1% of the time when the true effect is ≤ θ—crucially, this holds even at the boundary (effect = θ), not just when effect = 0. This is what makes the CI gate reliable: you won't get spurious failures on code that just barely meets your threshold.

*Bayesian inference* (Layer 2) gives you a posterior probability ("72% chance of a leak exceeding θ") and an effect estimate ("~50ns uniform shift"). This is far more informative than pass/fail, and the probability is calibrated: when we report 80%, approximately 80% of the posterior mass lies above θ.

By combining both:
- A reliable CI gate that won't flake on negligible effects
- Rich diagnostic information when you need to understand what's happening

They're computed from the same data and use the same threshold. In normal operation they agree; divergence indicates edge cases worth investigating.

### 2.3 Sample Splitting

The Bayesian layer requires estimating a covariance matrix and setting priors. If we estimate these from the same data we then test, we risk overfitting: the prior becomes tuned to the specific noise realization, inflating our confidence.

To avoid this, we split the data temporally (preserving measurement order):

- **Calibration set** (first 30%): Used to estimate covariance $\Sigma_0$ and set the prior scale
- **Inference set** (last 70%): Used for the CI gate (both test statistic and bootstrap critical value estimation) and Bayesian posterior computation

Why temporal rather than random split? Timing measurements exhibit autocorrelation—nearby samples are more similar than distant ones due to cache state, frequency scaling, etc. A random split would leak information; a temporal split keeps calibration and inference truly independent.

The 30/70 ratio balances two needs: enough calibration data for stable covariance estimation, enough inference data for statistical power. With 100k samples, the inference set still has 70k samples—plenty for precise quantile estimation.

**Minimum sample requirements:**

| Samples per class | Behavior |
|-------------------|----------|
| $\geq 50$ | Normal 30/70 split |
| 20–49 | Warning; use all data for both phases (accept double-dipping bias) |
| $< 20$ | Return Unmeasurable |

Below 50 samples, the split leaves too few calibration samples for reliable covariance estimation. We fall back to using all data for both phases, accepting that this makes the posterior slightly overconfident. Below 20, even this isn't meaningful.

### 2.4 Layer 1: CI Gate

The CI gate answers "is there a timing leak exceeding my concern threshold?" with controlled false positive rate. We use a relevant-hypothesis bootstrap test following SILENT [2], which properly controls FPR when the true effect is at or below the threshold.

#### The Relevant Hypotheses

Classical tests frame the hypotheses as "effect = 0" vs "effect ≠ 0". But we care about practical significance: does the effect exceed `min_effect_of_concern` (θ)? The relevant hypotheses are:

$$
H_0: d \leq \theta \quad \text{vs} \quad H_1: d > \theta
$$

where $d = \max_k |q_F(k) - q_R(k)|$ is the maximum absolute quantile difference.

**Why not just test "effect = 0" and add θ as a post-filter?**

SILENT explicitly warns against this common mistake: running a classical test (H₀: effect = 0), rejecting when significant, and *then* checking if the observed statistic exceeds θ. This two-step procedure does **not** control type-1 error at α.

The problem: at the boundary (true effect = θ), your estimator $\hat{d}$ is centered at θ. Half the time it lands above θ, half below. So you reject ~50% of the time at the boundary—regardless of what α you set for the classical test.

The fix: build θ into the hypothesis itself, and use a bootstrap that's calibrated to the relevant null (d ≤ θ), not the classical null (d = 0).

#### Test Statistic

For each quantile $k \in K = \{0.1, 0.2, \ldots, 0.9\}$, define:

$$
A_k = |\hat{q}_F(k) - \hat{q}_R(k)|
$$

The raw distance is $d = \max_k A_k$. Following SILENT (Section 5, equation before Algorithm 1), the test statistic is **studentized**:

$$
\hat{Q}_{max} = \max_{k \in K_{\text{sub}}^{\text{max}}} \frac{A_k - \theta}{\hat{\sigma}_k}
$$

where:
- $K_{\text{sub}}^{\text{max}} \subseteq K$ is a filtered subset of quantiles (see below)
- $\hat{\sigma}_k$ is the bootstrap-estimated standard deviation of $A_k$
- Division by $\hat{\sigma}_k$ normalizes each quantile's contribution by its variability

**Why studentize?** Without studentization, high-variance quantiles (often the tails) dominate the max, making the test insensitive to leaks in low-variance quantiles. Studentization gives each quantile equal "voting power" relative to its noise level.

#### Quantile Filtering (SILENT-style)

SILENT uses $\hat{\sigma}_k$ for both **filtering** (to construct $K_{\text{sub}}$) and **studentization** (inside the max). We filter quantiles in two steps:

**Step 1: Remove high-variance quantiles**

Quantiles with unusually high bootstrap variance (e.g., p90 on noisy data) can still dominate even after studentization. Following SILENT exactly:

$$
K_{\text{sub}} = \left\{ k \in K : \hat{\sigma}_k^2 \leq 5 \cdot \overline{\sigma^2} \right\}
$$

where $\overline{\sigma^2} = \frac{1}{|K|} \sum_{j \in K} \hat{\sigma}_j^2$ is the **mean** variance across quantiles. Also exclude quantiles with near-zero variance (< $10^{-10}$) to avoid division blow-ups.

**Step 2: Power-boost filter ($K_{\text{sub}}^{\text{max}}$)**

SILENT keeps only quantiles likely to contribute to detection. The exact criterion:

$$
K_{\text{sub}}^{\text{max}} = \left\{ k \in K_{\text{sub}} : A_k + 30 \cdot \hat{\sigma}_k \cdot \sqrt{\frac{(\log n)^{3/2}}{n}} \geq \theta \right\}
$$

The slack term $30 \cdot \sqrt{(\log n)^{3/2} / n}$ accounts for estimation uncertainty in $A_k$. This filter is applied by default (matching SILENT).

**Transparency:** Filtered quantiles are reported in `Diagnostics.filtered_quantiles` with reasons.

#### Algorithm (Continuous Timer Mode)

**Setup:**

1. Compute observed quantile differences $A_k = |\hat{q}_F(k) - \hat{q}_R(k)|$ for each $k \in K$
2. Compute raw distance $d = \max_k A_k$

**Bootstrap ($B = 2{,}000$ iterations for CI, $B = 10{,}000$ for thorough mode):**

SILENT's theory requires **paired resampling**: the same index set is used for both classes, preserving cross-covariance structure.

For each bootstrap iteration $b = 1, \ldots, B$:

1. Sample a set of block starting indices $I \subset \{1, \ldots, n - \ell + 1\}$
2. **Resample both classes using the same indices**: $F^{*(b)} = F[I]$, $R^{*(b)} = R[I]$
3. Compute bootstrap quantiles: $\hat{q}^{*(b)}_F(k)$, $\hat{q}^{*(b)}_R(k)$
4. Compute bootstrap differences: $A^{*(b)}_k = |\hat{q}^{*(b)}_F(k) - \hat{q}^{*(b)}_R(k)|$

**Critical:** Both classes use **block** bootstrap with **the same index set** (not separately sampled). This preserves:
- Autocorrelation structure within each class
- Cross-covariance between classes (common-mode drift)

SILENT's Theorem 2 guarantees bounded FPR under this scheme.

**Variance estimation (for filtering and studentization):**

From the bootstrap samples, compute per-quantile standard deviation:

$$
\hat{\sigma}_k = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B} \left( A^{*(b)}_k - \bar{A}_k \right)^2}, \quad \bar{A}_k = \frac{1}{B} \sum_{b=1}^{B} A^{*(b)}_k
$$

Use these to determine $K_{\text{sub}}^{\text{max}}$ (filtering) and for studentization in the test statistic.

**Centered, studentized bootstrap statistics:**

For each iteration $b$, compute:

$$
\hat{Q}^{*(b)}_{max} = \max_{k \in K_{\text{sub}}^{\text{max}}} \frac{A^{*(b)}_k - A_k}{\hat{\sigma}_k}
$$

Note: we subtract the *observed* $A_k$ (not θ) to center the bootstrap distribution around 0.

**Decision:**

1. Compute critical value: $c^*_{1-\alpha} = \text{Quantile}_{1-\alpha}(\{\hat{Q}^{*(1)}_{max}, \ldots, \hat{Q}^{*(B)}_{max}\})$
2. Compute test statistic: $\hat{Q}_{max} = \max_{k \in K_{\text{sub}}^{\text{max}}} \frac{A_k - \theta}{\hat{\sigma}_k}$
3. **Reject** (declare leak) if $\hat{Q}_{max} > c^*_{1-\alpha}$

**Note on SILENT's R implementation:** SILENT's R code includes an additional finite-sample bias correction using the Beta distribution, scaling with raw timing variance. We omit this correction because (1) Theorem 2's FPR guarantee holds for the uncorrected statistic, and (2) the correction's scaling with raw variance (rather than quantile-difference variance) causes false rejections when testing constant-time code with high measurement variance but identical class distributions.

#### Why This Works at the Boundary

The key insight is the **centered, studentized bootstrap**: $\hat{Q}^*_{max} = \max_k (A^*_k - A_k) / \hat{\sigma}_k$.

- The centering $(A^*_k - A_k)$ makes $\hat{Q}^*_{max}$ distributed around 0, regardless of the true effect size
- $\hat{Q}_{max}$ measures studentized excess over θ: $(A_k - \theta) / \hat{\sigma}_k$
- At the boundary (true effect = θ): observed $A_k \approx \theta$, so $\hat{Q}_{max} \approx 0$
- Since $\hat{Q}^*_{max}$ is centered at 0, we have $P(\hat{Q}_{max} > c^*_{1-\alpha}) \approx \alpha$ ✓

Away from the boundary:
- True effect < θ: $\hat{Q}_{max} < 0$ typically, so rejection rate < α (conservative)
- True effect > θ: $\hat{Q}_{max} > 0$, so power → 1 as n → ∞

#### Discrete Timer Mode

When the timer has low resolution, the continuous bootstrap CLT doesn't hold—quantile estimators behave differently with atoms (tied values). We switch to a discrete-aware mode that follows SILENT's Algorithm 4.

**Key difference from continuous mode:** Discrete mode uses **non-studentized** statistics with $\sqrt{m}$ scaling, unlike continuous mode's studentized form. This matches SILENT's separate treatment of discrete distributions.

**Trigger condition (SILENT's criterion):**

Discrete mode triggers when the **minimum uniqueness ratio** across both classes is below 10%:

$$
\min\left(\frac{|\text{unique}(F)|}{n_F}, \frac{|\text{unique}(R)|}{n_R}\right) < 0.10
$$

When triggered, set `Diagnostics.discrete_mode = true` and use the procedures below.

**Mid-distribution quantiles:**

Instead of the standard quantile estimator, use mid-distribution quantiles (via the `midquantile` function) which handle ties correctly:

$$
F_{\text{mid}}(x) = F(x) - \frac{1}{2}p(x), \quad \hat{q}^{\text{mid}}_k = F^{-1}_{\text{mid}}(k)
$$

where $p(x)$ is the probability mass at $x$. With integer ticks, compute quantiles from a histogram—this is efficient and numerically stable.

**m-out-of-n paired block bootstrap:**

Instead of resampling $n$ values, resample $m_1 = \lfloor n^{2/3} \rfloor$ values. Still use **paired** block resampling (same indices for both classes) to preserve cross-covariance.

**Asymmetric scaling (non-studentized):**

Standard m-out-of-n bootstrap theory (Bickel et al. 1997) states that $\sqrt{m}(\hat{\theta}^*_m - \hat{\theta}_n)$ approximates the distribution of $\sqrt{n}(\hat{\theta}_n - \theta)$. This means:

- **Bootstrap statistic** (scaled by $\sqrt{m_1}$, non-studentized):
$$
\hat{Q}^{*(b)} = \sqrt{m_1} \cdot \max_{k \in K_{\text{sub}}} \left( A^{*(b)}_k - A_k \right)
$$

- **Test statistic** (scaled by $\sqrt{n}$, non-studentized):
$$
\hat{Q} = \sqrt{n} \cdot \max_{k \in K_{\text{sub}}} (A_k - \theta)
$$

The asymmetry is intentional: the $\sqrt{m_1}$-scaled bootstrap distribution approximates the $\sqrt{n}$-scaled test statistic's distribution.

**Practical constraints on m:**

- **Minimum:** $m_1 \geq 400$ per class (enough to estimate deciles reliably)
- If $n < 2{,}000$: set $m_1 = \max(200, \lfloor 0.5n \rfloor)$ and add `QualityIssue::SmallSampleDiscrete`
- **Block length cap:** $\ell \leq m_1 / 5$ to avoid resamples being "one giant block"

**Work in ticks internally:**

In discrete mode, perform all computations in **ticks** (the timer's native unit), not nanoseconds:
- $A_k$, θ, effect sizes all in ticks
- Convert to nanoseconds only for display: "Δ ≈ 1.2 ticks ≈ 49ns"

This avoids introducing rounding errors from the tick-to-ns conversion factor.

**Threshold clamping:**

If the user's θ (in ns) would be < 1 tick after conversion, clamp to 1 tick and add `QualityIssue::ThresholdClamped`:

```rust
QualityIssue {
    code: IssueCode::ThresholdClamped,
    message: "θ=2ns clamped to 1 tick (41ns). Timer cannot resolve smaller effects.",
    guidance: "Use a cycle counter (perf/kperf) or accept reduced sensitivity.",
}
```

**Quantile filtering in discrete mode:**

High-variance filtering still applies, but with an adjustment for pinned quantiles. If a quantile has near-zero variance because both classes are pinned to the *same* tick, this is actually strong evidence of similarity—not noise to discard. In the Bayesian layer, consider treating such quantiles as having high precision centered at zero difference, rather than excluding them entirely.

**Bayesian layer in discrete mode:**

The Gaussian model is a rougher approximation with discrete data. We:
1. Estimate covariance using the same m-out-of-n bootstrap
2. Rescale: $\Sigma_n = \Sigma_m \cdot (m_1 / n)$
3. Report `QualityIssue::DiscreteTimer` on all Bayesian outputs
4. Frame leak_probability as "approximate posterior under Gaussianized model"

The CI gate (with m-out-of-n) remains the primary reliability guarantee.

**Validation requirements:**

Discrete mode is outside most practitioners' intuition. The implementation should include empirical validation:
- **Null at boundary:** true $d = \theta$ in ticks → gate fails ≈ α
- **Below threshold:** $d = 0.5\theta$ → fail rate ≤ α
- **Power curves:** $d = 1.5\theta$, $d = 2\theta$ → adequate power
- **Robustness:** AR(1) dependence, blocky drift, periodic noise
- **Tick size variation:** different timer resolutions

#### Batching Consistency

When using adaptive batching (§3.4), measurements are in batch-total scale. The threshold θ must be scaled to match:

$$
\theta_{\text{batch}} = K \cdot \theta
$$

where $K$ is the batch size. Use $\theta_{\text{batch}}$ in the CI gate computation.

**Caveat:** This scaling maintains consistent **threshold semantics**, but batching changes the microarchitectural regime. K operations executed together accumulate correlated state (branch predictor training, cache warming), so the distribution of batch totals is not simply K × the single-operation distribution. Do not assume decision invariance across batching strategies; treat large K as a regime change. The K ≤ 20 limit (§3.4) bounds but does not eliminate this effect.

The reported `max_distance_ns` in `CiGate` is always converted back to per-operation scale for interpretability.

#### FPR Guarantee

By construction, when the true effect $d \leq \theta$:

$$
\limsup_{n \to \infty} P(\hat{Q} > c^*_{1-\alpha}) \leq \alpha
$$

with equality when $d = \theta$ (the least favorable point in the null). This is the SILENT guarantee [2, Theorem 2].

**Monte Carlo error in critical value estimation:**

The bootstrap critical value $c^*_{1-\alpha}$ is estimated from B samples. For α = 0.01 (1% FPR), only ~B×α samples land beyond the $(1-\alpha)$ quantile. With B = 2,000, that's only ~20 samples in the tail, giving:

$$
\text{SE}(\hat{\alpha}) = \sqrt{\frac{\alpha(1-\alpha)}{B}} \approx 0.0022
$$

Relative to α = 0.01, this is **~22% relative error**—not negligible. This is a primary source of CI gate flakiness at strict α levels.

**Recommendation:** For stable 1% gates, use $B \geq 10{,}000$ (thorough mode). With B = 10,000, the relative error drops to ~10%, which is acceptable for most applications. The heuristic $B \gtrsim 50/\alpha$ ensures adequate tail resolution.

### 2.5 Layer 2: Bayesian Inference

The Bayesian layer provides what the CI gate cannot: a probability that there's a leak worth caring about, and an estimate of how big it is. We use a conjugate Gaussian model that admits closed-form posteriors—no MCMC required.

#### Design Matrix

We decompose the observed quantile differences into interpretable components:

- **Uniform shift** ($\mu$): All quantiles move by the same amount (e.g., a branch that adds constant overhead)
- **Tail effect** ($\tau$): Upper quantiles shift more than lower ones (e.g., cache misses that occur probabilistically)

The design matrix encodes this decomposition:

$$
X = \begin{bmatrix} \mathbf{1} & \mathbf{b}_{\text{tail}} \end{bmatrix}
$$

where:
- $\mathbf{1} = (1, 1, 1, 1, 1, 1, 1, 1, 1)^\top$ — a uniform shift moves all 9 quantiles equally
- $\mathbf{b}_{\text{tail}} = (-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5)^\top$ — a tail effect moves upper quantiles up and lower quantiles down, centered at zero

The tail basis is centered (sums to zero) so that $\mu$ and $\tau$ are orthogonal. An effect of $\mu = 10, \tau = 0$ means a pure 10ns uniform shift; $\mu = 0, \tau = 20$ means upper quantiles are 10ns slower while lower quantiles are 10ns faster than the mean.

**Design limitation:** The antisymmetric tail basis assumes symmetric "lower-faster, upper-slower" patterns. Real cache-timing leaks often show:
- Lower quantiles unchanged (~100 cycles baseline)
- Upper quantiles elevated (~1000 cycles from cache misses)

The model *can* fit this pattern (μ absorbs the mean shift, τ captures the spread), but the interpretation of τ becomes less intuitive. A future version could add a third basis capturing asymmetric upper-tail effects, though this increases model complexity (9×3 GLS is still cheap).

#### Model

We model the observed quantile differences as:

$$
\Delta = X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \Sigma_0)
$$

where $\beta = (\mu, \tau)^\top$ are the effect magnitudes and $\Sigma_0$ is the null covariance estimated via bootstrap (§2.6).

#### Prior Specification

We place a zero-centered Gaussian prior on $\beta$:

$$
\beta \sim \mathcal{N}(0, \Lambda_0), \quad \Lambda_0 = \text{diag}(\sigma_\mu^2, \sigma_\tau^2)
$$

The prior scale determines regularization strength. We set it proportional to the threshold:

$$
\sigma_\mu = \sigma_\tau = 2\theta
$$

where θ is the minimum effect of concern (from the attacker model). This ensures **calibrated priors**: $P(|\beta| > \theta \mid \text{prior}) \approx 62\%$, representing genuine uncertainty.

> **Rationale**: The previous formula $\max(2 \cdot \text{MDE}, \theta)$ caused poor calibration when MDE >> θ (noisy data with strict threshold). The prior became too diffuse, giving $P(|\beta| > \theta \mid \text{prior}) \approx 99\%$ before seeing any data—biasing toward "leak detected." By tying the prior scale directly to θ, we ensure the prior reflects genuine uncertainty about whether effects exceed the threshold.

The variance ratio quality gate (§3.4) catches cases where posterior ≈ prior, indicating the data wasn't informative enough to update our beliefs.

#### Posterior Distribution

With Gaussian likelihood and Gaussian prior, the posterior on $\beta$ is also Gaussian (standard conjugate result⁴):

$$
\beta \mid \Delta \sim \mathcal{N}(\beta_{\text{post}}, \Lambda_{\text{post}})
$$

where:

$$
\Lambda_{\text{post}} = \left( X^\top \Sigma_0^{-1} X + \Lambda_0^{-1} \right)^{-1}
$$

$$
\beta_{\text{post}} = \Lambda_{\text{post}} X^\top \Sigma_0^{-1} \Delta
$$

The posterior mean $\beta_{\text{post}} = (\mu, \tau)^\top$ gives the estimated shift and tail effects in nanoseconds.

⁴ Bishop, C. M. (2006). Pattern Recognition and Machine Learning, §3.3. Springer.

#### Leak Probability via Posterior Integration

The key question is: "what's the probability of a leak worth caring about?" The user specifies this threshold via `min_effect_of_concern` (θ). We compute:

$$
P(\text{significant leak} \mid \Delta) = P\bigl(\max_k |(X\beta)_k| > \theta \;\big|\; \Delta\bigr)
$$

Since the posterior is Gaussian, we compute this by Monte Carlo integration:

```
Draw N = 1000 samples β ~ N(β_post, Λ_post)
For each sample:
  pred = X @ β                    # predicted Δ vector (9 quantile differences)
  max_effect = max_k |pred[k]|    # same metric structure as CI gate
  if max_effect > θ: count++
leak_probability = count / N
```

**Calibration caveat:** This probability is **internally consistent under the Gaussian model**—if we report 80%, then 80% of the posterior predictive mass exceeds θ. However, it is not automatically **frequentist-calibrated** in the "long-run 80% of the time" sense unless:
- The Gaussian likelihood is well-specified (quantile estimators can be non-Gaussian with dependence or discrete timers)
- The linear basis $X\beta$ adequately captures the true effect structure

We validate calibration empirically on null and near-null benchmarks. The CI gate (not this layer) provides the bounded-FPR guarantee.

**Relationship to CI gate (approximate, not exact):**

The CI gate uses **absolute** quantile differences: $d = \max_k |A_k|$ where $A_k = |\hat{q}_F(k) - \hat{q}_R(k)|$.

The Bayesian layer uses **signed** differences $\Delta_k = \hat{q}_F(k) - \hat{q}_R(k)$ with a 2D linear model. The leak probability $P(\max_k |(X\beta)_k| > \theta)$ is a **model-based approximation** to the gate's statistic, not literally the same quantity.

In practice they usually agree because:
- When there's a real effect, both detect it
- The linear model $X\beta$ captures the dominant patterns (shift and tail)

But divergence is possible in edge cases (complex non-linear patterns, high uncertainty straddling θ).

**Interpreting the tail basis:**

With the tail basis $b_{\text{tail}} \in [-0.5, +0.5]$, the predicted difference at quantile $k$ is:

$$
(X\beta)_k = \mu + \tau \cdot b_{\text{tail}}(k)
$$

When $\mu = 0$, the maximum absolute effect from $\tau$ alone is $0.5 \cdot |\tau|$ (at the extreme quantiles). So a "10ns tail effect" produces at most ~5ns difference at any single quantile.

**Example with default `min_effect_of_concern = 10ns`:**

| Posterior mean (μ, τ) | Max predicted |Δ| | Approx. leak probability |
|-----------------------|------------------|--------------------------|
| (2ns, 1ns) | ~2.5ns | ~2% |
| (8ns, 4ns) | ~10ns | ~50% |
| (12ns, 5ns) | ~14.5ns | ~85% |

**Additional diagnostics (reported but not used for leak_probability):**

We also report marginal probabilities for interpretability:
- $P(|\mu| > \theta \mid \Delta)$: probability shift component exceeds threshold
- $P(|\tau| > \theta \mid \Delta)$: probability tail component exceeds threshold

These help diagnose *what kind* of leak exists but aren't used for the headline `leak_probability`.

#### Effect Estimation

We always report effect estimates (not just when leak_probability > 0.5). The posterior mean $\beta_{\text{post}}$ gives the point estimate; the posterior covariance $\Lambda_{\text{post}}$ quantifies uncertainty.

**Effect pattern classification:**

An effect component is "significant" if its magnitude exceeds twice its posterior standard error:

$$
\text{significant}(\mu) \iff |\mu| > 2 \cdot \sqrt{\Lambda_{\text{post}}[0,0]}
$$

| Shift significant? | Tail significant? | Pattern |
|--------------------|-------------------|---------|
| Yes | No | UniformShift |
| No | Yes | TailEffect |
| Yes | Yes | Mixed |
| No | No | Classify by relative magnitude: if $|\mu| > |\tau|$ then UniformShift, else TailEffect |

**Credible interval:** We draw 1000 samples from the posterior (reusing the leak probability samples), compute the total effect magnitude $\lVert\beta\rVert_2$ for each, and report the 2.5th and 97.5th percentiles as a 95% credible interval.

### 2.6 Covariance Estimation

Both the CI gate and Bayesian layer need the null covariance $\Sigma_0$, estimated via block bootstrap on the calibration set. Block bootstrap preserves autocorrelation structure (standard bootstrap assumes i.i.d., underestimating variance).

**Politis-White automatic block length selection:**

Following SILENT, we use the Politis-White algorithm³ to select the optimal block length, computing it separately for each class and taking the maximum:

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

For the stationary bootstrap:

$$
b^{\text{opt}} = \left( \frac{g^2}{(\hat{\sigma}^2)^2} \right)^{1/3} n^{1/3}
$$

Capped at $b_{\max} = \min(3\sqrt{n}, n/3)$ to prevent degenerate blocks.

³ Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection for the Dependent Bootstrap." Econometric Reviews 23(1):53–70. See also the correction: Patton, A., Politis, D. N., & White, H. (2009). Econometric Reviews 28(4):372–375.

**Procedure (calibration bootstrap for Σ₀):**

For each of $B = 2{,}000$ bootstrap iterations on the **calibration set**:
1. Generate block indices; apply to both classes (paired resampling)
2. Compute quantile vectors $\hat{q}_F^*$, $\hat{q}_R^*$ from resampled data
3. Record $\Delta^* = \hat{q}_F^* - \hat{q}_R^*$

The CI gate runs its own bootstrap on the **inference set** to compute critical values (§2.4). The calibration bootstrap here estimates Σ₀ for the Bayesian layer only.

Compute sample covariance of the $\Delta^*$ vectors directly using Welford's online algorithm⁵ (numerically stable for large $B$).

⁵ Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419–420. See also Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). "Algorithms for Computing the Sample Variance." The American Statistician 37(3):242–247 for the parallel/incremental generalization.

**Rescaling for inference set (30/70 split):**

Covariance scales as $1/n$. Since estimation uses the calibration set but inference uses a different set, rescale:

$$
\Sigma_0 = \hat{\Sigma}_{\text{cal}} \cdot \frac{n_{\text{cal}}}{n_{\text{inf}}}
$$

With 30/70 split: multiply by ~0.43. This shrinks the covariance (larger set = lower variance).

**Paired resampling and cross-covariance:**

We use **paired resampling**: generate one set of block indices, apply it to both per-class sequences independently:

```
F = [F₁, F₂, ..., Fₙ]  # Fixed-class samples in measurement order
R = [R₁, R₂, ..., Rₙ]  # Random-class samples in measurement order

indices = sample_block_indices(n, block_length)
F* = F[indices]
R* = R[indices]
```

This preserves cross-covariance from common-mode noise (thermal drift, frequency scaling) that affects both classes at similar experimental positions. Note: with randomized interleaving, Fᵢ and Rᵢ are not measured at the same instant, but at similar aggregate positions in the experiment. This matches SILENT's theoretical requirements.

**Numerical stability:**

In discrete mode with many ties, some quantiles may have zero variance, making $\Sigma_0$ ill-conditioned. Even if Cholesky succeeds, the inverse has huge values for near-zero variance elements, causing them to dominate the Bayesian regression incorrectly.

We regularize by ensuring a minimum diagonal value of 1% of mean variance, bounding the condition number to ~100:

$$
\sigma^2_i \leftarrow \max(\sigma^2_i, 0.01 \cdot \bar{\sigma}^2) + \varepsilon
$$

where $\bar{\sigma}^2 = \text{tr}(\Sigma)/9$ and $\varepsilon = 10^{-10} + \bar{\sigma}^2 \cdot 10^{-8}$.

If Cholesky still fails after regularization, set `MeasurementQuality::TooNoisy` and `leak_probability = 0.5` (maximally uncertain). The CI gate still runs.

### 2.7 Minimum Detectable Effect

The MDE answers: "what's the smallest effect I could reliably detect given the noise level?" Important for interpreting negative results: if MDE > θ, a pass means insufficient sensitivity, not safety.

**Derivation:**

Under the linear model with GLS estimator $\hat{\beta} = (X^\top \Sigma_0^{-1} X)^{-1} X^\top \Sigma_0^{-1} \Delta$, the MDE at significance $\alpha$ with 50% power is:

$$
\text{MDE}_\mu = z_{1-\alpha/2} \cdot \sqrt{\left( \mathbf{1}^\top \Sigma_0^{-1} \mathbf{1} \right)^{-1}}, \quad
\text{MDE}_\tau = z_{1-\alpha/2} \cdot \sqrt{\left( \mathbf{b}_{\text{tail}}^\top \Sigma_0^{-1} \mathbf{b}_{\text{tail}} \right)^{-1}}
$$

**Scaling:** MDE ∝ $1/\sqrt{n}$ (4× samples → 2× better). If MDE > `min_effect_of_concern`, consider more samples.

---

## 3. Measurement Model

This section describes how timing samples are collected: timer selection, warmup, interleaving, outlier handling, and adaptive batching for coarse-resolution platforms.

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

Before measurement begins, several sanity checks detect common problems that would invalidate results:

**Timer sanity**: Verify the timer is monotonic and has reasonable resolution. Abort if the timer appears broken.

**Harness sanity (fixed-vs-fixed)**: Split the fixed samples in half and run the full analysis pipeline. If a "leak" is detected between two halves of identical inputs, something is wrong with the test harness—perhaps the closure captures mutable state, or the timer has systematic bias. This catches bugs that would otherwise produce false positives.

**Generator overhead**: Measure the random input generator in isolation. If the fixed and random generators differ in cost by more than 10%, **abort with an error** (not just a warning). The measured difference would reflect generator cost rather than the operation under test. This check catches API misuse where inputs are generated inside the timed region (see §3.3). A well-designed API should make this check always pass.

### 3.2.1 Stationarity Check

SILENT explicitly lists non-stationarity as a known failure mode: if the measurement distribution drifts during the test (e.g., thermal throttling, background load changes, GC pauses), the bootstrap assumptions break down.

We implement a cheap rolling-variance check:

1. Divide samples into $W$ windows (default: 10 windows)
2. Compute median and IQR within each window
3. Flag `stationarity_suspect = true` if:
   - Max window median differs from min window median by > $\max(2 \times \text{IQR}, \text{drift\_floor})$
   - Or window variance increases/decreases monotonically by > 50%

The `drift_floor` prevents false positives in very quiet environments where IQR → 0. Default: $\text{drift\_floor} = 0.05 \times \text{global\_median}$ (5% of the overall median timing).

This doesn't prove non-stationarity (that would require more sophisticated tests), but it catches obvious drift patterns that explain "why did CI flake?"

**When stationarity is suspect:**

```rust
QualityIssue {
    code: IssueCode::StationaritySuspect,
    message: "Timing distribution appears to drift during measurement (median shift: 15ns)",
    guidance: "Try: longer warmup, CPU frequency pinning (disable turbo/C-states), or run during stable system load",
}
```

The test still produces results (we don't abort), but the quality is downgraded and the issue is surfaced prominently.

### 3.2.2 Dependence Estimation

SILENT emphasizes that timing measurements are often dependent due to caches, TLB, frequency scaling, etc. We estimate the dependence length $m$ using an autocorrelation-based heuristic.

**Critical:** Compute ACF **per class** (Fixed and Random separately), not on the interleaved sequence. The interleaved sequence has artificial structure from the measurement schedule (alternating F/R) that would contaminate the dependence estimate.

**Algorithm:**

1. Extract the Fixed-class subsequence $F_1, F_2, \ldots, F_{n/2}$ and Random-class subsequence $R_1, R_2, \ldots, R_{n/2}$
2. Demean each class: $\tilde{F}_i = F_i - \bar{F}$, $\tilde{R}_i = R_i - \bar{R}$
3. Compute sample autocorrelation $\rho_F(h)$ and $\rho_R(h)$ for lags $h = 1, 2, \ldots, h_{\max}$
4. Use the **maximum** of the two: $\hat{m} = \max(\hat{m}_F, \hat{m}_R)$ where $\hat{m}_c$ is the smallest $h$ where $|\rho_c(h)| < 2/\sqrt{n/2}$

This estimate is used for ESS computation and diagnostics. The actual block length for bootstrap uses the full Politis-White algorithm (§2.6).

**Effective sample size:**

$$
\text{ESS} \approx \frac{n}{1 + 2\sum_{h=1}^{\hat{m}} \rho(h)} \approx \frac{n}{\hat{m}}
$$

This is surfaced in `Diagnostics.effective_sample_size`. When ESS << n, the nominal sample count is misleading:

```rust
QualityIssue {
    code: IssueCode::HighDependence,
    message: "Estimated dependence length: 47 samples (ESS: 2,100 of 100,000)",
    guidance: "High autocorrelation reduces effective sample size. Try: isolate to dedicated core, disable hyperthreading, or increase sample count to compensate",
}
```

**Periodic interference check**: If lag-1 or lag-2 autocorrelation (in either class) exceeds 0.3, warn about periodic interference—likely from background processes, frequency scaling, or interrupt handlers. High autocorrelation inflates variance estimates and can cause both false positives and false negatives.

**CPU frequency governor (Linux)**: Check `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`. If not set to "performance", warn that frequency scaling may introduce noise. The "powersave" or "ondemand" governors cause the CPU to change frequency mid-measurement, adding variance unrelated to the code under test.

**Warmup**: Run the operation several times before measurement to warm caches, trigger JIT (if relevant), and stabilize frequency scaling. Default: 1,000 iterations.

### 3.3 Measurement Protocol

**Input pre-generation (critical):**

All inputs must be generated *before* the measurement loop begins. The timed region should contain only:
1. Input retrieval (loading a pre-generated value)
2. The operation under test
3. Output consumption (preventing dead-code elimination)

Input generation—especially random number generation—must happen outside the timed region. If random inputs are generated lazily inside the measurement closure, the generator overhead (often 20–100ns for cryptographic RNG) will be attributed to the operation, causing false positives.

```
// WRONG: Generator called inside timed region
for i in 0..n {
    let input = if schedule[i] == Fixed { &fixed } else { rng.gen() };  // ← RNG inside timing
    let t0 = timer.now();
    operation(input);
    let t1 = timer.now();
    record(t1 - t0);
}

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

The "Generator overhead" pre-flight check (§3.2) detects violations of this requirement, but proper API design should make violations impossible.

**Interleaved randomized sampling:**

Interleave Fixed and Random in shuffled order to prevent systematic drift from biasing one class.

**Outlier handling (winsorization):**

We **cap** (winsorize), not drop, outliers to preserve signal from tail-heavy leaks:

1. Compute $t_{\text{cap}}$ = 99.99th percentile from **pooled data** (Fixed ∪ Random)
2. Cap samples exceeding $t_{\text{cap}}$ (set $t = t_{\text{cap}}$)
3. Winsorization happens **before** quantile computation and bootstrap resampling

**Why pooled:** Using null-derived threshold is dangerous when Fixed is fast—Random's slow-path samples would be capped away, hiding real leaks.

**Quality thresholds:** >0.1% capped → warning; >1% → Acceptable; >5% → TooNoisy.

### 3.4 Adaptive Batching

On platforms with coarse timers (cntvct at 41ns), fast operations may complete in fewer ticks than needed for reliable measurement.

**When batching is needed:**

If a pilot measurement shows fewer than 5 ticks per call, enable batching.

**Batch size selection:**

Choose $K$ such that $K$ operations take approximately 50 ticks:

$$
K = \text{clamp}\left( \left\lceil \frac{50}{\text{ticks\_per\_call}} \right\rceil, 1, 20 \right)
$$

The maximum of 20 prevents microarchitectural artifacts (branch predictor, cache state) from accumulating across too many iterations.

**Interpretation caution for large K:**

When K > 10, interpret results more cautiously:
- The batch measures K *correlated* operations (predictor trained, caches warm)
- The distribution of batch totals may not relate simply to single-operation behavior
- Effects that appear in batched measurements may not manifest in single calls (and vice versa)
- Consider flagging results with K > 10 as requiring additional verification

**Effect scaling:**

When batching is enabled, the measured time is for $K$ operations. The reported effect size must be divided by $K$ to give per-operation estimates.

**Reporting:**

Results include batching metadata ($K$ value, achieved ticks, rationale) so users understand when batching was applied.

### 3.5 Measurability Thresholds

Some operations are too fast to measure reliably on a given platform.

**Thresholds:**

- If ticks per call $< 5$ (even with maximum batching): Return `Outcome::Unmeasurable`
- This typically means operations under ~200ns on Apple Silicon without kperf

**Unmeasurable result:**

Rather than returning unreliable statistics, the system explicitly indicates the limitation:

```rust
Outcome::Unmeasurable {
    operation_ns: 8.2,
    threshold_ns: 205.0,
    platform: "Apple Silicon (cntvct)",
    recommendation: "Run with sudo for kperf, or test a more complex operation"
}
```

### 3.6 Anomaly Detection

Users sometimes make mistakes that produce valid code but meaningless results—most commonly, capturing a pre-evaluated random value instead of regenerating it:

```rust
let value = rand::random();  // Evaluated once!
timing_test! {
    fixed: [0u8; 32],
    random: || value,  // Always returns the same thing
    test: |input| compare(&input),
}
```

**Detection mechanism:**

During measurement, track the uniqueness of random inputs by hashing the first 1,000 values generated. After measurement:

| Condition | Action |
|-----------|--------|
| All samples identical | Print error to stderr |
| $< 50\%$ unique | Print warning to stderr |
| Normal entropy | Silent |

**Type constraints:**

Anomaly detection requires the input type to be hashable. For types that don't support hashing (common in cryptography: scalars, field elements, big integers), detection is automatically skipped with zero overhead via autoref specialization.

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
    Timeout { current_probability: f64, samples_collected: usize },

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

The most important configuration choice is your **attacker model**, which determines what size leak you consider "negligible." SILENT's key insight [2] is that the right question isn't "is there any timing difference?" but "is the difference > θ under my threat model?"

**There is no single correct θ.** Your choice of preset is a statement about your threat model.

```rust
pub enum AttackerModel {
    /// Co-resident attacker with cycle-level timing.
    /// θ = 0.6 ns (~2 cycles @ 3GHz)
    ///
    /// Use for: SGX enclaves, cross-VM attacks, containers on shared hosts,
    /// hyperthreading siblings, local privilege escalation.
    ///
    /// Source: Kario argues even 1 cycle is detectable when attacker shares
    /// hardware. This is the strictest practical threshold.
    SharedHardware,

    /// Adjacent network attacker (LAN, HTTP/2 multiplexing).
    /// θ = 100 ns
    ///
    /// Use for: Internal services, microservices, HTTP/2 APIs where
    /// Timeless Timing Attacks apply (request multiplexing eliminates jitter).
    ///
    /// Source: Crosby et al. (2009) report ~100ns LAN accuracy.
    /// Van Goethem et al. (2020) show HTTP/2 enables similar precision remotely.
    AdjacentNetwork,

    /// Remote network attacker (general internet).
    /// θ = 50 μs
    ///
    /// Use for: Public APIs, web services, legacy HTTP/1.1 services.
    ///
    /// Source: Crosby et al. (2009) report 15–100μs internet accuracy;
    /// 50μs is a reasonable midpoint for general internet exposure.
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
| `AdjacentNetwork` | 100 ns | LAN, HTTP/2 (Timeless Timing Attacks) |
| `RemoteNetwork` | 50 μs | Internet, legacy services |
| `Research` | 0 | Profiling, debugging (not for CI) |

**Sources:**

- **Crosby et al. (2009)**: "Opportunities and Limits of Remote Timing Attacks." ACM TISSEC.
- **Van Goethem et al. (2020)**: "Timeless Timing Attacks." USENIX Security. Shows HTTP/2 request multiplexing enables LAN-like precision over the internet.
- **Kario**: Argues timing differences as small as one clock cycle can be detectable when sharing hardware.

**Recommended usage:**

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome};
use std::time::Duration;

// Choose based on deployment scenario
let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .time_budget(Duration::from_secs(30))
    .max_samples(100_000)
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

#### TimingOracle

The main configuration type. Use `for_attacker()` as the entry point:

```rust
impl TimingOracle {
    /// Create oracle for a specific attacker model (recommended entry point).
    pub fn for_attacker(model: AttackerModel) -> Self;

    // Adaptive sampling configuration
    pub fn time_budget(self, duration: Duration) -> Self;  // Max time to spend
    pub fn max_samples(self, n: usize) -> Self;            // Max samples per class
    pub fn batch_size(self, k: u32) -> Self;               // Force specific batch size

    // Decision thresholds
    pub fn pass_threshold(self, p: f64) -> Self;           // P(leak) < p → Pass (default 0.05)
    pub fn fail_threshold(self, p: f64) -> Self;           // P(leak) > p → Fail (default 0.95)

    // Run the test
    pub fn test<T, F>(self, inputs: InputPair<T>, operation: F) -> Outcome
    where
        F: FnMut(&T);
}
```

**Example configurations:**

```rust
use timing_oracle::{TimingOracle, AttackerModel};
use std::time::Duration;

// Quick check during development (10 second budget)
TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .time_budget(Duration::from_secs(10))

// Thorough CI check (60 second budget, up to 100k samples)
TimingOracle::for_attacker(AttackerModel::SharedHardware)
    .time_budget(Duration::from_secs(60))
    .max_samples(100_000)

// Custom thresholds for specific requirements
TimingOracle::for_attacker(AttackerModel::Custom { threshold_ns: 500.0 })
    .pass_threshold(0.01)   // Very confident pass
    .fail_threshold(0.99)   // Very confident fail
```

#### Key Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `time_budget` | 30 seconds | Maximum time to spend on analysis |
| `max_samples` | 100,000 | Maximum samples per class |
| `batch_size` | auto | Operations per timing measurement (auto-tuned) |
| `pass_threshold` | 0.05 | P(leak) below this → Pass |
| `fail_threshold` | 0.95 | P(leak) above this → Fail |

### 4.3 Macro API

```rust
use timing_oracle::{timing_test_checked, TimingOracle, AttackerModel, Outcome};
use std::time::Duration;

let outcome = timing_test_checked! {
    oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30)),
    baseline: || [0u8; 32],
    sample: || rand::random::<[u8; 32]>(),
    measure: |input| my_function(&input),
};

match outcome {
    Outcome::Pass { .. } => println!("No timing leak"),
    Outcome::Fail { exploitability, .. } => println!("Leak: {:?}", exploitability),
    Outcome::Inconclusive { reason, .. } => println!("Inconclusive"),
    Outcome::Unmeasurable { .. } => println!("Too fast to measure"),
}
```

The `sample` field requires a closure (`|| expr`)—this syntax mirrors the semantics.

**Multiple inputs**: Use tuple destructuring: `baseline: || (a, b), sample: || (ra, rb), measure: |(x, y)| ...`

### 4.4 Builder API

```rust
use timing_oracle::{TimingOracle, AttackerModel, Outcome, helpers::InputPair};
use std::time::Duration;

let inputs = InputPair::new(
    || [0u8; 32],
    || rand::random::<[u8; 32]>(),
);

let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
    .time_budget(Duration::from_secs(30))
    .test(inputs, |data| my_function(data));
```

`InputPair` separates input generation from measurement. Both closures return owned `T`.

| API | Returns | Best for |
|-----|---------|----------|
| `timing_test_checked!` | `Outcome` | Most users |
| Builder API | `Outcome` | Programmatic control |

---

## 5. Interpreting Results

### 5.1 Leak Probability

The `leak_probability` is $P(\max_k |(X\beta)_k| > \theta \mid \Delta)$—the posterior probability of an effect exceeding your concern threshold.

| Probability | Interpretation |
|-------------|----------------|
| $< 10\%$ | Probably safe |
| $10\%$–$50\%$ | Inconclusive |
| $50\%$–$90\%$ | Probably leaking |
| $> 90\%$ | Almost certainly leaking |

This is calibrated by construction under the Gaussian model (§2.5). It is *not* a frequentist p-value.

### 5.2 CI Gate

Binary pass/fail with controlled FPR ≤ α at the boundary (true effect = θ). Uses SILENT's relevant-hypothesis framework.

- **Pass**: $\hat{Q}_{max} \leq c^*_{1-\alpha}$
- **Fail**: Test statistic exceeded critical value

Key fields: `max_distance_ns` (raw max quantile difference), `q_hat_max` (test statistic), `margin` ($c^* - \hat{Q}_{max}$).

### 5.3 Effect Size

- **Shift (μ)**: Uniform timing difference. Cause: different code path, branch on secret.
- **Tail (τ)**: Upper quantiles affected more. Cause: cache misses, secret-dependent memory access.

Both reported in nanoseconds with 95% credible intervals.

### 5.4 Exploitability

Rough risk assessment based on Crosby et al. (2009):

| Level | Effect Size | Notes |
|-------|-------------|-------|
| Negligible | $< 100$ ns | LAN limit; requires ~10k+ queries |
| PossibleLAN | $100$–$500$ ns | LAN exploitable with ~1k–10k queries |
| LikelyLAN | $500$ ns – $20$ μs | Readily exploitable on LAN |
| PossibleRemote | $> 20$ μs | Potentially exploitable over internet |

**Caveat**: 2009 thresholds. Modern attacks may achieve better resolution. Treat as heuristics for prioritization.

### 5.5 Quality Assessment

Based primarily on MDE relative to `min_effect_of_concern`:

| MDE | Quality |
|-----|---------|
| $< 5$ ns | Excellent |
| $5$–$20$ ns | Good |
| $20$–$100$ ns | Acceptable |
| $> 100$ ns | TooNoisy |

`TooNoisy` triggers when: MDE > 100ns, winsorization > 5%, severe autocorrelation, or timer insufficient even with batching.

### 5.6 Reliability

A result is reliable if measurement completed AND (quality ≠ TooNoisy OR posterior is conclusive).

```rust
impl Outcome {
    pub fn is_reliable(&self) -> bool;
}
```

For CI: `skip_if_unreliable!` (fail-open) or `require_reliable!` (fail-closed).

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $\Delta$ | 9-dimensional vector of signed quantile differences (Bayesian layer) |
| $\Delta_p$ | Quantile difference at percentile $p$: $\hat{q}_F(p) - \hat{q}_R(p)$ |
| $A_k$ | Absolute quantile difference at level $k$: $|\hat{q}_F(k) - \hat{q}_R(k)|$ |
| $d$ | Max distance: $\max_k A_k$ |
| $\hat{Q}_{max}$ | CI gate test statistic (continuous): $\max_{k \in K_{\text{sub}}^{\text{max}}} (A_k - \theta) / \hat{\sigma}_k$ — studentized |
| $\hat{Q}$ | CI gate test statistic (discrete): $\sqrt{n} \cdot \max_{k \in K_{\text{sub}}} (A_k - \theta)$ — not studentized |
| $\hat{Q}^*_{max}$ | Bootstrap statistic (continuous): $\max_k (A^*_k - A_k) / \hat{\sigma}_k$ |
| $\hat{Q}^*$ | Bootstrap statistic (discrete): $\sqrt{m_1} \cdot \max_k (A^*_k - A_k)$ |
| $c^*_{1-\alpha}$ | Bootstrap critical value for CI gate |
| $K_{\text{sub}}$, $K_{\text{sub}}^{\text{max}}$ | Filtered subset of quantiles (high-variance excluded) |
| $\hat{\sigma}_k$ | Bootstrap-estimated standard deviation of $A_k$ (for filtering AND studentization) |
| $\Sigma_0$ | Null covariance matrix for Bayesian layer (estimated via bootstrap) |
| $X$ | $9 \times 2$ design matrix $[\mathbf{1} \mid \mathbf{b}_{\text{tail}}]$ |
| $\beta = (\mu, \tau)^\top$ | Effect parameters: uniform shift and tail effect |
| $\Lambda_0$ | Prior covariance for $\beta$ |
| $\Lambda_{\text{post}}$ | Posterior covariance for $\beta$ |
| $\theta$ | Practical significance threshold (`min_effect_of_concern`) |
| $\alpha$ | CI gate target false positive rate |
| MDE | Minimum detectable effect |
| $K$ | Batch size for adaptive batching |
| $n$ | Samples per class |
| $\hat{b}$ | Block length for bootstrap (Politis-White) |
| $m_1$ | Resample size for discrete mode: $\lfloor n^{2/3} \rfloor$ |
| $B$ | Bootstrap iterations |

**Note on continuous vs discrete mode:**
- Continuous mode uses **studentized** statistics (dividing by $\hat{\sigma}_k$), matching SILENT's Algorithm 1
- Discrete mode uses **non-studentized** $\sqrt{n}$-scaled statistics with m-out-of-n bootstrap, matching SILENT's Algorithm 4
- In discrete mode, bootstrap uses $\sqrt{m_1}$ while test statistic uses $\sqrt{n}$ (asymmetry per Bickel et al. 1997)

## Appendix B: Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| Deciles | $\{0.1, 0.2, \ldots, 0.9\}$ | Nine quantile positions |
| CI bootstrap iterations (default) | $2{,}000$ | Balance between accuracy and CI speed |
| CI bootstrap iterations (thorough) | $10{,}000$ | For local investigation / nightly |
| Covariance bootstrap iterations | $2{,}000$ | Same as CI gate default |
| Posterior samples | $1{,}000$ | For leak probability Monte Carlo integration |
| Block length | Politis-White | Automatic selection, max of both classes |
| Block length cap | $\min(3\sqrt{n}, n/3)$ | Prevent degenerate blocks |
| Variance filter ($K_{\text{sub}}$) | $5 \times \text{mean}(\sigma^2)$ | SILENT: exclude quantiles with $\hat{\sigma}_k^2 > 5 \cdot \overline{\sigma^2}$ |
| Power-boost slack ($K_{\text{sub}}^{\text{max}}$) | $30 \cdot \sqrt{(\log n)^{3/2}/n}$ | SILENT's exact formula |
| Continuous/discrete threshold | 10% uniqueness | Discrete mode if $\min(\text{unique}/n) < 0.10$ |
| Min ticks per call | $5$ | Below this, quantization noise dominates |
| Target ticks per batch | $50$ | Target for adaptive batching |
| Max batch size | $20$ | Limit microarchitectural artifacts |
| Default $\alpha$ | $0.01$ | 1% false positive rate |
| Default min effect | $10$ ns | Threshold for significant leak |

## Appendix C: References

**Statistical methodology:**

1. Dunsche, M., Lamp, M., & Pöpper, C. (2024). "With Great Power Come Great Side Channels: Statistical Timing Side-Channel Analyses with Bounded Type-1 Errors." USENIX Security. — RTLF bootstrap methodology

2. Dunsche, M., Lamp, M., & Pöpper, C. (2025). "SILENT: A New Lens on Statistics in Software Timing Side Channels." arXiv:2504.19821. — Extension supporting negligible leak thresholds

3. Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." Annals of Statistics. — Block bootstrap for autocorrelated data

4. Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap." JASA 89(428):1303–1313. — Stationary bootstrap foundations

5. Politis, D. N. & White, H. (2004). "Automatic Block-Length Selection for the Dependent Bootstrap." Econometric Reviews 23(1):53–70. — Automatic block length selection algorithm

6. Patton, A., Politis, D. N., & White, H. (2009). "Correction to 'Automatic Block-Length Selection for the Dependent Bootstrap'." Econometric Reviews 28(4):372–375. — Correction to [5]

7. Bickel, P. J., Götze, F., & van Zwet, W. R. (1997). "Resampling Fewer Than n Observations: Gains, Losses, and Remedies for Losses." Statistica Sinica 7:1–31. — m-out-of-n bootstrap theory; establishes that $\sqrt{m}(\hat{\theta}^*_m - \hat{\theta}_n)$ approximates the distribution of $\sqrt{n}(\hat{\theta}_n - \theta)$

8. Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages." The American Statistician 50(4):361–365. — Quantile estimator taxonomy; we use type 2 (inverse empirical CDF with averaging) for continuous mode

9. Bishop, C. M. (2006). Pattern Recognition and Machine Learning, Ch. 3. Springer. — Bayesian linear regression

10. Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419–420. — Online variance algorithm

11. Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). "Algorithms for Computing the Sample Variance." The American Statistician 37(3):242–247. — Parallel Welford extension

**Timing attacks:**

12. Reparaz, O., Balasch, J., & Verbauwhede, I. (2016). "Dude, is my code constant time?" DATE. — DudeCT methodology

13. Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009). "Opportunities and Limits of Remote Timing Attacks." ACM TISSEC 12(3):17. — Exploitability thresholds. Key findings: LAN resolution ~100ns with thousands of measurements; internet ~15–100µs.

14. Kario, H. — tlsfuzzer timing analysis. Argues that timing differences as small as one clock cycle can be detectable over local networks. Cited by SILENT as the "strict" alternative to Crosby's thresholds.

15. Bernstein, D. J. et al. (2024). "KyberSlash." — Timing vulnerability in Kyber reference implementation due to secret-dependent division. ~20-cycle leak on ARM Cortex-A7. Used by SILENT to demonstrate threshold-dependent conclusions.

**Additional statistical references:**

16. Geraci, M. & Jones, M. C. (2015). "Improved transformation-based quantile regression." Canadian Journal of Statistics 43(1):118-132. — `Qtools` package and mid-distribution quantiles for discrete data

**Existing tools:**

17. dudect (C): https://github.com/oreparaz/dudect
18. dudect-bencher (Rust): https://github.com/rozbb/dudect-bencher
