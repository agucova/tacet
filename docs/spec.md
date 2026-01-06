# timing-oracle Specification

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

**Quantile computation**: We use R-7 linear interpolation (the default in R and NumPy).² For sorted sample $x$ of size $n$:

$$
h = (n - 1) \cdot p
$$

$$
\hat{q}_p = x_{\lfloor h \rfloor} + (h - \lfloor h \rfloor) \cdot (x_{\lfloor h \rfloor + 1} - x_{\lfloor h \rfloor})
$$

Linear interpolation provides smoother estimates than direct order statistics, which matters for small calibration sets.

² Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages." The American Statistician 50(4):361–365. R-7 is estimator type 7 in their taxonomy.

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
- **Inference set** (last 70%): Used for the actual hypothesis test

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

Quantiles with unusually high bootstrap variance (e.g., p90 on noisy data) can still dominate even after studentization. Define:

$$
K_{\text{sub}} = \left\{ k \in K : \hat{\sigma}_k^2 < c_{\text{var}} \cdot \text{median}_{j \in K}(\hat{\sigma}_j^2) \right\}
$$

where $c_{\text{var}} = 4$ (quantiles with variance > 4× median are excluded). Also exclude quantiles with near-zero variance (< $10^{-10}$) to avoid division blow-ups.

**Step 2 (optional): Power-boost filter**

For additional power, SILENT suggests keeping only quantiles likely to contribute to detection:

$$
K_{\text{sub}}^{\text{max}} = \left\{ k \in K_{\text{sub}} : A_k > \theta - c_{\text{margin}} \cdot \hat{\sigma}_k \right\}
$$

where $c_{\text{margin}} \approx 2$. This focuses on quantiles that are "close to or above threshold." We make this optional (off by default) since it can make the test less conservative.

**Note:** The spec's filtering criteria differ in specifics from SILENT's equation 6. They are empirically validated rather than inheriting SILENT's formal properties directly.

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

#### Bootstrap Mode: Paired vs Separate-Class

**Paired mode (default, matches SILENT):**

Uses the same block indices for both classes. This preserves the joint bivariate structure that SILENT's theory depends on. The FPR guarantee $P(\text{reject} \mid d \leq \theta) \leq \alpha$ holds by Theorem 2.

**Separate-class mode (optional):**

Resamples Fixed and Random independently, destroying cross-covariance. This is a **variant** that does not carry SILENT's formal guarantee.

```rust
TimingOracle::new()
    .bootstrap_mode(BootstrapMode::SeparateClass)  // NOT recommended for CI
```

Separate-class mode **tends to be conservative** under common-mode drift conditions (when $\text{Cov}(\hat{q}_F, \hat{q}_R) > 0$), but this is not guaranteed:
- If cross-covariance is positive (typical): overestimates variance → conservative
- If cross-covariance is negative (rare): underestimates variance → inflated FPR

Use separate-class mode only for exploratory analysis where formal guarantees aren't required.

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

**Trigger conditions (any of):**

1. **duplicate_fraction ≥ 0.30**: More than 30% of samples share values with other samples
2. **tick_size ≥ 5ns**: Estimated timer resolution is coarse
3. **unique_values / n < 0.02**: Severe quantization (very few distinct values)

When triggered, set `Diagnostics.discrete_mode = true` and use the procedures below.

**Mid-distribution quantiles:**

Instead of the standard quantile estimator (R-7 interpolation), use mid-distribution quantiles which handle ties correctly:

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

The prior scale determines regularization strength. We set it adaptively:

$$
\sigma_\mu = \sigma_\tau = \max(2 \cdot \text{MDE}, \text{min\_effect\_of\_concern})
$$

where MDE is the minimum detectable effect (§2.7). This provides mild shrinkage toward zero without strongly biasing the estimates.

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

Both the CI gate and Bayesian layer need to know how much natural variability to expect in $\Delta$. This is captured by the null covariance $\Sigma_0$, estimated via block bootstrap on the calibration set.

**Why bootstrap instead of analytical formulas?**

Quantile estimators have complex covariance structures that depend on the unknown underlying density. Asymptotic formulas exist but require density estimation, which introduces its own errors. Bootstrap sidesteps this by directly resampling from the data.

**Why block bootstrap instead of standard bootstrap?**

Timing measurements are autocorrelated: nearby samples are more similar than distant ones due to cache state, branch predictor warmup, and frequency scaling. Standard bootstrap assumes i.i.d. samples; violating this leads to underestimated variance.

Block bootstrap preserves local correlation structure by resampling contiguous blocks rather than individual points. Block length scales as $n^{1/3}$ (Politis-Romano), with a constant that depends on the autocorrelation structure. We use:

$$
\hat{b} = \left\lceil c \cdot n^{1/3} \right\rceil, \quad c = 1.3 \text{ (default)}
$$

For $n = 30{,}000$ calibration samples, this gives blocks of ~40 samples. The constant $c$ is an engineering default; the theoretically optimal value depends on unknown spectral properties, but values in the range 1–2 work well empirically for timing data.³

³ Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap." JASA 89(428):1303–1313. The $n^{1/3}$ scaling is well-established; the multiplicative constant requires spectral density estimation for optimality.

**Procedure:**

For each of $B = 2{,}000$ bootstrap iterations:
1. Resample blocks with replacement from Fixed class → compute quantile vector $\hat{q}_F^*$
2. Resample blocks with replacement from Random class → compute quantile vector $\hat{q}_R^*$
3. Record $\Delta^* = \hat{q}_F^* - \hat{q}_R^*$

Compute sample covariance of the $\Delta^*$ vectors directly using Welford's online algorithm⁵ (numerically stable for large $B$).

⁵ Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419–420. See also Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). "Algorithms for Computing the Sample Variance." The American Statistician 37(3):242–247 for the parallel/incremental generalization.

**Critical: Rescaling for inference set (30/70 split)**

The bootstrap covariance $\hat{\Sigma}_{\text{cal}}$ is estimated at calibration sample size $n_{\text{cal}}$. Under standard asymptotics for sample quantiles, covariance scales as $1/n$. Since the Bayesian layer operates on the inference set with $n_{\text{inf}} \neq n_{\text{cal}}$, we must rescale:

$$
\Sigma_0 = \hat{\Sigma}_{\text{cal}} \cdot \frac{n_{\text{cal}}}{n_{\text{inf}}}
$$

With the default 30/70 split: $n_{\text{cal}} / n_{\text{inf}} = 0.3 / 0.7 \approx 0.43$. This **shrinks** the covariance for the larger inference set, which is correct—more samples means lower variance.

**Without this rescaling**, the Bayesian posterior would be too wide (overstating uncertainty), leading to:
- Underconfident leak probabilities
- Inflated MDE estimates
- Conservative but miscalibrated results

**Note on cross-covariance and bootstrap mode:**

An alternative approach estimates $\Sigma_F$ and $\Sigma_R$ separately, then computes $\Sigma_0 = \Sigma_F + \Sigma_R$. This ignores the cross-covariance term:

$$
\text{Var}(\Delta) = \Sigma_F + \Sigma_R - 2\,\text{Cov}(\hat{q}_F, \hat{q}_R)
$$

With interleaved measurements, common-mode noise (frequency drift, thermal effects) typically makes $\text{Cov}(\hat{q}_F, \hat{q}_R) > 0$, so ignoring cross-covariance **overestimates** variance—a conservative error for security applications.

Whether bootstrapping $\Delta^*$ captures cross-covariance depends on the resampling scheme:

- **Paired resampling** (default `BootstrapMode::Paired`): Uses same indices for both classes, preserving cross-covariance. Matches SILENT's theoretical requirements for the FPR guarantee.
- **Separate-class resampling** (`BootstrapMode::SeparateClass`): Resamples Fixed and Random independently, destroying cross-covariance. Tends to be conservative when cross-covariance is positive, but does not carry SILENT's formal guarantee.

For consistency, `bootstrap_mode` affects both the CI gate and covariance estimation.

**Numerical stability:**

The covariance matrix must be positive definite for Cholesky decomposition. Near-singular matrices can arise from limited bootstrap iterations or degenerate timing distributions. Stability is defined by Cholesky success; if ill-conditioned:

1. Add jitter to the diagonal:

$$
\varepsilon = 10^{-10} + \frac{\text{tr}(\Sigma)}{9} \cdot 10^{-8}
$$

The trace-scaled term adapts to the matrix's magnitude.

2. If Cholesky still fails after jitter:
   - Set `MeasurementQuality::TooNoisy` with issue explaining the failure
   - Set `leak_probability = 0.5` (maximally uncertain)
   - The CI gate still runs (it doesn't require $\Sigma_0$), so you get at least that result

**Implementation note:** Diagonal jitter is a simple fix. More principled approaches like Ledoit-Wolf shrinkage could improve stability and estimation accuracy, particularly with limited bootstrap iterations. This is an implementation choice; the specification is agnostic to the regularization method as long as the result is positive definite.

**Edge case handling:**

If fewer than 2 bootstrap iterations complete (should never happen in practice), do not return an identity matrix—this would imply "1 ns² variance" which is arbitrary. Instead, return an error or a conservatively large diagonal (e.g., $10^6 \cdot I$) and flag the result as unreliable.

### 2.7 Minimum Detectable Effect

The MDE answers: "given the noise level in this measurement, what's the smallest effect I could reliably detect?"

This is important for interpreting negative results. If the MDE is 50ns and you're concerned about 10ns effects, a passing test doesn't mean your code is safe—it means your measurement wasn't sensitive enough. You'd need more samples or a quieter environment.

**Derivation:**

Under our linear model $\Delta = X\beta + \varepsilon$ with $\text{Var}(\varepsilon) = \Sigma_0$, the GLS estimator is:

$$
\hat{\beta} = (X^\top \Sigma_0^{-1} X)^{-1} X^\top \Sigma_0^{-1} \Delta
$$

with variance $\text{Var}(\hat{\beta}) = (X^\top \Sigma_0^{-1} X)^{-1}$.

For a single-effect model (estimating $\mu$ assuming $\tau = 0$, or vice versa), the variance of the projected estimator is:

$$
\text{Var}(\hat{\mu}) = \left( \mathbf{1}^\top \Sigma_0^{-1} \mathbf{1} \right)^{-1}, \quad
\text{Var}(\hat{\tau}) = \left( \mathbf{b}_{\text{tail}}^\top \Sigma_0^{-1} \mathbf{b}_{\text{tail}} \right)^{-1}
$$

The MDE is the effect size detectable at significance level $\alpha$ with 50% power:

$$
\text{MDE}_\mu = z_{1-\alpha/2} \cdot \sqrt{\left( \mathbf{1}^\top \Sigma_0^{-1} \mathbf{1} \right)^{-1}}
$$

$$
\text{MDE}_\tau = z_{1-\alpha/2} \cdot \sqrt{\left( \mathbf{b}_{\text{tail}}^\top \Sigma_0^{-1} \mathbf{b}_{\text{tail}} \right)^{-1}}
$$

where $z_{0.975} \approx 1.96$ for $\alpha = 0.05$.

**Intuition:**

The term $\mathbf{1}^\top \Sigma_0^{-1} \mathbf{1}$ is the *precision* of the uniform-shift estimator. Larger precision (smaller variance) means smaller MDE. In the simple case of i.i.d. quantiles with $\Sigma_0 = \sigma^2 I$:

$$
\text{MDE}_\mu = z_{1-\alpha/2} \cdot \frac{\sigma}{\sqrt{9}} = z_{1-\alpha/2} \cdot \frac{\sigma}{3}
$$

Averaging 9 quantiles reduces the standard error by $\sqrt{9} = 3$, as expected.

**Interpretation:**

- MDE decreases with $\sqrt{n}$: 4× more samples → 2× better sensitivity
- MDE increases with timer noise: coarse timers mean larger MDE
- If MDE > min_effect_of_concern, consider more samples before trusting a "pass"

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

The block length for bootstrap is then $\ell = \max(\hat{m}, c \cdot n^{1/3})$ where $c \approx 1.3$.

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

Rather than measuring all Fixed samples then all Random samples, we interleave them in randomized order:

1. Generate a schedule: [Fixed, Random, Random, Fixed, ...] with equal counts, shuffled
2. Execute according to schedule, recording timestamps
3. Separate results by class for analysis

This prevents systematic drift (thermal throttling, frequency scaling) from biasing one class.

**Outlier handling (winsorization, not dropping):**

Extreme outliers (context switches, interrupts) can skew quantile estimates. However, **dropping outliers can delete real signal**—a tail-heavy leak might have its informative samples removed, causing false negatives.

We use **winsorization** (capping) instead of dropping:

1. Compute the 99.99th percentile from **pooled data** (Fixed ∪ Random combined) as $t_{\text{cap}}$
2. Cap (not drop) samples exceeding this threshold:
   - For each sample $t > t_{\text{cap}}$: set $t = t_{\text{cap}}$
3. Record the fraction capped for reporting

**Why pooled threshold, not null-derived:**

Using the Fixed-vs-Fixed harness threshold is **dangerous** when Fixed corresponds to a fast code path:
- If Fixed is fast (~100 cycles) and Random includes slow paths (~1000 cycles from cache misses)
- The null harness cap might be ~150 cycles
- All slow-path samples in Random get capped to 150
- Result: A real 900-cycle leak is reduced to ~50 cycles, potentially passing the test

The pooled threshold captures the full range of timing behavior from both classes, avoiding this failure mode.

**Why winsorize instead of drop:**
- Capping preserves sample count (important for bootstrap)
- Capped extreme values still contribute to upper quantiles (shifting them toward the cap)
- Dropping would silently remove potentially informative data

**Critical timing:** Winsorization must happen:
- **Before** computing observed quantiles $\hat{q}_F(k)$, $\hat{q}_R(k)$
- **Before** all bootstrap resampling (bootstrap operates on the winsorized dataset)

This ensures consistency: observed statistics and bootstrap statistics are computed on identically processed data.

**Quality signals:**
- If > 0.1% of samples are capped: `QualityIssue` warning (environmental noise)
- If > 1% capped: `MeasurementQuality::Acceptable` downgrade
- If > 5% capped: `MeasurementQuality::TooNoisy` (something is wrong)

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

### 3.7 Implementation Considerations

Several implementation details have statistical implications. Getting these wrong can introduce bias, inflate variance, or cause numerical instability.

#### Optimizer Barriers

Modern compilers aggressively optimize code, which can invalidate timing measurements in subtle ways:

- **Dead code elimination**: If the result of the operation isn't used, the compiler may remove it entirely
- **Code motion**: The compiler may move the operation outside the timed region
- **Constant folding**: With fixed inputs, the compiler may precompute results at compile time

The solution is `std::hint::black_box()`, which prevents the compiler from reasoning about a value's contents while having no runtime cost. Wrap both inputs and outputs:

```rust
let input = black_box(inputs.fixed());
let result = black_box(operation(&input));
```

For batched measurements, also use `compiler_fence(SeqCst)` between iterations to prevent the compiler from reordering or merging loop iterations. Without this, the compiler might transform a loop of K independent operations into something with different cache/predictor behavior.

**Statistical implication**: Without proper barriers, you may measure optimized-away code (artificially fast) or measure something other than what you intended. Both lead to invalid conclusions.

#### Quantile Computation

Quantiles are computed by sorting samples and interpolating. Two choices matter:

**Sorting algorithm**: Use unstable sort (`sort_unstable_by`) rather than stable sort. For timing data with many equal values (discrete ticks), stable sort's $O(n)$ extra memory and overhead is wasted. More importantly, benchmarking shows full sorting outperforms selection algorithms (like `select_nth_unstable`) for 9 quantiles due to cache behavior—the sorted array is reused 9 times.

**Comparison function**: Use `f64::total_cmp` rather than `partial_cmp`. While timing data shouldn't contain NaNs, using `partial_cmp(...).unwrap_or(Equal)` can silently corrupt sort ordering if NaNs appear (e.g., from division by zero in preprocessing). `total_cmp` provides a total ordering that handles NaNs consistently.

**Interpolation method**: Use R-7 linear interpolation (the default in R and NumPy). For sorted sample $x$ of size $n$ at probability $p$:

$$
h = (n-1) \cdot p, \quad \hat{q}_p = x_{\lfloor h \rfloor} + (h - \lfloor h \rfloor)(x_{\lfloor h \rfloor + 1} - x_{\lfloor h \rfloor})
$$

R-7 produces smoother quantile estimates than direct order statistics, which matters for the calibration set (smaller $n$). For $n > 1000$, the difference is negligible, but consistency across sample sizes avoids subtle biases.

**Statistical implication**: The choice of interpolation method affects quantile estimates at small sample sizes. Using a consistent, well-understood method (R-7) ensures our covariance estimates and thresholds are calibrated correctly.

#### Numerical Stability in Covariance Estimation

The covariance matrix $\Sigma_0$ is estimated from 2,000 bootstrap samples, each a 9-dimensional quantile vector. Naive computation (accumulate sum, then divide) suffers from catastrophic cancellation when the mean is large relative to the variance.

Use Welford's online algorithm, which maintains a running mean and sum of squared deviations:

```
for each sample x:
    n += 1
    delta = x - mean
    mean += delta / n
    M2 += delta * (x - mean)  # Note: uses updated mean
covariance = M2 / (n - 1)
```

This is numerically stable regardless of the mean's magnitude. For covariance matrices, extend to track cross-products.

**Statistical implication**: Numerical instability in covariance estimation propagates to the posterior calculation. A poorly estimated $\Sigma_0$ can cause the Cholesky decomposition to fail or produce incorrect posteriors. Welford's algorithm avoids this failure mode.

#### Batching and Microarchitectural Artifacts

When batching K operations together (§3.4), microarchitectural state accumulates across iterations. This can cause timing differences between classes even for constant-time code:

- **Branch predictor training**: K iterations with identical input trains predictors differently than K varied inputs
- **Cache state**: Same cache lines accessed K times vs K different access patterns
- **μop cache**: Same instruction sequence cached and optimized vs varied sequences

These are measurement artifacts, not timing leaks. We limit $K \leq 20$ to bound these effects—empirically, K=20 keeps artifacts below the noise floor for typical crypto operations.

Additional mitigations:

- Keep K identical across both classes (never batch Fixed at K=10 and Random at K=15)
- Pre-generate all random inputs before the timed region (generator cost doesn't confound measurement)
- Perform inference on batch totals, not per-call divided values—division reintroduces quantization noise

**Statistical implication**: Without these mitigations, batching can produce false positives (detecting "leaks" that are actually predictor/cache artifacts) or false negatives (artifacts masking real leaks). The K=20 limit and mitigations keep the false positive rate within the advertised $\alpha$.

---

## 4. Public API

This section describes the user-facing types and functions: result structures, configuration options, and three API tiers (macro, builder, raw) for different use cases.

### 4.1 Result Types

#### TestResult

The primary result type, returned directly by `timing_test!`:

```rust
pub struct TestResult {
    /// P(effect exceeds min_effect_of_concern), from 0.0 to 1.0
    pub leak_probability: f64,
    
    /// Effect size estimate with shift/tail decomposition
    pub effect: Effect,
    
    /// Exploitability assessment
    pub exploitability: Exploitability,
    
    /// Minimum detectable effect given noise level
    pub min_detectable_effect: MinDetectableEffect,
    
    /// CI gate result
    pub ci_gate: CiGate,
    
    /// Measurement quality assessment
    pub quality: MeasurementQuality,
    
    /// Detailed diagnostics (dependence, stationarity, filtered quantiles)
    pub diagnostics: Diagnostics,
    
    /// Fraction of samples capped (winsorized) as outliers
    pub winsorized_fraction: f64,
    
    /// The attacker model / threshold used
    pub attacker_model: AttackerModel,
}
```

#### Outcome

Returned by `timing_test_checked!` and the builder API for explicit unmeasurable handling:

```rust
pub enum Outcome {
    /// Analysis completed successfully
    Completed(TestResult),
    
    /// Operation too fast to measure on this platform
    Unmeasurable {
        operation_ns: f64,
        threshold_ns: f64,
        platform: String,
        recommendation: String,
    },
}
```

Most users should use `timing_test!` which panics on unmeasurable operations. Use `timing_test_checked!` or the builder API when you need to handle unmeasurable cases gracefully (e.g., conditional test skipping).

#### Effect

When a leak is detected, the effect is decomposed:

```rust
pub struct Effect {
    /// Uniform shift in nanoseconds (positive = fixed class slower)
    pub shift_ns: f64,
    
    /// Tail effect in nanoseconds (positive = fixed has heavier upper tail)
    pub tail_ns: f64,
    
    /// 95% credible interval for total effect magnitude
    pub credible_interval_ns: (f64, f64),
    
    /// Dominant pattern
    pub pattern: EffectPattern,
    
    /// Marginal probability: P(|μ| > θ | Δ)
    /// Diagnostic only—not used for headline leak_probability
    pub prob_shift_exceeds_threshold: f64,
    
    /// Marginal probability: P(|τ| > θ | Δ)  
    /// Diagnostic only—not used for headline leak_probability
    pub prob_tail_exceeds_threshold: f64,
}

pub enum EffectPattern {
    UniformShift,  // Branch, different code path
    TailEffect,    // Cache misses, memory access patterns
    Mixed,         // Both components significant
}
```

#### CiGate

The pass/fail decision with controlled FPR:

```rust
pub struct CiGate {
    /// True if Q_hat_max ≤ critical_value (no significant leak above threshold)
    pub passed: bool,
    
    /// Target FPR (default 0.01)
    pub alpha: f64,
    
    /// Practical significance threshold θ (min_effect_of_concern)
    pub threshold_theta_ns: f64,
    
    /// Observed test statistic
    /// - Continuous mode: max_k((A_k - θ) / σ_k) — studentized
    /// - Discrete mode: √n × max_k(A_k - θ) — not studentized
    pub q_hat_max: f64,
    
    /// Bootstrap critical value c*_{1-α}
    pub critical_value: f64,
    
    /// Raw max distance d = max_k |q_F(k) - q_R(k)| in nanoseconds
    pub max_distance_ns: f64,
    
    /// Margin: critical_value - q_hat_max (positive = passed with room to spare)
    pub margin: f64,
    
    /// Whether discrete timer mode was used (m-out-of-n bootstrap)
    pub discrete_mode: bool,
    
    /// Quantiles that were filtered out (transparency)
    pub filtered_quantile_count: usize,
}
```

The gate passes when `q_hat ≤ critical_value`. The `margin` field tells you how close you are to the boundary: positive means passed with room to spare, negative means failed. The `max_distance_ns` field gives the raw observed distance in nanoseconds for interpretability.

#### Exploitability

Heuristic assessment based on Crosby et al. (2009) and local attack research:

```rust
pub enum Exploitability {
    /// < 1 cycle: Essentially undetectable even locally
    Undetectable,
    
    /// 1–100 cycles: Exploitable by local attacker with direct access
    /// (SGX, shared hosting, local privilege escalation)
    ExploitableLocal,
    
    /// 100 ns – 500 ns: Exploitable on LAN with ~1k–10k queries
    ExploitableLAN,
    
    /// 500 ns – 20 μs: Readily exploitable on LAN, possibly remote
    LikelyExploitable,
    
    /// > 20 μs: Exploitable over internet
    ExploitableRemote,
}
```

**Caveat: 2009-era baseline**

The Crosby thresholds are from 2009. Modern stacks (better clocks, improved filtering, protocol-specific optimizations) can shift these boundaries in either direction. Treat this enum as a rough heuristic for prioritization, not a security guarantee.

**Connection to AttackerModel:**

The `Exploitability` assessment tells you who *could* exploit a detected leak. The `AttackerModel` configuration tells the library who you're *trying to defend against*. They work together:

| AttackerModel | θ (threshold) | CI gate fails if | Exploitability to watch |
|---------------|---------------|------------------|------------------------|
| `LocalCycles` | 2 cycles | effect > 2 cycles | ExploitableLocal+ |
| `LocalCoarseTimer` | 1 tick | effect > 1 tick | ExploitableLocal+ |
| `LANStrict` | 2 cycles | effect > 2 cycles | ExploitableLocal+ |
| `LANConservative` | 100 ns | effect > 100 ns | ExploitableLAN+ |
| `WANOptimistic` | 15 μs | effect > 15 μs | LikelyExploitable+ |
| `WANConservative` | 50 μs | effect > 50 μs | ExploitableRemote |
| `KyberSlashSentinel` | 10 cycles | effect > 10 cycles | ExploitableLocal+ |

A common pattern: configure for your threat model, but also check `exploitability` to understand worst-case risk. For example, with `LANConservative`, you might pass CI but still see `ExploitableLocal`—meaning a co-resident attacker could still exploit it.

#### MeasurementQuality

Assessment of result reliability:

```rust
pub enum MeasurementQuality {
    /// Confident in results
    Good,
    /// Results valid but noisier than ideal
    Acceptable { issues: Vec<QualityIssue> },
    /// Results may be unreliable
    TooNoisy { issues: Vec<QualityIssue> },
}

pub struct QualityIssue {
    pub code: IssueCode,
    pub message: String,
    pub guidance: String,  // Actionable next step
}

pub enum IssueCode {
    HighDependence,       // Autocorrelation reducing effective samples
    LowEffectiveSamples,  // ESS << nominal samples
    StationaritySuspect,  // Variance drift detected
    DiscreteTimer,        // Low resolution, using m-out-of-n mode
    SmallSampleDiscrete,  // n < 2000 in discrete mode, weak asymptotics
    HighGeneratorCost,    // Generator overhead > 10% of measurement
    LowUniqueInputs,      // Input generation may not be random
    QuantilesFiltered,    // Some quantiles excluded due to noise
    ThresholdClamped,     // θ collapsed to < 1 tick, clamped up
    HighWinsorRate,       // > 1% of samples capped (environmental noise)
}
```

**Example issue with guidance:**

```rust
QualityIssue {
    code: IssueCode::HighDependence,
    message: "Estimated dependence length: 47 samples (ESS: 2,100 of 100,000)",
    guidance: "Try: pin process to isolated core, disable turbo boost, increase warmup, or separate input generation to another thread",
}
```

#### Diagnostics

Detailed measurement diagnostics for debugging and transparency:

```rust
pub struct Diagnostics {
    /// Estimated dependence length (block size used for bootstrap)
    pub dependence_length: usize,
    
    /// Effective sample size (accounts for autocorrelation)
    /// ESS ≈ n / dependence_length
    pub effective_sample_size: usize,
    
    /// True if rolling variance test detected drift
    pub stationarity_suspect: bool,
    
    /// Quantiles excluded from CI gate (and why)
    pub filtered_quantiles: Vec<FilteredQuantile>,
    
    /// Whether discrete timer mode was triggered
    pub discrete_mode: bool,
    
    /// Timer resolution estimate (ns per tick)
    pub timer_resolution_ns: f64,
    
    /// Fraction of duplicate timing values (high = discrete timer)
    pub duplicate_fraction: f64,
}

pub struct FilteredQuantile {
    pub quantile: f64,           // e.g., 0.9
    pub reason: FilterReason,
    pub bootstrap_std: f64,      // σ_k estimate
}

pub enum FilterReason {
    /// Bootstrap variance too high relative to others
    HighVariance { relative_to_median: f64 },
    /// Bootstrap variance near zero (would cause numerical issues)
    NearZeroVariance,
}
```

These diagnostics are always computed and available in `TestResult`. They explain *why* results might be unreliable, not just *that* they are.

### 4.2 Configuration

#### Attacker Model Presets

The most important configuration choice is your **attacker model**, which determines what size leak you consider "negligible." SILENT's key insight [2] is that the right question isn't "is there any timing difference?" but "is the difference > θ under my threat model?"

**There is no single correct θ.** SILENT explicitly demonstrates this using KyberSlash: under Crosby-style thresholds (~100ns), the ~20-cycle leak isn't flagged as practically significant; under Kario-style thresholds (~1 cycle), it is. Your choice of preset is a statement about your threat model.

```rust
pub enum AttackerModel {
    // ═══════════════════════════════════════════════════════════════════
    // LOCAL ATTACKER PRESETS
    // ═══════════════════════════════════════════════════════════════════
    
    /// Local attacker with cycle-level timing (rdtsc, perf, kperf)
    /// θ = 2 cycles
    /// Use for: SGX enclaves, shared hosting, local privilege escalation
    /// Source: Kario argues even 1 cycle is detectable over LAN—so for
    /// same-host attackers, 1–2 cycles is the conservative stance.
    LocalCycles,
    
    /// Local attacker but only coarse timers available
    /// θ = 1 tick (whatever the timer resolution is)
    /// Use for: Sandboxed environments, tick-only measurement, noisy scheduling
    /// Rationale: This isn't "attacker is weaker"—it's "measurement primitive
    /// is weaker." Picks the smallest meaningful θ for your timer.
    LocalCoarseTimer,
    
    // ═══════════════════════════════════════════════════════════════════
    // LAN ATTACKER PRESETS
    // ═══════════════════════════════════════════════════════════════════
    
    /// Strict LAN attacker ("Kario-style")
    /// θ = 2 cycles
    /// Use for: High-security internal services where LAN attackers are capable
    /// Source: Kario argues even ~1 clock cycle can be detectable over LAN.
    /// SILENT explicitly contrasts this with Crosby's more relaxed view.
    /// Warning: May produce "TooNoisy" on coarse timers.
    LANStrict,
    
    /// Conservative LAN attacker ("Crosby-style")
    /// θ = 100 ns
    /// Use for: Internal services, database servers, microservices
    /// Source: Crosby et al. (2009) report attackers can measure with
    /// "accuracy as good as 100ns over a local network."
    LANConservative,
    
    // ═══════════════════════════════════════════════════════════════════
    // WAN ATTACKER PRESETS
    // ═══════════════════════════════════════════════════════════════════
    
    /// Optimistic WAN attacker (low-jitter environments)
    /// θ = 15 μs
    /// Use for: Same-region cloud, datacenter-to-datacenter, low-jitter paths
    /// Source: Best-case end of Crosby's 15–100μs internet range.
    /// Note: Less anchored than Conservative; use when you have reason to
    /// believe network conditions are favorable.
    WANOptimistic,
    
    /// Conservative WAN attacker (general internet)
    /// θ = 50 μs
    /// Use for: Public APIs, web services, general internet exposure
    /// Source: Crosby et al. (2009) report 15–100μs accuracy across internet;
    /// 50μs is a reasonable midpoint.
    WANConservative,
    
    // ═══════════════════════════════════════════════════════════════════
    // SPECIAL-PURPOSE PRESETS
    // ═══════════════════════════════════════════════════════════════════
    
    /// Calibrated to catch KyberSlash-class vulnerabilities
    /// θ = 10 cycles
    /// Use for: Post-quantum crypto, division-based leaks, "don't ignore
    /// 20-cycle class vulnerabilities"
    /// Source: SILENT characterizes KyberSlash as ~20 cycles on Raspberry Pi
    /// 2B (900MHz Cortex-A7). θ=10 ensures such leaks are non-negligible.
    KyberSlashSentinel,
    
    /// Research mode: detect any statistical difference (θ → 0)
    /// Warning: Will flag tiny, unexploitable differences. Not for CI.
    /// Use for: Profiling, debugging, academic analysis, finding any leak
    Research,
    
    // ═══════════════════════════════════════════════════════════════════
    // CUSTOM THRESHOLDS
    // ═══════════════════════════════════════════════════════════════════
    
    /// Custom threshold in nanoseconds
    CustomNs { threshold_ns: f64 },
    
    /// Custom threshold in cycles (more portable across CPUs)
    CustomCycles { threshold_cycles: u32 },
    
    /// Custom threshold in timer ticks (for tick-based timers)
    CustomTicks { threshold_ticks: u32 },
}
```

**Preset summary:**

| Preset | θ | Native unit | Threat model |
|--------|---|-------------|--------------|
| `LocalCycles` | 2 cycles | cycles | Co-resident attacker, strong timer |
| `LocalCoarseTimer` | 1 tick | ticks | Co-resident attacker, weak timer |
| `LANStrict` | 2 cycles | cycles | LAN attacker, Kario-style (capable) |
| `LANConservative` | 100 ns | ns | LAN attacker, Crosby-style |
| `WANOptimistic` | 15 μs | ns | Internet, low-jitter path |
| `WANConservative` | 50 μs | ns | Internet, general |
| `KyberSlashSentinel` | 10 cycles | cycles | Catch ~20-cycle leaks |
| `Research` | 0 | — | Detect any difference |

**The LAN dispute:** SILENT goes out of its way to show that "what counts as negligible on LAN?" is genuinely disputed. `LANStrict` vs `LANConservative` represents this fork explicitly—pick based on your threat model, not based on what's convenient.

**Sources:**

- **Crosby et al. (2009)**: "Opportunities and Limits of Remote Timing Attacks." ACM TISSEC. Reports ~100ns LAN accuracy, 15–100μs internet accuracy.
- **Kario**: Argues timing differences as small as one clock cycle can be detectable over local networks. SILENT cites this as the "strict" alternative to Crosby.
- **SILENT [2]**: Uses KyberSlash (~20 cycles on Raspberry Pi 2B @ 900MHz Cortex-A7) to demonstrate how conclusions flip based on θ choice.

**Recommended usage:**

```rust
// Public web API (internet attackers)
let result = timing_test! {
    oracle: TimingOracle::for_attacker(AttackerModel::WANConservative),
    fixed: || verify_signature(&secret_key, &msg),
    random: || verify_signature(&random_key, &msg),
};

// Internal microservice (conservative LAN model)
let result = timing_test! {
    oracle: TimingOracle::for_attacker(AttackerModel::LANConservative),
    // ...
};

// High-security internal service (strict LAN model)
let result = timing_test! {
    oracle: TimingOracle::for_attacker(AttackerModel::LANStrict),
    // ...
};

// SGX enclave or shared hosting
let result = timing_test! {
    oracle: TimingOracle::for_attacker(AttackerModel::LocalCycles),
    // ...
};

// Post-quantum crypto (catch KyberSlash-class leaks)
let result = timing_test! {
    oracle: TimingOracle::for_attacker(AttackerModel::KyberSlashSentinel),
    // ...
};
```

**Output always shows θ in context:**

```
CI gate: FAIL (α=1%) for leaks > θ=100ns [LANConservative]
Observed distance: d=247ns (≈741 cycles @ 3.0GHz)
Margin: d−θ = 147ns
```

#### Native Units for θ

**Critical:** θ should be applied in the timer's native units for robustness:

| Timer type | θ applied as | Rationale |
|------------|--------------|-----------|
| Cycle counter (rdtsc, perf, kperf) | cycles | "10 cycles" is portable across CPUs |
| Tick timer (cntvct_el0) | ticks | Avoids sub-tick thresholds |
| Nanosecond timer | ns | Direct comparison |

When a cycle-based preset (like `LocalCycles` or `LANStrict`) would collapse to < 1 tick on a coarse timer, the library warns and suggests alternatives:

```
Warning: LANStrict requires θ=2 cycles, but timer resolution is 41ns (~123 cycles).
Effective θ clamped to 1 tick. Options:
  - Use a cycle counter (perf/kperf) for this attacker model
  - Switch to LANConservative (θ=100ns) which is measurable on this timer
  - Accept reduced sensitivity with LocalCoarseTimer (θ=1 tick)
```

This prevents strict presets from being silently meaningless on coarse timers.

#### TimingOracle

The main configuration type with builder pattern:

```rust
impl TimingOracle {
    // Attacker-model presets (recommended)
    pub fn for_attacker(model: AttackerModel) -> Self;
    
    // Sample-count presets
    pub fn new() -> Self;       // 100k samples, thorough
    pub fn balanced() -> Self;  // 20k samples, good for CI
    pub fn quick() -> Self;     // 5k samples, development
    
    // Fine-grained configuration
    pub fn samples(self, n: usize) -> Self;
    pub fn warmup(self, n: usize) -> Self;
    pub fn ci_alpha(self, alpha: f64) -> Self;
    pub fn min_effect_of_concern_ns(self, ns: f64) -> Self;
    pub fn min_effect_of_concern_cycles(self, cycles: u32) -> Self;
    pub fn attacker_model(self, model: AttackerModel) -> Self;
    pub fn bootstrap_mode(self, mode: BootstrapMode) -> Self;
    pub fn bootstrap_iterations(self, b: usize) -> Self;
}

pub enum BootstrapMode {
    /// Resample Fixed and Random separately (default)
    /// Conservative: destroys positive cross-covariance, tighter FPR, lower power
    SeparateClass,
    
    /// Resample paired blocks from interleaved sequence
    /// Higher power when common-mode drift dominates
    /// Use for local investigation, not CI
    Paired,
}
```

**Combining presets:**

```rust
// LANConservative threat model, but fewer samples for faster CI
TimingOracle::for_attacker(AttackerModel::LANConservative)
    .samples(20_000)

// Maximum sensitivity for local investigation
TimingOracle::new()
    .bootstrap_mode(BootstrapMode::Paired)
    .bootstrap_iterations(10_000)
```

Output always shows both cycles and nanoseconds when a cycle counter is available:

```
Threshold: θ = 100ns (≈300 cycles @ 3.0GHz)
Threshold: θ = 10 cycles ≈ 3.3ns @ 3.0GHz
```

#### Key Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| samples | 100,000 | Samples per class |
| warmup | 1,000 | Warmup iterations |
| ci_alpha | 0.01 | CI gate false positive rate (α) |
| min_effect_of_concern | 10.0 ns | Threshold θ for practical significance; effects below this are considered negligible |
| attacker_model | LANConservative | Determines default θ; can be overridden by explicit min_effect_of_concern |
| bootstrap_mode | SeparateClass | How to resample; `Paired` for maximum power in local investigation |
| bootstrap_iterations | 2,000 | B for CI gate; use 10,000 for thorough mode |

### 4.3 Macro API

The `timing_test!` macro is the recommended API for most users. It returns `TestResult` directly and panics if the operation is unmeasurable.

#### Syntax

```rust
let result = timing_test! {
    // Optional: custom configuration
    oracle: TimingOracle::balanced(),
    
    // Optional: setup block
    setup: {
        let key = Aes128::new(&KEY);
    },
    
    // Required: fixed input (evaluated once)
    fixed: [0u8; 32],
    
    // Required: random generator (closure, called per sample)
    random: || rand::random::<[u8; 32]>(),
    
    // Required: test body (duplicated into both closures)
    test: |input| {
        key.encrypt(&input);
    },
};

// result is TestResult, not Outcome
assert!(result.ci_gate.passed);
```

**The `random` field requires an explicit closure** (`|| expr`). This is intentional: the syntax mirrors the semantics. A bare expression looks like it would be evaluated once, which is exactly the mistake we're preventing.

#### Checked Variant

For explicit unmeasurable handling, use `timing_test_checked!`:

```rust
let outcome = timing_test_checked! {
    fixed: [0u8; 32],
    random: || rand::random::<[u8; 32]>(),
    test: |input| operation(&input),
};

match outcome {
    Outcome::Completed(result) => assert!(result.ci_gate.passed),
    Outcome::Unmeasurable { recommendation, .. } => {
        eprintln!("Skipping: {}", recommendation);
    }
}
```

Use `timing_test_checked!` when:
- Running on platforms with coarse timers where unmeasurable is expected
- You want to skip rather than fail when measurement isn't possible
- Writing platform-adaptive test suites

#### Multiple Inputs

Use tuple destructuring:

```rust
timing_test! {
    fixed: ([0u8; 12], [0u8; 64]),
    random: || (rand::random(), rand::random()),
    test: |(nonce, plaintext)| {
        cipher.encrypt(&nonce, &plaintext);
    },
}
```

#### Compile-Time Errors

The macro catches common mistakes with clear messages:

| Mistake | Error |
|---------|-------|
| Missing `random:` field | "missing `random` field" with syntax suggestion |
| Typo in field name | "unknown field `rando`, expected one of: ..." |
| Missing closure on random | "`random` must be a closure" with fix suggestion |
| Type mismatch fixed/random | Standard Rust type error pointing at user's code |

### 4.4 Builder API

For users who prefer explicit code over macros:

```rust
let result = TimingTest::new()
    .oracle(TimingOracle::balanced())
    .fixed([0u8; 32])
    .random(|| rand::random::<[u8; 32]>())
    .test(|input| secret.ct_eq(&input))
    .run();
```

#### Methods

```rust
impl TimingTest {
    pub fn new() -> Self;
    pub fn oracle(self, oracle: TimingOracle) -> Self;  // Optional
    pub fn fixed<V: Clone>(self, value: V) -> Self;     // Required
    pub fn random<F: FnMut() -> V>(self, gen: F) -> Self;  // Required
    pub fn test<E: FnMut(V)>(self, body: E) -> Self;    // Required
    pub fn run(self) -> Outcome;
}
```

#### Error Handling

| Condition | Behavior |
|-----------|----------|
| Missing required field | Panic with clear message |
| Type mismatch | Compile error |
| Unmeasurable operation | Returns `Outcome::Unmeasurable` |

### 4.5 Raw API

For complex cases requiring full control:

```rust
use timing_oracle::{test, helpers::InputPair};

let inputs = InputPair::new([0u8; 32], || rand::random());

let result = test(
    || encrypt(&inputs.fixed()),
    || encrypt(&inputs.random()),
);
```

#### InputPair

Separates input generation from measurement:

```rust
impl<T: Clone, F: FnMut() -> T> InputPair<T, F> {
    pub fn new(fixed: T, generator: F) -> Self;
    pub fn fixed(&self) -> T;   // Clones fixed value
    pub fn random(&self) -> T;  // Calls generator
}
```

Both methods return owned `T` for symmetric usage.

**Anomaly detection**: If the type is hashable, tracks value uniqueness and warns about suspicious patterns. For non-hashable types, tracking is skipped automatically with zero overhead.

#### When to Use Each API

| API | Returns | Best for |
|-----|---------|----------|
| `timing_test!` | `TestResult` | Most users; panics if unmeasurable |
| `timing_test_checked!` | `Outcome` | Platform-adaptive tests; explicit unmeasurable handling |
| `TimingTest` builder | `Outcome` | Macro-averse; programmatic result access |
| `InputPair` + `test()` | `Outcome` | Multiple independent varying inputs; full control |

---

## 5. Interpreting Results

### 5.1 Leak Probability

The `leak_probability` answers: "what's the probability that this code has a timing leak exceeding `min_effect_of_concern`?"

This is computed by integrating the posterior distribution over effect sizes (§2.5). It's **calibrated**: when we report 80% probability, approximately 80% of the posterior mass lies above your concern threshold.

**Decision thresholds:**

| Probability | Interpretation | Action |
|-------------|----------------|--------|
| $< 10\%$ | Probably safe | Pass |
| $10\%$–$50\%$ | Inconclusive | Consider more samples or lower threshold |
| $50\%$–$90\%$ | Probably leaking | Investigate |
| $> 90\%$ | Almost certainly leaking | Fix required |

These are guidelines, not rules. Your risk tolerance may vary.

**What it depends on:**

- `min_effect_of_concern`: Effects below this threshold don't count as "leaks." Default 10ns.
- Measurement noise: Higher noise → wider posterior → more uncertainty.
- Sample size: More samples → narrower posterior → more decisive probabilities.

**What it is not:**

- Not a frequentist p-value
- Not the probability of being exploited (see `exploitability` for that)
- Not sensitive to "any nonzero effect"—only effects you've said you care about

### 5.2 CI Gate

The CI gate provides a binary pass/fail with controlled false positive rate, testing whether the effect exceeds `min_effect_of_concern` (θ) using the SILENT relevant-hypothesis framework.

**Semantics:**

- `passed: true` means $\hat{Q}_{max} \leq c^*_{1-\alpha}$ (the studentized excess-over-threshold doesn't exceed the bootstrap critical value)
- `passed: false` means the test statistic exceeded the critical value, indicating a likely leak above θ

**FPR guarantee:** When the true effect is ≤ θ, the gate fails at most α of the time. Crucially, this holds *at the boundary* (true effect = θ), not just when effect = 0. This is what distinguishes the SILENT approach from naive "test + filter" methods.

- Code with no timing leak: fails ≤ α ✓
- Code with a 5ns leak when θ = 10ns: fails ≤ α ✓ (below threshold)
- Code with a 10ns leak when θ = 10ns: fails ≈ α ✓ (at boundary—properly calibrated)
- Code with a 15ns leak when θ = 10ns: usually fails ✓ (above threshold, good power)

**Key fields:**

- `max_distance_ns`: The raw observed distance $d = \max_k |q_F(k) - q_R(k)|$ in nanoseconds—the most interpretable measure
- `q_hat_max`: The test statistic. In continuous mode: $\hat{Q}_{max} = \max_k (A_k - \theta) / \hat{\sigma}_k$ (studentized). In discrete mode: $\sqrt{n} \cdot \max_k(A_k - \theta)$ (not studentized)
- `margin`: $c^* - \hat{Q}_{max}$. Positive = passed with room to spare; negative = failed
- `discrete_mode`: Whether the m-out-of-n bootstrap with mid-quantiles was used (for low-resolution timers)
- `filtered_quantile_count`: How many quantiles were excluded due to high variance (transparency)

**Relationship to Bayesian layer:**

Both layers use the same threshold θ, so they typically agree:

| CI Gate | Posterior | Interpretation |
|---------|-----------|----------------|
| Pass | Low | Clean: effect is below threshold |
| Pass | High | Rare: posterior sees mass above θ that CI didn't flag (edge case) |
| Fail | High | Clear leak above threshold |
| Fail | Low | Rare: CI flagged something posterior doesn't (edge case) |

Divergence is uncommon because both use the same θ. When it happens, it's usually due to posterior uncertainty (wide credible interval straddling θ) vs the CI gate's hard cutoff.

### 5.3 Effect Size

When a leak is detected, the effect tells you *how big* and *what kind*:

**Shift ($\mu$)**: Uniform timing difference across all quantiles. Typical cause: different code path, branch on secret data.

**Tail ($\tau$)**: Upper quantiles affected more than lower. Typical cause: cache misses, memory access patterns depending on secret.

**Interpreting magnitude:**

The effect is in nanoseconds. A 50ns shift means the fixed input takes about 50ns longer on average than random inputs.

**Credible interval:**

The 95% credible interval gives uncertainty bounds. If it's [20ns, 80ns], you're 95% confident the true effect is in that range.

### 5.4 Exploitability

The `Exploitability` enum provides a rough assessment of practical risk, based on Crosby et al. (2009)⁶ which measured timing attack resolution across network conditions:

| Level | Effect Size | Meaning |
|-------|-------------|---------|
| Negligible | $< 100$ ns | At the limit of LAN exploitability; requires ~10k+ queries |
| PossibleLAN | $100$–$500$ ns | Exploitable on LAN with ~1k–10k queries |
| LikelyLAN | $500$ ns – $20$ μs | Readily exploitable on LAN with hundreds of queries |
| PossibleRemote | $> 20$ μs | Large enough to potentially exploit over internet |

⁶ Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009). "Opportunities and Limits of Remote Timing Attacks." ACM TISSEC 12(3):17. Key findings: LAN resolution "as good as 100ns" with thousands of measurements; internet resolution "15–100µs" with best hosts resolving ~30µs. Their empirical tests with 1,000 measurements showed ~200ns resolution on LAN and ~30–50µs over the internet.

**Caveats:**

- These are heuristics, not guarantees
- Actual exploitability depends on many factors (network jitter, attacker capabilities, protocol specifics)
- Modern attacks may achieve better resolution than the 2009 study
- Even "Negligible" leaks should be fixed if practical

### 5.5 Quality Assessment

The `MeasurementQuality` indicates confidence in the results, primarily based on the minimum detectable effect (MDE) relative to effects of practical concern:

| MDE | Quality | Interpretation |
|-----|---------|----------------|
| $< 5$ ns | Excellent | Can detect very small leaks |
| $5$–$20$ ns | Good | Sufficient for most applications |
| $20$–$100$ ns | Acceptable | May miss small leaks; consider more samples |
| $> 100$ ns | TooNoisy | Results unreliable; environment too noisy |

These thresholds assume a default `min_effect_of_concern` of 10ns. If you're concerned about smaller effects, you need correspondingly better quality.

**Other quality factors:**

**Good**: Normal conditions, MDE within acceptable range, results are reliable.

**Acceptable**: Minor issues detected but results are still valid:
- Moderately high outlier rate (1–5%)
- MDE somewhat elevated but still useful
- Slight autocorrelation in measurements

**TooNoisy**: Results may be unreliable:
- Very high outlier rate ($> 5\%$)
- MDE exceeds 100ns (can't detect practically relevant effects)
- Severe autocorrelation
- Timer resolution insufficient even with batching

When quality is `TooNoisy`, consider:
- Reducing system load (close browsers, stop background jobs)
- Increasing sample count ($4\times$ samples → $2\times$ better MDE)
- Running on dedicated hardware or in single-user mode
- Using a platform with better timer resolution (x86_64 with rdtsc, or ARM with perf_event)

### 5.6 Reliability Handling

Some tests may be unreliable on certain platforms due to timer limitations or environmental factors.

**Checking reliability:**

```rust
impl Outcome {
    /// True if results are trustworthy
    pub fn is_reliable(&self) -> bool;
}
```

A result is reliable if:
- Measurement completed (not `Unmeasurable`)
- Quality is not `TooNoisy`, OR posterior is conclusive ($< 0.1$ or $> 0.9$)

The rationale: conclusive posteriors overcame the noise, so the signal was strong enough.

**Handling unreliable results:**

For CI integration, use fail-open or fail-closed policies:

```rust
// Fail-open: unreliable tests are skipped (pass)
let result = skip_if_unreliable!(outcome, "test_name");

// Fail-closed: unreliable tests fail
let result = require_reliable!(outcome, "test_name");
```

Environment variable `TIMING_ORACLE_UNRELIABLE_POLICY` can override: set to `fail_closed` for stricter CI.

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
| Variance filter cutoff | $4 \times$ median | Exclude quantiles with $\hat{\sigma}_k^2 > 4 \cdot \text{median}$ |
| Min ticks per call | $5$ | Below this, quantization noise dominates |
| Target ticks per batch | $50$ | Target for adaptive batching |
| Max batch size | $20$ | Limit microarchitectural artifacts |
| Anomaly detection window | $1{,}000$ | Samples to track for uniqueness |
| Anomaly detection threshold | $0.5$ | Warn if $< 50\%$ unique values |
| Default $\alpha$ | $0.01$ | 1% false positive rate |
| Default min effect | $10$ ns | Threshold for significant leak |

## Appendix C: References

**Statistical methodology:**

1. Dunsche, M., Lamp, M., & Pöpper, C. (2024). "With Great Power Come Great Side Channels: Statistical Timing Side-Channel Analyses with Bounded Type-1 Errors." USENIX Security. — RTLF bootstrap methodology

2. Dunsche, M., Lamp, M., & Pöpper, C. (2025). "SILENT: A New Lens on Statistics in Software Timing Side Channels." arXiv:2504.19821. — Extension supporting negligible leak thresholds

3. Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." Annals of Statistics. — Block bootstrap for autocorrelated data

4. Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap." JASA 89(428):1303–1313. — Block length heuristics

5. Bickel, P. J., Götze, F., & van Zwet, W. R. (1997). "Resampling Fewer Than n Observations: Gains, Losses, and Remedies for Losses." Statistica Sinica 7:1–31. — m-out-of-n bootstrap theory; establishes that $\sqrt{m}(\hat{\theta}^*_m - \hat{\theta}_n)$ approximates the distribution of $\sqrt{n}(\hat{\theta}_n - \theta)$

6. Hyndman, R. J. & Fan, Y. (1996). "Sample quantiles in statistical packages." The American Statistician 50(4):361–365. — R-7 quantile interpolation

7. Bishop, C. M. (2006). Pattern Recognition and Machine Learning, Ch. 3. Springer. — Bayesian linear regression

8. Welford, B. P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." Technometrics 4(3):419–420. — Online variance algorithm

9. Chan, T. F., Golub, G. H., & LeVeque, R. J. (1983). "Algorithms for Computing the Sample Variance." The American Statistician 37(3):242–247. — Parallel Welford extension

**Timing attacks:**

10. Reparaz, O., Balasch, J., & Verbauwhede, I. (2016). "Dude, is my code constant time?" DATE. — DudeCT methodology

11. Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009). "Opportunities and Limits of Remote Timing Attacks." ACM TISSEC 12(3):17. — Exploitability thresholds. Key findings: LAN resolution ~100ns with thousands of measurements; internet ~15–100µs.

12. Kario, H. — tlsfuzzer timing analysis. Argues that timing differences as small as one clock cycle can be detectable over local networks. Cited by SILENT as the "strict" alternative to Crosby's thresholds.

13. Bernstein, D. J. et al. (2024). "KyberSlash." — Timing vulnerability in Kyber reference implementation due to secret-dependent division. ~20-cycle leak on ARM Cortex-A7. Used by SILENT to demonstrate threshold-dependent conclusions.

**Existing tools:**

14. dudect (C): https://github.com/oreparaz/dudect
15. dudect-bencher (Rust): https://github.com/rozbb/dudect-bencher
