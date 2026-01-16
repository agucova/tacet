# Bayesian Inference Layer

This document summarizes how the Bayesian layer (Layer 2) works in timing-oracle. While the CI gate provides a binary pass/fail decision with controlled false positive rate, the Bayesian layer provides interpretable outputs: a probability of a leak and an estimate of its magnitude.

## Purpose

The Bayesian layer answers questions the CI gate cannot:
- **"What's the probability of a timing leak?"** — A number between 0 and 1, not just pass/fail
- **"How big is the effect?"** — Estimated timing difference in nanoseconds
- **"What kind of leak is it?"** — Uniform shift vs tail effect

## Model Overview

### Effect Decomposition

Timing leaks manifest in two characteristic patterns:

1. **Uniform shift (μ)**: All quantiles move by the same amount
   - Typical cause: Different code path, branch on secret data
   - Example: An `if` statement that adds constant overhead

2. **Tail effect (τ)**: Upper quantiles shift more than lower ones
   - Typical cause: Cache misses that occur probabilistically
   - Example: Table lookups that sometimes hit L1, sometimes miss to L3

The observed quantile differences $\Delta \in \mathbb{R}^9$ (the 9 decile differences between baseline and sample classes) are modeled as:

$$\Delta = X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \Sigma_0)$$

where:
- $\beta = (\mu, \tau)^\top$ are the effect magnitudes
- $X$ is a $9 \times 2$ design matrix encoding shift and tail patterns
- $\Sigma_0$ is the null covariance (estimated via bootstrap)

### Design Matrix

$$X = \begin{bmatrix} \mathbf{1} & \mathbf{b}_{\text{tail}} \end{bmatrix}$$

- **Shift basis** $\mathbf{1} = (1, 1, \ldots, 1)^\top$: A uniform shift affects all quantiles equally
- **Tail basis** $\mathbf{b}_{\text{tail}} = (-0.5, -0.375, \ldots, 0.375, 0.5)^\top$: Centered at zero, upper quantiles positive, lower negative

The tail basis is orthogonalized against the shift basis under the $\Sigma_0^{-1}$-weighted inner product, ensuring $\mu$ and $\tau$ estimates are uncorrelated.

## Sample Splitting

To avoid overfitting, data is split temporally (preserving measurement order):

- **Calibration set (first 30%)**: Used to estimate covariance $\Sigma_0$ and set prior scale
- **Inference set (last 70%)**: Used for posterior computation

The temporal split (rather than random) preserves independence—timing measurements are autocorrelated, so a random split would leak information.

## Covariance Estimation

The null covariance $\Sigma_0$ is estimated via **paired block bootstrap** on the calibration set:

1. Block-resample the joint interleaved sequence (both classes together)
2. Split by class label after resampling
3. Compute quantile difference vector $\Delta^*$
4. Repeat B = 2,000 times
5. Compute sample covariance of $\Delta^*$ vectors

**Key insight**: Paired resampling preserves cross-covariance from common-mode noise. Resampling classes independently would overestimate variance.

The covariance is then rescaled for the inference set size:
$$\Sigma_0 \leftarrow \Sigma_0 \cdot \frac{n_{\text{cal}}}{n_{\text{inf}}}$$

## Prior Specification

A zero-centered Gaussian prior on $\beta$:

$$\beta \sim \mathcal{N}(0, \Lambda_0), \quad \Lambda_0 = \text{diag}(\sigma_\mu^2, \sigma_\tau^2)$$

The prior scale is set proportional to the threshold:
$$\sigma_\mu = \sigma_\tau = 2\theta$$

where θ is the minimum effect of concern (from the attacker model). This ensures **calibrated priors**: $P(|\beta| > \theta \mid \text{prior}) \approx 62\%$, representing genuine uncertainty rather than bias toward detection.

## Posterior Computation

With Gaussian likelihood and Gaussian prior, the posterior is also Gaussian (conjugate):

$$\beta \mid \Delta \sim \mathcal{N}(\beta_{\text{post}}, \Lambda_{\text{post}})$$

where:
$$\Lambda_{\text{post}} = \left( X^\top \Sigma_0^{-1} X + \Lambda_0^{-1} \right)^{-1}$$
$$\beta_{\text{post}} = \Lambda_{\text{post}} X^\top \Sigma_0^{-1} \Delta$$

This is computed in closed form—no MCMC required.

## Leak Probability

The key question: "What's the probability of a leak exceeding my concern threshold θ?"

$$P(\text{leak} > \theta \mid \Delta) = P\bigl(\max_k |(X\beta)_k| > \theta \;\big|\; \Delta\bigr)$$

This is computed via Monte Carlo integration:

```
Draw N = 1,000 samples β ~ N(β_post, Λ_post)
For each sample:
    pred = X @ β                    # predicted Δ vector
    max_effect = max_k |pred[k]|    # same metric as CI gate
    if max_effect > θ: count++
leak_probability = count / N
```

The probability is internally consistent under the Gaussian model: when we report 80%, approximately 80% of the posterior predictive mass exceeds θ.

## Effect Estimation

The posterior mean $\beta_{\text{post}} = (\mu, \tau)^\top$ gives point estimates in nanoseconds.

### Pattern Classification

An effect component is "significant" if it exceeds twice its posterior standard error:

$$\text{significant}(\mu) \iff |\mu| > 2 \cdot \sqrt{\Lambda_{\text{post}}[0,0]}$$

| Shift significant? | Tail significant? | Pattern |
|--------------------|-------------------|---------|
| Yes | No | UniformShift |
| No | Yes | TailEffect |
| Yes | Yes | Mixed |
| No | No | Classify by relative magnitude |

### Credible Interval

Draw 1,000 samples from the posterior, compute total effect magnitude $\lVert\beta\rVert_2$ for each, report 2.5th and 97.5th percentiles as a 95% credible interval.

## Relationship to CI Gate

The Bayesian layer and CI gate use the same data and threshold θ, but answer different questions:

| Layer | Question | Output |
|-------|----------|--------|
| CI Gate | Is there a leak > θ? | Pass/Fail (FPR controlled) |
| Bayesian | What's the probability and magnitude? | P(leak), effect size |

They usually agree. When they diverge:
- **Gate passes, posterior medium**: Small effect below detection threshold
- **Gate fails, posterior medium**: Detected but effect is small

The CI gate is authoritative for pass/fail decisions; the Bayesian layer explains magnitude.

## Limitations

1. **Gaussian assumption**: The model assumes quantile differences are approximately Gaussian. This holds asymptotically but may be rough with discrete timers or small samples.

2. **Linear basis**: The shift+tail basis assumes effects vary smoothly across quantiles. "Hockey stick" patterns (flat through P80, spike at P90) will still be detected by the CI gate but may be mischaracterized by the effect decomposition.

3. **Prior sensitivity**: The posterior probability depends on the prior. Users with different beliefs can use the raw data and covariance estimates to compute their own posteriors.

## Summary

The Bayesian layer provides interpretable leak detection:

1. **Decompose** observed quantile differences into shift (μ) and tail (τ) components
2. **Estimate** covariance via paired block bootstrap on calibration data
3. **Compute** closed-form Gaussian posterior on effect parameters
4. **Integrate** to get probability that max effect exceeds threshold θ
5. **Report** leak probability, effect estimates, pattern classification, and credible intervals

This gives users actionable information—not just "leak detected" but "72% probability of a ~50ns uniform shift"—while the CI gate provides the reliable pass/fail decision for automation.
