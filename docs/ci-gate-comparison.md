# CI Gate Statistical Approaches: RTLF vs SILENT

This document summarizes the high-level statistical approaches used for the frequentist CI gate in two versions of the timing-oracle specification.

## RTLF-Based Approach (Jan 1, 2026 - commit 2d115c82)

The older approach used a **block permutation test** inspired by the RTLF paper (Dunsche et al., USENIX Security 2024).

### Hypotheses

Classical null hypothesis testing:
- **H₀**: No timing difference (d = 0)
- **H₁**: Timing difference exists (d ≠ 0)

### Test Statistic

Simple maximum absolute quantile difference (non-studentized):

$$M = \max_{p \in \{0.1, 0.2, \ldots, 0.9\}} |\Delta_p|$$

where $\Delta_p = \hat{q}_{\text{Fixed}}(p) - \hat{q}_{\text{Random}}(p)$.

### Algorithm

1. **Compute observed statistic**: From actual class labels, compute $M = \max_p |\Delta_p|$

2. **Block permutation** (B = 10,000 iterations):
   - Divide measurement indices into contiguous blocks of length b
   - For each block, with probability 0.5, flip all labels (Fixed ↔ Random)
   - Recompute quantiles from permuted labels
   - Record $M^* = \max_p |\Delta^*_p|$

3. **Compute threshold**:
   $$\tau = \text{Quantile}_{1-\alpha}(\{M^{*(1)}, \ldots, M^{*(B)}\})$$

4. **Threshold flooring** (practical significance):
   $$\tau_{\text{effective}} = \max(\tau_\alpha, 1\text{ tick}, \text{min\_effect\_of\_concern})$$

5. **Decision**: Reject H₀ (declare leak) if $M > \tau_{\text{effective}}$

### Key Characteristics

- **No studentization**: All quantiles weighted equally regardless of their variance
- **No quantile filtering**: All 9 deciles always included
- **Classical null**: Tests whether effect = 0, with practical significance handled via threshold flooring
- **Permutation-based**: Null distribution estimated by permuting class labels within blocks
- **Threshold flooring**: Post-hoc adjustment to avoid flagging sub-threshold effects

---

## SILENT-Based Approach (Current - Jan 5, 2026+)

The current approach follows the SILENT framework (Dunsche et al., arXiv 2025), which extends RTLF to support **relevant hypothesis testing** with a practical significance threshold built into the hypothesis itself.

### Hypotheses

Relevant hypothesis testing:
- **H₀**: Effect is negligible (d ≤ θ)
- **H₁**: Effect exceeds threshold (d > θ)

where θ = `min_effect_of_concern` (the practical significance threshold).

### Test Statistic (Continuous Mode)

Studentized maximum statistic over filtered quantiles:

$$\hat{Q}_{\max} = \max_{k \in K_{\text{sub}}^{\text{max}}} \frac{A_k - \theta}{\hat{\sigma}_k}$$

where:
- $A_k = |\hat{q}_F(k) - \hat{q}_R(k)|$ is the absolute quantile difference
- $\hat{\sigma}_k$ is the bootstrap-estimated standard deviation of $A_k$
- $K_{\text{sub}}^{\text{max}}$ is a filtered subset of quantiles

### Quantile Filtering (SILENT-style)

Two-step filtering to improve power:

**Step 1 - Variance filter**:
$$K_{\text{sub}} = \{k \in K : \hat{\sigma}_k^2 \leq 5 \cdot \overline{\sigma^2}\}$$

Removes quantiles with unusually high variance that would dominate even after studentization.

**Step 2 - Power-boost filter**:
$$K_{\text{sub}}^{\text{max}} = \{k \in K_{\text{sub}} : A_k + 30\hat{\sigma}_k\sqrt{(\log n)^{3/2}/n} \geq \theta\}$$

Keeps only quantiles likely to contribute to detection.

### Algorithm

1. **Compute observed differences**: $A_k = |\hat{q}_F(k) - \hat{q}_R(k)|$ for each quantile

2. **Bootstrap** (B = 2,000 iterations, paired resampling):
   - Sample block indices, apply to **both classes using the same indices**
   - Compute bootstrap quantile differences $A_k^{*(b)}$

3. **Variance estimation**: From bootstrap samples, compute $\hat{\sigma}_k$ for each quantile

4. **Filter quantiles**: Apply variance and power-boost filters to get $K_{\text{sub}}^{\text{max}}$

5. **Centered bootstrap statistics**:
   $$\hat{Q}^{*(b)}_{\max} = \max_{k \in K_{\text{sub}}^{\text{max}}} \frac{A^{*(b)}_k - A_k}{\hat{\sigma}_k}$$

   Note: Centered at observed $A_k$, not at θ.

6. **Critical value**: $c^*_{1-\alpha} = \text{Quantile}_{1-\alpha}(\{\hat{Q}^{*(1)}_{\max}, \ldots, \hat{Q}^{*(B)}_{\max}\})$

7. **Test statistic**: $\hat{Q}_{\max} = \max_{k \in K_{\text{sub}}^{\text{max}}} \frac{A_k - \theta}{\hat{\sigma}_k}$

8. **Decision**: Reject H₀ (declare leak) if $\hat{Q}_{\max} > c^*_{1-\alpha}$

### Discrete Timer Mode

When timer resolution is coarse (< 10% unique values), SILENT uses a different approach:

- **Non-studentized** statistics with $\sqrt{n}$ scaling
- **m-out-of-n bootstrap** ($m = \lfloor n^{2/3} \rfloor$) instead of full-n bootstrap
- Asymmetric scaling: bootstrap uses $\sqrt{m}$, test statistic uses $\sqrt{n}$
- Mid-distribution quantiles to handle tied values

### Key Characteristics

- **Studentization**: Normalizes each quantile's contribution by its variability
- **Quantile filtering**: Excludes high-variance and low-power quantiles
- **Relevant null**: Tests d ≤ θ directly, not d = 0
- **Paired resampling**: Same indices for both classes, preserving cross-covariance
- **FPR guarantee at boundary**: When true effect = θ, FPR ≤ α (SILENT Theorem 2)

---

## Summary of Key Differences

| Aspect | RTLF Approach | SILENT Approach |
|--------|---------------|-----------------|
| **Null hypothesis** | d = 0 | d ≤ θ |
| **Threshold handling** | Post-hoc flooring | Built into hypothesis |
| **Test statistic** | $\max_k |A_k|$ | $\max_k (A_k - \theta)/\hat{\sigma}_k$ |
| **Studentization** | No | Yes (continuous mode) |
| **Quantile filtering** | No | Yes (variance + power-boost) |
| **Resampling** | Block permutation | Paired block bootstrap |
| **Discrete mode** | Not specified | m-out-of-n bootstrap |
| **FPR at boundary** | ~50% at d = θ | ≤ α at d = θ |

### Why the Change?

The RTLF approach with threshold flooring has a critical flaw: at the boundary (true effect = θ), the observed statistic $\hat{d}$ is centered at θ. The floored threshold $\tau = θ$ means you reject ~50% of the time at the boundary—regardless of what α you set. The FPR guarantee only holds when the true effect is well below θ.

SILENT's relevant-hypothesis approach fixes this by building θ into the hypothesis itself. The centered, studentized bootstrap ensures that at d = θ, the test statistic $\hat{Q}_{\max} \approx 0$ and the bootstrap distribution is also centered at 0, giving the correct α rejection rate at the boundary.

---

## References

1. Dunsche, M., Lamp, M., & Pöpper, C. (2024). "With Great Power Come Great Side Channels: Statistical Timing Side-Channel Analyses with Bounded Type-1 Errors." USENIX Security. — RTLF bootstrap methodology

2. Dunsche, M., Lamp, M., & Pöpper, C. (2025). "SILENT: A New Lens on Statistics in Software Timing Side Channels." arXiv:2504.19821. — Extension supporting negligible leak thresholds with bounded FPR at the boundary
