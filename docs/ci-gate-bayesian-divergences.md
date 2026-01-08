# CI Gate vs Bayesian Layer Divergence Analysis

This document catalogs observed divergences between the CI gate (Layer 1) and Bayesian inference (Layer 2) in timing-oracle, along with debug output to help diagnose root causes.

## Summary of Findings

**Direction of Disagreement**: In all cases observed, disagreements go in one direction:
- **CI gate passes** but **Bayesian shows elevated leak probability**
- No cases found where CI gate fails but Bayesian probability is low

This suggests the CI gate is systematically more conservative (fewer detections) than the Bayesian layer for edge cases.

## Hypothesis Categories

Three main hypotheses for divergence were proposed:

1. **Studentization Asymmetry**: SILENT's CI gate divides by σ̂ₖ per-quantile inside the max, while Bayesian GLS weights the entire linear model fit via Σ₀
2. **Quantile Filtering**: SILENT's power-boost filter can exclude quantiles aggressively
3. **Different Effect Definitions**: CI gate uses max|Aₖ| (raw quantile diff), Bayesian uses max|(Xβ)ₖ| (linear model prediction)

---

## Case 1: High-Variance Quantile Damping (Studentization Asymmetry)

**Test**: `known_safe` fixed-vs-fixed calibration trial
**Verdict**: CI gate Pass, Bayesian 75.1% leak probability

### Raw Data
```
ci_gate result=Pass threshold=447.79 max_observed=53.50 ns
bayes leak_probability=0.751 theta=10.00ns
```

### Debug Output
```
delta_infer=[0.77, 2.83, 0.51, 3.22, 53.50, 1.54, 0.0, -0.90, -0.26]
a_max_abs_delta=53.499

ci_gate_continuous sigmas=[1.42, 1.93, 2.04, 20.71, 54.78, 11.85, 2.01, 2.34, 2.90]
ci_gate_continuous filtered_indices=[3, 5]
ci_gate_continuous q_max=-0.3276 critical_value=7.9920

delta_z pooled=[1.07, 2.66, 0.52, 4.76, 78.94, 2.29, 0.0, 0.09, 0.02]
delta_z infer=[0.50, 1.65, 0.31, 0.50, 1.39, 0.41, 0.0, 0.34, 0.07]

bayes effect_mean shift=1.3ns tail=-1.2ns
bayes max_effect_ci=(0.60,6.75)ns
```

### Analysis

**Root cause: Studentization damping of high-variance quantile**

The key effect is at quantile 5 (index 4):
- Absolute difference: A₄ = 53.50ns (well above θ=10ns)
- Bootstrap variance: σ̂₄ = 54.78ns (very high)
- Studentized value: (53.50 - 10.00) / 54.78 = 0.79 (not significant)

The CI gate's per-quantile studentization effectively says: "Yes, 53.50ns is large, but so is the noise at this quantile, so it's not statistically significant."

Meanwhile, the Bayesian layer:
- Uses GLS with full covariance matrix
- Projects the 9D quantile difference onto 2D (shift, tail)
- The 53.50ns effect at a single quantile gets partially absorbed but still contributes
- 75.1% posterior mass above θ=10ns

**Diagnostic indicators**:
- Large discrepancy between `delta_z pooled` and `delta_z infer` at the same quantile
- `filtered_indices` showing only 2 of 9 quantiles used
- High variance at the quantile with the largest observed effect

---

## Case 2: Discrete Mode Numerical Instability

**Test**: `aes128_byte_pattern_independence`
**Verdict**: CI gate Pass, Bayesian 23.3% leak probability

### Raw Data
```
ci_gate result=Pass threshold=1.19 max_observed=1.00 ticks
bayes leak_probability=0.233 theta=1.00ticks
```

### Debug Output
```
discrete_mode=true min_uniqueness_ratio=0.004

delta_infer=[0.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0]
a_max_abs_delta=1.000

pooled_cov trace=97.89 min_diag=0.0002 max_diag=51.47
infer_cov trace=99.02 min_diag=0.0000 max_diag=71.80

pooled_quad cond_est=2.05e5
infer_quad cond_est=3.08e8

delta_z pooled=[0.0, 14.45, 0.0, 0.0, 12.31, 0.0, 0.42, 0.0, 0.14]
delta_z infer=[0.0, 11.51, 0.0, 0.0, 11.35, 0.0, 0.55, 0.0, 0.12]
```

### Analysis

**Root cause: Near-zero variance in discrete mode creates numerical instability**

With only 0.4% unique values:
- Many quantiles have identical values across all samples
- This creates near-zero diagonal entries in covariance: min_diag=0.0002
- Condition number explodes: 2×10⁵ to 3×10⁸

Effects:
- Some quantiles get inflated z-scores (14.45, 12.31) due to near-zero variance
- The regularization (1% of mean variance floor) isn't aggressive enough
- Bayesian posterior spreads out due to ill-conditioned covariance

**Diagnostic indicators**:
- `min_uniqueness_ratio` < 0.10 (discrete mode triggered)
- Extremely high `cond_est` values (>10⁵)
- Very small `min_diag` in covariance (<0.01)
- Some `delta_z` values are extremely large (>10)

---

## Case 3: Zero-Effect with Posterior Uncertainty

**Test**: `aes128_multiple_blocks_constant_time`
**Verdict**: CI gate Pass, Bayesian 36.1% leak probability

### Raw Data
```
ci_gate result=Pass threshold=1.41 max_observed=0.00 ticks
bayes leak_probability=0.361 theta=1.00ticks
```

### Debug Output
```
discrete_mode=true min_uniqueness_ratio=0.021

delta_infer=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
a_max_abs_delta=0.000

pooled_cov trace=3515.53 min_diag=0.0038 max_diag=2149.54
pooled_quad cond_est=5.06e5
infer_quad cond_est=3.86e5

delta_z pooled=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
delta_z infer=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

bayes_mahalanobis pooled=Some(0.0) infer=Some(0.0)
```

### Analysis

**Root cause: Ill-conditioned covariance inflates posterior uncertainty**

Even with zero observed effect:
- All quantile differences are exactly 0
- Mahalanobis distance is 0
- Yet Bayesian reports 36.1% leak probability

The mechanism:
- Covariance matrix has condition number ~5×10⁵
- This creates high uncertainty in some directions of the posterior
- Monte Carlo integration samples from this wide posterior
- 36.1% of samples exceed θ=1 tick despite zero point estimate

**Diagnostic indicators**:
- `delta_infer` is all zeros
- `bayes_mahalanobis` is 0 (no effect detected)
- High `cond_est` values (>10⁵)
- Non-zero `leak_probability` despite zero observed effect

---

## Case 4: Boundary Effect with Intermediate Probability

**Test**: Calibration trial
**Verdict**: CI gate Pass, Bayesian 42.8% leak probability

### Raw Data
```
ci_gate result=Pass threshold=52.55 max_observed=24.00 ticks
bayes leak_probability=0.428 theta=38.93ticks
```

### Debug Output
```
delta_infer=[-1.0, -2.0, -4.0, -17.0, -21.0, -24.0, -23.0, -23.0, -21.0]
a_max_abs_delta=24.000

mde shift=1.70ticks tail=3.58ticks

delta_z pooled=[1.24, 2.05, 3.17, 15.62, 23.51, 18.50, 10.40, 10.16, 7.83]
delta_z infer=[0.41, 0.48, 1.03, 3.93, 4.53, 6.26, 6.94, 8.36, 7.93]

bayes effect_mean shift=-2.03ticks tail=-3.18ticks
bayes effect_ci shift=(-4.35,0.30) tail=(-7.83,1.48)ticks
```

### Analysis

**Root cause: Effect below threshold but posterior uncertainty spans threshold**

Observed maximum effect: 24 ticks
Threshold θ: 38.93 ticks
Ratio: 24/38.93 = 0.62 (below threshold)

Yet Bayesian gives 42.8% probability because:
- The 95% CI for shift is (-4.35, 0.30), which crosses zero
- The 95% CI for tail is (-7.83, 1.48), also crossing zero
- Combined, there's substantial uncertainty about the true effect
- 42.8% of posterior samples exceed θ despite point estimate being below

This is actually **correct behavior**: the effect is genuinely uncertain, and the Bayesian layer is quantifying that uncertainty.

**Diagnostic indicators**:
- max_observed < theta (effect below threshold)
- Effect credible intervals crossing zero
- Intermediate leak_probability (10-50%)

---

## Case 5: Measurement Noise Misinterpreted as Leak (ECC Test)

**Test**: `x25519_multiple_operations_constant_time`
**Verdict**: CI gate Pass, Bayesian 95.5-98.4% leak probability
**Reality**: **NOT a real leak** - x25519_dalek is constant-time by design

### Raw Data (with coarse timer - cntvct)
```
ci_gate result=Pass threshold=487725.77 max_observed=25211.68 ns
bayes leak_probability=0.984 theta=50.00ns
```

### Raw Data (with high-resolution timer - kperf)
```
ci_gate result=Pass threshold=85959.88 max_observed=1243.94 ns
bayes leak_probability=0.955 theta=50.00ns
bayes leak_probability_zero_delta=0.894   <-- CRITICAL!
```

### Debug Output (kperf run)
```
delta_infer=[-178.05, -97.77, -123.41, -144.02, -195.30, -545.41, -1243.94, -980.82, 415.76]
a_max_abs_delta=1243.939

mde shift=160.69ns tail=352.77ns   <-- MDE >> theta!

bayes effect_ci shift=(-267.392,108.398) tail=(-404.317,420.666)ns
```

### Analysis

**Root cause: Measurement noise with high prior uncertainty**

**This is NOT a real timing leak.** The evidence:

1. **`leak_probability_zero_delta=0.894`**: Even with **zero observed effect**, the Bayesian layer would report 89.4% leak probability. The 95.5% is dominated by prior uncertainty, not observed data.

2. **Effect credible intervals cross zero**: shift=(-267.4, 108.4)ns, tail=(-404.3, 420.7)ns

3. **MDE >> θ**: Minimum detectable effect (160.7ns) is 3× the threshold (50ns)

4. **x25519_dalek uses the donna implementation**, specifically designed for constant-time operation

5. **CI gate passes** with margin (q_max=1.54 vs critical=5.03)

The CI gate is **correct** here - it's saying "the variance is so high that we can't conclude there's a leak." The Bayesian layer is also technically correct - it's saying "we're very uncertain" - but the high probability is misleading without context.

**Why the apparent 25μs effect with coarse timer?**

The original run used cntvct (41.67ns resolution) because kperf was locked:
- p90 showed 25,211ns effect with σ=90,430ns variance
- This is characteristic of **tail noise from system interference** (interrupts, scheduling)
- With kperf, the same quantile shows only 415ns effect

**Diagnostic indicators for measurement noise vs real leak**:
- `leak_probability_zero_delta` > 0.5 = prior dominates
- Effect credible intervals crossing zero
- MDE >> θ (insufficient precision)
- Variance much larger than effect at tail quantiles
- Different results with different timers (noise is timer-dependent)

---

## Case 6: Large Tail Effect Missed (ECC Scalar Mult)

**Test**: `x25519_scalar_mult_constant_time`
**Verdict**: CI gate Pass, Bayesian 84.8% leak probability

### Raw Data
```
ci_gate result=Pass threshold=26.90 max_observed=46.00 ticks
bayes leak_probability=0.848 theta=1.20ticks
```

### Debug Output
```
delta_infer=[0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 46.0]
a_max_abs_delta=46.000

discrete_mode=true min_uniqueness_ratio=0.089

pooled_cov trace=18421.42 min_diag=0.24 max_diag=18385.59
pooled_quad cond_est=4.61e5
infer_quad cond_est=4.43e5

delta_z pooled=[0.0, 1.80, 1.98, 2.05, 2.02, 1.86, 3.19, 0.18, 0.34]
delta_z infer=[0.0, 0.37, 0.60, 1.24, 1.08, 0.44, 0.65, 0.13, 0.09]

mde shift=1.17ticks tail=2.32ticks
```

### Analysis

**Root cause: Tail quantile effect damped by high variance in discrete mode**

Key observations:
- The p90 quantile has a 46-tick effect (38× the threshold of 1.2 ticks)
- But delta_z at p90 is only 0.34 (pooled) / 0.09 (infer)
- This indicates the variance at p90 is extremely high

The pattern:
- Lower quantiles (p10-p70) have small effects (0-2 ticks)
- P90 has massive effect (46 ticks) but also massive variance
- Studentization dampens the p90 contribution
- CI gate passes even though max_observed (46) > threshold (26.9)...

Wait, this is interesting: max_observed (46) is greater than CI gate threshold (26.9), but CI gate still passes. This is because:
- The "threshold" shown in output is computed differently
- The actual test is q_max vs critical_value (studentized comparison)
- The displayed threshold is for human interpretation, not the actual decision boundary

**Diagnostic indicators**:
- max_observed > displayed threshold but CI gate passes
- Very small delta_z at the quantile with largest effect
- High condition number (>10⁵) combined with discrete mode

---

## Case 7: θ=0 Research Mode (Artificial Divergence)

**Test**: `cache_line_boundary_effects`, `table_lookup_small_l1`
**Verdict**: CI gate Pass, Bayesian 100% leak probability

### Raw Data
```
# cache_line_boundary_effects
ci_gate result=Pass threshold=9.09 max_observed=3.00 ticks
bayes leak_probability=1.000 theta=0.00ticks

# table_lookup_small_l1
ci_gate result=Pass threshold=182.13 max_observed=7.59 ns
bayes leak_probability=1.000 theta=0.00ns
```

### Analysis

**Root cause: θ=0 makes any non-zero effect a "leak"**

When using Research mode (θ=0):
- CI gate has a non-zero effective threshold from studentization and bootstrap critical values
- Bayesian with θ=0 reports 100% for any non-zero observed effect

This is **artificial divergence** - the layers are asking different questions:
- CI gate: "Is the studentized effect significant?"
- Bayesian: "Is any effect > 0?" (always yes with noisy data)

**Diagnostic indicators**:
- `theta=0.00` in Bayesian output
- `leak_probability=1.000` regardless of CI gate result
- CI gate threshold much higher than max_observed

---

## Summary Table

| Case | CI Gate | Bayes | Root Cause | Direction |
|------|---------|-------|------------|-----------|
| High-variance damping | Pass | 75.1% | Studentization dampens high-σ quantile | Bayes more sensitive |
| Discrete instability | Pass | 23.3% | Near-zero variance creates numerical issues | Bayes more sensitive |
| Zero-effect inflation | Pass | 36.1% | Ill-conditioned covariance inflates posterior | Bayes more sensitive |
| Boundary uncertainty | Pass | 42.8% | Effect near threshold with high uncertainty | Bayes more sensitive |
| **Extreme tail damping (25μs)** | Pass | **98.4%** | 25μs effect at p90 with 90μs variance | **Bayes dramatically more sensitive** |
| **Discrete tail damping (46 ticks)** | Pass | **84.8%** | 46-tick effect at p90 in discrete mode | **Bayes more sensitive** |
| θ=0 Research mode | Pass | 100% | Different effective thresholds | Artificial |

## Recommendations

1. **For debugging divergences**: Check `cond_est`, `min_diag`, and `filtered_indices` in debug output

2. **For discrete mode**: Consider stronger covariance regularization when `cond_est` > 10⁴

3. **For CI reliability**: The CI gate's studentization is conservative by design (bounded FPR), but can miss effects when a single high-variance quantile dominates

4. **For interpretation**: When CI gate passes but Bayesian shows 20-80%, treat as "inconclusive" and consider:
   - Running with more samples
   - Checking the MDE relative to θ
   - Looking at individual quantile contributions

## Debug Variables Quick Reference

| Variable | Meaning | Warning Signs |
|----------|---------|---------------|
| `filtered_indices` | Quantiles used by CI gate | Only 1-2 quantiles remaining |
| `sigmas` | Per-quantile bootstrap σ | Very high values (>50ns) |
| `cond_est` | Covariance condition number | Values > 10⁴ |
| `min_diag` | Minimum covariance diagonal | Values < 0.01 |
| `delta_z` | Studentized quantile differences | Values > 10 indicate near-zero variance |
| `discrete_mode` | Whether discrete mode is active | Combined with high cond_est = instability |
| **`leak_probability_zero_delta`** | **Bayesian prob with Δ=0** | **Values > 0.5 = prior dominates posterior** |

### Critical Diagnostic: `leak_probability_zero_delta`

This variable shows what the Bayesian leak probability would be if the observed effect were exactly zero. It's crucial for interpreting high leak probabilities:

**If `leak_probability_zero_delta` is high (>0.5):**
- The Bayesian probability is dominated by **prior uncertainty**, not observed data
- A high `leak_probability` does NOT indicate evidence of a leak
- The measurement simply isn't precise enough to make strong claims

**Example from x25519_multiple_operations test:**
```
leak_probability=0.955          # 95.5% - looks like a leak!
leak_probability_zero_delta=0.894  # BUT: 89.4% even with zero effect!
```

This tells us 95.5% - 89.4% = 6.5% of the probability comes from actual observed data. The rest is prior uncertainty. This is NOT a real timing leak - it's just a noisy measurement where the Bayesian layer correctly reports "we're very uncertain."

**Interpretation guide:**

| leak_prob | zero_delta | Interpretation |
|-----------|------------|----------------|
| High | Low (<0.2) | Strong evidence of leak |
| High | Medium (0.2-0.5) | Moderate evidence, check MDE |
| High | High (>0.5) | **Prior dominates - inconclusive** |
| Low | Any | No leak (or below threshold) |

---

## Intentional Leak Tests (Contrast: Both Layers Agree)

For tests with **large, clear timing leaks**, both layers agree completely:

| Test | Injected Effect | CI Gate | Bayesian | Agreement |
|------|-----------------|---------|----------|-----------|
| `detects_branch_timing` | ~18,794 ticks | **Fail** | **100%** | Yes |
| `detects_early_exit_comparison` | ~12,523 ticks | **Fail** | **100%** | Yes |
| `exploitability_possible_lan` | ~52 ticks | **Fail** | **100%** | Yes |
| `exploitability_possible_remote` | ~1,199 ticks | **Fail** | **100%** | Yes |
| `effect_pattern_pure_uniform_shift` | ~500 cycles | **Fail** | **100%** | Yes |
| `effect_pattern_pure_tail` | ~2,000 cycles (15% prob) | **Fail** | **100%** | Yes |
| `exploitability_negligible` | ~5 ticks | **Pass** | **0%** | Yes |

**Key insight**: When the effect is large and clear (well above threshold), both layers agree:
- CI gate fails → Bayesian says 100% leak
- CI gate passes with effect well below threshold → Bayesian says 0% leak

**Divergences only occur in edge cases**:
- Effects near the threshold boundary
- High-variance quantiles that get studentization-damped
- Discrete mode with numerical instability
- Ill-conditioned covariance matrices

This is actually reassuring: the layers agree on clear-cut cases and only diverge when the statistical signal is genuinely ambiguous or when one layer's methodology handles an edge case differently.

---

## Conclusions for Hypothesis Evaluation

### 1. Studentization Asymmetry (CONFIRMED as major cause)

**Evidence**: Cases 1, 5, and 6 directly demonstrate this.

The CI gate's per-quantile studentization (dividing each A_k - θ by σ̂_k) can dramatically suppress large effects when the corresponding quantile has high variance. This is by design (SILENT's bounded FPR guarantee), but creates a blind spot:

- A 25μs effect at p90 gets studentized to 0.28 because σ̂₉₀ = 90μs
- The Bayesian layer's GLS doesn't do per-quantile studentization inside the max
- GLS weights by the full covariance matrix, which distributes the variance contribution differently

**Key insight**: The studentization happens **inside the max** for CI gate, but the Bayesian layer takes the max **after** GLS projection. These are fundamentally different operations.

### 2. Quantile Filtering (CONFIRMED as contributing factor)

**Evidence**: Case 1 shows `filtered_indices=[3, 5]` - only 2 of 9 quantiles used.

The power-boost filter can exclude quantiles that don't appear to contribute to detection. This is conservative by design, but means the CI gate may ignore quantiles where the Bayesian layer sees meaningful signal.

However, filtering alone doesn't explain the most extreme cases (5 and 6), where all or most quantiles were retained.

### 3. Different Effect Definitions (CONFIRMED as conceptual difference)

**Evidence**: All cases show the CI gate using max|A_k| while Bayesian uses max|(Xβ)_k|.

The 2D linear model (shift + tail) smooths across quantiles:
- An effect concentrated at one quantile gets partially absorbed into the tail component
- The predicted effect max|(Xβ)_k| is always ≤ max|A_k| due to smoothing
- But the Bayesian posterior can still place high probability mass above θ

### 4. Numerical Stability in Discrete Mode (CONFIRMED as pathology)

**Evidence**: Cases 2, 3, and 6 show condition numbers >10⁵.

In discrete mode with few unique values:
- Some quantiles have near-zero variance (pinned to same tick)
- Covariance matrix becomes ill-conditioned
- This can inflate posterior uncertainty (Case 3) or create extreme z-scores (Case 2)

---

## Which Layer is More Correct?

**For bounded FPR guarantee**: CI gate is correct by construction (SILENT's Theorem 2).

**For sensitivity to real effects**: The Bayesian layer appears more sensitive, but may have inflated false positive rate in edge cases.

**The divergences reveal a fundamental tradeoff**:
- CI gate: Conservative, bounded FPR, may miss large effects with high variance
- Bayesian: More sensitive, better calibrated probability estimates, but not FPR-controlled

**Recommendation for users**: When CI gate passes but Bayesian shows >50%, investigate the specific quantile contributions and variance structure before concluding "no leak."
