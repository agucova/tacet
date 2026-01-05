# Bayesian Calibration Bug Report

## Summary

The Bayesian inference layer produces excessive false positives (~22-40%) when testing null data (random vs random inputs) with kperf-enabled measurements on Apple Silicon. The issue arises because ultra-low minimum detectable effects (MDE < 2ns) cause random measurement noise to be interpreted as statistically and practically significant timing leaks.

## Background

The timing oracle uses a two-layer statistical approach:

1. **Layer 1 (CI Gate)**: Frequentist hypothesis test using max-statistic bootstrap on 9 quantile differences, with bounded false positive rate ≤ 2×α
2. **Layer 2 (Bayesian Inference)**: Computes posterior probability of timing leak given observed effect size

The Bayesian layer uses:
- **Prior**: `prior_no_leak` (default: 0.5) representing P(no leak) before seeing data
- **Prior scale**: `max(2×MDE, min_effect_ns)` representing the expected effect size under H₁ (leak hypothesis)
- **Likelihood ratio**: Based on comparing observed effect to prior scale
- **Output**: `leak_probability` = P(leak | data) via Bayes factor

## Test Setup

**Test**: `tests/calibration.rs::bayesian_calibration`
- **Purpose**: Verify Bayesian layer doesn't over-concentrate on high probabilities for null data
- **Method**: Run 100 trials on pure noise (baseline and sample both random)
- **Expected**: < 10% of trials should have `leak_probability > 0.9`
- **Platform**: Apple Silicon M-series with kperf PMU cycle counting (~0.3ns resolution)
- **Configuration**: `TimingOracle::quick()` (5k samples, 50 bootstrap iterations)

## Observed Behavior

**Failure**: 22% of trials produced `leak_probability > 0.9` (expected: < 10%)

**Sample diagnostic run (10 trials on null data with kperf)**:

```
Trial 1: leak_prob=99.7%  BF=1.02e3   MDE=1.0ns/2.2ns   effect=Some("2.0ns/-0.8ns")
Trial 2: leak_prob=91.6%  BF=3.27e1   MDE=1.0ns/2.7ns   effect=Some("1.3ns/-0.1ns")
Trial 3: leak_prob=2.6%   BF=7.87e-2  MDE=1.4ns/3.4ns   effect=None
Trial 4: leak_prob=0.0%   BF=1.34e-3  MDE=0.6ns/2.1ns   effect=None
Trial 5: leak_prob=1.1%   BF=3.27e-2  MDE=1.0ns/2.2ns   effect=None
Trial 6: leak_prob=95.4%  BF=6.27e1   MDE=0.5ns/1.2ns   effect=Some("-0.3ns/1.7ns")
Trial 7: leak_prob=100.0% BF=1.20e53  MDE=0.9ns/1.9ns   effect=Some("4.0ns/0.4ns")
Trial 8: leak_prob=0.2%   BF=5.05e-3  MDE=1.0ns/2.6ns   effect=None
Trial 9: leak_prob=0.9%   BF=2.59e-2  MDE=1.3ns/3.5ns   effect=None
Trial 10: leak_prob=3.0%  BF=9.15e-2  MDE=0.9ns/2.0ns   effect=None
```

**Result**: 4/10 trials (40%) had `leak_probability > 90%`

## Root Cause Analysis

### 1. Ultra-Low MDE with kperf

With kperf's ~0.3ns measurement resolution, the library achieves extremely low MDEs:
- Shift MDE: 0.5-1.4ns
- Tail MDE: 1.2-3.5ns

This is *technically* working as designed - kperf enables detection of sub-nanosecond timing differences.

### 2. Measurement Noise Exceeds MDE

Even on null data (no true effect), measurement noise produces small random variations:
- Observed effects: ±0.3ns to ±4.0ns
- These are purely random fluctuations, not real timing leaks

With MDE < 2ns, random noise routinely exceeds the minimum detectable effect.

### 3. Bayesian Interpretation of Small Effects

The Bayesian layer computes:
```
prior_scale = max(2×MDE, min_effect_ns)  // default min_effect_ns = 10.0
```

With kperf:
- MDE = 1.0ns → prior_scale = max(2.0ns, 10.0ns) = **10.0ns**
- Observed effect = 2.0ns (random noise)

However, looking at Trial 7 (BF=1.20e53), the Bayes factor suggests the observed 4.0ns effect is being treated as highly significant relative to the prior.

The issue appears to be:
- The effect size (shift_ns, tail_ns) is being compared to the prior scale
- Small MDEs mean small random fluctuations exceed statistical significance thresholds
- The Bayesian likelihood ratio heavily favors H₁ when effect ≫ MDE, even if effect is tiny in absolute terms (< 5ns)

### 4. Statistical vs Practical Significance

The library distinguishes:
- **Statistical significance**: Effect exceeds MDE (detected by CI gate + Bayesian layer)
- **Practical significance**: Effect has security implications (classified by exploitability)

Current exploitability thresholds:
- Negligible: < 100ns (not exploitable)
- PossibleLAN: 100-500ns
- LikelyLAN: 500ns-20μs
- PossibleRemote: > 20μs

A 2ns timing difference is:
- ✅ Statistically detectable with kperf
- ❌ Not practically significant (100× below exploitability threshold)
- ❌ Just measurement noise on null data

## Why This Happens

The Bayesian layer was likely designed with standard timers (~40ns resolution) in mind, where:
- Typical MDE: 20-100ns
- Prior scale: max(40-200ns, 10ns) = 40-200ns
- Random noise: ±10-30ns
- Random noise ≪ MDE, so false positives are rare

With kperf:
- Typical MDE: 0.5-2ns (20-100× smaller)
- Prior scale: max(1-4ns, 10ns) = 10ns (still conservative)
- Random noise: ±1-4ns (same absolute magnitude)
- **Random noise ≈ MDE**, so false positives are common

The Bayesian layer is correctly detecting statistically significant differences, but it's too sensitive when MDE is ultra-low.

## Questions for Specification Designer

1. **Prior Scale Minimum**: Should we enforce a minimum prior scale (e.g., 5-10ns) regardless of MDE to account for irreducible measurement noise? This would make the Bayesian layer more skeptical of tiny effects.

2. **MDE Calibration**: When MDE < 5ns (ultra-high resolution), should we:
   - Use a larger prior scale to be more conservative?
   - Require larger effect sizes before high leak probabilities?
   - Adjust the Bayesian model to account for measurement noise floors?

3. **Noise Floor Modeling**: Should the Bayesian layer explicitly model a "noise floor" (e.g., 2-5ns) below which effects are attributed to measurement noise rather than real timing differences?

4. **Two-Stage Interpretation**: Should we separate:
   - **Stage 1**: Statistical detection (is there *any* detectable difference?)
   - **Stage 2**: Practical assessment (is the difference *large enough to matter*?)

   Currently, the Bayesian layer focuses on statistical detection, but perhaps `leak_probability` should integrate practical significance (e.g., downweight probabilities when effect < 10ns).

5. **Platform-Specific Calibration**: Should the Bayesian inference parameters adapt to measurement resolution? For example:
   - Standard timer (40ns res): Use current parameters
   - kperf/perf (0.3ns res): Use more conservative prior scale or higher evidence threshold

## Current Workaround

The exploitability classification already handles this correctly:
- Effects < 100ns → `Exploitability::Negligible`
- User can check `result.exploitability` rather than just `leak_probability`

However, the Bayesian `leak_probability` should ideally not produce 90-100% confidence on negligible effects from null data.

## Proposed Solutions (for Discussion)

### Option A: Minimum Prior Scale
```rust
let prior_scale = max(max(2×MDE, 5.0), min_effect_ns);
// Ensures prior scale ≥ 5ns even with ultra-low MDE
```

### Option B: Noise Floor Model
```rust
let noise_floor = 2.0; // ns
if effect < noise_floor {
    // Attribute to measurement noise, reduce leak probability
}
```

### Option C: MDE-Relative Threshold
```rust
// Only trigger high leak_probability if effect > 5×MDE
let evidence_threshold = 5.0 * mde;
if effect < evidence_threshold {
    // Apply skeptical prior or reduce Bayes factor
}
```

### Option D: Integrate Exploitability
```rust
// Downweight leak_probability when exploitability is Negligible
if exploitability == Negligible && effect < 10.0 {
    leak_probability *= 0.1; // Strong skepticism on tiny effects
}
```

## Impact

- **Current behavior**: Library correctly detects statistically significant differences with kperf, but over-reports leak probability on practically insignificant noise
- **User impact**: Users with kperf see many false high leak probabilities on constant-time code
- **CI impact**: `bayesian_calibration` test fails, blocking release

## Recommendation

I recommend discussing with the spec designer whether:
1. The Bayesian layer should be more conservative when MDE is ultra-low
2. `leak_probability` should integrate practical significance, not just statistical significance
3. A minimum prior scale (5-10ns) is appropriate to account for irreducible measurement noise

The statistical machinery is working correctly in detecting differences, but the *interpretation* layer may need adjustment for ultra-high-resolution measurements.
