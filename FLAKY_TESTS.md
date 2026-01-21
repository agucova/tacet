# Flaky Tests Investigation (2026-01-20)

## False Positive Tests (standard timer - cntvct_el0)

These tests occasionally get confident `Fail` outcomes when they should `Pass`.
The oracle detects small effects (~30-40ns) near timer resolution and reports
high confidence, but these are likely measurement artifacts.

| Test | File | Failure Rate |
|------|------|--------------|
| `blake2b_512_constant_time` | `hash_timing.rs` | ~1/15 |
| `kyber768_decapsulate_constant_time` | `pqcrypto_timing.rs` | ~1/20 |

### Root Cause
- Timer resolution (~42ns cntvct_el0) is close to the detected effects
- Small systematic biases in measurement can appear as timing leaks
- These are "wrongly confident" results, not Inconclusive

### Potential Solutions
1. Use higher threshold (RemoteNetwork instead of AdjacentNetwork)
2. Add effect magnitude check before panicking on Fail
3. Investigate if there's a real intermittent timing issue in implementations

## kperf (PMU timer) Investigation Results

### Fixed Issues
1. **`detects_early_exit_comparison`** - Function return value wasn't wrapped in `black_box`, causing compiler to optimize away the comparison. Fixed by adding `std::hint::black_box()`.

2. **`force_discrete_mode_activates`** - `force_discrete_mode` config option was stored but never used. Fixed by propagating it through `CalibrationConfig` and applying the override in `calibrate()`.

3. **Regularization fallback** - When covariance matrix had cond > 10⁶, the code fell back to identity matrix (pure OLS), which lost per-quantile variance structure. This caused tail effects to be underestimated because all quantiles were weighted equally. Fixed by using diagonal matrix (weighted OLS) instead, preserving variance structure. Spec updated in §3.4.5 and §B.3.

### Remaining Flaky Tests (kperf)

| Test | Failure Rate | Issue |
|------|--------------|-------|
| `effect_pattern_mixed` | ~25% | Expected Mixed, sometimes gets UniformShift. Tail component occasionally measured smaller than expected. |
| `effect_pattern_pure_uniform_shift` | ~5% | Expected UniformShift, sometimes gets Mixed. Small tail measurement triggers Mixed classification. |
| `dilithium3_verify_constant_time` | ~60-75% | False positive - detects ~159ns timing leak in Dilithium3 verification (pqcrypto). |
| `test_pmu_measurement_consistency` | ~5% | CV occasionally exceeds 0.5 threshold due to system noise. |

### Root Cause Analysis

**`effect_pattern_mixed`**: The test injects 100ns shift + 500ns spike (15% probability), expecting Mixed pattern. With kperf's high precision, the tail measurement can vary between runs. After the regularization fix, this passes ~75% of the time.

**`effect_pattern_pure_uniform_shift`**: The test injects a pure 2000ns shift with no tail effect. Occasionally the posterior draws show enough tail component (>10ns) to trigger Mixed classification under the draw-based dominance rules.

**`dilithium3_verify_constant_time`**: The pqcrypto Dilithium3 implementation shows a measurable timing difference (~159ns) with kperf's high resolution. This could be:
- A real timing side-channel in the PQClean implementation
- Measurement artifacts from FFI overhead
- Cache effects during verification
