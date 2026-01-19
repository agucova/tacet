# mbedTLS False Positive Diagnostic Report

Generated: 2026-01-18

## 1. Dataset Information

- **Source**: SILENT paper (Dunsche et al.)
- **File**: `Wrong_second_byte_(0x02_set_to_0x17)vsCorrectly_formatted_PKCS#1_PMS_message.csv`
- **Context**: mbedTLS Bleichenbacher timing attack (PKCS#1 padding oracle)
- **Ground Truth**: SILENT reports 'Failed to reject' → NO timing leak
- **Issue**: Our tool incorrectly reports FAIL (false positive)

### Raw Data Statistics

| Metric | Value |
|--------|-------|
| Total samples per class | 500000 |
| Data unit | CPU cycles |
| Assumed CPU frequency | 3.0 GHz |
| Baseline mean | 7500880.11 ns (7.50 ms) |
| Test mean | 7499785.06 ns (7.50 ms) |
| Raw mean difference | -1095.05 ns |
| Baseline std | 1429857.29 ns (1.43 ms) |
| Test std | 1432965.43 ns (1.43 ms) |
| Coefficient of variation | 19.1% |

### Analysis Subset

Using first 50000 samples per class for detailed analysis.

## 2. SILENT's Analysis Parameters

From `Wrong_second_byte_..._summary_results.json`:

```json
{
  "alpha": 0.1,
  "bootstrap_samples": 1000,
  "delta": 5,
  "block_size": 2122,
  "test_statistic": 2.303,
  "threshold": 4.296,
  "decision": "Failed to reject"
}
```

**Key observations**:
- SILENT uses `block_size=2122` for 500,000 samples
- Block ratio: 0.0042 (0.42% of data)
- Equivalent for 50000 samples: 212
- Threshold in nanoseconds: 1.67 ns (5 cycles @ 3GHz)

## 3. Autocorrelation Analysis

### Lag-k Autocorrelation

| Lag | Baseline ACF | Test ACF |
|-----|--------------|----------|
| 1 | 0.9185 | 0.9180 |
| 2 | 0.8662 | 0.8663 |
| 5 | 0.7631 | 0.7611 |
| 10 | 0.6689 | 0.6576 |
| 20 | 0.5555 | 0.5446 |
| 50 | 0.3807 | 0.3688 |
| 100 | 0.2448 | 0.2458 |
| 200 | 0.1690 | 0.1724 |
| 500 | 0.0592 | 0.0240 |
| 1000 | 0.0167 | 0.0191 |

**Interpretation**: High autocorrelation at large lags indicates strong temporal dependence.
SILENT's block_size=2122 suggests dependence extends ~2000+ samples.

### Decorrelation Length Estimates

| Metric | Baseline | Test |
|--------|----------|------|
| Lag-1 ACF | 0.9185 | 0.9180 |
| Last lag with |ACF| > 0.05 | 4783 | 4858 |
| Suggested block size | 9566 | 9716 |

## 4. Block Size Estimation

| Method | Block Size | Ratio to SILENT |
|--------|------------|-----------------|
| SILENT (scaled to 50k) | 212 | 1.00x |
| Our Politis-White | 671 | 3.17x |
| Simple (1.3×n^⅓) | 48 | 0.23x |
| Max P-W (3×√n) | 671 | 3.17x |

**Critical Issue**: Our block size is 0.3x smaller than SILENT's!
This underestimates variance by approximately the same factor.

## 5. Bootstrap Covariance Analysis

### Our Bootstrap Results (block_size=271)

#### Variance Estimates (diagonal of Σ)

| Quantile | Variance | Std Error | 95% CI Width |
|----------|----------|-----------|--------------|
| 10% | 5.30e10 | 230213.2 ns | 902435.8 ns |
| 20% | 3.87e10 | 196654.0 ns | 770883.5 ns |
| 30% | 1.51e10 | 122909.9 ns | 481806.8 ns |
| 40% | 3.98e9 | 63075.2 ns | 247254.7 ns |
| 50% | 2.33e8 | 15253.3 ns | 59793.1 ns |
| 60% | 3.50e5 | 591.4 ns | 2318.2 ns |
| 70% | 2.42e5 | 492.1 ns | 1928.9 ns |
| 80% | 2.18e5 | 466.4 ns | 1828.3 ns |
| 90% | 2.99e5 | 547.2 ns | 2145.1 ns |

#### Correlation Matrix (selected elements)

| | 10% | 50% | 90% |
|---|-----|-----|-----|
| 10% | 1.000 | 0.034 | 0.438 |
| 50% | 0.034 | 1.000 | 0.146 |
| 90% | 0.438 | 0.146 | 1.000 |

**Note**: High inter-quantile correlations (0.4-0.7) indicate dependent data structure.

## 6. Quantile Difference Analysis

### Raw Quantile Differences (Δ = baseline - test)

| Quantile | Baseline | Test | Δ (ns) | SE | Z-score |
|----------|----------|------|--------|-----|---------|
| 10% | 5437105 | 5467877 | -30772.0 | 230213.2 | -0.13 |
| 20% | 8140542 | 8143763 | -3221.8 | 196654.0 | -0.02 |
| 30% | 8162975 | 8163208 | -232.7 | 122909.9 | -0.00 |
| 40% | 8172485 | 8172939 | -453.3 | 63075.2 | -0.01 |
| 50% | 8180636 | 8180852 | -216.7 | 15253.3 | -0.01 |
| 60% | 8188210 | 8188296 | -85.7 | 591.4 | -0.14 |
| 70% | 8195936 | 8196181 | -245.0 | 492.1 | -0.50 |
| 80% | 8205364 | 8205450 | -85.8 | 466.4 | -0.18 |
| 90% | 8219538 | 8219917 | -379.7 | 547.2 | -0.69 |

**Interpretation**:
- The 10% quantile shows a large difference (-30772 ns)
- But with SE=230213 ns, Z-score=-0.13
- Most quantiles have |Z| < 2, suggesting no significant effect

## 7. Gibbs Sampler Analysis

### θ = 1.67 ns

| Metric | Value |
|--------|-------|
| Leak probability | 100.0% |
| Decision | FAIL |
| Expected (SILENT) | PASS |
| Shift estimate | 0.00 ns |
| Tail estimate | 0.00 ns |
| 95% CI | [43.1, 17265.8] ns |
| Quality | TooNoisy |

### θ = 5 ns

| Metric | Value |
|--------|-------|
| Leak probability | 100.0% |
| Decision | FAIL |
| Expected (SILENT) | PASS |
| Shift estimate | 0.00 ns |
| Tail estimate | 0.00 ns |
| 95% CI | [56.9, 17689.8] ns |
| Quality | TooNoisy |

### θ = 100 ns

| Metric | Value |
|--------|-------|
| Leak probability | 100.0% |
| Decision | FAIL |
| Expected (SILENT) | PASS |
| Shift estimate | 0.00 ns |
| Tail estimate | 0.00 ns |
| 95% CI | [971.6, 19148.2] ns |
| Quality | TooNoisy |

### θ = 1000 ns

| Metric | Value |
|--------|-------|
| Leak probability | 100.0% |
| Decision | FAIL |
| Expected (SILENT) | PASS |
| Shift estimate | 0.00 ns |
| Tail estimate | 0.00 ns |
| 95% CI | [1528.6, 20826.2] ns |
| Quality | TooNoisy |

## 8. Variance Inflation Analysis

### Effective Sample Size by Block Size

| Block Size | Source | Effective n | Variance Multiplier |
|------------|--------|-------------|---------------------|
| 671 | Our P-W | 74 | 671.0x |
| 48 | Simple | 1041 | 48.0x |
| 212 | SILENT equiv | 235 | 212.0x |
| 9716 | ACF-based | 5 | 9716.0x |

**Key insight**: Using block_size=671 instead of 212 means our variance
estimates are ~0.3x too small, making signals appear 0.6x more significant.

## 9. Recommendations

### Potential Fixes

1. **Allow manual block size override** in `SinglePassConfig`
2. **Improve ACF-based block estimation** for extreme autocorrelation
3. **Add diagnostic warning** when computed block << decorrelation length
4. **Use conservative (larger) block sizes** as a safety margin

### Proposed Block Size Formula

```
// Current: Politis-White with max = 3√n
// Proposed: max(P-W, ACF-based, conservative_floor)
let acf_block = estimate_decorrelation_length(data) * 2;
let conservative_floor = (n as f64).sqrt() as usize; // √n
let block = pw_block.max(acf_block).max(conservative_floor);
```

### Data Quality Indicators for This Dataset

| Indicator | Baseline | Test | Concern? |
|-----------|----------|------|----------|
| Unique values | 46% | 46% | No |
| Lag-1 ACF | 0.918 | 0.918 | Yes (correlated) |
| CV | 19.1% | 19.1% | No |

---

*Report generated by timing-oracle mbedTLS diagnostic*

## 10. Test Suite Failures Analysis

### Overview

The full test suite shows **6 unexpected failures** in tests that should PASS (constant-time implementations):

| Test | Module | Expected | Actual |
|------|--------|----------|--------|
| `aes128_block_encrypt_constant_time` | aes_timing | PASS | FAIL |
| `aes128_different_keys_constant_time` | aes_timing | PASS | FAIL |
| `aes128_hamming_weight_independence` | aes_timing | PASS | FAIL |
| `ring_chacha20poly1305_constant_time` | aead_timing | PASS | FAIL |
| `async_block_on_overhead_symmetric` | async_timing | PASS | FAIL |
| `blake2b_512_constant_time` | hash_timing | PASS | FAIL |

### Root Cause: Incomplete Migration

**Critical Finding**: The test failures are from the **OLD v5.2 mixture prior**, not the new v5.4 Gibbs sampler!

```
Code Path Analysis:
├── single_pass.rs     → compute_bayes_gibbs()     ✓ Updated to v5.4
├── loop_runner.rs     → compute_bayes_factor_mixture()  ✗ Still v5.2!
└── step.rs (core)     → compute_bayes_factor()    ✗ Still old method!
```

The integration tests use `TimingOracle::for_attacker().test()` which routes through `loop_runner.rs`, which still uses the old mixture prior.

### Evidence from Debug Output

Test output shows old mixture prior debug messages:
```
[DEBUG bayes] log_ml_narrow = -52.89, log_ml_slab = -84.47
[DEBUG bayes] log_w1 (narrow) = -52.90, log_w2 (slab) = -89.08
```

These `log_ml_narrow` / `log_ml_slab` messages are from `compute_bayes_factor_mixture()`, not the new Gibbs sampler.

### Files Requiring Migration

| File | Current | Required |
|------|---------|----------|
| `timing-oracle/src/adaptive/loop_runner.rs:347` | `compute_bayes_factor_mixture()` | `compute_bayes_gibbs()` |
| `timing-oracle-core/src/adaptive/step.rs:405` | `compute_bayes_factor()` | `compute_bayes_gibbs()` |

### Implications

1. **The v5.4 Gibbs migration is incomplete** - only `single_pass.rs` was updated
2. **Test failures are from v5.2 code** - the old mixture prior that had the correlation issue
3. **Two separate issues**:
   - mbedTLS false positive (affects both v5.2 and v5.4 - block size estimation)
   - Test suite failures (v5.2 mixture prior being too aggressive)

### Sample Failure Details

#### blake2b_512_constant_time
```
Samples: 7000 per class
Quality: Good
Probability of leak: 100.0%
Effect: 270.9 ns Mixed
  Shift: -84.6 ns
  Tail:  -257.3 ns
  95% CI: 155.2–386.5 ns
```

This is detecting a ~270ns "effect" in BLAKE2b which is a constant-time implementation. The old mixture prior is producing false positives.

### Recommendations

1. **Complete the v5.4 migration** by updating:
   - `loop_runner.rs` to use `compute_bayes_gibbs()`
   - `step.rs` to use `compute_bayes_gibbs()`
   
2. **Re-run tests after migration** to see if Gibbs sampler fixes these false positives

3. **The mbedTLS issue is separate** - it's about block size estimation for highly autocorrelated external data, which affects both v5.2 and v5.4

---

## 11. Summary

### Two Distinct Issues

| Issue | Cause | Affects | Fix |
|-------|-------|---------|-----|
| **Test suite failures (6 FPs)** | Incomplete v5.4 migration; tests still use v5.2 mixture prior | Integration tests | Complete migration to Gibbs sampler |
| **mbedTLS false positive** | Block size underestimation for extreme autocorrelation | Both v5.2 and v5.4 | Improve block size estimation |

### Migration Status

```
v5.2 → v5.4 Migration Progress:
[✓] Phase 1: Create gibbs.rs module
[✓] Phase 2: Add t-prior calibration
[✓] Phase 3: Add compute_bayes_gibbs()
[✓] Phase 4: Update quality gates
[✓] Phase 5: Update Posterior struct
[~] Phase 6: Update wrappers (INCOMPLETE - only single_pass.rs done)
[ ] Phase 7: Validate (blocked by Phase 6)
```

### Next Steps

1. Update `loop_runner.rs` to use Gibbs sampler
2. Update `step.rs` to use Gibbs sampler  
3. Re-run full test suite
4. If tests pass, then investigate mbedTLS block size issue separately

---

*Report extended with test failure analysis*
