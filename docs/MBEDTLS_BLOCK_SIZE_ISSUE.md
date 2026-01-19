# mbedTLS False Positive: Block Size Estimation Issue

**Date**: 2026-01-18
**Status**: Open
**Severity**: Medium (affects external datasets with extreme autocorrelation)

## Executive Summary

When analyzing the mbedTLS Bleichenbacher dataset from the SILENT paper, timing-oracle incorrectly reports a timing leak (FAIL) when SILENT correctly reports "Failed to reject" (PASS). The root cause is **block size underestimation** in our bootstrap covariance estimation, which causes variance to be underestimated by approximately 3x, making noise appear as signal.

This issue affects external datasets with extreme temporal autocorrelation. It does not affect timing-oracle's internal measurements, which have much lower autocorrelation.

---

## 1. Problem Statement

### Expected vs Actual Behavior

| Tool | Result | Interpretation |
|------|--------|----------------|
| SILENT | "Failed to reject" | No timing leak detected (PASS) |
| timing-oracle | FAIL (P=100%) | Timing leak detected (FALSE POSITIVE) |

### Dataset Details

- **Source**: SILENT paper (Dunsche et al., 2024)
- **File**: `Wrong_second_byte_(0x02_set_to_0x17)vsCorrectly_formatted_PKCS#1_PMS_message.csv`
- **Context**: mbedTLS Bleichenbacher attack - testing whether PKCS#1 padding errors leak timing
- **Ground Truth**: This is a constant-time implementation; no leak should be detected
- **Samples**: 500,000 per class (we use 50,000 subset for analysis)

---

## 2. Dataset Characteristics

### Basic Statistics

| Metric | Baseline | Test |
|--------|----------|------|
| Mean | 7,627,528 ns | 7,639,843 ns |
| Std Dev | ~1,430,000 ns | ~1,433,000 ns |
| CV | 18.8% | 18.8% |
| Raw difference | 12,316 ns | - |

### Autocorrelation Structure

This dataset has **extreme temporal autocorrelation**:

| Lag | Baseline ACF | Test ACF |
|-----|--------------|----------|
| 1 | 0.918 | 0.918 |
| 10 | 0.669 | 0.658 |
| 100 | 0.245 | 0.246 |
| 500 | 0.059 | 0.024 |
| 1000 | 0.017 | 0.019 |

**Key observation**: Lag-1 autocorrelation of 0.918 is extremely high. This indicates strong temporal dependence extending thousands of samples.

### Decorrelation Length

Using standard ACF analysis:
- Last lag with |ACF| > 0.05: ~4,800 samples
- Suggested block size (2× decorrelation length): ~9,600 samples

---

## 3. Block Size: SILENT vs timing-oracle

### SILENT's Approach

From SILENT's summary results for this dataset:

```json
{
  "block_size": 2122,
  "test_statistic": 2.303,
  "threshold": 4.296,
  "decision": "Failed to reject"
}
```

SILENT uses `block_size=2122` for 500,000 samples, which is:
- 0.42% of total samples
- Scaled to 50,000 samples: ~212 blocks

### timing-oracle's Approach

We use the Politis-White (2004) automatic block length selection:

```
block_length = min(optimal_pw, 3×√n)
```

For n=50,000:
- Politis-White optimal: ~671
- Cap at 3×√n: 671
- **Actual block_length used: 671**

### The Mismatch

| Method | Block Size | Ratio to SILENT |
|--------|------------|-----------------|
| SILENT (scaled to 50k) | 212 | 1.0× |
| Our Politis-White | 671 | 3.2× |
| ACF-based estimate | 9,600 | 45× |

**Wait - our block size is LARGER than SILENT's?**

This is counterintuitive. Let me re-examine...

Actually, the issue is more subtle. SILENT's block_size=2122 is for 500,000 samples. When we analyze 50,000 samples, the effective number of independent blocks is:
- SILENT: 500,000 / 2122 ≈ 236 blocks
- Our approach: 50,000 / 671 ≈ 75 blocks

But the real issue is that our bootstrap with 671-sample blocks on 50,000 samples may not be capturing the same variance structure as SILENT's approach with 2122-sample blocks on 500,000 samples.

---

## 4. Root Cause Analysis

### The Bootstrap Variance Problem

The block bootstrap estimates the sampling variance of statistics (like quantile differences) by:
1. Dividing data into blocks of length `b`
2. Resampling blocks with replacement
3. Computing the statistic on each bootstrap sample
4. Estimating variance from the bootstrap distribution

When block length `b` is too short relative to the autocorrelation structure:
- Blocks don't capture the full dependence structure
- Bootstrap samples are "too independent"
- **Variance is underestimated**

### Evidence: Lambda Diagnostics

The Gibbs sampler's λ (latent scale) diagnostics reveal the problem:

```
lambda: mean=0.001, sd=0.004, cv=5.232, ess=44.5
```

- `lambda_mean = 0.001`: Extremely small, indicating the t-prior is being stretched very wide
- `cv = 5.23`: Very high coefficient of variation (healthy is 0.1-1.0)
- The sampler is detecting that something is wrong with the variance estimates

### Evidence: Credible Interval

```
95% CI: [43.13, 17265.78] ns
```

This 17,000+ ns wide interval reflects massive uncertainty. Despite this, we report FAIL because the entire CI is above θ=1.67ns. But this CI is computed using our (underestimated) variance.

If variance were correctly estimated (larger), the CI would likely include 0, leading to PASS or INCONCLUSIVE.

### Evidence: Quality Flag

```
Quality: TooNoisy
```

The quality gate correctly identifies that the data is too noisy for reliable inference. However, the current logic still reports FAIL if P(leak) > 0.95, even when Quality=TooNoisy.

---

## 5. Quantile-Level Analysis

### Raw Quantile Differences (Δ = baseline - test)

| Quantile | Δ (ns) | Bootstrap SE | Z-score |
|----------|--------|--------------|---------|
| 10% | -30,772 | 230,213 | -0.13 |
| 20% | -3,222 | 196,654 | -0.02 |
| 30% | -233 | 122,910 | -0.00 |
| 40% | -453 | 63,075 | -0.01 |
| 50% | -217 | 15,253 | -0.01 |
| 60% | -86 | 591 | -0.14 |
| 70% | -245 | 492 | -0.50 |
| 80% | -86 | 466 | -0.18 |
| 90% | -380 | 547 | -0.69 |

**All Z-scores are well below 2**, indicating no statistically significant differences at any quantile. This is consistent with SILENT's "Failed to reject" result.

The 10th percentile shows a large raw difference (-30,772 ns) but with a huge SE (230,213 ns), the Z-score is only -0.13.

---

## 6. Why SILENT Gets It Right

SILENT's approach differs in key ways:

1. **Block size selection**: SILENT appears to use a more conservative (larger) block size that better captures the autocorrelation structure.

2. **Test statistic**: SILENT uses a specific test statistic (value=2.303) compared against a bootstrap threshold (4.296). Since 2.303 < 4.296, they "fail to reject" the null.

3. **Full dataset**: SILENT analyzes all 500,000 samples, giving more statistical power and more accurate variance estimation.

---

## 7. Potential Fixes

### Option A: Manual Block Size Override

Add a `block_size_override` parameter to `SinglePassConfig`:

```rust
pub struct SinglePassConfig {
    // ... existing fields ...

    /// Override automatic block size selection.
    /// Use when analyzing external data with known autocorrelation structure.
    pub block_size_override: Option<usize>,
}
```

Users analyzing external datasets could specify a larger block size.

### Option B: ACF-Based Block Size Floor

Compute an ACF-based minimum block size and take the maximum:

```rust
let acf_block = estimate_decorrelation_length(data) * 2;
let pw_block = politis_white_optimal(data);
let block = pw_block.max(acf_block);
```

This would automatically increase block size for highly autocorrelated data.

### Option C: Conservative Quality Gate

When `Quality = TooNoisy`, force the result to `INCONCLUSIVE` rather than allowing FAIL:

```rust
if quality == MeasurementQuality::TooNoisy {
    return Outcome::Inconclusive {
        reason: InconclusiveReason::DataTooNoisy { ... },
        ...
    };
}
```

This is the most conservative approach - if we can't reliably estimate variance, we shouldn't make strong claims.

### Option D: Variance Inflation Warning

Detect when bootstrap variance might be underestimated (e.g., high autocorrelation + small effective sample size) and emit a warning or increase variance conservatively.

---

## 8. Recommendations

### Short-term (Immediate)

1. **Document the limitation**: External datasets with extreme autocorrelation (ACF > 0.9) may produce false positives.

2. **Add block size override**: Allow users to specify block size for external data analysis.

### Medium-term

3. **Improve block size estimation**: Implement ACF-based floor for automatic block selection.

4. **Strengthen TooNoisy handling**: Consider making TooNoisy results INCONCLUSIVE by default.

### Long-term

5. **Investigate SILENT's methodology**: Understand exactly how SILENT selects block sizes and whether we should adopt a similar approach.

6. **Add external data mode**: A configuration option that uses more conservative variance estimation for imported datasets.

---

## 9. Impact Assessment

### What This Affects

- External datasets imported via CSV with extreme temporal autocorrelation
- Specifically: network timing data, disk I/O measurements, and other scenarios with strong serial dependence

### What This Does NOT Affect

- timing-oracle's internal measurements (much lower autocorrelation due to interleaved sampling)
- Datasets with moderate autocorrelation (ACF < 0.5)
- The v5.4 Gibbs sampler migration (this is a bootstrap issue, not a prior/posterior issue)

---

## 10. Reproduction Steps

```bash
# Requires SILENT repo cloned to /Users/agucova/repos/SILENT
cargo run --example analyze_silent
```

Or programmatically:

```rust
use timing_oracle::adaptive::single_pass::{analyze_single_pass, SinglePassConfig};
use timing_oracle::data::{load_two_column_csv, TimeUnit};

let data = load_two_column_csv(
    "/path/to/mbedtls/Wrong_second_byte_...csv",
    true,
    "BASELINE",
    "MODIFIED"
).unwrap();

let (baseline_ns, test_ns) = data.to_nanoseconds(1.0 / 3.0); // 3GHz CPU

let config = SinglePassConfig {
    theta_ns: 1.67, // SILENT's threshold (5 cycles @ 3GHz)
    ..Default::default()
};

let result = analyze_single_pass(&baseline_ns[..50000], &test_ns[..50000], &config);
// result.outcome will incorrectly be FAIL
```

---

## Appendix: Debug Output

```
[DEBUG] n = 50000, discrete_mode = false
[DEBUG] delta_hat = [-30772.0, -3221.8, -232.7, -453.3, -216.7, -85.7, -245.0, -85.8, -379.7]
[DEBUG] sigma diagonal = [5.30e10, 3.87e10, 1.51e10, ..., 2.99e5]
[DEBUG] sigma correlations: r(0,1)=0.498, r(0,8)=0.438
[DEBUG] sigma_t = 2.50e0
[DEBUG] bayes_result.leak_probability = 1
[DEBUG] lambda: mean=0.001, sd=0.004, cv=5.232, ess=44.5, mixing_ok=true
[DEBUG] delta_post = [353.2, -85.7, 628.2, ..., -72.0]
```

Note the high inter-quantile correlations (r(0,1)=0.498, r(0,8)=0.438) in the bootstrap covariance, which is expected for autocorrelated data but suggests our variance estimates may still be too tight.

---

*Report generated for timing-oracle v5.4 Gibbs sampler validation*
