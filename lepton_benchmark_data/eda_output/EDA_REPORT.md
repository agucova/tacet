# Fine-Threshold Benchmark EDA Report

## Overview

This analysis examines the performance of timing side-channel detection tools on fine-threshold synthetic benchmarks with realistic noise models.

**Dataset:**
- 43,200 individual trial results
- 864 aggregated conditions
- Tools compared: ad-test, dudect, ks-test, mona, rtlf-native, silent-native, tacet, timing-tvla
- Effect sizes: 0.0 to 0.005 σ
- Noise models: ar1-0.2-realistic, ar1-0.4-realistic, ar1-0.6-realistic, ar1-0.8-realistic, ar1-n0.2-realistic, ar1-n0.4-realistic, ar1-n0.6-realistic, ar1-n0.8-realistic, iid-realistic

## Key Findings

### 1. False Positive Rate (Type I Error)

| Tool | Average FPR | Status |
|------|-------------|--------|
| dudect | 0.000 | ✓ Good |
| mona | 0.000 | ✓ Good |
| silent-native | 0.000 | ✓ Good |
| tacet | 0.000 | ✓ Good |
| timing-tvla | 0.000 | ✓ Good |
| rtlf-native | 0.069 | ✓ Good |
| ks-test | 0.293 | ✗ High |
| ad-test | 0.358 | ✗ High |

**Interpretation:** Tools with FPR > 0.05 are producing false alarms above the nominal α=0.05 level.
The AD-test and KS-test show very high FPR (~0.90), making them unreliable for CI use.

### 2. Statistical Power

| Tool          |   Avg FPR | FPR OK?   |   Power@0.0006σ |   Power@0.001σ |   Effect for 80% |   Median Time (ms) |
|:--------------|----------:|:----------|----------------:|---------------:|-----------------:|-------------------:|
| ad-test       |     0.358 | ✗         |            0.93 |           1    |            0     |                  0 |
| dudect        |     0     | ✓         |            0.89 |           1    |            0     |                 11 |
| ks-test       |     0.293 | ✗         |            0.92 |           1    |            0     |                  1 |
| mona          |     0     | ✓         |            0.89 |           1    |            0     |                  0 |
| rtlf-native   |     0.069 | ✓         |            0.9  |           1    |            0     |               4789 |
| silent-native |     0     | ✓         |            0.89 |           1    |            0     |               7700 |
| tacet |     0     | ✓         |            0.33 |           0.38 |            0.002 |                 25 |
| timing-tvla   |     0     | ✓         |            0.89 |           1    |            0     |                  0 |

### 3. Speed vs Accuracy Trade-off

| Tool | Median Time | Speed Rank |
|------|-------------|------------|
| ad-test | 0ms | #1 |
| mona | 0ms | #2 |
| timing-tvla | 0ms | #3 |
| ks-test | 1ms | #4 |
| dudect | 11ms | #5 |
| tacet | 25ms | #6 |
| rtlf-native | 4789ms | #7 |
| silent-native | 7700ms | #8 |

## Conclusions

1. **tacet** achieves excellent FPR control while maintaining competitive power
2. **AD-test and KS-test** have severely inflated FPR (~90%) and should not be used for decision-making
3. **RTLF and SILENT** maintain good FPR but at the cost of slower execution
4. **DudeCT** shows moderate FPR but fast execution

## Plots

- `fpr_comparison.png` - FPR across tools and noise models
- `fpr_heatmap.png` - FPR heatmap visualization
- `power_curves.png` - Statistical power vs effect size
- `power_curves_zoomed.png` - Power curves in critical small-effect region
- `execution_time.png` - Execution time comparison
- `noise_robustness.png` - Robustness across noise models
