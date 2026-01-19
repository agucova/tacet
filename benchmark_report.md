# Benchmark Results (quick)

## Configuration

- **Preset**: quick
- **Samples per class**: 1000
- **Datasets per point**: 3
- **Effect multipliers**: [0.0, 0.1]
- **Total execution time**: 0.1s

## False Positive Rate (effect = 0)

| Tool | iid |
|------|--------|
| timing-oracle | - |

## Power (Detection Rate)

| Tool | 0.1σ |
|------|------|
| timing-oracle | 0% |

## Detailed Results by Tool

### timing-oracle

| Pattern | Effect | Noise | Rate | 95% CI | Time (ms) |
|---------|--------|-------|------|--------|----------|
| shift | 0σ | iid | 0.0% | [0.0%, 56.2%] | 71 |
| shift | 0.1σ | iid | 0.0% | [0.0%, 56.2%] | 71 |

