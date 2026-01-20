# Benchmark Results (quick)

## Configuration

- **Preset**: quick
- **Samples per class**: 10000
- **Datasets per point**: 10
- **Effect multipliers**: [0.0, 0.5, 1.0, 2.0]
- **Total execution time**: 4.9s

## False Positive Rate (effect = 0)

| Tool | iid |
|------|--------|
| timing-oracle | - |

## Power (Detection Rate)

| Tool | 0.5σ | 1σ | 2σ |
|------|------|------|------|
| timing-oracle | 100% | 100% | 100% |

## Detailed Results by Tool

### timing-oracle

| Pattern | Effect | Noise | Rate | 95% CI | Time (ms) |
|---------|--------|-------|------|--------|----------|
| shift | 0σ | iid | 0.0% | [0.0%, 27.8%] | 1032 |
| shift | 0.5σ | iid | 100.0% | [72.2%, 100.0%] | 750 |
| shift | 1σ | iid | 100.0% | [72.2%, 100.0%] | 788 |
| shift | 2σ | iid | 100.0% | [72.2%, 100.0%] | 860 |
| tail | 0σ | iid | 0.0% | [0.0%, 27.8%] | 941 |
| tail | 0.5σ | iid | 100.0% | [72.2%, 100.0%] | 925 |
| tail | 1σ | iid | 100.0% | [72.2%, 100.0%] | 818 |
| tail | 2σ | iid | 100.0% | [72.2%, 100.0%] | 928 |

