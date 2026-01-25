# Statistical Validation Suite

This document describes the statistical validation suite for `tacet`, which rigorously validates the library's false positive rate (FPR), statistical power, Bayesian calibration, effect estimation accuracy, and coverage properties.

## Overview

The validation suite is organized into test tiers with different runtime/thoroughness tradeoffs:

| Tier | Runtime | Trials | Use Case |
|------|---------|--------|----------|
| **Iteration** | ~30 min | 50-100 | Quick iteration while developing |
| **Quick** | ~1-2 hours | 100-200 | PR checks, CI |
| **Full** | ~2-3 hours | 200-500 | Weekly validation |
| **Validation** | ~4+ hours | 500-1000 | Pre-release validation |

## Running the Validation Suite

### Quick Start (Iteration Tier)

For fast feedback during development:

```bash
# Set environment variables
export CALIBRATION_TIER=iteration
export CALIBRATION_DATA_DIR=./calibration_data

# Run the core tests
cargo test --release \
  --test calibration_fpr \
  --test calibration_power \
  --test calibration_bayesian \
  --test calibration_estimation \
  --test calibration_coverage

# Generate plots
uv run crates/tacet/scripts/plot_calibration.py ./calibration_data --output ./plots
```

### Full Validation

For comprehensive validation (run weekly or before releases):

```bash
# Set environment variables
export CALIBRATION_TIER=validation
export CALIBRATION_DATA_DIR=./calibration_data

# Run all tests including ignored validation tests
cargo test --release \
  --test calibration_fpr \
  --test calibration_power \
  --test calibration_bayesian \
  --test calibration_estimation \
  --test calibration_coverage \
  -- --ignored

# Generate plots
uv run crates/tacet/scripts/plot_calibration.py ./calibration_data --output ./plots
```

## Test Categories

### 1. False Positive Rate (FPR)

**Purpose:** Verify that the oracle does not report false positives when there is no timing difference.

**Files:** `calibration_fpr.rs`

**Method:**
- Test with identical inputs (random vs random, fixed vs fixed)
- Count the fraction of trials that incorrectly report a timing leak

**Acceptance Criteria:**
| Tier | Max FPR |
|------|---------|
| Iteration | ≤ 15% |
| Quick | ≤ 10% |
| Full | ≤ 7% |
| Validation | ≤ 5% |

### 2. Statistical Power

**Purpose:** Verify that the oracle detects timing differences when they exist.

**Files:** `calibration_power.rs`

**Method:**
- Inject known timing differences at various effect sizes (0.5θ, θ, 2θ, 5θ, 10θ)
- Count the fraction of trials that correctly detect the timing leak

**Effect Sizes Tested:**

| Model | θ | 0.5θ | θ | 2θ | 5θ | 10θ |
|-------|---|------|---|----|----|-----|
| Research | 50ns | 25ns | 50ns | 100ns | 250ns | 500ns |
| AdjacentNetwork | 100ns | 50ns | 100ns | 200ns | 500ns | 1000ns |
| RemoteNetwork | 50μs | 25μs | 50μs | 100μs | 250μs | 500μs |
| PostQuantumSentinel | 3.3ns | - | - | 10ns | 33ns | 100ns |
| SharedHardware | 0.6ns | - | - | 6ns | 30ns | 60ns |

**Acceptance Criteria:**
| Effect Size | Min Power (Quick) | Min Power (Validation) |
|-------------|-------------------|------------------------|
| 2×θ | ≥ 70% | ≥ 80% |
| 5×θ | ≥ 90% | ≥ 95% |
| 10×θ | ≥ 95% | ≥ 99% |

### 3. Bayesian Calibration

**Purpose:** Verify that stated probabilities match empirical frequencies (when oracle reports P(leak) = X%, approximately X% of those cases should be true positives).

**Files:** `calibration_bayesian.rs`

**Method:**
- Run trials at various true effect sizes (0, θ, 2θ)
- Bin trials by reported P(leak) into deciles
- For each bin, compute empirical true positive rate
- Compare stated probability to empirical rate

**Acceptance Criteria:**
| Tier | Max Mean Calibration Error | Max Single Bin Deviation |
|------|----------------------------|--------------------------|
| Iteration | ≤ 20% | ≤ 30% |
| Quick | ≤ 15% | ≤ 25% |
| Validation | ≤ 10% | ≤ 20% |

### 4. Effect Estimation Accuracy

**Purpose:** Verify that the oracle's effect estimates (shift_ns, tail_ns) accurately reflect the true injected timing differences.

**Files:** `calibration_estimation.rs`

**Method:**
- Inject known delays: 50ns, 100ns, 200ns, 500ns, 1μs
- Record estimated effect for each trial
- Compute bias (mean - true) and RMSE

**Acceptance Criteria:**
| Effect Size | Max Bias | Max RMSE |
|-------------|----------|----------|
| ≥ 2θ | ≤ 20% | ≤ 50% |
| ≥ 5θ | ≤ 15% | ≤ 40% |

### 5. Coverage Calibration

**Purpose:** Verify that 95% credible intervals contain the true value approximately 95% of the time.

**Files:** `calibration_coverage.rs`

**Method:**
- Inject known effects at various sizes
- Check if the reported CI contains the true injected value
- Compute coverage rate per effect size

**Acceptance Criteria:**
| Tier | Min Coverage |
|------|--------------|
| Iteration | ≥ 80% |
| Quick | ≥ 85% |
| Validation | ≥ 90% |

## Output Files

After running the validation suite with `CALIBRATION_DATA_DIR` set, CSV files are generated for each test. Running the plotting script generates:

| File | Description |
|------|-------------|
| `power_curve.png` | Power vs effect size (all models combined) |
| `power_curves_faceted.png` | Power curves, one panel per attacker model |
| `fpr_calibration.png` | FPR with confidence intervals |
| `bayesian_calibration.png` | Stated P(leak) vs empirical true positive rate |
| `coverage_calibration.png` | CI coverage by effect size |
| `effect_estimation.png` | Estimated vs true effect scatter plot |
| `estimation_bias.png` | Bias and RMSE by effect size |
| `summary.txt` | Text summary of all key metrics |

## Interpreting Results

### Power Curve

A healthy power curve should:
- Show increasing power with effect size (monotonic)
- Reach ≥70% power at 2×θ
- Reach ≥90% power at 5×θ
- Approach 100% power at 10×θ

### Bayesian Calibration Curve

A well-calibrated system should show points close to the diagonal identity line. Deviations indicate:
- **Points above diagonal:** System is underconfident (reports lower probability than warranted)
- **Points below diagonal:** System is overconfident (reports higher probability than actual)

### Coverage Plot

Bars should cluster around 95% (the nominal CI level). Coverage below 85% indicates the CIs are too narrow; coverage above 98% indicates the CIs are too conservative.

### Estimation Bias Plot

Points should cluster around 0% bias. The ±20% dashed lines indicate acceptable bias thresholds. RMSE annotations show estimation variance.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CALIBRATION_TIER` | Test tier: `iteration`, `quick`, `full`, `validation` | `quick` |
| `CALIBRATION_DATA_DIR` | Directory for CSV output | None (disabled) |
| `CALIBRATION_DISABLED` | Set to `1` to skip all calibration tests | None |
| `CALIBRATION_SEED` | Random seed for reproducibility | Current time |
| `TIMING_ATTACKER_MODEL` | Override attacker model for tests | Per-test default |

## Attacker Model Notes

Some attacker models require high-precision timers:

| Model | θ | PMU Required | Notes |
|-------|---|--------------|-------|
| SharedHardware | 0.6ns | Yes | Skip on coarse timers |
| PostQuantumSentinel | 3.3ns | Yes | Skip on coarse timers |
| AdjacentNetwork | 100ns | No | Primary validation target |
| RemoteNetwork | 50μs | No | Large delays, always measurable |
| Research | ~0 | No | For debugging, not CI |

PMU tests are marked as `#[ignore]` and require:
- **macOS:** `sudo` for kperf access
- **Linux:** `sudo` or `CAP_PERFMON` capability

## CI Integration

### GitHub Actions Example

```yaml
jobs:
  validation-quick:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run quick validation
        env:
          CALIBRATION_TIER: quick
          CALIBRATION_DATA_DIR: ./calibration_data
        run: |
          cargo test --release \
            --test calibration_fpr \
            --test calibration_power \
            --test calibration_coverage
      - uses: actions/upload-artifact@v4
        with:
          name: calibration-data
          path: calibration_data/

  validation-full:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'  # Weekly
    steps:
      - uses: actions/checkout@v4
      - name: Run full validation
        env:
          CALIBRATION_TIER: validation
          CALIBRATION_DATA_DIR: ./calibration_data
        run: |
          cargo test --release \
            --test calibration_fpr \
            --test calibration_power \
            --test calibration_bayesian \
            --test calibration_estimation \
            --test calibration_coverage \
            -- --ignored
```

## Troubleshooting

### Tests Skipped Due to Unmeasurable

If many tests show as unmeasurable:
1. Ensure you're running on bare metal (not in a VM or container with constrained resources)
2. Check that CPU frequency scaling is disabled (`cpupower frequency-set -g performance`)
3. Try using the `--release` flag for faster operations

### Low Power at Expected Effect Sizes

Possible causes:
1. High system noise (background processes)
2. Timer resolution insufficient for the effect size
3. Incorrect attacker model for the scenario

### Calibration Tests Timing Out

Increase `max_wall_ms` in the test configuration or use a faster tier.
