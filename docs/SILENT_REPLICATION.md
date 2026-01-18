# SILENT Paper Replication Plan

## Overview
This document outlines how to replicate the experiments from the SILENT paper
("A New Lens on Statistics in Software Timing Side Channels") using timing-oracle.

## 1. Synthetic Autocorrelation Experiments ✅ DONE

Our `calibration_autocorrelation.rs` tests already generate AR(1) correlated data
and produce comparable heatmaps. Results in `calibration_data_remote/`.

## 2. Real-World Dataset Experiments

### 2a. Data Format Adapter
SILENT uses CSV format:
```csv
V1,V2
X,1067424
Y,531296
...
```

We need a utility to convert this to timing-oracle's InputPair format.

### 2b. KyberSlash Dataset
**File**: `~/repos/SILENT/examples/paper-kyberslash/timing_measurements_1_comparison_-72_72.csv`
**Expected**: ~20 cycle timing difference (FAIL expected)
**SILENT params**: α=0.1, Δ varies (5-40 cycles)

To replicate:
```rust
// Load CSV data as two groups
let (x_samples, y_samples) = load_silent_csv("timing_measurements_1_comparison_-72_72.csv");

// Run timing-oracle with comparable threshold
// SILENT Δ=5 cycles ≈ ~1.7ns at 3GHz
let outcome = TimingOracle::for_attacker(AttackerModel::Custom { threshold_ns: 1.7 })
    .from_raw_samples(x_samples, y_samples);
```

### 2c. mbedTLS Bleichenbacher Dataset  
**File**: `~/repos/SILENT/examples/paper-mbedtls/Wrong_second_byte_...csv`
**Expected**: No timing leak (PASS expected - this was a false positive in RTLF)
**SILENT params**: α=0.1, Δ=5, B=1000

### 2d. Web Application Dataset
**File**: `~/repos/SILENT/examples/paper-web-application/web-diff-lan-10k.csv`
**Expected**: Timing leak detected (FAIL expected)

## 3. Implementation Tasks

### Task 1: CSV Data Loader
Add a utility to load SILENT-format CSV files:
```rust
pub fn load_silent_csv(path: &Path) -> (Vec<u64>, Vec<u64>) {
    // Parse V1,V2 format
    // Split by group label (X vs Y)
    // Return two sample vectors
}
```

### Task 2: Raw Sample Analysis Mode
Add ability to analyze pre-collected samples instead of running measurements:
```rust
impl TimingOracle {
    pub fn from_raw_samples(
        baseline: Vec<u64>,
        sample: Vec<u64>,
    ) -> Outcome {
        // Skip measurement phase
        // Go directly to statistical analysis
    }
}
```

### Task 3: Comparison Script
Create a script that:
1. Runs SILENT on each dataset
2. Runs timing-oracle on each dataset  
3. Compares decisions and metrics

## 4. Parameter Mapping

| SILENT | timing-oracle |
|--------|---------------|
| α (false positive rate) | pass_threshold (default 0.05) |
| Δ (min detectable effect) | threshold_ns (AttackerModel) |
| B (bootstrap samples) | Internal (2000 default) |
| n (sample size) | max_samples |

## 5. Expected Results Comparison

| Dataset | SILENT Result | Expected timing-oracle |
|---------|---------------|------------------------|
| KyberSlash | Rejected (leak) | FAIL (leak detected) |
| mbedTLS | Not rejected | PASS (no leak) |
| Web App | Rejected (leak) | FAIL (leak detected) |
