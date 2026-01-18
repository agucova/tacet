# Implementation Plan: CSV Loader and Raw Sample Analysis Mode

## Overview

This document outlines the implementation plan for adding raw sample analysis capabilities to timing-oracle, enabling analysis of pre-collected timing measurements (e.g., from SILENT paper datasets).

## Current Architecture

The existing data flow is:

```
InputPair<T> + operation closure
    ↓
Collector.collect() → Vec<Sample> (cycles)
    ↓
calibrate(baseline: &[u64], sample: &[u64], ns_per_tick, config)
    ↓
AdaptiveState (accumulates samples)
    ↓
run_adaptive(calibration, state, ns_per_tick, config)
    ↓
Outcome
```

**Key insight**: The `calibrate()` function already accepts raw `&[u64]` sample arrays. The infrastructure for raw sample analysis exists - we just need a clean API.

## Implementation Tasks

### Task 1: CSV Data Loader Module

**Location**: `crates/timing-oracle/src/data/mod.rs` (new module)

**Purpose**: Parse SILENT-format CSV files into sample vectors

```rust
// crates/timing-oracle/src/data/mod.rs

/// Errors that can occur during data loading.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    #[error("Missing group label in data (expected X and Y)")]
    MissingGroup,

    #[error("Insufficient samples: got {got}, need at least {min}")]
    InsufficientSamples { got: usize, min: usize },
}

/// Loaded timing data with two groups.
#[derive(Debug, Clone)]
pub struct TimingData {
    /// Samples for group X (typically baseline/control).
    pub x_samples: Vec<u64>,
    /// Samples for group Y (typically test/treatment).
    pub y_samples: Vec<u64>,
    /// Time unit of the samples (cycles, nanoseconds, etc.).
    pub unit: TimeUnit,
    /// Optional metadata from the file.
    pub metadata: Option<DataMetadata>,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum TimeUnit {
    #[default]
    Cycles,
    Nanoseconds,
    Microseconds,
}

/// Load SILENT-format CSV: "V1,V2" header with X/Y group labels.
///
/// Format:
/// ```csv
/// V1,V2
/// X,1067424
/// Y,531296
/// X,1051010
/// ...
/// ```
pub fn load_silent_csv(path: &Path) -> Result<TimingData, DataError> {
    // Implementation
}

/// Load generic two-column CSV with optional header.
/// Assumes column 1 = group label, column 2 = timing value.
pub fn load_two_column_csv(
    path: &Path,
    has_header: bool,
    x_label: &str,
    y_label: &str,
) -> Result<TimingData, DataError> {
    // Implementation
}

/// Load from two separate files (one per group).
pub fn load_separate_files(
    x_path: &Path,
    y_path: &Path,
    unit: TimeUnit,
) -> Result<TimingData, DataError> {
    // Implementation
}
```

**Dependencies**: None new (just `std::io`, `std::path`)

**Effort**: ~2-3 hours

---

### Task 2: Raw Sample Analysis API

**Location**: `crates/timing-oracle/src/oracle.rs` (extend existing)

**Purpose**: Add methods to analyze pre-collected samples

```rust
impl TimingOracle {
    /// Analyze pre-collected timing samples.
    ///
    /// This is useful for:
    /// - Analyzing data from external tools (SILENT, dudect, etc.)
    /// - Replaying historical measurements
    /// - Testing with synthetic/simulated data
    ///
    /// # Arguments
    /// * `data` - Loaded timing data with two sample groups
    /// * `ns_per_sample` - Nanoseconds per sample unit (1.0 if already in ns)
    ///
    /// # Example
    /// ```ignore
    /// use timing_oracle::{TimingOracle, AttackerModel, data::load_silent_csv};
    ///
    /// let data = load_silent_csv("kyberslash.csv")?;
    /// let outcome = TimingOracle::for_attacker(AttackerModel::SharedHardware)
    ///     .analyze_samples(data, 0.33);  // 0.33 ns/cycle at 3GHz
    /// ```
    pub fn analyze_samples(
        self,
        data: TimingData,
        ns_per_sample: f64,
    ) -> Outcome {
        self.analyze_raw_samples(
            &data.x_samples,
            &data.y_samples,
            ns_per_sample,
        )
    }

    /// Analyze raw sample arrays directly.
    ///
    /// Lower-level API for when you have samples in memory.
    pub fn analyze_raw_samples(
        self,
        baseline_samples: &[u64],
        test_samples: &[u64],
        ns_per_sample: f64,
    ) -> Outcome {
        // 1. Validate inputs
        // 2. Create calibration config (no preflight checks for pre-collected)
        // 3. Split samples: first N for calibration, rest for adaptive
        // 4. Run calibration phase
        // 5. Run adaptive loop with remaining samples
        // 6. Convert AdaptiveOutcome → Outcome
    }
}
```

**Key Design Decisions**:

1. **No preflight checks**: Pre-collected data can't be validated for timer sanity, autocorrelation source, etc. Skip preflight or run reduced checks.

2. **Sample splitting**: Need to decide how to split samples between calibration and adaptive phases:
   - Option A: Use first `calibration_samples` for calibration, rest for adaptive
   - Option B: Use all samples for calibration (single-pass mode)
   - **Recommendation**: Option B for simplicity - run calibration on all data, then compute posterior

3. **Single-pass mode**: For pre-collected data, we can't collect more samples. The adaptive loop becomes a single posterior computation rather than iterative sampling.

4. **Diagnostics adaptation**: Some diagnostics (drift detection, throughput) don't apply to pre-collected data. Need to handle gracefully.

**Effort**: ~4-5 hours

---

### Task 3: Single-Pass Analysis Mode

**Location**: `crates/timing-oracle/src/adaptive/single_pass.rs` (new file)

**Purpose**: Compute outcome from fixed sample set without adaptive iteration

```rust
/// Analyze a fixed set of samples in a single pass.
///
/// Unlike the adaptive loop, this doesn't collect additional samples.
/// Used for pre-collected data analysis.
pub fn analyze_fixed_samples(
    baseline_ns: &[f64],
    sample_ns: &[f64],
    theta_ns: f64,
    config: &AnalysisConfig,
) -> SinglePassResult {
    // 1. Compute quantile differences
    // 2. Estimate covariance via bootstrap
    // 3. Compute posterior P(leak > theta | data)
    // 4. Apply decision thresholds
    // 5. Return result with diagnostics
}

pub struct SinglePassResult {
    pub outcome: Outcome,
    pub posterior: PosteriorState,
    pub effect_estimate: EffectEstimate,
    pub quality: MeasurementQuality,
}
```

**Rationale**: The existing adaptive loop assumes we can collect more samples. For pre-collected data, we need a simpler single-pass computation.

**Effort**: ~3-4 hours

---

### Task 4: Unit/Time Conversion Utilities

**Location**: `crates/timing-oracle/src/data/units.rs`

**Purpose**: Handle different time units in input data

```rust
/// Convert samples from one time unit to nanoseconds.
pub fn to_nanoseconds(samples: &[u64], unit: TimeUnit, cpu_freq_ghz: Option<f64>) -> Vec<f64> {
    match unit {
        TimeUnit::Nanoseconds => samples.iter().map(|&s| s as f64).collect(),
        TimeUnit::Microseconds => samples.iter().map(|&s| s as f64 * 1000.0).collect(),
        TimeUnit::Cycles => {
            let ns_per_cycle = cpu_freq_ghz
                .map(|f| 1.0 / f)
                .unwrap_or(0.33); // Default ~3GHz
            samples.iter().map(|&s| s as f64 * ns_per_cycle).collect()
        }
    }
}

/// Estimate CPU frequency from cycle counts and wall-clock time.
/// Useful when analyzing data where frequency is unknown.
pub fn estimate_cpu_freq_ghz(cycles: &[u64], duration_ns: u64) -> f64 {
    let total_cycles: u64 = cycles.iter().sum();
    total_cycles as f64 / duration_ns as f64
}
```

**Effort**: ~1 hour

---

## Module Structure

```
crates/timing-oracle/src/
├── data/
│   ├── mod.rs          # Public API, TimingData struct
│   ├── csv.rs          # CSV parsing (SILENT format, generic)
│   ├── units.rs        # Time unit conversions
│   └── validation.rs   # Input validation
├── adaptive/
│   ├── ...existing...
│   └── single_pass.rs  # NEW: Single-pass analysis
└── oracle.rs           # Extended with analyze_samples()
```

## API Surface (Public)

```rust
// New public items in timing_oracle crate

pub mod data {
    pub struct TimingData { ... }
    pub enum TimeUnit { ... }
    pub enum DataError { ... }

    pub fn load_silent_csv(path: &Path) -> Result<TimingData, DataError>;
    pub fn load_two_column_csv(...) -> Result<TimingData, DataError>;
}

impl TimingOracle {
    pub fn analyze_samples(self, data: TimingData, ns_per_sample: f64) -> Outcome;
    pub fn analyze_raw_samples(self, baseline: &[u64], test: &[u64], ns_per_sample: f64) -> Outcome;
}
```

## Testing Strategy

### Unit Tests
1. CSV parsing with valid SILENT format
2. CSV parsing with malformed input
3. Empty/insufficient samples handling
4. Time unit conversions

### Integration Tests
1. **KyberSlash dataset**: Load → Analyze → Expect FAIL (leak detected)
2. **mbedTLS dataset**: Load → Analyze → Expect PASS (no leak, was FP in RTLF)
3. **Web app dataset**: Load → Analyze → Expect FAIL (leak detected)

### Comparison Tests
1. Run SILENT R script on dataset
2. Run timing-oracle on same dataset
3. Compare decisions and metrics

## Implementation Order

1. **Task 1: CSV Loader** (2-3 hours)
   - No dependencies on other tasks
   - Immediately testable with SILENT data files

2. **Task 4: Unit Conversions** (1 hour)
   - Simple utilities needed by other tasks

3. **Task 3: Single-Pass Analysis** (3-4 hours)
   - Core statistical computation
   - Can reuse existing `calibrate()` function

4. **Task 2: Raw Sample API** (4-5 hours)
   - Integrates all other components
   - Final public API

**Total estimated effort**: 10-13 hours

## Open Questions

1. **Should we expose the single-pass API publicly?**
   - Pro: More flexibility for advanced users
   - Con: More API surface to maintain
   - **Recommendation**: Keep private initially, expose if needed

2. **How to handle unknown CPU frequency?**
   - Option A: Require user to specify ns_per_sample
   - Option B: Default to common value (3GHz → 0.33 ns/cycle)
   - Option C: Try to estimate from data if timestamps available
   - **Recommendation**: Option A with Option B as default

3. **Should pre-collected analysis skip all preflight checks?**
   - Some checks (autocorrelation) could still be useful
   - Others (timer sanity) don't apply
   - **Recommendation**: Run subset of checks, flag in diagnostics

## Success Criteria

- [ ] Can load all three SILENT example datasets
- [ ] Results match SILENT's decisions (Rejected/Not rejected)
- [ ] Performance: < 5 seconds for 10K sample analysis
- [ ] Clear error messages for malformed input
- [ ] Documentation with examples
