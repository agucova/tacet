# Calibration Enhancement Plan

This plan adds two new calibration test suites to enable comparative claims against SILENT and other timing side-channel tools.

## Platform Requirements

**Target platform:** Linux ARM64 with `perf` feature (perf_event PMU access)

Prerequisites:
```bash
# Option 1: Run as root
sudo cargo test --features perf

# Option 2: Set perf_event_paranoid (persistent)
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
cargo test --features perf

# Option 3: Grant capabilities to test binary
cargo build --release
sudo setcap cap_perfmon=ep ./target/release/deps/calibration_*
```

**Note:** `kperf` is macOS-only. On Linux, we use `perf` (perf_event).

---

## Enhancement 1: Autocorrelation Robustness Tests

**Goal:** Replicate SILENT's Figure 1 - show that our oracle maintains type-1 error control across varying autocorrelation levels, unlike dudect/TVLA/RTLF/tlsfuzzer.

### Design

Generate synthetic timing data using an AR(1) process with controlled parameters:
- **μ (effect size):** 0, 0.5Δ, Δ, 2Δ (where Δ = θ = 100ns for AdjacentNetwork)
- **Φ (autocorrelation):** -0.8, -0.4, 0, 0.4, 0.8

For each (μ, Φ) cell:
1. Generate N trials of synthetic correlated data
2. Feed to the oracle (bypassing real measurement)
3. Record rejection rate
4. Verify type-1 error ≤ α when μ < Δ

### New File: `calibration_autocorrelation.rs`

```rust
//! Autocorrelation robustness tests.
//!
//! Validates that the oracle maintains type-1 error control under
//! varying levels of measurement autocorrelation, unlike classical
//! tools that assume IID samples.

mod calibration_utils;

use calibration_utils::{CalibrationConfig, Tier};

// AR(1) process parameters
const EFFECT_MULTIPLIERS: [f64; 4] = [0.0, 0.5, 1.0, 2.0];  // × θ
const AUTOCORRELATIONS: [f64; 5] = [-0.8, -0.4, 0.0, 0.4, 0.8];  // Φ

// θ for AdjacentNetwork
const THETA_NS: f64 = 100.0;

// Simulation parameters
const SAMPLES_PER_TRIAL: usize = 10_000;
const BASE_NOISE_STD: f64 = 50.0;  // σ in nanoseconds

/// Generate AR(1) correlated samples.
///
/// Y_n = Φ * Y_{n-1} + ε_n, where ε_n ~ N(0, σ²)
fn generate_ar1_samples(
    n: usize,
    phi: f64,
    mean_shift: f64,
    sigma: f64,
    rng: &mut impl rand::Rng,
) -> Vec<f64> {
    use rand_distr::{Distribution, Normal};

    let normal = Normal::new(0.0, sigma).unwrap();
    let mut samples = Vec::with_capacity(n);

    // Innovation variance adjusted for stationarity
    let innovation_var = sigma * sigma * (1.0 - phi * phi).max(0.01);
    let innovation_std = innovation_var.sqrt();
    let innovation = Normal::new(0.0, innovation_std).unwrap();

    let mut prev = 0.0;
    for _ in 0..n {
        let eps = innovation.sample(rng);
        let y = phi * prev + eps + mean_shift;
        samples.push(y);
        prev = y - mean_shift;  // Detrend for AR process
    }

    samples
}

#[test]
fn autocorrelation_robustness_quick() {
    // Quick version: subset of grid
    run_autocorrelation_grid(
        "autocorr_quick",
        &[0.0, 1.0, 2.0],      // μ multipliers
        &[-0.4, 0.0, 0.4],      // Φ values
        50,                      // trials per cell
    );
}

#[test]
#[ignore]
fn autocorrelation_robustness_validation() {
    // Full grid for publication
    run_autocorrelation_grid(
        "autocorr_validation",
        &EFFECT_MULTIPLIERS,
        &AUTOCORRELATIONS,
        200,  // trials per cell
    );
}

fn run_autocorrelation_grid(
    test_name: &str,
    mu_multipliers: &[f64],
    phi_values: &[f64],
    trials_per_cell: usize,
) {
    // Implementation feeds synthetic data to oracle
    // Records rejection rate for each (μ, Φ) cell
    // Outputs CSV for heatmap plotting
    todo!("Implement grid test")
}
```

### Implementation Steps

1. **Add AR(1) generator to `calibration_utils.rs`**
   - `generate_ar1_samples(n, phi, mean_shift, sigma, rng) -> Vec<f64>`
   - Handles edge cases (|Φ| near 1)

2. **Create synthetic data injection path**
   - Option A: Create `TimingOracle::test_with_raw_data()` that accepts pre-generated samples
   - Option B: Mock the measurement collector to return synthetic data
   - **Recommended:** Option A is cleaner for testing

3. **Implement heatmap data output**
   - CSV format: `mu_mult,phi,trials,rejections,rejection_rate,ci_low,ci_high`
   - One row per (μ, Φ) cell

4. **Add plotting script**
   - `scripts/plot_autocorrelation_heatmap.py`
   - Generates Figure 1-style heatmaps

### Acceptance Criteria

| Condition | Requirement |
|-----------|-------------|
| μ = 0, any Φ | Rejection rate ≤ α (type-1 error control) |
| μ = 0.5Δ, any Φ | Rejection rate ≤ α (still in H₀ for relevant hypothesis) |
| μ = Δ, any Φ | Rejection rate ≈ α (boundary) |
| μ = 2Δ, any Φ | Rejection rate ≥ 1-β (power) |

For Tier::Validation: α = 0.10, β = 0.10

---

## Enhancement 2: Fine-Grained Power Curves

**Goal:** Generate smooth power curves with finer effect size granularity and confidence intervals, suitable for publication.

### Design

Sweep effect sizes more finely around θ:
- **Effect sizes:** 0, 0.25θ, 0.5θ, 0.75θ, θ, 1.25θ, 1.5θ, 2θ, 3θ, 5θ, 10θ
- **Output:** Detection rate with Wilson 95% CI at each point
- **Additional metrics:** Median samples to decision, median wall time

### New File: `calibration_power_curve.rs`

```rust
//! Fine-grained power curve generation.
//!
//! Generates publication-quality power curves showing detection rate
//! vs. effect size with confidence intervals.

mod calibration_utils;

use calibration_utils::{
    busy_wait_ns, wilson_ci, CalibrationConfig, TrialRunner,
};
use timing_oracle::{AttackerModel, TimingOracle, Outcome};
use timing_oracle::helpers::InputPair;

// Fine-grained effect multipliers
const EFFECT_MULTIPLIERS: [f64; 11] = [
    0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0
];

/// Power curve data point
#[derive(Debug, Clone)]
struct PowerPoint {
    effect_mult: f64,
    effect_ns: f64,
    trials: usize,
    detections: usize,
    detection_rate: f64,
    ci_low: f64,
    ci_high: f64,
    median_samples: usize,
    median_time_ms: u64,
}

#[test]
fn power_curve_iteration() {
    // Quick check: 3 points
    run_power_curve(
        "power_curve_iteration",
        AttackerModel::AdjacentNetwork,
        100.0,  // θ
        &[0.0, 1.0, 2.0],
    );
}

#[test]
fn power_curve_quick() {
    if std::env::var("CALIBRATION_TIER").as_deref() == Ok("iteration") {
        return;
    }
    // Standard: 6 key points
    run_power_curve(
        "power_curve_quick",
        AttackerModel::AdjacentNetwork,
        100.0,
        &[0.0, 0.5, 1.0, 1.5, 2.0, 5.0],
    );
}

#[test]
#[ignore]
fn power_curve_validation_adjacent_network() {
    // Full curve for publication
    std::env::set_var("CALIBRATION_TIER", "validation");
    run_power_curve(
        "power_curve_validation_adjacent",
        AttackerModel::AdjacentNetwork,
        100.0,
        &EFFECT_MULTIPLIERS,
    );
}

#[test]
#[ignore]
fn power_curve_validation_remote_network() {
    std::env::set_var("CALIBRATION_TIER", "validation");
    run_power_curve(
        "power_curve_validation_remote",
        AttackerModel::RemoteNetwork,
        50_000.0,  // θ = 50μs
        &EFFECT_MULTIPLIERS,
    );
}

fn run_power_curve(
    test_name: &str,
    model: AttackerModel,
    theta_ns: f64,
    multipliers: &[f64],
) {
    let config = CalibrationConfig::from_env(test_name);
    let trials = config.tier.power_trials();

    let mut points: Vec<PowerPoint> = Vec::new();

    for &mult in multipliers {
        let effect_ns = (theta_ns * mult) as u64;
        let mut detections = 0;
        let mut samples_used: Vec<usize> = Vec::new();
        let mut times_ms: Vec<u64> = Vec::new();

        for _ in 0..trials {
            let start = std::time::Instant::now();

            let inputs = InputPair::new(|| false, || true);
            let outcome = TimingOracle::for_attacker(model)
                .max_samples(config.samples_per_trial)
                .time_budget(config.time_budget_per_trial)
                .test(inputs, |&should_delay| {
                    // Base operation
                    busy_wait_ns(2000);
                    if should_delay {
                        busy_wait_ns(effect_ns);
                    }
                });

            let elapsed_ms = start.elapsed().as_millis() as u64;
            times_ms.push(elapsed_ms);

            match &outcome {
                Outcome::Fail { samples_used: n, .. } => {
                    detections += 1;
                    samples_used.push(*n);
                }
                Outcome::Pass { samples_used: n, .. } |
                Outcome::Inconclusive { samples_used: n, .. } => {
                    samples_used.push(*n);
                }
                _ => {}
            }
        }

        let rate = detections as f64 / trials as f64;
        let (ci_low, ci_high) = wilson_ci(detections, trials, 0.05);

        samples_used.sort();
        times_ms.sort();

        points.push(PowerPoint {
            effect_mult: mult,
            effect_ns: effect_ns as f64,
            trials,
            detections,
            detection_rate: rate,
            ci_low,
            ci_high,
            median_samples: samples_used.get(trials / 2).copied().unwrap_or(0),
            median_time_ms: times_ms.get(trials / 2).copied().unwrap_or(0),
        });
    }

    // Output CSV
    export_power_curve_csv(test_name, &points);

    // Print summary table
    print_power_curve_table(test_name, theta_ns, &points);
}
```

### Implementation Steps

1. **Add `wilson_ci` to calibration_utils.rs** (if not present)
   ```rust
   pub fn wilson_ci(successes: usize, trials: usize, alpha: f64) -> (f64, f64)
   ```

2. **Create `PowerPoint` struct** with all metrics

3. **Implement CSV export**
   - `calibration_data/power_curve_{test_name}.csv`
   - Columns: effect_mult, effect_ns, trials, detections, rate, ci_low, ci_high, median_samples, median_time_ms

4. **Add plotting script**
   - `scripts/plot_power_curve.py`
   - X-axis: effect/θ
   - Y-axis: detection rate with CI band
   - Reference lines at α and 1-β

### Acceptance Criteria

| Effect Size | Requirement (Validation Tier) |
|-------------|------------------------------|
| 0 | Detection rate ≤ 7% (FPR) |
| 0.5θ | Detection rate ≤ 15% |
| θ | Detection rate in [10%, 50%] |
| 2θ | Detection rate ≥ 70% |
| 5θ | Detection rate ≥ 95% |
| 10θ | Detection rate ≥ 99% |

---

## Changes to calibration_utils.rs

### New Functions

```rust
/// Generate AR(1) correlated samples for autocorrelation testing.
pub fn generate_ar1_samples(
    n: usize,
    phi: f64,
    mean_shift: f64,
    sigma: f64,
    rng: &mut StdRng,
) -> Vec<f64>;

/// Compute Wilson score confidence interval for binomial proportion.
pub fn wilson_ci(successes: usize, trials: usize, alpha: f64) -> (f64, f64);

/// Export power curve data to CSV.
pub fn export_power_curve_csv(test_name: &str, points: &[PowerPoint]);

/// Export autocorrelation heatmap data to CSV.
pub fn export_autocorr_heatmap_csv(
    test_name: &str,
    results: &[AutocorrCell],
);
```

### New Tier Methods

```rust
impl Tier {
    /// Trials per cell for autocorrelation grid.
    pub fn autocorr_trials_per_cell(&self) -> usize {
        match self {
            Tier::Iteration => 30,
            Tier::Quick => 50,
            Tier::Full => 100,
            Tier::Validation => 200,
        }
    }

    /// Number of effect multipliers for power curve.
    pub fn power_curve_points(&self) -> &'static [f64] {
        match self {
            Tier::Iteration => &[0.0, 1.0, 2.0],
            Tier::Quick => &[0.0, 0.5, 1.0, 1.5, 2.0, 5.0],
            Tier::Full => &[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
            Tier::Validation => &[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0],
        }
    }
}
```

---

## Linux ARM64 Compatibility Checklist

### CRITICAL: Fix `busy_wait_ns` Counter Frequency

**Problem:** Current implementation hardcodes 24MHz (Apple Silicon). Linux ARM64 uses different frequencies:
- AWS Graviton2/3: typically 1GHz (1,000,000,000 Hz)
- Ampere Altra: varies
- Generic: read from `CNTFRQ_EL0` register

**Fix required in `calibration_utils.rs`:**

```rust
#[cfg(target_arch = "aarch64")]
fn get_counter_freq_hz() -> u64 {
    // On macOS, it's always 24MHz
    #[cfg(target_os = "macos")]
    {
        24_000_000
    }

    // On Linux, read from CNTFRQ_EL0 register
    #[cfg(target_os = "linux")]
    {
        let freq: u64;
        unsafe {
            core::arch::asm!("mrs {}, cntfrq_el0", out(reg) freq);
        }
        freq
    }
}

#[cfg(target_arch = "aarch64")]
lazy_static::lazy_static! {
    static ref COUNTER_FREQ_HZ: u64 = get_counter_freq_hz();
}

#[inline(never)]
pub fn busy_wait_ns(ns: u64) {
    // ... clamping code ...

    #[cfg(target_arch = "aarch64")]
    {
        // Calculate ticks: ticks = ns * freq / 1_000_000_000
        // To avoid overflow with large ns: ticks = (ns / 1000) * (freq / 1_000_000) + adjustment
        let freq = *COUNTER_FREQ_HZ;
        let ticks = if freq == 24_000_000 {
            // Apple Silicon fast path: ticks = ns * 24 / 1000
            (clamped * 24 + 999) / 1000
        } else {
            // General case: ticks = ns * freq / 1e9
            // Use u128 to avoid overflow
            ((clamped as u128 * freq as u128 + 999_999_999) / 1_000_000_000) as u64
        };

        let start: u64;
        unsafe {
            core::arch::asm!("mrs {}, cntvct_el0", out(reg) start);
        }

        loop {
            let now: u64;
            unsafe {
                core::arch::asm!("mrs {}, cntvct_el0", out(reg) now);
            }
            if now.wrapping_sub(start) >= ticks {
                break;
            }
            std::hint::spin_loop();
        }
    }
    // ... non-aarch64 fallback ...
}
```

### Other Checklist Items

1. **Timer backend selection**
   - [x] `perf` feature available for Linux ARM64
   - [ ] **Fix `busy_wait_ns` to read counter frequency from `CNTFRQ_EL0`** (CRITICAL)
   - [ ] Add `lazy_static` to dev-dependencies if not present

2. **busy_wait_ns verification**
   - [ ] Test on AWS Graviton (1GHz counter)
   - [ ] Test on Ampere Altra
   - [ ] Verify delay accuracy within ±10%

3. **perf_event permissions**
   - Default `perf_event_paranoid` may be 2 or higher on cloud instances
   - **Action:** Document how to set `perf_event_paranoid=1` or use `CAP_PERFMON`

4. **Test on actual Linux ARM64 hardware**
   - AWS Graviton2/3
   - Ampere Altra
   - Raspberry Pi 4/5

---

## Estimated Runtime (Validation Tier)

| Test | Grid Size | Trials | Est. Time |
|------|-----------|--------|-----------|
| autocorr_validation | 4×5=20 cells | 200/cell | ~2 hours |
| power_curve_validation_adjacent | 11 points | 200/point | ~1 hour |
| power_curve_validation_remote | 11 points | 200/point | ~1 hour |

**Total:** ~4 hours for full validation suite

---

## Output Files

```
calibration_data/
├── autocorr_heatmap_{test_name}.csv
├── power_curve_{test_name}.csv
└── ...

scripts/
├── plot_autocorrelation_heatmap.py
├── plot_power_curve.py
└── ...
```

---

## Implementation Order

1. **Phase 1: Infrastructure** (calibration_utils.rs)
   - Add `wilson_ci` function
   - Add `generate_ar1_samples` function
   - Fix `busy_wait_ns` for Linux ARM64 counter frequency
   - Add CSV export helpers

2. **Phase 2: Power Curves** (calibration_power_curve.rs)
   - Implement fine-grained power curve tests
   - Add plotting script
   - Verify on macOS, then Linux ARM64

3. **Phase 3: Autocorrelation** (calibration_autocorrelation.rs)
   - Create synthetic data injection mechanism
   - Implement grid test
   - Add heatmap plotting script
   - Verify against expected type-1 error behavior

4. **Phase 4: Validation Run**
   - Run full validation on Linux ARM64
   - Generate figures for comparison with SILENT
   - Document results
