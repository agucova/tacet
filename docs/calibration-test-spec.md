# Calibration Test Suite Technical Specification

## Overview / Context

This document specifies a deterministic, CI-safe calibration test suite for the `timing-oracle` Rust crate. The suite validates, per timer backend and attacker model:

1. **False Positive Rate (FPR)** under true-null inputs
2. **Power** as a function of injected timing effect size
3. **Bayesian credible interval coverage** for explicitly exposed posterior parameters
4. **Robustness** under bounded, opt-in CPU/memory contention

The suite produces structured JSONL output and uses explicit skip/fail policy to minimize flakes while still detecting regressions.

## Goals and Non-Goals

### Goals

- Deterministic-by-default test behavior given fixed env + fixed seed.
- CI-bounded runtime and resources with explicit caps.
- Clear, machine-readable pass/fail/skip reasons and metrics.
- Support Linux/macOS on x86_64 and aarch64; Windows is best-effort with explicit backend skips.

### Non-Goals

- Benchmarking absolute throughput/latency of the oracle.
- Verifying OS/kernel PMU configuration beyond "available and usable without elevated privileges".
- Classifier correctness of effect-pattern labels beyond statistical calibration metrics.

## Implementation Phases

### Phase 1: Core Infrastructure (No API Changes)

- `calibration_utils.rs`: config, RNG, trial runner, JSONL, stats helpers
- `calibration_fpr.rs`: FPR tests using existing API
- `calibration_power.rs`: Power tests using existing API
- Coverage tests: **DEFERRED** (requires API change)

### Phase 2: API Extension + Coverage

- Add `shift_ns_ci95` and `tail_ns_ci95` to `Effect` struct
- Implement `calibration_coverage.rs`

### Phase 3: Stress Tests

- `calibration_stress.rs`: bounded stress wrappers
- Enable after Phase 1 tests are stable

---

## System Architecture

### Layout

```
crates/timing-oracle/tests/
├── calibration_utils.rs      # Config, RNG, trial runner, JSONL, stats helpers
├── calibration_fpr.rs        # Null experiments (Phase 1)
├── calibration_power.rs      # Effect-size sweep (Phase 1)
├── calibration_coverage.rs   # Posterior CI coverage (Phase 2)
├── calibration_stress.rs     # Bounded stress wrappers (Phase 3)
└── calibration.rs            # Legacy tests (migrate to use common utilities)
```

### Tiering

Tier selected by `CALIBRATION_TIER` with defaults:

- If `CALIBRATION_TIER` set: use it
- Else if `CI` env var is exactly `true` or `1`: `quick`
- Else: `full`

| Tier | Max Wall Time | Samples/Trial | FPR Trials | Power Trials | Coverage Trials |
|------|---------------|---------------|------------|--------------|-----------------|
| quick | 3 min | 2,000 | 50 | 50 | 100 |
| full | 8 min | 5,000 | 100 | 100 | 200 |
| validation | 25 min | 10,000 | 500 | 200 | 500 |

---

## Component Design

### 1) Configuration Contract

All calibration tests MUST call `CalibrationConfig::from_env(test_name)` and MUST NOT read env vars directly elsewhere.

#### Environment Variables (normative)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CALIBRATION_TIER` | `quick\|full\|validation` | auto | Test tier |
| `CALIBRATION_SEED` | `u64` | auto | RNG seed |
| `CALIBRATION_TIMER` | `auto\|coarse\|pmu` | `auto` | Timer backend |
| `CALIBRATION_MAX_WALL_MS` | `u64` | tier-dependent | Max wall time (hard cap: 1,800,000) |
| `CALIBRATION_MAX_UNMEASURABLE_RATE` | `f64` | 0.20 | Max unmeasurable rate before skip |
| `CALIBRATION_MIN_COMPLETED_RATE` | `f64` | 0.90 | Min completion rate |
| `CALIBRATION_LOG_FORMAT` | `text\|jsonl` | `text` | Output format |
| `CALIBRATION_ENABLE_STRESS` | `0\|1` | `0` | Enable stress tests |
| `CALIBRATION_DISABLE_BATCHING` | `0\|1` | `0` | Disable adaptive batching |
| `CALIBRATION_MAX_STRESS_THREADS` | `usize` | 2 (CI) / 4 (local) | Max stress threads (hard cap: 4) |
| `CALIBRATION_MAX_MEMORY_MB` | `usize` | 128 / 256 | Max memory pressure (hard cap: 512) |
| `CALIBRATION_DISABLED` | `0\|1` | `0` | Skip all calibration tests |

All numeric values are clamped to compiled-in hard caps to prevent DoS.

#### Rust Types

```rust
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    pub tier: Tier,
    pub seed: u64,
    pub seed_source: SeedSource,
    pub timer: TimerBackend,
    pub max_wall_ms: u64,
    pub max_unmeasurable_rate: f64,
    pub min_completed_rate: f64,
    pub log_format: LogFormat,
    pub enable_stress: bool,
    pub disable_batching: bool,
    pub samples_per_trial: usize,
    pub time_budget_per_trial: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier { Quick, Full, Validation }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedSource { Fixed, DerivedFromTestName, Random }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerBackend { Coarse, Pmu }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat { Text, Jsonl }
```

#### Deterministic Seed Derivation

- If `CALIBRATION_SEED` set: use it (`SeedSource::Fixed`)
- Else if CI detected (`CI=true` or `CI=1`): `seed = fnv1a_64("timing-oracle:" + test_name)`
- Else: `seed = OS RNG` and print reproduction command

`fnv1a_64` is FNV-1a 64-bit hash over UTF-8 bytes.

### 2) Timer Backend Selection

`CALIBRATION_TIMER=auto` resolves deterministically:

1. If PMU backend is available *and usable without elevated privileges*: select PMU
2. Else: select coarse backend

### 3) Effect Injection Contract

```rust
/// Best-effort delay of at least `ns` nanoseconds.
/// Uses Instant-based spinning for determinism across platforms.
pub fn busy_wait_ns(ns: u64) {
    let clamped = ns.min(MAX_INJECT_NS);
    let start = Instant::now();
    let target = Duration::from_nanos(clamped);
    while start.elapsed() < target {
        std::hint::spin_loop();
    }
}
```

| Tier | MAX_INJECT_NS |
|------|---------------|
| Quick | 10,000 (10μs) |
| Full | 100,000 (100μs) |
| Validation | 1,000,000 (1ms) |

**Rationale**: `Instant`-based spinning is chosen over cycle-counter calibration for determinism. Cycle counters vary with DVFS, core migration, and VM jitter.

### 4) Oracle Output Contract

#### Current API (Phase 1)

```rust
pub enum Outcome {
    Pass { leak_probability: f64, effect: Effect, /* ... */ },
    Fail { leak_probability: f64, effect: Effect, /* ... */ },
    Inconclusive { leak_probability: f64, effect: Effect, /* ... */ },
    Unmeasurable { /* ... */ },
}

pub struct Effect {
    pub shift_ns: f64,
    pub tail_ns: f64,
    pub credible_interval_ns: (f64, f64),  // 95% CI for magnitude
    pub pattern: EffectPattern,
}
```

#### Extended API (Phase 2)

```rust
pub struct Effect {
    pub shift_ns: f64,
    pub tail_ns: f64,
    pub credible_interval_ns: (f64, f64),
    pub shift_ns_ci95: Option<(f64, f64)>,     // NEW
    pub tail_ns_ci95: Option<(f64, f64)>,      // NEW
    pub pattern: EffectPattern,
}
```

### 5) Decision Rule (single source of truth)

#### Outcome Classification

- `Pass`: `leak_probability <= 0.05`
- `Fail`: `leak_probability >= 0.95`
- `Inconclusive`: otherwise
- `Unmeasurable`: explicit

**For FPR**: `Fail` = false positive. `Pass` and `Inconclusive` = not false positive.

**For Power**: `Fail` = detection. `Pass` and `Inconclusive` = no detection.

### 6) FPR Validation (Phase 1)

#### Experiment

True-null input pair: both classes execute identical code with no injected delay.

#### Acceptance

Use Wilson CI upper bound: pass if `wilson_upper_95(failures, completed) <= max_fpr`.

| Tier | Max FPR |
|------|---------|
| Quick | 15% |
| Full | 10% |
| Validation | 7% |

### 7) Power Validation (Phase 1)

#### Effect Sizes (θ-relative)

| AttackerModel | θ | Test Effects (0.5θ, 1θ, 2θ, 5θ) |
|---------------|---|----------------------------------|
| Research | 50ns (nominal) | 25ns, 50ns, 100ns, 250ns |
| PostQuantumSentinel | 3.3ns | 2ns, 3ns, 7ns, 17ns |
| AdjacentNetwork | 100ns | 50ns, 100ns, 200ns, 500ns |
| WANConservative | 50μs | 25μs, 50μs, 100μs, 250μs |

#### Acceptance

| Effect | Requirement |
|--------|-------------|
| 0.5θ | Report only |
| 1θ | Report only (warn if <30%) |
| 2θ | Power ≥ 70% |
| 5θ | Power ≥ 90% (95% for validation) |

### 8) Coverage Validation (Phase 2)

#### Prerequisite

`Effect::shift_ns_ci95` must be exposed. Without this, tests skip with `"api_not_extended"`.

#### Experiment

Inject pure uniform shifts: 100ns, 300ns, 500ns. Check if `shift_ns_ci95` contains true value.

#### Acceptance

| Tier | Min Coverage |
|------|--------------|
| Quick | 80% |
| Full | 85% |
| Validation | 88% |

### 9) Stress Testing (Phase 3)

#### CPU Contention

Spawn up to `min(CALIBRATION_MAX_STRESS_THREADS, num_cpus/4)` threads with periodic yields.

#### Memory Pressure

Allocate up to `min(CALIBRATION_MAX_MEMORY_MB, 512)` MB and touch pages.

#### Acceptance Under Stress

- FPR max relaxed by +10% absolute
- Skip if unmeasurable rate > 30%

---

## JSONL Output Schema (v1)

```json
{
  "schema_version": 1,
  "test": "calibration_fpr::fpr_quick_random_vs_random",
  "tier": "quick",
  "timer": "coarse",
  "seed": 12345678,
  "seed_source": "derived_from_test_name",
  "requested_trials": 50,
  "completed_trials": 48,
  "unmeasurable_trials": 2,
  "pass": 45,
  "fail": 3,
  "inconclusive": 0,
  "metrics": {
    "fpr": 0.0625,
    "wilson_upper_95": 0.165
  },
  "thresholds": {
    "max_fpr": 0.15
  },
  "decision": "pass",
  "skip_reason": null,
  "wall_time_ms": 45000,
  "platform": {
    "arch": "x86_64",
    "os": "linux"
  }
}
```

---

## Error Handling Strategy

### Fail vs Skip Policy

| Condition | Decision |
|-----------|----------|
| Metric violates acceptance AND sufficient trials | **Fail** |
| `completed < min_completed_rate * requested` | **Skip** (`insufficient_completed_trials`) |
| `unmeasurable_rate > max_unmeasurable_rate` | **Skip** (`excessive_unmeasurable_rate`) |
| Wall time exceeded before min completion | **Skip** (`wall_time_exceeded`) |
| Required timer unavailable | **Skip** (`timer_unavailable`) |
| Invalid env var | **Skip** (`invalid_config`) |
| `CALIBRATION_DISABLED=1` | **Skip** (`disabled_by_env`) |
| Coverage test without API extension | **Skip** (`api_not_extended`) |

---

## Security Considerations

1. **Env var validation**: All numeric env vars parsed strictly; invalid = skip (not panic).
2. **Hard caps**: All resource limits have compiled-in maximums.
3. **No privilege escalation**: PMU probing never attempts sudo.
4. **Minimal logging**: No host-unique identifiers.

---

## Migration Plan

### Phase 1 (This PR)

- Add `calibration_utils.rs` with all infrastructure
- Add `calibration_fpr.rs` and `calibration_power.rs` as **non-blocking**
- Add CI workflow running `quick` tier
- Monitor flake/skip rates for 2 weeks

### Phase 2 (Follow-up PR)

- Extend `Effect` struct with `shift_ns_ci95`, `tail_ns_ci95`
- Add `calibration_coverage.rs` as **non-blocking**
- Make Phase 1 tests **blocking** once flake rate < 5%

### Phase 3 (Follow-up PR)

- Add `calibration_stress.rs` as **non-blocking**
- Make `full` tier tests **blocking**
- Enable weekly `validation` runs

### Rollback

- `CALIBRATION_DISABLED=1` in CI workflow disables all calibration tests immediately

---

## References

This specification was developed through adversarial debate between Claude (Opus 4.5) and GPT-5.2 (via Codex CLI), with 2 rounds of critique and revision.
