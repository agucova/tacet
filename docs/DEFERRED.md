# Deferred Items and Implementation Notes

This document tracks items deferred from the v4.1 spec implementation, UX improvements retained from the existing implementation, and remaining work.

## Completed v4.1 Changes

### Type System Updates
- [x] `Outcome::Research(ResearchOutcome)` - 5th variant for research mode
- [x] `ResearchStatus` enum with 5 variants:
  - `EffectDetected` - CI lower > theta_floor * 1.1
  - `NoEffectDetected` - CI includes floor, no clear effect
  - `ResolutionLimitReached` - Stuck at theta_floor
  - `QualityIssue(InconclusiveReason)` - Quality gate triggered
  - `BudgetExhausted` - Time/sample budget exhausted
- [x] `ResearchOutcome` struct with research-specific fields
- [x] `InconclusiveReason::ThresholdUnachievable` variant
- [x] `InconclusiveReason::ModelMismatch` variant
- [x] `IssueCode::ModelMismatch` for quality issues
- [x] `interpretation_caveat: Option<String>` on `EffectEstimate`
- [x] New fields on Pass/Fail/Inconclusive: `theta_user`, `theta_eff`, `theta_floor`
- [x] `Calibration` struct v4.1 fields: `c_floor`, `q_thresh`, `theta_tick`, `theta_eff`, `theta_floor_initial`, `rng_seed`

### Quality Gates
- [x] Quality gate ordering - gates checked BEFORE decision boundaries (spec Section 2.6)
- [x] ThresholdUnachievable gate (when theta_user < theta_floor(n_max))
- [x] ModelMismatch gate (Q > q_thresh, verdict-blocking)
- [x] chi-squared(7, 0.99) = 18.48 fallback for q_thresh

### Prior Scale
- [x] Prior covariance uses `theta_eff` (not `theta_user`) per spec Section 2.3.4
- [x] Prior scale factor 1.12 applied to effective threshold

### Measurement Floor
- [x] Analytical theta_floor computation: `c_floor / sqrt(n)`
- [x] theta_tick floor component from timer resolution
- [x] theta_eff = max(theta_user, theta_floor)

## UX Improvements Retained from Implementation

These implementation choices improve user experience compared to the spec:

### 1. Field Naming
- **`model_fit_chi2`** instead of spec's `model_fit_q` - clearer that it's a chi-squared statistic
- **`model_fit_threshold`** - explicit threshold field for comparison

### 2. Rich Diagnostics
- `preflight_warnings: Vec<PreflightWarningInfo>` with severity levels
- `timer_name: String` and `platform: String` for reproducibility
- `attacker_model: Option<String>` for context
- `quality_issues: Vec<QualityIssue>` with guidance

### 3. PreflightSeverity Distinction
- `Informational` (efficiency hints) vs `ResultUndermining` (assumption violations)
- Users can distinguish safe-to-ignore vs critical warnings

### 4. QualityIssue with IssueCode
- Structured `QualityIssue { code, message, guidance }`
- Typed `IssueCode` enum for programmatic handling

### 5. Environment Variable Support
- `TIMING_ORACLE_UNRELIABLE_POLICY` for CI configuration (fail-open/fail-closed)

### 6. Outcome Helper Methods
- `is_reliable()` - considers both quality and confidence level
- `handle_unreliable()` - policy-based handling
- `leak_probability()`, `effect()`, `quality()`, `diagnostics()` - unified access

## Recently Implemented (v4.1 completion)

### Implemented in v4.1 Migration

1. **Bootstrap-calibrated q_thresh** ✓
   - Now computes 99th percentile of Q* distribution during bootstrap
   - Two-pass approach: first pass computes covariance, second pass computes Q* distribution
   - Falls back to chi-squared(7, 0.99) = 18.48 only when inversion fails or discrete mode
   - Location: `crates/timing-oracle-core/src/statistics/covariance.rs`

2. **AcquisitionStream integration** ✓
   - Added `to_timing_samples()` adapter method to `AcquisitionStream`
   - Calibration now uses `AcquisitionStream` for interleaved sample construction
   - Location: `crates/timing-oracle-core/src/statistics/acquisition.rs`
   - Location: `crates/timing-oracle/src/adaptive/calibration.rs`

3. **ThresholdElevated QualityIssue** ✓
   - Emits when `theta_eff > theta_user` (measurement floor exceeds user threshold)
   - Provides clear guidance about using cycle counters for better resolution
   - Location: `crates/timing-oracle/src/oracle.rs` (build_diagnostics)

4. **CI-based stopping for research mode** ✓
   - Full implementation of `run_research_mode()` method
   - Stopping conditions: EffectDetected, NoEffectDetected, ResolutionLimitReached, QualityIssue, BudgetExhausted
   - Location: `crates/timing-oracle/src/oracle.rs`

## Remaining Deferred Items

### Lower Priority (Optional)

1. **Full AcquisitionStream in oracle measurement loop**
   - Current: Oracle stores samples in separate `Vec<u64>` for baseline/sample
   - Spec: Could use `AcquisitionStream` throughout measurement loop
   - Impact: Minor improvement to dependence estimation during collection
   - Note: Calibration already uses AcquisitionStream; oracle measurement is more complex refactor

2. **Bootstrap-calibrated q_thresh for discrete mode**
   - Current: Discrete mode uses chi-squared fallback (18.48)
   - Spec: Should compute Q* distribution using m-out-of-n bootstrap
   - Impact: Conservative fallback is acceptable for discrete data
   - Note: Discrete mode uses different bootstrap structure (per-class sequences)

### Test File Fixes ✓ COMPLETED

All test files have been updated with `Outcome::Research(_)` match arms:

- ✓ `tests/async_timing.rs`
- ✓ `tests/calibration_coverage.rs`
- ✓ `tests/calibration_estimation.rs`
- ✓ `tests/calibration_bayesian.rs`
- ✓ `tests/crypto_attacks.rs`
- ✓ `tests/discrete_mode.rs`
- ✓ `tests/hash_timing.rs`
- ✓ `tests/macro_tests.rs`
- ✓ `tests/pqcrypto_timing.rs`
- ✓ `tests/power_analysis.rs`
- ✓ `tests/rsa_timing.rs`
- ✓ `tests/rsa_vulnerability_assessment.rs`
- ✓ `tests/aead_timing.rs`
- ✓ `tests/aes_timing.rs`
- ✓ `tests/ecc_timing.rs`
- ✓ `tests/calibration_utils.rs`

### Benchmark Adapter Updates ✓ COMPLETED

- ✓ `benches/comp/adapters/timing_oracle_adapter.rs` - Research variant added
- ✓ `benches/oracle.rs` - Research variant added to match arms

## Breaking Changes Summary

| Change | Impact | Migration |
|--------|--------|-----------|
| `Outcome::Research` variant | Match arms break | Add new arm |
| New `InconclusiveReason` variants | Match arms break | Add new arms |
| New `Calibration` fields | Struct init breaks | Update `new()` calls |
| New `Outcome` fields | Destructuring breaks | Update pattern matches |
| `ResearchStatus` in core | New dependency | Import from core |

## Notes

- All packages compile successfully (timing-oracle, timing-oracle-core, timing-oracle-c)
- All examples compile and run successfully
- All test files updated with Research variant match arms
- Research mode fully integrated with CI-based stopping conditions
- Bootstrap-calibrated q_thresh implemented for standard mode
- AcquisitionStream integrated into calibration phase
- ThresholdElevated QualityIssue emitted when measurement floor exceeds user threshold
