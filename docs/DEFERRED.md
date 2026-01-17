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

## Deferred Items

### Not Implemented (Lower Priority)

1. **AcquisitionStream type** - Per-acquisition interleaved storage
   - Current: Per-class `Vec<u64>` storage
   - Spec: `Vec<(SampleClass, f64)>` acquisition stream
   - Impact: Slightly less accurate dependence estimation
   - Note: Current approach works well in practice

2. **Bootstrap-calibrated q_thresh** - Computing Q* distribution during bootstrap
   - Current: Uses chi-squared(7, 0.99) = 18.48 fallback everywhere
   - Spec: Should compute 99th percentile of Q* from bootstrap
   - Impact: Slightly conservative model fit threshold
   - TODOs remain in code: `// TODO: use calibration.q_thresh`

3. **ThresholdElevated QualityIssue** - Emitting when theta_eff > theta_user
   - Current: Not implemented
   - Spec: Should emit QualityIssue when threshold elevated
   - Impact: Less visibility into threshold elevation

4. **CI-based stopping for research mode** - Full implementation
   - Current: Basic research status variants defined
   - Spec: Stopping when CI.lower > 1.1 * theta_floor
   - Impact: Research mode not fully operational

### Test File Fixes Needed

The following test files need `Outcome::Research(_)` match arms added:

- `tests/async_timing.rs` (4 match statements)
- `tests/calibration_coverage.rs` (1 match statement)
- `tests/calibration_estimation.rs` (1 match statement)
- `tests/calibration_bayesian.rs` (1 match statement)
- `tests/crypto_attacks.rs` (~16 match statements)
- `tests/discrete_mode.rs` (2 match statements)
- `tests/hash_timing.rs` (10 match statements)
- `tests/macro_tests.rs` (3 match statements)
- `tests/pqcrypto_timing.rs` (12 match statements)
- `tests/power_analysis.rs` (5 match statements)
- `tests/rsa_timing.rs` (6 match statements)
- `tests/rsa_vulnerability_assessment.rs` (1 match statement)
- `tests/aead_timing.rs` (8 match statements)
- `tests/aes_timing.rs` (6 match statements)
- `tests/ecc_timing.rs` (7 match statements)

Pattern to add after each `Outcome::Unmeasurable` arm:
```rust
Outcome::Research(_) => {}  // or return, (), etc. depending on context
```

### Benchmark Adapter Updates

- `benches/comp/adapters/timing_oracle_adapter.rs` needs Research variant

## Breaking Changes Summary

| Change | Impact | Migration |
|--------|--------|-----------|
| `Outcome::Research` variant | Match arms break | Add new arm |
| New `InconclusiveReason` variants | Match arms break | Add new arms |
| New `Calibration` fields | Struct init breaks | Update `new()` calls |
| New `Outcome` fields | Destructuring breaks | Update pattern matches |
| `ResearchStatus` in core | New dependency | Import from core |

## Notes

- Main package compiles successfully
- All examples compile successfully
- Test suite needs match arm updates (not blocking for library users)
- Research mode is defined but not fully integrated into adaptive loop
