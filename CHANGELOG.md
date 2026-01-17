# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Spec §3.3.2: Block length selection now uses class-conditional ACF**
  - Previous: Politis-White on pooled acquisition stream (anti-conservative due to class alternation)
  - New: Class-conditional acquisition-lag ACF with conservative combination
  - Added safety floor (b_min = 10) and inflation factor for fragile regimes
  - Fixes elevated FPR (5-8% → 2-5% expected) under null hypothesis

- **Spec §3.8: Strengthened calibration validation requirements**
  - Added explicit FPR metrics: FPR_gated and FPR_overall
  - Recommended 500+ trials (up from 100) for stable estimates
  - Added normative acceptance criteria (FPR_gated ≤ 5%, FPR_overall ≤ 10%)
  - Added anti-conservative remediation escalation steps

### Documentation

- **Major restructure (spec v4.2 → v5.0):**
  - Specification (`docs/spec.md`) refactored to be language-agnostic with RFC 2119 terminology
  - Added Abstract Types section (§2) with pseudocode ADT definitions
  - Added API Design Principles section (§5) for preventing user mistakes
  - Moved implementation details to new `docs/implementation-guide.md`
  - Moved detailed interpretation guidance to `docs/guide.md`
  - Renamed `docs/api-reference.md` to `docs/api-rust.md`
  - Added `docs/api-c.md` for C/C++ bindings
  - Added `docs/api-go.md` for Go bindings

## [0.1.0] - 2025-01-05

### Added

- Initial release
- `TimingOracle` builder with configurable presets (`new`, `balanced`, `quick`, `calibration`)
- `timing_test!` macro - returns `TestResult` directly, panics if unmeasurable
- `timing_test_checked!` macro - returns `Outcome` for explicit unmeasurable handling
- Two-layer statistical analysis:
  - CI gate (frequentist): max-statistic bootstrap with bounded false positive rate
  - Bayesian layer: posterior probability of timing leak via Bayes factor
- Effect decomposition into `UniformShift`, `TailEffect`, or `Mixed` patterns
- Exploitability assessment: `Negligible`, `PossibleLAN`, `LikelyLAN`, `PossibleRemote`
- Measurement quality metrics and minimum detectable effect (MDE) estimation
- `InputPair` helper for generating baseline/sample test inputs
- `skip_if_unreliable!` and `require_reliable!` macros for handling noisy environments
- Adaptive batching for platforms with coarse timer resolution
- Preflight checks for measurement validation

### Features

- `parallel` (default) - Rayon-based parallel bootstrap (4-8x speedup)
- `kperf` (default, macOS ARM64) - PMU-based cycle counting via kperf (~1ns resolution, requires sudo)
- `perf` (default, Linux) - perf_event cycle counting (~1ns resolution, requires sudo/CAP_PERFMON)
- `macros` - Proc macros for ergonomic test syntax

### Platform Support

| Platform | Timer | Resolution |
|----------|-------|------------|
| x86_64 | rdtsc | ~0.3 ns |
| Apple Silicon | kperf | ~1 ns (with sudo) |
| Apple Silicon | cntvct | ~41 ns (fallback) |
| Linux ARM | perf_event | ~1 ns (with sudo) |
