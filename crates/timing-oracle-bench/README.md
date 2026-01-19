# timing-oracle-bench

Benchmark comparison infrastructure for evaluating timing-oracle against other timing side-channel detection tools.

## Overview

This crate provides:

1. **Synthetic data generation** - Create datasets with known statistical properties (null, shift, tail effects)
2. **Tool adapters** - Unified interface for comparing multiple detection tools
3. **Benchmark runner** - Automated FPR, power, and efficiency testing

## Methodology Comparison

| Adapter | Statistical Method | Assumptions | FPR Control | Handles Dependencies | Output |
|---------|-------------------|-------------|-------------|---------------------|--------|
| `TimingOracleAdapter` | Bayesian posterior + adaptive sampling | Non-parametric priors | Formal (posterior threshold) | Yes (covariance estimation) | P(leak), effect size, exploitability |
| `DudectAdapter` | Welch's t-test + percentile cropping | Gaussian after cropping | Threshold-based (&#124;t&#124; > 4.5) | No | Binary + t-statistic |
| `TimingTvlaAdapter` | Welch's t-test (no preprocessing) | Gaussian | Threshold-based (&#124;t&#124; > 4.5) | No | Binary + t-statistic |
| `KsTestAdapter` | Kolmogorov-Smirnov two-sample test | None (non-parametric) | Formal (α) | No | Binary + D statistic + p-value |
| `AndersonDarlingAdapter` | Anderson-Darling two-sample test | None (non-parametric) | Formal (α) | No | Binary + A² statistic + p-value |
| `MonaAdapter` | Crosby's box test (percentile boxes) | None (non-parametric) | None (deterministic) | No | Binary + box coordinates |
| `RtlfAdapter` | Bootstrap + quantile comparison | Non-parametric | Formal (α) | Yes (bootstrap) | Binary + p-value |
| `SilentAdapter` | Bootstrap + equivalence test | Non-parametric | Formal (α) | Yes (bootstrap) | Binary + test statistic |
| `TlsfuzzerAdapter` | Sign test + Wilcoxon | Paired samples | Formal (α) | Partial (pairing) | Binary + p-value |

### Key Differences

**Parametric vs Non-parametric:**
- *Parametric* (dudect, Timing-TVLA): Assume Gaussian distributions. Timing data is typically log-normal, so this assumption is often violated.
- *Non-parametric* (timing-oracle, RTLF, SILENT, Mona): Make minimal distribution assumptions.

**Dependency Handling:**
- Sequential timing measurements are correlated (CPU state, caches). Tools that ignore this (dudect, Timing-TVLA, Mona) may have inflated false positive rates.
- Bootstrap-based methods (RTLF, SILENT) and timing-oracle's covariance estimation handle dependencies.

**False Positive Rate (FPR) Control:**
- *Formal*: User specifies α, tool guarantees FPR ≤ α (timing-oracle, RTLF, SILENT, tlsfuzzer)
- *Threshold-based*: Fixed threshold (|t| > 4.5) with no formal FPR guarantee (dudect, Timing-TVLA)
- *None*: Deterministic test, FPR depends on data characteristics (Mona)

## Supported Tools

### Native Rust Implementations

| Adapter | Tool | Description |
|---------|------|-------------|
| `TimingOracleAdapter` | timing-oracle | Full Bayesian analysis with adaptive sampling |
| `DudectAdapter` | dudect | Welch's t-test with percentile cropping (Reparaz et al., 2017) |
| `TimingTvlaAdapter` | Timing-TVLA | Simplified TVLA for timing (single t-test, |t| > 4.5) |
| `KsTestAdapter` | KS test | Kolmogorov-Smirnov two-sample test |
| `AndersonDarlingAdapter` | AD test | Anderson-Darling two-sample test (tail-sensitive) |
| `MonaAdapter` | Mona | Crosby's box test (Crosby et al., 2009) |

### External Tool Adapters

These adapters wrap external tools and require additional dependencies:

| Adapter | Tool | Dependencies | Reference |
|---------|------|--------------|-----------|
| `RtlfAdapter` | RTLF | R + RTLF package | Dunsche et al., USENIX Security 2024 |
| `SilentAdapter` | SILENT | R + SILENT package | Dunsche et al., arXiv 2025 |
| `TlsfuzzerAdapter` | tlsfuzzer | Python + tlsfuzzer | Kario et al. |
| `RtlfDockerAdapter` | RTLF (Docker) | Docker/Podman | For isolated execution |

## Installation

### Native adapters only

```toml
[dependencies]
timing-oracle-bench = "0.1"
```

### With external tools (via devenv/Nix)

The external tools (RTLF, SILENT, tlsfuzzer) are packaged in `devenv.nix`:

```bash
# Enter development shell with all tools available
devenv shell

# Verify tools are available
rtlf --help
silent --help
python -m tlsfuzzer.analysis --help
```

## Usage

### Generate synthetic datasets

```rust
use timing_oracle_bench::{SyntheticConfig, EffectType, generate_dataset};

// Dataset with 5% mean shift (should detect leak)
let config = SyntheticConfig {
    samples_per_class: 30_000,
    effect: EffectType::Shift { percent: 5.0 },
    seed: 42,
    ..Default::default()
};

let dataset = generate_dataset(&config);
println!("Interleaved: {} samples", dataset.interleaved.len());
println!("Blocked: {} baseline, {} test",
    dataset.blocked.baseline.len(),
    dataset.blocked.test.len());
```

### Analyze with a single tool

```rust
use timing_oracle_bench::{TimingOracleAdapter, DudectAdapter, ToolAdapter};

let dataset = generate_dataset(&config);

// timing-oracle (uses interleaved data)
let to_adapter = TimingOracleAdapter::default();
let result = to_adapter.analyze(&dataset);
println!("timing-oracle: detected={}, P(leak)={:.4}",
    result.detected_leak,
    result.leak_probability.unwrap_or(0.0));

// dudect (uses blocked data)
let dudect = DudectAdapter::default();
let result = dudect.analyze(&dataset);
println!("dudect: detected={}, t={}",
    result.detected_leak,
    result.status);
```

### Run benchmark suite

```rust
use timing_oracle_bench::{
    BenchmarkRunner, TimingOracleAdapter, DudectAdapter,
    SyntheticConfig, EffectType,
};

let runner = BenchmarkRunner::new(vec![
    Box::new(TimingOracleAdapter::default()),
    Box::new(DudectAdapter::default()),
])
.datasets_per_config(100);

let configs = vec![
    ("null".into(), SyntheticConfig {
        effect: EffectType::Null,
        samples_per_class: 30_000,
        ..Default::default()
    }),
    ("shift-5pct".into(), SyntheticConfig {
        effect: EffectType::Shift { percent: 5.0 },
        samples_per_class: 30_000,
        ..Default::default()
    }),
];

let report = runner.run_full_suite(&configs);
println!("{}", report.to_markdown());
```

## Effect Types

| Effect | Description | Expected Detection |
|--------|-------------|-------------------|
| `Null` | Both classes identical | Should NOT detect (FPR test) |
| `Shift { percent }` | Mean shifted by X% | Should detect (power test) |
| `Tail` | Same mean, heavier tail | Tests tail sensitivity |
| `SameMean` | Same mean, different variance | Tests variance sensitivity |

## Data Formats

Tools receive data in their preferred format:

- **Interleaved**: `Vec<(Class, u64)>` - Random order, preserves temporal structure
- **Blocked**: `BlockedData { baseline: Vec<u64>, test: Vec<u64> }` - Grouped by class

```rust
// timing-oracle uses interleaved (exploits temporal correlations)
assert!(TimingOracleAdapter::default().uses_interleaved() == true);

// dudect/RTLF use blocked (traditional approach)
assert!(DudectAdapter::default().uses_interleaved() == false);
```

## External Tool Details

### RTLF (R)

Bootstrap-based percentile comparison from USENIX Security 2024.

```rust
use timing_oracle_bench::RtlfAdapter;

let rtlf = RtlfAdapter::new()
    .alpha(0.09)  // Significance level (default)
    .bootstrap_iterations(10_000);

let result = rtlf.analyze(&dataset);
// Exit codes: 10 = no leak, 11 = leak detected
```

### SILENT (R)

Bootstrap-based equivalence test from arXiv 2025.

```rust
use timing_oracle_bench::SilentAdapter;

let silent = SilentAdapter::new()
    .alpha(0.10)
    .delta(100);  // Relevant difference threshold

let result = silent.analyze(&dataset);
```

### tlsfuzzer (Python)

Sign test and Wilcoxon analysis from the tlsfuzzer project.

```rust
use timing_oracle_bench::TlsfuzzerAdapter;

let tlsfuzzer = TlsfuzzerAdapter::new()
    .alpha(0.05);

let result = tlsfuzzer.analyze(&dataset);
// Parses p-values from analysis output
```

### Mona / Crosby Box Test

Non-parametric test that checks for non-overlapping percentile boxes.

```rust
use timing_oracle_bench::MonaAdapter;

let mona = MonaAdapter::new()
    .min_box_size(5);  // Minimum box width in percentiles

let result = mona.analyze(&dataset);
// Returns the box [i%, j%] where distributions don't overlap
```

### Timing-TVLA

Simplified TVLA for timing analysis (single Welch's t-test).

```rust
use timing_oracle_bench::TimingTvlaAdapter;

let tvla = TimingTvlaAdapter::default();  // threshold = 4.5
// Or custom threshold:
let tvla = TimingTvlaAdapter::with_threshold(5.0);

let result = tvla.analyze(&dataset);
// result.status = "|t|=12.34, threshold=4.5"
```

**Note:** Timing-TVLA has known limitations for timing analysis (see [SILENT paper](https://arxiv.org/abs/2504.19821)):
- Assumes Gaussian distributions (timing data is typically log-normal)
- Ignores measurement dependencies (sequential samples are correlated)
- No formal FPR control (4.5 threshold is arbitrary)

For rigorous analysis, prefer RTLF, SILENT, or timing-oracle.

### Kolmogorov-Smirnov Test

Classic non-parametric two-sample test comparing empirical CDFs.

```rust
use timing_oracle_bench::KsTestAdapter;

let ks = KsTestAdapter::default();  // alpha = 0.05
// Or custom significance level:
let ks = KsTestAdapter::with_alpha(0.01);

let result = ks.analyze(&dataset);
// result.status = "D=0.1234, p=0.0456, alpha=0.05"
```

The KS test statistic D is the maximum absolute difference between the two empirical CDFs. It's sensitive to differences in location, scale, and shape, but less powerful than Anderson-Darling for tail differences.

### Anderson-Darling Test

Non-parametric two-sample test with increased sensitivity to tail differences.

```rust
use timing_oracle_bench::AndersonDarlingAdapter;

let ad = AndersonDarlingAdapter::default();  // alpha = 0.05
// Or custom significance level:
let ad = AndersonDarlingAdapter::with_alpha(0.01);

let result = ad.analyze(&dataset);
// result.status = "A²=3.4567, p=0.0123, alpha=0.05"
```

The Anderson-Darling test gives more weight to the tails of the distribution compared to KS, making it particularly useful for timing analysis where leaks often manifest as tail effects (occasional slow operations).

## References

- **timing-oracle**: This project
- **dudect**: Reparaz et al., ["Dude, is my code constant time?"](https://eprint.iacr.org/2016/1123) (USENIX Security 2017)
- **TVLA**: Goodwill et al., ["A testing methodology for side-channel resistance validation"](https://csrc.nist.gov/csrc/media/events/non-invasive-attack-testing-workshop/documents/08_goodwill.pdf) (NIST 2011) - originally for power analysis
- **Kolmogorov-Smirnov**: Smirnov, ["Table for estimating the goodness of fit of empirical distributions"](https://link.springer.com/article/10.1007/BF02288483) (1948)
- **Anderson-Darling**: Scholz & Stephens, ["K-sample Anderson-Darling tests"](https://www.jstor.org/stable/2288805) (JASA 1987)
- **RTLF**: Dunsche et al., ["With Great Power Come Great Side Channels"](https://www.usenix.org/conference/usenixsecurity24/presentation/dunsche) (USENIX Security 2024)
- **SILENT**: Dunsche et al., ["SILENT: A New Lens on Statistics in Software Timing Side Channels"](https://arxiv.org/abs/2504.19821) (arXiv 2025)
- **Mona/Box Test**: Crosby et al., ["Opportunities and Limits of Remote Timing Attacks"](https://www.cs.rice.edu/~dwallach/pub/crosby-timing2009.pdf) (2009)
- **tlsfuzzer**: Kario et al., [TLS timing analysis toolkit](https://tlsfuzzer.readthedocs.io/en/latest/timing-analysis.html)
- **t-test limitations**: Standaert, ["How (Not) to Use Welch's T-Test in Side-Channel Security Evaluations"](https://eprint.iacr.org/2017/138.pdf) (2017)

## License

MIT OR Apache-2.0
