# UniFFI Implementation Proposal for timing-oracle

## Executive Summary

This proposal outlines migrating timing-oracle's FFI layer from manual C bindings to [UniFFI](https://github.com/mozilla/uniffi-rs), Mozilla's multi-language bindings generator. This would provide:

- **Single source of truth**: Types defined once in Rust, generated for all targets
- **Reduced API drift**: No manual synchronization between Rust, C, and Go types
- **Future JS support**: Path to JavaScript/WASM bindings via uniffi-bindgen-js

**Target languages**: C++, Go, JavaScript (WASM)

---

## Binding Ecosystem Status

### Core UniFFI (Mozilla)

| Property | Status |
|----------|--------|
| Version | 0.28.3 stable, 0.29.4 latest |
| Maturity | Production (Firefox mobile/desktop) |
| Languages | Kotlin, Swift, Python, Ruby (official) |

### Third-Party Bindings

| Binding | Version | uniffi-rs Target | Maintainer | Maturity |
|---------|---------|------------------|------------|----------|
| **uniffi-bindgen-go** | v0.4.0 | 0.28.3 | NordSecurity | Young but active |
| **uniffi-bindgen-cpp** | v0.8.1 | 0.29.4 | NordSecurity | Active, C++20 required |
| **uniffi-bindgen-js** | v0.29.3-1 | ~0.29.x | Mozilla/community | Active, WASM support |

### Version Alignment Concern

The bindings target different uniffi-rs versions:
- Go: 0.28.3
- C++: 0.29.4
- JS: ~0.29.x

**Recommendation**: Pin to uniffi-rs 0.28.3 initially for Go compatibility, upgrade when uniffi-bindgen-go catches up.

---

## Architecture

### Current Architecture (Manual FFI)

```
timing-oracle-core (Rust)
        │
        ▼ manual conversion
timing-oracle-go/src/types.rs (Rust FFI structs)
        │
        ▼ manual sync
timing-oracle-go/include/timing_oracle_go.h (C header)
        │
        ▼ manual sync
go/timingoracle/internal/ffi/types.go (Go types)
        │
        ▼ manual conversion
go/timingoracle/result.go (Public Go types)
```

**Problem**: 4 places to update when types change. High drift risk.

### Proposed Architecture (UniFFI)

```
timing-oracle-core (Rust)
        │
        ▼ #[uniffi::export]
timing-oracle-uniffi (thin wrapper crate)
        │
        ├──► uniffi-bindgen-cpp ──► C++ headers + impl
        ├──► uniffi-bindgen-go  ──► Go package
        └──► uniffi-bindgen-js  ──► TypeScript + WASM
```

**Benefit**: Types defined once, generated everywhere.

---

## Type Mapping Analysis

### Types to Export

From `timing-oracle-core/src/result.rs` and `types.rs`:

#### Enums

| Rust Type | Complexity | UniFFI Support |
|-----------|------------|----------------|
| `Outcome` | High (enum with data) | ✅ Tagged unions |
| `InconclusiveReason` | High (enum with data) | ✅ Tagged unions |
| `EffectPattern` | Simple | ✅ Direct |
| `Exploitability` | Simple | ✅ Direct |
| `MeasurementQuality` | Simple | ✅ Direct |
| `AttackerModel` | Medium (Custom variant) | ✅ Tagged unions |
| `ResearchStatus` | Medium | ✅ Tagged unions |

#### Structs

| Rust Type | Fields | UniFFI Support |
|-----------|--------|----------------|
| `EffectEstimate` | 5 fields, tuple | ✅ With adjustment* |
| `Diagnostics` | ~40 fields | ✅ Direct |
| `ResearchOutcome` | 10 fields | ✅ Direct |
| `TopQuantile` | 4 fields | ✅ Direct |
| `QualityIssue` | 3 fields | ✅ Direct |

*UniFFI doesn't support tuples directly; `credible_interval_ns: (f64, f64)` needs a wrapper struct.

#### Potential Issues

1. **Tuple fields**: `credible_interval_ns: (f64, f64)` → Need `struct CredibleInterval { low: f64, high: f64 }`

2. **Option<Vec<T>>**: `top_quantiles: Option<Vec<TopQuantile>>` → Supported but check bindgen output

3. **Large enums**: `Outcome` has 5 variants with many fields → Works but generated code may be verbose

4. **String in enums**: `InconclusiveReason` variants contain `String` → Supported

---

## Implementation Plan

### Phase 1: Create uniffi Wrapper Crate (1-2 days)

Create `crates/timing-oracle-uniffi/`:

```rust
// src/lib.rs
use timing_oracle_core::result::*;
use timing_oracle_core::types::*;

// Re-export with UniFFI attributes
uniffi::setup_scaffolding!();

/// Credible interval bounds (UniFFI doesn't support tuples)
#[derive(uniffi::Record)]
pub struct CredibleInterval {
    pub low: f64,
    pub high: f64,
}

/// Effect estimate with UniFFI-compatible types
#[derive(uniffi::Record)]
pub struct EffectEstimateFFI {
    pub shift_ns: f64,
    pub tail_ns: f64,
    pub credible_interval: CredibleInterval,
    pub pattern: EffectPattern,
    pub interpretation_caveat: Option<String>,
}

// ... conversion impls ...

#[derive(uniffi::Enum)]
pub enum OutcomeFFI {
    Pass { /* fields */ },
    Fail { /* fields */ },
    Inconclusive { /* fields */ },
    Unmeasurable { /* fields */ },
    Research { outcome: ResearchOutcomeFFI },
}

/// Main analysis function
#[uniffi::export]
pub fn analyze(
    baseline: Vec<u64>,
    sample: Vec<u64>,
    config: ConfigFFI,
) -> Result<OutcomeFFI, AnalysisError> {
    // Convert inputs, call core, convert outputs
}
```

**Cargo.toml**:
```toml
[package]
name = "timing-oracle-uniffi"
version = "0.1.0"

[lib]
crate-type = ["cdylib", "staticlib"]

[dependencies]
timing-oracle-core = { path = "../timing-oracle-core" }
uniffi = { version = "0.28.3", features = ["cli"] }

[build-dependencies]
uniffi = { version = "0.28.3", features = ["build"] }
```

### Phase 2: Generate C++ Bindings (0.5 days)

```bash
cargo install uniffi-bindgen-cpp \
  --git https://github.com/NordSecurity/uniffi-bindgen-cpp \
  --tag v0.7.1+v0.28.3

uniffi-bindgen-cpp \
  --library target/release/libtiming_oracle_uniffi.a \
  --out-dir bindings/cpp
```

**Output structure**:
```
bindings/cpp/
├── timing_oracle_uniffi.hpp  # C++ header with classes
├── timing_oracle_uniffi.cpp  # Implementation
└── timing_oracle_uniffi.h    # C scaffolding
```

**C++ API feel**:
```cpp
#include "timing_oracle_uniffi.hpp"

auto config = timing_oracle::Config::adjacent_network();
auto result = timing_oracle::analyze(baseline, sample, config);

if (auto* pass = std::get_if<timing_oracle::Outcome::Pass>(&result)) {
    std::cout << "Leak probability: " << pass->leak_probability << std::endl;
}
```

### Phase 3: Generate Go Bindings (0.5 days)

```bash
cargo install uniffi-bindgen-go \
  --git https://github.com/NordSecurity/uniffi-bindgen-go \
  --tag v0.4.0+v0.28.3

uniffi-bindgen-go \
  --library target/release/libtiming_oracle_uniffi.a \
  --out-dir bindings/go
```

**Output structure**:
```
bindings/go/
└── timingoracle/
    ├── timingoracle.go     # Go bindings
    └── timingoracle.c      # CGo scaffolding
```

**Go API feel**:
```go
import "github.com/example/timing-oracle/bindings/go/timingoracle"

config := timingoracle.ConfigAdjacentNetwork()
result, err := timingoracle.Analyze(baseline, sample, config)
if err != nil {
    log.Fatal(err)
}

switch r := result.(type) {
case *timingoracle.OutcomePass:
    fmt.Printf("Leak probability: %.2f%%\n", r.LeakProbability*100)
case *timingoracle.OutcomeFail:
    fmt.Printf("Timing leak detected: %s\n", r.Exploitability)
}
```

### Phase 4: Generate JavaScript/WASM Bindings (1 day)

Using [uniffi-bindgen-react-native](https://github.com/aspect-build/aspect-frameworks) (being renamed to uniffi-bindgen-js):

```bash
# Install
cargo install uniffi-bindgen-react-native

# Generate WASM bindings
uniffi-bindgen-react-native \
  --library target/wasm32-unknown-unknown/release/libtiming_oracle_uniffi.a \
  --language wasm \
  --out-dir bindings/js
```

**Output structure**:
```
bindings/js/
├── src/
│   ├── index.ts           # TypeScript API
│   └── timing_oracle.wasm # WASM binary
├── package.json
└── tsconfig.json
```

**JavaScript API feel**:
```typescript
import { analyze, Config, Outcome } from 'timing-oracle';

const config = Config.adjacentNetwork();
const result = await analyze(baseline, sample, config);

if (result.type === 'Pass') {
    console.log(`Leak probability: ${result.leakProbability * 100}%`);
} else if (result.type === 'Fail') {
    console.log(`Timing leak: ${result.exploitability}`);
}
```

### Phase 5: Build System Integration (0.5 days)

**Makefile / justfile**:
```makefile
.PHONY: bindings bindings-cpp bindings-go bindings-js

bindings: bindings-cpp bindings-go bindings-js

bindings-cpp:
	cargo build --release -p timing-oracle-uniffi
	uniffi-bindgen-cpp --library target/release/libtiming_oracle_uniffi.a --out-dir bindings/cpp

bindings-go:
	uniffi-bindgen-go --library target/release/libtiming_oracle_uniffi.a --out-dir bindings/go

bindings-js:
	cargo build --release -p timing-oracle-uniffi --target wasm32-unknown-unknown
	uniffi-bindgen-react-native --library target/wasm32/release/libtiming_oracle_uniffi.a --language wasm --out-dir bindings/js
```

**CI integration**:
```yaml
# .github/workflows/bindings.yml
jobs:
  generate-bindings:
    steps:
      - run: make bindings
      - run: diff -r bindings/ bindings.expected/  # Detect drift
```

---

## Migration Strategy

### Option A: Big Bang (Recommended for this project)

1. Create uniffi wrapper crate
2. Generate all bindings
3. Update Go package to use generated bindings
4. Deprecate manual C bindings
5. Remove old FFI crates

**Pros**: Clean cut, no maintenance of two systems
**Cons**: More upfront work, risk if UniFFI has issues

### Option B: Gradual Migration

1. Create uniffi wrapper alongside existing FFI
2. Generate Go bindings, compare with manual
3. Switch Go to uniffi when validated
4. Add C++/JS bindings
5. Eventually remove manual FFI

**Pros**: Lower risk, can fall back
**Cons**: Maintaining two systems temporarily

---

## Risk Assessment

### High Confidence

- **C++ bindings**: uniffi-bindgen-cpp is mature, NordSecurity uses in production
- **Basic types**: Enums, structs, primitives all well-supported

### Medium Confidence

- **Go bindings**: v0.4.0 is young, but actively maintained by NordSecurity
- **Complex enums**: `Outcome` with 5 variants should work, but test thoroughly

### Lower Confidence

- **JS/WASM**: uniffi-bindgen-js is newer, less battle-tested
- **Version alignment**: May need to coordinate upgrades across 3 bindgen tools

### Mitigations

1. **Extensive testing**: Port existing Go tests to verify generated bindings
2. **Pin versions**: Lock uniffi-rs and all bindgen tools to compatible versions
3. **CI checks**: Fail CI if generated code differs from committed bindings
4. **Fallback plan**: Keep manual FFI code in a branch for 1-2 releases

---

## Effort Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| 1. UniFFI wrapper crate | 1-2 days | None |
| 2. C++ bindings | 0.5 days | Phase 1 |
| 3. Go bindings | 0.5 days | Phase 1 |
| 4. JS/WASM bindings | 1 day | Phase 1 |
| 5. Build integration | 0.5 days | Phases 2-4 |
| 6. Testing & validation | 1-2 days | All |
| **Total** | **5-7 days** | |

---

## Recommendation

**Proceed with UniFFI migration** using the Big Bang approach:

1. The type complexity is manageable (mostly straightforward structs/enums)
2. All three target languages have active binding generators
3. The single-source-of-truth benefit is significant for long-term maintenance
4. Mozilla's production use provides confidence in the core

**Key actions**:
1. Pin to uniffi-rs 0.28.3 for Go compatibility
2. Start with Go bindings (your current pain point)
3. Add C++ and JS incrementally
4. Maintain comprehensive tests to catch binding issues

---

## References

- [UniFFI Documentation](https://mozilla.github.io/uniffi-rs/)
- [uniffi-bindgen-go](https://github.com/NordSecurity/uniffi-bindgen-go)
- [uniffi-bindgen-cpp](https://github.com/NordSecurity/uniffi-bindgen-cpp)
- [uniffi-bindgen-react-native](https://github.com/aspect-build/aspect-frameworks) (uniffi-bindgen-js)
- [Mozilla Hacks: UniFFI JS](https://hacks.mozilla.org/2023/08/autogenerating-rust-js-bindings-with-uniffi/)
