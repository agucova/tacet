<p align="center">
  <img src="https://raw.githubusercontent.com/agucova/tacet/main/website/public/logo-dark-bg.svg" alt="tacet-go" width="340" />
</p>

<p align="center">
  <strong>Go bindings for tacet – detect side channels in cryptographic code.</strong>
</p>

<p align="center">
  <a href="https://pkg.go.dev/github.com/agucova/tacet/crates/tacet-go"><img src="https://pkg.go.dev/badge/github.com/agucova/tacet/crates/tacet-go.svg" alt="Go Reference"></a>
  <a href="https://github.com/agucova/tacet/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
</p>

---

## Installation

```bash
go get github.com/agucova/tacet/crates/tacet-go
go run github.com/agucova/tacet/crates/tacet-go/cmd/tacet-install@latest
```

The install command downloads the pre-built static library for your platform (~12MB) and places it where CGo can find it. This only needs to be run once.

**Requirements:** Go 1.22+ with CGo enabled.

### Platform Support

| Platform | Architecture | Status |
|----------|--------------|--------|
| macOS | ARM64 (Apple Silicon) | ✅ Supported |
| macOS | AMD64 (Intel) | ✅ Supported |
| Linux | ARM64 | ✅ Supported |
| Linux | AMD64 | ✅ Supported |

The library is statically linked, so binaries are self-contained with no runtime dependencies.

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    "time"

    tacet "github.com/agucova/tacet/crates/tacet-go"
)

func main() {
    result, err := tacet.Test(
        tacet.NewZeroGenerator(0),
        tacet.FuncOperation(func(input []byte) {
            myCryptoFunction(input)
        }),
        32, // input size in bytes
        tacet.WithAttacker(tacet.AdjacentNetwork),
        tacet.WithTimeBudget(30*time.Second),
    )
    if err != nil {
        log.Fatal(err)
    }

    switch result.Outcome {
    case tacet.Pass:
        fmt.Printf("No leak detected (P=%.1f%%)\n", result.LeakProbability*100)
    case tacet.Fail:
        fmt.Printf("Timing leak: %s\n", result.Exploitability)
    case tacet.Inconclusive:
        fmt.Printf("Inconclusive: %s\n", result.InconclusiveReason)
    }
}
```

## Attacker Models

Choose based on your threat scenario:

| Model | Threshold | Use Case |
|-------|-----------|----------|
| `SharedHardware` | 0.6 ns | SGX, containers, cross-VM |
| `PostQuantum` | 3.3 ns | Post-quantum crypto |
| `AdjacentNetwork` | 100 ns | LAN, HTTP/2 APIs |
| `RemoteNetwork` | 50 μs | Internet-exposed services |
| `Research` | 0 | Detect any difference |

## Documentation

See the full [API documentation](https://tacet.sh/api/go/) or the [user guide](https://tacet.sh/guides/user-guide/).

## Building from Source

If you prefer to build the native library yourself instead of downloading pre-built binaries:

### Prerequisites

- [Rust toolchain](https://rustup.rs/) (stable)
- Go 1.21+ with CGo enabled
- C compiler (clang or gcc)

### Build Steps

```bash
# Clone the repository
git clone https://github.com/agucova/tacet
cd tacet

# Build the C library
cargo build -p tacet-c --release

# Strip debug symbols (reduces size from ~26MB to ~12MB)
strip -S target/release/libtacet_c.a  # macOS
# or: strip --strip-debug target/release/libtacet_c.a  # Linux

# Copy to the appropriate platform directory
mkdir -p crates/tacet-go/internal/ffi/lib/$(go env GOOS)_$(go env GOARCH)
cp target/release/libtacet_c.a \
   crates/tacet-go/internal/ffi/lib/$(go env GOOS)_$(go env GOARCH)/

# Verify it works
cd crates/tacet-go
go test -v -short -run TestTimerWorks
```

### Specifying a Version

To download a specific version of the library:

```bash
TIMING_ORACLE_VERSION=v0.1.0 go generate github.com/agucova/tacet/crates/tacet-go/...
```

### Verifying Your Build

After building or downloading, verify the library works:

```bash
cd crates/tacet-go

# Run a quick test
go test -v -short -run TestTimerWorks

# Run the example
go run ./examples/simple
```

Expected output:
```
Timer: cntvct_el0 (41.67 ns resolution)  # ARM64
# or
Timer: rdtsc (0.29 ns resolution)        # x86_64
```

## Architecture

The Go bindings use CGo to call a statically-linked Rust library:

```
┌─────────────────────────────────────────────┐
│  Your Go Code                               │
├─────────────────────────────────────────────┤
│  tacet (Go)                          │
│  - Pure Go measurement loop                 │
│  - Platform-specific timers (asm)           │
├─────────────────────────────────────────────┤
│  internal/ffi (CGo)                         │
│  - Calls Rust via C ABI                     │
├─────────────────────────────────────────────┤
│  libtacet_c.a (Rust, static)        │
│  - Bayesian statistical analysis            │
│  - Calibration and adaptive sampling        │
└─────────────────────────────────────────────┘
```

The timing-critical measurement loop runs in pure Go with platform-specific assembly timers (`rdtsc` on x86_64, `cntvct_el0` on ARM64). The Rust library is only called for statistical analysis between batches, minimizing FFI overhead.

## License

MIT
