# timing-oracle-go

Go bindings for [timing-oracle](https://github.com/agucova/timing-oracle), a library for detecting timing side channels in cryptographic code.

## Installation

```bash
go get github.com/agucova/timing-oracle/crates/timing-oracle-go
go generate github.com/agucova/timing-oracle/crates/timing-oracle-go/...
```

The `go generate` command downloads the pre-built static library for your platform (~12MB). This only needs to be run once.

**Requirements:** Go 1.21+ with CGo enabled.

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

    timingoracle "github.com/agucova/timing-oracle/crates/timing-oracle-go"
)

func main() {
    result, err := timingoracle.Test(
        timingoracle.NewZeroGenerator(0),
        timingoracle.FuncOperation(func(input []byte) {
            myCryptoFunction(input)
        }),
        32, // input size in bytes
        timingoracle.WithAttacker(timingoracle.AdjacentNetwork),
        timingoracle.WithTimeBudget(30*time.Second),
    )
    if err != nil {
        log.Fatal(err)
    }

    switch result.Outcome {
    case timingoracle.Pass:
        fmt.Printf("No leak detected (P=%.1f%%)\n", result.LeakProbability*100)
    case timingoracle.Fail:
        fmt.Printf("Timing leak: %s\n", result.Exploitability)
    case timingoracle.Inconclusive:
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

See the full [API documentation](https://timing-oracle.dev/api/go/) or the [user guide](https://timing-oracle.dev/guides/user-guide/).

## Building from Source

If you prefer to build the native library yourself instead of downloading pre-built binaries:

### Prerequisites

- [Rust toolchain](https://rustup.rs/) (stable)
- Go 1.21+ with CGo enabled
- C compiler (clang or gcc)

### Build Steps

```bash
# Clone the repository
git clone https://github.com/agucova/timing-oracle
cd timing-oracle

# Build the C library
cargo build -p timing-oracle-c --release

# Strip debug symbols (reduces size from ~26MB to ~12MB)
strip -S target/release/libtiming_oracle_c.a  # macOS
# or: strip --strip-debug target/release/libtiming_oracle_c.a  # Linux

# Copy to the appropriate platform directory
mkdir -p crates/timing-oracle-go/internal/ffi/lib/$(go env GOOS)_$(go env GOARCH)
cp target/release/libtiming_oracle_c.a \
   crates/timing-oracle-go/internal/ffi/lib/$(go env GOOS)_$(go env GOARCH)/

# Verify it works
cd crates/timing-oracle-go
go test -v -short -run TestTimerWorks
```

### Specifying a Version

To download a specific version of the library:

```bash
TIMING_ORACLE_VERSION=v0.1.0 go generate github.com/agucova/timing-oracle/crates/timing-oracle-go/...
```

### Verifying Your Build

After building or downloading, verify the library works:

```bash
cd crates/timing-oracle-go

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
│  timingoracle (Go)                          │
│  - Pure Go measurement loop                 │
│  - Platform-specific timers (asm)           │
├─────────────────────────────────────────────┤
│  internal/ffi (CGo)                         │
│  - Calls Rust via C ABI                     │
├─────────────────────────────────────────────┤
│  libtiming_oracle_c.a (Rust, static)        │
│  - Bayesian statistical analysis            │
│  - Calibration and adaptive sampling        │
└─────────────────────────────────────────────┘
```

The timing-critical measurement loop runs in pure Go with platform-specific assembly timers (`rdtsc` on x86_64, `cntvct_el0` on ARM64). The Rust library is only called for statistical analysis between batches, minimizing FFI overhead.

## License

MIT
