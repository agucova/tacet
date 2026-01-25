# timing-oracle-go

Go bindings for [timing-oracle](https://github.com/agucova/timing-oracle), a library for detecting timing side channels in cryptographic code.

## Installation

```bash
go get github.com/agucova/timing-oracle/crates/timing-oracle-go
```

Pre-built static libraries are included for:
- macOS ARM64 (Apple Silicon)
- macOS AMD64 (Intel)
- Linux ARM64
- Linux AMD64

No Rust toolchain required.

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

See the full [Go API documentation](https://github.com/agucova/timing-oracle/blob/main/docs/api-go.md).

## Building from Source

If you need to build the native library yourself:

```bash
# Requires Rust toolchain
cargo build -p timing-oracle-c --release

# Strip debug symbols
strip -S target/release/libtiming_oracle_c.a

# Copy to the appropriate platform directory
cp target/release/libtiming_oracle_c.a \
   crates/timing-oracle-go/internal/ffi/lib/$(go env GOOS)_$(go env GOARCH)/
```

## License

MIT
