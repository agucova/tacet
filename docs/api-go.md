# timing-oracle Go API Reference

Go bindings for timing-oracle. For conceptual overview, see [guide.md](guide.md). For other language bindings, see [api-rust.md](api-rust.md) or [api-c.md](api-c.md).

## Table of Contents

- [Quick Reference](#quick-reference)
- [Test Function](#test-function)
- [Configuration Options](#configuration-options)
  - [Attacker Models](#attacker-models)
  - [Budget Options](#budget-options)
  - [Threshold Options](#threshold-options)
  - [Advanced Options](#advanced-options)
- [Generators and Operations](#generators-and-operations)
  - [Generator Interface](#generator-interface)
  - [Operation Interface](#operation-interface)
  - [Built-in Generators](#built-in-generators)
- [Result Types](#result-types)
  - [Outcome](#outcome)
  - [Result](#result)
  - [Effect](#effect)
  - [Quality](#quality)
  - [Exploitability](#exploitability)
  - [InconclusiveReason](#inclusivereason)
- [Analyze Function](#analyze-function)
- [Information Functions](#information-functions)
- [Architecture Notes](#architecture-notes)

---

## Quick Reference

```go
package main

import (
    "fmt"
    "log"
    "time"

    "github.com/agucova/timing-oracle/go/timingoracle"
)

func main() {
    result, err := timingoracle.Test(
        timingoracle.NewZeroGenerator(0),  // Baseline: all zeros
        timingoracle.FuncOperation(func(input []byte) {
            myCryptoFunction(input)
        }),
        32,  // Input size in bytes
        timingoracle.WithAttacker(timingoracle.AdjacentNetwork),
        timingoracle.WithTimeBudget(30 * time.Second),
    )
    if err != nil {
        log.Fatal(err)
    }

    switch result.Outcome {
    case timingoracle.Pass:
        fmt.Printf("No leak: P(leak)=%.1f%%\n", result.LeakProbability*100)
    case timingoracle.Fail:
        fmt.Printf("Leak: %.1f ns, %s\n", result.Effect.TotalNs(), result.Exploitability)
    case timingoracle.Inconclusive:
        fmt.Printf("Inconclusive: %s\n", result.InconclusiveReason)
    case timingoracle.Unmeasurable:
        fmt.Printf("Too fast: %s\n", result.Recommendation)
    }
}
```

---

## Test Function

```go
func Test(gen Generator, op Operation, inputSize int, opts ...Option) (*Result, error)
```

Runs a timing side-channel analysis on the given operation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `gen` | `Generator` | Input generator for baseline and sample classes |
| `op` | `Operation` | The operation to test for timing leaks |
| `inputSize` | `int` | Size of input buffer in bytes |
| `opts` | `...Option` | Functional options for configuration |

**Returns:** `*Result` containing the analysis outcome, or an error if the test fails to start.

**Example:**
```go
result, err := timingoracle.Test(
    myGenerator,
    myOperation,
    32,
    timingoracle.WithAttacker(timingoracle.SharedHardware),
    timingoracle.WithTimeBudget(60 * time.Second),
    timingoracle.WithMaxSamples(50000),
)
```

---

## Configuration Options

All configuration uses the functional options pattern.

### Attacker Models

```go
func WithAttacker(model AttackerModel) Option
```

Sets the threat model. Choose based on your deployment scenario:

| Model | Threshold | Use Case |
|-------|-----------|----------|
| `SharedHardware` | 0.6 ns (~2 cycles) | SGX, containers, cross-VM, hyperthreading |
| `PostQuantum` | 3.3 ns (~10 cycles) | Post-quantum crypto (ML-KEM, ML-DSA) |
| `AdjacentNetwork` | 100 ns | LAN services, HTTP/2 APIs |
| `RemoteNetwork` | 50 μs | Internet-exposed services |
| `Research` | 0 | Detect any difference (not for CI) |

**Example:**
```go
// For a LAN-exposed service
timingoracle.WithAttacker(timingoracle.AdjacentNetwork)

// For SGX or container environment
timingoracle.WithAttacker(timingoracle.SharedHardware)

// Get the threshold for a model
threshold := timingoracle.AdjacentNetwork.ThresholdNs()  // 100.0
```

### Budget Options

```go
func WithTimeBudget(d time.Duration) Option
func WithMaxSamples(n int) Option
```

| Option | Default | Description |
|--------|---------|-------------|
| `WithTimeBudget` | 30 seconds | Maximum time to spend on the test |
| `WithMaxSamples` | 100,000 | Maximum samples per class |

**Example:**
```go
timingoracle.WithTimeBudget(60 * time.Second)
timingoracle.WithMaxSamples(50000)
```

### Threshold Options

```go
func WithCustomThreshold(thresholdNs float64) Option
func WithPassThreshold(p float64) Option
func WithFailThreshold(p float64) Option
```

| Option | Default | Description |
|--------|---------|-------------|
| `WithCustomThreshold` | Model-dependent | Override threshold in nanoseconds |
| `WithPassThreshold` | 0.05 | P(leak) below this → Pass |
| `WithFailThreshold` | 0.95 | P(leak) above this → Fail |

**Example:**
```go
// Custom threshold of 500 nanoseconds
timingoracle.WithCustomThreshold(500.0)

// More confident decisions
timingoracle.WithPassThreshold(0.01)   // Pass if P(leak) < 1%
timingoracle.WithFailThreshold(0.99)   // Fail if P(leak) > 99%
```

### Advanced Options

```go
func WithSeed(seed uint64) Option
func WithCalibrationSamples(n int) Option
func WithBatchSize(n int) Option
func WithoutAdaptiveBatching() Option
```

| Option | Default | Description |
|--------|---------|-------------|
| `WithSeed` | 0 (system entropy) | Random seed for reproducibility |
| `WithCalibrationSamples` | 5,000 | Samples for calibration phase |
| `WithBatchSize` | 1,000 | Samples per adaptive batch |
| `WithoutAdaptiveBatching` | enabled | Disable automatic batching |

---

## Generators and Operations

### Generator Interface

```go
type Generator interface {
    // Generate fills output with test data.
    // baseline=true for baseline class, false for sample class.
    Generate(output []byte, baseline bool)
}
```

Generators create test inputs. They are called **outside** the timed region.

### Operation Interface

```go
type Operation interface {
    // Execute runs the operation being tested.
    // This is called inside the timed region.
    Execute(input []byte)
}
```

Operations are the code being tested for timing leaks. They are called **inside** the timed region.

### Built-in Generators

```go
// NewZeroGenerator creates a generator that returns zeros for baseline
// and random data for sample class.
func NewZeroGenerator(seed uint64) Generator

// FuncGenerator adapts a function to the Generator interface.
func FuncGenerator(fn func(output []byte, baseline bool)) Generator
```

**FuncOperation** adapts a function to the Operation interface:
```go
// FuncOperation adapts a function to the Operation interface.
func FuncOperation(fn func(input []byte)) Operation
```

**Example:**
```go
// Using built-in generator
gen := timingoracle.NewZeroGenerator(0)

// Using function adapter
gen := timingoracle.FuncGenerator(func(output []byte, baseline bool) {
    if baseline {
        for i := range output {
            output[i] = 0
        }
    } else {
        rand.Read(output)
    }
})

// Operation adapter
op := timingoracle.FuncOperation(func(input []byte) {
    subtle.ConstantTimeCompare(input, secret)
})
```

---

## Result Types

### Outcome

```go
type Outcome int

const (
    Pass Outcome = iota        // No timing leak detected
    Fail                       // Timing leak confirmed
    Inconclusive              // Could not reach a decision
    Unmeasurable              // Operation too fast to measure
)
```

### Result

```go
type Result struct {
    Outcome            Outcome
    LeakProbability    float64           // P(max|δ| > θ | data), 0.0-1.0
    Effect             Effect            // Effect breakdown
    Quality            Quality           // Measurement quality
    SamplesUsed        int               // Samples collected per class
    ElapsedTime        time.Duration     // Total test duration
    Exploitability     Exploitability    // Risk assessment (for Fail)
    InconclusiveReason InconclusiveReason // (for Inconclusive)
    MDEShiftNs         float64           // Minimum detectable shift
    MDETailNs          float64           // Minimum detectable tail effect
    TimerResolutionNs  float64           // Timer resolution
    ThetaUserNs        float64           // Requested threshold
    ThetaEffNs         float64           // Effective threshold used
    Recommendation     string            // Guidance (for Inconclusive/Unmeasurable)
}

// Helper methods
func (r *Result) IsConclusive() bool   // True if Pass or Fail
func (r *Result) IsMeasurable() bool   // True if not Unmeasurable
func (r *Result) String() string       // Human-readable summary
```

### Effect

```go
type Effect struct {
    ShiftNs float64       // Uniform shift component (ns)
    TailNs  float64       // Tail effect component (ns)
    CILow   float64       // 95% CI lower bound (ns)
    CIHigh  float64       // 95% CI upper bound (ns)
    Pattern EffectPattern // Pattern classification
}

func (e Effect) TotalNs() float64  // ShiftNs + TailNs
```

### EffectPattern

```go
type EffectPattern int

const (
    UniformShift EffectPattern = iota  // Constant difference across quantiles
    TailEffect                         // Upper quantiles affected more
    Mixed                              // Both components present
    Indeterminate                      // Cannot determine pattern
)
```

### Quality

```go
type Quality int

const (
    Excellent Quality = iota  // MDE < 5 ns
    Good                      // MDE 5-20 ns
    Poor                      // MDE 20-100 ns
    TooNoisy                  // MDE > 100 ns
)
```

### Exploitability

```go
type Exploitability int

const (
    Negligible    Exploitability = iota  // < 100 ns - very difficult
    PossibleLAN                          // 100-500 ns - may be exploitable on LAN
    LikelyLAN                            // 500 ns - 20 μs - likely exploitable on LAN
    PossibleRemote                       // > 20 μs - potentially exploitable remotely
)
```

### InconclusiveReason

```go
type InconclusiveReason int

const (
    ReasonNone InconclusiveReason = iota
    ReasonDataTooNoisy           // Posterior ≈ prior
    ReasonNotLearning            // KL divergence collapsed
    ReasonWouldTakeTooLong       // Projected time exceeds budget
    ReasonTimeBudgetExceeded     // Time budget exhausted
    ReasonSampleBudgetExceeded   // Sample limit reached
    ReasonConditionsChanged      // Drift detected
    ReasonThresholdUnachievable  // θ_user < θ_floor at max budget
    ReasonModelMismatch          // 2D model doesn't fit
)
```

---

## Analyze Function

```go
func Analyze(baseline, sample []uint64, opts ...Option) (*Result, error)
```

Runs one-shot analysis on pre-collected timing data. Use this when timing samples have been collected separately.

| Parameter | Type | Description |
|-----------|------|-------------|
| `baseline` | `[]uint64` | Timing samples for baseline class (in timer ticks) |
| `sample` | `[]uint64` | Timing samples for sample class (in timer ticks) |
| `opts` | `...Option` | Configuration options |

**Note:** Timing data should be raw timer ticks, not nanoseconds. The library converts based on timer frequency.

---

## Information Functions

```go
func Version() string           // Library version (e.g., "0.1.0")
func TimerName() string         // Timer name (e.g., "rdtsc", "cntvct_el0")
func TimerFrequency() uint64    // Timer frequency in Hz
func TimerResolutionNs() float64 // Approximate resolution in nanoseconds
```

---

## Architecture Notes

### Performance Design

The Go library separates concerns for optimal performance:

1. **Measurement loop**: Pure Go with platform-specific assembly timers
2. **Statistical analysis**: Rust library via CGo (called only between batches)

This ensures no FFI overhead during timing-critical measurement. The CGo boundary is only crossed:
- Once after calibration
- Once per adaptive batch (~1,000 samples)

### Platform Timers

| Platform | Timer | Resolution |
|----------|-------|------------|
| x86_64 | `rdtsc` | ~0.3 ns |
| ARM64 Linux | `cntvct_el0` | ~1-40 ns (varies by SoC) |
| ARM64 macOS | `cntvct_el0` | ~41 ns |

PMU-based timers are not exposed in the Go API; use the standard timers which are sufficient for most use cases.

### Thread Safety

The `Test` function is safe to call from multiple goroutines concurrently. Each call maintains independent state.

---

## Example: Testing Crypto Operations

```go
package main

import (
    "crypto/subtle"
    "fmt"
    "log"
    "time"

    "github.com/agucova/timing-oracle/go/timingoracle"
)

var secret = []byte("super-secret-key-32-bytes-long!!")

func main() {
    result, err := timingoracle.Test(
        timingoracle.NewZeroGenerator(42),
        timingoracle.FuncOperation(func(input []byte) {
            // This should be constant-time
            subtle.ConstantTimeCompare(input, secret)
        }),
        32,
        timingoracle.WithAttacker(timingoracle.AdjacentNetwork),
        timingoracle.WithTimeBudget(30*time.Second),
    )
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(result)  // Uses String() method

    if result.Outcome == timingoracle.Fail {
        fmt.Printf("\nDetails:\n")
        fmt.Printf("  Effect: %.2f ns (shift) + %.2f ns (tail)\n",
            result.Effect.ShiftNs, result.Effect.TailNs)
        fmt.Printf("  95%% CI: [%.2f, %.2f] ns\n",
            result.Effect.CILow, result.Effect.CIHigh)
        fmt.Printf("  Exploitability: %s\n", result.Exploitability)
    }
}
```

---

## Installation

```bash
# The Go module automatically pulls the Rust FFI library
go get github.com/agucova/timing-oracle/go/timingoracle
```

**Requirements:**
- Go 1.21 or later
- CGo enabled
- The timing-oracle-go Rust library (built automatically via go generate)

For building the Rust component manually:
```bash
cd crates/timing-oracle-go
cargo build --release
```
