# timing-oracle-c

C bindings for [timing-oracle](../timing-oracle/), a Bayesian timing side-channel detection library.

This crate provides a C API for detecting timing side channels in cryptographic code. It wraps the core Rust library and exposes a simple callback-based interface suitable for C, C++, and other languages with C FFI support.

For the underlying statistical methodology and theory, see the [main library documentation](../../README.md) and [specification](../../docs/spec.md).

## Quick Start

```c
#include <timing_oracle.h>
#include <string.h>
#include <stdlib.h>

static uint8_t secret[32];

// Generator: zeros for baseline, random for sample
void generator(void *ctx, bool is_baseline, uint8_t *out, size_t size) {
    if (is_baseline) {
        memset(out, 0, size);
    } else {
        arc4random_buf(out, size);  // Or read from /dev/urandom
    }
}

// Operation: the code being tested for timing leaks
void operation(void *ctx, const uint8_t *input, size_t size) {
    // Your cryptographic operation here
    my_crypto_function(input, secret, size);
}

int main(void) {
    // Create configuration
    to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
    config.time_budget_secs = 30.0;

    // Run test
    to_result_t result = to_test(&config, 32, generator, operation, NULL);

    // Check result
    if (result.outcome == TO_OUTCOME_FAIL) {
        printf("Timing leak detected! P(leak)=%.1f%%\n",
               result.leak_probability * 100.0);
    }

    // Clean up
    to_result_free(&result);
    return result.outcome == TO_OUTCOME_FAIL ? 1 : 0;
}
```

## Building

### From Source

```bash
# Build the library (from repository root)
cargo build --release -p timing-oracle-c

# Output files:
#   target/release/libtiming_oracle_c.a      (static library)
#   target/release/libtiming_oracle_c.dylib  (macOS dynamic library)
#   target/release/libtiming_oracle_c.so     (Linux dynamic library)
```

### Linking

**macOS:**
```bash
cc -I path/to/include your_code.c -L path/to/lib -ltiming_oracle_c \
   -framework Security -framework CoreFoundation -liconv
```

**Linux:**
```bash
cc -I path/to/include your_code.c -L path/to/lib -ltiming_oracle_c \
   -lpthread -ldl -lm
```

## API Overview

### Configuration

Create a configuration with `to_config_default()`, then customize as needed:

```c
to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
config.time_budget_secs = 60.0;    // Max test duration
config.max_samples = 50000;        // Max samples per class
config.pass_threshold = 0.01;      // More confident pass (P < 1%)
config.fail_threshold = 0.99;      // More confident fail (P > 99%)
```

### Attacker Models

Choose based on your threat model:

| Model | Threshold | Use Case |
|-------|-----------|----------|
| `TO_ATTACKER_SHARED_HARDWARE` | ~0.6 ns | SGX, containers, hyperthreading |
| `TO_ATTACKER_POST_QUANTUM` | ~3.3 ns | Post-quantum crypto |
| `TO_ATTACKER_ADJACENT_NETWORK` | 100 ns | LAN, HTTP/2 APIs |
| `TO_ATTACKER_REMOTE_NETWORK` | 50 μs | Public internet |
| `TO_ATTACKER_RESEARCH` | ~0 | Detect any difference |
| `TO_ATTACKER_CUSTOM` | User-defined | Specific requirements |

### Callbacks

The library uses two callbacks for test execution:

**Generator** (called outside timed region):
```c
void generator(void *ctx, bool is_baseline, uint8_t *output, size_t size);
```
- `is_baseline = true`: Generate baseline input (typically all zeros)
- `is_baseline = false`: Generate sample input (typically random)

**Operation** (called inside timed region):
```c
void operation(void *ctx, const uint8_t *input, size_t size);
```
- The cryptographic operation being tested
- Use `volatile` or compiler barriers to prevent optimization

### Running Tests

```c
// Standard test (automatic time tracking)
to_result_t result = to_test(&config, input_size, generator, operation, context);

// For no_std/embedded (external time tracking)
double elapsed = 0.0;
to_result_t result = to_test_with_time(&config, input_size, generator, operation,
                                        context, &elapsed);
```

### Handling Results

```c
switch (result.outcome) {
    case TO_OUTCOME_PASS:
        // No timing leak detected
        printf("P(leak) = %.2f%%\n", result.leak_probability * 100.0);
        break;

    case TO_OUTCOME_FAIL:
        // Timing leak confirmed
        printf("Leak: %.1f ns, exploitability: %s\n",
               result.effect.shift_ns,
               to_exploitability_str(result.exploitability));
        break;

    case TO_OUTCOME_INCONCLUSIVE:
        // Could not determine
        printf("Reason: %s\n",
               to_inconclusive_reason_str(result.inconclusive_reason));
        break;

    case TO_OUTCOME_UNMEASURABLE:
        // Operation too fast
        printf("Operation: %.1f ns (timer: %.1f ns resolution)\n",
               result.operation_ns, result.timer_resolution_ns);
        break;
}

// Always free the result
to_result_free(&result);
```

### String Utilities

Convert enum values to human-readable strings:

```c
to_outcome_str(result.outcome);           // "Pass", "Fail", etc.
to_quality_str(result.quality);           // "Excellent", "Good", etc.
to_exploitability_str(result.exploitability);
to_effect_pattern_str(result.effect.pattern);
to_inconclusive_reason_str(result.inconclusive_reason);
```

### Information Functions

```c
const char *version = to_version();           // e.g., "0.1.0"
const char *timer = to_timer_name();          // e.g., "rdtsc"
uint64_t freq = to_timer_frequency();         // Timer frequency in Hz
```

## Memory Management

**What to free:**
- Call `to_result_free(&result)` after processing any result
- The `recommendation` field is heap-allocated and freed by `to_result_free()`

**What NOT to free:**
- `timer_name`, `platform` - static strings
- Return values from `to_*_str()` functions - static strings

**Safe to call:**
- `to_result_free(NULL)` - no-op
- `to_result_free(&result)` multiple times - safe

## Platform Considerations

### Timers

The library supports multiple timer backends with automatic selection:

| Platform | Standard Timer | PMU Timer | PMU Resolution |
|----------|----------------|-----------|----------------|
| x86_64 Linux | `rdtsc` (~0.3 ns) | `perf_event` (~0.3 ns) | Requires sudo or CAP_PERFMON |
| ARM64 Linux | `cntvct_el0` (~1 ns) | `perf_event` (~0.3 ns) | Requires sudo or CAP_PERFMON |
| ARM64 macOS | `cntvct_el0` (~42 ns) | `kperf` (~0.3 ns) | Requires sudo |
| Other | `clock_gettime` (~1 μs) | N/A | N/A |

### Timer Preference

Control timer selection via the configuration:

```c
to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);

// Use PMU timer if available, otherwise fall back to standard (default)
config.timer_preference = TO_TIMER_AUTO;

// Always use standard timer (rdtsc/cntvct_el0), skip PMU detection
config.timer_preference = TO_TIMER_STANDARD;

// Require PMU timer, fail if unavailable
config.timer_preference = TO_TIMER_PREFER_PMU;
```

### Enabling PMU Timers

PMU timers provide the highest precision (~0.3 ns) but require elevated privileges:

**Linux (perf_event):**
```bash
# Option 1: Run with sudo
sudo ./your_program

# Option 2: Lower perf_event_paranoid (system-wide, persists until reboot)
echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid

# Option 3: Grant CAP_PERFMON to specific binary (persists)
sudo setcap cap_perfmon+ep ./your_program
```

**macOS (kperf):**
```bash
# kperf requires root - no alternative
sudo ./your_program
```

### Adaptive Batching

On platforms with coarse timer resolution, the library automatically batches operations together to achieve measurable timing differences. This is transparent to the caller.

## Examples

See the [`examples/`](examples/) directory:

| Example | Description |
|---------|-------------|
| [`simple.c`](examples/simple.c) | Basic usage |
| [`compare.c`](examples/compare.c) | Leaky vs constant-time comparison |
| [`attacker_models.c`](examples/attacker_models.c) | Threat model selection |
| [`result_handling.c`](examples/result_handling.c) | Complete result interpretation |
| [`custom_config.c`](examples/custom_config.c) | Configuration options |
| [`no_std_embedded.c`](examples/no_std_embedded.c) | External time tracking |
| [`crypto_hmac.c`](examples/crypto_hmac.c) | HMAC verification (OpenSSL) |

Build examples:
```bash
cd examples
make                    # Build core examples
make WITH_OPENSSL=1     # Include OpenSSL example
```

## C++ Support

For C++17 and later, a header-only wrapper is available:

```cpp
#include <timing_oracle.hpp>

auto result = timing_oracle::test(
    timing_oracle::AttackerModel::AdjacentNetwork,
    32,
    [](auto& output, bool baseline) {
        // Generator
    },
    [](const auto& input) {
        // Operation
    }
);
```

See [`include/timing_oracle.hpp`](include/timing_oracle.hpp) for details.

## See Also

- [Main library documentation](../../README.md)
- [API specification](../../docs/spec.md)
- [Bayesian methodology](../../docs/bayesian-layer.md)
- [Testing patterns](../../docs/testing-patterns.md)
