# timing-oracle C/C++ API Reference

C and C++ bindings for timing-oracle. For conceptual overview, see [guide.md](guide.md). For other language bindings, see [api-rust.md](api-rust.md) or [api-go.md](api-go.md).

## Table of Contents

- [Quick Reference](#quick-reference)
- [Configuration](#configuration)
  - [Creating Configuration](#creating-configuration)
  - [Attacker Models](#attacker-models)
  - [Configuration Fields](#configuration-fields)
- [Callbacks](#callbacks)
  - [Generator Callback](#generator-callback)
  - [Operation Callback](#operation-callback)
- [Running Tests](#running-tests)
- [Result Handling](#result-handling)
  - [Outcome Types](#outcome-types)
  - [Effect Estimate](#effect-estimate)
  - [Exploitability](#exploitability)
  - [Quality Assessment](#quality-assessment)
- [String Utilities](#string-utilities)
- [Information Functions](#information-functions)
- [Memory Management](#memory-management)
- [Platform Considerations](#platform-considerations)
- [C++ Wrapper](#c-wrapper)

---

## Quick Reference

```c
#include <timing_oracle.h>

// Generator: zeros for baseline, random for sample
void generator(void *ctx, bool is_baseline, uint8_t *out, size_t size) {
    if (is_baseline) {
        memset(out, 0, size);
    } else {
        arc4random_buf(out, size);
    }
}

// Operation: the code being tested
void operation(void *ctx, const uint8_t *input, size_t size) {
    my_crypto_function(input, size);
}

int main(void) {
    to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
    config.time_budget_secs = 30.0;

    to_result_t result = to_test(&config, 32, generator, operation, NULL);

    switch (result.outcome) {
        case TO_OUTCOME_PASS:
            printf("No leak: P(leak)=%.1f%%\n", result.leak_probability * 100.0);
            break;
        case TO_OUTCOME_FAIL:
            printf("Leak: %.1f ns\n", result.effect.shift_ns);
            break;
        case TO_OUTCOME_INCONCLUSIVE:
            printf("Inconclusive: %s\n", to_inconclusive_reason_str(result.inconclusive_reason));
            break;
        case TO_OUTCOME_UNMEASURABLE:
            printf("Too fast: %s\n", result.recommendation);
            break;
    }

    to_result_free(&result);
    return result.outcome == TO_OUTCOME_FAIL ? 1 : 0;
}
```

---

## Configuration

### Creating Configuration

```c
// Create with default settings for an attacker model
to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);

// Customize as needed
config.time_budget_secs = 60.0;
config.max_samples = 50000;
```

### Attacker Models

| Enum Value | Threshold | Use Case |
|------------|-----------|----------|
| `TO_ATTACKER_SHARED_HARDWARE` | ~0.6 ns | SGX, containers, hyperthreading |
| `TO_ATTACKER_POST_QUANTUM` | ~3.3 ns | Post-quantum crypto |
| `TO_ATTACKER_ADJACENT_NETWORK` | 100 ns | LAN, HTTP/2 APIs |
| `TO_ATTACKER_REMOTE_NETWORK` | 50 μs | Public internet |
| `TO_ATTACKER_RESEARCH` | ~0 | Detect any difference |
| `TO_ATTACKER_CUSTOM` | User-defined | Specific requirements |

For custom thresholds:
```c
to_config_t config = to_config_default(TO_ATTACKER_CUSTOM);
config.threshold_ns = 500.0;  // Custom threshold in nanoseconds
```

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `attacker_model` | `to_attacker_model_t` | Required | Threat model preset |
| `threshold_ns` | `double` | Model-dependent | Threshold in nanoseconds (for CUSTOM) |
| `time_budget_secs` | `double` | 30.0 | Maximum test duration |
| `max_samples` | `size_t` | 100000 | Maximum samples per class |
| `pass_threshold` | `double` | 0.05 | P(leak) below this → Pass |
| `fail_threshold` | `double` | 0.95 | P(leak) above this → Fail |
| `timer_preference` | `to_timer_preference_t` | AUTO | Timer selection mode |

---

## Callbacks

### Generator Callback

```c
typedef void (*to_generator_fn)(void *context, bool is_baseline, uint8_t *output, size_t size);
```

Called **outside** the timed region to generate test inputs.

| Parameter | Description |
|-----------|-------------|
| `context` | User-provided context pointer (from `to_test`) |
| `is_baseline` | `true` for baseline class, `false` for sample class |
| `output` | Buffer to write generated input |
| `size` | Size of output buffer in bytes |

**Example:**
```c
void generator(void *ctx, bool is_baseline, uint8_t *output, size_t size) {
    if (is_baseline) {
        memset(output, 0, size);  // All zeros for baseline
    } else {
        arc4random_buf(output, size);  // Random for sample
    }
}
```

### Operation Callback

```c
typedef void (*to_operation_fn)(void *context, const uint8_t *input, size_t size);
```

Called **inside** the timed region. This is the code being tested for timing leaks.

| Parameter | Description |
|-----------|-------------|
| `context` | User-provided context pointer (from `to_test`) |
| `input` | Input buffer (from generator) |
| `size` | Size of input buffer in bytes |

**Important:** Avoid compiler optimizations that might eliminate the operation:
```c
void operation(void *ctx, const uint8_t *input, size_t size) {
    volatile uint8_t result[32];
    my_crypto_function(input, result, size);
    // volatile prevents dead-code elimination
}
```

---

## Running Tests

### Standard Test

```c
to_result_t to_test(
    const to_config_t *config,
    size_t input_size,
    to_generator_fn generator,
    to_operation_fn operation,
    void *context
);
```

| Parameter | Description |
|-----------|-------------|
| `config` | Test configuration |
| `input_size` | Size of input buffer in bytes |
| `generator` | Input generator callback |
| `operation` | Operation to test |
| `context` | User context passed to callbacks |

### External Time Tracking

For embedded or no_std environments:

```c
to_result_t to_test_with_time(
    const to_config_t *config,
    size_t input_size,
    to_generator_fn generator,
    to_operation_fn operation,
    void *context,
    double *elapsed_secs
);
```

The `elapsed_secs` pointer is updated with the current elapsed time. Use this when the standard timer is unavailable.

---

## Result Handling

### Outcome Types

```c
typedef enum {
    TO_OUTCOME_PASS,         // No timing leak detected
    TO_OUTCOME_FAIL,         // Timing leak confirmed
    TO_OUTCOME_INCONCLUSIVE, // Could not determine
    TO_OUTCOME_UNMEASURABLE  // Operation too fast
} to_outcome_t;
```

### Result Structure

```c
typedef struct {
    to_outcome_t outcome;
    double leak_probability;      // P(leak > θ | data), 0.0-1.0
    to_effect_t effect;           // Effect breakdown
    to_exploitability_t exploitability;  // Risk assessment (for FAIL)
    to_quality_t quality;         // Measurement quality
    to_inconclusive_reason_t inconclusive_reason;  // (for INCONCLUSIVE)
    size_t samples_used;
    double theta_user;            // Requested threshold
    double theta_eff;             // Effective threshold used
    double theta_floor;           // Measurement floor
    double operation_ns;          // (for UNMEASURABLE)
    double timer_resolution_ns;   // (for UNMEASURABLE)
    const char *recommendation;   // (for UNMEASURABLE) - heap allocated
    const char *timer_name;       // Timer used - static string
    const char *platform;         // Platform name - static string
} to_result_t;
```

### Effect Estimate

```c
typedef struct {
    double shift_ns;              // Uniform shift component
    double tail_ns;               // Tail effect component
    double ci_lower_ns;           // 95% CI lower bound
    double ci_upper_ns;           // 95% CI upper bound
    to_effect_pattern_t pattern;  // Pattern classification
} to_effect_t;
```

**Effect Patterns:**

| Enum Value | Meaning |
|------------|---------|
| `TO_PATTERN_UNIFORM_SHIFT` | All quantiles shifted equally |
| `TO_PATTERN_TAIL_EFFECT` | Upper quantiles affected more |
| `TO_PATTERN_MIXED` | Both shift and tail present |
| `TO_PATTERN_COMPLEX` | Pattern doesn't fit 2D model |
| `TO_PATTERN_INDETERMINATE` | No significant effect |

### Exploitability

```c
typedef enum {
    TO_EXPLOITABILITY_SHARED_HARDWARE_ONLY,  // < 10 ns
    TO_EXPLOITABILITY_HTTP2_MULTIPLEXING,    // 10-100 ns
    TO_EXPLOITABILITY_STANDARD_REMOTE,       // 100 ns - 10 μs
    TO_EXPLOITABILITY_OBVIOUS_LEAK           // > 10 μs
} to_exploitability_t;
```

### Quality Assessment

```c
typedef enum {
    TO_QUALITY_EXCELLENT,  // MDE < 5 ns
    TO_QUALITY_GOOD,       // MDE 5-20 ns
    TO_QUALITY_POOR,       // MDE 20-100 ns
    TO_QUALITY_TOO_NOISY   // MDE > 100 ns
} to_quality_t;
```

### Inconclusive Reasons

```c
typedef enum {
    TO_INCONCLUSIVE_DATA_TOO_NOISY,
    TO_INCONCLUSIVE_NOT_LEARNING,
    TO_INCONCLUSIVE_WOULD_TAKE_TOO_LONG,
    TO_INCONCLUSIVE_THRESHOLD_UNACHIEVABLE,
    TO_INCONCLUSIVE_TIME_BUDGET_EXCEEDED,
    TO_INCONCLUSIVE_SAMPLE_BUDGET_EXCEEDED,
    TO_INCONCLUSIVE_CONDITIONS_CHANGED
} to_inconclusive_reason_t;
```

---

## String Utilities

Convert enum values to human-readable strings:

```c
const char *to_outcome_str(to_outcome_t outcome);
const char *to_quality_str(to_quality_t quality);
const char *to_exploitability_str(to_exploitability_t exploitability);
const char *to_effect_pattern_str(to_effect_pattern_t pattern);
const char *to_inconclusive_reason_str(to_inconclusive_reason_t reason);
```

All return static strings (do not free).

---

## Information Functions

```c
const char *to_version(void);       // Library version (e.g., "0.1.0")
const char *to_timer_name(void);    // Timer name (e.g., "rdtsc")
uint64_t to_timer_frequency(void);  // Timer frequency in Hz
```

---

## Memory Management

### What to Free

- **`to_result_free(&result)`** — Call after processing any result
- The `recommendation` field is heap-allocated and freed by `to_result_free()`

### What NOT to Free

- `timer_name`, `platform` — Static strings
- Return values from `to_*_str()` functions — Static strings

### Safe Calls

```c
to_result_free(NULL);       // No-op
to_result_free(&result);    // Safe to call multiple times
```

---

## Platform Considerations

### Timer Selection

Control timer selection via configuration:

```c
to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);

// Automatic selection (default) - uses PMU if available
config.timer_preference = TO_TIMER_AUTO;

// Force standard timer (rdtsc/cntvct_el0)
config.timer_preference = TO_TIMER_STANDARD;

// Require PMU timer, fail if unavailable
config.timer_preference = TO_TIMER_PREFER_PMU;
```

### Platform Timer Matrix

| Platform | Standard Timer | PMU Timer | PMU Requirements |
|----------|----------------|-----------|------------------|
| x86_64 Linux | `rdtsc` (~0.3 ns) | `perf_event` (~0.3 ns) | sudo or CAP_PERFMON |
| ARM64 Linux | `cntvct_el0` (~1-40 ns) | `perf_event` (~0.3 ns) | sudo or CAP_PERFMON |
| ARM64 macOS | `cntvct_el0` (~42 ns) | `kperf` (~0.3 ns) | sudo |
| Other | `clock_gettime` (~1 μs) | N/A | N/A |

### Enabling PMU Timers

**Linux:**
```bash
# Option 1: Run with sudo
sudo ./your_program

# Option 2: Lower perf_event_paranoid
echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid

# Option 3: Grant CAP_PERFMON
sudo setcap cap_perfmon+ep ./your_program
```

**macOS:**
```bash
# kperf requires root
sudo ./your_program
```

---

## C++ Wrapper

For C++17 and later, a header-only wrapper provides a more ergonomic interface:

```cpp
#include <timing_oracle.hpp>
#include <vector>
#include <random>

int main() {
    using namespace timing_oracle;

    auto result = test(
        AttackerModel::AdjacentNetwork,
        32,  // input size
        [](std::vector<uint8_t>& output, bool baseline) {
            if (baseline) {
                std::fill(output.begin(), output.end(), 0);
            } else {
                std::random_device rd;
                std::generate(output.begin(), output.end(), [&]{ return rd(); });
            }
        },
        [](const std::vector<uint8_t>& input) {
            my_crypto_function(input.data(), input.size());
        }
    );

    if (result.outcome == Outcome::Fail) {
        std::cerr << "Leak detected: " << result.effect.shift_ns << " ns\n";
        return 1;
    }
    return 0;
}
```

### C++ Types

The C++ wrapper provides type-safe enums and RAII resource management:

```cpp
namespace timing_oracle {
    enum class AttackerModel { SharedHardware, PostQuantum, AdjacentNetwork, RemoteNetwork, Research, Custom };
    enum class Outcome { Pass, Fail, Inconclusive, Unmeasurable };
    enum class Exploitability { SharedHardwareOnly, Http2Multiplexing, StandardRemote, ObviousLeak };
    // ...

    struct Result {
        Outcome outcome;
        double leak_probability;
        Effect effect;
        // ... (automatically freed on destruction)
    };
}
```

---

## Building and Linking

### From Source

```bash
cargo build --release -p timing-oracle-c

# Output files:
#   target/release/libtiming_oracle_c.a      (static)
#   target/release/libtiming_oracle_c.dylib  (macOS dynamic)
#   target/release/libtiming_oracle_c.so     (Linux dynamic)
```

### Linking

**macOS:**
```bash
cc -I include your_code.c -L lib -ltiming_oracle_c \
   -framework Security -framework CoreFoundation -liconv
```

**Linux:**
```bash
cc -I include your_code.c -L lib -ltiming_oracle_c \
   -lpthread -ldl -lm
```

---

## Examples

See [`crates/timing-oracle-c/examples/`](../crates/timing-oracle-c/examples/):

| Example | Description |
|---------|-------------|
| `simple.c` | Basic usage |
| `compare.c` | Leaky vs constant-time comparison |
| `attacker_models.c` | Threat model selection |
| `result_handling.c` | Complete result interpretation |
| `custom_config.c` | Configuration options |
| `no_std_embedded.c` | External time tracking |
| `crypto_hmac.c` | HMAC verification (OpenSSL) |
