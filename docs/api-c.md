# C API Guide

Tacet provides a C-compatible API for timing side-channel detection in C and C++ projects. The API is designed to be simple, safe, and zero-configuration.

## Installation

See the [installation script](../install.sh) for automated installation, or visit the [C bindings directory](../crates/tacet-c/README.md) for manual installation instructions.

## Quick Start

```c
#include <tacet.h>
#include <stdio.h>
#include <stdlib.h>

// Your cryptographic function to test
void my_crypto_function(const uint8_t *data, size_t len) {
    // ... your implementation ...
}

// Timer function (platform-specific)
uint64_t read_timer(void) {
#if defined(__aarch64__)
    uint64_t cnt;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(cnt));
    return cnt;
#elif defined(__x86_64__)
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    #error "Unsupported platform"
#endif
}

int main(void) {
    const size_t num_samples = 10000;
    uint64_t *baseline_samples = malloc(num_samples * sizeof(uint64_t));
    uint64_t *sample_samples = malloc(num_samples * sizeof(uint64_t));

    uint8_t baseline_data[32] = {0};  // All zeros
    uint8_t sample_data[32];

    // Collect timing samples
    for (size_t i = 0; i < num_samples; i++) {
        // Generate random sample data
        for (size_t j = 0; j < 32; j++) {
            sample_data[j] = rand() & 0xFF;
        }

        // Measure baseline (all zeros)
        uint64_t start = read_timer();
        my_crypto_function(baseline_data, 32);
        uint64_t end = read_timer();
        baseline_samples[i] = end - start;

        // Measure sample (random data)
        start = read_timer();
        my_crypto_function(sample_data, 32);
        end = read_timer();
        sample_samples[i] = end - start;
    }

    // Create configuration - timer frequency is auto-detected!
    struct ToConfig config = to_config_adjacent_network();

    // Analyze timing samples
    struct ToResult result;
    to_analyze(baseline_samples, sample_samples, num_samples, &config, &result);

    // Check result
    if (result.outcome == Pass) {
        printf("✓ No timing leak detected (P=%.1f%%)\n",
               result.leak_probability * 100);
    } else if (result.outcome == Fail) {
        printf("✗ Timing leak detected (P=%.1f%%, exploitability: %d)\n",
               result.leak_probability * 100, result.exploitability);
    } else if (result.outcome == Inconclusive) {
        printf("? Inconclusive (P=%.1f%%)\n", result.leak_probability * 100);
    } else {
        printf("⊘ Unmeasurable: %s\n", result.recommendation);
    }

    free(baseline_samples);
    free(sample_samples);
    return result.outcome == Fail ? 1 : 0;
}
```

## Timer Frequency (Automatic Detection)

**The timer frequency is automatically detected** — you don't need to configure it manually!

When `timer_frequency_hz` is left at its default value of 0, tacet automatically:
- Reads ARM64 `CNTFRQ_EL0` register with firmware validation
- Reads x86_64 TSC frequency from sysfs or CPUID
- Falls back to runtime calibration if needed

### Platform Support

| Platform | Timer | Detection Method | Typical Frequency |
|----------|-------|------------------|-------------------|
| ARM64 macOS | `cntvct_el0` | Read CNTFRQ_EL0 | 24 MHz (M1/M2), 1 GHz (M3+) |
| ARM64 Linux | `cntvct_el0` | Read CNTFRQ_EL0, detect platform | 1 GHz (Graviton), 54 MHz (RPi4) |
| x86_64 Linux | `rdtsc` | sysfs, CPUID, calibration | ~3 GHz (typical) |
| x86_64 macOS | `rdtsc` | sysctl, calibration | ~3 GHz (typical) |

### Optional: Explicit Detection

If you want to check the detected frequency for debugging:

```c
uint64_t freq = to_detect_timer_frequency();
printf("Timer frequency: %.2f MHz\n", freq / 1e6);
```

Example output:
- Apple Silicon M1/M2: `Timer frequency: 24.00 MHz`
- Apple Silicon M3+: `Timer frequency: 1000.00 MHz`
- AWS Graviton: `Timer frequency: 1000.00 MHz`
- Intel x86_64: `Timer frequency: 3000.00 MHz` (varies by CPU)

### Manual Override (Advanced)

Only set `timer_frequency_hz` manually if you:
- Know the exact frequency and want to skip detection
- Are using a non-standard timer
- Need deterministic behavior without calibration overhead

```c
struct ToConfig config = to_config_adjacent_network();
config.timer_frequency_hz = 3000000000ULL;  // 3 GHz (x86_64 typical)
```

Common values:
- `24000000` - 24 MHz (Apple Silicon M1/M2)
- `1000000000` - 1 GHz (Apple Silicon M3+, AWS Graviton)
- `3000000000` - 3 GHz (Intel x86_64, approximate)

## Attacker Models

Choose an attacker model based on your threat scenario. Cycle-based thresholds use a conservative 5 GHz reference frequency (assumes fast attacker hardware).

### Adjacent Network (100 ns threshold)
For LAN attackers or HTTP/2 APIs. Use this for internal microservices or APIs with request multiplexing.

```c
struct ToConfig config = to_config_adjacent_network();
```

### Shared Hardware (0.4 ns / ~2 cycles @ 5 GHz threshold)
For co-resident attackers with cycle-level timing access (SGX enclaves, containers, shared hosting).

```c
struct ToConfig config = to_config_shared_hardware();
```

### Post-Quantum (2.0 ns / ~10 cycles @ 5 GHz threshold)
For lattice-based cryptography (ML-KEM, ML-DSA). Catches KyberSlash-class timing leaks.

```c
struct ToConfig config = to_config_post_quantum();
```

### Remote Network (50 μs threshold)
For general internet exposure and legacy HTTP/1.1 services.

```c
struct ToConfig config = to_config_remote_network();
```

### Custom Threshold

```c
struct ToConfig config = to_config_adjacent_network();
config.custom_threshold_ns = 500.0;  // 500ns custom threshold
```

## Configuration Options

### Time Budget

Control how long the analysis runs:

```c
struct ToConfig config = to_config_adjacent_network();
config.time_budget_secs = 60.0;  // Run for up to 60 seconds
```

### Sample Limits

Set maximum number of samples:

```c
struct ToConfig config = to_config_adjacent_network();
config.max_samples = 100000;  // Collect at most 100k samples per class
```

### Decision Thresholds

Adjust statistical confidence levels:

```c
struct ToConfig config = to_config_adjacent_network();
config.pass_threshold = 0.01;   // More confident pass (default 0.05)
config.fail_threshold = 0.99;   // More confident fail (default 0.95)
```

## API Reference

### Configuration Functions

- `ToConfig to_config_adjacent_network()` - LAN attacker (100ns threshold)
- `ToConfig to_config_shared_hardware()` - Co-resident attacker (~2 cycles threshold)
- `ToConfig to_config_remote_network()` - Internet attacker (50μs threshold)
- `uint64_t to_detect_timer_frequency()` - Detect timer frequency in Hz

### Analysis Functions

- `ToError to_analyze(...)` - One-shot analysis of pre-collected samples
- `ToCalibration* to_calibrate(...)` - Calibrate for adaptive sampling loop
- `ToError to_step(...)` - One step of adaptive sampling loop
- `ToState* to_state_new()` - Create new adaptive state
- `void to_state_free(ToState*)` - Free adaptive state
- `void to_calibration_free(ToCalibration*)` - Free calibration data

### Result Types

```c
enum ToOutcome {
    Pass,          // No leak detected
    Fail,          // Leak confirmed
    Inconclusive,  // Could not reach decision
    Unmeasurable   // Operation too fast to measure
};

struct ToResult {
    enum ToOutcome outcome;
    double leak_probability;          // 0.0 to 1.0
    double shift_ns;                  // Uniform shift effect
    double tail_ns;                   // Tail distribution effect
    enum Exploitability exploitability;  // Negligible/LAN/Remote
    enum MeasurementQuality quality;     // Excellent/Good/Poor/TooNoisy
    char recommendation[256];         // Guidance for Unmeasurable
    uint64_t samples_used;            // Total samples collected
};
```

### Exploitability Levels

```c
enum Exploitability {
    Negligible = 0,    // < 100ns (likely not exploitable)
    PossibleLAN = 1,   // 100-500ns (exploitable on LAN)
    LikelyLAN = 2,     // 500ns-20μs (clearly exploitable on LAN)
    PossibleRemote = 3 // > 20μs (potentially exploitable remotely)
};
```

## Examples

See the [examples directory](../crates/tacet-c/examples/) for complete working examples:

- `simple.c` - Basic one-shot analysis
- `adaptive.c` - Adaptive sampling loop (C measurement, Rust analysis)

## Building and Linking

### With pkg-config

```bash
cc your_code.c $(pkg-config --cflags --libs tacet) -o your_program
```

### Manual linking

```bash
# macOS
cc your_code.c -ltacet_c -framework Security -framework CoreFoundation -o your_program

# Linux
cc your_code.c -ltacet_c -lpthread -ldl -lm -o your_program
```

## Best Practices

1. **Use zero-config**: Leave `timer_frequency_hz` at 0 for automatic detection
2. **Choose appropriate attacker model**: Match your threat scenario
3. **Collect enough samples**: At least 5,000 for calibration, 10,000+ for analysis
4. **Check for Unmeasurable**: Handle operations too fast to measure gracefully
5. **Interpret results carefully**: Low leak probability doesn't guarantee constant-time

## Common Pitfalls

### Timer Quantization

On platforms with coarse timer resolution (e.g., Apple Silicon's 42ns cntvct_el0), tacet automatically applies batching to compensate. No manual configuration needed.

### CPU Frequency Scaling

Tacet's Bayesian approach is robust to moderate CPU frequency scaling. For best results:
- Pin process to a single CPU core
- Disable CPU frequency scaling if possible
- Use longer time budgets (60s+) for noisy environments

### Cache Effects

Tacet detects timing differences, including those from cache effects. If your code has data-dependent cache access patterns, these will be detected as leaks.

## Troubleshooting

### "Unmeasurable" Result

If you get `Unmeasurable`, your operation is too fast for the available timer. Options:
1. Use a higher-precision timer (kperf on macOS, perf on Linux)
2. Test a slower operation (multiple rounds, larger inputs)
3. Accept that the operation is below measurement threshold

### "Inconclusive" Result

If analysis is inconclusive:
1. Increase time budget: `config.time_budget_secs = 120.0`
2. Increase sample limit: `config.max_samples = 200000`
3. Reduce system noise (close other applications, pin to CPU core)

### Unexpected Leak Detection

If tacet detects a leak you believe is false:
1. Check for data-dependent branches or memory access
2. Verify timer precision with `to_detect_timer_frequency()`
3. Try a different attacker model (e.g., RemoteNetwork instead of SharedHardware)
4. Review the effect size (`shift_ns`, `tail_ns`) - small effects may be acceptable

## Further Reading

- [Main specification](spec.md) - Statistical methodology
- [User guide](guide.md) - Detailed usage and interpretation
- [Implementation guide](implementation-guide.md) - Platform-specific details
- [Rust API reference](api-rust.md) - Rust equivalent API
