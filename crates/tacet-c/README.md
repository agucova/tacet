# tacet-c: C/C++ Bindings for Tacet

Statistical timing side-channel detection for C and C++ projects.

## Features

- **Zero configuration**: Timer frequency automatically detected on ARM64 and x86_64
- **Platform-aware**: Handles Apple Silicon (M1/M2/M3+), AWS Graviton, Intel/AMD x86_64
- **Robust detection**: Validates hardware registers, falls back to calibration
- **Same as Rust**: Uses identical statistical methodology as the Rust API
- **No manual tuning**: Works out-of-the-box on all supported platforms

## Installation

### Via install.sh (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/agucova/tacet/main/install.sh | bash
```

Or with custom prefix:
```bash
curl -fsSL https://raw.githubusercontent.com/agucova/tacet/main/install.sh | PREFIX=$HOME/.local bash
```

### Manual Installation

Download from [GitHub Releases](https://github.com/agucova/tacet/releases):
- `libtacet_c-{platform}.a` - Static library
- `tacet.h` - C header
- `tacet.hpp` - C++ wrapper (coming soon)
- `tacet-{platform}.pc` - pkg-config file

Install to your preferred location:
```bash
# Example: Install to /usr/local
sudo cp libtacet_c-darwin-arm64.a /usr/local/lib/libtacet_c.a
sudo cp tacet.h /usr/local/include/tacet/
sudo cp tacet-darwin-arm64.pc /usr/local/lib/pkgconfig/tacet.pc
```

## Quick Start

```c
#include <tacet.h>
#include <stdio.h>

// Your cryptographic function
void my_crypto(const uint8_t *data) {
    // ... implementation ...
}

// Platform-specific timer
uint64_t read_timer(void) {
#if defined(__aarch64__)
    uint64_t cnt;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(cnt));
    return cnt;
#elif defined(__x86_64__)
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#endif
}

int main(void) {
    // Collect timing samples (simplified)
    uint64_t baseline[10000], sample[10000];
    uint8_t data[32] = {0};

    for (size_t i = 0; i < 10000; i++) {
        uint64_t start = read_timer();
        my_crypto(data);
        baseline[i] = read_timer() - start;

        // Generate random data for sample class
        for (size_t j = 0; j < 32; j++) data[j] = rand() & 0xFF;

        start = read_timer();
        my_crypto(data);
        sample[i] = read_timer() - start;
    }

    // Create configuration - timer frequency is auto-detected!
    struct ToConfig config = to_config_adjacent_network();

    // Analyze
    struct ToResult result;
    to_analyze(baseline, sample, 10000, &config, &result);

    // Check result
    if (result.outcome == Pass) {
        printf("✓ No timing leak detected\n");
        return 0;
    } else if (result.outcome == Fail) {
        printf("✗ Timing leak detected: P=%.1f%%, exploitability=%d\n",
               result.leak_probability * 100, result.exploitability);
        return 1;
    }
}
```

## Zero-Configuration Timer Detection

tacet-c automatically detects your system's timer frequency — no manual configuration needed!

```c
// Optional: Check detected frequency for debugging
uint64_t freq = to_detect_timer_frequency();
printf("Timer frequency: %.2f MHz\n", freq / 1e6);

// Timer frequency is automatically detected during analysis
struct ToConfig config = to_config_adjacent_network();
// config.timer_frequency_hz is 0 (auto-detect)

struct ToResult result;
to_analyze(baseline, sample, count, &config, &result);  // Detects automatically
```

### How It Works

- **ARM64**: Reads `CNTFRQ_EL0` register, validates against known platforms
- **x86_64**: Reads TSC frequency from sysfs (Linux) or sysctl (macOS)
- **Fallback**: Runtime calibration if register reads fail

### Platform Support

| Platform | Timer | Frequency |
|----------|-------|-----------|
| Apple Silicon M1/M2 | `cntvct_el0` | 24 MHz (auto-detected) |
| Apple Silicon M3+ | `cntvct_el0` | 1 GHz (auto-detected) |
| AWS Graviton | `cntvct_el0` | 1 GHz (auto-detected) |
| Raspberry Pi 4 | `cntvct_el0` | 54 MHz (auto-detected) |
| Intel/AMD x86_64 | `rdtsc` | ~3 GHz (auto-detected) |

## Documentation

- **[C API Guide](../../docs/api-c.md)** - Comprehensive usage guide, zero-config examples
- **[Examples](./examples/)** - Working code examples (`simple.c`, `adaptive.c`)
- **[Header Reference](./include/tacet.h)** - Full API documentation in header comments

## Building Your Project

### With pkg-config (Recommended)

```bash
cc your_code.c $(pkg-config --cflags --libs tacet) -o your_program
```

### With CMake

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(TACET REQUIRED tacet)

add_executable(your_program your_code.c)
target_link_libraries(your_program ${TACET_LIBRARIES})
target_include_directories(your_program PRIVATE ${TACET_INCLUDE_DIRS})
```

### Manual Linking

```bash
# macOS
cc your_code.c -ltacet_c -framework Security -framework CoreFoundation -o your_program

# Linux
cc your_code.c -ltacet_c -lpthread -ldl -lm -o your_program
```

## Examples

See the [examples directory](./examples/) for complete working examples:

### simple.c - Basic One-Shot Analysis
```bash
just bindings c-example-simple
./crates/tacet-c/examples/simple
```

### adaptive.c - Adaptive Sampling Loop
```bash
just bindings c-example-adaptive
./crates/tacet-c/examples/adaptive
```

## API Overview

### Attacker Models

```c
// LAN attacker (100ns threshold)
struct ToConfig config = to_config_adjacent_network();

// Co-resident attacker (~2 cycles threshold)
struct ToConfig config = to_config_shared_hardware();

// Internet attacker (50μs threshold)
struct ToConfig config = to_config_remote_network();
```

### Analysis

```c
// One-shot analysis
struct ToResult result;
ToError err = to_analyze(baseline, sample, count, &config, &result);

// Adaptive sampling loop
ToCalibration *cal = to_calibrate(baseline, sample, 5000, &config, &err);
ToState *state = to_state_new();

ToStepResult step_result;
double elapsed = /* time since start */;
err = to_step(cal, state, new_baseline, new_sample, 1000, &config, elapsed, &step_result);

if (step_result.has_decision) {
    // Analysis complete
}

to_state_free(state);
to_calibration_free(cal);
```

### Timer Detection

```c
// Automatic detection (recommended)
struct ToConfig config = to_config_adjacent_network();
// config.timer_frequency_hz == 0 (auto-detect)

// Optional: Check detected frequency
uint64_t freq = to_detect_timer_frequency();
printf("Detected: %.2f MHz\n", freq / 1e6);

// Manual override (advanced)
config.timer_frequency_hz = 24000000;  // 24 MHz
```

## Building from Source

### Prerequisites

- Rust 1.90+ (for building)
- Cargo
- cbindgen (installed automatically via cargo)

### Build Steps

```bash
# Clone repository
git clone https://github.com/agucova/tacet.git
cd tacet

# Build C bindings
cargo build -p tacet-c --release

# Outputs:
# - target/release/libtacet_c.a - Static library
# - crates/tacet-c/include/tacet.h - C header
# - target/release/tacet.pc - pkg-config file
```

### Install Locally

```bash
# Install to /usr/local
sudo cp target/release/libtacet_c.a /usr/local/lib/
sudo cp crates/tacet-c/include/tacet.h /usr/local/include/tacet/
sudo cp target/release/tacet.pc /usr/local/lib/pkgconfig/

# Or install to custom prefix
PREFIX=$HOME/.local
cp target/release/libtacet_c.a $PREFIX/lib/
cp crates/tacet-c/include/tacet.h $PREFIX/include/tacet/
cp target/release/tacet.pc $PREFIX/lib/pkgconfig/
```

## Platform-Specific Notes

### macOS

- **Apple Silicon M1/M2**: 24 MHz counter, auto-detected
- **Apple Silicon M3+ (macOS 15+)**: 1 GHz counter (kernel-scaled), auto-detected
- **Intel**: 3 GHz TSC (typical), auto-detected via sysctl

### Linux

- **ARM64**: Reads CNTFRQ_EL0, detects Graviton/RPi platforms
- **x86_64**: Reads from `/sys/devices/system/cpu/cpu0/tsc_freq_khz` or CPUID
- **Fallback**: Runtime calibration if hardware detection fails

## Troubleshooting

### "Unmeasurable" Result

Your operation is too fast for the timer. Options:
1. Test a slower operation (multiple rounds, larger data)
2. Use a higher-precision timer (kperf/perf)
3. Accept that operation is below measurement threshold

### Wrong Frequency Detection

If automatic detection returns unexpected values:
```c
uint64_t freq = to_detect_timer_frequency();
printf("Detected: %llu Hz\n", freq);

// Override if needed
struct ToConfig config = to_config_adjacent_network();
config.timer_frequency_hz = /* your known frequency */;
```

## License

Mozilla Public License 2.0 (MPL-2.0) - see [LICENSE](../../LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md)

## Links

- [GitHub Repository](https://github.com/agucova/tacet)
- [C API Documentation](../../docs/api-c.md)
- [Rust API Documentation](https://docs.rs/tacet)
- [Issue Tracker](https://github.com/agucova/tacet/issues)
