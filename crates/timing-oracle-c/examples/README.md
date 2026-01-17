# timing-oracle C Examples

This directory contains example programs demonstrating the timing-oracle C API.

## Prerequisites

Build the timing-oracle-c library first:

```bash
cd ../../..  # Repository root
cargo build --release -p timing-oracle-c
```

## Building Examples

```bash
# Build all core examples
make

# Build including OpenSSL example
make WITH_OPENSSL=1

# Build a single example
make simple

# Clean
make clean
```

## Examples

### Core Examples (no external dependencies)

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `simple` | Minimal working example | Config, callbacks, result handling |
| `compare` | Leaky vs constant-time code | Early-exit vs XOR-accumulator patterns |
| `attacker_models` | All threat model presets | Threshold selection, sensitivity |
| `result_handling` | Complete result interpretation | All outcomes, effects, quality |
| `custom_config` | Configuration options | Budgets, thresholds, seeds |
| `no_std_embedded` | External time tracking | `to_test_with_time()` usage |

### Optional Examples (require OpenSSL)

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `crypto_hmac` | HMAC-SHA256 verification | Real crypto, security impact |

## Running Examples

```bash
# Run any example
./simple
./compare
./attacker_models
./result_handling
./custom_config
./no_std_embedded
./crypto_hmac    # Requires OpenSSL build
```

## Example Walkthrough

### 1. simple.c - Start Here

The simplest complete example. Shows:
- Creating a configuration with `to_config_default()`
- Implementing generator and operation callbacks
- Running a test with `to_test()`
- Interpreting the result
- Cleaning up with `to_result_free()`

### 2. compare.c - Understand Detection

Demonstrates what timing-oracle detects:
- **Leaky code**: Early-exit comparison (FAILS the test)
- **Safe code**: XOR-accumulator comparison (PASSES the test)

This is the pattern that matters for password verification, MAC validation, etc.

### 3. attacker_models.c - Choose Your Threat Model

Shows all 6 attacker models and when to use each:
- SharedHardware for SGX/containers
- PostQuantum for post-quantum crypto
- AdjacentNetwork for LAN/web APIs (default)
- RemoteNetwork for internet APIs
- Research for detecting any difference
- Custom for specific thresholds

### 4. result_handling.c - Full Result Analysis

Exhaustive handling of all result fields:
- All 4 outcome types (Pass, Fail, Inconclusive, Unmeasurable)
- Effect size decomposition (shift vs tail)
- Quality assessment
- Platform and timer information

### 5. custom_config.c - Advanced Configuration

All configuration parameters:
- Time and sample budgets
- Pass/fail probability thresholds
- Custom thresholds
- Reproducible results with seeds
- NULL config for defaults

### 6. no_std_embedded.c - Embedded/SGX Usage

Using `to_test_with_time()` for environments without std:
- External time tracking
- SGX enclave scenarios
- Embedded systems with custom timers

### 7. crypto_hmac.c - Real Crypto

Testing actual cryptographic code with OpenSSL:
- HMAC-SHA256 verification
- Leaky vs constant-time comparison impact
- Security implications (Lucky Thirteen style attacks)

## Troubleshooting

### Library not found

```
Error: Library not found at ../../../target/release/libtiming_oracle_c.a
```

Build the library first:
```bash
cd ../../.. && cargo build --release -p timing-oracle-c
```

### OpenSSL not found

```
fatal error: 'openssl/hmac.h' file not found
```

Install OpenSSL development libraries:
```bash
# macOS
brew install openssl

# Ubuntu/Debian
apt install libssl-dev

# Fedora
dnf install openssl-devel
```

### Tests always return INCONCLUSIVE

This usually means the operation is too fast or the system is too noisy:
- Increase `time_budget_secs`
- Reduce system load (close other programs)
- Try a less sensitive attacker model (e.g., AdjacentNetwork instead of SharedHardware)

### Operation is UNMEASURABLE

The operation completes faster than the timer can measure:
- **Use PMU timer** (recommended): Run with `sudo` for ~0.3ns resolution
  - macOS: `sudo ./your_program` (uses kperf)
  - Linux: `sudo ./your_program` (uses perf_event)
  - Alternative on Linux: `echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid`
- Use a more complex operation
- The library automatically batches operations on coarse-timer platforms
- Consider if the operation is security-relevant (very fast ops may not leak useful info)
