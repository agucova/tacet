# Testing Guide

This document describes the testing strategy for `timing-oracle` and how to run the test suite.

## Test Organization

### Unit Tests (`cargo test --lib`)

Fast tests that validate individual components:
- **Statistical functions**: Quantile computation, bootstrap, covariance
- **Analysis layers**: CI gate, Bayesian inference, effect decomposition
- **Measurement**: Timer calibration, sample collection, outlier filtering
- **Preflight checks**: Sanity, generator cost, autocorrelation, system

**Runtime:** ~3 seconds

### Integration Tests

#### Known Leaky (`tests/known_leaky.rs`)

Tests that **must detect** timing leaks:

```rust
cargo test --test known_leaky
```

**Tests:**
- `detects_early_exit_comparison` - Early-exit byte-by-byte comparison
- `detects_branch_timing` - Branch-based timing (if x == 0)

**Expected:**
- `leak_probability > 0.9`
- `ci_gate.passed == false`

**Runtime:** ~2-3 minutes each (100k samples)

#### Known Safe (`tests/known_safe.rs`)

Tests that **must not false-positive** on constant-time code:

```rust
cargo test --test known_safe
```

**Tests:**
- `no_false_positive_xor_compare` - Constant-time XOR-based comparison
- `no_false_positive_xor` - Simple XOR operation

**Expected:**
- `leak_probability < 0.5`
- `ci_gate.passed == true`

**Runtime:** ~2-3 minutes each (100k samples)

#### Calibration (`tests/calibration.rs`)

Statistical validation tests:

```rust
cargo test --test calibration
```

**Tests:**
- `ci_gate_fpr_calibration` - Verify false positive rate ≤ 2α (100 trials)
- `bayesian_calibration` - Verify posterior doesn't over-concentrate (100 trials)

**Expected:**
- CI gate: rejection rate ≤ 2% with α=0.01
- Bayesian: <10% of null trials have probability > 0.9

**Runtime:** ~5-10 minutes (many trials with quick mode)

#### Crypto Attacks (`tests/crypto_attacks.rs`)

Real-world cryptographic timing vulnerabilities:

```rust
cargo test --test crypto_attacks
```

**Test Categories (20 tests total):**

1. **Cache-Based Attacks (4 tests)**
   - `aes_sbox_timing_fast` - AES S-box lookup timing (10k samples, ~15s)
   - `aes_sbox_timing_thorough` - Thorough S-box test (100k samples, ~2-3 min) [ignored]
   - `cache_line_boundary_effects` - Cache line access patterns
   - `memory_access_pattern_leak` - Sequential vs random memory access

2. **Modular Exponentiation (2 tests)**
   - `modexp_square_and_multiply_timing` - Square-and-multiply with different Hamming weights (8k samples)
   - `modexp_bit_pattern_timing` - Bit pattern timing differences

3. **Table Lookup Timing (3 tests)**
   - `table_lookup_small_l1` - L1-resident table (should show minimal timing)
   - `table_lookup_medium_l2` - L2/L3 cache effects
   - `table_lookup_large_cache_thrash` - Cache thrashing with 4KB table

4. **Effect Pattern Validation (3 tests)** - x86_64/aarch64 only
   - `effect_pattern_pure_uniform_shift` - Validates UniformShift classification
   - `effect_pattern_pure_tail` - Validates TailEffect classification
   - `effect_pattern_mixed` - Validates Mixed pattern classification

5. **Exploitability Thresholds (4 tests)** - x86_64/aarch64 only
   - `exploitability_negligible` - <100ns delays
   - `exploitability_possible_lan` - 100-500ns delays
   - `exploitability_likely_lan` - 500ns-20μs delays
   - `exploitability_possible_remote` - >20μs delays

**Expected:**
- Leak detection tests: `leak_probability > 0.7-0.9`
- Effect patterns match theoretical predictions
- Exploitability classifications align with Crosby et al. thresholds

**Runtime:**
- Fast suite (~15 tests): 3-5 minutes
- With ignored tests: 15-20 minutes total

**Platform Notes:**
- Effect pattern and exploitability tests require x86_64 or aarch64
- Cache timing behavior varies by CPU architecture
- Results may differ between Intel, AMD, and ARM processors

#### Async Timing (`tests/async_timing.rs`)

Async/await and concurrent task timing validation:

```rust
cargo test --test async_timing
```

**Test Categories (9 tests total):**

1. **Baseline Tests (2 tests)** - Should pass, no false positives
   - `async_executor_overhead_no_false_positive` - Symmetric executor overhead (10k samples)
   - `async_block_on_overhead_symmetric` - Identical block_on() overhead (8k samples)

2. **Leak Detection (3 tests)** - Should detect timing leaks
   - `detects_conditional_await_timing` - Secret-dependent await (50k samples)
   - `detects_early_exit_async` - Async early-return comparison (50k samples)
   - `detects_secret_dependent_sleep` - Sleep duration varies with secret (100k samples) [ignored]

3. **Concurrent Tasks (2 tests)**
   - `concurrent_tasks_no_crosstalk` - Background tasks don't interfere (10k samples)
   - `detects_task_spawn_timing_leak` - Different spawn counts (50k samples) [ignored]

4. **Optional Thorough (2 tests)** - Informational [both ignored]
   - `tokio_single_vs_multi_thread_stability` - Noise comparison between runtimes
   - `async_workload_flag_effectiveness` - Flag behavior validation

**Expected:**
- Baseline: `ci_gate.passed == true`, `leak_probability < 0.5`
- Leak detection: `leak_probability > 0.7-0.8`
- Concurrent: No crosstalk from background tasks

**Runtime:**
- Fast suite (5 tests): 20-40 seconds
- With ignored tests: 5-10 minutes total

**Usage Pattern:**
```rust
let rt = tokio::runtime::Builder::new_current_thread()
    .enable_time()
    .build()
    .unwrap();

TimingOracle::new().test(
    || rt.block_on(async_fixed_op()),
    || rt.block_on(async_random_op())
);
```

#### AES Timing (`tests/aes_timing.rs`)

AES-128 encryption timing tests inspired by DudeCT's aes32 example:

```rust
cargo test --test aes_timing
```

**Test Categories (7 tests total):**

All tests use **DudeCT's two-class pattern**:
- **Class 0**: All-zero plaintexts/keys (0x00 repeated)
- **Class 1**: Random plaintexts/keys

1. **Block Encryption Tests (4 tests)**
   - `aes128_block_encrypt_constant_time` - Basic encryption with zeros vs random (100k samples, ~60s)
   - `aes128_different_keys_constant_time` - All-zero vs random key (50k samples)
   - `aes128_multiple_blocks_constant_time` - 4 blocks cumulative timing (20k samples)
   - `aes128_key_init_constant_time` - Key expansion timing (50k samples)

2. **Comparative Tests (3 tests)**
   - `aes128_hamming_weight_independence` - 0x00 vs 0xFF plaintexts (30k samples)
   - `aes128_byte_pattern_independence` - Sequential vs reverse patterns (30k samples)

**Expected:**
- Modern AES implementations (with AES-NI) should be constant-time
- May detect small timing differences but exploitability should be **Negligible**
- Tests allow `!ci_gate.passed` but assert on `Exploitability::Negligible`

**Runtime:**
- Full suite: 2-3 minutes

**Example Output:**
```
[aes128_block_encrypt_constant_time]
  leak_probability: 0.142
  ci_gate.passed: true
  exploitability: Negligible
```

#### ECC Timing (`tests/ecc_timing.rs`)

Curve25519/X25519 elliptic curve timing tests inspired by DudeCT's donna/donnabad examples:

```rust
cargo test --test ecc_timing
```

**Test Categories (8 tests total):**

All tests use **DudeCT's two-class pattern**:
- **Class 0**: All-zero scalars/basepoints (0x00 repeated)
- **Class 1**: Random scalars/basepoints

1. **Scalar Multiplication Tests (5 tests)**
   - `x25519_scalar_mult_constant_time` - Basic scalar mult with zeros vs random (50k samples, ~45s)
   - `x25519_different_basepoints_constant_time` - All-zero vs random basepoint (30k samples)
   - `x25519_multiple_operations_constant_time` - 3 operations cumulative timing (10k samples)
   - `x25519_scalar_clamping_constant_time` - Scalar clamping timing (20k samples)
   - `x25519_ecdh_exchange_constant_time` - Full ECDH key exchange (15k samples)

2. **Comparative Tests (2 tests)**
   - `x25519_hamming_weight_independence` - 0x00 vs 0xFF scalars (20k samples)
   - `x25519_byte_pattern_independence` - Sequential vs reverse patterns (20k samples)

**Expected:**
- Modern X25519 implementations (like x25519-dalek) should be constant-time
- May detect small timing differences but exploitability should be **Negligible** or **PossibleLAN**
- Tests allow `!ci_gate.passed` but assert on low exploitability

**Runtime:**
- Full suite: 2-3 minutes

**Example Output:**
```
[x25519_scalar_mult_constant_time]
  leak_probability: 0.089
  ci_gate.passed: true
  exploitability: Negligible
```

## Running Tests

### Quick Iteration

```bash
# Run only unit tests (fast)
cargo test --lib

# Run a specific integration test file
cargo test --test aes_timing
cargo test --test ecc_timing

# Run a specific test
cargo test --test known_leaky detects_early_exit_comparison
cargo test --test aes_timing aes128_block_encrypt_constant_time

# Run with nextest (faster parallel execution)
cargo nextest run
```

### Full Test Suite

```bash
# Run everything (unit + integration)
cargo test

# Run with single thread (more reliable timing)
cargo test -- --test-threads=1

# Run with nextest in timing profile (single-threaded)
cargo nextest run --profile timing
```

### Continuous Integration

For CI pipelines, use the full suite with single-threaded execution:

```bash
cargo test -- --test-threads=1
```

## Test Configuration

### Quick Mode

Integration tests use `TimingOracle::new()` with 100k samples for thorough validation. During development, use `TimingOracle::quick()` (5k samples) for faster iteration.

## Interpreting Test Failures

### Known Leaky Tests Fail to Detect

**Possible causes:**
1. High noise environment (check `result.quality`)
2. Insufficient sample size
3. Timer calibration issue

**Debug:**
```bash
# Run with verbose output
RUST_LOG=debug cargo test --test known_leaky -- --nocapture

# Check system configuration
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor  # Linux
```

### Known Safe Tests False-Positive

**Possible causes:**
1. Environmental interference
2. Unstable measurement conditions
3. Multiple testing effects

**Debug:**
1. Check preflight warnings in output
2. Run sanity check manually
3. Reduce alpha: `ci_alpha(0.001)`

### Calibration Tests Fail

**CI Gate FPR > 2α:**
- Indicates block bootstrap doesn't preserve null distribution
- May need adjustment to block size or resampling strategy

**Bayesian over-concentration:**
- Too many high probabilities on null data
- Check prior settings and sample splitting

## Platform-Specific Notes

### macOS / Apple Silicon

- Uses ARM `cntvct_el0` timer (~24 MHz)
- Lower "cycles per nanosecond" than x86 (0.024 vs 3.0)
- Results are still valid - conversion is handled automatically

### Linux

- Ensure CPU governor is set to "performance"
- Disable turbo boost for consistent results
- Consider CPU isolation for very precise measurements

### Windows

- Uses x86 `rdtsc` timer
- May have higher noise due to timer precision
- Run with administrator privileges for best results

## Benchmarks

Microbenchmarks for performance testing:

```bash
cargo bench
```

These test individual components (quantile computation, bootstrap, etc.) with small sample sizes to track performance regressions.

## Adding New Tests

### Unit Test Template

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_feature() {
        // Arrange
        let input = ...;

        // Act
        let result = my_function(input);

        // Assert
        assert!(result.is_valid());
    }
}
```

### Integration Test Template

```rust
// tests/my_test.rs
use timing_oracle::TimingOracle;

#[test]
fn test_my_timing_property() {
    let result = TimingOracle::new()
        .samples(100_000)
        .test(
            || fixed_operation(),
            || random_operation(),
        );

    assert!(/* your condition */,
            "Description with value: {}", result.leak_probability);
}
```

## Test Coverage

Run with coverage report:

```bash
# Install cargo-llvm-cov
cargo install cargo-llvm-cov

# Generate coverage
cargo llvm-cov --html

# View in browser
open target/llvm-cov/html/index.html
```

## Continuous Integration Setup

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable

      - name: Run unit tests
        run: cargo test --lib

      - name: Run integration tests
        run: cargo test --test known_leaky --test known_safe -- --test-threads=1

      - name: Run calibration tests
        run: cargo test --test calibration -- --test-threads=1
        # Optional: allow calibration tests to fail (they're statistical)
        continue-on-error: true
```

## Troubleshooting Test Issues

See [README.md § Troubleshooting](README.md#troubleshooting) for detailed debugging steps for common test failure scenarios.
