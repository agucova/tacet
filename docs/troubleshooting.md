# timing-oracle Troubleshooting Guide

This guide covers common issues and debugging strategies for timing-oracle.

For API documentation, see [api-reference.md](api-reference.md). For conceptual overview, see [guide.md](guide.md).

## Table of Contents

- [Measurement Issues](#measurement-issues)
  - [High Noise / "TooNoisy" Quality](#high-noise--toonoisy-quality)
  - [False Positives](#false-positives)
  - [Unable to Detect Known Leaks](#unable-to-detect-known-leaks)
  - [Virtual Machine Issues](#virtual-machine-issues)
- [Understanding Results](#understanding-results)
  - [Outcome Types](#outcome-types)
  - [Reliability Macros](#reliability-macros)

---

## Measurement Issues

### High Noise / "TooNoisy" Quality

**Symptoms:**
- `quality: TooNoisy` or `Poor`
- High minimum detectable effect (>100 ns)
- Inconsistent results

**Solutions:**

1. **Check CPU governor (Linux):**
   ```bash
   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
   # Should be "performance", not "powersave"
   sudo cpupower frequency-set -g performance
   ```

2. **Disable turbo boost:**
   ```bash
   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
   ```

3. **Check for background processes:**
   - Close unnecessary applications
   - Verify low system load with `top` or `htop`

4. **Run single-threaded:**
   ```bash
   cargo test --test timing_tests -- --test-threads=1
   ```

5. **Use cycle-accurate timers (if available):**

   On macOS ARM64, run with `sudo` to enable PMU-based timing:
   ```bash
   sudo cargo test
   ```

   On Linux, grant `CAP_PERFMON` or run with `sudo`.

### False Positives

**Symptoms:**
- Constant-time code flagged as leaky
- `leak_probability` high on XOR operations

**Solutions:**

1. **Run sanity check:**
   ```rust
   // Test identical inputs - should always pass
   let inputs = InputPair::new(
       || [0u8; 32],
       || [0u8; 32],  // Both identical
   );

   let outcome = TimingOracle::for_attacker(AttackerModel::Research)
       .test(inputs, |data| my_op(data));

   assert!(outcome.passed(), "Sanity check failed - environment issue");
   ```

2. **Check preflight warnings** - they indicate environmental issues

3. **Ensure identical code paths:**
   ```rust
   // The test closure must execute the same code for both inputs
   let inputs = InputPair::new(
       || [0u8; 32],
       || rand::random(),
   );

   let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
       .test(inputs, |data| encrypt(data));  // Same code path for both
   ```

4. **Check for state-dependent behavior:**
   - Ensure no global state modified between measurements
   - Verify caches are in consistent state

### Unable to Detect Known Leaks

**Symptoms:**
- Early-exit comparison shows `leak_probability < 0.5`
- Tests pass on leaky code

**Solutions:**

1. **Verify the leak exists:**
   ```rust
   // Manual inspection
   let mut fixed_times = vec![];
   let mut random_times = vec![];
   for _ in 0..1000 {
       let timer = Timer::new();
       fixed_times.push(timer.measure_cycles(|| fixed_op()));
       random_times.push(timer.measure_cycles(|| random_op()));
   }
   // Check if medians differ significantly
   ```

2. **Prevent compiler optimization:**
   ```rust
   use std::hint::black_box;

   let inputs = InputPair::new(|| [0u8; 32], || rand::random());
   let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
       .test(inputs, |data| {
           black_box(my_function(black_box(data)));
       });
   ```

3. **Use `#[inline(never)]`** on the function under test

4. **Use larger input sizes** - larger arrays mean longer loops and more detectable timing differences

5. **Use a stricter attacker model:**
   ```rust
   // SharedHardware has a ~0.6ns threshold vs AdjacentNetwork's 100ns
   TimingOracle::for_attacker(AttackerModel::SharedHardware)
   ```

### Virtual Machine Issues

**Symptoms:**
- Extremely high noise
- Unreliable timing measurements
- Preflight warnings about VM detected

**Solutions:**

1. **Run tests on bare metal when possible**

2. **If VM required, ensure:**
   - Dedicated CPU cores (no overcommit)
   - Nested virtualization disabled
   - High-resolution timer support enabled

3. **Accept limitations:**
   - VMs inherently have more timing noise
   - Focus on larger effect sizes (>100ns)

---

## Understanding Results

### Outcome Types

The oracle returns four possible outcomes:

| Outcome | Meaning | Action |
|---------|---------|--------|
| `Pass` | No timing leak detected | Code is likely constant-time for your threat model |
| `Fail` | Timing leak confirmed | Investigate and fix the vulnerability |
| `Inconclusive` | Cannot reach decision | Check the `reason` field for guidance |
| `Unmeasurable` | Operation too fast | Use cycle-accurate timer or test slower operation |

**Inconclusive reasons:**
- `DataTooNoisy` - Environment has too much noise
- `NotLearning` - Posterior not converging (borderline case)
- `TimeBudgetExceeded` - Ran out of time
- `SampleBudgetExceeded` - Hit sample limit

### Reliability Macros

For CI integration, use reliability macros to handle edge cases:

```rust
use timing_oracle::{TimingOracle, AttackerModel, skip_if_unreliable, require_reliable, helpers::InputPair};

#[test]
fn test_crypto() {
    let inputs = InputPair::new(|| [0u8; 32], || rand::random());
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .test(inputs, |data| my_crypto_op(data));

    // Option 1: Skip if unreliable (fail-open, good for noisy CI)
    let outcome = skip_if_unreliable!(outcome, "test_crypto");
    assert!(outcome.passed());

    // Option 2: Require reliable (fail-closed, for security-critical code)
    // let outcome = require_reliable!(outcome, "test_crypto");
    // assert!(outcome.passed());
}
```

**Environment variable override:**

```bash
# Force all tests to fail if unreliable
TIMING_ORACLE_UNRELIABLE_POLICY=fail_closed cargo test
```
