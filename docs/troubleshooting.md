# timing-oracle Troubleshooting Guide

This guide covers common issues, debugging strategies, and performance optimization for timing-oracle.

For API documentation, see [api-reference.md](api-reference.md). For conceptual overview, see [guide.md](guide.md).

## Table of Contents

- [Measurement Issues](#measurement-issues)
  - [High Noise / "TooNoisy" Quality](#high-noise--toonoisy-quality)
  - [False Positives on Noise](#false-positives-on-noise)
  - [Unable to Detect Known Leaks](#unable-to-detect-known-leaks)
  - [Virtual Machine Issues](#virtual-machine-issues)
- [Handling Unreliable Measurements](#handling-unreliable-measurements)
  - [Reliability Macros](#reliability-macros)
  - [Reliability Criteria](#reliability-criteria)
  - [Policy Selection](#policy-selection)
- [Handling Test Flakiness](#handling-test-flakiness)
- [Performance Tips](#performance-tips)
  - [Typical Runtime](#typical-runtime)
  - [Optimization Strategies](#optimization-strategies)
  - [Preset Comparison](#preset-comparison)

---

## Measurement Issues

### High Noise / "TooNoisy" Quality

**Symptoms:**
- `quality: TooNoisy` or `Poor`
- High `min_detectable_effect` (>100 ns)
- Unreliable or inconsistent results

**Solutions:**

1. **Increase sample count:**
   ```rust
   TimingOracle::new().samples(500_000)  // vs default 100k
   ```

2. **Check CPU governor (Linux):**
   ```bash
   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
   # Should be "performance", not "powersave"
   sudo cpupower frequency-set -g performance
   ```

3. **Disable turbo boost:**
   ```bash
   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
   ```

4. **Run on isolated CPU cores:**
   ```bash
   cargo test --test timing_tests -- --test-threads=1
   ```

5. **Check for background processes:**
   - Close unnecessary applications
   - Verify low system load with `top` or `htop`

6. **Use cycle-accurate timers (if available):**
   ```rust
   // macOS with sudo
   #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
   let timer = timing_oracle::measurement::kperf::PmuTimer::new()?;

   // Linux with sudo/CAP_PERFMON
   #[cfg(all(target_os = "linux", feature = "perf"))]
   let timer = timing_oracle::measurement::perf::LinuxPerfTimer::new()?;
   ```

### False Positives on Noise

**Symptoms:**
- Constant-time code flagged as leaky
- `leak_probability` high on XOR operations
- CI gate failing spuriously

**Solutions:**

1. **Tighten CI alpha:**
   ```rust
   TimingOracle::new().alpha(0.001)  // vs default 0.01
   ```

2. **Run sanity check manually:**
   ```rust
   // Split fixed samples in half - should not detect leak
   let result = TimingOracle::new()
       .test(|| my_fixed_op(), || my_fixed_op());

   assert!(result.ci_gate.passed, "Sanity check failed!");
   ```

3. **Check preflight warnings** - they'll indicate environmental issues

4. **Ensure identical code paths:**
   ```rust
   // Wrong - different code in each closure
   let result = test(
       || encrypt(&FIXED),
       || encrypt(&rand::random()),  // RNG overhead measured!
   );

   // Correct - use InputPair
   let inputs = InputPair::new([0u8; 32], || rand::random());
   let result = test(
       || encrypt(inputs.fixed()),
       || encrypt(inputs.random()),
   );
   ```

5. **Check for state-dependent behavior:**
   - Ensure no global state modified between measurements
   - Verify caches are in consistent state
   - Use interleaved measurement (default behavior)

### Unable to Detect Known Leaks

**Symptoms:**
- Early-exit comparison shows `leak_probability < 0.5`
- CI gate passes on leaky code
- Effect size too small to detect

**Solutions:**

1. **Increase sample count:**
   ```rust
   TimingOracle::new().samples(500_000)
   ```

2. **Ensure sufficient timing difference:**
   - Small leaks (<10ns) may require very large sample sizes
   - Verify leak is visible in raw timing data:
   ```rust
   // Manual inspection
   let mut fixed_times = vec![];
   let mut random_times = vec![];
   for _ in 0..1000 {
       let timer = Timer::new();
       fixed_times.push(timer.measure_cycles(|| fixed_op()));
       random_times.push(timer.measure_cycles(|| random_op()));
   }
   println!("Fixed median: {}", median(&fixed_times));
   println!("Random median: {}", median(&random_times));
   ```

3. **Disable compiler optimizations in test:**
   ```rust
   use std::hint::black_box;

   let result = TimingOracle::new().test(
       || black_box(my_function(black_box(&fixed_input))),
       || black_box(my_function(black_box(&random_input()))),
   );
   ```

4. **Check for optimizer eliminating the leak:**
   - Ensure function isn't inlined and optimized away
   - Use `#[inline(never)]` on the function under test
   - Verify with `--release` vs debug builds

5. **Use larger input sizes:**
   ```rust
   // 32-byte array might be too fast
   let secret = [0u8; 512];  // Larger array = longer loop = more detectable leak
   ```

### Virtual Machine Issues

**Symptoms:**
- Extremely high noise
- Unreliable timing measurements
- Preflight warnings about VM detected
- Timer resolution much worse than expected

**Solutions:**

1. **Run tests on bare metal when possible**

2. **If VM required, ensure:**
   - Dedicated CPU cores (no overcommit)
   - Nested virtualization disabled
   - High-resolution timer support enabled in hypervisor

3. **Configure VM for timing tests:**
   ```bash
   # Example for QEMU/KVM
   -cpu host,tsc-frequency=3000000000  # Fixed TSC frequency
   -smp 4,sockets=1,cores=4            # Dedicated cores
   ```

4. **Use larger sample sizes:**
   ```rust
   TimingOracle::new().samples(500_000)  // Compensate for VM noise
   ```

5. **Accept higher MDE in VM environments:**
   - VMs inherently have more noise
   - Focus on larger effect sizes (>100ns)

---

## Handling Unreliable Measurements

Timing tests are environment-dependent. Operations may be unmeasurable on coarse timers (e.g., Apple Silicon's ~42ns resolution), or measurement quality may degrade under system load.

### Reliability Macros

```rust
use timing_oracle::{TimingOracle, Outcome, skip_if_unreliable, require_reliable};

#[test]
fn test_cache_timing() {
    let result = TimingOracle::new().test(fixed, random);

    // Wrap result in Outcome for reliability checking
    let outcome = Outcome::Completed(result);

    // Skip assertions if measurement is unreliable (fail-open)
    // Prints "[SKIPPED] test_cache_timing: ..." and returns early
    let result = skip_if_unreliable!(outcome, "test_cache_timing");

    assert!(result.leak_probability > 0.5);
}

#[test]
fn test_critical_crypto() {
    let result = TimingOracle::new().test(fixed, random);
    let outcome = Outcome::Completed(result);

    // Panic if measurement is unreliable (fail-closed)
    // Use for security-critical tests that MUST produce reliable results
    let result = require_reliable!(outcome, "test_critical_crypto");

    assert!(result.leak_probability < 0.1);
}
```

### Reliability Criteria

A measurement is considered **reliable** if:

1. The operation was measurable (not `Outcome::Unmeasurable`)
2. Minimum detectable effect (MDE) is valid (> 0.01ns and finite)
3. AND one of:
   - Measurement quality is acceptable (not `TooNoisy`), OR
   - The posterior is conclusive (< 10% or > 90%)

**The key insight:** A conclusive posterior is trustworthy even with noisy measurements—the signal overcame the noise. Measurements are only unreliable when noise prevents reaching any conclusion.

```rust
// This is reliable (conclusive result despite noise)
// leak_probability: 0.98, quality: TooNoisy

// This is unreliable (inconclusive + noisy)
// leak_probability: 0.55, quality: TooNoisy

// This is unreliable (invalid MDE)
// min_detectable_effect: 0.0 ns (timer resolution failure)
```

### Policy Selection

| Policy | Macro | Use Case |
|--------|-------|----------|
| Fail-open | `skip_if_unreliable!` | Cache timing tests, tests on Apple Silicon, parallel CI |
| Fail-closed | `require_reliable!` | Security-critical code, dedicated quiet CI runners |

**Environment variable override:**

Set `TIMING_ORACLE_UNRELIABLE_POLICY` to override the default:
- `fail_open` - Skip unreliable tests (default)
- `fail_closed` - Fail unreliable tests

```bash
# Force all tests to fail if unreliable
TIMING_ORACLE_UNRELIABLE_POLICY=fail_closed cargo test
```

---

## Handling Test Flakiness

If tests occasionally fail/pass inconsistently, use hierarchical thresholds:

```rust
use timing_oracle::{Mode, FailCriterion};

// Use hierarchical thresholds
TimingOracle::ci_test()
    .mode(Mode::Full)
    .fail_on(FailCriterion::Either {
        probability: 0.99,  // Very high bar
    })
    .run(fixed_fn, random_fn)
    .unwrap_or_report();

// Or require both gate AND probability
TimingOracle::ci_test()
    .fail_on(FailCriterion::Both {
        probability: 0.9,
    })
    .run(fixed_fn, random_fn)
    .unwrap_or_report();
```

**Multiple test correction:**

With α = 0.01 per test, running N tests gives P(≥1 false positive) ≈ 1 - (1-α)^N.

For large test suites:
- Use stricter α (e.g., 0.001)
- Treat first failure as warning, require confirmation on re-run
- Use hierarchical gating: `leak_probability > 0.99` = fail, `> 0.9` = warn

**Flakiness debugging checklist:**
1. Check system load during failures
2. Verify no background processes interfering
3. Run with increased samples
4. Check for time-of-day effects (thermal throttling)
5. Consider using `skip_if_unreliable!` for inherently noisy tests

---

## Performance Tips

### Typical Runtime

For default configuration (100k samples):

| Operation Speed | Measurement Time | Total with Analysis |
|----------------|------------------|---------------------|
| Fast (10ns) | ~2 seconds | ~3 seconds |
| Medium (100ns) | ~20 seconds | ~21 seconds |
| Slow (1μs) | ~200 seconds | ~201 seconds |

Analysis overhead (bootstrap + Bayesian inference) is typically <1 second.

### Optimization Strategies

1. **Start with quick mode during development:**
   ```rust
   TimingOracle::quick()  // 5k samples, ~10x faster
   ```

2. **Use parallel feature for bootstrap:**
   ```toml
   [dependencies]
   timing-oracle = { version = "0.1", features = ["parallel"] }
   ```
   Speedup: ~2-4x on multi-core machines

3. **Reduce bootstrap iterations for faster CI:**
   ```rust
   TimingOracle::new()
       .ci_bootstrap_iterations(2_000)    // vs 10k default
       .cov_bootstrap_iterations(500)     // vs 2k default
   ```

4. **Adjust calibration fraction:**
   ```rust
   TimingOracle::new()
       .calibration_fraction(0.2)  // vs 0.3 default
       // More data for inference, less for calibration
   ```

5. **Use appropriate presets:**
   - Development: `.quick()`
   - CI: `.balanced()`
   - Final validation: `.new()` (default)
   - Calibration studies: `.calibration()`

### Preset Comparison

With parallel feature enabled on an 8-core machine:

| Preset | Samples | Bootstrap | Runtime | Use Case |
|--------|---------|-----------|---------|----------|
| Default | 100k | 10k CI, 2k cov | ~5s | Final validation |
| Balanced | 20k | 100 CI, 50 cov | ~1s | Production CI |
| Quick | 5k | 50 CI, 50 cov | ~0.3s | Development |
| Calibration | 2k | 30 CI, 20 cov | ~0.15s | Many trials |

**Speedup vs default:**
- Balanced: ~5x faster
- Quick: ~17x faster
- Calibration: ~33x faster

**Accuracy tradeoff:**
- Balanced maintains detection accuracy with slightly higher MDE
- Quick suitable for obvious leaks (>100ns effects)
- Calibration only for statistical studies, not production testing
