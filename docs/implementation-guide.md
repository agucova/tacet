# timing-oracle Implementation Guide

This guide provides detailed implementation guidance for the timing-oracle library. It covers platform-specific considerations, measurement protocols, and performance optimization techniques.

For the normative statistical specification, see [spec.md](spec.md). For language-specific API documentation, see [api-rust.md](api-rust.md), [api-c.md](api-c.md), or [api-go.md](api-go.md).

---

## 1. Timer Implementation

### 1.1 Platform-Specific Timers

timing-oracle uses the highest-resolution timer available on each platform:

| Platform | Standard Timer | Resolution | PMU Timer | PMU Resolution | PMU Requirements |
|----------|----------------|------------|-----------|----------------|------------------|
| x86_64 Linux | `rdtsc` | ~0.3 ns | `perf_event` | ~0.3 ns | sudo or CAP_PERFMON |
| x86_64 macOS | `rdtsc` | ~0.3 ns | N/A | N/A | N/A |
| ARM64 Linux | `cntvct_el0` | ~1-40 ns¹ | `perf_event` | ~0.3 ns | sudo or CAP_PERFMON |
| ARM64 macOS | `cntvct_el0` | ~41 ns | `kperf` | ~1 ns | sudo |

¹ ARM64 Linux resolution varies by SoC: ARMv8.6+ (Graviton4) ~1 ns, Ampere Altra ~40 ns, Raspberry Pi 4 ~18 ns.

### 1.2 Timer Selection Logic

Implementations SHOULD select timers in this order:

1. **PMU timer** if available and requested (highest precision)
2. **Platform-native cycle counter** (`rdtsc` on x86_64, `cntvct_el0` on ARM64)
3. **OS high-resolution timer** (`clock_gettime` on POSIX) as fallback

### 1.3 Cycle-to-Nanosecond Conversion

Results are reported in nanoseconds for interpretability. The conversion factor MUST be calibrated at startup:

1. Read the timer value
2. Sleep for a known duration (e.g., 10 ms)
3. Read the timer value again
4. Compute: `ns_per_tick = sleep_duration_ns / (end_tick - start_tick)`

For cycle counters tied to CPU frequency, use the nominal frequency from CPUID (x86) or device tree (ARM). Account for frequency scaling if the counter is not invariant.

### 1.4 Serialization Barriers

To prevent out-of-order execution from affecting measurements:

**x86_64:**
```asm
; Before rdtsc
mfence
lfence
rdtsc
; After rdtsc (if measuring end time)
rdtscp  ; or lfence after rdtsc
```

**ARM64:**
```asm
; Before cntvct_el0
isb
mrs x0, cntvct_el0
; After (if measuring end time)
isb
```

---

## 2. Pre-flight Checks

Pre-flight checks detect common problems before measurement begins. They help users diagnose setup issues without wasting time on invalid measurements.

### 2.1 Timer Sanity

Verify the timer behaves correctly:

1. **Monotonicity**: Read timer twice in succession; second reading MUST be ≥ first
2. **Reasonable rate**: Timer should advance at a plausible rate (not stuck, not wrapping unexpectedly)
3. **Resolution check**: Measure a trivial operation; if always returns 0 ticks, timer resolution is too coarse

If timer sanity fails, abort with a clear error message.

### 2.2 Harness Sanity (Fixed-vs-Fixed)

This check detects bugs in the test harness itself:

1. Split the baseline (fixed) samples in half
2. Run the full statistical analysis comparing the two halves
3. If a "leak" is detected between identical inputs, the harness has a problem

Common causes of harness sanity failures:
- Generator called inside the timed region
- Side effects from previous operations affecting timing
- Memory allocation during measurement

### 2.3 Stationarity Check

Non-stationarity (drift over time) violates bootstrap assumptions. Implement a rolling-variance check:

1. Divide samples into $W$ windows (default: 10)
2. Compute median and IQR within each window
3. Flag `stationarity_suspect = true` if:
   - Max window median differs from min by > $\max(2 \times \text{IQR}, 0.05 \times \text{global\_median})$
   - Or window variance changes monotonically by > 50%

When stationarity is suspect, continue the test but emit a quality warning. Results should be interpreted cautiously.

### 2.4 Warmup

Run the operation approximately 1,000 times before measurement to:
- Warm instruction and data caches
- Allow CPU frequency scaling to stabilize
- JIT-compile code paths (if applicable)

The warmup phase uses a mix of baseline and sample inputs to exercise all code paths.

---

## 3. Measurement Protocol

### 3.1 Input Pre-generation

All inputs MUST be generated *before* the measurement loop begins. Generating inputs inside the timed region causes false positives due to RNG timing variation.

**Correct pattern:**
```
1. Pre-generate all baseline inputs into array
2. Pre-generate all sample inputs into array
3. Create randomized schedule of which class to measure
4. For each measurement:
   a. Select input from pre-generated array based on schedule
   b. Start timer
   c. Execute operation
   d. Stop timer
   e. Record (class, timing) pair
```

### 3.2 Interleaved Randomized Sampling

Interleave baseline and sample measurements in randomized order to prevent systematic drift from biasing one class:

1. Create a schedule array with equal counts of baseline/sample labels
2. Shuffle the schedule using a seeded RNG (for reproducibility)
3. Execute measurements in schedule order

This ensures that if timing drifts over the measurement period, both classes are equally affected.

### 3.3 Acquisition Stream Storage

Store the full acquisition stream $\{(c_t, y_t)\}_{t=1}^{T}$ where:
- $c_t \in \{\text{Baseline}, \text{Sample}\}$ is the class label
- $y_t$ is the measured timing value
- $t$ is the acquisition index (time order)

The acquisition stream is essential for correct covariance estimation via stream-based block bootstrap (see spec §3.3.2). Do NOT separate samples by class before bootstrap resampling.

### 3.4 Outlier Handling (Winsorization)

Cap (don't drop) outliers to preserve signal from tail-heavy leaks while limiting the influence of extreme values:

1. Pool all timing samples (both classes)
2. Compute $t_{\text{cap}}$ = 99.99th percentile of pooled data
3. For each sample: $y_t \leftarrow \min(y_t, t_{\text{cap}})$
4. Winsorization happens before quantile computation

**Quality thresholds for capping rate:**
| Capped % | Quality | Action |
|----------|---------|--------|
| < 0.1% | Normal | Silent |
| 0.1–1% | Warning | Emit quality warning |
| 1–5% | Acceptable | Note in diagnostics |
| > 5% | TooNoisy | Quality gate triggered |

---

## 4. Adaptive Batching

On platforms with coarse timer resolution, fast operations may complete in fewer ticks than needed for reliable measurement. Adaptive batching addresses this.

### 4.1 When Batching is Needed

Enable batching when pilot measurement shows fewer than 5 ticks per operation call.

**Pilot measurement procedure:**
1. Run ~100 warmup iterations
2. Measure median ticks per operation
3. If median < 5 ticks, enable batching

### 4.2 Batch Size Selection

$$
K = \text{clamp}\left( \left\lceil \frac{50}{\text{ticks\_per\_call}} \right\rceil, 1, 20 \right)
$$

The maximum of 20 prevents microarchitectural artifacts (branch predictor saturation, cache effects) from accumulating within a batch.

### 4.3 Batched Measurement

When batching is enabled:
1. Execute the operation $K$ times consecutively
2. Record the total time for all $K$ executions as one sample
3. The threshold scales: $\theta_{\text{batch}} = K \cdot \theta$
4. Reported effects are divided by $K$ to give per-operation estimates

### 4.4 When to Disable Batching

Batching should be disabled when:
- Timer resolution is fine enough (≥5 ticks per call)
- Using PMU-based timers (kperf, perf_event)
- Operation is inherently slow (>210 ns on ~42 ns resolution timer)

---

## 5. Measurability Detection

Some operations are too fast to measure reliably even with maximum batching.

### 5.1 Unmeasurable Threshold

If ticks per call < 5 even with K=20 batching, return `Unmeasurable`:

$$
\text{ticks\_per\_call} < 5 \text{ with } K = 20 \implies \text{operation\_ns} < \frac{5 \times \text{tick\_ns}}{20}
$$

For a 42 ns timer, this means operations faster than ~10 ns cannot be reliably measured.

### 5.2 Unmeasurable Response

When an operation is unmeasurable, provide actionable guidance:
- Suggest using a PMU timer if available
- Suggest testing a more complex operation
- Report the operation's estimated timing and the timer's resolution

---

## 6. Anomaly Detection

Detect common user mistakes that would produce invalid results.

### 6.1 Input Uniqueness Check

Track uniqueness of sample (random) inputs by hashing the first 1,000 values:

| Condition | Action |
|-----------|--------|
| All identical | Error: generator is broken |
| < 50% unique | Warning: low entropy in inputs |
| ≥ 50% unique | Normal |

A common bug is capturing a random value once and reusing it:
```
// BUG: value evaluated once, reused for all samples
let value = rand::random();
generator = || value;  // Always returns the same thing!
```

### 6.2 Generator Cost Detection

If the generator takes significant time relative to the operation, it may affect results. Detect this by measuring generator overhead separately and warning if it exceeds a threshold (e.g., 10% of operation time).

---

## 7. Performance Optimization

### 7.1 Parallel Bootstrap

The 2,000-iteration block bootstrap is embarrassingly parallel. Use thread pools (rayon, OpenMP) to parallelize across CPU cores.

**Implementation notes:**
- Each thread needs its own RNG instance (seeded deterministically from iteration index)
- Accumulate covariance contributions using thread-local storage, then reduce
- Typical speedup: 4-8x on multi-core systems

### 7.2 Streaming Quantile Algorithms

For very large sample counts, computing quantiles via full sort becomes expensive. Consider:
- **P² algorithm**: O(1) space approximate quantiles (sufficient for MDE estimation)
- **t-digest**: Accurate tail quantiles with bounded memory
- **Incremental sorting**: Maintain partially sorted structure, finalize at quantile computation

For the default sample budget (100k), full sorting is acceptable.

### 7.3 Memory-Efficient Covariance Accumulation

Use Welford's online algorithm for numerically stable mean and covariance accumulation:

```
# For each bootstrap sample Δ*:
n += 1
delta = Δ* - mean
mean += delta / n
M2 += outer(delta, Δ* - mean)

# Final covariance:
Σ = M2 / (n - 1)
```

This avoids storing all bootstrap samples in memory.

---

## 8. Platform-Specific Notes

### 8.1 x86_64

**rdtsc behavior:**
- Modern CPUs have invariant TSC (constant rate regardless of frequency scaling)
- Check CPUID for `InvariantTSC` feature flag
- If not invariant, results may be affected by frequency scaling

**perf_event access:**
- Requires `CAP_PERFMON` capability or `perf_event_paranoid ≤ 2`
- To check: `cat /proc/sys/kernel/perf_event_paranoid`
- To enable: `echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid`

### 8.2 Apple Silicon (ARM64 macOS)

**cntvct_el0 characteristics:**
- Fixed 24 MHz frequency on M1/M2/M3 (~41.67 ns resolution)
- Not affected by performance/efficiency core differences
- Always available without privileges

**kperf access:**
- Requires root privileges (no capability alternative)
- Uses undocumented private framework
- May break across macOS versions

### 8.3 ARM64 Linux

**cntvct_el0 variations:**
- Frequency varies by SoC (check `/sys/devices/system/clocksource/*/available_clocksource`)
- Some cloud instances (Graviton4) have 1 GHz counters (~1 ns)
- Older SoCs may have 25-54 MHz counters (~18-40 ns)

**perf_event access:**
- Same as x86_64: requires `CAP_PERFMON` or `perf_event_paranoid ≤ 2`
- Hardware PMU support varies by SoC

---

## References

For the statistical methodology underlying these implementation details, see the [specification](spec.md).

For language-specific API documentation:
- [Rust API Reference](api-rust.md)
- [C/C++ API Reference](api-c.md)
- [Go API Reference](api-go.md)
