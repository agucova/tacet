# timing-oracle Implementation Notes

Supplementary implementation guidance for the timing-oracle specification. These notes cover performance optimizations, numerical stability, and platform-specific considerations that don't affect the statistical methodology but matter for a production implementation.

---

## Optimizer Barriers

Modern compilers aggressively optimize code, which can invalidate timing measurements:

- **Dead code elimination**: Unused results may be removed entirely
- **Code motion**: Operations may move outside the timed region
- **Constant folding**: Fixed inputs may be precomputed at compile time

**Solution:** Use `std::hint::black_box()` on both inputs and outputs:

```rust
let input = black_box(inputs.fixed());
let result = black_box(operation(&input));
```

For batched measurements, add `compiler_fence(SeqCst)` between iterations to prevent loop reordering/merging.

---

## Sorting and Quantile Computation

**Sorting**: Use `sort_unstable_by` with `f64::total_cmp`. Unstable sort avoids unnecessary memory overhead for timing data with many ties. `total_cmp` provides a total ordering that handles NaNs consistently (avoid `partial_cmp(...).unwrap_or(Equal)` which can corrupt sort order).

**Performance**: Full sorting outperforms selection algorithms (`select_nth_unstable`) for 9 quantiles due to cache behavior—the sorted array is reused 9 times.

---

## Numerical Stability in Covariance Estimation

The covariance matrix $\Sigma_0$ is estimated from 2,000 bootstrap samples. Naive computation (accumulate sum, then divide) suffers from catastrophic cancellation.

**Use Welford's online algorithm:**

```
for each sample x:
    n += 1
    delta = x - mean
    mean += delta / n
    M2 += delta * (x - mean)  # Uses updated mean
covariance = M2 / (n - 1)
```

For covariance matrices, extend to track cross-products. See Chan et al. (1983) for the parallel/incremental generalization.

---

## Batching Mitigations

When batching K operations (§3.4 of spec), microarchitectural state accumulates:

- **Branch predictor training**: K identical inputs vs K varied inputs
- **Cache state**: Same lines accessed K times vs varied patterns
- **μop cache**: Same instruction sequence cached vs varied

Mitigations:
- Limit $K \leq 20$ (empirically keeps artifacts below noise floor)
- Keep K identical across both classes
- Pre-generate all random inputs before timed region
- Perform inference on batch totals, not divided per-call values

---

## Platform-Specific Timer Access

### x86_64
- `rdtsc` / `rdtscp`: Cycle counter, ~1 cycle resolution
- Use `rdtscp` (serializing) or fence before `rdtsc`
- Convert cycles to ns using calibrated frequency

### AArch64 (macOS)
- `kperf`: PMU access via `mach_continuous_time()`, ~1ns resolution
- Requires entitlements on newer macOS versions
- Fallback: `cntvct_el0` (~41ns resolution)

### AArch64 (Linux)
- `perf_event_open`: PMU access, ~1 cycle resolution
- Requires `perf_event_paranoid <= 2` or `CAP_PERFMON`
- Fallback: `clock_gettime(CLOCK_MONOTONIC)`

### Portable Fallback
- `std::time::Instant`: Usually ~100ns resolution
- Adequate for operations > 1μs with batching

---

## RNG Considerations

**Counter-based RNG** (e.g., ChaCha) is preferred for reproducibility:
- Same seed + counter → same sequence
- Reproducible across runs for debugging
- Fast enough to not dominate measurement overhead

**Critical**: All random inputs must be pre-generated before the measurement loop. Generator calls inside the timed region confound the measurement.

---

## Memory Layout

For cache-friendly access during quantile computation:
- Store Fixed and Random samples in separate contiguous arrays
- Pre-sort once, reuse for all 9 quantiles
- For bootstrap: generate block indices, then gather—avoid scattered reads

---

## Parallelization Opportunities

Bootstrap iterations are embarrassingly parallel:
- Each iteration: resample → compute quantiles → record Δ*
- No data dependencies between iterations
- Use rayon or similar for B=10,000 thorough mode

Politis-White block length estimation is per-class and can run in parallel for Fixed and Random.

---

## Diagnostic Thresholds

| Diagnostic | Warning | Error |
|------------|---------|-------|
| Winsorized fraction | > 0.1% | > 5% |
| Stationarity drift | > 2×IQR (with floor) | — |
| ESS / n | < 0.5 | < 0.1 |
| Generator overhead diff | > 5% | > 10% |

---

## Testing the Implementation

### FPR Validation
Run null tests (Fixed-vs-Fixed with identical data) at scale:
- 10,000 runs at α=0.01 should yield ~100 failures (95% CI: ~80-120)
- Consistent over-rejection indicates bug in bootstrap or filtering

### Power Validation
Inject known effects at d=θ, d=1.5θ, d=2θ:
- d=θ: ~50% rejection (boundary)
- d=2θ: >90% rejection

### Discrete Mode Validation
Test with simulated discrete timers (quantize continuous data to ticks):
- Verify FPR holds at boundary
- Verify m-out-of-n scaling is correct

---

## Dependencies

Recommended crates:
- `statrs`: Beta CDF for finite-sample correction
- `rand_chacha`: Counter-based RNG
- `rayon`: Parallel bootstrap iterations (optional)
