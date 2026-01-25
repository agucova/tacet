# perf_mmap Counter Overflow Bug and Fix

## Executive Summary

**Critical Bug Found**: The custom perf_event mmap implementation had a race condition that produced garbage counter values (often near `i64::MAX = 9,223,372,036,854,775,807`), corrupting timing measurements - the core functionality of this library.

**Root Cause**: Failed to validate the `index` field before reading PMU registers. When `index==0` (counter not scheduled due to multiplexing or thread migration), we read stale PMU state from unrelated events, producing values like `9223372034710892537`.

**Impact**:
- Effect injection calibration failures (requesting 10μs → actual 140ms, 14,053× error)
- Random spurious timing measurements
- Unreliable leak detection

**Fix**: Added `index` and `pmc_width` validation in `perf_mmap.rs:try_read_counter()`.

## Technical Details

### The Bug

**File**: `crates/tacet/src/measurement/perf_mmap.rs`
**Function**: `MmapState::try_read_counter()`
**Lines**: 130-162 (before fix)

The perf_event mmap protocol requires checking the `index` field before reading PMU registers:

```c
struct perf_event_mmap_page {
    u32 index;        // PMU hardware counter index (0 = invalid!)
    i64 offset;       // Virtualization offset
    u16 pmc_width;    // Counter bit width
    // ...
};
```

**The kernel sets `index = 0` when**:
- Event is multiplexed out (kernel scheduled different event on this PMU)
- Thread migrated to different CPU (counter not mapped on new CPU)
- Event not yet started or disabled

**Our buggy code** (lines 144-150):
```rust
// Read mmap page fields
let offset = read_once!(page.offset);
let pmc_width = read_once!(page.pmc_width);

// Read PMU register via MRS instruction
let pmc_value = mrs_pmccntr_el0();  // ❌ ALWAYS reads, even if index==0!
```

### What Happens

1. **Normal operation** (`index != 0`):
   ```
   index = 3             // Using PMU counter #3
   offset = 1,000,000    // Kernel's virtualization offset
   pmc_value = 50,000    // Raw PMU register (32-bit on ARM)
   result = offset + sign_extend(pmc_value) = 1,050,000 cycles ✓
   ```

2. **Multiplexing/migration** (`index = 0`):
   ```
   index = 0             // ❌ Counter not available!
   offset = 1,000,000    // OUR event's offset
   pmc_value = ???       // ❌ DIFFERENT event's PMU state!
   result = offset + sign_extend(???) = GARBAGE
   ```

   When we read `PMCCNTR_EL0` with `index==0`, we get PMU state from whatever event is currently using that counter (or undefined state). Combining this with OUR event's offset produces nonsense:

   ```
   offset = 1,000,000
   pmc_value = 0xFFFFFFFF (from unrelated event)
   sign_extend(0xFFFFFFFF, 32bit) = -1
   result = 1,000,000 + (-1) = 999,999 cycles ❌
   ```

   Worse, if `offset` itself is large and sign extension produces a large negative number:
   ```
   offset = 9,223,372,036,000,000,000
   pmc_value = 0x7FFFFFFF
   sign_extend(0x7FFFFFFF, 32bit) = 2,147,483,647
   result = 9,223,372,038,147,483,647 ≈ i64::MAX ❌
   ```

### Observable Symptoms

**Effect Injection**:
```
[calibrate_with_perf] total_cycles=3600157 ✓
[calibrate_with_perf] total_cycles=9223372034710892537 ❌ (overflow!)
```

**Validation Warnings**:
```
WARNING: busy_wait_ns accuracy issue detected!
Requested: 10000ns, Actual: 140534996ns, Ratio: 14053.50x
```

**Benchmarks**:
```
FPR: 100% (should be 0%)       ❌ All measurements corrupted
Power: 0% detection            ❌ No useful measurements
```

## The Fix

### Part 1: Validate index before PMU read

**File**: `crates/tacet/src/measurement/perf_mmap.rs:186-198`

```rust
// CRITICAL: Validate index before reading PMU register
//
// index == 0 means the hardware counter is NOT currently available:
// - Event multiplexed out (kernel scheduled different event on this PMU)
// - Thread migrated to different CPU (counter not mapped on new CPU)
// - Event disabled or not yet started
//
// Reading PMCCNTR_EL0 with index==0 produces GARBAGE (often near i64::MAX)
// because we'd be combining offset from OUR event with PMU state from a
// DIFFERENT event. This is the root cause of spurious overflow values.
if index == 0 {
    return None;  // Retry until event is rescheduled
}

// Validate pmc_width to prevent sign extension bugs
if pmc_width == 0 || pmc_width > 64 {
    return None;  // Retry with valid metadata
}
```

### Part 2: Overflow detection in effect injection

**File**: `crates/tacet/src/helpers/effect.rs:131-158`

Even with index validation, spurious large cycle counts can occur. Added retry logic:

```rust
const MAX_REASONABLE_CYCLES: u64 = 10_000_000_000; // 10B cycles (~4s at 2.5GHz)

for _attempt in 0..3 {
    let total_cycles = calibration_timer.measure_cycles(|| {
        for _ in 0..100_000 {
            spin_bundle(1);
        }
    });

    // Reject obvious overflow/error values
    if total_cycles > MAX_REASONABLE_CYCLES {
        continue;  // Retry
    }

    let cost_per_unit = (total_cycles as f64 / 100_000.0) / timer.cycles_per_ns();
    return Self { cost_per_unit };
}

// If all attempts failed, fall back to Instant-based calibration
Self::calibrate_with_instant()
```

### Part 3: Defense in Depth - Thread Pinning

**File**: `crates/tacet/src/oracle.rs:580-590`

Oracle already uses CPU affinity pinning to reduce thread migration:

```rust
let _affinity_guard = if self.config.cpu_affinity {
    match AffinityGuard::try_pin() {
        AffinityResult::Pinned(guard) => Some(guard),
        AffinityResult::NotPinned { reason } => {
            tracing::warn!("CPU affinity unavailable: {}", reason);
            None
        }
    }
} else {
    None
};
```

**Platform behavior**:
- **Linux**: Enforced pinning via `sched_setaffinity` (no special privileges needed)
- **macOS**: Advisory hint via `THREAD_AFFINITY_POLICY` (kernel may still migrate)

**Why both**:
1. Thread pinning **reduces likelihood** of `index==0` (fewer migrations)
2. Index validation **handles** `index==0` when migrations do occur (defense in depth)

## Verification

### Test Results (Neoverse N1, ARM64 Linux)

**Before fix**:
```
FPR: 100%/95%/50% (false positives everywhere)
Power: 0% (no detection)
Effect injection: 10μs requested → 140ms actual (14,053× error)
```

**After fix**:
```
FPR: 0.0% across all noise models ✓
Power: 15-60% at various effect sizes ✓
Effect injection: No validation warnings ✓
Overflow values: Detected and rejected via retry ✓
```

### Test Commands

```bash
# Run calibration tests
cargo test --release -p tacet --test calibration -- --nocapture

# Run benchmark with realistic timing
cargo run --release -p tacet-bench -- --preset quick --tools tacet --realistic
```

## Documentation Updates

### perf_mmap.rs

Added comprehensive documentation to `MmapState::read_counter()` explaining:
- Retry conditions (seqlock, index==0, invalid pmc_width)
- Multiplexing & thread migration behavior
- Return values and error handling
- Caller expectations (use `saturating_sub`, expect retries)
- Performance characteristics (typical 2-4 cycles, retry 10-50ns, multiplexing ~1ms)

### Caller Contract

From `perf.rs:LinuxPerfTimer::measure_cycles()`:

```rust
// Note: read_counter() returns 0 if retry limit is exhausted (very rare).
// The counter value CAN legitimately be 0, so we cannot check for it here.
// If retry exhaustion occurs, read_counter() logs an error via tracing::error.
// The index validation (index==0 check) prevents garbage values from
// multiplexing/migration, which was the root cause of overflow bugs.
let start = mmap.read_counter();
// ... measured work ...
let end = mmap.read_counter();
return end.saturating_sub(start);
```

## Lessons Learned

1. **Always validate kernel metadata** before trusting hardware register reads
2. **Read the specification**: Linux perf_event mmap protocol clearly documents `index==0` as "counter unavailable"
3. **Defense in depth**: Thread pinning reduces `index==0`, but validation handles it when it occurs
4. **Overflow detection**: Even with correct implementation, add sanity checks for impossibly large values
5. **Test on target platform**: Bug only manifested on ARM64 Linux under load/multiplexing

## References

- [Linux perf_event mmap protocol](https://man7.org/linux/man-pages/man2/perf_event_open.2.html) (search for "mmap layout")
- [ARM PMU documentation](https://developer.arm.com/documentation/ddi0595/latest/)
- tacet issue: [Effect injection calibration failures on ARM64](#)
