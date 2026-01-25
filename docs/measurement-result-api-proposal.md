# Measurement Result API Proposal

## Problem

Current `measure_cycles()` returns `u64` with 0 as sentinel for errors:
- Cannot distinguish legitimate 0 cycles from retry exhaustion
- Silently corrupts measurements instead of failing explicitly
- No way for callers to skip invalid samples

## Proposed API

```rust
/// Error returned when measurement fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementError {
    /// Seqlock retry limit exceeded (1000 attempts).
    ///
    /// This indicates system is under extreme load or PMU event is being
    /// constantly multiplexed. Timing measurements are unreliable.
    RetryExhausted,

    /// Timer reset or read syscall failed.
    ///
    /// On Linux perf_event, this may indicate insufficient permissions
    /// or counter has been disabled.
    SyscallFailed,

    /// Measurement returned non-finite value (NaN/Inf).
    ///
    /// This can occur if cycles_per_ns calibration is invalid.
    NonFinite,
}

impl std::fmt::Display for MeasurementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RetryExhausted => write!(f, "PMU counter retry limit exceeded after 1000 attempts"),
            Self::SyscallFailed => write!(f, "perf_event syscall failed"),
            Self::NonFinite => write!(f, "measurement produced non-finite value"),
        }
    }
}

impl std::error::Error for MeasurementError {}

/// Result of a timing measurement.
pub type MeasurementResult = Result<u64, MeasurementError>;
```

## Timer Trait Changes

```rust
pub trait CycleTimer {
    /// Measure execution time in cycles.
    ///
    /// Returns `Ok(cycles)` on success, or `Err(e)` if measurement failed.
    /// Callers should skip samples that return `Err`.
    ///
    /// # Errors
    ///
    /// - `RetryExhausted`: mmap seqlock retry limit exceeded (extremely rare)
    /// - `SyscallFailed`: perf_event reset/read failed
    fn measure_cycles<F, T>(&mut self, f: F) -> MeasurementResult
    where
        F: FnOnce() -> T;

    /// Measure execution time in nanoseconds.
    ///
    /// Convenience wrapper around `measure_cycles()` with conversion.
    fn measure_ns<F, T>(&mut self, f: F) -> Result<f64, MeasurementError>
    where
        F: FnOnce() -> T,
    {
        self.measure_cycles(f).map(|cycles| self.cycles_to_ns(cycles))
    }
}
```

## Implementation (perf_mmap)

```rust
// perf_mmap.rs
pub fn read_counter(&self) -> Result<u64, MeasurementError> {
    const MAX_RETRIES: usize = 1000;

    for _ in 0..MAX_RETRIES {
        if let Some(val) = self.try_read_counter() {
            return Ok(val);
        }
    }

    // Log ONCE per retry exhaustion (not per retry attempt)
    tracing::error!(
        "perf_mmap seqlock retry exhausted after {} attempts - \
         system under extreme load or PMU event constantly multiplexed",
        MAX_RETRIES
    );
    Err(MeasurementError::RetryExhausted)
}

// perf.rs
pub fn measure_cycles<F, T>(&mut self, f: F) -> MeasurementResult
where
    F: FnOnce() -> T,
{
    #[cfg(feature = "perf-mmap")]
    if let Some(ref mmap) = self.mmap_state {
        // mmap path - errors are explicit
        let start = mmap.read_counter()?;  // Propagate error
        compiler_fence(Ordering::SeqCst);
        std::hint::black_box(f());
        compiler_fence(Ordering::SeqCst);
        let end = mmap.read_counter()?;    // Propagate error
        return Ok(end.saturating_sub(start));
    }

    // Fallback: syscall-based read
    if self.counter.reset().is_err() {
        return Err(MeasurementError::SyscallFailed);
    }
    compiler_fence(Ordering::SeqCst);
    std::hint::black_box(f());
    compiler_fence(Ordering::SeqCst);
    self.counter.read()
        .map_err(|_| MeasurementError::SyscallFailed)
}
```

## Collector Changes

```rust
// collector.rs
for class in schedule {
    let cycles = match class {
        Class::Baseline => self.timer.measure_cycles(&mut fixed),
        Class::Sample => self.timer.measure_cycles(&mut random),
    };

    match cycles {
        Ok(c) => samples.push(Sample::new(class, c)),
        Err(e) => {
            tracing::warn!("Measurement failed: {}, skipping sample", e);
            // Don't push invalid sample - continues with next measurement
        }
    }
}
```

## Oracle Changes

```rust
// oracle.rs - pilot loop
for i in 0..PILOT_SAMPLES.min(initial_samples) {
    let cycles = timer.measure_cycles(|| {
        operation(&baseline_inputs[i]);
        std::hint::black_box(());
    });

    match cycles {
        Ok(c) => pilot_cycles.push(c),
        Err(e) => {
            tracing::warn!("Pilot measurement {} failed: {}, skipping", i, e);
            // Continue to next pilot sample
        }
    }
}
```

## Migration Path

### Phase 1: Add new API alongside old (non-breaking)

```rust
// Add new method that returns Result
pub fn try_measure_cycles<F, T>(&mut self, f: F) -> MeasurementResult;

// Keep old method, implemented via new one
pub fn measure_cycles<F, T>(&mut self, f: F) -> u64 {
    self.try_measure_cycles(f).unwrap_or_else(|e| {
        tracing::error!("Measurement failed: {}", e);
        0  // Sentinel value for compatibility
    })
}
```

### Phase 2: Update callers to use try_measure_cycles

Update collector, oracle, and other internal code.

### Phase 3: Make try_measure_cycles the default (breaking change for v0.4.0)

Rename methods:
- `try_measure_cycles` → `measure_cycles`
- Old `measure_cycles` → deprecated or removed

## Benefits

1. **Explicit error handling** - No silent corruption
2. **Skip invalid samples** - Caller decides what to do
3. **No latency variation** - Failed measurements don't add fallback overhead
4. **Statistical integrity** - Invalid samples don't pollute analysis
5. **Better diagnostics** - Can log/count different error types

## Drawbacks

1. **Breaking API change** - All callers must update
2. **Ergonomics** - Must handle Result at call sites

## Alternative: Panic on Error

```rust
pub fn measure_cycles<F, T>(&mut self, f: F) -> u64 {
    let start = mmap.read_counter()
        .expect("perf_mmap retry exhausted - system unusable for timing");
    // ...
}
```

**Pros**: No API change
**Cons**: Can't recover, kills entire test run on one bad sample

## Recommendation

**Use Result-based API** because:
- Retry exhaustion is rare but not impossible
- Killing entire test run on one bad sample is too harsh
- Skipping bad samples is the right behavior for statistical analysis
- Makes error handling explicit and testable
