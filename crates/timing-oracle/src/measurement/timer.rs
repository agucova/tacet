//! Platform-specific high-resolution timing.
//!
//! Provides cycle-accurate timing using:
//! - x86_64: `lfence; rdtsc` with compiler fence
//! - aarch64: `isb; mrs cntvct_el0`
//! - Fallback: `std::time::Instant` for other platforms

use std::hint::black_box as std_black_box;
use std::time::Instant;

/// Wrapper around `std::hint::black_box` for preventing compiler optimizations.
///
/// Use this to wrap function calls being measured to prevent the compiler
/// from optimizing away the computation or reordering it relative to timing calls.
#[inline]
pub fn black_box<T>(x: T) -> T {
    std_black_box(x)
}

/// Read the CPU cycle counter with appropriate serialization.
///
/// On x86_64, this uses `lfence; rdtsc` to ensure all prior instructions
/// complete before reading the timestamp counter.
///
/// On aarch64, this uses `isb; mrs cntvct_el0` for the virtual timer count.
///
/// On other platforms, falls back to `Instant::now()` based measurement
/// (less precise but still functional).
#[inline]
pub fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        rdtsc_x86_64()
    }

    #[cfg(target_arch = "aarch64")]
    {
        rdtsc_aarch64()
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        rdtsc_fallback()
    }
}

/// x86_64 implementation using lfence + rdtsc.
#[cfg(target_arch = "x86_64")]
#[inline]
fn rdtsc_x86_64() -> u64 {
    // Compiler fence to prevent reordering
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

    let cycles: u64;
    unsafe {
        // lfence serializes instruction execution
        // rdtsc reads the timestamp counter
        std::arch::asm!(
            "lfence",
            "rdtsc",
            "shl rdx, 32",
            "or rax, rdx",
            out("rax") cycles,
            out("rdx") _,
            options(nostack, nomem),
        );
    }

    // Compiler fence after measurement
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

    cycles
}

/// aarch64 implementation using isb + mrs cntvct_el0.
#[cfg(target_arch = "aarch64")]
#[inline]
fn rdtsc_aarch64() -> u64 {
    // Compiler fence to prevent reordering
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

    let cycles: u64;
    unsafe {
        // isb ensures all prior instructions are complete
        // mrs reads the virtual timer count register
        std::arch::asm!(
            "isb",
            "mrs {}, cntvct_el0",
            out(reg) cycles,
            options(nostack, nomem),
        );
    }

    // Compiler fence after measurement
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

    cycles
}

/// Fallback implementation using std::time::Instant.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn rdtsc_fallback() -> u64 {
    // Use a static reference point for consistency within a run
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();

    let start = START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

/// Calibrate the cycle counter to determine cycles per nanosecond.
///
/// This runs a calibration loop to measure the relationship between
/// CPU cycles and wall-clock time.
///
/// # Returns
///
/// The estimated number of cycles per nanosecond. For a 3 GHz CPU,
/// this would return approximately 3.0.
pub fn cycles_per_ns() -> f64 {
    const CALIBRATION_MS: u64 = 1;
    const CALIBRATION_ITERATIONS: usize = 100;

    let mut ratios = Vec::with_capacity(CALIBRATION_ITERATIONS);

    for _ in 0..CALIBRATION_ITERATIONS {
        let start_cycles = rdtsc();
        let start_time = Instant::now();

        std::thread::sleep(std::time::Duration::from_millis(CALIBRATION_MS));

        let end_cycles = rdtsc();
        let elapsed_nanos = start_time.elapsed().as_nanos() as u64;

        if elapsed_nanos == 0 {
            continue;
        }

        let cycles = end_cycles.saturating_sub(start_cycles);
        ratios.push(cycles as f64 / elapsed_nanos as f64);
    }

    if ratios.is_empty() {
        return 3.0;
    }

    ratios.sort_by(|a, b| a.total_cmp(b));
    let mid = ratios.len() / 2;
    if ratios.len() % 2 == 0 {
        (ratios[mid - 1] + ratios[mid]) / 2.0
    } else {
        ratios[mid]
    }
}

/// Estimate the timer resolution in nanoseconds.
///
/// This measures the minimum observable time difference between
/// consecutive timer reads.
fn estimate_resolution_ns(cycles_per_ns: f64) -> f64 {
    // Platform-specific known resolutions
    #[cfg(target_arch = "aarch64")]
    {
        // ARM64 uses cntvct_el0 virtual timer. Frequency is IMPDEF before ARMv8.6,
        // standardized to 1 GHz in ARMv8.6+. Examples:
        // - Apple Silicon: 24 MHz (~42ns) - coarse
        // - Ampere Altra: 25 MHz (~40ns) - coarse
        // - AWS Graviton4 (ARMv9): 1 GHz (~1ns) - fine
        // - Raspberry Pi 4: 54 MHz (~18ns) - moderate
        // We detect resolution dynamically via cycles_per_ns calibration.
        if cycles_per_ns > 0.0 && cycles_per_ns < 0.1 {
            // Low cycles/ns indicates coarse timer (pre-ARMv8.6 or slow SoC)
            // Resolution is 1 tick in nanoseconds
            1.0 / cycles_per_ns
        } else {
            // High-frequency timer (ARMv8.6+ or empirical measurement)
            measure_timer_resolution(cycles_per_ns)
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // x86 TSC typically runs at CPU frequency
        // Resolution is essentially 1 cycle
        if cycles_per_ns > 0.0 {
            1.0 / cycles_per_ns
        } else {
            1.0 // Fallback to 1ns
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback: measure empirically
        measure_timer_resolution(cycles_per_ns)
    }
}

/// Empirically measure timer resolution by finding minimum non-zero difference.
fn measure_timer_resolution(cycles_per_ns: f64) -> f64 {
    let mut min_diff = u64::MAX;

    // Take multiple measurements to find the minimum tick
    for _ in 0..1000 {
        let t1 = rdtsc();
        let t2 = rdtsc();
        let diff = t2.saturating_sub(t1);
        if diff > 0 && diff < min_diff {
            min_diff = diff;
        }
    }

    if min_diff == u64::MAX || cycles_per_ns <= 0.0 {
        1.0 // Fallback
    } else {
        min_diff as f64 / cycles_per_ns
    }
}

/// High-level timer for measuring function execution.
///
/// Wraps the low-level cycle counter with calibration and
/// conversion to nanoseconds.
#[derive(Debug, Clone)]
pub struct Timer {
    /// Cycles per nanosecond for conversion.
    cycles_per_ns: f64,
    /// Estimated timer resolution in nanoseconds.
    resolution_ns: f64,
}

impl Timer {
    /// Create a new timer with automatic calibration.
    pub fn new() -> Self {
        let cpn = cycles_per_ns();
        let resolution = estimate_resolution_ns(cpn);
        Self {
            cycles_per_ns: cpn,
            resolution_ns: resolution,
        }
    }

    /// Create a timer with a known cycles-per-nanosecond value.
    ///
    /// Useful for testing or when calibration has already been done.
    pub fn with_cycles_per_ns(cycles_per_ns: f64) -> Self {
        let resolution = estimate_resolution_ns(cycles_per_ns);
        Self {
            cycles_per_ns,
            resolution_ns: resolution,
        }
    }

    /// Get the calibrated cycles per nanosecond.
    pub fn cycles_per_ns(&self) -> f64 {
        self.cycles_per_ns
    }

    /// Get the estimated timer resolution in nanoseconds.
    ///
    /// On x86_64, this is typically ~0.3-1ns (CPU frequency TSC).
    /// On aarch64, this varies by SoC: ~1ns on ARMv8.6+ (Graviton4),
    /// ~40ns on Apple Silicon/Ampere Altra, ~18ns on Raspberry Pi 4.
    pub fn resolution_ns(&self) -> f64 {
        self.resolution_ns
    }

    /// Suggest minimum iterations per sample based on timer resolution.
    ///
    /// For operations faster than the timer resolution, batching multiple
    /// iterations per measurement ensures reliable timing data.
    ///
    /// # Arguments
    ///
    /// * `target_resolution_ns` - Desired effective resolution (default: 10ns)
    ///
    /// # Returns
    ///
    /// Recommended iterations per sample (minimum 1).
    pub fn suggested_iterations(&self, target_resolution_ns: f64) -> usize {
        if self.resolution_ns <= target_resolution_ns {
            1
        } else {
            // Need enough iterations so that total time >> timer resolution
            // Aim for total measured time to be at least 10x timer resolution
            let multiplier = (self.resolution_ns * 10.0 / target_resolution_ns).ceil() as usize;
            multiplier.max(1)
        }
    }

    /// Measure the execution time of a function in cycles.
    ///
    /// Uses `black_box` to prevent optimization of the measured function.
    #[inline]
    pub fn measure_cycles<F, T>(&self, f: F) -> u64
    where
        F: FnOnce() -> T,
    {
        let start = rdtsc();
        black_box(f());
        let end = rdtsc();
        end.saturating_sub(start)
    }

    /// Measure the execution time of a function in nanoseconds.
    #[inline]
    pub fn measure_ns<F, T>(&self, f: F) -> f64
    where
        F: FnOnce() -> T,
    {
        let cycles = self.measure_cycles(f);
        self.cycles_to_ns(cycles)
    }

    /// Convert cycles to nanoseconds using calibrated ratio.
    #[inline]
    pub fn cycles_to_ns(&self, cycles: u64) -> f64 {
        cycles as f64 / self.cycles_per_ns
    }

    /// Measure batched iterations and return per-iteration cycles.
    ///
    /// Runs the function `iterations` times and returns the average
    /// cycles per iteration. This is useful when timer resolution is
    /// too coarse for single-iteration measurements.
    #[inline]
    pub fn measure_batched_cycles<F, T>(&self, iterations: usize, mut f: F) -> u64
    where
        F: FnMut() -> T,
    {
        if iterations <= 1 {
            return self.measure_cycles(f);
        }

        let start = rdtsc();
        for _ in 0..iterations {
            black_box(f());
        }
        let end = rdtsc();

        let total_cycles = end.saturating_sub(start);
        total_cycles / iterations as u64
    }

    /// Measure batched iterations and return per-iteration nanoseconds.
    #[inline]
    pub fn measure_batched_ns<F, T>(&self, iterations: usize, f: F) -> f64
    where
        F: FnMut() -> T,
    {
        let cycles = self.measure_batched_cycles(iterations, f);
        self.cycles_to_ns(cycles)
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdtsc_monotonic() {
        let a = rdtsc();
        let b = rdtsc();
        // Should be monotonically increasing (or at least not going backwards significantly)
        assert!(b >= a || a.saturating_sub(b) < 1000);
    }

    #[test]
    fn test_cycles_per_ns_reasonable() {
        let cpn = cycles_per_ns();
        // Should be between 0.01 GHz and 10 GHz
        // ARM64 cntvct_el0: 0.024-1.0 cycles/ns depending on SoC
        // x86_64 TSC: 1-5 cycles/ns (CPU frequency)
        assert!(cpn > 0.01 && cpn < 10.0, "cycles_per_ns = {}", cpn);
    }

    #[test]
    fn test_timer_measure() {
        let timer = Timer::new();
        let cycles = timer.measure_cycles(|| {
            // Use enough iterations to exceed timer resolution on all platforms
            // ARM timer resolution is ~41ns, so we need > 41ns of work
            let mut sum = 0u64;
            for i in 0..100_000 {
                sum = sum.wrapping_add(black_box(i));
            }
            black_box(sum)
        });
        assert!(cycles > 0, "cycles should be positive, got {}", cycles);
    }

    #[test]
    fn test_timer_resolution_reasonable() {
        let timer = Timer::new();
        let resolution = timer.resolution_ns();
        // Resolution should be between 0.1ns and 100ns
        // ARM: ~41ns, x86: ~0.3-1ns
        assert!(
            resolution > 0.1 && resolution < 100.0,
            "resolution_ns = {}",
            resolution
        );
    }

    #[test]
    fn test_suggested_iterations() {
        let timer = Timer::new();
        let suggested = timer.suggested_iterations(10.0);

        // On x86 (resolution ~0.3ns), should suggest 1
        // On ARM (resolution ~41ns), should suggest more
        assert!(suggested >= 1, "suggested = {}", suggested);

        #[cfg(target_arch = "aarch64")]
        {
            // ARM should suggest batching for 10ns target resolution
            assert!(
                suggested > 1,
                "ARM should suggest batching, got {}",
                suggested
            );
        }
    }

    #[test]
    fn test_batched_measurement() {
        let timer = Timer::new();

        // Batched measurement should give reasonable results
        let single = timer.measure_cycles(|| black_box(42));
        let batched = timer.measure_batched_cycles(100, || black_box(42));

        // Batched per-iteration should be similar to or less than single
        // (amortizes measurement overhead)
        assert!(batched <= single * 2, "single={}, batched={}", single, batched);
    }
}
