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
    const CALIBRATION_MS: u64 = 10;
    const CALIBRATION_ITERATIONS: u32 = 5;

    let mut total_cycles = 0u64;
    let mut total_nanos = 0u64;

    for _ in 0..CALIBRATION_ITERATIONS {
        let start_cycles = rdtsc();
        let start_time = Instant::now();

        // Busy wait for calibration period
        std::thread::sleep(std::time::Duration::from_millis(CALIBRATION_MS));

        let end_cycles = rdtsc();
        let elapsed_nanos = start_time.elapsed().as_nanos() as u64;

        total_cycles += end_cycles.saturating_sub(start_cycles);
        total_nanos += elapsed_nanos;
    }

    if total_nanos == 0 {
        // Fallback to a reasonable default (3 GHz)
        return 3.0;
    }

    total_cycles as f64 / total_nanos as f64
}

/// High-level timer for measuring function execution.
///
/// Wraps the low-level cycle counter with calibration and
/// conversion to nanoseconds.
#[derive(Debug, Clone)]
pub struct Timer {
    /// Cycles per nanosecond for conversion.
    cycles_per_ns: f64,
}

impl Timer {
    /// Create a new timer with automatic calibration.
    pub fn new() -> Self {
        Self {
            cycles_per_ns: cycles_per_ns(),
        }
    }

    /// Create a timer with a known cycles-per-nanosecond value.
    ///
    /// Useful for testing or when calibration has already been done.
    pub fn with_cycles_per_ns(cycles_per_ns: f64) -> Self {
        Self { cycles_per_ns }
    }

    /// Get the calibrated cycles per nanosecond.
    pub fn cycles_per_ns(&self) -> f64 {
        self.cycles_per_ns
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
        // Should be between 0.1 GHz and 10 GHz
        assert!(cpn > 0.1 && cpn < 10.0, "cycles_per_ns = {}", cpn);
    }

    #[test]
    fn test_timer_measure() {
        let timer = Timer::new();
        let cycles = timer.measure_cycles(|| {
            let mut sum = 0u64;
            for i in 0..1000 {
                sum = sum.wrapping_add(i);
            }
            black_box(sum)
        });
        assert!(cycles > 0);
    }
}
