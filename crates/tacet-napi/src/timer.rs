//! Platform-specific high-resolution timing for Node.js/Bun bindings.
//!
//! Provides cycle-accurate timing using:
//! - x86_64: `lfence; rdtsc`
//! - aarch64: `isb; mrs cntvct_el0`
//! - Fallback: `std::time::Instant`

#![allow(dead_code)]

use std::time::Instant;

/// Read the CPU cycle counter with appropriate serialization.
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
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

    let cycles: u64;
    unsafe {
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

    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    cycles
}

/// aarch64 implementation using isb + mrs cntvct_el0.
#[cfg(target_arch = "aarch64")]
#[inline]
fn rdtsc_aarch64() -> u64 {
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

    let cycles: u64;
    unsafe {
        std::arch::asm!(
            "isb",
            "mrs {}, cntvct_el0",
            out(reg) cycles,
            options(nostack, nomem),
        );
    }

    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    cycles
}

/// Fallback implementation using std::time::Instant.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn rdtsc_fallback() -> u64 {
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();

    let start = START.get_or_init(Instant::now);
    start.elapsed().as_nanos() as u64
}

/// Calibrate the cycle counter to determine cycles per nanosecond.
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
pub fn estimate_resolution_ns(cycles_per_ns: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        if cycles_per_ns > 0.0 && cycles_per_ns < 0.1 {
            1.0 / cycles_per_ns
        } else {
            measure_timer_resolution(cycles_per_ns)
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if cycles_per_ns > 0.0 {
            1.0 / cycles_per_ns
        } else {
            1.0
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        measure_timer_resolution(cycles_per_ns)
    }
}

/// Empirically measure timer resolution.
fn measure_timer_resolution(cycles_per_ns: f64) -> f64 {
    let mut min_diff = u64::MAX;

    for _ in 0..1000 {
        let t1 = rdtsc();
        let t2 = rdtsc();
        let diff = t2.saturating_sub(t1);
        if diff > 0 && diff < min_diff {
            min_diff = diff;
        }
    }

    if min_diff == u64::MAX || cycles_per_ns <= 0.0 {
        1.0
    } else {
        min_diff as f64 / cycles_per_ns
    }
}

/// Prevent compiler from optimizing away a value.
#[inline]
pub fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

/// Timer state for measurement.
#[derive(Debug, Clone)]
pub struct Timer {
    pub cycles_per_ns: f64,
    pub resolution_ns: f64,
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

    /// Convert cycles to nanoseconds.
    #[inline]
    pub fn cycles_to_ns(&self, cycles: u64) -> f64 {
        cycles as f64 / self.cycles_per_ns
    }

    /// Get timer frequency in Hz.
    pub fn frequency_hz(&self) -> u64 {
        (self.cycles_per_ns * 1_000_000_000.0) as u64
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}
