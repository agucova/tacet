//! PMU-based cycle counting for Apple Silicon using kperf.
//!
//! This module provides cycle-accurate timing on Apple Silicon by accessing
//! hardware performance counters through Apple's private kperf framework.
//!
//! # Requirements
//!
//! - macOS on Apple Silicon (M1/M2/M3)
//! - **Must run with sudo/root privileges**
//! - Enable with `--features kperf` (enabled by default)
//!
//! # Usage
//!
//! kperf requires root privileges. Build first, then run with sudo:
//!
//! ```bash
//! cargo build --release
//! sudo ./target/release/your_binary
//! ```
//!
//! ```rust,ignore
//! use timing_oracle::measurement::kperf::PmuTimer;
//!
//! match PmuTimer::new() {
//!     Ok(mut timer) => {
//!         let cycles = timer.measure_cycles(|| my_operation());
//!         println!("Took {} cycles", cycles);
//!     }
//!     Err(e) => {
//!         eprintln!("kperf unavailable: {}", e);
//!         // Fall back to standard timer...
//!     }
//! }
//! ```
//!
//! # How it works
//!
//! Apple Silicon CPUs have performance monitoring counters (PMCs) that count
//! actual CPU cycles. These are accessed through the undocumented kperf framework.
//! Unlike the virtual timer (cntvct_el0) which runs at 24 MHz, PMCs run at CPU
//! frequency (~3 GHz), providing ~100x better resolution.
//!
//! # Implementation Notes
//!
//! This module works around a bug in kperf-rs where `PerfCounter::reset()` calls
//! `kperf_reset()`, which is a **global** reset that stops all kpc counting system-wide,
//! rather than just resetting the counter value. We avoid `reset()` entirely and instead
//! manually track deltas between `read()` calls.
//!
//! See: <https://github.com/El-Naizin/rust-kperf/issues/1>

use std::sync::atomic::{compiler_fence, Ordering};

/// Error type for PMU initialization failures.
#[derive(Debug, Clone)]
pub enum PmuError {
    /// Not running on Apple Silicon
    UnsupportedPlatform,
    /// kperf framework not available
    FrameworkNotFound,
    /// Permission denied (need sudo)
    PermissionDenied,
    /// Counter configuration failed
    ConfigurationFailed(String),
    /// Another process holds exclusive PMU access (macOS kpc limitation)
    ConcurrentAccess,
}

impl std::fmt::Display for PmuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PmuError::UnsupportedPlatform => write!(f, "PMU timing requires Apple Silicon"),
            PmuError::FrameworkNotFound => write!(f, "kperf framework not found"),
            PmuError::PermissionDenied => write!(
                f,
                "kperf requires root privileges.\n\
                 \n\
                 To use cycle-accurate PMU timing:\n\
                 \n\
                 1. Build first:  cargo build --release\n\
                 2. Run with sudo: sudo ./target/release/your_binary\n\
                 \n\
                 Alternatively, the library will fall back to the standard timer with\n\
                 adaptive batching, which works for most cryptographic operations."
            ),
            PmuError::ConfigurationFailed(msg) => write!(f, "PMU configuration failed: {}", msg),
            PmuError::ConcurrentAccess => write!(
                f,
                "Another process holds exclusive PMU access.\n\
                 \n\
                 The macOS kpc API requires system-wide exclusive access to PMU counters.\n\
                 When running tests in parallel (e.g., via nextest), only one process can\n\
                 use kperf at a time.\n\
                 \n\
                 Solutions:\n\
                 1. Run with single thread: cargo nextest run --test-threads=1\n\
                 2. Use the kperf profile: cargo nextest run --profile kperf\n\
                 \n\
                 The library will automatically fall back to the standard timer."
            ),
        }
    }
}

impl std::error::Error for PmuError {}

/// PMU-based timer for cycle-accurate measurement on Apple Silicon.
///
/// This timer uses hardware performance counters to measure actual CPU cycles,
/// providing much better resolution than the virtual timer.
///
/// # Requirements
///
/// - Must run with sudo/root privileges
/// - Only works on Apple Silicon (M1/M2/M3)
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub struct PmuTimer {
    /// The underlying kperf performance counter
    counter: kperf_rs::PerfCounter,
    /// Estimated cycles per nanosecond (CPU frequency in GHz)
    cycles_per_ns: f64,
    /// Lock guard for exclusive PMU access - released when PmuTimer is dropped
    _lock_guard: super::kperf_lock::LockGuard,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl PmuTimer {
    /// Initialize PMU counters for cycle counting.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not running on Apple Silicon
    /// - kperf framework not available
    /// - Not running with sudo/root privileges
    /// - Another process holds exclusive PMU access
    pub fn new() -> Result<Self, PmuError> {
        use super::kperf_lock::{try_acquire_default, LockResult};

        // Step 1: Acquire exclusive lock for PMU access
        // This serializes kperf initialization across processes to avoid
        // conflicts with the macOS kpc API's system-wide exclusive access model.
        let lock_guard = match try_acquire_default() {
            LockResult::Acquired(guard) => guard,
            LockResult::Timeout => {
                return Err(PmuError::ConcurrentAccess);
            }
            LockResult::IoError(e) => {
                return Err(PmuError::ConfigurationFailed(format!(
                    "Failed to acquire kperf lock: {}",
                    e
                )));
            }
        };

        // Step 2: Check permissions
        kperf_rs::check_kpc_permission().map_err(|e| match e {
            kperf_rs::error::KperfError::PermissionDenied => PmuError::PermissionDenied,
            _ => PmuError::ConfigurationFailed(format!("{:?}", e)),
        })?;

        // Step 3: Build the performance counter for cycles
        let mut counter = kperf_rs::PerfCounterBuilder::new()
            .track_event(kperf_rs::event::Event::Cycles)
            .build_counter()
            .map_err(|e| PmuError::ConfigurationFailed(format!("{:?}", e)))?;

        // Step 4: Start counting
        counter
            .start()
            .map_err(|e| PmuError::ConfigurationFailed(format!("Failed to start: {:?}", e)))?;

        // Step 5: Calibrate cycles per nanosecond
        let cycles_per_ns = Self::calibrate(&mut counter);

        Ok(Self {
            counter,
            cycles_per_ns,
            _lock_guard: lock_guard,
        })
    }

    fn calibrate(counter: &mut kperf_rs::PerfCounter) -> f64 {
        use std::time::Instant;

        // IMPORTANT: Thread counters only count cycles when the thread is RUNNING.
        // Using sleep() doesn't work because the thread isn't consuming CPU cycles.
        // We must use a busy loop that actually burns CPU cycles.
        //
        // NOTE: We avoid calling counter.reset() because kperf-rs's reset() calls
        // kperf_reset() which is a GLOBAL reset that stops all kpc counting.
        // Instead, we manually track deltas between reads.

        let mut ratios = Vec::with_capacity(10);

        // Get initial counter value
        let mut prev_cycles = match counter.read() {
            Ok(c) => c,
            Err(_) => return 3.0, // Fallback if we can't read
        };

        for _ in 0..10 {
            let start_time = Instant::now();

            // Busy loop that burns ~1ms of CPU cycles
            // Use volatile-style operations to prevent optimization
            let mut dummy: u64 = 1;
            loop {
                // Simple arithmetic that can't be optimized away easily
                dummy = dummy.wrapping_mul(6364136223846793005).wrapping_add(1);
                std::hint::black_box(dummy);

                // Check wall clock time periodically
                // (checking every iteration would dominate measurement)
                if dummy & 0xFFFF == 0 && start_time.elapsed().as_micros() >= 1000 {
                    break;
                }
            }

            // Read cycles after busy work and compute delta
            let current_cycles = match counter.read() {
                Ok(c) => c,
                Err(_) => continue,
            };
            let elapsed_nanos = start_time.elapsed().as_nanos() as u64;
            let delta_cycles = current_cycles.saturating_sub(prev_cycles);
            prev_cycles = current_cycles;

            if elapsed_nanos > 0 && delta_cycles > 0 {
                ratios.push(delta_cycles as f64 / elapsed_nanos as f64);
            }
        }

        if ratios.is_empty() {
            eprintln!("Warning: PMU calibration failed, using fallback (3.0 cycles/ns)");
            return 3.0;
        }

        ratios.sort_by(|a, b| a.total_cmp(b));
        ratios[ratios.len() / 2]
    }

    /// Measure execution time in cycles.
    ///
    /// Returns 0 if the measurement fails (e.g., read error).
    #[inline]
    pub fn measure_cycles<F, T>(&mut self, f: F) -> u64
    where
        F: FnOnce() -> T,
    {
        // NOTE: We avoid calling counter.reset() because kperf-rs's reset() calls
        // kperf_reset() which is a GLOBAL reset that stops all kpc counting.
        // Instead, we read before and after and compute the delta.
        let start = match self.counter.read() {
            Ok(c) => c,
            Err(_) => return 0,
        };
        compiler_fence(Ordering::SeqCst);
        std::hint::black_box(f());
        compiler_fence(Ordering::SeqCst);
        let end = self.counter.read().unwrap_or(start);
        end.saturating_sub(start)
    }

    /// Convert cycles to nanoseconds.
    #[inline]
    pub fn cycles_to_ns(&self, cycles: u64) -> f64 {
        cycles as f64 / self.cycles_per_ns
    }

    /// Get the calibrated cycles per nanosecond.
    pub fn cycles_per_ns(&self) -> f64 {
        self.cycles_per_ns
    }

    /// Get the timer resolution in nanoseconds (~0.3ns for 3GHz CPU).
    pub fn resolution_ns(&self) -> f64 {
        1.0 / self.cycles_per_ns
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl std::fmt::Debug for PmuTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PmuTimer")
            .field("cycles_per_ns", &self.cycles_per_ns)
            .finish()
    }
}

// Stub implementation for non-Apple Silicon platforms
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[derive(Debug)]
pub struct PmuTimer {
    _private: (),
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl PmuTimer {
    /// PMU timer is only available on Apple Silicon.
    pub fn new() -> Result<Self, PmuError> {
        Err(PmuError::UnsupportedPlatform)
    }

    #[inline]
    pub fn measure_cycles<F, T>(&mut self, _f: F) -> u64
    where
        F: FnOnce() -> T,
    {
        0
    }

    #[inline]
    pub fn cycles_to_ns(&self, cycles: u64) -> f64 {
        cycles as f64
    }

    pub fn cycles_per_ns(&self) -> f64 {
        1.0
    }

    pub fn resolution_ns(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_pmu_timer_requires_root() {
        match PmuTimer::new() {
            Ok(_) => {
                eprintln!("PMU timer initialized (running as root)");
            }
            Err(PmuError::PermissionDenied) => {
                eprintln!("PMU timer requires root (expected)");
            }
            Err(e) => {
                eprintln!("PMU timer error: {}", e);
            }
        }
    }

    #[test]
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    fn test_pmu_unsupported_platform() {
        assert!(matches!(
            PmuTimer::new(),
            Err(PmuError::UnsupportedPlatform)
        ));
    }
}
