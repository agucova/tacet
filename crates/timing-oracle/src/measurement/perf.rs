//! PMU-based cycle counting for Linux using perf_event.
//!
//! This module provides cycle-accurate timing on Linux by accessing
//! hardware performance counters through the perf_event subsystem.
//!
//! # Requirements
//!
//! - Linux kernel with perf_event support
//! - **Must run with sudo/root privileges** OR have `CAP_PERFMON` capability OR
//!   `kernel.perf_event_paranoid <= 2`
//! - Enable with `--features perf` (enabled by default)
//!
//! # Usage
//!
//! perf_event may require elevated privileges. Build first, then run with sudo:
//!
//! ```bash
//! cargo build --release
//! sudo ./target/release/your_binary
//! ```
//!
//! Or grant capabilities to avoid needing sudo:
//!
//! ```bash
//! sudo setcap cap_perfmon=ep ./target/release/your_binary
//! ./target/release/your_binary  # No sudo needed
//! ```
//!
//! ```rust,ignore
//! use timing_oracle::measurement::perf::LinuxPerfTimer;
//!
//! match LinuxPerfTimer::new() {
//!     Ok(mut timer) => {
//!         let cycles = timer.measure_cycles(|| my_operation());
//!         println!("Took {} cycles", cycles);
//!     }
//!     Err(e) => {
//!         eprintln!("perf unavailable: {}", e);
//!         // Fall back to standard timer...
//!     }
//! }
//! ```
//!
//! # How it works
//!
//! Linux perf_event provides access to hardware performance monitoring counters (PMCs)
//! that count actual CPU cycles. Unlike coarse timers on some ARM64 SoCs, PMCs run at
//! CPU frequency (~1-5 GHz), providing sub-nanosecond resolution.

#[cfg(target_os = "linux")]
use std::sync::atomic::{compiler_fence, Ordering};

/// Error type for perf initialization failures.
#[derive(Debug, Clone)]
pub enum PerfError {
    /// Not running on Linux
    UnsupportedPlatform,
    /// Permission denied (need sudo or capabilities)
    PermissionDenied,
    /// Counter configuration failed
    ConfigurationFailed(String),
}

impl std::fmt::Display for PerfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerfError::UnsupportedPlatform => write!(f, "perf timing requires Linux"),
            PerfError::PermissionDenied => write!(
                f,
                "perf_event requires elevated privileges.\n\
                 \n\
                 To use cycle-accurate PMU timing, either:\n\
                 \n\
                 1. Run with sudo:\n\
                    cargo build --release\n\
                    sudo ./target/release/your_binary\n\
                 \n\
                 2. Grant CAP_PERFMON capability (kernel 5.8+):\n\
                    sudo setcap cap_perfmon=ep ./target/release/your_binary\n\
                 \n\
                 3. Lower perf_event_paranoid (system-wide, less secure):\n\
                    echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid\n\
                 \n\
                 Alternatively, the library will fall back to the standard timer with\n\
                 adaptive batching, which works for most cryptographic operations."
            ),
            PerfError::ConfigurationFailed(msg) => write!(f, "perf configuration failed: {}", msg),
        }
    }
}

impl std::error::Error for PerfError {}

/// Perf-based timer for cycle-accurate measurement on Linux.
///
/// This timer uses hardware performance counters to measure actual CPU cycles,
/// providing much better resolution than coarse system timers.
///
/// # Requirements
///
/// - May require sudo/root privileges or capabilities
/// - Only works on Linux
#[cfg(target_os = "linux")]
pub struct LinuxPerfTimer {
    /// The underlying perf_event counter
    counter: ::perf_event2::Counter,
    /// Estimated cycles per nanosecond (CPU frequency in GHz)
    cycles_per_ns: f64,
}

#[cfg(target_os = "linux")]
impl LinuxPerfTimer {
    /// Initialize perf counters for cycle counting.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not running on Linux
    /// - Insufficient permissions
    /// - Counter configuration fails
    pub fn new() -> Result<Self, PerfError> {
        use ::perf_event2::Builder;
        use ::perf_event2::events::Hardware;

        // Build the performance counter for CPU cycles
        let mut counter = Builder::new(Hardware::CPU_CYCLES)
            .build()
            .map_err(|e| {
                // perf_event2 returns io::Error
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    PerfError::PermissionDenied
                } else {
                    PerfError::ConfigurationFailed(format!("{:?}", e))
                }
            })?;

        // Enable counting
        counter.enable().map_err(|e| {
            PerfError::ConfigurationFailed(format!("Failed to enable: {:?}", e))
        })?;

        // Calibrate cycles per nanosecond
        let cycles_per_ns = Self::calibrate(&mut counter);

        Ok(Self {
            counter,
            cycles_per_ns,
        })
    }

    fn calibrate(counter: &mut ::perf_event2::Counter) -> f64 {
        use std::time::Instant;

        // IMPORTANT: Thread counters only count cycles when the thread is RUNNING.
        // Using sleep() doesn't work because the thread isn't consuming CPU cycles.
        // We must use a busy loop that actually burns CPU cycles.

        let mut ratios = Vec::with_capacity(10);
        for _ in 0..10 {
            // Reset counter to establish baseline
            if counter.reset().is_err() {
                continue;
            }

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
                if dummy & 0xFFFF == 0 {
                    if start_time.elapsed().as_micros() >= 1000 {
                        break;
                    }
                }
            }

            // Read cycles after busy work
            let cycles = match counter.read() {
                Ok(c) => c,
                Err(_) => continue,
            };
            let elapsed_nanos = start_time.elapsed().as_nanos() as u64;

            if elapsed_nanos > 0 && cycles > 0 {
                ratios.push(cycles as f64 / elapsed_nanos as f64);
            }
        }

        if ratios.is_empty() {
            eprintln!("Warning: perf calibration failed, using fallback (3.0 cycles/ns)");
            return 3.0;
        }

        ratios.sort_by(|a, b| a.total_cmp(b));
        ratios[ratios.len() / 2]
    }

    /// Measure execution time in cycles.
    ///
    /// Returns 0 if the measurement fails (e.g., reset or read error).
    #[inline]
    pub fn measure_cycles<F, T>(&mut self, f: F) -> u64
    where
        F: FnOnce() -> T,
    {
        // Reset to establish baseline - if this fails, we can't measure accurately
        if self.counter.reset().is_err() {
            return 0;
        }
        compiler_fence(Ordering::SeqCst);
        std::hint::black_box(f());
        compiler_fence(Ordering::SeqCst);
        self.counter.read().unwrap_or(0)
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

#[cfg(target_os = "linux")]
impl std::fmt::Debug for LinuxPerfTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinuxPerfTimer")
            .field("cycles_per_ns", &self.cycles_per_ns)
            .finish()
    }
}

// Stub implementation for non-Linux platforms
#[cfg(not(target_os = "linux"))]
/// Stub timer for non-Linux platforms.
///
/// This is a placeholder implementation that always returns errors.
#[derive(Debug)]
pub struct LinuxPerfTimer {
    _private: (),
}

#[cfg(not(target_os = "linux"))]
impl LinuxPerfTimer {
    /// perf timer is only available on Linux.
    pub fn new() -> Result<Self, PerfError> {
        Err(PerfError::UnsupportedPlatform)
    }

    /// Stub measurement (always returns 0).
    #[inline]
    pub fn measure_cycles<F, T>(&mut self, _f: F) -> u64
    where
        F: FnOnce() -> T,
    {
        0
    }

    /// Stub conversion (returns cycles as-is).
    #[inline]
    pub fn cycles_to_ns(&self, cycles: u64) -> f64 {
        cycles as f64
    }

    /// Stub cycles per nanosecond (returns 1.0).
    pub fn cycles_per_ns(&self) -> f64 {
        1.0
    }

    /// Stub resolution (returns 1.0).
    pub fn resolution_ns(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "linux")]
    fn test_perf_timer_permissions() {
        match LinuxPerfTimer::new() {
            Ok(_) => {
                eprintln!("perf timer initialized (sufficient permissions)");
            }
            Err(PerfError::PermissionDenied) => {
                eprintln!("perf timer requires elevated permissions (expected on some systems)");
            }
            Err(e) => {
                eprintln!("perf timer error: {}", e);
            }
        }
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    fn test_perf_unsupported_platform() {
        assert!(matches!(
            LinuxPerfTimer::new(),
            Err(PerfError::UnsupportedPlatform)
        ));
    }
}
