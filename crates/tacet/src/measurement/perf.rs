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
//! use tacet::measurement::perf::LinuxPerfTimer;
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

#[cfg(all(target_os = "linux", feature = "perf-mmap"))]
use super::perf_mmap::MmapState;

use super::error::{MeasurementError, MeasurementResult};

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
    /// The underlying perf_event counter (used for syscall fallback)
    counter: ::perf_event2::Counter,
    /// Estimated cycles per nanosecond (CPU frequency in GHz)
    cycles_per_ns: f64,
    /// mmap-based PMU access state (ARM64 Linux only, requires perf-mmap feature)
    #[cfg(feature = "perf-mmap")]
    mmap_state: Option<MmapState>,
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
        use ::perf_event2::events::Hardware;
        use ::perf_event2::Builder;

        // Build the performance counter for CPU cycles
        let mut counter = Builder::new(Hardware::CPU_CYCLES).build().map_err(|e| {
            // perf_event2 returns io::Error
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                PerfError::PermissionDenied
            } else {
                PerfError::ConfigurationFailed(format!("{:?}", e))
            }
        })?;

        // Enable counting
        counter
            .enable()
            .map_err(|e| PerfError::ConfigurationFailed(format!("Failed to enable: {:?}", e)))?;

        // Calibrate cycles per nanosecond
        let cycles_per_ns = Self::calibrate(&mut counter);

        // Try to setup mmap-based PMU access (non-fatal if fails)
        #[cfg(feature = "perf-mmap")]
        let mmap_state = Self::try_setup_mmap().ok();

        #[cfg(feature = "perf-mmap")]
        if mmap_state.is_some() {
            tracing::info!("perf-mmap enabled: using zero-overhead userspace PMU reads");
        }

        Ok(Self {
            counter,
            cycles_per_ns,
            #[cfg(feature = "perf-mmap")]
            mmap_state,
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
                if dummy & 0xFFFF == 0 && start_time.elapsed().as_micros() >= 1000 {
                    break;
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
            return 3.0;
        }

        ratios.sort_by(|a, b| a.total_cmp(b));
        ratios[ratios.len() / 2]
    }

    /// Measure execution time in cycles.
    ///
    /// On ARM64 Linux with the `perf-mmap` feature enabled, this uses zero-overhead
    /// userspace PMU reads via the `mrs` instruction. Otherwise, falls back to
    /// syscall-based reads.
    ///
    /// # Errors
    ///
    /// - `RetryExhausted`: mmap seqlock retry limit exceeded (1000 attempts)
    /// - `SyscallFailed`: perf_event reset or read syscall failed
    ///
    /// When a measurement fails, the entire sample should be skipped rather than
    /// using a sentinel value. Invalid samples corrupt statistical analysis.
    #[inline]
    pub fn measure_cycles<F, T>(&mut self, f: F) -> MeasurementResult
    where
        F: FnOnce() -> T,
    {
        #[cfg(feature = "perf-mmap")]
        if let Some(ref mmap) = self.mmap_state {
            // Zero-overhead mmap path (no syscalls)
            //
            // Propagate errors from read_counter() - if retry exhausted, caller
            // must skip this sample. The index validation (index==0 check) prevents
            // garbage values from multiplexing/migration.
            let start = mmap.read_counter()?;
            compiler_fence(Ordering::SeqCst);
            std::hint::black_box(f());
            compiler_fence(Ordering::SeqCst);
            let end = mmap.read_counter()?;
            return Ok(end.saturating_sub(start));
        }

        // Fallback: syscall-based read
        if self.counter.reset().is_err() {
            return Err(MeasurementError::SyscallFailed);
        }
        compiler_fence(Ordering::SeqCst);
        std::hint::black_box(f());
        compiler_fence(Ordering::SeqCst);
        self.counter
            .read()
            .map_err(|_| MeasurementError::SyscallFailed)
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

    /// Try to setup mmap-based PMU access (ARM64 Linux only).
    ///
    /// This is a best-effort attempt that silently fails if:
    /// - Kernel too old (<5.12)
    /// - Sysctl disabled (perf_user_access == 0)
    /// - Insufficient privileges
    /// - mmap fails
    /// - Virtualized environment without PMU
    ///
    /// Returns Ok(MmapState) on success, Err on any failure (graceful degradation).
    #[cfg(feature = "perf-mmap")]
    fn try_setup_mmap() -> Result<MmapState, Box<dyn std::error::Error>> {
        use perf_event_open_sys::bindings::{perf_event_attr, PERF_COUNT_HW_CPU_CYCLES, PERF_TYPE_HARDWARE};
        use std::os::unix::io::RawFd;

        // Prepare perf_event_attr structure
        let mut attr = unsafe { std::mem::zeroed::<perf_event_attr>() };
        attr.type_ = PERF_TYPE_HARDWARE;
        attr.size = std::mem::size_of::<perf_event_attr>() as u32;
        attr.config = PERF_COUNT_HW_CPU_CYCLES as u64;
        attr.__bindgen_anon_3.config1 = 0x3; // bit 0: userspace, bit 1: 64-bit counter
        attr.__bindgen_anon_1.sample_period = 0;
        attr.set_disabled(0);
        attr.set_exclude_kernel(1);
        attr.set_exclude_hv(1);

        // Get current CPU to avoid multiplexing when thread is pinned.
        // If the thread has been pinned via sched_setaffinity, binding the
        // perf_event to the same specific CPU prevents the kernel from
        // multiplexing it with events on other CPUs.
        #[cfg(target_os = "linux")]
        let cpu = unsafe {
            let cpu_id = libc::sched_getcpu();
            if cpu_id < 0 { -1 } else { cpu_id }
        };
        #[cfg(not(target_os = "linux"))]
        let cpu = -1;

        // Open perf event
        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                &attr as *const _,
                0,  // pid (0 = current thread)
                cpu, // cpu (specific CPU on Linux, -1 elsewhere)
                -1, // group_fd (-1 = no group)
                0,  // flags
            )
        };

        if fd < 0 {
            let err = std::io::Error::last_os_error();
            tracing::debug!("perf_event_open for mmap failed: {}", err);
            return Err(err.into());
        }

        let fd = fd as RawFd;

        // Setup mmap
        match unsafe { MmapState::new(fd) } {
            Ok(mmap_state) => {
                tracing::info!("ARM64 userspace PMU access enabled via mmap (perf-mmap feature)");
                Ok(mmap_state)
            }
            Err(e) => {
                tracing::debug!("mmap setup failed: {}", e);
                // Close the fd before returning error
                unsafe { libc::close(fd) };
                Err(Box::new(e))
            }
        }
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

    /// Stub measurement (always returns error).
    #[inline]
    pub fn measure_cycles<F, T>(&mut self, _f: F) -> MeasurementResult
    where
        F: FnOnce() -> T,
    {
        Err(MeasurementError::SyscallFailed)
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
                // perf timer initialized successfully
            }
            Err(PerfError::PermissionDenied) => {
                // perf timer requires elevated permissions (expected on some systems)
            }
            Err(_) => {
                // perf timer initialization failed
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
