//! Unified timer abstraction for cycle-accurate timing across platforms.
//!
//! This module provides:
//! - `BoxedTimer` - An enum wrapping all timer implementations
//! - `TimerSpec` - Specification for which timer to use
//!
//! Timer implementations:
//! - `Timer` - Standard platform timer (rdtsc/cntvct_el0)
//! - `PmuTimer` - macOS Apple Silicon PMU via kperf (requires sudo)
//! - `LinuxPerfTimer` - Linux perf_event PMU (requires sudo/CAP_PERFMON)

use super::Timer;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
use super::kperf::PmuTimer;

#[cfg(all(target_os = "linux", feature = "perf"))]
use super::perf::LinuxPerfTimer;

/// A polymorphic timer that can be any of the supported timer implementations.
///
/// This enum-based approach avoids trait object limitations while providing
/// a unified interface for all timer types.
pub enum BoxedTimer {
    /// Standard platform timer (rdtsc/cntvct_el0)
    Standard(Timer),

    /// macOS Apple Silicon PMU timer (kperf)
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
    Kperf(PmuTimer),

    /// Linux perf_event PMU timer
    #[cfg(all(target_os = "linux", feature = "perf"))]
    Perf(LinuxPerfTimer),
}

impl BoxedTimer {
    /// Measure execution time in cycles (or equivalent units).
    #[inline]
    pub fn measure_cycles<F, T>(&mut self, f: F) -> u64
    where
        F: FnOnce() -> T,
    {
        match self {
            BoxedTimer::Standard(t) => t.measure_cycles(f),
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
            BoxedTimer::Kperf(t) => t.measure_cycles(f),
            #[cfg(all(target_os = "linux", feature = "perf"))]
            BoxedTimer::Perf(t) => t.measure_cycles(f),
        }
    }

    /// Convert cycles to nanoseconds using calibrated ratio.
    #[inline]
    pub fn cycles_to_ns(&self, cycles: u64) -> f64 {
        match self {
            BoxedTimer::Standard(t) => t.cycles_to_ns(cycles),
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
            BoxedTimer::Kperf(t) => t.cycles_to_ns(cycles),
            #[cfg(all(target_os = "linux", feature = "perf"))]
            BoxedTimer::Perf(t) => t.cycles_to_ns(cycles),
        }
    }

    /// Get timer resolution in nanoseconds.
    pub fn resolution_ns(&self) -> f64 {
        match self {
            BoxedTimer::Standard(t) => t.resolution_ns(),
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
            BoxedTimer::Kperf(t) => t.resolution_ns(),
            #[cfg(all(target_os = "linux", feature = "perf"))]
            BoxedTimer::Perf(t) => t.resolution_ns(),
        }
    }

    /// Get the calibrated cycles per nanosecond.
    pub fn cycles_per_ns(&self) -> f64 {
        match self {
            BoxedTimer::Standard(t) => t.cycles_per_ns(),
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
            BoxedTimer::Kperf(t) => t.cycles_per_ns(),
            #[cfg(all(target_os = "linux", feature = "perf"))]
            BoxedTimer::Perf(t) => t.cycles_per_ns(),
        }
    }

    /// Timer name for diagnostics and metadata.
    pub fn name(&self) -> &'static str {
        match self {
            BoxedTimer::Standard(_) => {
                #[cfg(target_arch = "x86_64")]
                {
                    "rdtsc"
                }
                #[cfg(target_arch = "aarch64")]
                {
                    "cntvct_el0"
                }
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                {
                    "Instant"
                }
            }
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
            BoxedTimer::Kperf(_) => "kperf",
            #[cfg(all(target_os = "linux", feature = "perf"))]
            BoxedTimer::Perf(_) => "perf_event",
        }
    }
}

impl std::fmt::Debug for BoxedTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoxedTimer")
            .field("name", &self.name())
            .field("cycles_per_ns", &self.cycles_per_ns())
            .field("resolution_ns", &self.resolution_ns())
            .finish()
    }
}

/// Specification for which timer to use.
///
/// This enum allows `TimingOracle` to remain `Clone` while deferring
/// timer creation until `test()` is called.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TimerSpec {
    /// Auto-detect: try PMU first (if running as root), fall back to standard.
    ///
    /// This is the recommended default. When running with sudo, it automatically
    /// uses cycle-accurate PMU timing. Otherwise, it falls back to the standard
    /// platform timer with adaptive batching.
    #[default]
    Auto,

    /// Always use the standard platform timer.
    ///
    /// Uses rdtsc on x86_64, cntvct_el0 on ARM64. No elevated privileges required.
    /// On ARM64 with coarse timers (~42ns on Apple Silicon), adaptive batching
    /// compensates for resolution.
    Standard,

    /// Prefer PMU timer, panic if unavailable.
    ///
    /// Explicitly requests PMU timing (kperf on macOS, perf_event on Linux).
    /// Panics if PMU initialization fails (e.g., not running as root, concurrent access).
    /// Use this when you need to ensure PMU timing is actually being used.
    PreferPmu,
}

impl TimerSpec {
    /// Create a timer based on this specification.
    ///
    /// # PMU Auto-Detection
    ///
    /// When `Auto` or `PreferPmu` is specified:
    /// - On macOS ARM64: Tries kperf (requires sudo)
    /// - On Linux: Tries perf_event (requires sudo or CAP_PERFMON)
    /// - Falls back to standard timer if PMU unavailable
    pub fn create_timer(&self) -> BoxedTimer {
        match self {
            TimerSpec::Standard => BoxedTimer::Standard(Timer::new()),

            TimerSpec::Auto => {
                // Try PMU first on supported platforms, fall back silently
                #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
                {
                    use super::kperf::PmuError;
                    match PmuTimer::new() {
                        Ok(pmu) => return BoxedTimer::Kperf(pmu),
                        Err(PmuError::ConcurrentAccess) => {
                            // Another process holds the PMU lock - provide clear guidance
                            eprintln!(
                                "[timing-oracle] kperf: another process holds PMU lock. \
                                 Falling back to standard timer. \
                                 For exclusive kperf access, use: --test-threads=1"
                            );
                        }
                        Err(e) => {
                            // Log other PMU failures
                            eprintln!("[timing-oracle] kperf init failed: {:?}", e);
                        }
                    }
                }

                #[cfg(all(target_os = "linux", feature = "perf"))]
                {
                    match LinuxPerfTimer::new() {
                        Ok(perf) => return BoxedTimer::Perf(perf),
                        Err(e) => {
                            // Always log PMU failures - silent fallback hides bugs
                            eprintln!("[timing-oracle] perf init failed: {:?}", e);
                        }
                    }
                }

                // Fall back to standard timer
                BoxedTimer::Standard(Timer::new())
            }

            TimerSpec::PreferPmu => {
                // User explicitly requested PMU - fail hard if unavailable
                #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
                {
                    use super::kperf::PmuError;
                    match PmuTimer::new() {
                        Ok(pmu) => return BoxedTimer::Kperf(pmu),
                        Err(PmuError::ConcurrentAccess) => {
                            panic!(
                                "PreferPmu: kperf unavailable due to concurrent access. \
                                 Run with --test-threads=1 for exclusive PMU access, \
                                 or use TimerSpec::Auto to fall back to standard timer."
                            );
                        }
                        Err(e) => {
                            panic!("PreferPmu: kperf initialization failed: {:?}", e);
                        }
                    }
                }

                #[cfg(all(target_os = "linux", feature = "perf"))]
                {
                    match LinuxPerfTimer::new() {
                        Ok(perf) => return BoxedTimer::Perf(perf),
                        Err(e) => {
                            panic!("PreferPmu: perf_event initialization failed: {:?}", e);
                        }
                    }
                }

                // PMU not available on this platform
                #[cfg(not(any(
                    all(target_os = "macos", target_arch = "aarch64", feature = "kperf"),
                    all(target_os = "linux", feature = "perf")
                )))]
                {
                    panic!(
                        "PreferPmu: PMU timing not available on this platform. \
                         Use TimerSpec::Auto or TimerSpec::Standard instead."
                    );
                }
            }
        }
    }

    /// Check if PMU timing is available on this platform.
    ///
    /// Returns `true` if PMU can be initialized (i.e., running with sufficient privileges).
    pub fn pmu_available() -> bool {
        #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
        {
            if PmuTimer::new().is_ok() {
                return true;
            }
        }

        #[cfg(all(target_os = "linux", feature = "perf"))]
        {
            if LinuxPerfTimer::new().is_ok() {
                return true;
            }
        }

        false
    }
}

impl std::fmt::Display for TimerSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimerSpec::Auto => write!(f, "Auto"),
            TimerSpec::Standard => write!(f, "Standard"),
            TimerSpec::PreferPmu => write!(f, "PreferPmu"),
        }
    }
}
