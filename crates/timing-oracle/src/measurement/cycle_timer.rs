//! Unified timer abstraction for cycle-accurate timing across platforms.
//!
//! This module provides:
//! - `BoxedTimer` - An enum wrapping all timer implementations
//! - `TimerSpec` - Specification for which timer to use
//! - `TimerError` - Errors from timer selection
//!
//! # Timer Implementations
//!
//! | Platform        | Timer          | Resolution | Privileges      |
//! |-----------------|----------------|------------|-----------------|
//! | x86_64          | `rdtsc`        | ~0.3ns     | None            |
//! | ARM64 (macOS)   | `cntvct_el0`   | ~42ns      | None            |
//! | ARM64 (macOS)   | `kperf`        | ~0.3ns     | sudo            |
//! | ARM64 (Linux)   | `cntvct_el0`   | varies     | None            |
//! | Linux           | `perf_event`   | ~0.3ns     | sudo/CAP_PERFMON|
//! | All             | `std::Instant` | ~1µs       | None            |
//!
//! # User vs Power User APIs
//!
//! For most users, `TimerSpec::Auto` provides sensible defaults:
//! - Uses the best available timer for your platform
//! - Falls back gracefully when cycle-accurate timing is unavailable
//!
//! Power users (e.g., kernel developers) can select specific timers:
//! - Via enum variants: `TimerSpec::Rdtsc`, `TimerSpec::Kperf`, etc.
//! - Via runtime selection: `TimerSpec::by_name("kperf")?`

use super::Timer;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
use super::kperf::PmuTimer;

#[cfg(all(target_os = "linux", feature = "perf"))]
use super::perf::LinuxPerfTimer;

/// Error returned when timer selection fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimerError {
    /// Timer name is unknown or not available on this platform.
    UnknownOrUnavailable(String),
    /// Timer initialization failed (e.g., no privileges, concurrent access).
    InitializationFailed(String),
}

impl std::fmt::Display for TimerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimerError::UnknownOrUnavailable(name) => {
                write!(
                    f,
                    "Timer '{}' is unknown or not available on this platform. \
                     Available timers: {}",
                    name,
                    TimerSpec::available_names().join(", ")
                )
            }
            TimerError::InitializationFailed(msg) => {
                write!(f, "Timer initialization failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for TimerError {}

/// Reason why the timer fell back from high-precision cycle-accurate timing.
///
/// This is propagated to output formatters so recommendations are context-aware.
/// For example, "run with sudo" is only helpful when the issue is permissions,
/// not when the issue is concurrent access (e.g., parallel tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimerFallbackReason {
    /// No fallback occurred - using the requested timer.
    #[default]
    None,

    /// User explicitly requested the system timer.
    Requested,

    /// macOS: kperf is locked by another process (e.g., parallel tests).
    ConcurrentAccess,

    /// Not running as root/sudo - expected, not an error.
    NoPrivileges,

    /// Cycle-accurate timing initialization failed despite having elevated privileges.
    CycleCounterUnavailable,
}

impl TimerFallbackReason {
    /// Human-readable description for debug output.
    pub fn as_str(&self) -> Option<&'static str> {
        match self {
            TimerFallbackReason::None => None,
            TimerFallbackReason::Requested => Some("user requested"),
            TimerFallbackReason::ConcurrentAccess => Some("concurrent access"),
            TimerFallbackReason::NoPrivileges => Some("no sudo"),
            TimerFallbackReason::CycleCounterUnavailable => Some("unavailable"),
        }
    }
}

/// A polymorphic timer that can be any of the supported timer implementations.
///
/// This enum-based approach avoids trait object limitations while providing
/// a unified interface for all timer types.
#[allow(clippy::large_enum_variant)] // Timer size is unavoidable; avoid Box for hot path
pub enum BoxedTimer {
    /// Platform default timer (rdtsc on x86_64, cntvct_el0 on ARM64)
    Standard(Timer),

    /// macOS Apple Silicon cycle counter via kperf (PMCCNTR_EL0)
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
    Kperf(PmuTimer),

    /// Linux cycle counter via perf_event subsystem
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
///
/// # User-Friendly Variants
///
/// Most users should use one of these:
/// - [`Auto`](TimerSpec::Auto) - Best available timer (recommended)
/// - [`SystemTimer`](TimerSpec::SystemTimer) - Platform default, no privileges needed
/// - [`RequireCycleAccurate`](TimerSpec::RequireCycleAccurate) - Require high-precision or panic
///
/// # Platform-Specific Variants (Power Users)
///
/// For kernel developers or those who need specific timing primitives:
/// - [`Rdtsc`](TimerSpec::Rdtsc) - x86_64 Time Stamp Counter
/// - [`VirtualTimer`](TimerSpec::VirtualTimer) - ARM64 cntvct_el0
/// - [`Kperf`](TimerSpec::Kperf) - macOS ARM64 cycle counter via kperf
/// - [`PerfEvent`](TimerSpec::PerfEvent) - Linux cycle counter via perf_event
/// - [`StdInstant`](TimerSpec::StdInstant) - std::time::Instant (portable fallback)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TimerSpec {
    /// Auto-detect: try cycle-accurate timer first (if running as root), fall back to system timer.
    ///
    /// This is the recommended default. When running with sudo, it automatically
    /// uses cycle-accurate timing (~0.3ns). Otherwise, it falls back to the system
    /// timer with adaptive batching to compensate for coarser resolution.
    #[default]
    Auto,

    /// Always use the system timer.
    ///
    /// Uses the platform default timer:
    /// - x86_64: `rdtsc` (Time Stamp Counter, ~0.3ns resolution)
    /// - ARM64: `cntvct_el0` (Virtual Timer, ~1-42ns depending on SoC)
    ///
    /// No elevated privileges required. On ARM64 with coarse timers,
    /// adaptive batching compensates for resolution.
    SystemTimer,

    /// Require cycle-accurate timing, panic if unavailable.
    ///
    /// Explicitly requests cycle-accurate timing:
    /// - macOS ARM64: kperf (PMCCNTR_EL0, requires sudo)
    /// - Linux: perf_event (requires sudo or CAP_PERFMON)
    /// - x86_64: rdtsc (always available, no special privileges)
    ///
    /// Panics if initialization fails. Use this when you need to ensure
    /// high-precision timing is actually being used.
    RequireCycleAccurate,

    // ─────────────────────────────────────────────────────────────────────────
    // Platform-specific variants (power users)
    // ─────────────────────────────────────────────────────────────────────────

    /// Force x86_64 Time Stamp Counter (rdtsc).
    ///
    /// Resolution: ~0.3ns (cycle-accurate on modern CPUs)
    /// Privileges: None
    #[cfg(target_arch = "x86_64")]
    Rdtsc,

    /// Force ARM64 Virtual Timer Counter (cntvct_el0).
    ///
    /// Resolution: ~1-42ns (varies by SoC; ~42ns on Apple Silicon)
    /// Privileges: None
    #[cfg(target_arch = "aarch64")]
    VirtualTimer,

    /// Force macOS kperf cycle counter (PMCCNTR_EL0).
    ///
    /// Resolution: ~0.3ns (true CPU cycle counter)
    /// Privileges: sudo required
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
    Kperf,

    /// Force Linux perf_event cycle counter.
    ///
    /// Resolution: ~0.3ns (true CPU cycle counter)
    /// Privileges: sudo or CAP_PERFMON required
    #[cfg(all(target_os = "linux", feature = "perf"))]
    PerfEvent,

    /// Force std::time::Instant (portable fallback).
    ///
    /// Resolution: ~1µs (platform-dependent)
    /// Privileges: None
    ///
    /// This is the most portable option but has the lowest resolution.
    /// Only use when you need guaranteed portability over precision.
    StdInstant,
}

impl TimerSpec {
    /// Create a TimerSpec from a string name (for CLI/config use).
    ///
    /// Returns an error if the timer is not available on this platform.
    ///
    /// # Accepted Names
    ///
    /// | Input                           | TimerSpec               |
    /// |---------------------------------|-------------------------|
    /// | `"auto"`                        | `Auto`                  |
    /// | `"system"`, `"systemtimer"`     | `SystemTimer`           |
    /// | `"cycle"`, `"cycleaccurate"`    | `RequireCycleAccurate`  |
    /// | `"instant"`, `"std"`            | `StdInstant`            |
    /// | `"rdtsc"`, `"tsc"` (x86_64)     | `Rdtsc`                 |
    /// | `"cntvct"`, `"cntvct_el0"`, `"virtualtimer"` (ARM64) | `VirtualTimer` |
    /// | `"kperf"`, `"pmu"`, `"pmccntr"` (macOS ARM64) | `Kperf`    |
    /// | `"perf"`, `"perf_event"`, `"perfevent"` (Linux) | `PerfEvent` |
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let timer = TimerSpec::by_name("kperf")?;
    /// ```
    pub fn by_name(name: &str) -> Result<TimerSpec, TimerError> {
        match name.to_lowercase().as_str() {
            // User-friendly names (always available)
            "auto" => Ok(TimerSpec::Auto),
            "system" | "systemtimer" | "system_timer" => Ok(TimerSpec::SystemTimer),
            "cycle" | "cycleaccurate" | "cycle_accurate" => Ok(TimerSpec::RequireCycleAccurate),
            "instant" | "std" | "stdinstant" | "std_instant" => Ok(TimerSpec::StdInstant),

            // x86_64: rdtsc
            #[cfg(target_arch = "x86_64")]
            "rdtsc" | "tsc" => Ok(TimerSpec::Rdtsc),

            #[cfg(not(target_arch = "x86_64"))]
            "rdtsc" | "tsc" => Err(TimerError::UnknownOrUnavailable(name.to_string())),

            // ARM64: cntvct_el0
            #[cfg(target_arch = "aarch64")]
            "cntvct" | "cntvct_el0" | "virtualtimer" | "virtual_timer" => {
                Ok(TimerSpec::VirtualTimer)
            }

            #[cfg(not(target_arch = "aarch64"))]
            "cntvct" | "cntvct_el0" | "virtualtimer" | "virtual_timer" => {
                Err(TimerError::UnknownOrUnavailable(name.to_string()))
            }

            // macOS ARM64: kperf
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
            "kperf" | "pmu" | "pmccntr" | "pmccntr_el0" => Ok(TimerSpec::Kperf),

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "kperf")))]
            "kperf" | "pmu" | "pmccntr" | "pmccntr_el0" => {
                Err(TimerError::UnknownOrUnavailable(name.to_string()))
            }

            // Linux: perf_event
            #[cfg(all(target_os = "linux", feature = "perf"))]
            "perf" | "perf_event" | "perfevent" => Ok(TimerSpec::PerfEvent),

            #[cfg(not(all(target_os = "linux", feature = "perf")))]
            "perf" | "perf_event" | "perfevent" => {
                Err(TimerError::UnknownOrUnavailable(name.to_string()))
            }

            _ => Err(TimerError::UnknownOrUnavailable(name.to_string())),
        }
    }

    /// List available timer names for this platform.
    ///
    /// Returns names that can be passed to [`by_name`](Self::by_name).
    pub fn available_names() -> &'static [&'static str] {
        &[
            "auto",
            "system",
            "cycle",
            "instant",
            #[cfg(target_arch = "x86_64")]
            "rdtsc",
            #[cfg(target_arch = "aarch64")]
            "cntvct_el0",
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
            "kperf",
            #[cfg(all(target_os = "linux", feature = "perf"))]
            "perf_event",
        ]
    }

    /// Check if running with elevated privileges.
    ///
    /// Returns true if we have elevated privileges that would make fallback unexpected.
    fn has_elevated_privileges() -> bool {
        #[cfg(target_os = "macos")]
        {
            // On macOS, check if running as root
            std::process::Command::new("id")
                .arg("-u")
                .output()
                .map(|o| o.stdout == b"0\n")
                .unwrap_or(false)
        }

        #[cfg(target_os = "linux")]
        {
            // On Linux, check if running as root or have CAP_PERFMON
            // Simple check for running as root first
            std::process::Command::new("id")
                .arg("-u")
                .output()
                .map(|o| o.stdout == b"0\n")
                .unwrap_or(false)
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            false
        }
    }

    /// Create a timer based on this specification.
    ///
    /// # Cycle-Accurate Timer Auto-Detection
    ///
    /// When `Auto` or `RequireCycleAccurate` is specified:
    /// - On macOS ARM64: Tries kperf (requires sudo)
    /// - On Linux: Tries perf_event (requires sudo or CAP_PERFMON)
    /// - Falls back to system timer if cycle-accurate timing is unavailable
    ///
    /// Returns a tuple of (timer, fallback_reason) where fallback_reason indicates
    /// why the timer fell back from high-precision timing (if at all).
    pub fn create_timer(&self) -> (BoxedTimer, TimerFallbackReason) {
        match self {
            TimerSpec::SystemTimer => (
                BoxedTimer::Standard(Timer::new()),
                TimerFallbackReason::Requested,
            ),

            TimerSpec::StdInstant => {
                // StdInstant uses the same Timer implementation but is explicitly requested
                (
                    BoxedTimer::Standard(Timer::new()),
                    TimerFallbackReason::Requested,
                )
            }

            #[cfg(target_arch = "x86_64")]
            TimerSpec::Rdtsc => {
                // x86_64 rdtsc is always available
                (BoxedTimer::Standard(Timer::new()), TimerFallbackReason::None)
            }

            #[cfg(target_arch = "aarch64")]
            TimerSpec::VirtualTimer => {
                // ARM64 cntvct_el0 is always available
                (BoxedTimer::Standard(Timer::new()), TimerFallbackReason::None)
            }

            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
            TimerSpec::Kperf => {
                use super::kperf::PmuError;
                match PmuTimer::new() {
                    Ok(pmu) => (BoxedTimer::Kperf(pmu), TimerFallbackReason::None),
                    Err(PmuError::ConcurrentAccess) => {
                        panic!(
                            "Kperf: cycle counter locked by another process. \
                             Run with --test-threads=1 for exclusive access, \
                             or use TimerSpec::Auto to fall back to system timer."
                        );
                    }
                    Err(e) => {
                        panic!("Kperf: initialization failed: {:?}", e);
                    }
                }
            }

            #[cfg(all(target_os = "linux", feature = "perf"))]
            TimerSpec::PerfEvent => match LinuxPerfTimer::new() {
                Ok(perf) => (BoxedTimer::Perf(perf), TimerFallbackReason::None),
                Err(e) => {
                    panic!("PerfEvent: initialization failed: {:?}", e);
                }
            },

            TimerSpec::Auto => {
                #[allow(unused_variables)]
                let has_elevated = Self::has_elevated_privileges();

                // Try cycle-accurate timer first on supported platforms
                #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
                {
                    use super::kperf::PmuError;
                    match PmuTimer::new() {
                        Ok(pmu) => return (BoxedTimer::Kperf(pmu), TimerFallbackReason::None),
                        Err(PmuError::ConcurrentAccess) => {
                            tracing::warn!(
                                "Cycle counter (kperf) locked by another process. \
                                 Falling back to system timer."
                            );
                            eprintln!(
                                "\u{26A0} Cycle-accurate timing unavailable (concurrent access) \
                                 \u{2014} using coarse timer (~42ns).\n  \
                                 If using cargo test, run with --test-threads=1."
                            );
                            return (
                                BoxedTimer::Standard(Timer::new()),
                                TimerFallbackReason::ConcurrentAccess,
                            );
                        }
                        Err(_) if has_elevated => {
                            tracing::warn!(
                                "Running with elevated privileges but cycle counter (kperf) \
                                 unavailable. Falling back to system timer. Check system configuration."
                            );
                            eprintln!(
                                "\u{26A0} Cycle-accurate timing unavailable despite elevated \
                                 privileges \u{2014} using coarse timer (~42ns).\n  \
                                 Check system configuration."
                            );
                            return (
                                BoxedTimer::Standard(Timer::new()),
                                TimerFallbackReason::CycleCounterUnavailable,
                            );
                        }
                        Err(_) => {
                            // Normal fallback without elevated privileges
                            return (
                                BoxedTimer::Standard(Timer::new()),
                                TimerFallbackReason::NoPrivileges,
                            );
                        }
                    }
                }

                #[cfg(all(target_os = "linux", feature = "perf"))]
                {
                    match LinuxPerfTimer::new() {
                        Ok(perf) => return (BoxedTimer::Perf(perf), TimerFallbackReason::None),
                        Err(_) if has_elevated => {
                            tracing::warn!(
                                "Running with elevated privileges but cycle counter (perf_event) \
                                 unavailable. Falling back to system timer. Check kernel perf_event support."
                            );
                            eprintln!(
                                "\u{26A0} Cycle-accurate timing unavailable despite elevated \
                                 privileges \u{2014} using coarse timer.\n  \
                                 Check kernel perf_event support (CONFIG_PERF_EVENTS) or \
                                 perf_event_paranoid setting."
                            );
                            return (
                                BoxedTimer::Standard(Timer::new()),
                                TimerFallbackReason::CycleCounterUnavailable,
                            );
                        }
                        Err(_) => {
                            // Normal fallback without elevated privileges
                            return (
                                BoxedTimer::Standard(Timer::new()),
                                TimerFallbackReason::NoPrivileges,
                            );
                        }
                    }
                }

                // Fall back to system timer (non-ARM or no cycle counter feature)
                #[cfg(not(any(
                    all(target_os = "macos", target_arch = "aarch64", feature = "kperf"),
                    all(target_os = "linux", feature = "perf")
                )))]
                {
                    (BoxedTimer::Standard(Timer::new()), TimerFallbackReason::None)
                }
            }

            TimerSpec::RequireCycleAccurate => {
                // User explicitly requested cycle-accurate timing - fail hard if unavailable
                #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
                {
                    use super::kperf::PmuError;
                    match PmuTimer::new() {
                        Ok(pmu) => (BoxedTimer::Kperf(pmu), TimerFallbackReason::None),
                        Err(PmuError::ConcurrentAccess) => {
                            panic!(
                                "RequireCycleAccurate: kperf unavailable due to concurrent access. \
                                 Run with --test-threads=1 for exclusive access, \
                                 or use TimerSpec::Auto to fall back to system timer."
                            );
                        }
                        Err(e) => {
                            panic!(
                                "RequireCycleAccurate: kperf initialization failed: {:?}",
                                e
                            );
                        }
                    }
                }

                #[cfg(all(target_os = "linux", feature = "perf"))]
                {
                    match LinuxPerfTimer::new() {
                        Ok(perf) => return (BoxedTimer::Perf(perf), TimerFallbackReason::None),
                        Err(e) => {
                            panic!(
                                "RequireCycleAccurate: perf_event initialization failed: {:?}",
                                e
                            );
                        }
                    }
                }

                // On x86_64, rdtsc is already cycle-accurate
                #[cfg(all(
                    target_arch = "x86_64",
                    not(any(
                        all(target_os = "macos", target_arch = "aarch64", feature = "kperf"),
                        all(target_os = "linux", feature = "perf")
                    ))
                ))]
                {
                    (BoxedTimer::Standard(Timer::new()), TimerFallbackReason::None)
                }

                // Cycle-accurate timing not available on this platform
                #[cfg(not(any(
                    target_arch = "x86_64",
                    all(target_os = "macos", target_arch = "aarch64", feature = "kperf"),
                    all(target_os = "linux", feature = "perf")
                )))]
                {
                    panic!(
                        "RequireCycleAccurate: Cycle-accurate timing not available on this platform. \
                         Use TimerSpec::Auto or TimerSpec::SystemTimer instead."
                    );
                }
            }
        }
    }

    /// Check if cycle-accurate timing is available on this platform.
    ///
    /// Returns `true` if a cycle-accurate timer can be initialized
    /// (i.e., running with sufficient privileges on platforms that require them,
    /// or always true on x86_64 where rdtsc is available without privileges).
    pub fn cycle_accurate_available() -> bool {
        // x86_64 rdtsc is always cycle-accurate
        #[cfg(target_arch = "x86_64")]
        {
            true
        }

        // ARM64 macOS: check if kperf can be initialized
        #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
        {
            PmuTimer::new().is_ok()
        }

        // Linux with perf feature: check if perf_event can be initialized
        #[cfg(all(target_os = "linux", feature = "perf"))]
        {
            LinuxPerfTimer::new().is_ok()
        }

        // ARM64 without kperf feature (Linux without perf, or other platforms)
        #[cfg(all(
            target_arch = "aarch64",
            not(all(target_os = "macos", feature = "kperf")),
            not(all(target_os = "linux", feature = "perf"))
        ))]
        {
            false
        }

        // Other platforms (no cycle-accurate timing available)
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64"
        )))]
        {
            false
        }
    }

}

impl std::fmt::Display for TimerSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimerSpec::Auto => write!(f, "Auto"),
            TimerSpec::SystemTimer => write!(f, "SystemTimer"),
            TimerSpec::RequireCycleAccurate => write!(f, "RequireCycleAccurate"),
            TimerSpec::StdInstant => write!(f, "StdInstant"),
            #[cfg(target_arch = "x86_64")]
            TimerSpec::Rdtsc => write!(f, "Rdtsc"),
            #[cfg(target_arch = "aarch64")]
            TimerSpec::VirtualTimer => write!(f, "VirtualTimer"),
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
            TimerSpec::Kperf => write!(f, "Kperf"),
            #[cfg(all(target_os = "linux", feature = "perf"))]
            TimerSpec::PerfEvent => write!(f, "PerfEvent"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_by_name_user_friendly() {
        assert_eq!(TimerSpec::by_name("auto").unwrap(), TimerSpec::Auto);
        assert_eq!(
            TimerSpec::by_name("system").unwrap(),
            TimerSpec::SystemTimer
        );
        assert_eq!(
            TimerSpec::by_name("systemtimer").unwrap(),
            TimerSpec::SystemTimer
        );
        assert_eq!(
            TimerSpec::by_name("cycle").unwrap(),
            TimerSpec::RequireCycleAccurate
        );
        assert_eq!(
            TimerSpec::by_name("cycleaccurate").unwrap(),
            TimerSpec::RequireCycleAccurate
        );
        assert_eq!(
            TimerSpec::by_name("instant").unwrap(),
            TimerSpec::StdInstant
        );
        assert_eq!(TimerSpec::by_name("std").unwrap(), TimerSpec::StdInstant);
    }

    #[test]
    fn test_by_name_case_insensitive() {
        assert_eq!(TimerSpec::by_name("AUTO").unwrap(), TimerSpec::Auto);
        assert_eq!(TimerSpec::by_name("Auto").unwrap(), TimerSpec::Auto);
        assert_eq!(TimerSpec::by_name("SYSTEM").unwrap(), TimerSpec::SystemTimer);
    }

    #[test]
    fn test_by_name_unknown() {
        assert!(matches!(
            TimerSpec::by_name("unknown"),
            Err(TimerError::UnknownOrUnavailable(_))
        ));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_by_name_x86() {
        assert_eq!(TimerSpec::by_name("rdtsc").unwrap(), TimerSpec::Rdtsc);
        assert_eq!(TimerSpec::by_name("tsc").unwrap(), TimerSpec::Rdtsc);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_by_name_arm64() {
        assert_eq!(
            TimerSpec::by_name("cntvct_el0").unwrap(),
            TimerSpec::VirtualTimer
        );
        assert_eq!(
            TimerSpec::by_name("virtualtimer").unwrap(),
            TimerSpec::VirtualTimer
        );
    }

    #[test]
    fn test_available_names() {
        let names = TimerSpec::available_names();
        assert!(names.contains(&"auto"));
        assert!(names.contains(&"system"));
        assert!(names.contains(&"cycle"));
        assert!(names.contains(&"instant"));
    }

    #[test]
    fn test_fallback_reason_as_str() {
        assert_eq!(TimerFallbackReason::None.as_str(), None);
        assert_eq!(
            TimerFallbackReason::Requested.as_str(),
            Some("user requested")
        );
        assert_eq!(
            TimerFallbackReason::ConcurrentAccess.as_str(),
            Some("concurrent access")
        );
        assert_eq!(TimerFallbackReason::NoPrivileges.as_str(), Some("no sudo"));
        assert_eq!(
            TimerFallbackReason::CycleCounterUnavailable.as_str(),
            Some("unavailable")
        );
    }

    #[test]
    fn test_timer_error_display() {
        let err = TimerError::UnknownOrUnavailable("foo".to_string());
        let msg = err.to_string();
        assert!(msg.contains("foo"));
        assert!(msg.contains("Available timers"));
    }
}
