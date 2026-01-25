//! Unified timer abstraction for cycle-accurate timing across platforms.
//!
//! This module provides:
//! - [`BoxedTimer`] - An enum wrapping all timer implementations
//! - [`TimerSpec`] - Specification for which timer to use
//! - [`TimerError`] - Errors from timer selection
//!
//! # Timer Selection Rationale
//!
//! `TimerSpec::Auto` uses platform-specific logic to select the best timer:
//!
//! **On x86_64:** Uses `rdtsc` (invariant TSC). This is wall-clock time, no privileges
//! needed, and already high-precision (~0.3ns).
//!
//! **On ARM64:** Tries PMU timers (`kperf`, `perf_event`) first, falls back to
//! `cntvct_el0` if unavailable. ARM64 system timers are often too coarse (42ns on
//! Apple Silicon, 40ns on Neoverse N1), so we prioritize PMU access when available.
//! Falls back gracefully without sudo.
//!
//! **Why prefer wall-clock on x86_64?** Attackers measure wall-clock time, not CPU
//! cycles. `rdtsc` (invariant TSC) directly measures wall-clock time, matching what
//! attackers observe. PMU cycle counters measure CPU cycles, which can differ from
//! wall-clock due to frequency scaling.
//!
//! PMU-based timers remain available via explicit [`TimerSpec::Kperf`] or
//! [`TimerSpec::PerfEvent`] for microarchitectural research.
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
//! - Uses register-based timers that measure wall-clock time
//! - No elevated privileges required
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
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement fails:
    /// - `RetryExhausted`: PMU counter seqlock retry limit exceeded (perf_event mmap only)
    /// - `SyscallFailed`: Timer syscall failed (PMU timers only)
    ///
    /// When a measurement fails, the caller should skip the entire sample rather than
    /// using a sentinel value, as invalid measurements corrupt statistical analysis.
    #[inline]
    pub fn measure_cycles<F, T>(&mut self, f: F) -> super::error::MeasurementResult
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
/// - [`Auto`](TimerSpec::Auto) - System timer, works everywhere (recommended)
/// - [`SystemTimer`](TimerSpec::SystemTimer) - Platform default, no privileges needed
/// - [`RequireHighPrecision`](TimerSpec::RequireHighPrecision) - Require ≤1ns resolution, panic if unavailable
/// - [`RequireCycleAccurate`](TimerSpec::RequireCycleAccurate) - Require PMU cycle counter, panic if unavailable
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
    /// Auto-detect the best available timer for your platform.
    ///
    /// This is the recommended default. Behavior:
    /// - **x86_64**: Uses `rdtsc` (~0.3ns, no privileges needed)
    /// - **ARM64**: Tries PMU timer first (kperf/perf_event with sudo), falls back
    ///   to `cntvct_el0` if unavailable. Adaptive batching compensates for coarse
    ///   timers when needed.
    ///
    /// On ARM64 without sudo, falls back gracefully to system timer.
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

    /// Require high-precision timing (≤2ns resolution), panic if unavailable.
    ///
    /// Performs runtime detection: first checks if the system timer has sufficient
    /// resolution, then falls back to PMU timers if needed.
    ///
    /// - x86_64: `rdtsc` (~0.3ns) — always succeeds
    /// - ARM64 Linux ARMv8.6+ (Graviton4): `cntvct_el0` (~1ns) — succeeds without sudo
    /// - ARM64 Linux pre-ARMv8.6: falls back to `perf_event` (requires sudo/CAP_PERFMON)
    /// - ARM64 macOS: falls back to `kperf` (requires sudo)
    ///
    /// Panics if no high-precision timer is available. Use this for CI when you
    /// need robust timing measurements.
    RequireHighPrecision,

    /// Require PMU cycle counter, panic if unavailable.
    ///
    /// Explicitly requests PMU-based cycle counting:
    /// - macOS ARM64: kperf (PMCCNTR_EL0, requires sudo)
    /// - Linux: perf_event (requires sudo or CAP_PERFMON)
    /// - x86_64: rdtsc (always available, no special privileges)
    ///
    /// Panics if initialization fails. Use this for microarchitectural research
    /// when you need true CPU cycle counts rather than wall-clock time.
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
    /// | `"highprecision"`, `"high_precision"` | `RequireHighPrecision` |
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
            "highprecision" | "high_precision" | "requirehighprecision" => {
                Ok(TimerSpec::RequireHighPrecision)
            }
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
            "highprecision",
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

    /// Create a timer based on this specification.
    ///
    /// Returns a tuple of (timer, fallback_reason) where fallback_reason indicates
    /// why the timer fell back from high-precision timing (if at all).
    ///
    /// # Timer Selection
    ///
    /// - `Auto`: x86_64 uses rdtsc; ARM64 tries PMU (kperf/perf_event) first, falls back to cntvct_el0
    /// - `SystemTimer`: Always use register-based timer (rdtsc/cntvct_el0)
    /// - `RequireHighPrecision`: Use high-precision timer (rdtsc/kperf/perf_event), panic if unavailable
    /// - `RequireCycleAccurate`: Use PMU cycle counter (kperf/perf_event), panic if unavailable
    /// - Platform-specific variants: Use the specified timer directly
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
                (
                    BoxedTimer::Standard(Timer::new()),
                    TimerFallbackReason::None,
                )
            }

            #[cfg(target_arch = "aarch64")]
            TimerSpec::VirtualTimer => {
                // ARM64 cntvct_el0 is always available
                (
                    BoxedTimer::Standard(Timer::new()),
                    TimerFallbackReason::None,
                )
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
                // On x86_64: rdtsc is already high-precision, use it directly
                #[cfg(target_arch = "x86_64")]
                {
                    return (BoxedTimer::Standard(Timer::new()), TimerFallbackReason::None);
                }

                // On ARM64: try PMU timers first (cntvct_el0 is often coarse),
                // fall back to system timer if PMU unavailable
                #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
                {
                    use super::kperf::PmuError;
                    match PmuTimer::new() {
                        Ok(pmu) => (BoxedTimer::Kperf(pmu), TimerFallbackReason::None),
                        Err(PmuError::ConcurrentAccess) => {
                            tracing::warn!(
                                "Cycle counter (kperf) locked by another process. \
                                 Falling back to system timer."
                            );
                            (
                                BoxedTimer::Standard(Timer::new()),
                                TimerFallbackReason::ConcurrentAccess,
                            )
                        }
                        Err(_) => {
                            // No sudo or other error - fall back to system timer
                            (
                                BoxedTimer::Standard(Timer::new()),
                                TimerFallbackReason::NoPrivileges,
                            )
                        }
                    }
                }

                #[cfg(all(target_os = "linux", target_arch = "aarch64", feature = "perf"))]
                {
                    match LinuxPerfTimer::new() {
                        Ok(perf) => return (BoxedTimer::Perf(perf), TimerFallbackReason::None),
                        Err(_) => {
                            // No sudo or other error - fall back to system timer
                            (
                                BoxedTimer::Standard(Timer::new()),
                                TimerFallbackReason::NoPrivileges,
                            )
                        }
                    }
                }

                // Other platforms or no perf feature: use system timer
                #[cfg(not(any(
                    target_arch = "x86_64",
                    all(target_os = "macos", target_arch = "aarch64", feature = "kperf"),
                    all(target_os = "linux", target_arch = "aarch64", feature = "perf")
                )))]
                {
                    (BoxedTimer::Standard(Timer::new()), TimerFallbackReason::None)
                }
            }

            #[allow(clippy::needless_return)] // Returns needed in cfg blocks for early exit
            TimerSpec::RequireHighPrecision => {
                // Require ≤2ns resolution timer. Check system timer first (runtime detection),
                // then fall back to PMU timers if needed.
                const HIGH_PRECISION_THRESHOLD_NS: f64 = 2.0;

                // First, check if the system timer has sufficient resolution
                let system_timer = Timer::new();
                if system_timer.resolution_ns() <= HIGH_PRECISION_THRESHOLD_NS {
                    // System timer is high precision (x86_64 rdtsc, ARMv8.6+ cntvct_el0)
                    return (BoxedTimer::Standard(system_timer), TimerFallbackReason::None);
                }

                // System timer is too coarse, try PMU timers
                #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
                {
                    use super::kperf::PmuError;
                    match PmuTimer::new() {
                        Ok(pmu) => return (BoxedTimer::Kperf(pmu), TimerFallbackReason::None),
                        Err(PmuError::ConcurrentAccess) => {
                            panic!(
                                "RequireHighPrecision: System timer resolution ({:.1}ns) exceeds \
                                 {HIGH_PRECISION_THRESHOLD_NS}ns threshold, and kperf is locked by \
                                 another process. Run with --test-threads=1 for exclusive kperf access.",
                                system_timer.resolution_ns()
                            );
                        }
                        Err(e) => {
                            panic!(
                                "RequireHighPrecision: System timer resolution ({:.1}ns) exceeds \
                                 {HIGH_PRECISION_THRESHOLD_NS}ns threshold, and kperf initialization \
                                 failed: {:?}. Run with sudo for kperf access.",
                                system_timer.resolution_ns(),
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
                                "RequireHighPrecision: System timer resolution ({:.1}ns) exceeds \
                                 {HIGH_PRECISION_THRESHOLD_NS}ns threshold, and perf_event \
                                 initialization failed: {:?}. Run with sudo or CAP_PERFMON.",
                                system_timer.resolution_ns(),
                                e
                            );
                        }
                    }
                }

                // No PMU timer available, and system timer is too coarse
                #[cfg(not(any(
                    all(target_os = "macos", target_arch = "aarch64", feature = "kperf"),
                    all(target_os = "linux", feature = "perf")
                )))]
                {
                    panic!(
                        "RequireHighPrecision: System timer resolution ({:.1}ns) exceeds \
                         {HIGH_PRECISION_THRESHOLD_NS}ns threshold, and no PMU timer is available. \
                         On ARM64, build with --features kperf (macOS) or --features perf (Linux).",
                        system_timer.resolution_ns()
                    );
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
                            panic!("RequireCycleAccurate: kperf initialization failed: {:?}", e);
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
                    (
                        BoxedTimer::Standard(Timer::new()),
                        TimerFallbackReason::None,
                    )
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
            return true;
        }

        // ARM64 macOS: check if kperf can be initialized
        #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
        {
            PmuTimer::new().is_ok()
        }

        // Linux with perf feature: check if perf_event can be initialized
        #[cfg(all(target_os = "linux", feature = "perf"))]
        {
            return LinuxPerfTimer::new().is_ok();
        }

        // ARM64 without kperf feature (Linux without perf, or other platforms)
        #[cfg(all(
            target_arch = "aarch64",
            not(all(target_os = "macos", feature = "kperf")),
            not(all(target_os = "linux", feature = "perf"))
        ))]
        {
            return false;
        }

        // Other platforms (no cycle-accurate timing available)
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            return false;
        }
    }
}

impl std::fmt::Display for TimerSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimerSpec::Auto => write!(f, "Auto"),
            TimerSpec::SystemTimer => write!(f, "SystemTimer"),
            TimerSpec::RequireHighPrecision => write!(f, "RequireHighPrecision"),
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
            TimerSpec::by_name("highprecision").unwrap(),
            TimerSpec::RequireHighPrecision
        );
        assert_eq!(
            TimerSpec::by_name("high_precision").unwrap(),
            TimerSpec::RequireHighPrecision
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
        assert_eq!(
            TimerSpec::by_name("SYSTEM").unwrap(),
            TimerSpec::SystemTimer
        );
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
        assert!(names.contains(&"highprecision"));
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
