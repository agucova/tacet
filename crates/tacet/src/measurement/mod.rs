//! Measurement infrastructure for timing analysis.
//!
//! This module provides:
//! - High-resolution cycle counting with platform-specific implementations
//! - Sample collection with randomized interleaved design
//! - Symmetric outlier filtering for robust analysis
//! - Unified timer abstraction for cross-platform timing
//!
//! # Timer Selection Rationale
//!
//! `TimerSpec::Auto` uses platform-specific logic:
//! - **x86_64**: `rdtsc` (~0.3ns, wall-clock time, no privileges needed)
//! - **ARM64**: Tries PMU first (`kperf`/`perf_event` with sudo), falls back to
//!   `cntvct_el0` if unavailable. ARM64 system timers are often too coarse.
//!
//! **Why this matters:** On x86_64, `rdtsc` (invariant TSC) is already high-precision
//! and measures wall-clock time (what attackers observe). On ARM64, system timers
//! are often coarse (42ns on Apple Silicon, 40ns on Neoverse N1), so we prioritize
//! PMU access when available for better precision.
//!
//! PMU-based timers are available via explicit [`TimerSpec::Kperf`] or
//! [`TimerSpec::PerfEvent`] for microarchitectural research.
//!
//! # ARM64 Timer Resolution
//!
//! ARM64 timer resolution depends on the SoC's counter frequency:
//! - ARMv8.6+ (Graviton4): ~1ns (1 GHz mandated by spec)
//! - Apple Silicon: ~42ns (24 MHz)
//! - Ampere Altra: ~40ns (25 MHz)
//! - Raspberry Pi 4: ~18ns (54 MHz)
//!
//! On platforms with coarse resolution, adaptive batching compensates automatically.
//!
//! # Explicit Timer Selection
//!
//! Use [`TimerSpec`] to control timer selection:
//!
//! ```ignore
//! use tacet::{TimingOracle, TimerSpec};
//!
//! // Default: register-based timer (rdtsc/cntvct_el0)
//! let result = TimingOracle::new()
//!     .timer_spec(TimerSpec::Auto)
//!     .test(...);
//!
//! // Require high-precision timing (≤2ns), recommended for CI
//! // Uses runtime detection: system timer if sufficient, else PMU timer
//! let result = TimingOracle::new()
//!     .timer_spec(TimerSpec::RequireHighPrecision)
//!     .test(...);
//!
//! // Require PMU cycle counter (for microarchitectural research)
//! let result = TimingOracle::new()
//!     .timer_spec(TimerSpec::RequireCycleAccurate)
//!     .test(...);
//! ```
//!
//! # Platform-Specific Timers
//!
//! For kernel developers or microarchitectural research:
//!
//! ```ignore
//! use tacet::TimerSpec;
//!
//! // Select by name at runtime (for CLI tools)
//! let timer = TimerSpec::by_name("kperf")?;
//!
//! // Or use platform-specific variants directly
//! #[cfg(target_arch = "x86_64")]
//! let timer = TimerSpec::Rdtsc;
//!
//! #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
//! let timer = TimerSpec::Kperf;  // Requires sudo
//! ```

pub mod affinity;
mod collector;
mod cycle_timer;
mod error;
mod outlier;
#[cfg(feature = "thread-priority")]
pub mod priority;
mod timer;

#[cfg(all(feature = "kperf", target_os = "macos"))]
mod kperf_lock;

#[cfg(all(feature = "kperf", target_os = "macos"))]
pub mod kperf;

#[cfg(all(feature = "perf", target_os = "linux"))]
pub mod perf;

#[cfg(all(feature = "perf-mmap", target_os = "linux"))]
pub(crate) mod perf_mmap;

pub use collector::{
    Collector, Sample, MAX_BATCH_SIZE, MIN_TICKS_SINGLE_CALL, TARGET_TICKS_PER_BATCH,
};
pub use cycle_timer::{BoxedTimer, TimerError, TimerFallbackReason, TimerSpec};
pub use error::{MeasurementError, MeasurementResult};
pub use outlier::{filter_outliers, winsorize_f64, OutlierStats};
pub use timer::{black_box, cycles_per_ns, rdtsc, Timer};
