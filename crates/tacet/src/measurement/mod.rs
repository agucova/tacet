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
//! By default (`TimerSpec::Auto`), tacet uses register-based timers:
//! - **x86_64**: `rdtsc` instruction (~0.3ns resolution)
//! - **aarch64**: `cntvct_el0` virtual timer (resolution varies by SoC)
//!
//! These are preferred for timing side-channel detection because **attackers
//! measure wall-clock time**. When a remote attacker times your API, they observe
//! wall-clock duration. `rdtsc` (invariant TSC) and `cntvct_el0` directly measure
//! this, matching what attackers can observe.
//!
//! PMU-based timers (`kperf`, `perf_event`) measure CPU cycles, which can differ
//! from wall-clock time due to frequency scaling. They're available via explicit
//! [`TimerSpec::Kperf`] or [`TimerSpec::PerfEvent`] for microarchitectural research.
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

pub use collector::{
    Collector, Sample, MAX_BATCH_SIZE, MIN_TICKS_SINGLE_CALL, TARGET_TICKS_PER_BATCH,
};
pub use cycle_timer::{BoxedTimer, TimerError, TimerFallbackReason, TimerSpec};
pub use outlier::{filter_outliers, winsorize_f64, OutlierStats};
pub use timer::{black_box, cycles_per_ns, rdtsc, Timer};
