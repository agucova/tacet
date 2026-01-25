//! Measurement infrastructure for timing analysis.
//!
//! This module provides:
//! - High-resolution cycle counting with platform-specific implementations
//! - Sample collection with randomized interleaved design
//! - Symmetric outlier filtering for robust analysis
//! - Unified timer abstraction for cross-platform cycle-accurate timing
//!
//! # Timer Selection
//!
//! By default, timing uses platform timers:
//! - **x86_64**: `rdtsc` instruction (~0.3ns resolution, cycle-accurate)
//! - **aarch64**: `cntvct_el0` virtual timer (resolution varies by SoC)
//!
//! ARM64 timer resolution depends on the SoC's counter frequency:
//! - ARMv8.6+ (Graviton4): ~1ns (1 GHz mandated by spec)
//! - Apple Silicon: ~42ns (24 MHz)
//! - Ampere Altra: ~40ns (25 MHz)
//! - Raspberry Pi 4: ~18ns (54 MHz)
//!
//! # Automatic Cycle-Accurate Timer Detection
//!
//! When running with sudo/root privileges, the library automatically uses
//! cycle-accurate timing:
//! - **macOS ARM64**: kperf (PMCCNTR_EL0, ~0.3ns resolution)
//! - **Linux**: perf_event (~0.3ns resolution)
//!
//! No code changes needed - just run with sudo:
//!
//! ```bash
//! cargo build --release
//! sudo ./target/release/your_binary
//! ```
//!
//! # Manual Timer Selection
//!
//! Use `TimerSpec` to explicitly control timer selection:
//!
//! ```ignore
//! use timing_oracle::{TimingOracle, TimerSpec};
//!
//! // Force system timer (no cycle-accurate timing)
//! let result = TimingOracle::new()
//!     .timer_spec(TimerSpec::SystemTimer)
//!     .test(...);
//!
//! // Require cycle-accurate timing (panics if unavailable)
//! let result = TimingOracle::new()
//!     .timer_spec(TimerSpec::RequireCycleAccurate)
//!     .test(...);
//! ```
//!
//! # Power User: Platform-Specific Timers
//!
//! For kernel developers or those who need specific timing primitives:
//!
//! ```ignore
//! use timing_oracle::TimerSpec;
//!
//! // Select by name at runtime (for CLI tools)
//! let timer = TimerSpec::by_name("kperf")?;
//!
//! // Or use platform-specific variants directly
//! #[cfg(target_arch = "x86_64")]
//! let timer = TimerSpec::Rdtsc;
//!
//! #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
//! let timer = TimerSpec::Kperf;
//! ```

pub mod affinity;
pub mod priority;
mod collector;
mod cycle_timer;
mod outlier;
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
