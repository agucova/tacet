//! Measurement infrastructure for timing analysis.
//!
//! This module provides:
//! - High-resolution cycle counting with platform-specific implementations
//! - Sample collection with randomized interleaved design
//! - Symmetric outlier filtering for robust analysis
//! - Unified timer trait for cross-platform PMU support
//!
//! # Timer Selection
//!
//! By default, timing uses platform timers:
//! - **x86_64**: `rdtsc` instruction (~1ns resolution)
//! - **aarch64**: `cntvct_el0` virtual timer (resolution varies by SoC)
//!
//! ARM64 timer resolution depends on the SoC's counter frequency:
//! - ARMv8.6+ (Graviton4): ~1ns (1 GHz mandated by spec)
//! - Apple Silicon: ~42ns (24 MHz)
//! - Ampere Altra: ~40ns (25 MHz)
//! - Raspberry Pi 4: ~18ns (54 MHz)
//!
//! # Automatic PMU Detection
//!
//! When running with sudo/root privileges, the library automatically uses
//! cycle-accurate PMU timing:
//! - **macOS ARM64**: kperf (~0.3ns resolution)
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
//! // Force standard timer (no PMU)
//! let result = TimingOracle::new()
//!     .timer_spec(TimerSpec::Standard)
//!     .test(...);
//!
//! // Prefer PMU with fallback
//! let result = TimingOracle::new()
//!     .timer_spec(TimerSpec::PreferPmu)
//!     .test(...);
//! ```

mod collector;
mod cycle_timer;
mod outlier;
mod timer;

#[cfg(feature = "kperf")]
mod kperf_lock;

#[cfg(feature = "kperf")]
pub mod kperf;

#[cfg(feature = "perf")]
pub mod perf;

pub use collector::{Collector, Sample, MAX_BATCH_SIZE, MIN_TICKS_SINGLE_CALL, TARGET_TICKS_PER_BATCH};
pub use cycle_timer::{BoxedTimer, TimerSpec};
pub use outlier::{filter_outliers, OutlierStats};
pub use timer::{black_box, cycles_per_ns, rdtsc, Timer};
