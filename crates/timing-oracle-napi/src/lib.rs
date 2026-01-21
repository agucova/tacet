//! Node.js/Bun bindings for timing-oracle.
//!
//! This crate provides native bindings via napi-rs. The design is:
//!
//! **Native addon exports (this crate):**
//! - `rdtsc()` - Read CPU cycle counter (fast, ~1-2ns overhead)
//! - `calibrateTimer()` - Get timer info (cycles/ns, resolution)
//! - `analyze()` - One-shot analysis of pre-collected samples
//! - `calibrateSamples()` - Calibration phase for adaptive loop
//! - `adaptiveStepBatch()` - Single step of adaptive sampling
//!
//! **TypeScript implements:**
//! - Measurement loop with interleaved schedule
//! - Batch K detection (pilot phase)
//! - High-level `TimingOracle` class
//!
//! This keeps FFI overhead minimal - only 2 `rdtsc()` calls per measurement,
//! with the operation running in pure JS/TS.

#![deny(clippy::all)]

mod oracle;
mod timer;
mod types;

use napi_derive::napi;

// Re-export types
pub use types::*;

// Re-export oracle functions
pub use oracle::{
    adaptive_step_batch, analyze, calibrate_samples, AdaptiveState, AdaptiveStepResult, Calibration,
};

/// Read the CPU cycle counter.
///
/// This is a very fast operation (~1-2ns overhead) that reads:
/// - x86_64: `lfence; rdtsc` (CPU timestamp counter)
/// - aarch64: `isb; mrs cntvct_el0` (virtual timer count)
///
/// Returns raw timer ticks as a BigInt. Use `calibrateTimer()` to convert to nanoseconds.
///
/// # Example (TypeScript)
/// ```typescript
/// const start = rdtsc();
/// myOperation();
/// const elapsed = rdtsc() - start;
/// ```
#[napi]
pub fn rdtsc() -> i64 {
    timer::rdtsc() as i64
}

/// Get timer calibration information.
///
/// Performs a quick calibration (~100ms) to determine:
/// - `cyclesPerNs` - Timer ticks per nanosecond
/// - `resolutionNs` - Timer resolution in nanoseconds
/// - `frequencyHz` - Timer frequency in Hz
///
/// # Example (TypeScript)
/// ```typescript
/// const info = calibrateTimer();
/// const elapsedNs = Number(elapsed) / info.cyclesPerNs;
/// ```
#[napi]
pub fn calibrate_timer() -> TimerInfo {
    let t = timer::Timer::new();
    TimerInfo {
        cycles_per_ns: t.cycles_per_ns,
        resolution_ns: t.resolution_ns,
        frequency_hz: t.frequency_hz() as f64,
    }
}

/// Get the library version.
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Create a default configuration for a given attacker model.
#[napi]
pub fn default_config(attacker_model: AttackerModel) -> Config {
    Config {
        attacker_model,
        ..Config::default()
    }
}

/// Create a configuration for adjacent network attacker (100ns threshold).
#[napi]
pub fn config_adjacent_network() -> Config {
    default_config(AttackerModel::AdjacentNetwork)
}

/// Create a configuration for shared hardware attacker (0.6ns threshold).
#[napi]
pub fn config_shared_hardware() -> Config {
    default_config(AttackerModel::SharedHardware)
}

/// Create a configuration for remote network attacker (50us threshold).
#[napi]
pub fn config_remote_network() -> Config {
    default_config(AttackerModel::RemoteNetwork)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdtsc_monotonic() {
        let a = rdtsc();
        let b = rdtsc();
        assert!(b >= a || (a - b) < 1000);
    }

    #[test]
    fn test_timer_calibration() {
        let info = calibrate_timer();
        // Cycles per ns should be reasonable (0.01 - 10 GHz)
        assert!(info.cycles_per_ns > 0.01 && info.cycles_per_ns < 10.0);
        // Resolution should be reasonable (0.1 - 100 ns)
        assert!(info.resolution_ns > 0.1 && info.resolution_ns < 100.0);
    }

    #[test]
    fn test_default_config() {
        let config = default_config(AttackerModel::AdjacentNetwork);
        assert!((config.theta_ns() - 100.0).abs() < 1e-10);
    }
}
