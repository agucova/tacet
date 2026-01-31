//! WASM bindings for tacet.
//!
//! This crate provides WebAssembly bindings via wasm-bindgen. The design is:
//!
//! **WASM exports (this crate):**
//! - `analyze()` - One-shot analysis of pre-collected samples
//! - `calibrateSamples()` - Calibration phase for adaptive loop
//! - `adaptiveStepBatch()` - Single step of adaptive sampling
//! - `Calibration` - Opaque handle for calibration state
//! - `AdaptiveState` - Opaque handle for adaptive sampling state
//!
//! **TypeScript implements:**
//! - Measurement loop with interleaved schedule (uses JS timers)
//! - Batch K detection (pilot phase)
//! - High-level `TimingOracle` class
//!
//! This keeps WASM overhead minimal - the measurement loop runs in pure JS,
//! with only the statistical analysis performed in WASM.

#![deny(clippy::all)]

mod oracle;
mod types;

// Re-export types for use by wasm-bindgen
pub use types::*;

// Re-export oracle functions and classes
pub use oracle::{
    adaptive_step_batch, analyze, calibrate_samples, config_adjacent_network,
    config_remote_network, config_shared_hardware, default_config, free_calibration, version,
    AdaptiveState, Calibration,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = default_config(AttackerModel::AdjacentNetwork);
        assert!((config.theta_ns() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_presets() {
        let adjacent = config_adjacent_network();
        assert!((adjacent.theta_ns() - 100.0).abs() < 1e-10);

        let shared = config_shared_hardware();
        assert!((shared.theta_ns() - 0.4).abs() < 1e-10);

        let remote = config_remote_network();
        assert!((remote.theta_ns() - 50000.0).abs() < 1e-10);
    }
}
