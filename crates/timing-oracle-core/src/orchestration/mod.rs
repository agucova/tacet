//! Orchestration helpers for timing measurements.
//!
//! This module provides platform-independent logic for orchestrating
//! timing measurements, including:
//!
//! - **Pilot phase**: Determining optimal batch size K for timer resolution
//!
//! These helpers are used by both `timing-oracle` (Rust API) and
//! `timing-oracle-c` (C FFI) to share common decision logic.

mod pilot;

pub use pilot::{
    compute_batch_k, compute_batch_k_from_ticks, PilotDecision, UnmeasurableInfo,
    DEFAULT_MAX_BATCH_K, DEFAULT_TARGET_TICKS,
};
