//! Platform-specific timing tests
//!
//! Tests for PMU-based cycle counting (requires sudo):
//! - kperf: macOS ARM64 PMU
//! - perf: Linux perf_event

#[path = "platform/kperf.rs"]
mod kperf;
#[path = "platform/perf.rs"]
mod perf;
