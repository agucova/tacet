//! Output formatting for timing analysis results.
//!
//! This module provides formatters for displaying `TestResult` in different formats:
//! - Terminal: Human-readable output with colors and box drawing
//! - JSON: Machine-readable serialization

mod json;
mod terminal;

pub use json::{to_json, to_json_pretty};
pub use terminal::format_result;
