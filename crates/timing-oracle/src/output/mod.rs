//! Output formatting for timing analysis results.
//!
//! This module provides formatters for displaying `Outcome` in different formats:
//! - Terminal: Human-readable output with colors and box drawing
//! - JSON: Machine-readable serialization

mod json;
mod terminal;

pub use json::{to_json, to_json_pretty};
pub use terminal::{
    format_debug_summary, format_diagnostics_section, format_outcome, is_verbose,
};
