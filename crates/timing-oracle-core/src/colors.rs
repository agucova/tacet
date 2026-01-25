//! Terminal color support for formatting output.
//!
//! This module provides color formatting that works in two modes:
//!
//! 1. **With `std` feature (default)**: Uses the `colored` crate which automatically
//!    respects `NO_COLOR`, `TERM`, and TTY detection.
//!
//! 2. **Without `std`**: Returns plain text without any formatting.

extern crate alloc;
use alloc::string::String;

// =============================================================================
// With std: use the `colored` crate for proper TTY/NO_COLOR handling
// =============================================================================

#[cfg(feature = "std")]
use colored::Colorize;

#[cfg(feature = "std")]
pub fn green(s: &str) -> String {
    s.green().to_string()
}

#[cfg(feature = "std")]
pub fn red(s: &str) -> String {
    s.red().to_string()
}

#[cfg(feature = "std")]
pub fn yellow(s: &str) -> String {
    s.yellow().to_string()
}

#[cfg(feature = "std")]
pub fn cyan(s: &str) -> String {
    s.cyan().to_string()
}

#[cfg(feature = "std")]
pub fn bold(s: &str) -> String {
    s.bold().to_string()
}

#[cfg(feature = "std")]
pub fn dim(s: &str) -> String {
    s.dimmed().to_string()
}

#[cfg(feature = "std")]
pub fn bold_green(s: &str) -> String {
    s.green().bold().to_string()
}

#[cfg(feature = "std")]
pub fn bold_red(s: &str) -> String {
    s.red().bold().to_string()
}

#[cfg(feature = "std")]
pub fn bold_yellow(s: &str) -> String {
    s.yellow().bold().to_string()
}

#[cfg(feature = "std")]
pub fn bold_cyan(s: &str) -> String {
    s.cyan().bold().to_string()
}

// =============================================================================
// Without std: plain text (no colors)
// =============================================================================

#[cfg(not(feature = "std"))]
pub fn green(s: &str) -> String {
    String::from(s)
}

#[cfg(not(feature = "std"))]
pub fn red(s: &str) -> String {
    String::from(s)
}

#[cfg(not(feature = "std"))]
pub fn yellow(s: &str) -> String {
    String::from(s)
}

#[cfg(not(feature = "std"))]
pub fn cyan(s: &str) -> String {
    String::from(s)
}

#[cfg(not(feature = "std"))]
pub fn bold(s: &str) -> String {
    String::from(s)
}

#[cfg(not(feature = "std"))]
pub fn dim(s: &str) -> String {
    String::from(s)
}

#[cfg(not(feature = "std"))]
pub fn bold_green(s: &str) -> String {
    String::from(s)
}

#[cfg(not(feature = "std"))]
pub fn bold_red(s: &str) -> String {
    String::from(s)
}

#[cfg(not(feature = "std"))]
pub fn bold_yellow(s: &str) -> String {
    String::from(s)
}

#[cfg(not(feature = "std"))]
pub fn bold_cyan(s: &str) -> String {
    String::from(s)
}
