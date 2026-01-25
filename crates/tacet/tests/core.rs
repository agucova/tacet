//! Core validation tests
//!
//! These tests validate the fundamental correctness of the timing oracle:
//! - `known_leaky`: Tests that MUST detect timing leaks
//! - `known_safe`: Tests that MUST NOT false-positive on constant-time code

#[path = "core/known_leaky.rs"]
mod known_leaky;
#[path = "core/known_safe.rs"]
mod known_safe;
