//! Fast unit-style integration tests
//!
//! These tests validate configuration, types, and helpers without timing measurements.
//! They run quickly and are suitable for the "quick" nextest profile.

#[path = "unit/config_validation.rs"]
mod config_validation;
#[path = "unit/types.rs"]
mod types;
#[path = "unit/reliability.rs"]
mod reliability;
#[path = "unit/helpers.rs"]
mod helpers;
#[path = "unit/integration.rs"]
mod integration;
#[path = "unit/discrete_mode.rs"]
mod discrete_mode;
