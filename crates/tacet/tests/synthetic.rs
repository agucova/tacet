//! Synthetic and attack pattern tests
//!
//! Tests for artificial timing scenarios:
//! - attack_patterns: Cache effects, modexp, table lookups
//! - dudect_examples: DudeCT comparison examples
//! - power_analysis: Statistical power analysis

#[path = "synthetic/attack_patterns.rs"]
mod attack_patterns;
#[path = "synthetic/dudect_examples.rs"]
mod dudect_examples;
#[path = "synthetic/power_analysis.rs"]
mod power_analysis;
