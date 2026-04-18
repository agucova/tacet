//! Known-vulnerable crypto detection-rate testbed.
//!
//! Mirrors the structure of `tests/crypto.rs` (the constant-time FPR suite),
//! but every test here asserts that tacet **detects** a timing leak. Semantics
//! are inverted relative to the `crypto` binary:
//!
//! - `Outcome::Fail`  → test passes (true positive).
//! - `Outcome::Pass`  → test fails via `panic!` (false negative).
//! - `Outcome::Inconclusive` → recorded but does not fail the test (underpowered).
//!
//! Tests in this binary emit the same stdout tokens as the `crypto` binary so
//! that `scripts/measure_tpr.sh` can reuse `measure_fpr.sh`'s parser. See
//! `leaky/assert.rs` for the shared helper that enforces this mapping.
//!
//! Submodules:
//! - `injected`: controlled byte-conditional `busy_wait_ns` injection,
//!   sweeping delay magnitudes to produce a detection curve.

#[path = "leaky/assert.rs"]
mod assert;

#[path = "leaky/injected.rs"]
mod injected;
