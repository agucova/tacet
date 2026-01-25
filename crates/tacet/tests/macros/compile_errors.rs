//! Compile-time error tests for the timing_test! macro.
//!
//! These tests verify that the macro produces helpful error messages
//! when used incorrectly.

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
