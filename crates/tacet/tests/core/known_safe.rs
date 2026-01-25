//! Tests that must not false-positive on constant-time code.
//!
//! These tests verify that the oracle does NOT detect timing leaks in
//! properly implemented constant-time synthetic operations.
//!
//! Uses DudeCT's two-class pattern: fixed (non-pathological) vs random.
//! Operations are repeated enough times to be measurable with standard timers.
//!
//! A `Fail` result here would indicate a false positive bug in the oracle.

use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{skip_if_unreliable, AttackerModel, Outcome, TimingOracle};

// ============================================================================
// Test helpers
// ============================================================================

fn rand_bytes_64() -> [u8; 64] {
    let mut arr = [0u8; 64];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

fn rand_bytes_256() -> [u8; 256] {
    let mut arr = [0u8; 256];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

/// Non-pathological fixed pattern (mixed bits, not all zeros or ones)
const FIXED_64: [u8; 64] = [
    0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34,
    0xbe, 0x42, 0x27, 0x33, 0xc0, 0x90, 0x0b, 0x94, 0x15, 0xd1, 0x08, 0xc7, 0xf7, 0x9e, 0x24, 0xef,
    0x51, 0x72, 0x83, 0xcc, 0xf8, 0xc2, 0x9e, 0x49, 0x10, 0x48, 0xa1, 0x45, 0x68, 0x7f, 0x29, 0xd7,
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
];

const FIXED_256: [u8; 256] = {
    let mut arr = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        // Deterministic mixed-bit pattern
        arr[i] = ((i as u8).wrapping_mul(0x9D)).wrapping_add(0x37);
        i += 1;
    }
    arr
};

// Number of iterations to ensure measurable timing (~500ns+ per measurement)
const ITERATIONS: usize = 200;

// ============================================================================
// Synthetic constant-time operations
// ============================================================================

/// Constant-time XOR fold - identical operations regardless of data
#[inline(never)]
fn xor_fold(data: &[u8]) -> u8 {
    let mut acc = 0u8;
    for &byte in data {
        acc ^= byte;
    }
    std::hint::black_box(acc)
}

/// Constant-time comparison pattern (OR accumulator)
#[inline(never)]
fn ct_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc = 0u8;
    for i in 0..a.len() {
        acc |= a[i] ^ b[i];
    }
    std::hint::black_box(acc == 0)
}

/// Constant-time wrapping sum
#[inline(never)]
fn wrapping_sum(data: &[u8]) -> u64 {
    let mut sum = 0u64;
    for &byte in data {
        sum = sum.wrapping_add(byte as u64);
    }
    std::hint::black_box(sum)
}

// ============================================================================
// Tests: XOR fold (64 bytes × 200 iterations)
// ============================================================================

/// Test XOR fold on 64-byte buffer, repeated to be measurable.
///
/// DudeCT pattern: fixed (mixed bits) vs random.
/// XOR is constant-time - timing should not depend on data values.
#[test]
fn no_false_positive_xor_fold() {
    let inputs = InputPair::new(|| FIXED_64, rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            for _ in 0..ITERATIONS {
                xor_fold(data);
            }
        });

    eprintln!("\n[xor_fold]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "xor_fold");

    match &outcome {
        Outcome::Pass { .. } => {
            // Expected - no timing leak in constant-time code
        }
        Outcome::Fail {
            leak_probability, ..
        } => {
            panic!(
                "FALSE POSITIVE: XOR fold should be constant-time (leak_probability={:.3})",
                leak_probability
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("[INCONCLUSIVE] xor_fold: {}", reason);
        }
        _ => {}
    }
}

// ============================================================================
// Tests: Constant-time compare (64 bytes × 200 iterations)
// ============================================================================

/// Test constant-time comparison pattern.
///
/// Compares fixed secret against fixed/random data.
/// OR-accumulator pattern should be constant-time.
#[test]
fn no_false_positive_ct_compare() {
    let secret = FIXED_64;

    let inputs = InputPair::new(|| FIXED_64, rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            for _ in 0..ITERATIONS {
                ct_compare(&secret, data);
            }
        });

    eprintln!("\n[ct_compare]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "ct_compare");

    match &outcome {
        Outcome::Pass { .. } => {}
        Outcome::Fail {
            leak_probability, ..
        } => {
            panic!(
                "FALSE POSITIVE: CT compare should be constant-time (leak_probability={:.3})",
                leak_probability
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("[INCONCLUSIVE] ct_compare: {}", reason);
        }
        _ => {}
    }
}

// ============================================================================
// Tests: Wrapping sum (256 bytes × 200 iterations)
// ============================================================================

/// Test wrapping addition sum on 256-byte buffer.
///
/// Wrapping arithmetic is constant-time on all architectures.
#[test]
fn no_false_positive_wrapping_sum() {
    let inputs = InputPair::new(|| FIXED_256, rand_bytes_256);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            for _ in 0..ITERATIONS {
                wrapping_sum(data);
            }
        });

    eprintln!("\n[wrapping_sum]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "wrapping_sum");

    match &outcome {
        Outcome::Pass { .. } => {}
        Outcome::Fail {
            leak_probability, ..
        } => {
            panic!(
                "FALSE POSITIVE: Wrapping sum should be constant-time (leak_probability={:.3})",
                leak_probability
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("[INCONCLUSIVE] wrapping_sum: {}", reason);
        }
        _ => {}
    }
}

// ============================================================================
// Tests: Combined operations
// ============================================================================

/// Test combined constant-time operations.
///
/// XOR fold + wrapping sum should both be constant-time.
#[test]
fn no_false_positive_combined_ops() {
    let inputs = InputPair::new(|| FIXED_64, rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            for _ in 0..ITERATIONS {
                let xor = xor_fold(data);
                let sum = wrapping_sum(data);
                std::hint::black_box((xor, sum));
            }
        });

    eprintln!("\n[combined_ops]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "combined_ops");

    match &outcome {
        Outcome::Pass { .. } => {}
        Outcome::Fail {
            leak_probability, ..
        } => {
            panic!(
                "FALSE POSITIVE: Combined ops should be constant-time (leak_probability={:.3})",
                leak_probability
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("[INCONCLUSIVE] combined_ops: {}", reason);
        }
        _ => {}
    }
}
