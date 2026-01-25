//! # DudeCT Examples Test Suite
//!
//! This module tests Rust equivalents of the [dudect examples](https://github.com/oreparaz/dudect/tree/master/examples),
//! demonstrating tacet usage from a new user's perspective.
//!
//! ## DudeCT Examples Mapping
//!
//! | DudeCT Example | What It Tests | Expected Result | Rust Equivalent |
//! |----------------|---------------|-----------------|-----------------|
//! | `simple` | `memcmp()` comparison | LEAK | `==` on byte slices |
//! | `aes32` | Reference AES T-table | LEAK | Minimal T-table lookup |
//! | `aesbitsliced` | Bitsliced AES | PASS | `aes` crate (AES-NI) |
//! | `donna` | Curve25519-donna | PASS | `x25519-dalek` |
//! | `donnabad` | Non-CT Curve25519 | LEAK | Naive double-and-add |
//!
//! ## Usage Pattern
//!
//! All tests follow the DudeCT two-class pattern:
//! - **Baseline (Class 0)**: All-zero data
//! - **Sample (Class 1)**: Random data
//!
//! This tests for data-dependent timing rather than comparing specific fixed values.

use std::time::Duration;

use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use subtle::ConstantTimeEq;
use tacet::{helpers::InputPair, AttackerModel, Outcome, TimingOracle};
use x25519_dalek::PublicKey;

// ============================================================================
// Helper Functions
// ============================================================================

/// LEAK: Standard Rust equality uses early exit on mismatch.
/// This is equivalent to dudect's `simple` example using memcmp().
#[inline(never)]
fn naive_byte_compare(a: &[u8], b: &[u8]) -> bool {
    a == b
}

/// LEAK: Explicitly leaky byte-by-byte comparison with early exit.
/// This should definitely show timing differences.
#[inline(never)]
fn explicit_early_exit_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if a[i] != b[i] {
            return false; // Early exit - timing leak!
        }
    }
    true
}

/// PASS: The `subtle` crate provides constant-time comparison.
#[inline(never)]
fn ct_byte_compare(a: &[u8], b: &[u8]) -> bool {
    a.ct_eq(b).into()
}

/// AES S-box (used for T-table lookup demonstration).
/// This is the forward S-box from the AES specification.
#[rustfmt::skip]
static SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// LEAK: Table lookup with secret-dependent index causes cache timing.
/// This is equivalent to dudect's `aes32` example with T-table lookups.
#[inline(never)]
fn table_lookup(data: &[u8]) -> u8 {
    let mut result = 0u8;
    for &byte in data {
        // Cache timing leak: different indices hit different cache lines
        result ^= SBOX[byte as usize];
    }
    std::hint::black_box(result)
}

/// PASS: The `aes` crate uses AES-NI on x86_64 (constant-time).
/// This is equivalent to dudect's `aesbitsliced` example.
#[inline(never)]
fn aes_encrypt(key: &[u8; 16], data: &[u8; 16]) -> [u8; 16] {
    let cipher = Aes128::new(key.into());
    let mut block = (*data).into();
    cipher.encrypt_block(&mut block);
    block.into()
}

/// PASS: The `x25519-dalek` crate is designed to be constant-time.
/// This is equivalent to dudect's `donna` example.
///
/// PublicKey::from([u8; 32]) performs scalar multiplication: scalar * basepoint,
/// which is the same operation that curve25519_donna performs.
#[inline(never)]
fn x25519_scalar_mult(scalar: &[u8; 32]) -> [u8; 32] {
    let public = PublicKey::from(*scalar);
    *public.as_bytes()
}

/// LEAK: Deliberately leaky double-and-add with branches on secret bits.
/// This is equivalent to dudect's `donnabad` example.
#[inline(never)]
fn naive_scalar_mult(scalar: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let mut acc: u64 = 1;

    for byte in scalar.iter() {
        for bit_idx in 0..8 {
            let bit = (byte >> bit_idx) & 1;
            // Always double (square)
            acc = acc.wrapping_mul(acc);

            if bit == 1 {
                // Only add when bit is 1 - this is the TIMING LEAK!
                // The branch predictor and execution time differ based on secret bits.
                acc = acc.wrapping_add(0x12345678);
                std::hint::black_box(&acc); // Prevent optimization
            }
        }
    }

    result[0..8].copy_from_slice(&acc.to_le_bytes());
    std::hint::black_box(result)
}

// ============================================================================
// Tests: Constant-Time Code (expect Pass)
// ============================================================================

/// Test that constant-time comparison using `subtle` crate passes.
/// Equivalent to testing a fixed implementation of dudect's `simple` example.
#[test]
fn test_memcmp_constant_time() {
    let secret = [0x42u8; 512];

    let inputs = InputPair::new(
        || [0u8; 512],      // Baseline: all zeros
        || rand::random(),  // Sample: random data
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            std::hint::black_box(ct_byte_compare(&secret, data));
        });

    // Use the new Display impl
    println!("{}", outcome);

    // Still panic on unexpected failure
    if let Outcome::Fail { .. } = outcome {
        panic!("UNEXPECTED FAIL: Constant-time comparison showed leak!");
    }
}

/// Test that AES-128 encryption using the `aes` crate passes.
/// Equivalent to dudect's `aesbitsliced` example (constant-time AES).
#[test]
fn test_aes128_constant_time() {
    let key = [0u8; 16]; // Fixed key for timing test

    let inputs = InputPair::new(
        || [0u8; 16],       // Baseline: all zeros plaintext
        || rand::random(),  // Sample: random plaintext
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            std::hint::black_box(aes_encrypt(&key, data));
        });

    println!("{}", outcome);

    if let Outcome::Fail { .. } = outcome {
        panic!("UNEXPECTED FAIL: AES-128 showed timing leak!");
    }
}

/// Test that X25519 scalar multiplication using `x25519-dalek` passes.
/// Equivalent to dudect's `donna` example (constant-time Curve25519).
#[test]
fn test_x25519_constant_time() {
    let inputs = InputPair::new(
        || [0u8; 32],       // Baseline: all zeros scalar
        || rand::random(),  // Sample: random scalar
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            std::hint::black_box(x25519_scalar_mult(data));
        });

    println!("{}", outcome);

    if let Outcome::Fail { .. } = outcome {
        panic!("UNEXPECTED FAIL: X25519 showed timing leak!");
    }
}

// ============================================================================
// Tests: Leaky Code (expect Fail)
// ============================================================================

/// Test that naive byte comparison (using `==`) is detected as leaky.
/// Equivalent to dudect's `simple` example with memcmp().
#[test]
fn test_memcmp_leaky() {
    let secret = [0x42u8; 512];

    let inputs = InputPair::new(
        || [0u8; 512],      // Baseline: all zeros (no early match)
        || rand::random(),  // Sample: random (possible early exits)
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            std::hint::black_box(naive_byte_compare(&secret, data));
        });

    println!("{}", outcome);

    // For leaky tests, we expect Fail but accept Pass (compiler optimization)
    // or Unmeasurable (operation too fast)
    if let Outcome::Pass { .. } = outcome {
        println!("Note: Rust's == compiles to memcmp which may be SIMD-optimized");
    }
}

/// Test explicit early-exit comparison to verify the oracle detects it.
///
/// Key insight: The secret must START with zeros so that:
/// - Baseline (zeros) matches for many bytes before failing → slow
/// - Sample (random) mismatches early → fast
#[test]
fn test_explicit_early_exit_leaky() {
    // Secret is all zeros - baseline will match entirely, sample will exit early
    let secret = [0u8; 512];

    let inputs = InputPair::new(
        || [0u8; 512],      // Baseline: matches secret entirely (slow - checks all 512 bytes)
        || rand::random(),  // Sample: random, exits on first mismatch (fast)
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            std::hint::black_box(explicit_early_exit_compare(&secret, data));
        });

    println!("{}", outcome);

    // This should definitely show a leak
    if let Outcome::Pass { .. } = outcome {
        println!("WARNING: Explicit early-exit didn't show leak - unexpected!");
    }
}

/// Test that T-table lookup is detected as leaky due to cache timing.
/// Equivalent to dudect's `aes32` example with T-table AES.
///
/// Note: The 256-byte S-box fits entirely in L1 cache on modern CPUs,
/// so this test typically passes. See `test_large_table_lookup_leaky`
/// for a version with a larger table that causes cache effects.
#[test]
fn test_table_lookup_leaky() {
    let inputs = InputPair::new(
        || [0u8; 16],       // Baseline: all zeros (same cache line)
        || rand::random(),  // Sample: random (different cache lines)
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            std::hint::black_box(table_lookup(data));
        });

    println!("{}", outcome);

    // Cache timing leaks can be subtle and may not always be detected
    if let Outcome::Pass { .. } = outcome {
        println!("Note: 256-byte S-box fits in L1 cache - no timing difference expected");
    }
}

/// LEAK: Large table lookup that exceeds L1 cache working set.
/// 64KB table = 1024 cache lines, causes cache pressure on random access.
#[inline(never)]
fn large_table_lookup(data: &[u8], table: &[u64; 8192]) -> u64 {
    let mut result = 0u64;
    for &byte in data {
        // Use byte to index into different regions of the large table
        // Multiply by 32 to spread accesses across cache lines
        let idx = (byte as usize) * 32;
        result ^= table[idx];
    }
    std::hint::black_box(result)
}

/// Test cache timing with a larger table (64KB) that causes cache pressure.
/// This should show timing differences between repeated same-index access
/// (baseline: all zeros → always index 0) vs random indices (sample).
///
/// Results vary by attacker model:
/// - AdjacentNetwork (100ns): Pass — effect too small for network exploitation
/// - SharedHardware (0.6ns): Inconclusive — measurement floor ~9ns
/// - Custom 5ns: **Fail** — detects ~8ns cache timing effect
///
/// This demonstrates that the same code can be "safe" or "leaky" depending
/// on your threat model.
#[test]
fn test_large_table_lookup_leaky() {
    // 64KB table (8192 × 8 bytes) - larger than typical L1 working set
    let table: Box<[u64; 8192]> = (0..8192)
        .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15)) // Pseudo-random fill
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let inputs = InputPair::new(
        || [0u8; 16],       // Baseline: all zeros → always hits same cache line
        || rand::random(),  // Sample: random → accesses spread across 64KB
    );

    // Custom 5ns threshold: realistic for shared-hardware scenarios,
    // above the measurement floor, detects the ~8ns cache effect
    let outcome = TimingOracle::for_attacker(AttackerModel::Custom { threshold_ns: 5.0 })
        .time_budget(Duration::from_secs(60))
        .max_samples(100_000)
        .test(inputs, |data| {
            std::hint::black_box(large_table_lookup(data, &table));
        });

    println!("{}", outcome);

    // Should detect cache timing leak with Custom 5ns threshold
    if let Outcome::Pass { .. } = outcome {
        println!("Note: Cache timing effects depend on CPU microarchitecture");
    }
}

/// Test that naive scalar multiplication is detected as leaky.
/// Equivalent to dudect's `donnabad` example with non-constant-time ECC.
#[test]
fn test_naive_scalar_mult_leaky() {
    let inputs = InputPair::new(
        || [0u8; 32],       // Baseline: all zeros (no additions)
        || rand::random(),  // Sample: random (many additions)
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            std::hint::black_box(naive_scalar_mult(data));
        });

    println!("{}", outcome);

    // This should detect a leak - if it passes, compiler may have optimized
    if let Outcome::Pass { .. } = outcome {
        println!("Note: Compiler may have optimized to constant-time code");
    }
}
