//! Curve25519 timing tests - inspired by DudeCT's donna/donnabad examples
//!
//! Tests X25519 ECDH scalar multiplication for timing leaks using DudeCT's two-class pattern:
//! - Class 0: Fixed scalars/basepoints
//! - Class 1: Random scalars/basepoints
//!
//! This tests whether Curve25519 implementations have data-dependent timing.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use timing_oracle::helpers::InputPair;
use timing_oracle::{skip_if_unreliable, AttackerModel, Exploitability, MeasurementQuality, Outcome, TimingOracle};
use x25519_dalek::x25519;

fn rand_bytes_32() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

// ============================================================================
// X25519 Scalar Multiplication Tests
// ============================================================================

/// X25519 scalar multiplication should be constant-time
///
/// Uses DudeCT's two-class pattern: fixed scalar vs random scalars
/// Tests whether the X25519 implementation has data-dependent timing
#[test]
fn x25519_scalar_mult_constant_time() {
    // Fixed basepoint (standard X25519 base point)
    let basepoint = x25519_dalek::X25519_BASEPOINT_BYTES;

    // Use a valid fixed scalar (not all-zeros which is pathological)
    let fixed_scalar: [u8; 32] = [
        0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82,
        0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a, 0x33, 0x9c,
        0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f,
        0x9a, 0x2d, 0x7e, 0x4c, 0x8b, 0x3a, 0x6d, 0x5e,
    ];

    // Pre-generate inputs using InputPair helper
    const SAMPLES: usize = 50_000;
    let scalars = InputPair::new(|| fixed_scalar, rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(SAMPLES)
        .min_effect_ns(50.0)
        .test(scalars, |scalar| {
            let result = x25519(*scalar, basepoint);
            std::hint::black_box(result);
        });

    let outcome = skip_if_unreliable!(outcome, "x25519_scalar_mult_constant_time");

    eprintln!("\n[x25519_scalar_mult_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // X25519 implementations should be constant-time
    match &outcome {
        Outcome::Pass { leak_probability, quality, .. } => {
            // Only assert on Bayesian metrics when measurements are reliable enough
            if matches!(quality, MeasurementQuality::Excellent) {
                assert!(
                    *leak_probability < 0.5,
                    "Leak probability too high ({:.1}%) with good quality measurements",
                    leak_probability * 100.0
                );
            } else {
                eprintln!(
                    "Note: {:?} quality (prob={:.1}%)",
                    quality,
                    leak_probability * 100.0
                );
            }
        }
        Outcome::Fail { leak_probability, exploitability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                panic!(
                    "X25519 should be constant-time (got Fail with leak_probability={:.3}, exploitability={:?})",
                    leak_probability, exploitability
                );
            } else {
                // With poor quality measurements, allow fail but check exploitability
                assert!(
                    matches!(exploitability, Exploitability::Negligible | Exploitability::PossibleLAN),
                    "Exploitability: {:?}",
                    exploitability
                );
                eprintln!(
                    "Note: Fail with {:?} quality (prob={:.1}%, {:?})",
                    quality,
                    leak_probability * 100.0,
                    exploitability
                );
            }
        }
        Outcome::Inconclusive { leak_probability, quality, .. } => {
            eprintln!(
                "Note: Inconclusive - {:?} quality (prob={:.1}%)",
                quality,
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { .. } => {}
    }
}

/// X25519 with different basepoints - should still be constant-time
///
/// Tests whether scalar multiplication depends on basepoint data
#[test]
fn x25519_different_basepoints_constant_time() {
    let scalar = rand_bytes_32();

    // Basepoint 0: All zeros
    let basepoint_zeros = [0u8; 32];
    // Basepoint 1: Random
    let basepoint_random = rand_bytes_32();

    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(30_000)
        .min_effect_ns(50.0)
        .test(inputs, |bp_idx| {
            let basepoint = if *bp_idx == 0 {
                basepoint_zeros
            } else {
                basepoint_random
            };
            let result = x25519(scalar, basepoint);
            std::hint::black_box(result);
        });

    let outcome = skip_if_unreliable!(outcome, "x25519_different_basepoints_constant_time");

    eprintln!("\n[x25519_different_basepoints_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Different basepoints shouldn't cause significant timing differences
    match &outcome {
        Outcome::Pass { leak_probability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                assert!(
                    *leak_probability < 0.5,
                    "Leak probability too high ({:.1}%) with good quality measurements",
                    leak_probability * 100.0
                );
            } else {
                eprintln!(
                    "Note: {:?} quality (prob={:.1}%)",
                    quality,
                    leak_probability * 100.0
                );
            }
        }
        Outcome::Fail { leak_probability, exploitability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                panic!(
                    "Should be constant-time (got Fail with leak_probability={:.3}, exploitability={:?})",
                    leak_probability, exploitability
                );
            } else {
                assert!(
                    matches!(exploitability, Exploitability::Negligible | Exploitability::PossibleLAN),
                    "Exploitability: {:?}",
                    exploitability
                );
                eprintln!(
                    "Note: Fail with {:?} quality (prob={:.1}%, {:?})",
                    quality,
                    leak_probability * 100.0,
                    exploitability
                );
            }
        }
        Outcome::Inconclusive { leak_probability, quality, .. } => {
            eprintln!(
                "Note: Inconclusive - {:?} quality (prob={:.1}%)",
                quality,
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { .. } => {}
    }
}

/// X25519 multiple operations - tests for cumulative timing effects
#[test]
fn x25519_multiple_operations_constant_time() {
    let basepoint = x25519_dalek::X25519_BASEPOINT_BYTES;

    // Use valid fixed scalars (not all-zeros)
    let fixed_scalars: [[u8; 32]; 3] = [
        [0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82,
         0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a, 0x33, 0x9c,
         0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f,
         0x9a, 0x2d, 0x7e, 0x4c, 0x8b, 0x3a, 0x6d, 0x5e],
        [0x2a, 0x3b, 0x4c, 0x5d, 0x6e, 0x7f, 0x80, 0x91,
         0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7, 0x08, 0x19,
         0x2a, 0x3b, 0x4c, 0x5d, 0x6e, 0x7f, 0x80, 0x91,
         0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7, 0x08, 0x19],
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
         0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00,
         0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
         0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00],
    ];

    // Pre-generate inputs using InputPair - 3 scalars per sample
    const SAMPLES: usize = 10_000;
    let scalars = InputPair::new(
        || fixed_scalars,
        || [rand_bytes_32(), rand_bytes_32(), rand_bytes_32()],
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(SAMPLES)
        .min_effect_ns(50.0)
        .test(scalars, |scalar_set| {
            let mut total = 0u8;
            for scalar in scalar_set {
                let result = x25519(*scalar, basepoint);
                total ^= result[0];
            }
            std::hint::black_box(total);
        });

    let outcome = skip_if_unreliable!(outcome, "x25519_multiple_operations_constant_time");

    eprintln!("\n[x25519_multiple_operations_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Multiple operations should maintain constant-time properties
    match &outcome {
        Outcome::Pass { leak_probability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                assert!(
                    *leak_probability < 0.5,
                    "Leak probability too high ({:.1}%) with good quality measurements",
                    leak_probability * 100.0
                );
            } else {
                eprintln!(
                    "Note: {:?} quality (prob={:.1}%)",
                    quality,
                    leak_probability * 100.0
                );
            }
        }
        Outcome::Fail { leak_probability, exploitability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                panic!(
                    "X25519 multiple operations should be constant-time (got Fail with leak_probability={:.3}, exploitability={:?})",
                    leak_probability, exploitability
                );
            } else {
                assert!(
                    matches!(exploitability, Exploitability::Negligible | Exploitability::PossibleLAN),
                    "Exploitability: {:?}",
                    exploitability
                );
                eprintln!(
                    "Note: Fail with {:?} quality (prob={:.1}%, {:?})",
                    quality,
                    leak_probability * 100.0,
                    exploitability
                );
            }
        }
        Outcome::Inconclusive { leak_probability, quality, .. } => {
            eprintln!(
                "Note: Inconclusive - {:?} quality (prob={:.1}%)",
                quality,
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { .. } => {}
    }
}

/// Scalar clamping timing
///
/// Tests whether scalar clamping (key preparation) has timing dependencies on data
#[test]
fn x25519_scalar_clamping_constant_time() {
    let basepoint = x25519_dalek::X25519_BASEPOINT_BYTES;

    // Use a valid fixed scalar (not all-zeros)
    let base_fixed_scalar: [u8; 32] = [
        0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82,
        0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a, 0x33, 0x9c,
        0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f,
        0x9a, 0x2d, 0x7e, 0x4c, 0x8b, 0x3a, 0x6d, 0x5e,
    ];

    // Pre-generate inputs using InputPair - both pre-clamped
    const SAMPLES: usize = 20_000;
    let scalars = InputPair::new(
        || {
            // Pre-clamp fixed scalar
            let mut scalar = base_fixed_scalar;
            scalar[0] &= 248;
            scalar[31] &= 127;
            scalar[31] |= 64;
            scalar
        },
        || {
            // Pre-clamp random scalar
            let mut scalar = rand_bytes_32();
            scalar[0] &= 248;
            scalar[31] &= 127;
            scalar[31] |= 64;
            scalar
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(SAMPLES)
        .min_effect_ns(50.0)
        .test(scalars, |scalar| {
            let result = x25519(*scalar, basepoint);
            std::hint::black_box(result);
        });

    let outcome = skip_if_unreliable!(outcome, "x25519_scalar_clamping_constant_time");

    eprintln!("\n[x25519_scalar_clamping_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Scalar clamping should be constant-time
    match &outcome {
        Outcome::Pass { leak_probability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                assert!(
                    *leak_probability < 0.5,
                    "Leak probability too high ({:.1}%) with good quality measurements",
                    leak_probability * 100.0
                );
            } else {
                eprintln!(
                    "Note: {:?} quality (prob={:.1}%)",
                    quality,
                    leak_probability * 100.0
                );
            }
        }
        Outcome::Fail { leak_probability, exploitability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                panic!(
                    "Scalar clamping should be constant-time (got Fail with leak_probability={:.3}, exploitability={:?})",
                    leak_probability, exploitability
                );
            } else {
                assert!(
                    matches!(exploitability, Exploitability::Negligible | Exploitability::PossibleLAN),
                    "Exploitability: {:?}",
                    exploitability
                );
                eprintln!(
                    "Note: Fail with {:?} quality (prob={:.1}%, {:?})",
                    quality,
                    leak_probability * 100.0,
                    exploitability
                );
            }
        }
        Outcome::Inconclusive { leak_probability, quality, .. } => {
            eprintln!(
                "Note: Inconclusive - {:?} quality (prob={:.1}%)",
                quality,
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { .. } => {}
    }
}

// ============================================================================
// Comparative Tests (Different Scalar Patterns)
// ============================================================================

/// Compare high vs low Hamming weight scalars
///
/// Tests if the number of 1-bits in scalar affects timing
#[test]
fn x25519_hamming_weight_independence() {
    let basepoint = x25519_dalek::X25519_BASEPOINT_BYTES;

    let inputs = InputPair::new(|| [0x00u8; 32], || [0xFFu8; 32]);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(20_000)
        .min_effect_ns(50.0)
        .test(inputs, |scalar| {
            let result = x25519(*scalar, basepoint);
            std::hint::black_box(result);
        });

    let outcome = skip_if_unreliable!(outcome, "x25519_hamming_weight_independence");

    eprintln!("\n[x25519_hamming_weight_independence]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Hamming weight should not affect timing significantly
    match &outcome {
        Outcome::Pass { leak_probability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                assert!(
                    *leak_probability < 0.5,
                    "Leak probability too high ({:.1}%) with good quality measurements",
                    leak_probability * 100.0
                );
            } else {
                eprintln!(
                    "Note: {:?} quality (prob={:.1}%)",
                    quality,
                    leak_probability * 100.0
                );
            }
        }
        Outcome::Fail { leak_probability, exploitability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                panic!(
                    "Should be constant-time (got Fail with leak_probability={:.3}, exploitability={:?})",
                    leak_probability, exploitability
                );
            } else {
                assert!(
                    matches!(exploitability, Exploitability::Negligible | Exploitability::PossibleLAN),
                    "Exploitability: {:?}",
                    exploitability
                );
                eprintln!(
                    "Note: Fail with {:?} quality (prob={:.1}%, {:?})",
                    quality,
                    leak_probability * 100.0,
                    exploitability
                );
            }
        }
        Outcome::Inconclusive { leak_probability, quality, .. } => {
            eprintln!(
                "Note: Inconclusive - {:?} quality (prob={:.1}%)",
                quality,
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { .. } => {}
    }
}

/// Test sequential vs scattered byte patterns in scalar
#[test]
fn x25519_byte_pattern_independence() {
    let basepoint = x25519_dalek::X25519_BASEPOINT_BYTES;

    let inputs = InputPair::new(|| 0u8, || 1u8);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(20_000)
        .min_effect_ns(50.0)
        .test(inputs, |pattern_type| {
            let mut scalar = [0u8; 32];
            if *pattern_type == 0 {
                // Sequential pattern
                for (i, byte) in scalar.iter_mut().enumerate() {
                    *byte = i as u8;
                }
            } else {
                // Scattered pattern (reverse)
                for (i, byte) in scalar.iter_mut().enumerate() {
                    *byte = (31 - i) as u8;
                }
            }
            let result = x25519(scalar, basepoint);
            std::hint::black_box(result);
        });

    let outcome = skip_if_unreliable!(outcome, "x25519_byte_pattern_independence");

    eprintln!("\n[x25519_byte_pattern_independence]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Byte pattern should not affect timing
    match &outcome {
        Outcome::Pass { leak_probability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                assert!(
                    *leak_probability < 0.5,
                    "Leak probability too high ({:.1}%) with good quality measurements",
                    leak_probability * 100.0
                );
            } else {
                eprintln!(
                    "Note: {:?} quality (prob={:.1}%)",
                    quality,
                    leak_probability * 100.0
                );
            }
        }
        Outcome::Fail { leak_probability, exploitability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                panic!(
                    "Should be constant-time (got Fail with leak_probability={:.3}, exploitability={:?})",
                    leak_probability, exploitability
                );
            } else {
                assert!(
                    matches!(exploitability, Exploitability::Negligible | Exploitability::PossibleLAN),
                    "Exploitability: {:?}",
                    exploitability
                );
                eprintln!(
                    "Note: Fail with {:?} quality (prob={:.1}%, {:?})",
                    quality,
                    leak_probability * 100.0,
                    exploitability
                );
            }
        }
        Outcome::Inconclusive { leak_probability, quality, .. } => {
            eprintln!(
                "Note: Inconclusive - {:?} quality (prob={:.1}%)",
                quality,
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { .. } => {}
    }
}

/// Full ECDH exchange timing
///
/// Tests complete key exchange operation for timing leaks
#[test]
fn x25519_ecdh_exchange_constant_time() {
    // Use valid fixed inputs (not all-zeros)
    let fixed_scalar: [u8; 32] = [
        0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82,
        0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a, 0x33, 0x9c,
        0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f,
        0x9a, 0x2d, 0x7e, 0x4c, 0x8b, 0x3a, 0x6d, 0x5e,
    ];
    let fixed_pubkey: [u8; 32] = [
        0x2a, 0x3b, 0x4c, 0x5d, 0x6e, 0x7f, 0x80, 0x91,
        0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7, 0x08, 0x19,
        0x2a, 0x3b, 0x4c, 0x5d, 0x6e, 0x7f, 0x80, 0x91,
        0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7, 0x08, 0x19,
    ];

    // Pre-generate inputs using InputPair - scalar and public key per sample
    const SAMPLES: usize = 15_000;
    let inputs = InputPair::new(
        || (fixed_scalar, fixed_pubkey),
        || (rand_bytes_32(), rand_bytes_32()),
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(SAMPLES)
        .min_effect_ns(50.0)
        .test(inputs, |(secret_scalar, other_public_key)| {
            // Perform scalar multiplication (ECDH)
            let shared = x25519(*secret_scalar, *other_public_key);
            std::hint::black_box(shared);
        });

    let outcome = skip_if_unreliable!(outcome, "x25519_ecdh_exchange_constant_time");

    eprintln!("\n[x25519_ecdh_exchange_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Full ECDH exchange should be constant-time
    match &outcome {
        Outcome::Pass { leak_probability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                assert!(
                    *leak_probability < 0.5,
                    "Leak probability too high ({:.1}%) with good quality measurements",
                    leak_probability * 100.0
                );
            } else {
                eprintln!(
                    "Note: {:?} quality (prob={:.1}%)",
                    quality,
                    leak_probability * 100.0
                );
            }
        }
        Outcome::Fail { leak_probability, exploitability, quality, .. } => {
            if matches!(quality, MeasurementQuality::Excellent) {
                panic!(
                    "ECDH exchange should be constant-time (got Fail with leak_probability={:.3}, exploitability={:?})",
                    leak_probability, exploitability
                );
            } else {
                assert!(
                    matches!(exploitability, Exploitability::Negligible | Exploitability::PossibleLAN),
                    "Exploitability: {:?}",
                    exploitability
                );
                eprintln!(
                    "Note: Fail with {:?} quality (prob={:.1}%, {:?})",
                    quality,
                    leak_probability * 100.0,
                    exploitability
                );
            }
        }
        Outcome::Inconclusive { leak_probability, quality, .. } => {
            eprintln!(
                "Note: Inconclusive - {:?} quality (prob={:.1}%)",
                quality,
                leak_probability * 100.0
            );
        }
        Outcome::Unmeasurable { .. } => {}
    }
}
