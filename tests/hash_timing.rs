//! Hash function timing tests
//!
//! Tests cryptographic hash functions for timing side channels using DudeCT's two-class pattern:
//! - Class 0: Fixed input (e.g., all zeros)
//! - Class 1: Random input
//!
//! This tests whether hash implementations have data-dependent timing.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use blake2::{Blake2b512, Blake2s256, Digest as Blake2Digest};
use sha3::{Sha3_256, Sha3_384, Sha3_512};
use timing_oracle::helpers::InputPair;
use timing_oracle::{skip_if_unreliable, Exploitability, TimingOracle};

fn rand_bytes_32() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

fn rand_bytes_64() -> [u8; 64] {
    let mut arr = [0u8; 64];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

fn rand_bytes_128() -> [u8; 128] {
    let mut arr = [0u8; 128];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

// ============================================================================
// SHA-3 Family Tests
// ============================================================================

/// SHA3-256 should be constant-time
///
/// Tests whether the Keccak sponge construction has data-dependent timing
#[test]
fn sha3_256_constant_time() {
    const SAMPLES: usize = 50_000;

    // Use non-pathological fixed input
    let fixed_input: [u8; 32] = [
        0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07,
        0x34, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
        0xee, 0xff,
    ];

    let inputs = InputPair::new(|| fixed_input, rand_bytes_32);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .min_effect_ns(50.0)
        .test(inputs, |data| {
            let hash = Sha3_256::digest(data);
            std::hint::black_box(hash[0]);
        });

    let result = skip_if_unreliable!(outcome, "sha3_256_constant_time");

    eprintln!("\n[sha3_256_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // SHA-3 should be constant-time
    assert!(
        result.ci_gate.passed(),
        "SHA3-256 should be constant-time"
    );

    // Informational: warn if probability is high, but CI gate is authoritative
    if result.leak_probability > 0.5 {
        eprintln!(
            "Note: High leak_probability ({:.1}%) despite CI gate pass - expected with noisy measurements",
            result.leak_probability * 100.0
        );
    }

    assert!(matches!(result.exploitability,
        Exploitability::Negligible | Exploitability::PossibleLAN),
        "Exploitability should be low, got: {:?}", result.exploitability);
}

/// SHA3-384 should be constant-time
#[test]
fn sha3_384_constant_time() {
    const SAMPLES: usize = 50_000;

    let fixed_input: [u8; 64] = [0x5a; 64]; // Non-pathological pattern
    let inputs = InputPair::new(|| fixed_input, rand_bytes_64);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .min_effect_ns(50.0)
        .test(inputs, |data| {
            let hash = Sha3_384::digest(data);
            std::hint::black_box(hash[0]);
        });

    let result = skip_if_unreliable!(outcome, "sha3_384_constant_time");

    eprintln!("\n[sha3_384_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed(),
        "SHA3-384 should be constant-time"
    );

    // Informational: warn if probability is high, but CI gate is authoritative
    if result.leak_probability > 0.5 {
        eprintln!(
            "Note: High leak_probability ({:.1}%) despite CI gate pass - expected with noisy measurements",
            result.leak_probability * 100.0
        );
    }

    assert!(matches!(result.exploitability,
        Exploitability::Negligible | Exploitability::PossibleLAN),
        "Exploitability should be low, got: {:?}", result.exploitability);
}

/// SHA3-512 should be constant-time
#[test]
fn sha3_512_constant_time() {
    const SAMPLES: usize = 50_000;

    let fixed_input: [u8; 64] = [0xa5; 64]; // Non-pathological pattern
    let inputs = InputPair::new(|| fixed_input, rand_bytes_64);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .min_effect_ns(50.0)
        .test(inputs, |data| {
            let hash = Sha3_512::digest(data);
            std::hint::black_box(hash[0]);
        });

    let result = skip_if_unreliable!(outcome, "sha3_512_constant_time");

    eprintln!("\n[sha3_512_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed(),
        "SHA3-512 should be constant-time"
    );

    // Informational: warn if probability is high, but CI gate is authoritative
    if result.leak_probability > 0.5 {
        eprintln!(
            "Note: High leak_probability ({:.1}%) despite CI gate pass - expected with noisy measurements",
            result.leak_probability * 100.0
        );
    }

    assert!(matches!(result.exploitability,
        Exploitability::Negligible | Exploitability::PossibleLAN),
        "Exploitability should be low, got: {:?}", result.exploitability);
}

/// SHA3-256 with same-length but different data
///
/// Tests constant-time property for fixed-length inputs
/// Note: SHA3 is NOT constant-time for varying lengths by design
#[test]
fn sha3_256_data_independence() {
    const SAMPLES: usize = 30_000;

    // Test with 128-byte inputs (same length, different data)
    let fixed_input: [u8; 128] = [0x42; 128];
    let inputs = InputPair::new(|| fixed_input, rand_bytes_128);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .min_effect_ns(50.0)
        .test(inputs, |data| {
            let hash = Sha3_256::digest(data);
            std::hint::black_box(hash[0]);
        });

    let result = skip_if_unreliable!(outcome, "sha3_256_data_independence");

    eprintln!("\n[sha3_256_data_independence]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed(),
        "SHA3-256 should be constant-time for same-length inputs"
    );

    // Informational: warn if probability is high, but CI gate is authoritative
    if result.leak_probability > 0.5 {
        eprintln!(
            "Note: High leak_probability ({:.1}%) despite CI gate pass - expected with noisy measurements",
            result.leak_probability * 100.0
        );
    }

    assert!(matches!(result.exploitability,
        Exploitability::Negligible | Exploitability::PossibleLAN),
        "Exploitability should be low, got: {:?}", result.exploitability);
}

// ============================================================================
// BLAKE2 Family Tests
// ============================================================================

/// BLAKE2b-512 should be constant-time
///
/// BLAKE2b is a high-speed hash function, widely used in crypto applications
#[test]
fn blake2b_512_constant_time() {
    const SAMPLES: usize = 50_000;

    let fixed_input: [u8; 64] = [0x73; 64]; // Non-pathological pattern
    let inputs = InputPair::new(|| fixed_input, rand_bytes_64);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .min_effect_ns(50.0)
        .test(inputs, |data| {
            let hash = Blake2b512::digest(data);
            std::hint::black_box(hash[0]);
        });

    let result = skip_if_unreliable!(outcome, "blake2b_512_constant_time");

    eprintln!("\n[blake2b_512_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed(),
        "BLAKE2b-512 should be constant-time"
    );

    // Informational: warn if probability is high, but CI gate is authoritative
    if result.leak_probability > 0.5 {
        eprintln!(
            "Note: High leak_probability ({:.1}%) despite CI gate pass - expected with noisy measurements",
            result.leak_probability * 100.0
        );
    }

    assert!(matches!(result.exploitability,
        Exploitability::Negligible | Exploitability::PossibleLAN),
        "Exploitability should be low, got: {:?}", result.exploitability);
}

/// BLAKE2s-256 should be constant-time
///
/// BLAKE2s is optimized for 8-32 bit platforms
#[test]
fn blake2s_256_constant_time() {
    const SAMPLES: usize = 50_000;

    let fixed_input: [u8; 32] = [0xb7; 32]; // Non-pathological pattern
    let inputs = InputPair::new(|| fixed_input, rand_bytes_32);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .min_effect_ns(50.0)
        .test(inputs, |data| {
            let hash = Blake2s256::digest(data);
            std::hint::black_box(hash[0]);
        });

    let result = skip_if_unreliable!(outcome, "blake2s_256_constant_time");

    eprintln!("\n[blake2s_256_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed(),
        "BLAKE2s-256 should be constant-time"
    );

    // Informational: warn if probability is high, but CI gate is authoritative
    if result.leak_probability > 0.5 {
        eprintln!(
            "Note: High leak_probability ({:.1}%) despite CI gate pass - expected with noisy measurements",
            result.leak_probability * 100.0
        );
    }

    assert!(matches!(result.exploitability,
        Exploitability::Negligible | Exploitability::PossibleLAN),
        "Exploitability should be low, got: {:?}", result.exploitability);
}

// ============================================================================
// Comparative Tests (Different Input Patterns)
// ============================================================================

/// Compare high vs low Hamming weight inputs for SHA3
///
/// Tests if the number of 1-bits in input affects timing
#[test]
fn sha3_256_hamming_weight_independence() {
    const SAMPLES: usize = 30_000;

    let inputs = InputPair::new(|| [0x00u8; 32], || [0xFFu8; 32]);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .min_effect_ns(50.0)
        .test(inputs, |data| {
            let hash = Sha3_256::digest(data);
            std::hint::black_box(hash[0]);
        });

    let result = skip_if_unreliable!(outcome, "sha3_256_hamming_weight_independence");

    eprintln!("\n[sha3_256_hamming_weight_independence]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // Primary check: CI gate should pass
    assert!(
        result.ci_gate.passed(),
        "Hamming weight should not affect timing"
    );

    // Informational: warn if probability is high, but CI gate is authoritative
    if result.leak_probability > 0.5 {
        eprintln!(
            "Note: High leak_probability ({:.1}%) despite CI gate pass - expected with noisy measurements",
            result.leak_probability * 100.0
        );
    }

    // Secondary check: exploitability should be low
    assert!(
        matches!(
            result.exploitability,
            Exploitability::Negligible | Exploitability::PossibleLAN
        ),
        "Exploitability: {:?}", result.exploitability
    );
}

/// Compare high vs low Hamming weight inputs for BLAKE2b
#[test]
fn blake2b_hamming_weight_independence() {
    const SAMPLES: usize = 30_000;

    let inputs = InputPair::new(|| [0x00u8; 64], || [0xFFu8; 64]);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .min_effect_ns(50.0)
        .test(inputs, |data| {
            let hash = Blake2b512::digest(data);
            std::hint::black_box(hash[0]);
        });

    let result = skip_if_unreliable!(outcome, "blake2b_hamming_weight_independence");

    eprintln!("\n[blake2b_hamming_weight_independence]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // Primary check: CI gate should pass
    assert!(
        result.ci_gate.passed(),
        "Hamming weight should not affect timing"
    );

    // Informational: warn if probability is high, but CI gate is authoritative
    if result.leak_probability > 0.5 {
        eprintln!(
            "Note: High leak_probability ({:.1}%) despite CI gate pass - expected with noisy measurements",
            result.leak_probability * 100.0
        );
    }

    // Secondary check: exploitability should be low
    assert!(
        matches!(
            result.exploitability,
            Exploitability::Negligible | Exploitability::PossibleLAN
        ),
        "Exploitability: {:?}", result.exploitability
    );
}

// ============================================================================
// Incremental Hashing Tests
// ============================================================================

/// Test incremental/streaming hash updates
///
/// Many applications process data in chunks - this tests whether
/// update boundaries affect timing
#[test]
fn sha3_256_incremental_constant_time() {
    use sha3::Digest;
    const SAMPLES: usize = 30_000;

    let fixed_chunks: ([u8; 32], [u8; 32]) = ([0x11; 32], [0x22; 32]);

    let inputs = InputPair::new(|| fixed_chunks, || (rand_bytes_32(), rand_bytes_32()));

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .min_effect_ns(50.0)
        .test(inputs, |(data1, data2)| {
            let mut hasher = Sha3_256::new();
            hasher.update(data1);
            hasher.update(data2);
            let hash = hasher.finalize();
            std::hint::black_box(hash[0]);
        });

    let result = skip_if_unreliable!(outcome, "sha3_256_incremental_constant_time");

    eprintln!("\n[sha3_256_incremental_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed(),
        "SHA3-256 incremental hashing should be constant-time"
    );

    // Informational: warn if probability is high, but CI gate is authoritative
    if result.leak_probability > 0.5 {
        eprintln!(
            "Note: High leak_probability ({:.1}%) despite CI gate pass - expected with noisy measurements",
            result.leak_probability * 100.0
        );
    }

    assert!(matches!(result.exploitability,
        Exploitability::Negligible | Exploitability::PossibleLAN),
        "Exploitability should be low, got: {:?}", result.exploitability);
}

/// Test BLAKE2b incremental hashing
#[test]
fn blake2b_incremental_constant_time() {
    use blake2::Digest;
    const SAMPLES: usize = 30_000;

    let fixed_chunks: ([u8; 64], [u8; 64]) = ([0x33; 64], [0x44; 64]);

    let inputs = InputPair::new(|| fixed_chunks, || (rand_bytes_64(), rand_bytes_64()));

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .min_effect_ns(50.0)
        .test(inputs, |(data1, data2)| {
            let mut hasher = Blake2b512::new();
            hasher.update(data1);
            hasher.update(data2);
            let hash = hasher.finalize();
            std::hint::black_box(hash[0]);
        });

    let result = skip_if_unreliable!(outcome, "blake2b_incremental_constant_time");

    eprintln!("\n[blake2b_incremental_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed(),
        "BLAKE2b incremental hashing should be constant-time"
    );

    // Informational: warn if probability is high, but CI gate is authoritative
    if result.leak_probability > 0.5 {
        eprintln!(
            "Note: High leak_probability ({:.1}%) despite CI gate pass - expected with noisy measurements",
            result.leak_probability * 100.0
        );
    }

    assert!(matches!(result.exploitability,
        Exploitability::Negligible | Exploitability::PossibleLAN),
        "Exploitability should be low, got: {:?}", result.exploitability);
}
