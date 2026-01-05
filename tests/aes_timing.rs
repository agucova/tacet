//! AES timing tests - inspired by DudeCT's aes32 example
//!
//! Tests full AES-128 encryption for timing leaks using DudeCT's two-class pattern:
//! - Class 0: Fixed plaintexts
//! - Class 1: Random plaintexts
//!
//! This tests whether AES implementations have data-dependent timing.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use timing_oracle::helpers::InputPair;
use timing_oracle::TimingOracle;

fn rand_bytes_16() -> [u8; 16] {
    let mut arr = [0u8; 16];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

// ============================================================================
// AES-128 Block Cipher Tests
// ============================================================================

/// AES-128 encryption should be constant-time
///
/// Uses DudeCT's two-class pattern: fixed plaintexts vs random plaintexts
/// Tests whether the AES implementation has data-dependent timing
#[test]
fn aes128_block_encrypt_constant_time() {
    // Fixed key for all encryptions
    let key = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f,
        0x3c,
    ];

    let cipher = Aes128::new(&key.into());

    // Use a non-pathological fixed plaintext
    let fixed_plaintext: [u8; 16] = [
        0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
        0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34,
    ];

    // Pre-generate inputs using InputPair helper
    const SAMPLES: usize = 100_000;
    let inputs = InputPair::new(|| fixed_plaintext, rand_bytes_16);

    let outcome = TimingOracle::new()
        .samples(SAMPLES)
        .alpha(0.01)
        .test(inputs, |plaintext| {
            let mut block = plaintext.to_owned().into();
            cipher.encrypt_block(&mut block);
            std::hint::black_box(block[0]);
        });

    eprintln!("\n[aes128_block_encrypt_constant_time]");
    if let timing_oracle::Outcome::Completed(ref r) = outcome {
        eprintln!("{}", timing_oracle::output::format_result(r));
    }

    // Modern AES implementations should be constant-time
    let result = timing_oracle::skip_if_unreliable!(outcome, "aes128_block_encrypt_constant_time");

    assert!(
        result.ci_gate.passed,
        "AES-128 should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
    assert!(
        matches!(
            result.exploitability,
            timing_oracle::Exploitability::Negligible | timing_oracle::Exploitability::PossibleLAN
        ),
        "AES-128 should have negligible exploitability (got {:?})",
        result.exploitability
    );
    assert!(
        !matches!(result.quality, timing_oracle::MeasurementQuality::TooNoisy),
        "Measurement quality should not be TooNoisy (got {:?})",
        result.quality
    );
}

/// AES-128 with different keys - should still be constant-time
///
/// Tests whether key schedule or encryption depends on key material
#[test]
fn aes128_different_keys_constant_time() {
    // Use non-pathological fixed keys (not all-zeros)
    let key1: [u8; 16] = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
    ];
    let key2: [u8; 16] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    ];

    let cipher1 = Aes128::new(&key1.into());
    let cipher2 = Aes128::new(&key2.into());

    let plaintext = [0x01u8; 16]; // Fixed plaintext

    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::new()
        .samples(50_000)
        .test(inputs, |key_idx| {
            let mut block = plaintext.into();
            if *key_idx == 0 {
                cipher1.encrypt_block(&mut block);
            } else {
                cipher2.encrypt_block(&mut block);
            }
            std::hint::black_box(block[0]);
        });

    eprintln!("\n[aes128_different_keys_constant_time]");
    if let timing_oracle::Outcome::Completed(ref r) = outcome {
        eprintln!("{}", timing_oracle::output::format_result(r));
    }

    // Different keys shouldn't cause timing differences
    let result = timing_oracle::skip_if_unreliable!(outcome, "aes128_different_keys_constant_time");

    assert!(
        result.ci_gate.passed,
        "AES-128 with different keys should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
    assert!(
        matches!(
            result.exploitability,
            timing_oracle::Exploitability::Negligible | timing_oracle::Exploitability::PossibleLAN
        ),
        "AES with different keys should have low exploitability (got {:?})",
        result.exploitability
    );
    assert!(
        !matches!(result.quality, timing_oracle::MeasurementQuality::TooNoisy),
        "Measurement quality should not be TooNoisy (got {:?})",
        result.quality
    );
}

/// AES-128 multiple blocks - tests for cumulative timing effects
#[test]
fn aes128_multiple_blocks_constant_time() {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::Rng;

    // Use deterministic seed for reproducibility
    let mut rng = StdRng::seed_from_u64(0x1234_5678_9ABC_DEF0);

    // Generate deterministic key
    let mut key = [0u8; 16];
    for byte in &mut key {
        *byte = rng.random();
    }
    let cipher = Aes128::new(&key.into());

    // Use non-pathological fixed blocks (not all-zeros)
    let fixed_blocks: [[u8; 16]; 4] = [
        [0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
         0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34],
        [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
         0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff],
        [0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87,
         0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f],
        [0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
         0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a],
    ];

    // Pre-generate inputs using InputPair - 4 blocks per sample
    // Use seeded RNG for deterministic random blocks
    const SAMPLES: usize = 20_000;
    let mut rng_blocks = StdRng::seed_from_u64(0xFEDC_BA98_7654_3210);
    let mut random_blocks_vec: Vec<[[u8; 16]; 4]> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let mut blocks = [[0u8; 16]; 4];
        for block in &mut blocks {
            for byte in block {
                *byte = rng_blocks.random();
            }
        }
        random_blocks_vec.push(blocks);
    }

    let idx = std::cell::Cell::new(0);
    let inputs = InputPair::new(
        || fixed_blocks,
        move || {
            let i = idx.get();
            idx.set((i + 1) % SAMPLES);
            random_blocks_vec[i]
        },
    );

    // Use alpha=0.05 to reproduce original flakiness (default is 0.01)
    let outcome = TimingOracle::new()
        .samples(SAMPLES)
        .alpha(0.05)
        .test(inputs, |blocks| {
            let mut total = 0u8;
            for b in blocks {
                let mut block = (*b).into();
                cipher.encrypt_block(&mut block);
                total ^= block[0];
            }
            std::hint::black_box(total);
        });

    eprintln!("\n[aes128_multiple_blocks_constant_time]");
    if let timing_oracle::Outcome::Completed(ref r) = outcome {
        eprintln!("{}", timing_oracle::output::format_result(r));

        // Debug: Show CI gate details to understand boundary behavior
        eprintln!("\nCI Gate Debug:");
        eprintln!("  Passed: {}", r.ci_gate.passed);
        eprintln!("  Alpha: {}", r.ci_gate.alpha);
        eprintln!("  Max observed: {:.2}", r.ci_gate.max_observed);
        eprintln!("  Threshold: {:.2}", r.ci_gate.threshold);
        let margin = r.ci_gate.threshold - r.ci_gate.max_observed;
        eprintln!("  Margin: {:.2} {}", margin, if margin < 0.0 { "FAIL" } else { "pass" });
        eprintln!("  Per-quantile observed:");
        for i in 0..9 {
            eprintln!(
                "    Q{}: obs={:7.2}",
                i + 1,
                r.ci_gate.observed[i],
            );
        }
    }

    // Multiple block encryption should maintain constant-time properties
    let result = timing_oracle::skip_if_unreliable!(outcome, "aes128_multiple_blocks_constant_time");

    assert!(
        result.ci_gate.passed,
        "AES-128 multiple blocks should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
    assert!(
        matches!(
            result.exploitability,
            timing_oracle::Exploitability::Negligible | timing_oracle::Exploitability::PossibleLAN
        ),
        "AES-128 multiple blocks should have low exploitability (got {:?})",
        result.exploitability
    );
    assert!(
        !matches!(result.quality, timing_oracle::MeasurementQuality::TooNoisy),
        "Measurement quality should not be TooNoisy (got {:?})",
        result.quality
    );
}

/// AES key expansion timing
///
/// Tests whether key expansion (KeyInit) has timing dependencies on key material
#[test]
fn aes128_key_init_constant_time() {
    // Use non-pathological fixed key (not all-zeros)
    let fixed_key: [u8; 16] = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
    ];

    // Pre-generate keys using InputPair helper
    const SAMPLES: usize = 50_000;
    let keys = InputPair::new(|| fixed_key, rand_bytes_16);

    let outcome = TimingOracle::new()
        .samples(SAMPLES)
        .test(keys, |key| {
            let cipher = Aes128::new(key.into());
            std::hint::black_box(cipher);
        });

    eprintln!("\n[aes128_key_init_constant_time]");
    if let timing_oracle::Outcome::Completed(ref r) = outcome {
        eprintln!("{}", timing_oracle::output::format_result(r));
    }

    // Key initialization should be constant-time
    let result = timing_oracle::skip_if_unreliable!(outcome, "aes128_key_init_constant_time");

    assert!(
        result.ci_gate.passed,
        "AES-128 key init should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
    assert!(
        matches!(
            result.exploitability,
            timing_oracle::Exploitability::Negligible | timing_oracle::Exploitability::PossibleLAN
        ),
        "AES-128 key init should have low exploitability (got {:?})",
        result.exploitability
    );
    assert!(
        !matches!(result.quality, timing_oracle::MeasurementQuality::TooNoisy),
        "Measurement quality should not be TooNoisy (got {:?})",
        result.quality
    );
}

// ============================================================================
// Comparative Tests (Different Input Patterns)
// ============================================================================

/// Compare high vs low Hamming weight plaintexts
///
/// Tests if the number of 1-bits in plaintext affects timing
#[test]
fn aes128_hamming_weight_independence() {
    let key = rand_bytes_16();
    let cipher = Aes128::new(&key.into());

    let inputs = InputPair::new(|| [0x00u8; 16], || [0xFFu8; 16]);

    let outcome = TimingOracle::new()
        .samples(30_000)
        .test(inputs, |plaintext| {
            let mut block = plaintext.to_owned().into();
            cipher.encrypt_block(&mut block);
            std::hint::black_box(block[0]);
        });

    eprintln!("\n[aes128_hamming_weight_independence]");
    if let timing_oracle::Outcome::Completed(ref r) = outcome {
        eprintln!("{}", timing_oracle::output::format_result(r));
    }

    // Hamming weight should not affect timing
    let result = timing_oracle::skip_if_unreliable!(outcome, "aes128_hamming_weight_independence");

    assert!(
        result.ci_gate.passed,
        "AES-128 Hamming weight should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
    assert!(
        matches!(
            result.exploitability,
            timing_oracle::Exploitability::Negligible | timing_oracle::Exploitability::PossibleLAN
        ),
        "Hamming weight should not affect timing (got {:?})",
        result.exploitability
    );
    assert!(
        !matches!(result.quality, timing_oracle::MeasurementQuality::TooNoisy),
        "Measurement quality should not be TooNoisy (got {:?})",
        result.quality
    );
}

/// Test sequential vs scattered byte patterns
#[test]
fn aes128_byte_pattern_independence() {
    let key = rand_bytes_16();
    let cipher = Aes128::new(&key.into());

    let inputs = InputPair::new(|| 0u8, || 1u8);

    let outcome = TimingOracle::new()
        .samples(30_000)
        .test(inputs, |pattern_type| {
            let mut block = [0u8; 16];
            if *pattern_type == 0 {
                // Sequential pattern
                for i in 0..16 {
                    block[i] = i as u8;
                }
            } else {
                // Scattered pattern (reverse)
                for i in 0..16 {
                    block[i] = (15 - i) as u8;
                }
            }
            let mut block = block.into();
            cipher.encrypt_block(&mut block);
            std::hint::black_box(block[0]);
        });

    eprintln!("\n[aes128_byte_pattern_independence]");
    if let timing_oracle::Outcome::Completed(ref r) = outcome {
        eprintln!("{}", timing_oracle::output::format_result(r));
    }

    // Byte patterns should not affect timing
    let result = timing_oracle::skip_if_unreliable!(outcome, "aes128_byte_pattern_independence");

    assert!(
        result.ci_gate.passed,
        "AES-128 byte pattern should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
    assert!(
        matches!(
            result.exploitability,
            timing_oracle::Exploitability::Negligible | timing_oracle::Exploitability::PossibleLAN
        ),
        "Byte patterns should not affect timing (got {:?})",
        result.exploitability
    );
    assert!(
        !matches!(result.quality, timing_oracle::MeasurementQuality::TooNoisy),
        "Measurement quality should not be TooNoisy (got {:?})",
        result.quality
    );
}
