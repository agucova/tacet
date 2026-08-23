//! Orion library timing tests
//!
//! Tests orion (https://github.com/orion-rs/orion), a pure Rust cryptography library
//! that explicitly claims to be constant-time for all operations.
//!
//! **Critical validation**: This test suite verifies orion's constant-time claims.
//! Any Fail outcomes indicate potential timing side-channel vulnerabilities in a
//! library that explicitly guarantees constant-time operation.
//!
//! Uses DudeCT's two-class pattern:
//! - Class 0: All-zero data (vec![0u8; 32])
//! - Class 1: Random data
//!
//! This pattern tests for data-dependent timing rather than specific value comparisons.
//!
//! ## Orion API Notes
//!
//! Orion provides both high-level (`orion::*`) and low-level (`orion::hazardous::*`) APIs:
//! - High-level: BLAKE2b-based MAC (auth), XChaCha20-Poly1305 (aead), BLAKE2b hash (hash), Argon2i (pwhash)
//! - Low-level: HMAC-SHA512, Poly1305, and other primitives
//!
//! We test both to validate constant-time claims across all implementations.

use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{skip_if_unreliable, AttackerModel, Exploitability, Outcome, TimingOracle};

fn rand_bytes_32() -> Vec<u8> {
    let mut vec = vec![0u8; 32];
    for byte in &mut vec {
        *byte = rand::random();
    }
    vec
}

// ============================================================================
// BLAKE2b-MAC (auth) Tests
// ============================================================================

/// BLAKE2b-MAC (auth) should be constant-time
///
/// Tests whether orion::auth timing depends on message content
#[test]
fn orion_auth_constant_time() {
    use orion::auth;

    let secret_key = auth::SecretKey::default();

    let inputs = InputPair::new(|| vec![0u8; 32], rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(10))
        .max_samples(100_000)
        .test(inputs, |message| {
            let tag = auth::authenticate(&secret_key, message).unwrap();
            std::hint::black_box(tag.unprotected_as_bytes()[0]);
        });

    eprintln!("\n[orion_auth_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "orion_auth_constant_time");

    match &outcome {
        Outcome::Pass { .. } => {
            // Good - orion's constant-time claim holds
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            assert!(
                matches!(
                    exploitability,
                    Exploitability::SharedHardwareOnly | Exploitability::Http2Multiplexing
                ),
                "BLAKE2b-MAC should have negligible exploitability (got {:?})",
                exploitability
            );
            panic!(
                "CRITICAL: Orion auth (BLAKE2b-MAC) claims constant-time but shows leak (P={:.3})",
                leak_probability
            );
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            eprintln!(
                "Warning: Inconclusive result (leak_probability={:.3})",
                leak_probability
            );
        }
        Outcome::Unmeasurable { .. } => {}
        Outcome::Research(_) => {}
    }
}

/// BLAKE2b-MAC Hamming weight independence
///
/// Tests if the number of 1-bits in the message affects timing
#[test]
fn orion_auth_hamming_weight() {
    use orion::auth;

    let secret_key = auth::SecretKey::default();

    let inputs = InputPair::new(|| vec![0x00u8; 32], || vec![0xFFu8; 32]);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(10))
        .max_samples(100_000)
        .test(inputs, |message| {
            let tag = auth::authenticate(&secret_key, message).unwrap();
            std::hint::black_box(tag.unprotected_as_bytes()[0]);
        });

    eprintln!("\n[orion_auth_hamming_weight]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "orion_auth_hamming_weight");

    match &outcome {
        Outcome::Pass { .. } => {
            // Good - Hamming weight doesn't affect timing
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            assert!(
                matches!(
                    exploitability,
                    Exploitability::SharedHardwareOnly | Exploitability::Http2Multiplexing
                ),
                "Hamming weight should not affect timing (got {:?})",
                exploitability
            );
            panic!(
                "CRITICAL: Orion auth shows Hamming weight dependency (P={:.3})",
                leak_probability
            );
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            eprintln!(
                "Warning: Inconclusive result (leak_probability={:.3})",
                leak_probability
            );
        }
        Outcome::Unmeasurable { .. } => {}
        Outcome::Research(_) => {}
    }
}

// ============================================================================
// BLAKE2b Hash Tests
// ============================================================================

/// BLAKE2b hashing should be constant-time
///
/// Tests whether BLAKE2b timing depends on input data
#[test]
fn orion_hash_constant_time() {
    use orion::hash;

    let inputs = InputPair::new(|| vec![0u8; 32], rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(10))
        .max_samples(100_000)
        .test(inputs, |message| {
            let digest = hash::digest(message).unwrap();
            std::hint::black_box(digest.as_ref()[0]);
        });

    eprintln!("\n[orion_hash_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "orion_hash_constant_time");

    match &outcome {
        Outcome::Pass { .. } => {
            // Good - BLAKE2b is constant-time
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            assert!(
                matches!(
                    exploitability,
                    Exploitability::SharedHardwareOnly | Exploitability::Http2Multiplexing
                ),
                "BLAKE2b should have negligible exploitability (got {:?})",
                exploitability
            );
            panic!(
                "CRITICAL: Orion BLAKE2b hash claims constant-time but shows leak (P={:.3})",
                leak_probability
            );
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            eprintln!(
                "Warning: Inconclusive result (leak_probability={:.3})",
                leak_probability
            );
        }
        Outcome::Unmeasurable { .. } => {}
        Outcome::Research(_) => {}
    }
}

/// BLAKE2b Hamming weight independence
#[test]
fn orion_hash_hamming_weight() {
    use orion::hash;

    let inputs = InputPair::new(|| vec![0x00u8; 64], || vec![0xFFu8; 64]);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(10))
        .max_samples(100_000)
        .test(inputs, |message| {
            let digest = hash::digest(message).unwrap();
            std::hint::black_box(digest.as_ref()[0]);
        });

    eprintln!("\n[orion_hash_hamming_weight]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "orion_hash_hamming_weight");

    match &outcome {
        Outcome::Pass { .. } => {
            // Good - Hamming weight doesn't affect timing
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            assert!(
                matches!(
                    exploitability,
                    Exploitability::SharedHardwareOnly | Exploitability::Http2Multiplexing
                ),
                "Hamming weight should not affect timing (got {:?})",
                exploitability
            );
            panic!(
                "CRITICAL: Orion BLAKE2b shows Hamming weight dependency (P={:.3})",
                leak_probability
            );
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            eprintln!(
                "Warning: Inconclusive result (leak_probability={:.3})",
                leak_probability
            );
        }
        Outcome::Unmeasurable { .. } => {}
        Outcome::Research(_) => {}
    }
}

// ============================================================================
// XChaCha20-Poly1305 AEAD Tests
// ============================================================================

/// XChaCha20-Poly1305 encryption should be constant-time
///
/// Tests whether AEAD encryption timing depends on plaintext content
#[test]
fn orion_aead_encrypt_constant_time() {
    use orion::aead;

    let secret_key = aead::SecretKey::default();

    let inputs = InputPair::new(|| vec![0u8; 32], rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(10))
        .max_samples(100_000)
        .test(inputs, |plaintext| {
            let ciphertext = aead::seal(&secret_key, plaintext).unwrap();
            std::hint::black_box(ciphertext[0]);
        });

    eprintln!("\n[orion_aead_encrypt_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "orion_aead_encrypt_constant_time");

    match &outcome {
        Outcome::Pass { .. } => {
            // Good - encryption is constant-time
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            assert!(
                matches!(
                    exploitability,
                    Exploitability::SharedHardwareOnly | Exploitability::Http2Multiplexing
                ),
                "XChaCha20-Poly1305 should have negligible exploitability (got {:?})",
                exploitability
            );
            panic!(
                "CRITICAL: Orion XChaCha20-Poly1305 encryption shows leak (P={:.3})",
                leak_probability
            );
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            eprintln!(
                "Warning: Inconclusive result (leak_probability={:.3})",
                leak_probability
            );
        }
        Outcome::Unmeasurable { .. } => {}
        Outcome::Research(_) => {}
    }
}

/// XChaCha20-Poly1305 decryption should be constant-time
///
/// Tests whether AEAD decryption timing depends on ciphertext content
#[test]
fn orion_aead_decrypt_constant_time() {
    use orion::aead;

    let secret_key = aead::SecretKey::default();

    // Pre-encrypt two different plaintexts
    let ciphertext1 = aead::seal(&secret_key, &[0u8; 32]).unwrap();
    let ciphertext2 = aead::seal(&secret_key, &rand_bytes_32()).unwrap();

    let inputs = InputPair::new(move || ciphertext1.clone(), move || ciphertext2.clone());

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(10))
        .max_samples(100_000)
        .test(inputs, |ciphertext| {
            let plaintext = aead::open(&secret_key, ciphertext).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    eprintln!("\n[orion_aead_decrypt_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "orion_aead_decrypt_constant_time");

    match &outcome {
        Outcome::Pass { .. } => {
            // Good - decryption is constant-time
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            assert!(
                matches!(
                    exploitability,
                    Exploitability::SharedHardwareOnly | Exploitability::Http2Multiplexing
                ),
                "XChaCha20-Poly1305 should have negligible exploitability (got {:?})",
                exploitability
            );
            panic!(
                "CRITICAL: Orion XChaCha20-Poly1305 decryption shows leak (P={:.3})",
                leak_probability
            );
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            eprintln!(
                "Warning: Inconclusive result (leak_probability={:.3})",
                leak_probability
            );
        }
        Outcome::Unmeasurable { .. } => {}
        Outcome::Research(_) => {}
    }
}

// ============================================================================
// Argon2i Password Hashing Tests
// ============================================================================

/// Argon2i password hashing should be constant-time
///
/// CRITICAL TEST: Password hashing is a high-value timing attack target.
/// Uses stricter SharedHardware threshold (~2 cycles @ 5 GHz).
#[test]
#[ignore = "password hashing is too slow for routine test runs"]
fn orion_pwhash_constant_time() {
    use orion::pwhash;

    let inputs = InputPair::new(|| vec![0u8; 32], rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::SharedHardware)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .max_samples(50_000)
        .test(inputs, |password| {
            let pw = pwhash::Password::from_slice(password).unwrap();
            let hash = pwhash::hash_password(&pw, 3, 1 << 16).unwrap();
            std::hint::black_box(hash.unprotected_as_bytes()[0]);
        });

    eprintln!("\n[orion_pwhash_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "orion_pwhash_constant_time");

    match &outcome {
        Outcome::Pass { .. } => {
            // Good - Argon2i is constant-time
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            panic!(
                "CRITICAL: Orion pwhash (Argon2i) shows timing leak in password hashing (P={:.3}, exploit={:?})",
                leak_probability, exploitability
            );
        }
        Outcome::Inconclusive {
            leak_probability, ..
        } => {
            eprintln!(
                "Warning: Inconclusive result (leak_probability={:.3})",
                leak_probability
            );
        }
        Outcome::Unmeasurable { .. } => {}
        Outcome::Research(_) => {}
    }
}
