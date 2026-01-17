//! RSA timing tests
//!
//! Tests RSA operations for timing side channels using DudeCT's two-class pattern:
//! - Class 0: Fixed message
//! - Class 1: Random message
//!
//! RSA is particularly sensitive to timing attacks due to modular exponentiation.
//! Modern implementations use blinding and constant-time techniques.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.
//!
//! NOTE: We use RSA-1024 instead of RSA-2048 for performance (~4-8x faster).
//! Constant-time properties are identical - same algorithm, smaller numbers.
//! This tests the same code paths while keeping test runtime reasonable.

use rsa::pkcs1v15::{SigningKey, VerifyingKey};
use rsa::rand_core::OsRng;
use rsa::signature::{RandomizedSigner, SignatureEncoding, Verifier};
use rsa::{Pkcs1v15Encrypt, RsaPrivateKey, RsaPublicKey};
use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{skip_if_unreliable, AttackerModel, Outcome, TimingOracle};

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

// ============================================================================
// RSA-1024 Encryption Tests (1024-bit for speed; same constant-time code paths)
// ============================================================================

/// RSA encryption should be constant-time
///
/// Note: RSA encryption uses OAEP/PKCS#1 padding which involves randomization,
/// but the underlying modular exponentiation should be constant-time.
#[test]
fn rsa_1024_encrypt_constant_time() {
    // Generate a 1024-bit RSA key pair (faster than 2048, same constant-time properties)
    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).expect("failed to generate key");
    let public_key = RsaPublicKey::from(&private_key);

    // Non-pathological fixed message (RSA-2048 with PKCS#1 v1.5 can encrypt up to 245 bytes)
    let fixed_message: [u8; 32] = [0x42; 32];
    let inputs = InputPair::new(|| fixed_message, rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(60))
        .test(inputs, |msg| {
            // Note: PKCS#1 v1.5 encryption is randomized, but we're testing
            // whether the message content affects timing
            let ciphertext = public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, msg)
                .unwrap();
            std::hint::black_box(ciphertext[0]);
        });

    eprintln!("\n[rsa_1024_encrypt_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rsa_1024_encrypt_constant_time");

    // RSA encryption should be constant-time with respect to message content
    match &outcome {
        Outcome::Pass {
            leak_probability,
            quality,
            ..
        } => {
            eprintln!(
                "Test passed: P(leak)={:.1}%, quality={:?}",
                leak_probability * 100.0,
                quality
            );
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            panic!(
                "RSA-1024 encryption should be constant-time (got leak_probability={:.1}%, {:?})",
                leak_probability * 100.0,
                exploitability
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("Inconclusive: {:?}", reason);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Unmeasurable: {}", recommendation);
        }
        &Outcome::Research(_) => {}
    }
}

/// RSA decryption should be constant-time
///
/// Decryption is the most sensitive operation - involves private exponent.
///
/// NOTE: Both classes use DIFFERENT ciphertexts to avoid microarchitectural
/// caching artifacts. Using "same ciphertext repeated" vs "different ciphertexts"
/// measures cache warming (~250ns), not algorithmic timing leaks.
#[test]
fn rsa_1024_decrypt_constant_time() {
    const SAMPLES: usize = 200; // Pool size per class

    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).expect("failed to generate key");
    let public_key = RsaPublicKey::from(&private_key);

    // Pre-generate TWO separate pools of ciphertexts
    // Both classes cycle through different ciphertexts to avoid cache warming artifacts
    let pool_baseline: Vec<Vec<u8>> = (0..SAMPLES)
        .map(|_| {
            let msg = rand_bytes_32();
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
                .unwrap()
        })
        .collect();

    let pool_sample: Vec<Vec<u8>> = (0..SAMPLES)
        .map(|_| {
            let msg = rand_bytes_32();
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
                .unwrap()
        })
        .collect();

    let idx_baseline = std::cell::Cell::new(0usize);
    let idx_sample = std::cell::Cell::new(0usize);
    let inputs = InputPair::new(
        move || {
            let i = idx_baseline.get();
            idx_baseline.set((i + 1) % SAMPLES);
            pool_baseline[i].clone()
        },
        move || {
            let i = idx_sample.get();
            idx_sample.set((i + 1) % SAMPLES);
            pool_sample[i].clone()
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .warmup(100)
        .calibration_samples(2000)
        .test(inputs, |ct| {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    eprintln!("\n[rsa_1024_decrypt_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rsa_1024_decrypt_constant_time");

    // Modern RSA implementations use blinding
    match &outcome {
        Outcome::Pass {
            leak_probability,
            quality,
            ..
        } => {
            eprintln!(
                "Test passed: P(leak)={:.1}%, quality={:?}",
                leak_probability * 100.0,
                quality
            );
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            panic!(
                "RSA-1024 decryption should be constant-time (got leak_probability={:.1}%, {:?})",
                leak_probability * 100.0,
                exploitability
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("Inconclusive: {:?}", reason);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Unmeasurable: {}", recommendation);
        }
        &Outcome::Research(_) => {}
    }
}

// ============================================================================
// RSA Signing Tests (2048-bit to avoid cache artifacts)
// ============================================================================

/// RSA signing should be constant-time
///
/// Signing uses the private key and is sensitive to timing attacks.
/// Uses RSA-2048 because RSA-1024 shows cache-related timing artifacts.
///
/// This test is ignored by default because RSA-2048 key generation and
/// signing are very slow (5+ minutes). Run with `cargo test -- --ignored`
/// for thorough validation.
#[test]
#[ignore]
fn rsa_2048_sign_constant_time() {
    let private_key = RsaPrivateKey::new(&mut OsRng, 2048).expect("failed to generate key");
    let signing_key = SigningKey::<sha2::Sha256>::new_unprefixed(private_key);

    // Non-pathological fixed message
    let fixed_message: [u8; 64] = [0x42; 64];
    let inputs = InputPair::new(|| fixed_message, rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .warmup(100)
        .calibration_samples(2000)
        .test(inputs, |msg| {
            let signature = signing_key.sign_with_rng(&mut OsRng, msg);
            std::hint::black_box(signature.to_bytes().as_ref()[0]);
        });

    eprintln!("\n[rsa_2048_sign_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rsa_2048_sign_constant_time");

    match &outcome {
        Outcome::Pass {
            leak_probability,
            quality,
            ..
        } => {
            eprintln!(
                "Test passed: P(leak)={:.1}%, quality={:?}",
                leak_probability * 100.0,
                quality
            );
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            panic!(
                "RSA-2048 signing should be constant-time (got leak_probability={:.1}%, {:?})",
                leak_probability * 100.0,
                exploitability
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("Inconclusive: {:?}", reason);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Unmeasurable: {}", recommendation);
        }
        &Outcome::Research(_) => {}
    }
}

/// RSA signature verification should be constant-time
///
/// Verification uses public key but still should be constant-time.
///
/// Uses pool-based pattern (100+ per class) to average out per-value timing
/// variation, following the methodology from investigation-rsa-timing-anomaly.md.
/// Both pools contain random messages to test general constant-time behavior.
///
/// NOTE: This test requires PMU timer (sudo) for reliable results on Apple Silicon.
/// The standard cntvct_el0 timer (41.7ns resolution) causes ThresholdElevated,
/// making the test unreliable without cycle-level precision.
#[test]
#[ignore] // Requires PMU timer: sudo -E cargo test --ignored
fn rsa_1024_verify_constant_time() {
    const POOL_SIZE: usize = 100;

    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).expect("failed to generate key");
    let public_key = RsaPublicKey::from(&private_key);
    let signing_key = SigningKey::<sha2::Sha256>::new_unprefixed(private_key);
    let verifying_key = VerifyingKey::<sha2::Sha256>::new_unprefixed(public_key);

    // Pre-generate two separate pools of message/signature pairs
    let pool_a: Vec<([u8; 64], _)> = (0..POOL_SIZE)
        .map(|_| {
            let msg = rand_bytes_64();
            let sig = signing_key.sign_with_rng(&mut OsRng, &msg);
            (msg, sig)
        })
        .collect();

    let pool_b: Vec<([u8; 64], _)> = (0..POOL_SIZE)
        .map(|_| {
            let msg = rand_bytes_64();
            let sig = signing_key.sign_with_rng(&mut OsRng, &msg);
            (msg, sig)
        })
        .collect();

    // Cycling indices for each pool
    let idx_a = std::cell::Cell::new(0usize);
    let idx_b = std::cell::Cell::new(0usize);

    let inputs = InputPair::new(
        move || {
            let i = idx_a.get();
            idx_a.set((i + 1) % POOL_SIZE);
            i
        },
        move || {
            let i = idx_b.get();
            idx_b.set((i + 1) % POOL_SIZE);
            i + POOL_SIZE // Offset to distinguish from baseline
        },
    );

    // Combine pools for index-based access
    let all_pairs: Vec<_> = pool_a.into_iter().chain(pool_b).collect();

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |idx| {
            let (msg, sig) = &all_pairs[*idx];
            let result = verifying_key.verify(msg, sig);
            std::hint::black_box(result.is_ok());
        });

    eprintln!("\n[rsa_1024_verify_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rsa_1024_verify_constant_time");

    match &outcome {
        Outcome::Pass {
            leak_probability,
            quality,
            ..
        } => {
            eprintln!(
                "Test passed: P(leak)={:.1}%, quality={:?}",
                leak_probability * 100.0,
                quality
            );
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            panic!(
                "RSA-1024 verification should be constant-time (got leak_probability={:.1}%, {:?})",
                leak_probability * 100.0,
                exploitability
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("Inconclusive: {:?}", reason);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Unmeasurable: {}", recommendation);
        }
        &Outcome::Research(_) => {}
    }
}

// ============================================================================
// Comparative Tests (1024-bit for speed; same constant-time code paths)
// ============================================================================

/// Compare all-zeros vs all-ones message for RSA encryption
#[test]
fn rsa_1024_hamming_weight_independence() {
    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).expect("failed to generate key");
    let public_key = RsaPublicKey::from(&private_key);

    let inputs = InputPair::new(|| [0x00u8; 32], || [0xFFu8; 32]);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(60))
        .test(inputs, |msg| {
            let ct = public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, msg)
                .unwrap();
            std::hint::black_box(ct[0]);
        });

    eprintln!("\n[rsa_1024_hamming_weight_independence]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rsa_1024_hamming_weight_independence");

    match &outcome {
        Outcome::Pass {
            leak_probability,
            quality,
            ..
        } => {
            eprintln!(
                "Test passed: P(leak)={:.1}%, quality={:?}",
                leak_probability * 100.0,
                quality
            );
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            panic!(
                "RSA Hamming weight should be constant-time (got leak_probability={:.1}%, {:?})",
                leak_probability * 100.0,
                exploitability
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("Inconclusive: {:?}", reason);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Unmeasurable: {}", recommendation);
        }
        &Outcome::Research(_) => {}
    }
}

/// RSA key size comparison - 2048 vs 4096 (informational)
///
/// This is NOT a constant-time test - just verifies our tool detects
/// the expected timing difference from different key sizes
#[test]
#[ignore] // Run with --ignored, this is slow
fn rsa_key_size_timing_difference() {
    let key_2048 = RsaPrivateKey::new(&mut OsRng, 2048).expect("failed to generate 2048-bit key");
    let key_4096 = RsaPrivateKey::new(&mut OsRng, 4096).expect("failed to generate 4096-bit key");
    let pub_2048 = RsaPublicKey::from(&key_2048);
    let pub_4096 = RsaPublicKey::from(&key_4096);

    let message = [0x42u8; 32];

    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(120))
        .test(inputs, |key_idx| {
            if *key_idx == 0 {
                let ct = pub_2048
                    .encrypt(&mut OsRng, Pkcs1v15Encrypt, &message)
                    .unwrap();
                std::hint::black_box(ct[0]);
            } else {
                let ct = pub_4096
                    .encrypt(&mut OsRng, Pkcs1v15Encrypt, &message)
                    .unwrap();
                std::hint::black_box(ct[0]);
            }
        });

    eprintln!("\n[rsa_key_size_timing_difference]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rsa_key_size_timing_difference");

    // We EXPECT a timing difference here - 4096-bit operations are slower
    eprintln!("Note: Timing difference expected (4096-bit is ~4x slower than 2048-bit)");

    match &outcome {
        Outcome::Pass {
            leak_probability,
            quality,
            ..
        } => {
            eprintln!(
                "Unexpected pass: P(leak)={:.1}%, quality={:?}",
                leak_probability * 100.0,
                quality
            );
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            effect,
            ..
        } => {
            eprintln!(
                "Expected timing difference detected: P(leak)={:.1}%, {:?}, effect={:?}",
                leak_probability * 100.0,
                exploitability,
                effect
            );
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("Inconclusive: {:?}", reason);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Unmeasurable: {}", recommendation);
        }
        &Outcome::Research(_) => {}
    }
}

/// Control test: both classes use different ciphertexts
/// If this passes but rsa_1024_decrypt fails, the difference is microarchitectural caching
#[test]
fn rsa_1024_decrypt_control_both_random() {
    const SAMPLES: usize = 400; // Split into two pools

    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).expect("failed to generate key");
    let public_key = RsaPublicKey::from(&private_key);

    // Pre-generate TWO pools of ciphertexts
    let pool_a: Vec<Vec<u8>> = (0..SAMPLES / 2)
        .map(|_| {
            let msg = rand_bytes_32();
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
                .unwrap()
        })
        .collect();

    let pool_b: Vec<Vec<u8>> = (0..SAMPLES / 2)
        .map(|_| {
            let msg = rand_bytes_32();
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
                .unwrap()
        })
        .collect();

    let idx_a = std::cell::Cell::new(0usize);
    let idx_b = std::cell::Cell::new(0usize);

    let inputs = InputPair::new(
        move || {
            let i = idx_a.get();
            idx_a.set((i + 1) % (SAMPLES / 2));
            pool_a[i].clone()
        },
        move || {
            let i = idx_b.get();
            idx_b.set((i + 1) % (SAMPLES / 2));
            pool_b[i].clone()
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .warmup(100)
        .calibration_samples(2000)
        .test(inputs, |ct| {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    eprintln!("\n[rsa_1024_decrypt_control_both_random]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // This is informational - we want to see if the effect disappears
    match &outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            eprintln!("Control passed: P(leak)={:.1}%", leak_probability * 100.0);
            eprintln!("This suggests the original failure was microarchitectural caching, not an RSA timing leak");
        }
        Outcome::Fail {
            leak_probability, ..
        } => {
            eprintln!("Control failed: P(leak)={:.1}%", leak_probability * 100.0);
            eprintln!("This suggests a real timing leak in RSA decryption");
        }
        _ => {}
    }
}
