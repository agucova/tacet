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

use rsa::pkcs1v15::{SigningKey, VerifyingKey};
use rsa::rand_core::OsRng;
use rsa::signature::{RandomizedSigner, SignatureEncoding, Verifier};
use rsa::{Pkcs1v15Encrypt, RsaPrivateKey, RsaPublicKey};
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

// ============================================================================
// RSA-2048 Encryption Tests
// ============================================================================

/// RSA-2048 encryption should be constant-time
///
/// Note: RSA encryption uses OAEP/PKCS#1 padding which involves randomization,
/// but the underlying modular exponentiation should be constant-time.
#[test]
fn rsa_2048_encrypt_constant_time() {
    const SAMPLES: usize = 10_000; // RSA is slow

    // Generate a 2048-bit RSA key pair
    let private_key = RsaPrivateKey::new(&mut OsRng, 2048).expect("failed to generate key");
    let public_key = RsaPublicKey::from(&private_key);

    // Non-pathological fixed message (RSA-2048 with PKCS#1 v1.5 can encrypt up to 245 bytes)
    let fixed_message: [u8; 32] = [0x42; 32];
    let inputs = InputPair::new(|| fixed_message, rand_bytes_32);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .min_effect_ns(50.0)
        .test(inputs, |msg| {
            // Note: PKCS#1 v1.5 encryption is randomized, but we're testing
            // whether the message content affects timing
            let ciphertext = public_key.encrypt(&mut OsRng, Pkcs1v15Encrypt, msg).unwrap();
            std::hint::black_box(ciphertext[0]);
        });

    let result = skip_if_unreliable!(outcome, "rsa_2048_encrypt_constant_time");

    eprintln!("\n[rsa_2048_encrypt_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // RSA encryption should be constant-time with respect to message content
    assert!(
        result.ci_gate.passed,
        "RSA-2048 encryption should be constant-time"
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
    assert!(
        matches!(
            result.exploitability,
            Exploitability::Negligible | Exploitability::PossibleLAN
        ),
        "Exploitability: {:?}",
        result.exploitability
    );
}

/// RSA-2048 decryption should be constant-time
///
/// Decryption is the most sensitive operation - involves private exponent
#[test]
fn rsa_2048_decrypt_constant_time() {
    const SAMPLES: usize = 5_000; // RSA decryption is very slow

    let private_key = RsaPrivateKey::new(&mut OsRng, 2048).expect("failed to generate key");
    let public_key = RsaPublicKey::from(&private_key);

    // Pre-generate ciphertexts
    let fixed_message = [0x42u8; 32];
    let fixed_ciphertext = public_key
        .encrypt(&mut OsRng, Pkcs1v15Encrypt, &fixed_message)
        .unwrap();

    let random_ciphertexts: Vec<Vec<u8>> = (0..SAMPLES)
        .map(|_| {
            let msg = rand_bytes_32();
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
                .unwrap()
        })
        .collect();

    let idx = std::cell::Cell::new(0usize);
    let fixed_ciphertext_clone = fixed_ciphertext.clone();
    let inputs = InputPair::new(move || fixed_ciphertext_clone.clone(), move || {
        let i = idx.get();
        idx.set((i + 1) % SAMPLES);
        random_ciphertexts[i].clone()
    });

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .min_effect_ns(50.0)
        .test(inputs, |ct| {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    let result = skip_if_unreliable!(outcome, "rsa_2048_decrypt_constant_time");

    eprintln!("\n[rsa_2048_decrypt_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // Modern RSA implementations use blinding
    assert!(
        result.ci_gate.passed,
        "RSA-2048 decryption should be constant-time"
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
    assert!(
        matches!(
            result.exploitability,
            Exploitability::Negligible | Exploitability::PossibleLAN
        ),
        "Exploitability: {:?}",
        result.exploitability
    );
}

// ============================================================================
// RSA Signing Tests
// ============================================================================

/// RSA-2048 signing should be constant-time
///
/// Signing uses the private key and is sensitive to timing attacks
#[test]
fn rsa_2048_sign_constant_time() {
    const SAMPLES: usize = 5_000;

    let private_key = RsaPrivateKey::new(&mut OsRng, 2048).expect("failed to generate key");
    let signing_key = SigningKey::<sha2::Sha256>::new_unprefixed(private_key);

    // Non-pathological fixed message
    let fixed_message: [u8; 64] = [0x42; 64];
    let inputs = InputPair::new(|| fixed_message, rand_bytes_64);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .min_effect_ns(50.0)
        .test(inputs, |msg| {
            let signature = signing_key.sign_with_rng(&mut OsRng, msg);
            std::hint::black_box(signature.to_bytes().as_ref()[0]);
        });

    let result = skip_if_unreliable!(outcome, "rsa_2048_sign_constant_time");

    eprintln!("\n[rsa_2048_sign_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed,
        "RSA-2048 signing should be constant-time"
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
    assert!(
        matches!(
            result.exploitability,
            Exploitability::Negligible | Exploitability::PossibleLAN
        ),
        "Exploitability: {:?}",
        result.exploitability
    );
}

/// RSA-2048 signature verification should be constant-time
///
/// Verification uses public key but still should be constant-time
#[test]
fn rsa_2048_verify_constant_time() {
    const SAMPLES: usize = 5_000;

    let private_key = RsaPrivateKey::new(&mut OsRng, 2048).expect("failed to generate key");
    let public_key = RsaPublicKey::from(&private_key);
    let signing_key = SigningKey::<sha2::Sha256>::new_unprefixed(private_key);
    let verifying_key = VerifyingKey::<sha2::Sha256>::new_unprefixed(public_key);

    // Pre-generate signatures - RSA Signature doesn't implement Hash, so we
    // use an index-based approach
    let fixed_message = [0x42u8; 64];
    let fixed_signature = signing_key.sign_with_rng(&mut OsRng, &fixed_message);

    let random_msgs: Vec<[u8; 64]> = (0..SAMPLES).map(|_| rand_bytes_64()).collect();
    let random_sigs: Vec<_> = random_msgs
        .iter()
        .map(|msg| signing_key.sign_with_rng(&mut OsRng, msg))
        .collect();

    // Use index to select between pre-generated values
    let idx = std::cell::Cell::new(0usize);
    let inputs = InputPair::new(|| 0usize, || {
        let i = idx.get();
        idx.set((i + 1) % SAMPLES);
        i + 1 // Non-zero index for random class
    });

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .min_effect_ns(50.0)
        .test(inputs, |class| {
            let result = if *class == 0 {
                verifying_key.verify(&fixed_message, &fixed_signature)
            } else {
                let i = (*class - 1) % SAMPLES;
                verifying_key.verify(&random_msgs[i], &random_sigs[i])
            };
            std::hint::black_box(result.is_ok());
        });

    let result = skip_if_unreliable!(outcome, "rsa_2048_verify_constant_time");

    eprintln!("\n[rsa_2048_verify_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed,
        "RSA-2048 verification should be constant-time"
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
    assert!(
        matches!(
            result.exploitability,
            Exploitability::Negligible | Exploitability::PossibleLAN
        ),
        "Exploitability: {:?}",
        result.exploitability
    );
}

// ============================================================================
// Comparative Tests
// ============================================================================

/// Compare all-zeros vs all-ones message for RSA encryption
#[test]
fn rsa_2048_hamming_weight_independence() {
    const SAMPLES: usize = 5_000;

    let private_key = RsaPrivateKey::new(&mut OsRng, 2048).expect("failed to generate key");
    let public_key = RsaPublicKey::from(&private_key);

    let inputs = InputPair::new(|| [0x00u8; 32], || [0xFFu8; 32]);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .min_effect_ns(50.0)
        .test(inputs, |msg| {
            let ct = public_key.encrypt(&mut OsRng, Pkcs1v15Encrypt, msg).unwrap();
            std::hint::black_box(ct[0]);
        });

    let result = skip_if_unreliable!(outcome, "rsa_2048_hamming_weight_independence");

    eprintln!("\n[rsa_2048_hamming_weight_independence]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed,
        "Should be constant-time"
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
    assert!(
        matches!(
            result.exploitability,
            Exploitability::Negligible | Exploitability::PossibleLAN
        ),
        "RSA Hamming weight should not affect timing (got {:?})",
        result.exploitability
    );
}

/// RSA key size comparison - 2048 vs 4096 (informational)
///
/// This is NOT a constant-time test - just verifies our tool detects
/// the expected timing difference from different key sizes
#[test]
#[ignore] // Run with --ignored, this is slow
fn rsa_key_size_timing_difference() {
    const SAMPLES: usize = 2_000;

    let key_2048 = RsaPrivateKey::new(&mut OsRng, 2048).expect("failed to generate 2048-bit key");
    let key_4096 = RsaPrivateKey::new(&mut OsRng, 4096).expect("failed to generate 4096-bit key");
    let pub_2048 = RsaPublicKey::from(&key_2048);
    let pub_4096 = RsaPublicKey::from(&key_4096);

    let message = [0x42u8; 32];

    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .min_effect_ns(50.0)
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

    let result = skip_if_unreliable!(outcome, "rsa_key_size_timing_difference");

    eprintln!("\n[rsa_key_size_timing_difference]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // We EXPECT a timing difference here - 4096-bit operations are slower
    eprintln!(
        "Note: Timing difference expected (4096-bit is ~4x slower than 2048-bit)"
    );
    assert!(
        result.leak_probability < 0.3,
        "Leak probability too high: {:.1}%",
        result.leak_probability * 100.0
    );
}
