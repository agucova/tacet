//! LibreSSL/OpenSSL timing tests
//!
//! Tests cryptographic implementations from LibreSSL/OpenSSL via FFI bindings.
//! Uses DudeCT's two-class pattern:
//! - Baseline: All zeros or fixed values
//! - Sample: Random values
//!
//! CRITICAL: FFI harness verification is essential. Always run sanity checks first.
//!
//! **NOTE**: These tests require LibreSSL/OpenSSL development files.
//! On macOS with devenv: `devenv shell` will provide libressl.dev
//! Environment variable needed: `OPENSSL_DIR=/nix/store/.../libressl-X.Y.Z`
//! Find path with: `find /nix/store -maxdepth 1 -name "*libressl-*" -type d | grep -v "\\-nc\\|\\-man" | head -1`
//!
//! Operations tested:
//! - RSA-2048 PKCS#1 v1.5 decryption (Bleichenbacher-class attacks like MARVIN)
//! - RSA-2048 OAEP decryption
//! - ECDSA P-256 signing and verification
//! - AES-256-GCM encryption (software fallback paths)

use openssl::ec::{EcGroup, EcKey};
use openssl::hash::MessageDigest;
use openssl::nid::Nid;
use openssl::pkey::PKey;
use openssl::rsa::{Padding, Rsa};
use openssl::sign::{Signer, Verifier};
use openssl::symm::{encrypt_aead, Cipher};
use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{skip_if_unreliable, AttackerModel, Outcome, TimingOracle};

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
// RSA PKCS#1 v1.5 Tests (Bleichenbacher-class, MARVIN)
// ============================================================================

/// LibreSSL RSA PKCS#1 v1.5 decryption should be constant-time
///
/// Tests for timing side channels in PKCS#1 v1.5 padding validation.
/// Historical vulnerabilities: Bleichenbacher (1998), ROBOT (2017), MARVIN (2023)
///
/// Uses pool-based pattern to avoid caching artifacts.
#[test]
fn libressl_rsa_2048_pkcs1v15_decrypt_constant_time() {
    const POOL_SIZE: usize = 100;

    // Generate RSA-2048 key pair
    let rsa = Rsa::generate(2048).expect("failed to generate RSA key");
    let private_key = PKey::from_rsa(rsa.clone()).expect("failed to create private key");
    let public_key = PKey::from_rsa(rsa).expect("failed to create public key");

    // Pre-generate two pools of ciphertexts
    let pool_baseline: Vec<Vec<u8>> = (0..POOL_SIZE)
        .map(|_| {
            let msg = rand_bytes_32();
            let mut encrypted = vec![0u8; 256]; // RSA-2048 = 256 bytes
            let rsa = public_key.rsa().unwrap();
            let len = rsa
                .public_encrypt(&msg, &mut encrypted, Padding::PKCS1)
                .unwrap();
            encrypted.truncate(len);
            encrypted
        })
        .collect();

    let pool_sample: Vec<Vec<u8>> = (0..POOL_SIZE)
        .map(|_| {
            let msg = rand_bytes_32();
            let mut encrypted = vec![0u8; 256];
            let rsa = public_key.rsa().unwrap();
            let len = rsa
                .public_encrypt(&msg, &mut encrypted, Padding::PKCS1)
                .unwrap();
            encrypted.truncate(len);
            encrypted
        })
        .collect();

    let idx_baseline = std::cell::Cell::new(0usize);
    let idx_sample = std::cell::Cell::new(0usize);

    let inputs = InputPair::new(
        move || {
            let i = idx_baseline.get();
            idx_baseline.set((i + 1) % POOL_SIZE);
            pool_baseline[i].clone()
        },
        move || {
            let i = idx_sample.get();
            idx_sample.set((i + 1) % POOL_SIZE);
            pool_sample[i].clone()
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(60))
        .warmup(100)
        .calibration_samples(2000)
        .test(inputs, |ciphertext| {
            let rsa = private_key.rsa().unwrap();
            let mut decrypted = vec![0u8; 256];
            // PKCS#1 v1.5 padding - historical source of timing leaks
            let result = rsa.private_decrypt(ciphertext, &mut decrypted, Padding::PKCS1);
            let _ = std::hint::black_box(result);
        });

    eprintln!("\n[libressl_rsa_2048_pkcs1v15_decrypt_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libressl_rsa_2048_pkcs1v15_decrypt_constant_time");

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
            effect,
            ..
        } => {
            eprintln!("⚠️  TIMING LEAK DETECTED in LibreSSL RSA PKCS#1 v1.5 decryption");
            eprintln!(
                "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                leak_probability * 100.0,
                exploitability,
                effect.max_effect_ns
            );
            eprintln!("   This may be a variant of MARVIN (CVE-2023-50782) or similar timing leak");
            panic!("RSA PKCS#1 v1.5 decryption timing leak - potential security vulnerability");
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("Inconclusive: {:?}", reason);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Unmeasurable: {}", recommendation);
        }
        Outcome::Research(_) => {}
    }
}

// ============================================================================
// RSA OAEP Tests
// ============================================================================

/// LibreSSL RSA OAEP decryption should be constant-time
///
/// OAEP is more robust than PKCS#1 v1.5 but still requires constant-time implementation.
#[test]
fn libressl_rsa_2048_oaep_decrypt_constant_time() {
    const POOL_SIZE: usize = 100;

    let rsa = Rsa::generate(2048).expect("failed to generate RSA key");
    let private_key = PKey::from_rsa(rsa.clone()).expect("failed to create private key");
    let public_key = PKey::from_rsa(rsa).expect("failed to create public key");

    // Pre-generate two pools of ciphertexts with OAEP padding
    let pool_baseline: Vec<Vec<u8>> = (0..POOL_SIZE)
        .map(|_| {
            let msg = rand_bytes_32();
            let mut encrypted = vec![0u8; 256];
            let rsa = public_key.rsa().unwrap();
            let len = rsa
                .public_encrypt(&msg, &mut encrypted, Padding::PKCS1_OAEP)
                .unwrap();
            encrypted.truncate(len);
            encrypted
        })
        .collect();

    let pool_sample: Vec<Vec<u8>> = (0..POOL_SIZE)
        .map(|_| {
            let msg = rand_bytes_32();
            let mut encrypted = vec![0u8; 256];
            let rsa = public_key.rsa().unwrap();
            let len = rsa
                .public_encrypt(&msg, &mut encrypted, Padding::PKCS1_OAEP)
                .unwrap();
            encrypted.truncate(len);
            encrypted
        })
        .collect();

    let idx_baseline = std::cell::Cell::new(0usize);
    let idx_sample = std::cell::Cell::new(0usize);

    let inputs = InputPair::new(
        move || {
            let i = idx_baseline.get();
            idx_baseline.set((i + 1) % POOL_SIZE);
            pool_baseline[i].clone()
        },
        move || {
            let i = idx_sample.get();
            idx_sample.set((i + 1) % POOL_SIZE);
            pool_sample[i].clone()
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(60))
        .warmup(100)
        .calibration_samples(2000)
        .test(inputs, |ciphertext| {
            let rsa = private_key.rsa().unwrap();
            let mut decrypted = vec![0u8; 256];
            let result = rsa.private_decrypt(ciphertext, &mut decrypted, Padding::PKCS1_OAEP);
            let _ = std::hint::black_box(result);
        });

    eprintln!("\n[libressl_rsa_2048_oaep_decrypt_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libressl_rsa_2048_oaep_decrypt_constant_time");

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
            effect,
            ..
        } => {
            eprintln!("⚠️  TIMING LEAK DETECTED in LibreSSL RSA OAEP decryption");
            eprintln!(
                "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                leak_probability * 100.0,
                exploitability,
                effect.max_effect_ns
            );
            panic!("RSA OAEP decryption timing leak");
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("Inconclusive: {:?}", reason);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Unmeasurable: {}", recommendation);
        }
        Outcome::Research(_) => {}
    }
}

// ============================================================================
// ECDSA P-256 Tests (Multiple 2024 CVEs in other implementations)
// ============================================================================

/// LibreSSL ECDSA P-256 signing should be constant-time
///
/// ECDSA is particularly sensitive to timing attacks - nonce generation and
/// modular inversion can leak the private key. Multiple CVEs in 2024.
#[test]
fn libressl_ecdsa_p256_sign_constant_time() {
    // Generate P-256 key pair
    let group = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1).expect("failed to create EC group");
    let ec_key = EcKey::generate(&group).expect("failed to generate EC key");
    let private_key = PKey::from_ec_key(ec_key).expect("failed to create private key");

    // Use zeros vs random pattern for message
    let inputs = InputPair::new(|| [0u8; 32], rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(60))
        .warmup(100)
        .test(inputs, |message| {
            let mut signer = Signer::new(MessageDigest::sha256(), &private_key)
                .expect("failed to create signer");
            signer.update(message).unwrap();
            let signature = signer.sign_to_vec().unwrap();
            std::hint::black_box(signature[0]);
        });

    eprintln!("\n[libressl_ecdsa_p256_sign_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libressl_ecdsa_p256_sign_constant_time");

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
            effect,
            ..
        } => {
            eprintln!("⚠️  TIMING LEAK DETECTED in LibreSSL ECDSA P-256 signing");
            eprintln!(
                "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                leak_probability * 100.0,
                exploitability,
                effect.max_effect_ns
            );
            eprintln!("   Similar to CVEs in other ECDSA implementations (2024)");
            panic!("ECDSA P-256 signing timing leak - can reveal private key");
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("Inconclusive: {:?}", reason);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Unmeasurable: {}", recommendation);
        }
        Outcome::Research(_) => {}
    }
}

/// LibreSSL ECDSA P-256 verification should be constant-time
///
/// While verification doesn't directly involve the private key, it should
/// still be constant-time to avoid information leakage.
#[test]
fn libressl_ecdsa_p256_verify_constant_time() {
    const POOL_SIZE: usize = 100;

    // Generate P-256 key pair
    let group = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1).expect("failed to create EC group");
    let ec_key = EcKey::generate(&group).expect("failed to generate EC key");
    let private_key = PKey::from_ec_key(ec_key.clone()).expect("failed to create private key");
    let public_key = PKey::from_ec_key(ec_key).expect("failed to create public key");

    // Pre-generate two pools of message/signature pairs
    let pool_baseline: Vec<([u8; 32], Vec<u8>)> = (0..POOL_SIZE)
        .map(|_| {
            let msg = rand_bytes_32();
            let mut signer = Signer::new(MessageDigest::sha256(), &private_key).unwrap();
            signer.update(&msg).unwrap();
            let sig = signer.sign_to_vec().unwrap();
            (msg, sig)
        })
        .collect();

    let pool_sample: Vec<([u8; 32], Vec<u8>)> = (0..POOL_SIZE)
        .map(|_| {
            let msg = rand_bytes_32();
            let mut signer = Signer::new(MessageDigest::sha256(), &private_key).unwrap();
            signer.update(&msg).unwrap();
            let sig = signer.sign_to_vec().unwrap();
            (msg, sig)
        })
        .collect();

    let idx_baseline = std::cell::Cell::new(0usize);
    let idx_sample = std::cell::Cell::new(0usize);

    let inputs = InputPair::new(
        move || {
            let i = idx_baseline.get();
            idx_baseline.set((i + 1) % POOL_SIZE);
            i
        },
        move || {
            let i = idx_sample.get();
            idx_sample.set((i + 1) % POOL_SIZE);
            i + POOL_SIZE
        },
    );

    let all_pairs: Vec<_> = pool_baseline.into_iter().chain(pool_sample).collect();

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |idx| {
            let (msg, sig) = &all_pairs[*idx];
            let mut verifier = Verifier::new(MessageDigest::sha256(), &public_key).unwrap();
            verifier.update(msg).unwrap();
            let result = verifier.verify(sig);
            let _ = std::hint::black_box(result);
        });

    eprintln!("\n[libressl_ecdsa_p256_verify_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libressl_ecdsa_p256_verify_constant_time");

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
                "ECDSA P-256 verification should be constant-time (got leak_probability={:.1}%, {:?})",
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
        Outcome::Research(_) => {}
    }
}

// ============================================================================
// AES-256-GCM Tests (Software fallback paths)
// ============================================================================

/// LibreSSL AES-256-GCM encryption should be constant-time
///
/// Tests software fallback paths (no AES-NI) for data-dependent timing.
#[test]
fn libressl_aes_256_gcm_encrypt_constant_time() {
    let key = rand_bytes_32();
    let cipher = Cipher::aes_256_gcm();

    let nonce_counter = std::sync::atomic::AtomicU64::new(0);
    let inputs = InputPair::new(|| [0u8; 64], rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |plaintext| {
            // Generate unique nonce for each encryption
            let n = nonce_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut nonce = [0u8; 12];
            nonce[..8].copy_from_slice(&n.to_le_bytes());

            let mut tag = vec![0u8; 16];
            let ciphertext = encrypt_aead(cipher, &key, Some(&nonce), &[], plaintext, &mut tag)
                .expect("encryption failed");
            std::hint::black_box(ciphertext[0]);
            std::hint::black_box(tag[0]);
        });

    eprintln!("\n[libressl_aes_256_gcm_encrypt_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libressl_aes_256_gcm_encrypt_constant_time");

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
            effect,
            ..
        } => {
            eprintln!("⚠️  TIMING LEAK DETECTED in LibreSSL AES-256-GCM encryption");
            eprintln!(
                "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                leak_probability * 100.0,
                exploitability,
                effect.max_effect_ns
            );
            panic!("AES-256-GCM encryption timing leak");
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("Inconclusive: {:?}", reason);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Unmeasurable: {}", recommendation);
        }
        Outcome::Research(_) => {}
    }
}

// ============================================================================
// Harness Verification Tests
// ============================================================================

/// Sanity check: identical inputs should pass
///
/// FFI overhead can introduce noise - verify harness is working correctly
#[test]
fn libressl_harness_sanity_check() {
    let rsa = Rsa::generate(2048).expect("failed to generate RSA key");
    let private_key = PKey::from_rsa(rsa.clone()).expect("failed to create private key");
    let public_key = PKey::from_rsa(rsa).expect("failed to create public key");

    // Pre-generate a single ciphertext
    let msg = rand_bytes_32();
    let mut encrypted = vec![0u8; 256];
    let rsa = public_key.rsa().unwrap();
    let len = rsa
        .public_encrypt(&msg, &mut encrypted, Padding::PKCS1)
        .unwrap();
    encrypted.truncate(len);

    // Both classes return the same ciphertext
    let inputs = InputPair::new(|| encrypted.clone(), || encrypted.clone());

    let outcome = TimingOracle::for_attacker(AttackerModel::Research)
        .time_budget(Duration::from_secs(10))
        .test(inputs, |ciphertext| {
            let rsa = private_key.rsa().unwrap();
            let mut decrypted = vec![0u8; 256];
            let result = rsa.private_decrypt(ciphertext, &mut decrypted, Padding::PKCS1);
            let _ = std::hint::black_box(result);
        });

    eprintln!("\n[libressl_harness_sanity_check]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    match &outcome {
        Outcome::Pass { .. } => {
            eprintln!("Sanity check passed - harness is working correctly");
        }
        Outcome::Fail { .. } => {
            panic!("Sanity check failed - identical inputs should not show timing difference");
        }
        _ => {}
    }
}
