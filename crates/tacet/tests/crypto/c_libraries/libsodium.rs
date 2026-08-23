//! Libsodium timing tests
//!
//! Tests cryptographic implementations from Libsodium via sodiumoxide FFI bindings.
//! Uses DudeCT's two-class pattern:
//! - Baseline: All zeros or fixed values
//! - Sample: Random values
//!
//! Libsodium is designed for misuse resistance and constant-time operation.
//! These tests validate those properties.
//!
//! **NOTE**: These tests require Libsodium development files.
//! On macOS with devenv: `devenv shell` will provide libsodium.dev
//! The sodiumoxide crate should automatically find libsodium in the nix environment.
//!
//! Operations tested:
//! - Ed25519 signing and verification
//! - X25519 key exchange
//! - crypto_box (authenticated encryption)
//! - crypto_secretbox (symmetric authenticated encryption)

use sodiumoxide::crypto::box_::{gen_keypair, open, seal, Nonce};
use sodiumoxide::crypto::scalarmult::{scalarmult, GroupElement, Scalar};
use sodiumoxide::crypto::secretbox;
use sodiumoxide::crypto::sign::ed25519;
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
// Ed25519 Tests (Signature scheme)
// ============================================================================

/// Libsodium Ed25519 signing should be constant-time
///
/// Ed25519 is designed to be constant-time. This validates that property.
/// Uses DudeCT pattern: zeros vs random for messages.
#[test]
fn libsodium_ed25519_sign_constant_time() {
    // Initialize sodiumoxide
    sodiumoxide::init().expect("failed to initialize sodiumoxide");

    // Generate Ed25519 keypair
    let (_pk, sk) = ed25519::gen_keypair();

    // Use zeros vs random pattern for message
    let inputs = InputPair::new(|| [0u8; 64], rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(60))
        .warmup(100)
        .test(inputs, |message| {
            let signature = ed25519::sign_detached(message, &sk);
            std::hint::black_box(signature.as_ref()[0]);
        });

    eprintln!("\n[libsodium_ed25519_sign_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libsodium_ed25519_sign_constant_time");

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
            eprintln!("⚠️  TIMING LEAK DETECTED in Libsodium Ed25519 signing");
            eprintln!(
                "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                leak_probability * 100.0,
                exploitability,
                effect.max_effect_ns
            );
            panic!("Ed25519 signing timing leak - unexpected for Libsodium");
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

/// Libsodium Ed25519 verification should be constant-time
///
/// Validates constant-time properties of Ed25519 signature verification.
#[test]
fn libsodium_ed25519_verify_constant_time() {
    const POOL_SIZE: usize = 100;

    sodiumoxide::init().expect("failed to initialize sodiumoxide");

    let (pk, sk) = ed25519::gen_keypair();

    // Pre-generate two pools of message/signature pairs
    let pool_baseline: Vec<([u8; 64], ed25519::Signature)> = (0..POOL_SIZE)
        .map(|_| {
            let msg = rand_bytes_64();
            let sig = ed25519::sign_detached(&msg, &sk);
            (msg, sig)
        })
        .collect();

    let pool_sample: Vec<([u8; 64], ed25519::Signature)> = (0..POOL_SIZE)
        .map(|_| {
            let msg = rand_bytes_64();
            let sig = ed25519::sign_detached(&msg, &sk);
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
            let result = ed25519::verify_detached(sig, msg, &pk);
            std::hint::black_box(result);
        });

    eprintln!("\n[libsodium_ed25519_verify_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libsodium_ed25519_verify_constant_time");

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
                "Ed25519 verification should be constant-time (got leak_probability={:.1}%, {:?})",
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
// X25519 Tests (Key exchange)
// ============================================================================

/// Libsodium X25519 scalar multiplication should be constant-time
///
/// Tests the core X25519 ECDH operation for data-dependent timing.
/// Uses zeros vs random pattern for scalar bytes (not Scalar type which doesn't implement Hash).
#[test]
fn libsodium_x25519_scalar_mult_constant_time() {
    sodiumoxide::init().expect("failed to initialize sodiumoxide");

    // Use standard X25519 basepoint
    let basepoint = GroupElement([
        9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ]);

    // Use a fixed non-zero scalar vs random scalars (using byte arrays)
    let fixed_scalar_bytes: [u8; 32] = [
        0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82, 0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a, 0x33,
        0x9c, 0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f, 0x9a, 0x2d, 0x7e, 0x4c, 0x8b, 0x3a,
        0x6d, 0x5e,
    ];

    let inputs = InputPair::new(move || fixed_scalar_bytes, rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |scalar_bytes| {
            let scalar = Scalar(*scalar_bytes);
            let result = scalarmult(&scalar, &basepoint);
            if let Ok(point) = result {
                std::hint::black_box(point.0[0]);
            }
        });

    eprintln!("\n[libsodium_x25519_scalar_mult_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libsodium_x25519_scalar_mult_constant_time");

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
            eprintln!("⚠️  TIMING LEAK DETECTED in Libsodium X25519 scalar multiplication");
            eprintln!(
                "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                leak_probability * 100.0,
                exploitability,
                effect.max_effect_ns
            );
            panic!("X25519 scalar multiplication timing leak");
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
// crypto_box Tests (Authenticated encryption)
// ============================================================================

/// Libsodium crypto_box encryption should be constant-time
///
/// crypto_box combines X25519 ECDH + XSalsa20 + Poly1305.
/// Tests for data-dependent timing in the combined operation.
#[test]
fn libsodium_crypto_box_encrypt_constant_time() {
    sodiumoxide::init().expect("failed to initialize sodiumoxide");

    // Generate keypairs for sender and receiver
    let (_sender_pk, sender_sk) = gen_keypair();
    let (receiver_pk, _receiver_sk) = gen_keypair();

    let nonce_counter = std::sync::atomic::AtomicU64::new(0);
    let inputs = InputPair::new(|| [0u8; 64], rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |plaintext| {
            // Generate unique nonce for each encryption
            let n = nonce_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut nonce_bytes = [0u8; 24];
            nonce_bytes[..8].copy_from_slice(&n.to_le_bytes());
            let nonce = Nonce(nonce_bytes);

            let ciphertext = seal(plaintext, &nonce, &receiver_pk, &sender_sk);
            std::hint::black_box(ciphertext[0]);
        });

    eprintln!("\n[libsodium_crypto_box_encrypt_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libsodium_crypto_box_encrypt_constant_time");

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
            eprintln!("⚠️  TIMING LEAK DETECTED in Libsodium crypto_box encryption");
            eprintln!(
                "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                leak_probability * 100.0,
                exploitability,
                effect.max_effect_ns
            );
            panic!("crypto_box encryption timing leak");
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

/// Libsodium crypto_box decryption should be constant-time
///
/// Decryption involves MAC verification which must be constant-time.
#[test]
fn libsodium_crypto_box_decrypt_constant_time() {
    const POOL_SIZE: usize = 100;

    sodiumoxide::init().expect("failed to initialize sodiumoxide");

    let (sender_pk, sender_sk) = gen_keypair();
    let (receiver_pk, receiver_sk) = gen_keypair();

    // Pre-generate two pools of ciphertexts
    let pool_baseline: Vec<(Vec<u8>, Nonce)> = (0..POOL_SIZE)
        .map(|i| {
            let plaintext = rand_bytes_64();
            let mut nonce_bytes = [0u8; 24];
            nonce_bytes[..8].copy_from_slice(&(i as u64).to_le_bytes());
            let nonce = Nonce(nonce_bytes);
            let ciphertext = seal(&plaintext, &nonce, &receiver_pk, &sender_sk);
            (ciphertext, nonce)
        })
        .collect();

    let pool_sample: Vec<(Vec<u8>, Nonce)> = (0..POOL_SIZE)
        .map(|i| {
            let plaintext = rand_bytes_64();
            let mut nonce_bytes = [0u8; 24];
            nonce_bytes[..8].copy_from_slice(&((i + POOL_SIZE) as u64).to_le_bytes());
            let nonce = Nonce(nonce_bytes);
            let ciphertext = seal(&plaintext, &nonce, &receiver_pk, &sender_sk);
            (ciphertext, nonce)
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
        .time_budget(Duration::from_secs(30))
        .test(inputs, |(ciphertext, nonce)| {
            let result = open(ciphertext, nonce, &sender_pk, &receiver_sk);
            std::hint::black_box(result.is_ok());
        });

    eprintln!("\n[libsodium_crypto_box_decrypt_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libsodium_crypto_box_decrypt_constant_time");

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
            eprintln!("⚠️  TIMING LEAK DETECTED in Libsodium crypto_box decryption");
            eprintln!(
                "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                leak_probability * 100.0,
                exploitability,
                effect.max_effect_ns
            );
            panic!("crypto_box decryption timing leak - MAC verification may leak");
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
// crypto_secretbox Tests (Symmetric authenticated encryption)
// ============================================================================

/// Libsodium crypto_secretbox encryption should be constant-time
///
/// crypto_secretbox uses XSalsa20 + Poly1305 for authenticated encryption.
#[test]
fn libsodium_crypto_secretbox_encrypt_constant_time() {
    sodiumoxide::init().expect("failed to initialize sodiumoxide");

    let key = secretbox::gen_key();

    let nonce_counter = std::sync::atomic::AtomicU64::new(0);
    let inputs = InputPair::new(|| [0u8; 64], rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |plaintext| {
            let n = nonce_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut nonce_bytes = [0u8; 24];
            nonce_bytes[..8].copy_from_slice(&n.to_le_bytes());
            let nonce = secretbox::Nonce(nonce_bytes);

            let ciphertext = secretbox::seal(plaintext, &nonce, &key);
            std::hint::black_box(ciphertext[0]);
        });

    eprintln!("\n[libsodium_crypto_secretbox_encrypt_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libsodium_crypto_secretbox_encrypt_constant_time");

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
            eprintln!("⚠️  TIMING LEAK DETECTED in Libsodium crypto_secretbox encryption");
            eprintln!(
                "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                leak_probability * 100.0,
                exploitability,
                effect.max_effect_ns
            );
            panic!("crypto_secretbox encryption timing leak");
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

/// Libsodium crypto_secretbox decryption should be constant-time
///
/// MAC verification must be constant-time to avoid timing oracle attacks.
#[test]
fn libsodium_crypto_secretbox_decrypt_constant_time() {
    const POOL_SIZE: usize = 100;

    sodiumoxide::init().expect("failed to initialize sodiumoxide");

    let key = secretbox::gen_key();

    // Pre-generate two pools of ciphertexts
    let pool_baseline: Vec<(Vec<u8>, secretbox::Nonce)> = (0..POOL_SIZE)
        .map(|i| {
            let plaintext = rand_bytes_64();
            let mut nonce_bytes = [0u8; 24];
            nonce_bytes[..8].copy_from_slice(&(i as u64).to_le_bytes());
            let nonce = secretbox::Nonce(nonce_bytes);
            let ciphertext = secretbox::seal(&plaintext, &nonce, &key);
            (ciphertext, nonce)
        })
        .collect();

    let pool_sample: Vec<(Vec<u8>, secretbox::Nonce)> = (0..POOL_SIZE)
        .map(|i| {
            let plaintext = rand_bytes_64();
            let mut nonce_bytes = [0u8; 24];
            nonce_bytes[..8].copy_from_slice(&((i + POOL_SIZE) as u64).to_le_bytes());
            let nonce = secretbox::Nonce(nonce_bytes);
            let ciphertext = secretbox::seal(&plaintext, &nonce, &key);
            (ciphertext, nonce)
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
        .time_budget(Duration::from_secs(30))
        .test(inputs, |(ciphertext, nonce)| {
            let result = secretbox::open(ciphertext, nonce, &key);
            std::hint::black_box(result.is_ok());
        });

    eprintln!("\n[libsodium_crypto_secretbox_decrypt_constant_time]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "libsodium_crypto_secretbox_decrypt_constant_time");

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
                "crypto_secretbox decryption should be constant-time (got leak_probability={:.1}%, {:?})",
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
// Harness Verification Tests
// ============================================================================

/// Sanity check: identical inputs should pass
#[test]
fn libsodium_harness_sanity_check() {
    sodiumoxide::init().expect("failed to initialize sodiumoxide");

    let (_pk, sk) = ed25519::gen_keypair();

    // Use identical message for both classes
    let fixed_msg = [0x42u8; 64];
    let inputs = InputPair::new(|| fixed_msg, || fixed_msg);

    let outcome = TimingOracle::for_attacker(AttackerModel::Research)
        .time_budget(Duration::from_secs(10))
        .test(inputs, |message| {
            let signature = ed25519::sign_detached(message, &sk);
            std::hint::black_box(signature.as_ref()[0]);
        });

    eprintln!("\n[libsodium_harness_sanity_check]");
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
