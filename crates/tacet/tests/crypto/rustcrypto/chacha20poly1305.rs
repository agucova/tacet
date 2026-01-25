//! crypto::rustcrypto::chacha20poly1305
//!
//! ChaCha20-Poly1305 AEAD timing tests using the RustCrypto implementation.
//! Crate: chacha20poly1305
//! Family: AEAD
//! Expected: All Pass (constant-time implementation)
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{skip_if_unreliable, AttackerModel, Exploitability, Outcome, TimingOracle};

fn rand_bytes_64() -> [u8; 64] {
    let mut arr = [0u8; 64];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

// ============================================================================
// Core ChaCha20-Poly1305 Tests
// ============================================================================

/// ChaCha20-Poly1305 encryption should be constant-time
#[test]
fn rustcrypto_chacha20poly1305_encrypt_ct() {
    let key_bytes: [u8; 32] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
        0x1e, 0x1f,
    ];
    let cipher = ChaCha20Poly1305::new(&key_bytes.into());

    let nonce_counter = std::sync::atomic::AtomicU64::new(0);
    let fixed_plaintext: [u8; 64] = [0x42; 64];
    let inputs = InputPair::new(|| fixed_plaintext, rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |plaintext| {
            let nonce_value = nonce_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut nonce_bytes = [0u8; 12];
            nonce_bytes[..8].copy_from_slice(&nonce_value.to_le_bytes());
            let nonce = Nonce::from_slice(&nonce_bytes);

            let ciphertext = cipher.encrypt(nonce, plaintext.as_ref()).unwrap();
            std::hint::black_box(ciphertext[0]);
        });

    eprintln!("\n[rustcrypto_chacha20poly1305_encrypt_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_chacha20poly1305_encrypt_ct");

    match &outcome {
        Outcome::Pass { leak_probability, .. } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail { leak_probability, exploitability, .. } => {
            panic!(
                "ChaCha20-Poly1305 encryption should be constant-time (got leak_probability={:.1}%, {:?})",
                leak_probability * 100.0, exploitability
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

    if let Some(exp) = get_exploitability(&outcome) {
        assert!(
            matches!(exp, Exploitability::SharedHardwareOnly | Exploitability::Http2Multiplexing),
            "ChaCha20-Poly1305 should have low exploitability (got {:?})",
            exp
        );
    }
}

/// ChaCha20-Poly1305 decryption should be constant-time
#[test]
fn rustcrypto_chacha20poly1305_decrypt_ct() {
    const SAMPLES: usize = 30_000;

    let key_bytes: [u8; 32] = [0x5a; 32];
    let cipher = ChaCha20Poly1305::new(&key_bytes.into());
    let nonce = Nonce::from_slice(&[0u8; 12]);

    let fixed_plaintext = [0x42u8; 64];
    let fixed_ciphertext = cipher.encrypt(nonce, fixed_plaintext.as_ref()).unwrap();

    let random_ciphertexts: Vec<Vec<u8>> = (0..SAMPLES)
        .map(|_| {
            let pt = rand_bytes_64();
            cipher.encrypt(nonce, pt.as_ref()).unwrap()
        })
        .collect();

    let idx = std::cell::Cell::new(0usize);
    let fixed_ciphertext_clone = fixed_ciphertext.clone();
    let inputs = InputPair::new(
        move || fixed_ciphertext_clone.clone(),
        move || {
            let i = idx.get();
            idx.set((i + 1) % SAMPLES);
            random_ciphertexts[i].clone()
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |ct| {
            let plaintext = cipher.decrypt(nonce, ct.as_ref()).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    eprintln!("\n[rustcrypto_chacha20poly1305_decrypt_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_chacha20poly1305_decrypt_ct");

    match &outcome {
        Outcome::Pass { leak_probability, .. } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail { leak_probability, exploitability, .. } => {
            panic!(
                "ChaCha20-Poly1305 decryption should be constant-time (got leak_probability={:.1}%, {:?})",
                leak_probability * 100.0, exploitability
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

    if let Some(exp) = get_exploitability(&outcome) {
        assert!(
            matches!(exp, Exploitability::SharedHardwareOnly | Exploitability::Http2Multiplexing),
            "ChaCha20-Poly1305 decryption should have low exploitability (got {:?})",
            exp
        );
    }
}

/// ChaCha20-Poly1305 with varying nonces should be constant-time
#[test]
fn rustcrypto_chacha20poly1305_nonce_ct() {
    let key_bytes: [u8; 32] = [0x73; 32];
    let cipher = ChaCha20Poly1305::new(&key_bytes.into());
    let plaintext = [0x42u8; 64];

    let nonces = InputPair::new(|| [0x00u8; 12], || [0xFFu8; 12]);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(nonces, |nonce_bytes| {
            let nonce = Nonce::from_slice(nonce_bytes);
            let ciphertext = cipher.encrypt(nonce, plaintext.as_ref()).unwrap();
            std::hint::black_box(ciphertext[0]);
        });

    eprintln!("\n[rustcrypto_chacha20poly1305_nonce_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_chacha20poly1305_nonce_ct");

    match &outcome {
        Outcome::Pass { leak_probability, .. } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail { leak_probability, exploitability, effect, .. } => {
            panic!(
                "ChaCha20-Poly1305 nonce should not affect timing (got leak_probability={:.1}%, {:?}, effect={:?})",
                leak_probability * 100.0, exploitability, effect
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

    if let Some(exp) = get_exploitability(&outcome) {
        assert!(
            matches!(exp, Exploitability::SharedHardwareOnly | Exploitability::Http2Multiplexing),
            "ChaCha20-Poly1305 nonce independence should have low exploitability (got {:?})",
            exp
        );
    }
}

/// Compare all-zeros vs all-ones plaintext for ChaCha20-Poly1305
#[test]
fn rustcrypto_chacha20poly1305_hamming() {
    let key_bytes: [u8; 32] = [0x42; 32];
    let cipher = ChaCha20Poly1305::new(&key_bytes.into());
    let nonce_counter = std::sync::atomic::AtomicU64::new(0);

    let inputs = InputPair::new(|| [0x00u8; 64], || [0xFFu8; 64]);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |plaintext| {
            let nonce_value = nonce_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut nonce_bytes = [0u8; 12];
            nonce_bytes[..8].copy_from_slice(&nonce_value.to_le_bytes());
            let nonce = Nonce::from_slice(&nonce_bytes);

            let ct = cipher.encrypt(nonce, plaintext.as_ref()).unwrap();
            std::hint::black_box(ct[0]);
        });

    eprintln!("\n[rustcrypto_chacha20poly1305_hamming]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_chacha20poly1305_hamming");

    match &outcome {
        Outcome::Pass { leak_probability, .. } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail { leak_probability, exploitability, .. } => {
            panic!(
                "ChaCha20-Poly1305 Hamming weight should be constant-time (got leak_probability={:.1}%, {:?})",
                leak_probability * 100.0, exploitability
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

    if let Some(exp) = get_exploitability(&outcome) {
        assert!(
            matches!(exp, Exploitability::SharedHardwareOnly | Exploitability::Http2Multiplexing),
            "ChaCha20-Poly1305 Hamming weight should not affect timing (got {:?})",
            exp
        );
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn get_exploitability(outcome: &Outcome) -> Option<Exploitability> {
    match outcome {
        Outcome::Fail { exploitability, .. } => Some(*exploitability),
        _ => None,
    }
}
