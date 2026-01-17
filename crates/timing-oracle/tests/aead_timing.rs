//! AEAD (Authenticated Encryption with Associated Data) timing tests
//!
//! Tests AEAD constructions for timing side channels using DudeCT's two-class pattern:
//! - Class 0: Fixed plaintext/nonce
//! - Class 1: Random plaintext/nonce
//!
//! AEAD is critical for TLS, messaging, and file encryption.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use ring::aead::{self, LessSafeKey, UnboundKey, AES_256_GCM, CHACHA20_POLY1305};
use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{skip_if_unreliable, AttackerModel, Exploitability, Outcome, TimingOracle};

#[allow(dead_code)]
fn rand_bytes_12() -> [u8; 12] {
    let mut arr = [0u8; 12];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

#[allow(dead_code)]
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
// ChaCha20-Poly1305 Tests (RustCrypto)
// ============================================================================

/// ChaCha20-Poly1305 encryption should be constant-time
///
/// ChaCha20 is designed to be constant-time, Poly1305 MAC as well
#[test]
fn chacha20poly1305_encrypt_constant_time() {
    // Fixed key
    let key_bytes: [u8; 32] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
        0x1e, 0x1f,
    ];
    let cipher = ChaCha20Poly1305::new(&key_bytes.into());

    // Use atomic counter to generate unique nonces (avoid nonce reuse)
    let nonce_counter = std::sync::atomic::AtomicU64::new(0);

    // Non-pathological fixed plaintext
    let fixed_plaintext: [u8; 64] = [0x42; 64];
    let inputs = InputPair::new(|| fixed_plaintext, rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |plaintext| {
            // Generate unique nonce for each encryption
            let nonce_value = nonce_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut nonce_bytes = [0u8; 12];
            nonce_bytes[..8].copy_from_slice(&nonce_value.to_le_bytes());
            let nonce = Nonce::from_slice(&nonce_bytes);

            let ciphertext = cipher.encrypt(nonce, plaintext.as_ref()).unwrap();
            std::hint::black_box(ciphertext[0]);
        });

    eprintln!("\n[chacha20poly1305_encrypt_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "chacha20poly1305_encrypt_constant_time");

    match &outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
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

    // Check exploitability and quality if conclusive
    if let Some(exp) = get_exploitability(&outcome) {
        assert!(
            matches!(
                exp,
                Exploitability::Negligible | Exploitability::PossibleLAN
            ),
            "ChaCha20-Poly1305 should have low exploitability (got {:?})",
            exp
        );
    }
}

/// ChaCha20-Poly1305 decryption should be constant-time
#[test]
fn chacha20poly1305_decrypt_constant_time() {
    const SAMPLES: usize = 30_000;

    let key_bytes: [u8; 32] = [0x5a; 32];
    let cipher = ChaCha20Poly1305::new(&key_bytes.into());
    let nonce = Nonce::from_slice(&[0u8; 12]);

    // Pre-generate ciphertexts
    let fixed_plaintext = [0x42u8; 64];
    let fixed_ciphertext = cipher.encrypt(nonce, fixed_plaintext.as_ref()).unwrap();

    // Generate random ciphertexts
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

    eprintln!("\n[chacha20poly1305_decrypt_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "chacha20poly1305_decrypt_constant_time");

    match &outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
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
            matches!(
                exp,
                Exploitability::Negligible | Exploitability::PossibleLAN
            ),
            "ChaCha20-Poly1305 decryption should have low exploitability (got {:?})",
            exp
        );
    }
}

/// ChaCha20-Poly1305 with varying nonces should be constant-time
#[test]
fn chacha20poly1305_nonce_independence() {
    let key_bytes: [u8; 32] = [0x73; 32];
    let cipher = ChaCha20Poly1305::new(&key_bytes.into());

    let plaintext = [0x42u8; 64];

    // Nonce A (all zeros) vs Nonce B (all ones)
    // Both deterministic to avoid cache locality differences
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

    eprintln!("\n[chacha20poly1305_nonce_independence]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "chacha20poly1305_nonce_independence");

    match &outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            effect,
            ..
        } => {
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
            matches!(
                exp,
                Exploitability::Negligible | Exploitability::PossibleLAN
            ),
            "ChaCha20-Poly1305 nonce independence should have low exploitability (got {:?})",
            exp
        );
    }
}

// ============================================================================
// AES-256-GCM Tests (ring)
// ============================================================================

/// AES-256-GCM encryption should be constant-time
///
/// ring uses hardware AES-NI when available
#[test]
fn aes_256_gcm_encrypt_constant_time() {
    let key_bytes: [u8; 32] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
        0x1e, 0x1f,
    ];

    let unbound_key = UnboundKey::new(&AES_256_GCM, &key_bytes).unwrap();
    let key = LessSafeKey::new(unbound_key);

    // Fixed nonce (12 bytes)
    let nonce_bytes: [u8; 12] = [0u8; 12];

    // Non-pathological fixed plaintext
    let fixed_plaintext: [u8; 64] = [0x42; 64];
    let inputs = InputPair::new(|| fixed_plaintext, rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |plaintext| {
            let nonce = aead::Nonce::assume_unique_for_key(nonce_bytes);
            let mut in_out = plaintext.to_vec();
            let tag = key
                .seal_in_place_separate_tag(nonce, aead::Aad::empty(), &mut in_out)
                .unwrap();
            std::hint::black_box(tag.as_ref()[0]);
        });

    eprintln!("\n[aes_256_gcm_encrypt_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "aes_256_gcm_encrypt_constant_time");

    match &outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            panic!(
                "AES-256-GCM encryption should be constant-time (got leak_probability={:.1}%, {:?})",
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
            matches!(
                exp,
                Exploitability::Negligible | Exploitability::PossibleLAN
            ),
            "AES-256-GCM encryption should have low exploitability (got {:?})",
            exp
        );
    }
}

/// AES-256-GCM decryption should be constant-time
#[test]
fn aes_256_gcm_decrypt_constant_time() {
    const SAMPLES: usize = 30_000;

    let key_bytes: [u8; 32] = [0x5a; 32];
    let unbound_key = UnboundKey::new(&AES_256_GCM, &key_bytes).unwrap();
    let key = LessSafeKey::new(unbound_key);

    let nonce_bytes: [u8; 12] = [0u8; 12];

    // Pre-generate ciphertexts with tags
    let fixed_plaintext = [0x42u8; 64];
    let mut fixed_ct = fixed_plaintext.to_vec();
    let fixed_nonce = aead::Nonce::assume_unique_for_key(nonce_bytes);
    key.seal_in_place_append_tag(fixed_nonce, aead::Aad::empty(), &mut fixed_ct)
        .unwrap();

    let random_cts: Vec<Vec<u8>> = (0..SAMPLES)
        .map(|_| {
            let pt = rand_bytes_64();
            let mut ct = pt.to_vec();
            let nonce = aead::Nonce::assume_unique_for_key(nonce_bytes);
            key.seal_in_place_append_tag(nonce, aead::Aad::empty(), &mut ct)
                .unwrap();
            ct
        })
        .collect();

    let idx = std::cell::Cell::new(0usize);
    let fixed_ct_clone = fixed_ct.clone();
    let inputs = InputPair::new(
        move || fixed_ct_clone.clone(),
        move || {
            let i = idx.get();
            idx.set((i + 1) % SAMPLES);
            random_cts[i].clone()
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |ct| {
            let nonce = aead::Nonce::assume_unique_for_key(nonce_bytes);
            let mut ct_mut = ct.clone();
            let pt = key
                .open_in_place(nonce, aead::Aad::empty(), &mut ct_mut)
                .unwrap();
            std::hint::black_box(pt[0]);
        });

    eprintln!("\n[aes_256_gcm_decrypt_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "aes_256_gcm_decrypt_constant_time");

    match &outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            panic!(
                "AES-256-GCM decryption should be constant-time (got leak_probability={:.1}%, {:?})",
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
            matches!(
                exp,
                Exploitability::Negligible | Exploitability::PossibleLAN
            ),
            "AES-256-GCM decryption should have low exploitability (got {:?})",
            exp
        );
    }
}

// ============================================================================
// ChaCha20-Poly1305 via ring
// ============================================================================

/// ChaCha20-Poly1305 via ring should be constant-time
#[test]
fn ring_chacha20poly1305_constant_time() {
    let key_bytes: [u8; 32] = [0x73; 32];
    let unbound_key = UnboundKey::new(&CHACHA20_POLY1305, &key_bytes).unwrap();
    let key = LessSafeKey::new(unbound_key);

    let nonce_bytes: [u8; 12] = [0u8; 12];

    let fixed_plaintext: [u8; 64] = [0x42; 64];
    let inputs = InputPair::new(|| fixed_plaintext, rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |plaintext| {
            let nonce = aead::Nonce::assume_unique_for_key(nonce_bytes);
            let mut in_out = plaintext.to_vec();
            let tag = key
                .seal_in_place_separate_tag(nonce, aead::Aad::empty(), &mut in_out)
                .unwrap();
            std::hint::black_box(tag.as_ref()[0]);
        });

    eprintln!("\n[ring_chacha20poly1305_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "ring_chacha20poly1305_constant_time");

    match &outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            panic!(
                "ring ChaCha20-Poly1305 should be constant-time (got leak_probability={:.1}%, {:?})",
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
            matches!(
                exp,
                Exploitability::Negligible | Exploitability::PossibleLAN
            ),
            "ring ChaCha20-Poly1305 should have low exploitability (got {:?})",
            exp
        );
    }
}

// ============================================================================
// Comparative Tests
// ============================================================================

/// Compare all-zeros vs all-ones plaintext for ChaCha20-Poly1305
#[test]
fn chacha20poly1305_hamming_weight_independence() {
    let key_bytes: [u8; 32] = [0x42; 32];
    let cipher = ChaCha20Poly1305::new(&key_bytes.into());

    // Use atomic counter to generate unique nonces (avoid nonce reuse)
    let nonce_counter = std::sync::atomic::AtomicU64::new(0);

    let inputs = InputPair::new(|| [0x00u8; 64], || [0xFFu8; 64]);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |plaintext| {
            // Generate unique nonce for each encryption
            let nonce_value = nonce_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut nonce_bytes = [0u8; 12];
            nonce_bytes[..8].copy_from_slice(&nonce_value.to_le_bytes());
            let nonce = Nonce::from_slice(&nonce_bytes);

            let ct = cipher.encrypt(nonce, plaintext.as_ref()).unwrap();
            std::hint::black_box(ct[0]);
        });

    eprintln!("\n[chacha20poly1305_hamming_weight_independence]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "chacha20poly1305_hamming_weight_independence");

    match &outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
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
            matches!(
                exp,
                Exploitability::Negligible | Exploitability::PossibleLAN
            ),
            "ChaCha20-Poly1305 Hamming weight should not affect timing (got {:?})",
            exp
        );
    }
}

/// Compare all-zeros vs all-ones plaintext for AES-GCM
#[test]
fn aes_gcm_hamming_weight_independence() {
    let key_bytes: [u8; 32] = [0x42; 32];
    let unbound_key = UnboundKey::new(&AES_256_GCM, &key_bytes).unwrap();
    let key = LessSafeKey::new(unbound_key);

    // Use atomic counter to generate unique nonces (avoid nonce reuse)
    let nonce_counter = std::sync::atomic::AtomicU64::new(0);

    let inputs = InputPair::new(|| [0x00u8; 64], || [0xFFu8; 64]);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |plaintext| {
            // Generate unique nonce for each encryption
            let nonce_value = nonce_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut nonce_bytes = [0u8; 12];
            nonce_bytes[..8].copy_from_slice(&nonce_value.to_le_bytes());
            let nonce = aead::Nonce::assume_unique_for_key(nonce_bytes);

            let mut in_out = plaintext.to_vec();
            let tag = key
                .seal_in_place_separate_tag(nonce, aead::Aad::empty(), &mut in_out)
                .unwrap();
            std::hint::black_box(tag.as_ref()[0]);
        });

    eprintln!("\n[aes_gcm_hamming_weight_independence]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "aes_gcm_hamming_weight_independence");

    match &outcome {
        Outcome::Pass {
            leak_probability, ..
        } => {
            eprintln!("Test passed: P(leak)={:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail {
            leak_probability,
            exploitability,
            ..
        } => {
            panic!(
                "AES-GCM Hamming weight should be constant-time (got leak_probability={:.1}%, {:?})",
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
            matches!(
                exp,
                Exploitability::Negligible | Exploitability::PossibleLAN
            ),
            "AES-GCM Hamming weight should not affect timing (got {:?})",
            exp
        );
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Extract exploitability from an outcome (only available for Fail variant)
fn get_exploitability(outcome: &Outcome) -> Option<Exploitability> {
    match outcome {
        Outcome::Fail { exploitability, .. } => Some(*exploitability),
        _ => None,
    }
}
