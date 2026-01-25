//! crypto::ring::chacha20poly1305
//!
//! ChaCha20-Poly1305 AEAD timing tests using the ring crate.
//! Crate: ring
//! Family: AEAD
//! Expected: All Pass (constant-time implementation)
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use ring::aead::{self, LessSafeKey, UnboundKey, CHACHA20_POLY1305};
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

/// ChaCha20-Poly1305 via ring should be constant-time
#[test]
fn ring_chacha20poly1305_encrypt_ct() {
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

    eprintln!("\n[ring_chacha20poly1305_encrypt_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "ring_chacha20poly1305_encrypt_ct");

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
                Exploitability::SharedHardwareOnly | Exploitability::Http2Multiplexing
            ),
            "ring ChaCha20-Poly1305 should have low exploitability (got {:?})",
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
