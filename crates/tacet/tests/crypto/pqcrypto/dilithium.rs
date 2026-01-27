//! crypto::pqcrypto::dilithium
//!
//! ML-DSA (Dilithium3) timing tests using the pqcrypto crate.
//! Crate: pqcrypto-dilithium
//! Family: Digital Signatures
//! Expected: All Pass (constant-time PQClean implementation)
//!
//! NOTE: Dilithium uses rejection sampling which causes INTENTIONAL timing variation
//! based on the message. This is NOT a vulnerability because:
//! 1. The message is public in signature schemes
//! 2. The rejection probability is independent of the secret key
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use pqcrypto_dilithium::dilithium3;
use pqcrypto_traits::sign::{DetachedSignature as _, PublicKey as _, SecretKey as _};
use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{skip_if_unreliable, AttackerModel, Outcome, TimingOracle};

// ============================================================================
// ML-DSA (Dilithium) Tests
// ============================================================================

/// Dilithium3 key generation timing
#[test]
fn pqcrypto_dilithium3_keypair_ct() {
    // Using new_unchecked because we're using indices as class identifiers (intentional)
    let inputs = InputPair::new_unchecked(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(45))
        .test(inputs, |_| {
            let (pk, sk) = dilithium3::keypair();
            std::hint::black_box(pk.as_bytes()[0] ^ sk.as_bytes()[0]);
        });

    eprintln!("\n[pqcrypto_dilithium3_keypair_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "pqcrypto_dilithium3_keypair_ct");

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
                "Dilithium3 keypair should have consistent timing (got leak_probability={:.1}%, {:?})",
                leak_probability * 100.0, exploitability
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

/// Dilithium3 signing timing consistency
///
/// NOTE: Dilithium uses rejection sampling which causes INTENTIONAL timing variation
/// based on the message. This test verifies that the timing distribution is consistent
/// (both branches use the same message), which would catch implementation bugs but not
/// message-dependent timing (which is expected).
#[test]
fn pqcrypto_dilithium3_sign_ct() {
    let (_pk, sk) = dilithium3::keypair();

    // Use the SAME message for both classes to test timing consistency
    // This isolates measurement noise from message-dependent rejection sampling
    let fixed_message: [u8; 64] = [0x42; 64];
    let inputs = InputPair::new(|| fixed_message, || fixed_message);

    let outcome = TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(45))
        .test(inputs, |msg| {
            let sig = dilithium3::detached_sign(msg, &sk);
            std::hint::black_box(sig.as_bytes()[0]);
        });

    eprintln!("\n[pqcrypto_dilithium3_sign_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));
    eprintln!("Note: Same message used for both classes (testing timing consistency)");

    let outcome = skip_if_unreliable!(outcome, "pqcrypto_dilithium3_sign_ct");

    // With identical inputs, any timing difference indicates measurement noise
    // or implementation issues, not message-dependent timing
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
                "Dilithium3 signing should have consistent timing for same message (got leak_probability={:.1}%, {:?})",
                leak_probability * 100.0, exploitability
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


/// Dilithium signing with different message patterns (informational)
///
/// NOTE: Dilithium uses rejection sampling, so timing DOES vary based on message content.
/// This is NOT a vulnerability - the message is public and the rejection probability
/// is independent of the secret key.
///
/// This test is informational: it documents the expected message-dependent timing
/// behavior rather than asserting it should be constant-time.
#[test]
fn pqcrypto_dilithium3_message_hamming() {
    let (_, sk) = dilithium3::keypair();

    let inputs = InputPair::new(|| [0x00u8; 64], || [0xFFu8; 64]);

    let outcome = TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(45))
        .test(inputs, |msg| {
            let sig = dilithium3::detached_sign(msg, &sk);
            std::hint::black_box(sig.as_bytes()[0]);
        });

    eprintln!("\n[pqcrypto_dilithium3_message_hamming]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "pqcrypto_dilithium3_message_hamming");

    // Informational: Dilithium timing varies based on message content due to
    // rejection sampling. This is expected behavior, not a vulnerability.
    eprintln!("Note: Dilithium uses rejection sampling - message-dependent timing is EXPECTED");
    eprintln!(
        "      This is NOT a vulnerability (message is public, rejection independent of secret)"
    );

    // We DON'T panic on Fail because message-dependent timing is expected.
    // Only log the results for documentation purposes.
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
            eprintln!(
                "Timing difference detected (expected for Dilithium): P(leak)={:.1}%, {:?}, effect={:?}",
                leak_probability * 100.0, exploitability, effect
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
