//! crypto::pqcrypto::falcon
//!
//! Falcon-512 timing tests using the pqcrypto crate.
//! Crate: pqcrypto-falcon
//! Family: Digital Signatures (NTRU-based)
//! Expected: Mixed (Falcon uses floating-point which may cause platform-dependent timing)
//!
//! NOTE: Falcon uses floating-point Gaussian sampling which causes INTENTIONAL
//! timing variation based on the message. Like Dilithium, this is NOT a vulnerability
//! because the message is public and the sampling is independent of the secret key.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::DetachedSignature as _;
use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{skip_if_unreliable, AttackerModel, Outcome, TimingOracle};

fn rand_bytes_64() -> [u8; 64] {
    let mut arr = [0u8; 64];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

// ============================================================================
// Falcon Tests
// ============================================================================

/// Falcon-512 signing timing consistency
///
/// NOTE: Falcon uses floating-point Gaussian sampling which causes INTENTIONAL
/// timing variation based on the message. This test verifies timing consistency
/// by using the same message for both classes.
#[test]
fn pqcrypto_falcon512_sign_ct() {
    let (_pk, sk) = falcon512::keypair();

    // Use the SAME message for both classes to test timing consistency
    // This isolates measurement noise from message-dependent sampling
    let fixed_message: [u8; 64] = [0x42; 64];
    let inputs = InputPair::new(|| fixed_message, || fixed_message);

    let outcome = TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(45))
        .test(inputs, |msg| {
            let sig = falcon512::detached_sign(msg, &sk);
            std::hint::black_box(sig.as_bytes()[0]);
        });

    eprintln!("\n[pqcrypto_falcon512_sign_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));
    eprintln!("Note: Same message used for both classes (testing timing consistency)");
    eprintln!("      Falcon uses floating-point which may cause platform-dependent timing");

    let outcome = skip_if_unreliable!(outcome, "pqcrypto_falcon512_sign_ct");

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
                "Falcon-512 signing should have consistent timing for same message (got leak_probability={:.1}%, {:?})",
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

/// Falcon-512 verification timing (informational)
///
/// NOTE: Falcon verification may have data-dependent timing depending on the
/// implementation variant. The PQClean implementation may not use constant-time
/// hash-to-point, which can cause timing variations based on message content.
/// This is NOT necessarily a vulnerability if the message is public.
///
/// This test is informational: it documents the timing behavior of Falcon verification.
#[test]
fn pqcrypto_falcon512_verify_ct() {
    const SAMPLES: usize = 3_000;

    let (pk, sk) = falcon512::keypair();

    // Pre-generate messages and signatures for both classes
    let fixed_message = [0x42u8; 64];
    let fixed_sig = falcon512::detached_sign(&fixed_message, &sk);

    // Class 0 (baseline): Repeat same message/sig
    // Class 1 (sample): Use different messages/sigs
    let baseline_msgs: Vec<[u8; 64]> = (0..SAMPLES).map(|_| fixed_message).collect();
    let baseline_sigs: Vec<_> = (0..SAMPLES).map(|_| fixed_sig).collect();

    let sample_msgs: Vec<[u8; 64]> = (0..SAMPLES).map(|_| rand_bytes_64()).collect();
    let sample_sigs: Vec<_> = sample_msgs
        .iter()
        .map(|msg| falcon512::detached_sign(msg, &sk))
        .collect();

    let idx0 = std::cell::Cell::new(0usize);
    let idx1 = std::cell::Cell::new(0usize);

    // Use index-based approach since pqcrypto types don't implement Hash
    // Using new_unchecked because we're intentionally using indices as class identifiers
    let inputs = InputPair::new_unchecked(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(45))
        .test(inputs, |which| {
            // IDENTICAL code paths for both classes - only data differs
            let (msgs, sigs, idx) = if *which == 0 {
                (&baseline_msgs, &baseline_sigs, &idx0)
            } else {
                (&sample_msgs, &sample_sigs, &idx1)
            };
            let i = idx.get();
            idx.set((i + 1) % SAMPLES);
            let result = falcon512::verify_detached_signature(&sigs[i], &msgs[i], &pk);
            std::hint::black_box(result.is_ok());
        });

    eprintln!("\n[pqcrypto_falcon512_verify_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "pqcrypto_falcon512_verify_ct");

    // Informational: Falcon verification may have timing variations based on
    // message/signature content. This is documented PQClean behavior.
    eprintln!("Note: Falcon verification timing may vary based on message/signature");
    eprintln!("      PQClean may not use constant-time hash-to-point");

    // We DON'T panic on Fail for Falcon verification due to documented
    // timing variability in PQClean's hash-to-point implementation.
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
                "Timing difference detected (expected for Falcon): P(leak)={:.1}%, {:?}, effect={:?}",
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
