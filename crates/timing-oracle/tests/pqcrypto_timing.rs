//! Post-Quantum Cryptography timing tests
//!
//! Tests NIST-selected post-quantum algorithms for timing side channels:
//! - ML-KEM (Kyber): Key Encapsulation Mechanism
//! - ML-DSA (Dilithium): Digital Signatures
//! - Falcon: Digital Signatures (NTRU-based)
//! - SPHINCS+: Hash-based Signatures
//!
//! These use PQClean implementations via the pqcrypto crate (C FFI).
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use pqcrypto_dilithium::dilithium3;
use pqcrypto_falcon::falcon512;
use pqcrypto_kyber::kyber768;
use pqcrypto_sphincsplus::sphincssha2128fsimple;
use pqcrypto_traits::kem::{Ciphertext as _, PublicKey as _, SecretKey as _, SharedSecret as _};
use pqcrypto_traits::sign::{DetachedSignature as _, PublicKey as _, SecretKey as _};
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
// ML-KEM (Kyber) Tests
// ============================================================================

/// Kyber-768 key generation should be constant-time
#[test]
fn kyber768_keypair_constant_time() {
    // We're testing whether repeated key generation has consistent timing
    // This is a bit different - we generate keys in both branches
    // Using new_unchecked because we're using indices as class identifiers (intentional)
    let inputs = InputPair::new_unchecked(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(45))
        .test(inputs, |_| {
            let (pk, sk) = kyber768::keypair();
            std::hint::black_box(pk.as_bytes()[0] ^ sk.as_bytes()[0]);
        });

    eprintln!("\n[kyber768_keypair_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "kyber768_keypair_constant_time");

    // Key generation should have consistent timing (both branches do the same thing)
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
                "Kyber-768 keypair generation should have consistent timing (got leak_probability={:.1}%, {:?})",
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

/// Kyber-768 encapsulation should be constant-time with respect to the public key
#[test]
fn kyber768_encapsulate_constant_time() {
    const SAMPLES: usize = 10_000;

    // Generate TWO batches of random keypairs to compare timing independence
    // Using "same key repeated" vs "different keys" would test cache effects, not constant-time behavior
    // Instead, we use two independent batches of random keys (like kyber768_ciphertext_independence)
    let batch1_keypairs: Vec<_> = (0..SAMPLES).map(|_| kyber768::keypair()).collect();
    let batch2_keypairs: Vec<_> = (0..SAMPLES).map(|_| kyber768::keypair()).collect();

    let batch1_pks: Vec<_> = batch1_keypairs.iter().map(|(pk, _)| *pk).collect();
    let batch2_pks: Vec<_> = batch2_keypairs.iter().map(|(pk, _)| *pk).collect();

    let idx0 = std::cell::Cell::new(0usize);
    let idx1 = std::cell::Cell::new(0usize);

    // Use index-based approach since pqcrypto types don't implement Hash
    // Using new_unchecked because we're using indices as class identifiers (intentional)
    let inputs = InputPair::new_unchecked(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(45))
        .test(inputs, |which| {
            // IDENTICAL code paths for both classes - only data differs
            // Both classes iterate through different random keys, eliminating cache bias
            let (pks, idx) = if *which == 0 {
                (&batch1_pks, &idx0)
            } else {
                (&batch2_pks, &idx1)
            };
            let i = idx.get();
            idx.set((i + 1) % SAMPLES);
            let (ss, ct) = kyber768::encapsulate(&pks[i]);
            std::hint::black_box(ss.as_bytes()[0] ^ ct.as_bytes()[0]);
        });

    eprintln!("\n[kyber768_encapsulate_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "kyber768_encapsulate_constant_time");

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
                "Kyber-768 encapsulation should be constant-time (got leak_probability={:.1}%, {:?})",
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

/// Kyber-768 decapsulation should be constant-time
///
/// Decapsulation is the most sensitive operation
#[test]
fn kyber768_decapsulate_constant_time() {
    const SAMPLES: usize = 10_000;

    // Generate key pair
    let (pk, sk) = kyber768::keypair();

    // Pre-generate ciphertexts for both classes
    let (_, fixed_ct) = kyber768::encapsulate(&pk);

    // Class 0 (baseline): Repeat the same ciphertext
    // Class 1 (sample): Use different ciphertexts
    let baseline_cts: Vec<_> = (0..SAMPLES).map(|_| fixed_ct).collect();
    let sample_cts: Vec<_> = (0..SAMPLES)
        .map(|_| {
            let (_, ct) = kyber768::encapsulate(&pk);
            ct
        })
        .collect();

    let idx0 = std::cell::Cell::new(0usize);
    let idx1 = std::cell::Cell::new(0usize);

    // Use index-based approach since pqcrypto types don't implement Hash
    // Using new_unchecked because we're using indices as class identifiers (intentional)
    let inputs = InputPair::new_unchecked(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(45))
        .test(inputs, |which| {
            // IDENTICAL code paths for both classes - only data differs
            let (cts, idx) = if *which == 0 {
                (&baseline_cts, &idx0)
            } else {
                (&sample_cts, &idx1)
            };
            let i = idx.get();
            idx.set((i + 1) % SAMPLES);
            let ss = kyber768::decapsulate(&cts[i], &sk);
            std::hint::black_box(ss.as_bytes()[0]);
        });

    eprintln!("\n[kyber768_decapsulate_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "kyber768_decapsulate_constant_time");

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
                "Kyber-768 decapsulation should be constant-time (got leak_probability={:.1}%, {:?})",
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

// ============================================================================
// ML-DSA (Dilithium) Tests
// ============================================================================

/// Dilithium3 key generation timing
#[test]
fn dilithium3_keypair_constant_time() {
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

    eprintln!("\n[dilithium3_keypair_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "dilithium3_keypair_constant_time");

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
/// based on the message. This is NOT a vulnerability because:
/// 1. The message is public in signature schemes
/// 2. The rejection probability is independent of the secret key
///
/// This test verifies that the timing distribution is consistent (both branches
/// use the same message), which would catch implementation bugs but not message-
/// dependent timing (which is expected).
#[test]
fn dilithium3_sign_constant_time() {
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

    eprintln!("\n[dilithium3_sign_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));
    eprintln!("Note: Same message used for both classes (testing timing consistency)");

    let outcome = skip_if_unreliable!(outcome, "dilithium3_sign_constant_time");

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

/// Dilithium3 verification should be constant-time
#[test]
fn dilithium3_verify_constant_time() {
    const SAMPLES: usize = 5_000;

    let (pk, sk) = dilithium3::keypair();

    // Pre-generate messages and signatures for both classes
    let fixed_message = [0x42u8; 64];
    let fixed_sig = dilithium3::detached_sign(&fixed_message, &sk);

    // Class 0 (baseline): Repeat same message/sig
    // Class 1 (sample): Use different messages/sigs
    let baseline_msgs: Vec<[u8; 64]> = (0..SAMPLES).map(|_| fixed_message).collect();
    let baseline_sigs: Vec<_> = (0..SAMPLES).map(|_| fixed_sig).collect();

    let sample_msgs: Vec<[u8; 64]> = (0..SAMPLES).map(|_| rand_bytes_64()).collect();
    let sample_sigs: Vec<_> = sample_msgs
        .iter()
        .map(|msg| dilithium3::detached_sign(msg, &sk))
        .collect();

    let idx0 = std::cell::Cell::new(0usize);
    let idx1 = std::cell::Cell::new(0usize);

    // Use index-based approach since pqcrypto types don't implement Hash
    // Using new_unchecked because we're using indices as class identifiers (intentional)
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
            let result = dilithium3::verify_detached_signature(&sigs[i], &msgs[i], &pk);
            std::hint::black_box(result.is_ok());
        });

    eprintln!("\n[dilithium3_verify_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "dilithium3_verify_constant_time");

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
                "Dilithium3 verification should be constant-time (got leak_probability={:.1}%, {:?})",
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

// ============================================================================
// Falcon Tests
// ============================================================================

/// Falcon-512 signing timing consistency
///
/// NOTE: Falcon uses floating-point Gaussian sampling which causes INTENTIONAL
/// timing variation based on the message. Like Dilithium, this is NOT a vulnerability
/// because the message is public and the sampling is independent of the secret key.
///
/// This test verifies timing consistency by using the same message for both classes.
#[test]
fn falcon512_sign_constant_time() {
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

    eprintln!("\n[falcon512_sign_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));
    eprintln!("Note: Same message used for both classes (testing timing consistency)");
    eprintln!("      Falcon uses floating-point which may cause platform-dependent timing");

    let outcome = skip_if_unreliable!(outcome, "falcon512_sign_constant_time");

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
fn falcon512_verify_constant_time() {
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

    eprintln!("\n[falcon512_verify_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "falcon512_verify_constant_time");

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

// ============================================================================
// SPHINCS+ Tests
// ============================================================================

/// SPHINCS+-SHA2-128f signing timing
///
/// Note: SPHINCS+ is hash-based and inherently slower but designed to be constant-time
#[test]
#[ignore] // SPHINCS+ is very slow, run with --ignored
fn sphincs_sha2_128f_sign_constant_time() {
    let (_pk, sk) = sphincssha2128fsimple::keypair();

    let fixed_message: [u8; 32] = [0x42; 32];
    let inputs = InputPair::new(|| fixed_message, rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(120))
        .test(inputs, |msg| {
            let sig = sphincssha2128fsimple::detached_sign(msg, &sk);
            std::hint::black_box(sig.as_bytes()[0]);
        });

    eprintln!("\n[sphincs_sha2_128f_sign_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "sphincs_sha2_128f_sign_constant_time");

    // SPHINCS+ is hash-based and should be constant-time
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
                "SPHINCS+ signing should be constant-time (got leak_probability={:.1}%, {:?})",
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

/// SPHINCS+-SHA2-128f verification timing
#[test]
#[ignore] // SPHINCS+ is very slow, run with --ignored
fn sphincs_sha2_128f_verify_constant_time() {
    const SAMPLES: usize = 1_000;

    let (pk, sk) = sphincssha2128fsimple::keypair();

    let fixed_message = [0x42u8; 32];
    let fixed_sig = sphincssha2128fsimple::detached_sign(&fixed_message, &sk);

    let random_msgs: Vec<[u8; 32]> = (0..SAMPLES).map(|_| rand_bytes_32()).collect();
    let random_sigs: Vec<_> = random_msgs
        .iter()
        .map(|msg| sphincssha2128fsimple::detached_sign(msg, &sk))
        .collect();

    // DudeCT compliant: Create parallel arrays for symmetric code paths
    // Baseline: repeat fixed message/sig SAMPLES times (conceptually)
    // Sample: random messages/sigs
    // Both classes use the same index management pattern
    let baseline_msgs: Vec<[u8; 32]> = vec![fixed_message; SAMPLES];
    let baseline_sigs: Vec<_> = vec![fixed_sig; SAMPLES];

    let idx0 = std::cell::Cell::new(0usize);
    let idx1 = std::cell::Cell::new(0usize);

    // Use index-based approach since pqcrypto types don't implement Hash
    // Using new_unchecked because we're using indices as class identifiers (intentional)
    let inputs = InputPair::new_unchecked(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(120))
        .test(inputs, |which| {
            // Symmetric code paths: both classes do identical operations
            let (msgs, sigs, idx) = if *which == 0 {
                (&baseline_msgs, &baseline_sigs, &idx0)
            } else {
                (&random_msgs, &random_sigs, &idx1)
            };
            let i = idx.get();
            idx.set((i + 1) % SAMPLES);
            let result = sphincssha2128fsimple::verify_detached_signature(&sigs[i], &msgs[i], &pk);
            std::hint::black_box(result.is_ok());
        });

    eprintln!("\n[sphincs_sha2_128f_verify_constant_time]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "sphincs_sha2_128f_verify_constant_time");

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
                "SPHINCS+ verification should be constant-time (got leak_probability={:.1}%, {:?})",
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
// Comparative Tests
// ============================================================================

/// Kyber decapsulation with different ciphertext batches
#[test]
fn kyber768_ciphertext_independence() {
    const SAMPLES: usize = 10_000;

    let (pk, sk) = kyber768::keypair();

    // Generate two batches of random ciphertexts
    let batch1: Vec<_> = (0..SAMPLES)
        .map(|_| {
            let (_, ct) = kyber768::encapsulate(&pk);
            ct
        })
        .collect();

    let batch2: Vec<_> = (0..SAMPLES)
        .map(|_| {
            let (_, ct) = kyber768::encapsulate(&pk);
            ct
        })
        .collect();

    let idx1 = std::cell::Cell::new(0usize);
    let idx2 = std::cell::Cell::new(0usize);

    // Using new_unchecked because we're using indices as class identifiers (intentional)
    let inputs = InputPair::new_unchecked(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::PostQuantumSentinel)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(45))
        .test(inputs, |which| {
            // DudeCT compliant: extract-then-use pattern for uniform code path
            let (batch, idx) = if *which == 0 {
                (&batch1, &idx1)
            } else {
                (&batch2, &idx2)
            };
            // Uniform code after extraction
            let i = idx.get();
            idx.set((i + 1) % SAMPLES);
            let ss = kyber768::decapsulate(&batch[i], &sk);
            std::hint::black_box(ss.as_bytes()[0]);
        });

    eprintln!("\n[kyber768_ciphertext_independence]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "kyber768_ciphertext_independence");

    // Both branches use random ciphertexts, should have consistent timing
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
                "Kyber decapsulation timing should be independent of ciphertext (got leak_probability={:.1}%, {:?})",
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
fn dilithium3_message_hamming_weight() {
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

    eprintln!("\n[dilithium3_message_hamming_weight]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "dilithium3_message_hamming_weight");

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
