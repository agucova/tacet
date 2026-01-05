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
use timing_oracle::helpers::InputPair;
use timing_oracle::TimingOracle;

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
    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::balanced()
        .samples(10_000)
        .alpha(0.01)
        .test(inputs, |_| {
            let (pk, sk) = kyber768::keypair();
            std::hint::black_box(pk.as_bytes()[0] ^ sk.as_bytes()[0]);
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[kyber768_keypair_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // Key generation should have consistent timing (both branches do the same thing)
    assert!(
        result.ci_gate.passed,
        "Kyber-768 keypair generation should have consistent timing (got leak_probability={:.3})",
        result.leak_probability
    );
}

/// Kyber-768 encapsulation should be constant-time with respect to the public key
#[test]
fn kyber768_encapsulate_constant_time() {
    const SAMPLES: usize = 10_000;

    // Generate a fixed key pair
    let (pk_fixed, _sk_fixed) = kyber768::keypair();

    // Generate random key pairs for comparison
    let random_keypairs: Vec<_> = (0..SAMPLES).map(|_| kyber768::keypair()).collect();

    let idx = std::cell::Cell::new(0usize);

    // Use index-based approach since pqcrypto types don't implement Hash
    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .test(inputs, |which| {
            let pk = if *which == 0 {
                &pk_fixed
            } else {
                let i = idx.get();
                idx.set((i + 1) % SAMPLES);
                &random_keypairs[i].0
            };
            let (ss, ct) = kyber768::encapsulate(pk);
            std::hint::black_box(ss.as_bytes()[0] ^ ct.as_bytes()[0]);
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[kyber768_encapsulate_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed,
        "Kyber-768 encapsulation should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
}

/// Kyber-768 decapsulation should be constant-time
///
/// Decapsulation is the most sensitive operation
#[test]
fn kyber768_decapsulate_constant_time() {
    const SAMPLES: usize = 10_000;

    // Generate key pair
    let (pk, sk) = kyber768::keypair();

    // Pre-generate ciphertexts
    let (_, fixed_ct) = kyber768::encapsulate(&pk);

    let random_cts: Vec<_> = (0..SAMPLES)
        .map(|_| {
            let (_, ct) = kyber768::encapsulate(&pk);
            ct
        })
        .collect();

    let idx = std::cell::Cell::new(0usize);

    // Use index-based approach since pqcrypto types don't implement Hash
    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .test(inputs, |which| {
            let ct = if *which == 0 {
                &fixed_ct
            } else {
                let i = idx.get();
                idx.set((i + 1) % SAMPLES);
                &random_cts[i]
            };
            let ss = kyber768::decapsulate(ct, &sk);
            std::hint::black_box(ss.as_bytes()[0]);
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[kyber768_decapsulate_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed,
        "Kyber-768 decapsulation should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
}

// ============================================================================
// ML-DSA (Dilithium) Tests
// ============================================================================

/// Dilithium3 key generation timing
#[test]
fn dilithium3_keypair_constant_time() {
    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::balanced()
        .samples(5_000)
        .alpha(0.01)
        .test(inputs, |_| {
            let (pk, sk) = dilithium3::keypair();
            std::hint::black_box(pk.as_bytes()[0] ^ sk.as_bytes()[0]);
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[dilithium3_keypair_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed,
        "Dilithium3 keypair should have consistent timing (got leak_probability={:.3})",
        result.leak_probability
    );
}

/// Dilithium3 signing should be constant-time with respect to message content
#[test]
fn dilithium3_sign_constant_time() {
    const SAMPLES: usize = 5_000;

    let (_pk, sk) = dilithium3::keypair();

    let fixed_message: [u8; 64] = [0x42; 64];
    let inputs = InputPair::new(|| fixed_message, rand_bytes_64);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .test(inputs, |msg| {
            let sig = dilithium3::detached_sign(msg, &sk);
            std::hint::black_box(sig.as_bytes()[0]);
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[dilithium3_sign_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed,
        "Dilithium3 signing should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
}

/// Dilithium3 verification should be constant-time
#[test]
fn dilithium3_verify_constant_time() {
    const SAMPLES: usize = 5_000;

    let (pk, sk) = dilithium3::keypair();

    // Pre-generate signatures
    let fixed_message = [0x42u8; 64];
    let fixed_sig = dilithium3::detached_sign(&fixed_message, &sk);

    let random_msgs: Vec<[u8; 64]> = (0..SAMPLES).map(|_| rand_bytes_64()).collect();
    let random_sigs: Vec<_> = random_msgs
        .iter()
        .map(|msg| dilithium3::detached_sign(msg, &sk))
        .collect();

    let idx = std::cell::Cell::new(0usize);

    // Use index-based approach since pqcrypto types don't implement Hash
    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .test(inputs, |which| {
            let (msg, sig) = if *which == 0 {
                (&fixed_message[..], &fixed_sig)
            } else {
                let i = idx.get();
                idx.set((i + 1) % SAMPLES);
                (&random_msgs[i][..], &random_sigs[i])
            };
            let result = dilithium3::verify_detached_signature(sig, msg, &pk);
            std::hint::black_box(result.is_ok());
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[dilithium3_verify_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed,
        "Dilithium3 verification should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
}

// ============================================================================
// Falcon Tests
// ============================================================================

/// Falcon-512 signing should be constant-time
#[test]
fn falcon512_sign_constant_time() {
    const SAMPLES: usize = 3_000; // Falcon is slower

    let (_pk, sk) = falcon512::keypair();

    let fixed_message: [u8; 64] = [0x42; 64];
    let inputs = InputPair::new(|| fixed_message, rand_bytes_64);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .test(inputs, |msg| {
            let sig = falcon512::detached_sign(msg, &sk);
            std::hint::black_box(sig.as_bytes()[0]);
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[falcon512_sign_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // Note: Falcon uses floating-point internally and may have timing variations
    // This test documents the current behavior
    eprintln!("Note: Falcon uses floating-point which may cause platform-dependent timing");
}

/// Falcon-512 verification should be constant-time
#[test]
fn falcon512_verify_constant_time() {
    const SAMPLES: usize = 3_000;

    let (pk, sk) = falcon512::keypair();

    let fixed_message = [0x42u8; 64];
    let fixed_sig = falcon512::detached_sign(&fixed_message, &sk);

    let random_msgs: Vec<[u8; 64]> = (0..SAMPLES).map(|_| rand_bytes_64()).collect();
    let random_sigs: Vec<_> = random_msgs
        .iter()
        .map(|msg| falcon512::detached_sign(msg, &sk))
        .collect();

    let idx = std::cell::Cell::new(0usize);

    // Use index-based approach since pqcrypto types don't implement Hash
    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .test(inputs, |which| {
            let (msg, sig) = if *which == 0 {
                (&fixed_message[..], &fixed_sig)
            } else {
                let i = idx.get();
                idx.set((i + 1) % SAMPLES);
                (&random_msgs[i][..], &random_sigs[i])
            };
            let result = falcon512::verify_detached_signature(sig, msg, &pk);
            std::hint::black_box(result.is_ok());
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[falcon512_verify_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));
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
    const SAMPLES: usize = 500; // SPHINCS+ is VERY slow

    let (_pk, sk) = sphincssha2128fsimple::keypair();

    let fixed_message: [u8; 32] = [0x42; 32];
    let inputs = InputPair::new(|| fixed_message, rand_bytes_32);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .test(inputs, |msg| {
            let sig = sphincssha2128fsimple::detached_sign(msg, &sk);
            std::hint::black_box(sig.as_bytes()[0]);
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[sphincs_sha2_128f_sign_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // SPHINCS+ is hash-based and should be constant-time
    assert!(
        result.ci_gate.passed,
        "SPHINCS+ signing should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
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

    let idx = std::cell::Cell::new(0usize);

    // Use index-based approach since pqcrypto types don't implement Hash
    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .test(inputs, |which| {
            let (msg, sig) = if *which == 0 {
                (&fixed_message[..], &fixed_sig)
            } else {
                let i = idx.get();
                idx.set((i + 1) % SAMPLES);
                (&random_msgs[i][..], &random_sigs[i])
            };
            let result = sphincssha2128fsimple::verify_detached_signature(sig, msg, &pk);
            std::hint::black_box(result.is_ok());
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[sphincs_sha2_128f_verify_constant_time]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.ci_gate.passed,
        "SPHINCS+ verification should be constant-time (got leak_probability={:.3})",
        result.leak_probability
    );
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

    let inputs = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .test(inputs, |which| {
            if *which == 0 {
                let i = idx1.get();
                idx1.set((i + 1) % SAMPLES);
                let ss = kyber768::decapsulate(&batch1[i], &sk);
                std::hint::black_box(ss.as_bytes()[0]);
            } else {
                let i = idx2.get();
                idx2.set((i + 1) % SAMPLES);
                let ss = kyber768::decapsulate(&batch2[i], &sk);
                std::hint::black_box(ss.as_bytes()[0]);
            }
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[kyber768_ciphertext_independence]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // Both branches use random ciphertexts, should have consistent timing
    assert!(
        result.ci_gate.passed,
        "Kyber decapsulation timing should be independent of ciphertext (got leak_probability={:.3})",
        result.leak_probability
    );
}

/// Dilithium signing with different message patterns
#[test]
fn dilithium3_message_hamming_weight() {
    const SAMPLES: usize = 5_000;

    let (_, sk) = dilithium3::keypair();

    let inputs = InputPair::new(|| [0x00u8; 64], || [0xFFu8; 64]);

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .test(inputs, |msg| {
            let sig = dilithium3::detached_sign(msg, &sk);
            std::hint::black_box(sig.as_bytes()[0]);
        });

    let result = outcome.unwrap_completed();

    eprintln!("\n[dilithium3_message_hamming_weight]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        matches!(
            result.exploitability,
            timing_oracle::Exploitability::Negligible | timing_oracle::Exploitability::PossibleLAN
        ),
        "Dilithium3 message Hamming weight should not affect timing (got {:?})",
        result.exploitability
    );
}
