//! crypto::pqcrypto::kyber
//!
//! ML-KEM (Kyber-768) timing tests using the pqcrypto crate.
//! Crate: pqcrypto-kyber
//! Family: KEM (Key Encapsulation Mechanism)
//! Expected: All Pass (constant-time PQClean implementation)
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use pqcrypto_kyber::kyber768;
use pqcrypto_traits::kem::{Ciphertext as _, PublicKey as _, SecretKey as _, SharedSecret as _};
use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{skip_if_unreliable, AttackerModel, Outcome, TimingOracle};

// ============================================================================
// ML-KEM (Kyber) Tests
// ============================================================================

/// Kyber-768 key generation should be constant-time
#[test]
fn pqcrypto_kyber768_keypair_ct() {
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

    eprintln!("\n[pqcrypto_kyber768_keypair_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "pqcrypto_kyber768_keypair_ct");

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
fn pqcrypto_kyber768_encapsulate_ct() {
    const SAMPLES: usize = 10_000;

    // Generate TWO batches of random keypairs to compare timing independence
    // Using "same key repeated" vs "different keys" would test cache effects, not constant-time behavior
    // Instead, we use two independent batches of random keys
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

    eprintln!("\n[pqcrypto_kyber768_encapsulate_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "pqcrypto_kyber768_encapsulate_ct");

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
fn pqcrypto_kyber768_decapsulate_ct() {
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

    eprintln!("\n[pqcrypto_kyber768_decapsulate_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "pqcrypto_kyber768_decapsulate_ct");

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

/// Kyber decapsulation with different ciphertext batches
#[test]
fn pqcrypto_kyber768_ciphertext_independence() {
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

    eprintln!("\n[pqcrypto_kyber768_ciphertext_independence]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "pqcrypto_kyber768_ciphertext_independence");

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
