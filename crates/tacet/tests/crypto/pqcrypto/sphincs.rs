//! crypto::pqcrypto::sphincs
//!
//! SPHINCS+-SHA2-128f timing tests using the pqcrypto crate.
//! Crate: pqcrypto-sphincsplus
//! Family: Hash-based Signatures
//! Expected: All Pass (constant-time hash-based implementation)
//!
//! Note: SPHINCS+ is hash-based and inherently slower but designed to be constant-time.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use pqcrypto_sphincsplus::sphincssha2128fsimple;
use pqcrypto_traits::sign::DetachedSignature as _;
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

// ============================================================================
// SPHINCS+ Tests
// ============================================================================

/// SPHINCS+-SHA2-128f signing timing
///
/// Note: SPHINCS+ is hash-based and inherently slower but designed to be constant-time
#[test]
#[ignore] // SPHINCS+ is very slow, run with --ignored
fn pqcrypto_sphincs_sha2_128f_sign_ct() {
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

    eprintln!("\n[pqcrypto_sphincs_sha2_128f_sign_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "pqcrypto_sphincs_sha2_128f_sign_ct");

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
fn pqcrypto_sphincs_sha2_128f_verify_ct() {
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

    eprintln!("\n[pqcrypto_sphincs_sha2_128f_verify_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "pqcrypto_sphincs_sha2_128f_verify_ct");

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
