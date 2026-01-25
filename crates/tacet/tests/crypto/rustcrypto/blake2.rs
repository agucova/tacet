//! crypto::rustcrypto::blake2
//!
//! BLAKE2 hash function timing tests using the RustCrypto implementation.
//! Crate: blake2
//! Family: Hash
//! Expected: All Pass (constant-time implementation)
//!
//! BLAKE2b is optimized for 64-bit platforms, BLAKE2s for 8-32 bit platforms.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use blake2::{Blake2b512, Blake2s256, Digest};
use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{skip_if_unreliable, AttackerModel, Exploitability, Outcome, TimingOracle};

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
// Core BLAKE2 Tests
// ============================================================================

/// BLAKE2b-512 should be constant-time
#[test]
fn rustcrypto_blake2b512_ct() {
    let fixed_input: [u8; 64] = [0x73; 64];
    let inputs = InputPair::new(|| fixed_input, rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            let hash = Blake2b512::digest(data);
            std::hint::black_box(hash[0]);
        });

    eprintln!("\n[rustcrypto_blake2b512_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_blake2b512_ct");

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
                "BLAKE2b-512 should be constant-time, got leak_probability={:.1}%, exploitability={:?}",
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
        &Outcome::Research(_) => {}
    }

    if let Some(Exploitability::StandardRemote) | Some(Exploitability::ObviousLeak) =
        get_exploitability(&outcome)
    {
        panic!(
            "Exploitability should be low, got: {:?}",
            get_exploitability(&outcome)
        );
    }
}

/// BLAKE2s-256 should be constant-time
#[test]
fn rustcrypto_blake2s256_ct() {
    let fixed_input: [u8; 32] = [0xb7; 32];
    let inputs = InputPair::new(|| fixed_input, rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            let hash = Blake2s256::digest(data);
            std::hint::black_box(hash[0]);
        });

    eprintln!("\n[rustcrypto_blake2s256_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_blake2s256_ct");

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
                "BLAKE2s-256 should be constant-time, got leak_probability={:.1}%, exploitability={:?}",
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
        &Outcome::Research(_) => {}
    }

    if let Some(Exploitability::StandardRemote) | Some(Exploitability::ObviousLeak) =
        get_exploitability(&outcome)
    {
        panic!(
            "Exploitability should be low, got: {:?}",
            get_exploitability(&outcome)
        );
    }
}

// ============================================================================
// Comparative Tests
// ============================================================================

/// Compare high vs low Hamming weight inputs for BLAKE2b
#[test]
fn rustcrypto_blake2b_hamming() {
    let inputs = InputPair::new(|| [0x00u8; 64], || [0xFFu8; 64]);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            let hash = Blake2b512::digest(data);
            std::hint::black_box(hash[0]);
        });

    eprintln!("\n[rustcrypto_blake2b_hamming]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_blake2b_hamming");

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
                "Hamming weight should not affect timing, got leak_probability={:.1}%, exploitability={:?}",
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
        &Outcome::Research(_) => {}
    }

    if let Some(Exploitability::StandardRemote) | Some(Exploitability::ObviousLeak) =
        get_exploitability(&outcome)
    {
        panic!(
            "Exploitability should be low, got: {:?}",
            get_exploitability(&outcome)
        );
    }
}

// ============================================================================
// Incremental Hashing Tests
// ============================================================================

/// Test BLAKE2b incremental hashing
#[test]
fn rustcrypto_blake2b_incremental_ct() {
    let fixed_chunks: ([u8; 64], [u8; 64]) = ([0x33; 64], [0x44; 64]);

    let inputs = InputPair::new(|| fixed_chunks, || (rand_bytes_64(), rand_bytes_64()));

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |(data1, data2)| {
            let mut hasher = Blake2b512::new();
            hasher.update(data1);
            hasher.update(data2);
            let hash = hasher.finalize();
            std::hint::black_box(hash[0]);
        });

    eprintln!("\n[rustcrypto_blake2b_incremental_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_blake2b_incremental_ct");

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
                "BLAKE2b incremental hashing should be constant-time, got leak_probability={:.1}%, exploitability={:?}",
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
        &Outcome::Research(_) => {}
    }

    if let Some(Exploitability::StandardRemote) | Some(Exploitability::ObviousLeak) =
        get_exploitability(&outcome)
    {
        panic!(
            "Exploitability should be low, got: {:?}",
            get_exploitability(&outcome)
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
