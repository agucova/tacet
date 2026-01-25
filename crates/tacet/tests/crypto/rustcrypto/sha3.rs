//! crypto::rustcrypto::sha3
//!
//! SHA-3 (Keccak) hash function timing tests using the RustCrypto implementation.
//! Crate: sha3
//! Family: Hash
//! Expected: All Pass (constant-time implementation)
//!
//! Tests whether the Keccak sponge construction has data-dependent timing.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA differs.
//! Pre-generate inputs outside closures to avoid measuring RNG time.

use sha3::{Digest, Sha3_256, Sha3_384, Sha3_512};
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

fn rand_bytes_128() -> [u8; 128] {
    let mut arr = [0u8; 128];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

// ============================================================================
// Core SHA-3 Tests
// ============================================================================

/// SHA3-256 should be constant-time
#[test]
fn rustcrypto_sha3_256_ct() {
    let fixed_input: [u8; 32] = [
        0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07,
        0x34, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
        0xee, 0xff,
    ];

    let inputs = InputPair::new(|| fixed_input, rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            let hash = Sha3_256::digest(data);
            std::hint::black_box(hash[0]);
        });

    eprintln!("\n[rustcrypto_sha3_256_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_sha3_256_ct");

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
                "SHA3-256 should be constant-time, got leak_probability={:.1}%, exploitability={:?}",
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

/// SHA3-384 should be constant-time
#[test]
fn rustcrypto_sha3_384_ct() {
    let fixed_input: [u8; 64] = [0x5a; 64];
    let inputs = InputPair::new(|| fixed_input, rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            let hash = Sha3_384::digest(data);
            std::hint::black_box(hash[0]);
        });

    eprintln!("\n[rustcrypto_sha3_384_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_sha3_384_ct");

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
                "SHA3-384 should be constant-time, got leak_probability={:.1}%, exploitability={:?}",
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

/// SHA3-512 should be constant-time
#[test]
fn rustcrypto_sha3_512_ct() {
    let fixed_input: [u8; 64] = [0xa5; 64];
    let inputs = InputPair::new(|| fixed_input, rand_bytes_64);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            let hash = Sha3_512::digest(data);
            std::hint::black_box(hash[0]);
        });

    eprintln!("\n[rustcrypto_sha3_512_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_sha3_512_ct");

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
                "SHA3-512 should be constant-time, got leak_probability={:.1}%, exploitability={:?}",
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
// Data Independence Tests
// ============================================================================

/// SHA3-256 with same-length but different data
///
/// Tests constant-time property for fixed-length inputs
/// Note: SHA3 is NOT constant-time for varying lengths by design
#[test]
fn rustcrypto_sha3_256_data_independence() {
    let fixed_input: [u8; 128] = [0x42; 128];
    let inputs = InputPair::new(|| fixed_input, rand_bytes_128);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            let hash = Sha3_256::digest(data);
            std::hint::black_box(hash[0]);
        });

    eprintln!("\n[rustcrypto_sha3_256_data_independence]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_sha3_256_data_independence");

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
                "SHA3-256 should be constant-time for same-length inputs, got leak_probability={:.1}%, exploitability={:?}",
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

/// Compare high vs low Hamming weight inputs for SHA3
///
/// Tests if the number of 1-bits in input affects timing
#[test]
fn rustcrypto_sha3_256_hamming() {
    let inputs = InputPair::new(|| [0x00u8; 32], || [0xFFu8; 32]);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            let hash = Sha3_256::digest(data);
            std::hint::black_box(hash[0]);
        });

    eprintln!("\n[rustcrypto_sha3_256_hamming]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_sha3_256_hamming");

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

/// Test incremental/streaming hash updates
///
/// Many applications process data in chunks - this tests whether
/// update boundaries affect timing
#[test]
fn rustcrypto_sha3_256_incremental_ct() {
    let fixed_chunks: ([u8; 32], [u8; 32]) = ([0x11; 32], [0x22; 32]);

    let inputs = InputPair::new(|| fixed_chunks, || (rand_bytes_32(), rand_bytes_32()));

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |(data1, data2)| {
            let mut hasher = Sha3_256::new();
            hasher.update(data1);
            hasher.update(data2);
            let hash = hasher.finalize();
            std::hint::black_box(hash[0]);
        });

    eprintln!("\n[rustcrypto_sha3_256_incremental_ct]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    let outcome = skip_if_unreliable!(outcome, "rustcrypto_sha3_256_incremental_ct");

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
                "SHA3-256 incremental hashing should be constant-time, got leak_probability={:.1}%, exploitability={:?}",
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
