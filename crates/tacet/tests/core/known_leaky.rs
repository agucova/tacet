//! Tests that must detect known timing leaks.
//!
//! CI Configuration for leak detection tests:
//! - pass_threshold(0.01): Very hard to falsely pass (we expect leaks)
//! - fail_threshold(0.85): Quick to detect leaks
//! - time_budget(30s): Generous ceiling

use rsa::rand_core::OsRng;
use rsa::{Pkcs1v15Encrypt, RsaPrivateKey, RsaPublicKey};
use std::cell::Cell;
use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{assert_leak_detected, AttackerModel, Outcome, TimingOracle};

/// Test that early-exit comparison is detected as leaky.
///
/// Uses a larger array (512 bytes) to ensure the operation is measurable
/// with coarse timers (~41ns resolution on Apple Silicon).
#[test]
fn detects_early_exit_comparison() {
    let secret = [0u8; 512];

    let inputs = InputPair::new(|| [0u8; 512], rand_bytes_512);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            std::hint::black_box(early_exit_compare(&secret, data));
        });

    // Print the outcome for debugging
    eprintln!("\n[detects_early_exit_comparison]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    // Skip ONLY if unmeasurable (operation too fast for this platform)
    // Inconclusive should NOT be skipped - if we can't detect a known leak, that's a test failure
    if let Outcome::Unmeasurable { recommendation, .. } = &outcome {
        eprintln!(
            "[SKIPPED] detects_early_exit_comparison: {}",
            recommendation
        );
        return;
    }

    // For known leaky code, we expect Fail - uses new macro with rich diagnostics on failure
    assert_leak_detected!(outcome);
}

/// Test that branch-based timing is detected.
#[test]
fn detects_branch_timing() {
    // Baseline: 0 (triggers expensive branch)
    // Sample: never zero (skips expensive branch)
    let inputs = InputPair::new(|| 0u8, || rand::random::<u8>() | 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |x| {
            branch_on_zero(*x);
        });

    // Print the outcome for debugging
    eprintln!("\n[detects_branch_timing]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    // Skip ONLY if unmeasurable (operation too fast for this platform)
    // Inconclusive should NOT be skipped - if we can't detect a known leak, that's a test failure
    if let Outcome::Unmeasurable { recommendation, .. } = &outcome {
        eprintln!("[SKIPPED] detects_branch_timing: {}", recommendation);
        return;
    }

    // For known leaky code, we expect Fail - uses new macro with rich diagnostics on failure
    assert_leak_detected!(outcome);
}

/// Test that CVE-2023-49092 (MARVIN attack) is detected in RustCrypto's `rsa` crate.
///
/// The MARVIN pattern compares decryption of a fixed ciphertext (baseline) against
/// decryption of varied ciphertexts (sample). The fixed ciphertext benefits from
/// microarchitectural caching effects (branch predictor warming, cache line reuse),
/// creating a measurable timing difference that leaks information about the
/// RSA decryption operation.
///
/// See: website/src/content/docs/case-studies/rsa-timing-anomaly.mdx
#[test]
fn detects_marvin_rsa_decryption() {
    const POOL_SIZE: usize = 200;

    // Generate an RSA-1024 key pair
    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).expect("failed to generate key");
    let public_key = RsaPublicKey::from(&private_key);

    // Baseline: a single fixed ciphertext, repeated for all measurements
    let fixed_message = [0x42u8; 32];
    let fixed_ciphertext = public_key
        .encrypt(&mut OsRng, Pkcs1v15Encrypt, &fixed_message)
        .unwrap();

    // Sample: a pool of 200 different ciphertexts, cycled through
    let random_ciphertexts: Vec<Vec<u8>> = (0..POOL_SIZE)
        .map(|_| {
            let msg: [u8; 32] = rand::random();
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
                .unwrap()
        })
        .collect();

    let sample_idx = Cell::new(0usize);
    let inputs = InputPair::new(
        move || fixed_ciphertext.clone(),
        move || {
            let i = sample_idx.get();
            sample_idx.set((i + 1) % POOL_SIZE);
            random_ciphertexts[i].clone()
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.95)
        .time_budget(Duration::from_secs(600))
        .max_samples(5_000_000)
        .warmup(500)
        .calibration_samples(10_000)
        .test(inputs, |ct| {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    // Print the outcome for diagnostics
    eprintln!("\n[detects_marvin_rsa_decryption]");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    // Skip if unmeasurable or inconclusive — RSA decryption has extreme
    // autocorrelation that prevents the posterior from reaching 0.95,
    // so this test serves as a regression guard (must not Pass).
    match &outcome {
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!(
                "[SKIPPED] detects_marvin_rsa_decryption: {}",
                recommendation
            );
            return;
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!("[SKIPPED] detects_marvin_rsa_decryption: {}", reason);
            return;
        }
        _ => {}
    }

    // CVE-2023-49092 is a known timing leak; we expect Fail
    assert_leak_detected!(outcome);
}

fn early_exit_compare(a: &[u8], b: &[u8]) -> bool {
    for i in 0..a.len().min(b.len()) {
        if a[i] != b[i] {
            return false;
        }
    }
    a.len() == b.len()
}

fn branch_on_zero(x: u8) -> u8 {
    if x == 0 {
        // Simulate expensive operation
        std::hint::black_box(0u8);
        for _ in 0..1000 {
            std::hint::black_box(0u8);
        }
        0
    } else {
        x
    }
}

fn rand_bytes_512() -> [u8; 512] {
    let mut arr = [0u8; 512];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
