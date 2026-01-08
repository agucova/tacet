//! Example: Testing AES-256-GCM encryption for timing leaks.
//!
//! This demonstrates the CORRECT way to test crypto operations:
//! - Pre-generate all inputs before measurement
//! - Single operation closure executes identical code path for both classes
//! - Only the input data differs

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use timing_oracle::helpers;

fn main() {
    let key = Key::<Aes256Gcm>::from_slice(&[0u8; 32]);
    let cipher = Aes256Gcm::new(key);
    let nonce = Nonce::from_slice(&[0u8; 12]);

    // Pre-generate inputs: fixed (all zeros) and random (generated per sample)
    let plaintexts = helpers::byte_vecs(1024);

    let result = timing_oracle::TimingOracle::new()
        .samples(50_000)
        .test(plaintexts, |input| {
            // Single operation closure: receives &Vec<u8> for current class
            // Performs identical encryption operation for both fixed and random inputs
            std::hint::black_box(
                cipher
                    .encrypt(nonce, input.as_slice())
                    .expect("encryption should succeed")
            );
        })
        .unwrap_completed();

    println!("Leak probability: {:.2}%", result.leak_probability * 100.0);
    println!("CI gate: {}", if result.ci_gate.passed() { "PASS" } else { "FAIL" });

    if let Some(effect) = result.effect {
        println!("Effect size: {:.1} ns", effect.shift_ns);
    }
}
