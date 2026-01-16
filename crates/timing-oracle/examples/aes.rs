//! Example: Testing AES-256-GCM encryption for timing leaks.
//!
//! This demonstrates the CORRECT way to test crypto operations:
//! - Pre-generate all inputs before measurement
//! - Single operation closure executes identical code path for both classes
//! - Only the input data differs

use std::time::Duration;
use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use timing_oracle::{helpers, AttackerModel, Outcome, TimingOracle};

fn main() {
    let key = Key::<Aes256Gcm>::from_slice(&[0u8; 32]);
    let cipher = Aes256Gcm::new(key);
    let nonce = Nonce::from_slice(&[0u8; 12]);

    // Pre-generate inputs: fixed (all zeros) and random (generated per sample)
    let plaintexts = helpers::byte_vecs(1024);

    println!("Testing AES-256-GCM encryption for timing leaks...");
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .test(plaintexts, |input| {
            // Single operation closure: receives &Vec<u8> for current class
            // Performs identical encryption operation for both fixed and random inputs
            std::hint::black_box(
                cipher
                    .encrypt(nonce, input.as_slice())
                    .expect("encryption should succeed"),
            );
        });

    match outcome {
        Outcome::Pass {
            leak_probability,
            effect,
            ..
        } => {
            println!("Result: PASS");
            println!("Leak probability: {:.2}%", leak_probability * 100.0);
            println!(
                "Effect size: shift={:.1}ns, tail={:.1}ns",
                effect.shift_ns, effect.tail_ns
            );
        }
        Outcome::Fail {
            leak_probability,
            effect,
            exploitability,
            ..
        } => {
            println!("Result: FAIL");
            println!("Leak probability: {:.2}%", leak_probability * 100.0);
            println!("Effect size: {:.1}ns", effect.shift_ns);
            println!("Exploitability: {:?}", exploitability);
        }
        Outcome::Inconclusive {
            leak_probability,
            reason,
            ..
        } => {
            println!("Result: INCONCLUSIVE");
            println!("Leak probability: {:.2}%", leak_probability * 100.0);
            println!("Reason: {:?}", reason);
        }
        Outcome::Unmeasurable {
            recommendation, ..
        } => {
            println!("Could not measure: {}", recommendation);
        }
    }
}
