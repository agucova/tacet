//! Standalone orion test to verify compilation
//!
//! This is a minimal test to ensure orion integration works correctly
//! without dependencies on other crypto libraries that may have linking issues.

use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{AttackerModel, TimingOracle};

fn rand_bytes_32() -> Vec<u8> {
    let mut vec = vec![0u8; 32];
    for byte in &mut vec {
        *byte = rand::random();
    }
    vec
}

#[test]
fn orion_auth_compiles() {
    use orion::auth;

    let secret_key = auth::SecretKey::default();

    let inputs = InputPair::new(|| vec![0u8; 32], rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(5))
        .max_samples(10_000)
        .test(inputs, |message| {
            let tag = auth::authenticate(&secret_key, message).unwrap();
            std::hint::black_box(tag.unprotected_as_bytes()[0]);
        });

    eprintln!(
        "Orion auth test outcome: {}",
        tacet::output::format_outcome(&outcome)
    );
}

#[test]
fn orion_hash_compiles() {
    use orion::hash;

    let inputs = InputPair::new(|| vec![0u8; 32], rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(5))
        .max_samples(10_000)
        .test(inputs, |message| {
            let digest = hash::digest(message).unwrap();
            std::hint::black_box(digest.as_ref()[0]);
        });

    eprintln!(
        "Orion hash test outcome: {}",
        tacet::output::format_outcome(&outcome)
    );
}

#[test]
fn orion_aead_compiles() {
    use orion::aead;

    let secret_key = aead::SecretKey::default();

    let inputs = InputPair::new(|| vec![0u8; 32], rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(5))
        .max_samples(10_000)
        .test(inputs, |plaintext| {
            let ciphertext = aead::seal(&secret_key, plaintext).unwrap();
            std::hint::black_box(ciphertext[0]);
        });

    eprintln!(
        "Orion aead test outcome: {}",
        tacet::output::format_outcome(&outcome)
    );
}
