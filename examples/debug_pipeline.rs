//! Minimal reproducible example for debugging pipeline steps.
//!
//! Run with:
//!   TO_DEBUG_PIPELINE=1 cargo run --example debug_pipeline

use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::output::format_outcome;
use timing_oracle::{AttackerModel, TimingOracle};

fn rand_bytes_32() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

fn main() {
    println!("=== X25519 Multiple-Op Pipeline Debug ===\n");

    let basepoint = x25519_dalek::X25519_BASEPOINT_BYTES;

    let fixed_scalars: [[u8; 32]; 3] = [
        [0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82,
         0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a, 0x33, 0x9c,
         0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f,
         0x9a, 0x2d, 0x7e, 0x4c, 0x8b, 0x3a, 0x6d, 0x5e],
        [0x2a, 0x3b, 0x4c, 0x5d, 0x6e, 0x7f, 0x80, 0x91,
         0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7, 0x08, 0x19,
         0x2a, 0x3b, 0x4c, 0x5d, 0x6e, 0x7f, 0x80, 0x91,
         0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7, 0x08, 0x19],
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
         0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00,
         0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
         0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00],
    ];

    // Cache-controlled inputs: both classes draw from the same pre-generated pool.
    // This avoids "fixed vs random" cache artifacts while still differing data.
    const SAMPLES: usize = 10_000;
    let mut pool: Vec<[[u8; 32]; 3]> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        pool.push([rand_bytes_32(), rand_bytes_32(), rand_bytes_32()]);
    }
    pool[0] = fixed_scalars;

    let idx = std::cell::Cell::new(0usize);
    let inputs = InputPair::new(
        || {
            let i = idx.get();
            idx.set((i + 1) % SAMPLES);
            pool[i]
        },
        || {
            let i = idx.get();
            idx.set((i + 1) % SAMPLES);
            pool[(i + SAMPLES / 2) % SAMPLES]
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .max_samples(10_000)
        .min_effect_ns(50.0)
        .test(inputs, |scalar_set| {
            let mut total = 0u8;
            for scalar in scalar_set {
                let result = x25519_dalek::x25519(*scalar, basepoint);
                total ^= result[0];
            }
            std::hint::black_box(total);
        });

    println!("{}", format_outcome(&outcome));
}
