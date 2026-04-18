//! Controlled byte-conditional timing-leak injection.
//!
//! Wraps an otherwise-constant-time AES-128 encrypt with a secret-dependent
//! `busy_wait_ns(delay)` call. The delay fires only when the plaintext's first
//! byte has its high bit set, leaking roughly one bit of secret per call.
//!
//! This is a ground-truth synthetic leak: the magnitude is known by
//! construction, so detection rate as a function of `delay_ns` is a direct
//! power curve. Reviewer context: USENIX Sec'26 reviewer C requested
//! "insert branch and cache-dependent operations into the existing libraries";
//! this is the branch-dependent variant with a sweepable magnitude.
//!
//! Threshold: `AdjacentNetwork` (θ = 100 ns). Effects ≥ 100 ns should be
//! detected; smaller effects should yield Inconclusive rather than missed
//! detections — that's the calibrated-verdict claim we want to demonstrate.

use std::time::Duration;

use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use tacet::helpers::effect::{busy_wait_ns, init_effect_injection};
use tacet::helpers::InputPair;
use tacet::{AttackerModel, TimingOracle};

use super::assert::record_detection_outcome;

/// Canonical fixed AES-128 key used for all injected-shift tests.
/// (Value is arbitrary — NIST AES test vector from FIPS 197.)
const KEY: [u8; 16] = [
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
];

fn rand_bytes_16() -> [u8; 16] {
    let mut arr = [0u8; 16];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

/// Shared test body. Runs an AES-128 block encrypt with a conditional
/// `busy_wait_ns(delay_ns)` triggered by the first plaintext byte's top bit.
fn run_injected_shift_test(test_name: &str, delay_ns: u64) {
    // Required before calling busy_wait_ns on this thread.
    init_effect_injection();

    let cipher = Aes128::new(&KEY.into());

    // DudeCT two-class pattern: fixed all-zero baseline vs. random sample.
    // The baseline's first byte is 0x00 (never triggers the delay).
    // Random samples have first byte ∈ [0, 255] uniformly, so the condition
    // `pt[0] != 0` fires on 255/256 ≈ 99.6% of random samples.
    //
    // Design note: fire-on-nonzero (rather than, e.g., fire-on-top-bit) is
    // chosen so that the effective W1 distance between the two classes
    // approaches delay_ns exactly. This gives a clean ground-truth
    // delay_ns ↔ W1 mapping for the power curve. A 50%-trigger design
    // would yield W1 ≈ delay_ns / 2 and make the curve misleading.
    let inputs = InputPair::new(|| [0u8; 16], rand_bytes_16);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.05)
        .fail_threshold(0.95)
        .time_budget(Duration::from_secs(60))
        .test(inputs, |pt| {
            let mut block = (*pt).into();
            cipher.encrypt_block(&mut block);
            // Class-discriminating leak: 0% on baseline, ~99.6% on sample.
            // Effective W1 ≈ delay_ns.
            if pt[0] != 0 {
                busy_wait_ns(delay_ns);
            }
            std::hint::black_box(block[0]);
        });

    record_detection_outcome(&outcome, test_name);
}

// ============================================================================
// Delay sweep: 2, 5, 20, 50, 100, 500 ns
// ============================================================================
//
// Expectation under AdjacentNetwork (θ = 100 ns):
//   2,5 ns   → dominated by platform timer resolution; expect Inconclusive
//              (ThresholdElevated) on ARM, possibly PASS on x86_64.
//              Per the calibration claim, we should NOT get false negatives
//              disguised as Pass.
//   20 ns    → below θ but above measurement floor on x86_64 clean; mostly
//              Inconclusive.
//   50 ns    → near θ; detection rate should climb.
//   100 ns   → at θ; expect majority Fail under clean conditions.
//   500 ns   → well above θ; expect near-100% Fail under all conditions.

#[test]
fn injected_shift_2ns() {
    run_injected_shift_test("injected_shift_2ns", 2);
}

#[test]
fn injected_shift_5ns() {
    run_injected_shift_test("injected_shift_5ns", 5);
}

#[test]
fn injected_shift_20ns() {
    run_injected_shift_test("injected_shift_20ns", 20);
}

#[test]
fn injected_shift_50ns() {
    run_injected_shift_test("injected_shift_50ns", 50);
}

#[test]
fn injected_shift_100ns() {
    run_injected_shift_test("injected_shift_100ns", 100);
}

#[test]
fn injected_shift_500ns() {
    run_injected_shift_test("injected_shift_500ns", 500);
}
