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
fn injected_shift_200ns() {
    run_injected_shift_test("injected_shift_200ns", 200);
}

#[test]
fn injected_shift_500ns() {
    run_injected_shift_test("injected_shift_500ns", 500);
}

#[test]
fn injected_shift_5000ns() {
    run_injected_shift_test("injected_shift_5000ns", 5000);
}

// Baselines filled in at d ≥ 200 ns for the amplification overlay, where
// `busy_wait_ns` per-call overhead is ≤5% of nominal (measured on EPYC @ 5 GHz
// via `examples/busy_wait_calibration.rs`: +13 ns constant per call, which
// saturates actual-vs-nominal at d ≤ ~10 ns).
#[test]
fn injected_shift_1000ns() {
    run_injected_shift_test("injected_shift_1000ns", 1000);
}

#[test]
fn injected_shift_2000ns() {
    run_injected_shift_test("injected_shift_2000ns", 2000);
}

#[test]
fn injected_shift_20000ns() {
    run_injected_shift_test("injected_shift_20000ns", 20000);
}

// ============================================================================
// Operation-loop amplification sweep (ShowTime-style, Rokicki et al.)
// ============================================================================
//
// Wraps the same conditional `busy_wait_ns(delay_ns)` in a k-iteration loop,
// modelling per-op amplification: an attacker who invokes the leaky op k times
// per query multiplies the per-query W_1 by k.
//
// Design rationale: `busy_wait_ns` has a ~13 ns per-call overhead on x86_64
// (measured at 5 GHz EPYC), so small nominal d values don't actually inject
// what their names claim (d ∈ {1, 2, 3, 5} all land at ~19 ns actual). We
// therefore fix d = 200 ns — where actual = 211 ns (5% overhead, negligible
// relative to the k-loop integer multiplier) — and sweep k ∈ {5, 10, 25, 100}.
// Effective per-query delays are {~1055, ~2110, ~5275, ~21100} ns, each within
// 5% of matched single-op baselines {1000, 2000, 5000, 20000}. Overlaying the
// two sets tests the scaling-law prediction that detection depends only on
// actual effective delay d·k, not on (d, k) separately.
//
// Threshold remains AdjacentNetwork (θ = 100 ns). Baseline k = 1 at d = 200
// already exists as `injected_shift_200ns` equivalent — we reuse
// `injected_shift_1000ns`, 2000, 5000, 20000 above as the comparison points.

fn run_amplified_shift_test(test_name: &str, delay_ns: u64, k: u32) {
    init_effect_injection();

    let cipher = Aes128::new(&KEY.into());
    let inputs = InputPair::new(|| [0u8; 16], rand_bytes_16);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.05)
        .fail_threshold(0.95)
        .time_budget(Duration::from_secs(60))
        .test(inputs, |pt| {
            let mut block = (*pt).into();
            cipher.encrypt_block(&mut block);
            // k-fold amplification of the per-op leak. Effective W1 ≈ k · delay_ns.
            if pt[0] != 0 {
                for _ in 0..k {
                    busy_wait_ns(delay_ns);
                }
            }
            std::hint::black_box(block[0]);
        });

    record_detection_outcome(&outcome, test_name);
}

#[test]
fn injected_shift_5ns_k10() {
    run_amplified_shift_test("injected_shift_5ns_k10", 5, 10);
}

#[test]
fn injected_shift_5ns_k100() {
    run_amplified_shift_test("injected_shift_5ns_k100", 5, 100);
}

#[test]
fn injected_shift_5ns_k1000() {
    run_amplified_shift_test("injected_shift_5ns_k1000", 5, 1000);
}

// d = 200 ns base: actual effective delay ≈ 211, 1055, 2110, 5275, 21100 ns
// for k ∈ {1, 5, 10, 25, 100}, matching baselines {200, 1000, 2000, 5000, 20000}
// within 5%.

#[test]
fn injected_shift_200ns_k5() {
    run_amplified_shift_test("injected_shift_200ns_k5", 200, 5);
}

#[test]
fn injected_shift_200ns_k10() {
    run_amplified_shift_test("injected_shift_200ns_k10", 200, 10);
}

#[test]
fn injected_shift_200ns_k25() {
    run_amplified_shift_test("injected_shift_200ns_k25", 200, 25);
}

#[test]
fn injected_shift_200ns_k100() {
    run_amplified_shift_test("injected_shift_200ns_k100", 200, 100);
}

// Second amplification base at d = 1000 ns (per-call overhead ≈ 1%), k ∈
// {2, 5, 20} → actual effective ≈ {2023, 5058, 20230} ns, matching baselines
// {2000, 5000, 20000} ns within ≤ 2%. Orthogonal overlay: confirms the
// scaling law isn't an artifact of the d = 200 base choice.

#[test]
fn injected_shift_1000ns_k2() {
    run_amplified_shift_test("injected_shift_1000ns_k2", 1000, 2);
}

#[test]
fn injected_shift_1000ns_k5() {
    run_amplified_shift_test("injected_shift_1000ns_k5", 1000, 5);
}

#[test]
fn injected_shift_1000ns_k20() {
    run_amplified_shift_test("injected_shift_1000ns_k20", 1000, 20);
}
