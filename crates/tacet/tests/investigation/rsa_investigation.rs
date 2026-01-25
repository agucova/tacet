//! Investigation tests for RSA timing anomaly.
//!
//! These tests explore the timing differences observed between different
//! test patterns. Results are printed for analysis - tests don't assert.

use rsa::pkcs1v15::Pkcs1v15Encrypt;
use rsa::rand_core::OsRng;
use rsa::{RsaPrivateKey, RsaPublicKey};
use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{AttackerModel, TimingOracle};

fn rand_bytes_32() -> [u8; 32] {
    rand::random()
}

/// Experiment A: Both classes use THE SAME pool (should show ~0 effect)
#[test]
#[ignore] // Known RSA timing side-channel (RUSTSEC-2023-0071), see docs/investigation-rsa-timing-anomaly.md
fn exp_a_same_pool_both_classes() {
    const POOL_SIZE: usize = 200;

    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    // Single shared pool
    let shared_pool: Vec<Vec<u8>> = (0..POOL_SIZE)
        .map(|_| {
            let msg = rand_bytes_32();
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
                .unwrap()
        })
        .collect();

    let pool_clone = shared_pool.clone();
    let idx_a = std::cell::Cell::new(0usize);
    let idx_b = std::cell::Cell::new(0usize);

    let inputs = InputPair::new(
        move || {
            let i = idx_a.get();
            idx_a.set((i + 1) % POOL_SIZE);
            shared_pool[i].clone()
        },
        move || {
            let i = idx_b.get();
            idx_b.set((i + 1) % POOL_SIZE);
            pool_clone[i].clone()
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |ct| {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    eprintln!("\n=== Experiment A: Same pool for both classes ===");
    eprintln!("Expected: ~0 effect (identical data)");
    eprintln!("{}", tacet::output::format_outcome(&outcome));
}

/// Experiment B: Pool size variation - 10 items
#[test]
#[ignore] // Known RSA timing side-channel (RUSTSEC-2023-0071), see docs/investigation-rsa-timing-anomaly.md
fn exp_b_pool_size_10() {
    run_with_pool_size(10, "B");
}

/// Experiment C: Pool size variation - 100 items
#[test]
#[ignore] // Known RSA timing side-channel (RUSTSEC-2023-0071), see docs/investigation-rsa-timing-anomaly.md
fn exp_c_pool_size_100() {
    run_with_pool_size(100, "C");
}

/// Experiment D: Pool size variation - 1000 items
#[test]
#[ignore] // Known RSA timing side-channel (RUSTSEC-2023-0071), see docs/investigation-rsa-timing-anomaly.md
fn exp_d_pool_size_1000() {
    run_with_pool_size(1000, "D");
}

fn run_with_pool_size(pool_size: usize, label: &str) {
    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    let pool_a: Vec<Vec<u8>> = (0..pool_size)
        .map(|_| {
            let msg = rand_bytes_32();
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
                .unwrap()
        })
        .collect();

    let pool_b: Vec<Vec<u8>> = (0..pool_size)
        .map(|_| {
            let msg = rand_bytes_32();
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
                .unwrap()
        })
        .collect();

    let idx_a = std::cell::Cell::new(0usize);
    let idx_b = std::cell::Cell::new(0usize);

    let inputs = InputPair::new(
        move || {
            let i = idx_a.get();
            idx_a.set((i + 1) % pool_size);
            pool_a[i].clone()
        },
        move || {
            let i = idx_b.get();
            idx_b.set((i + 1) % pool_size);
            pool_b[i].clone()
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |ct| {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    eprintln!(
        "\n=== Experiment {}: Pool size {} (both classes different pools) ===",
        label, pool_size
    );
    eprintln!("{}", tacet::output::format_outcome(&outcome));
}

/// Experiment E: RSA-2048 with original pattern (same ct repeated)
#[test]
#[ignore] // Known RSA timing side-channel (RUSTSEC-2023-0071), see docs/investigation-rsa-timing-anomaly.md
fn exp_e_rsa2048_same_repeated() {
    let private_key = RsaPrivateKey::new(&mut OsRng, 2048).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    let fixed_ct = public_key
        .encrypt(&mut OsRng, Pkcs1v15Encrypt, &[0x42u8; 32])
        .unwrap();

    let random_cts: Vec<Vec<u8>> = (0..200)
        .map(|_| {
            let msg = rand_bytes_32();
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
                .unwrap()
        })
        .collect();

    let idx = std::cell::Cell::new(0usize);
    let fixed_ct_clone = fixed_ct.clone();

    let inputs = InputPair::new(
        move || fixed_ct_clone.clone(),
        move || {
            let i = idx.get();
            idx.set((i + 1) % 200);
            random_cts[i].clone()
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |ct| {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    eprintln!("\n=== Experiment E: RSA-2048, same ciphertext repeated ===");
    eprintln!("{}", tacet::output::format_outcome(&outcome));
}

/// Experiment F: RSA encrypt (public key op) with original pattern
#[test]
fn exp_f_encrypt_same_repeated() {
    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    let fixed_msg = [0x42u8; 32];

    let inputs = InputPair::new(|| fixed_msg, rand_bytes_32);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |msg| {
            let ct = public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, msg)
                .unwrap();
            std::hint::black_box(ct[0]);
        });

    eprintln!("\n=== Experiment F: RSA-1024 encrypt, fixed vs random message ===");
    eprintln!("{}", tacet::output::format_outcome(&outcome));
}

/// Experiment G: Simple XOR with original pattern (known constant-time)
#[test]
fn exp_g_xor_same_repeated() {
    let secret = [0xABu8; 32];

    let inputs = InputPair::new(
        || [0x00u8; 32], // Fixed: all zeros
        rand_bytes_32,   // Random
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |data| {
            let mut result = 0u8;
            for i in 0..32 {
                result |= secret[i] ^ data[i];
            }
            std::hint::black_box(result);
        });

    eprintln!("\n=== Experiment G: XOR (constant-time), fixed zeros vs random ===");
    eprintln!("{}", tacet::output::format_outcome(&outcome));
}

/// Experiment H: Pool size 1 for both classes (extreme case)
#[test]
#[ignore] // Known RSA timing side-channel (RUSTSEC-2023-0071), see docs/investigation-rsa-timing-anomaly.md
fn exp_h_pool_size_1_both() {
    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    // Each class has exactly ONE ciphertext
    let ct_a = public_key
        .encrypt(&mut OsRng, Pkcs1v15Encrypt, &rand_bytes_32())
        .unwrap();
    let ct_b = public_key
        .encrypt(&mut OsRng, Pkcs1v15Encrypt, &rand_bytes_32())
        .unwrap();

    let inputs = InputPair::new(move || ct_a.clone(), move || ct_b.clone());

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |ct| {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    eprintln!("\n=== Experiment H: Pool size 1 for BOTH classes ===");
    eprintln!("Baseline: same ct_a repeated, Sample: same ct_b repeated");
    eprintln!("{}", tacet::output::format_outcome(&outcome));
}

// =============================================================================
// SHAPE ANALYSIS EXPERIMENTS
// =============================================================================

/// Count leading zero bytes in a byte slice
fn count_leading_zeros(bytes: &[u8]) -> usize {
    bytes.iter().take_while(|&&b| b == 0).count()
}

/// Get the most significant byte (first non-zero, or 0 if all zeros)
fn msb_value(bytes: &[u8]) -> u8 {
    *bytes.iter().find(|&&b| b != 0).unwrap_or(&0)
}

/// Compute Hamming weight (popcount) of all bytes
fn hamming_weight(bytes: &[u8]) -> u32 {
    bytes.iter().map(|b| b.count_ones()).sum()
}

/// Experiment I1: Correlate ciphertext shape with decryption timing
///
/// This test generates many ciphertexts, measures decryption time for each,
/// and computes correlation with shape properties (leading zeros, MSB, Hamming weight).
#[test]
#[ignore] // Known RSA timing side-channel (RUSTSEC-2023-0071), see docs/investigation-rsa-timing-anomaly.md
fn exp_i1_shape_correlation() {
    use std::time::Instant;

    const NUM_CIPHERTEXTS: usize = 100;
    const REPS_PER_CT: usize = 100;

    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    eprintln!("\n=== Experiment I1: Shape Correlation Analysis ===");
    eprintln!(
        "Generating {} ciphertexts, {} repetitions each...",
        NUM_CIPHERTEXTS, REPS_PER_CT
    );

    // Generate ciphertexts and collect shape properties
    let mut data: Vec<(Vec<u8>, usize, u8, u32, f64)> = Vec::with_capacity(NUM_CIPHERTEXTS);

    for i in 0..NUM_CIPHERTEXTS {
        let msg = rand_bytes_32();
        let ct = public_key
            .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
            .unwrap();

        let leading_zeros = count_leading_zeros(&ct);
        let msb = msb_value(&ct);
        let hw = hamming_weight(&ct);

        // Warmup
        for _ in 0..10 {
            let _ = private_key.decrypt(Pkcs1v15Encrypt, &ct).unwrap();
        }

        // Measure timing (mean of REPS_PER_CT repetitions)
        let start = Instant::now();
        for _ in 0..REPS_PER_CT {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, &ct).unwrap();
            std::hint::black_box(plaintext[0]);
        }
        let elapsed = start.elapsed();
        let mean_ns = elapsed.as_nanos() as f64 / REPS_PER_CT as f64;

        data.push((ct, leading_zeros, msb, hw, mean_ns));

        if (i + 1) % 20 == 0 {
            eprintln!("  Progress: {}/{}", i + 1, NUM_CIPHERTEXTS);
        }
    }

    // Output CSV header and data
    eprintln!("\n--- Raw Data (CSV) ---");
    eprintln!("idx,leading_zeros,msb_value,hamming_weight,mean_timing_ns");
    for (i, (_, lz, msb, hw, timing)) in data.iter().enumerate() {
        eprintln!("{},{},{},{},{:.1}", i, lz, msb, hw, timing);
    }

    // Compute correlations
    let timings: Vec<f64> = data.iter().map(|(_, _, _, _, t)| *t).collect();
    let leading_zeros: Vec<f64> = data.iter().map(|(_, lz, _, _, _)| *lz as f64).collect();
    let msb_values: Vec<f64> = data.iter().map(|(_, _, msb, _, _)| *msb as f64).collect();
    let hamming_weights: Vec<f64> = data.iter().map(|(_, _, _, hw, _)| *hw as f64).collect();

    let r_lz = pearson_correlation(&timings, &leading_zeros);
    let r_msb = pearson_correlation(&timings, &msb_values);
    let r_hw = pearson_correlation(&timings, &hamming_weights);

    eprintln!("\n--- Correlation Analysis ---");
    eprintln!(
        "Pearson correlation (timing vs leading_zeros): r = {:.4}",
        r_lz
    );
    eprintln!(
        "Pearson correlation (timing vs msb_value):     r = {:.4}",
        r_msb
    );
    eprintln!(
        "Pearson correlation (timing vs hamming_weight): r = {:.4}",
        r_hw
    );

    // Summary statistics
    let mean_timing: f64 = timings.iter().sum::<f64>() / timings.len() as f64;
    let min_timing = timings.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_timing = timings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_timing - min_timing;

    eprintln!("\n--- Timing Summary ---");
    eprintln!("Mean: {:.1} ns", mean_timing);
    eprintln!("Min:  {:.1} ns", min_timing);
    eprintln!("Max:  {:.1} ns", max_timing);
    eprintln!("Range: {:.1} ns", range);

    // Find fastest and slowest ciphertexts
    let (fastest_idx, _) = data
        .iter()
        .enumerate()
        .min_by(|a, b| a.1 .4.partial_cmp(&b.1 .4).unwrap())
        .unwrap();
    let (slowest_idx, _) = data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1 .4.partial_cmp(&b.1 .4).unwrap())
        .unwrap();

    eprintln!("\n--- Extreme Cases ---");
    eprintln!(
        "Fastest (idx {}): lz={}, msb=0x{:02X}, hw={}, timing={:.1}ns",
        fastest_idx,
        data[fastest_idx].1,
        data[fastest_idx].2,
        data[fastest_idx].3,
        data[fastest_idx].4
    );
    eprintln!(
        "Slowest (idx {}): lz={}, msb=0x{:02X}, hw={}, timing={:.1}ns",
        slowest_idx,
        data[slowest_idx].1,
        data[slowest_idx].2,
        data[slowest_idx].3,
        data[slowest_idx].4
    );

    eprintln!("\n--- Interpretation ---");
    if r_lz.abs() > 0.5 || r_msb.abs() > 0.5 || r_hw.abs() > 0.5 {
        eprintln!("STRONG correlation found! Shape hypothesis supported.");
        if r_lz.abs() > 0.5 {
            eprintln!("  - Leading zeros strongly correlated with timing");
        }
        if r_msb.abs() > 0.5 {
            eprintln!("  - MSB value strongly correlated with timing");
        }
        if r_hw.abs() > 0.5 {
            eprintln!("  - Hamming weight strongly correlated with timing");
        }
    } else if r_lz.abs() > 0.3 || r_msb.abs() > 0.3 || r_hw.abs() > 0.3 {
        eprintln!("MODERATE correlation found. Shape hypothesis partially supported.");
    } else {
        eprintln!("WEAK correlation. Shape hypothesis not strongly supported.");
        eprintln!("Consider other factors (blinding, implementation details).");
    }
}

/// Compute Pearson correlation coefficient between two vectors
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|a| a * a).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Experiment I2: Compare two ciphertexts with similar magnitude (same MSB range)
///
/// If the shape hypothesis is correct, two ciphertexts with the same leading byte
/// should show minimal timing difference compared to Exp H's ~211ns.
#[test]
#[ignore] // Known RSA timing side-channel (RUSTSEC-2023-0071), see docs/investigation-rsa-timing-anomaly.md
fn exp_i2_fixed_magnitude() {
    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    eprintln!("\n=== Experiment I2: Fixed Magnitude Comparison ===");
    eprintln!("Finding two ciphertexts with same leading byte (0x80-0xFF range)...");

    // Generate ciphertexts until we find two with the same leading byte in the high range
    let mut ct_a: Option<Vec<u8>> = None;
    let mut ct_b: Option<Vec<u8>> = None;
    let mut target_msb: Option<u8> = None;
    let mut attempts = 0;

    while ct_b.is_none() && attempts < 10000 {
        let msg = rand_bytes_32();
        let ct = public_key
            .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
            .unwrap();
        let msb = msb_value(&ct);
        let lz = count_leading_zeros(&ct);

        // Only consider ciphertexts with MSB in 0x80-0xFF range (full magnitude)
        if msb >= 0x80 && lz == 0 {
            if ct_a.is_none() {
                ct_a = Some(ct);
                target_msb = Some(msb);
                eprintln!("  Found first ciphertext with MSB=0x{:02X}", msb);
            } else if target_msb == Some(msb) {
                ct_b = Some(ct);
                eprintln!("  Found second ciphertext with MSB=0x{:02X}", msb);
            }
        }
        attempts += 1;
    }

    let ct_a = ct_a.expect("Could not find first ciphertext");
    let ct_b = ct_b.expect("Could not find matching ciphertext");

    eprintln!("Found matching pair after {} attempts", attempts);
    eprintln!(
        "CT_A: lz={}, msb=0x{:02X}, hw={}",
        count_leading_zeros(&ct_a),
        msb_value(&ct_a),
        hamming_weight(&ct_a)
    );
    eprintln!(
        "CT_B: lz={}, msb=0x{:02X}, hw={}",
        count_leading_zeros(&ct_b),
        msb_value(&ct_b),
        hamming_weight(&ct_b)
    );

    // Run the 1-vs-1 test (same pattern as Exp H)
    let inputs = InputPair::new(move || ct_a.clone(), move || ct_b.clone());

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |ct| {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    eprintln!("\n--- Results ---");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    eprintln!("\n--- Interpretation ---");
    eprintln!("Compare to Experiment H (~211ns effect with random ciphertexts):");
    eprintln!("If effect is much smaller here, magnitude/shape hypothesis is confirmed.");
    eprintln!("If effect is similar, other factors are at play.");
}

/// Experiment I3: Stratified comparison - low MSB vs high MSB ciphertexts
///
/// If magnitude matters, ciphertexts with small MSB (0x01-0x3F) should decrypt
/// differently than those with large MSB (0xC0-0xFF).
#[test]
#[ignore] // Run with: sudo cargo test exp_i3 -- --ignored --nocapture
fn exp_i3_msb_stratified() {
    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    const POOL_SIZE: usize = 50;

    eprintln!("\n=== Experiment I3: MSB-Stratified Comparison ===");
    eprintln!(
        "Building pool of {} 'low MSB' (0x01-0x3F) ciphertexts...",
        POOL_SIZE
    );

    // Build pool of low-MSB ciphertexts
    let mut low_msb_pool: Vec<Vec<u8>> = Vec::with_capacity(POOL_SIZE);
    let mut attempts = 0;
    while low_msb_pool.len() < POOL_SIZE && attempts < 100000 {
        let msg = rand_bytes_32();
        let ct = public_key
            .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
            .unwrap();
        let msb = msb_value(&ct);
        if msb > 0 && msb <= 0x3F {
            low_msb_pool.push(ct);
        }
        attempts += 1;
    }
    eprintln!(
        "  Found {} low-MSB ciphertexts in {} attempts",
        low_msb_pool.len(),
        attempts
    );

    eprintln!(
        "Building pool of {} 'high MSB' (0xC0-0xFF) ciphertexts...",
        POOL_SIZE
    );

    // Build pool of high-MSB ciphertexts
    let mut high_msb_pool: Vec<Vec<u8>> = Vec::with_capacity(POOL_SIZE);
    attempts = 0;
    while high_msb_pool.len() < POOL_SIZE && attempts < 100000 {
        let msg = rand_bytes_32();
        let ct = public_key
            .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
            .unwrap();
        let msb = msb_value(&ct);
        if msb >= 0xC0 {
            high_msb_pool.push(ct);
        }
        attempts += 1;
    }
    eprintln!(
        "  Found {} high-MSB ciphertexts in {} attempts",
        high_msb_pool.len(),
        attempts
    );

    if low_msb_pool.len() < POOL_SIZE || high_msb_pool.len() < POOL_SIZE {
        eprintln!("WARNING: Could not build full pools. Results may be unreliable.");
    }

    let low_pool_size = low_msb_pool.len();
    let high_pool_size = high_msb_pool.len();

    let idx_low = std::cell::Cell::new(0usize);
    let idx_high = std::cell::Cell::new(0usize);

    let inputs = InputPair::new(
        move || {
            let i = idx_low.get();
            idx_low.set((i + 1) % low_pool_size);
            low_msb_pool[i].clone()
        },
        move || {
            let i = idx_high.get();
            idx_high.set((i + 1) % high_pool_size);
            high_msb_pool[i].clone()
        },
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |ct| {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    eprintln!("\n--- Results ---");
    eprintln!("Baseline: low MSB pool (0x01-0x3F), Sample: high MSB pool (0xC0-0xFF)");
    eprintln!("{}", tacet::output::format_outcome(&outcome));

    eprintln!("\n--- Interpretation ---");
    eprintln!("If significant effect detected, magnitude strongly affects timing.");
    eprintln!("The direction (positive/negative shift) indicates which decrypts faster.");
}

/// Comprehensive shape analysis experiment - RUN WITH SUDO
///
/// This test runs I1, I2, and baseline H experiments and produces a summary.
/// Requires sudo for PMU timer (kperf) access for reliable results.
///
/// Run with: sudo cargo test exp_comprehensive -- --ignored --nocapture
#[test]
#[ignore]
fn exp_comprehensive_shape_analysis() {
    use std::time::Instant;

    eprintln!("\n");
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║     RSA TIMING ANOMALY - COMPREHENSIVE SHAPE ANALYSIS        ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();

    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    // ========================================================================
    // PART 1: Per-ciphertext timing analysis (shape correlation)
    // ========================================================================
    eprintln!("┌──────────────────────────────────────────────────────────────┐");
    eprintln!("│ PART 1: Per-Ciphertext Timing Analysis                       │");
    eprintln!("└──────────────────────────────────────────────────────────────┘");

    const NUM_CTS: usize = 50;
    const REPS: usize = 200;

    eprintln!(
        "Generating {} ciphertexts, {} measurements each...",
        NUM_CTS, REPS
    );

    let mut ct_data: Vec<(Vec<u8>, u8, u32, f64)> = Vec::with_capacity(NUM_CTS);

    for i in 0..NUM_CTS {
        let msg = rand_bytes_32();
        let ct = public_key
            .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
            .unwrap();
        let msb = msb_value(&ct);
        let hw = hamming_weight(&ct);

        // Warmup
        for _ in 0..20 {
            let _ = private_key.decrypt(Pkcs1v15Encrypt, &ct).unwrap();
        }

        // Measure
        let start = Instant::now();
        for _ in 0..REPS {
            let p = private_key.decrypt(Pkcs1v15Encrypt, &ct).unwrap();
            std::hint::black_box(p[0]);
        }
        let mean_ns = start.elapsed().as_nanos() as f64 / REPS as f64;

        ct_data.push((ct, msb, hw, mean_ns));

        if (i + 1) % 10 == 0 {
            eprintln!("  Progress: {}/{}", i + 1, NUM_CTS);
        }
    }

    // Compute correlations
    let timings: Vec<f64> = ct_data.iter().map(|(_, _, _, t)| *t).collect();
    let msb_vals: Vec<f64> = ct_data.iter().map(|(_, m, _, _)| *m as f64).collect();
    let hw_vals: Vec<f64> = ct_data.iter().map(|(_, _, h, _)| *h as f64).collect();

    let r_msb = pearson_correlation(&timings, &msb_vals);
    let r_hw = pearson_correlation(&timings, &hw_vals);

    let mean_t: f64 = timings.iter().sum::<f64>() / timings.len() as f64;
    let min_t = timings.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_t = timings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    eprintln!();
    eprintln!(
        "  Timing: mean={:.0}ns, min={:.0}ns, max={:.0}ns, range={:.0}ns",
        mean_t,
        min_t,
        max_t,
        max_t - min_t
    );
    eprintln!("  Correlation with MSB:     r = {:.4}", r_msb);
    eprintln!("  Correlation with Hamming: r = {:.4}", r_hw);

    // ========================================================================
    // PART 2: Oracle tests comparing different patterns
    // ========================================================================
    eprintln!();
    eprintln!("┌──────────────────────────────────────────────────────────────┐");
    eprintln!("│ PART 2: Oracle Pattern Comparison                            │");
    eprintln!("└──────────────────────────────────────────────────────────────┘");

    // Test 2a: Two random ciphertexts (baseline, like Exp H)
    let ct_rand_a = public_key
        .encrypt(&mut OsRng, Pkcs1v15Encrypt, &rand_bytes_32())
        .unwrap();
    let ct_rand_b = public_key
        .encrypt(&mut OsRng, Pkcs1v15Encrypt, &rand_bytes_32())
        .unwrap();

    eprintln!();
    eprintln!("Test 2a: Two random ciphertexts (1 vs 1)");
    eprintln!(
        "  CT_A: msb=0x{:02X}, hw={}",
        msb_value(&ct_rand_a),
        hamming_weight(&ct_rand_a)
    );
    eprintln!(
        "  CT_B: msb=0x{:02X}, hw={}",
        msb_value(&ct_rand_b),
        hamming_weight(&ct_rand_b)
    );

    let inputs = InputPair::new(move || ct_rand_a.clone(), move || ct_rand_b.clone());

    let outcome_2a = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |ct| {
            let p = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(p[0]);
        });

    eprintln!("{}", tacet::output::format_outcome(&outcome_2a));

    // Test 2b: Two ciphertexts with same MSB
    eprintln!();
    eprintln!("Test 2b: Two ciphertexts with SAME MSB");

    let mut ct_same_a: Option<Vec<u8>> = None;
    let mut ct_same_b: Option<Vec<u8>> = None;
    let mut target_msb: Option<u8> = None;

    for _ in 0..5000 {
        let ct = public_key
            .encrypt(&mut OsRng, Pkcs1v15Encrypt, &rand_bytes_32())
            .unwrap();
        let msb = msb_value(&ct);
        if msb >= 0x80 {
            if ct_same_a.is_none() {
                ct_same_a = Some(ct);
                target_msb = Some(msb);
            } else if target_msb == Some(msb) {
                ct_same_b = Some(ct);
                break;
            }
        }
    }

    if let (Some(cta), Some(ctb)) = (ct_same_a, ct_same_b) {
        eprintln!(
            "  CT_A: msb=0x{:02X}, hw={}",
            msb_value(&cta),
            hamming_weight(&cta)
        );
        eprintln!(
            "  CT_B: msb=0x{:02X}, hw={}",
            msb_value(&ctb),
            hamming_weight(&ctb)
        );

        let inputs = InputPair::new(move || cta.clone(), move || ctb.clone());

        let outcome_2b = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .pass_threshold(0.15)
            .fail_threshold(0.99)
            .time_budget(Duration::from_secs(30))
            .test(inputs, |ct| {
                let p = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
                std::hint::black_box(p[0]);
            });

        eprintln!("{}", tacet::output::format_outcome(&outcome_2b));
    } else {
        eprintln!("  Could not find matching MSB pair");
    }

    // Test 2c: Two pools of 100 ciphertexts (control)
    eprintln!();
    eprintln!("Test 2c: Two pools of 100 ciphertexts (100 vs 100)");

    let pool_a: Vec<Vec<u8>> = (0..100)
        .map(|_| {
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &rand_bytes_32())
                .unwrap()
        })
        .collect();
    let pool_b: Vec<Vec<u8>> = (0..100)
        .map(|_| {
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &rand_bytes_32())
                .unwrap()
        })
        .collect();

    let idx_a = std::cell::Cell::new(0usize);
    let idx_b = std::cell::Cell::new(0usize);

    let inputs = InputPair::new(
        move || {
            let i = idx_a.get();
            idx_a.set((i + 1) % 100);
            pool_a[i].clone()
        },
        move || {
            let i = idx_b.get();
            idx_b.set((i + 1) % 100);
            pool_b[i].clone()
        },
    );

    let outcome_2c = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |ct| {
            let p = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            std::hint::black_box(p[0]);
        });

    eprintln!("{}", tacet::output::format_outcome(&outcome_2c));

    // ========================================================================
    // SUMMARY
    // ========================================================================
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║                        SUMMARY                                ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Shape Correlation Analysis:");
    eprintln!(
        "  - MSB correlation:     r = {:.4} ({})",
        r_msb,
        if r_msb.abs() > 0.5 {
            "STRONG"
        } else if r_msb.abs() > 0.3 {
            "MODERATE"
        } else {
            "WEAK"
        }
    );
    eprintln!(
        "  - Hamming correlation: r = {:.4} ({})",
        r_hw,
        if r_hw.abs() > 0.5 {
            "STRONG"
        } else if r_hw.abs() > 0.3 {
            "MODERATE"
        } else {
            "WEAK"
        }
    );
    eprintln!();
    eprintln!("If correlations are weak and 1-vs-1 tests show large effects,");
    eprintln!("the timing variation is NOT due to simple shape properties.");
    eprintln!("Other factors to investigate: blinding, Montgomery reduction,");
    eprintln!("cache effects, or implementation-specific behavior.");
}

// =============================================================================
// IMPLEMENTATION COMPARISON EXPERIMENTS
// =============================================================================

// Experiment C1 (ring comparison) temporarily disabled due to sha2 type compatibility issues
// TODO: Fix the ring vs RustCrypto signing comparison

/// Experiment B1: Blinding effectiveness test
///
/// RSA blinding should cause timing to vary randomly even for the same input.
/// We measure the same ciphertext many times and look at the variance.
#[test]
#[ignore] // Run with: sudo cargo test exp_b1 -- --ignored --nocapture
fn exp_b1_blinding_variance() {
    use std::time::Instant;

    eprintln!("\n");
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║     BLINDING EFFECTIVENESS TEST                              ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("If blinding is working, repeated decryptions of the same ciphertext");
    eprintln!("should show timing variance (each uses a different random blinding factor).");
    eprintln!();

    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    // Single ciphertext
    let ct = public_key
        .encrypt(&mut OsRng, Pkcs1v15Encrypt, &rand_bytes_32())
        .unwrap();

    // Warmup
    for _ in 0..100 {
        let _ = private_key.decrypt(Pkcs1v15Encrypt, &ct).unwrap();
    }

    // Collect individual timing samples
    const SAMPLES: usize = 1000;
    let mut timings = Vec::with_capacity(SAMPLES);

    for _ in 0..SAMPLES {
        let start = Instant::now();
        let p = private_key.decrypt(Pkcs1v15Encrypt, &ct).unwrap();
        let elapsed = start.elapsed().as_nanos() as f64;
        std::hint::black_box(p[0]);
        timings.push(elapsed);
    }

    // Compute statistics
    let mean: f64 = timings.iter().sum::<f64>() / timings.len() as f64;
    let variance: f64 =
        timings.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (timings.len() - 1) as f64;
    let std_dev = variance.sqrt();
    let cv = std_dev / mean * 100.0; // Coefficient of variation

    let min = timings.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = timings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Sort for percentiles
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p10 = timings[SAMPLES / 10];
    let p50 = timings[SAMPLES / 2];
    let p90 = timings[SAMPLES * 9 / 10];

    eprintln!("Single ciphertext decrypted {} times:", SAMPLES);
    eprintln!();
    eprintln!("  Mean:   {:.0} ns", mean);
    eprintln!("  StdDev: {:.0} ns", std_dev);
    eprintln!("  CV:     {:.2}%", cv);
    eprintln!();
    eprintln!("  Min:    {:.0} ns", min);
    eprintln!("  P10:    {:.0} ns", p10);
    eprintln!("  P50:    {:.0} ns", p50);
    eprintln!("  P90:    {:.0} ns", p90);
    eprintln!("  Max:    {:.0} ns", max);
    eprintln!("  Range:  {:.0} ns", max - min);
    eprintln!();

    // Interpretation
    if cv > 5.0 {
        eprintln!("High variance (CV > 5%) suggests blinding IS adding randomness.");
        eprintln!("But this also means timing varies significantly per-decrypt.");
    } else if cv > 1.0 {
        eprintln!("Moderate variance (CV 1-5%). Some blinding effect present.");
    } else {
        eprintln!("Low variance (CV < 1%). Timing is very consistent.");
        eprintln!("Either blinding has minimal timing impact, or it's not active.");
    }

    // Compare to expected effect size
    eprintln!();
    eprintln!("For context: Experiment H showed ~450-600ns timing DIFFERENCE");
    eprintln!(
        "between different ciphertexts. If blinding variance ({:.0}ns StdDev)",
        std_dev
    );
    eprintln!("is smaller than the inter-ciphertext difference, the per-ciphertext");
    eprintln!("timing variation is a real, persistent property of each ciphertext.");
}

/// Experiment B2: Multi-ciphertext persistence test
///
/// Measures multiple ciphertexts repeatedly to see if each has a PERSISTENT
/// mean timing or if the variation is just noise.
#[test]
#[ignore] // Run with: sudo cargo test exp_b2 -- --ignored --nocapture
fn exp_b2_persistence_test() {
    use std::time::Instant;

    eprintln!("\n");
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║     PERSISTENCE TEST: Do ciphertexts have stable means?      ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();

    let private_key = RsaPrivateKey::new(&mut OsRng, 1024).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    const NUM_CTS: usize = 10;
    const REPS_PER_CT: usize = 500;

    eprintln!(
        "Measuring {} ciphertexts, {} decryptions each...",
        NUM_CTS, REPS_PER_CT
    );
    eprintln!();

    // Generate ciphertexts
    let ciphertexts: Vec<Vec<u8>> = (0..NUM_CTS)
        .map(|_| {
            public_key
                .encrypt(&mut OsRng, Pkcs1v15Encrypt, &rand_bytes_32())
                .unwrap()
        })
        .collect();

    // Warmup all ciphertexts
    for ct in &ciphertexts {
        for _ in 0..20 {
            let _ = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
        }
    }

    // Measure each ciphertext
    let mut ct_stats: Vec<(f64, f64, f64, f64)> = Vec::new(); // (mean, std, min, max)

    for (i, ct) in ciphertexts.iter().enumerate() {
        let mut timings = Vec::with_capacity(REPS_PER_CT);

        for _ in 0..REPS_PER_CT {
            let start = Instant::now();
            let p = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            let elapsed = start.elapsed().as_nanos() as f64;
            std::hint::black_box(p[0]);
            timings.push(elapsed);
        }

        let mean: f64 = timings.iter().sum::<f64>() / timings.len() as f64;
        let variance: f64 =
            timings.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (timings.len() - 1) as f64;
        let std_dev = variance.sqrt();
        let min = timings.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = timings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        ct_stats.push((mean, std_dev, min, max));

        eprintln!(
            "  CT[{}]: mean={:.0}ns, std={:.0}ns, range=[{:.0}, {:.0}]",
            i, mean, std_dev, min, max
        );
    }

    // Analyze inter-ciphertext variation
    let means: Vec<f64> = ct_stats.iter().map(|(m, _, _, _)| *m).collect();
    let overall_mean: f64 = means.iter().sum::<f64>() / means.len() as f64;
    let inter_ct_variance: f64 = means
        .iter()
        .map(|m| (m - overall_mean).powi(2))
        .sum::<f64>()
        / (means.len() - 1) as f64;
    let inter_ct_std = inter_ct_variance.sqrt();

    let min_mean = means.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_mean = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_range = max_mean - min_mean;

    // Average within-ciphertext std
    let avg_within_std: f64 =
        ct_stats.iter().map(|(_, s, _, _)| *s).sum::<f64>() / ct_stats.len() as f64;

    eprintln!();
    eprintln!("┌──────────────────────────────────────────────────────────────┐");
    eprintln!("│ ANALYSIS                                                     │");
    eprintln!("└──────────────────────────────────────────────────────────────┘");
    eprintln!();
    eprintln!("  Overall mean:           {:.0} ns", overall_mean);
    eprintln!(
        "  Inter-ciphertext StdDev: {:.0} ns (variation BETWEEN ciphertexts)",
        inter_ct_std
    );
    eprintln!(
        "  Avg within-CT StdDev:    {:.0} ns (blinding noise)",
        avg_within_std
    );
    eprintln!("  Mean range (max-min):    {:.0} ns", mean_range);
    eprintln!();
    eprintln!("  Fastest CT mean: {:.0} ns", min_mean);
    eprintln!("  Slowest CT mean: {:.0} ns", max_mean);
    eprintln!();

    // Key ratio: inter-CT variation vs within-CT noise
    let signal_to_noise = inter_ct_std / (avg_within_std / (REPS_PER_CT as f64).sqrt());

    eprintln!("┌──────────────────────────────────────────────────────────────┐");
    eprintln!("│ INTERPRETATION                                               │");
    eprintln!("└──────────────────────────────────────────────────────────────┘");
    eprintln!();
    eprintln!("  Signal-to-noise ratio: {:.2}", signal_to_noise);
    eprintln!("    (inter-CT StdDev / standard error of mean)");
    eprintln!();

    if signal_to_noise > 3.0 {
        eprintln!("  ✓ STRONG PERSISTENCE: Ciphertexts have reliably different mean times.");
        eprintln!(
            "    The ~{:.0}ns inter-CT variation is NOT just noise.",
            inter_ct_std
        );
        eprintln!("    Each ciphertext has an intrinsic 'speed' property.");
    } else if signal_to_noise > 1.5 {
        eprintln!("  ~ MODERATE PERSISTENCE: Some real variation between ciphertexts.");
        eprintln!("    Effect is detectable but could be partially noise.");
    } else {
        eprintln!("  ✗ WEAK PERSISTENCE: Inter-CT variation is mostly noise.");
        eprintln!("    The oracle may be detecting statistical fluctuations.");
    }

    eprintln!();
    eprintln!("  For context:");
    eprintln!(
        "    - Blinding adds ~{:.0}ns StdDev per-measurement",
        avg_within_std
    );
    eprintln!(
        "    - With {} measurements, SE of mean = {:.0}ns",
        REPS_PER_CT,
        avg_within_std / (REPS_PER_CT as f64).sqrt()
    );
    eprintln!("    - Inter-CT spread = {:.0}ns", mean_range);
    eprintln!(
        "    - This spread is {:.1}x the standard error",
        mean_range / (avg_within_std / (REPS_PER_CT as f64).sqrt())
    );
}
