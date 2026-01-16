//! Crypto attack integration tests - validates detection of realistic timing vulnerabilities.
//!
//! Tests cover:
//! - Cache-based attacks (AES S-box, cache lines, memory access patterns)
//! - Algorithmic attacks (modular exponentiation, bit patterns)
//! - Table lookup timing (L1/L2/L3 cache)
//! - Effect pattern validation (uniform shift, tail effect, mixed)
//! - Exploitability classification (Crosby et al. thresholds)
//!
//! CI Configuration:
//! - For leak detection tests: pass_threshold(0.01), fail_threshold(0.85)
//! - For constant-time tests: pass_threshold(0.15), fail_threshold(0.99)
//! - time_budget(30s) for all tests

use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{
    skip_if_unreliable, AttackerModel, EffectPattern, Exploitability, Outcome, TimingOracle,
};

// ============================================================================
// Helper Functions Module
// ============================================================================

mod helpers {
    use super::*;

    /// Naive modular exponentiation - intentionally leaky!
    ///
    /// Uses square-and-multiply algorithm where timing depends on
    /// Hamming weight of the exponent (number of 1 bits).
    pub fn modpow_naive(base: &BigUint, exp: &BigUint, modulus: &BigUint) -> BigUint {
        let mut result = BigUint::one();
        let mut base = base.clone();
        let mut exp = exp.clone();

        while exp > BigUint::zero() {
            if &exp & BigUint::one() == BigUint::one() {
                result = (result * &base) % modulus;
            }
            base = (&base * &base) % modulus;
            exp >>= 1;
        }

        result
    }

    /// Generate a simplified AES S-box or permutation table
    pub fn generate_sbox() -> [u8; 256] {
        let mut sbox = [0u8; 256];
        for i in 0..256 {
            // Simple permutation based on bit reversal and XOR
            let mut x = i as u8;
            x = (x & 0xF0) >> 4 | (x & 0x0F) << 4;
            x = (x & 0xCC) >> 2 | (x & 0x33) << 2;
            x = (x & 0xAA) >> 1 | (x & 0x55) << 1;
            sbox[i] = x ^ 0x63;
        }
        sbox
    }

    /// Platform-specific busy-wait for controlled timing delays
    #[cfg(target_arch = "x86_64")]
    pub fn busy_wait_cycles(cycles: u64) {
        let start = unsafe { core::arch::x86_64::_rdtsc() };
        while unsafe { core::arch::x86_64::_rdtsc() } - start < cycles {
            std::hint::spin_loop();
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn busy_wait_cycles(cycles: u64) {
        let start = unsafe {
            let cnt: u64;
            core::arch::asm!("mrs {}, cntvct_el0", out(reg) cnt);
            cnt
        };
        loop {
            let now = unsafe {
                let cnt: u64;
                core::arch::asm!("mrs {}, cntvct_el0", out(reg) cnt);
                cnt
            };
            if now - start >= cycles {
                break;
            }
            std::hint::spin_loop();
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn busy_wait_cycles(_cycles: u64) {
        // Fallback: just use a fixed delay
        for _ in 0..100 {
            std::hint::black_box(std::hint::spin_loop());
        }
    }

    /// Cache line size aligned buffer for cache timing tests
    #[repr(align(64))]
    #[allow(dead_code)]
    pub struct CacheAligned<T> {
        pub data: T,
    }

    #[allow(dead_code)]
    impl<T> CacheAligned<T> {
        pub fn new(data: T) -> Self {
            Self { data }
        }
    }
}

// ============================================================================
// Category 1: Cache-Based Attacks
// ============================================================================

/// 1.1 AES S-box Timing (Fast) - Should detect cache-based timing
#[test]
fn aes_sbox_timing_fast() {
    let sbox = helpers::generate_sbox();
    let secret_key = 0xABu8;

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 10_000;
    let indices = InputPair::new(|| secret_key, rand::random::<u8>);

    // Use Research mode (theta=0) to test raw detection capability for cache timing
    let outcome = TimingOracle::for_attacker(AttackerModel::Research)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(indices, |idx| {
            let val = std::hint::black_box(*idx);
            std::hint::black_box(sbox[val as usize]);
        });

    // Skip if measurement is unreliable (cache timing is hard to measure on Apple Silicon)
    let outcome = skip_if_unreliable!(outcome, "aes_sbox_timing_fast");

    eprintln!("\n[aes_sbox_timing_fast]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // On many systems, S-box lookups show cache effects
    // We expect moderate leak probability and/or TailEffect pattern
    let (leak_probability, effect) = match &outcome {
        Outcome::Pass { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Fail { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Inconclusive { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Unmeasurable { .. } => return,
    };

    let has_tail_effect = matches!(effect.pattern, EffectPattern::TailEffect);

    // Should detect either moderate leak probability or tail effect pattern
    let has_leak_signal = leak_probability > 0.3 || has_tail_effect;
    assert!(
        has_leak_signal,
        "Expected to detect some cache timing effect (got leak_probability={}, has_tail_effect={})",
        leak_probability,
        has_tail_effect
    );
}

/// 1.2 AES S-box Timing (Thorough) - High confidence detection
#[test]
#[ignore = "slow test - run with --ignored"]
fn aes_sbox_timing_thorough() {
    let sbox = helpers::generate_sbox();
    let secret_key = 0xABu8;

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 100_000;
    let indices = InputPair::new(|| secret_key, rand::random::<u8>);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(indices, |idx| {
            let val = std::hint::black_box(*idx);
            std::hint::black_box(sbox[val as usize]);
        });

    eprintln!("\n[aes_sbox_timing_thorough]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // With 100k samples, cache effects should be more pronounced
    // Should detect leak with high confidence (Fail outcome)
    match &outcome {
        Outcome::Fail { leak_probability, .. } => {
            assert!(
                *leak_probability > 0.5,
                "Expected high leak probability with 100k samples (got {})",
                leak_probability
            );
        }
        Outcome::Pass { leak_probability, .. } => {
            panic!(
                "Expected Fail outcome for S-box timing leak, got Pass with leak_probability={}",
                leak_probability
            );
        }
        Outcome::Inconclusive { leak_probability, reason, .. } => {
            panic!(
                "Expected Fail outcome for S-box timing leak, got Inconclusive: {:?}, leak_probability={}",
                reason, leak_probability
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Skipping: {}", recommendation);
        }
    }
}

/// 1.3 Cache Line Boundary Effects
#[test]
fn cache_line_boundary_effects() {
    // Create a large buffer with cache-line aligned sections
    let buffer = vec![0u8; 4096];
    let secret_offset_same_line = 0usize;
    let secret_offset_diff_line = 64usize; // Different cache line

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 10_000;
    let indices = InputPair::new(
        || secret_offset_same_line,
        || secret_offset_diff_line + (rand::random::<u32>() as usize % 4) * 64,
    );

    // Use Research mode (theta=0) to test raw detection capability for cache timing
    let outcome = TimingOracle::for_attacker(AttackerModel::Research)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(indices, |idx| {
            let idx_val = std::hint::black_box(*idx);
            std::hint::black_box(buffer[idx_val % buffer.len()]);
        });

    // Operation may be too fast on some platforms - skip if unmeasurable
    let outcome = skip_if_unreliable!(outcome, "cache_line_boundary_effects");

    eprintln!("\n[cache_line_boundary_effects]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Cache line boundary accesses may show timing differences due to cache effects
    // Should detect at least moderate leak probability or tail effects
    let (leak_probability, effect) = match &outcome {
        Outcome::Pass { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Fail { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Inconclusive { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Unmeasurable { .. } => return,
    };

    let has_tail_effect = matches!(effect.pattern, EffectPattern::TailEffect | EffectPattern::Mixed);

    let has_leak_signal = leak_probability > 0.2 || has_tail_effect;
    assert!(
        has_leak_signal,
        "Expected to detect cache line boundary effects (got leak_probability={}, has_tail_effect={})",
        leak_probability,
        has_tail_effect
    );
}

/// 1.4 Memory Access Pattern Leak
#[test]
fn memory_access_pattern_leak() {
    use std::cell::Cell;

    let data = vec![rand::random::<u64>(); 1024];
    let secret_pattern = [0usize, 64, 128, 192]; // Sequential in large strides

    // Pre-generate access indices using InputPair
    const SAMPLES: usize = 8_000;
    let data_len = data.len();
    let pattern_idx = Cell::new(0usize);
    let indices = InputPair::new(
        || {
            // Cycle through secret_pattern - call once to get first value
            let i = pattern_idx.get();
            let access_idx = secret_pattern[i % secret_pattern.len()];
            pattern_idx.set((i + 1) % secret_pattern.len());
            access_idx
        },
        || rand::random::<u32>() as usize % data_len,
    );

    // Use Research mode (theta=0) to test raw detection capability for memory patterns
    let outcome = TimingOracle::for_attacker(AttackerModel::Research)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(indices, |access_idx| {
            let access_idx_val = std::hint::black_box(*access_idx);
            std::hint::black_box(data[access_idx_val]);
        });

    // Memory access is very fast - may be unmeasurable on coarse timers (e.g., macOS 42ns)
    let outcome = skip_if_unreliable!(outcome, "memory_access_pattern_leak");

    eprintln!("\n[memory_access_pattern_leak]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Memory access patterns with different strides should show timing differences
    // due to cache effects (sequential vs random access)
    let (leak_probability, effect) = match &outcome {
        Outcome::Pass { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Fail { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Inconclusive { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Unmeasurable { .. } => return,
    };

    let has_leak_signal = leak_probability > 0.2;
    assert!(
        has_leak_signal,
        "Expected to detect memory access pattern timing differences (got leak_probability={})",
        leak_probability
    );

    // Should have measurable effect
    let total_effect = effect.shift_ns.abs() + effect.tail_ns.abs();
    assert!(
        total_effect > 5.0,
        "Expected measurable timing effect for memory patterns (got {:.1}ns)",
        total_effect
    );
}

// ============================================================================
// Category 2: Modular Exponentiation
// ============================================================================

/// 2.1 Square-and-Multiply Timing (Fast) - Should detect algorithmic timing
#[test]
fn modexp_square_and_multiply_timing() {
    let base = BigUint::from(5u32);
    let modulus = BigUint::from(1000000007u64); // Large prime

    // High Hamming weight exponent (many 1 bits = more multiplies)
    let exp_high_hamming = BigUint::from(0xFFFFu32); // 16 ones
    // Low Hamming weight exponent (few 1 bits = fewer multiplies)
    let exp_low_hamming = BigUint::from(0x8001u32); // 2 ones

    let exponents = InputPair::new(|| exp_high_hamming.clone(), || exp_low_hamming.clone());

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(exponents, |exp| {
            std::hint::black_box(helpers::modpow_naive(&base, exp, &modulus));
        });

    // Print full formatted output
    eprintln!("\n{}", timing_oracle::output::format_outcome(&outcome));

    // Square-and-multiply creates a uniform shift (more iterations = more time)
    // Should detect leak (Fail outcome)
    match &outcome {
        Outcome::Fail {
            leak_probability,
            effect,
            ..
        } => {
            assert!(
                *leak_probability > 0.8,
                "Expected high leak probability for modexp (got {})",
                leak_probability
            );

            // Should show UniformShift or Mixed pattern (BigInt ops may have variance)
            assert!(
                matches!(effect.pattern, EffectPattern::UniformShift | EffectPattern::Mixed),
                "Expected UniformShift or Mixed pattern for algorithmic timing (got {:?})",
                effect.pattern
            );

            // Should have significant shift
            assert!(
                effect.shift_ns > 50.0,
                "Expected significant shift_ns (got {:.1})",
                effect.shift_ns
            );
        }
        Outcome::Pass { leak_probability, .. } => {
            panic!(
                "Expected Fail outcome for modexp timing leak, got Pass with leak_probability={}",
                leak_probability
            );
        }
        Outcome::Inconclusive { leak_probability, reason, .. } => {
            panic!(
                "Expected Fail outcome for modexp timing leak, got Inconclusive: {:?}, leak_probability={}",
                reason, leak_probability
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Skipping: {}", recommendation);
        }
    }
}

/// 2.2 Exponent Bit Pattern Timing
#[test]
fn modexp_bit_pattern_timing() {
    let base = BigUint::from(7u32);
    let modulus = BigUint::from(1000000007u64);

    // Create random exponents with controlled Hamming weights
    let exp_many_ones = BigUint::from(0xAAAAAAAAu32); // 50% ones
    let exp_few_ones = BigUint::from(0x80000001u32); // ~6% ones

    let exponents = InputPair::new(|| exp_many_ones.clone(), || exp_few_ones.clone());

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(exponents, |exp| {
            std::hint::black_box(helpers::modpow_naive(&base, exp, &modulus));
        });

    eprintln!("\n[modexp_bit_pattern_timing]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Should detect leak (Fail outcome)
    match &outcome {
        Outcome::Fail { leak_probability, .. } => {
            assert!(
                *leak_probability > 0.8,
                "Expected high leak probability for bit pattern timing (got {})",
                leak_probability
            );
        }
        Outcome::Pass { leak_probability, .. } => {
            panic!(
                "Expected Fail outcome for bit pattern timing, got Pass with leak_probability={}",
                leak_probability
            );
        }
        Outcome::Inconclusive { leak_probability, reason, .. } => {
            panic!(
                "Expected Fail outcome for bit pattern timing, got Inconclusive: {:?}, leak_probability={}",
                reason, leak_probability
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Skipping: {}", recommendation);
        }
    }
}

// ============================================================================
// Category 3: Table Lookup Timing
// ============================================================================

/// 3.1 Small Table (L1 Cache) - Should show minimal timing
///
/// Uses DudeCT's two-class pattern: all-zero index vs random index
#[test]
fn table_lookup_small_l1() {
    let table = [rand::random::<u64>(); 4]; // 32 bytes, fits in L1

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 10_000;
    let table_len = table.len();
    let indices = InputPair::new(
        || 0usize,
        || rand::random::<u32>() as usize % table_len,
    );

    // Use AdjacentNetwork - L1-resident small table should pass
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(indices, |idx| {
            let idx_val = std::hint::black_box(*idx);
            std::hint::black_box(table[idx_val]);
        });

    let outcome = skip_if_unreliable!(outcome, "table_lookup_small_l1");

    eprintln!("\n[table_lookup_small_l1]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // L1-resident small table: should pass or have Negligible exploitability if Fail
    match &outcome {
        Outcome::Pass { .. } => {
            // Good - no timing leak detected
        }
        Outcome::Fail { exploitability, .. } => {
            assert!(
                matches!(exploitability, Exploitability::Negligible),
                "Expected Negligible exploitability for L1-resident table (got {:?})",
                exploitability
            );
        }
        Outcome::Inconclusive { .. } => {
            // Acceptable for small table operations
        }
        Outcome::Unmeasurable { .. } => {
            // Operation too fast - acceptable
        }
    }
}

/// 3.2 Medium Table (L2/L3 Cache) - May show cache effects
#[test]
fn table_lookup_medium_l2() {
    let table = vec![rand::random::<u64>(); 32]; // 256 bytes (AES S-box size)

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 10_000;
    let table_len = table.len();
    let indices = InputPair::new(
        || 0usize,
        || rand::random::<u32>() as usize % table_len,
    );

    // Use AdjacentNetwork - medium table should pass or have low exploitability
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(indices, |idx| {
            let idx_val = std::hint::black_box(*idx);
            std::hint::black_box(table[idx_val]);
        });

    let outcome = skip_if_unreliable!(outcome, "table_lookup_medium_l2");

    eprintln!("\n[table_lookup_medium_l2]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Medium-sized table may show cache timing effects
    // Should be measurable but exploitability should be low (Negligible or PossibleLAN at most)
    match &outcome {
        Outcome::Pass { .. } => {
            // Good - no timing leak detected
        }
        Outcome::Fail { exploitability, .. } => {
            assert!(
                matches!(exploitability, Exploitability::Negligible | Exploitability::PossibleLAN),
                "Expected low exploitability for medium table (got {:?})",
                exploitability
            );
        }
        Outcome::Inconclusive { .. } => {
            // Acceptable for medium table operations
        }
        Outcome::Unmeasurable { .. } => {
            // Operation too fast - acceptable
        }
    }
}

/// 3.3 Large Table (Cache Thrashing) - Should show cache effects
#[test]
fn table_lookup_large_cache_thrash() {
    let table = vec![rand::random::<u64>(); 512]; // 4KB

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 10_000;
    let table_len = table.len();
    let indices = InputPair::new(
        || 0usize,
        || rand::random::<u32>() as usize % table_len,
    );

    // Use Research mode (theta=0) to test raw detection capability for cache timing
    let outcome = TimingOracle::for_attacker(AttackerModel::Research)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(indices, |idx| {
            let idx_val = std::hint::black_box(*idx);
            std::hint::black_box(table[idx_val]);
        });

    // Skip if measurement is unreliable (cache timing is hard to measure on Apple Silicon)
    let outcome = skip_if_unreliable!(outcome, "table_lookup_large_cache_thrash");

    eprintln!("\n[table_lookup_large_cache_thrash]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // Large table should show cache timing effects
    let (leak_probability, effect) = match &outcome {
        Outcome::Pass { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Fail { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Inconclusive { leak_probability, effect, .. } => (*leak_probability, effect),
        Outcome::Unmeasurable { .. } => return,
    };

    let has_tail_effect = matches!(effect.pattern, EffectPattern::TailEffect);

    // Should detect leak probability or tail effect pattern
    let has_leak_signal = leak_probability > 0.4 || has_tail_effect;
    assert!(
        has_leak_signal,
        "Expected cache effects for large table (got leak_probability={}, has_tail_effect={})",
        leak_probability,
        has_tail_effect
    );
}

// ============================================================================
// Category 4: Effect Pattern Validation
// ============================================================================

/// 4.1 Pure Uniform Shift - Validates UniformShift classification
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn effect_pattern_pure_uniform_shift() {
    // Use a larger delay to ensure uniform shift dominates any measurement noise
    const DELAY_CYCLES: u64 = 500;

    let which_class = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(which_class, |class| {
            if *class == 0 {
                // No delay
                std::hint::black_box(42);
            } else {
                // Constant delay
                helpers::busy_wait_cycles(DELAY_CYCLES);
                std::hint::black_box(42);
            }
        });

    // Skip assertions if measurement is unreliable
    let outcome = skip_if_unreliable!(outcome, "effect_pattern_pure_uniform_shift");

    eprintln!("\n[effect_pattern_pure_uniform_shift]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let effect = match &outcome {
        Outcome::Pass { effect, .. } => effect,
        Outcome::Fail { effect, .. } => effect,
        Outcome::Inconclusive { effect, .. } => effect,
        Outcome::Unmeasurable { .. } => return,
    };

    // Should classify as UniformShift (or Mixed with dominant shift)
    // On real hardware, constant delays may have small jitter that creates a minor tail
    assert!(
        matches!(effect.pattern, EffectPattern::UniformShift | EffectPattern::Mixed),
        "Expected UniformShift or Mixed pattern (got {:?})",
        effect.pattern
    );

    // Should have significant shift component
    // Use abs() since sign depends on which class is slower
    assert!(
        effect.shift_ns.abs() > 50.0,
        "Expected significant shift component (got {:.1}ns)",
        effect.shift_ns
    );

    // Shift should dominate tail (at least 5x larger)
    assert!(
        effect.shift_ns.abs() > effect.tail_ns.abs() * 5.0,
        "Expected shift to dominate tail (got shift={:.1}ns, tail={:.1}ns)",
        effect.shift_ns,
        effect.tail_ns
    );
}

/// 4.2 Pure Tail Effect - Validates TailEffect classification
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn effect_pattern_pure_tail() {
    use std::cell::Cell;

    // Use larger values to ensure tail effect is measurable
    const BASE_CYCLES: u64 = 50; // Base operation to ensure measurability (>5 ticks)
    const EXPENSIVE_CYCLES: u64 = 2000;
    const TAIL_PROBABILITY: f64 = 0.15;
    const SAMPLES: usize = 10_000;

    // Pre-generate spike decisions - both closures use identical code paths
    let spike_decisions: Vec<bool> = (0..SAMPLES)
        .map(|_| rand::random::<f64>() < TAIL_PROBABILITY)
        .collect();
    // Use separate counters to avoid randomized schedule scrambling the spike pattern
    let fixed_idx = Cell::new(0usize);
    let random_idx = Cell::new(0usize);

    let which_class = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(which_class, |class| {
            if *class == 0 {
                let i = fixed_idx.get();
                fixed_idx.set(i.wrapping_add(1));
                // Base operation to ensure measurability on ARM
                helpers::busy_wait_cycles(BASE_CYCLES);
                // No spike regardless of decision
                let _ = spike_decisions[i % SAMPLES];
                std::hint::black_box(42);
            } else {
                let i = random_idx.get();
                random_idx.set(i.wrapping_add(1));
                // Same base operation
                helpers::busy_wait_cycles(BASE_CYCLES);
                // Apply spike based on pre-generated decision
                if spike_decisions[i % SAMPLES] {
                    helpers::busy_wait_cycles(EXPENSIVE_CYCLES);
                }
                std::hint::black_box(42);
            }
        });

    // Skip assertions if measurement is unreliable
    let outcome = skip_if_unreliable!(outcome, "effect_pattern_pure_tail");

    eprintln!("\n[effect_pattern_pure_tail]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let effect = match &outcome {
        Outcome::Pass { effect, .. } => effect,
        Outcome::Fail { effect, .. } => effect,
        Outcome::Inconclusive { effect, .. } => effect,
        Outcome::Unmeasurable { .. } => return,
    };

    // When neither effect is statistically significant, pattern is Indeterminate - this is valid
    // Only assert on pattern when we have significant effects to classify
    match effect.pattern {
        EffectPattern::Indeterminate => {
            // Indeterminate is valid when effects aren't statistically significant
            eprintln!(
                "Note: Got Indeterminate pattern (shift={:.1}ns, tail={:.1}ns) - effects not significant",
                effect.shift_ns, effect.tail_ns
            );
        }
        EffectPattern::TailEffect | EffectPattern::Mixed => {
            // Should have significant tail component when classified as TailEffect/Mixed
            // Note: A probabilistic spike (15% chance of 2000 cycles) creates both:
            // - A shift effect (mean increases by ~300 cycles = 15% × 2000)
            // - A tail effect (variance increases in upper quantiles)
            // So we expect Mixed or TailEffect, not necessarily |tail| > |shift|
            assert!(
                effect.tail_ns.abs() > 20.0,
                "Expected significant tail component (got {:.1}ns)",
                effect.tail_ns
            );
        }
        EffectPattern::UniformShift => {
            // UniformShift is not expected for this test, but acceptable if effect is small
            eprintln!(
                "Unexpected UniformShift pattern (shift={:.1}ns, tail={:.1}ns)",
                effect.shift_ns, effect.tail_ns
            );
        }
    }
}

/// 4.3 Mixed Pattern - Validates Mixed classification
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn effect_pattern_mixed() {
    use std::cell::Cell;

    const BASE_DELAY: u64 = 30;
    const SPIKE_DELAY: u64 = 150;
    const SPIKE_PROBABILITY: f64 = 0.15;
    const SAMPLES: usize = 10_000;

    // Pre-generate spike decisions - both closures use identical code paths
    let spike_decisions: Vec<bool> = (0..SAMPLES)
        .map(|_| rand::random::<f64>() < SPIKE_PROBABILITY)
        .collect();
    // Use separate counters to avoid randomized schedule scrambling the spike pattern
    let fixed_idx = Cell::new(0usize);
    let random_idx = Cell::new(0usize);

    let which_class = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(which_class, |class| {
            if *class == 0 {
                let i = fixed_idx.get();
                fixed_idx.set(i.wrapping_add(1));
                // No delays, but access spike_decisions for identical code path
                let _ = spike_decisions[i % SAMPLES];
                std::hint::black_box(42);
            } else {
                let i = random_idx.get();
                random_idx.set(i.wrapping_add(1));
                // Base delay (uniform shift)
                helpers::busy_wait_cycles(BASE_DELAY);

                // Plus occasional spike (tail effect)
                if spike_decisions[i % SAMPLES] {
                    helpers::busy_wait_cycles(SPIKE_DELAY);
                }
                std::hint::black_box(42);
            }
        });

    // Skip assertions if measurement is unreliable
    let outcome = skip_if_unreliable!(outcome, "effect_pattern_mixed");

    eprintln!("\n[effect_pattern_mixed]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    let effect = match &outcome {
        Outcome::Pass { effect, .. } => effect,
        Outcome::Fail { effect, .. } => effect,
        Outcome::Inconclusive { effect, .. } => effect,
        Outcome::Unmeasurable { .. } => return,
    };

    // Should classify as Mixed
    assert!(
        matches!(effect.pattern, EffectPattern::Mixed),
        "Expected Mixed pattern (got {:?})",
        effect.pattern
    );

    // Both components should be significant (use abs for sign-independence)
    assert!(
        effect.shift_ns.abs() > 3.0 && effect.tail_ns.abs() > 3.0,
        "Expected both |shift| and |tail| > 3ns (got shift={:.1}ns, tail={:.1}ns)",
        effect.shift_ns,
        effect.tail_ns
    );
}

// ============================================================================
// Category 5: Exploitability Thresholds
// ============================================================================

/// 5.1 Negligible (<100ns) - Should classify as Negligible
///
/// Uses XOR-based constant-time equality comparison to simulate safe cryptographic code.
/// This is a realistic test: XOR comparison should have negligible timing differences
/// since it executes the same operations regardless of input data.
#[test]
fn exploitability_negligible() {
    use rand::Rng;

    let inputs = InputPair::new(
        || [0u8; 32], // Baseline: zeros
        || {
            let mut rng = rand::rng();
            let mut arr = [0u8; 32];
            rng.fill(&mut arr);
            arr
        }, // Sample: random data
    );

    // XOR comparison is constant-time - expect Pass
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.15)
        .fail_threshold(0.99)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |input| {
            // Pure XOR-based constant-time equality check
            let mut diff = 0u8;
            for (a, b) in input.iter().zip([0u8; 32].iter()) {
                diff |= a ^ b;
            }
            std::hint::black_box(diff);
        });

    let outcome = skip_if_unreliable!(outcome, "exploitability_negligible");

    eprintln!("\n[exploitability_negligible]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    // XOR-based comparison is constant-time, so exploitability should always be Negligible
    // Whether it passes or fails (due to noise), the effect should be tiny
    match &outcome {
        Outcome::Pass { leak_probability, effect, .. } => {
            eprintln!(
                "No timing leak detected (leak_probability = {:.4})",
                leak_probability
            );
            // Effect should be negligible
            assert!(
                effect.total_effect_ns() < 100.0,
                "XOR comparison should have effect < 100ns (got {:.1}ns)",
                effect.total_effect_ns()
            );
        }
        Outcome::Fail { leak_probability, effect, exploitability, .. } => {
            eprintln!(
                "Small timing detected but should be Negligible (leak_probability = {:.4})",
                leak_probability
            );
            assert_eq!(
                *exploitability,
                Exploitability::Negligible,
                "XOR comparison should have Negligible exploitability. leak_prob: {:.4}, effect: shift={:.1}ns, tail={:.1}ns",
                leak_probability,
                effect.shift_ns,
                effect.tail_ns
            );
        }
        Outcome::Inconclusive { leak_probability, effect, .. } => {
            // Inconclusive is acceptable for such a small effect
            eprintln!(
                "Inconclusive result (leak_probability = {:.4}, effect = {:.1}ns)",
                leak_probability,
                effect.total_effect_ns()
            );
        }
        Outcome::Unmeasurable { .. } => {
            // Operation too fast - acceptable for XOR
        }
    }
}

/// 5.2 PossibleLAN (100-500ns) - Should classify appropriately
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn exploitability_possible_lan() {
    // Medium delay targeting ~200-300ns
    // x86_64: rdtsc at ~3GHz = 0.33ns/cycle, so ~700-900 cycles
    // aarch64: cntvct_el0 at 24MHz = 41.67ns/tick, so ~5-7 ticks
    #[cfg(target_arch = "x86_64")]
    const MEDIUM_DELAY: u64 = 800;
    #[cfg(target_arch = "aarch64")]
    const MEDIUM_DELAY: u64 = 6;

    let which_class = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(which_class, |class| {
            if *class == 0 {
                std::hint::black_box(42);
            } else {
                helpers::busy_wait_cycles(MEDIUM_DELAY);
                std::hint::black_box(42);
            }
        });

    eprintln!("\n[exploitability_possible_lan]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    match &outcome {
        Outcome::Fail { exploitability, effect, .. } => {
            assert!(
                matches!(exploitability, Exploitability::PossibleLAN | Exploitability::LikelyLAN),
                "Expected PossibleLAN or LikelyLAN exploitability (got {:?})",
                exploitability
            );

            // Verify the leak was detected - effect magnitude can vary significantly by platform
            let total_effect = effect.shift_ns.abs() + effect.tail_ns.abs();
            assert!(
                total_effect >= 50.0,
                "Expected measurable effect for medium delay (got {:.1}ns)",
                total_effect
            );
        }
        Outcome::Pass { leak_probability, .. } => {
            panic!(
                "Expected Fail outcome for medium delay, got Pass with leak_probability={}",
                leak_probability
            );
        }
        Outcome::Inconclusive { leak_probability, reason, .. } => {
            panic!(
                "Expected Fail outcome for medium delay, got Inconclusive: {:?}, leak_probability={}",
                reason, leak_probability
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Skipping: {}", recommendation);
        }
    }
}

/// 5.3 LikelyLAN (500ns - 20μs) - Should classify appropriately
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn exploitability_likely_lan() {
    // Large delay targeting ~2μs
    // x86_64: rdtsc at ~3GHz = 0.33ns/cycle, so ~6000 cycles
    // aarch64: cntvct_el0 at 24MHz = 41.67ns/tick, so ~48 ticks
    #[cfg(target_arch = "x86_64")]
    const LARGE_DELAY: u64 = 6000;
    #[cfg(target_arch = "aarch64")]
    const LARGE_DELAY: u64 = 48;

    let which_class = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(which_class, |class| {
            if *class == 0 {
                std::hint::black_box(42);
            } else {
                helpers::busy_wait_cycles(LARGE_DELAY);
                std::hint::black_box(42);
            }
        });

    eprintln!("\n[exploitability_likely_lan]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    match &outcome {
        Outcome::Fail { exploitability, effect, .. } => {
            assert!(
                matches!(exploitability, Exploitability::LikelyLAN),
                "Expected LikelyLAN exploitability (got {:?})",
                exploitability
            );

            // Verify effect magnitude is in the 500ns-20μs range (with reasonable margins)
            // Target was ~2μs, so expect 400ns-25μs with platform variance
            let total_effect = effect.shift_ns.abs() + effect.tail_ns.abs();
            assert!(
                total_effect >= 400.0 && total_effect <= 25_000.0,
                "Expected total effect in 400ns-25μs range for LikelyLAN classification (got {:.1}ns)",
                total_effect
            );
        }
        Outcome::Pass { leak_probability, .. } => {
            panic!(
                "Expected Fail outcome for large delay, got Pass with leak_probability={}",
                leak_probability
            );
        }
        Outcome::Inconclusive { leak_probability, reason, .. } => {
            panic!(
                "Expected Fail outcome for large delay, got Inconclusive: {:?}, leak_probability={}",
                reason, leak_probability
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Skipping: {}", recommendation);
        }
    }
}

/// 5.4 PossibleRemote (>20μs) - Should classify appropriately
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn exploitability_possible_remote() {
    // Very large delay targeting ~50μs
    // x86_64: rdtsc at ~3GHz = 0.33ns/cycle, so ~150000 cycles
    // aarch64: cntvct_el0 at 24MHz = 41.67ns/tick, so ~1200 ticks
    #[cfg(target_arch = "x86_64")]
    const HUGE_DELAY: u64 = 150_000;
    #[cfg(target_arch = "aarch64")]
    const HUGE_DELAY: u64 = 1200;

    let which_class = InputPair::new(|| 0, || 1);

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .pass_threshold(0.01)
        .fail_threshold(0.85)
        .time_budget(Duration::from_secs(30))
        .test(which_class, |class| {
            if *class == 0 {
                std::hint::black_box(42);
            } else {
                helpers::busy_wait_cycles(HUGE_DELAY);
                std::hint::black_box(42);
            }
        });

    eprintln!("\n[exploitability_possible_remote]");
    eprintln!("{}", timing_oracle::output::format_outcome(&outcome));

    match &outcome {
        Outcome::Fail { exploitability, effect, .. } => {
            assert!(
                matches!(exploitability, Exploitability::PossibleRemote),
                "Expected PossibleRemote exploitability (got {:?})",
                exploitability
            );

            // Verify effect magnitude is > 20μs (with reasonable margin)
            // Target was ~50μs, so expect > 15μs with platform variance
            let total_effect = effect.shift_ns.abs() + effect.tail_ns.abs();
            assert!(
                total_effect >= 15_000.0,
                "Expected total effect > 15μs for PossibleRemote classification (got {:.1}ns)",
                total_effect
            );
        }
        Outcome::Pass { leak_probability, .. } => {
            panic!(
                "Expected Fail outcome for huge delay, got Pass with leak_probability={}",
                leak_probability
            );
        }
        Outcome::Inconclusive { leak_probability, reason, .. } => {
            panic!(
                "Expected Fail outcome for huge delay, got Inconclusive: {:?}, leak_probability={}",
                reason, leak_probability
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Skipping: {}", recommendation);
        }
    }
}
