//! Temporary diagnostic test for MDE issue

use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::statistics::compute_deciles;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};

fn generate_sbox() -> [u8; 256] {
    let mut sbox = [0u8; 256];
    for i in 0..256 {
        sbox[i] = (i as u8).wrapping_mul(0x1D).wrapping_add(0x63);
    }
    sbox
}

/// Check raw timing data first
#[test]
fn debug_raw_timing() {
    use timing_oracle::measurement::{Collector, Timer};

    let sbox = generate_sbox();
    let secret_key = 0xABu8;

    let indices = InputPair::new(|| secret_key, || rand::random::<u8>());

    let timer = Timer::new();
    let collector = Collector::with_timer(timer.clone(), 100);

    let (fixed_cycles, random_cycles, _batching_info) = collector.collect_separated(
        1000,
        || {
            let val = std::hint::black_box(indices.baseline());
            std::hint::black_box(sbox[val as usize])
        },
        || {
            let val = std::hint::black_box(indices.sample());
            std::hint::black_box(sbox[val as usize])
        },
    );

    // Convert to ns
    let fixed_ns: Vec<f64> = fixed_cycles
        .iter()
        .map(|&c| timer.cycles_to_ns(c))
        .collect();
    let random_ns: Vec<f64> = random_cycles
        .iter()
        .map(|&c| timer.cycles_to_ns(c))
        .collect();

    // Count unique values
    let mut fixed_sorted = fixed_ns.clone();
    fixed_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    fixed_sorted.dedup();

    let mut random_sorted = random_ns.clone();
    random_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    random_sorted.dedup();

    eprintln!("\n=== RAW TIMING DIAGNOSTICS ===");
    eprintln!("Cycles per ns: {}", timer.cycles_per_ns());
    eprintln!("Fixed samples: {}", fixed_ns.len());
    eprintln!("Random samples: {}", random_ns.len());
    eprintln!("Fixed unique values: {}", fixed_sorted.len());
    eprintln!("Random unique values: {}", random_sorted.len());

    // Show first 20 unique values
    eprintln!(
        "\nFixed unique values (first 20): {:?}",
        &fixed_sorted[..fixed_sorted.len().min(20)]
    );
    eprintln!(
        "Random unique values (first 20): {:?}",
        &random_sorted[..random_sorted.len().min(20)]
    );

    // Basic stats
    let fixed_min = fixed_ns.iter().cloned().fold(f64::INFINITY, f64::min);
    let fixed_max = fixed_ns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let random_min = random_ns.iter().cloned().fold(f64::INFINITY, f64::min);
    let random_max = random_ns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    eprintln!("\nFixed range: {} to {} ns", fixed_min, fixed_max);
    eprintln!("Random range: {} to {} ns", random_min, random_max);

    // Compute deciles
    let fixed_deciles = compute_deciles(&fixed_ns);
    let random_deciles = compute_deciles(&random_ns);

    eprintln!("\nFixed deciles:  {:?}", fixed_deciles.as_slice());
    eprintln!("Random deciles: {:?}", random_deciles.as_slice());
    eprintln!(
        "Difference:     {:?}",
        (fixed_deciles - random_deciles).as_slice()
    );
    eprintln!("==============================\n");
}

#[test]
fn debug_mde_issue() {
    let sbox = generate_sbox();
    let secret_key = 0xABu8;

    const SAMPLES: usize = 10_000;
    let indices = InputPair::new(|| secret_key, || rand::random::<u8>());

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(10))
        .max_samples(SAMPLES)
        .test(indices, |idx| {
        std::hint::black_box(());
        std::hint::black_box(sbox[*idx as usize]);
    });

    eprintln!("\n=== DIAGNOSTIC OUTPUT ===");

    match outcome {
        Outcome::Pass { leak_probability, effect, samples_used, quality, diagnostics } => {
            eprintln!("Outcome: PASS");
            eprintln!("Samples used: {}", samples_used);
            eprintln!("Quality: {:?}", quality);
            eprintln!("Discrete mode: {}", diagnostics.discrete_mode);
            eprintln!();
            eprintln!("Leak probability: {:.3}", leak_probability);
            eprintln!("Effect shift_ns: {:.2}", effect.shift_ns);
            eprintln!("Effect tail_ns: {:.2}", effect.tail_ns);
            eprintln!("Effect pattern: {:?}", effect.pattern);
            eprintln!("Effect CI: {:?}", effect.credible_interval_ns);
        }
        Outcome::Fail { leak_probability, effect, exploitability, samples_used, quality, diagnostics } => {
            eprintln!("Outcome: FAIL");
            eprintln!("Samples used: {}", samples_used);
            eprintln!("Quality: {:?}", quality);
            eprintln!("Discrete mode: {}", diagnostics.discrete_mode);
            eprintln!();
            eprintln!("Leak probability: {:.3}", leak_probability);
            eprintln!("Exploitability: {:?}", exploitability);
            eprintln!("Effect shift_ns: {:.2}", effect.shift_ns);
            eprintln!("Effect tail_ns: {:.2}", effect.tail_ns);
            eprintln!("Effect pattern: {:?}", effect.pattern);
            eprintln!("Effect CI: {:?}", effect.credible_interval_ns);
        }
        Outcome::Inconclusive { reason, leak_probability, effect, samples_used, quality, diagnostics } => {
            eprintln!("Outcome: INCONCLUSIVE");
            eprintln!("Reason: {:?}", reason);
            eprintln!("Samples used: {}", samples_used);
            eprintln!("Quality: {:?}", quality);
            eprintln!("Discrete mode: {}", diagnostics.discrete_mode);
            eprintln!();
            eprintln!("Leak probability: {:.3}", leak_probability);
            eprintln!("Effect shift_ns: {:.2}", effect.shift_ns);
            eprintln!("Effect tail_ns: {:.2}", effect.tail_ns);
            eprintln!("Effect pattern: {:?}", effect.pattern);
            eprintln!("Effect CI: {:?}", effect.credible_interval_ns);
        }
        Outcome::Unmeasurable { operation_ns, threshold_ns, platform, recommendation } => {
            eprintln!("Outcome: UNMEASURABLE");
            eprintln!("Operation: {:.1}ns", operation_ns);
            eprintln!("Threshold: {:.1}ns", threshold_ns);
            eprintln!("Platform: {}", platform);
            eprintln!("Recommendation: {}", recommendation);
            return; // Skip the rest
        }
    }

    eprintln!("=========================\n");
}
