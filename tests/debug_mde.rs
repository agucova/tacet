//! Temporary diagnostic test for MDE issue

use timing_oracle::helpers::InputPair;
use timing_oracle::statistics::compute_deciles;
use timing_oracle::TimingOracle;

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

    let outcome = TimingOracle::new().samples(SAMPLES).test(indices, |idx| {
        std::hint::black_box(());
        std::hint::black_box(sbox[*idx as usize]);
    });

    let result = timing_oracle::skip_if_unreliable!(outcome, "debug_mde_issue");

    eprintln!("\n=== DIAGNOSTIC OUTPUT ===");
    eprintln!("Samples per class: {}", result.metadata.samples_per_class);
    eprintln!("Cycles per ns: {}", result.metadata.cycles_per_ns);
    eprintln!("Timer: {}", result.metadata.timer);
    eprintln!(
        "Timer resolution: {} ns",
        result.metadata.timer_resolution_ns
    );
    eprintln!(
        "Batching enabled: {}, k={}",
        result.metadata.batching.enabled, result.metadata.batching.k
    );
    eprintln!("Outlier fraction: {}", result.outlier_fraction);
    eprintln!();
    eprintln!("MDE shift_ns: {}", result.min_detectable_effect.shift_ns);
    eprintln!("MDE tail_ns: {}", result.min_detectable_effect.tail_ns);
    eprintln!("Quality: {:?}", result.quality);
    eprintln!();
    eprintln!("CI Gate passed: {}", result.ci_gate.passed());
    eprintln!("CI Gate alpha: {}", result.ci_gate.alpha);
    eprintln!("CI Gate threshold: {:.2}", result.ci_gate.threshold);
    eprintln!("CI Gate max_observed: {:.2}", result.ci_gate.max_observed);
    eprintln!("CI Gate observed: {:?}", result.ci_gate.observed);
    eprintln!();
    eprintln!("Leak probability: {}", result.leak_probability);
    eprintln!("Exploitability: {:?}", result.exploitability);
    if let Some(ref effect) = result.effect {
        eprintln!("Effect shift_ns: {}", effect.shift_ns);
        eprintln!("Effect tail_ns: {}", effect.tail_ns);
        eprintln!("Effect pattern: {:?}", effect.pattern);
        eprintln!("Effect CI: {:?}", effect.credible_interval_ns);
    }
    eprintln!("=========================\n");
}
