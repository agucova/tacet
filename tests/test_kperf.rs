//! Test PMU-based timing with kperf (requires sudo)

#![cfg(feature = "kperf")]

use timing_oracle::measurement::kperf::PmuTimer;
use timing_oracle::measurement::Timer;

#[test]
fn test_pmu_vs_standard_timer() {
    eprintln!("\n=== PMU vs Standard Timer Comparison ===\n");

    // Standard timer (cntvct_el0)
    let std_timer = Timer::new();
    eprintln!("Standard timer (cntvct_el0):");
    eprintln!("  Cycles per ns: {:.4}", std_timer.cycles_per_ns());
    eprintln!("  Resolution: {:.2} ns", std_timer.resolution_ns());

    // PMU timer (requires root)
    match PmuTimer::new() {
        Ok(mut pmu_timer) => {
            eprintln!("\nPMU timer (kperf):");
            eprintln!("  Cycles per ns: {:.4}", pmu_timer.cycles_per_ns());
            eprintln!("  Resolution: {:.4} ns", pmu_timer.resolution_ns());

            // Measure a simple operation with both timers
            let iterations = 1000;

            // PMU measurements
            let mut pmu_samples: Vec<u64> = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let cycles = pmu_timer.measure_cycles(|| {
                    std::hint::black_box(42u64.wrapping_mul(17))
                });
                pmu_samples.push(cycles);
            }

            // Standard timer measurements
            let mut std_samples: Vec<u64> = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let cycles = std_timer.measure_cycles(|| {
                    std::hint::black_box(42u64.wrapping_mul(17))
                });
                std_samples.push(cycles);
            }

            // Analyze PMU samples
            pmu_samples.sort();
            let pmu_min = pmu_samples[0];
            let pmu_max = pmu_samples[iterations - 1];
            let pmu_median = pmu_samples[iterations / 2];
            let pmu_unique: std::collections::HashSet<_> = pmu_samples.iter().collect();

            // Analyze standard samples
            std_samples.sort();
            let std_min = std_samples[0];
            let std_max = std_samples[iterations - 1];
            let std_median = std_samples[iterations / 2];
            let std_unique: std::collections::HashSet<_> = std_samples.iter().collect();

            eprintln!("\nMeasurement comparison ({} samples):", iterations);
            eprintln!("\n  PMU timer:");
            eprintln!("    Min: {} cycles ({:.2} ns)", pmu_min, pmu_timer.cycles_to_ns(pmu_min));
            eprintln!("    Median: {} cycles ({:.2} ns)", pmu_median, pmu_timer.cycles_to_ns(pmu_median));
            eprintln!("    Max: {} cycles ({:.2} ns)", pmu_max, pmu_timer.cycles_to_ns(pmu_max));
            eprintln!("    Unique values: {}", pmu_unique.len());

            eprintln!("\n  Standard timer:");
            eprintln!("    Min: {} cycles ({:.2} ns)", std_min, std_timer.cycles_to_ns(std_min));
            eprintln!("    Median: {} cycles ({:.2} ns)", std_median, std_timer.cycles_to_ns(std_median));
            eprintln!("    Max: {} cycles ({:.2} ns)", std_max, std_timer.cycles_to_ns(std_max));
            eprintln!("    Unique values: {}", std_unique.len());

            eprintln!("\n  Resolution improvement: {:.1}x",
                std_timer.resolution_ns() / pmu_timer.resolution_ns());

            eprintln!("\n===========================================\n");
        }
        Err(e) => {
            eprintln!("\nPMU timer error: {}", e);
            eprintln!("(Run with sudo to enable PMU access)");
            eprintln!("\n===========================================\n");
        }
    }
}
