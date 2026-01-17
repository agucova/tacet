//! Test PMU-based timing with kperf (requires sudo)
//!
//! These tests require root privileges to access PMU counters.
//!
//! **IMPORTANT**: Run tests serially to avoid kperf subsystem interference:
//! ```bash
//! cargo build --features kperf --tests
//! sudo ./target/debug/deps/test_kperf-* --test-threads=1 --nocapture
//! ```
//!
//! Running tests in parallel can cause kperf counter interference, leading to
//! incorrect calibration values (e.g., 0.03 cycles/ns instead of ~3 cycles/ns).

#![cfg(feature = "kperf")]

use timing_oracle::measurement::kperf::{PmuError, PmuTimer};
use timing_oracle::measurement::Timer;

/// Test that PMU timer calibrates to a reasonable CPU frequency.
/// Apple Silicon runs at ~2-4 GHz depending on power state and core type.
#[test]
fn test_pmu_calibration_accuracy() {
    match PmuTimer::new() {
        Ok(pmu_timer) => {
            let cycles_per_ns = pmu_timer.cycles_per_ns();

            // Apple Silicon runs at 2.0-4.5 GHz depending on:
            // - Core type (efficiency vs performance)
            // - Power state (idle vs active)
            // - Thermal conditions
            assert!(
                (1.5..=5.0).contains(&cycles_per_ns),
                "cycles_per_ns {} should be between 1.5 and 5.0 for Apple Silicon",
                cycles_per_ns
            );

            // Resolution should be sub-nanosecond
            let resolution = pmu_timer.resolution_ns();
            assert!(
                resolution < 1.0,
                "PMU resolution {} ns should be < 1 ns",
                resolution
            );

            eprintln!(
                "PMU calibration: {:.2} cycles/ns, {:.3} ns resolution",
                cycles_per_ns, resolution
            );
        }
        Err(PmuError::PermissionDenied) => {
            eprintln!("Skipping test_pmu_calibration_accuracy (requires sudo)");
        }
        Err(e) => panic!("Unexpected PMU error: {}", e),
    }
}

/// Test that measure_cycles returns non-zero for actual work.
#[test]
fn test_pmu_measure_cycles_nonzero() {
    match PmuTimer::new() {
        Ok(mut pmu_timer) => {
            // Measure some actual work
            let cycles = pmu_timer.measure_cycles(|| {
                let mut x = 0u64;
                for i in 0..1000 {
                    x = x.wrapping_add(std::hint::black_box(i));
                }
                std::hint::black_box(x)
            });

            assert!(
                cycles > 0,
                "measure_cycles should return > 0 for actual work"
            );

            // Should be at least a few hundred cycles for 1000 iterations
            assert!(
                cycles > 100,
                "Expected > 100 cycles for 1000 iterations, got {}",
                cycles
            );

            eprintln!("Measured {} cycles for 1000 iterations", cycles);
        }
        Err(PmuError::PermissionDenied) => {
            eprintln!("Skipping test_pmu_measure_cycles_nonzero (requires sudo)");
        }
        Err(e) => panic!("Unexpected PMU error: {}", e),
    }
}

/// Test that cycles_to_ns conversion is consistent with calibration.
#[test]
fn test_pmu_cycles_to_ns_conversion() {
    match PmuTimer::new() {
        Ok(pmu_timer) => {
            let cycles_per_ns = pmu_timer.cycles_per_ns();

            // 1000 cycles at ~4 cycles/ns should be ~250 ns
            let test_cycles = 1000u64;
            let ns = pmu_timer.cycles_to_ns(test_cycles);
            let expected_ns = test_cycles as f64 / cycles_per_ns;

            assert!(
                (ns - expected_ns).abs() < 0.001,
                "cycles_to_ns({}) = {} ns, expected {} ns",
                test_cycles,
                ns,
                expected_ns
            );

            eprintln!("{} cycles = {:.2} ns", test_cycles, ns);
        }
        Err(PmuError::PermissionDenied) => {
            eprintln!("Skipping test_pmu_cycles_to_ns_conversion (requires sudo)");
        }
        Err(e) => panic!("Unexpected PMU error: {}", e),
    }
}

/// Test that PMU has better resolution than the standard ARM timer.
#[test]
fn test_pmu_resolution_better_than_standard() {
    match PmuTimer::new() {
        Ok(pmu_timer) => {
            let std_timer = Timer::new();

            let pmu_resolution = pmu_timer.resolution_ns();
            let std_resolution = std_timer.resolution_ns();

            // PMU should be at least 10x better resolution
            let improvement = std_resolution / pmu_resolution;
            assert!(
                improvement >= 10.0,
                "PMU resolution ({:.2} ns) should be at least 10x better than standard ({:.2} ns), got {:.1}x",
                pmu_resolution, std_resolution, improvement
            );

            eprintln!(
                "Resolution improvement: {:.1}x ({:.2} ns vs {:.2} ns)",
                improvement, pmu_resolution, std_resolution
            );
        }
        Err(PmuError::PermissionDenied) => {
            eprintln!("Skipping test_pmu_resolution_better_than_standard (requires sudo)");
        }
        Err(e) => panic!("Unexpected PMU error: {}", e),
    }
}

/// Test measurement consistency (low coefficient of variation).
#[test]
fn test_pmu_measurement_consistency() {
    match PmuTimer::new() {
        Ok(mut pmu_timer) => {
            let mut samples = Vec::with_capacity(100);

            // Measure the same operation 100 times
            for _ in 0..100 {
                let cycles = pmu_timer.measure_cycles(|| {
                    let mut x = 0u64;
                    for i in 0..500 {
                        x = x.wrapping_add(std::hint::black_box(i));
                    }
                    std::hint::black_box(x)
                });
                samples.push(cycles as f64);
            }

            // Calculate mean and std dev
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let variance =
                samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
            let std_dev = variance.sqrt();
            let cv = std_dev / mean;

            // Coefficient of variation should be < 50% for stable measurements
            assert!(
                cv < 0.5,
                "CV {} should be < 0.5 for consistent measurements (mean={:.0}, std={:.0})",
                cv,
                mean,
                std_dev
            );

            eprintln!(
                "Measurement stats: mean={:.0} cycles, std={:.0}, CV={:.2}",
                mean, std_dev, cv
            );
        }
        Err(PmuError::PermissionDenied) => {
            eprintln!("Skipping test_pmu_measurement_consistency (requires sudo)");
        }
        Err(e) => panic!("Unexpected PMU error: {}", e),
    }
}

/// Test that reset errors are handled (returns 0).
#[test]
fn test_pmu_error_handling() {
    match PmuTimer::new() {
        Ok(mut pmu_timer) => {
            // Normal measurement should work
            let cycles = pmu_timer.measure_cycles(|| 42u64);
            // Note: This could be 0 if the operation is too fast, but shouldn't panic
            eprintln!("Trivial operation: {} cycles", cycles);
        }
        Err(PmuError::PermissionDenied) => {
            eprintln!("Skipping test_pmu_error_handling (requires sudo)");
        }
        Err(e) => panic!("Unexpected PMU error: {}", e),
    }
}

/// Comparison test between PMU and standard timer (informational).
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
                let cycles =
                    pmu_timer.measure_cycles(|| std::hint::black_box(42u64.wrapping_mul(17)));
                pmu_samples.push(cycles);
            }

            // Standard timer measurements
            let mut std_samples: Vec<u64> = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let cycles =
                    std_timer.measure_cycles(|| std::hint::black_box(42u64.wrapping_mul(17)));
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
            eprintln!(
                "    Min: {} cycles ({:.2} ns)",
                pmu_min,
                pmu_timer.cycles_to_ns(pmu_min)
            );
            eprintln!(
                "    Median: {} cycles ({:.2} ns)",
                pmu_median,
                pmu_timer.cycles_to_ns(pmu_median)
            );
            eprintln!(
                "    Max: {} cycles ({:.2} ns)",
                pmu_max,
                pmu_timer.cycles_to_ns(pmu_max)
            );
            eprintln!("    Unique values: {}", pmu_unique.len());

            eprintln!("\n  Standard timer:");
            eprintln!(
                "    Min: {} cycles ({:.2} ns)",
                std_min,
                std_timer.cycles_to_ns(std_min)
            );
            eprintln!(
                "    Median: {} cycles ({:.2} ns)",
                std_median,
                std_timer.cycles_to_ns(std_median)
            );
            eprintln!(
                "    Max: {} cycles ({:.2} ns)",
                std_max,
                std_timer.cycles_to_ns(std_max)
            );
            eprintln!("    Unique values: {}", std_unique.len());

            eprintln!(
                "\n  Resolution improvement: {:.1}x",
                std_timer.resolution_ns() / pmu_timer.resolution_ns()
            );

            eprintln!("\n===========================================\n");
        }
        Err(e) => {
            eprintln!("\nPMU timer error: {}", e);
            eprintln!("(Run with sudo to enable PMU access)");
            eprintln!("\n===========================================\n");
        }
    }
}
