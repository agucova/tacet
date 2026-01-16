//! Integration tests for Linux perf feature.
//!
//! Run with: cargo test --features perf --test test_perf
//!
//! Note: These tests may require elevated permissions (sudo) on some systems.

#![cfg(feature = "perf")]

use timing_oracle::measurement::perf::{LinuxPerfTimer, PerfError};

#[test]
#[cfg(target_os = "linux")]
fn test_perf_timer_initialization() {
    // This test checks if perf timer can be initialized
    // It may fail with PermissionDenied on systems without proper permissions
    match LinuxPerfTimer::new() {
        Ok(timer) => {
            // Successfully initialized
            eprintln!("perf timer initialized successfully");
            eprintln!("  cycles_per_ns: {:.2}", timer.cycles_per_ns());
            eprintln!("  resolution_ns: {:.2}", timer.resolution_ns());

            // Verify reasonable values
            assert!(
                timer.cycles_per_ns() > 0.5 && timer.cycles_per_ns() < 10.0,
                "cycles_per_ns should be reasonable: {}",
                timer.cycles_per_ns()
            );
            assert!(
                timer.resolution_ns() > 0.1 && timer.resolution_ns() < 5.0,
                "resolution_ns should be sub-nanosecond to few nanoseconds: {}",
                timer.resolution_ns()
            );
        }
        Err(PerfError::PermissionDenied) => {
            eprintln!("perf timer requires elevated permissions (expected on some systems)");
            eprintln!("Run with: sudo -E cargo test --features perf --test test_perf");
        }
        Err(e) => {
            panic!("Unexpected perf error: {}", e);
        }
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_perf_timer_measurement() {
    // Skip if permissions aren't available
    let mut timer = match LinuxPerfTimer::new() {
        Ok(t) => t,
        Err(PerfError::PermissionDenied) => {
            eprintln!("Skipping measurement test - requires elevated permissions");
            return;
        }
        Err(e) => panic!("Unexpected error: {}", e),
    };

    // Measure a simple operation
    let cycles = timer.measure_cycles(|| {
        let mut sum = 0u64;
        for i in 0..10000 {
            sum = sum.wrapping_add(std::hint::black_box(i));
        }
        std::hint::black_box(sum)
    });

    eprintln!("Measured {} cycles for 10k additions", cycles);
    assert!(cycles > 0, "Should measure non-zero cycles");

    // Verify conversion
    let ns = timer.cycles_to_ns(cycles);
    eprintln!("  = {:.1} ns", ns);
    assert!(ns > 0.0, "Should convert to positive nanoseconds");
}

#[test]
#[cfg(target_os = "linux")]
fn test_perf_timer_consistency() {
    // Skip if permissions aren't available
    let mut timer = match LinuxPerfTimer::new() {
        Ok(t) => t,
        Err(PerfError::PermissionDenied) => {
            eprintln!("Skipping consistency test - requires elevated permissions");
            return;
        }
        Err(e) => panic!("Unexpected error: {}", e),
    };

    // Measure the same operation 100 times to get good statistics
    let mut samples = Vec::with_capacity(100);
    for _ in 0..100 {
        let cycles = timer.measure_cycles(|| {
            let mut sum = 0u64;
            for i in 0..500 {
                sum = sum.wrapping_add(std::hint::black_box(i));
            }
            std::hint::black_box(sum)
        });
        samples.push(cycles as f64);
    }

    // All measurements should be non-zero
    assert!(
        samples.iter().all(|&c| c > 0.0),
        "All measurements should be non-zero"
    );

    // Calculate mean and std dev
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / samples.len() as f64;
    let std_dev = variance.sqrt();
    let cv = std_dev / mean;

    eprintln!("Measurement stats: mean={:.0} cycles, std={:.0}, CV={:.2}",
        mean, std_dev, cv);

    // Coefficient of variation should be < 50% for stable measurements
    assert!(
        cv < 0.5,
        "CV {} should be < 0.5 for consistent measurements (mean={:.0}, std={:.0})",
        cv, mean, std_dev
    );
}

#[test]
#[cfg(not(target_os = "linux"))]
fn test_perf_unsupported_platform() {
    // On non-Linux platforms, should return UnsupportedPlatform
    let result = LinuxPerfTimer::new();
    assert!(
        matches!(result, Err(PerfError::UnsupportedPlatform)),
        "Should return UnsupportedPlatform on non-Linux"
    );
}

#[test]
#[cfg(target_os = "linux")]
fn test_perf_timer_zero_work() {
    // Skip if permissions aren't available
    let mut timer = match LinuxPerfTimer::new() {
        Ok(t) => t,
        Err(PerfError::PermissionDenied) => {
            eprintln!("Skipping zero-work test - requires elevated permissions");
            return;
        }
        Err(e) => panic!("Unexpected error: {}", e),
    };

    // Measure essentially no work
    let cycles = timer.measure_cycles(|| {
        std::hint::black_box(42)
    });

    eprintln!("Measured {} cycles for minimal work", cycles);
    // Should be a small number (may be 0 or very small on some systems)
    assert!(
        cycles < 10000,
        "Minimal work should take < 10k cycles, got {}",
        cycles
    );
}

#[test]
#[cfg(target_os = "linux")]
fn test_perf_cycles_to_ns_conversion() {
    // Skip if permissions aren't available
    let timer = match LinuxPerfTimer::new() {
        Ok(t) => t,
        Err(PerfError::PermissionDenied) => {
            eprintln!("Skipping conversion test - requires elevated permissions");
            return;
        }
        Err(e) => panic!("Unexpected error: {}", e),
    };

    let cycles_per_ns = timer.cycles_per_ns();

    // Test conversion consistency
    let test_cycles = 1000u64;
    let ns = timer.cycles_to_ns(test_cycles);
    let expected_ns = test_cycles as f64 / cycles_per_ns;

    assert!(
        (ns - expected_ns).abs() < 0.001,
        "cycles_to_ns({}) = {} ns, expected {} ns",
        test_cycles, ns, expected_ns
    );

    eprintln!("{} cycles = {:.2} ns (at {:.2} cycles/ns)",
        test_cycles, ns, cycles_per_ns);
}

#[test]
#[cfg(target_os = "linux")]
fn test_perf_vs_standard_timer() {
    use timing_oracle::measurement::Timer;

    eprintln!("\n=== Perf vs Standard Timer Comparison ===\n");

    // Standard timer
    let std_timer = Timer::new();
    eprintln!("Standard timer:");
    eprintln!("  Cycles per ns: {:.4}", std_timer.cycles_per_ns());
    eprintln!("  Resolution: {:.2} ns", std_timer.resolution_ns());

    // Perf timer (requires permissions)
    match LinuxPerfTimer::new() {
        Ok(mut perf_timer) => {
            eprintln!("\nPerf timer (perf_event):");
            eprintln!("  Cycles per ns: {:.4}", perf_timer.cycles_per_ns());
            eprintln!("  Resolution: {:.4} ns", perf_timer.resolution_ns());

            // Measure a simple operation with both timers
            let iterations = 1000;

            // Perf measurements
            let mut perf_samples: Vec<u64> = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let cycles = perf_timer.measure_cycles(|| {
                    std::hint::black_box(42u64.wrapping_mul(17))
                });
                perf_samples.push(cycles);
            }

            // Standard timer measurements
            let mut std_samples: Vec<u64> = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let cycles = std_timer.measure_cycles(|| {
                    std::hint::black_box(42u64.wrapping_mul(17))
                });
                std_samples.push(cycles);
            }

            // Analyze perf samples
            perf_samples.sort();
            let perf_min = perf_samples[0];
            let perf_max = perf_samples[iterations - 1];
            let perf_median = perf_samples[iterations / 2];
            let perf_unique: std::collections::HashSet<_> = perf_samples.iter().collect();

            // Analyze standard samples
            std_samples.sort();
            let std_min = std_samples[0];
            let std_max = std_samples[iterations - 1];
            let std_median = std_samples[iterations / 2];
            let std_unique: std::collections::HashSet<_> = std_samples.iter().collect();

            eprintln!("\nMeasurement comparison ({} samples):", iterations);
            eprintln!("\n  Perf timer:");
            eprintln!("    Min: {} cycles ({:.2} ns)", perf_min, perf_timer.cycles_to_ns(perf_min));
            eprintln!("    Median: {} cycles ({:.2} ns)", perf_median, perf_timer.cycles_to_ns(perf_median));
            eprintln!("    Max: {} cycles ({:.2} ns)", perf_max, perf_timer.cycles_to_ns(perf_max));
            eprintln!("    Unique values: {}", perf_unique.len());

            eprintln!("\n  Standard timer:");
            eprintln!("    Min: {} cycles ({:.2} ns)", std_min, std_timer.cycles_to_ns(std_min));
            eprintln!("    Median: {} cycles ({:.2} ns)", std_median, std_timer.cycles_to_ns(std_median));
            eprintln!("    Max: {} cycles ({:.2} ns)", std_max, std_timer.cycles_to_ns(std_max));
            eprintln!("    Unique values: {}", std_unique.len());

            // Check if perf has better granularity
            if perf_timer.resolution_ns() < std_timer.resolution_ns() {
                let improvement = std_timer.resolution_ns() / perf_timer.resolution_ns();
                eprintln!("\n  Resolution improvement: {:.1}x", improvement);
            } else {
                eprintln!("\n  Note: Standard timer has similar or better resolution on this system");
            }

            eprintln!("\n===========================================\n");
        }
        Err(PerfError::PermissionDenied) => {
            eprintln!("\nPerf timer error: Permission denied");
            eprintln!("(Run with sudo to enable perf access)");
            eprintln!("\n===========================================\n");
        }
        Err(e) => panic!("Unexpected perf error: {}", e),
    }
}
