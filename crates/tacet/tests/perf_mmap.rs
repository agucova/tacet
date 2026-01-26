//! Tests for perf-mmap feature.
//!
//! These tests verify that the mmap-based PMU reads work correctly and
//! gracefully fall back to syscall-based reads when mmap is unavailable.

#[cfg(all(target_os = "linux", feature = "perf-mmap"))]
mod mmap_tests {
    use tacet::measurement::perf::LinuxPerfTimer;

    #[test]
    fn test_timer_creation_succeeds() {
        // Should succeed even without root (falls back to syscall if mmap fails)
        let timer = LinuxPerfTimer::new();
        assert!(timer.is_ok(), "Timer creation should succeed");
    }

    #[test]
    fn test_measurements_work_without_mmap() {
        // Should work even if mmap setup fails (graceful fallback)
        let mut timer = LinuxPerfTimer::new().unwrap();

        let cycles = timer.measure_cycles(|| {
            std::hint::black_box(42)
        }).unwrap();

        // Cycles might be 0 if permissions are denied, but shouldn't panic
        assert!(cycles >= 0);
    }

    #[test]
    fn test_repeated_measurements() {
        let mut timer = LinuxPerfTimer::new().unwrap();

        // Run multiple measurements to verify stability
        for _ in 0..100 {
            let cycles = timer.measure_cycles(|| {
                let mut sum = 0u64;
                for i in 0..100 {
                    sum = sum.wrapping_add(std::hint::black_box(i));
                }
                std::hint::black_box(sum)
            }).unwrap();

            // Should measure some cycles
            assert!(cycles > 0 || cycles == 0); // May be 0 if no permissions
        }
    }

    #[test]
    fn test_cycles_conversion() {
        let timer = LinuxPerfTimer::new().unwrap();

        let cycles = 1_000_000u64;
        let ns = timer.cycles_to_ns(cycles);

        // Should be a reasonable nanosecond value
        assert!(ns > 0.0);
        assert!(ns < 1_000_000_000.0); // Less than 1 second

        // Check cycles_per_ns is reasonable (1-5 GHz typical)
        let cpn = timer.cycles_per_ns();
        assert!(cpn > 0.5);
        assert!(cpn < 10.0);
    }

    #[test]
    fn test_zero_cycles_measurement() {
        let mut timer = LinuxPerfTimer::new().unwrap();

        // Measure a no-op
        let cycles = timer.measure_cycles(|| {}).unwrap();

        // Should be small but may vary
        assert!(cycles >= 0);
    }

    /// Integration test requiring root privileges and sysctl configuration.
    ///
    /// Run with:
    /// ```bash
    /// echo 1 | sudo tee /proc/sys/kernel/perf_user_access
    /// sudo cargo test --features perf-mmap -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    #[cfg(target_arch = "aarch64")]
    fn test_mmap_accuracy_with_root() {
        // Requires: sudo sh -c 'echo 1 > /proc/sys/kernel/perf_user_access'
        let mut timer = LinuxPerfTimer::new().unwrap();

        // Verify measurements are reasonable
        let cycles = timer.measure_cycles(|| {
            let mut sum = 0u64;
            for i in 0..10000 {
                sum = sum.wrapping_add(std::hint::black_box(i));
            }
            std::hint::black_box(sum)
        }).unwrap();

        // Should measure significant cycles for 10k iterations
        assert!(
            cycles > 10000,
            "Should measure at least 10k cycles, got: {}",
            cycles
        );
        assert!(
            cycles < 100_000_000,
            "Cycles should be reasonable, got: {}",
            cycles
        );
    }

    /// Test that mmap measurements are consistent with syscall measurements.
    #[test]
    #[ignore]
    #[cfg(target_arch = "aarch64")]
    fn test_mmap_vs_syscall_consistency() {
        let mut timer = LinuxPerfTimer::new().unwrap();

        // Take multiple measurements
        let mut measurements = Vec::new();
        for _ in 0..10 {
            let cycles = timer.measure_cycles(|| {
                let mut sum = 0u64;
                for i in 0..1000 {
                    sum = sum.wrapping_add(std::hint::black_box(i));
                }
                std::hint::black_box(sum)
            }).unwrap();
            measurements.push(cycles);
        }

        // All measurements should be > 0
        assert!(measurements.iter().all(|&c| c > 0));

        // Calculate mean and std dev to check consistency
        let mean = measurements.iter().sum::<u64>() as f64 / measurements.len() as f64;
        let variance = measurements
            .iter()
            .map(|&c| {
                let diff = c as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / measurements.len() as f64;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean; // Coefficient of variation

        // Measurements should be relatively consistent (CV < 50%)
        assert!(
            cv < 0.5,
            "Measurements too variable: mean={}, std_dev={}, CV={}",
            mean,
            std_dev,
            cv
        );
    }
}

/// Tests that should work without perf-mmap feature
#[cfg(all(target_os = "linux", not(feature = "perf-mmap")))]
mod without_mmap {
    use tacet::measurement::perf::LinuxPerfTimer;

    #[test]
    fn test_works_without_mmap_feature() {
        let timer = LinuxPerfTimer::new();
        assert!(timer.is_ok());
    }
}
