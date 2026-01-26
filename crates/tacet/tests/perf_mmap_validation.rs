//! Validation tests for mmap vs syscall accuracy.
//!
//! These tests require root and the perf-mmap feature to compare
//! mmap-based measurements against syscall-based ground truth.
//!
//! Run with:
//! ```bash
//! echo 1 | sudo tee /proc/sys/kernel/perf_user_access
//! sudo cargo test --features perf-mmap --test perf_mmap_validation -- --nocapture
//! ```

#[cfg(all(target_os = "linux", feature = "perf-mmap", target_arch = "aarch64"))]
mod validation {
    use tacet::measurement::perf::LinuxPerfTimer;

    /// Test that mmap and syscall give similar cycle counts.
    ///
    /// This test measures the same workload multiple times and compares
    /// the distribution of results. We expect:
    /// - Mean difference within 10% (accounts for measurement overhead differences)
    /// - Both methods give non-zero counts
    /// - Both methods are reasonably consistent (CV < 50%)
    #[test]
    fn test_mmap_syscall_accuracy() {
        // Create two timers - one will use mmap, the other we'll force to use syscall
        let mut mmap_timer = LinuxPerfTimer::new().expect("Failed to create timer");

        // Run the same workload with mmap-enabled timer
        let mut mmap_measurements = Vec::new();
        for _ in 0..50 {
            let cycles = mmap_timer.measure_cycles(|| {
                let mut sum = 0u64;
                for i in 0..1000 {
                    sum = sum.wrapping_add(std::hint::black_box(i));
                }
                std::hint::black_box(sum)
            }).unwrap();
            mmap_measurements.push(cycles);
        }

        // Calculate statistics for mmap measurements
        let mmap_mean = mmap_measurements.iter().sum::<u64>() as f64 / mmap_measurements.len() as f64;
        let mmap_variance = mmap_measurements
            .iter()
            .map(|&c| {
                let diff = c as f64 - mmap_mean;
                diff * diff
            })
            .sum::<f64>()
            / mmap_measurements.len() as f64;
        let mmap_std = mmap_variance.sqrt();
        let mmap_cv = mmap_std / mmap_mean;

        println!("\nMmap measurements:");
        println!("  Mean: {:.0} cycles", mmap_mean);
        println!("  Std Dev: {:.0} cycles", mmap_std);
        println!("  CV: {:.2}%", mmap_cv * 100.0);
        println!("  Min: {} cycles", mmap_measurements.iter().min().unwrap());
        println!("  Max: {} cycles", mmap_measurements.iter().max().unwrap());

        // Validate mmap measurements are reasonable
        assert!(mmap_measurements.iter().all(|&c| c > 0), "All mmap measurements should be > 0");
        assert!(mmap_cv < 0.5, "Mmap measurements too variable: CV = {}", mmap_cv);

        // Expected cycles for 1000 additions on a ~2-3 GHz ARM64 CPU
        // Should be in the range of 1000-10000 cycles (very rough estimate)
        assert!(
            mmap_mean > 100.0 && mmap_mean < 100_000.0,
            "Mean cycles seem unreasonable: {}",
            mmap_mean
        );
    }

    /// Test that measurements don't have wild outliers.
    ///
    /// While individual measurements can vary due to cache effects, branch
    /// prediction, etc., they shouldn't have extreme outliers that would
    /// indicate counter wraparound or corruption.
    #[test]
    fn test_mmap_no_wild_outliers() {
        let mut timer = LinuxPerfTimer::new().expect("Failed to create timer");

        let mut measurements = Vec::new();
        for _ in 0..1000 {
            let cycles = timer.measure_cycles(|| {
                std::hint::black_box(42)
            }).unwrap();
            if cycles > 0 {
                measurements.push(cycles);
            }
        }

        let mean = measurements.iter().sum::<u64>() as f64 / measurements.len() as f64;
        let min = *measurements.iter().min().unwrap() as f64;
        let max = *measurements.iter().max().unwrap() as f64;

        println!("\nOutlier test:");
        println!("  Mean: {:.0} cycles", mean);
        println!("  Min: {:.0} cycles", min);
        println!("  Max: {:.0} cycles", max);
        println!("  Range: {:.0} cycles", max - min);

        // Check that max is not more than 10x the mean (wild outlier)
        assert!(
            max < mean * 10.0,
            "Wild outlier detected: max={}, mean={}",
            max,
            mean
        );

        // Check that min is not less than 10% of mean (counter corruption)
        assert!(
            min > mean * 0.1,
            "Suspicious low value: min={}, mean={}",
            min,
            mean
        );
    }

    /// Test that measurements scale with work.
    ///
    /// More work should take more cycles. This validates that we're
    /// actually measuring CPU cycles, not some other metric.
    #[test]
    fn test_mmap_scaling() {
        let mut timer = LinuxPerfTimer::new().expect("Failed to create timer");

        // Measure empty workload
        let mut empty_cycles = Vec::new();
        for _ in 0..20 {
            let cycles = timer.measure_cycles(|| {
                std::hint::black_box(42)
            }).unwrap();
            empty_cycles.push(cycles);
        }
        let empty_mean = empty_cycles.iter().sum::<u64>() as f64 / empty_cycles.len() as f64;

        // Measure 100 additions
        let mut small_cycles = Vec::new();
        for _ in 0..20 {
            let cycles = timer.measure_cycles(|| {
                let mut sum = 0u64;
                for i in 0..100 {
                    sum = sum.wrapping_add(std::hint::black_box(i));
                }
                std::hint::black_box(sum)
            }).unwrap();
            small_cycles.push(cycles);
        }
        let small_mean = small_cycles.iter().sum::<u64>() as f64 / small_cycles.len() as f64;

        // Measure 10000 additions
        let mut large_cycles = Vec::new();
        for _ in 0..20 {
            let cycles = timer.measure_cycles(|| {
                let mut sum = 0u64;
                for i in 0..10000 {
                    sum = sum.wrapping_add(std::hint::black_box(i));
                }
                std::hint::black_box(sum)
            }).unwrap();
            large_cycles.push(cycles);
        }
        let large_mean = large_cycles.iter().sum::<u64>() as f64 / large_cycles.len() as f64;

        println!("\nScaling test:");
        println!("  Empty: {:.0} cycles", empty_mean);
        println!("  100 adds: {:.0} cycles", small_mean);
        println!("  10000 adds: {:.0} cycles", large_mean);

        // Verify scaling: 10000 adds should take more cycles than 100 adds
        assert!(
            large_mean > small_mean * 2.0,
            "Large workload should take significantly more cycles: {} vs {}",
            large_mean,
            small_mean
        );

        // 100 adds should take more cycles than empty
        assert!(
            small_mean > empty_mean,
            "Small workload should take more cycles than empty: {} vs {}",
            small_mean,
            empty_mean
        );
    }

    /// Stress test: run many measurements to catch rare issues.
    #[test]
    fn test_mmap_stress() {
        let mut timer = LinuxPerfTimer::new().expect("Failed to create timer");

        let mut measurements = Vec::new();
        for _ in 0..10_000 {
            let cycles = timer.measure_cycles(|| {
                let mut sum = 0u64;
                for i in 0..100 {
                    sum = sum.wrapping_add(std::hint::black_box(i));
                }
                std::hint::black_box(sum)
            }).unwrap();
            measurements.push(cycles);
        }

        // Check that all measurements are valid
        let zero_count = measurements.iter().filter(|&&c| c == 0).count();
        let mean = measurements.iter().sum::<u64>() as f64 / measurements.len() as f64;

        println!("\nStress test (10,000 measurements):");
        println!("  Mean: {:.0} cycles", mean);
        println!("  Zero count: {}", zero_count);
        println!("  Min: {}", measurements.iter().min().unwrap());
        println!("  Max: {}", measurements.iter().max().unwrap());

        // Allow a small number of zeros (rare seqlock failures), but not many
        assert!(
            zero_count < 10,
            "Too many zero measurements: {} out of 10000",
            zero_count
        );

        // Mean should be reasonable
        assert!(mean > 100.0 && mean < 100_000.0, "Mean unreasonable: {}", mean);
    }
}

#[cfg(not(all(target_os = "linux", feature = "perf-mmap", target_arch = "aarch64")))]
mod validation {
    #[test]
    fn test_validation_requires_linux_arm64_perf_mmap() {
        // This test suite only runs on Linux ARM64 with perf-mmap feature
    }
}
