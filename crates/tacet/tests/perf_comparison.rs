//! Comparison test to verify mmap and syscall give similar results.
//!
//! Run this twice to compare:
//! ```bash
//! # Syscall baseline
//! sudo cargo test --release -p tacet --test perf_comparison -- --nocapture > syscall_results.txt
//!
//! # mmap version
//! sudo cargo test --release -p tacet --features perf-mmap --test perf_comparison -- --nocapture > mmap_results.txt
//!
//! # Compare the outputs
//! diff -u syscall_results.txt mmap_results.txt
//! ```

#[cfg(all(target_os = "linux", feature = "perf"))]
mod comparison {
    use tacet::measurement::perf::LinuxPerfTimer;

    #[test]
    fn perf_comparison_workload_1() {
        let mut timer = match LinuxPerfTimer::new() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to create timer: {}", e);
                return;
            }
        };

        // Run the workload many times to get stable statistics
        let mut measurements = Vec::new();
        for _ in 0..100 {
            let cycles = timer.measure_cycles(|| {
                let mut sum = 0u64;
                for i in 0..1000 {
                    sum = sum.wrapping_add(std::hint::black_box(i));
                }
                std::hint::black_box(sum)
            });
            measurements.push(cycles);
        }

        // Compute statistics
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
        let min = *measurements.iter().min().unwrap();
        let max = *measurements.iter().max().unwrap();
        let median = {
            let mut sorted = measurements.clone();
            sorted.sort();
            sorted[sorted.len() / 2]
        };

        // Output in a parseable format
        println!("\n=== WORKLOAD_1: 1000 adds ===");
        println!("MEAN: {:.2}", mean);
        println!("MEDIAN: {}", median);
        println!("STDDEV: {:.2}", std_dev);
        println!("MIN: {}", min);
        println!("MAX: {}", max);
        println!("CV: {:.4}", std_dev / mean);
        println!("SAMPLES: {}", measurements.len());

        // Features detected
        #[cfg(feature = "perf-mmap")]
        println!("MODE: mmap");

        #[cfg(not(feature = "perf-mmap"))]
        println!("MODE: syscall");
    }

    #[test]
    fn perf_comparison_workload_2() {
        let mut timer = match LinuxPerfTimer::new() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to create timer: {}", e);
                return;
            }
        };

        // Smaller workload
        let mut measurements = Vec::new();
        for _ in 0..100 {
            let cycles = timer.measure_cycles(|| {
                let mut sum = 0u64;
                for i in 0..100 {
                    sum = sum.wrapping_add(std::hint::black_box(i));
                }
                std::hint::black_box(sum)
            });
            measurements.push(cycles);
        }

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
        let min = *measurements.iter().min().unwrap();
        let max = *measurements.iter().max().unwrap();
        let median = {
            let mut sorted = measurements.clone();
            sorted.sort();
            sorted[sorted.len() / 2]
        };

        println!("\n=== WORKLOAD_2: 100 adds ===");
        println!("MEAN: {:.2}", mean);
        println!("MEDIAN: {}", median);
        println!("STDDEV: {:.2}", std_dev);
        println!("MIN: {}", min);
        println!("MAX: {}", max);
        println!("CV: {:.4}", std_dev / mean);
        println!("SAMPLES: {}", measurements.len());

        #[cfg(feature = "perf-mmap")]
        println!("MODE: mmap");

        #[cfg(not(feature = "perf-mmap"))]
        println!("MODE: syscall");
    }

    #[test]
    fn perf_comparison_workload_3() {
        let mut timer = match LinuxPerfTimer::new() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to create timer: {}", e);
                return;
            }
        };

        // Large workload
        let mut measurements = Vec::new();
        for _ in 0..100 {
            let cycles = timer.measure_cycles(|| {
                let mut sum = 0u64;
                for i in 0..10000 {
                    sum = sum.wrapping_add(std::hint::black_box(i));
                }
                std::hint::black_box(sum)
            });
            measurements.push(cycles);
        }

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
        let min = *measurements.iter().min().unwrap();
        let max = *measurements.iter().max().unwrap();
        let median = {
            let mut sorted = measurements.clone();
            sorted.sort();
            sorted[sorted.len() / 2]
        };

        println!("\n=== WORKLOAD_3: 10000 adds ===");
        println!("MEAN: {:.2}", mean);
        println!("MEDIAN: {}", median);
        println!("STDDEV: {:.2}", std_dev);
        println!("MIN: {}", min);
        println!("MAX: {}", max);
        println!("CV: {:.4}", std_dev / mean);
        println!("SAMPLES: {}", measurements.len());

        #[cfg(feature = "perf-mmap")]
        println!("MODE: mmap");

        #[cfg(not(feature = "perf-mmap"))]
        println!("MODE: syscall");
    }
}

#[cfg(not(all(target_os = "linux", feature = "perf")))]
mod comparison {
    #[test]
    fn comparison_requires_linux_perf() {
        // Test only runs on Linux with perf feature
    }
}
