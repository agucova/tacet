//! Stress calibration tests.
//!
//! These tests verify that the timing oracle maintains reasonable behavior
//! under CPU and memory pressure. Stress tests are disabled by default and
//! require CALIBRATION_ENABLE_STRESS=1 to run.
//!
//! Under stress conditions:
//! - FPR max is relaxed by +10% absolute
//! - Skip if unmeasurable rate > 30%
//!
//! See docs/calibration-test-spec.md for the full specification.

mod calibration_utils;

use calibration_utils::{CalibrationConfig, Decision, TrialRunner, select_attacker_model};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use timing_oracle::helpers::InputPair;
use timing_oracle::TimingOracle;

// =============================================================================
// STRESS CONFIGURATION
// =============================================================================

/// Maximum stress threads (hard cap per spec).
const MAX_STRESS_THREADS: usize = 4;

/// Maximum memory pressure in MB (hard cap per spec).
const MAX_MEMORY_MB: usize = 512;

/// Get the number of stress threads to use.
fn stress_threads() -> usize {
    let from_env: usize = std::env::var("CALIBRATION_MAX_STRESS_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    // Cap at num_cpus/4 and MAX_STRESS_THREADS
    let num_cpus = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    from_env.min(num_cpus / 4).min(MAX_STRESS_THREADS).max(1)
}

/// Get the memory pressure in MB to use.
fn stress_memory_mb() -> usize {
    let from_env: usize = std::env::var("CALIBRATION_MAX_MEMORY_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    from_env.min(MAX_MEMORY_MB)
}

/// Check if stress tests are enabled.
fn stress_enabled() -> bool {
    std::env::var("CALIBRATION_ENABLE_STRESS").as_deref() == Ok("1")
}

// =============================================================================
// STRESS GENERATORS
// =============================================================================

/// CPU stress worker that performs busy work with periodic yields.
fn cpu_stress_worker(stop: Arc<AtomicBool>, work_done: Arc<AtomicUsize>) {
    let mut counter: u64 = 0;
    while !stop.load(Ordering::Relaxed) {
        // Do some busy work
        for _ in 0..10_000 {
            counter = counter.wrapping_mul(6364136223846793005).wrapping_add(1);
            std::hint::black_box(counter);
        }
        work_done.fetch_add(1, Ordering::Relaxed);

        // Yield periodically to avoid starving the test thread
        std::thread::yield_now();
    }
}

/// Memory pressure generator that allocates and touches pages.
struct MemoryPressure {
    /// Allocated memory blocks.
    _blocks: Vec<Vec<u8>>,
    /// Total allocated size in bytes.
    size_bytes: usize,
}

impl MemoryPressure {
    /// Allocate memory pressure up to the specified MB.
    fn new(size_mb: usize) -> Self {
        let size_bytes = size_mb * 1024 * 1024;

        // Allocate in 1MB blocks to avoid huge allocations
        let block_size = 1024 * 1024;
        let num_blocks = size_mb;

        let mut blocks = Vec::with_capacity(num_blocks);

        for _ in 0..num_blocks {
            // Allocate and touch each page to ensure actual memory usage
            let mut block = vec![0u8; block_size];

            // Touch every 4KB page
            let page_size = 4096;
            for i in (0..block_size).step_by(page_size) {
                block[i] = 0xFF;
            }

            blocks.push(block);
        }

        Self {
            _blocks: blocks,
            size_bytes,
        }
    }

    fn size_mb(&self) -> usize {
        self.size_bytes / (1024 * 1024)
    }
}

/// Stress context that manages CPU and memory pressure.
struct StressContext {
    stop: Arc<AtomicBool>,
    threads: Vec<thread::JoinHandle<()>>,
    work_counters: Vec<Arc<AtomicUsize>>,
    memory: Option<MemoryPressure>,
}

impl StressContext {
    /// Create a new stress context with the specified CPU threads and memory.
    fn new(cpu_threads: usize, memory_mb: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let mut threads = Vec::with_capacity(cpu_threads);
        let mut work_counters = Vec::with_capacity(cpu_threads);

        // Start CPU stress threads
        for _ in 0..cpu_threads {
            let stop_clone = Arc::clone(&stop);
            let work_done = Arc::new(AtomicUsize::new(0));
            let work_done_clone = Arc::clone(&work_done);

            let handle = thread::spawn(move || {
                cpu_stress_worker(stop_clone, work_done_clone);
            });

            threads.push(handle);
            work_counters.push(work_done);
        }

        // Allocate memory pressure
        let memory = if memory_mb > 0 {
            Some(MemoryPressure::new(memory_mb))
        } else {
            None
        };

        Self {
            stop,
            threads,
            work_counters,
            memory,
        }
    }

    /// Get the number of CPU stress threads.
    fn cpu_threads(&self) -> usize {
        self.threads.len()
    }

    /// Get the memory pressure size in MB.
    fn memory_mb(&self) -> usize {
        self.memory.as_ref().map(|m| m.size_mb()).unwrap_or(0)
    }

    /// Get total work done across all threads.
    #[allow(dead_code)]
    fn total_work(&self) -> usize {
        self.work_counters.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }
}

impl Drop for StressContext {
    fn drop(&mut self) {
        // Signal threads to stop
        self.stop.store(true, Ordering::Relaxed);

        // Join all threads
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }
}

// =============================================================================
// STRESS TESTS
// =============================================================================

/// FPR under CPU stress.
///
/// Verifies that the oracle maintains bounded FPR even under CPU contention.
/// FPR threshold is relaxed by +10% absolute under stress.
#[test]
fn stress_fpr_cpu_contention() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[stress_fpr_cpu_contention] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    if !stress_enabled() {
        eprintln!("[stress_fpr_cpu_contention] Skipped: CALIBRATION_ENABLE_STRESS != 1");
        return;
    }

    let test_name = "stress_fpr_cpu_contention";
    let config = CalibrationConfig::from_env(test_name);
    let mut rng = config.rng();

    let cpu_threads = stress_threads();
    let trials = config.tier.fpr_trials() / 2;  // Fewer trials under stress

    // Start CPU stress
    let _stress = StressContext::new(cpu_threads, 0);

    eprintln!(
        "[{}] Starting {} trials with {} CPU stress threads (tier: {})",
        test_name, trials, cpu_threads, config.tier
    );

    let mut runner = TrialRunner::new(test_name, config.clone(), trials);

    for trial in 0..trials {
        if runner.should_stop() {
            eprintln!("[{}] Early stop at trial {}", test_name, trial);
            break;
        }

        // Both classes use fixed data - true null hypothesis
        let inputs = InputPair::new(
            || [0u8; 32],
            || [0u8; 32],
        );

        let outcome = TimingOracle::for_attacker(select_attacker_model(test_name))
            .max_samples(config.samples_per_trial)
            .time_budget(config.time_budget_per_trial)
            .test(inputs, |data| {
                let mut acc: u64 = 0;
                for _ in 0..100 {
                    for &b in data.iter() {
                        acc = acc.wrapping_add(std::hint::black_box(b) as u64);
                    }
                }
                std::hint::black_box(acc);
            });

        runner.record(&outcome);

        if (trial + 1) % 10 == 0 || trial + 1 == trials {
            eprintln!(
                "[{}] Trial {}/{}: {} failures ({:.1}% FPR)",
                test_name,
                trial + 1,
                trials,
                runner.fail_count(),
                runner.fpr() * 100.0
            );
        }

        rng = StdRng::seed_from_u64(rng.random());
    }

    // Stress tests allow +10% FPR, and higher unmeasurable rate
    let fpr = runner.fpr();
    let max_fpr_stressed = config.tier.max_fpr() + 0.10;
    let unmeasurable_rate = runner.unmeasurable_rate();

    let decision = if unmeasurable_rate > 0.30 {
        Decision::Skip("excessive_unmeasurable_rate_under_stress".into())
    } else if runner.completed_rate() < 0.50 {
        Decision::Skip("insufficient_completed_trials".into())
    } else if fpr <= max_fpr_stressed {
        Decision::Pass
    } else {
        Decision::Fail(format!(
            "FPR {:.1}% exceeds stressed max {:.0}%",
            fpr * 100.0,
            max_fpr_stressed * 100.0
        ))
    };

    let (_, report) = runner.finalize_fpr();
    report.print(&config);

    eprintln!(
        "[{}] Final: FPR {:.1}% (max {:.0}%), unmeasurable {:.1}% - {}",
        test_name,
        fpr * 100.0,
        max_fpr_stressed * 100.0,
        unmeasurable_rate * 100.0,
        decision
    );

    match decision {
        Decision::Pass => eprintln!("[{}] PASSED", test_name),
        Decision::Skip(reason) => eprintln!("[{}] SKIPPED: {}", test_name, reason),
        Decision::Fail(reason) => panic!("[{}] FAILED: {}", test_name, reason),
    }
}

/// FPR under memory pressure.
///
/// Verifies that the oracle maintains bounded FPR under memory pressure.
#[test]
fn stress_fpr_memory_pressure() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[stress_fpr_memory_pressure] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    if !stress_enabled() {
        eprintln!("[stress_fpr_memory_pressure] Skipped: CALIBRATION_ENABLE_STRESS != 1");
        return;
    }

    let test_name = "stress_fpr_memory_pressure";
    let config = CalibrationConfig::from_env(test_name);
    let mut rng = config.rng();

    let memory_mb = stress_memory_mb();
    let trials = config.tier.fpr_trials() / 2;

    // Start memory pressure
    let _stress = StressContext::new(0, memory_mb);

    eprintln!(
        "[{}] Starting {} trials with {}MB memory pressure (tier: {})",
        test_name, trials, memory_mb, config.tier
    );

    let mut runner = TrialRunner::new(test_name, config.clone(), trials);

    for trial in 0..trials {
        if runner.should_stop() {
            eprintln!("[{}] Early stop at trial {}", test_name, trial);
            break;
        }

        let inputs = InputPair::new(
            || [0u8; 32],
            || [0u8; 32],
        );

        let outcome = TimingOracle::for_attacker(select_attacker_model(test_name))
            .max_samples(config.samples_per_trial)
            .time_budget(config.time_budget_per_trial)
            .test(inputs, |data| {
                let mut acc: u64 = 0;
                for _ in 0..100 {
                    for &b in data.iter() {
                        acc = acc.wrapping_add(std::hint::black_box(b) as u64);
                    }
                }
                std::hint::black_box(acc);
            });

        runner.record(&outcome);

        if (trial + 1) % 10 == 0 || trial + 1 == trials {
            eprintln!(
                "[{}] Trial {}/{}: {} failures ({:.1}% FPR)",
                test_name,
                trial + 1,
                trials,
                runner.fail_count(),
                runner.fpr() * 100.0
            );
        }

        rng = StdRng::seed_from_u64(rng.random());
    }

    let fpr = runner.fpr();
    let max_fpr_stressed = config.tier.max_fpr() + 0.10;
    let unmeasurable_rate = runner.unmeasurable_rate();

    let decision = if unmeasurable_rate > 0.30 {
        Decision::Skip("excessive_unmeasurable_rate_under_stress".into())
    } else if runner.completed_rate() < 0.50 {
        Decision::Skip("insufficient_completed_trials".into())
    } else if fpr <= max_fpr_stressed {
        Decision::Pass
    } else {
        Decision::Fail(format!(
            "FPR {:.1}% exceeds stressed max {:.0}%",
            fpr * 100.0,
            max_fpr_stressed * 100.0
        ))
    };

    let (_, report) = runner.finalize_fpr();
    report.print(&config);

    eprintln!(
        "[{}] Final: FPR {:.1}% (max {:.0}%), unmeasurable {:.1}% - {}",
        test_name,
        fpr * 100.0,
        max_fpr_stressed * 100.0,
        unmeasurable_rate * 100.0,
        decision
    );

    match decision {
        Decision::Pass => eprintln!("[{}] PASSED", test_name),
        Decision::Skip(reason) => eprintln!("[{}] SKIPPED: {}", test_name, reason),
        Decision::Fail(reason) => panic!("[{}] FAILED: {}", test_name, reason),
    }
}

/// FPR under combined CPU and memory stress.
///
/// Most stressful test: both CPU contention and memory pressure.
#[test]
fn stress_fpr_combined() {
    if CalibrationConfig::is_disabled() {
        eprintln!("[stress_fpr_combined] Skipped: CALIBRATION_DISABLED=1");
        return;
    }

    if !stress_enabled() {
        eprintln!("[stress_fpr_combined] Skipped: CALIBRATION_ENABLE_STRESS != 1");
        return;
    }

    let test_name = "stress_fpr_combined";
    let config = CalibrationConfig::from_env(test_name);
    let mut rng = config.rng();

    let cpu_threads = stress_threads();
    let memory_mb = stress_memory_mb();
    let trials = config.tier.fpr_trials() / 2;

    // Start combined stress
    let _stress = StressContext::new(cpu_threads, memory_mb);

    eprintln!(
        "[{}] Starting {} trials with {} CPU threads + {}MB memory (tier: {})",
        test_name, trials, cpu_threads, memory_mb, config.tier
    );

    let mut runner = TrialRunner::new(test_name, config.clone(), trials);

    for trial in 0..trials {
        if runner.should_stop() {
            eprintln!("[{}] Early stop at trial {}", test_name, trial);
            break;
        }

        let inputs = InputPair::new(
            || [0u8; 32],
            || [0u8; 32],
        );

        let outcome = TimingOracle::for_attacker(select_attacker_model(test_name))
            .max_samples(config.samples_per_trial)
            .time_budget(config.time_budget_per_trial)
            .test(inputs, |data| {
                let mut acc: u64 = 0;
                for _ in 0..100 {
                    for &b in data.iter() {
                        acc = acc.wrapping_add(std::hint::black_box(b) as u64);
                    }
                }
                std::hint::black_box(acc);
            });

        runner.record(&outcome);

        if (trial + 1) % 10 == 0 || trial + 1 == trials {
            eprintln!(
                "[{}] Trial {}/{}: {} failures ({:.1}% FPR)",
                test_name,
                trial + 1,
                trials,
                runner.fail_count(),
                runner.fpr() * 100.0
            );
        }

        rng = StdRng::seed_from_u64(rng.random());
    }

    let fpr = runner.fpr();
    let max_fpr_stressed = config.tier.max_fpr() + 0.10;
    let unmeasurable_rate = runner.unmeasurable_rate();

    let decision = if unmeasurable_rate > 0.30 {
        Decision::Skip("excessive_unmeasurable_rate_under_stress".into())
    } else if runner.completed_rate() < 0.50 {
        Decision::Skip("insufficient_completed_trials".into())
    } else if fpr <= max_fpr_stressed {
        Decision::Pass
    } else {
        Decision::Fail(format!(
            "FPR {:.1}% exceeds stressed max {:.0}%",
            fpr * 100.0,
            max_fpr_stressed * 100.0
        ))
    };

    let (_, report) = runner.finalize_fpr();
    report.print(&config);

    eprintln!(
        "[{}] Final: FPR {:.1}% (max {:.0}%), unmeasurable {:.1}% - {}",
        test_name,
        fpr * 100.0,
        max_fpr_stressed * 100.0,
        unmeasurable_rate * 100.0,
        decision
    );

    match decision {
        Decision::Pass => eprintln!("[{}] PASSED", test_name),
        Decision::Skip(reason) => eprintln!("[{}] SKIPPED: {}", test_name, reason),
        Decision::Fail(reason) => panic!("[{}] FAILED: {}", test_name, reason),
    }
}
