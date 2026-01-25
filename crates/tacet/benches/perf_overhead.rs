//! Benchmark for perf_event measurement overhead.
//!
//! This benchmark measures the overhead of `measure_cycles()` with an empty
//! workload to quantify the performance difference between syscall-based
//! and mmap-based PMU reads.
//!
//! Expected results:
//! - Syscall-based (default): ~2000ns per measurement
//! - mmap-based (with perf-mmap feature): ~300ns per measurement
//! - Speedup: ~7x
//!
//! Run with:
//! ```bash
//! # Without mmap (syscall baseline)
//! cargo bench --bench perf_overhead
//!
//! # With mmap (requires root and sysctl)
//! echo 1 | sudo tee /proc/sys/kernel/perf_user_access
//! sudo cargo bench --bench perf_overhead --features perf-mmap
//! ```

use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(target_os = "linux")]
use std::hint::black_box;

#[cfg(target_os = "linux")]
use tacet::measurement::perf::LinuxPerfTimer;

#[cfg(target_os = "linux")]
fn bench_measurement_overhead(c: &mut Criterion) {
    let mut timer = match LinuxPerfTimer::new() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to create timer: {}", e);
            eprintln!("This benchmark requires elevated privileges (sudo) or CAP_PERFMON.");
            return;
        }
    };

    c.bench_function("measure_cycles_empty", |b| {
        b.iter(|| {
            timer.measure_cycles(|| black_box(42))
        })
    });

    c.bench_function("measure_cycles_100_adds", |b| {
        b.iter(|| {
            timer.measure_cycles(|| {
                let mut sum = 0u64;
                for i in 0..100 {
                    sum = sum.wrapping_add(black_box(i));
                }
                black_box(sum)
            })
        })
    });

    c.bench_function("measure_cycles_1000_adds", |b| {
        b.iter(|| {
            timer.measure_cycles(|| {
                let mut sum = 0u64;
                for i in 0..1000 {
                    sum = sum.wrapping_add(black_box(i));
                }
                black_box(sum)
            })
        })
    });
}

#[cfg(target_os = "linux")]
fn bench_raw_counter_read(c: &mut Criterion) {
    let mut timer = match LinuxPerfTimer::new() {
        Ok(t) => t,
        Err(_) => return,
    };

    // Benchmark just the counter read overhead (no workload)
    c.bench_function("counter_read_only", |b| {
        b.iter(|| {
            timer.measure_cycles(|| {})
        })
    });
}

#[cfg(target_os = "linux")]
fn bench_timer_calibration(c: &mut Criterion) {
    c.bench_function("timer_creation", |b| {
        b.iter(|| {
            black_box(LinuxPerfTimer::new().ok())
        })
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    benches,
    bench_measurement_overhead,
    bench_raw_counter_read,
    bench_timer_calibration
);

#[cfg(not(target_os = "linux"))]
fn bench_noop(_c: &mut Criterion) {
    // This benchmark is only available on Linux
}

#[cfg(not(target_os = "linux"))]
criterion_group!(benches, bench_noop);

criterion_main!(benches);
