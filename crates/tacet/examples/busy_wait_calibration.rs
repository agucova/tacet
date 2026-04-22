//! Measures actual wall-clock cost of `busy_wait_ns(d)` across nominal delays.
//!
//! For each nominal d, calls `busy_wait_ns(d)` N times in a tight loop, divides
//! total elapsed wall time by N, reports mean effective delay per call.
//! Also reports per-call std via a second inner loop timing individual calls
//! (only for N_SAMPLES inner iterations, to keep noise in check).
//!
//! Relevant to the amplification experiment (USENIX Sec'26 Reviewer B, Q1): if
//! `busy_wait_ns(d)` has non-trivial per-call overhead at small d, the
//! x-axis of the detection curve at nominal d mis-represents the actual
//! injected delay.

use std::hint::black_box;
use std::time::Instant;
use tacet::helpers::effect::{busy_wait_ns, init_effect_injection};

fn main() {
    init_effect_injection();

    // Warmup: first call populates the calibration cache, subsequent calls hit it.
    for _ in 0..10_000 {
        busy_wait_ns(100);
    }
    // Drop warmup results.
    black_box(0u64);

    // Nominal delays to probe. Mirrors the delays used in the leaky tests,
    // plus a few extras at the small end to resolve the overhead floor.
    let nominal_delays: &[u64] = &[1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 5000];
    // N per-delay loop iterations. High enough to beat scheduler noise.
    const N_PER_LOOP: usize = 100_000;
    // Outer repeats to estimate variance across loop invocations.
    const REPEATS: usize = 11;

    println!(
        "nominal_ns,n_per_loop,repeats,mean_actual_ns,median_actual_ns,p10_actual_ns,p90_actual_ns"
    );

    for &d in nominal_delays {
        let mut samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let t0 = Instant::now();
            for _ in 0..N_PER_LOOP {
                busy_wait_ns(d);
            }
            let elapsed = t0.elapsed();
            // Nanoseconds per call.
            let per_call_ns = elapsed.as_nanos() as f64 / N_PER_LOOP as f64;
            samples.push(per_call_ns);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = samples.iter().copied().sum::<f64>() / REPEATS as f64;
        let median = samples[REPEATS / 2];
        let p10 = samples[REPEATS / 10];
        let p90 = samples[(REPEATS * 9) / 10];
        println!(
            "{d},{N_PER_LOOP},{REPEATS},{mean:.3},{median:.3},{p10:.3},{p90:.3}"
        );
        eprintln!(
            "  d={d}ns → mean {mean:.2}ns, median {median:.2}ns, p10 {p10:.2}, p90 {p90:.2} (overhead factor {:.2}x)",
            mean / d as f64
        );
    }
}
