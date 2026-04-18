//! Collect real crypto timing measurements into BlockedData format.
//!
//! This module provides a minimal helper that runs a user-supplied crypto
//! closure under tacet's measurement infrastructure and produces raw
//! interleaved/blocked timing samples. The output shape mirrors what
//! `SweepRunner` feeds adapters for synthetic data, so every `ToolAdapter`
//! can consume it unchanged.
//!
//! The resulting `CollectedBlocked` carries the actual timer resolution so
//! downstream adapters (tacet via `analyze_raw_samples_with_resolution`)
//! can respect the quantization floor.
//!
//! # Intended use
//!
//! Registry entries in `crypto_registry::{tier1, tier2}` capture a crypto
//! setup (pools, keys, ciphers) and call [`run_collection`] with their
//! per-sample operation closure.

use crate::BlockedData;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::hint::black_box;
use std::time::Instant;
use tacet::measurement::affinity::AffinityGuard;
use tacet::measurement::TimerSpec;
use tacet::Class;

/// Raw measurements collected from a real crypto operation.
#[derive(Debug, Clone)]
pub struct CollectedBlocked {
    /// Baseline-class samples in nanoseconds.
    pub baseline_ns: Vec<f64>,
    /// Test-class (sample) samples in nanoseconds.
    pub test_ns: Vec<f64>,
    /// Timer tick size in nanoseconds (for tacet's `θ_tick`).
    pub timer_resolution_ns: f64,
    /// Wall-clock collection time in milliseconds.
    pub collection_time_ms: u64,
    /// Timer used for measurement (for CSV metadata).
    pub timer_name: &'static str,
}

impl CollectedBlocked {
    /// View as a `BlockedData` of `u64` nanoseconds, consumable by existing
    /// tool adapters.
    ///
    /// Every crypto operation in the registry takes ≥10 ns, so rounding to
    /// `u64` is within the tools' native precision. See `plan §Implementation`
    /// option (a) for the discussion.
    pub fn to_blocked_u64(&self) -> BlockedData {
        BlockedData {
            baseline: self.baseline_ns.iter().map(|&v| v.round() as u64).collect(),
            test: self.test_ns.iter().map(|&v| v.round() as u64).collect(),
        }
    }
}

/// Run a crypto operation under tacet's measurement harness and return the
/// interleaved timings split by class.
///
/// # Parameters
/// - `seed`: deterministic RNG seed for the interleaving schedule.
/// - `samples_per_class`: number of samples drawn from each class.
/// - `warmup_iterations`: warmup runs (alternating class) before collection.
/// - `op`: the per-sample operation. Receives the class so the closure can
///   dispatch into its own pre-built baseline/sample input pool. Both
///   branches MUST execute the same shape of work; only data may differ.
///
/// # Returns
/// A [`CollectedBlocked`] with per-class real-nanosecond timings and the
/// underlying timer resolution. All measurement, pinning, and timer setup
/// happens inside this function.
///
/// The closure is invoked inside `BoxedTimer::measure_cycles`; its work is
/// what gets timed. The caller is responsible for setup / tear-down outside
/// the closure (see `realistic::collect_realistic_dataset` for the analogue
/// on synthetic data).
pub fn run_collection<F>(
    seed: u64,
    samples_per_class: usize,
    warmup_iterations: usize,
    mut op: F,
) -> CollectedBlocked
where
    F: FnMut(Class),
{
    let wall_start = Instant::now();

    // Pin to the current CPU. On Linux this is critical for perf_event mmap
    // (migrations blank counters); on macOS it stabilises rdtsc/cntvct.
    let _affinity_guard = match AffinityGuard::try_pin() {
        tacet::measurement::affinity::AffinityResult::Pinned(g) => Some(g),
        tacet::measurement::affinity::AffinityResult::NotPinned { .. } => None,
    };

    let (mut timer, _fallback_reason) = TimerSpec::Auto.create_timer();
    let timer_name = timer.name();
    let resolution_ns = timer.resolution_ns();

    let mut rng = StdRng::seed_from_u64(seed);

    // Build interleaving schedule: equal baseline/sample, shuffled.
    let mut plan: Vec<Class> = Vec::with_capacity(samples_per_class * 2);
    plan.extend(std::iter::repeat_n(Class::Baseline, samples_per_class));
    plan.extend(std::iter::repeat_n(Class::Sample, samples_per_class));
    plan.shuffle(&mut rng);

    // Warmup: alternate classes so caches / branch predictors reach steady
    // state before we start recording.
    for i in 0..warmup_iterations {
        let class = if i % 2 == 0 {
            Class::Baseline
        } else {
            Class::Sample
        };
        black_box(timer.measure_cycles(|| op(class)).ok());
    }

    let mut baseline_ns: Vec<f64> = Vec::with_capacity(samples_per_class);
    let mut test_ns: Vec<f64> = Vec::with_capacity(samples_per_class);

    for &class in &plan {
        let result = timer.measure_cycles(|| op(class));
        let Ok(cycles) = result else {
            // Skip failed measurements (e.g., perf_event RetryExhausted) —
            // they would corrupt the distribution. Rare on x86_64 rdtsc.
            continue;
        };
        let ns = timer.cycles_to_ns(cycles);
        match class {
            Class::Baseline => baseline_ns.push(ns),
            Class::Sample => test_ns.push(ns),
        }
    }

    CollectedBlocked {
        baseline_ns,
        test_ns,
        timer_resolution_ns: resolution_ns,
        collection_time_ms: wall_start.elapsed().as_millis() as u64,
        timer_name,
    }
}
