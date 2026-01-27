//! Effect injection utilities for timing tests and benchmarks.
//!
//! This module provides standardized ways to inject controlled timing effects
//! for testing timing side-channel detection tools. It uses platform-specific
//! high-precision busy-waiting to create accurate, measurable delays.
//!
//! # Usage
//!
//! ```ignore
//! use tacet::helpers::effect::{busy_wait_ns, EffectInjector};
//!
//! // Simple delay injection
//! busy_wait_ns(100); // Wait ~100 nanoseconds
//!
//! // With safety limits
//! let injector = EffectInjector::with_max_ns(10_000); // Max 10μs
//! injector.delay_ns(500); // Inject 500ns delay
//! ```
//!
//! # Platform Support
//!
//! | Platform | Timer | Resolution | Notes |
//! |----------|-------|------------|-------|
//! | aarch64 macOS | `cntvct_el0` | ~42ns | Direct assembly (kperf can't be used in parallel) |
//! | aarch64 Linux | `perf_event` | ~0.3ns | PMU cycles via mmap (requires sudo or fallback) |
//! | x86_64 Linux/macOS | RDTSC | ~0.3ns | ~3GHz TSC typical |
//! | Other | `Instant::now()` | ~50ns | Fallback, prints warning |
//!
//! The module automatically selects the best available timer and warns if
//! falling back to imprecise methods.

use std::sync::atomic::{AtomicU64, Ordering};
#[allow(unused_imports)]
use std::sync::OnceLock;
// Used by x86_64 TSC calibration and fallback path (conditionally compiled)
// Also used by tests on all platforms
#[allow(unused_imports)]
use std::time::{Duration, Instant};

// AtomicBool only used for fallback warning
#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))
)))]
use std::sync::atomic::AtomicBool;

// Linux ARM64: Use perf timer for accurate effect injection
#[cfg(all(target_os = "linux", target_arch = "aarch64", feature = "perf"))]
use crate::measurement::perf::LinuxPerfTimer;

// Linux ARM64: Get or initialize the effect timer
#[cfg(all(target_os = "linux", target_arch = "aarch64", feature = "perf"))]
fn get_effect_timer() -> Option<&'static LinuxPerfTimer> {
    use std::sync::OnceLock;
    use crate::measurement::perf::LinuxPerfTimer;

    static EFFECT_TIMER: OnceLock<Option<LinuxPerfTimer>> = OnceLock::new();

    EFFECT_TIMER.get_or_init(|| {
        // Try to create a perf_event timer for effect calibration
        // This uses syscall-based reads (not mmap) so no PMU multiplexing issues
        LinuxPerfTimer::new().ok()
    }).as_ref()
}

// =============================================================================
// BUNDLE CALIBRATION FOR EFFECT INJECTION
// =============================================================================

use std::cell::{Cell, LazyCell};

thread_local! {
    /// Track whether effect injection was explicitly initialized on this thread.
    ///
    /// Used to enforce explicit initialization before measurements to prevent
    /// calibration overhead from interfering with timing tests.
    static EFFECT_INITIALIZED: Cell<bool> = const { Cell::new(false) };

    /// Thread-local calibrated bundle costs for effect injection.
    ///
    /// Calibrates once per thread on first use. Uses perf timer if available,
    /// falls back to Instant-based measurement.
    static BUNDLE_CALIBRATION: LazyCell<Option<BundleCalibration>> = LazyCell::new(|| {
        eprintln!("[BUNDLE_CALIBRATION] Initializing...");

        // Try perf timer first (Linux ARM64 with perf feature)
        #[cfg(all(target_os = "linux", target_arch = "aarch64", feature = "perf"))]
        if let Some(timer) = get_effect_timer() {
            eprintln!("[BUNDLE_CALIBRATION] Using perf timer");
            return Some(BundleCalibration::calibrate_with_perf(timer));
        }

        // Fallback: calibrate using Instant (less accurate but always available)
        eprintln!("[BUNDLE_CALIBRATION] Using Instant timer (perf not available)");
        Some(BundleCalibration::calibrate_with_instant())
    });
}

/// Calibrated cost per spin iteration.
///
/// Pre-measures the cost of executing spin_bundle(1) many times,
/// allowing runtime execution without measurement overhead.
#[derive(Clone, Debug)]
#[allow(dead_code)] // Used only in perf-enabled configurations
struct BundleCalibration {
    /// Nanoseconds per single spin iteration.
    ///
    /// Derived from averaging 100,000+ samples to achieve sub-timer precision.
    cost_per_unit: f64,
}

impl BundleCalibration {
    /// Calibrate bundle costs using perf timer (most accurate).
    ///
    /// Uses large sample counts to achieve sub-timer-resolution precision.
    /// Even with perf_event cycle counting, measuring many iterations averages
    /// out any measurement overhead.
    #[cfg(all(target_os = "linux", target_arch = "aarch64", feature = "perf"))]
    fn calibrate_with_perf(timer: &LinuxPerfTimer) -> Self {
        const BASE_SIZE: u64 = 1;
        const LARGE_SAMPLE_COUNT: usize = 100_000;
        const MAX_REASONABLE_CYCLES: u64 = 10_000_000_000; // 10B cycles (~4s at 2.5GHz)

        let mut calibration_timer = LinuxPerfTimer::new().unwrap_or_else(|_| {
            panic!("Failed to create perf timer during calibration")
        });

        // Retry up to 3 times if we get bogus overflow values or measurement errors
        for _attempt in 0..3 {
            let result = calibration_timer.measure_cycles(|| {
                for _ in 0..LARGE_SAMPLE_COUNT {
                    spin_bundle(BASE_SIZE);
                }
            });

            // Skip invalid measurements
            let total_cycles = match result {
                Ok(cycles) => cycles,
                Err(e) => {
                    eprintln!("[calibrate_with_perf] Measurement error: {:?}, retrying", e);
                    continue;
                }
            };

            eprintln!("[calibrate_with_perf] total_cycles={}", total_cycles);

            // Reject obvious overflow/error values (close to u64::MAX or i64::MAX)
            if total_cycles > MAX_REASONABLE_CYCLES {
                eprintln!("[calibrate_with_perf] Rejected overflow value, retrying");
                continue;
            }

            let avg_cycles = total_cycles as f64 / LARGE_SAMPLE_COUNT as f64;
            let cost_per_unit = avg_cycles / timer.cycles_per_ns();

            eprintln!("[calibrate_with_perf] SUCCESS: cost_per_unit={:.2}ns (avg_cycles={:.2}, cycles_per_ns={:.2})",
                      cost_per_unit, avg_cycles, timer.cycles_per_ns());

            return Self { cost_per_unit };
        }

        // If all attempts failed, fall back to Instant-based calibration
        eprintln!("[calibrate_with_perf] All attempts failed, falling back to Instant");
        Self::calibrate_with_instant()
    }

    /// Calibrate bundle costs using Instant (fallback, less accurate).
    ///
    /// Uses large sample counts to achieve sub-timer-resolution precision.
    /// Even with a 42ns timer, measuring 100,000 iterations gives sub-ns accuracy
    /// as quantization errors average out.
    fn calibrate_with_instant() -> Self {
        const BASE_SIZE: u64 = 1;
        const LARGE_SAMPLE_COUNT: usize = 100_000;

        eprintln!("[calibrate_with_instant] Starting calibration...");

        let start = Instant::now();
        for _ in 0..LARGE_SAMPLE_COUNT {
            spin_bundle(BASE_SIZE);
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        let cost_per_unit = elapsed_ns / LARGE_SAMPLE_COUNT as f64;

        eprintln!("[calibrate_with_instant] SUCCESS: cost_per_unit={:.2}ns", cost_per_unit);

        Self { cost_per_unit }
    }

    /// Calculate exact number of iterations needed for target delay.
    ///
    /// Returns the iteration count that minimizes error vs target_ns.
    /// Uses the calibrated cost per unit iteration for sub-timer precision.
    #[allow(dead_code)] // Used only in bundle-calibrated configurations
    fn iterations_for_target(&self, target_ns: u64) -> u64 {
        if target_ns == 0 || self.cost_per_unit <= 0.0 {
            return 0;
        }

        let exact_iters = (target_ns as f64) / self.cost_per_unit;
        exact_iters.round().max(0.0) as u64
    }
}

/// Execute a spin-loop bundle without measurement.
///
/// This is the core operation that gets calibrated and executed at runtime.
/// Uses spin_loop + black_box to create measurable delay without optimization.
#[inline(never)]
fn spin_bundle(iterations: u64) {
    for _ in 0..iterations {
        std::hint::spin_loop();
        std::hint::black_box(());
    }
}

/// Initialize effect injection calibration.
///
/// **REQUIRED**: Must be called before using `busy_wait_ns()` or `EffectInjector`.
/// Calling effect injection functions without initialization will panic.
///
/// This function performs one-time calibration of bundle costs for the current thread.
/// Call this **before** running timing tests that use effect injection to avoid
/// calibration overhead during measurements.
///
/// # When to Use
///
/// - **In tests**: Call at the start of each test that uses effect injection
/// - **In benchmarks**: Call during setup, before measurement phase
/// - **In libraries**: Call in initialization code before effect injection
///
/// # Performance
///
/// Takes approximately 10-50ms depending on timer availability:
/// - Linux ARM64 with perf: ~10ms (uses PMU)
/// - Other platforms: ~20-50ms (uses Instant)
///
/// # Thread Safety
///
/// Calibration is thread-local. Call once per thread that will inject effects.
///
/// # Panics
///
/// Does not panic. However, calling `busy_wait_ns()` without initialization will panic.
///
/// # Example
///
/// ```ignore
/// use tacet::helpers::effect::{init_effect_injection, busy_wait_ns};
///
/// // In test setup - REQUIRED
/// init_effect_injection();
///
/// // Now effect injection is calibrated
/// busy_wait_ns(100);
/// ```
pub fn init_effect_injection() {
    // Mark as initialized for this thread
    EFFECT_INITIALIZED.with(|initialized| initialized.set(true));

    // Trigger calibration by accessing the thread-local
    BUNDLE_CALIBRATION.with(|_| {
        // Calibration happens in LazyCell::new() closure
    });

    // Run validation during initialization (not during first busy_wait_ns call)
    // This ensures all overhead happens upfront, keeping busy_wait_ns latency consistent
    validate_busy_wait_once();
}

// =============================================================================
// PLATFORM DETECTION AND WARNINGS
// =============================================================================

/// Track whether we've already printed the fallback warning.
#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))
)))]
static FALLBACK_WARNING_PRINTED: AtomicBool = AtomicBool::new(false);

/// Print a one-time warning when using the imprecise fallback timer.
#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))
)))]
fn warn_imprecise_fallback() {
    if !FALLBACK_WARNING_PRINTED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[tacet::effect] WARNING: Using Instant::now() fallback for busy_wait_ns. \
             This has ~50ns overhead and poor precision. For accurate effect injection, \
             use aarch64 or x86_64 on Linux/macOS."
        );
    }
}

/// Returns the name of the timer backend being used.
pub fn timer_backend_name() -> &'static str {
    #[cfg(all(target_os = "linux", target_arch = "aarch64", feature = "perf"))]
    {
        if get_effect_timer().is_some() {
            "perf_event"
        } else {
            "cntvct_el0"
        }
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "cntvct_el0"
    }
    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    {
        "rdtsc"
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))
    )))]
    {
        "instant_fallback"
    }
}

/// Returns true if using a precise hardware counter, false if using fallback.
pub fn using_precise_timer() -> bool {
    #[cfg(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))
    ))]
    {
        true
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))
    )))]
    {
        false
    }
}

/// Returns the detected counter frequency in Hz.
///
/// This is useful for debugging and understanding the timer precision.
/// The frequency determines the minimum measurable delay:
/// - 24 MHz → ~42ns resolution (Apple Silicon)
/// - 1 GHz → ~1ns resolution (AWS Graviton)
/// - 3 GHz → ~0.33ns resolution (x86_64 TSC)
///
/// # Returns
/// - The counter frequency in Hz for precise timers
/// - 0 for the fallback timer (no meaningful frequency)
///
/// # Note
/// This function is re-exported from `tacet_core::timer` for backward compatibility.
pub use tacet_core::timer::counter_frequency_hz;

/// Returns the timer resolution in nanoseconds.
///
/// This is the theoretical minimum delay that can be measured,
/// calculated as 1e9 / frequency.
///
/// # Returns
/// - Resolution in nanoseconds (e.g., 0.33 for 3GHz TSC)
/// - f64::INFINITY for the fallback timer
///
/// # Note
/// This function is re-exported from `tacet_core::timer` for backward compatibility.
pub use tacet_core::timer::timer_resolution_ns;

/// Compute minimum reliably injectable effect based on timer resolution.
///
/// Returns the smallest effect size that can be distinguished from
/// timer quantization noise. This ensures injected effects span multiple
/// quantization levels for reliable measurement.
///
/// # Formula
///
/// `min_injectable = timer_resolution × 5.0`
///
/// The 5× multiplier ensures effects span at least 5 quantization levels,
/// providing clear separation between effect sizes.
///
/// # Returns
///
/// Minimum injectable effect in nanoseconds, clamped to [20ns, 500ns] for
/// cross-platform consistency.
pub fn min_injectable_effect_ns() -> f64 {
    let resolution = timer_resolution_ns();

    // Conservative multiplier: 5× timer resolution
    // This ensures effects span multiple quantization levels
    let floor = resolution * 5.0;

    // Clamp to reasonable range:
    // - Minimum: 20ns (allows testing on most platforms)
    // - Maximum: 500ns (avoids making effects too large)
    floor.clamp(20.0, 500.0)
}

// =============================================================================
// ARM64 COUNTER ACCESS
// =============================================================================

/// Get the ARM64 counter frequency in Hz.
///
/// # Platform Notes
///
/// - **macOS (Apple Silicon)**: Reads from CNTFRQ_EL0 register.
///   - M1/M2 chips: 24 MHz (24,000,000 Hz)
///   - M3/M4+ chips (macOS 15+): 1 GHz (1,000,000,000 Hz)
///     See: https://developer.apple.com/documentation/macos-release-notes/macos-15-release-notes
///
/// - **Linux**: Reads from CNTFRQ_EL0 register. This can be unreliable on some
///   platforms due to firmware bugs (e.g., some older Raspberry Pi firmware,
///   early QEMU versions). We validate the value and fall back to known
///   frequencies for common platforms.
///
/// - **Known ARM64 frequencies**:
///   - Apple M3/M4+: 1 GHz (1,000,000,000 Hz)
///   - Apple M1/M2: 24 MHz (24,000,000 Hz)
///   - AWS Graviton: 1 GHz (1,000,000,000 Hz)
///   - Raspberry Pi 4: 54 MHz (54,000,000 Hz)
///   - QEMU default: 62.5 MHz (62,500,000 Hz)
///   - Generic ARM: Often 24 MHz or 100 MHz
#[cfg(target_arch = "aarch64")]
fn get_aarch64_counter_freq_hz() -> u64 {
    // Read counter frequency from CNTFRQ_EL0 register (works on both macOS and Linux)
    let freq: u64;
    unsafe {
        core::arch::asm!("mrs {}, cntfrq_el0", out(reg) freq);
    }

    // Validate: CNTFRQ_EL0 can be incorrectly programmed by firmware
    // Valid range: 1 MHz to 10 GHz
    if is_reasonable_aarch64_freq(freq) {
        return freq;
    }

    // CNTFRQ_EL0 looks wrong - try platform-specific fallbacks
    eprintln!(
        "[tacet::effect] WARNING: CNTFRQ_EL0 returned suspicious value: {} Hz",
        freq
    );

    #[cfg(target_os = "linux")]
    if let Some(platform_freq) = detect_aarch64_platform_freq() {
        eprintln!(
            "[tacet::effect] INFO: Using platform-detected frequency: {} Hz",
            platform_freq
        );
        return platform_freq;
    }

    #[cfg(target_os = "macos")]
    {
        // Fallback for older macOS or unusual configurations
        // Try 1GHz first (M3+), then 24MHz (M1/M2)
        eprintln!("[tacet::effect] INFO: Using macOS fallback frequency: 1 GHz");
        1_000_000_000
    }

    // Ultimate fallback: calibrate against Instant::now()
    #[cfg(target_os = "linux")]
    {
        eprintln!("[tacet::effect] INFO: Calibrating ARM64 counter frequency...");
        return calibrate_aarch64_frequency();
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        // Fallback: assume 24MHz (may be inaccurate)
        24_000_000
    }
}

/// Check if an ARM64 counter frequency is reasonable (1 MHz to 10 GHz).
#[cfg(target_arch = "aarch64")]
fn is_reasonable_aarch64_freq(freq: u64) -> bool {
    (1_000_000..=10_000_000_000).contains(&freq)
}

/// Try to detect ARM64 platform from /proc/cpuinfo and return known frequency.
///
/// NOTE: This is only used as a fallback when CNTFRQ_EL0 returns an unreasonable
/// value (outside 1 MHz - 10 GHz range). If CNTFRQ_EL0 is valid, it will be used
/// directly, even if it differs from these platform defaults.
///
/// For example, Neoverse systems may report 25 MHz, 50 MHz, or 1 GHz depending on
/// the SoC integrator's choice - all are valid if reported by CNTFRQ_EL0.
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn detect_aarch64_platform_freq() -> Option<u64> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let cpuinfo_lower = cpuinfo.to_lowercase();

    // NOTE: These are fallback defaults for when CNTFRQ_EL0 is broken
    // Actual systems may use different frequencies - trust CNTFRQ_EL0 if valid!

    // AWS Graviton (Neoverse cores) - common default
    if cpuinfo_lower.contains("neoverse") || cpuinfo_lower.contains("graviton") {
        return Some(1_000_000_000); // 1 GHz (common, but not universal)
    }

    // Raspberry Pi 4 (Cortex-A72)
    if cpuinfo_lower.contains("raspberry pi 4") || cpuinfo_lower.contains("bcm2711") {
        return Some(54_000_000); // 54 MHz
    }

    // Raspberry Pi 3 (Cortex-A53)
    if cpuinfo_lower.contains("raspberry pi 3") || cpuinfo_lower.contains("bcm2837") {
        return Some(19_200_000); // 19.2 MHz (crystal oscillator)
    }

    // Ampere Altra
    if cpuinfo_lower.contains("ampere") || cpuinfo_lower.contains("altra") {
        return Some(1_000_000_000); // 1 GHz
    }

    // Qualcomm Snapdragon (various)
    if cpuinfo_lower.contains("qualcomm") || cpuinfo_lower.contains("snapdragon") {
        return Some(19_200_000); // Common 19.2 MHz
    }

    None
}

/// Calibrate ARM64 counter frequency by measuring against Instant::now().
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn calibrate_aarch64_frequency() -> u64 {
    use std::time::{Duration, Instant};

    const SAMPLES: usize = 5;
    const SLEEP_MS: u64 = 20;

    let mut frequencies = Vec::with_capacity(SAMPLES);

    for _ in 0..SAMPLES {
        let start_cnt: u64;
        unsafe {
            core::arch::asm!("mrs {}, cntvct_el0", out(reg) start_cnt);
        }
        let start_instant = Instant::now();

        std::thread::sleep(Duration::from_millis(SLEEP_MS));

        let end_cnt: u64;
        unsafe {
            core::arch::asm!("mrs {}, cntvct_el0", out(reg) end_cnt);
        }
        let elapsed_ns = start_instant.elapsed().as_nanos() as u64;

        let cnt_delta = end_cnt.wrapping_sub(start_cnt);
        let freq = ((cnt_delta as u128 * 1_000_000_000) / elapsed_ns as u128) as u64;

        if is_reasonable_aarch64_freq(freq) {
            frequencies.push(freq);
        }
    }

    if frequencies.is_empty() {
        eprintln!("[tacet::effect] WARNING: ARM64 calibration failed. Using 24MHz estimate.");
        return 24_000_000;
    }

    // Use median for robustness
    frequencies.sort_unstable();
    let median = frequencies[frequencies.len() / 2];

    eprintln!(
        "[tacet::effect] INFO: ARM64 counter frequency calibrated to {:.2} MHz",
        median as f64 / 1_000_000.0
    );

    median
}

/// Cached counter frequency for ARM64.
#[cfg(target_arch = "aarch64")]
static AARCH64_COUNTER_FREQ: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// Get the cached ARM64 counter frequency.
#[cfg(target_arch = "aarch64")]
fn aarch64_counter_freq() -> u64 {
    *AARCH64_COUNTER_FREQ.get_or_init(get_aarch64_counter_freq_hz)
}

/// Convert nanoseconds to counter ticks for ARM64.
///
/// Note: On Apple Silicon, the physical clock runs at 24MHz regardless of the
/// reported frequency. On M3+ with macOS 15+, the kernel scales CNTVCT_EL0 values
/// so they appear to run at 1GHz (incrementing by ~41.67 per physical tick).
/// The actual measurement granularity remains ~42ns on all Apple Silicon.
#[cfg(target_arch = "aarch64")]
#[inline]
fn ns_to_ticks_aarch64(ns: u64) -> u64 {
    let freq = aarch64_counter_freq();

    if freq == 1_000_000_000 {
        // 1GHz fast path (M3+ with macOS 15+, AWS Graviton, etc.): 1 tick = 1 ns
        ns
    } else if freq == 24_000_000 {
        // 24MHz fast path (M1/M2 Apple Silicon): ticks = ns * 24 / 1000
        (ns * 24).div_ceil(1000)
    } else {
        // General case: ticks = ns * freq / 1e9
        // Use u128 to avoid overflow for large ns values
        ((ns as u128 * freq as u128).div_ceil(1_000_000_000)) as u64
    }
}

// =============================================================================
// X86_64 RDTSC ACCESS
// =============================================================================

/// Cached TSC frequency for x86_64 (Hz).
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
static X86_64_TSC_FREQ: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// Get x86_64 TSC frequency using the most reliable method available.
///
/// Priority order:
/// 1. Linux: /sys/devices/system/cpu/cpu0/tsc_freq_khz (kernel-provided, most accurate)
/// 2. Linux: /proc/cpuinfo "cpu MHz" (may reflect current P-state, not TSC)
/// 3. macOS: sysctl machdep.tsc.frequency
/// 4. Fallback: calibration against Instant::now()
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn get_x86_64_tsc_freq_hz() -> u64 {
    // Try platform-specific reliable sources first
    #[cfg(target_os = "linux")]
    if let Some(freq) = get_tsc_freq_linux() {
        return freq;
    }

    #[cfg(target_os = "macos")]
    if let Some(freq) = get_tsc_freq_macos() {
        return freq;
    }

    // Fallback: calibration
    calibrate_tsc_frequency()
}

/// Linux: Try to get TSC frequency from sysfs or CPUID.
///
/// Note: We intentionally do NOT use /proc/cpuinfo "cpu MHz" because:
/// - It may report current frequency (varies with scaling/turbo) or boot frequency
/// - TSC on modern Intel CPUs with invariant TSC runs at base frequency
/// - See: https://lwn.net/Articles/162548/
///
/// Fallback chain:
/// 1. sysfs tsc_freq_khz (most reliable, but not always available)
/// 2. CPUID leaf 0x16 base frequency (Skylake+, accurate for invariant TSC)
/// 3. Calibration against Instant::now() (always accurate, but slower)
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn get_tsc_freq_linux() -> Option<u64> {
    // Method 1: sysfs tsc_freq_khz (most reliable when available)
    // Only exposed in some kernels (Google's, or with tsc_freq_khz driver)
    if let Ok(content) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/tsc_freq_khz") {
        if let Ok(khz) = content.trim().parse::<u64>() {
            let freq = khz * 1000;
            if is_reasonable_tsc_freq(freq) {
                return Some(freq);
            }
        }
    }

    // Method 2: CPUID leaf 0x16 - Processor Frequency Information (Skylake+)
    // Returns base frequency in MHz, which matches TSC on invariant TSC systems
    // Only use if CPU has invariant TSC (constant_tsc flag)
    if has_invariant_tsc() {
        if let Some(freq) = get_cpuid_base_freq() {
            if is_reasonable_tsc_freq(freq) {
                return Some(freq);
            }
        }
    }

    // Fall back to calibration
    None
}

/// Check if CPU has invariant TSC (constant rate regardless of frequency scaling).
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn has_invariant_tsc() -> bool {
    // Check CPUID.80000007H:EDX[8] - Invariant TSC
    // Also indicated by constant_tsc and nonstop_tsc flags in /proc/cpuinfo
    let result: u32;
    unsafe {
        // Note: ebx is reserved by LLVM for PIC, so we save/restore it manually
        core::arch::asm!(
            "push rbx",
            "mov eax, 0x80000007",
            "cpuid",
            "pop rbx",
            out("edx") result,
            out("eax") _,
            out("ecx") _,
            options(nostack)
        );
    }
    (result & (1 << 8)) != 0
}

/// Get processor base frequency from CPUID leaf 0x16 (Skylake+).
/// Returns frequency in Hz, or None if not available.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn get_cpuid_base_freq() -> Option<u64> {
    // First check if leaf 0x16 is supported
    let max_leaf: u32;
    unsafe {
        // Note: ebx is reserved by LLVM for PIC, so we save/restore it manually
        core::arch::asm!(
            "push rbx",
            "mov eax, 0",
            "cpuid",
            "pop rbx",
            out("eax") max_leaf,
            out("ecx") _,
            out("edx") _,
            options(nostack)
        );
    }

    if max_leaf < 0x16 {
        return None; // CPUID leaf 0x16 not supported (pre-Skylake)
    }

    // CPUID leaf 0x16: Processor Frequency Information
    // EAX = Base frequency in MHz
    // EBX = Max frequency in MHz
    // ECX = Bus/reference frequency in MHz
    let base_mhz: u32;
    unsafe {
        core::arch::asm!(
            "push rbx",
            "mov eax, 0x16",
            "cpuid",
            "pop rbx",
            out("eax") base_mhz,
            out("ecx") _,
            out("edx") _,
            options(nostack)
        );
    }

    if base_mhz > 0 {
        Some(base_mhz as u64 * 1_000_000)
    } else {
        None
    }
}

/// macOS: Try to get TSC frequency from sysctl.
#[cfg(all(target_arch = "x86_64", target_os = "macos"))]
fn get_tsc_freq_macos() -> Option<u64> {
    // Try machdep.tsc.frequency first (most accurate)
    if let Some(freq) = sysctl_read_u64("machdep.tsc.frequency") {
        if is_reasonable_tsc_freq(freq) {
            return Some(freq);
        }
    }

    // Fallback to hw.cpufrequency (base frequency, usually matches TSC)
    if let Some(freq) = sysctl_read_u64("hw.cpufrequency") {
        if is_reasonable_tsc_freq(freq) {
            return Some(freq);
        }
    }

    None
}

/// Read a u64 value from sysctl on macOS.
#[cfg(all(target_arch = "x86_64", target_os = "macos"))]
fn sysctl_read_u64(name: &str) -> Option<u64> {
    use std::process::Command;

    let output = Command::new("sysctl").arg("-n").arg(name).output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().parse::<u64>().ok()
}

/// Check if a TSC frequency is reasonable (500 MHz to 10 GHz).
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn is_reasonable_tsc_freq(freq: u64) -> bool {
    (500_000_000..=10_000_000_000).contains(&freq)
}

/// Fallback: Calibrate TSC frequency by measuring against Instant.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn calibrate_tsc_frequency() -> u64 {
    eprintln!("[tacet::effect] INFO: Calibrating TSC frequency (no sysfs/sysctl available)...");

    // Use multiple samples for better accuracy
    const SAMPLES: usize = 5;
    const SLEEP_MS: u64 = 20;

    let mut frequencies = Vec::with_capacity(SAMPLES);

    for _ in 0..SAMPLES {
        let start_tsc = rdtsc();
        let start_instant = Instant::now();

        std::thread::sleep(Duration::from_millis(SLEEP_MS));

        let end_tsc = rdtsc();
        let elapsed_ns = start_instant.elapsed().as_nanos() as u64;

        let tsc_delta = end_tsc.wrapping_sub(start_tsc);
        let freq = ((tsc_delta as u128 * 1_000_000_000) / elapsed_ns as u128) as u64;

        if is_reasonable_tsc_freq(freq) {
            frequencies.push(freq);
        }
    }

    if frequencies.is_empty() {
        eprintln!("[tacet::effect] WARNING: TSC calibration failed. Using 3GHz estimate.");
        return 3_000_000_000;
    }

    // Use median for robustness
    frequencies.sort_unstable();
    let median = frequencies[frequencies.len() / 2];

    eprintln!(
        "[tacet::effect] INFO: TSC frequency calibrated to {:.2} GHz",
        median as f64 / 1_000_000_000.0
    );

    median
}

/// Get cached x86_64 TSC frequency.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn x86_64_tsc_freq() -> u64 {
    *X86_64_TSC_FREQ.get_or_init(get_x86_64_tsc_freq_hz)
}

/// Read x86_64 TSC (Time Stamp Counter).
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[inline]
fn rdtsc() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        // RDTSC returns TSC in EDX:EAX
        core::arch::asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nostack, nomem)
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// Convert nanoseconds to TSC ticks for x86_64.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[inline]
fn ns_to_ticks_x86_64(ns: u64) -> u64 {
    let freq = x86_64_tsc_freq();
    // ticks = ns * freq / 1e9
    ((ns as u128 * freq as u128).div_ceil(1_000_000_000)) as u64
}

// =============================================================================
// BUSY WAIT IMPLEMENTATION
// =============================================================================

/// Global maximum delay limit (safety feature).
/// Default: 1ms. Can be adjusted via `set_global_max_delay_ns`.
static GLOBAL_MAX_DELAY_NS: AtomicU64 = AtomicU64::new(1_000_000);

/// Track whether we've validated busy_wait accuracy.
static BUSY_WAIT_VALIDATED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Validate busy_wait_ns accuracy on first use.
///
/// Runs a quick sanity check to ensure the counter frequency is correct.
/// If the actual delay differs significantly from the requested delay,
/// prints a warning (once) to help diagnose misconfigured timers.
fn validate_busy_wait_once() {
    BUSY_WAIT_VALIDATED.get_or_init(|| {
        // Skip validation on fallback platforms (no precise timer)
        if !using_precise_timer() {
            return true;
        }

        // Use a moderate delay that's measurable but fast
        const TARGET_NS: u64 = 10_000; // 10μs

        let start = Instant::now();

        // Temporarily bypass validation to avoid recursion
        #[cfg(target_arch = "aarch64")]
        {
            busy_wait_ns_aarch64(TARGET_NS);
        }
        #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
        {
            busy_wait_ns_x86_64(TARGET_NS);
        }

        let actual_ns = start.elapsed().as_nanos() as u64;
        let ratio = actual_ns as f64 / TARGET_NS as f64;

        // Allow wide tolerance: 0.3x to 10x
        // - Below 0.3x suggests frequency is way too high (e.g., using 24MHz when it's 1GHz)
        // - Above 10x suggests frequency is way too low or severe system interference
        if !(0.3..=10.0).contains(&ratio) {
            eprintln!(
                "[tacet::effect] WARNING: busy_wait_ns accuracy issue detected!\n\
                 Requested: {}ns, Actual: {}ns, Ratio: {:.2}x\n\
                 Detected frequency: {} Hz ({})\n\
                 This may indicate incorrect counter frequency detection.\n\
                 Effect injection timing may be inaccurate.",
                TARGET_NS,
                actual_ns,
                ratio,
                counter_frequency_hz(),
                timer_backend_name()
            );
        }

        true
    });
}

/// Set the global maximum delay limit.
///
/// This is a safety feature to prevent accidentally injecting very long delays.
/// Default is 1ms (1,000,000 ns).
///
/// # Arguments
/// * `max_ns` - Maximum delay in nanoseconds (clamped to 10ms hard limit)
pub fn set_global_max_delay_ns(max_ns: u64) {
    // Hard limit: 10ms to prevent runaway delays
    let clamped = max_ns.min(10_000_000);
    GLOBAL_MAX_DELAY_NS.store(clamped, Ordering::Relaxed);
}

/// Get the current global maximum delay limit.
pub fn global_max_delay_ns() -> u64 {
    GLOBAL_MAX_DELAY_NS.load(Ordering::Relaxed)
}

/// Busy-wait for approximately the specified number of nanoseconds.
///
/// Uses platform-specific high-precision timing:
/// - ARM64: Direct counter register access (`cntvct_el0`)
/// - x86_64 Linux/macOS: RDTSC instruction
/// - Other: `Instant::now()` busy-loop (prints warning on first use)
///
/// The delay is clamped to the global maximum (default 1ms).
///
/// # Arguments
/// * `ns` - Delay duration in nanoseconds
///
/// # Platform Notes
///
/// | Platform | Method | Resolution |
/// |----------|--------|------------|
/// | aarch64 macOS (all) | cntvct_el0 | ~42ns (physical 24MHz clock) |
/// | aarch64 Linux | cntvct_el0 | ~1ns (1GHz typical) |
/// | x86_64 Linux/macOS | RDTSC | ~0.3ns (3GHz typical) |
/// | Other | Instant::now() | ~50ns (prints warning) |
///
/// Note: On Apple Silicon M3+ with macOS 15+, the counter is scaled to appear
/// as 1GHz, but the physical resolution remains ~42ns.
#[inline(never)]
pub fn busy_wait_ns(ns: u64) {
    // Require explicit initialization to prevent calibration during measurements
    EFFECT_INITIALIZED.with(|initialized| {
        if !initialized.get() {
            panic!(
                "Effect injection not initialized! Call init_effect_injection() before using busy_wait_ns().\n\
                 \n\
                 Example:\n\
                 use tacet::helpers::init_effect_injection;\n\
                 \n\
                 init_effect_injection();  // Required before first use\n\
                 busy_wait_ns(100);        // Now safe to use\n\
                 \n\
                 This ensures calibration doesn't interfere with timing measurements."
            );
        }
    });

    let max = GLOBAL_MAX_DELAY_NS.load(Ordering::Relaxed);
    let clamped = ns.min(max);

    if clamped == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        busy_wait_ns_aarch64(clamped);
    }

    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    {
        busy_wait_ns_x86_64(clamped);
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))
    )))]
    {
        warn_imprecise_fallback();
        busy_wait_ns_instant(clamped);
    }
}

/// ARM64-specific busy-wait using counter register.
///
/// Uses pre-calibrated bundle execution to avoid double measurement.
/// Falls back to frequency-based or Instant-based timing if calibration unavailable.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn busy_wait_ns_aarch64(ns: u64) {
    if ns == 0 {
        return;
    }

    // Try calibrated bundles first (avoids double measurement)
    let used_bundles = BUNDLE_CALIBRATION.with(|cal| {
        if let Some(cal) = cal.as_ref() {
            let iterations = cal.iterations_for_target(ns);

            if iterations > 0 {
                // Execute exact number of iterations (no repetitions needed)
                // Calibration measured cost per single iteration with high precision
                spin_bundle(iterations);
            }

            true
        } else {
            false
        }
    });

    if used_bundles {
        return;
    }

    // Fallback 1: Frequency-based timing using cntvct_el0
    // This works on all ARM64 platforms and doesn't require measurement during execution
    let freq = aarch64_counter_freq();
    if freq > 0 {
        let ticks = ns_to_ticks_aarch64(ns);

        let start: u64;
        unsafe {
            core::arch::asm!("mrs {}, cntvct_el0", out(reg) start);
        }

        loop {
            let now: u64;
            unsafe {
                core::arch::asm!("mrs {}, cntvct_el0", out(reg) now);
            }
            if now.wrapping_sub(start) >= ticks {
                break;
            }
            std::hint::spin_loop();
        }
        return;
    }

    // Fallback 2: Instant-based (least accurate, always available)
    let start = Instant::now();
    let target = Duration::from_nanos(ns);
    while start.elapsed() < target {
        std::hint::spin_loop();
    }
}

/// x86_64-specific busy-wait using RDTSC.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[inline(never)]
fn busy_wait_ns_x86_64(ns: u64) {
    let ticks = ns_to_ticks_x86_64(ns);

    let start = rdtsc();

    loop {
        let now = rdtsc();
        if now.wrapping_sub(start) >= ticks {
            break;
        }
        std::hint::spin_loop();
    }
}

/// Fallback busy-wait using Instant::now().
/// This is less precise (~50ns overhead) and should be avoided.
#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))
)))]
#[inline(never)]
fn busy_wait_ns_instant(ns: u64) {
    let start = Instant::now();
    let target = Duration::from_nanos(ns);
    while start.elapsed() < target {
        std::hint::spin_loop();
    }
}

// =============================================================================
// EFFECT INJECTOR
// =============================================================================

/// Controlled effect injector for timing tests.
///
/// Provides a safe, configurable way to inject timing effects with
/// built-in safety limits and common patterns.
///
/// # Example
///
/// ```ignore
/// use tacet::helpers::effect::EffectInjector;
///
/// let injector = EffectInjector::new();
///
/// // In a timing test:
/// TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
///     .test(inputs, |&should_delay| {
///         injector.conditional_delay(should_delay, 200);
///     });
/// ```
#[derive(Debug, Clone)]
pub struct EffectInjector {
    /// Maximum delay this injector will apply (in nanoseconds).
    max_ns: u64,
}

impl Default for EffectInjector {
    fn default() -> Self {
        Self::new()
    }
}

impl EffectInjector {
    /// Create a new effect injector with default safety limit (100μs).
    ///
    /// **Important**: Call `init_effect_injection()` before creating the first
    /// EffectInjector to ensure calibration doesn't interfere with timing measurements.
    pub fn new() -> Self {
        // Ensure calibration happens (if not already done)
        init_effect_injection();
        Self { max_ns: 100_000 }
    }

    /// Create an effect injector with a custom maximum delay.
    ///
    /// # Arguments
    /// * `max_ns` - Maximum delay in nanoseconds
    pub fn with_max_ns(max_ns: u64) -> Self {
        // Ensure calibration happens (if not already done)
        init_effect_injection();
        Self { max_ns }
    }

    /// Inject a fixed delay.
    ///
    /// The delay is clamped to this injector's maximum.
    #[inline]
    pub fn delay_ns(&self, ns: u64) {
        let clamped = ns.min(self.max_ns);
        busy_wait_ns(clamped);
    }

    /// Inject a conditional delay (common pattern for timing tests).
    ///
    /// If `condition` is true, delays by `delay_ns`.
    /// If false, no delay is applied.
    ///
    /// This is the standard pattern for baseline vs sample timing tests:
    /// - Baseline: condition=false (no delay)
    /// - Sample: condition=true (delay applied)
    #[inline]
    pub fn conditional_delay(&self, condition: bool, delay_ns: u64) {
        if condition {
            self.delay_ns(delay_ns);
        }
        // Always execute black_box to equalize non-timing side effects
        std::hint::black_box(());
    }

    /// Inject a delay proportional to a value.
    ///
    /// Useful for simulating data-dependent timing where larger values
    /// take longer to process.
    ///
    /// # Arguments
    /// * `value` - The value to scale by
    /// * `ns_per_unit` - Nanoseconds of delay per unit of value
    #[inline]
    pub fn proportional_delay(&self, value: u64, ns_per_unit: f64) {
        let delay = (value as f64 * ns_per_unit) as u64;
        self.delay_ns(delay);
    }

    /// Inject a delay based on Hamming weight (number of 1 bits).
    ///
    /// Simulates timing leaks where processing time depends on
    /// the number of set bits in the data.
    ///
    /// # Arguments
    /// * `data` - Byte slice to compute Hamming weight from
    /// * `ns_per_bit` - Nanoseconds of delay per set bit
    #[inline]
    pub fn hamming_weight_delay(&self, data: &[u8], ns_per_bit: f64) {
        let weight: u32 = data.iter().map(|b| b.count_ones()).sum();
        let delay = (weight as f64 * ns_per_bit) as u64;
        self.delay_ns(delay);
    }

    /// Get the maximum delay this injector will apply.
    pub fn max_ns(&self) -> u64 {
        self.max_ns
    }
}

// =============================================================================
// EFFECT TYPES FOR BENCHMARKING
// =============================================================================

/// Predefined effect patterns for standardized benchmarking.
///
/// These match common timing side-channel scenarios and provide
/// consistent test conditions across different tools.
#[derive(Debug, Clone, Copy)]
pub enum BenchmarkEffect {
    /// No effect - both classes have identical timing.
    /// Used for FPR (false positive rate) testing.
    Null,

    /// Fixed delay added to sample class.
    /// The most common and easily detectable pattern.
    FixedDelay {
        /// Delay in nanoseconds added to sample class
        delay_ns: u64,
    },

    /// Delay as a multiple of the attacker model threshold θ.
    /// Useful for power curve testing.
    ThetaMultiple {
        /// Multiplier (e.g., 2.0 = 2×θ)
        multiplier: f64,
        /// Base θ in nanoseconds
        theta_ns: f64,
    },

    /// Early-exit pattern: baseline takes longer than sample.
    /// Simulates early-exit comparison where matching data is slower.
    EarlyExit {
        /// Maximum delay (applied to baseline/matching case)
        max_delay_ns: u64,
    },

    /// Hamming weight dependent: delay proportional to set bits.
    HammingWeight {
        /// Nanoseconds per set bit
        ns_per_bit: f64,
    },

    /// Bimodal mixture: occasional slow operations.
    /// Models cache misses, branch mispredictions, syscalls.
    /// With probability `slow_prob`, adds `slow_delay_ns` instead of nothing.
    Bimodal {
        /// Probability of slow operation (e.g., 0.05 = 5%)
        slow_prob: f64,
        /// Delay in nanoseconds for slow operations
        slow_delay_ns: u64,
    },

    /// Variable delay sampled from a distribution.
    /// Simulates variance differences between classes.
    /// Delay is sampled from N(mean_ns, std_ns) clamped to [0, max_ns].
    VariableDelay {
        /// Mean delay in nanoseconds
        mean_ns: u64,
        /// Standard deviation in nanoseconds
        std_ns: u64,
    },

    /// Tail effect: occasional large delays.
    /// Similar to Bimodal but with a multiplicative tail.
    /// Models occasional OS interrupts, page faults, etc.
    TailEffect {
        /// Base delay in nanoseconds (always applied)
        base_delay_ns: u64,
        /// Probability of tail event (e.g., 0.05 = 5%)
        tail_prob: f64,
        /// Multiplier for tail events (e.g., 10.0 = 10x base)
        tail_mult: f64,
    },
}

impl BenchmarkEffect {
    /// Get the expected delay difference in nanoseconds (sample - baseline).
    ///
    /// Returns None for effects that depend on input data or are stochastic.
    pub fn expected_difference_ns(&self) -> Option<f64> {
        match self {
            BenchmarkEffect::Null => Some(0.0),
            BenchmarkEffect::FixedDelay { delay_ns } => Some(*delay_ns as f64),
            BenchmarkEffect::ThetaMultiple {
                multiplier,
                theta_ns,
            } => Some(multiplier * theta_ns),
            BenchmarkEffect::EarlyExit { max_delay_ns } => Some(-(*max_delay_ns as f64)), // Baseline slower
            BenchmarkEffect::HammingWeight { .. } => None, // Data-dependent
            // Stochastic effects return expected value
            BenchmarkEffect::Bimodal {
                slow_prob,
                slow_delay_ns,
            } => Some(slow_prob * (*slow_delay_ns as f64)),
            BenchmarkEffect::VariableDelay { mean_ns, .. } => Some(*mean_ns as f64),
            BenchmarkEffect::TailEffect {
                base_delay_ns,
                tail_prob,
                tail_mult,
            } => {
                // Expected value: base + prob * (mult - 1) * base
                let base = *base_delay_ns as f64;
                Some(base * (1.0 + tail_prob * (tail_mult - 1.0)))
            }
        }
    }

    /// Create a bimodal effect (occasional slow operations).
    pub fn bimodal(slow_prob: f64, slow_delay_ns: u64) -> Self {
        BenchmarkEffect::Bimodal {
            slow_prob,
            slow_delay_ns,
        }
    }

    /// Standard bimodal pattern: 5% probability of 10μs delay.
    pub fn bimodal_default() -> Self {
        Self::bimodal(0.05, 10_000)
    }

    /// Create a variable delay effect.
    pub fn variable_delay(mean_ns: u64, std_ns: u64) -> Self {
        BenchmarkEffect::VariableDelay { mean_ns, std_ns }
    }

    /// Create a tail effect (occasional large delays).
    pub fn tail_effect(base_delay_ns: u64, tail_prob: f64, tail_mult: f64) -> Self {
        BenchmarkEffect::TailEffect {
            base_delay_ns,
            tail_prob,
            tail_mult,
        }
    }

    /// Standard tail effect: 100ns base, 5% probability of 10x multiplier.
    pub fn tail_effect_default() -> Self {
        Self::tail_effect(100, 0.05, 10.0)
    }

    /// Create a FixedDelay effect at a specific θ multiple.
    pub fn at_theta_multiple(multiplier: f64, theta_ns: f64) -> Self {
        BenchmarkEffect::ThetaMultiple {
            multiplier,
            theta_ns,
        }
    }

    /// Common effect sizes for AdjacentNetwork (θ = 100ns).
    pub fn adjacent_network_effects() -> Vec<(f64, Self)> {
        vec![
            (0.5, Self::FixedDelay { delay_ns: 50 }),
            (1.0, Self::FixedDelay { delay_ns: 100 }),
            (2.0, Self::FixedDelay { delay_ns: 200 }),
            (5.0, Self::FixedDelay { delay_ns: 500 }),
            (10.0, Self::FixedDelay { delay_ns: 1000 }),
        ]
    }

    /// Common effect sizes for Research mode (small effects).
    pub fn research_effects() -> Vec<(f64, Self)> {
        vec![
            (1.0, Self::FixedDelay { delay_ns: 50 }),
            (2.0, Self::FixedDelay { delay_ns: 100 }),
            (5.0, Self::FixedDelay { delay_ns: 250 }),
            (10.0, Self::FixedDelay { delay_ns: 500 }),
            (20.0, Self::FixedDelay { delay_ns: 1000 }),
        ]
    }
}

/// Apply a benchmark effect to a measurement closure.
///
/// Returns a closure that can be used with `TimingOracle::test()`.
///
/// # Example
///
/// ```ignore
/// use tacet::helpers::effect::{BenchmarkEffect, apply_effect};
///
/// let effect = BenchmarkEffect::FixedDelay { delay_ns: 200 };
/// let inputs = InputPair::new(|| false, || true);
///
/// TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
///     .test(inputs, apply_effect(effect));
/// ```
pub fn apply_effect(effect: BenchmarkEffect) -> impl Fn(&bool) {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};
    use std::cell::RefCell;

    let injector = EffectInjector::new();

    // Thread-local RNG for stochastic effects
    thread_local! {
        static RNG: RefCell<rand::rngs::ThreadRng> = RefCell::new(rand::rng());
    }

    move |&is_sample: &bool| {
        match effect {
            BenchmarkEffect::Null => {
                std::hint::black_box(());
            }
            BenchmarkEffect::FixedDelay { delay_ns } => {
                injector.conditional_delay(is_sample, delay_ns);
            }
            BenchmarkEffect::ThetaMultiple {
                multiplier,
                theta_ns,
            } => {
                let delay = (multiplier * theta_ns) as u64;
                injector.conditional_delay(is_sample, delay);
            }
            BenchmarkEffect::EarlyExit { max_delay_ns } => {
                // Early exit: baseline (is_sample=false) is SLOWER
                injector.conditional_delay(!is_sample, max_delay_ns);
            }
            BenchmarkEffect::HammingWeight { .. } => {
                // This variant needs byte data, not bool
                // Use apply_effect_bytes for data-dependent effects
                std::hint::black_box(());
            }
            BenchmarkEffect::Bimodal {
                slow_prob,
                slow_delay_ns,
            } => {
                if is_sample {
                    RNG.with(|rng| {
                        if rng.borrow_mut().random::<f64>() < slow_prob {
                            injector.delay_ns(slow_delay_ns);
                        }
                    });
                }
            }
            BenchmarkEffect::VariableDelay { mean_ns, std_ns } => {
                if is_sample {
                    RNG.with(|rng| {
                        // Sample from normal distribution, clamp to >= 0
                        let normal = Normal::new(mean_ns as f64, std_ns as f64)
                            .unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
                        let delay: f64 = normal.sample(&mut *rng.borrow_mut());
                        let delay = delay.max(0.0) as u64;
                        injector.delay_ns(delay);
                    });
                }
            }
            BenchmarkEffect::TailEffect {
                base_delay_ns,
                tail_prob,
                tail_mult,
            } => {
                if is_sample {
                    RNG.with(|rng| {
                        let is_tail = rng.borrow_mut().random::<f64>() < tail_prob;
                        let delay = if is_tail {
                            (base_delay_ns as f64 * tail_mult) as u64
                        } else {
                            base_delay_ns
                        };
                        injector.delay_ns(delay);
                    });
                }
            }
        }
    }
}

/// Apply a data-dependent benchmark effect.
///
/// For effects that depend on the actual input data (like Hamming weight).
pub fn apply_effect_bytes(effect: BenchmarkEffect) -> impl Fn(&[u8; 32]) {
    let injector = EffectInjector::new();

    move |data: &[u8; 32]| {
        match effect {
            BenchmarkEffect::HammingWeight { ns_per_bit } => {
                injector.hamming_weight_delay(data, ns_per_bit);
            }
            BenchmarkEffect::FixedDelay { delay_ns } => {
                // For fixed delay with byte data, delay if not all zeros
                let is_sample = data.iter().any(|&b| b != 0);
                injector.conditional_delay(is_sample, delay_ns);
            }
            _ => {
                std::hint::black_box(data);
            }
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_busy_wait_returns_quickly_for_zero() {
        init_effect_injection();
        let start = Instant::now();
        busy_wait_ns(0);
        let elapsed = start.elapsed();
        // Very generous bound - we're just checking it doesn't hang
        assert!(
            elapsed < Duration::from_millis(10),
            "busy_wait_ns(0) took {:?}, should be nearly instant",
            elapsed
        );
    }

    #[test]
    #[ignore] // Timing-sensitive test, flaky in CI due to calibration overhead
    fn test_busy_wait_respects_global_limit() {
        init_effect_injection();
        // Set a small limit
        let old_max = global_max_delay_ns();
        set_global_max_delay_ns(1000); // 1μs

        let start = Instant::now();
        busy_wait_ns(1_000_000); // Request 1ms, should be clamped to 1μs
        let elapsed = start.elapsed();

        // Should complete in well under 1ms
        assert!(elapsed < Duration::from_millis(1));

        // Restore
        set_global_max_delay_ns(old_max);
    }

    #[test]
    fn test_busy_wait_approximate_accuracy() {
        init_effect_injection();
        // Test that a 10μs delay takes at least some time and finishes eventually.
        // Wide bounds due to scheduler jitter, especially in CI environments.
        let start = Instant::now();
        busy_wait_ns(10_000);
        let elapsed = start.elapsed();

        // Should take at least 1μs (sanity check it's not instant)
        // and less than 10ms (sanity check it's not infinite)
        assert!(
            elapsed >= Duration::from_micros(1),
            "Too fast: {:?}",
            elapsed
        );
        assert!(
            elapsed < Duration::from_millis(10),
            "Too slow: {:?}",
            elapsed
        );
    }

    #[test]
    #[ignore] // Flaky in CI - timing comparisons are inherently unreliable
    fn test_effect_injector_conditional() {
        let injector = EffectInjector::with_max_ns(100_000);

        // Use larger delays to overcome measurement noise
        // Average multiple iterations
        let mut sum_false = Duration::ZERO;
        let mut sum_true = Duration::ZERO;
        for _ in 0..10 {
            let start = Instant::now();
            injector.conditional_delay(false, 50_000);
            sum_false += start.elapsed();

            let start = Instant::now();
            injector.conditional_delay(true, 50_000);
            sum_true += start.elapsed();
        }

        // True should take noticeably longer on average
        assert!(
            sum_true > sum_false,
            "Expected true ({:?}) > false ({:?})",
            sum_true,
            sum_false
        );
    }

    #[test]
    fn test_effect_injector_clamping() {
        let injector = EffectInjector::with_max_ns(1000); // 1μs max

        let start = Instant::now();
        injector.delay_ns(1_000_000); // Request 1ms
        let elapsed = start.elapsed();

        // Should be clamped to ~1μs, not 1ms
        assert!(elapsed < Duration::from_millis(1));
    }

    #[test]
    fn test_benchmark_effect_expected_difference() {
        assert_eq!(BenchmarkEffect::Null.expected_difference_ns(), Some(0.0));

        assert_eq!(
            BenchmarkEffect::FixedDelay { delay_ns: 100 }.expected_difference_ns(),
            Some(100.0)
        );

        assert_eq!(
            BenchmarkEffect::ThetaMultiple {
                multiplier: 2.0,
                theta_ns: 100.0
            }
            .expected_difference_ns(),
            Some(200.0)
        );

        assert_eq!(
            BenchmarkEffect::EarlyExit { max_delay_ns: 500 }.expected_difference_ns(),
            Some(-500.0)
        );

        assert_eq!(
            BenchmarkEffect::HammingWeight { ns_per_bit: 1.0 }.expected_difference_ns(),
            None
        );
    }

    // =========================================================================
    // Platform detection tests
    // =========================================================================

    #[test]
    fn test_timer_backend_name() {
        let name = timer_backend_name();
        // Should be one of the known backends
        assert!(
            name == "cntvct_el0" || name == "rdtsc" || name == "instant_fallback",
            "Unknown timer backend: {}",
            name
        );
    }

    #[test]
    fn test_using_precise_timer() {
        let precise = using_precise_timer();
        let backend = timer_backend_name();

        // Consistency check: precise should match backend
        if backend == "instant_fallback" {
            assert!(!precise, "Fallback should not be precise");
        } else {
            assert!(precise, "Hardware timer should be precise");
        }
    }

    #[test]
    fn test_platform_detection_consistency() {
        // These should be consistent with each other
        let backend = timer_backend_name();
        let precise = using_precise_timer();

        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(backend, "cntvct_el0");
            assert!(precise);
        }

        #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
        {
            assert_eq!(backend, "rdtsc");
            assert!(precise);
        }
    }

    // =========================================================================
    // EffectInjector tests
    // =========================================================================

    #[test]
    fn test_effect_injector_default() {
        let injector = EffectInjector::default();
        assert_eq!(injector.max_ns(), 100_000); // Default is 100μs
    }

    #[test]
    fn test_effect_injector_custom_max() {
        let injector = EffectInjector::with_max_ns(5000);
        assert_eq!(injector.max_ns(), 5000);
    }

    #[test]
    fn test_hamming_weight_calculation() {
        let injector = EffectInjector::with_max_ns(100_000);

        // Test with known hamming weights
        let all_zeros = [0u8; 32];
        let all_ones = [0xFFu8; 32]; // 32 * 8 = 256 bits set
        let half_set = [0xAAu8; 32]; // 32 * 4 = 128 bits set (alternating)

        // Verify hamming weight logic (not timing, just calculation)
        let hw_zeros: u32 = all_zeros.iter().map(|b| b.count_ones()).sum();
        let hw_ones: u32 = all_ones.iter().map(|b| b.count_ones()).sum();
        let hw_half: u32 = half_set.iter().map(|b| b.count_ones()).sum();

        assert_eq!(hw_zeros, 0);
        assert_eq!(hw_ones, 256);
        assert_eq!(hw_half, 128);

        // Test that the injector doesn't panic
        injector.hamming_weight_delay(&all_zeros, 1.0);
        injector.hamming_weight_delay(&all_ones, 1.0);
        injector.hamming_weight_delay(&half_set, 1.0);
    }

    #[test]
    fn test_proportional_delay_calculation() {
        let injector = EffectInjector::with_max_ns(100_000);

        // Test that proportional delay doesn't panic with various inputs
        injector.proportional_delay(0, 10.0);
        injector.proportional_delay(100, 10.0);
        injector.proportional_delay(1000, 0.1);
    }

    // =========================================================================
    // BenchmarkEffect tests
    // =========================================================================

    #[test]
    fn test_benchmark_effect_at_theta_multiple() {
        let effect = BenchmarkEffect::at_theta_multiple(2.0, 100.0);
        assert_eq!(effect.expected_difference_ns(), Some(200.0));

        let effect = BenchmarkEffect::at_theta_multiple(0.5, 100.0);
        assert_eq!(effect.expected_difference_ns(), Some(50.0));
    }

    #[test]
    fn test_benchmark_effect_presets() {
        let adjacent = BenchmarkEffect::adjacent_network_effects();
        assert!(!adjacent.is_empty());

        // Check that multipliers increase
        for window in adjacent.windows(2) {
            assert!(window[0].0 < window[1].0, "Multipliers should increase");
        }

        let research = BenchmarkEffect::research_effects();
        assert!(!research.is_empty());
    }

    #[test]
    fn test_apply_effect_null() {
        let effect_fn = apply_effect(BenchmarkEffect::Null);
        // Should not panic
        effect_fn(&false);
        effect_fn(&true);
    }

    #[test]
    fn test_apply_effect_fixed_delay() {
        let effect_fn = apply_effect(BenchmarkEffect::FixedDelay { delay_ns: 100 });
        // Should not panic
        effect_fn(&false);
        effect_fn(&true);
    }

    #[test]
    fn test_apply_effect_bytes_hamming() {
        let effect_fn = apply_effect_bytes(BenchmarkEffect::HammingWeight { ns_per_bit: 1.0 });

        let zeros = [0u8; 32];
        let ones = [0xFFu8; 32];

        // Should not panic
        effect_fn(&zeros);
        effect_fn(&ones);
    }

    #[test]
    fn test_benchmark_effect_bimodal() {
        // Test construction
        let effect = BenchmarkEffect::bimodal(0.1, 1000);
        match effect {
            BenchmarkEffect::Bimodal {
                slow_prob,
                slow_delay_ns,
            } => {
                assert!((slow_prob - 0.1).abs() < 1e-9);
                assert_eq!(slow_delay_ns, 1000);
            }
            _ => panic!("Expected Bimodal variant"),
        }

        // Test expected difference: slow_prob * slow_delay_ns
        assert_eq!(effect.expected_difference_ns(), Some(100.0)); // 0.1 * 1000

        // Test default constructor
        let default = BenchmarkEffect::bimodal_default();
        match default {
            BenchmarkEffect::Bimodal {
                slow_prob,
                slow_delay_ns,
            } => {
                assert!((slow_prob - 0.05).abs() < 1e-9);
                assert_eq!(slow_delay_ns, 10_000);
            }
            _ => panic!("Expected Bimodal variant"),
        }
        assert_eq!(default.expected_difference_ns(), Some(500.0)); // 0.05 * 10_000
    }

    #[test]
    fn test_benchmark_effect_variable_delay() {
        // Test construction
        let effect = BenchmarkEffect::variable_delay(500, 100);
        match effect {
            BenchmarkEffect::VariableDelay { mean_ns, std_ns } => {
                assert_eq!(mean_ns, 500);
                assert_eq!(std_ns, 100);
            }
            _ => panic!("Expected VariableDelay variant"),
        }

        // Test expected difference: mean_ns
        assert_eq!(effect.expected_difference_ns(), Some(500.0));
    }

    #[test]
    fn test_benchmark_effect_tail_effect() {
        // Test construction
        let effect = BenchmarkEffect::tail_effect(100, 0.1, 5.0);
        match effect {
            BenchmarkEffect::TailEffect {
                base_delay_ns,
                tail_prob,
                tail_mult,
            } => {
                assert_eq!(base_delay_ns, 100);
                assert!((tail_prob - 0.1).abs() < 1e-9);
                assert!((tail_mult - 5.0).abs() < 1e-9);
            }
            _ => panic!("Expected TailEffect variant"),
        }

        // Test expected difference: base_delay * (1 - tail_prob) + base_delay * tail_mult * tail_prob
        // = 100 * 0.9 + 100 * 5.0 * 0.1 = 90 + 50 = 140
        assert_eq!(effect.expected_difference_ns(), Some(140.0));

        // Test default constructor
        let default = BenchmarkEffect::tail_effect_default();
        match default {
            BenchmarkEffect::TailEffect {
                base_delay_ns,
                tail_prob,
                tail_mult,
            } => {
                assert_eq!(base_delay_ns, 100);
                assert!((tail_prob - 0.05).abs() < 1e-9);
                assert!((tail_mult - 10.0).abs() < 1e-9);
            }
            _ => panic!("Expected TailEffect variant"),
        }
        // Expected: 100 * 0.95 + 100 * 10.0 * 0.05 = 95 + 50 = 145
        assert_eq!(default.expected_difference_ns(), Some(145.0));
    }

    #[test]
    fn test_apply_effect_bimodal() {
        let effect_fn = apply_effect(BenchmarkEffect::Bimodal {
            slow_prob: 0.1,
            slow_delay_ns: 1000,
        });
        // Should not panic - test multiple calls to exercise both branches
        for _ in 0..20 {
            effect_fn(&false);
            effect_fn(&true);
        }
    }

    #[test]
    fn test_apply_effect_variable_delay() {
        let effect_fn = apply_effect(BenchmarkEffect::VariableDelay {
            mean_ns: 500,
            std_ns: 100,
        });
        // Should not panic - test multiple calls
        for _ in 0..20 {
            effect_fn(&false);
            effect_fn(&true);
        }
    }

    #[test]
    fn test_apply_effect_tail_effect() {
        let effect_fn = apply_effect(BenchmarkEffect::TailEffect {
            base_delay_ns: 100,
            tail_prob: 0.1,
            tail_mult: 5.0,
        });
        // Should not panic - test multiple calls to exercise both branches
        for _ in 0..20 {
            effect_fn(&false);
            effect_fn(&true);
        }
    }

    #[test]
    fn test_stochastic_effects_are_stochastic() {
        // Verify that stochastic effects actually produce variation
        use std::time::Instant;

        // For Bimodal, run many iterations and check we get some variation
        let effect_fn = apply_effect(BenchmarkEffect::Bimodal {
            slow_prob: 0.5, // 50% chance to be slow
            slow_delay_ns: 10_000,
        });

        let mut times = Vec::with_capacity(50);
        for _ in 0..50 {
            let start = Instant::now();
            effect_fn(&true);
            times.push(start.elapsed().as_nanos() as f64);
        }

        // With 50% probability and 50 samples, we should see both fast and slow
        let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // There should be meaningful variation (max should be at least 2x min)
        // This test may occasionally fail due to randomness, but with 50% prob
        // and 50 samples, the chance of all being identical is vanishingly small
        assert!(
            max_time > min_time * 1.5,
            "Bimodal effect should show variation: min={:.0}ns, max={:.0}ns",
            min_time,
            max_time
        );
    }

    // =========================================================================
    // Busy wait accuracy tests (statistical)
    // =========================================================================

    #[test]
    fn test_busy_wait_multiple_calls() {
        init_effect_injection();
        // Run multiple iterations to verify consistency
        let delay_ns = 5000; // 5μs
        let iterations = 10;
        let mut total_elapsed = Duration::ZERO;

        for _ in 0..iterations {
            let start = Instant::now();
            busy_wait_ns(delay_ns);
            total_elapsed += start.elapsed();
        }

        let avg_elapsed_ns = total_elapsed.as_nanos() as f64 / iterations as f64;

        // Average should be at least 80% of requested (some calls might be fast)
        // and not more than 100x (sanity check)
        assert!(
            avg_elapsed_ns >= delay_ns as f64 * 0.5,
            "Average too fast: {}ns for {}ns request",
            avg_elapsed_ns,
            delay_ns
        );
        assert!(
            avg_elapsed_ns < delay_ns as f64 * 100.0,
            "Average too slow: {}ns for {}ns request",
            avg_elapsed_ns,
            delay_ns
        );
    }

    #[test]
    #[ignore] // Requires dedicated CPU time for accurate measurement
    fn test_busy_wait_monotonicity() {
        // Longer delays should take longer (statistical test)
        let delays = [1000u64, 5000, 10000, 50000]; // 1μs to 50μs
        let mut times = Vec::new();

        for &delay in &delays {
            let start = Instant::now();
            for _ in 0..5 {
                busy_wait_ns(delay);
            }
            times.push(start.elapsed().as_nanos() as f64 / 5.0);
        }

        // Each larger delay should generally take longer
        for i in 1..times.len() {
            assert!(
                times[i] > times[i - 1] * 0.5, // Allow some variance
                "Delay {} ({:.0}ns) should be longer than {} ({:.0}ns)",
                delays[i],
                times[i],
                delays[i - 1],
                times[i - 1]
            );
        }
    }

    // =========================================================================
    // Global limit tests
    // =========================================================================

    #[test]
    fn test_global_max_delay_hard_limit() {
        let old_max = global_max_delay_ns();

        // Try to set above hard limit (10ms)
        set_global_max_delay_ns(100_000_000); // 100ms

        // Should be clamped to 10ms
        assert_eq!(global_max_delay_ns(), 10_000_000);

        // Restore
        set_global_max_delay_ns(old_max);
    }

    #[test]
    fn test_global_max_delay_normal() {
        let old_max = global_max_delay_ns();

        set_global_max_delay_ns(500_000); // 500μs
        assert_eq!(global_max_delay_ns(), 500_000);

        // Restore
        set_global_max_delay_ns(old_max);
    }

    // =========================================================================
    // Frequency detection tests
    // =========================================================================

    #[test]
    fn test_counter_frequency_reasonable() {
        let freq = counter_frequency_hz();
        let backend = timer_backend_name();

        if backend == "instant_fallback" {
            assert_eq!(freq, 0, "Fallback should report 0 frequency");
        } else {
            // Hardware counters should report reasonable frequencies
            // Range: 1 MHz to 10 GHz
            assert!(
                freq >= 1_000_000,
                "Frequency {} Hz is too low for {}",
                freq,
                backend
            );
            assert!(
                freq <= 10_000_000_000,
                "Frequency {} Hz is too high for {}",
                freq,
                backend
            );

            // Platform-specific sanity checks
            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            {
                // Apple Silicon: 24MHz on M1/M2, 1GHz on M3/M4+
                assert!(
                    freq == 24_000_000 || freq == 1_000_000_000,
                    "Apple Silicon should be 24MHz (M1/M2) or 1GHz (M3+), got {} Hz",
                    freq
                );
            }

            #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
            {
                // x86_64 TSC is typically 1-5 GHz
                assert!(
                    freq >= 1_000_000_000,
                    "x86_64 TSC {} Hz seems too low",
                    freq
                );
            }
        }
    }

    #[test]
    fn test_timer_resolution_reasonable() {
        let resolution = timer_resolution_ns();
        let backend = timer_backend_name();

        if backend == "instant_fallback" {
            assert!(
                resolution.is_infinite(),
                "Fallback should report infinite resolution"
            );
        } else {
            // Hardware counters should have sub-microsecond resolution
            assert!(
                resolution > 0.0,
                "Resolution should be positive for {}",
                backend
            );
            assert!(
                resolution < 1000.0,
                "Resolution {} ns is too coarse for {}",
                resolution,
                backend
            );

            // Platform-specific checks
            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            {
                // Apple Silicon: 24MHz → ~41.67ns (M1/M2), 1GHz → 1ns (M3/M4+)
                let is_m1_m2 = (resolution - 41.67).abs() < 1.0;
                let is_m3_plus = resolution < 2.0;
                assert!(
                    is_m1_m2 || is_m3_plus,
                    "Apple Silicon resolution should be ~42ns (M1/M2) or ~1ns (M3+), got {}",
                    resolution
                );
            }

            #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
            {
                // x86_64: 1-5 GHz → 0.2-1.0ns typical
                assert!(
                    resolution < 10.0,
                    "x86_64 resolution {} ns seems too coarse",
                    resolution
                );
            }
        }
    }

    #[test]
    fn test_frequency_consistency() {
        // Multiple calls should return the same (cached) value
        let freq1 = counter_frequency_hz();
        let freq2 = counter_frequency_hz();
        assert_eq!(freq1, freq2, "Frequency should be consistent (cached)");

        let res1 = timer_resolution_ns();
        let res2 = timer_resolution_ns();
        assert_eq!(res1, res2, "Resolution should be consistent");
    }

    // =========================================================================
    // Minimum injectable effect tests
    // =========================================================================

    #[test]
    fn test_min_injectable_effect_reasonable() {
        let min_effect = min_injectable_effect_ns();
        assert!(min_effect >= 20.0, "Min effect should be at least 20ns");
        assert!(min_effect <= 500.0, "Min effect should not exceed 500ns");
    }

    #[test]
    fn test_min_injectable_effect_above_resolution() {
        let resolution = timer_resolution_ns();
        let min_effect = min_injectable_effect_ns();
        assert!(
            min_effect >= resolution * 3.0,
            "Min effect should be at least 3× timer resolution for reliable measurement"
        );
    }

    // =========================================================================
    // Frequency accuracy validation
    // =========================================================================

    /// Validate that our detected frequency matches reality by measuring a known delay.
    ///
    /// This test would catch cases where:
    /// - CNTFRQ_EL0 is incorrectly programmed by firmware
    /// - TSC frequency detection returns wrong value
    /// - Platform detection guesses wrong frequency
    #[test]
    fn test_frequency_accuracy_validation() {
        let backend = timer_backend_name();
        if backend == "instant_fallback" {
            // Can't validate fallback - it has no frequency
            return;
        }

        let freq = counter_frequency_hz();
        if freq == 0 {
            return;
        }

        // Use a longer delay for more accurate measurement
        const TARGET_DELAY_MS: u64 = 50;
        const TARGET_DELAY_NS: u64 = TARGET_DELAY_MS * 1_000_000;

        // Expected ticks for this delay
        let expected_ticks = ((TARGET_DELAY_NS as u128 * freq as u128) / 1_000_000_000) as u64;

        // Measure actual ticks during a real delay
        let (start_ticks, end_ticks) = measure_ticks_during_sleep(TARGET_DELAY_MS);
        let actual_ticks = end_ticks.wrapping_sub(start_ticks);

        // Calculate ratio - should be close to 1.0 if frequency is correct
        let ratio = actual_ticks as f64 / expected_ticks as f64;

        // Allow 20% tolerance for sleep jitter and measurement overhead
        // If frequency is wildly wrong (e.g., 24MHz vs 1GHz), ratio would be ~40x off
        assert!(
            ratio > 0.8 && ratio < 1.2,
            "Frequency validation failed!\n\
             Backend: {}\n\
             Detected frequency: {} Hz ({:.2} MHz)\n\
             Target delay: {} ms\n\
             Expected ticks: {}\n\
             Actual ticks: {}\n\
             Ratio: {:.3} (should be ~1.0)\n\
             This suggests the detected frequency is incorrect.",
            backend,
            freq,
            freq as f64 / 1e6,
            TARGET_DELAY_MS,
            expected_ticks,
            actual_ticks,
            ratio
        );
    }

    /// Measure counter ticks during a sleep of known duration.
    #[cfg(target_arch = "aarch64")]
    fn measure_ticks_during_sleep(sleep_ms: u64) -> (u64, u64) {
        let start: u64;
        unsafe {
            core::arch::asm!("mrs {}, cntvct_el0", out(reg) start);
        }

        std::thread::sleep(Duration::from_millis(sleep_ms));

        let end: u64;
        unsafe {
            core::arch::asm!("mrs {}, cntvct_el0", out(reg) end);
        }

        (start, end)
    }

    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    fn measure_ticks_during_sleep(sleep_ms: u64) -> (u64, u64) {
        let start = rdtsc();
        std::thread::sleep(Duration::from_millis(sleep_ms));
        let end = rdtsc();
        (start, end)
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))
    )))]
    fn measure_ticks_during_sleep(_sleep_ms: u64) -> (u64, u64) {
        (0, 0) // Fallback - test will skip
    }

    /// More stringent accuracy test - validates busy_wait_ns actually waits correct time.
    #[test]
    #[ignore] // Timing-sensitive test, flaky in CI due to calibration overhead
    fn test_busy_wait_accuracy_validation() {
        init_effect_injection();
        let backend = timer_backend_name();
        if backend == "instant_fallback" {
            return;
        }

        // Test that busy_wait_ns(10000) actually waits ~10μs
        const TARGET_NS: u64 = 10_000;
        const ITERATIONS: usize = 20;

        let mut total_elapsed_ns: u128 = 0;

        for _ in 0..ITERATIONS {
            let start = Instant::now();
            busy_wait_ns(TARGET_NS);
            total_elapsed_ns += start.elapsed().as_nanos();
        }

        let avg_elapsed_ns = total_elapsed_ns / ITERATIONS as u128;
        let ratio = avg_elapsed_ns as f64 / TARGET_NS as f64;

        // Should be at least 0.5x (allowing for measurement overhead eating into time)
        // and at most 5x (allowing for scheduler delays)
        // If frequency is 40x wrong, ratio would be ~0.025 or ~40
        assert!(
            ratio > 0.5 && ratio < 5.0,
            "busy_wait_ns accuracy validation failed!\n\
             Backend: {}\n\
             Detected frequency: {} Hz\n\
             Target delay: {} ns\n\
             Average actual delay: {} ns\n\
             Ratio: {:.2} (should be 1.0-2.0 typically)\n\
             This suggests the counter frequency is incorrect.",
            backend,
            counter_frequency_hz(),
            TARGET_NS,
            avg_elapsed_ns,
            ratio
        );
    }
}
