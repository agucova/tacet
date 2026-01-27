//! Platform timer frequency detection.
//!
//! Provides automatic detection of system timer frequencies for ARM64 and x86_64.
//! This module is no-std compatible for basic register reading, with std-dependent
//! calibration routines available when the `std` feature is enabled.

#[cfg(target_arch = "aarch64")]
use core::arch::asm;

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
use core::arch::asm;

// =============================================================================
// ARM64 (aarch64) Generic Timer
// =============================================================================

/// Get ARM64 counter frequency in Hz.
///
/// Reads the CNTFRQ_EL0 register and validates the result. Falls back to
/// platform detection or calibration if the register value is suspicious.
///
/// # Platform Notes
/// - Apple Silicon M1/M2: 24 MHz
/// - Apple Silicon M3+ (macOS 15+): 1 GHz (kernel-scaled)
/// - AWS Graviton: 1 GHz
/// - Raspberry Pi 4: 54 MHz
///
/// # Returns
/// Counter frequency in Hz, or a reasonable fallback if detection fails.
#[cfg(target_arch = "aarch64")]
pub fn get_aarch64_counter_freq_hz() -> u64 {
    // One-time initialization with validation (not thread-local)
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static VALIDATED_FREQ: OnceLock<u64> = OnceLock::new();

        *VALIDATED_FREQ.get_or_init(|| {
            // Read counter frequency from CNTFRQ_EL0 register
            let cntfrq: u64;
            unsafe {
                asm!("mrs {}, cntfrq_el0", out(reg) cntfrq);
            }

            // Quick validation: is the frequency value reasonable?
            if !is_reasonable_aarch64_freq(cntfrq) {
                eprintln!(
                    "[tacet-core] WARNING: CNTFRQ_EL0 returned suspicious value: {} Hz",
                    cntfrq
                );
                eprintln!("[tacet-core] Calibrating ARM64 counter frequency...");
                return calibrate_aarch64_frequency();
            }

            // Sanity check: does CNTFRQ_EL0 match runtime calibration?
            // This catches virtualization issues where CNTFRQ_EL0 is incorrectly programmed
            let calibrated = calibrate_aarch64_frequency();
            let ratio = calibrated as f64 / cntfrq as f64;

            // Allow 10% tolerance for measurement noise
            if !(0.9..=1.1).contains(&ratio) {
                eprintln!(
                    "[tacet-core] WARNING: CNTFRQ_EL0 ({} Hz / {:.2} MHz) differs from calibrated frequency ({} Hz / {:.2} MHz) by {:.1}%",
                    cntfrq, cntfrq as f64 / 1e6,
                    calibrated, calibrated as f64 / 1e6,
                    (ratio - 1.0) * 100.0
                );
                eprintln!(
                    "[tacet-core] This typically indicates virtualization (VM/CI) with misconfigured timers."
                );
                eprintln!(
                    "[tacet-core] Using calibrated frequency: {} Hz ({:.2} MHz)",
                    calibrated, calibrated as f64 / 1e6
                );
                return calibrated;
            }

            // CNTFRQ_EL0 matches calibration - trust it
            cntfrq
        })
    }

    // No-std fallback: trust CNTFRQ_EL0 (can't calibrate without std::time)
    #[cfg(not(feature = "std"))]
    {
        let freq: u64;
        unsafe {
            asm!("mrs {}, cntfrq_el0", out(reg) freq);
        }

        if is_reasonable_aarch64_freq(freq) {
            freq
        } else {
            24_000_000 // Assume 24MHz (Apple Silicon M1/M2)
        }
    }
}

/// Check if an ARM64 counter frequency is reasonable (1 MHz to 10 GHz).
#[cfg(target_arch = "aarch64")]
#[inline]
fn is_reasonable_aarch64_freq(freq: u64) -> bool {
    (1_000_000..=10_000_000_000).contains(&freq)
}

/// Try to detect ARM64 platform from /proc/cpuinfo and return known frequency.
///
/// This is only used as a fallback when CNTFRQ_EL0 returns an unreasonable value.
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "linux"))]
fn detect_aarch64_platform_freq() -> Option<u64> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let cpuinfo_lower = cpuinfo.to_lowercase();

    // AWS Graviton (Neoverse cores) - common default
    if cpuinfo_lower.contains("neoverse") || cpuinfo_lower.contains("graviton") {
        return Some(1_000_000_000); // 1 GHz
    }

    // Raspberry Pi 4 (Cortex-A72)
    if cpuinfo_lower.contains("raspberry pi 4") || cpuinfo_lower.contains("bcm2711") {
        return Some(54_000_000); // 54 MHz
    }

    None
}

/// Calibrate ARM64 counter frequency against std::time::Instant.
///
/// Takes multiple samples and returns the median for robustness.
#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[allow(dead_code)]
fn calibrate_aarch64_frequency() -> u64 {
    use std::time::{Duration, Instant};

    const SAMPLES: usize = 5;
    const SLEEP_MS: u64 = 20;

    let mut frequencies = Vec::with_capacity(SAMPLES);

    for _ in 0..SAMPLES {
        let start_cnt: u64;
        unsafe {
            asm!("mrs {}, cntvct_el0", out(reg) start_cnt);
        }
        let start_instant = Instant::now();

        std::thread::sleep(Duration::from_millis(SLEEP_MS));

        let end_cnt: u64;
        unsafe {
            asm!("mrs {}, cntvct_el0", out(reg) end_cnt);
        }
        let elapsed_ns = start_instant.elapsed().as_nanos() as u64;

        let cnt_delta = end_cnt.wrapping_sub(start_cnt);
        let freq = ((cnt_delta as u128 * 1_000_000_000) / elapsed_ns as u128) as u64;

        if is_reasonable_aarch64_freq(freq) {
            frequencies.push(freq);
        }
    }

    if frequencies.is_empty() {
        eprintln!("[tacet-core] WARNING: ARM64 calibration failed. Using 24MHz estimate.");
        return 24_000_000;
    }

    // Use median for robustness
    frequencies.sort_unstable();
    let median = frequencies[frequencies.len() / 2];

    eprintln!(
        "[tacet-core] ARM64 counter frequency calibrated to {:.2} MHz",
        median as f64 / 1_000_000.0
    );

    median
}

// =============================================================================
// x86_64 RDTSC
// =============================================================================

/// Get x86_64 TSC frequency in Hz.
///
/// Tries multiple methods in priority order:
/// 1. Linux: /sys/devices/system/cpu/cpu0/tsc_freq_khz (most reliable)
/// 2. Linux: CPUID leaf 0x16 base frequency (Skylake+, if invariant TSC)
/// 3. macOS: sysctl machdep.tsc.frequency
/// 4. Fallback: calibration against std::time::Instant
///
/// # Returns
/// TSC frequency in Hz, or a reasonable estimate if detection fails.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
pub fn get_x86_64_tsc_freq_hz() -> u64 {
    // One-time initialization (not thread-local)
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static VALIDATED_FREQ: OnceLock<u64> = OnceLock::new();

        *VALIDATED_FREQ.get_or_init(|| {
            // Try platform-specific reliable sources first
            #[cfg(target_os = "linux")]
            if let Some(freq) = get_tsc_freq_linux() {
                return freq;
            }

            #[cfg(target_os = "macos")]
            if let Some(freq) = get_tsc_freq_macos() {
                return freq;
            }

            // Fallback: runtime calibration
            calibrate_tsc_frequency()
        })
    }

    // No-std fallback: assume 3GHz (common Intel base frequency)
    #[cfg(not(feature = "std"))]
    {
        3_000_000_000
    }
}

/// Linux: Try to get TSC frequency from sysfs or CPUID.
#[cfg(all(feature = "std", target_arch = "x86_64", target_os = "linux"))]
fn get_tsc_freq_linux() -> Option<u64> {
    // Method 1: sysfs tsc_freq_khz (most reliable when available)
    if let Ok(content) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/tsc_freq_khz") {
        if let Ok(khz) = content.trim().parse::<u64>() {
            let freq = khz * 1000;
            if is_reasonable_tsc_freq(freq) {
                return Some(freq);
            }
        }
    }

    // Method 2: CPUID leaf 0x16 - Processor Frequency Information (Skylake+)
    if has_invariant_tsc() {
        if let Some(freq) = get_cpuid_base_freq() {
            if is_reasonable_tsc_freq(freq) {
                return Some(freq);
            }
        }
    }

    None
}

/// Check if CPU has invariant TSC (constant rate regardless of frequency scaling).
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn has_invariant_tsc() -> bool {
    // Check CPUID.80000007H:EDX[8] - Invariant TSC
    let result: u32;
    unsafe {
        asm!(
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
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn get_cpuid_base_freq() -> Option<u64> {
    // First check if leaf 0x16 is supported
    let max_leaf: u32;
    unsafe {
        asm!(
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
        return None;
    }

    // CPUID leaf 0x16: Processor Frequency Information
    // EAX = Base frequency in MHz
    let base_mhz: u32;
    unsafe {
        asm!(
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

    if base_mhz == 0 {
        return None;
    }

    Some(base_mhz as u64 * 1_000_000)
}

/// macOS: Try to get TSC frequency from sysctl.
#[cfg(all(feature = "std", target_arch = "x86_64", target_os = "macos"))]
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
#[cfg(all(feature = "std", target_arch = "x86_64", target_os = "macos"))]
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
#[inline]
fn is_reasonable_tsc_freq(freq: u64) -> bool {
    (500_000_000..=10_000_000_000).contains(&freq)
}

/// Calibrate TSC frequency by measuring against std::time::Instant.
#[cfg(all(feature = "std", target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn calibrate_tsc_frequency() -> u64 {
    use std::time::{Duration, Instant};

    eprintln!("[tacet-core] Calibrating TSC frequency (no sysfs/sysctl available)...");

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
        eprintln!("[tacet-core] WARNING: TSC calibration failed. Using 3GHz estimate.");
        return 3_000_000_000;
    }

    // Use median for robustness
    frequencies.sort_unstable();
    let median = frequencies[frequencies.len() / 2];

    eprintln!(
        "[tacet-core] TSC frequency calibrated to {:.2} GHz",
        median as f64 / 1_000_000_000.0
    );

    median
}

/// Read x86_64 TSC (Time Stamp Counter).
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[inline]
fn rdtsc() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nostack, nomem)
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

// =============================================================================
// Public API
// =============================================================================

/// Automatically detect the system timer frequency in Hz.
///
/// This function uses platform-specific detection:
/// - **ARM64**: Reads CNTFRQ_EL0 with validation and fallbacks
/// - **x86_64**: Reads TSC frequency from sysfs/CPUID with calibration fallback
///
/// # Returns
/// - Timer frequency in Hz for platforms with cycle counters
/// - 0 for platforms without cycle counters (fallback timer)
///
/// # Examples
/// ```
/// let freq = tacet_core::timer::counter_frequency_hz();
/// println!("Timer frequency: {} Hz ({:.2} MHz)", freq, freq as f64 / 1e6);
/// ```
pub fn counter_frequency_hz() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        get_aarch64_counter_freq_hz()
    }

    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    {
        get_x86_64_tsc_freq_hz()
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))
    )))]
    {
        0 // Fallback platforms have no meaningful counter frequency
    }
}

/// Returns the timer resolution in nanoseconds.
///
/// This is the theoretical minimum delay that can be measured,
/// calculated as 1e9 / frequency.
///
/// # Returns
/// - Resolution in nanoseconds (e.g., 0.33 for 3GHz TSC, 42 for 24MHz Apple Silicon)
/// - f64::INFINITY for fallback platforms without cycle counters
///
/// # Examples
/// ```
/// let resolution = tacet_core::timer::timer_resolution_ns();
/// println!("Timer resolution: {:.2} ns", resolution);
/// ```
pub fn timer_resolution_ns() -> f64 {
    let freq = counter_frequency_hz();
    if freq == 0 {
        f64::INFINITY
    } else {
        1_000_000_000.0 / freq as f64
    }
}
