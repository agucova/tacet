//! Time unit conversion utilities.
//!
//! Provides conversions between different time units used in timing measurements.

/// Time unit for timing samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeUnit {
    /// CPU cycles (requires frequency for ns conversion).
    #[default]
    Cycles,

    /// Nanoseconds (no conversion needed).
    Nanoseconds,

    /// Microseconds (multiply by 1000 for ns).
    Microseconds,

    /// Milliseconds (multiply by 1_000_000 for ns).
    Milliseconds,

    /// Timer ticks at a specific frequency.
    Ticks {
        /// Ticks per second (Hz).
        frequency_hz: u64,
    },
}

impl TimeUnit {
    /// Get the conversion factor to nanoseconds.
    ///
    /// For `Cycles`, uses the provided CPU frequency.
    /// For other units, the frequency parameter is ignored.
    ///
    /// # Arguments
    /// * `cpu_freq_ghz` - CPU frequency in GHz (only used for `Cycles`)
    ///
    /// # Returns
    /// Nanoseconds per unit.
    pub fn ns_per_unit(&self, cpu_freq_ghz: Option<f64>) -> f64 {
        match self {
            TimeUnit::Cycles => {
                // ns/cycle = 1 / (GHz) = 1 / (cycles/ns)
                let freq = cpu_freq_ghz.unwrap_or(3.0); // Default 3 GHz
                1.0 / freq
            }
            TimeUnit::Nanoseconds => 1.0,
            TimeUnit::Microseconds => 1_000.0,
            TimeUnit::Milliseconds => 1_000_000.0,
            TimeUnit::Ticks { frequency_hz } => {
                // ns/tick = 1e9 / frequency_hz
                1e9 / (*frequency_hz as f64)
            }
        }
    }

    /// Create a Ticks unit from a frequency in MHz.
    pub fn ticks_mhz(freq_mhz: u64) -> Self {
        TimeUnit::Ticks {
            frequency_hz: freq_mhz * 1_000_000,
        }
    }

    /// Create a Ticks unit from a frequency in Hz.
    pub fn ticks_hz(freq_hz: u64) -> Self {
        TimeUnit::Ticks {
            frequency_hz: freq_hz,
        }
    }
}

/// Convert samples to nanoseconds.
///
/// # Arguments
/// * `samples` - Raw timing samples
/// * `unit` - Time unit of the samples
/// * `cpu_freq_ghz` - CPU frequency in GHz (only used if unit is Cycles)
///
/// # Returns
/// Samples converted to nanoseconds.
pub fn to_nanoseconds(samples: &[u64], unit: TimeUnit, cpu_freq_ghz: Option<f64>) -> Vec<f64> {
    let factor = unit.ns_per_unit(cpu_freq_ghz);
    samples.iter().map(|&s| s as f64 * factor).collect()
}

/// Convert f64 samples to nanoseconds.
pub fn to_nanoseconds_f64(samples: &[f64], unit: TimeUnit, cpu_freq_ghz: Option<f64>) -> Vec<f64> {
    let factor = unit.ns_per_unit(cpu_freq_ghz);
    samples.iter().map(|&s| s * factor).collect()
}

/// Estimate CPU frequency from cycle counts and wall-clock duration.
///
/// # Arguments
/// * `total_cycles` - Total CPU cycles measured
/// * `duration_ns` - Wall-clock duration in nanoseconds
///
/// # Returns
/// Estimated CPU frequency in GHz.
pub fn estimate_cpu_freq_ghz(total_cycles: u64, duration_ns: u64) -> f64 {
    if duration_ns == 0 {
        return 3.0; // Default fallback
    }
    total_cycles as f64 / duration_ns as f64
}

/// Common CPU frequency presets.
pub mod presets {
    /// Intel/AMD ~3.0 GHz (common desktop/server).
    pub const FREQ_3GHZ: f64 = 3.0;

    /// Intel/AMD ~2.5 GHz (laptop power-saving).
    pub const FREQ_2_5GHZ: f64 = 2.5;

    /// Apple M1/M2 performance cores ~3.2 GHz.
    pub const FREQ_APPLE_PERF: f64 = 3.2;

    /// Apple M1/M2 efficiency cores ~2.0 GHz.
    pub const FREQ_APPLE_EFF: f64 = 2.0;

    /// ARM Cortex-A72 ~1.5 GHz (Raspberry Pi 4).
    pub const FREQ_RPI4: f64 = 1.5;

    /// ARM Cortex-A7 ~900 MHz (Raspberry Pi 2).
    pub const FREQ_RPI2: f64 = 0.9;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycles_conversion() {
        let samples = vec![3000u64, 6000, 9000];

        // At 3 GHz: 1 cycle = 0.333... ns
        let ns = to_nanoseconds(&samples, TimeUnit::Cycles, Some(3.0));

        assert!((ns[0] - 1000.0).abs() < 0.01);
        assert!((ns[1] - 2000.0).abs() < 0.01);
        assert!((ns[2] - 3000.0).abs() < 0.01);
    }

    #[test]
    fn test_nanoseconds_passthrough() {
        let samples = vec![100u64, 200, 300];
        let ns = to_nanoseconds(&samples, TimeUnit::Nanoseconds, None);

        assert_eq!(ns, vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn test_microseconds_conversion() {
        let samples = vec![1u64, 2, 3];
        let ns = to_nanoseconds(&samples, TimeUnit::Microseconds, None);

        assert_eq!(ns, vec![1000.0, 2000.0, 3000.0]);
    }

    #[test]
    fn test_ticks_conversion() {
        // 25 MHz counter (like ARM cntvct_el0)
        let samples = vec![25u64, 50, 100];
        let ns = to_nanoseconds(&samples, TimeUnit::ticks_mhz(25), None);

        // 1 tick = 40 ns at 25 MHz
        assert!((ns[0] - 1000.0).abs() < 0.01); // 25 * 40 = 1000
        assert!((ns[1] - 2000.0).abs() < 0.01); // 50 * 40 = 2000
        assert!((ns[2] - 4000.0).abs() < 0.01); // 100 * 40 = 4000
    }

    #[test]
    fn test_estimate_freq() {
        // 3 billion cycles in 1 second = 3 GHz
        let freq = estimate_cpu_freq_ghz(3_000_000_000, 1_000_000_000);
        assert!((freq - 3.0).abs() < 0.01);
    }
}
