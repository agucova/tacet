//! Type aliases and common types.

use nalgebra::{SMatrix, SVector};

/// 9x9 covariance matrix for quantile differences.
pub type Matrix9 = SMatrix<f64, 9, 9>;

/// 9-dimensional vector for quantile differences.
pub type Vector9 = SVector<f64, 9>;

/// 9x2 design matrix [ones | b_tail] for effect decomposition.
pub type Matrix9x2 = SMatrix<f64, 9, 2>;

/// 2x2 matrix for effect covariance.
pub type Matrix2 = SMatrix<f64, 2, 2>;

/// 2-dimensional vector for effect parameters (shift, tail).
pub type Vector2 = SVector<f64, 2>;

/// Input class identifier for timing measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Class {
    /// Baseline input (typically constant) that establishes the reference timing.
    Baseline,
    /// Sample input (typically varied) for comparison against baseline.
    Sample,
}

/// A timing sample with its class label, preserving measurement order.
///
/// Used for joint resampling in covariance estimation, which preserves
/// temporal pairing between baseline and sample measurements.
#[derive(Debug, Clone, Copy)]
pub struct TimingSample {
    /// Timing value in nanoseconds.
    pub time_ns: f64,
    /// Which class this sample belongs to.
    pub class: Class,
}

/// Attacker model presets for timing analysis thresholds.
///
/// There is no single correct threshold (θ). Your choice of preset is a statement
/// about your threat model. SILENT explicitly demonstrates this using KyberSlash:
/// under Crosby-style thresholds (~100ns), a ~20-cycle leak isn't flagged as
/// practically significant; under Kario-style thresholds (~1 cycle), it is.
///
/// # Sources
///
/// - **Crosby et al. (2009)**: "Opportunities and Limits of Remote Timing Attacks."
///   Reports ~100ns LAN accuracy, 15–100μs internet accuracy.
/// - **Kario**: Argues timing differences as small as one clock cycle can be
///   detectable over local networks.
/// - **SILENT [2]**: Uses KyberSlash (~20 cycles on Raspberry Pi 2B) to demonstrate
///   how conclusions flip based on θ choice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttackerModel {
    // ═══════════════════════════════════════════════════════════════════
    // LOCAL ATTACKER PRESETS
    // ═══════════════════════════════════════════════════════════════════
    /// Local attacker with cycle-level timing (rdtsc, perf, kperf).
    ///
    /// θ = 2 cycles
    ///
    /// Use for: SGX enclaves, shared hosting, local privilege escalation.
    /// Kario argues even 1 cycle is detectable over LAN—for same-host
    /// attackers, 1–2 cycles is the conservative stance.
    LocalCycles,

    /// Local attacker but only coarse timers available.
    ///
    /// θ = 1 tick (whatever the timer resolution is)
    ///
    /// Use for: Sandboxed environments, tick-only measurement, noisy scheduling.
    /// This isn't "attacker is weaker"—it's "measurement primitive is weaker."
    LocalCoarseTimer,

    // ═══════════════════════════════════════════════════════════════════
    // LAN ATTACKER PRESETS
    // ═══════════════════════════════════════════════════════════════════
    /// Strict LAN attacker ("Kario-style").
    ///
    /// θ = 2 cycles
    ///
    /// Use for: High-security internal services where LAN attackers are capable.
    /// Kario argues even ~1 clock cycle can be detectable over LAN.
    /// Warning: May produce "TooNoisy" on coarse timers.
    LANStrict,

    /// Conservative LAN attacker ("Crosby-style").
    ///
    /// θ = 100 ns
    ///
    /// Use for: Internal services, database servers, microservices.
    /// Crosby et al. (2009) report attackers can measure with "accuracy as
    /// good as 100ns over a local network."
    LANConservative,

    // ═══════════════════════════════════════════════════════════════════
    // WAN ATTACKER PRESETS
    // ═══════════════════════════════════════════════════════════════════
    /// Optimistic WAN attacker (low-jitter environments).
    ///
    /// θ = 15 μs
    ///
    /// Use for: Same-region cloud, datacenter-to-datacenter, low-jitter paths.
    /// Best-case end of Crosby's 15–100μs internet range.
    WANOptimistic,

    /// Conservative WAN attacker (general internet).
    ///
    /// θ = 50 μs
    ///
    /// Use for: Public APIs, web services, general internet exposure.
    /// Crosby et al. (2009) report 15–100μs accuracy across internet;
    /// 50μs is a reasonable midpoint.
    WANConservative,

    // ═══════════════════════════════════════════════════════════════════
    // SPECIAL-PURPOSE PRESETS
    // ═══════════════════════════════════════════════════════════════════
    /// Calibrated to catch KyberSlash-class vulnerabilities.
    ///
    /// θ = 10 cycles
    ///
    /// Use for: Post-quantum crypto, division-based leaks.
    /// SILENT characterizes KyberSlash as ~20 cycles on Raspberry Pi 2B
    /// (900MHz Cortex-A7). θ=10 ensures such leaks are non-negligible.
    KyberSlashSentinel,

    /// Research mode: detect any statistical difference (θ → 0).
    ///
    /// Warning: Will flag tiny, unexploitable differences. Not for CI.
    /// Use for: Profiling, debugging, academic analysis, finding any leak.
    Research,

    // ═══════════════════════════════════════════════════════════════════
    // CUSTOM THRESHOLDS
    // ═══════════════════════════════════════════════════════════════════
    /// Custom threshold in nanoseconds.
    CustomNs {
        /// Threshold in nanoseconds.
        threshold_ns: f64,
    },

    /// Custom threshold in cycles (more portable across CPUs).
    CustomCycles {
        /// Threshold in CPU cycles.
        threshold_cycles: u32,
    },

    /// Custom threshold in timer ticks (for tick-based timers).
    CustomTicks {
        /// Threshold in timer ticks.
        threshold_ticks: u32,
    },
}

impl Default for AttackerModel {
    /// Default to LANConservative (100ns) - a sensible default for most use cases.
    fn default() -> Self {
        AttackerModel::LANConservative
    }
}

impl AttackerModel {
    /// Convert the attacker model to a threshold in nanoseconds.
    ///
    /// For cycle-based presets, uses the provided CPU frequency (in GHz).
    /// For tick-based presets, uses the provided timer resolution (in ns).
    ///
    /// Returns `None` if the model requires runtime information not provided.
    pub fn to_threshold_ns(&self, cpu_freq_ghz: Option<f64>, timer_resolution_ns: Option<f64>) -> Option<f64> {
        match self {
            // Nanosecond-based presets
            AttackerModel::LANConservative => Some(100.0),
            AttackerModel::WANOptimistic => Some(15_000.0),      // 15 μs
            AttackerModel::WANConservative => Some(50_000.0),    // 50 μs

            // Cycle-based presets (need CPU frequency)
            AttackerModel::LocalCycles | AttackerModel::LANStrict => {
                cpu_freq_ghz.map(|freq| 2.0 / freq) // 2 cycles to ns
            }
            AttackerModel::KyberSlashSentinel => {
                cpu_freq_ghz.map(|freq| 10.0 / freq) // 10 cycles to ns
            }

            // Research mode: effectively zero threshold
            AttackerModel::Research => Some(0.0),

            // Tick-based preset (needs timer resolution)
            AttackerModel::LocalCoarseTimer => timer_resolution_ns,

            // Custom presets
            AttackerModel::CustomNs { threshold_ns } => Some(*threshold_ns),
            AttackerModel::CustomCycles { threshold_cycles } => {
                cpu_freq_ghz.map(|freq| *threshold_cycles as f64 / freq)
            }
            AttackerModel::CustomTicks { threshold_ticks } => {
                timer_resolution_ns.map(|res| *threshold_ticks as f64 * res)
            }
        }
    }

    /// Check if this preset is cycle-based (requires CPU frequency for conversion).
    pub fn is_cycle_based(&self) -> bool {
        matches!(
            self,
            AttackerModel::LocalCycles
                | AttackerModel::LANStrict
                | AttackerModel::KyberSlashSentinel
                | AttackerModel::CustomCycles { .. }
        )
    }

    /// Check if this preset is tick-based (requires timer resolution for conversion).
    pub fn is_tick_based(&self) -> bool {
        matches!(
            self,
            AttackerModel::LocalCoarseTimer | AttackerModel::CustomTicks { .. }
        )
    }

    /// Get a human-readable description of this attacker model.
    pub fn description(&self) -> &'static str {
        match self {
            AttackerModel::LocalCycles => "Local attacker with cycle-level timing (θ=2 cycles)",
            AttackerModel::LocalCoarseTimer => "Local attacker with coarse timer (θ=1 tick)",
            AttackerModel::LANStrict => "Strict LAN attacker, Kario-style (θ=2 cycles)",
            AttackerModel::LANConservative => "Conservative LAN attacker, Crosby-style (θ=100ns)",
            AttackerModel::WANOptimistic => "Optimistic WAN attacker, low-jitter (θ=15μs)",
            AttackerModel::WANConservative => "Conservative WAN attacker, general internet (θ=50μs)",
            AttackerModel::KyberSlashSentinel => "KyberSlash-class detector (θ=10 cycles)",
            AttackerModel::Research => "Research mode, detect any difference (θ→0)",
            AttackerModel::CustomNs { threshold_ns } => {
                // Static string, can't include dynamic value
                if *threshold_ns < 1000.0 {
                    "Custom threshold (nanoseconds)"
                } else if *threshold_ns < 1_000_000.0 {
                    "Custom threshold (microseconds)"
                } else {
                    "Custom threshold (milliseconds)"
                }
            }
            AttackerModel::CustomCycles { .. } => "Custom threshold (cycles)",
            AttackerModel::CustomTicks { .. } => "Custom threshold (ticks)",
        }
    }
}
