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

/// Attacker model determines the minimum effect threshold (θ) for leak detection.
///
/// Choose based on your threat model - this is the most important configuration choice.
/// There is no single correct threshold; your choice is a statement about who you're
/// defending against.
///
/// # Sources
///
/// - **Crosby et al. (2009)**: "Opportunities and Limits of Remote Timing Attacks."
///   Reports ~100ns LAN accuracy, 15–100μs internet accuracy.
/// - **Van Goethem et al. (2020)**: "Timeless Timing Attacks" (USENIX Security).
///   Achieved 100ns accuracy over the internet using HTTP/2 request multiplexing.
/// - **Flush+Reload, Prime+Probe literature**: Documents cycle-level timing attacks
///   on shared hardware (SGX, cross-VM, containers).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AttackerModel {
    /// Attacker shares physical hardware with the target.
    ///
    /// θ = 2 cycles (~0.6ns @ 3GHz)
    ///
    /// Use for: SGX enclaves, cross-VM on shared cache, co-located containers,
    /// hyperthreading neighbors, shared hosting.
    ///
    /// Sources: Flush+Reload, Prime+Probe literature
    SharedHardware,

    /// Catch KyberSlash-class timing leaks in post-quantum cryptography.
    ///
    /// θ = 10 cycles (~3.3ns @ 3GHz)
    ///
    /// Use for: ML-KEM (Kyber), ML-DSA (Dilithium), and other lattice-based
    /// cryptography where ~20 cycle leaks have been shown exploitable.
    ///
    /// This provides a 2x safety margin: set θ at 10 cycles to reliably
    /// catch 20+ cycle leaks.
    ///
    /// Sources: KyberSlash (Jancar et al. 2024)
    PostQuantumSentinel,

    /// Attacker on same local network, or using HTTP/2 concurrent requests.
    ///
    /// θ = 100ns
    ///
    /// Use for: Internal services, microservices, or any HTTP/2 endpoint.
    ///
    /// Note: "Timeless Timing Attacks" (USENIX 2020) achieved 100ns over
    /// the internet using HTTP/2 request multiplexing. The LAN/WAN
    /// distinction is weaker than previously thought.
    ///
    /// Sources: Crosby et al. 2009 (LAN), Van Goethem et al. 2020 (HTTP/2)
    #[default]
    AdjacentNetwork,

    /// Attacker over the internet using traditional timing techniques.
    ///
    /// θ = 50μs
    ///
    /// Use for: Public APIs without HTTP/2, legacy services, high-jitter paths.
    ///
    /// Sources: Crosby et al. 2009 (15-100μs range)
    RemoteNetwork,

    /// Detect any measurable timing difference.
    ///
    /// θ → 0 (clamped to timer resolution)
    ///
    /// Warning: Will flag tiny, unexploitable differences. Not for CI.
    /// Use for: Profiling, debugging, academic analysis, finding any leak.
    Research,

    /// Custom threshold in nanoseconds.
    Custom {
        /// Threshold in nanoseconds.
        threshold_ns: f64,
    },
}

impl AttackerModel {
    /// Convert attacker model to threshold in nanoseconds.
    pub fn to_threshold_ns(&self) -> f64 {
        match self {
            AttackerModel::SharedHardware => 0.6,      // ~2 cycles @ 3GHz
            AttackerModel::PostQuantumSentinel => 3.3, // ~10 cycles @ 3GHz (catches 20+ cycle leaks)
            AttackerModel::AdjacentNetwork => 100.0,
            AttackerModel::RemoteNetwork => 50_000.0, // 50μs
            AttackerModel::Research => 0.0,           // Will be clamped to timer resolution
            AttackerModel::Custom { threshold_ns } => *threshold_ns,
        }
    }

    /// Get a human-readable description of this attacker model.
    pub fn description(&self) -> &'static str {
        match self {
            AttackerModel::SharedHardware => "shared hardware (SGX, cross-VM, containers)",
            AttackerModel::PostQuantumSentinel => "post-quantum sentinel (KyberSlash-class)",
            AttackerModel::AdjacentNetwork => "adjacent network (LAN, HTTP/2)",
            AttackerModel::RemoteNetwork => "remote network (internet)",
            AttackerModel::Research => "research (detect any difference)",
            AttackerModel::Custom { .. } => "custom threshold",
        }
    }
}
