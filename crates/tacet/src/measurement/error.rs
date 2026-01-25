//! Error types for timing measurements.

/// Error returned when a timing measurement fails.
///
/// Measurements can fail for various reasons related to PMU counter availability,
/// system load, or invalid timer state. When a measurement fails, the entire
/// sample should be skipped rather than using a sentinel value like 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MeasurementError {
    /// Seqlock retry limit exceeded (1000 attempts).
    ///
    /// This occurs when the perf_event mmap page is constantly being updated
    /// by the kernel, preventing a consistent read. Indicates either:
    /// - System under extreme load
    /// - PMU event being constantly multiplexed
    /// - Thread being rapidly migrated between CPUs
    ///
    /// This is extremely rare (<1 in 10M measurements) under normal conditions.
    RetryExhausted,

    /// Timer syscall failed (reset or read).
    ///
    /// On Linux perf_event, this may indicate:
    /// - Insufficient permissions (need sudo or CAP_PERFMON)
    /// - Counter has been disabled
    /// - File descriptor closed unexpectedly
    SyscallFailed,

    /// Measurement returned non-finite value (NaN or Infinity).
    ///
    /// This indicates a bug in timer calibration (invalid cycles_per_ns)
    /// or arithmetic overflow during conversion.
    NonFinite,
}

impl std::fmt::Display for MeasurementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RetryExhausted => write!(
                f,
                "PMU counter seqlock retry limit exceeded - system under extreme load"
            ),
            Self::SyscallFailed => write!(f, "perf_event syscall failed"),
            Self::NonFinite => write!(f, "measurement produced non-finite value (NaN/Inf)"),
        }
    }
}

impl std::error::Error for MeasurementError {}

/// Result type for timing measurements.
pub type MeasurementResult = Result<u64, MeasurementError>;
