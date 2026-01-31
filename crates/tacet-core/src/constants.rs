//! Mathematical constants used throughout the crate.

/// Default deterministic seed for RNG operations.
///
/// This seed ensures reproducibility: same seed + same data = same result.
/// The value `0x74696D696E67` is "timing" encoded in ASCII.
pub const DEFAULT_SEED: u64 = 0x74696D696E67;

/// Decile percentiles for quantile computation.
pub const DECILES: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// Natural log of 2*pi, used in multivariate normal log-pdf computation.
pub const LOG_2PI: f64 = 1.8378770664093453;

// =============================================================================
// Default configuration constants (spec §3.4)
// =============================================================================

/// Default pass threshold for Bayesian decision: P(leak) < 0.05 means Pass.
pub const DEFAULT_PASS_THRESHOLD: f64 = 0.05;

/// Default fail threshold for Bayesian decision: P(leak) > 0.95 means Fail.
pub const DEFAULT_FAIL_THRESHOLD: f64 = 0.95;

/// Default number of bootstrap iterations for covariance estimation.
pub const DEFAULT_BOOTSTRAP_ITERATIONS: usize = 2000;

/// Default number of calibration samples per class.
pub const DEFAULT_CALIBRATION_SAMPLES: usize = 5000;

/// Default batch size for adaptive sampling loop.
pub const DEFAULT_BATCH_SIZE: usize = 1000;

/// Default maximum samples per class before stopping.
pub const DEFAULT_MAX_SAMPLES: usize = 1_000_000;

/// Default time budget in seconds.
pub const DEFAULT_TIME_BUDGET_SECS: u64 = 60;
