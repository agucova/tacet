//! Mathematical constants used throughout the crate.

/// Default deterministic seed for RNG operations.
///
/// This seed ensures reproducibility: same seed + same data = same result.
/// The value `0x74696D696E67` is "timing" encoded in ASCII.
pub const DEFAULT_SEED: u64 = 0x74696D696E67;

/// Decile percentiles for quantile computation.
pub const DECILES: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// Unit vector for uniform shift detection.
pub const ONES: [f64; 9] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// Centered tail basis vector for tail effect detection.
/// This is orthogonal to ONES, allowing independent estimation of shift vs tail effects.
pub const B_TAIL: [f64; 9] = [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5];

/// Natural log of 2*pi, used in multivariate normal log-pdf computation.
pub const LOG_2PI: f64 = 1.8378770664093453;
