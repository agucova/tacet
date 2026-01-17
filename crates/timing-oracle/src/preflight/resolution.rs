//! Timer sanity check utility.
//!
//! This module provides the platform-specific `timer_sanity_check` function
//! that requires access to the timer implementation.
//!
//! The resolution check logic itself is in `timing-oracle-core::preflight::resolution`.

use timing_oracle_core::preflight::ResolutionWarning;

/// Warning type for non-monotonic timer (re-exported for convenience).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TimerSanityWarning {
    /// Timer is not monotonic (returned negative duration).
    ///
    /// **Severity**: ResultUndermining
    ///
    /// The timer is fundamentally broken. All measurements are unreliable.
    NonMonotonic {
        /// Number of non-monotonic steps detected.
        violations: usize,
        /// Largest negative jump in cycles.
        max_jump_cycles: u64,
    },
}

impl TimerSanityWarning {
    /// Check if this warning undermines result confidence.
    pub fn is_result_undermining(&self) -> bool {
        true // Non-monotonic timer is always result-undermining
    }

    /// Convert to a ResolutionWarning from core (for unified handling).
    pub fn to_resolution_warning(&self) -> ResolutionWarning {
        match self {
            // Note: Core ResolutionWarning doesn't have NonMonotonic,
            // so we map to InsufficientResolution as a placeholder.
            // The actual handling should use this type directly.
            TimerSanityWarning::NonMonotonic { .. } => ResolutionWarning::InsufficientResolution {
                unique_values: 0,
                total_samples: 0,
                zero_fraction: 1.0,
                timer_resolution_ns: f64::INFINITY,
            },
        }
    }
}

/// Perform basic timer sanity check (spec §3.2).
///
/// Verifies monotonicity by taking 1000 consecutive timestamps.
///
/// # Arguments
///
/// * `_timer` - The timer to check (currently unused, uses rdtsc directly)
///
/// # Returns
///
/// `Some(TimerSanityWarning)` if non-monotonicity detected, `None` otherwise.
pub fn timer_sanity_check(_timer: &crate::measurement::BoxedTimer) -> Option<TimerSanityWarning> {
    let mut violations = 0;
    let mut max_jump = 0;

    let mut last = crate::measurement::rdtsc();
    for _ in 0..1000 {
        let current = crate::measurement::rdtsc();
        if current < last {
            violations += 1;
            max_jump = max_jump.max(last - current);
        }
        last = current;
    }

    if violations > 0 {
        return Some(TimerSanityWarning::NonMonotonic {
            violations,
            max_jump_cycles: max_jump,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement::{BoxedTimer, Timer};

    #[test]
    fn test_timer_sanity_check() {
        let timer = BoxedTimer::Standard(Timer::new());
        let result = timer_sanity_check(&timer);
        // Should not detect non-monotonicity on a working system
        assert!(
            result.is_none(),
            "Timer should be monotonic on a working system"
        );
    }
}
