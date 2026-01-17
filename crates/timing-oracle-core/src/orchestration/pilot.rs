//! Pilot phase decision logic for batch size selection.
//!
//! The pilot phase determines the optimal batch size K to achieve sufficient
//! timer resolution for accurate measurements. This module provides platform-independent
//! logic for making this decision based on operation timing characteristics.

extern crate alloc;

use crate::math::ceil;
use alloc::string::String;

/// Default minimum ticks per measurement for reliable quantization.
pub const DEFAULT_TARGET_TICKS: f64 = 50.0;

/// Default maximum batch size.
pub const DEFAULT_MAX_BATCH_K: usize = 20;

/// Result of pilot phase analysis.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct PilotDecision {
    /// Selected batch size K.
    pub batch_k: usize,

    /// Whether batching is enabled (K > 1).
    pub batching_enabled: bool,

    /// Estimated ticks per batch with the selected K.
    pub ticks_per_batch: f64,

    /// If Some, the operation is too fast to measure reliably.
    pub unmeasurable: Option<UnmeasurableInfo>,

    /// Human-readable explanation of the decision.
    pub rationale: String,
}

/// Information about an unmeasurable operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "std", derive(serde::Serialize, serde::Deserialize))]
pub struct UnmeasurableInfo {
    /// Median operation time in nanoseconds.
    pub operation_ns: f64,

    /// Minimum measurable threshold in nanoseconds.
    pub threshold_ns: f64,

    /// Ticks per single call.
    pub ticks_per_call: f64,
}

/// Compute the optimal batch size K for timing measurements.
///
/// Given the median operation time and timer resolution, determines the
/// smallest K that achieves the target ticks per measurement.
///
/// # Arguments
///
/// * `median_cycles` - Median operation time in cycles from pilot measurements
/// * `cycles_per_ns` - Timer's cycles per nanosecond conversion factor
/// * `timer_resolution_ns` - Timer resolution in nanoseconds
/// * `target_ticks_per_batch` - Target minimum ticks per measurement (default 50)
/// * `max_batch_k` - Maximum allowed batch size (default 20)
///
/// # Returns
///
/// A `PilotDecision` with the selected batch size and rationale.
pub fn compute_batch_k(
    median_cycles: u64,
    cycles_per_ns: f64,
    timer_resolution_ns: f64,
    target_ticks_per_batch: f64,
    max_batch_k: usize,
) -> PilotDecision {
    // Convert median cycles to nanoseconds
    let median_ns = median_cycles as f64 / cycles_per_ns;

    // Calculate ticks per call (how many timer ticks per single operation)
    let ticks_per_call = median_ns / timer_resolution_ns;

    compute_batch_k_from_ticks(
        ticks_per_call,
        median_ns,
        timer_resolution_ns,
        target_ticks_per_batch,
        max_batch_k,
    )
}

/// Compute the optimal batch size K from ticks-per-call.
///
/// This is the core decision logic, separated for easier testing.
///
/// # Arguments
///
/// * `ticks_per_call` - How many timer ticks per single operation
/// * `median_ns` - Median operation time in nanoseconds (for rationale)
/// * `timer_resolution_ns` - Timer resolution in nanoseconds
/// * `target_ticks_per_batch` - Target minimum ticks per measurement
/// * `max_batch_k` - Maximum allowed batch size
///
/// # Returns
///
/// A `PilotDecision` with the selected batch size and rationale.
pub fn compute_batch_k_from_ticks(
    ticks_per_call: f64,
    median_ns: f64,
    timer_resolution_ns: f64,
    target_ticks_per_batch: f64,
    max_batch_k: usize,
) -> PilotDecision {
    use alloc::format;

    if ticks_per_call >= target_ticks_per_batch {
        // No batching needed - individual measurements have enough resolution
        return PilotDecision {
            batch_k: 1,
            batching_enabled: false,
            ticks_per_batch: ticks_per_call,
            unmeasurable: None,
            rationale: format!(
                "no batching needed ({:.1} ticks/call >= {:.0} target)",
                ticks_per_call, target_ticks_per_batch
            ),
        };
    }

    // Need batching to achieve target tick density
    let k_raw = ceil(target_ticks_per_batch / ticks_per_call) as usize;
    let k_attempt = k_raw.clamp(1, max_batch_k);
    let actual_ticks = ticks_per_call * k_attempt as f64;

    // Check if even with max batching we're still below measurability threshold
    if actual_ticks < target_ticks_per_batch {
        let threshold_ns = timer_resolution_ns * target_ticks_per_batch / k_attempt as f64;
        let rationale = format!(
            "UNMEASURABLE: {:.1} ticks/batch < {:.0} minimum even at K={} (op ~{:.2}ns, threshold ~{:.2}ns)",
            actual_ticks, target_ticks_per_batch, k_attempt, median_ns, threshold_ns
        );

        return PilotDecision {
            batch_k: 1, // Revert to k=1 for reporting unmeasurable
            batching_enabled: false,
            ticks_per_batch: ticks_per_call,
            unmeasurable: Some(UnmeasurableInfo {
                operation_ns: median_ns,
                threshold_ns,
                ticks_per_call,
            }),
            rationale,
        };
    }

    // Batching achieves target
    let rationale = format!(
        "K={} ({:.1} ticks/batch, {:.2} ticks/call, timer res {:.1}ns)",
        k_attempt, actual_ticks, ticks_per_call, timer_resolution_ns
    );

    PilotDecision {
        batch_k: k_attempt,
        batching_enabled: k_attempt > 1,
        ticks_per_batch: actual_ticks,
        unmeasurable: None,
        rationale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_batching_needed() {
        let decision = compute_batch_k_from_ticks(
            100.0, // 100 ticks per call
            1000.0, 10.0, 50.0, // target 50 ticks
            20,
        );

        assert_eq!(decision.batch_k, 1);
        assert!(!decision.batching_enabled);
        assert!(decision.unmeasurable.is_none());
    }

    #[test]
    fn test_batching_needed() {
        let decision = compute_batch_k_from_ticks(
            10.0, // 10 ticks per call
            100.0, 10.0, 50.0, // target 50 ticks
            20,
        );

        // Need K=5 to get 50 ticks (10 * 5 = 50)
        assert_eq!(decision.batch_k, 5);
        assert!(decision.batching_enabled);
        assert!(decision.unmeasurable.is_none());
        assert!((decision.ticks_per_batch - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_unmeasurable() {
        let decision = compute_batch_k_from_ticks(
            0.5, // 0.5 ticks per call (very fast)
            5.0, 10.0, 50.0, // target 50 ticks
            20,   // max K=20
        );

        // Even at K=20, only get 10 ticks
        assert!(decision.unmeasurable.is_some());
        let info = decision.unmeasurable.unwrap();
        assert!((info.operation_ns - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_max_batch_k_cap() {
        let decision = compute_batch_k_from_ticks(
            5.0, // 5 ticks per call
            50.0, 10.0, 200.0, // target 200 ticks
            20,    // max K=20
        );

        // Would need K=40, but capped at 20
        // K=20 gives 100 ticks, which is < 200 target
        assert!(decision.unmeasurable.is_some());
    }
}
