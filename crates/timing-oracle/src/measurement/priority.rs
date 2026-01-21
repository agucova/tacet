//! Thread priority elevation for reduced preemption during measurement.
//!
//! Elevating thread priority helps reduce timing noise from preemption by
//! other processes. This module provides best-effort priority elevation
//! that fails silently if privileges are insufficient.
//!
//! Uses the `thread-priority` crate for cross-platform support.
//!
//! # Platform Behavior
//!
//! - **Linux**: Sets thread priority via pthread or nice value
//! - **macOS**: Sets thread QoS class or priority
//! - **Windows**: Sets thread priority class
//! - **Other Unix**: Falls back to nice value where available
//!
//! # Example
//!
//! ```ignore
//! use timing_oracle::measurement::priority::{PriorityGuard, PriorityResult};
//!
//! // Elevate priority (RAII - auto-restores on drop)
//! let guard = match PriorityGuard::try_elevate() {
//!     PriorityResult::Elevated(guard) => Some(guard),
//!     PriorityResult::NotElevated { reason } => {
//!         eprintln!("Priority elevation not available: {}", reason);
//!         None
//!     }
//! };
//!
//! // ... perform timing measurements ...
//!
//! // Guard dropped here, original priority restored
//! ```

use thread_priority::{ThreadPriority, ThreadPriorityValue};

/// Result of attempting to elevate thread priority.
#[derive(Debug)]
pub enum PriorityResult {
    /// Successfully elevated priority; keep guard alive during measurement.
    Elevated(PriorityGuard),
    /// Could not elevate priority; measurement continues at normal priority.
    NotElevated {
        /// Human-readable explanation of why elevation was not possible.
        reason: String,
    },
}

/// RAII guard that restores original thread priority when dropped.
pub struct PriorityGuard {
    /// Original thread priority to restore on drop.
    original_priority: ThreadPriority,
    /// Whether we successfully changed priority.
    priority_changed: bool,
}

impl PriorityGuard {
    /// Try to elevate the current thread's priority.
    ///
    /// Returns `PriorityResult::Elevated(guard)` on success,
    /// or `PriorityResult::NotElevated { reason }` if elevation was not possible.
    pub fn try_elevate() -> PriorityResult {
        // Get current priority to restore later
        let original_priority = match thread_priority::get_current_thread_priority() {
            Ok(p) => p,
            Err(e) => {
                return PriorityResult::NotElevated {
                    reason: format!("Failed to get current thread priority: {:?}", e),
                };
            }
        };

        // Try to set a higher priority
        // We use a moderately high priority (not max, to avoid starving system threads)
        // ThreadPriorityValue ranges from 0-99 on most platforms, with higher = more priority
        let target_priority = match ThreadPriorityValue::try_from(75u8) {
            Ok(v) => ThreadPriority::Crossplatform(v),
            Err(_) => {
                return PriorityResult::NotElevated {
                    reason: "Failed to create valid priority value".to_string(),
                };
            }
        };

        match thread_priority::set_current_thread_priority(target_priority) {
            Ok(()) => {
                tracing::debug!(
                    "Elevated thread priority from {:?} to {:?}",
                    original_priority,
                    target_priority
                );
                PriorityResult::Elevated(PriorityGuard {
                    original_priority,
                    priority_changed: true,
                })
            }
            Err(e) => {
                // Priority elevation failed - this is expected without elevated privileges
                // on many systems. Return success with priority_changed = false so we
                // still restore on drop (no-op).
                tracing::debug!("Thread priority elevation failed (expected without privileges): {:?}", e);
                PriorityResult::NotElevated {
                    reason: format!(
                        "Priority elevation requires elevated privileges: {:?}",
                        e
                    ),
                }
            }
        }
    }
}

impl Drop for PriorityGuard {
    fn drop(&mut self) {
        if self.priority_changed {
            if let Err(e) =
                thread_priority::set_current_thread_priority(self.original_priority.clone())
            {
                tracing::warn!("Failed to restore thread priority: {:?}", e);
            } else {
                tracing::debug!("Restored thread priority to {:?}", self.original_priority);
            }
        }
    }
}

impl std::fmt::Debug for PriorityGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PriorityGuard")
            .field("original_priority", &self.original_priority)
            .field("priority_changed", &self.priority_changed)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_elevate() {
        // Should either succeed or fail gracefully
        let result = PriorityGuard::try_elevate();
        match result {
            PriorityResult::Elevated(guard) => {
                println!("Successfully elevated priority: {:?}", guard);
                // Guard dropped here, should restore
            }
            PriorityResult::NotElevated { reason } => {
                println!(
                    "Priority not elevated (expected without privileges): {}",
                    reason
                );
            }
        }
    }

    #[test]
    fn test_elevate_and_restore() {
        // Acquire priority, then drop and verify no errors
        let guard = PriorityGuard::try_elevate();
        if let PriorityResult::Elevated(g) = guard {
            // Do some work
            std::hint::black_box(42);
            // Drop guard
            drop(g);
            // Should be able to elevate again
            let guard2 = PriorityGuard::try_elevate();
            // May or may not succeed depending on system state
            match guard2 {
                PriorityResult::Elevated(_) => println!("Re-elevated successfully"),
                PriorityResult::NotElevated { reason } => {
                    println!("Re-elevation failed (ok): {}", reason)
                }
            }
        }
    }
}
