//! CPU affinity pinning for reduced timing noise.
//!
//! Thread migration between CPU cores can introduce significant timing noise
//! due to cache invalidation and different core frequencies. This module provides
//! RAII-based CPU affinity pinning that automatically activates during timing
//! measurements.
//!
//! # Platform Behavior
//!
//! - **Linux**: Uses `sched_setaffinity` to pin the thread to its current CPU.
//!   This is enforced by the kernel and no special privileges are required.
//! - **macOS**: Uses `thread_policy_set` with `THREAD_AFFINITY_POLICY` to set
//!   an affinity hint. This is advisory only—the kernel may still migrate the
//!   thread, but tends to keep threads with the same affinity tag together.
//!
//! # Example
//!
//! ```ignore
//! use timing_oracle::measurement::affinity::{AffinityGuard, AffinityResult};
//!
//! // Pin to current CPU (RAII - auto-restores on drop)
//! let guard = match AffinityGuard::try_pin() {
//!     AffinityResult::Pinned(guard) => Some(guard),
//!     AffinityResult::NotPinned { reason } => {
//!         eprintln!("CPU affinity not available: {}", reason);
//!         None
//!     }
//! };
//!
//! // ... perform timing measurements ...
//!
//! // Guard dropped here, original affinity restored (Linux) or hint cleared (macOS)
//! ```

/// Result of attempting to pin CPU affinity.
#[derive(Debug)]
pub enum AffinityResult {
    /// Successfully pinned to current CPU; keep guard alive during measurement.
    Pinned(AffinityGuard),
    /// Could not pin affinity; measurement continues without pinning.
    NotPinned {
        /// Human-readable explanation of why pinning was not possible.
        reason: String,
    },
}

/// RAII guard that restores original CPU affinity when dropped.
///
/// On Linux, the original affinity mask is restored on drop.
/// On macOS, the affinity hint is advisory and no restore is needed.
pub struct AffinityGuard {
    /// Original CPU affinity mask to restore on drop (Linux only).
    #[cfg(target_os = "linux")]
    original_mask: libc::cpu_set_t,
    /// The CPU we pinned to (Linux only).
    #[cfg(target_os = "linux")]
    _pinned_cpu: usize,

    /// Mach thread port used for setting affinity (macOS only).
    #[cfg(target_os = "macos")]
    _thread_port: u32,
}

impl AffinityGuard {
    /// Try to pin the current thread to its current CPU.
    ///
    /// Returns `AffinityResult::Pinned(guard)` on success, or
    /// `AffinityResult::NotPinned { reason }` if pinning is not available
    /// or fails for any reason.
    pub fn try_pin() -> AffinityResult {
        #[cfg(target_os = "linux")]
        {
            Self::try_pin_linux()
        }

        #[cfg(target_os = "macos")]
        {
            Self::try_pin_macos()
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            AffinityResult::NotPinned {
                reason: "CPU affinity not supported on this platform".to_string(),
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn try_pin_linux() -> AffinityResult {
        use std::mem::MaybeUninit;

        unsafe {
            // Get current affinity mask
            let mut original_mask = MaybeUninit::<libc::cpu_set_t>::uninit();
            let result = libc::sched_getaffinity(
                0, // current thread
                std::mem::size_of::<libc::cpu_set_t>(),
                original_mask.as_mut_ptr(),
            );

            if result != 0 {
                return AffinityResult::NotPinned {
                    reason: format!(
                        "sched_getaffinity failed: {}",
                        std::io::Error::last_os_error()
                    ),
                };
            }

            let original_mask = original_mask.assume_init();

            // Get current CPU
            let current_cpu = libc::sched_getcpu();
            if current_cpu < 0 {
                return AffinityResult::NotPinned {
                    reason: format!(
                        "sched_getcpu failed: {}",
                        std::io::Error::last_os_error()
                    ),
                };
            }

            // Create new mask with only current CPU
            let mut new_mask: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut new_mask);
            libc::CPU_SET(current_cpu as usize, &mut new_mask);

            // Set affinity to current CPU only
            let result = libc::sched_setaffinity(
                0, // current thread
                std::mem::size_of::<libc::cpu_set_t>(),
                &new_mask,
            );

            if result != 0 {
                return AffinityResult::NotPinned {
                    reason: format!(
                        "sched_setaffinity failed: {}",
                        std::io::Error::last_os_error()
                    ),
                };
            }

            tracing::debug!("Pinned thread to CPU {}", current_cpu);

            AffinityResult::Pinned(AffinityGuard {
                original_mask,
                _pinned_cpu: current_cpu as usize,
            })
        }
    }

    #[cfg(target_os = "macos")]
    fn try_pin_macos() -> AffinityResult {
        // macOS thread affinity is advisory only via thread_policy_set.
        // We use THREAD_AFFINITY_POLICY to hint that this thread should
        // stay on its current core. The kernel may still migrate it.
        //
        // Note: This often fails with KERN_POLICY_STATIC (46) on newer macOS
        // or certain process configurations. This is expected and the code
        // degrades gracefully.

        unsafe {
            let thread_port = libc::pthread_mach_thread_np(libc::pthread_self());
            if thread_port == 0 {
                return AffinityResult::NotPinned {
                    reason: "Failed to get mach thread port".to_string(),
                };
            }

            // THREAD_AFFINITY_POLICY uses an affinity_tag to group threads.
            // Threads with the same tag are kept on the same core when possible.
            // We use a unique tag per call to hint "keep this thread on current core".
            #[repr(C)]
            struct ThreadAffinityPolicy {
                affinity_tag: i32,
            }

            const THREAD_AFFINITY_POLICY: u32 = 4;
            const THREAD_AFFINITY_POLICY_COUNT: u32 = 1;

            // Use a non-zero tag to enable affinity hint.
            // The specific value doesn't matter much since we're just trying to
            // reduce migration, not coordinate between threads.
            let policy = ThreadAffinityPolicy {
                affinity_tag: 1, // Non-zero enables affinity hint
            };

            // thread_policy_set is in the mach crate, but we can call it via libc
            extern "C" {
                fn thread_policy_set(
                    thread: u32,
                    flavor: u32,
                    policy_info: *const i32,
                    count: u32,
                ) -> i32;
            }

            let result = thread_policy_set(
                thread_port,
                THREAD_AFFINITY_POLICY,
                &policy.affinity_tag as *const i32,
                THREAD_AFFINITY_POLICY_COUNT,
            );

            if result != 0 {
                // KERN_SUCCESS = 0
                // Common error codes:
                // - 46 (KERN_POLICY_STATIC): Policy cannot be changed (common on newer macOS)
                // - 4 (KERN_INVALID_ARGUMENT): Invalid argument
                let reason = match result {
                    46 => "macOS kernel policy is static (KERN_POLICY_STATIC) - affinity hints not supported on this system".to_string(),
                    4 => "Invalid argument to thread_policy_set".to_string(),
                    _ => format!("thread_policy_set failed with code {}", result),
                };
                return AffinityResult::NotPinned { reason };
            }

            tracing::debug!("Set macOS thread affinity hint (advisory)");

            AffinityResult::Pinned(AffinityGuard {
                _thread_port: thread_port,
            })
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for AffinityGuard {
    fn drop(&mut self) {
        unsafe {
            // Restore original affinity mask
            let result = libc::sched_setaffinity(
                0,
                std::mem::size_of::<libc::cpu_set_t>(),
                &self.original_mask,
            );

            if result != 0 {
                tracing::warn!(
                    "Failed to restore CPU affinity: {}",
                    std::io::Error::last_os_error()
                );
            } else {
                tracing::debug!("Restored original CPU affinity");
            }
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for AffinityGuard {
    fn drop(&mut self) {
        // macOS affinity is advisory - no explicit restore needed.
        // The hint is associated with the thread and will be cleared
        // when the thread exits or can be overwritten.
        tracing::debug!("macOS affinity hint released (advisory, no restore needed)");
    }
}

// Implement Debug manually
impl std::fmt::Debug for AffinityGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(target_os = "linux")]
        {
            f.debug_struct("AffinityGuard")
                .field("pinned_cpu", &self._pinned_cpu)
                .field("platform", &"linux")
                .finish()
        }

        #[cfg(target_os = "macos")]
        {
            f.debug_struct("AffinityGuard")
                .field("thread_port", &self._thread_port)
                .field("platform", &"macos (advisory)")
                .finish()
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            f.debug_struct("AffinityGuard")
                .field("platform", &"unsupported")
                .finish()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_pin() {
        // Should either succeed or fail gracefully
        let result = AffinityGuard::try_pin();
        match result {
            AffinityResult::Pinned(guard) => {
                println!("Successfully pinned: {:?}", guard);
                // Guard dropped here, should restore
            }
            AffinityResult::NotPinned { reason } => {
                println!("Not pinned (expected on some platforms): {}", reason);
            }
        }
    }

    #[test]
    fn test_pin_and_restore() {
        // Acquire affinity, then drop and verify no errors
        let guard = AffinityGuard::try_pin();
        if let AffinityResult::Pinned(g) = guard {
            // Do some work
            std::hint::black_box(42);
            // Drop guard
            drop(g);
            // Should be able to pin again
            let guard2 = AffinityGuard::try_pin();
            assert!(
                matches!(guard2, AffinityResult::Pinned(_)),
                "Should be able to pin again after restore"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_linux_pin_to_current_cpu() {
        use std::mem::MaybeUninit;

        let guard = AffinityGuard::try_pin();
        if let AffinityResult::Pinned(ref g) = guard {
            // Verify we're pinned to a single CPU
            unsafe {
                let mut mask = MaybeUninit::<libc::cpu_set_t>::uninit();
                let result = libc::sched_getaffinity(
                    0,
                    std::mem::size_of::<libc::cpu_set_t>(),
                    mask.as_mut_ptr(),
                );
                assert_eq!(result, 0, "sched_getaffinity should succeed");

                let mask = mask.assume_init();
                let mut count = 0;
                for i in 0..libc::CPU_SETSIZE as usize {
                    if libc::CPU_ISSET(i, &mask) {
                        count += 1;
                    }
                }
                assert_eq!(count, 1, "Should be pinned to exactly one CPU");
                assert_eq!(
                    g._pinned_cpu,
                    (0..libc::CPU_SETSIZE as usize)
                        .find(|&i| libc::CPU_ISSET(i, &mask))
                        .unwrap(),
                    "Pinned CPU should match"
                );
            }
        }
    }
}
