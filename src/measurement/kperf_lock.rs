//! File-based locking for kperf exclusive PMU access.
//!
//! The macOS kpc API requires exclusive system-wide access to PMU counters via
//! `kpc_force_all_ctrs_set(1)`. When multiple processes try to initialize kperf
//! simultaneously (e.g., nextest running tests in parallel), only one can succeed.
//!
//! This module provides file-based locking to serialize kperf initialization across
//! processes, with graceful fallback when the lock cannot be acquired.
//!
//! # Deadlock Resistance
//!
//! - Uses `LOCK_NB` (non-blocking) with timeout - never blocks indefinitely
//! - `flock` auto-releases when process exits (even on crash/SIGKILL)
//! - Single resource (one lock file) - circular dependencies impossible

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::time::{Duration, Instant};

extern crate libc;

/// Lock file path for kperf PMU access serialization.
/// Using /tmp ensures world-writable access and local filesystem (not NFS).
const LOCK_FILE_PATH: &str = "/tmp/timing-oracle-kperf.lock";

/// Default timeout for acquiring the lock.
/// 200ms is generous for typical test initialization while not blocking too long.
const DEFAULT_LOCK_TIMEOUT: Duration = Duration::from_millis(200);

/// Result of attempting to acquire the kperf lock.
#[derive(Debug)]
pub enum LockResult {
    /// Lock acquired successfully; holder must keep the guard alive.
    Acquired(LockGuard),
    /// Lock acquisition timed out; another process holds PMU access.
    Timeout,
    /// Lock file could not be created/opened.
    IoError(io::Error),
}

/// RAII guard that releases the lock when dropped.
///
/// The lock is automatically released when:
/// - This guard is dropped
/// - The process exits (normally or via signal)
/// - The file descriptor is closed
pub struct LockGuard {
    file: File,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        // flock is automatically released when file is closed,
        // but explicit unlock is good practice
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

// Implement Debug manually to avoid exposing file internals
impl std::fmt::Debug for LockGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LockGuard")
            .field("locked", &true)
            .finish()
    }
}

/// Try to acquire exclusive lock for kperf PMU access.
///
/// # Arguments
///
/// * `timeout` - Maximum time to wait for the lock
///
/// # Returns
///
/// - `LockResult::Acquired(guard)` - Lock acquired, keep guard alive while using PMU
/// - `LockResult::Timeout` - Another process holds the lock
/// - `LockResult::IoError(e)` - Failed to create/open lock file
pub fn try_acquire(timeout: Duration) -> LockResult {
    // Open/create lock file with world-writable permissions (0666)
    // This ensures any user can acquire the lock, even if a previous run
    // created it as root. The mode is masked by umask on creation.
    let file = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o666)
        .open(LOCK_FILE_PATH)
    {
        Ok(f) => f,
        Err(e) => return LockResult::IoError(e),
    };

    let fd = file.as_raw_fd();
    let start = Instant::now();

    // Try non-blocking lock in a loop with small sleeps
    loop {
        // LOCK_EX = exclusive lock, LOCK_NB = non-blocking
        let result = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };

        if result == 0 {
            // Lock acquired successfully
            return LockResult::Acquired(LockGuard { file });
        }

        // Check errno - EWOULDBLOCK means lock held by another process
        let errno = io::Error::last_os_error();
        if errno.kind() != io::ErrorKind::WouldBlock {
            // Unexpected error
            return LockResult::IoError(errno);
        }

        // Check if we've exceeded timeout
        if start.elapsed() >= timeout {
            return LockResult::Timeout;
        }

        // Sleep briefly before retrying (10ms)
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Acquire lock with default timeout (200ms).
pub fn try_acquire_default() -> LockResult {
    try_acquire(DEFAULT_LOCK_TIMEOUT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_acquire_release() {
        // Should be able to acquire lock
        let result = try_acquire_default();
        assert!(
            matches!(result, LockResult::Acquired(_)),
            "Should acquire lock: {:?}",
            result
        );

        // Guard dropped here, lock released
    }

    #[test]
    fn test_lock_contention_same_process() {
        // flock operates per-file-descriptor, not per-process.
        // Opening a new fd to the same file creates a separate lock that
        // will contend with the first one, even within the same process.
        let guard1 = try_acquire_default();
        assert!(matches!(guard1, LockResult::Acquired(_)));

        // Second acquire (different fd) should timeout since first holds lock
        let result = try_acquire(Duration::from_millis(50));
        assert!(
            matches!(result, LockResult::Timeout),
            "Expected timeout, got: {:?}",
            result
        );

        // After dropping first guard, should be able to acquire
        drop(guard1);
        let guard3 = try_acquire_default();
        assert!(matches!(guard3, LockResult::Acquired(_)));
    }
}
