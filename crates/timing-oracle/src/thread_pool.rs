//! Custom thread pool configuration for parallel operations.
//!
//! This module provides a shared thread pool with increased stack size
//! to prevent stack overflow when running many concurrent timing tests.

#[cfg(feature = "parallel")]
use rayon::ThreadPool;

#[cfg(feature = "parallel")]
use std::sync::OnceLock;

#[cfg(feature = "parallel")]
static THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Get or initialize the shared thread pool with custom configuration.
///
/// The thread pool is configured with:
/// - Stack size: 8 MB (vs rayon's default 2 MB)
/// - Thread count: Number of logical CPUs
///
/// This prevents stack overflow when running many concurrent timing tests,
/// each of which performs parallel bootstrap operations.
#[cfg(feature = "parallel")]
pub fn get_thread_pool() -> &'static ThreadPool {
    THREAD_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .stack_size(8 * 1024 * 1024) // 8 MB stack per thread
            .build()
            .expect("Failed to build custom thread pool")
    })
}

/// Execute a parallel operation using the custom thread pool.
///
/// This ensures all parallel operations in the library use the same
/// thread pool with appropriate stack configuration.
#[cfg(feature = "parallel")]
pub fn install<OP, R>(op: OP) -> R
where
    OP: FnOnce() -> R + Send,
    R: Send,
{
    get_thread_pool().install(op)
}

#[cfg(not(feature = "parallel"))]
pub fn install<OP, R>(op: OP) -> R
where
    OP: FnOnce() -> R,
{
    // No parallel feature - just execute directly
    op()
}
