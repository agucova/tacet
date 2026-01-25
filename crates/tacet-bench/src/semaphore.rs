//! Simple semaphore implementation for limiting concurrent operations.
//!
//! Used to limit concurrent timer creation in realistic mode benchmarks.

use std::sync::{Arc, Condvar, Mutex};

/// A simple counting semaphore.
#[derive(Clone)]
pub struct Semaphore {
    inner: Arc<SemaphoreInner>,
}

struct SemaphoreInner {
    permits: Mutex<usize>,
    condvar: Condvar,
}

impl Semaphore {
    /// Create a new semaphore with the given number of permits.
    pub fn new(permits: usize) -> Self {
        Self {
            inner: Arc::new(SemaphoreInner {
                permits: Mutex::new(permits),
                condvar: Condvar::new(),
            }),
        }
    }

    /// Acquire a permit, blocking if none are available.
    ///
    /// Returns a guard that releases the permit when dropped.
    pub fn acquire(&self) -> SemaphoreGuard {
        let mut permits = self.inner.permits.lock().unwrap();
        while *permits == 0 {
            permits = self.inner.condvar.wait(permits).unwrap();
        }
        *permits -= 1;
        SemaphoreGuard {
            semaphore: self.clone(),
        }
    }

    /// Release a permit.
    fn release(&self) {
        let mut permits = self.inner.permits.lock().unwrap();
        *permits += 1;
        self.inner.condvar.notify_one();
    }
}

/// RAII guard that releases a semaphore permit when dropped.
pub struct SemaphoreGuard {
    semaphore: Semaphore,
}

impl Drop for SemaphoreGuard {
    fn drop(&mut self) {
        self.semaphore.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_semaphore_limits_concurrency() {
        let sem = Semaphore::new(2);
        let counter = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];
        for _ in 0..10 {
            let sem = sem.clone();
            let counter = counter.clone();
            let max_concurrent = max_concurrent.clone();

            let handle = thread::spawn(move || {
                let _guard = sem.acquire();
                let current = counter.fetch_add(1, Ordering::SeqCst) + 1;

                // Update max if this is higher
                let mut max = max_concurrent.load(Ordering::SeqCst);
                while current > max {
                    match max_concurrent.compare_exchange(
                        max,
                        current,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => break,
                        Err(x) => max = x,
                    }
                }

                thread::sleep(Duration::from_millis(10));
                counter.fetch_sub(1, Ordering::SeqCst);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should never have had more than 2 concurrent
        assert!(max_concurrent.load(Ordering::SeqCst) <= 2);
    }
}
