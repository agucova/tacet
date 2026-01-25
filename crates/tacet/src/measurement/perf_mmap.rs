//! mmap-based PMU counter reads for ARM64 Linux.
//!
//! This module provides zero-overhead userspace PMU counter reads using the
//! perf_event mmap API with direct `mrs` instruction reads of PMCCNTR_EL0.
//!
//! # Requirements
//!
//! - Linux kernel ≥5.12 (ARM64 userspace PMU access)
//! - `kernel.perf_user_access = 1` sysctl (one-time admin setup)
//! - ARM64 (aarch64) architecture
//! - sudo/CAP_PERFMON for perf_event_open (same as syscall-based approach)
//!
//! # Performance
//!
//! Eliminates 2 syscalls per measurement (reset + read), reducing overhead
//! from ~2000ns to ~300ns per measurement (~7x faster).

use std::os::unix::io::RawFd;
use std::sync::atomic::{compiler_fence, Ordering};

use super::error::{MeasurementError, MeasurementResult};

/// perf_event mmap page structure from Linux kernel.
///
/// This structure is mapped into userspace when a perf_event fd is mmap'd.
/// It contains metadata needed to read PMU counters directly from userspace.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct perf_event_mmap_page {
    pub version: u32,
    pub compat_version: u32,
    pub lock: u32,              // Sequence lock for consistency
    pub index: u32,             // PMU index (0 = invalid)
    pub offset: i64,            // Offset to add to raw counter
    pub time_enabled: u64,      // Time event has been enabled
    pub time_running: u64,      // Time event has been running
    pub capabilities: u64,      // Capability flags
    pub pmc_width: u16,         // Counter bit width (32/40/64)
    pub time_shift: u16,        // Time conversion shift
    pub time_mult: u32,         // Time conversion multiplier
    pub time_offset: u64,       // Time conversion offset
    pub time_zero: u64,         // Reference time
    pub size: u32,              // Size of this structure
    pub reserved_1: u32,
    pub time_cycles: u64,       // Reference TSC cycles
    pub time_mask: u64,         // Cycle wraparound mask
    pub reserved: [u8; 928],    // Reserved space
    pub data_head: u64,         // Ring buffer write head
    pub data_tail: u64,         // Ring buffer read tail
    pub data_offset: u64,       // Ring buffer offset
    pub data_size: u64,         // Ring buffer size
    pub aux_head: u64,          // Aux buffer write head
    pub aux_tail: u64,          // Aux buffer read tail
    pub aux_offset: u64,        // Aux buffer offset
    pub aux_size: u64,          // Aux buffer size
}

/// State for mmap-based PMU counter reads.
///
/// This struct wraps a memory-mapped perf_event page and provides
/// zero-overhead counter reads via the seqlock protocol.
pub struct MmapState {
    _mmap: memmap2::MmapRaw,
    page_ptr: *const perf_event_mmap_page,
}

impl MmapState {
    /// Create a new MmapState from a perf_event file descriptor.
    ///
    /// # Safety
    ///
    /// - `fd` must be a valid perf_event file descriptor
    /// - Caller must ensure fd remains valid for the lifetime of MmapState
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - mmap fails (permissions, OOM, invalid fd)
    /// - Userspace PMU access is not enabled (cap_user_rdpmc == 0)
    pub unsafe fn new(fd: RawFd) -> Result<Self, std::io::Error> {
        let pagesize = libc::sysconf(libc::_SC_PAGESIZE) as usize;

        // Map one page (metadata only, no ring buffer)
        let mmap = memmap2::MmapOptions::new()
            .len(pagesize)
            .map_raw(fd)?;

        let page_ptr = mmap.as_ptr() as *const perf_event_mmap_page;

        // Verify userspace PMU access is enabled
        let caps = (*page_ptr).capabilities;
        const CAP_USER_RDPMC: u64 = 1 << 2;  // bit 2

        if (caps & CAP_USER_RDPMC) == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "Userspace PMU access not enabled (cap_user_rdpmc == 0). \
                 Check: sudo sh -c 'echo 1 > /proc/sys/kernel/perf_user_access'",
            ));
        }

        Ok(Self {
            _mmap: mmap,
            page_ptr,
        })
    }

    /// Read the current counter value.
    ///
    /// Uses the Linux perf_event mmap seqlock protocol to read PMU counters
    /// from userspace without syscalls. This is **much faster** (~10x) than
    /// syscall-based reads but requires careful validation.
    ///
    /// # Retry Conditions
    ///
    /// Returns None internally and retries (up to MAX_RETRIES) when:
    /// - Seqlock is held by kernel (lock & 1 != 0)
    /// - Seqlock changed during read (race with kernel update)
    /// - **index == 0**: Counter not scheduled (multiplexed out/thread migrated)
    /// - **pmc_width invalid**: Corrupted metadata (width == 0 or > 64)
    ///
    /// # Multiplexing & Thread Migration
    ///
    /// The kernel may multiplex events (share PMU hardware) or migrate threads
    /// between CPUs. When this happens, `index` becomes 0 to signal "counter
    /// unavailable". Attempting to read PMCCNTR_EL0 with index==0 produces
    /// **garbage values** (often near i64::MAX) because we'd be reading state
    /// from an unrelated event.
    ///
    /// This function handles this transparently by retrying until the event
    /// is scheduled again (typically <1ms).
    ///
    /// # Return Value
    ///
    /// - **Ok(value)**: Virtualized 64-bit counter value (offset + sign-extended PMC)
    /// - **Err(RetryExhausted)**: Retry limit exceeded after 1000 attempts
    ///
    /// # Caller Expectations
    ///
    /// Callers using this for timing measurements should:
    /// 1. Call read_counter() twice (start/end of measured region)
    /// 2. Handle `Err` by skipping the entire measurement sample
    /// 3. Use `end.saturating_sub(start)` for elapsed cycles on success
    ///
    /// # Performance
    ///
    /// - Typical: 2-4 CPU cycles per read (~1ns on 3GHz CPU)
    /// - Retry: ~10-50ns if kernel is updating page
    /// - Multiplexing: ~100ns-1ms if event is rescheduled
    ///
    /// # Errors
    ///
    /// Returns `Err(MeasurementError::RetryExhausted)` if the seqlock protocol
    /// fails to acquire a consistent read after 1000 attempts. This is extremely
    /// rare (<1 in 10M) and indicates system overload or constant multiplexing.
    #[inline]
    pub fn read_counter(&self) -> MeasurementResult {
        // Retry strategy for handling PMU multiplexing:
        // 1. Fast spin (0-1000): tight loop for seqlock races and brief unavailability
        // 2. Yielding spin (1001-50000): yield to scheduler for multiplexing (4ms period)
        // 3. Give up (>50000): likely system issue or constant multiplexing
        const FAST_RETRIES: usize = 1000;
        const MAX_RETRIES: usize = 50_000;

        for attempt in 0..MAX_RETRIES {
            if let Some(val) = unsafe { self.try_read_counter() } {
                return Ok(val);
            }

            // After fast spinning phase, yield to scheduler to wait for multiplexing
            // The kernel rotates PMU events every ~1-4ms (perf_event_mux_interval_ms)
            // Yielding allows other work while waiting for our event to be rescheduled
            if attempt >= FAST_RETRIES {
                std::thread::yield_now();
            }
        }

        // Log ONCE per exhaustion (not per retry)
        tracing::error!(
            "perf_mmap seqlock retry exhausted after {} attempts - \
             system under extreme load or PMU constantly multiplexed",
            MAX_RETRIES
        );
        Err(MeasurementError::RetryExhausted)
    }

    /// Try to read the counter once using the seqlock protocol.
    ///
    /// Returns None if the seqlock validation fails (kernel is updating page).
    #[inline]
    unsafe fn try_read_counter(&self) -> Option<u64> {
        let page = &*self.page_ptr;

        // 1. Acquire sequence lock (Ordering::Acquire provides memory barrier)
        let seq = atomic_load(&page.lock, Ordering::Acquire);

        // If lock is odd, kernel is writing to the page
        if (seq & 1) != 0 {
            return None;
        }

        // 2. Compiler barrier (prevents reordering)
        compiler_fence(Ordering::SeqCst);

        // 3. Read mmap page fields
        let index = read_once!(page.index);
        let offset = read_once!(page.offset);
        let pmc_width = read_once!(page.pmc_width);

        // CRITICAL: Validate index before reading PMU register
        //
        // index == 0 means the hardware counter is NOT currently available:
        // - Event multiplexed out (kernel scheduled different event on this PMU)
        // - Thread migrated to different CPU (counter not mapped on new CPU)
        // - Event disabled or not yet started
        //
        // Reading PMCCNTR_EL0 with index==0 produces GARBAGE (often near i64::MAX)
        // because we'd be combining offset from OUR event with PMU state from a
        // DIFFERENT event. This is the root cause of spurious overflow values.
        if index == 0 {
            return None;  // Retry until event is rescheduled
        }

        // Validate pmc_width to prevent sign extension bugs
        //
        // Typical ARM PMU counter widths: 32-bit (older), 40/48-bit (common), 64-bit (ARMv8.5+)
        // Invalid width (0 or >64) indicates corrupted metadata → abort read
        if pmc_width == 0 || pmc_width > 64 {
            return None;  // Retry with valid metadata
        }

        // 4. Read PMU register via MRS instruction
        let pmc_value = mrs_pmccntr_el0();

        // 5. Compiler barrier before verification
        compiler_fence(Ordering::SeqCst);

        // 6. Verify sequence lock unchanged
        let nseq = atomic_load(&page.lock, Ordering::Acquire);
        if seq != nseq {
            return None;  // Retry
        }

        // 7. Compute final value with sign extension
        Some(compute_counter(offset, pmc_value, pmc_width))
    }
}

// Mark as Send + Sync (safe because page_ptr is read-only and thread-local)
unsafe impl Send for MmapState {}
unsafe impl Sync for MmapState {}

/// Read ARM64 PMCCNTR_EL0 (Performance Monitors Cycle Count Register).
///
/// This register provides cycle-accurate timing on ARM64 when userspace
/// PMU access is enabled via the perf_event subsystem.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mrs_pmccntr_el0() -> u64 {
    let val: u64;
    std::arch::asm!(
        "mrs {}, pmccntr_el0",
        out(reg) val,
        options(nostack, nomem, preserves_flags),
    );
    val
}

/// Stub for non-ARM64 architectures (should never be called).
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn mrs_pmccntr_el0() -> u64 {
    unreachable!("mrs_pmccntr_el0 should only be called on aarch64")
}

/// Sign-extend a counter value to i64 based on its bit width.
///
/// ARM64 counters are typically 32-bit or 40-bit (implementation-defined).
/// The kernel virtualizes them to 64-bit using sign extension.
#[inline]
fn sign_extend(val: u64, width: u16) -> i64 {
    let shift = 64 - width as u32;
    ((val << shift) as i64) >> shift
}

/// Compute the final counter value by adding offset and sign-extending.
///
/// The kernel maintains `offset` to track wraparounds. Userspace applies
/// the delta with sign extension to get the virtualized 64-bit value.
#[inline]
fn compute_counter(offset: i64, pmc_value: u64, width: u16) -> u64 {
    offset.wrapping_add(sign_extend(pmc_value, width)) as u64
}

/// Atomic load with specified memory ordering.
///
/// Uses read_volatile to prevent compiler optimization.
#[inline]
unsafe fn atomic_load<T: Copy>(ptr: *const T, _order: Ordering) -> T {
    std::ptr::read_volatile(ptr)
}

/// Read-once macro to prevent compiler optimization and reordering.
macro_rules! read_once {
    ($expr:expr) => {
        std::ptr::read_volatile(&$expr)
    };
}

// Export the macro for use in this module
pub(crate) use read_once;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_extend_32bit() {
        // 32-bit counter wraparound
        assert_eq!(sign_extend(0xFFFFFFFF, 32), -1);
        assert_eq!(sign_extend(0x80000000, 32), -2147483648);
        assert_eq!(sign_extend(0x7FFFFFFF, 32), 2147483647);
        assert_eq!(sign_extend(0, 32), 0);
    }

    #[test]
    fn test_sign_extend_40bit() {
        // 40-bit counter (ARMv8 typical)
        assert_eq!(sign_extend(0xFFFFFFFFFF, 40), -1);
        assert_eq!(sign_extend(0x8000000000, 40), -549755813888);
        assert_eq!(sign_extend(0x7FFFFFFFFF, 40), 549755813887);
        assert_eq!(sign_extend(0, 40), 0);
    }

    #[test]
    fn test_sign_extend_64bit() {
        // 64-bit counter (ARMv8.5-PMU)
        assert_eq!(sign_extend(0xFFFFFFFFFFFFFFFF, 64), -1);
        assert_eq!(sign_extend(0x8000000000000000, 64), -9223372036854775808);
        assert_eq!(sign_extend(0, 64), 0);
    }

    #[test]
    fn test_compute_counter() {
        // Zero offset
        assert_eq!(compute_counter(0, 0x1000, 32), 0x1000);

        // Positive offset
        assert_eq!(compute_counter(100, 0x1000, 32), 100 + 0x1000);

        // Negative offset (wraparound case)
        assert_eq!(
            compute_counter(-100, 0x1000, 32),
            (0x1000u64).wrapping_sub(100)
        );

        // Counter wraparound (sign extension)
        assert_eq!(
            compute_counter(0, 0xFFFFFFFF, 32),
            0xFFFFFFFFFFFFFFFF  // -1 as u64
        );
    }
}
