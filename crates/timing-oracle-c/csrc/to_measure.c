/**
 * @file to_measure.c
 * @brief Timing-critical measurement loop for timing-oracle-c
 *
 * This file contains the only C code in the library: the hot measurement loop
 * that needs maximum control over timing and minimal overhead.
 *
 * Platform-specific timer reads are implemented using inline assembly where
 * possible, with a fallback to clock_gettime for portability.
 */

#include "to_measure.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ============================================================================
 * Platform-specific timer implementations
 * ============================================================================ */

#if defined(__x86_64__) || defined(_M_X64)
/* x86-64: Use RDTSC with LFENCE for serialization */

#include <time.h>

static inline uint64_t read_timer(void) {
    unsigned int lo, hi;
    /* LFENCE ensures all prior loads complete before RDTSC */
    __asm__ volatile (
        "lfence\n\t"
        "rdtsc"
        : "=a"(lo), "=d"(hi)
        :
        : "memory"
    );
    return ((uint64_t)hi << 32) | lo;
}

static const char* timer_name = "rdtsc";

/* Cached TSC frequency (computed once on first call) */
static uint64_t cached_tsc_freq = 0;

/* Calibrate TSC frequency by measuring against clock_gettime.
 * This provides a more accurate frequency than hardcoding 3GHz. */
static uint64_t calibrate_tsc_frequency(void) {
    struct timespec ts_start, ts_end;
    uint64_t tsc_start, tsc_end;

    /* Measure for ~10ms */
    const uint64_t target_ns = 10000000ULL;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    tsc_start = read_timer();

    /* Busy-wait for target duration */
    uint64_t elapsed_ns = 0;
    while (elapsed_ns < target_ns) {
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        elapsed_ns = (uint64_t)(ts_end.tv_sec - ts_start.tv_sec) * 1000000000ULL
                   + (uint64_t)(ts_end.tv_nsec - ts_start.tv_nsec);
    }

    tsc_end = read_timer();

    uint64_t tsc_delta = tsc_end - tsc_start;
    if (elapsed_ns == 0) {
        return 3000000000ULL; /* Fallback to 3 GHz */
    }

    /* freq = tsc_delta / (elapsed_ns / 1e9) = tsc_delta * 1e9 / elapsed_ns */
    return (tsc_delta * 1000000000ULL) / elapsed_ns;
}

static uint64_t get_timer_freq(void) {
    if (cached_tsc_freq == 0) {
        cached_tsc_freq = calibrate_tsc_frequency();
    }
    return cached_tsc_freq;
}

#elif defined(__aarch64__) || defined(_M_ARM64)
/* ARM64: Use CNTVCT_EL0 (virtual counter) with ISB for serialization */

static inline uint64_t read_timer(void) {
    uint64_t val;
    /* ISB ensures all prior instructions complete before reading counter */
    __asm__ volatile (
        "isb\n\t"
        "mrs %0, cntvct_el0"
        : "=r"(val)
        :
        : "memory"
    );
    return val;
}

static const char* timer_name = "cntvct_el0";

/* Read CNTFRQ_EL0 for actual frequency */
static uint64_t get_timer_freq(void) {
    uint64_t freq;
    __asm__ volatile ("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

#elif defined(__i386__) || defined(_M_IX86)
/* x86 32-bit: Use RDTSC */

#include <time.h>

static inline uint64_t read_timer(void) {
    unsigned int lo, hi;
    __asm__ volatile (
        "lfence\n\t"
        "rdtsc"
        : "=a"(lo), "=d"(hi)
        :
        : "memory"
    );
    return ((uint64_t)hi << 32) | lo;
}

static const char* timer_name = "rdtsc";

/* Cached TSC frequency (computed once on first call) */
static uint64_t cached_tsc_freq_i386 = 0;

static uint64_t calibrate_tsc_frequency_i386(void) {
    struct timespec ts_start, ts_end;
    uint64_t tsc_start, tsc_end;
    const uint64_t target_ns = 10000000ULL;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    tsc_start = read_timer();

    uint64_t elapsed_ns = 0;
    while (elapsed_ns < target_ns) {
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        elapsed_ns = (uint64_t)(ts_end.tv_sec - ts_start.tv_sec) * 1000000000ULL
                   + (uint64_t)(ts_end.tv_nsec - ts_start.tv_nsec);
    }

    tsc_end = read_timer();
    uint64_t tsc_delta = tsc_end - tsc_start;
    if (elapsed_ns == 0) {
        return 3000000000ULL;
    }
    return (tsc_delta * 1000000000ULL) / elapsed_ns;
}

static uint64_t get_timer_freq(void) {
    if (cached_tsc_freq_i386 == 0) {
        cached_tsc_freq_i386 = calibrate_tsc_frequency_i386();
    }
    return cached_tsc_freq_i386;
}

#elif defined(__arm__) && !defined(__aarch64__)
/* ARM 32-bit: Use clock_gettime as PMCCNTR requires privileged access.
 * The cycle counter (PMCCNTR) can be used if enabled, but typically
 * requires kernel module support to enable user-space access. */

#include <time.h>

static inline uint64_t read_timer(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static const char* timer_name = "clock_gettime";

static uint64_t get_timer_freq(void) {
    return 1000000000ULL; /* 1 GHz (nanoseconds) */
}

#else
/* Fallback: Use clock_gettime (POSIX) */

#include <time.h>

static inline uint64_t read_timer(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static const char* timer_name = "clock_gettime";

static uint64_t get_timer_freq(void) {
    return 1000000000ULL; /* 1 GHz (nanoseconds) */
}

#endif

/* ============================================================================
 * Public API
 * ============================================================================ */

size_t to_collect_batch(
    to_generator_fn generator,
    to_operation_fn operation,
    void *ctx,
    uint8_t *input_buffer,
    size_t input_size,
    const bool *schedule,
    size_t count,
    uint64_t *out_timings,
    size_t batch_k
) {
    /* Validate required parameters */
    if (generator == NULL || operation == NULL) {
        return 0;
    }
    if (input_buffer == NULL || schedule == NULL || out_timings == NULL) {
        return 0;
    }
    if (count == 0 || input_size == 0 || batch_k == 0) {
        return 0;
    }

    size_t n_baseline = 0;

    for (size_t i = 0; i < count; i++) {
        bool is_baseline = schedule[i];
        if (is_baseline) {
            n_baseline++;
        }

        /* Generate input OUTSIDE timed region */
        generator(ctx, is_baseline, input_buffer, input_size);

        /* === TIMED REGION === */
        uint64_t start = read_timer();

        for (size_t k = 0; k < batch_k; k++) {
            operation(ctx, input_buffer, input_size);
        }

#if defined(__aarch64__) || defined(_M_ARM64)
        /* Data Memory Barrier to ensure all memory operations from the
         * operation complete before reading the end time. This prevents
         * out-of-order execution from affecting timing measurements. */
        __asm__ volatile ("dmb ish" ::: "memory");
#endif

        uint64_t end = read_timer();
        /* === END TIMED REGION === */

        out_timings[i] = end - start;
    }

    return n_baseline;
}

const char* to_get_timer_name(void) {
    return timer_name;
}

uint64_t to_get_timer_frequency(void) {
    return get_timer_freq();
}

uint64_t to_read_timer(void) {
    return read_timer();
}
