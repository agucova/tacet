/**
 * @file common.h
 * @brief Shared utilities for timing-oracle C examples.
 *
 * This header provides cross-platform utilities used by all examples:
 * - Random byte generation
 * - Compiler optimization barriers
 * - Result printing helpers
 */

#ifndef TIMING_ORACLE_EXAMPLES_COMMON_H
#define TIMING_ORACLE_EXAMPLES_COMMON_H

#include "timing_oracle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ============================================================================
 * Random byte generation (cross-platform)
 * ============================================================================ */

/**
 * @brief Fill buffer with cryptographically random bytes.
 *
 * Uses arc4random_buf on macOS/BSD, /dev/urandom on Linux.
 */
static inline void random_bytes(uint8_t *buf, size_t n) {
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    arc4random_buf(buf, n);
#else
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) {
        size_t read = fread(buf, 1, n, f);
        (void)read;  /* Silence unused warning */
        fclose(f);
    } else {
        /* Fallback: not cryptographically secure, but works */
        for (size_t i = 0; i < n; i++) {
            buf[i] = (uint8_t)(rand() & 0xFF);
        }
    }
#endif
}

/* ============================================================================
 * Compiler optimization barriers
 * ============================================================================ */

/**
 * @brief Prevent compiler from optimizing away a value or operation.
 *
 * Use this after operations to ensure the compiler doesn't eliminate
 * "unused" results, which would defeat the purpose of timing tests.
 */
#if defined(__GNUC__) || defined(__clang__)
#define DO_NOT_OPTIMIZE(x) __asm__ volatile("" : : "r,m"(x) : "memory")
#else
/* MSVC and others: use volatile */
static inline void do_not_optimize_impl(void *p) {
    volatile char *vp = (volatile char *)p;
    (void)*vp;
}
#define DO_NOT_OPTIMIZE(x) do_not_optimize_impl((void *)&(x))
#endif

/**
 * @brief Memory barrier to prevent reordering around timed regions.
 */
#if defined(__GNUC__) || defined(__clang__)
#define MEMORY_BARRIER() __asm__ volatile("" ::: "memory")
#else
#define MEMORY_BARRIER() ((void)0)
#endif

/* ============================================================================
 * Result printing utilities
 * ============================================================================ */

/**
 * @brief Print a brief summary of the result (1-2 lines).
 */
static inline void print_result_summary(const to_result_t *r) {
    printf("Outcome: %s | P(leak)=%.1f%% | Samples=%zu | Time=%.1fs\n",
           to_outcome_str(r->outcome),
           r->leak_probability * 100.0,
           r->samples_used,
           r->elapsed_secs);
}

/**
 * @brief Print the full result with all fields.
 */
static inline void print_result_full(const to_result_t *r) {
    printf("=== Timing Oracle Result ===\n");
    printf("Outcome:          %s\n", to_outcome_str(r->outcome));
    printf("Leak probability: %.2f%%\n", r->leak_probability * 100.0);
    printf("Quality:          %s\n", to_quality_str(r->quality));
    printf("Samples/class:    %zu\n", r->samples_used);
    printf("Elapsed time:     %.2f seconds\n", r->elapsed_secs);
    printf("Timer:            %s (%.2f ns resolution)\n",
           r->timer_name, r->timer_resolution_ns);
    printf("Platform:         %s\n", r->platform);
    printf("Adaptive batch:   %s\n", r->adaptive_batching_used ? "yes" : "no");

    /* Effect size details */
    printf("\nEffect size:\n");
    printf("  Shift:    %.2f ns\n", r->effect.shift_ns);
    printf("  Tail:     %.2f ns\n", r->effect.tail_ns);
    printf("  95%% CI:   [%.2f, %.2f] ns\n", r->effect.ci_low_ns, r->effect.ci_high_ns);
    printf("  Pattern:  %s\n", to_effect_pattern_str(r->effect.pattern));

    /* Outcome-specific fields */
    switch (r->outcome) {
        case TO_OUTCOME_FAIL:
            printf("\nExploitability: %s\n", to_exploitability_str(r->exploitability));
            break;
        case TO_OUTCOME_INCONCLUSIVE:
            printf("\nReason: %s\n", to_inconclusive_reason_str(r->inconclusive_reason));
            break;
        case TO_OUTCOME_UNMEASURABLE:
            printf("\nOperation time: %.2f ns (too fast)\n", r->operation_ns);
            break;
        default:
            break;
    }

    /* Recommendation if present */
    if (r->recommendation && strlen(r->recommendation) > 0) {
        printf("\nRecommendation: %s\n", r->recommendation);
    }
}

/**
 * @brief Print result with color-coded outcome (ANSI terminals).
 */
static inline void print_result_colored(const to_result_t *r) {
    const char *color;
    switch (r->outcome) {
        case TO_OUTCOME_PASS:
            color = "\033[32m";  /* Green */
            break;
        case TO_OUTCOME_FAIL:
            color = "\033[31m";  /* Red */
            break;
        case TO_OUTCOME_INCONCLUSIVE:
            color = "\033[33m";  /* Yellow */
            break;
        case TO_OUTCOME_UNMEASURABLE:
            color = "\033[35m";  /* Magenta */
            break;
        default:
            color = "";
    }
    printf("%s%s\033[0m: P(leak)=%.1f%%, %zu samples, %.1fs\n",
           color, to_outcome_str(r->outcome),
           r->leak_probability * 100.0,
           r->samples_used,
           r->elapsed_secs);
}

/* ============================================================================
 * Common generator pattern
 * ============================================================================ */

/**
 * @brief Standard generator: zeros for baseline, random for sample.
 *
 * This is the most common input pattern for timing side-channel testing.
 * Can be used directly as a generator callback when no context is needed.
 */
static inline void standard_generator(void *ctx, bool is_baseline,
                                       uint8_t *output, size_t size) {
    (void)ctx;  /* Unused */
    if (is_baseline) {
        memset(output, 0, size);
    } else {
        random_bytes(output, size);
    }
}

#endif /* TIMING_ORACLE_EXAMPLES_COMMON_H */
