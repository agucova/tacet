/**
 * @file test_helpers.h
 * @brief Test helper functions for tacet C tests.
 */

#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Platform-specific high-resolution timer */
#if defined(__x86_64__) || defined(_M_X64)
static inline uint64_t read_timer(void) {
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
#elif defined(__aarch64__)
static inline uint64_t read_timer(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}
#else
#error "Unsupported architecture"
#endif

/* Compiler barrier to prevent reordering */
#define DO_NOT_OPTIMIZE(x) __asm__ volatile("" : : "r,m"(x) : "memory")

/**
 * @brief Early-exit byte comparison (KNOWN LEAKY).
 * Returns early on first mismatch.
 */
static inline bool leaky_compare(const uint8_t *a, const uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Constant-time XOR fold (KNOWN SAFE).
 * Processes all bytes regardless of content.
 */
static inline uint8_t xor_fold(const uint8_t *data, size_t n) {
    uint8_t result = 0;
    for (size_t i = 0; i < n; i++) {
        result ^= data[i];
    }
    DO_NOT_OPTIMIZE(result);
    return result;
}

/**
 * @brief Constant-time comparison (KNOWN SAFE).
 * Compares all bytes, no early exit.
 */
static inline bool constant_time_compare(const uint8_t *a, const uint8_t *b, size_t n) {
    volatile uint8_t diff = 0;
    for (size_t i = 0; i < n; i++) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

/**
 * @brief Fill buffer with all zeros.
 */
static inline void fill_zeros(uint8_t *buf, size_t size) {
    for (size_t i = 0; i < size; i++) {
        buf[i] = 0;
    }
}

/**
 * @brief Fill buffer with pseudo-random bytes.
 * Uses a simple LCG for reproducibility.
 */
static inline void fill_random(uint8_t *buf, size_t size, uint32_t *seed) {
    for (size_t i = 0; i < size; i++) {
        *seed = *seed * 1103515245 + 12345;
        buf[i] = (uint8_t)((*seed >> 16) & 0xFF);
    }
}

/**
 * @brief Collect timing samples for leaky comparison.
 * Baseline: zeros (matches secret on first byte for random secret)
 * Sample: random (exits later on average)
 */
void collect_leaky_samples(
    uint64_t *baseline_out,
    uint64_t *sample_out,
    size_t count,
    const uint8_t *secret,
    size_t secret_len
);

/**
 * @brief Collect timing samples for XOR fold (safe operation).
 * Baseline: zeros
 * Sample: random
 */
void collect_xor_samples(
    uint64_t *baseline_out,
    uint64_t *sample_out,
    size_t count,
    size_t data_len
);

/**
 * @brief Collect timing samples for constant-time comparison.
 * Baseline: zeros
 * Sample: random
 */
void collect_ct_compare_samples(
    uint64_t *baseline_out,
    uint64_t *sample_out,
    size_t count,
    const uint8_t *secret,
    size_t secret_len
);

/**
 * @brief Collect samples with artificial shift (for testing leak detection).
 * Adds a fixed delay to sample timings.
 */
void collect_shifted_samples(
    uint64_t *baseline_out,
    uint64_t *sample_out,
    size_t count,
    uint64_t shift_ticks
);

/**
 * @brief Collect identical samples (for testing no false positives).
 * Both classes do the same operation on same data.
 */
void collect_identical_samples(
    uint64_t *baseline_out,
    uint64_t *sample_out,
    size_t count
);

#endif /* TEST_HELPERS_H */
