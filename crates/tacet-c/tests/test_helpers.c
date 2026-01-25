/**
 * @file test_helpers.c
 * @brief Implementation of test helper functions for tacet C tests.
 */

#include "test_helpers.h"
#include <stdlib.h>
#include <string.h>

/* Static seed for random number generation */
static uint32_t g_seed = 12345;

/**
 * @brief Collect timing samples for leaky comparison.
 * Baseline: zeros (exits early when compared to random secret)
 * Sample: random (exits later on average)
 */
void collect_leaky_samples(
    uint64_t *baseline_out,
    uint64_t *sample_out,
    size_t count,
    const uint8_t *secret,
    size_t secret_len
) {
    uint8_t *input = (uint8_t *)malloc(secret_len);
    if (!input) return;

    for (size_t i = 0; i < count; i++) {
        /* Baseline measurement: all zeros */
        fill_zeros(input, secret_len);
        uint64_t start = read_timer();
        bool result = leaky_compare(input, secret, secret_len);
        uint64_t end = read_timer();
        DO_NOT_OPTIMIZE(result);
        baseline_out[i] = end - start;

        /* Sample measurement: random data */
        fill_random(input, secret_len, &g_seed);
        start = read_timer();
        result = leaky_compare(input, secret, secret_len);
        end = read_timer();
        DO_NOT_OPTIMIZE(result);
        sample_out[i] = end - start;
    }

    free(input);
}

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
) {
    uint8_t *input = (uint8_t *)malloc(data_len);
    if (!input) return;

    for (size_t i = 0; i < count; i++) {
        /* Baseline measurement: all zeros */
        fill_zeros(input, data_len);
        uint64_t start = read_timer();
        uint8_t result = xor_fold(input, data_len);
        uint64_t end = read_timer();
        DO_NOT_OPTIMIZE(result);
        baseline_out[i] = end - start;

        /* Sample measurement: random data */
        fill_random(input, data_len, &g_seed);
        start = read_timer();
        result = xor_fold(input, data_len);
        end = read_timer();
        DO_NOT_OPTIMIZE(result);
        sample_out[i] = end - start;
    }

    free(input);
}

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
) {
    uint8_t *input = (uint8_t *)malloc(secret_len);
    if (!input) return;

    for (size_t i = 0; i < count; i++) {
        /* Baseline measurement: all zeros */
        fill_zeros(input, secret_len);
        uint64_t start = read_timer();
        bool result = constant_time_compare(input, secret, secret_len);
        uint64_t end = read_timer();
        DO_NOT_OPTIMIZE(result);
        baseline_out[i] = end - start;

        /* Sample measurement: random data */
        fill_random(input, secret_len, &g_seed);
        start = read_timer();
        result = constant_time_compare(input, secret, secret_len);
        end = read_timer();
        DO_NOT_OPTIMIZE(result);
        sample_out[i] = end - start;
    }

    free(input);
}

/**
 * @brief Collect samples with artificial shift (for testing leak detection).
 * Adds a fixed delay to sample timings.
 */
void collect_shifted_samples(
    uint64_t *baseline_out,
    uint64_t *sample_out,
    size_t count,
    uint64_t shift_ticks
) {
    uint8_t data[32];
    fill_zeros(data, 32);

    for (size_t i = 0; i < count; i++) {
        /* Baseline measurement */
        uint64_t start = read_timer();
        uint8_t result = xor_fold(data, 32);
        uint64_t end = read_timer();
        DO_NOT_OPTIMIZE(result);
        baseline_out[i] = end - start;

        /* Sample measurement with artificial shift */
        fill_random(data, 32, &g_seed);
        start = read_timer();
        result = xor_fold(data, 32);
        end = read_timer();
        DO_NOT_OPTIMIZE(result);
        /* Add artificial delay to simulate timing leak */
        sample_out[i] = (end - start) + shift_ticks;
    }
}

/**
 * @brief Collect identical samples (for testing no false positives).
 * Both classes do the same operation on same data.
 */
void collect_identical_samples(
    uint64_t *baseline_out,
    uint64_t *sample_out,
    size_t count
) {
    uint8_t data[32];
    fill_zeros(data, 32);

    for (size_t i = 0; i < count; i++) {
        /* Baseline measurement */
        uint64_t start = read_timer();
        uint8_t result = xor_fold(data, 32);
        uint64_t end = read_timer();
        DO_NOT_OPTIMIZE(result);
        baseline_out[i] = end - start;

        /* Sample measurement: identical operation on identical data */
        start = read_timer();
        result = xor_fold(data, 32);
        end = read_timer();
        DO_NOT_OPTIMIZE(result);
        sample_out[i] = end - start;
    }
}
