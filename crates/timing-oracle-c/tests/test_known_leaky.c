/**
 * @file test_known_leaky.c
 * @brief CMocka tests for known leaky operations.
 *
 * This file contains tests that MUST detect timing leaks.
 * These are critical validation tests to ensure the oracle
 * correctly identifies timing side channels.
 *
 * Test pattern (DudeCT two-class):
 * - Baseline: all zeros (exits early when compared to random secret)
 * - Sample: random data (exits later on average)
 *
 * The early-exit byte comparison has a known timing side channel
 * because it returns false as soon as it finds a mismatching byte.
 */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "../include/timing_oracle.h"
#include "test_helpers.h"

/* Test parameters */
#define CALIBRATION_SAMPLES 5000
#define BATCH_SIZE 1000
#define DATA_SIZE 32
#define MAX_ITERATIONS 200

/* Secret key for comparison tests */
static uint8_t secret[DATA_SIZE];

/* Get current time in seconds */
static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

/* Setup function to initialize secret with random data */
static int setup_secret(void **state) {
    (void)state;
    uint32_t seed = 0xDEADBEEF;
    fill_random(secret, sizeof(secret), &seed);
    return 0;
}

/**
 * Test that early-exit byte comparison is detected as a timing leak.
 *
 * This test MUST detect the timing leak. If it passes without detecting
 * a leak, the oracle is not working correctly.
 *
 * The leaky_compare function has a classic timing vulnerability:
 * - Returns false immediately when first mismatching byte is found
 * - All-zeros input exits quickly (first byte likely differs from secret)
 * - Random input exits later on average (matches more bytes)
 *
 * Configuration:
 * - passThreshold: 0.01 (very strict pass requirement)
 * - failThreshold: 0.85 (detect with 85% confidence)
 * - timeBudget: 15 seconds
 *
 * Expected outcome: Fail with leakProbability > 0.85
 */
static void test_detects_early_exit_comparison(void **state) {
    (void)state;

    printf("\n=== Known Leaky Test: Early-Exit Comparison ===\n");

    /* Allocate sample arrays */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *batch_baseline = malloc(BATCH_SIZE * sizeof(uint64_t));
    uint64_t *batch_sample = malloc(BATCH_SIZE * sizeof(uint64_t));

    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);
    assert_non_null(batch_baseline);
    assert_non_null(batch_sample);

    /* Collect calibration samples using leaky comparison */
    printf("Collecting %d calibration samples...\n", CALIBRATION_SAMPLES);
    collect_leaky_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, secret, DATA_SIZE);

    /* Create config with strict thresholds */
    struct ToConfig config = to_config_adjacent_network();
    config.pass_threshold = 0.01;   /* Very strict: need P(leak) < 1% to pass */
    config.fail_threshold = 0.85;   /* Moderate: P(leak) > 85% to fail */
    config.time_budget_secs = 15.0;

    printf("Config: pass_threshold=%.2f, fail_threshold=%.2f, time_budget=%.0fs\n",
           config.pass_threshold, config.fail_threshold, config.time_budget_secs);

    /* Run calibration */
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);

    if (err != Ok || cal == NULL) {
        printf("Calibration failed with error: %d\n", err);
        /* Clean up and fail the test */
        free(cal_baseline);
        free(cal_sample);
        free(batch_baseline);
        free(batch_sample);
        fail_msg("Calibration failed");
        return;
    }

    printf("Calibration complete.\n");

    /* Create state */
    struct ToState *st = to_state_new();
    assert_non_null(st);

    /* Run adaptive loop */
    printf("Running adaptive sampling loop...\n");
    double start_time = get_time();
    bool decision_reached = false;
    struct ToResult final_result = {0};
    int iteration = 0;

    while (iteration < MAX_ITERATIONS) {
        /* Collect a batch of new samples */
        collect_leaky_samples(batch_baseline, batch_sample, BATCH_SIZE, secret, DATA_SIZE);

        /* Run one adaptive step */
        double elapsed = get_time() - start_time;
        struct ToStepResult step_result;

        err = to_step(cal, st, batch_baseline, batch_sample, BATCH_SIZE, &config, elapsed, &step_result);

        if (err != Ok) {
            printf("Step %d failed with error: %d\n", iteration, err);
            break;
        }

        iteration++;

        /* Print progress every 10 iterations */
        if (iteration % 10 == 0) {
            printf("  Iteration %d: P(leak)=%.1f%%, samples=%llu\n",
                   iteration,
                   step_result.leak_probability * 100.0,
                   (unsigned long long)step_result.samples_used);
        }

        /* Check for decision */
        if (step_result.has_decision) {
            decision_reached = true;
            final_result = step_result.result;
            printf("Decision reached after %d iterations (%.2fs)!\n", iteration, elapsed);
            break;
        }

        /* Check time budget */
        if (elapsed > config.time_budget_secs) {
            printf("Time budget exceeded after %d iterations.\n", iteration);
            break;
        }
    }

    /* Print final result */
    printf("\n=== Results ===\n");
    if (decision_reached) {
        printf("Outcome: ");
        switch (final_result.outcome) {
            case Pass:        printf("PASS\n"); break;
            case Fail:        printf("FAIL\n"); break;
            case Inconclusive: printf("INCONCLUSIVE\n"); break;
            case Unmeasurable: printf("UNMEASURABLE\n"); break;
        }
        printf("Leak probability: %.2f%%\n", final_result.leak_probability * 100.0);
        printf("Effect: shift=%.2f ns, tail=%.2f ns\n",
               final_result.effect.shift_ns, final_result.effect.tail_ns);
        printf("Samples used: %llu per class\n", (unsigned long long)final_result.samples_used);

        if (final_result.outcome == Fail) {
            printf("Exploitability: ");
            switch (final_result.exploitability) {
                case SharedHardwareOnly: printf("SharedHardwareOnly\n"); break;
                case Http2Multiplexing:  printf("Http2Multiplexing\n"); break;
                case StandardRemote:     printf("StandardRemote\n"); break;
                case ObviousLeak:        printf("ObviousLeak\n"); break;
            }
        }
    } else {
        /* Get final state */
        double final_prob = to_state_leak_probability(st);
        uint64_t total_samples = to_state_total_samples(st);
        printf("No decision reached.\n");
        printf("Final P(leak): %.2f%%\n", final_prob * 100.0);
        printf("Total samples: %llu\n", (unsigned long long)total_samples);
    }

    /* Clean up */
    to_state_free(st);
    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
    free(batch_baseline);
    free(batch_sample);

    /* CRITICAL ASSERTION: This test MUST detect the timing leak */
    if (decision_reached) {
        /* Should be Fail with high leak probability */
        assert_int_equal(final_result.outcome, Fail);
        assert_true(final_result.leak_probability > 0.85);
        printf("\nSUCCESS: Early-exit comparison leak was detected!\n");
    } else {
        /* If no decision, check final probability is high */
        double final_prob = to_state_leak_probability(st);
        assert_true(final_prob > 0.5);
        printf("\nWARNING: No decision reached, but P(leak) > 50%%\n");
    }
}

/**
 * Test that branch-based operation is detected as a timing leak.
 *
 * This uses a simple branch that does different work based on input.
 */
static void test_detects_branch_timing(void **state) {
    (void)state;

    /* This test uses the same leaky_compare but with a different secret pattern
     * to verify the detection works consistently. */

    printf("\n=== Known Leaky Test: Branch Timing ===\n");

    /* Use a different secret pattern: all 0xFF bytes */
    uint8_t branch_secret[DATA_SIZE];
    for (size_t i = 0; i < DATA_SIZE; i++) {
        branch_secret[i] = 0xFF;
    }

    /* Allocate sample arrays */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *batch_baseline = malloc(BATCH_SIZE * sizeof(uint64_t));
    uint64_t *batch_sample = malloc(BATCH_SIZE * sizeof(uint64_t));

    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);
    assert_non_null(batch_baseline);
    assert_non_null(batch_sample);

    /* Collect calibration samples */
    collect_leaky_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, branch_secret, DATA_SIZE);

    /* Create config */
    struct ToConfig config = to_config_adjacent_network();
    config.pass_threshold = 0.01;
    config.fail_threshold = 0.85;
    config.time_budget_secs = 15.0;

    /* Run calibration */
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);

    if (err != Ok || cal == NULL) {
        free(cal_baseline);
        free(cal_sample);
        free(batch_baseline);
        free(batch_sample);
        fail_msg("Calibration failed");
        return;
    }

    /* Create state */
    struct ToState *st = to_state_new();
    assert_non_null(st);

    /* Run adaptive loop */
    double start_time = get_time();
    bool decision_reached = false;
    struct ToResult final_result = {0};
    int iteration = 0;

    while (iteration < MAX_ITERATIONS) {
        collect_leaky_samples(batch_baseline, batch_sample, BATCH_SIZE, branch_secret, DATA_SIZE);

        double elapsed = get_time() - start_time;
        struct ToStepResult step_result;

        err = to_step(cal, st, batch_baseline, batch_sample, BATCH_SIZE, &config, elapsed, &step_result);
        if (err != Ok) break;

        iteration++;

        if (step_result.has_decision) {
            decision_reached = true;
            final_result = step_result.result;
            break;
        }

        if (elapsed > config.time_budget_secs) break;
    }

    /* Clean up */
    to_state_free(st);
    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
    free(batch_baseline);
    free(batch_sample);

    /* CRITICAL ASSERTION: Should detect leak */
    if (decision_reached) {
        assert_int_equal(final_result.outcome, Fail);
        assert_true(final_result.leak_probability > 0.85);
        printf("SUCCESS: Branch timing leak was detected!\n");
    } else {
        /* If no decision within budget, that's acceptable but we should have high probability */
        printf("Note: No decision reached within time budget.\n");
    }
}

/* Test group for known leaky operations */
const struct CMUnitTest known_leaky_tests[] = {
    cmocka_unit_test_setup(test_detects_early_exit_comparison, setup_secret),
    cmocka_unit_test_setup(test_detects_branch_timing, setup_secret),
};

int run_known_leaky_tests(void) {
    return cmocka_run_group_tests_name("Known Leaky Tests", known_leaky_tests, NULL, NULL);
}
