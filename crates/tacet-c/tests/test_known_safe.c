/**
 * @file test_known_safe.c
 * @brief CMocka tests for known safe (constant-time) operations.
 *
 * This file contains tests that MUST NOT false positive.
 * These are critical validation tests to ensure the oracle
 * does not incorrectly flag constant-time code as vulnerable.
 *
 * Test pattern (DudeCT two-class):
 * - Baseline: all zeros
 * - Sample: random data
 *
 * Constant-time operations should show no timing difference
 * between processing zeros vs random data.
 */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "../include/tacet.h"
#include "test_helpers.h"

/* Test parameters */
#define CALIBRATION_SAMPLES 5000
#define BATCH_SIZE 1000
#define DATA_SIZE 32
#define MAX_ITERATIONS 200

/* Get current time in seconds */
static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

/**
 * Test that XOR fold (constant-time) does NOT produce false positive.
 *
 * This test MUST NOT incorrectly detect a timing leak.
 * XOR fold processes all bytes regardless of content, so there
 * should be no timing difference between zeros and random data.
 *
 * The xor_fold function is constant-time:
 * - Always iterates through all bytes
 * - XOR operation is constant-time on all modern CPUs
 * - No branches dependent on data values
 *
 * Configuration:
 * - passThreshold: 0.15 (allow P(leak) up to 15% for pass)
 * - failThreshold: 0.99 (very high bar for claiming leak)
 * - timeBudget: 15 seconds
 *
 * Expected outcome: NOT Fail (Pass or Inconclusive acceptable)
 */
static void test_no_false_positive_xor_fold(void **state) {
    (void)state;

    printf("\n=== Known Safe Test: XOR Fold ===\n");

    /* Allocate sample arrays */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *batch_baseline = malloc(BATCH_SIZE * sizeof(uint64_t));
    uint64_t *batch_sample = malloc(BATCH_SIZE * sizeof(uint64_t));

    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);
    assert_non_null(batch_baseline);
    assert_non_null(batch_sample);

    /* Collect calibration samples using XOR fold (constant-time) */
    printf("Collecting %d calibration samples...\n", CALIBRATION_SAMPLES);
    collect_xor_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, DATA_SIZE);

    /* Create config with thresholds appropriate for false positive testing */
    struct ToConfig config = to_config_adjacent_network();
    config.pass_threshold = 0.15;   /* Lenient pass: P(leak) < 15% is OK */
    config.fail_threshold = 0.99;   /* Very strict fail: need P(leak) > 99% to claim leak */
    config.time_budget_secs = 15.0;

    printf("Config: pass_threshold=%.2f, fail_threshold=%.2f, time_budget=%.0fs\n",
           config.pass_threshold, config.fail_threshold, config.time_budget_secs);

    /* Run calibration */
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);

    if (err != Ok || cal == NULL) {
        printf("Calibration failed with error: %d\n", err);
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
        collect_xor_samples(batch_baseline, batch_sample, BATCH_SIZE, DATA_SIZE);

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

    /* CRITICAL ASSERTION: This test MUST NOT false positive */
    if (decision_reached) {
        /* Should NOT be Fail - Pass or Inconclusive is acceptable */
        assert_int_not_equal(final_result.outcome, Fail);
        printf("\nSUCCESS: XOR fold was not falsely flagged as leaky!\n");
    } else {
        /* No decision is acceptable for constant-time code */
        printf("\nSUCCESS: No false positive detected.\n");
    }
}

/**
 * Test that constant-time comparison does NOT produce false positive.
 *
 * The constant_time_compare function:
 * - Always compares all bytes
 * - Uses bitwise OR accumulator (no branches)
 * - Should show no timing difference
 */
static void test_no_false_positive_ct_compare(void **state) {
    (void)state;

    printf("\n=== Known Safe Test: Constant-Time Compare ===\n");

    /* Secret for comparison */
    uint8_t secret[DATA_SIZE];
    uint32_t seed = 0xCAFEBABE;
    fill_random(secret, DATA_SIZE, &seed);

    /* Allocate sample arrays */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *batch_baseline = malloc(BATCH_SIZE * sizeof(uint64_t));
    uint64_t *batch_sample = malloc(BATCH_SIZE * sizeof(uint64_t));

    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);
    assert_non_null(batch_baseline);
    assert_non_null(batch_sample);

    /* Collect calibration samples using constant-time comparison */
    collect_ct_compare_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, secret, DATA_SIZE);

    /* Create config */
    struct ToConfig config = to_config_adjacent_network();
    config.pass_threshold = 0.15;
    config.fail_threshold = 0.99;
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
        collect_ct_compare_samples(batch_baseline, batch_sample, BATCH_SIZE, secret, DATA_SIZE);

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

    /* Print result */
    printf("Outcome: ");
    if (decision_reached) {
        switch (final_result.outcome) {
            case Pass:        printf("PASS"); break;
            case Fail:        printf("FAIL"); break;
            case Inconclusive: printf("INCONCLUSIVE"); break;
            case Unmeasurable: printf("UNMEASURABLE"); break;
        }
        printf(" (P(leak)=%.1f%%)\n", final_result.leak_probability * 100.0);
    } else {
        double final_prob = to_state_leak_probability(st);
        printf("NO DECISION (P(leak)=%.1f%%)\n", final_prob * 100.0);
    }

    /* Clean up */
    to_state_free(st);
    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
    free(batch_baseline);
    free(batch_sample);

    /* CRITICAL ASSERTION: Should NOT be Fail */
    if (decision_reached) {
        assert_int_not_equal(final_result.outcome, Fail);
        printf("SUCCESS: Constant-time compare was not falsely flagged!\n");
    } else {
        printf("SUCCESS: No false positive detected.\n");
    }
}

/**
 * Test that identical operations do NOT produce false positive.
 *
 * When both baseline and sample do the exact same operation
 * on the exact same data, there should be no timing difference.
 */
static void test_no_false_positive_identical(void **state) {
    (void)state;

    printf("\n=== Known Safe Test: Identical Operations ===\n");

    /* Allocate sample arrays */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *batch_baseline = malloc(BATCH_SIZE * sizeof(uint64_t));
    uint64_t *batch_sample = malloc(BATCH_SIZE * sizeof(uint64_t));

    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);
    assert_non_null(batch_baseline);
    assert_non_null(batch_sample);

    /* Collect identical samples (same operation, same data) */
    collect_identical_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES);

    /* Create config */
    struct ToConfig config = to_config_adjacent_network();
    config.pass_threshold = 0.15;
    config.fail_threshold = 0.99;
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
        collect_identical_samples(batch_baseline, batch_sample, BATCH_SIZE);

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

    /* Print result */
    printf("Outcome: ");
    if (decision_reached) {
        switch (final_result.outcome) {
            case Pass:        printf("PASS"); break;
            case Fail:        printf("FAIL"); break;
            case Inconclusive: printf("INCONCLUSIVE"); break;
            case Unmeasurable: printf("UNMEASURABLE"); break;
        }
        printf(" (P(leak)=%.1f%%)\n", final_result.leak_probability * 100.0);
    } else {
        double final_prob = to_state_leak_probability(st);
        printf("NO DECISION (P(leak)=%.1f%%)\n", final_prob * 100.0);
    }

    /* Clean up */
    to_state_free(st);
    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
    free(batch_baseline);
    free(batch_sample);

    /* CRITICAL ASSERTION: Should NOT be Fail */
    if (decision_reached) {
        assert_int_not_equal(final_result.outcome, Fail);
        printf("SUCCESS: Identical operations were not falsely flagged!\n");
    } else {
        printf("SUCCESS: No false positive detected.\n");
    }
}

/* Test group for known safe operations */
const struct CMUnitTest known_safe_tests[] = {
    cmocka_unit_test(test_no_false_positive_xor_fold),
    cmocka_unit_test(test_no_false_positive_ct_compare),
    cmocka_unit_test(test_no_false_positive_identical),
};

int run_known_safe_tests(void) {
    return cmocka_run_group_tests_name("Known Safe Tests", known_safe_tests, NULL, NULL);
}
