/**
 * @file test_memory_safety.c
 * @brief CMocka tests for tacet memory safety.
 *
 * Tests for:
 * - State lifecycle (new/use/free)
 * - Calibration lifecycle (calibrate/free)
 * - Double-free safety
 * - Resource cleanup
 */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>

#include "../include/tacet.h"
#include "test_helpers.h"

/* Calibration samples */
#define CALIBRATION_SAMPLES 5000

/* Batch size */
#define BATCH_SIZE 1000

/**
 * Test state lifecycle: new, use, free.
 */
static void test_state_lifecycle(void **state) {
    (void)state;

    /* Create state */
    struct ToState *st = to_state_new();
    assert_non_null(st);

    /* Use state: query functions should work */
    uint64_t samples = to_state_total_samples(st);
    assert_int_equal(samples, 0);

    double prob = to_state_leak_probability(st);
    assert_true(prob >= 0.0 && prob <= 1.0);

    /* Free state */
    to_state_free(st);

    /* If we get here without crash/error, test passes */
    assert_true(1);
}

/**
 * Test multiple states can be created and freed independently.
 */
static void test_multiple_states(void **state) {
    (void)state;

    /* Create multiple states */
    struct ToState *st1 = to_state_new();
    struct ToState *st2 = to_state_new();
    struct ToState *st3 = to_state_new();

    assert_non_null(st1);
    assert_non_null(st2);
    assert_non_null(st3);

    /* States should be distinct */
    assert_ptr_not_equal(st1, st2);
    assert_ptr_not_equal(st2, st3);
    assert_ptr_not_equal(st1, st3);

    /* Free in different order than creation */
    to_state_free(st2);
    to_state_free(st1);
    to_state_free(st3);

    assert_true(1);
}

/**
 * Test calibration lifecycle: calibrate, use, free.
 */
static void test_calibration_lifecycle(void **state) {
    (void)state;

    /* Allocate sample arrays */
    uint64_t *baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    assert_non_null(baseline);
    assert_non_null(sample);

    /* Collect samples */
    collect_xor_samples(baseline, sample, CALIBRATION_SAMPLES, 32);

    /* Create config */
    struct ToConfig config = to_config_adjacent_network();

    /* Create calibration */
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(baseline, sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    /* Use calibration: create state and run one step */
    struct ToState *st = to_state_new();
    assert_non_null(st);

    uint64_t *batch_baseline = malloc(BATCH_SIZE * sizeof(uint64_t));
    uint64_t *batch_sample = malloc(BATCH_SIZE * sizeof(uint64_t));
    assert_non_null(batch_baseline);
    assert_non_null(batch_sample);

    collect_xor_samples(batch_baseline, batch_sample, BATCH_SIZE, 32);

    struct ToStepResult result;
    err = to_step(cal, st, batch_baseline, batch_sample, BATCH_SIZE, &config, 1.0, &result);
    assert_int_equal(err, Ok);

    /* Free resources */
    to_state_free(st);
    to_calibration_free(cal);
    free(baseline);
    free(sample);
    free(batch_baseline);
    free(batch_sample);

    assert_true(1);
}

/**
 * Test that multiple calibrations can coexist.
 */
static void test_multiple_calibrations(void **state) {
    (void)state;

    /* Allocate sample arrays */
    uint64_t *baseline1 = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *sample1 = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *baseline2 = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *sample2 = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    assert_non_null(baseline1);
    assert_non_null(sample1);
    assert_non_null(baseline2);
    assert_non_null(sample2);

    /* Collect samples */
    collect_xor_samples(baseline1, sample1, CALIBRATION_SAMPLES, 32);
    collect_xor_samples(baseline2, sample2, CALIBRATION_SAMPLES, 32);

    /* Create configs */
    struct ToConfig config1 = to_config_adjacent_network();
    struct ToConfig config2 = to_config_shared_hardware();

    /* Create multiple calibrations */
    enum ToError err1, err2;
    struct ToCalibration *cal1 = to_calibrate(baseline1, sample1, CALIBRATION_SAMPLES, &config1, &err1);
    struct ToCalibration *cal2 = to_calibrate(baseline2, sample2, CALIBRATION_SAMPLES, &config2, &err2);

    assert_int_equal(err1, Ok);
    assert_int_equal(err2, Ok);
    assert_non_null(cal1);
    assert_non_null(cal2);

    /* Calibrations should be distinct */
    assert_ptr_not_equal(cal1, cal2);

    /* Free in reverse order */
    to_calibration_free(cal2);
    to_calibration_free(cal1);

    free(baseline1);
    free(sample1);
    free(baseline2);
    free(sample2);

    assert_true(1);
}

/**
 * Test that double-free of state is safe.
 * Note: This relies on the API setting the pointer to NULL or handling gracefully.
 * In practice, the user should set their pointer to NULL after free.
 */
static void test_double_free_state_safe(void **state) {
    (void)state;

    /* Create state */
    struct ToState *st = to_state_new();
    assert_non_null(st);

    /* Free once */
    to_state_free(st);

    /* Free NULL should be safe */
    to_state_free(NULL);

    /* Note: We cannot safely test actual double-free of the same pointer
     * as that's undefined behavior. The API documents that NULL is safe. */
    assert_true(1);
}

/**
 * Test that double-free of calibration is safe.
 * Note: This relies on the API setting the pointer to NULL or handling gracefully.
 * In practice, the user should set their pointer to NULL after free.
 */
static void test_double_free_calibration_safe(void **state) {
    (void)state;

    /* Allocate sample arrays */
    uint64_t *baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    assert_non_null(baseline);
    assert_non_null(sample);

    /* Collect samples */
    collect_xor_samples(baseline, sample, CALIBRATION_SAMPLES, 32);

    /* Create config and calibration */
    struct ToConfig config = to_config_adjacent_network();
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(baseline, sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    /* Free once */
    to_calibration_free(cal);

    /* Free NULL should be safe */
    to_calibration_free(NULL);

    free(baseline);
    free(sample);

    assert_true(1);
}

/**
 * Test full workflow lifecycle.
 */
static void test_full_workflow_lifecycle(void **state) {
    (void)state;

    /* Allocate all arrays */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *batch_baseline = malloc(BATCH_SIZE * sizeof(uint64_t));
    uint64_t *batch_sample = malloc(BATCH_SIZE * sizeof(uint64_t));

    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);
    assert_non_null(batch_baseline);
    assert_non_null(batch_sample);

    /* Collect calibration samples */
    collect_xor_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, 32);

    /* Create config */
    struct ToConfig config = to_config_adjacent_network();
    config.time_budget_secs = 10.0;

    /* Create calibration */
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    /* Create state */
    struct ToState *st = to_state_new();
    assert_non_null(st);

    /* Run a few steps */
    for (int i = 0; i < 5; i++) {
        collect_xor_samples(batch_baseline, batch_sample, BATCH_SIZE, 32);

        struct ToStepResult result;
        err = to_step(cal, st, batch_baseline, batch_sample, BATCH_SIZE, &config, (double)i, &result);
        assert_int_equal(err, Ok);

        /* Verify result is valid */
        assert_true(result.leak_probability >= 0.0 && result.leak_probability <= 1.0);
        assert_true(result.samples_used > 0);

        if (result.has_decision) {
            break;
        }
    }

    /* Clean up in correct order */
    to_state_free(st);
    to_calibration_free(cal);

    free(cal_baseline);
    free(cal_sample);
    free(batch_baseline);
    free(batch_sample);

    assert_true(1);
}

/**
 * Test state can be reused across multiple calibrations.
 * Note: This tests whether creating a new state for each calibration works.
 */
static void test_state_per_calibration(void **state) {
    (void)state;

    /* Allocate sample arrays */
    uint64_t *baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *batch_baseline = malloc(BATCH_SIZE * sizeof(uint64_t));
    uint64_t *batch_sample = malloc(BATCH_SIZE * sizeof(uint64_t));

    assert_non_null(baseline);
    assert_non_null(sample);
    assert_non_null(batch_baseline);
    assert_non_null(batch_sample);

    struct ToConfig config = to_config_adjacent_network();
    enum ToError err;

    /* First calibration */
    collect_xor_samples(baseline, sample, CALIBRATION_SAMPLES, 32);
    struct ToCalibration *cal1 = to_calibrate(baseline, sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal1);

    struct ToState *st1 = to_state_new();
    assert_non_null(st1);

    collect_xor_samples(batch_baseline, batch_sample, BATCH_SIZE, 32);
    struct ToStepResult result1;
    err = to_step(cal1, st1, batch_baseline, batch_sample, BATCH_SIZE, &config, 1.0, &result1);
    assert_int_equal(err, Ok);

    /* Clean up first */
    to_state_free(st1);
    to_calibration_free(cal1);

    /* Second calibration with new state */
    collect_xor_samples(baseline, sample, CALIBRATION_SAMPLES, 32);
    struct ToCalibration *cal2 = to_calibrate(baseline, sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal2);

    struct ToState *st2 = to_state_new();
    assert_non_null(st2);

    collect_xor_samples(batch_baseline, batch_sample, BATCH_SIZE, 32);
    struct ToStepResult result2;
    err = to_step(cal2, st2, batch_baseline, batch_sample, BATCH_SIZE, &config, 1.0, &result2);
    assert_int_equal(err, Ok);

    /* Clean up second */
    to_state_free(st2);
    to_calibration_free(cal2);

    free(baseline);
    free(sample);
    free(batch_baseline);
    free(batch_sample);

    assert_true(1);
}

/* Test group for memory safety */
const struct CMUnitTest memory_safety_tests[] = {
    cmocka_unit_test(test_state_lifecycle),
    cmocka_unit_test(test_multiple_states),
    cmocka_unit_test(test_calibration_lifecycle),
    cmocka_unit_test(test_multiple_calibrations),
    cmocka_unit_test(test_double_free_state_safe),
    cmocka_unit_test(test_double_free_calibration_safe),
    cmocka_unit_test(test_full_workflow_lifecycle),
    cmocka_unit_test(test_state_per_calibration),
};

int run_memory_safety_tests(void) {
    return cmocka_run_group_tests_name("Memory Safety Tests", memory_safety_tests, NULL, NULL);
}
