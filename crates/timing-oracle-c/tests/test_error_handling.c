/**
 * @file test_error_handling.c
 * @brief CMocka tests for timing-oracle error handling.
 *
 * Tests for:
 * - Null pointer safety (free functions)
 * - Error code propagation
 * - Edge cases
 */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>

#include "../include/timing_oracle.h"
#include "test_helpers.h"

/* Calibration samples */
#define CALIBRATION_SAMPLES 5000

/**
 * Test that to_state_free with NULL does not crash.
 */
static void test_state_free_null_safe(void **state) {
    (void)state;

    /* Should not crash when passed NULL */
    to_state_free(NULL);

    /* If we get here without crashing, the test passes */
    assert_true(1);
}

/**
 * Test that to_calibration_free with NULL does not crash.
 */
static void test_calibration_free_null_safe(void **state) {
    (void)state;

    /* Should not crash when passed NULL */
    to_calibration_free(NULL);

    /* If we get here without crashing, the test passes */
    assert_true(1);
}

/**
 * Test that to_step with null calibration returns correct error.
 */
static void test_step_null_calibration(void **state) {
    (void)state;

    struct ToState *st = to_state_new();
    assert_non_null(st);

    uint64_t baseline[100];
    uint64_t sample[100];
    struct ToConfig config = to_config_adjacent_network();
    struct ToStepResult result;

    enum ToError err = to_step(NULL, st, baseline, sample, 100, &config, 1.0, &result);
    assert_int_equal(err, NullPointer);

    to_state_free(st);
}

/**
 * Test that to_step with null state returns correct error.
 */
static void test_step_null_state(void **state) {
    (void)state;

    /* Create valid calibration first */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);

    collect_xor_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, 32);

    struct ToConfig config = to_config_adjacent_network();
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    uint64_t baseline[100];
    uint64_t sample[100];
    struct ToStepResult result;

    err = to_step(cal, NULL, baseline, sample, 100, &config, 1.0, &result);
    assert_int_equal(err, NullPointer);

    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
}

/**
 * Test that to_step with null baseline returns correct error.
 */
static void test_step_null_baseline(void **state) {
    (void)state;

    /* Create valid calibration first */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);

    collect_xor_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, 32);

    struct ToConfig config = to_config_adjacent_network();
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    struct ToState *st = to_state_new();
    assert_non_null(st);

    uint64_t sample[100];
    struct ToStepResult result;

    err = to_step(cal, st, NULL, sample, 100, &config, 1.0, &result);
    assert_int_equal(err, NullPointer);

    to_state_free(st);
    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
}

/**
 * Test that to_step with null sample returns correct error.
 */
static void test_step_null_sample(void **state) {
    (void)state;

    /* Create valid calibration first */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);

    collect_xor_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, 32);

    struct ToConfig config = to_config_adjacent_network();
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    struct ToState *st = to_state_new();
    assert_non_null(st);

    uint64_t baseline[100];
    struct ToStepResult result;

    err = to_step(cal, st, baseline, NULL, 100, &config, 1.0, &result);
    assert_int_equal(err, NullPointer);

    to_state_free(st);
    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
}

/**
 * Test that to_step with null config returns correct error.
 */
static void test_step_null_config(void **state) {
    (void)state;

    /* Create valid calibration first */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);

    collect_xor_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, 32);

    struct ToConfig config = to_config_adjacent_network();
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    struct ToState *st = to_state_new();
    assert_non_null(st);

    uint64_t baseline[100];
    uint64_t sample[100];
    struct ToStepResult result;

    err = to_step(cal, st, baseline, sample, 100, NULL, 1.0, &result);
    assert_int_equal(err, NullPointer);

    to_state_free(st);
    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
}

/**
 * Test that to_step with null result returns correct error.
 */
static void test_step_null_result(void **state) {
    (void)state;

    /* Create valid calibration first */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);

    collect_xor_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, 32);

    struct ToConfig config = to_config_adjacent_network();
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    struct ToState *st = to_state_new();
    assert_non_null(st);

    uint64_t baseline[100];
    uint64_t sample[100];

    err = to_step(cal, st, baseline, sample, 100, &config, 1.0, NULL);
    assert_int_equal(err, NullPointer);

    to_state_free(st);
    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
}

/**
 * Test that to_step with zero count returns correct error.
 */
static void test_step_zero_count(void **state) {
    (void)state;

    /* Create valid calibration first */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);

    collect_xor_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, 32);

    struct ToConfig config = to_config_adjacent_network();
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    struct ToState *st = to_state_new();
    assert_non_null(st);

    uint64_t baseline[100];
    uint64_t sample[100];
    struct ToStepResult result;

    err = to_step(cal, st, baseline, sample, 0, &config, 1.0, &result);
    /* Zero-count step is a no-op, returns Ok (not an error) */
    assert_int_equal(err, Ok);

    to_state_free(st);
    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
}

/**
 * Test that to_calibrate with insufficient samples returns correct error.
 */
static void test_calibrate_insufficient_samples(void **state) {
    (void)state;

    uint64_t baseline[10] = {100, 101, 99, 100, 100, 101, 99, 100, 100, 101};
    uint64_t sample[10] = {100, 101, 99, 100, 100, 101, 99, 100, 100, 101};
    struct ToConfig config = to_config_adjacent_network();

    enum ToError err;
    struct ToCalibration *cal = to_calibrate(baseline, sample, 10, &config, &err);

    /* Should return NotEnoughSamples error for very small counts */
    assert_int_equal(err, NotEnoughSamples);
    assert_null(cal);
}

/**
 * Test that to_analyze with all null pointers returns error.
 */
static void test_analyze_all_null(void **state) {
    (void)state;

    enum ToError err = to_analyze(NULL, NULL, 100, NULL, NULL);
    assert_int_equal(err, NullPointer);
}

/**
 * Test error codes are distinct values.
 */
static void test_error_codes_distinct(void **state) {
    (void)state;

    /* Verify all error codes have distinct values */
    assert_int_not_equal(Ok, NullPointer);
    assert_int_not_equal(Ok, InvalidConfig);
    assert_int_not_equal(Ok, CalibrationFailed);
    assert_int_not_equal(Ok, AnalysisFailed);
    assert_int_not_equal(Ok, NotEnoughSamples);

    assert_int_not_equal(NullPointer, InvalidConfig);
    assert_int_not_equal(NullPointer, CalibrationFailed);
    assert_int_not_equal(NullPointer, AnalysisFailed);
    assert_int_not_equal(NullPointer, NotEnoughSamples);

    assert_int_not_equal(InvalidConfig, CalibrationFailed);
    assert_int_not_equal(InvalidConfig, AnalysisFailed);
    assert_int_not_equal(InvalidConfig, NotEnoughSamples);

    assert_int_not_equal(CalibrationFailed, AnalysisFailed);
    assert_int_not_equal(CalibrationFailed, NotEnoughSamples);

    assert_int_not_equal(AnalysisFailed, NotEnoughSamples);
}

/**
 * Test outcome codes are distinct values.
 */
static void test_outcome_codes_distinct(void **state) {
    (void)state;

    /* Verify all outcome codes have distinct values */
    assert_int_not_equal(Pass, Fail);
    assert_int_not_equal(Pass, Inconclusive);
    assert_int_not_equal(Pass, Unmeasurable);

    assert_int_not_equal(Fail, Inconclusive);
    assert_int_not_equal(Fail, Unmeasurable);

    assert_int_not_equal(Inconclusive, Unmeasurable);
}

/* Test group for error handling */
const struct CMUnitTest error_handling_tests[] = {
    cmocka_unit_test(test_state_free_null_safe),
    cmocka_unit_test(test_calibration_free_null_safe),
    cmocka_unit_test(test_step_null_calibration),
    cmocka_unit_test(test_step_null_state),
    cmocka_unit_test(test_step_null_baseline),
    cmocka_unit_test(test_step_null_sample),
    cmocka_unit_test(test_step_null_config),
    cmocka_unit_test(test_step_null_result),
    cmocka_unit_test(test_step_zero_count),
    cmocka_unit_test(test_calibrate_insufficient_samples),
    cmocka_unit_test(test_analyze_all_null),
    cmocka_unit_test(test_error_codes_distinct),
    cmocka_unit_test(test_outcome_codes_distinct),
};

int run_error_handling_tests(void) {
    return cmocka_run_group_tests_name("Error Handling Tests", error_handling_tests, NULL, NULL);
}
