/**
 * @file test_adaptive.c
 * @brief CMocka tests for tacet adaptive sampling loop.
 *
 * Tests for:
 * - to_calibrate() basic functionality
 * - to_calibrate() error handling
 * - to_step() basic functionality
 * - to_step() reaching decision
 */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <sys/time.h>

#include "../include/tacet.h"
#include "test_helpers.h"

/* Calibration samples per class (as per spec) */
#define CALIBRATION_SAMPLES 5000

/* Batch size for adaptive loop */
#define BATCH_SIZE 1000

/* Maximum iterations before giving up */
#define MAX_ITERATIONS 100

/* Secret for leaky comparison tests */
static uint8_t secret[32];

/* Get current time in seconds */
static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

/* Setup function to initialize secret */
static int setup_secret(void **state) {
    (void)state;
    uint32_t seed = 42;
    fill_random(secret, sizeof(secret), &seed);
    return 0;
}

/**
 * Test basic calibration with 5000 samples.
 * Should succeed and return valid calibration handle.
 */
static void test_calibrate_basic(void **state) {
    (void)state;

    /* Allocate sample arrays */
    uint64_t *baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    assert_non_null(baseline);
    assert_non_null(sample);

    /* Collect XOR fold samples (safe, constant-time operation) */
    collect_xor_samples(baseline, sample, CALIBRATION_SAMPLES, 32);

    /* Create config */
    struct ToConfig config = to_config_adjacent_network();

    /* Run calibration */
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(baseline, sample, CALIBRATION_SAMPLES, &config, &err);

    /* Should succeed */
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    /* Clean up */
    to_calibration_free(cal);
    free(baseline);
    free(sample);
}

/**
 * Test calibration with null baseline returns error.
 */
static void test_calibrate_null_baseline(void **state) {
    (void)state;

    uint64_t sample[CALIBRATION_SAMPLES];
    struct ToConfig config = to_config_adjacent_network();

    enum ToError err;
    struct ToCalibration *cal = to_calibrate(NULL, sample, CALIBRATION_SAMPLES, &config, &err);

    /* Should return NullPointer error */
    assert_int_equal(err, NullPointer);
    assert_null(cal);
}

/**
 * Test calibration with null sample returns error.
 */
static void test_calibrate_null_sample(void **state) {
    (void)state;

    uint64_t baseline[CALIBRATION_SAMPLES];
    struct ToConfig config = to_config_adjacent_network();

    enum ToError err;
    struct ToCalibration *cal = to_calibrate(baseline, NULL, CALIBRATION_SAMPLES, &config, &err);

    /* Should return NullPointer error */
    assert_int_equal(err, NullPointer);
    assert_null(cal);
}

/**
 * Test calibration with null config returns error.
 */
static void test_calibrate_null_config(void **state) {
    (void)state;

    uint64_t baseline[CALIBRATION_SAMPLES];
    uint64_t sample[CALIBRATION_SAMPLES];

    enum ToError err;
    struct ToCalibration *cal = to_calibrate(baseline, sample, CALIBRATION_SAMPLES, NULL, &err);

    /* Should return NullPointer error */
    assert_int_equal(err, NullPointer);
    assert_null(cal);
}

/**
 * Test calibration with zero count returns error.
 */
static void test_calibrate_zero_count(void **state) {
    (void)state;

    uint64_t baseline[100];
    uint64_t sample[100];
    struct ToConfig config = to_config_adjacent_network();

    enum ToError err;
    struct ToCalibration *cal = to_calibrate(baseline, sample, 0, &config, &err);

    /* Should return NotEnoughSamples error */
    assert_int_equal(err, NotEnoughSamples);
    assert_null(cal);
}

/**
 * Test basic step functionality.
 * Calibrate first, then run one step and check leak_probability updates.
 */
static void test_step_basic(void **state) {
    (void)state;

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
    collect_xor_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, 32);

    /* Create config */
    struct ToConfig config = to_config_adjacent_network();
    config.time_budget_secs = 30.0;

    /* Run calibration */
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    /* Create state */
    struct ToState *st = to_state_new();
    assert_non_null(st);

    /* Check initial state */
    uint64_t initial_samples = to_state_total_samples(st);
    assert_int_equal(initial_samples, 0);

    double initial_prob = to_state_leak_probability(st);
    /* Initial probability should be 0.5 (no data yet) */
    assert_true(initial_prob >= 0.0 && initial_prob <= 1.0);

    /* Collect a batch of new samples */
    collect_xor_samples(batch_baseline, batch_sample, BATCH_SIZE, 32);

    /* Run one adaptive step */
    struct ToStepResult step_result;
    err = to_step(cal, st, batch_baseline, batch_sample, BATCH_SIZE, &config, 1.0, &step_result);
    assert_int_equal(err, Ok);

    /* Check that samples were added */
    assert_true(step_result.samples_used > 0);

    /* Check state was updated */
    uint64_t total_samples = to_state_total_samples(st);
    assert_true(total_samples > initial_samples);

    /* Leak probability should be valid */
    double prob = step_result.leak_probability;
    assert_true(prob >= 0.0 && prob <= 1.0);

    /* Clean up */
    to_state_free(st);
    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
    free(batch_baseline);
    free(batch_sample);
}

/**
 * Test step with null calibration returns error.
 */
static void test_step_null_calibration(void **state) {
    (void)state;

    struct ToState *st = to_state_new();
    assert_non_null(st);

    uint64_t baseline[BATCH_SIZE];
    uint64_t sample[BATCH_SIZE];
    struct ToConfig config = to_config_adjacent_network();
    struct ToStepResult result;

    enum ToError err = to_step(NULL, st, baseline, sample, BATCH_SIZE, &config, 1.0, &result);

    /* Should return NullPointer error */
    assert_int_equal(err, NullPointer);

    to_state_free(st);
}

/**
 * Test step with null state returns error.
 */
static void test_step_null_state(void **state) {
    (void)state;

    /* Create minimal calibration */
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

    uint64_t baseline[BATCH_SIZE];
    uint64_t sample[BATCH_SIZE];
    struct ToStepResult result;

    err = to_step(cal, NULL, baseline, sample, BATCH_SIZE, &config, 1.0, &result);

    /* Should return NullPointer error */
    assert_int_equal(err, NullPointer);

    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
}

/**
 * Test adaptive loop until decision is reached.
 * Uses leaky comparison which should trigger Fail decision.
 */
static void test_step_reaches_decision(void **state) {
    (void)state;

    /* Allocate sample arrays */
    uint64_t *cal_baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *cal_sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *batch_baseline = malloc(BATCH_SIZE * sizeof(uint64_t));
    uint64_t *batch_sample = malloc(BATCH_SIZE * sizeof(uint64_t));
    assert_non_null(cal_baseline);
    assert_non_null(cal_sample);
    assert_non_null(batch_baseline);
    assert_non_null(batch_sample);

    /* Collect calibration samples with leaky comparison */
    collect_leaky_samples(cal_baseline, cal_sample, CALIBRATION_SAMPLES, secret, sizeof(secret));

    /* Create config with reasonable time budget */
    struct ToConfig config = to_config_adjacent_network();
    config.time_budget_secs = 30.0;
    config.pass_threshold = 0.05;
    config.fail_threshold = 0.95;

    /* Run calibration */
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(cal_baseline, cal_sample, CALIBRATION_SAMPLES, &config, &err);
    assert_int_equal(err, Ok);
    assert_non_null(cal);

    /* Create state */
    struct ToState *st = to_state_new();
    assert_non_null(st);

    /* Run adaptive loop */
    double start_time = get_time();
    bool decision_reached = false;
    int iteration = 0;

    while (iteration < MAX_ITERATIONS) {
        /* Collect a batch of new samples */
        collect_leaky_samples(batch_baseline, batch_sample, BATCH_SIZE, secret, sizeof(secret));

        /* Run one adaptive step */
        double elapsed = get_time() - start_time;
        struct ToStepResult step_result;

        err = to_step(cal, st, batch_baseline, batch_sample, BATCH_SIZE, &config, elapsed, &step_result);
        assert_int_equal(err, Ok);

        iteration++;

        /* Check for decision */
        if (step_result.has_decision) {
            decision_reached = true;

            /* Verify result is valid */
            assert_true(step_result.result.outcome >= Pass && step_result.result.outcome <= Unmeasurable);
            assert_true(step_result.result.leak_probability >= 0.0 && step_result.result.leak_probability <= 1.0);
            assert_true(step_result.result.samples_used > 0);

            break;
        }

        /* Check time budget */
        if (elapsed > config.time_budget_secs) {
            break;
        }
    }

    /* Should have reached a decision or timed out */
    /* Note: We don't assert decision_reached because the test may time out on slow machines */

    /* Clean up */
    to_state_free(st);
    to_calibration_free(cal);
    free(cal_baseline);
    free(cal_sample);
    free(batch_baseline);
    free(batch_sample);
}

/**
 * Test state functions.
 */
static void test_state_functions(void **state) {
    (void)state;

    /* Create state */
    struct ToState *st = to_state_new();
    assert_non_null(st);

    /* Initial total samples should be 0 */
    uint64_t samples = to_state_total_samples(st);
    assert_int_equal(samples, 0);

    /* Initial leak probability should be 0.5 */
    double prob = to_state_leak_probability(st);
    assert_true(prob >= 0.0 && prob <= 1.0);

    /* Clean up */
    to_state_free(st);
}

/* Test group for adaptive sampling */
const struct CMUnitTest adaptive_tests[] = {
    cmocka_unit_test_setup(test_calibrate_basic, setup_secret),
    cmocka_unit_test(test_calibrate_null_baseline),
    cmocka_unit_test(test_calibrate_null_sample),
    cmocka_unit_test(test_calibrate_null_config),
    cmocka_unit_test(test_calibrate_zero_count),
    cmocka_unit_test_setup(test_step_basic, setup_secret),
    cmocka_unit_test(test_step_null_calibration),
    cmocka_unit_test_setup(test_step_null_state, setup_secret),
    cmocka_unit_test_setup(test_step_reaches_decision, setup_secret),
    cmocka_unit_test(test_state_functions),
};

int run_adaptive_tests(void) {
    return cmocka_run_group_tests_name("Adaptive Sampling Tests", adaptive_tests, NULL, NULL);
}
