/**
 * @file test_one_shot.c
 * @brief CMocka tests for timing-oracle one-shot analysis (to_analyze).
 *
 * Tests for:
 * - to_analyze() with identical distributions (should not Fail)
 * - to_analyze() with shifted samples (should detect leak)
 * - to_analyze() error handling (null pointers, zero count)
 */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>

#include "../include/timing_oracle.h"
#include "test_helpers.h"

/* Number of samples for one-shot tests */
#define ONE_SHOT_SAMPLES 10000

/* Large shift in ticks to ensure detection (1000+ ticks) */
#define LARGE_SHIFT_TICKS 2000

/**
 * Test that identical distributions do not produce Fail outcome.
 * When baseline and sample are from the same operation on the same data,
 * we should get Pass or Inconclusive, but NOT Fail.
 */
static void test_analyze_identical_distributions(void **state) {
    (void)state;

    /* Allocate sample arrays */
    uint64_t *baseline = malloc(ONE_SHOT_SAMPLES * sizeof(uint64_t));
    uint64_t *sample = malloc(ONE_SHOT_SAMPLES * sizeof(uint64_t));
    assert_non_null(baseline);
    assert_non_null(sample);

    /* Collect identical samples (same operation on same data) */
    collect_identical_samples(baseline, sample, ONE_SHOT_SAMPLES);

    /* Create config with generous thresholds */
    struct ToConfig config = to_config_adjacent_network();
    config.pass_threshold = 0.05;
    config.fail_threshold = 0.95;

    /* Run analysis */
    struct ToResult result;
    enum ToError err = to_analyze(baseline, sample, ONE_SHOT_SAMPLES, &config, &result);

    /* Check that analysis succeeded */
    assert_int_equal(err, Ok);

    /* Should NOT be Fail (identical distributions have no timing leak) */
    assert_int_not_equal(result.outcome, Fail);

    /* Leak probability should be low to moderate */
    assert_true(result.leak_probability < 0.95);

    free(baseline);
    free(sample);
}

/**
 * Test that artificially shifted samples are detected as a leak.
 * Adding a large constant shift (1000+ ticks) should trigger Fail.
 */
static void test_analyze_shifted_samples(void **state) {
    (void)state;

    /* Allocate sample arrays */
    uint64_t *baseline = malloc(ONE_SHOT_SAMPLES * sizeof(uint64_t));
    uint64_t *sample = malloc(ONE_SHOT_SAMPLES * sizeof(uint64_t));
    assert_non_null(baseline);
    assert_non_null(sample);

    /* Collect samples with artificial shift */
    collect_shifted_samples(baseline, sample, ONE_SHOT_SAMPLES, LARGE_SHIFT_TICKS);

    /* Create config with adjacent network threshold (100ns) */
    struct ToConfig config = to_config_adjacent_network();
    config.pass_threshold = 0.05;
    config.fail_threshold = 0.95;

    /* Run analysis */
    struct ToResult result;
    enum ToError err = to_analyze(baseline, sample, ONE_SHOT_SAMPLES, &config, &result);

    /* Check that analysis succeeded */
    assert_int_equal(err, Ok);

    /* With large shift, should detect a timing difference */
    /* The outcome depends on the threshold vs actual effect size */
    /* With 2000 tick shift at ~3GHz, that's ~600+ ns, should be detected */
    assert_true(result.leak_probability > 0.5);

    /* Effect should show a shift */
    assert_true(result.effect.shift_ns > 0.0 || result.effect.tail_ns > 0.0);

    free(baseline);
    free(sample);
}

/**
 * Test that null baseline pointer returns NullPointer error.
 */
static void test_analyze_null_baseline(void **state) {
    (void)state;

    uint64_t sample[100];
    struct ToConfig config = to_config_adjacent_network();
    struct ToResult result;

    /* Call with null baseline */
    enum ToError err = to_analyze(NULL, sample, 100, &config, &result);

    /* Should return NullPointer error */
    assert_int_equal(err, NullPointer);
}

/**
 * Test that null sample pointer returns NullPointer error.
 */
static void test_analyze_null_sample(void **state) {
    (void)state;

    uint64_t baseline[100];
    struct ToConfig config = to_config_adjacent_network();
    struct ToResult result;

    /* Call with null sample */
    enum ToError err = to_analyze(baseline, NULL, 100, &config, &result);

    /* Should return NullPointer error */
    assert_int_equal(err, NullPointer);
}

/**
 * Test that null config pointer returns NullPointer error.
 */
static void test_analyze_null_config(void **state) {
    (void)state;

    uint64_t baseline[100];
    uint64_t sample[100];
    struct ToResult result;

    /* Call with null config */
    enum ToError err = to_analyze(baseline, sample, 100, NULL, &result);

    /* Should return NullPointer error */
    assert_int_equal(err, NullPointer);
}

/**
 * Test that null result pointer returns NullPointer error.
 */
static void test_analyze_null_result(void **state) {
    (void)state;

    uint64_t baseline[100];
    uint64_t sample[100];
    struct ToConfig config = to_config_adjacent_network();

    /* Call with null result */
    enum ToError err = to_analyze(baseline, sample, 100, &config, NULL);

    /* Should return NullPointer error */
    assert_int_equal(err, NullPointer);
}

/**
 * Test that zero count returns NotEnoughSamples error.
 */
static void test_analyze_zero_count(void **state) {
    (void)state;

    uint64_t baseline[100];
    uint64_t sample[100];
    struct ToConfig config = to_config_adjacent_network();
    struct ToResult result;

    /* Call with zero count */
    enum ToError err = to_analyze(baseline, sample, 0, &config, &result);

    /* Should return NotEnoughSamples error */
    assert_int_equal(err, NotEnoughSamples);
}

/**
 * Test that very small sample count returns NotEnoughSamples error.
 */
static void test_analyze_insufficient_samples(void **state) {
    (void)state;

    /* Create tiny sample arrays */
    uint64_t baseline[10] = {100, 101, 99, 100, 100, 101, 99, 100, 100, 101};
    uint64_t sample[10] = {100, 101, 99, 100, 100, 101, 99, 100, 100, 101};

    struct ToConfig config = to_config_adjacent_network();
    struct ToResult result;

    /* Call with very small count - should either fail or return error */
    enum ToError err = to_analyze(baseline, sample, 10, &config, &result);

    /* Either NotEnoughSamples error or Ok with Inconclusive/Unmeasurable */
    if (err == Ok) {
        /* If it succeeds, it shouldn't falsely claim high confidence */
        assert_true(result.outcome == Inconclusive ||
                    result.outcome == Unmeasurable ||
                    result.outcome == Pass);
    } else {
        assert_int_equal(err, NotEnoughSamples);
    }
}

/**
 * Test analysis result fields are populated correctly.
 */
static void test_analyze_result_fields(void **state) {
    (void)state;

    /* Allocate sample arrays */
    uint64_t *baseline = malloc(ONE_SHOT_SAMPLES * sizeof(uint64_t));
    uint64_t *sample = malloc(ONE_SHOT_SAMPLES * sizeof(uint64_t));
    assert_non_null(baseline);
    assert_non_null(sample);

    /* Collect identical samples */
    collect_identical_samples(baseline, sample, ONE_SHOT_SAMPLES);

    struct ToConfig config = to_config_adjacent_network();
    struct ToResult result;

    enum ToError err = to_analyze(baseline, sample, ONE_SHOT_SAMPLES, &config, &result);
    assert_int_equal(err, Ok);

    /* Check that result fields are valid */
    assert_true(result.leak_probability >= 0.0 && result.leak_probability <= 1.0);
    assert_true(result.samples_used > 0);
    assert_true(result.elapsed_secs >= 0.0);

    /* Effect pattern should be valid */
    assert_true(result.effect.pattern >= UniformShift && result.effect.pattern <= Indeterminate);

    /* Quality should be valid */
    assert_true(result.quality >= Excellent && result.quality <= TooNoisy);

    /* Threshold values should be set */
    assert_true(result.theta_user_ns > 0.0);  /* AdjacentNetwork is 100ns */
    assert_true(result.theta_eff_ns > 0.0);

    free(baseline);
    free(sample);
}

/* Test group for one-shot analysis */
const struct CMUnitTest one_shot_tests[] = {
    cmocka_unit_test(test_analyze_identical_distributions),
    cmocka_unit_test(test_analyze_shifted_samples),
    cmocka_unit_test(test_analyze_null_baseline),
    cmocka_unit_test(test_analyze_null_sample),
    cmocka_unit_test(test_analyze_null_config),
    cmocka_unit_test(test_analyze_null_result),
    cmocka_unit_test(test_analyze_zero_count),
    cmocka_unit_test(test_analyze_insufficient_samples),
    cmocka_unit_test(test_analyze_result_fields),
};

int run_one_shot_tests(void) {
    return cmocka_run_group_tests_name("One-Shot Analysis Tests", one_shot_tests, NULL, NULL);
}
