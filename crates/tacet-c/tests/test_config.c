/**
 * @file test_config.c
 * @brief CMocka tests for tacet configuration functions.
 *
 * Tests for:
 * - to_config_default()
 * - to_config_shared_hardware()
 * - to_config_adjacent_network()
 * - to_config_remote_network()
 * - to_attacker_threshold_ns()
 */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <string.h>

#include "../include/tacet.h"

/* Threshold values from spec */
#define SHARED_HARDWARE_THRESHOLD_NS 0.6
#define POST_QUANTUM_THRESHOLD_NS 3.3
#define ADJACENT_NETWORK_THRESHOLD_NS 100.0
#define REMOTE_NETWORK_THRESHOLD_NS 50000.0  /* 50 microseconds */

/* Tolerance for floating point comparisons */
#define EPSILON 0.001

/**
 * Test that to_config_default with AdjacentNetwork returns correct defaults.
 */
static void test_config_default_adjacent_network(void **state) {
    (void)state;

    struct ToConfig config = to_config_default(AdjacentNetwork);

    /* Check attacker model */
    assert_int_equal(config.attacker_model, AdjacentNetwork);

    /* Check default thresholds */
    assert_true(fabs(config.pass_threshold - 0.05) < EPSILON);
    assert_true(fabs(config.fail_threshold - 0.95) < EPSILON);

    /* Check custom_threshold_ns is 0 (using model default) */
    assert_true(fabs(config.custom_threshold_ns - 0.0) < EPSILON);

    /* Check that to_attacker_threshold_ns returns correct value */
    double threshold = to_attacker_threshold_ns(AdjacentNetwork);
    assert_true(fabs(threshold - ADJACENT_NETWORK_THRESHOLD_NS) < EPSILON);
}

/**
 * Test that to_config_shared_hardware returns correct threshold.
 */
static void test_config_shared_hardware_threshold(void **state) {
    (void)state;

    struct ToConfig config = to_config_shared_hardware();

    /* Check attacker model */
    assert_int_equal(config.attacker_model, SharedHardware);

    /* Check threshold value via to_attacker_threshold_ns */
    double threshold = to_attacker_threshold_ns(SharedHardware);
    assert_true(fabs(threshold - SHARED_HARDWARE_THRESHOLD_NS) < EPSILON);

    /* Check default decision thresholds */
    assert_true(fabs(config.pass_threshold - 0.05) < EPSILON);
    assert_true(fabs(config.fail_threshold - 0.95) < EPSILON);
}

/**
 * Test that to_config_remote_network returns correct threshold.
 */
static void test_config_remote_network_threshold(void **state) {
    (void)state;

    struct ToConfig config = to_config_remote_network();

    /* Check attacker model */
    assert_int_equal(config.attacker_model, RemoteNetwork);

    /* Check threshold value via to_attacker_threshold_ns */
    double threshold = to_attacker_threshold_ns(RemoteNetwork);
    assert_true(fabs(threshold - REMOTE_NETWORK_THRESHOLD_NS) < EPSILON);

    /* Check default decision thresholds */
    assert_true(fabs(config.pass_threshold - 0.05) < EPSILON);
    assert_true(fabs(config.fail_threshold - 0.95) < EPSILON);
}

/**
 * Test that PostQuantum attacker model has correct threshold.
 */
static void test_config_post_quantum_threshold(void **state) {
    (void)state;

    struct ToConfig config = to_config_default(PostQuantum);

    /* Check attacker model */
    assert_int_equal(config.attacker_model, PostQuantum);

    /* Check threshold value via to_attacker_threshold_ns */
    double threshold = to_attacker_threshold_ns(PostQuantum);
    assert_true(fabs(threshold - POST_QUANTUM_THRESHOLD_NS) < EPSILON);
}

/**
 * Test custom threshold configuration.
 */
static void test_config_custom_threshold(void **state) {
    (void)state;

    struct ToConfig config = to_config_default(AdjacentNetwork);

    /* Set custom threshold */
    double custom_ns = 500.0;  /* 500 nanoseconds */
    config.custom_threshold_ns = custom_ns;

    /* Verify custom threshold is set */
    assert_true(fabs(config.custom_threshold_ns - custom_ns) < EPSILON);
}

/**
 * Test Research attacker model (theta -> 0).
 */
static void test_config_research_threshold(void **state) {
    (void)state;

    struct ToConfig config = to_config_default(Research);

    /* Check attacker model */
    assert_int_equal(config.attacker_model, Research);

    /* Research threshold should be very small (approaching 0) */
    double threshold = to_attacker_threshold_ns(Research);
    assert_true(threshold < 0.1);  /* Should be essentially 0 */
}

/**
 * Test that config parameters can be modified.
 */
static void test_config_modification(void **state) {
    (void)state;

    struct ToConfig config = to_config_adjacent_network();

    /* Modify time budget */
    config.time_budget_secs = 60.0;
    assert_true(fabs(config.time_budget_secs - 60.0) < EPSILON);

    /* Modify max samples */
    config.max_samples = 50000;
    assert_int_equal(config.max_samples, 50000);

    /* Modify decision thresholds */
    config.pass_threshold = 0.01;
    config.fail_threshold = 0.99;
    assert_true(fabs(config.pass_threshold - 0.01) < EPSILON);
    assert_true(fabs(config.fail_threshold - 0.99) < EPSILON);

    /* Modify seed */
    config.seed = 42;
    assert_int_equal(config.seed, 42);

    /* Modify timer frequency */
    config.timer_frequency_hz = 3000000000ULL;  /* 3 GHz */
    assert_int_equal(config.timer_frequency_hz, 3000000000ULL);
}

/**
 * Test to_version returns non-null string.
 */
static void test_version_non_null(void **state) {
    (void)state;

    const char *version = to_version();
    assert_non_null(version);
    assert_true(strlen(version) > 0);
}

/* Test group for config functions */
const struct CMUnitTest config_tests[] = {
    cmocka_unit_test(test_config_default_adjacent_network),
    cmocka_unit_test(test_config_shared_hardware_threshold),
    cmocka_unit_test(test_config_remote_network_threshold),
    cmocka_unit_test(test_config_post_quantum_threshold),
    cmocka_unit_test(test_config_custom_threshold),
    cmocka_unit_test(test_config_research_threshold),
    cmocka_unit_test(test_config_modification),
    cmocka_unit_test(test_version_non_null),
};

int run_config_tests(void) {
    return cmocka_run_group_tests_name("Config Tests", config_tests, NULL, NULL);
}
