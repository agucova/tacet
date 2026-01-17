/**
 * @file custom_config.c
 * @brief Advanced configuration options for timing-oracle.
 *
 * This example demonstrates all configuration parameters:
 * - Attacker model and custom thresholds
 * - Time and sample budgets
 * - Decision thresholds (pass/fail)
 * - Reproducibility with seeds
 *
 * Build:
 *   make custom_config
 *
 * Run:
 *   ./custom_config
 */

#include "common.h"

#define INPUT_SIZE 32

static uint8_t secret[INPUT_SIZE];

/* Simple XOR operation for testing */
static void xor_operation(void *ctx, const uint8_t *input, size_t size) {
    (void)ctx;
    uint8_t acc = 0;
    for (size_t i = 0; i < size; i++) {
        acc ^= input[i] ^ secret[i];
    }
    DO_NOT_OPTIMIZE(acc);
}

/**
 * @brief Print the current configuration.
 */
static void print_config(const to_config_t *cfg) {
    printf("Configuration:\n");
    printf("  Attacker model: ");
    switch (cfg->attacker_model) {
        case TO_ATTACKER_SHARED_HARDWARE:
            printf("SharedHardware (~0.6 ns)\n");
            break;
        case TO_ATTACKER_POST_QUANTUM:
            printf("PostQuantum (~3.3 ns)\n");
            break;
        case TO_ATTACKER_ADJACENT_NETWORK:
            printf("AdjacentNetwork (100 ns)\n");
            break;
        case TO_ATTACKER_REMOTE_NETWORK:
            printf("RemoteNetwork (50 us)\n");
            break;
        case TO_ATTACKER_RESEARCH:
            printf("Research (detect any diff)\n");
            break;
        case TO_ATTACKER_CUSTOM:
            printf("Custom (%.1f ns)\n", cfg->custom_threshold_ns);
            break;
    }
    printf("  Time budget: %.1f seconds\n",
           cfg->time_budget_secs > 0 ? cfg->time_budget_secs : 30.0);
    printf("  Max samples: %zu%s\n",
           cfg->max_samples > 0 ? cfg->max_samples : 100000,
           cfg->max_samples == 0 ? " (default)" : "");
    printf("  Pass threshold: %.2f (declare PASS when P(leak) < %.0f%%)\n",
           cfg->pass_threshold, cfg->pass_threshold * 100);
    printf("  Fail threshold: %.2f (declare FAIL when P(leak) > %.0f%%)\n",
           cfg->fail_threshold, cfg->fail_threshold * 100);
    printf("  Seed: %s\n",
           cfg->seed == 0 ? "random (system entropy)" : "fixed");
    if (cfg->seed != 0) {
        printf("         Value: %llu\n", (unsigned long long)cfg->seed);
    }
    printf("\n");
}

int main(void) {
    printf("timing-oracle Custom Configuration Example\n");
    printf("==========================================\n\n");

    random_bytes(secret, INPUT_SIZE);

    /* ========================================
     * Example 1: Default configuration
     * ======================================== */
    printf("=== Example 1: Default Configuration ===\n\n");
    {
        to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
        print_config(&config);

        printf("Running test...\n");
        to_result_t result = to_test(&config, INPUT_SIZE, standard_generator,
                                     xor_operation, NULL);
        print_result_summary(&result);
        to_result_free(&result);
    }

    /* ========================================
     * Example 2: Quick scan (short budget)
     * ======================================== */
    printf("\n=== Example 2: Quick Scan (5 second budget) ===\n\n");
    {
        to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
        config.time_budget_secs = 5.0;
        config.max_samples = 10000;  /* Also limit samples */
        print_config(&config);

        printf("Running test...\n");
        to_result_t result = to_test(&config, INPUT_SIZE, standard_generator,
                                     xor_operation, NULL);
        print_result_summary(&result);
        to_result_free(&result);
    }

    /* ========================================
     * Example 3: High confidence (strict thresholds)
     * ======================================== */
    printf("\n=== Example 3: High Confidence (strict thresholds) ===\n\n");
    {
        to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
        config.pass_threshold = 0.01;  /* Pass only if P(leak) < 1% */
        config.fail_threshold = 0.99;  /* Fail only if P(leak) > 99% */
        config.time_budget_secs = 15.0;
        print_config(&config);

        printf("Note: Stricter thresholds require more samples to decide.\n");
        printf("Running test...\n");
        to_result_t result = to_test(&config, INPUT_SIZE, standard_generator,
                                     xor_operation, NULL);
        print_result_summary(&result);
        to_result_free(&result);
    }

    /* ========================================
     * Example 4: Custom threshold
     * ======================================== */
    printf("\n=== Example 4: Custom Threshold (25 ns) ===\n\n");
    {
        to_config_t config = to_config_default(TO_ATTACKER_CUSTOM);
        config.custom_threshold_ns = 25.0;  /* Only effective with CUSTOM */
        config.time_budget_secs = 10.0;
        print_config(&config);

        printf("Running test...\n");
        to_result_t result = to_test(&config, INPUT_SIZE, standard_generator,
                                     xor_operation, NULL);
        print_result_summary(&result);
        to_result_free(&result);
    }

    /* ========================================
     * Example 5: Reproducible results (fixed seed)
     * ======================================== */
    printf("\n=== Example 5: Reproducible Results (fixed seed) ===\n\n");
    {
        to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
        config.seed = 12345;  /* Fixed seed for reproducibility */
        config.time_budget_secs = 10.0;
        print_config(&config);

        printf("Running test twice with same seed...\n\n");

        /* First run */
        to_result_t result1 = to_test(&config, INPUT_SIZE, standard_generator,
                                      xor_operation, NULL);
        printf("Run 1: ");
        print_result_summary(&result1);

        /* Second run (same config) */
        to_result_t result2 = to_test(&config, INPUT_SIZE, standard_generator,
                                      xor_operation, NULL);
        printf("Run 2: ");
        print_result_summary(&result2);

        printf("\nNote: With fixed seed, the sample schedule is deterministic.\n");
        printf("Results may still differ due to timing measurement variance.\n");

        to_result_free(&result1);
        to_result_free(&result2);
    }

    /* ========================================
     * Example 6: NULL config (use all defaults)
     * ======================================== */
    printf("\n=== Example 6: NULL Config (all defaults) ===\n\n");
    {
        printf("Passing NULL for config uses all default values.\n");
        printf("Default attacker model: AdjacentNetwork (100 ns)\n\n");

        printf("Running test...\n");
        to_result_t result = to_test(NULL, INPUT_SIZE, standard_generator,
                                     xor_operation, NULL);
        print_result_summary(&result);
        to_result_free(&result);
    }

    printf("\n=== Configuration Summary ===\n\n");
    printf("to_config_t fields:\n");
    printf("  attacker_model      - Threat model (determines threshold)\n");
    printf("  custom_threshold_ns - Custom threshold (only with CUSTOM model)\n");
    printf("  time_budget_secs    - Max test duration (0 = 30s default)\n");
    printf("  max_samples         - Max samples per class (0 = 100k default)\n");
    printf("  pass_threshold      - P(leak) threshold for PASS (default: 0.05)\n");
    printf("  fail_threshold      - P(leak) threshold for FAIL (default: 0.95)\n");
    printf("  seed                - RNG seed (0 = system entropy)\n");

    return 0;
}
