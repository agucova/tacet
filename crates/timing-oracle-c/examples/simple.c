/**
 * @file simple.c
 * @brief Basic timing-oracle usage example.
 *
 * This example demonstrates the minimal workflow for detecting timing
 * side channels in a simple comparison function.
 *
 * Build:
 *   make simple
 *
 * Run:
 *   ./simple
 */

#include "common.h"

/* Size of data to compare */
#define DATA_SIZE 32

/* Secret key stored in the "application" */
static uint8_t secret_key[DATA_SIZE];

/**
 * @brief A comparison function with a timing side channel.
 *
 * This is the code we want to test. It has an early-exit pattern
 * that leaks timing information: comparisons fail faster when the
 * first bytes differ than when later bytes differ.
 */
static bool leaky_compare(const uint8_t *a, const uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return false;  /* Early exit - timing leak! */
        }
    }
    return true;
}

/**
 * @brief Generator callback: creates test inputs.
 *
 * For baseline: all zeros (maximally different from random secret)
 * For sample: random bytes (some will partially match the secret)
 */
static void generator(void *ctx, bool is_baseline, uint8_t *output, size_t size) {
    (void)ctx;
    if (is_baseline) {
        memset(output, 0, size);
    } else {
        random_bytes(output, size);
    }
}

/**
 * @brief Operation callback: the code being timed.
 *
 * This is called inside the timed region. We compare the input
 * against our secret key.
 */
static void operation(void *ctx, const uint8_t *input, size_t size) {
    (void)ctx;
    bool result = leaky_compare(input, secret_key, size);
    DO_NOT_OPTIMIZE(result);  /* Prevent optimization */
}

int main(void) {
    printf("timing-oracle simple example\n");
    printf("Library version: %s\n", to_version());
    printf("Timer: %s\n\n", to_timer_name());

    /* Initialize secret key with random data */
    random_bytes(secret_key, DATA_SIZE);

    /* Create configuration for adjacent network attacker (100ns threshold) */
    to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
    config.time_budget_secs = 15.0;  /* Limit to 15 seconds for demo */

    printf("Testing leaky_compare() for timing side channels...\n");
    printf("Attacker model: Adjacent Network (theta=100ns)\n");
    printf("Time budget: %.0f seconds\n\n", config.time_budget_secs);

    /* Run the timing test */
    to_result_t result = to_test(
        &config,
        DATA_SIZE,
        generator,
        operation,
        NULL  /* No context needed */
    );

    /* Display results */
    print_result_full(&result);

    /* Interpret the outcome */
    printf("\n");
    switch (result.outcome) {
        case TO_OUTCOME_PASS:
            printf("No timing leak detected at this threshold.\n");
            break;
        case TO_OUTCOME_FAIL:
            printf("TIMING LEAK DETECTED!\n");
            printf("This code should not be used in production.\n");
            break;
        case TO_OUTCOME_INCONCLUSIVE:
            printf("Could not determine - try with more time/samples.\n");
            break;
        case TO_OUTCOME_UNMEASURABLE:
            printf("Operation too fast to measure - try batching.\n");
            break;
    }

    /* Clean up */
    to_result_free(&result);

    return (result.outcome == TO_OUTCOME_FAIL) ? 1 : 0;
}
