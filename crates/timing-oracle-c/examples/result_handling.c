/**
 * @file result_handling.c
 * @brief Complete guide to interpreting timing-oracle results.
 *
 * This example demonstrates how to handle all possible outcomes
 * and extract useful diagnostic information from results.
 *
 * Outcomes:
 *   PASS          - No timing leak detected (P(leak) < 5%)
 *   FAIL          - Timing leak confirmed (P(leak) > 95%)
 *   INCONCLUSIVE  - Cannot determine (5% < P(leak) < 95%)
 *   UNMEASURABLE  - Operation too fast for the timer
 *
 * Build:
 *   make result_handling
 *
 * Run:
 *   ./result_handling
 */

#include "common.h"

#define INPUT_SIZE 32

static uint8_t secret[INPUT_SIZE];

/* Different operations to demonstrate different outcomes */

/* Leaky operation - will FAIL */
static void leaky_operation(void *ctx, const uint8_t *input, size_t size) {
    (void)ctx;
    for (size_t i = 0; i < size; i++) {
        if (input[i] != secret[i]) {
            DO_NOT_OPTIMIZE(i);
            return;
        }
    }
}

/* Safe operation - will PASS */
static void safe_operation(void *ctx, const uint8_t *input, size_t size) {
    (void)ctx;
    uint8_t acc = 0;
    for (size_t i = 0; i < size; i++) {
        acc |= input[i] ^ secret[i];
    }
    DO_NOT_OPTIMIZE(acc);
}

/* Very fast operation - will be UNMEASURABLE */
static void fast_operation(void *ctx, const uint8_t *input, size_t size) {
    (void)ctx;
    (void)input;
    (void)size;
    /* Almost nothing here - too fast to measure */
    volatile int x = 1;
    (void)x;
}

/**
 * @brief Demonstrates exhaustive result handling.
 *
 * This function shows how to properly handle all outcome types
 * and extract all useful information from the result.
 */
static void handle_result_exhaustively(const to_result_t *r) {
    printf("\n========================================\n");
    printf("RESULT ANALYSIS\n");
    printf("========================================\n\n");

    /* 1. Basic outcome */
    printf("1. OUTCOME: %s\n", to_outcome_str(r->outcome));
    printf("   Leak probability: %.2f%%\n", r->leak_probability * 100.0);

    /* 2. Measurement stats */
    printf("\n2. MEASUREMENT STATISTICS:\n");
    printf("   Samples per class: %zu\n", r->samples_used);
    printf("   Elapsed time: %.2f seconds\n", r->elapsed_secs);
    printf("   Throughput: %.0f samples/sec\n",
           r->elapsed_secs > 0 ? (double)r->samples_used / r->elapsed_secs : 0);

    /* 3. Platform info */
    printf("\n3. PLATFORM INFO:\n");
    printf("   Timer: %s\n", r->timer_name);
    printf("   Resolution: %.2f ns\n", r->timer_resolution_ns);
    printf("   Platform: %s\n", r->platform);
    printf("   Adaptive batching: %s\n", r->adaptive_batching_used ? "enabled" : "disabled");

    /* 4. Quality assessment */
    printf("\n4. MEASUREMENT QUALITY: %s\n", to_quality_str(r->quality));
    switch (r->quality) {
        case TO_QUALITY_EXCELLENT:
            printf("   MDE < 5 ns - Can detect very small leaks\n");
            break;
        case TO_QUALITY_GOOD:
            printf("   MDE 5-20 ns - Suitable for most tests\n");
            break;
        case TO_QUALITY_POOR:
            printf("   MDE 20-100 ns - May miss small leaks\n");
            break;
        case TO_QUALITY_TOO_NOISY:
            printf("   MDE > 100 ns - Consider reducing system noise\n");
            break;
    }

    /* 5. Effect size decomposition */
    printf("\n5. EFFECT SIZE ANALYSIS:\n");
    printf("   Total effect: %.2f ns\n", r->effect.shift_ns + r->effect.tail_ns);
    printf("   - Shift component: %.2f ns\n", r->effect.shift_ns);
    printf("   - Tail component: %.2f ns\n", r->effect.tail_ns);
    printf("   95%% Credible interval: [%.2f, %.2f] ns\n",
           r->effect.ci_low_ns, r->effect.ci_high_ns);
    printf("   Pattern: %s\n", to_effect_pattern_str(r->effect.pattern));

    /* Pattern interpretation */
    switch (r->effect.pattern) {
        case TO_EFFECT_UNIFORM_SHIFT:
            printf("   -> Consistent timing difference (e.g., different code path)\n");
            break;
        case TO_EFFECT_TAIL_EFFECT:
            printf("   -> Variable timing in upper quantiles (e.g., cache effects)\n");
            break;
        case TO_EFFECT_MIXED:
            printf("   -> Both shift and tail effects present\n");
            break;
        case TO_EFFECT_INDETERMINATE:
            printf("   -> Pattern could not be determined\n");
            break;
    }

    /* 6. Outcome-specific handling */
    printf("\n6. OUTCOME-SPECIFIC DETAILS:\n");

    switch (r->outcome) {
        case TO_OUTCOME_PASS:
            printf("   Status: SECURE (within threshold)\n");
            printf("   The operation shows no detectable timing leak.\n");
            printf("   Safe for use in the target threat model.\n");
            break;

        case TO_OUTCOME_FAIL:
            printf("   Status: VULNERABLE - TIMING LEAK DETECTED\n");
            printf("   Exploitability: %s\n", to_exploitability_str(r->exploitability));

            switch (r->exploitability) {
                case TO_EXPLOIT_NEGLIGIBLE:
                    printf("   -> Effect < 100 ns: Low practical risk\n");
                    break;
                case TO_EXPLOIT_POSSIBLE_LAN:
                    printf("   -> Effect 100-500 ns: Exploitable on LAN with effort\n");
                    break;
                case TO_EXPLOIT_LIKELY_LAN:
                    printf("   -> Effect 500ns-20us: Easily exploitable on LAN\n");
                    break;
                case TO_EXPLOIT_POSSIBLE_REMOTE:
                    printf("   -> Effect > 20 us: Potentially exploitable remotely\n");
                    break;
            }
            break;

        case TO_OUTCOME_INCONCLUSIVE:
            printf("   Status: UNDETERMINED\n");
            printf("   Reason: %s\n", to_inconclusive_reason_str(r->inconclusive_reason));

            switch (r->inconclusive_reason) {
                case TO_INCONCLUSIVE_DATA_TOO_NOISY:
                    printf("   -> Reduce system noise or increase samples\n");
                    break;
                case TO_INCONCLUSIVE_NOT_LEARNING:
                    printf("   -> Data not informative enough\n");
                    break;
                case TO_INCONCLUSIVE_WOULD_TAKE_TOO_LONG:
                    printf("   -> Estimated time exceeds budget\n");
                    break;
                case TO_INCONCLUSIVE_TIME_BUDGET_EXCEEDED:
                    printf("   -> Increase time_budget_secs\n");
                    break;
                case TO_INCONCLUSIVE_SAMPLE_BUDGET_EXCEEDED:
                    printf("   -> Increase max_samples\n");
                    break;
                case TO_INCONCLUSIVE_CONDITIONS_CHANGED:
                    printf("   -> System conditions changed during test\n");
                    break;
                case TO_INCONCLUSIVE_THRESHOLD_UNACHIEVABLE:
                    printf("   -> Requested threshold below measurement floor\n");
                    break;
                case TO_INCONCLUSIVE_MODEL_MISMATCH:
                    printf("   -> Statistical model doesn't fit the data\n");
                    break;
            }
            break;

        case TO_OUTCOME_UNMEASURABLE:
            printf("   Status: CANNOT MEASURE\n");
            printf("   Measured operation time: %.2f ns\n", r->operation_ns);
            printf("   Timer resolution: %.2f ns\n", r->timer_resolution_ns);
            printf("   -> Operation is too fast relative to timer precision\n");
            printf("   -> Consider batching multiple operations together\n");
            break;
    }

    /* 7. Recommendation */
    if (r->recommendation && strlen(r->recommendation) > 0) {
        printf("\n7. RECOMMENDATION:\n");
        printf("   %s\n", r->recommendation);
    }
}

int main(void) {
    printf("timing-oracle Result Handling Example\n");
    printf("=====================================\n");

    random_bytes(secret, INPUT_SIZE);

    /* Test 1: Leaky operation -> FAIL */
    printf("\n\n>>> TEST 1: Leaky operation (expecting FAIL)\n");
    {
        to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
        config.time_budget_secs = 10.0;

        to_result_t result = to_test(&config, INPUT_SIZE, standard_generator,
                                     leaky_operation, NULL);
        handle_result_exhaustively(&result);
        to_result_free(&result);
    }

    /* Test 2: Safe operation -> PASS */
    printf("\n\n>>> TEST 2: Safe operation (expecting PASS)\n");
    {
        to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
        config.time_budget_secs = 10.0;

        to_result_t result = to_test(&config, INPUT_SIZE, standard_generator,
                                     safe_operation, NULL);
        handle_result_exhaustively(&result);
        to_result_free(&result);
    }

    /* Test 3: Fast operation -> UNMEASURABLE */
    printf("\n\n>>> TEST 3: Very fast operation (expecting UNMEASURABLE)\n");
    {
        to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
        config.time_budget_secs = 5.0;

        to_result_t result = to_test(&config, INPUT_SIZE, standard_generator,
                                     fast_operation, NULL);
        handle_result_exhaustively(&result);
        to_result_free(&result);
    }

    printf("\n\nDone. See above for detailed result handling examples.\n");
    return 0;
}
