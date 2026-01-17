/**
 * @file compare.c
 * @brief Demonstrates detection of leaky vs constant-time comparison.
 *
 * This example shows the difference between:
 * 1. A leaky comparison with early exit (FAILS the test)
 * 2. A constant-time comparison using XOR accumulator (PASSES the test)
 *
 * This pattern is critical for password verification, MAC validation,
 * and any security-sensitive comparison.
 *
 * Build:
 *   make compare
 *
 * Run:
 *   ./compare
 */

#include "common.h"

#define KEY_SIZE 32

/* Secret key for comparison */
static uint8_t secret_key[KEY_SIZE];

/* ============================================================================
 * Comparison implementations
 * ============================================================================ */

/**
 * @brief INSECURE: Early-exit comparison.
 *
 * This is how NOT to compare secrets. The function returns as soon as
 * a mismatch is found, leaking the position of the first differing byte
 * through timing.
 *
 * Exploitability: An attacker can guess bytes one at a time, checking
 * whether the comparison took longer (indicating the guessed byte matched).
 */
static bool leaky_compare(const uint8_t *a, const uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return false;  /* Timing leak: early return */
        }
    }
    return true;
}

/**
 * @brief SECURE: Constant-time comparison using XOR accumulator.
 *
 * This compares all bytes regardless of where mismatches occur.
 * The timing is the same whether the first byte differs or the last.
 *
 * Pattern: XOR all byte differences into an accumulator, then check
 * if the accumulator is zero at the end.
 */
static bool constant_time_compare(const uint8_t *a, const uint8_t *b, size_t n) {
    uint8_t acc = 0;
    for (size_t i = 0; i < n; i++) {
        acc |= a[i] ^ b[i];
    }
    return acc == 0;
}

/* ============================================================================
 * Test harness
 * ============================================================================ */

/* Context to select which comparison function to test */
typedef struct {
    bool (*compare_fn)(const uint8_t *, const uint8_t *, size_t);
    const char *name;
} test_context_t;

static void generator(void *ctx, bool is_baseline, uint8_t *output, size_t size) {
    (void)ctx;
    if (is_baseline) {
        memset(output, 0, size);
    } else {
        random_bytes(output, size);
    }
}

static void operation(void *ctx, const uint8_t *input, size_t size) {
    test_context_t *test = (test_context_t *)ctx;
    bool result = test->compare_fn(input, secret_key, size);
    DO_NOT_OPTIMIZE(result);
}

/**
 * @brief Run a timing test on a comparison function.
 */
static to_outcome_t test_comparison(
    const char *name,
    bool (*compare_fn)(const uint8_t *, const uint8_t *, size_t)
) {
    test_context_t ctx = {
        .compare_fn = compare_fn,
        .name = name
    };

    printf("\n--- Testing: %s ---\n", name);

    to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
    config.time_budget_secs = 15.0;

    to_result_t result = to_test(&config, KEY_SIZE, generator, operation, &ctx);

    print_result_colored(&result);

    if (result.outcome == TO_OUTCOME_FAIL) {
        printf("  Effect: %.1f ns shift, %.1f ns tail\n",
               result.effect.shift_ns, result.effect.tail_ns);
        printf("  Pattern: %s\n", to_effect_pattern_str(result.effect.pattern));
    }

    to_outcome_t outcome = result.outcome;
    to_result_free(&result);
    return outcome;
}

int main(void) {
    printf("timing-oracle comparison example\n");
    printf("================================\n");
    printf("\nThis example tests two comparison implementations:\n");
    printf("1. Leaky (early-exit): should FAIL\n");
    printf("2. Constant-time (XOR): should PASS\n");

    /* Initialize secret key */
    random_bytes(secret_key, KEY_SIZE);

    /* Test the leaky implementation */
    to_outcome_t leaky_outcome = test_comparison("leaky_compare (early-exit)", leaky_compare);

    /* Test the constant-time implementation */
    to_outcome_t ct_outcome = test_comparison("constant_time_compare (XOR)", constant_time_compare);

    /* Summary */
    printf("\n=== Summary ===\n");
    printf("Leaky comparison:        %s\n",
           leaky_outcome == TO_OUTCOME_FAIL ? "FAIL (as expected)" :
           leaky_outcome == TO_OUTCOME_PASS ? "PASS (unexpected!)" : "INCONCLUSIVE");
    printf("Constant-time comparison: %s\n",
           ct_outcome == TO_OUTCOME_PASS ? "PASS (as expected)" :
           ct_outcome == TO_OUTCOME_FAIL ? "FAIL (unexpected!)" : "INCONCLUSIVE");

    /* Validation check */
    if (leaky_outcome == TO_OUTCOME_FAIL && ct_outcome == TO_OUTCOME_PASS) {
        printf("\nBoth tests behaved as expected.\n");
        return 0;
    } else {
        printf("\nNote: Results may vary based on platform and conditions.\n");
        return 1;
    }
}
