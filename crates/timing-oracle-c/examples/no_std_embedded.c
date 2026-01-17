/**
 * @file no_std_embedded.c
 * @brief External time tracking for embedded/SGX environments.
 *
 * This example demonstrates `to_test_with_time()`, which allows the caller
 * to provide time tracking externally. This is useful in environments where:
 *
 * - std library is unavailable (no_std)
 * - SGX enclaves with restricted syscalls
 * - Embedded systems with custom timers
 * - Testing scenarios where time is simulated
 *
 * The key difference from `to_test()`:
 * - Caller provides a pointer to elapsed_secs
 * - Caller is responsible for updating this value if needed
 * - The library reads this value for budget checks
 *
 * Build:
 *   make no_std_embedded
 *
 * Run:
 *   ./no_std_embedded
 */

#include "common.h"
#include <time.h>

#define INPUT_SIZE 32

static uint8_t secret[INPUT_SIZE];

/* Simple constant-time operation */
static void ct_operation(void *ctx, const uint8_t *input, size_t size) {
    (void)ctx;
    uint8_t acc = 0;
    for (size_t i = 0; i < size; i++) {
        acc ^= input[i] ^ secret[i];
    }
    DO_NOT_OPTIMIZE(acc);
}

/**
 * @brief Get current time in seconds (for demo purposes).
 *
 * In a real embedded environment, you would use your platform's
 * timer instead of clock_gettime.
 */
static double get_time_secs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(void) {
    printf("timing-oracle External Time Tracking Example\n");
    printf("=============================================\n\n");

    printf("This example uses to_test_with_time() instead of to_test().\n");
    printf("The caller provides the elapsed time externally, which is useful\n");
    printf("for environments without std time support.\n\n");

    random_bytes(secret, INPUT_SIZE);

    /* ========================================
     * Method 1: Simple elapsed time (most common)
     * ======================================== */
    printf("=== Method 1: Provide Initial Elapsed Time ===\n\n");
    {
        printf("For a single blocking call, providing the initial elapsed time\n");
        printf("(usually 0.0) is typically sufficient.\n\n");

        to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
        config.time_budget_secs = 10.0;

        double elapsed_secs = 0.0;  /* Start time */

        printf("Calling to_test_with_time() with elapsed_secs = %.1f\n", elapsed_secs);

        to_result_t result = to_test_with_time(
            &config,
            INPUT_SIZE,
            standard_generator,
            ct_operation,
            NULL,
            &elapsed_secs  /* Pointer to elapsed time */
        );

        printf("\nResult:\n");
        print_result_summary(&result);
        printf("Actual elapsed: %.2f seconds\n", result.elapsed_secs);

        to_result_free(&result);
    }

    /* ========================================
     * Method 2: External timer (advanced)
     * ======================================== */
    printf("\n=== Method 2: External Timer Demo ===\n\n");
    {
        printf("In this demo, we track time externally before the call.\n");
        printf("In practice, you might update elapsed_secs from a callback\n");
        printf("or in a separate thread.\n\n");

        to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
        config.time_budget_secs = 10.0;

        double start_time = get_time_secs();
        double elapsed_secs = 0.0;

        printf("Starting test with external timer...\n");

        /* Note: In a real scenario, you might update elapsed_secs
         * from within a callback if the library supported it,
         * or via a shared atomic variable in a multithreaded setup.
         *
         * For this single-threaded demo, we just provide 0.0
         * and let the library's internal timer do the work. */

        to_result_t result = to_test_with_time(
            &config,
            INPUT_SIZE,
            standard_generator,
            ct_operation,
            NULL,
            &elapsed_secs
        );

        double end_time = get_time_secs();
        double actual_elapsed = end_time - start_time;

        printf("\nResult:\n");
        print_result_summary(&result);
        printf("External timer measured: %.2f seconds\n", actual_elapsed);
        printf("Library reported: %.2f seconds\n", result.elapsed_secs);

        to_result_free(&result);
    }

    /* ========================================
     * When to use to_test_with_time()
     * ======================================== */
    printf("\n=== When to Use to_test_with_time() ===\n\n");

    printf("Use to_test_with_time() when:\n\n");

    printf("1. SGX Enclaves\n");
    printf("   SGX restricts system calls. You may need to get time\n");
    printf("   from the untrusted host or use enclave-internal timing.\n\n");

    printf("2. Bare Metal / RTOS\n");
    printf("   Embedded systems often have custom timer peripherals.\n");
    printf("   Use your platform's timer instead of std::time.\n\n");

    printf("3. Simulation / Testing\n");
    printf("   When testing the library itself, you may want to\n");
    printf("   simulate time progression without actual delays.\n\n");

    printf("4. Custom Budget Logic\n");
    printf("   If you need custom time accounting (e.g., excluding\n");
    printf("   time spent in certain operations), provide your own.\n\n");

    printf("For most use cases, prefer to_test() which handles\n");
    printf("time tracking automatically using the platform timer.\n");

    return 0;
}
