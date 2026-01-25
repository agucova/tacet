/**
 * @file simple_callback.c
 * @brief Callback-based timing-oracle usage example.
 *
 * This example demonstrates the new callback-based API using to_test(),
 * which is the recommended approach for most use cases. It handles all
 * the adaptive sampling logic internally.
 *
 * The C code provides:
 * - A callback function for sample collection
 * - High-resolution timing using rdtsc/cntvct
 * - Input generation
 *
 * The library handles:
 * - Calibration
 * - Adaptive sampling loop
 * - Decision making
 *
 * Build and run (from repo root):
 *   just bindings c-run-simple-callback
 *
 * Or manually:
 *   just bindings c-example-simple-callback
 *   ./crates/timing-oracle-c/examples/simple_callback
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#include "../include/timing_oracle.h"

/* Size of data to compare */
#define DATA_SIZE 32

/* Platform-specific high-resolution timer */
#if defined(__x86_64__) || defined(_M_X64)
static inline uint64_t read_timer(void) {
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
#elif defined(__aarch64__)
static inline uint64_t read_timer(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}
#else
#error "Unsupported architecture"
#endif

/* Compiler barrier to prevent reordering */
#define DO_NOT_OPTIMIZE(x) __asm__ volatile("" : : "r,m"(x) : "memory")

/* Secret key stored in the "application" */
static uint8_t secret_key[DATA_SIZE];

/**
 * @brief A comparison function with a timing side channel.
 *
 * This is the code we want to test. It has an early-exit pattern
 * that leaks timing information.
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
 * @brief Fill buffer with random bytes.
 */
static void random_bytes(uint8_t *buf, size_t size) {
    for (size_t i = 0; i < size; i++) {
        buf[i] = (uint8_t)rand();
    }
}

/**
 * @brief Callback function for sample collection.
 *
 * This is called by to_test() to collect timing samples. The callback
 * should fill baseline_out and sample_out with 'count' samples each
 * using interleaved sampling for best statistical properties.
 */
static void collect_samples(
    uint64_t *baseline_out,
    uint64_t *sample_out,
    size_t count,
    void *user_ctx
) {
    (void)user_ctx;  /* Unused in this example */

    uint8_t input[DATA_SIZE];

    for (size_t i = 0; i < count; i++) {
        /* Baseline measurement: all-zero input */
        memset(input, 0, DATA_SIZE);
        uint64_t start = read_timer();
        bool result = leaky_compare(input, secret_key, DATA_SIZE);
        uint64_t end = read_timer();
        DO_NOT_OPTIMIZE(result);
        baseline_out[i] = end - start;

        /* Sample measurement: random input */
        random_bytes(input, DATA_SIZE);
        start = read_timer();
        result = leaky_compare(input, secret_key, DATA_SIZE);
        end = read_timer();
        DO_NOT_OPTIMIZE(result);
        sample_out[i] = end - start;
    }
}

int main(void) {
    printf("timing-oracle callback example\n");
    printf("Library version: %s\n\n", to_version());

    /* Seed random number generator */
    srand((unsigned int)time(NULL));

    /* Initialize secret key with random data */
    random_bytes(secret_key, DATA_SIZE);

    /* Create configuration, then allow env overrides */
    struct ToConfig config = to_config_default(AdjacentNetwork);
    config.time_budget_secs = 30.0;
    config.max_samples = 100000;
    config = to_config_from_env(config);  /* CI can override via TO_* env vars */

    printf("Configuration:\n");
    printf("  Time budget: %.0f seconds\n", config.time_budget_secs);
    printf("  Max samples: %llu\n", (unsigned long long)config.max_samples);
    printf("  Threshold: %.1f ns\n\n", to_attacker_threshold_ns(config.attacker_model));

    printf("Running timing test (callback-based)...\n\n");

    /* Run the complete test using callback API */
    struct ToResult result;
    enum ToError err = to_test(&config, collect_samples, NULL, &result);

    if (err != Ok) {
        fprintf(stderr, "Test failed with error code: %d\n", err);
        return 1;
    }

    /* Display results */
    printf("=== Analysis Results ===\n\n");
    printf("Outcome: ");
    switch (result.outcome) {
        case Pass:        printf("PASS\n"); break;
        case Fail:        printf("FAIL\n"); break;
        case Inconclusive: printf("INCONCLUSIVE\n"); break;
        case Unmeasurable: printf("UNMEASURABLE\n"); break;
    }

    printf("Leak probability: %.2f%%\n", result.leak_probability * 100.0);
    printf("Effect estimate: %.2f ns (shift) + %.2f ns (tail)\n",
           result.effect.shift_ns, result.effect.tail_ns);
    printf("Samples used: %llu per class\n", (unsigned long long)result.samples_used);
    printf("Elapsed time: %.2f seconds\n", result.elapsed_secs);

    printf("Measurement quality: ");
    switch (result.quality) {
        case Excellent: printf("Excellent\n"); break;
        case Good:      printf("Good\n"); break;
        case Poor:      printf("Poor\n"); break;
        case TooNoisy:  printf("Too Noisy\n"); break;
    }

    if (result.outcome == Fail) {
        printf("\nExploitability: ");
        switch (result.exploitability) {
            case SharedHardwareOnly: printf("SharedHardwareOnly\n"); break;
            case Http2Multiplexing:  printf("Http2Multiplexing\n"); break;
            case StandardRemote:     printf("StandardRemote\n"); break;
            case ObviousLeak:        printf("ObviousLeak\n"); break;
        }
    }

    if (result.outcome == Inconclusive) {
        printf("\nInconclusive reason: ");
        switch (result.inconclusive_reason) {
            case None:                 printf("None\n"); break;
            case DataTooNoisy:         printf("DataTooNoisy\n"); break;
            case NotLearning:          printf("NotLearning\n"); break;
            case WouldTakeTooLong:     printf("WouldTakeTooLong\n"); break;
            case TimeBudgetExceeded:   printf("TimeBudgetExceeded\n"); break;
            case SampleBudgetExceeded: printf("SampleBudgetExceeded\n"); break;
            case ConditionsChanged:    printf("ConditionsChanged\n"); break;
            case ThresholdElevated:    printf("ThresholdElevated\n"); break;
        }
    }

    printf("\nThresholds:\n");
    printf("  User threshold: %.2f ns\n", result.theta_user_ns);
    printf("  Effective threshold: %.2f ns\n", result.theta_eff_ns);
    printf("  Measurement floor: %.2f ns\n", result.theta_floor_ns);

    /* Interpret the outcome */
    printf("\n");
    switch (result.outcome) {
        case Pass:
            printf("No timing leak detected at this threshold.\n");
            break;
        case Fail:
            printf("TIMING LEAK DETECTED!\n");
            printf("This code should not be used in production.\n");
            break;
        case Inconclusive:
            printf("Could not determine - try increasing time budget.\n");
            break;
        case Unmeasurable:
            printf("Operation too fast to measure reliably.\n");
            break;
    }

    return (result.outcome == Fail) ? 1 : 0;
}
