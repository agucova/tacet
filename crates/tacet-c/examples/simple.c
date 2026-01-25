/**
 * @file simple.c
 * @brief Basic tacet usage example using one-shot analysis.
 *
 * This example demonstrates the minimal workflow for detecting timing
 * side channels using the to_analyze() function with pre-collected samples.
 *
 * The C code handles:
 * - High-resolution timing using rdtsc/cntvct
 * - Sample collection with interleaving
 * - Input generation
 *
 * Then calls Rust for statistical analysis via to_analyze().
 *
 * Build and run (from repo root):
 *   just bindings c-run-simple
 *
 * Or manually:
 *   just bindings c-example-simple
 *   ./crates/tacet-c/examples/simple
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#include "../include/tacet.h"

/* Number of samples per class */
#define NUM_SAMPLES 10000

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
 * @brief Fill buffer with random bytes.
 */
static void random_bytes(uint8_t *buf, size_t size) {
    for (size_t i = 0; i < size; i++) {
        buf[i] = (uint8_t)rand();
    }
}

int main(void) {
    printf("tacet simple example (one-shot analysis)\n");
    printf("Library version: %s\n\n", to_version());

    /* Seed random number generator */
    srand((unsigned int)time(NULL));

    /* Initialize secret key with random data */
    random_bytes(secret_key, DATA_SIZE);

    /* Allocate sample arrays */
    uint64_t *baseline_samples = malloc(NUM_SAMPLES * sizeof(uint64_t));
    uint64_t *sample_samples = malloc(NUM_SAMPLES * sizeof(uint64_t));
    uint8_t input[DATA_SIZE];

    if (!baseline_samples || !sample_samples) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    printf("Collecting %d samples per class...\n", NUM_SAMPLES);

    /* Collect baseline samples (all-zero input) */
    memset(input, 0, DATA_SIZE);
    for (size_t i = 0; i < NUM_SAMPLES; i++) {
        uint64_t start = read_timer();
        bool result = leaky_compare(input, secret_key, DATA_SIZE);
        uint64_t end = read_timer();
        DO_NOT_OPTIMIZE(result);
        baseline_samples[i] = end - start;
    }

    /* Collect sample samples (random input) */
    for (size_t i = 0; i < NUM_SAMPLES; i++) {
        random_bytes(input, DATA_SIZE);
        uint64_t start = read_timer();
        bool result = leaky_compare(input, secret_key, DATA_SIZE);
        uint64_t end = read_timer();
        DO_NOT_OPTIMIZE(result);
        sample_samples[i] = end - start;
    }

    printf("Samples collected. Running analysis...\n\n");

    /* Create configuration for adjacent network attacker (100ns threshold) */
    struct ToConfig config = to_config_adjacent_network();

    /* Run analysis */
    struct ToResult result;
    enum ToError err = to_analyze(
        baseline_samples,
        sample_samples,
        NUM_SAMPLES,
        &config,
        &result
    );

    if (err != Ok) {
        fprintf(stderr, "Analysis failed with error code: %d\n", err);
        free(baseline_samples);
        free(sample_samples);
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
    printf("Samples used: %llu\n", (unsigned long long)result.samples_used);
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
            printf("Could not determine - try with more samples.\n");
            break;
        case Unmeasurable:
            printf("Operation too fast to measure reliably.\n");
            break;
    }

    /* Clean up */
    free(baseline_samples);
    free(sample_samples);

    return (result.outcome == Fail) ? 1 : 0;
}
