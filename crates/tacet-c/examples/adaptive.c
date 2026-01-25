/**
 * @file adaptive.c
 * @brief Adaptive sampling loop example using tacet C API.
 *
 * This example demonstrates the adaptive sampling workflow:
 * 1. Collect calibration samples
 * 2. Initialize calibration and state
 * 3. Loop: collect batches, call to_step(), check for decision
 * 4. Clean up
 *
 * This approach is more efficient than one-shot analysis as it
 * stops as soon as a statistically significant decision is reached.
 *
 * Build and run (from repo root):
 *   just bindings c-run-adaptive
 *
 * Or manually:
 *   just bindings c-example-adaptive
 *   ./crates/tacet-c/examples/adaptive
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "../include/tacet.h"

/* Calibration samples per class */
#define CALIBRATION_SAMPLES 5000

/* Batch size for adaptive loop */
#define BATCH_SIZE 1000

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

/* Get current time in seconds */
static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

/* Secret key stored in the "application" */
static uint8_t secret_key[DATA_SIZE];

/**
 * @brief A comparison function with a timing side channel.
 */
static bool leaky_compare(const uint8_t *a, const uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return false;
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
 * @brief Collect a batch of timing samples.
 *
 * Uses interleaved sampling for better statistical properties.
 */
static void collect_batch(
    uint64_t *baseline_out,
    uint64_t *sample_out,
    size_t count
) {
    uint8_t input[DATA_SIZE];

    for (size_t i = 0; i < count; i++) {
        /* Baseline sample (all zeros) */
        memset(input, 0, DATA_SIZE);
        uint64_t start = read_timer();
        bool result = leaky_compare(input, secret_key, DATA_SIZE);
        uint64_t end = read_timer();
        DO_NOT_OPTIMIZE(result);
        baseline_out[i] = end - start;

        /* Sample sample (random data) */
        random_bytes(input, DATA_SIZE);
        start = read_timer();
        result = leaky_compare(input, secret_key, DATA_SIZE);
        end = read_timer();
        DO_NOT_OPTIMIZE(result);
        sample_out[i] = end - start;
    }
}

int main(void) {
    printf("tacet adaptive sampling example\n");
    printf("Library version: %s\n\n", to_version());

    /* Seed random number generator */
    srand((unsigned int)time(NULL));

    /* Initialize secret key with random data */
    random_bytes(secret_key, DATA_SIZE);

    /* Create configuration */
    struct ToConfig config = to_config_adjacent_network();
    config.time_budget_secs = 30.0;

    printf("Attacker model: Adjacent Network (theta=100ns)\n");
    printf("Time budget: %.0f seconds\n\n", config.time_budget_secs);

    /* Allocate sample arrays */
    uint64_t *baseline = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *sample = malloc(CALIBRATION_SAMPLES * sizeof(uint64_t));
    uint64_t *batch_baseline = malloc(BATCH_SIZE * sizeof(uint64_t));
    uint64_t *batch_sample = malloc(BATCH_SIZE * sizeof(uint64_t));

    if (!baseline || !sample || !batch_baseline || !batch_sample) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    /* Phase 1: Collect calibration samples */
    printf("Phase 1: Collecting %d calibration samples per class...\n",
           CALIBRATION_SAMPLES);
    collect_batch(baseline, sample, CALIBRATION_SAMPLES);

    /* Run calibration */
    enum ToError err;
    struct ToCalibration *cal = to_calibrate(
        baseline, sample, CALIBRATION_SAMPLES, &config, &err
    );

    if (!cal) {
        fprintf(stderr, "Calibration failed with error code: %d\n", err);
        goto cleanup;
    }

    printf("Calibration complete.\n\n");

    /* Create adaptive state */
    struct ToState *state = to_state_new();
    if (!state) {
        fprintf(stderr, "Failed to create state\n");
        to_calibration_free(cal);
        goto cleanup;
    }

    /* Phase 2: Adaptive sampling loop */
    printf("Phase 2: Running adaptive sampling loop...\n");
    double start_time = get_time();
    int iteration = 0;

    while (1) {
        /* Collect a batch of new samples */
        collect_batch(batch_baseline, batch_sample, BATCH_SIZE);

        /* Run one adaptive step */
        double elapsed = get_time() - start_time;
        struct ToStepResult step_result;

        err = to_step(
            cal, state,
            batch_baseline, batch_sample, BATCH_SIZE,
            &config, elapsed,
            &step_result
        );

        if (err != Ok) {
            fprintf(stderr, "Step failed with error code: %d\n", err);
            break;
        }

        iteration++;

        /* Print progress */
        printf("\r  Iteration %d: P(leak)=%.1f%%, samples=%llu",
               iteration,
               step_result.leak_probability * 100.0,
               (unsigned long long)step_result.samples_used);
        fflush(stdout);

        /* Check for decision */
        if (step_result.has_decision) {
            printf("\n\nDecision reached after %.2f seconds!\n\n", elapsed);

            /* Display results */
            struct ToResult *result = &step_result.result;

            printf("=== Analysis Results ===\n\n");
            printf("Outcome: ");
            switch (result->outcome) {
                case Pass:        printf("PASS\n"); break;
                case Fail:        printf("FAIL\n"); break;
                case Inconclusive: printf("INCONCLUSIVE\n"); break;
                case Unmeasurable: printf("UNMEASURABLE\n"); break;
            }

            printf("Leak probability: %.2f%%\n", result->leak_probability * 100.0);
            printf("Effect estimate: %.2f ns (shift) + %.2f ns (tail)\n",
                   result->effect.shift_ns, result->effect.tail_ns);
            printf("Samples used: %llu per class\n",
                   (unsigned long long)result->samples_used);
            printf("Elapsed time: %.2f seconds\n", result->elapsed_secs);

            if (result->outcome == Fail) {
                printf("\nExploitability: ");
                switch (result->exploitability) {
                    case SharedHardwareOnly: printf("SharedHardwareOnly\n"); break;
                    case Http2Multiplexing:  printf("Http2Multiplexing\n"); break;
                    case StandardRemote:     printf("StandardRemote\n"); break;
                    case ObviousLeak:        printf("ObviousLeak\n"); break;
                }
            }

            if (result->outcome == Inconclusive) {
                printf("\nReason: ");
                switch (result->inconclusive_reason) {
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

            err = (result->outcome == Fail) ? 1 : 0;
            break;
        }

        /* Check time budget */
        if (elapsed > config.time_budget_secs) {
            printf("\n\nTime budget exceeded.\n");
            err = 0;
            break;
        }
    }

    /* Clean up */
    to_state_free(state);
    to_calibration_free(cal);

cleanup:
    free(baseline);
    free(sample);
    free(batch_baseline);
    free(batch_sample);

    return err;
}
