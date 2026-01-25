/**
 * @file test_main.c
 * @brief CMocka test runner for tacet C API tests.
 *
 * This file includes all test groups and runs them in sequence.
 * Each test group is defined in its own file and provides a
 * run_*_tests() function that returns the number of failures.
 */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/tacet.h"

/* External test group runners */
extern int run_config_tests(void);
extern int run_one_shot_tests(void);
extern int run_adaptive_tests(void);
extern int run_error_handling_tests(void);
extern int run_memory_safety_tests(void);
extern int run_known_leaky_tests(void);
extern int run_known_safe_tests(void);

/**
 * Print usage information.
 */
static void print_usage(const char *prog_name) {
    printf("Usage: %s [OPTIONS] [TEST_GROUP...]\n", prog_name);
    printf("\n");
    printf("Run tacet C API tests.\n");
    printf("\n");
    printf("Test groups:\n");
    printf("  config          Configuration tests\n");
    printf("  one_shot        One-shot analysis tests\n");
    printf("  adaptive        Adaptive sampling tests\n");
    printf("  error           Error handling tests\n");
    printf("  memory          Memory safety tests\n");
    printf("  known_leaky     Known leaky operation tests (slow)\n");
    printf("  known_safe      Known safe operation tests (slow)\n");
    printf("  all             Run all tests (default)\n");
    printf("  fast            Run fast tests only (excludes known_leaky, known_safe)\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help      Show this help message\n");
    printf("  -v, --version   Show library version\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                    Run all tests\n", prog_name);
    printf("  %s fast               Run fast tests only\n", prog_name);
    printf("  %s config error       Run config and error tests\n", prog_name);
    printf("  %s known_leaky        Run only known leaky tests\n", prog_name);
}

int main(int argc, char **argv) {
    int total_failures = 0;

    /* Flags for which test groups to run */
    bool run_config = false;
    bool run_one_shot = false;
    bool run_adaptive = false;
    bool run_error = false;
    bool run_memory = false;
    bool run_leaky = false;
    bool run_safe = false;
    bool run_all = false;
    bool run_fast = false;

    /* Parse command line arguments */
    if (argc == 1) {
        /* No arguments: run all tests */
        run_all = true;
    } else {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
                print_usage(argv[0]);
                return 0;
            } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
                printf("tacet version: %s\n", to_version());
                return 0;
            } else if (strcmp(argv[i], "all") == 0) {
                run_all = true;
            } else if (strcmp(argv[i], "fast") == 0) {
                run_fast = true;
            } else if (strcmp(argv[i], "config") == 0) {
                run_config = true;
            } else if (strcmp(argv[i], "one_shot") == 0) {
                run_one_shot = true;
            } else if (strcmp(argv[i], "adaptive") == 0) {
                run_adaptive = true;
            } else if (strcmp(argv[i], "error") == 0) {
                run_error = true;
            } else if (strcmp(argv[i], "memory") == 0) {
                run_memory = true;
            } else if (strcmp(argv[i], "known_leaky") == 0) {
                run_leaky = true;
            } else if (strcmp(argv[i], "known_safe") == 0) {
                run_safe = true;
            } else {
                fprintf(stderr, "Unknown test group: %s\n", argv[i]);
                fprintf(stderr, "Use --help for usage information.\n");
                return 1;
            }
        }
    }

    /* Set flags based on shortcuts */
    if (run_all) {
        run_config = true;
        run_one_shot = true;
        run_adaptive = true;
        run_error = true;
        run_memory = true;
        run_leaky = true;
        run_safe = true;
    } else if (run_fast) {
        run_config = true;
        run_one_shot = true;
        run_adaptive = true;
        run_error = true;
        run_memory = true;
        /* Exclude slow tests */
        run_leaky = false;
        run_safe = false;
    }

    /* Print header */
    printf("========================================\n");
    printf("  tacet C API Test Suite\n");
    printf("  Library version: %s\n", to_version());
    printf("========================================\n\n");

    /* Run selected test groups */
    if (run_config) {
        printf(">>> Running Config Tests...\n");
        total_failures += run_config_tests();
        printf("\n");
    }

    if (run_one_shot) {
        printf(">>> Running One-Shot Analysis Tests...\n");
        total_failures += run_one_shot_tests();
        printf("\n");
    }

    if (run_adaptive) {
        printf(">>> Running Adaptive Sampling Tests...\n");
        total_failures += run_adaptive_tests();
        printf("\n");
    }

    if (run_error) {
        printf(">>> Running Error Handling Tests...\n");
        total_failures += run_error_handling_tests();
        printf("\n");
    }

    if (run_memory) {
        printf(">>> Running Memory Safety Tests...\n");
        total_failures += run_memory_safety_tests();
        printf("\n");
    }

    if (run_leaky) {
        printf(">>> Running Known Leaky Tests (this may take a while)...\n");
        total_failures += run_known_leaky_tests();
        printf("\n");
    }

    if (run_safe) {
        printf(">>> Running Known Safe Tests (this may take a while)...\n");
        total_failures += run_known_safe_tests();
        printf("\n");
    }

    /* Print summary */
    printf("========================================\n");
    if (total_failures == 0) {
        printf("  All tests PASSED!\n");
    } else {
        printf("  %d test(s) FAILED\n", total_failures);
    }
    printf("========================================\n");

    return total_failures;
}
