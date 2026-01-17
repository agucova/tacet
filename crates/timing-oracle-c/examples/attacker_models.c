/**
 * @file attacker_models.c
 * @brief Demonstrates different attacker models and their thresholds.
 *
 * timing-oracle supports multiple threat models, each with a different
 * sensitivity threshold (theta). This example shows how to choose the
 * right model for your use case and how the threshold affects detection.
 *
 * Attacker Models:
 *   SharedHardware   - theta = 0.6 ns  - SGX, containers, hyperthreading
 *   PostQuantum      - theta = 3.3 ns  - Post-quantum crypto protection
 *   AdjacentNetwork  - theta = 100 ns  - LAN, HTTP/2 (Timeless Timing)
 *   RemoteNetwork    - theta = 50 us   - General internet APIs
 *   Research         - theta -> 0      - Detect any difference (not for CI)
 *   Custom           - user-defined    - Specific requirements
 *
 * Build:
 *   make attacker_models
 *
 * Run:
 *   ./attacker_models
 */

#include "common.h"

#define INPUT_SIZE 32

/* Secret for testing */
static uint8_t secret[INPUT_SIZE];

/**
 * @brief Operation with a small timing leak (~50ns).
 *
 * This simulates code with a moderate timing difference that would be:
 * - Detected by SharedHardware, PostQuantum, AdjacentNetwork
 * - Possibly missed by RemoteNetwork (50us threshold)
 */
static void operation_with_small_leak(void *ctx, const uint8_t *input, size_t size) {
    (void)ctx;
    uint8_t acc = 0;

    /* XOR operation (constant-time base) */
    for (size_t i = 0; i < size; i++) {
        acc ^= input[i] ^ secret[i];
    }

    /* Small timing leak: extra work if input has specific property */
    if (input[0] > 0) {
        volatile int dummy = 0;
        for (int i = 0; i < 50; i++) {
            dummy += i;
        }
        DO_NOT_OPTIMIZE(dummy);
    }

    DO_NOT_OPTIMIZE(acc);
}

/* Attacker model descriptions */
static const struct {
    to_attacker_model_t model;
    const char *name;
    const char *description;
    const char *use_case;
} attacker_models[] = {
    {
        TO_ATTACKER_SHARED_HARDWARE,
        "SharedHardware",
        "theta = ~0.6 ns (~2 CPU cycles)",
        "SGX enclaves, containers, hyperthreading attacks"
    },
    {
        TO_ATTACKER_POST_QUANTUM,
        "PostQuantum",
        "theta = ~3.3 ns (~10 cycles)",
        "Post-quantum crypto (KyberSlash-class attacks)"
    },
    {
        TO_ATTACKER_ADJACENT_NETWORK,
        "AdjacentNetwork",
        "theta = 100 ns",
        "LAN/HTTP2 (Timeless Timing Attacks)"
    },
    {
        TO_ATTACKER_REMOTE_NETWORK,
        "RemoteNetwork",
        "theta = 50 us",
        "Public internet APIs"
    },
    {
        TO_ATTACKER_RESEARCH,
        "Research",
        "theta -> 0 (detect any difference)",
        "Academic research (not for CI)"
    },
};

#define NUM_MODELS (sizeof(attacker_models) / sizeof(attacker_models[0]))

int main(void) {
    printf("timing-oracle Attacker Models Example\n");
    printf("=====================================\n\n");

    printf("This example tests the same operation against different attacker models.\n");
    printf("The operation has a small (~50ns) timing leak, demonstrating how\n");
    printf("different thresholds affect detection sensitivity.\n\n");

    /* Initialize secret */
    random_bytes(secret, INPUT_SIZE);

    /* Print model overview */
    printf("Attacker Models Overview:\n");
    printf("-------------------------\n");
    for (size_t i = 0; i < NUM_MODELS; i++) {
        printf("  %-18s %s\n", attacker_models[i].name, attacker_models[i].description);
        printf("  %18s Use: %s\n", "", attacker_models[i].use_case);
        printf("\n");
    }

    /* Test each model */
    printf("Testing operation against each model:\n");
    printf("--------------------------------------\n");

    for (size_t i = 0; i < NUM_MODELS; i++) {
        printf("\n[%zu/%zu] %s (%s)\n",
               i + 1, NUM_MODELS,
               attacker_models[i].name,
               attacker_models[i].description);

        to_config_t config = to_config_default(attacker_models[i].model);
        config.time_budget_secs = 10.0;  /* Quick test */

        to_result_t result = to_test(
            &config,
            INPUT_SIZE,
            standard_generator,
            operation_with_small_leak,
            NULL
        );

        printf("  Result: ");
        print_result_colored(&result);
        printf("  Effect: %.1f ns (95%% CI: [%.1f, %.1f] ns)\n",
               result.effect.shift_ns + result.effect.tail_ns,
               result.effect.ci_low_ns,
               result.effect.ci_high_ns);

        to_result_free(&result);
    }

    /* Custom threshold example */
    printf("\n[Custom] User-defined threshold (25 ns)\n");
    {
        to_config_t config = to_config_default(TO_ATTACKER_CUSTOM);
        config.custom_threshold_ns = 25.0;
        config.time_budget_secs = 10.0;

        to_result_t result = to_test(
            &config,
            INPUT_SIZE,
            standard_generator,
            operation_with_small_leak,
            NULL
        );

        printf("  Result: ");
        print_result_colored(&result);

        to_result_free(&result);
    }

    /* Guidance */
    printf("\n=== Choosing an Attacker Model ===\n\n");
    printf("Choose based on your deployment scenario:\n\n");
    printf("  SharedHardware   - Your code runs in SGX, containers, or shares\n");
    printf("                     hardware with untrusted code.\n\n");
    printf("  PostQuantum      - You're implementing post-quantum crypto and\n");
    printf("                     want to guard against KyberSlash-class attacks.\n\n");
    printf("  AdjacentNetwork  - Your crypto is exposed via LAN or HTTP/2.\n");
    printf("                     This is the default and covers most web apps.\n\n");
    printf("  RemoteNetwork    - Your API is only accessible over the internet\n");
    printf("                     with typical network latency.\n\n");
    printf("  Research         - You want to detect ANY timing difference,\n");
    printf("                     regardless of practical exploitability.\n");
    printf("                     Not recommended for CI (may never pass).\n\n");
    printf("  Custom           - You have specific requirements or want to\n");
    printf("                     match a known attack threshold.\n");

    return 0;
}
