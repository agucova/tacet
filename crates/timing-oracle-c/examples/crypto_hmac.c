/**
 * @file crypto_hmac.c
 * @brief Real-world crypto testing: HMAC verification timing.
 *
 * This example tests HMAC-SHA256 verification for timing side channels.
 * It demonstrates how to test actual cryptographic operations using
 * timing-oracle with OpenSSL.
 *
 * The security issue this detects:
 * - Comparing HMAC tags byte-by-byte with early exit leaks timing
 * - Attackers can forge valid MACs by trying bytes one at a time
 * - This is a real vulnerability (CVE-2013-0169 "Lucky Thirteen")
 *
 * Requirements:
 * - OpenSSL development libraries
 * - Build with: make WITH_OPENSSL=1 crypto_hmac
 *
 * Build:
 *   make WITH_OPENSSL=1 crypto_hmac
 *
 * Run:
 *   ./crypto_hmac
 */

#ifdef WITH_OPENSSL

#include "common.h"
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#define KEY_SIZE 32
#define MSG_SIZE 64
#define TAG_SIZE 32  /* SHA-256 output */

/* Context for HMAC tests */
typedef struct {
    uint8_t key[KEY_SIZE];
    uint8_t message[MSG_SIZE];
    uint8_t expected_tag[TAG_SIZE];
    bool (*verify_fn)(const uint8_t *, const uint8_t *, size_t);
} hmac_context_t;

/* ============================================================================
 * Comparison functions
 * ============================================================================ */

/**
 * @brief INSECURE: Early-exit comparison (timing leak).
 */
static bool leaky_verify(const uint8_t *computed, const uint8_t *expected, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (computed[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief SECURE: Constant-time comparison.
 */
static bool ct_verify(const uint8_t *computed, const uint8_t *expected, size_t len) {
    uint8_t acc = 0;
    for (size_t i = 0; i < len; i++) {
        acc |= computed[i] ^ expected[i];
    }
    return acc == 0;
}

/* ============================================================================
 * Test callbacks
 * ============================================================================ */

/**
 * @brief Generator: creates HMAC tags to verify.
 *
 * Baseline: compute correct HMAC (should match)
 * Sample: use random bytes as "tag" (will not match)
 *
 * This tests whether verification timing differs based on where
 * the mismatch occurs.
 */
static void hmac_generator(void *ctx, bool is_baseline, uint8_t *output, size_t size) {
    hmac_context_t *hctx = (hmac_context_t *)ctx;
    (void)size;  /* We know size == TAG_SIZE */

    if (is_baseline) {
        /* Correct tag - will match */
        memcpy(output, hctx->expected_tag, TAG_SIZE);
    } else {
        /* Random "tag" - will mismatch at various positions */
        RAND_bytes(output, TAG_SIZE);
    }
}

/**
 * @brief Operation: verify HMAC tag.
 *
 * 1. Compute HMAC of the message
 * 2. Compare with provided tag (using selected comparison function)
 */
static void hmac_operation(void *ctx, const uint8_t *input, size_t size) {
    hmac_context_t *hctx = (hmac_context_t *)ctx;
    (void)size;

    /* Compute HMAC */
    uint8_t computed_tag[TAG_SIZE];
    unsigned int len = TAG_SIZE;
    HMAC(EVP_sha256(),
         hctx->key, KEY_SIZE,
         hctx->message, MSG_SIZE,
         computed_tag, &len);

    /* Verify using selected comparison */
    bool result = hctx->verify_fn(computed_tag, input, TAG_SIZE);
    DO_NOT_OPTIMIZE(result);
}

/**
 * @brief Run timing test on HMAC verification.
 */
static to_outcome_t test_hmac_verify(
    const char *name,
    hmac_context_t *ctx,
    bool (*verify_fn)(const uint8_t *, const uint8_t *, size_t)
) {
    ctx->verify_fn = verify_fn;

    printf("\n--- Testing: %s ---\n", name);

    to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
    config.time_budget_secs = 15.0;

    to_result_t result = to_test(&config, TAG_SIZE, hmac_generator, hmac_operation, ctx);

    print_result_colored(&result);

    if (result.outcome == TO_OUTCOME_FAIL) {
        printf("  Exploitability: %s\n", to_exploitability_str(result.exploitability));
        printf("  Effect: %.1f ns (pattern: %s)\n",
               result.effect.shift_ns + result.effect.tail_ns,
               to_effect_pattern_str(result.effect.pattern));
    }

    to_outcome_t outcome = result.outcome;
    to_result_free(&result);
    return outcome;
}

int main(void) {
    printf("timing-oracle HMAC Verification Example\n");
    printf("=======================================\n\n");

    printf("This example tests HMAC-SHA256 verification for timing leaks.\n");
    printf("A leaky comparison allows attackers to forge MACs byte-by-byte.\n\n");

    /* Initialize OpenSSL */
    #if OPENSSL_VERSION_NUMBER < 0x10100000L
    OpenSSL_add_all_algorithms();
    #endif

    /* Set up context */
    hmac_context_t ctx;
    RAND_bytes(ctx.key, KEY_SIZE);
    RAND_bytes(ctx.message, MSG_SIZE);

    /* Compute expected HMAC */
    unsigned int len = TAG_SIZE;
    HMAC(EVP_sha256(),
         ctx.key, KEY_SIZE,
         ctx.message, MSG_SIZE,
         ctx.expected_tag, &len);

    printf("Key size: %d bytes\n", KEY_SIZE);
    printf("Message size: %d bytes\n", MSG_SIZE);
    printf("HMAC-SHA256 tag size: %d bytes\n\n", TAG_SIZE);

    /* Test leaky verification */
    to_outcome_t leaky_outcome = test_hmac_verify(
        "INSECURE: memcmp-style (early exit)",
        &ctx,
        leaky_verify
    );

    /* Test constant-time verification */
    to_outcome_t ct_outcome = test_hmac_verify(
        "SECURE: constant-time comparison",
        &ctx,
        ct_verify
    );

    /* Summary */
    printf("\n=== Summary ===\n");
    printf("Leaky verification:   %s\n",
           leaky_outcome == TO_OUTCOME_FAIL ? "FAIL (vulnerable)" :
           leaky_outcome == TO_OUTCOME_PASS ? "PASS (unexpected!)" : "INCONCLUSIVE");
    printf("Constant-time verification: %s\n",
           ct_outcome == TO_OUTCOME_PASS ? "PASS (secure)" :
           ct_outcome == TO_OUTCOME_FAIL ? "FAIL (unexpected!)" : "INCONCLUSIVE");

    printf("\n=== Security Notes ===\n");
    printf("- Always use constant-time comparison for MACs/signatures\n");
    printf("- OpenSSL's CRYPTO_memcmp() is constant-time\n");
    printf("- Never use memcmp() for cryptographic comparisons\n");
    printf("- This vulnerability enables Lucky Thirteen style attacks\n");

    return 0;
}

#else /* !WITH_OPENSSL */

#include <stdio.h>

int main(void) {
    printf("timing-oracle HMAC Example\n");
    printf("==========================\n\n");
    printf("This example requires OpenSSL.\n");
    printf("Build with: make WITH_OPENSSL=1 crypto_hmac\n");
    return 0;
}

#endif /* WITH_OPENSSL */
