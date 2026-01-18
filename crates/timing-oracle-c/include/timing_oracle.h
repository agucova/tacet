/**
 * @file timing_oracle.h
 * @brief C API for timing-oracle: Statistical timing side-channel detection
 *
 * This library provides Bayesian statistical testing for detecting timing side
 * channels in cryptographic code. It wraps the timing-oracle-core Rust library.
 *
 * @section Usage
 *
 * 1. Create a configuration with `to_config_default()`
 * 2. Implement generator and operation callbacks
 * 3. Call `to_test()` with your callbacks
 * 4. Handle the result based on `outcome`
 * 5. Free the result with `to_result_free()`
 *
 * @section Example
 *
 * @code{.c}
 * void my_generator(void *ctx, bool is_baseline, uint8_t *output, size_t size) {
 *     if (is_baseline) {
 *         memset(output, 0, size);  // All zeros for baseline
 *     } else {
 *         arc4random_buf(output, size);  // Random for sample
 *     }
 * }
 *
 * void my_operation(void *ctx, const uint8_t *input, size_t size) {
 *     my_crypto_function(input, size);
 * }
 *
 * int main(void) {
 *     to_config_t config = to_config_default(TO_ATTACKER_ADJACENT_NETWORK);
 *     config.time_budget_secs = 30.0;
 *
 *     to_result_t result = to_test(&config, 32, my_generator, my_operation, NULL);
 *
 *     if (result.outcome == TO_OUTCOME_FAIL) {
 *         printf("Timing leak detected! P(leak)=%.1f%%\n",
 *                result.leak_probability * 100.0);
 *     }
 *
 *     to_result_free(&result);
 *     return result.outcome == TO_OUTCOME_FAIL ? 1 : 0;
 * }
 * @endcode
 */

#ifndef TIMING_ORACLE_H
#define TIMING_ORACLE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Version
 * ============================================================================ */

#define TO_VERSION_MAJOR 0
#define TO_VERSION_MINOR 1
#define TO_VERSION_PATCH 0

/* ============================================================================
 * Attacker Models (from docs/spec.md Section 4.2)
 *
 * Choose based on your threat model:
 * - SharedHardware: Co-resident attacker (SGX, containers, hyperthreading)
 * - PostQuantum: Post-quantum crypto (KyberSlash-class attacks)
 * - AdjacentNetwork: LAN or HTTP/2 (Timeless Timing Attacks)
 * - RemoteNetwork: General internet
 * - Research: Detect any difference (not for CI)
 * - Custom: User-specified threshold
 * ============================================================================ */

typedef enum {
    /** theta = 0.6 ns (~2 cycles @ 3GHz) - SGX, cross-VM, containers */
    TO_ATTACKER_SHARED_HARDWARE = 0,
    /** theta = 3.3 ns (~10 cycles) - Post-quantum crypto */
    TO_ATTACKER_POST_QUANTUM = 1,
    /** theta = 100 ns - LAN, HTTP/2 (Timeless Timing Attacks) */
    TO_ATTACKER_ADJACENT_NETWORK = 2,
    /** theta = 50 us - General internet */
    TO_ATTACKER_REMOTE_NETWORK = 3,
    /** theta -> 0 - Detect any difference (not for CI) */
    TO_ATTACKER_RESEARCH = 4,
    /** User-specified threshold via custom_threshold_ns */
    TO_ATTACKER_CUSTOM = 5
} to_attacker_model_t;

/* ============================================================================
 * Timer Preference
 *
 * Controls which timer implementation to use for measurements:
 * - Auto: Try PMU first (perf/kperf), fall back to standard (rdtsc/cntvct)
 * - Standard: Always use standard timer (rdtsc on x86, cntvct on ARM)
 * - PreferPmu: Use PMU timer or fail if unavailable
 *
 * PMU timers provide ~0.3ns resolution but require elevated privileges:
 * - Linux: sudo, CAP_PERFMON, or perf_event_paranoid <= 2
 * - macOS: sudo (kperf requires root)
 * ============================================================================ */

typedef enum {
    /** Try PMU first (perf/kperf), fall back to standard timer */
    TO_TIMER_AUTO = 0,
    /** Always use standard timer (rdtsc/cntvct_el0) */
    TO_TIMER_STANDARD = 1,
    /** Require PMU timer, fail if unavailable */
    TO_TIMER_PREFER_PMU = 2
} to_timer_pref_t;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Configuration for the timing test.
 *
 * Use `to_config_default()` to create a configuration with sensible defaults,
 * then override specific fields as needed.
 */
typedef struct {
    /** Attacker model to use (determines threshold theta). */
    to_attacker_model_t attacker_model;
    /** Custom threshold in nanoseconds (only used if attacker_model == CUSTOM). */
    double custom_threshold_ns;
    /** Maximum samples per class. 0 = default (100,000). */
    size_t max_samples;
    /** Time budget in seconds. 0 = default (30s). */
    double time_budget_secs;
    /** Pass threshold for leak probability. Default: 0.05. */
    double pass_threshold;
    /** Fail threshold for leak probability. Default: 0.95. */
    double fail_threshold;
    /** Random seed. 0 = use system entropy. */
    uint64_t seed;
    /** Timer preference. Default: TO_TIMER_AUTO. */
    to_timer_pref_t timer_preference;
    /** Calibration samples per class. 0 = default (5,000). */
    size_t calibration_samples;
    /** Batch size for adaptive sampling. 0 = default (1,000). */
    size_t batch_size;
    /** Bootstrap iterations for covariance estimation. 0 = default (2,000). */
    size_t bootstrap_iterations;
} to_config_t;

/**
 * @brief Create a default configuration for the given attacker model.
 *
 * @param model The attacker model to use
 * @return Configuration with sensible defaults
 */
to_config_t to_config_default(to_attacker_model_t model);

/* ============================================================================
 * Callbacks
 * ============================================================================ */

/**
 * @brief Callback type for generating input data.
 *
 * This function is called to generate input data for each measurement.
 * The input should differ based on `is_baseline`:
 * - When `is_baseline` is true: generate baseline class input (typically all zeros)
 * - When `is_baseline` is false: generate sample class input (typically random)
 *
 * This function is called OUTSIDE the timed region, so its overhead does not
 * affect measurements.
 *
 * @param context User-provided context pointer (passed through from to_test)
 * @param is_baseline true for baseline class, false for sample class
 * @param output Buffer to write generated input into
 * @param output_size Size of the output buffer in bytes
 */
typedef void (*to_input_generator_fn)(
    void *context,
    bool is_baseline,
    uint8_t *output,
    size_t output_size
);

/**
 * @brief Callback type for the operation to time.
 *
 * This function is the operation being tested for timing side channels.
 * It is called inside the timed region with input generated by the generator.
 *
 * @param context User-provided context pointer (passed through from to_test)
 * @param input Input data buffer (generated by generator callback)
 * @param input_size Size of input data in bytes
 */
typedef void (*to_operation_fn)(
    void *context,
    const uint8_t *input,
    size_t input_size
);

/* ============================================================================
 * Result Types (from docs/spec.md Section 4.1)
 * ============================================================================ */

/**
 * @brief Test outcome.
 *
 * Four possible outcomes:
 * - PASS: No timing leak detected within threshold theta
 * - FAIL: Timing leak detected exceeding threshold theta
 * - INCONCLUSIVE: Could not reach a decision (see inconclusive_reason)
 * - UNMEASURABLE: Operation too fast to measure reliably
 */
typedef enum {
    /** No timing leak detected within threshold theta. */
    TO_OUTCOME_PASS = 0,
    /** Timing leak detected exceeding threshold theta. */
    TO_OUTCOME_FAIL = 1,
    /** Could not reach a decision (quality gate triggered). */
    TO_OUTCOME_INCONCLUSIVE = 2,
    /** Operation too fast to measure reliably. */
    TO_OUTCOME_UNMEASURABLE = 3,
    /** Research mode result (spec Section 3.6) - returned when attacker_model is Research. */
    TO_OUTCOME_RESEARCH = 4
} to_outcome_t;

/**
 * @brief Research mode status (spec Section 3.6).
 *
 * Research mode (AttackerModel::Research) doesn't make Pass/Fail decisions.
 * Instead, it characterizes the timing behavior with respect to the measurement floor.
 * Only valid when outcome == TO_OUTCOME_RESEARCH.
 */
typedef enum {
    /** CI clearly above theta_floor - timing difference detected. */
    TO_RESEARCH_EFFECT_DETECTED = 0,
    /** CI clearly below theta_floor - no timing difference above noise. */
    TO_RESEARCH_NO_EFFECT_DETECTED = 1,
    /** Hit timer resolution limit; theta_floor is as good as it gets. */
    TO_RESEARCH_RESOLUTION_LIMIT_REACHED = 2,
    /** Data quality issue detected (see inconclusive_reason for details). */
    TO_RESEARCH_QUALITY_ISSUE = 3,
    /** Ran out of time/samples before reaching conclusion. */
    TO_RESEARCH_BUDGET_EXHAUSTED = 4
} to_research_status_t;

/**
 * @brief Reason for inconclusive result.
 *
 * Only valid when outcome == TO_OUTCOME_INCONCLUSIVE.
 */
typedef enum {
    /** Posterior approximately equals prior after calibration; data not informative. */
    TO_INCONCLUSIVE_DATA_TOO_NOISY = 0,
    /** Posterior stopped updating despite new data. */
    TO_INCONCLUSIVE_NOT_LEARNING = 1,
    /** Estimated time to decision exceeds budget. */
    TO_INCONCLUSIVE_WOULD_TAKE_TOO_LONG = 2,
    /** Time budget exhausted. */
    TO_INCONCLUSIVE_TIME_BUDGET_EXCEEDED = 3,
    /** Sample limit reached. */
    TO_INCONCLUSIVE_SAMPLE_BUDGET_EXCEEDED = 4,
    /** Measurement conditions changed during test. */
    TO_INCONCLUSIVE_CONDITIONS_CHANGED = 5,
    /** Requested threshold is below measurement floor (v4.1). */
    TO_INCONCLUSIVE_THRESHOLD_UNACHIEVABLE = 6,
    /** Model doesn't fit the data - 2D model insufficient (v4.1). */
    TO_INCONCLUSIVE_MODEL_MISMATCH = 7
} to_inconclusive_reason_t;

/**
 * @brief Pattern of timing effect.
 *
 * The library decomposes timing differences into:
 * - Uniform shift: Same difference across all quantiles
 * - Tail effect: Difference concentrated in upper quantiles
 * - Mixed: Both components present
 */
typedef enum {
    /** Uniform shift across all quantiles. */
    TO_EFFECT_UNIFORM_SHIFT = 0,
    /** Effect concentrated in tails (upper quantiles). */
    TO_EFFECT_TAIL_EFFECT = 1,
    /** Both shift and tail components present. */
    TO_EFFECT_MIXED = 2,
    /** Cannot determine pattern. */
    TO_EFFECT_INDETERMINATE = 3
} to_effect_pattern_t;

/**
 * @brief Exploitability assessment (for Fail outcomes).
 *
 * Based on Timeless Timing Attacks (Van Goethem et al., 2020) research.
 */
typedef enum {
    /** < 10 ns - Requires shared hardware (SGX, containers) to exploit. */
    TO_EXPLOIT_SHARED_HARDWARE_ONLY = 0,
    /** 10-100 ns - Exploitable via HTTP/2 request multiplexing. */
    TO_EXPLOIT_HTTP2_MULTIPLEXING = 1,
    /** 100 ns - 10 us - Exploitable with standard remote timing. */
    TO_EXPLOIT_STANDARD_REMOTE = 2,
    /** > 10 us - Obvious leak, trivially exploitable. */
    TO_EXPLOIT_OBVIOUS_LEAK = 3
} to_exploitability_t;

/**
 * @brief Measurement quality assessment.
 *
 * Based on the Minimum Detectable Effect (MDE).
 */
typedef enum {
    /** MDE < 5 ns - Excellent measurement precision. */
    TO_QUALITY_EXCELLENT = 0,
    /** MDE 5-20 ns - Good precision. */
    TO_QUALITY_GOOD = 1,
    /** MDE 20-100 ns - Poor precision. */
    TO_QUALITY_POOR = 2,
    /** MDE > 100 ns - Too noisy for reliable detection. */
    TO_QUALITY_TOO_NOISY = 3
} to_quality_t;

/**
 * @brief Effect size estimate.
 *
 * Decomposes the timing difference into shift and tail components.
 */
typedef struct {
    /** Uniform shift component in nanoseconds. */
    double shift_ns;
    /** Tail effect component in nanoseconds. */
    double tail_ns;
    /** 95% credible interval lower bound. */
    double ci_low_ns;
    /** 95% credible interval upper bound. */
    double ci_high_ns;
    /** Pattern of the effect. */
    to_effect_pattern_t pattern;
    /** Whether there is a projection mismatch caveat. When true, the 2D effect
     * summary may not fully capture the timing difference. */
    bool has_interpretation_caveat;
    /** Top quantile indices (0-8) that contribute most to mismatch.
     * Only valid when has_interpretation_caveat is true. Set to 255 for unused. */
    uint8_t top_quantiles[3];
} to_effect_t;

/** Maximum number of warnings in diagnostics. */
#define TO_MAX_WARNINGS 8
/** Maximum warning message length. */
#define TO_MAX_WARNING_LEN 128

/**
 * @brief Detailed diagnostics for the test result (spec Section 2.8).
 *
 * Contains measurement quality metrics, calibration info, and diagnostic flags.
 */
typedef struct {
    /** Block length used for bootstrap resampling. */
    size_t dependence_length;
    /** Effective sample size accounting for autocorrelation. */
    size_t effective_sample_size;
    /** Ratio of post-test variance to calibration variance.
     * Values outside [0.5, 2.0] indicate drift. */
    double stationarity_ratio;
    /** Whether stationarity check passed. */
    bool stationarity_ok;
    /** Projection mismatch Q statistic. High values indicate 2D summary is approximate. */
    double projection_mismatch_q;
    /** Threshold for projection mismatch (calibrated at 99th percentile). */
    double projection_mismatch_threshold;
    /** Whether projection mismatch is acceptable. */
    bool projection_mismatch_ok;
    /** Outlier rate for baseline class (fraction filtered). */
    double outlier_rate_baseline;
    /** Outlier rate for sample class (fraction filtered). */
    double outlier_rate_sample;
    /** Whether outlier rates are balanced between classes. */
    bool outlier_asymmetry_ok;
    /** Whether discrete mode was used (low timer resolution). */
    bool discrete_mode;
    /** Fraction of duplicate timing values. */
    double duplicate_fraction;
    /** Whether preflight checks passed. */
    bool preflight_ok;
    /** Number of samples used in calibration phase. */
    size_t calibration_samples;
    /** Total time spent in seconds. */
    double total_time_secs;
    /** RNG seed used for bootstrap (0 if random). */
    uint64_t seed;
    /** User-specified threshold (theta_user) in nanoseconds. */
    double theta_user;
    /** Effective threshold (theta_eff) in nanoseconds.
     * May be higher than theta_user due to measurement floor. */
    double theta_eff;
    /** Measurement floor (theta_floor) in nanoseconds.
     * Minimum detectable effect given current noise. */
    double theta_floor;
    /** Number of warnings. */
    size_t warning_count;
    /** Warning messages (null-terminated strings). */
    char warnings[TO_MAX_WARNINGS][TO_MAX_WARNING_LEN];
} to_diagnostics_t;

/**
 * @brief Test result.
 *
 * Contains the outcome and all diagnostic information.
 * Must be freed with `to_result_free()` after use.
 */
typedef struct {
    /** Test outcome. */
    to_outcome_t outcome;

    /** Leak probability: P(max_k |(X*beta)_k| > theta | data). */
    double leak_probability;

    /** Effect size estimate. */
    to_effect_t effect;

    /** Measurement quality. */
    to_quality_t quality;

    /** Number of samples used per class. */
    size_t samples_used;

    /** Time spent in seconds. */
    double elapsed_secs;

    /** Exploitability (only valid if outcome == FAIL). */
    to_exploitability_t exploitability;

    /** Inconclusive reason (only valid if outcome == INCONCLUSIVE). */
    to_inconclusive_reason_t inconclusive_reason;

    /** Measured operation time in ns (only valid if outcome == UNMEASURABLE). */
    double operation_ns;

    /** Timer resolution in nanoseconds. */
    double timer_resolution_ns;

    /** Recommendation string (owned, must be freed via to_result_free; NULL if not applicable). */
    const char *recommendation;

    /** Timer name (static string, do not free). */
    const char *timer_name;

    /** Platform description (static string, do not free). */
    const char *platform;

    /** Whether adaptive batching was used. */
    bool adaptive_batching_used;

    /** Detailed diagnostics (spec Section 2.8). */
    to_diagnostics_t diagnostics;

    /* Research mode fields (only valid if outcome == TO_OUTCOME_RESEARCH) */

    /** Research mode status (only valid if outcome == TO_OUTCOME_RESEARCH). */
    to_research_status_t research_status;

    /** Maximum effect across quantiles in nanoseconds (only valid if outcome == TO_OUTCOME_RESEARCH). */
    double max_effect_ns;

    /** 95% CI lower bound for maximum effect (only valid if outcome == TO_OUTCOME_RESEARCH). */
    double max_effect_ci_low;

    /** 95% CI upper bound for maximum effect (only valid if outcome == TO_OUTCOME_RESEARCH). */
    double max_effect_ci_high;

    /** Whether the effect is detectable: CI lower bound > theta_floor (only valid if outcome == TO_OUTCOME_RESEARCH). */
    bool research_detectable;

    /** Whether model mismatch was detected in research mode (only valid if outcome == TO_OUTCOME_RESEARCH). */
    bool research_model_mismatch;
} to_result_t;

/* ============================================================================
 * Main API
 * ============================================================================ */

/**
 * @brief Run a timing test.
 *
 * This is the main entry point for timing side-channel detection.
 *
 * The test:
 * 1. Collects timing samples using interleaved baseline/sample measurements
 * 2. Computes quantile differences (9 deciles)
 * 3. Uses Bayesian inference to compute P(timing leak > threshold)
 * 4. Returns Pass, Fail, Inconclusive, or Unmeasurable
 *
 * @param config Test configuration (NULL for defaults)
 * @param input_size Size of input data in bytes
 * @param generator Callback to generate input data
 * @param operation Callback for the operation to time
 * @param context User context passed to callbacks (can be NULL)
 * @return Test result (must be freed with to_result_free)
 */
to_result_t to_test(
    const to_config_t *config,
    size_t input_size,
    to_input_generator_fn generator,
    to_operation_fn operation,
    void *context
);

/**
 * @brief Run a timing test with caller-provided time tracking (no_std compatible).
 *
 * This API is available in both standard and no_std builds. The caller is
 * responsible for tracking elapsed time and providing it via the `elapsed_secs`
 * pointer. The library will read this value whenever it needs the current time.
 *
 * For most use cases, prefer `to_test()` which handles time tracking automatically.
 * Use this function in environments without std (e.g., SGX enclaves, embedded).
 *
 * @param config Test configuration (NULL for defaults)
 * @param input_size Size of input data in bytes
 * @param generator Callback to generate input data
 * @param operation Callback for the operation to time
 * @param context User context passed to callbacks (can be NULL)
 * @param elapsed_secs Pointer to elapsed time in seconds (must remain valid during call)
 * @return Test result (must be freed with to_result_free)
 *
 * @note For the adaptive loop to work correctly, `elapsed_secs` should be updated
 *       before calling. In practice, since this is a single blocking call, providing
 *       the initial elapsed time (usually 0.0) is typically sufficient.
 */
to_result_t to_test_with_time(
    const to_config_t *config,
    size_t input_size,
    to_input_generator_fn generator,
    to_operation_fn operation,
    void *context,
    const double *elapsed_secs
);

/**
 * @brief Free a result's owned resources.
 *
 * Call this when done with a result to free any allocated memory.
 * Safe to call multiple times on the same result.
 *
 * @param result Pointer to result to free (can be NULL)
 */
void to_result_free(to_result_t *result);

/* ============================================================================
 * String Conversion Utilities
 * ============================================================================ */

/**
 * @brief Get string representation of outcome.
 * @param outcome The outcome value
 * @return Static string (do not free)
 */
const char *to_outcome_str(to_outcome_t outcome);

/**
 * @brief Get string representation of research status.
 * @param status The research status value
 * @return Static string (do not free)
 */
const char *to_research_status_str(to_research_status_t status);

/**
 * @brief Get string representation of effect pattern.
 * @param pattern The pattern value
 * @return Static string (do not free)
 */
const char *to_effect_pattern_str(to_effect_pattern_t pattern);

/**
 * @brief Get string representation of exploitability.
 * @param exploitability The exploitability value
 * @return Static string (do not free)
 */
const char *to_exploitability_str(to_exploitability_t exploitability);

/**
 * @brief Get string representation of quality.
 * @param quality The quality value
 * @return Static string (do not free)
 */
const char *to_quality_str(to_quality_t quality);

/**
 * @brief Get string representation of inconclusive reason.
 * @param reason The reason value
 * @return Static string (do not free)
 */
const char *to_inconclusive_reason_str(to_inconclusive_reason_t reason);

/* ============================================================================
 * Information
 * ============================================================================ */

/**
 * @brief Get the library version string.
 * @return Static version string (do not free)
 */
const char *to_version(void);

/**
 * @brief Get the name of the timer being used.
 *
 * Returns the platform-specific timer name:
 * - "rdtsc" on x86_64
 * - "cntvct_el0" on ARM64
 * - "clock_gettime" on other platforms
 *
 * @return Static string (do not free)
 */
const char *to_timer_name(void);

/**
 * @brief Get the timer frequency in Hz.
 *
 * This is an estimate used for converting timer ticks to nanoseconds.
 *
 * @return Timer frequency in Hz
 */
uint64_t to_timer_frequency(void);

#ifdef __cplusplus
}
#endif

#endif /* TIMING_ORACLE_H */
