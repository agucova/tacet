/*
 * timing_oracle_go.h - C FFI for timing-oracle statistical analysis (Go bindings)
 *
 * This library provides Bayesian statistical analysis for detecting timing side
 * channels. Unlike timing-oracle-c, this FFI does NOT include measurement
 * functionality - the Go side collects timing samples and passes raw timing
 * data (as uint64 ticks) to these analysis functions.
 *
 * Architecture:
 *   The measurement loop should be implemented in pure Go to avoid FFI overhead
 *   during timing-critical code. This library handles only the statistical
 *   analysis, which is not timing-sensitive.
 *
 * Usage:
 *   1. Go collects timing samples (baseline and sample class) using platform timers
 *   2. Go calls togo_calibrate() with initial samples
 *   3. Go collects more samples in batches
 *   4. Go calls togo_adaptive_step() after each batch
 *   5. Repeat until decision is reached (return code 1)
 *   6. Go calls togo_calibration_free() and togo_result_free() to clean up
 *
 * See the Go package documentation for complete examples.
 */

#ifndef TIMING_ORACLE_GO_H
#define TIMING_ORACLE_GO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Version
 * ============================================================================ */

#define TOGO_VERSION_MAJOR 0
#define TOGO_VERSION_MINOR 1
#define TOGO_VERSION_PATCH 0

/* ============================================================================
 * Attacker Models
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
    TOGO_ATTACKER_SHARED_HARDWARE = 0,
    /** theta = 3.3 ns (~10 cycles) - Post-quantum crypto */
    TOGO_ATTACKER_POST_QUANTUM = 1,
    /** theta = 100 ns - LAN, HTTP/2 (Timeless Timing Attacks) */
    TOGO_ATTACKER_ADJACENT_NETWORK = 2,
    /** theta = 50 us - General internet */
    TOGO_ATTACKER_REMOTE_NETWORK = 3,
    /** theta -> 0 - Detect any difference (not for CI) */
    TOGO_ATTACKER_RESEARCH = 4,
    /** User-specified threshold via custom_threshold_ns */
    TOGO_ATTACKER_CUSTOM = 5
} togo_attacker_model_t;

/* ============================================================================
 * Configuration
 * ============================================================================ */

/**
 * @brief Configuration for the timing analysis.
 *
 * Use `togo_config_default()` to create a configuration with sensible defaults,
 * then override specific fields as needed.
 */
typedef struct {
    /** Attacker model to use (determines threshold theta). */
    togo_attacker_model_t attacker_model;
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
    /** Random seed. 0 = use deterministic seed from data. */
    uint64_t seed;
    /** Timer frequency in Hz (for converting ticks to nanoseconds). */
    uint64_t timer_frequency_hz;
} togo_config_t;

/**
 * @brief Create a default configuration for the given attacker model.
 *
 * @param model The attacker model to use
 * @return Configuration with sensible defaults
 */
togo_config_t togo_config_default(togo_attacker_model_t model);

/* ============================================================================
 * Result Types
 * ============================================================================ */

/**
 * @brief Test outcome.
 */
typedef enum {
    /** No timing leak detected within threshold theta. */
    TOGO_OUTCOME_PASS = 0,
    /** Timing leak detected exceeding threshold theta. */
    TOGO_OUTCOME_FAIL = 1,
    /** Could not reach a decision (quality gate triggered). */
    TOGO_OUTCOME_INCONCLUSIVE = 2,
    /** Operation too fast to measure reliably. */
    TOGO_OUTCOME_UNMEASURABLE = 3
} togo_outcome_t;

/**
 * @brief Reason for inconclusive result.
 */
typedef enum {
    /** Not applicable (not inconclusive). */
    TOGO_INCONCLUSIVE_NONE = 0,
    /** Posterior approximately equals prior; data not informative. */
    TOGO_INCONCLUSIVE_DATA_TOO_NOISY = 1,
    /** Posterior stopped updating despite new data. */
    TOGO_INCONCLUSIVE_NOT_LEARNING = 2,
    /** Estimated time to decision exceeds budget. */
    TOGO_INCONCLUSIVE_WOULD_TAKE_TOO_LONG = 3,
    /** Time budget exhausted. */
    TOGO_INCONCLUSIVE_TIME_BUDGET_EXCEEDED = 4,
    /** Sample limit reached. */
    TOGO_INCONCLUSIVE_SAMPLE_BUDGET_EXCEEDED = 5,
    /** Measurement conditions changed during test. */
    TOGO_INCONCLUSIVE_CONDITIONS_CHANGED = 6,
    /** Requested threshold is below measurement floor. */
    TOGO_INCONCLUSIVE_THRESHOLD_UNACHIEVABLE = 7,
    /** Model doesn't fit the data - 2D model insufficient. */
    TOGO_INCONCLUSIVE_MODEL_MISMATCH = 8
} togo_inconclusive_reason_t;

/**
 * @brief Pattern of timing effect.
 */
typedef enum {
    /** Uniform shift across all quantiles. */
    TOGO_EFFECT_UNIFORM_SHIFT = 0,
    /** Effect concentrated in tails (upper quantiles). */
    TOGO_EFFECT_TAIL_EFFECT = 1,
    /** Both shift and tail components present. */
    TOGO_EFFECT_MIXED = 2,
    /** Cannot determine pattern. */
    TOGO_EFFECT_INDETERMINATE = 3
} togo_effect_pattern_t;

/**
 * @brief Exploitability assessment (for Fail outcomes).
 */
typedef enum {
    /** < 100 ns - Negligible practical impact. */
    TOGO_EXPLOIT_NEGLIGIBLE = 0,
    /** 100-500 ns - Possible on LAN with many measurements. */
    TOGO_EXPLOIT_POSSIBLE_LAN = 1,
    /** 500 ns - 20 us - Likely exploitable on LAN. */
    TOGO_EXPLOIT_LIKELY_LAN = 2,
    /** > 20 us - Potentially exploitable remotely. */
    TOGO_EXPLOIT_POSSIBLE_REMOTE = 3
} togo_exploitability_t;

/**
 * @brief Measurement quality assessment.
 */
typedef enum {
    /** MDE < 5 ns - Excellent measurement precision. */
    TOGO_QUALITY_EXCELLENT = 0,
    /** MDE 5-20 ns - Good precision. */
    TOGO_QUALITY_GOOD = 1,
    /** MDE 20-100 ns - Poor precision. */
    TOGO_QUALITY_POOR = 2,
    /** MDE > 100 ns - Too noisy for reliable detection. */
    TOGO_QUALITY_TOO_NOISY = 3
} togo_quality_t;

/**
 * @brief Effect size estimate.
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
    togo_effect_pattern_t pattern;
} togo_effect_t;

/**
 * @brief Diagnostics information for debugging and quality assessment.
 *
 * Contains detailed information about the measurement process (spec §2.8).
 */
typedef struct {
    /* Core diagnostics */

    /** Block length used for bootstrap resampling. */
    size_t dependence_length;
    /** Effective sample size accounting for autocorrelation. */
    size_t effective_sample_size;
    /** Ratio of post-test variance to calibration variance. */
    double stationarity_ratio;
    /** Whether stationarity check passed. */
    bool stationarity_ok;
    /** Projection mismatch Q statistic. */
    double projection_mismatch_q;
    /** Whether projection mismatch is acceptable. */
    bool projection_mismatch_ok;

    /* Timer diagnostics */

    /** Whether discrete mode was used (low timer resolution). */
    bool discrete_mode;
    /** Timer resolution in nanoseconds. */
    double timer_resolution_ns;

    /* v5.4 Gibbs sampler lambda diagnostics */

    /** Total number of Gibbs iterations. */
    size_t gibbs_iters_total;
    /** Number of burn-in iterations. */
    size_t gibbs_burnin;
    /** Number of retained samples. */
    size_t gibbs_retained;
    /** Posterior mean of latent scale lambda. */
    double lambda_mean;
    /** Posterior standard deviation of lambda. */
    double lambda_sd;
    /** Coefficient of variation of lambda (lambda_sd / lambda_mean). */
    double lambda_cv;
    /** Effective sample size of lambda chain. */
    double lambda_ess;
    /** Whether lambda chain mixed well (CV >= 0.1 AND ESS >= 20). */
    bool lambda_mixing_ok;

    /* v5.6 Gibbs sampler kappa diagnostics */

    /** Posterior mean of likelihood precision kappa. */
    double kappa_mean;
    /** Posterior standard deviation of kappa. */
    double kappa_sd;
    /** Coefficient of variation of kappa (kappa_sd / kappa_mean). */
    double kappa_cv;
    /** Effective sample size of kappa chain. */
    double kappa_ess;
    /** Whether kappa chain mixed well (CV >= 0.1 AND ESS >= 20). */
    bool kappa_mixing_ok;
} togo_diagnostics_t;

/**
 * @brief Analysis result.
 *
 * Contains the outcome and all diagnostic information.
 * Must be freed with `togo_result_free()` if recommendation is non-NULL.
 */
typedef struct {
    /** Test outcome. */
    togo_outcome_t outcome;

    /** Leak probability: P(max_k |(X*beta)_k| > theta | data). */
    double leak_probability;

    /** Effect size estimate. */
    togo_effect_t effect;

    /** Measurement quality. */
    togo_quality_t quality;

    /** Number of samples used per class. */
    size_t samples_used;

    /** Time spent in seconds. */
    double elapsed_secs;

    /** Exploitability (only valid if outcome == FAIL). */
    togo_exploitability_t exploitability;

    /** Inconclusive reason (only valid if outcome == INCONCLUSIVE). */
    togo_inconclusive_reason_t inconclusive_reason;

    /** Minimum detectable effect (shift) in nanoseconds. */
    double mde_shift_ns;

    /** Minimum detectable effect (tail) in nanoseconds. */
    double mde_tail_ns;

    /** Timer resolution in nanoseconds. */
    double timer_resolution_ns;

    /** User's requested threshold (theta) in nanoseconds. */
    double theta_user_ns;

    /** Effective threshold after floor adjustment in nanoseconds. */
    double theta_eff_ns;

    /** Measurement floor (theta_floor) in nanoseconds.
     * Minimum detectable effect given current noise. */
    double theta_floor_ns;

    /** Threshold at which decision was made (for Pass/Fail). */
    double decision_threshold_ns;

    /** Recommendation string (owned, must be freed via togo_result_free; NULL if not applicable). */
    const char *recommendation;

    /** Detailed diagnostics (optional). */
    togo_diagnostics_t diagnostics;

    /** Whether diagnostics are populated. */
    bool has_diagnostics;
} togo_result_t;

/* ============================================================================
 * Opaque State Types
 * ============================================================================ */

/**
 * @brief Opaque calibration state handle.
 *
 * Created by `togo_calibrate()`, freed by `togo_calibration_free()`.
 */
typedef struct {
    /** Opaque pointer to internal state (do not modify). */
    void *ptr;
} togo_calibration_t;

/**
 * @brief Adaptive sampling state.
 *
 * Tracks cumulative samples and posterior evolution between adaptive steps.
 * Created by `togo_adaptive_state_new()`, freed by `togo_adaptive_state_free()`.
 */
typedef struct {
    /** Total baseline samples collected (read-only, updated by togo_adaptive_step). */
    size_t total_baseline;
    /** Total sample class samples collected (read-only, updated by togo_adaptive_step). */
    size_t total_sample;
    /** Current leak probability estimate (read-only, updated by togo_adaptive_step). */
    double current_probability;
    /** Opaque pointer to internal state (do not modify). */
    void *ptr;
} togo_adaptive_state_t;

/* ============================================================================
 * Calibration API
 * ============================================================================ */

/**
 * @brief Run calibration phase on initial timing samples.
 *
 * This function performs the calibration phase:
 * 1. Computes quantile differences between baseline and sample classes
 * 2. Estimates covariance matrix via block bootstrap
 * 3. Sets up prior distribution for Bayesian inference
 * 4. Returns calibration state for use in adaptive steps
 *
 * @param baseline Pointer to baseline timing samples (raw ticks/cycles)
 * @param baseline_len Number of baseline samples (minimum 100)
 * @param sample Pointer to sample class timing samples (raw ticks/cycles)
 * @param sample_len Number of sample class samples (minimum 100)
 * @param config Analysis configuration
 * @param calibration Output: calibration state handle
 *
 * @return
 *   - 0: Success
 *   - -1: Null pointer
 *   - -2: Insufficient samples (< 100)
 *   - -3: Internal error
 */
int togo_calibrate(
    const uint64_t *baseline,
    size_t baseline_len,
    const uint64_t *sample,
    size_t sample_len,
    const togo_config_t *config,
    togo_calibration_t *calibration
);

/**
 * @brief Free calibration state.
 *
 * Safe to call multiple times on the same calibration.
 *
 * @param calibration Pointer to calibration to free (can be NULL)
 */
void togo_calibration_free(togo_calibration_t *calibration);

/* ============================================================================
 * Adaptive Sampling API
 * ============================================================================ */

/**
 * @brief Create a new adaptive state for tracking samples across batches.
 *
 * @return New adaptive state (must be freed with togo_adaptive_state_free)
 */
togo_adaptive_state_t togo_adaptive_state_new(void);

/**
 * @brief Free adaptive state.
 *
 * Safe to call multiple times on the same state.
 *
 * @param state Pointer to state to free (can be NULL)
 */
void togo_adaptive_state_free(togo_adaptive_state_t *state);

/**
 * @brief Run one adaptive step with a new batch of samples.
 *
 * This function updates the posterior distribution with new timing data and
 * checks if a decision can be made.
 *
 * @param calibration Calibration state from togo_calibrate()
 * @param baseline Pointer to new baseline timing samples
 * @param baseline_len Number of new baseline samples
 * @param sample Pointer to new sample class timing samples
 * @param sample_len Number of new sample class samples
 * @param config Analysis configuration
 * @param elapsed_secs Elapsed time in seconds (measured by caller)
 * @param state In/out adaptive state (updated on each call)
 * @param result Output result (only filled if decision reached)
 *
 * @return
 *   - 0: Continue sampling (no decision yet)
 *   - 1: Decision reached (result is filled)
 *   - -1: Null pointer
 *   - -2: Invalid calibration
 *   - -3: Internal error
 */
int togo_adaptive_step(
    const togo_calibration_t *calibration,
    const uint64_t *baseline,
    size_t baseline_len,
    const uint64_t *sample,
    size_t sample_len,
    const togo_config_t *config,
    double elapsed_secs,
    togo_adaptive_state_t *state,
    togo_result_t *result
);

/* ============================================================================
 * One-shot Analysis API
 * ============================================================================ */

/**
 * @brief Run complete analysis on pre-collected timing data.
 *
 * This is a convenience function that runs calibration and adaptive analysis
 * in a single call. Use the separate calibrate/adaptive_step functions for
 * incremental analysis with fresh batches.
 *
 * @param baseline Pointer to all baseline timing samples
 * @param baseline_len Number of baseline samples (minimum 100)
 * @param sample Pointer to all sample class timing samples
 * @param sample_len Number of sample class samples (minimum 100)
 * @param config Analysis configuration
 * @param result Output result
 *
 * @return
 *   - 0: Success
 *   - -1: Null pointer
 *   - -2: Insufficient samples (< 100)
 *   - -3: Internal error
 */
int togo_analyze(
    const uint64_t *baseline,
    size_t baseline_len,
    const uint64_t *sample,
    size_t sample_len,
    const togo_config_t *config,
    togo_result_t *result
);

/* ============================================================================
 * Cleanup
 * ============================================================================ */

/**
 * @brief Free a result's owned resources.
 *
 * Call this when done with a result to free the recommendation string.
 * Safe to call multiple times on the same result.
 *
 * @param result Pointer to result to free (can be NULL)
 */
void togo_result_free(togo_result_t *result);

/* ============================================================================
 * String Conversion Utilities
 * ============================================================================ */

/**
 * @brief Get string representation of outcome.
 * @param outcome The outcome value
 * @return Static string (do not free)
 */
const char *togo_outcome_str(togo_outcome_t outcome);

/**
 * @brief Get string representation of effect pattern.
 * @param pattern The pattern value
 * @return Static string (do not free)
 */
const char *togo_effect_pattern_str(togo_effect_pattern_t pattern);

/**
 * @brief Get string representation of exploitability.
 * @param exploitability The exploitability value
 * @return Static string (do not free)
 */
const char *togo_exploitability_str(togo_exploitability_t exploitability);

/**
 * @brief Get string representation of quality.
 * @param quality The quality value
 * @return Static string (do not free)
 */
const char *togo_quality_str(togo_quality_t quality);

/**
 * @brief Get string representation of inconclusive reason.
 * @param reason The reason value
 * @return Static string (do not free)
 */
const char *togo_inconclusive_reason_str(togo_inconclusive_reason_t reason);

/* ============================================================================
 * Information
 * ============================================================================ */

/**
 * @brief Get the library version string.
 * @return Static version string (do not free)
 */
const char *togo_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TIMING_ORACLE_GO_H */
