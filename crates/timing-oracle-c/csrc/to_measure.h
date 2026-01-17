/**
 * @file to_measure.h
 * @brief Internal measurement loop for timing-oracle-c
 *
 * This header declares the timing-critical measurement functions called from
 * the Rust FFI layer. These functions are implemented in C to ensure maximum
 * control over the hot loop and platform-specific timer access.
 *
 * Supported timers:
 * - x86_64: rdtsc (standard), perf_event (PMU, requires elevated privileges)
 * - ARM64 Linux: cntvct_el0 (standard), perf_event (PMU, requires elevated privileges)
 * - ARM64 macOS: cntvct_el0 (standard), kperf (PMU, requires root)
 */

#ifndef TO_MEASURE_H
#define TO_MEASURE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Timer Preference
 * ============================================================================ */

/**
 * @brief Timer preference for measurement.
 *
 * Controls which timer implementation to use:
 * - AUTO: Try PMU first (perf/kperf), fall back to standard (rdtsc/cntvct)
 * - STANDARD: Always use standard timer (rdtsc on x86, cntvct on ARM)
 * - PREFER_PMU: Use PMU timer or fail if unavailable
 */
typedef enum {
    TO_TIMER_PREF_AUTO = 0,      /**< Try PMU first, fall back to standard */
    TO_TIMER_PREF_STANDARD = 1,  /**< Always use standard timer */
    TO_TIMER_PREF_PREFER_PMU = 2 /**< Require PMU timer */
} to_timer_pref_t;

/**
 * @brief Active timer type (internal).
 */
typedef enum {
    TO_TIMER_TYPE_RDTSC = 0,        /**< x86_64 RDTSC */
    TO_TIMER_TYPE_CNTVCT = 1,       /**< ARM64 CNTVCT_EL0 */
    TO_TIMER_TYPE_PERF = 2,         /**< Linux perf_event */
    TO_TIMER_TYPE_KPERF = 3,        /**< macOS kperf */
    TO_TIMER_TYPE_CLOCK_GETTIME = 4 /**< Fallback clock_gettime */
} to_timer_type_t;

/**
 * @brief Initialize the timer subsystem.
 *
 * Must be called before to_collect_batch() if using non-default timer.
 * Safe to call multiple times; subsequent calls are no-ops.
 *
 * @param pref Timer preference (AUTO, STANDARD, or PREFER_PMU)
 * @return 0 on success, -1 if PREFER_PMU but PMU unavailable
 */
int to_timer_init(to_timer_pref_t pref);

/**
 * @brief Clean up the timer subsystem.
 *
 * Releases any resources (file descriptors, etc.) acquired during init.
 * Safe to call multiple times or without prior init.
 */
void to_timer_cleanup(void);

/**
 * @brief Get the currently active timer type.
 *
 * @return The timer type being used for measurements
 */
to_timer_type_t to_get_timer_type(void);

/**
 * @brief Get the cycles-per-nanosecond ratio for the active timer.
 *
 * For PMU timers, this is calibrated at init time.
 * For standard timers, this is derived from timer frequency.
 *
 * @return Cycles per nanosecond (e.g., 3.0 for 3GHz CPU)
 */
double to_get_cycles_per_ns(void);

/**
 * @brief Callback type for generating input data.
 *
 * @param context User-provided context pointer
 * @param is_baseline true for baseline class, false for sample class
 * @param output Buffer to write generated input into
 * @param output_size Size of the output buffer in bytes
 */
typedef void (*to_generator_fn)(
    void *context,
    bool is_baseline,
    uint8_t *output,
    size_t output_size
);

/**
 * @brief Callback type for the operation to time.
 *
 * @param context User-provided context pointer
 * @param input Input data buffer
 * @param input_size Size of input data in bytes
 */
typedef void (*to_operation_fn)(
    void *context,
    const uint8_t *input,
    size_t input_size
);

/**
 * @brief Collect a batch of timing measurements.
 *
 * This is the timing-critical measurement loop. It:
 * 1. Generates inputs using the generator callback (OUTSIDE timed region)
 * 2. Reads the high-resolution timer
 * 3. Calls the operation batch_k times
 * 4. Reads the timer again
 * 5. Stores the elapsed time
 *
 * The interleaving schedule determines which samples are baseline vs sample.
 *
 * @param generator Callback to generate input data
 * @param operation Callback for the operation to time
 * @param ctx User context passed to callbacks
 * @param input_buffer Pre-allocated buffer for input data
 * @param input_size Size of input data
 * @param schedule Array of booleans: true = baseline, false = sample
 * @param count Number of measurements to take
 * @param out_timings Output array for timing measurements (in timer ticks)
 * @param batch_k Number of operation invocations per measurement
 *
 * @return Number of baseline samples collected
 */
size_t to_collect_batch(
    to_generator_fn generator,
    to_operation_fn operation,
    void *ctx,
    uint8_t *input_buffer,
    size_t input_size,
    const bool *schedule,
    size_t count,
    uint64_t *out_timings,
    size_t batch_k
);

/**
 * @brief Get the name of the timer being used.
 *
 * @return Static string describing the timer (e.g., "rdtsc", "cntvct_el0")
 */
const char* to_get_timer_name(void);

/**
 * @brief Get the approximate timer frequency in Hz.
 *
 * This is an estimate used for converting ticks to nanoseconds.
 *
 * @return Timer frequency in Hz, or 0 if unknown
 */
uint64_t to_get_timer_frequency(void);

/**
 * @brief Read the current timer value.
 *
 * Exposed for calibration purposes.
 *
 * @return Current timer value in native units (cycles/ticks)
 */
uint64_t to_read_timer(void);

#ifdef __cplusplus
}
#endif

#endif /* TO_MEASURE_H */
