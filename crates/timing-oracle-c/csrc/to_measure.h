/**
 * @file to_measure.h
 * @brief Internal measurement loop for timing-oracle-c
 *
 * This header declares the timing-critical measurement functions called from
 * the Rust FFI layer. These functions are implemented in C to ensure maximum
 * control over the hot loop and platform-specific timer access.
 */

#ifndef TO_MEASURE_H
#define TO_MEASURE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

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
