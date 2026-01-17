/**
 * @file to_measure.c
 * @brief Timing-critical measurement loop for timing-oracle-c
 *
 * This file contains the timing-critical C code for the library:
 * - Platform-specific timer reads (rdtsc, cntvct_el0, perf, kperf)
 * - The hot measurement loop with minimal overhead
 *
 * Timer support:
 * - x86_64: rdtsc (standard), perf_event (PMU)
 * - ARM64 Linux: cntvct_el0 (standard), perf_event (PMU)
 * - ARM64 macOS: cntvct_el0 (standard), kperf (PMU)
 */

#include "to_measure.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================================
 * Platform detection
 * ============================================================================ */

#if defined(__linux__)
#define PLATFORM_LINUX 1
#else
#define PLATFORM_LINUX 0
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#define PLATFORM_MACOS_ARM64 1
#else
#define PLATFORM_MACOS_ARM64 0
#endif

/* ============================================================================
 * Global timer state
 * ============================================================================ */

static to_timer_type_t current_timer_type = TO_TIMER_TYPE_CLOCK_GETTIME;
static bool timer_initialized = false;
static double cached_cycles_per_ns = 0.0;

/* ============================================================================
 * Standard timer implementations (always available)
 * ============================================================================ */

#if defined(__x86_64__) || defined(_M_X64)

static inline uint64_t read_rdtsc(void) {
    unsigned int lo, hi;
    __asm__ volatile (
        "lfence\n\t"
        "rdtsc"
        : "=a"(lo), "=d"(hi)
        :
        : "memory"
    );
    return ((uint64_t)hi << 32) | lo;
}

static uint64_t get_rdtsc_freq(void) {
    struct timespec ts_start, ts_end;
    uint64_t tsc_start, tsc_end;
    const uint64_t target_ns = 10000000ULL;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    tsc_start = read_rdtsc();

    uint64_t elapsed_ns = 0;
    while (elapsed_ns < target_ns) {
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        elapsed_ns = (uint64_t)(ts_end.tv_sec - ts_start.tv_sec) * 1000000000ULL
                   + (uint64_t)(ts_end.tv_nsec - ts_start.tv_nsec);
    }

    tsc_end = read_rdtsc();
    uint64_t tsc_delta = tsc_end - tsc_start;
    if (elapsed_ns == 0) return 3000000000ULL;
    return (tsc_delta * 1000000000ULL) / elapsed_ns;
}

#define DEFAULT_TIMER_TYPE TO_TIMER_TYPE_RDTSC
#define read_standard_timer() read_rdtsc()
#define get_standard_timer_freq() get_rdtsc_freq()

#elif defined(__aarch64__) || defined(_M_ARM64)

static inline uint64_t read_cntvct(void) {
    uint64_t val;
    __asm__ volatile (
        "isb\n\t"
        "mrs %0, cntvct_el0"
        : "=r"(val)
        :
        : "memory"
    );
    return val;
}

static uint64_t get_cntvct_freq(void) {
    uint64_t freq;
    __asm__ volatile ("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

#define DEFAULT_TIMER_TYPE TO_TIMER_TYPE_CNTVCT
#define read_standard_timer() read_cntvct()
#define get_standard_timer_freq() get_cntvct_freq()

#elif defined(__i386__) || defined(_M_IX86)

static inline uint64_t read_rdtsc_32(void) {
    unsigned int lo, hi;
    __asm__ volatile (
        "lfence\n\t"
        "rdtsc"
        : "=a"(lo), "=d"(hi)
        :
        : "memory"
    );
    return ((uint64_t)hi << 32) | lo;
}

static uint64_t get_rdtsc_freq_32(void) {
    struct timespec ts_start, ts_end;
    uint64_t tsc_start, tsc_end;
    const uint64_t target_ns = 10000000ULL;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    tsc_start = read_rdtsc_32();

    uint64_t elapsed_ns = 0;
    while (elapsed_ns < target_ns) {
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        elapsed_ns = (uint64_t)(ts_end.tv_sec - ts_start.tv_sec) * 1000000000ULL
                   + (uint64_t)(ts_end.tv_nsec - ts_start.tv_nsec);
    }

    tsc_end = read_rdtsc_32();
    uint64_t tsc_delta = tsc_end - tsc_start;
    if (elapsed_ns == 0) return 3000000000ULL;
    return (tsc_delta * 1000000000ULL) / elapsed_ns;
}

#define DEFAULT_TIMER_TYPE TO_TIMER_TYPE_RDTSC
#define read_standard_timer() read_rdtsc_32()
#define get_standard_timer_freq() get_rdtsc_freq_32()

#else

static inline uint64_t read_clock_gettime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

#define DEFAULT_TIMER_TYPE TO_TIMER_TYPE_CLOCK_GETTIME
#define read_standard_timer() read_clock_gettime()
#define get_standard_timer_freq() 1000000000ULL

#endif

/* ============================================================================
 * Linux perf_event implementation
 * ============================================================================ */

#if PLATFORM_LINUX

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>

static int perf_fd = -1;

static int perf_init(void) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof(pe);
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    perf_fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (perf_fd < 0) {
        return -1;
    }

    ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, 0);
    return 0;
}

static void perf_cleanup(void) {
    if (perf_fd >= 0) {
        close(perf_fd);
        perf_fd = -1;
    }
}

static inline uint64_t read_perf(void) {
    uint64_t count = 0;
    if (perf_fd >= 0) {
        read(perf_fd, &count, sizeof(count));
    }
    return count;
}

#else

static int perf_init(void) { return -1; }
static void perf_cleanup(void) {}
static inline uint64_t read_perf(void) { return 0; }

#endif /* PLATFORM_LINUX */

/* ============================================================================
 * macOS kperf implementation
 * ============================================================================ */

#if PLATFORM_MACOS_ARM64

#include <dlfcn.h>
#include <string.h>

/* kperf constants */
#define KPC_CLASS_FIXED_MASK         (1 << 0)
#define KPC_CLASS_CONFIGURABLE_MASK  (1 << 1)

/* Function pointer types */
typedef int (*kpc_force_all_ctrs_set_fn)(int);
typedef int (*kpc_set_counting_fn)(uint32_t);
typedef int (*kpc_set_thread_counting_fn)(uint32_t);
typedef int (*kpc_get_thread_counters_fn)(uint32_t, uint32_t, uint64_t*);
typedef uint32_t (*kpc_get_counter_count_fn)(uint32_t);

/* Function pointers (loaded at runtime) */
static kpc_force_all_ctrs_set_fn kpc_force_all_ctrs_set = NULL;
static kpc_set_counting_fn kpc_set_counting = NULL;
static kpc_set_thread_counting_fn kpc_set_thread_counting = NULL;
static kpc_get_thread_counters_fn kpc_get_thread_counters = NULL;
static kpc_get_counter_count_fn kpc_get_counter_count = NULL;

static void *kperf_handle = NULL;
static uint32_t kperf_counter_count = 0;

/* Buffer for reading counters - fixed counters include cycles */
static uint64_t kperf_counters[32];

static int kperf_init(void) {
    /* Load kperf framework */
    kperf_handle = dlopen(
        "/System/Library/PrivateFrameworks/kperf.framework/kperf",
        RTLD_NOW
    );
    if (!kperf_handle) {
        return -1;
    }

    /* Load function pointers */
    kpc_force_all_ctrs_set = (kpc_force_all_ctrs_set_fn)dlsym(kperf_handle, "kpc_force_all_ctrs_set");
    kpc_set_counting = (kpc_set_counting_fn)dlsym(kperf_handle, "kpc_set_counting");
    kpc_set_thread_counting = (kpc_set_thread_counting_fn)dlsym(kperf_handle, "kpc_set_thread_counting");
    kpc_get_thread_counters = (kpc_get_thread_counters_fn)dlsym(kperf_handle, "kpc_get_thread_counters");
    kpc_get_counter_count = (kpc_get_counter_count_fn)dlsym(kperf_handle, "kpc_get_counter_count");

    if (!kpc_force_all_ctrs_set || !kpc_set_counting ||
        !kpc_set_thread_counting || !kpc_get_thread_counters ||
        !kpc_get_counter_count) {
        dlclose(kperf_handle);
        kperf_handle = NULL;
        return -1;
    }

    /* Force all counters accessible (requires root) */
    if (kpc_force_all_ctrs_set(1) != 0) {
        dlclose(kperf_handle);
        kperf_handle = NULL;
        return -1;
    }

    /* Get counter count for fixed class */
    kperf_counter_count = kpc_get_counter_count(KPC_CLASS_FIXED_MASK);
    if (kperf_counter_count == 0 || kperf_counter_count > 32) {
        kperf_counter_count = 10;  /* Reasonable default */
    }

    /* Enable counting for fixed counters (includes CPU cycles) */
    uint32_t classes = KPC_CLASS_FIXED_MASK;
    kpc_set_counting(classes);
    kpc_set_thread_counting(classes);

    return 0;
}

static void kperf_cleanup(void) {
    if (kperf_handle) {
        /* Disable counting */
        if (kpc_set_counting) kpc_set_counting(0);
        if (kpc_set_thread_counting) kpc_set_thread_counting(0);

        dlclose(kperf_handle);
        kperf_handle = NULL;
    }
    kpc_force_all_ctrs_set = NULL;
    kpc_set_counting = NULL;
    kpc_set_thread_counting = NULL;
    kpc_get_thread_counters = NULL;
    kpc_get_counter_count = NULL;
}

static inline uint64_t read_kperf(void) {
    if (kpc_get_thread_counters) {
        kpc_get_thread_counters(0, kperf_counter_count, kperf_counters);
        /* Fixed counter 0 is typically CPU cycles on Apple Silicon */
        return kperf_counters[0];
    }
    return 0;
}

#else

static int kperf_init(void) { return -1; }
static void kperf_cleanup(void) {}
static inline uint64_t read_kperf(void) { return 0; }

#endif /* PLATFORM_MACOS_ARM64 */

/* ============================================================================
 * Calibration
 * ============================================================================ */

/* Sort helper for median calculation */
static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double calibrate_cycles_per_ns(void) {
    const int iterations = 10;
    double ratios[10];

    for (int i = 0; i < iterations; i++) {
        struct timespec ts_start, ts_end;
        uint64_t cycles_start, cycles_end;

        clock_gettime(CLOCK_MONOTONIC, &ts_start);

        /* Read cycles based on current timer type */
        switch (current_timer_type) {
        case TO_TIMER_TYPE_PERF:
            cycles_start = read_perf();
            break;
        case TO_TIMER_TYPE_KPERF:
            cycles_start = read_kperf();
            break;
        default:
            cycles_start = read_standard_timer();
            break;
        }

        /* Busy wait ~1ms */
        volatile uint64_t dummy = 0;
        for (int j = 0; j < 1000000; j++) {
            dummy += j;
        }
        (void)dummy;

        /* Read cycles again */
        switch (current_timer_type) {
        case TO_TIMER_TYPE_PERF:
            cycles_end = read_perf();
            break;
        case TO_TIMER_TYPE_KPERF:
            cycles_end = read_kperf();
            break;
        default:
            cycles_end = read_standard_timer();
            break;
        }

        clock_gettime(CLOCK_MONOTONIC, &ts_end);

        uint64_t ns = (uint64_t)(ts_end.tv_sec - ts_start.tv_sec) * 1000000000ULL
                    + (uint64_t)(ts_end.tv_nsec - ts_start.tv_nsec);

        if (ns > 0) {
            ratios[i] = (double)(cycles_end - cycles_start) / (double)ns;
        } else {
            ratios[i] = 3.0;  /* Fallback: 3 GHz */
        }
    }

    /* Return median */
    qsort(ratios, iterations, sizeof(double), compare_double);
    return ratios[iterations / 2];
}

/* ============================================================================
 * Unified timer read (hot path)
 * ============================================================================ */

static inline uint64_t read_timer_unified(void) {
    switch (current_timer_type) {
    case TO_TIMER_TYPE_PERF:
        return read_perf();
    case TO_TIMER_TYPE_KPERF:
        return read_kperf();
    case TO_TIMER_TYPE_RDTSC:
    case TO_TIMER_TYPE_CNTVCT:
        return read_standard_timer();
    case TO_TIMER_TYPE_CLOCK_GETTIME:
    default:
        return read_standard_timer();
    }
}

/* ============================================================================
 * Public API: Timer management
 * ============================================================================ */

int to_timer_init(to_timer_pref_t pref) {
    if (timer_initialized) {
        return 0;  /* Already initialized */
    }

    /* Set default timer type for this platform */
    current_timer_type = DEFAULT_TIMER_TYPE;

    if (pref == TO_TIMER_PREF_STANDARD) {
        /* User explicitly wants standard timer */
        timer_initialized = true;
        cached_cycles_per_ns = (double)get_standard_timer_freq() / 1e9;
        return 0;
    }

    /* Try to initialize PMU timer */
    int pmu_result = -1;

#if PLATFORM_LINUX
    pmu_result = perf_init();
    if (pmu_result == 0) {
        current_timer_type = TO_TIMER_TYPE_PERF;
    }
#elif PLATFORM_MACOS_ARM64
    pmu_result = kperf_init();
    if (pmu_result == 0) {
        current_timer_type = TO_TIMER_TYPE_KPERF;
    }
#endif

    if (pmu_result != 0 && pref == TO_TIMER_PREF_PREFER_PMU) {
        /* PMU required but not available */
        return -1;
    }

    /* Calibrate cycles/ns ratio */
    if (current_timer_type == TO_TIMER_TYPE_PERF ||
        current_timer_type == TO_TIMER_TYPE_KPERF) {
        cached_cycles_per_ns = calibrate_cycles_per_ns();
    } else {
        cached_cycles_per_ns = (double)get_standard_timer_freq() / 1e9;
    }

    timer_initialized = true;
    return 0;
}

void to_timer_cleanup(void) {
    if (!timer_initialized) {
        return;
    }

#if PLATFORM_LINUX
    perf_cleanup();
#endif

#if PLATFORM_MACOS_ARM64
    kperf_cleanup();
#endif

    current_timer_type = DEFAULT_TIMER_TYPE;
    timer_initialized = false;
    cached_cycles_per_ns = 0.0;
}

to_timer_type_t to_get_timer_type(void) {
    return current_timer_type;
}

double to_get_cycles_per_ns(void) {
    if (cached_cycles_per_ns > 0.0) {
        return cached_cycles_per_ns;
    }
    return (double)get_standard_timer_freq() / 1e9;
}

/* ============================================================================
 * Public API: Measurement
 * ============================================================================ */

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
) {
    /* Validate required parameters */
    if (generator == NULL || operation == NULL) {
        return 0;
    }
    if (input_buffer == NULL || schedule == NULL || out_timings == NULL) {
        return 0;
    }
    if (count == 0 || input_size == 0 || batch_k == 0) {
        return 0;
    }

    /* Auto-initialize if needed */
    if (!timer_initialized) {
        to_timer_init(TO_TIMER_PREF_AUTO);
    }

    size_t n_baseline = 0;

    for (size_t i = 0; i < count; i++) {
        bool is_baseline = schedule[i];
        if (is_baseline) {
            n_baseline++;
        }

        /* Generate input OUTSIDE timed region */
        generator(ctx, is_baseline, input_buffer, input_size);

        /* === TIMED REGION === */
        uint64_t start = read_timer_unified();

        for (size_t k = 0; k < batch_k; k++) {
            operation(ctx, input_buffer, input_size);
        }

#if defined(__aarch64__) || defined(_M_ARM64)
        /* Data Memory Barrier to ensure all memory operations from the
         * operation complete before reading the end time. */
        __asm__ volatile ("dmb ish" ::: "memory");
#endif

        uint64_t end = read_timer_unified();
        /* === END TIMED REGION === */

        out_timings[i] = end - start;
    }

    return n_baseline;
}

/* ============================================================================
 * Public API: Timer info
 * ============================================================================ */

const char* to_get_timer_name(void) {
    switch (current_timer_type) {
    case TO_TIMER_TYPE_RDTSC:
        return "rdtsc";
    case TO_TIMER_TYPE_CNTVCT:
        return "cntvct_el0";
    case TO_TIMER_TYPE_PERF:
        return "perf";
    case TO_TIMER_TYPE_KPERF:
        return "kperf";
    case TO_TIMER_TYPE_CLOCK_GETTIME:
    default:
        return "clock_gettime";
    }
}

uint64_t to_get_timer_frequency(void) {
    switch (current_timer_type) {
    case TO_TIMER_TYPE_PERF:
    case TO_TIMER_TYPE_KPERF:
        /* For PMU timers, derive from calibrated cycles/ns */
        return (uint64_t)(cached_cycles_per_ns * 1e9);
    default:
        return get_standard_timer_freq();
    }
}

uint64_t to_read_timer(void) {
    return read_timer_unified();
}
