/**
 * @file timing_oracle.hpp
 * @brief Header-only C++ wrapper for timing-oracle C API
 *
 * This provides a modern C++ interface wrapping the C bindings with:
 * - RAII resource management
 * - std::span for array views
 * - Exception-based error handling
 * - Move semantics (no copies for opaque handles)
 *
 * Usage:
 *   #include "timing_oracle.hpp"
 *   using namespace timing_oracle;
 *
 *   auto config = config_adjacent_network();
 *   auto result = analyze(baseline_span, sample_span, config);
 *
 * Requires C++20 for std::span. Compile with -std=c++20.
 */

#pragma once

extern "C" {
#include "timing_oracle.h"
}

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace timing_oracle {

// ============================================================================
// Exception Types
// ============================================================================

/**
 * @brief Base exception class for timing-oracle errors.
 */
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message, ToError code = ToError::Ok)
        : std::runtime_error(message), error_code_(code) {}

    ToError error_code() const noexcept { return error_code_; }

private:
    ToError error_code_;
};

/**
 * @brief Exception thrown when a null pointer is encountered.
 */
class NullPointerError : public Error {
public:
    NullPointerError() : Error("Null pointer", ToError::NullPointer) {}
};

/**
 * @brief Exception thrown when configuration is invalid.
 */
class InvalidConfigError : public Error {
public:
    InvalidConfigError() : Error("Invalid configuration", ToError::InvalidConfig) {}
};

/**
 * @brief Exception thrown when calibration fails.
 */
class CalibrationError : public Error {
public:
    CalibrationError() : Error("Calibration failed", ToError::CalibrationFailed) {}
};

/**
 * @brief Exception thrown when analysis fails.
 */
class AnalysisError : public Error {
public:
    AnalysisError() : Error("Analysis failed", ToError::AnalysisFailed) {}
};

/**
 * @brief Exception thrown when there are not enough samples.
 */
class NotEnoughSamplesError : public Error {
public:
    NotEnoughSamplesError() : Error("Not enough samples", ToError::NotEnoughSamples) {}
};

/**
 * @brief Throws an appropriate exception for the given error code.
 */
inline void throw_on_error(ToError err) {
    switch (err) {
    case ToError::Ok:
        return;
    case ToError::NullPointer:
        throw NullPointerError();
    case ToError::InvalidConfig:
        throw InvalidConfigError();
    case ToError::CalibrationFailed:
        throw CalibrationError();
    case ToError::AnalysisFailed:
        throw AnalysisError();
    case ToError::NotEnoughSamples:
        throw NotEnoughSamplesError();
    default:
        throw Error("Unknown error", err);
    }
}

// ============================================================================
// RAII Wrapper Classes
// ============================================================================

/**
 * @brief RAII wrapper for ToState (adaptive sampling state).
 *
 * Manages the lifecycle of a ToState pointer. Movable but not copyable.
 */
class State {
public:
    /**
     * @brief Create a new adaptive state.
     * @throws Error if allocation fails.
     */
    State() : ptr_(to_state_new()) {
        if (!ptr_) {
            throw Error("Failed to allocate state");
        }
    }

    ~State() {
        if (ptr_) {
            to_state_free(ptr_);
        }
    }

    // Non-copyable
    State(const State&) = delete;
    State& operator=(const State&) = delete;

    // Movable
    State(State&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    State& operator=(State&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                to_state_free(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get the total number of samples collected (both classes combined).
     */
    uint64_t total_samples() const {
        return to_state_total_samples(ptr_);
    }

    /**
     * @brief Get the current leak probability estimate.
     * @return 0.5 if no posterior has been computed yet.
     */
    double leak_probability() const {
        return to_state_leak_probability(ptr_);
    }

    /**
     * @brief Get the underlying raw pointer.
     */
    ToState* get() const noexcept { return ptr_; }

    /**
     * @brief Check if the state is valid (non-null).
     */
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    ToState* ptr_;
};

/**
 * @brief RAII wrapper for ToCalibration (calibration data).
 *
 * Manages the lifecycle of a ToCalibration pointer. Movable but not copyable.
 */
class Calibration {
public:
    /**
     * @brief Wrap an existing calibration pointer (takes ownership).
     */
    explicit Calibration(ToCalibration* p) noexcept : ptr_(p) {}

    ~Calibration() {
        if (ptr_) {
            to_calibration_free(ptr_);
        }
    }

    // Non-copyable
    Calibration(const Calibration&) = delete;
    Calibration& operator=(const Calibration&) = delete;

    // Movable
    Calibration(Calibration&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    Calibration& operator=(Calibration&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                to_calibration_free(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get the underlying raw pointer.
     */
    ToCalibration* get() const noexcept { return ptr_; }

    /**
     * @brief Check if the calibration is valid (non-null).
     */
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    ToCalibration* ptr_;
};

// ============================================================================
// Configuration Functions
// ============================================================================

/**
 * @brief Create a configuration for the AdjacentNetwork attacker model.
 *
 * theta = 100 ns - LAN, HTTP/2 (Timeless Timing Attacks)
 */
inline ToConfig config_adjacent_network() {
    return to_config_adjacent_network();
}

/**
 * @brief Create a configuration for the SharedHardware attacker model.
 *
 * theta = 0.6 ns (~2 cycles @ 3GHz) - SGX, cross-VM, containers
 */
inline ToConfig config_shared_hardware() {
    return to_config_shared_hardware();
}

/**
 * @brief Create a configuration for the RemoteNetwork attacker model.
 *
 * theta = 50 us - General internet
 */
inline ToConfig config_remote_network() {
    return to_config_remote_network();
}

/**
 * @brief Create a default configuration for the given attacker model.
 */
inline ToConfig config_default(ToAttackerModel model) {
    return to_config_default(model);
}

/**
 * @brief Get the threshold in nanoseconds for an attacker model.
 */
inline double attacker_threshold_ns(ToAttackerModel model) {
    return to_attacker_threshold_ns(model);
}

// ============================================================================
// Analysis Functions
// ============================================================================

/**
 * @brief Run calibration on initial samples.
 *
 * This should be called once at the start with calibration samples
 * (typically 5000 per class). The returned calibration handle is used
 * for subsequent step() calls.
 *
 * @param baseline Array of baseline timing samples (in timer ticks)
 * @param sample Array of sample timing samples (in timer ticks)
 * @param config Configuration
 * @return Calibration object (RAII managed)
 * @throws CalibrationError if calibration fails
 * @throws NotEnoughSamplesError if not enough samples provided
 */
inline Calibration calibrate(
    std::span<const uint64_t> baseline,
    std::span<const uint64_t> sample,
    const ToConfig& config
) {
    if (baseline.size() != sample.size()) {
        throw Error("Baseline and sample arrays must have the same size");
    }

    ToError err = ToError::Ok;
    ToCalibration* cal = to_calibrate(
        baseline.data(),
        sample.data(),
        baseline.size(),
        &config,
        &err
    );

    if (err != ToError::Ok || !cal) {
        if (err == ToError::Ok) {
            throw CalibrationError();
        }
        throw_on_error(err);
    }

    return Calibration(cal);
}

/**
 * @brief Analyze pre-collected timing samples (one-shot).
 *
 * This is a convenience function for one-shot analysis when you already
 * have timing data collected. For adaptive sampling, use calibrate() + step().
 *
 * @param baseline Array of baseline timing samples (in timer ticks)
 * @param sample Array of sample timing samples (in timer ticks)
 * @param config Configuration
 * @return Analysis result
 * @throws AnalysisError if analysis fails
 * @throws NotEnoughSamplesError if not enough samples provided
 */
inline ToResult analyze(
    std::span<const uint64_t> baseline,
    std::span<const uint64_t> sample,
    const ToConfig& config
) {
    if (baseline.size() != sample.size()) {
        throw Error("Baseline and sample arrays must have the same size");
    }

    ToResult result{};
    ToError err = to_analyze(
        baseline.data(),
        sample.data(),
        baseline.size(),
        &config,
        &result
    );

    throw_on_error(err);
    return result;
}

/**
 * @brief Run one adaptive step with a batch of new samples.
 *
 * Call this in a loop after calibrate(). Each call processes a batch
 * of new timing samples and updates the posterior probability.
 *
 * @param calibration Calibration data from calibrate()
 * @param state Adaptive state (will be updated)
 * @param baseline Array of new baseline timing samples (in timer ticks)
 * @param sample Array of new sample timing samples (in timer ticks)
 * @param config Configuration
 * @param elapsed_secs Total elapsed time since start
 * @return Step result (check has_decision for termination)
 * @throws Error if step fails
 */
inline ToStepResult step(
    const Calibration& calibration,
    State& state,
    std::span<const uint64_t> baseline,
    std::span<const uint64_t> sample,
    const ToConfig& config,
    double elapsed_secs
) {
    if (baseline.size() != sample.size()) {
        throw Error("Baseline and sample arrays must have the same size");
    }

    ToStepResult result{};
    ToError err = to_step(
        calibration.get(),
        state.get(),
        baseline.data(),
        sample.data(),
        baseline.size(),
        &config,
        elapsed_secs,
        &result
    );

    throw_on_error(err);
    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Get the library version string.
 */
inline std::string version() {
    return std::string(to_version());
}

/**
 * @brief Get the library version as a string_view (no allocation).
 */
inline std::string_view version_view() {
    return std::string_view(to_version());
}

// ============================================================================
// Enum to String Conversions
// ============================================================================

/**
 * @brief Convert ToOutcome to a human-readable string.
 */
inline const char* outcome_to_string(ToOutcome outcome) {
    switch (outcome) {
    case ToOutcome::Pass:        return "Pass";
    case ToOutcome::Fail:        return "Fail";
    case ToOutcome::Inconclusive: return "Inconclusive";
    case ToOutcome::Unmeasurable: return "Unmeasurable";
    default:                     return "Unknown";
    }
}

/**
 * @brief Convert ToMeasurementQuality to a human-readable string.
 */
inline const char* quality_to_string(ToMeasurementQuality quality) {
    switch (quality) {
    case ToMeasurementQuality::Excellent: return "Excellent";
    case ToMeasurementQuality::Good:      return "Good";
    case ToMeasurementQuality::Poor:      return "Poor";
    case ToMeasurementQuality::TooNoisy:  return "TooNoisy";
    default:                              return "Unknown";
    }
}

/**
 * @brief Convert ToExploitability to a human-readable string.
 */
inline const char* exploitability_to_string(ToExploitability exploitability) {
    switch (exploitability) {
    case ToExploitability::SharedHardwareOnly: return "SharedHardwareOnly";
    case ToExploitability::Http2Multiplexing:  return "Http2Multiplexing";
    case ToExploitability::StandardRemote:     return "StandardRemote";
    case ToExploitability::ObviousLeak:        return "ObviousLeak";
    default:                                   return "Unknown";
    }
}

/**
 * @brief Convert ToEffectPattern to a human-readable string.
 */
inline const char* pattern_to_string(ToEffectPattern pattern) {
    switch (pattern) {
    case ToEffectPattern::UniformShift:  return "UniformShift";
    case ToEffectPattern::TailEffect:    return "TailEffect";
    case ToEffectPattern::Mixed:         return "Mixed";
    case ToEffectPattern::Indeterminate: return "Indeterminate";
    default:                             return "Unknown";
    }
}

/**
 * @brief Convert ToAttackerModel to a human-readable string.
 */
inline const char* attacker_model_to_string(ToAttackerModel model) {
    switch (model) {
    case ToAttackerModel::SharedHardware:  return "SharedHardware";
    case ToAttackerModel::PostQuantum:     return "PostQuantum";
    case ToAttackerModel::AdjacentNetwork: return "AdjacentNetwork";
    case ToAttackerModel::RemoteNetwork:   return "RemoteNetwork";
    case ToAttackerModel::Research:        return "Research";
    default:                               return "Unknown";
    }
}

/**
 * @brief Convert ToInconclusiveReason to a human-readable string.
 */
inline const char* inconclusive_reason_to_string(ToInconclusiveReason reason) {
    switch (reason) {
    case ToInconclusiveReason::None:                 return "None";
    case ToInconclusiveReason::DataTooNoisy:         return "DataTooNoisy";
    case ToInconclusiveReason::NotLearning:          return "NotLearning";
    case ToInconclusiveReason::WouldTakeTooLong:     return "WouldTakeTooLong";
    case ToInconclusiveReason::TimeBudgetExceeded:   return "TimeBudgetExceeded";
    case ToInconclusiveReason::SampleBudgetExceeded: return "SampleBudgetExceeded";
    case ToInconclusiveReason::ConditionsChanged:    return "ConditionsChanged";
    case ToInconclusiveReason::ThresholdElevated:    return "ThresholdElevated";
    default:                                         return "Unknown";
    }
}

} // namespace timing_oracle
