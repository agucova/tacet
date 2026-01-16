/**
 * @file timing_oracle.hpp
 * @brief C++ header-only wrapper for timing-oracle
 *
 * This header provides a modern C++ interface for timing side-channel detection.
 * It wraps the C API with type-safe enums, lambdas, builder pattern, and
 * std::variant for result handling.
 *
 * @section Requirements
 * - C++17 or later (for std::variant, std::optional, if constexpr)
 * - Link against libtiming_oracle
 *
 * @section Example
 *
 * @code{.cpp}
 * #include <timing_oracle.hpp>
 * #include <iostream>
 *
 * int main() {
 *     using namespace timing_oracle;
 *
 *     // Define input generators
 *     auto inputs = InputPair<32>(
 *         []() { return std::array<uint8_t, 32>{}; },  // Baseline: zeros
 *         []() {
 *             std::array<uint8_t, 32> arr;
 *             std::generate(arr.begin(), arr.end(), std::rand);
 *             return arr;
 *         }  // Sample: random
 *     );
 *
 *     // Run the test
 *     auto outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
 *         .time_budget(std::chrono::seconds(30))
 *         .test(inputs, [](std::span<const uint8_t> data) {
 *             my_crypto_function(data.data(), data.size());
 *         });
 *
 *     // Handle result with std::visit
 *     std::visit(overloaded{
 *         [](const Pass& p) {
 *             std::cout << "Pass: P(leak)=" << p.leak_probability * 100 << "%\n";
 *         },
 *         [](const Fail& f) {
 *             std::cout << "FAIL: " << to_string(f.exploitability) << "\n";
 *             std::exit(1);
 *         },
 *         [](const Inconclusive& i) {
 *             std::cout << "Inconclusive: " << to_string(i.reason) << "\n";
 *         },
 *         [](const Unmeasurable& u) {
 *             std::cout << "Unmeasurable: " << u.recommendation << "\n";
 *         }
 *     }, outcome);
 *
 *     return 0;
 * }
 * @endcode
 */

#ifndef TIMING_ORACLE_HPP
#define TIMING_ORACLE_HPP

#include "timing_oracle.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>

namespace timing_oracle {

// ============================================================================
// Scoped Enums
// ============================================================================

/**
 * @brief Attacker model determining the detection threshold.
 *
 * Choose based on your threat model:
 * - SharedHardware: Co-resident attacker with cycle-level timing
 * - PostQuantum: Post-quantum crypto attacks (KyberSlash-class)
 * - AdjacentNetwork: LAN or HTTP/2 (Timeless Timing Attacks)
 * - RemoteNetwork: General internet exposure
 * - Research: Detect any difference (not for CI)
 */
enum class AttackerModel {
    SharedHardware,    ///< theta = 0.6 ns (~2 cycles)
    PostQuantum,       ///< theta = 3.3 ns (~10 cycles)
    AdjacentNetwork,   ///< theta = 100 ns
    RemoteNetwork,     ///< theta = 50 us
    Research           ///< theta -> 0
};

/**
 * @brief Pattern of timing effect.
 */
enum class EffectPattern {
    UniformShift,   ///< Uniform shift across all quantiles
    TailEffect,     ///< Effect concentrated in tails
    Mixed,          ///< Both shift and tail components
    Indeterminate   ///< Cannot determine pattern
};

/**
 * @brief Exploitability assessment for detected leaks.
 */
enum class Exploitability {
    Negligible,     ///< < 100 ns
    PossibleLAN,    ///< 100-500 ns
    LikelyLAN,      ///< 500 ns - 20 us
    PossibleRemote  ///< > 20 us
};

/**
 * @brief Measurement quality assessment.
 */
enum class Quality {
    Excellent,  ///< MDE < 5 ns
    Good,       ///< MDE 5-20 ns
    Poor,       ///< MDE 20-100 ns
    TooNoisy    ///< MDE > 100 ns
};

/**
 * @brief Reason for inconclusive result.
 */
enum class InconclusiveReason {
    DataTooNoisy,           ///< Posterior ≈ prior
    NotLearning,            ///< KL divergence collapsed
    WouldTakeTooLong,       ///< Projected time exceeds budget
    TimeBudgetExceeded,     ///< Time budget exhausted
    SampleBudgetExceeded,   ///< Sample limit reached
    ConditionsChanged       ///< Measurement conditions changed
};

// ============================================================================
// String Conversion
// ============================================================================

inline std::string_view to_string(AttackerModel model) {
    switch (model) {
        case AttackerModel::SharedHardware: return "SharedHardware";
        case AttackerModel::PostQuantum: return "PostQuantum";
        case AttackerModel::AdjacentNetwork: return "AdjacentNetwork";
        case AttackerModel::RemoteNetwork: return "RemoteNetwork";
        case AttackerModel::Research: return "Research";
    }
    return "Unknown";
}

inline std::string_view to_string(EffectPattern pattern) {
    switch (pattern) {
        case EffectPattern::UniformShift: return "UniformShift";
        case EffectPattern::TailEffect: return "TailEffect";
        case EffectPattern::Mixed: return "Mixed";
        case EffectPattern::Indeterminate: return "Indeterminate";
    }
    return "Unknown";
}

inline std::string_view to_string(Exploitability exploit) {
    switch (exploit) {
        case Exploitability::Negligible: return "Negligible";
        case Exploitability::PossibleLAN: return "PossibleLAN";
        case Exploitability::LikelyLAN: return "LikelyLAN";
        case Exploitability::PossibleRemote: return "PossibleRemote";
    }
    return "Unknown";
}

inline std::string_view to_string(Quality quality) {
    switch (quality) {
        case Quality::Excellent: return "Excellent";
        case Quality::Good: return "Good";
        case Quality::Poor: return "Poor";
        case Quality::TooNoisy: return "TooNoisy";
    }
    return "Unknown";
}

inline std::string_view to_string(InconclusiveReason reason) {
    switch (reason) {
        case InconclusiveReason::DataTooNoisy: return "DataTooNoisy";
        case InconclusiveReason::NotLearning: return "NotLearning";
        case InconclusiveReason::WouldTakeTooLong: return "WouldTakeTooLong";
        case InconclusiveReason::TimeBudgetExceeded: return "TimeBudgetExceeded";
        case InconclusiveReason::SampleBudgetExceeded: return "SampleBudgetExceeded";
        case InconclusiveReason::ConditionsChanged: return "ConditionsChanged";
    }
    return "Unknown";
}

// ============================================================================
// Result Types
// ============================================================================

/**
 * @brief Effect size estimate with decomposition.
 */
struct Effect {
    double shift_ns;        ///< Uniform shift component (nanoseconds)
    double tail_ns;         ///< Tail effect component (nanoseconds)
    double ci_low_ns;       ///< 95% credible interval lower bound
    double ci_high_ns;      ///< 95% credible interval upper bound
    EffectPattern pattern;  ///< Effect pattern
};

/**
 * @brief Pass outcome: no timing leak detected.
 */
struct Pass {
    double leak_probability;    ///< P(leak > theta | data)
    Effect effect;              ///< Effect estimate
    Quality quality;            ///< Measurement quality
    std::size_t samples_used;   ///< Number of samples per class
    double elapsed_secs;        ///< Time spent
};

/**
 * @brief Fail outcome: timing leak detected.
 */
struct Fail {
    double leak_probability;    ///< P(leak > theta | data)
    Effect effect;              ///< Effect estimate
    Exploitability exploitability;  ///< Exploitability assessment
    Quality quality;            ///< Measurement quality
    std::size_t samples_used;   ///< Number of samples per class
    double elapsed_secs;        ///< Time spent
};

/**
 * @brief Inconclusive outcome: could not reach decision.
 */
struct Inconclusive {
    InconclusiveReason reason;  ///< Why inconclusive
    double leak_probability;    ///< Current P(leak > theta | data)
    Effect effect;              ///< Current effect estimate
    Quality quality;            ///< Measurement quality
    std::size_t samples_used;   ///< Number of samples per class
    double elapsed_secs;        ///< Time spent
};

/**
 * @brief Unmeasurable outcome: operation too fast.
 */
struct Unmeasurable {
    double operation_ns;        ///< Measured operation time
    double timer_resolution_ns; ///< Timer resolution
    std::string recommendation; ///< Suggested action
};

/**
 * @brief Test outcome variant.
 */
using Outcome = std::variant<Pass, Fail, Inconclusive, Unmeasurable>;

// ============================================================================
// Input Pair Helper
// ============================================================================

/**
 * @brief Pair of input generators for baseline and sample classes.
 *
 * @tparam N Size of input in bytes
 */
template <std::size_t N>
class InputPair {
public:
    using Generator = std::function<std::array<uint8_t, N>()>;

    /**
     * @brief Construct input pair from generators.
     *
     * @param baseline Generator for baseline class (typically all zeros)
     * @param sample Generator for sample class (typically random)
     */
    InputPair(Generator baseline, Generator sample)
        : baseline_(std::move(baseline))
        , sample_(std::move(sample))
    {}

    std::array<uint8_t, N> generate_baseline() const { return baseline_(); }
    std::array<uint8_t, N> generate_sample() const { return sample_(); }
    static constexpr std::size_t size() { return N; }

private:
    Generator baseline_;
    Generator sample_;
};

// ============================================================================
// Overloaded Visitor Helper
// ============================================================================

/**
 * @brief Helper for std::visit with multiple lambdas.
 *
 * Usage:
 * @code
 * std::visit(overloaded{
 *     [](const Pass& p) { ... },
 *     [](const Fail& f) { ... },
 *     ...
 * }, outcome);
 * @endcode
 */
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// ============================================================================
// Internal Conversion Helpers
// ============================================================================

namespace detail {

inline to_attacker_model_t to_c_model(AttackerModel model) {
    switch (model) {
        case AttackerModel::SharedHardware: return TO_ATTACKER_SHARED_HARDWARE;
        case AttackerModel::PostQuantum: return TO_ATTACKER_POST_QUANTUM;
        case AttackerModel::AdjacentNetwork: return TO_ATTACKER_ADJACENT_NETWORK;
        case AttackerModel::RemoteNetwork: return TO_ATTACKER_REMOTE_NETWORK;
        case AttackerModel::Research: return TO_ATTACKER_RESEARCH;
    }
    return TO_ATTACKER_ADJACENT_NETWORK;
}

inline EffectPattern from_c_pattern(to_effect_pattern_t pattern) {
    switch (pattern) {
        case TO_EFFECT_UNIFORM_SHIFT: return EffectPattern::UniformShift;
        case TO_EFFECT_TAIL_EFFECT: return EffectPattern::TailEffect;
        case TO_EFFECT_MIXED: return EffectPattern::Mixed;
        default: return EffectPattern::Indeterminate;
    }
}

inline Exploitability from_c_exploit(to_exploitability_t exploit) {
    switch (exploit) {
        case TO_EXPLOIT_NEGLIGIBLE: return Exploitability::Negligible;
        case TO_EXPLOIT_POSSIBLE_LAN: return Exploitability::PossibleLAN;
        case TO_EXPLOIT_LIKELY_LAN: return Exploitability::LikelyLAN;
        default: return Exploitability::PossibleRemote;
    }
}

inline Quality from_c_quality(to_quality_t quality) {
    switch (quality) {
        case TO_QUALITY_EXCELLENT: return Quality::Excellent;
        case TO_QUALITY_GOOD: return Quality::Good;
        case TO_QUALITY_POOR: return Quality::Poor;
        default: return Quality::TooNoisy;
    }
}

inline InconclusiveReason from_c_reason(to_inconclusive_reason_t reason) {
    switch (reason) {
        case TO_INCONCLUSIVE_DATA_TOO_NOISY: return InconclusiveReason::DataTooNoisy;
        case TO_INCONCLUSIVE_NOT_LEARNING: return InconclusiveReason::NotLearning;
        case TO_INCONCLUSIVE_WOULD_TAKE_TOO_LONG: return InconclusiveReason::WouldTakeTooLong;
        case TO_INCONCLUSIVE_TIME_BUDGET_EXCEEDED: return InconclusiveReason::TimeBudgetExceeded;
        case TO_INCONCLUSIVE_SAMPLE_BUDGET_EXCEEDED: return InconclusiveReason::SampleBudgetExceeded;
        default: return InconclusiveReason::ConditionsChanged;
    }
}

inline Effect from_c_effect(const to_effect_t& e) {
    return Effect{
        e.shift_ns,
        e.tail_ns,
        e.ci_low_ns,
        e.ci_high_ns,
        from_c_pattern(e.pattern)
    };
}

inline Outcome from_c_result(to_result_t& result) {
    Outcome outcome;

    Effect effect = from_c_effect(result.effect);
    Quality quality = from_c_quality(result.quality);

    switch (result.outcome) {
        case TO_OUTCOME_PASS:
            outcome = Pass{
                result.leak_probability,
                effect,
                quality,
                result.samples_used,
                result.elapsed_secs
            };
            break;

        case TO_OUTCOME_FAIL:
            outcome = Fail{
                result.leak_probability,
                effect,
                from_c_exploit(result.exploitability),
                quality,
                result.samples_used,
                result.elapsed_secs
            };
            break;

        case TO_OUTCOME_INCONCLUSIVE:
            outcome = Inconclusive{
                from_c_reason(result.inconclusive_reason),
                result.leak_probability,
                effect,
                quality,
                result.samples_used,
                result.elapsed_secs
            };
            break;

        case TO_OUTCOME_UNMEASURABLE:
            outcome = Unmeasurable{
                result.operation_ns,
                result.timer_resolution_ns,
                result.recommendation ? result.recommendation : ""
            };
            break;
    }

    to_result_free(&result);
    return outcome;
}

// Context for C callbacks wrapping C++ callables
template <std::size_t N>
struct CallbackContext {
    const InputPair<N>* inputs;
    std::function<void(std::span<const uint8_t>)>* operation;
    std::array<uint8_t, N> buffer;
};

template <std::size_t N>
void generator_wrapper(void* ctx, bool is_baseline, uint8_t* output, std::size_t) {
    auto* context = static_cast<CallbackContext<N>*>(ctx);
    auto data = is_baseline
        ? context->inputs->generate_baseline()
        : context->inputs->generate_sample();
    std::copy(data.begin(), data.end(), output);
}

template <std::size_t N>
void operation_wrapper(void* ctx, const uint8_t* input, std::size_t size) {
    auto* context = static_cast<CallbackContext<N>*>(ctx);
    (*context->operation)(std::span<const uint8_t>(input, size));
}

} // namespace detail

// ============================================================================
// TimingOracle Builder
// ============================================================================

/**
 * @brief Builder for timing tests.
 *
 * Usage:
 * @code
 * auto outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
 *     .time_budget(std::chrono::seconds(30))
 *     .max_samples(100000)
 *     .test(inputs, operation);
 * @endcode
 */
class TimingOracle {
public:
    /**
     * @brief Create a timing oracle for the given attacker model.
     *
     * @param model Attacker model determining detection threshold
     */
    static TimingOracle for_attacker(AttackerModel model) {
        return TimingOracle(model);
    }

    /**
     * @brief Set the time budget.
     *
     * @param budget Maximum time for the test
     */
    template <typename Rep, typename Period>
    TimingOracle& time_budget(std::chrono::duration<Rep, Period> budget) {
        time_budget_secs_ = std::chrono::duration<double>(budget).count();
        return *this;
    }

    /**
     * @brief Set the maximum number of samples per class.
     */
    TimingOracle& max_samples(std::size_t n) {
        max_samples_ = n;
        return *this;
    }

    /**
     * @brief Set the pass threshold for leak probability.
     *
     * @param threshold Pass if P(leak) < threshold (default 0.05)
     */
    TimingOracle& pass_threshold(double threshold) {
        pass_threshold_ = threshold;
        return *this;
    }

    /**
     * @brief Set the fail threshold for leak probability.
     *
     * @param threshold Fail if P(leak) > threshold (default 0.95)
     */
    TimingOracle& fail_threshold(double threshold) {
        fail_threshold_ = threshold;
        return *this;
    }

    /**
     * @brief Set a custom threshold in nanoseconds.
     *
     * This changes the attacker model to Custom.
     */
    TimingOracle& custom_threshold(double threshold_ns) {
        custom_threshold_ns_ = threshold_ns;
        return *this;
    }

    /**
     * @brief Set the random seed.
     *
     * @param seed 0 = use system entropy
     */
    TimingOracle& seed(uint64_t s) {
        seed_ = s;
        return *this;
    }

    /**
     * @brief Run the timing test.
     *
     * @tparam N Size of input in bytes
     * @param inputs Input pair with baseline and sample generators
     * @param operation Operation to test (called with std::span<const uint8_t>)
     * @return Test outcome
     */
    template <std::size_t N>
    Outcome test(
        const InputPair<N>& inputs,
        std::function<void(std::span<const uint8_t>)> operation
    ) {
        // Build configuration
        to_config_t config;
        if (custom_threshold_ns_) {
            config = to_config_default(TO_ATTACKER_CUSTOM);
            config.custom_threshold_ns = *custom_threshold_ns_;
        } else {
            config = to_config_default(detail::to_c_model(model_));
        }

        if (time_budget_secs_) config.time_budget_secs = *time_budget_secs_;
        if (max_samples_) config.max_samples = *max_samples_;
        if (pass_threshold_) config.pass_threshold = *pass_threshold_;
        if (fail_threshold_) config.fail_threshold = *fail_threshold_;
        if (seed_) config.seed = *seed_;

        // Set up callback context
        detail::CallbackContext<N> ctx{&inputs, &operation, {}};

        // Call C API
        to_result_t result = to_test(
            &config,
            N,
            detail::generator_wrapper<N>,
            detail::operation_wrapper<N>,
            &ctx
        );

        return detail::from_c_result(result);
    }

private:
    explicit TimingOracle(AttackerModel model) : model_(model) {}

    AttackerModel model_;
    std::optional<double> time_budget_secs_;
    std::optional<std::size_t> max_samples_;
    std::optional<double> pass_threshold_;
    std::optional<double> fail_threshold_;
    std::optional<double> custom_threshold_ns_;
    std::optional<uint64_t> seed_;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Get the library version.
 */
inline std::string_view version() {
    return to_version();
}

/**
 * @brief Get the timer name.
 */
inline std::string_view timer_name() {
    return to_timer_name();
}

/**
 * @brief Get the timer frequency in Hz.
 */
inline uint64_t timer_frequency() {
    return to_timer_frequency();
}

} // namespace timing_oracle

#endif // TIMING_ORACLE_HPP
