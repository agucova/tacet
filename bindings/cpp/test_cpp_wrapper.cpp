/**
 * @file test_cpp_wrapper.cpp
 * @brief Tests for the timing_oracle.hpp C++ wrapper.
 *
 * This file tests the modern C++ wrapper over the C bindings:
 * - Version retrieval
 * - Configuration creation
 * - State lifecycle and RAII
 * - One-shot analyze() with synthetic data
 * - Adaptive calibrate() + step() loop
 *
 * Build (from repo root, after building C library):
 *   clang++ -std=c++20 -c bindings/cpp/test_cpp_wrapper.cpp \
 *           -I crates/timing-oracle-c/include -I bindings/cpp
 *
 * Build and link (requires built library):
 *   clang++ -std=c++20 bindings/cpp/test_cpp_wrapper.cpp \
 *           -I crates/timing-oracle-c/include -I bindings/cpp \
 *           -L target/release -ltiming_oracle_c -o test_cpp_wrapper
 */

#include "timing_oracle.hpp"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <vector>

using namespace timing_oracle;

// ============================================================================
// Test Utilities
// ============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    static struct Test_##name { \
        Test_##name() { \
            printf("Running %s... ", #name); \
            fflush(stdout); \
            try { \
                test_##name(); \
                printf("PASSED\n"); \
                tests_passed++; \
            } catch (const std::exception& e) { \
                printf("FAILED: %s\n", e.what()); \
                tests_failed++; \
            } catch (...) { \
                printf("FAILED: unknown exception\n"); \
                tests_failed++; \
            } \
        } \
    } test_instance_##name; \
    static void test_##name()

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error("Assertion failed: " #cond); \
        } \
    } while (0)

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            throw std::runtime_error("Assertion failed: " #a " == " #b); \
        } \
    } while (0)

#define ASSERT_NE(a, b) \
    do { \
        if ((a) == (b)) { \
            throw std::runtime_error("Assertion failed: " #a " != " #b); \
        } \
    } while (0)

#define ASSERT_THROWS(expr, ExceptionType) \
    do { \
        bool caught = false; \
        try { \
            expr; \
        } catch (const ExceptionType&) { \
            caught = true; \
        } catch (...) { \
            throw std::runtime_error("Wrong exception type for: " #expr); \
        } \
        if (!caught) { \
            throw std::runtime_error("Expected exception not thrown: " #expr); \
        } \
    } while (0)

// ============================================================================
// Tests
// ============================================================================

TEST(version) {
    auto ver = version();
    ASSERT(!ver.empty());
    // Should be a valid semver-ish string
    ASSERT(ver.find('.') != std::string::npos);
}

TEST(version_view) {
    auto ver = version_view();
    ASSERT(!ver.empty());
    ASSERT(ver.find('.') != std::string::npos);
}

TEST(config_adjacent_network) {
    auto cfg = config_adjacent_network();
    ASSERT_EQ(cfg.attacker_model, ToAttackerModel::AdjacentNetwork);
    // Threshold should be set by the library
}

TEST(config_shared_hardware) {
    auto cfg = config_shared_hardware();
    ASSERT_EQ(cfg.attacker_model, ToAttackerModel::SharedHardware);
}

TEST(config_remote_network) {
    auto cfg = config_remote_network();
    ASSERT_EQ(cfg.attacker_model, ToAttackerModel::RemoteNetwork);
}

TEST(config_default) {
    auto cfg = config_default(ToAttackerModel::Research);
    ASSERT_EQ(cfg.attacker_model, ToAttackerModel::Research);
}

TEST(attacker_threshold) {
    // Adjacent network should have 100ns threshold
    double theta = attacker_threshold_ns(ToAttackerModel::AdjacentNetwork);
    ASSERT(theta > 0);
    // 100ns expected
    ASSERT(theta >= 50 && theta <= 200);
}

TEST(state_lifecycle) {
    // Create and destroy
    {
        State s;
        ASSERT(s);
        ASSERT(s.get() != nullptr);
    }
    // Destructor should have freed
}

TEST(state_initial_values) {
    State s;
    ASSERT_EQ(s.total_samples(), 0u);
    // Initial leak probability should be 0.5 (uninformed prior)
    ASSERT(s.leak_probability() >= 0.4 && s.leak_probability() <= 0.6);
}

TEST(state_move) {
    State s1;
    auto* ptr = s1.get();
    ASSERT(ptr != nullptr);

    // Move construct
    State s2(std::move(s1));
    ASSERT_EQ(s2.get(), ptr);
    ASSERT_EQ(s1.get(), nullptr);

    // Move assign
    State s3;
    s3 = std::move(s2);
    ASSERT_EQ(s3.get(), ptr);
    ASSERT_EQ(s2.get(), nullptr);
}

TEST(calibration_move) {
    // Generate synthetic calibration data
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> dist(100.0, 10.0);

    std::vector<uint64_t> baseline(5000);
    std::vector<uint64_t> sample(5000);

    for (size_t i = 0; i < 5000; i++) {
        baseline[i] = static_cast<uint64_t>(std::max(1.0, dist(rng)));
        sample[i] = static_cast<uint64_t>(std::max(1.0, dist(rng)));
    }

    auto cfg = config_adjacent_network();

    Calibration c1 = calibrate(baseline, sample, cfg);
    auto* ptr = c1.get();
    ASSERT(ptr != nullptr);

    // Move construct
    Calibration c2(std::move(c1));
    ASSERT_EQ(c2.get(), ptr);
    ASSERT_EQ(c1.get(), nullptr);

    // Move assign
    Calibration c3 = calibrate(baseline, sample, cfg);
    c3 = std::move(c2);
    ASSERT_EQ(c3.get(), ptr);
    ASSERT_EQ(c2.get(), nullptr);
}

TEST(analyze_identical_data) {
    // Identical baseline and sample should pass (no leak)
    std::vector<uint64_t> data(10000, 100);

    auto cfg = config_adjacent_network();
    auto result = analyze(data, data, cfg);

    // Should pass or be inconclusive, but not fail
    ASSERT(result.outcome == ToOutcome::Pass ||
           result.outcome == ToOutcome::Inconclusive);
}

TEST(analyze_distinct_data) {
    // Distinct distributions should be detected
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist_baseline(100.0, 10.0);
    std::normal_distribution<double> dist_sample(200.0, 10.0); // 100ns difference

    std::vector<uint64_t> baseline(10000);
    std::vector<uint64_t> sample(10000);

    for (size_t i = 0; i < 10000; i++) {
        baseline[i] = static_cast<uint64_t>(std::max(1.0, dist_baseline(rng)));
        sample[i] = static_cast<uint64_t>(std::max(1.0, dist_sample(rng)));
    }

    auto cfg = config_adjacent_network();
    auto result = analyze(baseline, sample, cfg);

    // Should detect the difference
    ASSERT(result.outcome == ToOutcome::Fail);
    ASSERT(result.leak_probability > 0.9);
}

TEST(analyze_size_mismatch) {
    std::vector<uint64_t> baseline(100, 100);
    std::vector<uint64_t> sample(50, 100);

    auto cfg = config_adjacent_network();
    ASSERT_THROWS(analyze(baseline, sample, cfg), Error);
}

TEST(adaptive_loop_simple) {
    // Test the adaptive loop with synthetic data (identical distributions)
    std::mt19937_64 rng(99);
    std::normal_distribution<double> dist(100.0, 10.0);

    // Calibration phase
    std::vector<uint64_t> cal_baseline(5000);
    std::vector<uint64_t> cal_sample(5000);

    for (size_t i = 0; i < 5000; i++) {
        cal_baseline[i] = static_cast<uint64_t>(std::max(1.0, dist(rng)));
        cal_sample[i] = static_cast<uint64_t>(std::max(1.0, dist(rng)));
    }

    auto cfg = config_adjacent_network();
    cfg.time_budget_secs = 5.0; // Short budget for test

    auto calibration = calibrate(cal_baseline, cal_sample, cfg);
    ASSERT(calibration);

    State state;
    ASSERT(state);

    // Run a few adaptive steps
    std::vector<uint64_t> batch_baseline(1000);
    std::vector<uint64_t> batch_sample(1000);

    bool reached_decision = false;
    double elapsed = 0.0;

    for (int i = 0; i < 10 && !reached_decision; i++) {
        // Generate batch
        for (size_t j = 0; j < 1000; j++) {
            batch_baseline[j] = static_cast<uint64_t>(std::max(1.0, dist(rng)));
            batch_sample[j] = static_cast<uint64_t>(std::max(1.0, dist(rng)));
        }

        elapsed += 0.1; // Simulate time passing
        auto step_result = step(calibration, state, batch_baseline, batch_sample, cfg, elapsed);

        if (step_result.has_decision) {
            reached_decision = true;
            // Identical distributions should pass
            ASSERT(step_result.result.outcome == ToOutcome::Pass ||
                   step_result.result.outcome == ToOutcome::Inconclusive);
        }

        // State should track samples
        ASSERT(state.total_samples() > 0);
    }
}

TEST(adaptive_loop_with_leak) {
    // Test the adaptive loop detecting a leak
    std::mt19937_64 rng(123);
    std::normal_distribution<double> dist_baseline(100.0, 5.0);
    std::normal_distribution<double> dist_sample(250.0, 5.0); // 150ns difference - clear leak

    // Calibration phase
    std::vector<uint64_t> cal_baseline(5000);
    std::vector<uint64_t> cal_sample(5000);

    for (size_t i = 0; i < 5000; i++) {
        cal_baseline[i] = static_cast<uint64_t>(std::max(1.0, dist_baseline(rng)));
        cal_sample[i] = static_cast<uint64_t>(std::max(1.0, dist_sample(rng)));
    }

    auto cfg = config_adjacent_network();
    cfg.time_budget_secs = 10.0;

    auto calibration = calibrate(cal_baseline, cal_sample, cfg);
    State state;

    std::vector<uint64_t> batch_baseline(1000);
    std::vector<uint64_t> batch_sample(1000);

    bool reached_decision = false;
    double elapsed = 0.0;

    for (int i = 0; i < 20 && !reached_decision; i++) {
        for (size_t j = 0; j < 1000; j++) {
            batch_baseline[j] = static_cast<uint64_t>(std::max(1.0, dist_baseline(rng)));
            batch_sample[j] = static_cast<uint64_t>(std::max(1.0, dist_sample(rng)));
        }

        elapsed += 0.1;
        auto step_result = step(calibration, state, batch_baseline, batch_sample, cfg, elapsed);

        if (step_result.has_decision) {
            reached_decision = true;
            // Should detect the leak
            ASSERT_EQ(step_result.result.outcome, ToOutcome::Fail);
            ASSERT(step_result.result.leak_probability > 0.9);
        }
    }

    // Should have reached a decision
    ASSERT(reached_decision);
}

TEST(enum_to_string_outcome) {
    ASSERT(std::strcmp(outcome_to_string(ToOutcome::Pass), "Pass") == 0);
    ASSERT(std::strcmp(outcome_to_string(ToOutcome::Fail), "Fail") == 0);
    ASSERT(std::strcmp(outcome_to_string(ToOutcome::Inconclusive), "Inconclusive") == 0);
    ASSERT(std::strcmp(outcome_to_string(ToOutcome::Unmeasurable), "Unmeasurable") == 0);
}

TEST(enum_to_string_quality) {
    ASSERT(std::strcmp(quality_to_string(ToMeasurementQuality::Excellent), "Excellent") == 0);
    ASSERT(std::strcmp(quality_to_string(ToMeasurementQuality::Good), "Good") == 0);
    ASSERT(std::strcmp(quality_to_string(ToMeasurementQuality::Poor), "Poor") == 0);
    ASSERT(std::strcmp(quality_to_string(ToMeasurementQuality::TooNoisy), "TooNoisy") == 0);
}

TEST(enum_to_string_exploitability) {
    ASSERT(std::strcmp(exploitability_to_string(ToExploitability::SharedHardwareOnly), "SharedHardwareOnly") == 0);
    ASSERT(std::strcmp(exploitability_to_string(ToExploitability::Http2Multiplexing), "Http2Multiplexing") == 0);
    ASSERT(std::strcmp(exploitability_to_string(ToExploitability::StandardRemote), "StandardRemote") == 0);
    ASSERT(std::strcmp(exploitability_to_string(ToExploitability::ObviousLeak), "ObviousLeak") == 0);
}

TEST(enum_to_string_pattern) {
    ASSERT(std::strcmp(pattern_to_string(ToEffectPattern::UniformShift), "UniformShift") == 0);
    ASSERT(std::strcmp(pattern_to_string(ToEffectPattern::TailEffect), "TailEffect") == 0);
    ASSERT(std::strcmp(pattern_to_string(ToEffectPattern::Mixed), "Mixed") == 0);
    ASSERT(std::strcmp(pattern_to_string(ToEffectPattern::Indeterminate), "Indeterminate") == 0);
}

TEST(enum_to_string_attacker_model) {
    ASSERT(std::strcmp(attacker_model_to_string(ToAttackerModel::SharedHardware), "SharedHardware") == 0);
    ASSERT(std::strcmp(attacker_model_to_string(ToAttackerModel::PostQuantum), "PostQuantum") == 0);
    ASSERT(std::strcmp(attacker_model_to_string(ToAttackerModel::AdjacentNetwork), "AdjacentNetwork") == 0);
    ASSERT(std::strcmp(attacker_model_to_string(ToAttackerModel::RemoteNetwork), "RemoteNetwork") == 0);
    ASSERT(std::strcmp(attacker_model_to_string(ToAttackerModel::Research), "Research") == 0);
}

TEST(enum_to_string_inconclusive_reason) {
    ASSERT(std::strcmp(inconclusive_reason_to_string(ToInconclusiveReason::None), "None") == 0);
    ASSERT(std::strcmp(inconclusive_reason_to_string(ToInconclusiveReason::DataTooNoisy), "DataTooNoisy") == 0);
    ASSERT(std::strcmp(inconclusive_reason_to_string(ToInconclusiveReason::NotLearning), "NotLearning") == 0);
    ASSERT(std::strcmp(inconclusive_reason_to_string(ToInconclusiveReason::WouldTakeTooLong), "WouldTakeTooLong") == 0);
    ASSERT(std::strcmp(inconclusive_reason_to_string(ToInconclusiveReason::TimeBudgetExceeded), "TimeBudgetExceeded") == 0);
    ASSERT(std::strcmp(inconclusive_reason_to_string(ToInconclusiveReason::SampleBudgetExceeded), "SampleBudgetExceeded") == 0);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST(config_from_env) {
    // Just test that it returns a valid config (env vars not set in test)
    auto base = config_default(ToAttackerModel::AdjacentNetwork);
    auto cfg = config_from_env(base);
    // Without env vars set, should be unchanged
    ASSERT_EQ(cfg.attacker_model, base.attacker_model);
    ASSERT(cfg.time_budget_secs == base.time_budget_secs);
}

// ============================================================================
// Oracle Builder Tests
// ============================================================================

TEST(oracle_factory_methods) {
    // Test forAttacker factory method with different attacker models
    auto o1 = Oracle::forAttacker(ToAttackerModel::AdjacentNetwork);
    ASSERT_EQ(o1.config().attacker_model, ToAttackerModel::AdjacentNetwork);

    auto o2 = Oracle::forAttacker(ToAttackerModel::SharedHardware);
    ASSERT_EQ(o2.config().attacker_model, ToAttackerModel::SharedHardware);

    auto o3 = Oracle::forAttacker(ToAttackerModel::RemoteNetwork);
    ASSERT_EQ(o3.config().attacker_model, ToAttackerModel::RemoteNetwork);

    auto o4 = Oracle::forAttacker(ToAttackerModel::Research);
    ASSERT_EQ(o4.config().attacker_model, ToAttackerModel::Research);
}

TEST(oracle_builder_methods) {
    using namespace std::chrono_literals;

    // Test builder chain
    auto oracle = Oracle::forAttacker(ToAttackerModel::AdjacentNetwork)
        .timeBudget(60s)
        .maxSamples(50000)
        .passThreshold(0.01)
        .failThreshold(0.99)
        .seed(12345)
        .thresholdNs(500.0);

    const auto& cfg = oracle.config();
    ASSERT(cfg.time_budget_secs >= 59.0 && cfg.time_budget_secs <= 61.0);
    ASSERT_EQ(cfg.max_samples, 50000u);
    ASSERT(cfg.pass_threshold >= 0.009 && cfg.pass_threshold <= 0.011);
    ASSERT(cfg.fail_threshold >= 0.989 && cfg.fail_threshold <= 0.991);
    ASSERT_EQ(cfg.seed, 12345u);
    ASSERT(cfg.custom_threshold_ns >= 499.0 && cfg.custom_threshold_ns <= 501.0);
}

TEST(oracle_builder_immutability) {
    // Builder methods should not modify original
    auto o1 = Oracle::forAttacker(ToAttackerModel::AdjacentNetwork);
    auto o2 = o1.maxSamples(1000);

    ASSERT_NE(o1.config().max_samples, 1000u);
    ASSERT_EQ(o2.config().max_samples, 1000u);
}

TEST(oracle_chrono_duration) {
    using namespace std::chrono_literals;

    // Test various duration types
    auto o1 = Oracle::forAttacker(ToAttackerModel::AdjacentNetwork).timeBudget(30s);
    ASSERT(o1.config().time_budget_secs >= 29.0 && o1.config().time_budget_secs <= 31.0);

    auto o2 = Oracle::forAttacker(ToAttackerModel::AdjacentNetwork).timeBudget(2min);
    ASSERT(o2.config().time_budget_secs >= 119.0 && o2.config().time_budget_secs <= 121.0);

    auto o3 = Oracle::forAttacker(ToAttackerModel::AdjacentNetwork).timeBudget(500ms);
    ASSERT(o3.config().time_budget_secs >= 0.49 && o3.config().time_budget_secs <= 0.51);
}

TEST(oracle_from_env) {
    auto oracle = Oracle::forAttacker(ToAttackerModel::AdjacentNetwork).fromEnv();
    // Just test it compiles and runs
    ASSERT(oracle.config().time_budget_secs > 0);
}

TEST(oracle_threshold_ns) {
    auto o1 = Oracle::forAttacker(ToAttackerModel::AdjacentNetwork);
    ASSERT(o1.thresholdNs() >= 50 && o1.thresholdNs() <= 200);  // ~100ns

    auto o2 = o1.thresholdNs(500.0);
    ASSERT(o2.thresholdNs() >= 499.0 && o2.thresholdNs() <= 501.0);
}

TEST(oracle_analyze) {
    // Test Oracle::analyze method
    std::vector<uint64_t> data(10000, 100);

    auto result = Oracle::forAttacker(ToAttackerModel::AdjacentNetwork)
        .analyze(data, data);
    ASSERT(result.outcome == ToOutcome::Pass ||
           result.outcome == ToOutcome::Inconclusive);
}

TEST(oracle_test_callback) {
    // Test Oracle::test method with callback
    using namespace std::chrono_literals;

    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(100.0, 10.0);

    auto result = Oracle::forAttacker(ToAttackerModel::AdjacentNetwork)
        .timeBudget(5s)
        .maxSamples(50000)
        .test([&](std::span<uint64_t> baseline, std::span<uint64_t> sample) {
            // Generate identical distributions (should pass)
            for (size_t i = 0; i < baseline.size(); i++) {
                baseline[i] = static_cast<uint64_t>(std::max(1.0, dist(rng)));
                sample[i] = static_cast<uint64_t>(std::max(1.0, dist(rng)));
            }
        });

    // Identical distributions should pass or be inconclusive
    ASSERT(result.outcome == ToOutcome::Pass ||
           result.outcome == ToOutcome::Inconclusive);
}

TEST(oracle_test_callback_with_leak) {
    // Test Oracle::test detecting a leak
    using namespace std::chrono_literals;

    std::mt19937_64 rng(123);
    std::normal_distribution<double> dist_baseline(100.0, 5.0);
    std::normal_distribution<double> dist_sample(250.0, 5.0);  // 150ns difference

    auto result = Oracle::forAttacker(ToAttackerModel::AdjacentNetwork)
        .timeBudget(10s)
        .maxSamples(100000)
        .test([&](std::span<uint64_t> baseline, std::span<uint64_t> sample) {
            for (size_t i = 0; i < baseline.size(); i++) {
                baseline[i] = static_cast<uint64_t>(std::max(1.0, dist_baseline(rng)));
                sample[i] = static_cast<uint64_t>(std::max(1.0, dist_sample(rng)));
            }
        });

    // Should detect the leak
    ASSERT_EQ(result.outcome, ToOutcome::Fail);
    ASSERT(result.leak_probability > 0.9);
}

// ============================================================================
// Stream Operator Tests
// ============================================================================

TEST(stream_operator_outcome) {
    std::ostringstream oss;
    oss << ToOutcome::Pass;
    ASSERT_EQ(oss.str(), std::string("Pass"));
}

TEST(stream_operator_quality) {
    std::ostringstream oss;
    oss << ToMeasurementQuality::Good;
    ASSERT_EQ(oss.str(), std::string("Good"));
}

TEST(stream_operator_exploitability) {
    std::ostringstream oss;
    oss << ToExploitability::StandardRemote;
    ASSERT_EQ(oss.str(), std::string("StandardRemote"));
}

TEST(stream_operator_pattern) {
    std::ostringstream oss;
    oss << ToEffectPattern::UniformShift;
    ASSERT_EQ(oss.str(), std::string("UniformShift"));
}

TEST(stream_operator_attacker_model) {
    std::ostringstream oss;
    oss << ToAttackerModel::AdjacentNetwork;
    ASSERT_EQ(oss.str(), std::string("AdjacentNetwork"));
}

TEST(stream_operator_inconclusive_reason) {
    std::ostringstream oss;
    oss << ToInconclusiveReason::TimeBudgetExceeded;
    ASSERT_EQ(oss.str(), std::string("TimeBudgetExceeded"));
}

TEST(stream_operator_effect) {
    ToEffect effect{};
    effect.shift_ns = 10.5;
    effect.tail_ns = 5.25;
    effect.ci_low_ns = 8.0;
    effect.ci_high_ns = 18.0;
    effect.pattern = ToEffectPattern::Mixed;

    std::ostringstream oss;
    oss << effect;
    std::string s = oss.str();

    ASSERT(s.find("10.5") != std::string::npos);  // shift_ns
    ASSERT(s.find("5.25") != std::string::npos);  // tail_ns
    ASSERT(s.find("Mixed") != std::string::npos);  // pattern
}

TEST(stream_operator_result) {
    ToResult result{};
    result.outcome = ToOutcome::Pass;
    result.leak_probability = 0.02;
    result.samples_used = 5000;
    result.quality = ToMeasurementQuality::Excellent;

    std::ostringstream oss;
    oss << result;
    std::string s = oss.str();

    ASSERT(s.find("Pass") != std::string::npos);
    ASSERT(s.find("2.00%") != std::string::npos);  // 0.02 * 100
    ASSERT(s.find("5000") != std::string::npos);
    ASSERT(s.find("Excellent") != std::string::npos);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("\n=== timing_oracle.hpp C++ Wrapper Tests ===\n\n");
    printf("Library version: %s\n\n", version().c_str());

    // Tests run automatically via static initialization

    printf("\n=== Summary ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);

    return (tests_failed > 0) ? 1 : 0;
}
