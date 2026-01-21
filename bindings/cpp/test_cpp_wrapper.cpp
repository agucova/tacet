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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
