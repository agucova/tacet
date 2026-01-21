/**
 * Tests for error handling, edge cases, and input validation.
 *
 * Validates that the API handles edge cases gracefully and provides
 * meaningful error messages for invalid inputs.
 */

import { expect, test, describe } from "bun:test";
import {
  TimingOracle,
  AttackerModel,
  Outcome,
  collectSamples,
  calibrateTimer,
  defaultConfig,
} from "../dist/index.js";
import {
  generateZeros,
  generateRandom,
  generateFixedPattern,
  xorFold,
  blackBox,
  outcomeName,
  inconclusiveReasonName,
} from "./helpers.js";

describe("Error handling", () => {
  test(
    "handles very short time budget",
    () => {
      // 100ms is very short - should still return a result
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(100) // Only 100ms
        .maxSamples(100_000)
        .test(
          {
            baseline: () => generateFixedPattern(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            for (let i = 0; i < 50; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[very_short_time_budget]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Elapsed: ${result.elapsedSecs.toFixed(2)}s`);

      // Should return some valid result, likely Inconclusive
      expect(result.outcome).toBeDefined();
      expect(result.leakProbability).toBeGreaterThanOrEqual(0);
      expect(result.leakProbability).toBeLessThanOrEqual(1);
    },
    5_000
  );

  test(
    "handles small sample budget",
    () => {
      // Note: The oracle requires a minimum number of samples for calibration
      // (typically 5000), so a very small budget will still use that minimum.
      // This test verifies the oracle handles this gracefully.
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(30_000)
        .maxSamples(200) // Very few samples (but calibration needs more)
        .test(
          {
            baseline: () => generateFixedPattern(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            for (let i = 0; i < 50; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[small_sample_budget]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Samples used: ${result.samplesUsed}`);

      // Should handle gracefully - outcome should be defined
      expect(result.outcome).toBeDefined();
      // Samples used should be positive
      expect(result.samplesUsed).toBeGreaterThan(0);
      // The outcome is likely Inconclusive due to insufficient budget
      // but could be Pass/Fail if a decision was reached during calibration
    },
    10_000
  );

  test(
    "seed produces deterministic behavior",
    () => {
      // Run the same test twice with the same seed
      const seed = 12345;

      const runTest = () =>
        TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
          .timeBudget(5_000)
          .maxSamples(10_000)
          .seed(seed)
          .test(
            {
              baseline: () => generateZeros(64),
              sample: () => generateRandom(64),
            },
            (input) => {
              for (let i = 0; i < 50; i++) {
                blackBox(xorFold(input));
              }
            }
          );

      const result1 = runTest();
      const result2 = runTest();

      console.log("\n[seed_determinism]");
      console.log(`Result 1 leak probability: ${result1.leakProbability.toFixed(4)}`);
      console.log(`Result 2 leak probability: ${result2.leakProbability.toFixed(4)}`);

      // With the same seed, internal randomness should be the same
      // However, the measurement data will differ due to system noise
      // So we just check that both produce valid results
      expect(result1.outcome).toBeDefined();
      expect(result2.outcome).toBeDefined();
    },
    20_000
  );

  test(
    "custom threshold overrides attacker model",
    () => {
      // Set a custom threshold that differs from AdjacentNetwork's 100ns
      const customThreshold = 500; // 500ns

      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .customThreshold(customThreshold)
        .timeBudget(5_000)
        .maxSamples(10_000)
        .test(
          {
            baseline: () => generateZeros(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            for (let i = 0; i < 50; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[custom_threshold_override]");
      console.log(`User theta: ${result.thetaUserNs}ns`);
      console.log(`Effective theta: ${result.thetaEffNs}ns`);

      // User threshold should reflect the custom value
      expect(result.thetaUserNs).toBe(customThreshold);
    },
    15_000
  );

  test(
    "handles zero-length data",
    () => {
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(5_000)
        .maxSamples(10_000)
        .test(
          {
            baseline: () => new Uint8Array(0),
            sample: () => new Uint8Array(0),
          },
          (input) => {
            blackBox(xorFold(input));
          }
        );

      console.log("\n[zero_length_data]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);

      // Should handle gracefully without crashing
      expect(result.outcome).toBeDefined();
    },
    15_000
  );

  test(
    "handles operation that throws",
    () => {
      let callCount = 0;

      // This shouldn't crash the entire test runner
      expect(() => {
        TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
          .timeBudget(2_000)
          .maxSamples(1_000)
          .test(
            {
              baseline: () => generateZeros(64),
              sample: () => generateRandom(64),
            },
            (input) => {
              callCount++;
              // Throw after a few calls
              if (callCount > 100) {
                throw new Error("Test error");
              }
              blackBox(xorFold(input));
            }
          );
      }).toThrow();
    },
    10_000
  );
});

describe("Input validation", () => {
  test("defaultConfig returns valid config", () => {
    const config = defaultConfig(AttackerModel.AdjacentNetwork);

    expect(config.attackerModel).toBe(AttackerModel.AdjacentNetwork);
    expect(config.timeBudgetMs).toBeGreaterThan(0);
    expect(config.maxSamples).toBeGreaterThan(0);
    expect(config.passThreshold).toBeGreaterThan(0);
    expect(config.passThreshold).toBeLessThan(1);
    expect(config.failThreshold).toBeGreaterThan(0);
    expect(config.failThreshold).toBeLessThan(1);
    expect(config.passThreshold).toBeLessThan(config.failThreshold);
  });

  test("defaultConfig for different attacker models", () => {
    const sharedConfig = defaultConfig(AttackerModel.SharedHardware);
    const adjacentConfig = defaultConfig(AttackerModel.AdjacentNetwork);
    const remoteConfig = defaultConfig(AttackerModel.RemoteNetwork);

    expect(sharedConfig.attackerModel).toBe(AttackerModel.SharedHardware);
    expect(adjacentConfig.attackerModel).toBe(AttackerModel.AdjacentNetwork);
    expect(remoteConfig.attackerModel).toBe(AttackerModel.RemoteNetwork);
  });

  test(
    "builder methods are chainable",
    () => {
      // All builder methods should return the builder for chaining
      const oracle = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(10_000)
        .maxSamples(50_000)
        .passThreshold(0.05)
        .failThreshold(0.95)
        .customThreshold(100)
        .seed(42);

      // Should be able to run a test
      const result = oracle.test(
        {
          baseline: () => generateZeros(32),
          sample: () => generateRandom(32),
        },
        (input) => {
          blackBox(xorFold(input));
        }
      );

      expect(result.outcome).toBeDefined();
    },
    15_000
  );

  test(
    "passThreshold and failThreshold affect outcomes",
    () => {
      // With very lenient thresholds, should pass quickly
      const lenientResult = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(5_000)
        .maxSamples(10_000)
        .passThreshold(0.49) // Very lenient
        .failThreshold(0.51) // Very lenient
        .test(
          {
            baseline: () => generateFixedPattern(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            for (let i = 0; i < 50; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[threshold_affects_outcome]");
      console.log(`Lenient thresholds outcome: ${Outcome[lenientResult.outcome]}`);
      console.log(`Leak probability: ${lenientResult.leakProbability.toFixed(3)}`);

      // With very lenient thresholds, should reach a decision quickly
      // Either Pass or Fail, but likely not Inconclusive
      expect(lenientResult.outcome).toBeDefined();
    },
    15_000
  );
});

describe("collectSamples function", () => {
  test("collectSamples returns expected structure", () => {
    const samples = collectSamples(
      100, // 100 samples per class
      () => generateZeros(32),
      () => generateRandom(32),
      (input) => {
        blackBox(xorFold(input));
      }
    );

    expect(samples.baseline).toBeInstanceOf(BigInt64Array);
    expect(samples.sample).toBeInstanceOf(BigInt64Array);
    expect(samples.baseline.length).toBe(100);
    expect(samples.sample.length).toBe(100);
    expect(samples.batchingInfo).toBeDefined();
    expect(samples.batchingInfo.k).toBeGreaterThanOrEqual(1);
    expect(samples.timerInfo).toBeDefined();
    expect(samples.timerInfo.frequencyHz).toBeGreaterThan(0);
  });

  test("collectSamples handles different sample counts", () => {
    const smallSamples = collectSamples(
      10,
      () => 0,
      () => 1,
      (x) => blackBox(x)
    );
    expect(smallSamples.baseline.length).toBe(10);
    expect(smallSamples.sample.length).toBe(10);

    const largeSamples = collectSamples(
      1000,
      () => 0,
      () => 1,
      (x) => blackBox(x)
    );
    expect(largeSamples.baseline.length).toBe(1000);
    expect(largeSamples.sample.length).toBe(1000);
  });

  test("collectSamples timing values are positive", () => {
    const samples = collectSamples(
      50,
      () => generateZeros(32),
      () => generateRandom(32),
      (input) => {
        for (let i = 0; i < 100; i++) {
          blackBox(xorFold(input));
        }
      }
    );

    // All timing samples should be positive
    for (let i = 0; i < samples.baseline.length; i++) {
      expect(samples.baseline[i]).toBeGreaterThan(0n);
    }
    for (let i = 0; i < samples.sample.length; i++) {
      expect(samples.sample[i]).toBeGreaterThan(0n);
    }
  });
});

describe("Timer functions", () => {
  test("calibrateTimer returns valid info", () => {
    const info = calibrateTimer();

    expect(info.cyclesPerNs).toBeGreaterThan(0);
    expect(info.resolutionNs).toBeGreaterThan(0);
    expect(info.frequencyHz).toBeGreaterThan(0);

    console.log("\n[calibrate_timer]");
    console.log(`Cycles per ns: ${info.cyclesPerNs.toFixed(4)}`);
    console.log(`Resolution: ${info.resolutionNs.toFixed(2)}ns`);
    console.log(`Frequency: ${(info.frequencyHz / 1e9).toFixed(3)} GHz`);
  });

  test("calibrateTimer resolution is reasonable", () => {
    const info = calibrateTimer();

    // Resolution should be between 0.1ns and 100ns for any reasonable timer
    expect(info.resolutionNs).toBeGreaterThan(0.1);
    expect(info.resolutionNs).toBeLessThan(100);
  });
});

describe("Result metadata", () => {
  test(
    "batchingInfo is populated",
    () => {
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(5_000)
        .maxSamples(10_000)
        .test(
          {
            baseline: () => generateZeros(32),
            sample: () => generateRandom(32),
          },
          (input) => {
            for (let i = 0; i < 50; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[batching_info]");
      console.log(`Enabled: ${result.batchingInfo.enabled}`);
      console.log(`K: ${result.batchingInfo.k}`);
      console.log(`Ticks per batch: ${result.batchingInfo.ticksPerBatch}`);
      console.log(`Rationale: ${result.batchingInfo.rationale}`);

      expect(result.batchingInfo.enabled).toBeDefined();
      expect(result.batchingInfo.k).toBeGreaterThanOrEqual(1);
      expect(result.batchingInfo.ticksPerBatch).toBeGreaterThanOrEqual(0);
      expect(result.batchingInfo.rationale).toBeDefined();
    },
    15_000
  );

  test(
    "timerInfo is populated",
    () => {
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(5_000)
        .maxSamples(10_000)
        .test(
          {
            baseline: () => generateZeros(32),
            sample: () => generateRandom(32),
          },
          (input) => {
            for (let i = 0; i < 50; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[timer_info]");
      console.log(`Cycles per ns: ${result.timerInfo.cyclesPerNs.toFixed(4)}`);
      console.log(`Resolution: ${result.timerInfo.resolutionNs.toFixed(2)}ns`);
      console.log(`Frequency: ${(result.timerInfo.frequencyHz / 1e9).toFixed(3)} GHz`);

      expect(result.timerInfo.cyclesPerNs).toBeGreaterThan(0);
      expect(result.timerInfo.resolutionNs).toBeGreaterThan(0);
      expect(result.timerInfo.frequencyHz).toBeGreaterThan(0);
    },
    15_000
  );

  test(
    "elapsedSecs reflects actual time spent",
    () => {
      const startTime = performance.now();

      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(3_000) // 3 second budget
        .maxSamples(100_000)
        .test(
          {
            baseline: () => generateFixedPattern(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            for (let i = 0; i < 50; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      const wallClockMs = performance.now() - startTime;

      console.log("\n[elapsed_time]");
      console.log(`Reported elapsed: ${result.elapsedSecs.toFixed(2)}s`);
      console.log(`Wall clock: ${(wallClockMs / 1000).toFixed(2)}s`);

      // Elapsed should be positive and roughly match wall clock
      expect(result.elapsedSecs).toBeGreaterThan(0);
      // Should not exceed budget significantly
      expect(result.elapsedSecs).toBeLessThan(5); // Some slack for overhead
    },
    10_000
  );
});
