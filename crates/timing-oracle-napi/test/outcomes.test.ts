/**
 * Tests for all outcome types, effect patterns, and exploitability levels.
 *
 * Validates that the API returns correctly structured results for each
 * possible outcome type.
 */

import { expect, test, describe } from "bun:test";
import {
  TimingOracle,
  AttackerModel,
  Outcome,
  InconclusiveReason,
  EffectPattern,
  Exploitability,
  MeasurementQuality,
} from "../dist/index.js";
import {
  generateZeros,
  generateRandom,
  generateFixedPattern,
  leakyCompare,
  xorFold,
  blackBox,
  outcomeName,
  inconclusiveReasonName,
  effectPatternName,
  exploitabilityName,
  measurementQualityName,
} from "./helpers.js";

describe("Outcome types", () => {
  test(
    "Pass outcome has expected fields",
    () => {
      // Use identical operations to maximize chance of Pass
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(10_000)
        .maxSamples(30_000)
        .passThreshold(0.15)
        .failThreshold(0.99)
        .test(
          {
            baseline: () => generateFixedPattern(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            // Constant-time operation
            for (let i = 0; i < 100; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[pass_outcome_fields]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Leak probability: ${result.leakProbability}`);

      // Check all expected fields are present
      expect(result.outcome).toBeDefined();
      expect(result.leakProbability).toBeGreaterThanOrEqual(0);
      expect(result.leakProbability).toBeLessThanOrEqual(1);
      expect(result.effect).toBeDefined();
      expect(result.effect.shiftNs).toBeDefined();
      expect(result.effect.tailNs).toBeDefined();
      expect(result.effect.pattern).toBeDefined();
      expect(result.quality).toBeDefined();
      expect(result.samplesUsed).toBeGreaterThan(0);
      expect(result.elapsedSecs).toBeGreaterThan(0);
      expect(result.diagnostics).toBeDefined();
      expect(result.batchingInfo).toBeDefined();
      expect(result.timerInfo).toBeDefined();

      // If Pass, leak probability should be low
      if (result.outcome === Outcome.Pass) {
        expect(result.leakProbability).toBeLessThan(0.5);
      }
    },
    20_000
  );

  test(
    "Fail outcome has exploitability",
    () => {
      // Use a clearly leaky operation
      const secret = generateZeros(512);

      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(15_000)
        .maxSamples(50_000)
        .passThreshold(0.01)
        .failThreshold(0.85)
        .test(
          {
            baseline: () => generateZeros(512),
            sample: () => generateRandom(512),
          },
          (input) => {
            blackBox(leakyCompare(secret, input));
          }
        );

      console.log("\n[fail_outcome_exploitability]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Exploitability: ${exploitabilityName(result.exploitability)}`);

      if (result.outcome === Outcome.Unmeasurable) {
        console.log(`[SKIPPED] Operation unmeasurable`);
        return;
      }

      if (result.outcome === Outcome.Fail) {
        // Fail outcome should have exploitability set
        expect(result.exploitability).toBeDefined();
        expect(result.leakProbability).toBeGreaterThan(0.85);

        // Effect should be non-trivial
        const totalEffect = Math.sqrt(
          result.effect.shiftNs ** 2 + result.effect.tailNs ** 2
        );
        expect(totalEffect).toBeGreaterThan(0);
      }
    },
    30_000
  );

  test(
    "Inconclusive with tight time budget",
    () => {
      // Very short time budget should result in Inconclusive
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(500) // Only 500ms
        .maxSamples(100_000)
        .passThreshold(0.01)
        .failThreshold(0.99)
        .test(
          {
            baseline: () => generateFixedPattern(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            for (let i = 0; i < 100; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[inconclusive_tight_budget]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Inconclusive reason: ${inconclusiveReasonName(result.inconclusiveReason)}`);

      // With such a tight budget, we expect Inconclusive
      if (result.outcome === Outcome.Inconclusive) {
        expect(result.inconclusiveReason).toBeDefined();
        expect(result.inconclusiveReason).not.toBe(InconclusiveReason.None);
      }

      // Should still have valid probability
      expect(result.leakProbability).toBeGreaterThanOrEqual(0);
      expect(result.leakProbability).toBeLessThanOrEqual(1);
    },
    10_000
  );

  test(
    "Inconclusive with tight sample budget",
    () => {
      // Very few samples with strict thresholds
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(30_000)
        .maxSamples(500) // Very few samples
        .passThreshold(0.01)
        .failThreshold(0.99)
        .test(
          {
            baseline: () => generateFixedPattern(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            for (let i = 0; i < 100; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[inconclusive_tight_samples]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Samples used: ${result.samplesUsed}`);
      console.log(`Inconclusive reason: ${inconclusiveReasonName(result.inconclusiveReason)}`);

      // May be inconclusive due to sample budget
      if (result.outcome === Outcome.Inconclusive) {
        expect(result.inconclusiveReason).not.toBe(InconclusiveReason.None);
      }
    },
    10_000
  );

  test("all InconclusiveReason values are accessible", () => {
    // Verify enum values are exported correctly
    expect(InconclusiveReason.None).toBe(0);
    expect(InconclusiveReason.DataTooNoisy).toBe(1);
    expect(InconclusiveReason.NotLearning).toBe(2);
    expect(InconclusiveReason.WouldTakeTooLong).toBe(3);
    expect(InconclusiveReason.TimeBudgetExceeded).toBe(4);
    expect(InconclusiveReason.SampleBudgetExceeded).toBe(5);
    expect(InconclusiveReason.ConditionsChanged).toBe(6);
    expect(InconclusiveReason.ThresholdElevated).toBe(7);
  });
});

describe("Effect patterns", () => {
  test("all EffectPattern values are accessible", () => {
    expect(EffectPattern.UniformShift).toBe(0);
    expect(EffectPattern.TailEffect).toBe(1);
    expect(EffectPattern.Mixed).toBe(2);
    expect(EffectPattern.Indeterminate).toBe(3);
  });

  test(
    "effect pattern is determined for detected leaks",
    () => {
      const secret = generateZeros(512);

      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(15_000)
        .maxSamples(50_000)
        .passThreshold(0.01)
        .failThreshold(0.85)
        .test(
          {
            baseline: () => generateZeros(512),
            sample: () => generateRandom(512),
          },
          (input) => {
            blackBox(leakyCompare(secret, input));
          }
        );

      console.log("\n[effect_pattern_detection]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Effect pattern: ${effectPatternName(result.effect.pattern)}`);
      console.log(`Shift: ${result.effect.shiftNs.toFixed(2)}ns`);
      console.log(`Tail: ${result.effect.tailNs.toFixed(2)}ns`);

      if (result.outcome === Outcome.Fail) {
        // Pattern should be determined
        expect(result.effect.pattern).toBeDefined();
        // Credible interval should be set
        expect(result.effect.credibleIntervalLow).toBeDefined();
        expect(result.effect.credibleIntervalHigh).toBeDefined();
      }
    },
    30_000
  );

  test(
    "effect has credible interval",
    () => {
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(10_000)
        .maxSamples(30_000)
        .test(
          {
            baseline: () => generateFixedPattern(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            for (let i = 0; i < 100; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[effect_credible_interval]");
      console.log(`CI: [${result.effect.credibleIntervalLow.toFixed(2)}, ${result.effect.credibleIntervalHigh.toFixed(2)}]`);

      // Credible interval fields should exist
      expect(result.effect.credibleIntervalLow).toBeDefined();
      expect(result.effect.credibleIntervalHigh).toBeDefined();
      // High should be >= low
      expect(result.effect.credibleIntervalHigh).toBeGreaterThanOrEqual(
        result.effect.credibleIntervalLow
      );
    },
    20_000
  );
});

describe("Exploitability levels", () => {
  test("all Exploitability values are accessible", () => {
    expect(Exploitability.SharedHardwareOnly).toBe(0);
    expect(Exploitability.Http2Multiplexing).toBe(1);
    expect(Exploitability.StandardRemote).toBe(2);
    expect(Exploitability.ObviousLeak).toBe(3);
  });

  test(
    "exploitability is set appropriately for detected leaks",
    () => {
      const secret = generateZeros(512);

      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(15_000)
        .maxSamples(50_000)
        .passThreshold(0.01)
        .failThreshold(0.85)
        .test(
          {
            baseline: () => generateZeros(512),
            sample: () => generateRandom(512),
          },
          (input) => {
            blackBox(leakyCompare(secret, input));
          }
        );

      console.log("\n[exploitability_appropriate]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Exploitability: ${exploitabilityName(result.exploitability)}`);
      console.log(`Total effect: ${Math.sqrt(result.effect.shiftNs ** 2 + result.effect.tailNs ** 2).toFixed(2)}ns`);

      if (result.outcome === Outcome.Fail) {
        // Exploitability should be set for failures
        expect(result.exploitability).toBeDefined();

        // Exploitability should be based on effect magnitude
        const totalEffectNs = Math.sqrt(
          result.effect.shiftNs ** 2 + result.effect.tailNs ** 2
        );

        // Map expected exploitability based on documented thresholds
        if (totalEffectNs < 10) {
          expect(result.exploitability).toBe(Exploitability.SharedHardwareOnly);
        } else if (totalEffectNs < 100) {
          expect(result.exploitability).toBe(Exploitability.Http2Multiplexing);
        } else if (totalEffectNs < 10_000) {
          expect(result.exploitability).toBe(Exploitability.StandardRemote);
        } else {
          expect(result.exploitability).toBe(Exploitability.ObviousLeak);
        }
      }
    },
    30_000
  );
});

describe("Measurement quality", () => {
  test("all MeasurementQuality values are accessible", () => {
    expect(MeasurementQuality.Excellent).toBe(0);
    expect(MeasurementQuality.Good).toBe(1);
    expect(MeasurementQuality.Poor).toBe(2);
    expect(MeasurementQuality.TooNoisy).toBe(3);
  });

  test(
    "quality is reported in results",
    () => {
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(10_000)
        .maxSamples(30_000)
        .test(
          {
            baseline: () => generateFixedPattern(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            for (let i = 0; i < 100; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[measurement_quality]");
      console.log(`Quality: ${measurementQualityName(result.quality)}`);
      console.log(`MDE shift: ${result.mdeShiftNs.toFixed(2)}ns`);
      console.log(`MDE tail: ${result.mdeTailNs.toFixed(2)}ns`);

      // Quality should be set
      expect(result.quality).toBeDefined();
      expect(result.quality).toBeGreaterThanOrEqual(MeasurementQuality.Excellent);
      expect(result.quality).toBeLessThanOrEqual(MeasurementQuality.TooNoisy);
    },
    20_000
  );
});

describe("Diagnostics", () => {
  test(
    "diagnostics are populated",
    () => {
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(10_000)
        .maxSamples(30_000)
        .test(
          {
            baseline: () => generateFixedPattern(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            for (let i = 0; i < 100; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[diagnostics_populated]");
      console.log(`Dependence length: ${result.diagnostics.dependenceLength}`);
      console.log(`Effective sample size: ${result.diagnostics.effectiveSampleSize}`);
      console.log(`Discrete mode: ${result.diagnostics.discreteMode}`);
      console.log(`Timer resolution: ${result.diagnostics.timerResolutionNs.toFixed(2)}ns`);

      // Check all diagnostic fields
      expect(result.diagnostics.dependenceLength).toBeDefined();
      expect(result.diagnostics.effectiveSampleSize).toBeDefined();
      expect(result.diagnostics.stationarityRatio).toBeDefined();
      expect(result.diagnostics.stationarityOk).toBeDefined();
      expect(result.diagnostics.projectionMismatchQ).toBeDefined();
      expect(result.diagnostics.projectionMismatchOk).toBeDefined();
      expect(result.diagnostics.discreteMode).toBeDefined();
      expect(result.diagnostics.timerResolutionNs).toBeDefined();
      expect(result.diagnostics.lambdaMean).toBeDefined();
      expect(result.diagnostics.lambdaMixingOk).toBeDefined();
    },
    20_000
  );
});

describe("Attacker models", () => {
  test("all AttackerModel values are accessible", () => {
    expect(AttackerModel.SharedHardware).toBe(0);
    expect(AttackerModel.PostQuantum).toBe(1);
    expect(AttackerModel.AdjacentNetwork).toBe(2);
    expect(AttackerModel.RemoteNetwork).toBe(3);
    expect(AttackerModel.Research).toBe(4);
  });

  test(
    "different attacker models produce different thresholds",
    () => {
      const runTest = (model: AttackerModel) =>
        TimingOracle.forAttacker(model)
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

      const resultShared = runTest(AttackerModel.SharedHardware);
      const resultAdjacent = runTest(AttackerModel.AdjacentNetwork);
      const resultRemote = runTest(AttackerModel.RemoteNetwork);

      console.log("\n[attacker_model_thresholds]");
      console.log(`SharedHardware theta: ${resultShared.thetaUserNs}ns`);
      console.log(`AdjacentNetwork theta: ${resultAdjacent.thetaUserNs}ns`);
      console.log(`RemoteNetwork theta: ${resultRemote.thetaUserNs}ns`);

      // Different models should have different thresholds
      // SharedHardware (0.6ns) < AdjacentNetwork (100ns) < RemoteNetwork (50000ns)
      expect(resultShared.thetaUserNs).toBeLessThan(resultAdjacent.thetaUserNs);
      expect(resultAdjacent.thetaUserNs).toBeLessThan(resultRemote.thetaUserNs);
    },
    30_000
  );
});
