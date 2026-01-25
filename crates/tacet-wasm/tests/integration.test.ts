/**
 * Integration tests - actually run timing measurements.
 */

import { describe, test, expect, beforeAll } from "bun:test";
import {
  initializeWasm,
  TimingOracle,
  AttackerModelValues,
  OutcomeValues,
  collectSamples,
  calibrateTimer,
  analyze,
} from "../dist/index.js";

describe("Full Measurement Flow", () => {
  beforeAll(async () => {
    await initializeWasm();
  });

  test("collectSamples works", () => {
    const result = collectSamples(
      100, // small sample for speed
      () => new Uint8Array(32).fill(0),
      () => crypto.getRandomValues(new Uint8Array(32)),
      (input) => {
        // Simple operation: XOR all bytes
        let sum = 0;
        for (const b of input) sum ^= b;
        return sum;
      }
    );

    expect(result.baseline).toBeInstanceOf(BigInt64Array);
    expect(result.sample).toBeInstanceOf(BigInt64Array);
    expect(result.baseline.length).toBe(100);
    expect(result.sample.length).toBe(100);
    expect(result.batchingInfo).toBeDefined();
    expect(result.timerInfo).toBeDefined();
  });

  test("analyze works on collected samples", () => {
    const timerInfo = calibrateTimer();

    // Collect some samples
    const samples = collectSamples(
      500,
      () => new Uint8Array(32).fill(0),
      () => crypto.getRandomValues(new Uint8Array(32)),
      (input) => {
        let sum = 0;
        for (const b of input) sum ^= b;
        return sum;
      }
    );

    // Run analysis
    const result = analyze(
      samples.baseline,
      samples.sample,
      {
        attackerModel: AttackerModelValues.AdjacentNetwork,
        maxSamples: 100000,
        timeBudgetMs: 30000,
        passThreshold: 0.05,
        failThreshold: 0.95,
        seed: undefined,
        customThresholdNs: undefined,
      },
      timerInfo.frequencyHz
    );

    expect(result).toBeDefined();
    expect(result.outcome).toBeDefined();
    expect(typeof result.leakProbability).toBe("number");
    expect(result.leakProbability).toBeGreaterThanOrEqual(0);
    expect(result.leakProbability).toBeLessThanOrEqual(1);
  });

  test("TimingOracle.test() runs full flow", () => {
    const result = TimingOracle
      .forAttacker(AttackerModelValues.AdjacentNetwork)
      .timeBudget(5000) // 5 seconds max
      .maxSamples(10000) // limit samples for speed
      .test(
        {
          baseline: () => new Uint8Array(32).fill(0),
          sample: () => crypto.getRandomValues(new Uint8Array(32)),
        },
        (input) => {
          // Constant-time XOR - should pass
          let sum = 0;
          for (const b of input) sum ^= b;
          return sum;
        }
      );

    expect(result).toBeDefined();
    expect(result.outcome).toBeDefined();
    expect([
      OutcomeValues.Pass,
      OutcomeValues.Fail,
      OutcomeValues.Inconclusive,
      OutcomeValues.Unmeasurable,
    ]).toContain(result.outcome);

    expect(typeof result.leakProbability).toBe("number");
    expect(result.samplesUsed).toBeGreaterThan(0);
    expect(result.batchingInfo).toBeDefined();
    expect(result.timerInfo).toBeDefined();

    // Check helper methods work
    expect(typeof result.isPass()).toBe("boolean");
    expect(typeof result.isFail()).toBe("boolean");
    expect(typeof result.leakProbabilityPercent()).toBe("string");
    expect(typeof result.toString()).toBe("string");
  });
});
