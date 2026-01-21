/**
 * Basic tests for timing-oracle Node.js bindings.
 *
 * Run with: bun test
 */

import { expect, test, describe } from "bun:test";
import {
  rdtsc,
  calibrateTimer,
  version,
  AttackerModel,
  Outcome,
  defaultConfig,
  collectSamples,
  TimingOracle,
} from "../dist/index.js";

describe("Timer functions", () => {
  test("rdtsc returns positive number", () => {
    const tsc = rdtsc();
    expect(typeof tsc).toBe("number");
    expect(tsc).toBeGreaterThan(0);
  });

  test("rdtsc is monotonic", () => {
    const t1 = rdtsc();
    const t2 = rdtsc();
    expect(t2).toBeGreaterThanOrEqual(t1);
  });

  test("calibrateTimer returns valid info", () => {
    const info = calibrateTimer();

    expect(info.cyclesPerNs).toBeGreaterThan(0);
    expect(info.resolutionNs).toBeGreaterThan(0);
    expect(info.frequencyHz).toBeGreaterThan(0);
  });
});

describe("Version and config", () => {
  test("version returns string", () => {
    const v = version();
    expect(typeof v).toBe("string");
    expect(v.length).toBeGreaterThan(0);
  });

  test("defaultConfig returns valid config", () => {
    const config = defaultConfig(AttackerModel.AdjacentNetwork);

    expect(config.timeBudgetMs).toBeGreaterThan(0);
    expect(config.maxSamples).toBeGreaterThan(0);
    expect(config.passThreshold).toBeGreaterThan(0);
    expect(config.passThreshold).toBeLessThan(1);
    expect(config.failThreshold).toBeGreaterThan(0);
    expect(config.failThreshold).toBeLessThan(1);
  });
});

describe("Measurement loop", () => {
  test("collectSamples collects data", () => {
    const samples = collectSamples(
      100, // 100 samples per class
      () => 0, // baseline: returns 0
      () => 1, // sample: returns 1
      (_input) => {
        // Simple operation
        let x = 0;
        for (let i = 0; i < 100; i++) x += i;
        return x;
      }
    );

    expect(samples.baseline.length).toBe(100);
    expect(samples.sample.length).toBe(100);
    expect(samples.batchingInfo.k).toBeGreaterThanOrEqual(1);
    expect(samples.timerInfo.frequencyHz).toBeGreaterThan(0);
  });
});

describe("TimingOracle", () => {
  test("builder pattern works", () => {
    const oracle = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
      .timeBudget(5000)
      .maxSamples(10000)
      .passThreshold(0.05)
      .failThreshold(0.95);

    expect(oracle).toBeDefined();
  });

  test("test runs and returns result", () => {
    // Use a very fast test with minimal samples
    const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
      .timeBudget(2000) // 2 second max
      .maxSamples(5000) // Quick test
      .test(
        {
          baseline: () => 0,
          sample: () => 1,
        },
        (_input) => {
          // Operation with no timing difference
          let x = 0;
          for (let i = 0; i < 50; i++) x += i;
        }
      );

    // Should return a valid result
    expect(result.outcome).toBeDefined();
    expect(result.leakProbability).toBeGreaterThanOrEqual(0);
    expect(result.leakProbability).toBeLessThanOrEqual(1);
    expect(result.batchingInfo).toBeDefined();
    expect(result.timerInfo).toBeDefined();
  });
});
