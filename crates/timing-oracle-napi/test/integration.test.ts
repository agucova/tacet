/**
 * Integration tests for known leaky and known safe operations.
 *
 * These tests validate that the timing oracle correctly detects timing leaks
 * in vulnerable code and does not produce false positives on constant-time code.
 *
 * Test patterns follow DudeCT's two-class methodology:
 * - Baseline: Fixed pattern (all zeros or deterministic)
 * - Sample: Random data
 */

import { expect, test, describe } from "bun:test";
import {
  TimingOracle,
  AttackerModel,
  Outcome,
} from "../dist/index.js";
import {
  leakyCompare,
  xorFold,
  constantTimeCompare,
  branchOnSecretBit,
  generateZeros,
  generateRandom,
  generateFixedPattern,
  blackBox,
  outcomeName,
} from "./helpers.js";

// Size for test arrays - larger arrays make timing differences more pronounced
const ARRAY_SIZE = 512;

// Number of iterations to make operations measurable with coarse timers
const ITERATIONS = 200;

describe("Known leaky operations", () => {
  test(
    "detects early-exit comparison",
    () => {
      // Baseline: zeros (will match early bytes of secret zeros, then exit)
      // Sample: random (will differ at random position, earlier exit on average)
      const secret = generateZeros(ARRAY_SIZE);

      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .passThreshold(0.01) // Very hard to falsely pass (we expect leaks)
        .failThreshold(0.85) // Quick to detect leaks
        .timeBudget(15_000) // 15 seconds
        .maxSamples(50_000)
        .test(
          {
            baseline: () => generateZeros(ARRAY_SIZE), // Matches secret
            sample: () => generateRandom(ARRAY_SIZE), // Differs early
          },
          (input) => {
            // Early-exit comparison is LEAKY
            blackBox(leakyCompare(secret, input));
          }
        );

      console.log("\n[detects_early_exit_comparison]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Leak probability: ${(result.leakProbability * 100).toFixed(2)}%`);
      console.log(`Effect: shift=${result.effect.shiftNs.toFixed(2)}ns, tail=${result.effect.tailNs.toFixed(2)}ns`);

      // Skip if unmeasurable (operation too fast for this platform)
      if (result.outcome === Outcome.Unmeasurable) {
        console.log(`[SKIPPED] Operation unmeasurable: ${result.recommendation}`);
        return;
      }

      // For known leaky code, we expect Fail
      expect(result.outcome).toBe(Outcome.Fail);
      expect(result.leakProbability).toBeGreaterThan(0.85);
    },
    30_000 // 30s timeout
  );

  test(
    "detects branch on secret bit",
    () => {
      // Baseline: high bit clear (fast path)
      // Sample: random (50% chance high bit set - slow path)
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .passThreshold(0.01)
        .failThreshold(0.85)
        .timeBudget(15_000)
        .maxSamples(50_000)
        .test(
          {
            baseline: () => {
              const arr = generateZeros(32);
              arr[0] = 0x00; // High bit clear
              return arr;
            },
            sample: () => {
              const arr = generateRandom(32);
              arr[0] |= 0x80; // High bit set (forces slow path)
              return arr;
            },
          },
          (input) => {
            // Branch on secret bit is LEAKY
            blackBox(branchOnSecretBit(input));
          }
        );

      console.log("\n[detects_branch_on_secret_bit]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Leak probability: ${(result.leakProbability * 100).toFixed(2)}%`);

      // Skip if unmeasurable
      if (result.outcome === Outcome.Unmeasurable) {
        console.log(`[SKIPPED] Operation unmeasurable: ${result.recommendation}`);
        return;
      }

      expect(result.outcome).toBe(Outcome.Fail);
      expect(result.leakProbability).toBeGreaterThan(0.85);
    },
    30_000
  );

  test(
    "detects early-exit with larger size difference",
    () => {
      // Using a 1KB array to make the timing difference more pronounced
      const secret = generateZeros(1024);

      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .passThreshold(0.01)
        .failThreshold(0.85)
        .timeBudget(15_000)
        .maxSamples(50_000)
        .test(
          {
            baseline: () => generateZeros(1024), // Matches all bytes
            sample: () => generateRandom(1024), // Exits early
          },
          (input) => {
            blackBox(leakyCompare(secret, input));
          }
        );

      console.log("\n[detects_early_exit_large]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Leak probability: ${(result.leakProbability * 100).toFixed(2)}%`);

      if (result.outcome === Outcome.Unmeasurable) {
        console.log(`[SKIPPED] Operation unmeasurable: ${result.recommendation}`);
        return;
      }

      expect(result.outcome).toBe(Outcome.Fail);
      expect(result.leakProbability).toBeGreaterThan(0.85);
    },
    30_000
  );
});

describe("Known safe operations", () => {
  test(
    "no false positive on XOR fold",
    () => {
      // DudeCT pattern: fixed (mixed bits) vs random
      // XOR is constant-time - timing should not depend on data values
      const fixedPattern = generateFixedPattern(64);

      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .passThreshold(0.15) // More lenient for safe code
        .failThreshold(0.99) // Very hard to falsely fail
        .timeBudget(15_000)
        .maxSamples(50_000)
        .test(
          {
            baseline: () => fixedPattern,
            sample: () => generateRandom(64),
          },
          (input) => {
            // XOR fold repeated to be measurable
            for (let i = 0; i < ITERATIONS; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[no_false_positive_xor_fold]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Leak probability: ${(result.leakProbability * 100).toFixed(2)}%`);
      console.log(`Effect: shift=${result.effect.shiftNs.toFixed(2)}ns, tail=${result.effect.tailNs.toFixed(2)}ns`);

      if (result.outcome === Outcome.Unmeasurable) {
        console.log(`[SKIPPED] Operation unmeasurable: ${result.recommendation}`);
        return;
      }

      // For constant-time code, we should NOT get Fail
      expect(result.outcome).not.toBe(Outcome.Fail);

      if (result.outcome === Outcome.Inconclusive) {
        console.log(`[INCONCLUSIVE] Reason: ${result.inconclusiveReason}`);
      }
    },
    30_000
  );

  test(
    "no false positive on constant-time compare",
    () => {
      const secret = generateFixedPattern(64);

      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .passThreshold(0.15)
        .failThreshold(0.99)
        .timeBudget(15_000)
        .maxSamples(50_000)
        .test(
          {
            baseline: () => generateFixedPattern(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            // OR-accumulator pattern is constant-time
            for (let i = 0; i < ITERATIONS; i++) {
              blackBox(constantTimeCompare(secret, input));
            }
          }
        );

      console.log("\n[no_false_positive_ct_compare]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Leak probability: ${(result.leakProbability * 100).toFixed(2)}%`);

      if (result.outcome === Outcome.Unmeasurable) {
        console.log(`[SKIPPED] Operation unmeasurable: ${result.recommendation}`);
        return;
      }

      expect(result.outcome).not.toBe(Outcome.Fail);

      if (result.outcome === Outcome.Inconclusive) {
        console.log(`[INCONCLUSIVE] Reason: ${result.inconclusiveReason}`);
      }
    },
    30_000
  );

  test(
    "no false positive on identical operations",
    () => {
      // Both baseline and sample do the exact same thing
      // This should always pass - no timing difference possible
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .passThreshold(0.15)
        .failThreshold(0.99)
        .timeBudget(10_000)
        .maxSamples(30_000)
        .test(
          {
            baseline: () => generateRandom(64),
            sample: () => generateRandom(64),
          },
          (input) => {
            // Identical operation for both classes
            for (let i = 0; i < ITERATIONS; i++) {
              blackBox(xorFold(input));
            }
          }
        );

      console.log("\n[no_false_positive_identical]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Leak probability: ${(result.leakProbability * 100).toFixed(2)}%`);

      if (result.outcome === Outcome.Unmeasurable) {
        console.log(`[SKIPPED] Operation unmeasurable: ${result.recommendation}`);
        return;
      }

      expect(result.outcome).not.toBe(Outcome.Fail);
    },
    20_000
  );

  test(
    "no false positive on wrapping arithmetic",
    () => {
      // Wrapping addition is constant-time on all architectures
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .passThreshold(0.15)
        .failThreshold(0.99)
        .timeBudget(15_000)
        .maxSamples(50_000)
        .test(
          {
            baseline: () => generateFixedPattern(256),
            sample: () => generateRandom(256),
          },
          (input) => {
            let sum = 0;
            for (let iter = 0; iter < ITERATIONS; iter++) {
              for (let i = 0; i < input.length; i++) {
                sum = (sum + input[i]) | 0;
              }
            }
            blackBox(sum);
          }
        );

      console.log("\n[no_false_positive_wrapping]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Leak probability: ${(result.leakProbability * 100).toFixed(2)}%`);

      if (result.outcome === Outcome.Unmeasurable) {
        console.log(`[SKIPPED] Operation unmeasurable: ${result.recommendation}`);
        return;
      }

      expect(result.outcome).not.toBe(Outcome.Fail);
    },
    30_000
  );
});

describe("Edge cases", () => {
  test(
    "handles very fast operations gracefully",
    () => {
      // Very simple operation that may be unmeasurable
      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(5_000)
        .maxSamples(10_000)
        .test(
          {
            baseline: () => 0,
            sample: () => 1,
          },
          (input) => {
            blackBox(input + 1);
          }
        );

      console.log("\n[handles_fast_operations]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);

      // Should return some valid outcome
      expect(result.outcome).toBeDefined();
      expect(result.leakProbability).toBeGreaterThanOrEqual(0);
      expect(result.leakProbability).toBeLessThanOrEqual(1);
    },
    15_000
  );

  test(
    "works with larger arrays",
    () => {
      const size = 4096;
      const secret = generateFixedPattern(size);

      const result = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
        .timeBudget(10_000)
        .maxSamples(30_000)
        .test(
          {
            baseline: () => generateFixedPattern(size),
            sample: () => generateRandom(size),
          },
          (input) => {
            blackBox(constantTimeCompare(secret, input));
          }
        );

      console.log("\n[handles_large_arrays]");
      console.log(`Outcome: ${outcomeName(result.outcome)}`);
      console.log(`Samples used: ${result.samplesUsed}`);

      expect(result.outcome).toBeDefined();
      expect(result.samplesUsed).toBeGreaterThan(0);
    },
    20_000
  );
});
