/**
 * Simple example: Testing crypto operations for timing leaks.
 *
 * Run with: bun examples/simple.ts
 */

import { TimingOracle, AttackerModel, calibrateTimer } from "../dist/index.js";
import crypto from "node:crypto";

// Show timer info
const timerInfo = calibrateTimer();
console.log("Timer info:");
console.log(`  Resolution: ${timerInfo.resolutionNs.toFixed(1)}ns`);
console.log(`  Frequency: ${(timerInfo.frequencyHz / 1e9).toFixed(2)} GHz\n`);

// Test 1: SHA-256 hashing (should PASS - constant-time)
console.log("Test 1: SHA-256 hashing (should PASS)...\n");

const result1 = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
  .timeBudget(15_000) // 15 seconds
  .maxSamples(50_000)
  .showProgress()
  .test(
    {
      baseline: () => Buffer.alloc(64, 0), // All zeros
      sample: () => crypto.randomBytes(64), // Random data
    },
    (input) => {
      // SHA-256 is constant-time
      crypto.createHash("sha256").update(input).digest();
    }
  );

console.log("\nResult:", result1.outcomeString());
console.log("Leak probability:", result1.leakProbabilityPercent());
console.log("Batching:", result1.batchingInfo.rationale);
console.log(`Effect: shift=${result1.effect.shiftNs.toFixed(2)}ns, tail=${result1.effect.tailNs.toFixed(2)}ns`);

if (result1.isPass()) {
  console.log("PASS: No timing leak detected\n");
} else if (result1.isFail()) {
  console.log("FAIL: Timing leak detected!");
  console.log("Exploitability:", result1.exploitabilityString());
} else if (result1.isInconclusive()) {
  console.log("INCONCLUSIVE:", result1.inconclusiveReasonString(), "\n");
} else {
  console.log("UNMEASURABLE:", result1.recommendation, "\n");
}

// Test 2: Early-exit comparison (should FAIL - timing leak)
console.log("---\nTest 2: Early-exit comparison (should FAIL)...\n");

const secret = Buffer.from("supersecretkey!!");

const result2 = TimingOracle.forAttacker(AttackerModel.AdjacentNetwork)
  .timeBudget(15_000)
  .maxSamples(50_000)
  .showProgress()
  .test(
    {
      baseline: () => Buffer.from(secret), // Matches secret (fast path)
      sample: () => crypto.randomBytes(16), // Usually differs early (slow path)
    },
    (input) => {
      // INSECURE: Early-exit comparison (has timing leak!)
      for (let i = 0; i < input.length; i++) {
        if (input[i] !== secret[i]) {
          return; // Early exit on mismatch - TIMING LEAK
        }
      }
    }
  );

console.log("\nResult:", result2.outcomeString());
console.log("Leak probability:", result2.leakProbabilityPercent());
console.log("Batching:", result2.batchingInfo.rationale);
console.log(`Effect: shift=${result2.effect.shiftNs.toFixed(2)}ns, tail=${result2.effect.tailNs.toFixed(2)}ns`);

if (result2.isPass()) {
  console.log("PASS: No timing leak detected (unexpected!)\n");
} else if (result2.isFail()) {
  console.log("FAIL: Timing leak detected! (expected)");
  console.log("Exploitability:", result2.exploitabilityString());
} else if (result2.isInconclusive()) {
  console.log("INCONCLUSIVE:", result2.inconclusiveReasonString(), "\n");
} else {
  console.log("UNMEASURABLE:", result2.recommendation, "\n");
}

// Demonstrate assertion style
console.log("\n---\nTest 3: Assertion style (assertNoLeak)...\n");

try {
  result2.assertNoLeak();
  console.log("No leak detected");
} catch (e) {
  console.log("Caught TimingLeakError:", (e as Error).message);
}
