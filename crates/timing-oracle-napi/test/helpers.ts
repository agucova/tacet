/**
 * Test helper functions for timing-oracle integration tests.
 *
 * Provides both leaky and constant-time operations for validation testing.
 */

import {
  Outcome,
  InconclusiveReason,
  EffectPattern,
  Exploitability,
  MeasurementQuality,
} from "../dist/index.js";

/**
 * Convert Outcome enum value to string (const enums don't have reverse mappings).
 */
export function outcomeName(outcome: Outcome): string {
  switch (outcome) {
    case Outcome.Pass:
      return "Pass";
    case Outcome.Fail:
      return "Fail";
    case Outcome.Inconclusive:
      return "Inconclusive";
    case Outcome.Unmeasurable:
      return "Unmeasurable";
    default:
      return `Unknown(${outcome})`;
  }
}

/**
 * Convert InconclusiveReason enum value to string.
 */
export function inconclusiveReasonName(reason: InconclusiveReason): string {
  switch (reason) {
    case InconclusiveReason.None:
      return "None";
    case InconclusiveReason.DataTooNoisy:
      return "DataTooNoisy";
    case InconclusiveReason.NotLearning:
      return "NotLearning";
    case InconclusiveReason.WouldTakeTooLong:
      return "WouldTakeTooLong";
    case InconclusiveReason.TimeBudgetExceeded:
      return "TimeBudgetExceeded";
    case InconclusiveReason.SampleBudgetExceeded:
      return "SampleBudgetExceeded";
    case InconclusiveReason.ConditionsChanged:
      return "ConditionsChanged";
    case InconclusiveReason.ThresholdElevated:
      return "ThresholdElevated";
    default:
      return `Unknown(${reason})`;
  }
}

/**
 * Convert EffectPattern enum value to string.
 */
export function effectPatternName(pattern: EffectPattern): string {
  switch (pattern) {
    case EffectPattern.UniformShift:
      return "UniformShift";
    case EffectPattern.TailEffect:
      return "TailEffect";
    case EffectPattern.Mixed:
      return "Mixed";
    case EffectPattern.Indeterminate:
      return "Indeterminate";
    default:
      return `Unknown(${pattern})`;
  }
}

/**
 * Convert Exploitability enum value to string.
 */
export function exploitabilityName(exploitability: Exploitability): string {
  switch (exploitability) {
    case Exploitability.SharedHardwareOnly:
      return "SharedHardwareOnly";
    case Exploitability.Http2Multiplexing:
      return "Http2Multiplexing";
    case Exploitability.StandardRemote:
      return "StandardRemote";
    case Exploitability.ObviousLeak:
      return "ObviousLeak";
    default:
      return `Unknown(${exploitability})`;
  }
}

/**
 * Convert MeasurementQuality enum value to string.
 */
export function measurementQualityName(quality: MeasurementQuality): string {
  switch (quality) {
    case MeasurementQuality.Excellent:
      return "Excellent";
    case MeasurementQuality.Good:
      return "Good";
    case MeasurementQuality.Poor:
      return "Poor";
    case MeasurementQuality.TooNoisy:
      return "TooNoisy";
    default:
      return `Unknown(${quality})`;
  }
}

/**
 * Generate a Uint8Array filled with zeros.
 */
export function generateZeros(size: number): Uint8Array {
  return new Uint8Array(size);
}

/**
 * Generate a Uint8Array filled with random bytes.
 */
export function generateRandom(size: number): Uint8Array {
  const arr = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = Math.floor(Math.random() * 256);
  }
  return arr;
}

/**
 * LEAKY: Early-exit comparison.
 *
 * This function has a timing side-channel because it returns early
 * when a mismatch is found. The timing reveals information about
 * which byte position differs.
 */
export function leakyCompare(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false; // Early exit - TIMING LEAK!
    }
  }
  return true;
}

/**
 * SAFE: Constant-time XOR fold.
 *
 * XORs all bytes together. This operation is constant-time because
 * XOR takes the same time regardless of operand values, and we always
 * process all bytes.
 */
export function xorFold(data: Uint8Array): number {
  let acc = 0;
  for (let i = 0; i < data.length; i++) {
    acc ^= data[i];
  }
  return acc;
}

/**
 * SAFE: Constant-time comparison using OR accumulator.
 *
 * Compares two arrays in constant time by XORing corresponding bytes
 * and ORing the results into an accumulator. We always process all
 * bytes regardless of whether a mismatch has been found.
 */
export function constantTimeCompare(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) {
    return false;
  }
  let acc = 0;
  for (let i = 0; i < a.length; i++) {
    acc |= a[i] ^ b[i];
  }
  return acc === 0;
}

/**
 * LEAKY: Branch on secret bit.
 *
 * This function has a timing side-channel because it executes different
 * code paths based on the value of the first bit.
 */
export function branchOnSecretBit(data: Uint8Array): number {
  if ((data[0] & 0x80) !== 0) {
    // Expensive branch for high bit set
    let x = 0;
    for (let i = 0; i < 1000; i++) {
      x += i;
    }
    return x;
  } else {
    // Fast path for high bit clear
    return 0;
  }
}

/**
 * LEAKY: Variable-time loop based on data.
 *
 * The number of iterations depends on the first byte value.
 */
export function variableTimeLoop(data: Uint8Array): number {
  const iterations = data[0];
  let sum = 0;
  for (let i = 0; i < iterations; i++) {
    sum += i;
  }
  return sum;
}

/**
 * SAFE: Constant-time wrapping sum.
 *
 * Addition is constant-time and we always process all bytes.
 */
export function wrappingSum(data: Uint8Array): number {
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    sum = (sum + data[i]) | 0; // Wrapping 32-bit addition
  }
  return sum;
}

/**
 * Non-pathological fixed pattern (mixed bits) for baseline testing.
 * Using a deterministic pattern avoids issues with all-zeros or all-ones.
 */
export function generateFixedPattern(size: number): Uint8Array {
  const arr = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    // Deterministic mixed-bit pattern
    arr[i] = ((i * 0x9d) + 0x37) & 0xff;
  }
  return arr;
}

/**
 * Add a constant delay to all operations (for testing UniformShift pattern).
 */
export function busyWait(ns: number): void {
  const start = performance.now();
  const targetMs = ns / 1_000_000;
  while (performance.now() - start < targetMs) {
    // Busy wait
  }
}

/**
 * Prevent dead code elimination.
 */
export function blackBox<T>(value: T): T {
  return value;
}

/**
 * Run an operation multiple times to make it measurable with coarse timers.
 */
export function repeated<T>(
  iterations: number,
  fn: () => T
): () => T {
  return () => {
    let result: T;
    for (let i = 0; i < iterations; i++) {
      result = fn();
    }
    return result!;
  };
}
