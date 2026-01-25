/**
 * @tacet/js - Timing side-channel detection for JavaScript/TypeScript
 *
 * Detect timing leaks in cryptographic code using statistical analysis.
 * Works in Node.js, Bun, and Deno (not browsers - they lack high-precision timers).
 *
 * @example
 * ```typescript
 * import { TimingOracle, AttackerModelValues, OutcomeValues } from '@tacet/js';
 * import crypto from 'crypto';
 *
 * const result = TimingOracle
 *   .forAttacker(AttackerModelValues.AdjacentNetwork)
 *   .timeBudget(30_000)
 *   .test(
 *     {
 *       baseline: () => Buffer.alloc(32, 0),
 *       sample: () => crypto.randomBytes(32),
 *     },
 *     (input) => myCryptoFunction(input)
 *   );
 *
 * switch (result.outcome) {
 *   case OutcomeValues.Pass:
 *     console.log(`No leak detected: P(leak) = ${result.leakProbability}`);
 *     break;
 *   case OutcomeValues.Fail:
 *     console.error(`Timing leak detected! Exploitability: ${result.exploitability}`);
 *     process.exit(1);
 *     break;
 *   case OutcomeValues.Inconclusive:
 *     console.warn(`Inconclusive: ${result.inconclusiveReason}`);
 *     break;
 * }
 * ```
 *
 * @packageDocumentation
 */

// Auto-initialize WASM on module load (requires top-level await support)
import { initializeWasm } from "./oracle.js";
await initializeWasm();

// Re-export everything from oracle (which includes collector and all types)
export {
  // WASM initialization
  initializeWasm,
  // High-level API
  TimingOracle,
  TimingTestResult,
  type InputPair,
  type TestResult,
  // Errors
  TimingOracleError,
  TimingLeakError,
  CalibrationError,
  InsufficientSamplesError,
  // Measurement loop
  collectSamples,
  collectBatches,
  calibrateTimer,
  type BatchingInfo,
  type CollectedSamples,
  // Enum value objects (runtime constants)
  AttackerModelValues,
  OutcomeValues,
  InconclusiveReasonValues,
  EffectPatternValues,
  ExploitabilityValues,
  MeasurementQualityValues,
  // Low-level functions
  analyze,
  calibrateSamples,
  adaptiveStepBatch,
  AdaptiveState,
  Calibration,
  version,
  defaultConfig,
  configAdjacentNetwork,
  configSharedHardware,
  configRemoteNetwork,
} from "./oracle.js";

// Re-export types separately
export type {
  AttackerModel,
  Outcome,
  InconclusiveReason,
  EffectPattern,
  Exploitability,
  MeasurementQuality,
  Config,
  AnalysisResult,
  EffectEstimate,
  Diagnostics,
  TimerInfo,
  AdaptiveStepResult,
} from "./oracle.js";
