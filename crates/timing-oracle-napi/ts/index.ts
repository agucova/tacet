/**
 * @timing-oracle/js - Timing side-channel detection for Node.js/Bun
 *
 * Detect timing leaks in cryptographic code using statistical analysis.
 *
 * @example
 * ```typescript
 * import { TimingOracle, AttackerModel, Outcome } from '@timing-oracle/js';
 * import crypto from 'crypto';
 *
 * const result = TimingOracle
 *   .forAttacker(AttackerModel.AdjacentNetwork)
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
 *   case Outcome.Pass:
 *     console.log(`No leak detected: P(leak) = ${result.leakProbability}`);
 *     break;
 *   case Outcome.Fail:
 *     console.error(`Timing leak detected! Exploitability: ${result.exploitability}`);
 *     process.exit(1);
 *     break;
 *   case Outcome.Inconclusive:
 *     console.warn(`Inconclusive: ${result.inconclusiveReason}`);
 *     break;
 * }
 * ```
 *
 * @packageDocumentation
 */

// Re-export everything from oracle (which includes collector and native re-exports)
export {
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
  type BatchingInfo,
  type CollectedSamples,
  // Enums
  AttackerModel,
  Outcome,
  InconclusiveReason,
  EffectPattern,
  Exploitability,
  MeasurementQuality,
  // Types
  type Config,
  type AnalysisResult,
  type EffectEstimate,
  type Diagnostics,
  type TimerInfo,
  type AdaptiveStepResult,
  // Low-level functions
  rdtsc,
  calibrateTimer,
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
