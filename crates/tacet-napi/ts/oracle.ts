/**
 * High-level TimingOracle API.
 *
 * Provides a builder-pattern interface for timing side-channel detection.
 */

import {
  type Config,
  type AnalysisResult,
  AttackerModel,
  analyze,
  calibrateSamples,
  adaptiveStepBatch,
  AdaptiveState,
  defaultConfig,
} from "../index.js";

import {
  collectSamples,
  collectBatches,
  type BatchingInfo,
  type TimerInfo,
} from "./collector.js";

import { TimingTestResult } from "./result.js";

/** Input pair generators for baseline and sample classes. */
export interface InputPair<T> {
  /** Generate baseline input (typically all zeros). */
  baseline: () => T;
  /** Generate sample input (typically random). */
  sample: () => T;
}

/** Extended result with additional metadata (deprecated, use TimingTestResult). */
export interface TestResult extends AnalysisResult {
  /** Batching info from pilot phase. */
  batchingInfo: BatchingInfo;
  /** Timer calibration info. */
  timerInfo: TimerInfo;
}

/**
 * TimingOracle - Builder-pattern API for timing side-channel detection.
 *
 * @example
 * ```typescript
 * import { TimingOracle, AttackerModel, Outcome } from '@tacet/js';
 * import crypto from 'crypto';
 *
 * const result = TimingOracle
 *   .forAttacker(AttackerModel.AdjacentNetwork)
 *   .timeBudget(30_000)  // 30 seconds
 *   .maxSamples(100_000)
 *   .test(
 *     {
 *       baseline: () => Buffer.alloc(32, 0),
 *       sample: () => crypto.randomBytes(32),
 *     },
 *     (input) => myCryptoFunction(input)
 *   );
 *
 * if (result.outcome === Outcome.Fail) {
 *   console.log(`Timing leak detected: ${result.exploitability}`);
 * }
 * ```
 */
export class TimingOracle {
  private config: Config;
  private _showProgress = false;

  private constructor(attackerModel: AttackerModel) {
    this.config = defaultConfig(attackerModel);
  }

  /**
   * Create a TimingOracle for a specific attacker model.
   *
   * @param model The attacker model (determines threshold theta)
   *
   * Available models:
   * - `SharedHardware` (0.6ns) - SGX, cross-VM, containers
   * - `PostQuantum` (3.3ns) - Post-quantum crypto
   * - `AdjacentNetwork` (100ns) - LAN, HTTP/2
   * - `RemoteNetwork` (50μs) - General internet
   * - `Research` (0) - Detect any difference
   */
  static forAttacker(model: AttackerModel): TimingOracle {
    return new TimingOracle(model);
  }

  /**
   * Set a custom threshold in nanoseconds.
   *
   * This overrides the threshold from the attacker model.
   */
  customThreshold(thresholdNs: number): this {
    this.config.customThresholdNs = thresholdNs;
    return this;
  }

  /**
   * Set the time budget in milliseconds.
   *
   * Default: 30,000 (30 seconds)
   */
  timeBudget(ms: number): this {
    this.config.timeBudgetMs = ms;
    return this;
  }

  /**
   * Set the maximum samples per class.
   *
   * Default: 100,000
   */
  maxSamples(n: number): this {
    this.config.maxSamples = n;
    return this;
  }

  /**
   * Set the pass threshold for leak probability.
   *
   * If P(leak) < passThreshold, the test passes.
   * Default: 0.05 (5%)
   */
  passThreshold(p: number): this {
    this.config.passThreshold = p;
    return this;
  }

  /**
   * Set the fail threshold for leak probability.
   *
   * If P(leak) > failThreshold, the test fails.
   * Default: 0.95 (95%)
   */
  failThreshold(p: number): this {
    this.config.failThreshold = p;
    return this;
  }

  /**
   * Set the random seed for reproducibility.
   */
  seed(s: number): this {
    this.config.seed = s;
    return this;
  }

  /**
   * Enable progress output to stderr.
   *
   * When enabled, prints progress updates during the test:
   * `[12.3s] P(leak)=2.1%, samples=45000`
   *
   * @param enabled Whether to show progress (default: true)
   */
  showProgress(enabled: boolean = true): this {
    this._showProgress = enabled;
    return this;
  }

  /**
   * Run the timing test.
   *
   * This is the main entry point. It:
   * 1. Runs pilot phase to detect batch size K
   * 2. Collects calibration samples
   * 3. Runs adaptive sampling until decision or budget exceeded
   *
   * @param inputs Input generators for baseline and sample classes
   * @param operation The operation to test
   * @returns Test result with outcome, effect size, and diagnostics
   */
  test<T>(inputs: InputPair<T>, operation: (input: T) => void): TimingTestResult {
    const startTime = performance.now();
    const timeBudgetMs = this.config.timeBudgetMs;
    const maxSamples = this.config.maxSamples;

    // Calibration phase: collect initial samples
    const calibrationSamples = 5000;
    const calibrationResult = collectSamples(
      calibrationSamples,
      inputs.baseline,
      inputs.sample,
      operation
    );

    const timerInfo = calibrationResult.timerInfo;
    const batchingInfo = calibrationResult.batchingInfo;

    // Accumulate all samples for final analysis if needed
    const allBaseline: bigint[] = Array.from(calibrationResult.baseline);
    const allSample: bigint[] = Array.from(calibrationResult.sample);

    // Run calibration
    const calibration = calibrateSamples(
      calibrationResult.baseline,
      calibrationResult.sample,
      this.config,
      timerInfo.frequencyHz
    );

    // Create adaptive state
    const state = new AdaptiveState();

    // Add calibration samples to state
    const elapsedSecs = (performance.now() - startTime) / 1000;
    let stepResult = adaptiveStepBatch(
      calibration,
      state,
      calibrationResult.baseline,
      calibrationResult.sample,
      this.config,
      elapsedSecs
    );

    // Check if we got a decision from calibration samples alone
    if (stepResult.isDecision && stepResult.result) {
      if (this._showProgress) {
        process.stderr.write("\n");
      }
      return new TimingTestResult(stepResult.result, batchingInfo, timerInfo);
    }

    // Adaptive loop
    const batchSize = 1000;
    const batchGenerator = collectBatches(
      batchSize,
      inputs.baseline,
      inputs.sample,
      operation
    );

    let totalSamples = calibrationSamples;

    for (const batch of batchGenerator) {
      totalSamples += batchSize;
      const currentElapsed = (performance.now() - startTime) / 1000;

      // Check budgets
      if (currentElapsed * 1000 >= timeBudgetMs) {
        break;
      }
      if (totalSamples >= maxSamples) {
        break;
      }

      // Accumulate samples for final analysis
      for (const v of batch.baseline) allBaseline.push(v);
      for (const v of batch.sample) allSample.push(v);

      // Run adaptive step
      stepResult = adaptiveStepBatch(
        calibration,
        state,
        batch.baseline,
        batch.sample,
        this.config,
        currentElapsed
      );

      // Show progress if enabled
      if (this._showProgress) {
        const pct = (stepResult.currentProbability * 100).toFixed(1);
        const n = state.totalBaseline;
        const elapsed = currentElapsed.toFixed(1);
        process.stderr.write(`\r[${elapsed}s] P(leak)=${pct}%, samples=${n}`);
      }

      if (stepResult.isDecision && stepResult.result) {
        if (this._showProgress) {
          process.stderr.write("\n");
        }
        return new TimingTestResult(stepResult.result, batchingInfo, timerInfo);
      }
    }

    // Budget exceeded - run final analysis on ALL collected samples
    if (this._showProgress) {
      process.stderr.write("\n");
    }

    const finalBaseline = new BigInt64Array(allBaseline);
    const finalSample = new BigInt64Array(allSample);

    const finalResult = analyze(
      finalBaseline,
      finalSample,
      this.config,
      timerInfo.frequencyHz
    );

    return new TimingTestResult(finalResult, batchingInfo, timerInfo);
  }

  /**
   * Analyze pre-collected timing data.
   *
   * Use this if you've already collected timing samples externally.
   *
   * @param baseline Baseline timing samples (raw ticks)
   * @param sample Sample timing samples (raw ticks)
   * @param timerFrequencyHz Timer frequency in Hz
   * @returns Analysis result
   */
  analyze(
    baseline: BigInt64Array,
    sample: BigInt64Array,
    timerFrequencyHz: number
  ): AnalysisResult {
    return analyze(baseline, sample, this.config, timerFrequencyHz);
  }
}

// Re-export types and enums from native module
export {
  AttackerModel,
  Outcome,
  InconclusiveReason,
  EffectPattern,
  Exploitability,
  MeasurementQuality,
  type Config,
  type AnalysisResult,
  type EffectEstimate,
  type Diagnostics,
} from "../index.js";

// Re-export low-level functions from native module
export {
  rdtsc,
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
  type AdaptiveStepResult,
} from "../index.js";

// Re-export collector (includes calibrateTimer and TimerInfo)
export {
  collectSamples,
  collectBatches,
  calibrateTimer,
  type BatchingInfo,
  type CollectedSamples,
  type TimerInfo,
} from "./collector.js";

// Re-export result wrapper
export { TimingTestResult } from "./result.js";

// Re-export error classes
export {
  TimingOracleError,
  TimingLeakError,
  CalibrationError,
  InsufficientSamplesError,
} from "./errors.js";
