/**
 * Result wrapper class with helper methods.
 */

import type { AnalysisResult, EffectEstimate, Diagnostics } from "./types.js";
import type {
  Outcome,
  InconclusiveReason,
  Exploitability,
  MeasurementQuality,
  EffectPattern,
} from "./types.js";
import {
  OutcomeValues,
  ExploitabilityValues,
  MeasurementQualityValues,
  InconclusiveReasonValues,
  EffectPatternValues,
} from "./types.js";
import type { BatchingInfo, TimerInfo } from "./collector.js";
import { TimingLeakError } from "./errors.js";

// Display name mappings for string enum values
const OUTCOME_DISPLAY: Record<Outcome, string> = {
  pass: "Pass",
  fail: "Fail",
  inconclusive: "Inconclusive",
  unmeasurable: "Unmeasurable",
};

const EXPLOITABILITY_DISPLAY: Record<Exploitability, string> = {
  sharedHardwareOnly: "SharedHardwareOnly",
  http2Multiplexing: "Http2Multiplexing",
  standardRemote: "StandardRemote",
  obviousLeak: "ObviousLeak",
};

const QUALITY_DISPLAY: Record<MeasurementQuality, string> = {
  excellent: "Excellent",
  good: "Good",
  poor: "Poor",
  tooNoisy: "TooNoisy",
};

const INCONCLUSIVE_DISPLAY: Record<InconclusiveReason, string> = {
  none: "None",
  dataTooNoisy: "DataTooNoisy",
  notLearning: "NotLearning",
  wouldTakeTooLong: "WouldTakeTooLong",
  timeBudgetExceeded: "TimeBudgetExceeded",
  sampleBudgetExceeded: "SampleBudgetExceeded",
  conditionsChanged: "ConditionsChanged",
  thresholdElevated: "ThresholdElevated",
};

const PATTERN_DISPLAY: Record<EffectPattern, string> = {
  uniformShift: "UniformShift",
  tailEffect: "TailEffect",
  mixed: "Mixed",
  indeterminate: "Indeterminate",
};

/**
 * Wrapper around AnalysisResult with helper methods for ergonomic result handling.
 *
 * @example
 * ```typescript
 * const result = TimingOracle
 *   .forAttacker(AttackerModel.AdjacentNetwork)
 *   .test(inputs, operation);
 *
 * // Check outcome
 * if (result.isFail()) {
 *   console.error(`Leak: ${result.exploitabilityString()}`);
 * }
 *
 * // Or assert directly
 * result.assertNoLeak();  // throws TimingLeakError if Fail
 *
 * // Or log summary
 * console.log(result.toString());
 * ```
 */
export class TimingTestResult {
  constructor(
    private readonly raw: AnalysisResult,
    /** Batching configuration from pilot phase. */
    readonly batchingInfo: BatchingInfo,
    /** Timer calibration info. */
    readonly timerInfo: TimerInfo,
  ) {}

  // Forward all AnalysisResult properties as getters

  /** Test outcome. */
  get outcome(): Outcome {
    return this.raw.outcome;
  }

  /** Leak probability: P(max_k |(X*beta)_k| > theta | data). */
  get leakProbability(): number {
    return this.raw.leakProbability;
  }

  /** Effect size estimate. */
  get effect(): EffectEstimate {
    return this.raw.effect;
  }

  /** Measurement quality. */
  get quality(): MeasurementQuality {
    return this.raw.quality;
  }

  /** Number of samples used per class. */
  get samplesUsed(): number {
    return this.raw.samplesUsed;
  }

  /** Time spent in seconds. */
  get elapsedSecs(): number {
    return this.raw.elapsedSecs;
  }

  /** Exploitability (only meaningful if outcome == Fail). */
  get exploitability(): Exploitability {
    return this.raw.exploitability;
  }

  /** Inconclusive reason (only meaningful if outcome == Inconclusive). */
  get inconclusiveReason(): InconclusiveReason {
    return this.raw.inconclusiveReason;
  }

  /** Minimum detectable effect (shift) in nanoseconds. */
  get mdeShiftNs(): number {
    return this.raw.mdeShiftNs;
  }

  /** Minimum detectable effect (tail) in nanoseconds. */
  get mdeTailNs(): number {
    return this.raw.mdeTailNs;
  }

  /** Timer resolution in nanoseconds. */
  get timerResolutionNs(): number {
    return this.raw.timerResolutionNs;
  }

  /** User's requested threshold (theta) in nanoseconds. */
  get thetaUserNs(): number {
    return this.raw.thetaUserNs;
  }

  /** Effective threshold after floor adjustment in nanoseconds. */
  get thetaEffNs(): number {
    return this.raw.thetaEffNs;
  }

  /** Recommendation string (empty if not applicable). */
  get recommendation(): string {
    return this.raw.recommendation;
  }

  /** Detailed diagnostics. */
  get diagnostics(): Diagnostics {
    return this.raw.diagnostics;
  }

  // Predicate methods

  /** True if outcome is Pass. */
  isPass(): boolean {
    return this.outcome === OutcomeValues.Pass;
  }

  /** True if outcome is Fail. */
  isFail(): boolean {
    return this.outcome === OutcomeValues.Fail;
  }

  /** True if outcome is Inconclusive. */
  isInconclusive(): boolean {
    return this.outcome === OutcomeValues.Inconclusive;
  }

  /** True if outcome is Unmeasurable. */
  isUnmeasurable(): boolean {
    return this.outcome === OutcomeValues.Unmeasurable;
  }

  /** True if outcome is Pass or Fail (a decision was reached). */
  isConclusive(): boolean {
    return this.isPass() || this.isFail();
  }

  // String formatters

  /** Get outcome as a string: 'Pass', 'Fail', 'Inconclusive', or 'Unmeasurable'. */
  outcomeString(): string {
    return OUTCOME_DISPLAY[this.outcome];
  }

  /** Get exploitability as a string. */
  exploitabilityString(): string {
    return EXPLOITABILITY_DISPLAY[this.exploitability];
  }

  /** Get measurement quality as a string. */
  qualityString(): string {
    return QUALITY_DISPLAY[this.quality];
  }

  /** Get inconclusive reason as a string. */
  inconclusiveReasonString(): string {
    return INCONCLUSIVE_DISPLAY[this.inconclusiveReason];
  }

  /** Get effect pattern as a string. */
  effectPatternString(): string {
    return PATTERN_DISPLAY[this.effect.pattern];
  }

  // Helper methods

  /** Format leak probability as a percentage string (e.g., "12.3%"). */
  leakProbabilityPercent(): string {
    return `${(this.leakProbability * 100).toFixed(1)}%`;
  }

  /** Total effect magnitude in nanoseconds (sqrt of shift² + tail²). */
  totalEffectNs(): number {
    return Math.sqrt(this.effect.shiftNs ** 2 + this.effect.tailNs ** 2);
  }

  /**
   * Assert that no timing leak was detected.
   *
   * Throws `TimingLeakError` if outcome is Fail.
   * Returns `this` for chaining if outcome is not Fail.
   *
   * @example
   * ```typescript
   * result.assertNoLeak();  // throws if Fail
   * ```
   */
  assertNoLeak(): this {
    if (this.isFail()) {
      throw new TimingLeakError(this);
    }
    return this;
  }

  /**
   * Get a human-readable summary string.
   *
   * @example
   * ```typescript
   * console.log(result.toString());
   * // → "Pass: P(leak)=1.2%"
   * // → "Fail: P(leak)=99.8%, effect=142.3ns (Http2Multiplexing)"
   * // → "Inconclusive: P(leak)=45.2% (DataTooNoisy)"
   * ```
   */
  toString(): string {
    let msg = `${this.outcomeString()}: P(leak)=${this.leakProbabilityPercent()}`;
    if (this.isFail()) {
      msg += `, effect=${this.totalEffectNs().toFixed(1)}ns (${this.exploitabilityString()})`;
    }
    if (this.isInconclusive()) {
      msg += ` (${this.inconclusiveReasonString()})`;
    }
    return msg;
  }
}
