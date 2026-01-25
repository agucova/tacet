/**
 * Typed error classes for timing-oracle.
 */

// Forward declaration to avoid circular import
// TimingTestResult is imported dynamically where needed
export interface TimingTestResultLike {
  leakProbabilityPercent(): string;
  totalEffectNs(): number;
  exploitabilityString(): string;
}

/**
 * Base error class for all timing-oracle errors.
 */
export class TimingOracleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TimingOracleError";
  }
}

/**
 * Thrown when a timing leak is detected.
 *
 * Contains the full result for inspection.
 *
 * @example
 * ```typescript
 * try {
 *   result.assertNoLeak();
 * } catch (e) {
 *   if (e instanceof TimingLeakError) {
 *     console.error(`Leak: ${e.result.exploitabilityString()}`);
 *   }
 * }
 * ```
 */
export class TimingLeakError extends TimingOracleError {
  constructor(readonly result: TimingTestResultLike) {
    super(
      `Timing leak detected: P(leak)=${result.leakProbabilityPercent()}, ` +
        `effect=${result.totalEffectNs().toFixed(1)}ns (${result.exploitabilityString()})`,
    );
    this.name = "TimingLeakError";
  }
}

/**
 * Thrown when calibration fails.
 */
export class CalibrationError extends TimingOracleError {
  constructor(message: string) {
    super(`Calibration failed: ${message}`);
    this.name = "CalibrationError";
  }
}

/**
 * Thrown when there are insufficient samples for analysis.
 */
export class InsufficientSamplesError extends TimingOracleError {
  constructor(
    readonly baseline: number,
    readonly sample: number,
    readonly required: number = 100,
  ) {
    super(
      `Insufficient samples: need ${required}, got ${baseline} baseline and ${sample} sample`,
    );
    this.name = "InsufficientSamplesError";
  }
}
