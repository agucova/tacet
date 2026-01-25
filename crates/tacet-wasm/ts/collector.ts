/**
 * TypeScript measurement loop implementation.
 *
 * This runs entirely in JS/TS with zero FFI overhead.
 * Uses runtime-native high-precision timers (Bun.nanoseconds or process.hrtime).
 */

// Runtime-native high-precision timer (no FFI)
declare const Bun: { nanoseconds: () => number } | undefined;

const isBun = typeof Bun !== "undefined";

/** Get current time in nanoseconds using the fastest available timer. */
const now: () => number = isBun
  ? () => Bun!.nanoseconds()
  : () => Number(process.hrtime.bigint());

/** Timer calibration info. */
export interface TimerInfo {
  /** Cycles per nanosecond (1.0 for JS timers since they return ns directly). */
  cyclesPerNs: number;
  /** Timer resolution in nanoseconds. */
  resolutionNs: number;
  /** Timer frequency in Hz. */
  frequencyHz: number;
}

/**
 * Calibrate the JS timer to detect resolution.
 *
 * Measures the minimum observable time difference to estimate timer resolution.
 */
export function calibrateTimer(): TimerInfo {
  // Measure minimum observable differences
  const diffs: number[] = [];

  for (let i = 0; i < 1000; i++) {
    const t1 = now();
    const t2 = now();
    const diff = t2 - t1;
    if (diff > 0) {
      diffs.push(diff);
    }
  }

  // Sort and take median of non-zero differences
  diffs.sort((a, b) => a - b);
  const resolutionNs: number = diffs.length > 0
    ? diffs[Math.floor(diffs.length / 2)]!
    : 1; // Default to 1ns if all diffs were 0

  return {
    cyclesPerNs: 1.0, // JS timers return nanoseconds directly
    resolutionNs,
    frequencyHz: 1e9, // 1 GHz (1 tick = 1 ns)
  };
}

/** Target ticks per batch for stable inference. */
const TARGET_TICKS_PER_BATCH = 50;

/** Maximum batch size to limit microarchitectural artifacts. */
const MAX_BATCH_SIZE = 20;

/** Number of pilot samples for batch K detection. */
const PILOT_SAMPLES = 100;

/** Warmup iterations before measurement. */
const WARMUP_ITERATIONS = 100;

/** Batching configuration from pilot phase. */
export interface BatchingInfo {
  /** Whether batching is enabled. */
  enabled: boolean;
  /** Batch size K (1 if disabled). */
  k: number;
  /** Ticks per batch. */
  ticksPerBatch: number;
  /** Human-readable rationale. */
  rationale: string;
}

/** Collected timing samples. */
export interface CollectedSamples {
  /** Baseline class timing samples (raw ticks). */
  baseline: BigInt64Array;
  /** Sample class timing samples (raw ticks). */
  sample: BigInt64Array;
  /** Batching configuration used. */
  batchingInfo: BatchingInfo;
  /** Timer info. */
  timerInfo: TimerInfo;
}

/**
 * Create a randomized interleaved schedule.
 *
 * Returns an array of booleans where true = baseline, false = sample.
 * The schedule is shuffled to prevent systematic biases.
 */
function createSchedule(samplesPerClass: number): boolean[] {
  const schedule: boolean[] = [];

  // First half: baseline (true)
  for (let i = 0; i < samplesPerClass; i++) {
    schedule.push(true);
  }
  // Second half: sample (false)
  for (let i = 0; i < samplesPerClass; i++) {
    schedule.push(false);
  }

  // Fisher-Yates shuffle
  for (let i = schedule.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp: boolean = schedule[i]!;
    schedule[i] = schedule[j]!;
    schedule[j] = temp;
  }

  return schedule;
}

/**
 * Run pilot phase to detect optimal batch size K.
 *
 * Measures operation duration and selects K to achieve TARGET_TICKS_PER_BATCH.
 */
function pilotPhase<T>(
  baselineGen: () => T,
  sampleGen: () => T,
  operation: (input: T) => void,
  timerInfo: TimerInfo
): BatchingInfo {
  // Warmup
  for (let i = 0; i < WARMUP_ITERATIONS; i++) {
    operation(baselineGen());
    operation(sampleGen());
  }

  // Pilot: measure individual operations (in nanoseconds)
  const pilotNs: number[] = [];

  for (let i = 0; i < PILOT_SAMPLES; i++) {
    // Baseline
    const inputB = baselineGen();
    const startB = now();
    operation(inputB);
    const endB = now();
    pilotNs.push(endB - startB);

    // Sample
    const inputS = sampleGen();
    const startS = now();
    operation(inputS);
    const endS = now();
    pilotNs.push(endS - startS);
  }

  // Compute median
  pilotNs.sort((a, b) => a - b);
  const medianNs = pilotNs[Math.floor(pilotNs.length / 2)]!;

  // Calculate ticks per call (ticks = ns / resolution)
  const ticksPerCall = medianNs / timerInfo.resolutionNs;

  // Select K
  if (ticksPerCall >= TARGET_TICKS_PER_BATCH) {
    return {
      enabled: false,
      k: 1,
      ticksPerBatch: ticksPerCall,
      rationale: `no batching needed (${ticksPerCall.toFixed(1)} ticks/call >= ${TARGET_TICKS_PER_BATCH} target)`,
    };
  }

  const kRaw = Math.ceil(TARGET_TICKS_PER_BATCH / ticksPerCall);
  const k = Math.min(Math.max(1, kRaw), MAX_BATCH_SIZE);
  const actualTicks = ticksPerCall * k;

  return {
    enabled: k > 1,
    k,
    ticksPerBatch: actualTicks,
    rationale: `K=${k} (${actualTicks.toFixed(1)} ticks/batch, ${ticksPerCall.toFixed(2)} ticks/call, timer res ${timerInfo.resolutionNs.toFixed(1)}ns)`,
  };
}

/**
 * Collect timing samples using interleaved measurement.
 *
 * This is the core measurement loop. It:
 * 1. Runs pilot phase to detect batch size K
 * 2. Creates randomized interleaved schedule
 * 3. Measures each operation with pure JS timing (zero FFI overhead)
 *
 * @param samplesPerClass Number of samples to collect per class
 * @param baselineGen Generator for baseline input (typically all zeros)
 * @param sampleGen Generator for sample input (typically random)
 * @param operation The operation to measure
 * @returns Collected timing samples
 */
export function collectSamples<T>(
  samplesPerClass: number,
  baselineGen: () => T,
  sampleGen: () => T,
  operation: (input: T) => void
): CollectedSamples {
  const timerInfo = calibrateTimer();

  // Pilot phase: detect batch K
  const batchingInfo = pilotPhase(baselineGen, sampleGen, operation, timerInfo);
  const k = batchingInfo.k;

  // Create schedule
  const schedule = createSchedule(samplesPerClass);

  // Allocate result arrays (BigInt64Array for compatibility with analysis functions)
  const baseline = new BigInt64Array(samplesPerClass);
  const sample = new BigInt64Array(samplesPerClass);
  let baselineIdx = 0;
  let sampleIdx = 0;

  // Measurement loop
  for (const isBaseline of schedule) {
    // Generate input OUTSIDE timed region
    const input = isBaseline ? baselineGen() : sampleGen();

    // TIMED REGION - pure JS, zero FFI overhead
    let elapsedNs: number;
    if (k === 1) {
      const start = now();
      operation(input);
      elapsedNs = now() - start;
    } else {
      const start = now();
      for (let i = 0; i < k; i++) {
        operation(input);
      }
      elapsedNs = now() - start;
    }

    // Store result as BigInt (for compatibility with Rust analysis)
    if (isBaseline) {
      baseline[baselineIdx++] = BigInt(Math.round(elapsedNs));
    } else {
      sample[sampleIdx++] = BigInt(Math.round(elapsedNs));
    }
  }

  return {
    baseline,
    sample,
    batchingInfo,
    timerInfo,
  };
}

/**
 * Collect samples in batches for adaptive analysis.
 *
 * This is a generator that yields batches of samples, allowing the caller
 * to run adaptive analysis between batches.
 *
 * @param batchSize Number of samples per batch per class
 * @param baselineGen Generator for baseline input
 * @param sampleGen Generator for sample input
 * @param operation The operation to measure
 */
export function* collectBatches<T>(
  batchSize: number,
  baselineGen: () => T,
  sampleGen: () => T,
  operation: (input: T) => void
): Generator<{
  baseline: BigInt64Array;
  sample: BigInt64Array;
  timerInfo: TimerInfo;
  batchingInfo: BatchingInfo;
}> {
  const timerInfo = calibrateTimer();

  // Pilot phase (only once)
  const batchingInfo = pilotPhase(baselineGen, sampleGen, operation, timerInfo);
  const k = batchingInfo.k;

  while (true) {
    const schedule = createSchedule(batchSize);
    const baseline = new BigInt64Array(batchSize);
    const sample = new BigInt64Array(batchSize);
    let baselineIdx = 0;
    let sampleIdx = 0;

    for (const isBaseline of schedule) {
      const input = isBaseline ? baselineGen() : sampleGen();

      // TIMED REGION - pure JS, zero FFI overhead
      let elapsedNs: number;
      if (k === 1) {
        const start = now();
        operation(input);
        elapsedNs = now() - start;
      } else {
        const start = now();
        for (let i = 0; i < k; i++) {
          operation(input);
        }
        elapsedNs = now() - start;
      }

      if (isBaseline) {
        baseline[baselineIdx++] = BigInt(Math.round(elapsedNs));
      } else {
        sample[sampleIdx++] = BigInt(Math.round(elapsedNs));
      }
    }

    yield { baseline, sample, timerInfo, batchingInfo };
  }
}
