# @tacet/js

Timing side-channel detection for JavaScript/TypeScript. Detect timing leaks in cryptographic code using statistical analysis.

This is a WASM-based implementation that works in **Node.js**, **Bun**, and **Deno**.

> **Note:** Browsers are not supported because they lack high-precision timers (due to Spectre mitigations).

## Installation

```bash
# npm
npm install @tacet/js

# bun
bun add @tacet/js

# deno/jsr
deno add jsr:@tacet/js
```

## Quick Start

```typescript
import { TimingOracle, AttackerModelValues, OutcomeValues } from "@tacet/js";
import crypto from "crypto";

// Test your cryptographic function
const result = TimingOracle.forAttacker(AttackerModelValues.AdjacentNetwork)
  .timeBudget(30_000) // 30 seconds
  .test(
    {
      baseline: () => Buffer.alloc(32, 0), // All zeros
      sample: () => crypto.randomBytes(32), // Random data
    },
    (input) => myCryptoFunction(input)
  );

// Handle the result
switch (result.outcome) {
  case OutcomeValues.Pass:
    console.log(`No leak detected: P(leak) = ${result.leakProbability}`);
    break;
  case OutcomeValues.Fail:
    console.error(`Timing leak detected! Exploitability: ${result.exploitability}`);
    process.exit(1);
    break;
  case OutcomeValues.Inconclusive:
    console.warn(`Inconclusive: ${result.inconclusiveReason}`);
    break;
}
```

## Attacker Models

Choose your attacker model based on your threat scenario:

| Model | Threshold | Use Case |
|-------|-----------|----------|
| `SharedHardware` | 0.6 ns | SGX, cross-VM, containers |
| `AdjacentNetwork` | 100 ns | LAN, HTTP/2 APIs |
| `RemoteNetwork` | 50 μs | Internet-facing services |
| `Research` | 0 | Detect any difference |

## API Reference

### `TimingOracle`

Builder-pattern API for timing tests:

```typescript
const oracle = TimingOracle.forAttacker(AttackerModelValues.AdjacentNetwork)
  .timeBudget(30_000)      // Max time in ms
  .maxSamples(100_000)     // Max samples to collect
  .passThreshold(0.05)     // P(leak) threshold for Pass
  .failThreshold(0.95);    // P(leak) threshold for Fail
```

### `TimingTestResult`

The result object contains:

- `outcome`: "pass" | "fail" | "inconclusive" | "unmeasurable"
- `leakProbability`: Posterior probability of a timing leak
- `effectNs`: Estimated timing difference in nanoseconds
- `exploitability`: "negligible" | "possibleLan" | "likelyLan" | "possibleRemote"
- `quality`: "excellent" | "good" | "poor" | "tooNoisy"

### Low-Level API

For advanced usage, you can use the low-level functions directly:

```typescript
import {
  collectSamples,
  calibrateSamples,
  analyze,
  defaultConfig,
} from "@tacet/js";

// Collect timing samples
const samples = collectSamples(inputPair, measureFn, 5000);

// Calibrate and analyze
const calibration = calibrateSamples(
  samples.baselineSamples,
  samples.sampleSamples,
  samples.nsPerTick,
  defaultConfig()
);

const result = analyze(calibration, defaultConfig());
```

## License

MPL-2.0
