/**
 * Basic tests for tacet-wasm WASM module.
 */

import { describe, test, expect, beforeAll } from "bun:test";
import {
  initializeWasm,
  version,
  defaultConfig,
  configAdjacentNetwork,
  configSharedHardware,
  configRemoteNetwork,
  calibrateTimer,
  AttackerModelValues,
  OutcomeValues,
} from "../dist/index.js";
import type { Config, AttackerModel } from "../dist/index.js";

describe("WASM Module", () => {
  beforeAll(async () => {
    await initializeWasm();
  });

  test("version returns a string", () => {
    const v = version();
    expect(typeof v).toBe("string");
    expect(v.length).toBeGreaterThan(0);
  });

  test("defaultConfig returns valid config", () => {
    const config = defaultConfig(AttackerModelValues.AdjacentNetwork);
    expect(config).toBeDefined();
    expect(config.attackerModel).toBe("adjacentNetwork");
    expect(typeof config.maxSamples).toBe("number");
    expect(typeof config.timeBudgetMs).toBe("number");
    expect(typeof config.passThreshold).toBe("number");
    expect(typeof config.failThreshold).toBe("number");
  });

  test("configAdjacentNetwork returns valid config", () => {
    const config = configAdjacentNetwork();
    expect(config).toBeDefined();
    expect(config.attackerModel).toBe("adjacentNetwork");
  });

  test("configSharedHardware returns valid config", () => {
    const config = configSharedHardware();
    expect(config).toBeDefined();
    expect(config.attackerModel).toBe("sharedHardware");
  });

  test("configRemoteNetwork returns valid config", () => {
    const config = configRemoteNetwork();
    expect(config).toBeDefined();
    expect(config.attackerModel).toBe("remoteNetwork");
  });
});

describe("Timer Calibration", () => {
  test("calibrateTimer returns valid info", () => {
    const info = calibrateTimer();
    expect(info).toBeDefined();
    expect(typeof info.cyclesPerNs).toBe("number");
    expect(typeof info.resolutionNs).toBe("number");
    expect(typeof info.frequencyHz).toBe("number");
    expect(info.cyclesPerNs).toBeGreaterThan(0);
    expect(info.resolutionNs).toBeGreaterThan(0);
    expect(info.frequencyHz).toBeGreaterThan(0);
  });
});

describe("Enum Values", () => {
  test("AttackerModelValues has all variants", () => {
    expect(AttackerModelValues.SharedHardware).toBe("sharedHardware");
    expect(AttackerModelValues.PostQuantum).toBe("postQuantum");
    expect(AttackerModelValues.AdjacentNetwork).toBe("adjacentNetwork");
    expect(AttackerModelValues.RemoteNetwork).toBe("remoteNetwork");
    expect(AttackerModelValues.Research).toBe("research");
  });

  test("OutcomeValues has all variants", () => {
    expect(OutcomeValues.Pass).toBe("pass");
    expect(OutcomeValues.Fail).toBe("fail");
    expect(OutcomeValues.Inconclusive).toBe("inconclusive");
    expect(OutcomeValues.Unmeasurable).toBe("unmeasurable");
  });
});
