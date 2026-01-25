# timing-oracle Test Suite

## Quick Reference

| Profile | Command | Tests | Time | Use Case |
|---------|---------|-------|------|----------|
| quick | `cargo nextest run --profile quick` | ~300 | ~5s | Fast iteration |
| smoke | `cargo nextest run --profile smoke` | ~310 | ~30s | Pre-commit check |
| crypto | `cargo nextest run --profile crypto` | ~56 | ~10min | Crypto validation |
| full | `cargo nextest run --profile full` | ~350 | ~30min | CI full suite |

## Test Categories

### Core (6 tests)
Baseline validation that the oracle works correctly.

| File | Tests | Description |
|------|-------|-------------|
| `core/known_leaky.rs` | 2 | MUST detect timing leaks (early-exit, branches) |
| `core/known_safe.rs` | 4 | MUST NOT false-positive on constant-time code |

### Crypto (30 tests)
Real cryptographic library timing validation, organized by crate.

| Crate | Family | Tests | File |
|-------|--------|-------|------|
| **RustCrypto** | | | |
| aes | Symmetric | 6 | `crypto/rustcrypto/aes.rs` |
| sha3 | Hash | 6 | `crypto/rustcrypto/sha3.rs` |
| blake2 | Hash | 4 | `crypto/rustcrypto/blake2.rs` |
| rsa | Asymmetric | 7 | `crypto/rustcrypto/rsa.rs` |
| **dalek** | | | |
| x25519-dalek | ECC | 7 | `crypto/dalek/x25519.rs` |

*Note: Additional crypto tests in `aead_timing.rs` (8) and `pqcrypto_timing.rs` (12) pending migration.*

### Synthetic (28 tests)
Artificial scenarios testing specific oracle behaviors.

| File | Tests | Description |
|------|-------|-------------|
| `synthetic/attack_patterns.rs` | 20 | Cache effects, modexp, table lookups |
| `synthetic/dudect_examples.rs` | 4 | DudeCT comparison examples |
| `synthetic/power_analysis.rs` | 4 | Statistical power analysis |

### Calibration (50 tests)
Statistical validation (slow, excluded from default runs).

| File | Tests | Description |
|------|-------|-------------|
| `calibration/fpr.rs` | 4 | False positive rate validation |
| `calibration/power.rs` | 11 | Detection power across effect sizes |
| `calibration/bayesian.rs` | 5 | Posterior calibration |
| `calibration/coverage.rs` | 5 | Credible interval coverage |
| `calibration/estimation.rs` | 5 | Effect estimation accuracy |
| `calibration/autocorrelation.rs` | 4 | Autocorrelation robustness |
| `calibration/power_curve.rs` | 5 | Fine-grained power curves |
| `calibration/stress.rs` | 3 | Behavior under CPU/memory pressure |
| `calibration/utils.rs` | - | Shared utilities (not tests) |

### Unit (176 tests)
Fast validation of config, types, and helpers (no timing measurements).

| File | Tests | Description |
|------|-------|-------------|
| `unit/config_validation.rs` | 58 | Configuration validation |
| `unit/types.rs` | 57 | Type tests (Outcome, Effect, etc.) |
| `unit/reliability.rs` | 30 | Reliability handling |
| `unit/helpers.rs` | 8 | Helper function tests |
| `unit/integration.rs` | 5 | End-to-end workflows |
| `unit/discrete_mode.rs` | 9 | Discrete timing mode |

### Platform (7 tests)
PMU-based timing (requires sudo).

| File | Tests | Platform |
|------|-------|----------|
| `platform/kperf.rs` | 4 | macOS ARM64 |
| `platform/perf.rs` | 3 | Linux |

### Runtime (11 tests)
Async/await and concurrent task timing.

| File | Tests | Description |
|------|-------|-------------|
| `runtime/async_timing.rs` | 9 | Async/await timing |
| `runtime/concurrency.rs` | 2 | Thread pool, background tasks |

### Macros (35 tests)
Proc macro syntax and compile-time errors.

| File | Tests | Description |
|------|-------|-------------|
| `macros/syntax.rs` | 34 | Macro syntax variations |
| `macros/compile_errors.rs` | 1 | trybuild compile-fail tests |

### Investigation (22 tests)
One-off debugging and research tests (excluded from CI).

| File | Tests | Description |
|------|-------|-------------|
| `investigation/rsa_investigation.rs` | 14 | RSA timing analysis |
| `investigation/rsa_vulnerability.rs` | 8 | RSA vulnerability assessment |

## Running Tests

```bash
# Fast iteration (unit tests only)
cargo nextest run --profile quick

# Pre-commit check (includes sample crypto)
cargo nextest run --profile smoke

# All crypto tests
cargo nextest run --profile crypto

# Specific crate tests
cargo nextest run --profile rustcrypto
cargo nextest run --profile dalek
cargo nextest run --profile pqcrypto

# Platform-specific (requires sudo)
sudo -E cargo nextest run --profile platform

# Full CI suite
cargo nextest run --profile full

# Calibration suite (slow)
cargo nextest run --profile calibration
```

## Adding New Crypto Tests

When adding tests for a new crypto library:

1. Create a new file under `crypto/{ecosystem}/` (e.g., `crypto/rustcrypto/ed25519.rs`)
2. Add the module to the parent entry point (e.g., `crypto/rustcrypto.rs`)
3. Name test functions with the pattern: `{crate}_{primitive}_{operation}_ct`
4. Use `_ct` suffix for constant-time tests, `_leaky` for intentionally leaky tests
5. Update this README with the new test count

## Test Naming Convention

```
{crate}_{primitive}_{operation}[_{property}]

Examples:
- rustcrypto_sha3_256_ct
- dalek_x25519_scalar_mult_ct
- ring_aes256gcm_encrypt_ct
- pqcrypto_kyber768_encapsulate_ct
```

Suffixes:
- `_ct` — constant-time test (expects Pass)
- `_leaky` — intentionally leaky (expects Fail)
- `_hamming` — Hamming weight independence
- `_pattern` — byte pattern independence
- `_nonce` — nonce independence
