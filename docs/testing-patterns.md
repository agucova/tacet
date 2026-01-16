# Testing Patterns Guide

**Comprehensive, example-driven patterns for testing cryptographic code with timing-oracle.**

---

## Table of Contents

1. [Introduction](#1-introduction) ← **Start here**
2. [Basic Patterns](#2-basic-patterns) - Foundation for 80% of use cases
3. [Input Generation Patterns](#3-input-generation-patterns) - Master the #1 mistake area
4. [Crypto-Specific Patterns](#4-crypto-specific-patterns) ⚠️ - Real-world crypto testing
   - [4.1 AEAD Ciphers](#41-aead-ciphers-critical) ⚠️ **Critical: Nonce reuse pitfall**
   - [4.2 Block Ciphers](#42-block-ciphers)
   - [4.3 Hash Functions](#43-hash-functions)
   - [4.4 Elliptic Curve Cryptography](#44-elliptic-curve-cryptography)
   - [4.5 RSA](#45-rsa)
   - [4.6 Post-Quantum Cryptography](#46-post-quantum-cryptography)
5. [Advanced Patterns](#5-advanced-patterns) - Async, complex types, state
6. [Integration Patterns](#6-integration-patterns) - CI/CD, test organization
7. [Troubleshooting Patterns](#7-troubleshooting-patterns) - Fixing common issues
8. [Quick Reference](#8-quick-reference) - Fast lookup catalog

---

## 1. Introduction

### 1.1 What This Document Covers

This guide provides **practical, copy-pasteable testing patterns** for using `timing-oracle` in real-world scenarios. Every code example is:

- ✓ **Compilable and runnable** - extracted from the actual test suite
- ✓ **Production-ready** - demonstrates best practices
- ✓ **Well-commented** - explains WHY, not just WHAT

**This document is NOT**:
- A conceptual overview (see [`guide.md`](./guide.md))
- An API reference (see [`api-reference.md`](./api-reference.md))
- A troubleshooting FAQ (see [`troubleshooting.md`](./troubleshooting.md))

**This document IS**:
- A practical cookbook of working patterns
- A compilation of lessons learned from real usage
- A guide to avoiding common pitfalls (especially AEAD nonce reuse ⚠️)

### 1.2 Navigation Guide - Where Should I Start?

Choose your starting point based on your situation:

| **Your Situation** | **Start Here** | **Estimated Time** |
|-------------------|---------------|-------------------|
| "I'm new to timing-oracle" | [§2 Basic Patterns](#2-basic-patterns) | 10-15 min |
| "I want to test AES/ChaCha20/AES-GCM" | [§4.1 AEAD Ciphers](#41-aead-ciphers-critical) ⚠️ | 15-20 min |
| "I'm testing other crypto primitives" | [§4 Crypto-Specific](#4-crypto-specific-patterns) | 20-30 min |
| "I have async/await code" | [§5.1 Async Patterns](#51-async-await-code) | 10-15 min |
| "My tests are flaky/unreliable" | [§7 Troubleshooting](#7-troubleshooting-patterns) | 15-25 min |
| "I need CI/CD integration" | [§6 Integration Patterns](#6-integration-patterns) | 10-15 min |
| "Quick pattern lookup" | [§8 Quick Reference](#8-quick-reference) | 2-5 min |

### 1.3 How to Use This Document

**For Newcomers:**
1. Read [§2 Basic Patterns](#2-basic-patterns) to understand the foundation
2. Read [§3 Input Generation](#3-input-generation-patterns) - this is where 90% of mistakes happen
3. Jump to the crypto primitive you're testing in [§4](#4-crypto-specific-patterns)
4. Bookmark [§8 Quick Reference](#8-quick-reference) for future lookups

**For Experienced Users:**
- Use [§8 Quick Reference](#8-quick-reference) to find patterns quickly
- Check [§4.1 AEAD](#41-aead-ciphers-critical) if testing authenticated encryption ⚠️
- Consult [§7 Troubleshooting](#7-troubleshooting-patterns) when tests behave unexpectedly

**For Code Reviewers:**
- Check [§3.5 Common Mistakes](#35-common-mistakes-checklist) - the anti-pattern catalog
- Verify AEAD tests use atomic nonces (§4.1)
- Ensure InputPair uses closures, not captured values (§3.5)

### 1.4 Visual Conventions

This document uses consistent visual markers:

- ❌ **Wrong** - Do NOT use this pattern (common mistake)
- ✓ **Correct** - Recommended pattern to use
- ⚠️ **Warning** - Critical pitfall or security concern
- 💡 **Tip** - Performance or usability improvement

**Example formatting:**
```rust
// ❌ WRONG: Captured nonce gets reused
let nonce = [0u8; 12];
let inputs = InputPair::new(|| plaintext, || plaintext);

// ✓ CORRECT: Atomic counter generates unique nonces
let nonce_counter = AtomicU64::new(0);
let inputs = InputPair::new(|| {
    let nonce_val = nonce_counter.fetch_add(1, Ordering::Relaxed);
    // ... use nonce_val
}, || { /* ... */ });
```

### 1.5 Code Example Structure

Each pattern follows this structure:

1. **Pattern name** - Descriptive title
2. **Use case** - When to use this pattern
3. **Complete example** - Copy-pasteable working code
4. **Key points** - What makes this pattern work
5. **Common mistakes** - What to avoid (❌/✓ comparison)
6. **References** - Source test file and related sections

### 1.6 Relationship to Other Documentation

| **Document** | **Purpose** | **When to Read** |
|-------------|-------------|------------------|
| [`README.md`](../README.md) | Quick start, installation, overview | First time using the library |
| [`guide.md`](./guide.md) | Conceptual explanation, methodology | Understanding how it works |
| [`api-reference.md`](./api-reference.md) | API parameters, types, configuration | Customizing test behavior |
| **`testing-patterns.md`** (this doc) | **Practical examples, patterns** | **Writing actual tests** |
| [`troubleshooting.md`](./troubleshooting.md) | Debugging, performance tuning | Tests failing or unreliable |
| [`TESTING.md`](../TESTING.md) | Library's own test suite structure | Contributing to the library |

**Reading order for new users:**
1. `README.md` - Get oriented
2. `guide.md` - Understand concepts
3. **`testing-patterns.md`** ← You are here - Write your first test
4. `api-reference.md` - Fine-tune configuration
5. `troubleshooting.md` - Fix issues as they arise

---

**Next:** [§2 Basic Patterns](#2-basic-patterns) - Foundation patterns for 80% of use cases

---

## 2. Basic Patterns

**Foundation patterns that cover 80% of use cases.** Start here if you're new to timing-oracle.

### 2.1 Pattern: Simplest Constant-Time Test

**Use case:** Verify that a simple operation (like XOR or array copy) is constant-time.

**Complete example:**
```rust
use timing_oracle::{TimingOracle, helpers::InputPair};

fn main() {
    // Test that XOR is constant-time (baseline test)
    let inputs = InputPair::new(
        || [0x00u8; 64],  // Baseline: all zeros
        || [0xFFu8; 64],  // Sample: all ones
    );

    let outcome = TimingOracle::balanced()
        .samples(30_000)
        .test(inputs, |data| {
            let result = data.iter().fold(0u8, |acc, &b| acc ^ b);
            std::hint::black_box(result);  // Prevent compiler optimizations
        });

    let result = outcome.unwrap_completed();

    println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
    println!("CI gate passed: {}", result.ci_gate.passed);
    println!("Exploitability: {:?}", result.exploitability);

    // Assert the operation is constant-time
    assert!(result.ci_gate.passed, "XOR should be constant-time");
    assert!(result.leak_probability < 0.3, "False positive: leak probability too high");
}
```

**Key points:**
- Use `.balanced()` preset for fast tests (~1-2 seconds)
- `black_box()` prevents compiler from optimizing away the operation
- Both inputs go through identical code path (XOR fold)
- Only the DATA differs (0x00 vs 0xFF)

**Reference:** See `examples/baseline_test.rs:4-20`

---

### 2.2 Pattern: DudeCT Two-Class Pattern (Fixed vs Random)

**Use case:** Standard pattern for testing secret-dependent operations. Tests if timing depends on input data.

**Complete example:**
```rust
use timing_oracle::{TimingOracle, helpers::InputPair, Outcome};

fn main() {
    let secret = [0xABu8; 32];

    // DudeCT pattern: fixed input vs random input
    let inputs = InputPair::new(
        || [0xABu8; 32],         // Baseline: fixed value (matches secret)
        || rand::random::<[u8; 32]>(),  // Sample: random data
    );

    let outcome = TimingOracle::balanced()
        .alpha(0.01)  // 1% false positive rate
        .test(inputs, |data| {
            // Both closures execute IDENTICAL code
            compare_bytes(&secret, data);
        });

    match outcome {
        Outcome::Completed(result) => {
            println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
            println!("CI gate: {}", if result.ci_gate.passed { "PASS" } else { "FAIL" });

            // For constant-time code, expect:
            // - CI gate passes
            // - Low leak probability (<30%)
            assert!(result.ci_gate.passed);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Could not measure: {}", recommendation);
        }
    }
}

/// Constant-time comparison using XOR accumulator
fn compare_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc = 0u8;
    for i in 0..a.len() {
        acc |= a[i] ^ b[i];
    }
    acc == 0
}
```

**Key points:**
- **Baseline class:** Fixed value (typically matches expected/secret value)
- **Sample class:** Random data (different from baseline)
- Both closures execute the SAME code path
- Tests whether timing depends on data content
- This is the recommended pattern for most crypto tests

**Common mistake:**
```rust
// ❌ WRONG: Testing different operations
let inputs = InputPair::new(
    || fast_path(),    // Different code!
    || slow_path(),    // Different code!
);

// ✓ CORRECT: Same operation, different data
let inputs = InputPair::new(
    || fixed_data,     // Same code,
    || random_data,    // different data
);
```

**Reference:** See `examples/compare.rs:14-42`, `tests/aes_timing.rs:63-94`

---

### 2.3 Pattern: InputPair - Avoiding RNG Overhead

**Use case:** Generating test inputs without measuring RNG overhead.

**The #1 mistake:** Generating random data INSIDE the measured closure.

**❌ WRONG - Measures RNG overhead:**
```rust
// This measures rand::random() overhead!
let outcome = TimingOracle::balanced().test_with_baseline(
    [0u8; 32],
    |_input| {
        let data = rand::random::<[u8; 32]>();  // ❌ RNG measured!
        crypto_function(&data);
    },
);
```

**✓ CORRECT - Pre-generate inputs:**
```rust
use timing_oracle::helpers::InputPair;

// InputPair generates inputs BEFORE timing starts
let inputs = InputPair::new(
    || [0u8; 32],                    // Baseline generator (closure)
    || rand::random::<[u8; 32]>(),  // Sample generator (closure)
);

let outcome = TimingOracle::balanced().test(inputs, |data| {
    crypto_function(data);  // Only this is measured
});
```

**Key points:**
- `InputPair` calls generator closures BEFORE timing starts
- All inputs are pre-generated in randomized order
- The measured closure receives **pre-generated** data
- Both baseline and sample use closures (not values!)

**Why closures matter:**
```rust
// ❌ WRONG: Captures a single pre-evaluated value
let random_value = rand::random::<[u8; 32]>();  // Generated ONCE
let inputs = InputPair::new(
    || [0u8; 32],
    || random_value,  // ❌ Same value every time!
);

// ✓ CORRECT: Closure generates fresh value each call
let inputs = InputPair::new(
    || [0u8; 32],
    || rand::random::<[u8; 32]>(),  // ✓ New value each time
);
```

**Reference:** See `examples/simple.rs:16-26`, §3 for advanced input patterns

---

### 2.4 Pattern: The Assertion Pyramid

**Use case:** Comprehensive validation of test results using multiple confidence layers.

**Complete example:**
```rust
use timing_oracle::{TimingOracle, helpers::InputPair, Exploitability, MeasurementQuality};

fn test_crypto_operation() {
    let inputs = InputPair::new(|| [0u8; 32], || rand::random::<[u8; 32]>());

    let outcome = TimingOracle::balanced()
        .alpha(0.01)
        .test(inputs, |data| {
            my_crypto_function(data);
        });

    let result = outcome.unwrap_completed();

    // Layer 1: CI Gate (Frequentist Test)
    // - Most conservative layer
    // - Controls false positive rate at `alpha` level
    assert!(
        result.ci_gate.passed,
        "CI gate failed: timing leak detected (max_stat={:.2} > threshold={:.2})",
        result.ci_gate.max_stat,
        result.ci_gate.threshold
    );

    // Layer 2: Leak Probability (Bayesian Confidence)
    // - Posterior probability of timing leak
    // - More nuanced than binary CI gate
    assert!(
        result.leak_probability < 0.3,
        "High leak probability: {:.1}% suggests timing dependence",
        result.leak_probability * 100.0
    );

    // Layer 3: Exploitability Assessment
    // - Practical impact classification
    // - Considers effect size in nanoseconds
    assert!(
        matches!(result.exploitability, Exploitability::Negligible),
        "Effect size too large: {:?}",
        result.exploitability
    );

    // Layer 4: Measurement Quality
    // - Validates test had sufficient statistical power
    // - Ensures results are trustworthy
    assert!(
        matches!(result.quality, MeasurementQuality::Excellent | MeasurementQuality::Good),
        "Poor measurement quality: {:?}. Consider increasing samples or checking system noise.",
        result.quality
    );

    // Optional: Effect size inspection
    if let Some(effect) = &result.effect {
        println!("Effect detected: {:.1}ns {:?}",
                 effect.shift_ns.abs() + effect.tail_ns.abs(),
                 effect.pattern);
    }
}

fn my_crypto_function(_data: &[u8]) {
    // Your crypto operation here
}
```

**The pyramid (bottom to top):**

```
                 ▲
                / \
               /   \
              /  4  \    Layer 4: Measurement Quality
             /-------\   (Did we have sufficient power?)
            /    3   \   Layer 3: Exploitability
           /---------\   (Is the effect practically significant?)
          /     2    \   Layer 2: Leak Probability (Bayesian)
         /-----------\   (How confident are we there's a leak?)
        /      1     \   Layer 1: CI Gate (Frequentist)
       /-------------\   (Did we reject null hypothesis?)
      /_______________\
```

**When to use each layer:**

| **Layer** | **Use For** | **Typical Threshold** |
|-----------|-------------|---------------------|
| CI Gate | Binary pass/fail decision | `.passed == true` |
| Leak Probability | Confidence assessment | `< 0.3` (baseline), `> 0.7` (leak detection) |
| Exploitability | Security risk evaluation | `Negligible` for constant-time code |
| Quality | Test reliability check | `Excellent` or `Good` |

**Key points:**
- CI gate is the **strictest** layer (use for production tests)
- Leak probability provides **nuance** (useful for debugging)
- Exploitability translates to **real-world impact**
- Quality ensures the test itself was **reliable**

**Reference:** See `tests/crypto_attacks.rs:63-110` for comprehensive examples

---

### 2.5 Pattern: Handling Unreliable Measurements

**Use case:** Gracefully handling operations that cannot be reliably measured (too fast, too noisy, or unmeasurable).

**Problem:** Some operations complete in <10ns or have high system noise, making timing measurements unreliable.

**Solution 1: Fail-Open with `skip_if_unreliable!` (Recommended for CI)**

```rust
use timing_oracle::{TimingOracle, helpers::InputPair, skip_if_unreliable};

#[test]
fn test_very_fast_operation() {
    let inputs = InputPair::new(|| 42u64, || rand::random::<u64>());

    let outcome = TimingOracle::balanced().test(inputs, |&x| {
        // Very fast operation - might be unmeasurable
        let result = x.wrapping_add(1);
        std::hint::black_box(result);
    });

    // Skip test gracefully if measurement is unreliable
    // Returns early without failing the test
    let result = skip_if_unreliable!(outcome, "test_very_fast_operation");

    // If we get here, measurement was reliable
    assert!(result.ci_gate.passed);
}
```

**Solution 2: Fail-Closed with `require_reliable!` (Strict Mode)**

```rust
use timing_oracle::{TimingOracle, helpers::InputPair, require_reliable};

#[test]
fn test_critical_crypto_operation() {
    let inputs = InputPair::new(|| [0u8; 32], || rand::random::<[u8; 32]>());

    let outcome = TimingOracle::balanced().test(inputs, |data| {
        critical_crypto_operation(data);
    });

    // PANIC if measurement is unreliable
    // Use for critical security tests that MUST be measured
    let result = require_reliable!(outcome, "test_critical_crypto_operation");

    assert!(result.ci_gate.passed);
}

fn critical_crypto_operation(_data: &[u8]) {}
```

**Solution 3: Manual Pattern Matching**

```rust
use timing_oracle::{Outcome, TimingOracle, helpers::InputPair};

#[test]
fn test_with_custom_handling() {
    let inputs = InputPair::new(|| [0u8; 32], || rand::random::<[u8; 32]>());

    let outcome = TimingOracle::balanced().test(inputs, |data| {
        my_operation(data);
    });

    match outcome {
        Outcome::Completed(result) => {
            // Normal assertion path
            assert!(result.ci_gate.passed);
            assert!(result.leak_probability < 0.3);
        }
        Outcome::Unmeasurable { reason, recommendation } => {
            // Custom handling based on reason
            eprintln!("Measurement unreliable: {}", recommendation);

            match reason {
                timing_oracle::UnmeasurableReason::TooFast => {
                    // Maybe batch the operation
                    eprintln!("Operation too fast - consider batching");
                }
                timing_oracle::UnmeasurableReason::HighNoise => {
                    // System noise too high - inform user
                    eprintln!("High system noise detected");
                }
                _ => {
                    eprintln!("Other unmeasurable reason: {:?}", reason);
                }
            }
        }
    }
}

fn my_operation(_data: &[u8]) {}
```

**Choosing the right approach:**

| **Pattern** | **Use When** | **Behavior on Unreliable** |
|-------------|-------------|---------------------------|
| `skip_if_unreliable!` | CI tests, optional operations | Returns early (test passes) |
| `require_reliable!` | Critical security tests | Panics (test fails) |
| Manual `match` | Custom handling needed | User-defined behavior |

**Environment variable override:**

```bash
# Force strict mode globally
export TIMING_ORACLE_UNRELIABLE_POLICY=fail_closed

# Now skip_if_unreliable! will panic instead of skipping
cargo test
```

**Key points:**
- Use `skip_if_unreliable!` for **most tests** (pragmatic)
- Use `require_reliable!` for **security-critical** tests (strict)
- Manual matching for **custom logic** (debugging)
- Operations <10ns or on noisy systems may be unmeasurable

**Reference:** See `src/macros.rs`, `tests/integration.rs:51-78`

---

**Next:** [§3 Input Generation Patterns](#3-input-generation-patterns) - Master the #1 mistake area

---

## 3. Input Generation Patterns

**Master the #1 mistake area**: 90% of timing-oracle issues come from incorrectinput generation. This section teaches how to generate inputs correctly.

### 3.1 Pattern: Fixed-Size Arrays (Convenience Helper)

**Use case:** Testing cryptographic operations on 32-byte data (keys, hashes, scalars).

**Complete example:**
```rust
use timing_oracle::{TimingOracle, helpers::byte_arrays_32};

#[test]
fn test_hash_function() {
    // Convenience helper for [u8; 32] arrays
    // Baseline: [0u8; 32]
    // Sample: rand::random::<[u8; 32]>()
    let inputs = byte_arrays_32();

    let outcome = TimingOracle::balanced().test(inputs, |data| {
        sha3_256(data);
    });

    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}

fn sha3_256(_input: &[u8; 32]) {
    // Your hash function here
}
```

**Equivalent manual form:**
```rust
use timing_oracle::helpers::InputPair;

let inputs = InputPair::new(
    || [0u8; 32],                    // Baseline: all zeros
    || rand::random::<[u8; 32]>(),  // Sample: random bytes
);
```

**Key points:**
- `byte_arrays_32()` is a convenience for the common case
- Automatically uses DudeCT pattern (zeros vs random)
- Use manual `InputPair::new()` for custom baseline values

**Other array sizes:**
```rust
// For 16-byte arrays (AES blocks)
let inputs = InputPair::new(
    || [0u8; 16],
    || rand::random::<[u8; 16]>(),
);

// For 64-byte arrays (SHA-512)
let inputs = InputPair::new(
    || [0u8; 64],
    || rand::random::<[u8; 64]>(),
);

// For custom fixed values
let inputs = InputPair::new(
    || [0xABu8; 32],  // Fixed value (not all zeros)
    || rand::random::<[u8; 32]>(),
);
```

**Reference:** See `src/helpers.rs:374-376`

---

### 3.2 Pattern: Variable-Size Vectors

**Use case:** Testing operations on variable-length data (variable-length plaintexts, messages).

**Complete example:**
```rust
use timing_oracle::{TimingOracle, helpers::byte_vecs};

#[test]
fn test_variable_length_encryption() {
    // Convenience helper for Vec<u8> of specific length
    let inputs = byte_vecs(1024);  // 1KB vectors

    let outcome = TimingOracle::balanced().test(inputs, |data| {
        encrypt_message(data);
    });

    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}

fn encrypt_message(_data: &[u8]) {
    // Your encryption here
}
```

**Equivalent manual form:**
```rust
use timing_oracle::helpers::InputPair;

let inputs = InputPair::new(
    move || vec![0u8; 1024],  // Baseline: zeros
    move || {
        (0..1024)
            .map(|_| rand::random::<u8>())
            .collect()
    },
);
```

**Variable-length pattern:**
```rust
// Testing multiple lengths (run separate tests)
for len in [64, 256, 1024, 4096] {
    let inputs = byte_vecs(len);

    let outcome = TimingOracle::balanced().test(inputs, |data| {
        process(data);
    });

    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed, "Failed at length {}", len);
}

fn process(_data: &[u8]) {}
```

**Key points:**
- `byte_vecs(len)` creates vectors of specific length
- Both baseline and sample have SAME length (only data differs)
- Use `move ||` closures to capture `len` variable
- Test different lengths in separate test runs

**Reference:** See `src/helpers.rs:394-399`

---

### 3.3 Pattern: Custom Closures with State (Cell Pattern)

**Use case:** Generating sequential or stateful test inputs without capturing values.

**Complete example:**
```rust
use std::cell::Cell;
use timing_oracle::{TimingOracle, helpers::InputPair};

#[test]
fn test_with_sequential_nonces() {
    // Cell allows interior mutability in closures
    let nonce_counter = Cell::new(0u64);

    let inputs = InputPair::new(
        move || {
            let n = nonce_counter.get();
            nonce_counter.set(n + 1);
            let mut nonce = [0u8; 12];
            nonce[..8].copy_from_slice(&n.to_le_bytes());
            nonce
        },
        move || {
            // Sample can also use the counter if needed
            rand::random::<[u8; 12]>()
        },
    );

    let outcome = TimingOracle::balanced().test(inputs, |nonce| {
        encrypt_with_nonce(nonce);
    });

    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}

fn encrypt_with_nonce(_nonce: &[u8; 12]) {
    // Your AEAD encryption here
}
```

**Why Cell?**
```rust
// ❌ WRONG: Won't compile - closures need FnMut
let mut counter = 0u64;
let inputs = InputPair::new(
    || {
        counter += 1;  // Error: captures mutable reference
        counter
    },
    || rand::random(),
);

// ✓ CORRECT: Cell provides interior mutability
use std::cell::Cell;

let counter = Cell::new(0u64);
let inputs = InputPair::new(
    || {
        let val = counter.get();
        counter.set(val + 1);
        val
    },
    || rand::random(),
);
```

**Other stateful patterns:**
```rust
use std::cell::Cell;

// Rotating through predefined values
let pool = vec![[0xAAu8; 32], [0xBBu8; 32], [0xCCu8; 32]];
let idx = Cell::new(0);
let inputs = InputPair::new(
    || {
        let i = idx.get();
        idx.set((i + 1) % pool.len());
        pool[i]
    },
    || rand::random::<[u8; 32]>(),
);

// Random but deterministic (seeded RNG)
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cell::RefCell;

let rng = RefCell::new(StdRng::seed_from_u64(12345));
let inputs = InputPair::new(
    || [0u8; 32],
    || {
        let mut rng = rng.borrow_mut();
        let mut arr = [0u8; 32];
        for byte in &mut arr {
            *byte = rng.random();
        }
        arr
    },
);
```

**Key points:**
- Use `Cell<T>` for copy types (u64, usize, etc.)
- Use `RefCell<T>` for non-copy types (RNG, Vec, etc.)
- Interior mutability allows state changes inside `Fn` closures
- Both baseline and sample can share state if needed

**Reference:** See `tests/aead_timing.rs:69-76` for real usage

---

### 3.4 Pattern: Pre-Generation Pool (Expensive Inputs)

**Use case:** When input generation is expensive (RSA keys, large structures), pre-generate a pool and cycle through it.

**Complete example:**
```rust
use std::cell::Cell;
use timing_oracle::{TimingOracle, helpers::InputPair};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn test_with_expensive_inputs() {
    const POOL_SIZE: usize = 1000;

    // Pre-generate expensive inputs ONCE
    let mut rng = StdRng::seed_from_u64(0x1234_5678);
    let mut input_pool: Vec<[u8; 128]> = Vec::with_capacity(POOL_SIZE);

    for _ in 0..POOL_SIZE {
        let mut data = [0u8; 128];
        for byte in &mut data {
            *byte = rng.random();
        }
        input_pool.push(data);
    }

    // Cycle through pre-generated pool using Cell index
    let idx = Cell::new(0);
    let inputs = InputPair::new(
        || [0u8; 128],  // Baseline: simple
        move || {
            let i = idx.get();
            idx.set((i + 1) % POOL_SIZE);
            input_pool[i]  // Returns pre-generated value
        },
    );

    let outcome = TimingOracle::balanced()
        .samples(20_000)  // More samples than pool size - will cycle
        .test(inputs, |data| {
            expensive_operation(data);
        });

    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}

fn expensive_operation(_data: &[u8; 128]) {
    // Your expensive crypto operation
}
```

**When to use:**

| **Input Type** | **Generation Cost** | **Pattern** |
|---------------|-------------------|-------------|
| `[u8; 32]` | <1μs | Direct: `\|\| rand::random()` |
| Large structs | 1-100μs | Pool: Pre-generate 1,000+ values |
| RSA keys | >1ms | Pool: Pre-generate 100-500 keys |
| ECC points | 10-100μs | Depends: Pool if >10μs |

**Key points:**
- Pre-generate pool OUTSIDE of InputPair closures
- Use `Cell<usize>` to cycle through pool index
- Pool can be smaller than total samples (will wrap around)
- Ensures generation cost doesn't dominate measurement
- Use seeded RNG for deterministic pools (reproducibility)

**Reference:** See `tests/aes_timing.rs:182-211` for real usage

---

### 3.5 Common Mistakes Checklist

#### ❌ Mistake #1: The Capture Bug (MOST COMMON)

**Problem:** Capturing a pre-evaluated value instead of a closure.

```rust
// ❌ WRONG: Captures single pre-evaluated value
let random_data = rand::random::<[u8; 32]>();  // Evaluated ONCE
let inputs = InputPair::new(
    || [0u8; 32],
    || random_data,  // ❌ Returns SAME value every time!
);
```

**Detection:** timing-oracle will warn:
```
ANOMALY: sample() returned identical values for all 10000 samples.

Common causes:
1. CLOSURE CAPTURE BUG: You may have captured a pre-evaluated value.
   ❌ Bad:  let val = random(); InputPair::new(|| baseline, || val)
   ✓  Good: InputPair::new(|| baseline, || random())
```

**Fix:**
```rust
// ✓ CORRECT: Closure generates new value each call
let inputs = InputPair::new(
    || [0u8; 32],
    || rand::random::<[u8; 32]>(),  // ✓ New value every time
);
```

---

#### ❌ Mistake #2: Generating Inputs Inside Measured Closure

**Problem:** RNG overhead is measured, drowning out the signal.

```rust
// ❌ WRONG: Measures rand::random() overhead
let outcome = TimingOracle::balanced().test(
    InputPair::new(|| (), || ()),
    |_| {
        let data = rand::random::<[u8; 32]>();  // ❌ Measured!
        crypto_function(&data);
    },
);
```

**Fix:**
```rust
// ✓ CORRECT: Pre-generate inputs
let inputs = InputPair::new(
    || [0u8; 32],
    || rand::random::<[u8; 32]>(),  // Generated BEFORE timing
);

let outcome = TimingOracle::balanced().test(inputs, |data| {
    crypto_function(data);  // ✓ Only this is measured
});

fn crypto_function(_data: &[u8; 32]) {}
```

---

#### ❌ Mistake #3: Different Code Paths

**Problem:** Testing different operations instead of data-dependent timing.

```rust
// ❌ WRONG: Baseline and sample execute DIFFERENT code
let inputs = InputPair::new(
    || true,   // Baseline takes fast path
    || false,  // Sample takes slow path
);

let outcome = TimingOracle::balanced().test(inputs, |&secret| {
    if secret {
        fast_operation();    // Different code!
    } else {
        slow_operation();    // Different code!
    }
});

fn fast_operation() {}
fn slow_operation() {}
```

**Fix:**
```rust
// ✓ CORRECT: Same code path, different data
let inputs = InputPair::new(
    || [0xABu8; 32],              // Baseline data
    || rand::random::<[u8; 32]>(),  // Sample data
);

let outcome = TimingOracle::balanced().test(inputs, |data| {
    crypto_operation(data);  // ✓ Same operation for both
});

fn crypto_operation(_data: &[u8; 32]) {}
```

---

#### ❌ Mistake #4: Using Values Instead of Closures

**Problem:** Passing values directly instead of closures.

```rust
// ❌ WRONG: Won't compile
let baseline_value = [0u8; 32];
let sample_value = rand::random::<[u8; 32]>();
let inputs = InputPair::new(baseline_value, sample_value);
// Error: expected closure, found array
```

**Fix:**
```rust
// ✓ CORRECT: Wrap in closures
let baseline_value = [0u8; 32];
let inputs = InputPair::new(
    || baseline_value,              // ✓ Closure returns value
    || rand::random::<[u8; 32]>(),  // ✓ Closure generates value
);
```

---

### 3.6 Pattern: Non-Hashable Types (Crypto Primitives)

**Use case:** Testing with types that don't implement `Hash` (elliptic curve scalars, field elements, big integers).

**Problem:** Some crypto types don't implement `Hash`:
```rust
// Won't compile if Scalar doesn't implement Hash
let inputs = InputPair::new(
    || Scalar::zero(),
    || Scalar::random(&mut rng),
);
// Error: Scalar doesn't implement Hash
```

**Solution: Use `new_untracked()`**

```rust
use timing_oracle::helpers::InputPair;

#[test]
fn test_ecc_scalar_multiplication() {
    // For types that don't implement Hash
    let inputs = InputPair::new_untracked(
        || Scalar::zero(),
        || Scalar::random(&mut thread_rng()),
    );

    let outcome = TimingOracle::balanced().test(inputs, |scalar| {
        scalar_multiply(scalar);
    });

    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}

// Placeholder types for example
struct Scalar;
impl Scalar {
    fn zero() -> Self { Scalar }
    fn random(_rng: &mut impl rand::Rng) -> Self { Scalar }
}

fn scalar_multiply(_scalar: &Scalar) {}
```

**Key points:**
- `new_untracked()` disables anomaly detection
- Use for crypto types without `Hash` implementation
- No warning about repeated values (detection disabled)
- Still pre-generates inputs correctly

**Alternative: Implement Hash wrapper:**
```rust
use std::hash::{Hash, Hasher};

#[derive(Clone)]
struct HashableScalar(Scalar);

impl Hash for HashableScalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the byte representation
        self.0.to_bytes().hash(state);
    }
}

// Now can use InputPair::new() with anomaly detection
let inputs = InputPair::new(
    || HashableScalar(Scalar::zero()),
    || HashableScalar(Scalar::random(&mut thread_rng())),
);
```

**Reference:** See `src/helpers.rs:311-333`

---

### 3.7 Pattern: Intentional Deterministic Inputs

**Use case:** Testing operations where sample inputs are intentionally fixed (testing nonce independence, pre-generated signatures).

**Problem:** Anomaly detection will warn about identical values:

```rust
let inputs = InputPair::new(
    || [0x00u8; 12],  // Nonce A
    || [0xFFu8; 12],  // Nonce B (same every time)
);
// After test: "ANOMALY: sample() returned identical values..."
```

**Solution: Use `new_unchecked()` to suppress warnings**

```rust
use timing_oracle::helpers::InputPair;

#[test]
fn test_nonce_independence() {
    // Intentionally using fixed nonces - this is OK
    let inputs = InputPair::new_unchecked(
        || [0x00u8; 12],  // Fixed nonce A
        || [0xFFu8; 12],  // Fixed nonce B
    );

    let outcome = TimingOracle::balanced().test(inputs, |nonce| {
        encrypt_with_fixed_key(nonce);
    });

    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}

fn encrypt_with_fixed_key(_nonce: &[u8; 12]) {
    // Encryption with same key, different nonces
}
```

**When to use `new_unchecked()`:**

| **Use Case** | **Baseline** | **Sample** | **Use `new_unchecked()`?** |
|-------------|--------------|------------|---------------------------|
| Standard DudeCT | `[0u8; 32]` | `rand::random()` | No - use `new()` |
| Nonce independence | Fixed nonce A | Fixed nonce B | Yes ✓ |
| Pre-generated sigs | Signature 1 | Signature 2 | Yes ✓ |
| Key comparison | Key A | Key B | Yes ✓ |

**Key points:**
- `new_unchecked()` = `new()` but disables anomaly detection
- Use when you INTENTIONALLY use deterministic sample inputs
- Still pre-generates inputs correctly (no performance impact)
- Prevents false warnings in legitimate use cases

**Reference:** See `src/helpers.rs:156-164`

---

**Next:** [§4 Crypto-Specific Patterns](#4-crypto-specific-patterns) - Real-world crypto testing ⚠️

---

## 4. Crypto-Specific Patterns

**Real-world cryptographic testing patterns.** Each crypto primitive has unique testing requirements and common pitfalls.

**Contents:**
- [4.1 AEAD Ciphers](#41-aead-ciphers-critical) ⚠️ **CRITICAL - Nonce reuse pitfall**
- [4.2 Block Ciphers](#42-block-ciphers)
- [4.3 Hash Functions](#43-hash-functions)
- [4.4 Elliptic Curve Cryptography](#44-elliptic-curve-cryptography)
- [4.5 RSA](#45-rsa)
- [4.6 Post-Quantum Cryptography](#46-post-quantum-cryptography)

---

### 4.1 AEAD Ciphers (CRITICAL)

⚠️ **CRITICAL SECTION**: AEAD testing has a common pitfall that violates security requirements AND produces measurement artifacts. Read carefully.

**AEAD = Authenticated Encryption with Associated Data**
- Examples: ChaCha20-Poly1305, AES-GCM, AES-256-GCM
- Use cases: TLS, QUIC, messaging, file encryption
- **Security requirement**: Each (key, nonce) pair must encrypt AT MOST ONE message

---

#### 4.1.1 THE NONCE REUSE BUG ⚠️

**The #1 AEAD testing mistake:**

```rust
// ❌ CATASTROPHICALLY WRONG: Nonce reused for every encryption
let cipher = ChaCha20Poly1305::new(&key.into());
let nonce = [0u8; 12];  // ❌ SAME nonce for all encryptions!

let inputs = InputPair::new(|| plaintext_a, || plaintext_b);

let outcome = TimingOracle::balanced().test(inputs, |plaintext| {
    // ❌ Encrypts 20,000+ messages with SAME nonce!
    // Violates AEAD security + creates measurement artifacts
    let ciphertext = cipher.encrypt(&nonce.into(), plaintext).unwrap();
    std::hint::black_box(ciphertext[0]);
});
```

**Why this is catastrophic:**

1. **Security violation**: Reusing (key, nonce) breaks AEAD confidentiality
   - ChaCha20-Poly1305: XOR keystream reuse → plaintext recovery
   - AES-GCM: Counter mode reuse → authentication forgery

2. **Measurement artifacts**: Nonce reuse can create CPU cache effects
   - Same nonce → same AES-CTR counter blocks → cache hits
   - Different nonce → different blocks → cache misses
   - Creates false timing dependencies on nonce value

**The fix:** Generate **unique nonces** for each encryption using an atomic counter.

---

#### 4.1.2 Pattern: AEAD Encryption with Atomic Nonce Counter ✓

**Use case:** Testing AEAD encryption (ChaCha20-Poly1305, AES-GCM) with unique nonces.

**Complete example:**
```rust
use chacha20poly1305::{ChaCha20Poly1305, Nonce, aead::{Aead, KeyInit}};
use std::sync::atomic::{AtomicU64, Ordering};
use timing_oracle::{TimingOracle, helpers::InputPair};

#[test]
fn test_chacha20poly1305_encryption() {
    // Fixed key
    let key_bytes = [0x42u8; 32];
    let cipher = ChaCha20Poly1305::new(&key_bytes.into());

    // ✓ CORRECT: Atomic counter for unique nonces
    let nonce_counter = AtomicU64::new(0);

    // Fixed vs random plaintexts (DudeCT pattern)
    let fixed_plaintext = [0x00u8; 64];
    let inputs = InputPair::new(
        || fixed_plaintext,
        || rand::random::<[u8; 64]>(),
    );

    let outcome = TimingOracle::balanced()
        .samples(30_000)
        .alpha(0.01)
        .test(inputs, |plaintext| {
            // Generate UNIQUE nonce for each encryption
            let nonce_value = nonce_counter.fetch_add(1, Ordering::Relaxed);
            let mut nonce_bytes = [0u8; 12];
            nonce_bytes[..8].copy_from_slice(&nonce_value.to_le_bytes());
            let nonce = Nonce::from_slice(&nonce_bytes);

            // Each encryption uses a DIFFERENT nonce
            let ciphertext = cipher.encrypt(nonce, plaintext.as_ref()).unwrap();
            std::hint::black_box(ciphertext[0]);
        });

    let result = outcome.unwrap_completed();

    assert!(result.ci_gate.passed);
    assert!(result.leak_probability < 0.3);
}
```

**Key points:**
- `AtomicU64::new(0)` creates nonce counter
- `fetch_add(1, Ordering::Relaxed)` increments atomically
- Each encryption gets unique nonce (0, 1, 2, ...)
- Encode counter into 12-byte nonce (little-endian)
- Works for 30,000+ samples (fits in u64)

**Why atomic?**
```rust
// ❌ WRONG: Cell doesn't work (not Clone)
let nonce_counter = Cell::new(0u64);
let inputs = InputPair::new(
    || ...,
    || ...,
);
// Error: AtomicU64 moved into closure but needed in test closure

// ✓ CORRECT: AtomicU64 can be shared between closures
let nonce_counter = AtomicU64::new(0);
// Both InputPair closures and test closure can access it
```

**Reference:** See `tests/aead_timing.rs:63-82`

---

#### 4.1.3 Pattern: AEAD Decryption with Pre-Generated Ciphertexts

**Use case:** Testing AEAD decryption timing.

**Complete example:**
```rust
use chacha20poly1305::{ChaCha20Poly1305, Nonce, aead::{Aead, KeyInit}};
use std::cell::Cell;
use timing_oracle::{TimingOracle, helpers::InputPair};

#[test]
fn test_chacha20poly1305_decryption() {
    const SAMPLES: usize = 30_000;

    let key_bytes = [0x42u8; 32];
    let cipher = ChaCha20Poly1305::new(&key_bytes.into());
    let nonce = Nonce::from_slice(&[0u8; 12]);

    // Pre-generate ciphertexts ONCE (outside InputPair)
    let fixed_plaintext = [0x00u8; 64];
    let fixed_ciphertext = cipher
        .encrypt(nonce, fixed_plaintext.as_ref())
        .unwrap();

    let random_ciphertexts: Vec<Vec<u8>> = (0..SAMPLES)
        .map(|_| {
            let pt = rand::random::<[u8; 64]>();
            cipher.encrypt(nonce, pt.as_ref()).unwrap()
        })
        .collect();

    // Cycle through pre-generated pool
    let idx = Cell::new(0usize);
    let fixed_ciphertext_clone = fixed_ciphertext.clone();
    let inputs = InputPair::new(
        move || fixed_ciphertext_clone.clone(),
        move || {
            let i = idx.get();
            idx.set((i + 1) % SAMPLES);
            random_ciphertexts[i].clone()
        },
    );

    let outcome = TimingOracle::balanced()
        .samples(SAMPLES)
        .alpha(0.01)
        .test(inputs, |ciphertext| {
            let plaintext = cipher.decrypt(nonce, ciphertext.as_ref()).unwrap();
            std::hint::black_box(plaintext[0]);
        });

    let result = outcome.unwrap_completed();

    assert!(result.ci_gate.passed);
    assert!(result.leak_probability < 0.3);
}
```

**Key points:**
- Pre-generate ALL ciphertexts before measurement
- Avoids measuring encryption overhead during decryption test
- Use Cell index to cycle through pre-generated pool
- Pool size matches sample count (no wrapping needed)

**Why pre-generate?**

| **Approach** | **Overhead** | **Correctness** |
|-------------|-------------|----------------|
| Generate in closure | Measures encryption + decryption | ❌ Wrong |
| Pre-generate pool | Measures only decryption | ✓ Correct |

**Reference:** See `tests/aead_timing.rs:126-144`

---

#### 4.1.4 Pattern: Testing Nonce Independence

**Use case:** Verify that encryption timing doesn't depend on nonce value.

**Complete example:**
```rust
use chacha20poly1305::{ChaCha20Poly1305, Nonce, aead::{Aead, KeyInit}};
use timing_oracle::{TimingOracle, helpers::InputPair};

#[test]
fn test_nonce_independence() {
    let key_bytes = [0x42u8; 32];
    let cipher = ChaCha20Poly1305::new(&key_bytes.into());

    let plaintext = [0x00u8; 64];  // Fixed plaintext

    // Test two DIFFERENT nonces (deterministic)
    // Use new_unchecked to suppress "identical values" warning
    let nonces = InputPair::new(
        || [0x00u8; 12],  // Nonce A (all zeros)
        || [0xFFu8; 12],  // Nonce B (all ones)
    );

    let outcome = TimingOracle::balanced()
        .samples(30_000)
        .test(nonces, |nonce_bytes| {
            let nonce = Nonce::from_slice(nonce_bytes);
            let ciphertext = cipher.encrypt(nonce, plaintext.as_ref()).unwrap();
            std::hint::black_box(ciphertext[0]);
        });

    let result = outcome.unwrap_completed();

    // For nonce independence:
    // - CI gate should pass (no statistically significant difference)
    // - Exploitability should be Negligible (any difference is tiny)
    assert!(result.ci_gate.passed);
    assert!(matches!(
        result.exploitability,
        timing_oracle::Exploitability::Negligible
    ));
}
```

**Key points:**
- Use FIXED plaintext (only nonce varies)
- Use deterministic nonces (0x00 vs 0xFF)
- Focus on exploitability, not just leak probability
- Small effects (5-10ns) may be statistically detectable but not exploitable

**Why deterministic nonces?**
- Avoids cache locality differences from random generation
- Makes test reproducible
- Both nonces are equally "pathological" (all same bits)

**Reference:** See `tests/aead_timing.rs:189-219`

---

#### 4.1.5 Quick Reference: AEAD Patterns

| **Test Type** | **Input Varies** | **Nonce Strategy** | **Key Point** |
|--------------|------------------|-------------------|--------------|
| Encryption | Plaintext | Atomic counter | ✓ Unique nonce per encryption |
| Decryption | Ciphertext | Fixed nonce OK | Pre-generate ciphertexts |
| Nonce independence | Nonce | Two fixed nonces | Fixed plaintext |
| Key independence | Key | Fixed nonce OK | Different keys, same plaintext |

**Common AEAD libraries:**

```rust
// ChaCha20-Poly1305 (RustCrypto)
use chacha20poly1305::{ChaCha20Poly1305, Nonce, aead::{Aead, KeyInit}};
let cipher = ChaCha20Poly1305::new(&key.into());
let ct = cipher.encrypt(nonce, plaintext).unwrap();

// AES-256-GCM (ring)
use ring::aead::{self, LessSafeKey, UnboundKey, Nonce, AES_256_GCM};
let unbound_key = UnboundKey::new(&AES_256_GCM, &key).unwrap();
let key = LessSafeKey::new(unbound_key);
let nonce = Nonce::assume_unique_for_key(nonce_bytes);
key.seal_in_place_append_tag(nonce, aead::Aad::empty(), &mut plaintext).unwrap();
```

---

### 4.2 Block Ciphers

**Use case:** Testing block cipher implementations (AES-128/256, etc.)

**Complete example:**
```rust
use aes::Aes128;
use aes::cipher::{BlockEncrypt, KeyInit};
use timing_oracle::{TimingOracle, helpers::InputPair};

#[test]
fn test_aes128_encryption() {
    // Fixed key
    let key_bytes = [0x42u8; 16];
    let cipher = Aes128::new(&key_bytes.into());

    // Fixed vs random plaintexts (DudeCT pattern)
    let inputs = InputPair::new(
        || [0x00u8; 16],  // Fixed block (all zeros)
        || rand::random::<[u8; 16]>(),  // Random blocks
    );

    let outcome = TimingOracle::balanced()
        .samples(30_000)
        .test(inputs, |plaintext| {
            let mut block = (*plaintext).into();
            cipher.encrypt_block(&mut block);
            std::hint::black_box(block[0]);
        });

    let result = outcome.unwrap_completed();

    assert!(result.ci_gate.passed);
    assert!(result.leak_probability < 0.3);
}
```

**Key points:**
- Use DudeCT pattern: fixed block vs random blocks
- Test single-block encryption first
- For multi-block: use cipher modes (CBC, CTR) with proper IV handling
- Modern AES-NI implementations should be constant-time

**Reference:** See `tests/aes_timing.rs:63-94`

---

### 4.3 Hash Functions

**Use case:** Testing hash function implementations (SHA-3, BLAKE2, etc.)

**Complete example:**
```rust
use sha3::{Sha3_256, Digest};
use timing_oracle::{TimingOracle, helpers::byte_arrays_32};

#[test]
fn test_sha3_256_constant_time() {
    let inputs = byte_arrays_32();  // Fixed vs random 32-byte inputs

    let outcome = TimingOracle::balanced()
        .samples(30_000)
        .test(inputs, |data| {
            let mut hasher = Sha3_256::new();
            hasher.update(data);
            let result = hasher.finalize();
            std::hint::black_box(result[0]);
        });

    let result = outcome.unwrap_completed();

    assert!(result.ci_gate.passed);
    assert!(result.leak_probability < 0.3);
}
```

**Key points:**
- One-shot hashing: create hasher, update, finalize
- For incremental hashing: test update() and finalize() separately
- Hash functions should be constant-time (no secret-dependent branches)
- Test with various input sizes if relevant

**Variable-length example:**
```rust
use timing_oracle::helpers::byte_vecs;

let inputs = byte_vecs(1024);  // 1KB messages
let outcome = TimingOracle::balanced().test(inputs, |data| {
    let hash = sha3::Sha3_256::digest(data);
    std::hint::black_box(hash[0]);
});
```

**Reference:** See `tests/hash_timing.rs:12-52`

---

### 4.4 Elliptic Curve Cryptography

**Use case:** Testing ECC operations (scalar multiplication, ECDH)

**Complete example:**
```rust
use x25519_dalek::{x25519, X25519_BASEPOINT_BYTES};
use timing_oracle::{TimingOracle, helpers::InputPair};

#[test]
fn test_x25519_scalar_mult() {
    let basepoint = X25519_BASEPOINT_BYTES;

    // ⚠️ IMPORTANT: Use valid scalar (NOT all-zeros which is pathological)
    let fixed_scalar: [u8; 32] = [
        0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82,
        0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a, 0x33, 0x9c,
        0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f,
        0x9a, 0x2d, 0x7e, 0x4c, 0x8b, 0x3a, 0x6d, 0x5e,
    ];

    let inputs = InputPair::new(
        || fixed_scalar,
        || rand::random::<[u8; 32]>(),
    );

    let outcome = TimingOracle::balanced()
        .samples(50_000)
        .test(inputs, |scalar| {
            let result = x25519(*scalar, basepoint);
            std::hint::black_box(result);
        });

    let result = outcome.unwrap_completed();

    assert!(result.ci_gate.passed);
    assert!(result.leak_probability < 0.3);
}
```

**Key points:**
- ⚠️ **Critical**: DON'T use all-zeros scalar (pathological edge case)
- Use non-pathological fixed scalar (mixed bits)
- X25519 should be constant-time by design
- Test scalar multiplication separately from full ECDH

**ECDH key exchange pattern:**
```rust
#[test]
fn test_ecdh_key_exchange() {
    let alice_secret = [0x42u8; 32];  // Alice's fixed secret
    let bob_public = x25519_dalek::X25519_BASEPOINT_BYTES;

    let inputs = InputPair::new(
        || alice_secret,
        || rand::random::<[u8; 32]>(),  // Different Alice secrets
    );

    let outcome = TimingOracle::balanced().test(inputs, |secret| {
        let shared = x25519(*secret, bob_public);
        std::hint::black_box(shared[0]);
    });

    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}
```

**Reference:** See `tests/ecc_timing.rs:33-81`

---

### 4.5 RSA

**Use case:** Testing RSA encryption/decryption/signing

**Complete example:**
```rust
use rsa::{RsaPrivateKey, RsaPublicKey, Pkcs1v15Encrypt};
use rsa::pkcs1v15::Pkcs1v15Sign;
use timing_oracle::{TimingOracle, helpers::byte_arrays_32};

#[test]
fn test_rsa_encryption() {
    let mut rng = rand::thread_rng();

    // Generate RSA keypair (do this ONCE outside measurement)
    let bits = 2048;
    let private_key = RsaPrivateKey::new(&mut rng, bits).unwrap();
    let public_key = RsaPublicKey::from(&private_key);

    let inputs = byte_arrays_32();  // 32-byte plaintexts

    // Use .quick() preset - RSA is slow (1-10ms per operation)
    let outcome = TimingOracle::quick()
        .samples(5_000)  // Fewer samples for slow operations
        .test(inputs, |plaintext| {
            let ciphertext = public_key
                .encrypt(&mut rand::thread_rng(), Pkcs1v15Encrypt, plaintext)
                .unwrap();
            std::hint::black_box(ciphertext[0]);
        });

    let result = outcome.unwrap_completed();

    assert!(result.ci_gate.passed);
    assert!(result.leak_probability < 0.3);
}
```

**Key points:**
- RSA operations are SLOW (1-10ms each)
- Use `.quick()` preset and fewer samples (5,000-10,000)
- Generate keypair ONCE outside measurement
- Testing decryption: pre-generate ciphertexts (like AEAD pattern)
- Blinding should prevent timing attacks in modern implementations

**Reference:** See `tests/rsa_timing.rs:15-60`

---

### 4.6 Post-Quantum Cryptography

**Use case:** Testing PQ crypto (ML-KEM, ML-DSA, Falcon, SPHINCS+)

**Complete example (ML-KEM encapsulation):**
```rust
use pqcrypto_kyber::kyber768::*;
use timing_oracle::{TimingOracle, helpers::InputPair};

#[test]
fn test_mlkem_encapsulation() {
    // Generate keypair ONCE
    let (public_key, _secret_key) = keypair();

    // Test encapsulation with random data
    let inputs = InputPair::new(
        || [0u8; 32],  // Fixed random seed
        || rand::random::<[u8; 32]>(),
    );

    let outcome = TimingOracle::balanced()
        .samples(20_000)
        .test(inputs, |seed| {
            // Encapsulate with deterministic RNG from seed
            let (ciphertext, _shared_secret) = encapsulate(&public_key);
            std::hint::black_box(ciphertext.as_bytes()[0]);
        });

    let result = outcome.unwrap_completed();

    assert!(result.ci_gate.passed);
    assert!(result.leak_probability < 0.3);
}
```

**Key points:**
- PQ crypto operations are SLOWER than classical crypto
- ML-KEM (Kyber): ~100μs for encap/decap
- ML-DSA (Dilithium): ~500μs for sign, ~200μs for verify
- Use `.balanced()` or `.quick()` presets
- Test encapsulation, decapsulation, signing, verification separately

**Reference:** See `tests/pqcrypto_timing.rs:14-68`

---

**Next:** [§5 Advanced Patterns](#5-advanced-patterns) - Async, complex types, state

---

## 5. Advanced Patterns

**Advanced testing scenarios:** Async/await code, complex types, stateful operations, and batch testing.

### 5.1 Async/Await Code

**Use case:** Testing async operations (async fn, futures, Tokio tasks) for timing leaks.

**Key principle:** Create the runtime ONCE outside the measured closure, then use `block_on()` inside.

#### Pattern: Single-Threaded Tokio Runtime (Recommended)

**Use case:** Minimal jitter for precise measurements.

**Complete example:**
```rust
use timing_oracle::{TimingOracle, helpers::InputPair};
use tokio::runtime::Runtime;
use tokio::time::{sleep, Duration};

#[test]
fn test_async_constant_time_operation() {
    // Create runtime ONCE (outside measurement)
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("failed to create runtime");

    // Fixed vs random inputs (DudeCT pattern)
    let fixed_input: [u8; 32] = [0x42; 32];
    let inputs = InputPair::new(
        || fixed_input,
        || rand::random::<[u8; 32]>(),
    );

    let outcome = TimingOracle::balanced()
        .samples(10_000)
        .test(inputs, |data| {
            // Use block_on() to bridge async → sync
            rt.block_on(async {
                // Your async operation here
                std::hint::black_box(data);
            })
        });

    let result = outcome.unwrap_completed();

    assert!(result.ci_gate.passed);
    assert!(result.leak_probability < 0.3);
}
```

**Key points:**
- Create runtime ONCE outside `test()` closure
- Use `.new_current_thread()` for lowest noise (single-threaded)
- Call `.enable_time()` if using tokio::time (sleep, interval, etc.)
- Use `rt.block_on(async { ... })` inside measured closure
- Runtime overhead is constant (same for all samples)

**Reference:** See `tests/async_timing.rs:22-27`, `tests/async_timing.rs:56-93`

---

#### Pattern: Multi-Threaded Runtime (Stress Testing)

**Use case:** Testing under concurrent load or simulating production conditions.

**Complete example:**
```rust
use timing_oracle::{TimingOracle, helpers::InputPair};
use tokio::runtime::Runtime;
use tokio::time::{sleep, Duration};

#[test]
fn test_async_with_background_tasks() {
    // Multi-threaded runtime (2 workers)
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_time()
        .build()
        .expect("failed to create runtime");

    let fixed_input: [u8; 32] = [0x42; 32];
    let inputs = InputPair::new(
        || fixed_input,
        || rand::random::<[u8; 32]>(),
    );

    let outcome = TimingOracle::balanced()
        .samples(10_000)
        .test(inputs, |data| {
            rt.block_on(async {
                // Spawn background tasks to simulate load
                for _ in 0..10 {
                    tokio::spawn(async {
                        sleep(Duration::from_micros(100)).await;
                    });
                }
                std::hint::black_box(data);
            })
        });

    let result = outcome.unwrap_completed();

    // Background tasks may increase noise
    assert!(result.ci_gate.passed);
}
```

**When to use:**

| **Runtime Type** | **Use For** | **Noise Level** |
|-----------------|-------------|----------------|
| Single-threaded | Precise measurements | Low |
| Multi-threaded (2 workers) | Stress testing | Medium |
| Multi-threaded (4+ workers) | Production simulation | High |

**Reference:** See `tests/async_timing.rs:30-36`, `tests/async_timing.rs:250-302`

---

#### Pattern: Detecting Async Timing Leaks

**Use case:** Testing for secret-dependent await patterns or conditional async operations.

**Complete example:**
```rust
use timing_oracle::{TimingOracle, helpers::InputPair};
use tokio::runtime::Runtime;
use tokio::time::{sleep, Duration};

#[test]
fn detects_conditional_await_timing() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();

    // Secret-dependent logic: true vs false
    let inputs = InputPair::new(|| true, || false);

    let outcome = TimingOracle::balanced()
        .samples(50_000)
        .test(inputs, |secret| {
            rt.block_on(async {
                if *secret {
                    // Extra await when secret is true (timing leak!)
                    sleep(Duration::from_nanos(100)).await;
                }
                sleep(Duration::from_micros(5)).await;
                std::hint::black_box(42);
            })
        });

    let result = outcome.unwrap_completed();

    // Should detect the leak
    assert!(!result.ci_gate.passed, "Should detect conditional await");
    assert!(result.leak_probability > 0.7);
}
```

**Reference:** See `tests/async_timing.rs:139-170`

---

### 5.2 Complex Types

**Use case:** Testing with structs, tuples, or custom types.

**Complete example:**
```rust
use timing_oracle::{TimingOracle, helpers::InputPair};

#[derive(Clone)]
struct CryptoContext {
    key: [u8; 32],
    nonce: [u8; 12],
    associated_data: Vec<u8>,
}

#[test]
fn test_with_struct() {
    // Fixed context
    let fixed_ctx = CryptoContext {
        key: [0x42; 32],
        nonce: [0x00; 12],
        associated_data: vec![],
    };

    let inputs = InputPair::new(
        move || fixed_ctx.clone(),
        || CryptoContext {
            key: rand::random(),
            nonce: rand::random(),
            associated_data: vec![],
        },
    );

    let outcome = TimingOracle::balanced().test(inputs, |ctx| {
        encrypt_with_context(ctx);
    });

    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}

fn encrypt_with_context(_ctx: &CryptoContext) {
    // Your encryption logic
}
```

**Tuple pattern:**
```rust
// Testing with tuples (key, plaintext)
let inputs = InputPair::new(
    || ([0x42u8; 32], [0x00u8; 64]),  // Fixed (key, plaintext)
    || (rand::random::<[u8; 32]>(), rand::random::<[u8; 64]>()),
);

let outcome = TimingOracle::balanced().test(inputs, |(key, plaintext)| {
    encrypt(key, plaintext);
});

fn encrypt(_key: &[u8; 32], _plaintext: &[u8; 64]) {}
```

**Key points:**
- Struct must implement `Clone` and `Hash` (or use `new_untracked()`)
- Use `move ||` closures to capture struct by value
- For borrowed data: use lifetimes or `'static` data

---

### 5.3 Stateful Operations

**Use case:** Testing operations that maintain state (caches, databases, connection pools).

**Pattern: `test_with_state()` API**

```rust
use timing_oracle::{TimingOracle, helpers::InputPair};
use std::collections::HashMap;

#[test]
fn test_stateful_cache() {
    // Shared mutable state
    let mut cache: HashMap<[u8; 32], Vec<u8>> = HashMap::new();

    let inputs = InputPair::new(
        || [0x00u8; 32],  // Cache hit (warm)
        || rand::random::<[u8; 32]>(),  // Cache miss (cold)
    );

    // Pre-warm cache with fixed key
    cache.insert([0x00u8; 32], vec![0xAB; 64]);

    let outcome = TimingOracle::balanced()
        .samples(20_000)
        .test_with_state(&mut cache, inputs, |cache, key| {
            // Access cache (may hit or miss)
            let value = cache.entry(*key).or_insert_with(|| vec![0xCD; 64]);
            std::hint::black_box(value[0]);
        });

    let result = outcome.unwrap_completed();

    // Cache timing may or may not leak (implementation-dependent)
    // This example would likely show a leak due to cache hit/miss difference
    eprintln!("Leak probability: {:.1}%", result.leak_probability * 100.0);
}
```

**Key points:**
- Use `test_with_state(&mut state, inputs, |state, input| {...})`
- State persists across all samples (realistic testing)
- Useful for: caches, database connections, connection pools
- Pre-warm state if testing hot path behavior

**Reference:** See `guide.md` examples

---

### 5.4 Batch Testing

**Use case:** Testing cumulative effects over multiple operations.

**Complete example:**
```rust
use timing_oracle::{TimingOracle, helpers::InputPair};

#[test]
fn test_multiple_aes_blocks() {
    use aes::Aes128;
    use aes::cipher::{BlockEncrypt, KeyInit};

    let key_bytes = [0x42u8; 16];
    let cipher = Aes128::new(&key_bytes.into());

    // Test 4 blocks per sample
    let fixed_blocks = [[0x00u8; 16]; 4];
    let inputs = InputPair::new(
        || fixed_blocks,
        || [rand::random::<[u8; 16]>(); 4],
    );

    let outcome = TimingOracle::balanced().test(inputs, |blocks| {
        for block_data in blocks {
            let mut block = (*block_data).into();
            cipher.encrypt_block(&mut block);
            std::hint::black_box(block[0]);
        }
    });

    let result = outcome.unwrap_completed();

    assert!(result.ci_gate.passed);
    assert!(result.leak_probability < 0.3);
}
```

**Key points:**
- Test multiple operations per sample (realistic workload)
- Cumulative timing may amplify small leaks
- Useful for: block cipher modes, hash tree construction, signature batching

**Reference:** See `tests/aes_timing.rs:176-223`

---

**Next:** [§6 Integration Patterns](#6-integration-patterns) - CI/CD, test organization

---

## 6. Integration Patterns

**Integrating timing-oracle into your development workflow:** CI/CD, test organization, environment configuration.

### 6.1 CI/CD Integration

**Use case:** Running timing tests in continuous integration pipelines.

**Pattern: Environment Variable Overrides**

```rust
use timing_oracle::{TimingOracle, helpers::byte_arrays_32};

#[test]
fn test_crypto_operation_ci() {
    let inputs = byte_arrays_32();

    // Use .from_env() to allow CI to override settings via environment variables
    let outcome = TimingOracle::balanced()
        .from_env()  // Reads TO_SAMPLES, TO_ALPHA, TO_MIN_EFFECT_NS
        .test(inputs, |data| {
            crypto_operation(data);
        });

    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}

fn crypto_operation(_data: &[u8; 32]) {}
```

**CI configuration examples:**

```bash
# GitHub Actions (.github/workflows/test.yml)
- name: Run timing tests
  run: cargo test --test crypto_timing
  env:
    TO_SAMPLES: "10000"      # Fewer samples for faster CI
    TO_ALPHA: "0.05"         # More lenient for noisy CI environments
    TO_MIN_EFFECT_NS: "100"  # Higher noise floor

# GitLab CI (.gitlab-ci.yml)
test:timing:
  script:
    - cargo test --test timing_tests
  variables:
    TO_SAMPLES: "10000"
    TO_ALPHA: "0.05"

# Local development (faster iteration)
export TO_SAMPLES=5000
export TO_ALPHA=0.1
cargo test
```

**Key points:**
- `.from_env()` reads environment variables to configure the oracle
- Use fewer samples in CI (faster, but less power)
- Use appropriate `AttackerModel` for your threat scenario

---

### 6.2 Test Organization

**Pattern: Shared Test Configuration**

```rust
// tests/common/mod.rs
use timing_oracle::TimingOracle;

pub fn ci_oracle() -> TimingOracle {
    TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .from_env()
}

pub fn thorough_oracle() -> TimingOracle {
    TimingOracle::new()
        .samples(100_000)
        .alpha(0.001)
}

// tests/aes_timing.rs
mod common;

#[test]
fn test_aes_encryption() {
    let inputs = byte_arrays_32();
    let outcome = common::ci_oracle().test(inputs, |data| {
        aes_encrypt(data);
    });
    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}

fn byte_arrays_32() -> timing_oracle::helpers::InputPair<[u8; 32], impl FnMut() -> [u8; 32], impl FnMut() -> [u8; 32]> {
    timing_oracle::helpers::InputPair::new(|| [0u8; 32], || rand::random())
}

fn aes_encrypt(_data: &[u8; 32]) {}
```

**Pattern: Test Categories with `#[ignore]`**

```rust
// Fast tests (run by default)
#[test]
fn quick_aes_test() {
    let outcome = TimingOracle::quick().test(/* ... */);
    // ...
}

// Thorough tests (run with --ignored)
#[test]
#[ignore = "slow test - run with --ignored"]
fn thorough_aes_test() {
    let outcome = TimingOracle::new()
        .samples(200_000)
        .test(/* ... */);
    // ...
}

// Run all tests including ignored:
// cargo test -- --ignored
```

**Key points:**
- Create shared oracle configurations in `tests/common/mod.rs`
- Use `#[ignore]` for slow/thorough tests
- Organize tests by crypto primitive (aes_timing.rs, rsa_timing.rs, etc.)
- Run fast tests in PR checks, thorough tests nightly

---

### 6.3 Platform-Specific Configuration

**Pattern: Conditional Timer Selection**

```rust
use timing_oracle::TimingOracle;

#[test]
fn test_with_best_timer() {
    let inputs = byte_arrays_32();

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let oracle = {
        // Use PMU timer on macOS ARM64 (requires sudo)
        use timing_oracle::measurement::PmuTimer;
        TimingOracle::balanced()
            .timer(PmuTimer::new().expect("failed to initialize PMU timer"))
    };

    #[cfg(target_os = "linux")]
    let oracle = {
        // Use perf_event timer on Linux (requires sudo or CAP_PERFMON)
        use timing_oracle::measurement::LinuxPerfTimer;
        TimingOracle::balanced()
            .timer(LinuxPerfTimer::new().expect("failed to initialize perf timer"))
    };

    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        target_os = "linux"
    )))]
    let oracle = TimingOracle::balanced();

    let outcome = oracle.test(inputs, |data| {
        crypto_operation(data);
    });

    let result = outcome.unwrap_completed();
    assert!(result.ci_gate.passed);
}

fn byte_arrays_32() -> timing_oracle::helpers::InputPair<[u8; 32], impl FnMut() -> [u8; 32], impl FnMut() -> [u8; 32]> {
    timing_oracle::helpers::InputPair::new(|| [0u8; 32], || rand::random())
}

fn crypto_operation(_data: &[u8; 32]) {}
```

**CI workflow with sudo:**

```yaml
# GitHub Actions
- name: Run timing tests with PMU
  run: sudo -E cargo test --test timing_tests
  # -E preserves environment variables
```

**Key points:**
- PMU/perf timers require elevated privileges
- Use `#[cfg]` for platform-specific timer selection
- CI needs `sudo -E` to run tests with advanced timers
- Standard timer works everywhere (no sudo needed)

---

**Next:** [§7 Troubleshooting Patterns](#7-troubleshooting-patterns) - Fixing common issues

---

## 7. Troubleshooting Patterns

**Fixing common issues:** Unmeasurable operations, high noise, false positives/negatives.

### 7.1 Unmeasurable Operations

**Problem:** Operation completes in <10ns or system noise is too high.

**Pattern: Graceful Handling**

```rust
use timing_oracle::{TimingOracle, helpers::InputPair, skip_if_unreliable};

#[test]
fn test_very_fast_operation() {
    let inputs = InputPair::new(|| 42u64, || rand::random::<u64>());

    let outcome = TimingOracle::balanced().test(inputs, |&x| {
        let result = x.wrapping_add(1);  // Very fast (~1ns)
        std::hint::black_box(result);
    });

    // Skip test if unmeasurable (returns early)
    let result = skip_if_unreliable!(outcome, "test_very_fast_operation");

    assert!(result.ci_gate.passed);
}
```

**Alternative: Require Reliable**

```rust
use timing_oracle::require_reliable;

#[test]
fn test_critical_operation() {
    let outcome = TimingOracle::balanced().test(/* ... */);

    // PANIC if unmeasurable (strict mode)
    let result = require_reliable!(outcome, "test_critical_operation");

    assert!(result.ci_gate.passed);
}
```

**When operation is too fast:**
- Option 1: Batch the operation (test 10-100 iterations per sample)
- Option 2: Use `skip_if_unreliable!` (pragmatic)
- Option 3: Accept that it's too fast to have exploitable timing

---

### 7.2 High Noise / Poor Measurement Quality

**Problem:** `MeasurementQuality::TooNoisy` or very high MDE.

**Solution 1: Increase Sample Count**

```rust
let outcome = TimingOracle::new()
    .samples(200_000)  // 10x more samples
    .test(inputs, |data| {
        crypto_operation(data);
    });

fn crypto_operation(_data: &[u8; 32]) {}
```

**Solution 2: System Sanity Check**

```bash
# Check CPU governor (Linux)
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Should be "performance", not "powersave"

# Set to performance mode
sudo cpupower frequency-set -g performance

# Disable turbo boost (reduces variance)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pmu/turbo_boost

# Isolate CPU cores (advanced)
# Add to kernel boot params: isolcpus=2,3
# Run tests on isolated cores: taskset -c 2 cargo test
```

**Solution 3: Use Cycle-Accurate Timers**

```rust
// macOS ARM64 with sudo
use timing_oracle::measurement::PmuTimer;

let outcome = TimingOracle::balanced()
    .timer(PmuTimer::new()?)
    .test(inputs, |data| {
        crypto_operation(data);
    });

fn crypto_operation(_data: &[u8; 32]) {}
```

**Key points:**
- Increase samples for higher statistical power
- Check system configuration (CPU governor, turbo boost)
- Use PMU/perf timers for best precision (requires sudo)
- Consider running on dedicated hardware (not VMs/containers)

---

### 7.3 False Positives

**Problem:** Constant-time code fails CI gate (leak_probability > 0.7 but code is correct).

**Diagnosis:**

```rust
let outcome = TimingOracle::balanced().test(inputs, |data| {
    constant_time_operation(data);
});

if let timing_oracle::Outcome::Completed(ref r) = outcome {
    eprintln!("{}", timing_oracle::output::format_result(r));
    // Check:
    // - Effect size (is it tiny? <5ns)
    // - Exploitability (is it Negligible?)
    // - Pattern (UniformShift suggests measurement artifact)
}

fn constant_time_operation(_data: &[u8; 32]) {}
```

**Common causes:**

1. **Different code paths** (not the same operation!)
   ```rust
   // ❌ WRONG
   let inputs = InputPair::new(|| true, || false);
   test(inputs, |secret| {
       if *secret { fast_path(); } else { slow_path(); }
   });

   fn fast_path() {}
   fn slow_path() {}
   ```

2. **State leakage** between samples
   ```rust
   // ❌ WRONG: Cache state persists
   static mut CACHE: [u8; 1024] = [0; 1024];
   test(inputs, |key| {
       unsafe { CACHE[key[0] as usize] = 1; }  // Leaks state!
   });
   ```

3. **Compiler optimization** differences
   ```rust
   // Force no optimization for timing-critical code
   #[inline(never)]
   fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
       // ...
       true
   }
   ```

---

### 7.4 False Negatives

**Problem:** Leaky code passes CI gate (should fail but doesn't).

**Diagnosis:**

1. **Increase sample count**
   ```rust
   let outcome = TimingOracle::new()
       .samples(200_000)  // More power
       .test(inputs, |data| {
           potentially_leaky_operation(data);
       });

   fn potentially_leaky_operation(_data: &[u8; 32]) {}
   ```

2. **Increase input size**
   ```rust
   // Larger inputs → larger effects
   let inputs = byte_vecs(4096);  // 4KB instead of 32 bytes
   ```

3. **Prevent inlining**
   ```rust
   #[inline(never)]
   fn leaky_comparison(a: &[u8], b: &[u8]) -> bool {
       for i in 0..a.len() {
           if a[i] != b[i] {
               return false;  // Early exit = timing leak
           }
       }
       true
   }
   ```

4. **Run in release mode**
   ```bash
   cargo test --release --test timing_tests
   ```

**Key points:**
- Increase samples/input size for more power
- Disable inlining for operations you want to measure
- Test in release mode (optimizations may expose leaks)
- Check that inputs actually differ (not captured bug!)

---

### 7.5 Platform-Specific Issues

**Apple Silicon (42ns timer resolution):**

Problem: Coarse timer creates quantization.

Solution: timing-oracle automatically enables adaptive batching (K=2-10 iterations per sample).

```rust
// No code changes needed - automatic batching
let outcome = TimingOracle::balanced().test(inputs, |data| {
    crypto_operation(data);  // Measured in batches if needed
});

fn crypto_operation(_data: &[u8; 32]) {}
```

**Virtual Machines:**

Problem: Hypervisor adds jitter.

Solution: Use bare metal for production timing tests, or use a more lenient attacker model:

```rust
let outcome = TimingOracle::for_attacker(AttackerModel::RemoteNetwork)  // 50μs threshold
    .test(inputs, |data| {
        crypto_operation(data);
    });

fn crypto_operation(_data: &[u8; 32]) {}
```

**Flaky CI Tests:**

Problem: Tests pass locally but fail in CI.

Solution:
1. Use `skip_if_unreliable!` (pragmatic)
2. Increase `alpha` via `TO_ALPHA=0.05`
3. Run on dedicated CI runners (not shared)

---

**Next:** [§8 Quick Reference](#8-quick-reference) - Fast lookup catalog

---

## 8. Quick Reference

**Fast lookup catalog** of all testing patterns organized by use case.

### 8.1 Basic Patterns Catalog

| **Pattern** | **Use Case** | **Code Snippet** | **Section** |
|------------|-------------|-----------------|-----------|
| **Simplest test** | Verify XOR/array copy constant-time | `InputPair::new(\|\| [0x00; 64], \|\| [0xFF; 64])` | [§2.1](#21-pattern-simplest-constant-time-test) |
| **DudeCT two-class** | Standard fixed vs random | `InputPair::new(\|\| fixed, \|\| rand::random())` | [§2.2](#22-pattern-dudect-two-class-pattern-fixed-vs-random) |
| **Avoiding RNG overhead** | Pre-generate inputs | Use `InputPair::new()`, NOT inside measured closure | [§2.3](#23-pattern-inputpair---avoiding-rng-overhead) |
| **Assertion pyramid** | 4-layer validation | ci_gate + leak_probability + exploitability + quality | [§2.4](#24-pattern-the-assertion-pyramid) |
| **Unreliable handling** | Graceful failure | `skip_if_unreliable!(outcome, "test_name")` | [§2.5](#25-pattern-handling-unreliable-measurements) |

### 8.2 Input Generation Catalog

| **Pattern** | **Use Case** | **Code Snippet** | **Section** |
|------------|-------------|-----------------|-----------|
| **32-byte arrays** | Hash/key testing | `byte_arrays_32()` | [§3.1](#31-pattern-fixed-size-arrays-convenience-helper) |
| **Variable vectors** | Variable-length data | `byte_vecs(1024)` | [§3.2](#32-pattern-variable-size-vectors) |
| **Sequential nonces** | AEAD encryption | `AtomicU64::new(0)`, `fetch_add(1, Relaxed)` | [§3.3](#33-pattern-custom-closures-with-state-cell-pattern) |
| **Pre-gen pool** | Expensive inputs (RSA keys) | Pre-generate Vec, cycle with `Cell<usize>` | [§3.4](#34-pattern-pre-generation-pool-expensive-inputs) |
| **Non-hashable types** | ECC scalars, field elements | `InputPair::new_untracked()` | [§3.6](#36-pattern-non-hashable-types-crypto-primitives) |
| **Intentional deterministic** | Nonce independence testing | `InputPair::new_unchecked()` | [§3.7](#37-pattern-intentional-deterministic-inputs) |

### 8.3 Crypto Patterns Catalog

#### AEAD Ciphers ⚠️ CRITICAL

| **Operation** | **Nonce Strategy** | **Key Pattern** | **Section** |
|--------------|-------------------|----------------|-----------|
| **Encryption** | `AtomicU64` counter | ✓ **Unique nonce per encryption** | [§4.1.2](#412-pattern-aead-encryption-with-atomic-nonce-counter-) |
| **Decryption** | Fixed nonce OK | Pre-generate ciphertext pool | [§4.1.3](#413-pattern-aead-decryption-with-pre-generated-ciphertexts) |
| **Nonce independence** | Two fixed nonces (0x00 vs 0xFF) | Fixed plaintext | [§4.1.4](#414-pattern-testing-nonce-independence) |

**⚠️ NEVER REUSE NONCES** - Always use atomic counter for AEAD encryption tests!

#### Other Crypto Primitives

| **Primitive** | **Key Point** | **Sample Count** | **Section** |
|--------------|--------------|----------------|-----------|
| **AES/Block ciphers** | DudeCT: fixed vs random blocks | 30,000 | [§4.2](#42-block-ciphers) |
| **SHA-3/Hashes** | One-shot or incremental | 30,000 | [§4.3](#43-hash-functions) |
| **X25519/ECC** | ⚠️ DON'T use all-zeros scalar | 50,000 | [§4.4](#44-elliptic-curve-cryptography) |
| **RSA** | Use `.quick()`, fewer samples | 5,000 | [§4.5](#45-rsa) |
| **ML-KEM/PQ** | Slower, use `.balanced()` | 20,000 | [§4.6](#46-post-quantum-cryptography) |

### 8.4 Advanced Patterns Catalog

| **Pattern** | **Use Case** | **Key Code** | **Section** |
|------------|-------------|-------------|-----------|
| **Async (single-thread)** | Precise async measurements | `Builder::new_current_thread()`, `rt.block_on()` | [§5.1](#51-asyncawait-code) |
| **Async (multi-thread)** | Stress testing | `Builder::new_multi_thread().worker_threads(2)` | [§5.1](#51-asyncawait-code) |
| **Structs/tuples** | Complex types | Implement `Clone + Hash`, use `move \|\|` | [§5.2](#52-complex-types) |
| **Stateful operations** | Caches, DB connections | `test_with_state(&mut state, inputs, \|state, input\| {...})` | [§5.3](#53-stateful-operations) |
| **Batch testing** | Multiple ops per sample | Loop inside measured closure | [§5.4](#54-batch-testing) |

### 8.5 CI/CD Integration Catalog

| **Pattern** | **Use Case** | **Code/Config** | **Section** |
|------------|-------------|----------------|-----------|
| **Env var overrides** | CI flexibility | `.from_env()` + `TO_SAMPLES=10000` | [§6.1](#61-cicd-integration) |
| **Shared config** | Reusable oracles | `tests/common/mod.rs` with helper functions | [§6.2](#62-test-organization) |
| **Slow tests** | Thorough validation | `#[ignore = "slow test"]`, run with `--ignored` | [§6.2](#62-test-organization) |
| **PMU/perf timers** | Best precision | `#[cfg]` + `sudo -E cargo test` | [§6.3](#63-platform-specific-configuration) |

### 8.6 Troubleshooting Catalog

| **Problem** | **Symptom** | **Solution** | **Section** |
|------------|-----------|-------------|-----------|
| **Unmeasurable** | `Outcome::Unmeasurable` | `skip_if_unreliable!()` or batch operations | [§7.1](#71-unmeasurable-operations) |
| **High noise** | `MeasurementQuality::TooNoisy` | Increase samples, check CPU governor, use PMU timer | [§7.2](#72-high-noise--poor-measurement-quality) |
| **False positive** | Safe code fails CI gate | Check: different code paths, state leakage, compiler optimizations | [§7.3](#73-false-positives) |
| **False negative** | Leaky code passes | Increase samples, prevent inlining, test in release mode | [§7.4](#74-false-negatives) |
| **Apple Silicon** | 42ns timer quantization | Automatic adaptive batching (no code changes) | [§7.5](#75-platform-specific-issues) |
| **Flaky CI** | Passes locally, fails in CI | Use `skip_if_unreliable!`, increase `TO_ALPHA=0.05` | [§7.5](#75-platform-specific-issues) |

### 8.7 Common Mistakes Checklist

Check your code against these common pitfalls:

#### ❌ Input Generation Mistakes

- [ ] **Captured value bug**: `let val = rand::random(); InputPair::new(|| baseline, || val)` ❌
  - ✓ Fix: `InputPair::new(|| baseline, || rand::random())` ✓

- [ ] **RNG inside closure**: Generating random data INSIDE measured closure ❌
  - ✓ Fix: Use `InputPair` to pre-generate inputs ✓

- [ ] **Different code paths**: Baseline and sample execute different operations ❌
  - ✓ Fix: Same operation, only DATA differs ✓

#### ⚠️ AEAD-Specific Mistakes

- [ ] **Nonce reuse**: Fixed nonce for all encryptions ❌❌❌ **CATASTROPHIC**
  - ✓ Fix: `AtomicU64::new(0)`, `fetch_add(1, Ordering::Relaxed)` ✓

#### ❌ ECC Mistakes

- [ ] **All-zeros scalar**: Using `[0u8; 32]` as fixed scalar ❌
  - ✓ Fix: Use non-pathological fixed scalar (mixed bits) ✓

#### ❌ Async Mistakes

- [ ] **Runtime inside closure**: Creating Tokio runtime inside measured closure ❌
  - ✓ Fix: Create runtime ONCE outside, use `rt.block_on()` inside ✓

#### ❌ General Mistakes

- [ ] **Measuring RNG**: Not pre-generating inputs ❌
- [ ] **State leakage**: Global state persists between samples ❌
- [ ] **Values instead of closures**: Passing values directly to `InputPair::new()` ❌

### 8.8 Quick Decision Tree

**"Which pattern should I use?"**

```
START: What are you testing?
│
├─ AEAD (ChaCha20-Poly1305, AES-GCM)?
│  └─ Use AtomicU64 nonce counter → §4.1 ⚠️
│
├─ Block cipher (AES)?
│  └─ DudeCT pattern, 30k samples → §4.2
│
├─ Hash function?
│  └─ byte_arrays_32(), 30k samples → §4.3
│
├─ ECC (X25519)?
│  └─ Valid scalar (NOT all-zeros!), 50k samples → §4.4
│
├─ RSA?
│  └─ .quick() preset, 5k samples → §4.5
│
├─ PQ crypto?
│  └─ .balanced(), 20k samples → §4.6
│
├─ Async/await code?
│  └─ Create runtime ONCE, use block_on() → §5.1
│
├─ Custom struct/complex type?
│  └─ Implement Clone+Hash, use move || → §5.2
│
├─ Operation with state (cache, DB)?
│  └─ Use test_with_state() → §5.3
│
├─ Very fast operation (<10ns)?
│  └─ Use skip_if_unreliable!() → §7.1
│
└─ Basic test?
   └─ Start with §2 (Basic Patterns)
```

### 8.9 Preset Selection Guide

| **Preset** | **Samples** | **Runtime** | **Use For** |
|-----------|------------|-----------|------------|
| `.new()` | 100,000 | ~5-10s | Production validation, final checks |
| `.balanced()` ⭐ | 20,000 | ~1-2s | **Most tests (recommended)** |
| `.quick()` | 5,000 | ~0.2-0.5s | Development iteration, slow ops (RSA) |
| `.calibration()` | 2,000 | ~0.1-0.2s | Running 100+ trials in test suites |

**Recommendation:** Start with `.balanced()` for 80% of use cases.

---

## End of Testing Patterns Guide

**You've reached the end!** You now have comprehensive patterns for:
- ✓ Basic testing (§2)
- ✓ Input generation (§3) - avoiding the #1 mistake
- ✓ Crypto-specific patterns (§4) - including AEAD nonce reuse ⚠️
- ✓ Advanced scenarios (§5) - async, complex types, state
- ✓ CI/CD integration (§6)
- ✓ Troubleshooting (§7)
- ✓ Quick reference (§8) - fast lookups

**Next steps:**
- For conceptual understanding: Read [`guide.md`](./guide.md)
- For API details: See [`api-reference.md`](./api-reference.md)
- For debugging: Consult [`troubleshooting.md`](./troubleshooting.md)
- For questions: [GitHub Discussions](https://github.com/anthropics/timing-oracle/discussions)

**Happy testing!** 🔒⏱️

