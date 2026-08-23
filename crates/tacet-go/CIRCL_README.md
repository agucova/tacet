# Cloudflare circl Post-Quantum Cryptography Tests

## Summary

**Status:** ✅ Tests Implemented, ❌ Blocked by tacet-go FFI Bug

Cloudflare's circl library tests have been fully implemented (580 lines, 12 tests) but **cannot run due to the critical FFI bug affecting all tacet-go tests**.

## Why Test circl?

[Cloudflare circl](https://github.com/cloudflare/circl) is a production-grade post-quantum cryptography library used in:
- **Cloudflare's infrastructure** (serving millions of requests/second)
- **TLS 1.3 post-quantum experiments**
- **Real-world deployment** at scale

Testing circl provides:
1. **Cross-language validation** of post-quantum crypto (Rust `pqcrypto` vs Go `circl`)
2. **Production implementation testing** (not just academic code)
3. **Paper credibility** for post-quantum claims

## What Was Implemented

**Location:** `crates/tacet-go/circl_test.go` (580 lines)

**Test Coverage (12 tests):**

### Kyber (ML-KEM) Tests
1. `TestCircl_Kyber512_Decapsulation` - CRITICAL private key operation
2. `TestCircl_Kyber512_Encapsulation` - Public key operation (informational)
3. `TestCircl_Kyber768_Decapsulation` - Most deployed parameter set
4. `TestCircl_Kyber1024_Decapsulation` - Highest security parameter set

**Test Pattern:** Baseline uses unmodified valid ciphertext, sample class XORs ciphertext with random input to test constant-time behavior with invalid/malformed ciphertexts.

**Threat Model:** `PostQuantum` (~10 cycles @ 5 GHz, 2.0ns) per KyberSlash research.

### Dilithium (ML-DSA) Tests
5. `TestCircl_Dilithium2_Signing` - Same message (tests timing consistency)
6. `TestCircl_Dilithium2_MessageHamming` - Different messages (informational, rejection sampling expected)
7. `TestCircl_Dilithium3_Signing` - Most deployed parameter set
8. `TestCircl_Dilithium5_Signing` - Highest security parameter set

**Note:** Dilithium uses rejection sampling which causes INTENTIONAL message-dependent timing variation. This is NOT a vulnerability as the message is public and rejection probability is independent of the secret key.

### Classical Crypto Tests (Comparison Baseline)
9. `TestCircl_X25519_ScalarMult` - Compare with golang.org/x/crypto/curve25519
10. `TestCircl_Ed25519_Signing` - Compare with stdlib crypto/ed25519

## Previously Reported FFI Bug (resolved)

An earlier revision of this file reported that all Go tests produced garbage
(`P(leak) = 100%` on identical inputs, negative effect sizes, nonsensical sample
counts) and blamed the FFI result conversion. That diagnosis was wrong. The
struct layout, enum mapping, class labelling, and tick-to-nanosecond conversion
are correct and are now pinned by deterministic regression tests in
`analysis_groundtruth_test.go`.

The real defect was in the measurement loop: the batch size was selected by
timing only the baseline class, which for asymmetric operations folded twenty
calls into every measurement and added enough serial dependence to collapse the
posterior onto its prior. See the "Status and Known Limitations" section of
`README.md` for the full list of what changed.

## Expected Results

Based on the test design:

**Kyber Decapsulation:**
- Should PASS or be INCONCLUSIVE (depending on circl's implementation)
- Any FAIL indicates potential KyberSlash-class timing leak

**Dilithium Signing (Same Message):**
- Should PASS - rejection sampling timing is message-dependent but NOT key-dependent
- Only fails if there's key-dependent timing (serious vulnerability)

**Dilithium Signing (Different Messages):**
- Expected to FAIL or be INCONCLUSIVE - rejection sampling causes intentional timing variation
- This is informational only, NOT a vulnerability

**X25519/Ed25519:**
- Should PASS - both are designed for constant-time operation
- Serves as validation baseline against stdlib implementations

## How to Run

```bash
cd crates/tacet-go

# Run all circl tests
go test -run TestCircl -v

# Run specific post-quantum test
go test -run TestCircl_Kyber768_Decapsulation -v

# Run with extended timeout (post-quantum crypto is slow)
go test -run TestCircl -v -timeout 30m
```

## Dependencies

```go
require github.com/cloudflare/circl v1.6.3
```

circl automatically pulls in:
- `golang.org/x/sys v0.40.0` (system calls)

## Documentation

See `CIRCL_INVESTIGATION_REPORT.md` for:
- Full test design rationale
- Threat model selection (why PostQuantum vs SharedHardware)
- Rejection sampling explanation
- Expected vs actual behavior analysis

## References

- [Cloudflare circl](https://github.com/cloudflare/circl)
- [KyberSlash](https://kyberslash.cr.yp.to/) - Timing attacks on Kyber implementations
- [ML-KEM FIPS 203](https://csrc.nist.gov/pubs/fips/203/final) - NIST standard
- [ML-DSA FIPS 204](https://csrc.nist.gov/pubs/fips/204/final) - NIST standard

## Status Updates

**2026-02-05:** Tests implemented, reported as blocked by a tacet-go FFI bug.

**2026-08-22:** The FFI bug report was a misdiagnosis. The batch-size selection
in the Go measurement loop was fixed, along with the missing `Unmeasurable`
verdict, input pre-generation, and measurement environment setup. The whole Go
test suite, including these circl tests, passes.
