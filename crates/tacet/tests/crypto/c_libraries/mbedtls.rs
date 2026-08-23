//! mbedTLS timing tests
//!
//! Tests cryptographic implementations from mbedTLS (ARM Mbed TLS) via FFI bindings.
//! Uses DudeCT's two-class pattern:
//! - Baseline: All zeros or fixed values
//! - Sample: Random values
//!
//! CRITICAL: FFI harness verification is essential. Always run sanity checks first.
//!
//! **NOTE**: These tests require mbedTLS development files.
//! On NixOS/macOS with Nix: mbedTLS is available in the Nix store.
//! Environment variable needed: `MBEDTLS_DIR=/nix/store/.../mbedtls-X.Y.Z`
//! Find path with: `find /nix/store -maxdepth 1 -name "*mbedtls-*" -type d | head -1`
//!
//! Operations tested:
//! - RSA-2048 PKCS#1 v1.5 decryption (Bleichenbacher-class attacks like MARVIN)
//! - RSA-2048 OAEP decryption
//! - ECDSA P-256 signing
//! - AES-256-GCM encryption (software fallback paths)

// The FFI surface is declared in full; not every binding is exercised by the tests.
#![allow(dead_code)]

use std::ffi::c_void;
use std::ptr;
use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{skip_if_unreliable, AttackerModel, Outcome, TimingOracle};

// ============================================================================
// mbedTLS FFI Bindings
// ============================================================================

// Link against mbedtls crypto library
#[link(name = "mbedcrypto")]
extern "C" {}

// Common types
type MbedtlsRngFunction = unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> i32;

// Use repr(C) with large opaque byte arrays to ensure proper alignment
// Sizes determined from mbedTLS headers
#[repr(C, align(8))]
struct MbedtlsRsaContext {
    _data: [u8; 1024], // Oversized to be safe
}

#[repr(C, align(8))]
struct MbedtlsEcdsaContext {
    _data: [u8; 512], // Oversized to be safe
}

#[repr(C, align(8))]
struct MbedtlsGcmContext {
    _data: [u8; 512], // Oversized to be safe
}

#[repr(C, align(8))]
struct MbedtlsEntropyContext {
    _data: [u8; 512], // Oversized to be safe
}

#[repr(C, align(8))]
struct MbedtlsCtrDrbgContext {
    _data: [u8; 512], // Oversized to be safe
}

// ECP group IDs from mbedtls/ecp.h
// typedef enum mbedtls_ecp_group_id
const MBEDTLS_ECP_DP_SECP256R1: i32 = 3;

// MD type IDs from mbedtls/md.h
const MBEDTLS_MD_NONE: i32 = 0;
const MBEDTLS_MD_SHA256: i32 = 9;

// RSA padding modes from mbedtls/rsa.h
const MBEDTLS_RSA_PKCS_V15: i32 = 0;
const MBEDTLS_RSA_PKCS_V21: i32 = 1;

// mbedTLS cipher IDs from mbedtls/cipher.h
const MBEDTLS_CIPHER_ID_AES: u32 = 2;

// Entropy source callback type
type MbedtlsEntropyFSource = unsafe extern "C" fn(*mut c_void, *mut u8, usize, *mut usize) -> i32;

extern "C" {
    // Entropy and RNG functions
    fn mbedtls_entropy_init(ctx: *mut MbedtlsEntropyContext);
    fn mbedtls_entropy_free(ctx: *mut MbedtlsEntropyContext);
    fn mbedtls_entropy_func(ctx: *mut c_void, output: *mut u8, len: usize) -> i32;
    fn mbedtls_entropy_add_source(
        ctx: *mut MbedtlsEntropyContext,
        f_source: MbedtlsEntropyFSource,
        p_source: *mut c_void,
        threshold: usize,
        strong: i32,
    ) -> i32;

    fn mbedtls_ctr_drbg_init(ctx: *mut MbedtlsCtrDrbgContext);
    fn mbedtls_ctr_drbg_free(ctx: *mut MbedtlsCtrDrbgContext);
    fn mbedtls_ctr_drbg_seed(
        ctx: *mut MbedtlsCtrDrbgContext,
        f_entropy: unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> i32,
        p_entropy: *mut c_void,
        custom: *const u8,
        len: usize,
    ) -> i32;
    fn mbedtls_ctr_drbg_random(ctx: *mut c_void, output: *mut u8, len: usize) -> i32;

    // RSA functions
    fn mbedtls_rsa_init(ctx: *mut MbedtlsRsaContext);
    fn mbedtls_rsa_free(ctx: *mut MbedtlsRsaContext);
    fn mbedtls_rsa_set_padding(ctx: *mut MbedtlsRsaContext, padding: i32, hash_id: i32) -> i32;
    fn mbedtls_rsa_gen_key(
        ctx: *mut MbedtlsRsaContext,
        f_rng: MbedtlsRngFunction,
        p_rng: *mut c_void,
        nbits: u32,
        exponent: i32,
    ) -> i32;
    fn mbedtls_rsa_pkcs1_encrypt(
        ctx: *mut MbedtlsRsaContext,
        f_rng: MbedtlsRngFunction,
        p_rng: *mut c_void,
        ilen: usize,
        input: *const u8,
        output: *mut u8,
    ) -> i32;
    fn mbedtls_rsa_pkcs1_decrypt(
        ctx: *mut MbedtlsRsaContext,
        f_rng: MbedtlsRngFunction,
        p_rng: *mut c_void,
        olen: *mut usize,
        input: *const u8,
        output: *mut u8,
        output_max_len: usize,
    ) -> i32;
    fn mbedtls_rsa_rsaes_oaep_encrypt(
        ctx: *mut MbedtlsRsaContext,
        f_rng: MbedtlsRngFunction,
        p_rng: *mut c_void,
        label: *const u8,
        label_len: usize,
        ilen: usize,
        input: *const u8,
        output: *mut u8,
    ) -> i32;
    fn mbedtls_rsa_rsaes_oaep_decrypt(
        ctx: *mut MbedtlsRsaContext,
        f_rng: MbedtlsRngFunction,
        p_rng: *mut c_void,
        label: *const u8,
        label_len: usize,
        olen: *mut usize,
        input: *const u8,
        output: *mut u8,
        output_max_len: usize,
    ) -> i32;

    // ECDSA functions
    fn mbedtls_ecdsa_init(ctx: *mut MbedtlsEcdsaContext);
    fn mbedtls_ecdsa_free(ctx: *mut MbedtlsEcdsaContext);
    fn mbedtls_ecdsa_genkey(
        ctx: *mut MbedtlsEcdsaContext,
        gid: i32,
        f_rng: MbedtlsRngFunction,
        p_rng: *mut c_void,
    ) -> i32;
    fn mbedtls_ecdsa_write_signature(
        ctx: *mut MbedtlsEcdsaContext,
        md_alg: i32,
        hash: *const u8,
        hlen: usize,
        sig: *mut u8,
        sig_size: usize,
        slen: *mut usize,
        f_rng: MbedtlsRngFunction,
        p_rng: *mut c_void,
    ) -> i32;

    // GCM functions
    fn mbedtls_gcm_init(ctx: *mut MbedtlsGcmContext);
    fn mbedtls_gcm_free(ctx: *mut MbedtlsGcmContext);
    fn mbedtls_gcm_setkey(
        ctx: *mut MbedtlsGcmContext,
        cipher: u32,
        key: *const u8,
        keybits: u32,
    ) -> i32;
    fn mbedtls_gcm_crypt_and_tag(
        ctx: *mut MbedtlsGcmContext,
        mode: i32,
        length: usize,
        iv: *const u8,
        iv_len: usize,
        add: *const u8,
        add_len: usize,
        input: *const u8,
        output: *mut u8,
        tag_len: usize,
        tag: *mut u8,
    ) -> i32;
}

// Helper functions
fn rand_bytes_32() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

fn rand_bytes_64() -> [u8; 64] {
    let mut arr = [0u8; 64];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

// Simple entropy callback using Rust's RNG
// This bypasses the entropy pool and directly provides entropy to CTR_DRBG
unsafe extern "C" fn rust_entropy_callback(_ctx: *mut c_void, output: *mut u8, len: usize) -> i32 {
    for i in 0..len {
        *output.add(i) = rand::random();
    }
    0 // Success
}

// Helper to initialize CTR_DRBG context with simple Rust RNG entropy
// Returns ctr_drbg_ctx as heap-allocated vector
// We skip the entropy pool and use Rust's RNG directly
unsafe fn init_rng() -> Vec<u8> {
    // Allocate CTR_DRBG context only
    let mut ctr_drbg_mem = vec![0u8; 512];
    let ctr_drbg_ctx = ctr_drbg_mem.as_mut_ptr() as *mut MbedtlsCtrDrbgContext;

    // Initialize CTR_DRBG
    mbedtls_ctr_drbg_init(ctr_drbg_ctx);

    // Seed CTR_DRBG with Rust RNG as entropy source
    let pers = b"tacet_timing_test";
    let ret = mbedtls_ctr_drbg_seed(
        ctr_drbg_ctx,
        rust_entropy_callback,
        ptr::null_mut(), // No entropy context needed
        pers.as_ptr(),
        pers.len(),
    );

    assert_eq!(ret, 0, "CTR_DRBG seed failed with error: {}", ret);

    ctr_drbg_mem
}

// Free RNG context
unsafe fn free_rng(ctr_drbg_mem: &mut Vec<u8>) {
    let ctr_drbg_ctx = ctr_drbg_mem.as_mut_ptr() as *mut MbedtlsCtrDrbgContext;
    mbedtls_ctr_drbg_free(ctr_drbg_ctx);
}

// ============================================================================
// RSA PKCS#1 v1.5 Tests (Bleichenbacher-class, MARVIN)
// ============================================================================

/// mbedTLS RSA PKCS#1 v1.5 decryption should be constant-time
///
/// Tests for timing side channels in PKCS#1 v1.5 padding validation.
/// Historical vulnerabilities: Bleichenbacher (1998), ROBOT (2017), MARVIN (2023)
///
/// Uses pool-based pattern to avoid caching artifacts.
#[test]
fn mbedtls_rsa_2048_pkcs1v15_decrypt_constant_time() {
    const POOL_SIZE: usize = 100;

    unsafe {
        // Initialize RNG (CTR_DRBG with Rust RNG entropy)
        let mut ctr_drbg_mem = init_rng();
        let ctr_drbg_ctx = ctr_drbg_mem.as_mut_ptr() as *mut c_void;

        // Allocate RSA context (opaque struct, we need heap allocation)
        let ctx_size = 1024; // Approximate size
        let mut ctx_mem = vec![0u8; ctx_size];
        let ctx = ctx_mem.as_mut_ptr() as *mut MbedtlsRsaContext;

        mbedtls_rsa_init(ctx);

        // Generate RSA-2048 key pair
        let ret = mbedtls_rsa_gen_key(ctx, mbedtls_ctr_drbg_random, ctr_drbg_ctx, 2048, 65537);
        assert_eq!(ret, 0, "RSA key generation failed");

        // Pre-generate two pools of ciphertexts
        let pool_baseline: Vec<Vec<u8>> = (0..POOL_SIZE)
            .map(|_| {
                let msg = rand_bytes_32();
                let mut encrypted = vec![0u8; 256]; // RSA-2048 = 256 bytes
                let ret = mbedtls_rsa_pkcs1_encrypt(
                    ctx,
                    mbedtls_ctr_drbg_random,
                    ctr_drbg_ctx,
                    msg.len(),
                    msg.as_ptr(),
                    encrypted.as_mut_ptr(),
                );
                assert_eq!(ret, 0, "RSA encryption failed");
                encrypted
            })
            .collect();

        let pool_sample: Vec<Vec<u8>> = (0..POOL_SIZE)
            .map(|_| {
                let msg = rand_bytes_32();
                let mut encrypted = vec![0u8; 256];
                let ret = mbedtls_rsa_pkcs1_encrypt(
                    ctx,
                    mbedtls_ctr_drbg_random,
                    ctr_drbg_ctx,
                    msg.len(),
                    msg.as_ptr(),
                    encrypted.as_mut_ptr(),
                );
                assert_eq!(ret, 0, "RSA encryption failed");
                encrypted
            })
            .collect();

        let idx_baseline = std::cell::Cell::new(0usize);
        let idx_sample = std::cell::Cell::new(0usize);

        let inputs = InputPair::new(
            move || {
                let i = idx_baseline.get();
                idx_baseline.set((i + 1) % POOL_SIZE);
                pool_baseline[i].clone()
            },
            move || {
                let i = idx_sample.get();
                idx_sample.set((i + 1) % POOL_SIZE);
                pool_sample[i].clone()
            },
        );

        let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .pass_threshold(0.15)
            .fail_threshold(0.99)
            .time_budget(Duration::from_secs(60))
            .warmup(100)
            .calibration_samples(2000)
            .test(inputs, |ciphertext| {
                let mut decrypted = vec![0u8; 256];
                let mut olen = 0usize;
                // PKCS#1 v1.5 padding - historical source of timing leaks
                let result = mbedtls_rsa_pkcs1_decrypt(
                    ctx,
                    mbedtls_ctr_drbg_random,
                    ctr_drbg_ctx,
                    &mut olen,
                    ciphertext.as_ptr(),
                    decrypted.as_mut_ptr(),
                    256,
                );
                std::hint::black_box(result);
            });

        eprintln!("\n[mbedtls_rsa_2048_pkcs1v15_decrypt_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        mbedtls_rsa_free(ctx);
        free_rng(&mut ctr_drbg_mem);

        let outcome =
            skip_if_unreliable!(outcome, "mbedtls_rsa_2048_pkcs1v15_decrypt_constant_time");

        match &outcome {
            Outcome::Pass {
                leak_probability,
                quality,
                ..
            } => {
                eprintln!(
                    "Test passed: P(leak)={:.1}%, quality={:?}",
                    leak_probability * 100.0,
                    quality
                );
            }
            Outcome::Fail {
                leak_probability,
                exploitability,
                effect,
                ..
            } => {
                eprintln!("⚠️  TIMING LEAK DETECTED in mbedTLS RSA PKCS#1 v1.5 decryption");
                eprintln!(
                    "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                    leak_probability * 100.0,
                    exploitability,
                    effect.max_effect_ns
                );
                eprintln!(
                    "   This may be a variant of MARVIN (CVE-2023-50782) or similar timing leak"
                );
                panic!("RSA PKCS#1 v1.5 decryption timing leak - potential security vulnerability");
            }
            Outcome::Inconclusive { reason, .. } => {
                eprintln!("Inconclusive: {:?}", reason);
            }
            Outcome::Unmeasurable { recommendation, .. } => {
                eprintln!("Unmeasurable: {}", recommendation);
            }
            Outcome::Research(_) => {}
        }
    }
}

// ============================================================================
// RSA OAEP Tests
// ============================================================================

/// mbedTLS RSA OAEP decryption should be constant-time
///
/// OAEP is more robust than PKCS#1 v1.5 but still requires constant-time implementation.
#[test]
fn mbedtls_rsa_2048_oaep_decrypt_constant_time() {
    const POOL_SIZE: usize = 100;

    unsafe {
        // Initialize RNG (CTR_DRBG with Rust RNG entropy)
        let mut ctr_drbg_mem = init_rng();
        let ctr_drbg_ctx = ctr_drbg_mem.as_mut_ptr() as *mut c_void;

        let ctx_size = 1024;
        let mut ctx_mem = vec![0u8; ctx_size];
        let ctx = ctx_mem.as_mut_ptr() as *mut MbedtlsRsaContext;

        mbedtls_rsa_init(ctx);

        let ret = mbedtls_rsa_gen_key(ctx, mbedtls_ctr_drbg_random, ctr_drbg_ctx, 2048, 65537);
        assert_eq!(ret, 0, "RSA key generation failed");

        // Set padding mode to PKCS#1 v2.1 (OAEP) with SHA-256
        let ret = mbedtls_rsa_set_padding(ctx, MBEDTLS_RSA_PKCS_V21, MBEDTLS_MD_SHA256);
        assert_eq!(ret, 0, "RSA set padding failed");

        // Pre-generate two pools of ciphertexts with OAEP padding
        let pool_baseline: Vec<Vec<u8>> = (0..POOL_SIZE)
            .map(|_| {
                let msg = rand_bytes_32();
                let mut encrypted = vec![0u8; 256];
                let ret = mbedtls_rsa_rsaes_oaep_encrypt(
                    ctx,
                    mbedtls_ctr_drbg_random,
                    ctr_drbg_ctx,
                    ptr::null(),
                    0,
                    msg.len(),
                    msg.as_ptr(),
                    encrypted.as_mut_ptr(),
                );
                assert_eq!(ret, 0, "RSA OAEP encryption failed");
                encrypted
            })
            .collect();

        let pool_sample: Vec<Vec<u8>> = (0..POOL_SIZE)
            .map(|_| {
                let msg = rand_bytes_32();
                let mut encrypted = vec![0u8; 256];
                let ret = mbedtls_rsa_rsaes_oaep_encrypt(
                    ctx,
                    mbedtls_ctr_drbg_random,
                    ctr_drbg_ctx,
                    ptr::null(),
                    0,
                    msg.len(),
                    msg.as_ptr(),
                    encrypted.as_mut_ptr(),
                );
                assert_eq!(ret, 0, "RSA OAEP encryption failed");
                encrypted
            })
            .collect();

        let idx_baseline = std::cell::Cell::new(0usize);
        let idx_sample = std::cell::Cell::new(0usize);

        let inputs = InputPair::new(
            move || {
                let i = idx_baseline.get();
                idx_baseline.set((i + 1) % POOL_SIZE);
                pool_baseline[i].clone()
            },
            move || {
                let i = idx_sample.get();
                idx_sample.set((i + 1) % POOL_SIZE);
                pool_sample[i].clone()
            },
        );

        let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .pass_threshold(0.15)
            .fail_threshold(0.99)
            .time_budget(Duration::from_secs(60))
            .warmup(100)
            .calibration_samples(2000)
            .test(inputs, |ciphertext| {
                let mut decrypted = vec![0u8; 256];
                let mut olen = 0usize;
                let result = mbedtls_rsa_rsaes_oaep_decrypt(
                    ctx,
                    mbedtls_ctr_drbg_random,
                    ctr_drbg_ctx,
                    ptr::null(),
                    0,
                    &mut olen,
                    ciphertext.as_ptr(),
                    decrypted.as_mut_ptr(),
                    256,
                );
                std::hint::black_box(result);
            });

        eprintln!("\n[mbedtls_rsa_2048_oaep_decrypt_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        mbedtls_rsa_free(ctx);
        free_rng(&mut ctr_drbg_mem);

        let outcome = skip_if_unreliable!(outcome, "mbedtls_rsa_2048_oaep_decrypt_constant_time");

        match &outcome {
            Outcome::Pass {
                leak_probability,
                quality,
                ..
            } => {
                eprintln!(
                    "Test passed: P(leak)={:.1}%, quality={:?}",
                    leak_probability * 100.0,
                    quality
                );
            }
            Outcome::Fail {
                leak_probability,
                exploitability,
                effect,
                ..
            } => {
                eprintln!("⚠️  TIMING LEAK DETECTED in mbedTLS RSA OAEP decryption");
                eprintln!(
                    "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                    leak_probability * 100.0,
                    exploitability,
                    effect.max_effect_ns
                );
                panic!("RSA OAEP decryption timing leak");
            }
            Outcome::Inconclusive { reason, .. } => {
                eprintln!("Inconclusive: {:?}", reason);
            }
            Outcome::Unmeasurable { recommendation, .. } => {
                eprintln!("Unmeasurable: {}", recommendation);
            }
            Outcome::Research(_) => {}
        }
    }
}

// ============================================================================
// ECDSA P-256 Tests
// ============================================================================

/// mbedTLS ECDSA P-256 signing should be constant-time
///
/// ECDSA is particularly sensitive to timing attacks - nonce generation and
/// modular inversion can leak the private key.
#[test]
fn mbedtls_ecdsa_p256_sign_constant_time() {
    unsafe {
        // Initialize RNG (CTR_DRBG with Rust RNG entropy)
        let mut ctr_drbg_mem = init_rng();
        let ctr_drbg_ctx = ctr_drbg_mem.as_mut_ptr() as *mut c_void;

        let ctx_size = 512;
        let mut ctx_mem = vec![0u8; ctx_size];
        let ctx = ctx_mem.as_mut_ptr() as *mut MbedtlsEcdsaContext;

        mbedtls_ecdsa_init(ctx);

        // Generate P-256 key pair
        let ret = mbedtls_ecdsa_genkey(
            ctx,
            MBEDTLS_ECP_DP_SECP256R1,
            mbedtls_ctr_drbg_random,
            ctr_drbg_ctx,
        );
        assert_eq!(ret, 0, "ECDSA key generation failed with error: {}", ret);

        // Use zeros vs random pattern for message
        let inputs = InputPair::new(|| [0u8; 32], rand_bytes_32);

        let outcome = TimingOracle::for_attacker(AttackerModel::SharedHardware)
            .pass_threshold(0.15)
            .fail_threshold(0.99)
            .time_budget(Duration::from_secs(60))
            .warmup(100)
            .test(inputs, |message| {
                let mut sig = vec![0u8; 139]; // ECDSA P-256 signature max size
                let mut sig_len = 0usize;
                let ret = mbedtls_ecdsa_write_signature(
                    ctx,
                    MBEDTLS_MD_SHA256,
                    message.as_ptr(),
                    message.len(),
                    sig.as_mut_ptr(),
                    sig.len(),
                    &mut sig_len,
                    mbedtls_ctr_drbg_random,
                    ctr_drbg_ctx,
                );
                std::hint::black_box(ret);
                std::hint::black_box(sig[0]);
            });

        eprintln!("\n[mbedtls_ecdsa_p256_sign_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        mbedtls_ecdsa_free(ctx);
        free_rng(&mut ctr_drbg_mem);

        let outcome = skip_if_unreliable!(outcome, "mbedtls_ecdsa_p256_sign_constant_time");

        match &outcome {
            Outcome::Pass {
                leak_probability,
                quality,
                ..
            } => {
                eprintln!(
                    "Test passed: P(leak)={:.1}%, quality={:?}",
                    leak_probability * 100.0,
                    quality
                );
            }
            Outcome::Fail {
                leak_probability,
                exploitability,
                effect,
                ..
            } => {
                eprintln!("⚠️  TIMING LEAK DETECTED in mbedTLS ECDSA P-256 signing");
                eprintln!(
                    "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                    leak_probability * 100.0,
                    exploitability,
                    effect.max_effect_ns
                );
                panic!("ECDSA P-256 signing timing leak - can reveal private key");
            }
            Outcome::Inconclusive { reason, .. } => {
                eprintln!("Inconclusive: {:?}", reason);
            }
            Outcome::Unmeasurable { recommendation, .. } => {
                eprintln!("Unmeasurable: {}", recommendation);
            }
            Outcome::Research(_) => {}
        }
    }
}

// ============================================================================
// AES-256-GCM Tests
// ============================================================================

/// mbedTLS AES-256-GCM encryption should be constant-time
///
/// Tests software fallback paths (no AES-NI) for data-dependent timing.
#[test]
fn mbedtls_aes_256_gcm_encrypt_constant_time() {
    unsafe {
        let ctx_size = 512;
        let mut ctx_mem = vec![0u8; ctx_size];
        let ctx = ctx_mem.as_mut_ptr() as *mut MbedtlsGcmContext;

        mbedtls_gcm_init(ctx);

        let key = rand_bytes_32();
        let ret = mbedtls_gcm_setkey(ctx, MBEDTLS_CIPHER_ID_AES, key.as_ptr(), 256);
        assert_eq!(ret, 0, "GCM setkey failed");

        let nonce_counter = std::sync::atomic::AtomicU64::new(0);
        let inputs = InputPair::new(|| [0u8; 64], rand_bytes_64);

        let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .pass_threshold(0.15)
            .fail_threshold(0.99)
            .time_budget(Duration::from_secs(30))
            .test(inputs, |plaintext| {
                // Generate unique nonce for each encryption
                let n = nonce_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let mut nonce = [0u8; 12];
                nonce[..8].copy_from_slice(&n.to_le_bytes());

                let mut ciphertext = vec![0u8; 64];
                let mut tag = vec![0u8; 16];
                let ret = mbedtls_gcm_crypt_and_tag(
                    ctx,
                    1, // MBEDTLS_GCM_ENCRYPT
                    plaintext.len(),
                    nonce.as_ptr(),
                    nonce.len(),
                    ptr::null(),
                    0,
                    plaintext.as_ptr(),
                    ciphertext.as_mut_ptr(),
                    tag.len(),
                    tag.as_mut_ptr(),
                );
                std::hint::black_box(ret);
                std::hint::black_box(ciphertext[0]);
                std::hint::black_box(tag[0]);
            });

        eprintln!("\n[mbedtls_aes_256_gcm_encrypt_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        mbedtls_gcm_free(ctx);

        let outcome = skip_if_unreliable!(outcome, "mbedtls_aes_256_gcm_encrypt_constant_time");

        match &outcome {
            Outcome::Pass {
                leak_probability,
                quality,
                ..
            } => {
                eprintln!(
                    "Test passed: P(leak)={:.1}%, quality={:?}",
                    leak_probability * 100.0,
                    quality
                );
            }
            Outcome::Fail {
                leak_probability,
                exploitability,
                effect,
                ..
            } => {
                eprintln!("⚠️  TIMING LEAK DETECTED in mbedTLS AES-256-GCM encryption");
                eprintln!(
                    "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                    leak_probability * 100.0,
                    exploitability,
                    effect.max_effect_ns
                );
                panic!("AES-256-GCM encryption timing leak");
            }
            Outcome::Inconclusive { reason, .. } => {
                eprintln!("Inconclusive: {:?}", reason);
            }
            Outcome::Unmeasurable { recommendation, .. } => {
                eprintln!("Unmeasurable: {}", recommendation);
            }
            Outcome::Research(_) => {}
        }
    }
}

// ============================================================================
// Harness Verification Tests
// ============================================================================

/// Sanity check: identical inputs should pass
///
/// FFI overhead can introduce noise - verify harness is working correctly
#[test]
fn mbedtls_harness_sanity_check() {
    unsafe {
        // Initialize RNG (CTR_DRBG with Rust RNG entropy)
        let mut ctr_drbg_mem = init_rng();
        let ctr_drbg_ctx = ctr_drbg_mem.as_mut_ptr() as *mut c_void;

        let ctx_size = 1024;
        let mut ctx_mem = vec![0u8; ctx_size];
        let ctx = ctx_mem.as_mut_ptr() as *mut MbedtlsRsaContext;

        mbedtls_rsa_init(ctx);

        let ret = mbedtls_rsa_gen_key(ctx, mbedtls_ctr_drbg_random, ctr_drbg_ctx, 2048, 65537);
        assert_eq!(ret, 0, "RSA key generation failed");

        // Pre-generate a single ciphertext
        let msg = rand_bytes_32();
        let mut encrypted = vec![0u8; 256];
        let ret = mbedtls_rsa_pkcs1_encrypt(
            ctx,
            mbedtls_ctr_drbg_random,
            ctr_drbg_ctx,
            msg.len(),
            msg.as_ptr(),
            encrypted.as_mut_ptr(),
        );
        assert_eq!(ret, 0, "RSA encryption failed");

        // Both classes return the same ciphertext
        let inputs = InputPair::new(|| encrypted.clone(), || encrypted.clone());

        let outcome = TimingOracle::for_attacker(AttackerModel::Research)
            .time_budget(Duration::from_secs(10))
            .test(inputs, |ciphertext| {
                let mut decrypted = vec![0u8; 256];
                let mut olen = 0usize;
                let result = mbedtls_rsa_pkcs1_decrypt(
                    ctx,
                    mbedtls_ctr_drbg_random,
                    ctr_drbg_ctx,
                    &mut olen,
                    ciphertext.as_ptr(),
                    decrypted.as_mut_ptr(),
                    256,
                );
                std::hint::black_box(result);
            });

        eprintln!("\n[mbedtls_harness_sanity_check]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        mbedtls_rsa_free(ctx);
        free_rng(&mut ctr_drbg_mem);

        match &outcome {
            Outcome::Pass { .. } => {
                eprintln!("Sanity check passed - harness is working correctly");
            }
            Outcome::Fail { .. } => {
                panic!("Sanity check failed - identical inputs should not show timing difference");
            }
            _ => {}
        }
    }
}
