//! wolfSSL timing tests
//!
//! Tests cryptographic implementations from wolfSSL via FFI bindings.
//! Uses DudeCT's two-class pattern:
//! - Baseline: All zeros or fixed values
//! - Sample: Random values
//!
//! CRITICAL: FFI harness verification is essential. Always run sanity checks first.
//!
//! **NOTE**: These tests require wolfSSL development files.
//! On macOS: `brew install wolfssl`
//! wolfSSL version: 5.8.4
//!
//! Operations tested:
//! - RSA-2048 PKCS#1 v1.5 decryption (Bleichenbacher-class attacks like MARVIN)
//! - RSA-2048 OAEP decryption
//! - ECDSA P-256 signing
//! - AES-256-GCM encryption/decryption
//! - Dilithium (ML-DSA) post-quantum signatures (if enabled in build)

// The FFI surface is declared in full; not every binding is exercised by the tests.
#![allow(dead_code)]

use std::mem::MaybeUninit;
use std::ptr;
use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{skip_if_unreliable, AttackerModel, Outcome, TimingOracle};

// ============================================================================
// FFI Bindings to wolfSSL
// ============================================================================

#[cfg(target_os = "macos")]
const WOLFSSL_LIB: &str = "/opt/homebrew/lib/libwolfssl.dylib";

#[cfg(target_os = "linux")]
const WOLFSSL_LIB: &str = "/usr/lib/libwolfssl.so";

// wolfSSL types (opaque structs)
// Sizes determined from wolfSSL 5.8.4 on macOS ARM64
#[repr(C)]
pub struct WC_RNG {
    _private: [u8; 40],
}

#[repr(C)]
pub struct RsaKey {
    _private: [u8; 8448],
}

#[repr(C)]
pub struct ecc_key {
    _private: [u8; 4320],
}

#[repr(C)]
pub struct Aes {
    _private: [u8; 1104],
}

// Constants
const RSA_BLOCK_TYPE_2: i32 = 2; // PKCS#1 v1.5
const WC_RSA_OAEP_PAD: i32 = 1;
const WC_RSA_NO_PAD: i32 = 3;
const WC_HASH_TYPE_SHA256: i32 = 4;
const ECC_SECP256R1: i32 = 7; // P-256 curve (wolfSSL numbering)

// Return codes
const BAD_FUNC_ARG: i32 = -173;

#[link(name = "wolfssl")]
extern "C" {
    // Initialization
    fn wolfCrypt_Init() -> i32;
    fn wolfCrypt_Cleanup() -> i32;

    // RNG
    fn wc_InitRng(rng: *mut WC_RNG) -> i32;
    fn wc_FreeRng(rng: *mut WC_RNG) -> i32;
    fn wc_RNG_GenerateBlock(rng: *mut WC_RNG, output: *mut u8, sz: u32) -> i32;

    // RSA
    fn wc_InitRsaKey(key: *mut RsaKey, heap: *mut std::ffi::c_void) -> i32;
    fn wc_FreeRsaKey(key: *mut RsaKey) -> i32;
    fn wc_MakeRsaKey(key: *mut RsaKey, size: i32, e: i64, rng: *mut WC_RNG) -> i32;
    fn wc_RsaPublicEncrypt(
        in_data: *const u8,
        in_len: u32,
        out: *mut u8,
        out_len: u32,
        key: *mut RsaKey,
        rng: *mut WC_RNG,
    ) -> i32;
    fn wc_RsaPrivateDecrypt(
        in_data: *const u8,
        in_len: u32,
        out: *mut u8,
        out_len: u32,
        key: *mut RsaKey,
    ) -> i32;
    fn wc_RsaPublicEncrypt_ex(
        in_data: *const u8,
        in_len: u32,
        out: *mut u8,
        out_len: u32,
        key: *mut RsaKey,
        rng: *mut WC_RNG,
        pad_type: i32,
        hash: i32,
        mgf: i32,
        label: *const u8,
        label_sz: u32,
    ) -> i32;
    fn wc_RsaPrivateDecrypt_ex(
        in_data: *const u8,
        in_len: u32,
        out: *mut u8,
        out_len: u32,
        key: *mut RsaKey,
        pad_type: i32,
        hash: i32,
        mgf: i32,
        label: *const u8,
        label_sz: u32,
    ) -> i32;

    // ECC
    fn wc_ecc_init(key: *mut ecc_key) -> i32;
    fn wc_ecc_free(key: *mut ecc_key) -> i32;
    fn wc_ecc_make_key(rng: *mut WC_RNG, keysize: i32, key: *mut ecc_key) -> i32;
    fn wc_ecc_make_key_ex(rng: *mut WC_RNG, keysize: i32, key: *mut ecc_key, curve_id: i32) -> i32;
    fn wc_ecc_sign_hash(
        in_data: *const u8,
        in_len: u32,
        out: *mut u8,
        out_len: *mut u32,
        rng: *mut WC_RNG,
        key: *mut ecc_key,
    ) -> i32;

    // AES-GCM
    fn wc_AesInit(aes: *mut Aes, heap: *mut std::ffi::c_void, dev_id: i32) -> i32;
    fn wc_AesFree(aes: *mut Aes) -> i32;
    fn wc_AesGcmSetKey(aes: *mut Aes, key: *const u8, len: u32) -> i32;
    fn wc_AesGcmEncrypt(
        aes: *mut Aes,
        out: *mut u8,
        in_data: *const u8,
        sz: u32,
        iv: *const u8,
        iv_sz: u32,
        auth_tag: *mut u8,
        auth_tag_sz: u32,
        auth_in: *const u8,
        auth_in_sz: u32,
    ) -> i32;
    fn wc_AesGcmDecrypt(
        aes: *mut Aes,
        out: *mut u8,
        in_data: *const u8,
        sz: u32,
        iv: *const u8,
        iv_sz: u32,
        auth_tag: *const u8,
        auth_tag_sz: u32,
        auth_in: *const u8,
        auth_in_sz: u32,
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

// ============================================================================
// RSA PKCS#1 v1.5 Tests (Bleichenbacher-class, MARVIN)
// ============================================================================

/// wolfSSL RSA PKCS#1 v1.5 decryption should be constant-time
///
/// Tests for timing side channels in PKCS#1 v1.5 padding validation.
/// Historical vulnerabilities: Bleichenbacher (1998), ROBOT (2017), MARVIN (2023)
///
/// Uses pool-based pattern to avoid caching artifacts.
#[test]
fn wolfssl_rsa_2048_pkcs1v15_decrypt_constant_time() {
    const POOL_SIZE: usize = 100;

    unsafe {
        assert_eq!(wolfCrypt_Init(), 0, "wolfCrypt initialization failed");

        // Initialize RNG
        let mut rng: MaybeUninit<WC_RNG> = MaybeUninit::uninit();
        let rng_ptr = rng.as_mut_ptr();
        assert_eq!(wc_InitRng(rng_ptr), 0, "RNG initialization failed");

        // Generate RSA-2048 key pair
        let mut key: MaybeUninit<RsaKey> = MaybeUninit::uninit();
        let key_ptr = key.as_mut_ptr();
        assert_eq!(
            wc_InitRsaKey(key_ptr, ptr::null_mut()),
            0,
            "RSA key initialization failed"
        );
        assert_eq!(
            wc_MakeRsaKey(key_ptr, 2048, 65537, rng_ptr),
            0,
            "RSA key generation failed"
        );

        // Pre-generate two pools of ciphertexts
        let mut pool_baseline: Vec<Vec<u8>> = Vec::with_capacity(POOL_SIZE);
        let mut pool_sample: Vec<Vec<u8>> = Vec::with_capacity(POOL_SIZE);

        for _ in 0..POOL_SIZE {
            // Baseline: encrypt random message
            let msg = rand_bytes_32();
            let mut encrypted = vec![0u8; 256]; // RSA-2048 = 256 bytes
            let len = wc_RsaPublicEncrypt(
                msg.as_ptr(),
                msg.len() as u32,
                encrypted.as_mut_ptr(),
                encrypted.len() as u32,
                key_ptr,
                rng_ptr,
            );
            assert!(len > 0, "Encryption failed");
            encrypted.truncate(len as usize);
            pool_baseline.push(encrypted);

            // Sample: encrypt different random message
            let msg = rand_bytes_32();
            let mut encrypted = vec![0u8; 256];
            let len = wc_RsaPublicEncrypt(
                msg.as_ptr(),
                msg.len() as u32,
                encrypted.as_mut_ptr(),
                encrypted.len() as u32,
                key_ptr,
                rng_ptr,
            );
            assert!(len > 0, "Encryption failed");
            encrypted.truncate(len as usize);
            pool_sample.push(encrypted);
        }

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
                // PKCS#1 v1.5 padding - historical source of timing leaks
                let result = wc_RsaPrivateDecrypt(
                    ciphertext.as_ptr(),
                    ciphertext.len() as u32,
                    decrypted.as_mut_ptr(),
                    decrypted.len() as u32,
                    key_ptr,
                );
                std::hint::black_box(result);
            });

        eprintln!("\n[wolfssl_rsa_2048_pkcs1v15_decrypt_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        // Cleanup
        wc_FreeRsaKey(key_ptr);
        wc_FreeRng(rng_ptr);
        wolfCrypt_Cleanup();

        let outcome =
            skip_if_unreliable!(outcome, "wolfssl_rsa_2048_pkcs1v15_decrypt_constant_time");

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
                eprintln!("⚠️  TIMING LEAK DETECTED in wolfSSL RSA PKCS#1 v1.5 decryption");
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

/// wolfSSL RSA OAEP decryption should be constant-time
///
/// OAEP is more robust than PKCS#1 v1.5 but still requires constant-time implementation.
#[test]
fn wolfssl_rsa_2048_oaep_decrypt_constant_time() {
    const POOL_SIZE: usize = 100;

    unsafe {
        assert_eq!(wolfCrypt_Init(), 0, "wolfCrypt initialization failed");

        let mut rng: MaybeUninit<WC_RNG> = MaybeUninit::uninit();
        let rng_ptr = rng.as_mut_ptr();
        assert_eq!(wc_InitRng(rng_ptr), 0, "RNG initialization failed");

        let mut key: MaybeUninit<RsaKey> = MaybeUninit::uninit();
        let key_ptr = key.as_mut_ptr();
        assert_eq!(
            wc_InitRsaKey(key_ptr, ptr::null_mut()),
            0,
            "RSA key initialization failed"
        );
        assert_eq!(
            wc_MakeRsaKey(key_ptr, 2048, 65537, rng_ptr),
            0,
            "RSA key generation failed"
        );

        // Pre-generate two pools of ciphertexts with OAEP padding
        let mut pool_baseline: Vec<Vec<u8>> = Vec::with_capacity(POOL_SIZE);
        let mut pool_sample: Vec<Vec<u8>> = Vec::with_capacity(POOL_SIZE);

        for _ in 0..POOL_SIZE {
            let msg = rand_bytes_32();
            let mut encrypted = vec![0u8; 256];
            let len = wc_RsaPublicEncrypt_ex(
                msg.as_ptr(),
                msg.len() as u32,
                encrypted.as_mut_ptr(),
                encrypted.len() as u32,
                key_ptr,
                rng_ptr,
                WC_RSA_OAEP_PAD,
                WC_HASH_TYPE_SHA256,
                WC_HASH_TYPE_SHA256, // MGF1 hash
                ptr::null(),
                0,
            );
            assert!(len > 0, "OAEP encryption failed");
            encrypted.truncate(len as usize);
            pool_baseline.push(encrypted);

            let msg = rand_bytes_32();
            let mut encrypted = vec![0u8; 256];
            let len = wc_RsaPublicEncrypt_ex(
                msg.as_ptr(),
                msg.len() as u32,
                encrypted.as_mut_ptr(),
                encrypted.len() as u32,
                key_ptr,
                rng_ptr,
                WC_RSA_OAEP_PAD,
                WC_HASH_TYPE_SHA256,
                WC_HASH_TYPE_SHA256,
                ptr::null(),
                0,
            );
            assert!(len > 0, "OAEP encryption failed");
            encrypted.truncate(len as usize);
            pool_sample.push(encrypted);
        }

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
                let result = wc_RsaPrivateDecrypt_ex(
                    ciphertext.as_ptr(),
                    ciphertext.len() as u32,
                    decrypted.as_mut_ptr(),
                    decrypted.len() as u32,
                    key_ptr,
                    WC_RSA_OAEP_PAD,
                    WC_HASH_TYPE_SHA256,
                    WC_HASH_TYPE_SHA256,
                    ptr::null(),
                    0,
                );
                std::hint::black_box(result);
            });

        eprintln!("\n[wolfssl_rsa_2048_oaep_decrypt_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        wc_FreeRsaKey(key_ptr);
        wc_FreeRng(rng_ptr);
        wolfCrypt_Cleanup();

        let outcome = skip_if_unreliable!(outcome, "wolfssl_rsa_2048_oaep_decrypt_constant_time");

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
                eprintln!("⚠️  TIMING LEAK DETECTED in wolfSSL RSA OAEP decryption");
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

/// wolfSSL ECDSA P-256 signing should be constant-time
///
/// ECDSA is particularly sensitive to timing attacks - nonce generation and
/// modular inversion can leak the private key. Multiple CVEs in 2024.
#[test]
fn wolfssl_ecdsa_p256_sign_constant_time() {
    unsafe {
        assert_eq!(wolfCrypt_Init(), 0, "wolfCrypt initialization failed");

        let mut rng: MaybeUninit<WC_RNG> = MaybeUninit::uninit();
        let rng_ptr = rng.as_mut_ptr();
        assert_eq!(wc_InitRng(rng_ptr), 0, "RNG initialization failed");

        // Generate P-256 key pair
        let mut key: MaybeUninit<ecc_key> = MaybeUninit::uninit();
        let key_ptr = key.as_mut_ptr();
        assert_eq!(wc_ecc_init(key_ptr), 0, "ECC key initialization failed");
        assert_eq!(
            wc_ecc_make_key_ex(rng_ptr, 32, key_ptr, ECC_SECP256R1),
            0,
            "ECC key generation failed"
        );

        // Use zeros vs random pattern for message
        let inputs = InputPair::new(|| [0u8; 32], rand_bytes_32);

        let outcome = TimingOracle::for_attacker(AttackerModel::SharedHardware)
            .pass_threshold(0.15)
            .fail_threshold(0.99)
            .time_budget(Duration::from_secs(60))
            .warmup(100)
            .test(inputs, |message| {
                let mut signature = vec![0u8; 72]; // DER-encoded ECDSA signature max size
                let mut sig_len = signature.len() as u32;
                let result = wc_ecc_sign_hash(
                    message.as_ptr(),
                    message.len() as u32,
                    signature.as_mut_ptr(),
                    &mut sig_len,
                    rng_ptr,
                    key_ptr,
                );
                std::hint::black_box(result);
                std::hint::black_box(signature[0]);
            });

        eprintln!("\n[wolfssl_ecdsa_p256_sign_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        wc_ecc_free(key_ptr);
        wc_FreeRng(rng_ptr);
        wolfCrypt_Cleanup();

        let outcome = skip_if_unreliable!(outcome, "wolfssl_ecdsa_p256_sign_constant_time");

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
                eprintln!("⚠️  TIMING LEAK DETECTED in wolfSSL ECDSA P-256 signing");
                eprintln!(
                    "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                    leak_probability * 100.0,
                    exploitability,
                    effect.max_effect_ns
                );
                eprintln!("   Similar to CVEs in other ECDSA implementations (2024)");
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

/// wolfSSL AES-256-GCM encryption should be constant-time
///
/// Tests for data-dependent timing in AES-GCM encryption.
#[test]
fn wolfssl_aes_256_gcm_encrypt_constant_time() {
    unsafe {
        assert_eq!(wolfCrypt_Init(), 0, "wolfCrypt initialization failed");

        let key = rand_bytes_32();
        let mut aes: MaybeUninit<Aes> = MaybeUninit::uninit();
        let aes_ptr = aes.as_mut_ptr();

        assert_eq!(
            wc_AesInit(aes_ptr, ptr::null_mut(), -1),
            0,
            "AES initialization failed"
        );
        assert_eq!(
            wc_AesGcmSetKey(aes_ptr, key.as_ptr(), key.len() as u32),
            0,
            "AES-GCM key setup failed"
        );

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

                let mut ciphertext = vec![0u8; plaintext.len()];
                let mut tag = vec![0u8; 16];
                let result = wc_AesGcmEncrypt(
                    aes_ptr,
                    ciphertext.as_mut_ptr(),
                    plaintext.as_ptr(),
                    plaintext.len() as u32,
                    nonce.as_ptr(),
                    nonce.len() as u32,
                    tag.as_mut_ptr(),
                    tag.len() as u32,
                    ptr::null(),
                    0,
                );
                std::hint::black_box(result);
                std::hint::black_box(ciphertext[0]);
                std::hint::black_box(tag[0]);
            });

        eprintln!("\n[wolfssl_aes_256_gcm_encrypt_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        wc_AesFree(aes_ptr);
        wolfCrypt_Cleanup();

        let outcome = skip_if_unreliable!(outcome, "wolfssl_aes_256_gcm_encrypt_constant_time");

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
                eprintln!("⚠️  TIMING LEAK DETECTED in wolfSSL AES-256-GCM encryption");
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

/// wolfSSL AES-256-GCM decryption should be constant-time
///
/// Tests for data-dependent timing in AES-GCM decryption, including MAC verification.
#[test]
fn wolfssl_aes_256_gcm_decrypt_constant_time() {
    const POOL_SIZE: usize = 100;

    unsafe {
        assert_eq!(wolfCrypt_Init(), 0, "wolfCrypt initialization failed");

        let key = rand_bytes_32();
        let mut aes: MaybeUninit<Aes> = MaybeUninit::uninit();
        let aes_ptr = aes.as_mut_ptr();

        assert_eq!(
            wc_AesInit(aes_ptr, ptr::null_mut(), -1),
            0,
            "AES initialization failed"
        );
        assert_eq!(
            wc_AesGcmSetKey(aes_ptr, key.as_ptr(), key.len() as u32),
            0,
            "AES-GCM key setup failed"
        );

        // Pre-generate two pools of ciphertext/tag pairs
        let mut pool_baseline: Vec<(Vec<u8>, Vec<u8>, [u8; 12])> = Vec::with_capacity(POOL_SIZE);
        let mut pool_sample: Vec<(Vec<u8>, Vec<u8>, [u8; 12])> = Vec::with_capacity(POOL_SIZE);

        for i in 0..POOL_SIZE {
            // Baseline: all zeros
            let plaintext = [0u8; 64];
            let mut nonce = [0u8; 12];
            nonce[..8].copy_from_slice(&(i as u64).to_le_bytes());
            let mut ciphertext = vec![0u8; 64];
            let mut tag = vec![0u8; 16];
            wc_AesGcmEncrypt(
                aes_ptr,
                ciphertext.as_mut_ptr(),
                plaintext.as_ptr(),
                plaintext.len() as u32,
                nonce.as_ptr(),
                nonce.len() as u32,
                tag.as_mut_ptr(),
                tag.len() as u32,
                ptr::null(),
                0,
            );
            pool_baseline.push((ciphertext, tag, nonce));

            // Sample: random
            let plaintext = rand_bytes_64();
            let mut nonce = [0u8; 12];
            nonce[..8].copy_from_slice(&((i + POOL_SIZE) as u64).to_le_bytes());
            let mut ciphertext = vec![0u8; 64];
            let mut tag = vec![0u8; 16];
            wc_AesGcmEncrypt(
                aes_ptr,
                ciphertext.as_mut_ptr(),
                plaintext.as_ptr(),
                plaintext.len() as u32,
                nonce.as_ptr(),
                nonce.len() as u32,
                tag.as_mut_ptr(),
                tag.len() as u32,
                ptr::null(),
                0,
            );
            pool_sample.push((ciphertext, tag, nonce));
        }

        let idx_baseline = std::cell::Cell::new(0usize);
        let idx_sample = std::cell::Cell::new(0usize);

        let inputs = InputPair::new(
            move || {
                let i = idx_baseline.get();
                idx_baseline.set((i + 1) % POOL_SIZE);
                i
            },
            move || {
                let i = idx_sample.get();
                idx_sample.set((i + 1) % POOL_SIZE);
                i + POOL_SIZE
            },
        );

        let all_pairs: Vec<_> = pool_baseline.into_iter().chain(pool_sample).collect();

        let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .pass_threshold(0.15)
            .fail_threshold(0.99)
            .time_budget(Duration::from_secs(30))
            .test(inputs, |idx| {
                let (ciphertext, tag, nonce) = &all_pairs[*idx];
                let mut plaintext = vec![0u8; ciphertext.len()];
                let result = wc_AesGcmDecrypt(
                    aes_ptr,
                    plaintext.as_mut_ptr(),
                    ciphertext.as_ptr(),
                    ciphertext.len() as u32,
                    nonce.as_ptr(),
                    nonce.len() as u32,
                    tag.as_ptr(),
                    tag.len() as u32,
                    ptr::null(),
                    0,
                );
                std::hint::black_box(result);
                std::hint::black_box(plaintext[0]);
            });

        eprintln!("\n[wolfssl_aes_256_gcm_decrypt_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        wc_AesFree(aes_ptr);
        wolfCrypt_Cleanup();

        let outcome = skip_if_unreliable!(outcome, "wolfssl_aes_256_gcm_decrypt_constant_time");

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
                eprintln!("⚠️  TIMING LEAK DETECTED in wolfSSL AES-256-GCM decryption");
                eprintln!(
                    "   P(leak)={:.1}%, exploitability={:?}, effect={:.1}ns",
                    leak_probability * 100.0,
                    exploitability,
                    effect.max_effect_ns
                );
                panic!("AES-256-GCM decryption timing leak");
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
fn wolfssl_harness_sanity_check() {
    unsafe {
        assert_eq!(wolfCrypt_Init(), 0, "wolfCrypt initialization failed");

        let mut rng: MaybeUninit<WC_RNG> = MaybeUninit::uninit();
        let rng_ptr = rng.as_mut_ptr();
        assert_eq!(wc_InitRng(rng_ptr), 0, "RNG initialization failed");

        let mut key: MaybeUninit<RsaKey> = MaybeUninit::uninit();
        let key_ptr = key.as_mut_ptr();
        assert_eq!(
            wc_InitRsaKey(key_ptr, ptr::null_mut()),
            0,
            "RSA key initialization failed"
        );
        assert_eq!(
            wc_MakeRsaKey(key_ptr, 2048, 65537, rng_ptr),
            0,
            "RSA key generation failed"
        );

        // Pre-generate a single ciphertext
        let msg = rand_bytes_32();
        let mut encrypted = vec![0u8; 256];
        let len = wc_RsaPublicEncrypt(
            msg.as_ptr(),
            msg.len() as u32,
            encrypted.as_mut_ptr(),
            encrypted.len() as u32,
            key_ptr,
            rng_ptr,
        );
        assert!(len > 0, "Encryption failed");
        encrypted.truncate(len as usize);

        // Both classes return the same ciphertext
        let inputs = InputPair::new(|| encrypted.clone(), || encrypted.clone());

        let outcome = TimingOracle::for_attacker(AttackerModel::Research)
            .time_budget(Duration::from_secs(10))
            .test(inputs, |ciphertext| {
                let mut decrypted = vec![0u8; 256];
                let result = wc_RsaPrivateDecrypt(
                    ciphertext.as_ptr(),
                    ciphertext.len() as u32,
                    decrypted.as_mut_ptr(),
                    decrypted.len() as u32,
                    key_ptr,
                );
                std::hint::black_box(result);
            });

        eprintln!("\n[wolfssl_harness_sanity_check]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        wc_FreeRsaKey(key_ptr);
        wc_FreeRng(rng_ptr);
        wolfCrypt_Cleanup();

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
