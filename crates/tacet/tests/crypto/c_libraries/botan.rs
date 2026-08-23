//! Botan timing tests
//!
//! Tests cryptographic implementations from Botan (modern C++11/14 crypto library) via C FFI.
//! Uses DudeCT's two-class pattern:
//! - Baseline: All zeros or fixed values
//! - Sample: Random values
//!
//! CRITICAL: FFI harness verification is essential. Always run sanity checks first.
//!
//! **NOTE**: These tests require Botan development files.
//! On macOS: `brew install botan`
//! Botan version: 3.10.0
//!
//! Operations tested:
//! - RSA-2048 PKCS#1 v1.5 decryption (Bleichenbacher-class attacks like MARVIN)
//! - RSA-2048 OAEP decryption (SHA-256)
//! - ECDSA P-256 signing
//! - AES-256-GCM encryption/decryption
//!
//! Botan is the only modern C++ crypto library in the test suite. It emphasizes
//! constant-time implementations and side-channel resistance.

// Botan's FFI takes NUL-terminated byte strings as *const u8, so explicit b"...\0"
// literals are used instead of c"..." (which yields *const c_char).
#![allow(clippy::manual_c_str_literals)]

use std::ptr;
use std::time::Duration;
use tacet::helpers::InputPair;
use tacet::{skip_if_unreliable, AttackerModel, Outcome, TimingOracle};

// ============================================================================
// FFI Bindings to Botan
// ============================================================================

// Botan opaque types
type BotanRng = *mut std::ffi::c_void;
type BotanPrivkey = *mut std::ffi::c_void;
type BotanPubkey = *mut std::ffi::c_void;
type BotanPkOpDecrypt = *mut std::ffi::c_void;
type BotanPkOpEncrypt = *mut std::ffi::c_void;
type BotanPkOpSign = *mut std::ffi::c_void;
type BotanCipher = *mut std::ffi::c_void;

// Error codes
const BOTAN_FFI_SUCCESS: i32 = 0;

#[link(name = "botan-3")]
extern "C" {
    // RNG
    fn botan_rng_init(rng: *mut BotanRng, rng_type: *const u8) -> i32;
    fn botan_rng_destroy(rng: BotanRng) -> i32;

    // Private key operations
    fn botan_privkey_create(
        key: *mut BotanPrivkey,
        algo_name: *const u8,
        algo_params: *const u8,
        rng: BotanRng,
    ) -> i32;
    fn botan_privkey_destroy(key: BotanPrivkey) -> i32;
    fn botan_privkey_export_pubkey(pubkey: *mut BotanPubkey, privkey: BotanPrivkey) -> i32;
    fn botan_pubkey_destroy(key: BotanPubkey) -> i32;

    // RSA decrypt operations
    fn botan_pk_op_decrypt_create(
        op: *mut BotanPkOpDecrypt,
        key: BotanPrivkey,
        padding: *const u8,
        flags: u32,
    ) -> i32;
    fn botan_pk_op_decrypt_destroy(op: BotanPkOpDecrypt) -> i32;
    fn botan_pk_op_decrypt(
        op: BotanPkOpDecrypt,
        out: *mut u8,
        out_len: *mut usize,
        ciphertext: *const u8,
        ciphertext_len: usize,
    ) -> i32;

    // RSA encrypt operations (for test setup)
    fn botan_pk_op_encrypt_create(
        op: *mut BotanPkOpEncrypt,
        key: BotanPubkey,
        padding: *const u8,
        flags: u32,
    ) -> i32;
    fn botan_pk_op_encrypt_destroy(op: BotanPkOpEncrypt) -> i32;
    fn botan_pk_op_encrypt_output_length(
        op: BotanPkOpEncrypt,
        ptext_len: usize,
        ctext_len: *mut usize,
    ) -> i32;
    fn botan_pk_op_encrypt(
        op: BotanPkOpEncrypt,
        rng: BotanRng,
        out: *mut u8,
        out_len: *mut usize,
        plaintext: *const u8,
        plaintext_len: usize,
    ) -> i32;

    // ECDSA signing
    fn botan_pk_op_sign_create(
        op: *mut BotanPkOpSign,
        key: BotanPrivkey,
        hash_and_padding: *const u8,
        flags: u32,
    ) -> i32;
    fn botan_pk_op_sign_destroy(op: BotanPkOpSign) -> i32;
    fn botan_pk_op_sign_update(op: BotanPkOpSign, data: *const u8, data_len: usize) -> i32;
    fn botan_pk_op_sign_finish(
        op: BotanPkOpSign,
        rng: BotanRng,
        sig: *mut u8,
        sig_len: *mut usize,
    ) -> i32;
    fn botan_pk_op_sign_output_length(op: BotanPkOpSign, olen: *mut usize) -> i32;

    // Symmetric cipher operations
    fn botan_cipher_init(cipher: *mut BotanCipher, name: *const u8, flags: u32) -> i32;
    fn botan_cipher_destroy(cipher: BotanCipher) -> i32;
    fn botan_cipher_set_key(cipher: BotanCipher, key: *const u8, key_len: usize) -> i32;
    fn botan_cipher_start(cipher: BotanCipher, nonce: *const u8, nonce_len: usize) -> i32;
    fn botan_cipher_update(
        cipher: BotanCipher,
        flags: u32,
        output: *mut u8,
        output_written: *mut usize,
        output_size: usize,
        input_bytes: *const u8,
        input_size: usize,
        input_consumed: *mut usize,
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

// RAII wrappers for Botan objects
struct BotanRngHandle(BotanRng);

impl BotanRngHandle {
    fn new() -> Result<Self, &'static str> {
        unsafe {
            let mut rng: BotanRng = ptr::null_mut();
            let rc = botan_rng_init(&mut rng, b"system\0".as_ptr());
            if rc == BOTAN_FFI_SUCCESS {
                Ok(BotanRngHandle(rng))
            } else {
                Err("Failed to initialize RNG")
            }
        }
    }

    fn as_ptr(&self) -> BotanRng {
        self.0
    }
}

impl Drop for BotanRngHandle {
    fn drop(&mut self) {
        unsafe {
            botan_rng_destroy(self.0);
        }
    }
}

struct BotanPrivkeyHandle(BotanPrivkey);

impl BotanPrivkeyHandle {
    fn create_rsa(rng: &BotanRngHandle, bits: usize) -> Result<Self, &'static str> {
        unsafe {
            let mut key: BotanPrivkey = ptr::null_mut();
            let algo_params = format!("{}\0", bits);
            let rc = botan_privkey_create(
                &mut key,
                b"RSA\0".as_ptr(),
                algo_params.as_ptr(),
                rng.as_ptr(),
            );
            if rc == BOTAN_FFI_SUCCESS {
                Ok(BotanPrivkeyHandle(key))
            } else {
                Err("Failed to create RSA private key")
            }
        }
    }

    fn create_ecdsa(rng: &BotanRngHandle, curve: &str) -> Result<Self, &'static str> {
        unsafe {
            let mut key: BotanPrivkey = ptr::null_mut();
            let curve_param = format!("{}\0", curve);
            let rc = botan_privkey_create(
                &mut key,
                b"ECDSA\0".as_ptr(),
                curve_param.as_ptr(),
                rng.as_ptr(),
            );
            if rc == BOTAN_FFI_SUCCESS {
                Ok(BotanPrivkeyHandle(key))
            } else {
                Err("Failed to create ECDSA private key")
            }
        }
    }

    fn export_pubkey(&self) -> Result<BotanPubkeyHandle, &'static str> {
        unsafe {
            let mut pubkey: BotanPubkey = ptr::null_mut();
            let rc = botan_privkey_export_pubkey(&mut pubkey, self.0);
            if rc == BOTAN_FFI_SUCCESS {
                Ok(BotanPubkeyHandle(pubkey))
            } else {
                Err("Failed to export public key")
            }
        }
    }

    fn as_ptr(&self) -> BotanPrivkey {
        self.0
    }
}

impl Drop for BotanPrivkeyHandle {
    fn drop(&mut self) {
        unsafe {
            botan_privkey_destroy(self.0);
        }
    }
}

struct BotanPubkeyHandle(BotanPubkey);

impl BotanPubkeyHandle {
    fn as_ptr(&self) -> BotanPubkey {
        self.0
    }
}

impl Drop for BotanPubkeyHandle {
    fn drop(&mut self) {
        unsafe {
            botan_pubkey_destroy(self.0);
        }
    }
}

// ============================================================================
// RSA PKCS#1 v1.5 Tests (Bleichenbacher-class, MARVIN)
// ============================================================================

/// Botan RSA PKCS#1 v1.5 decryption should be constant-time
///
/// Tests for timing side channels in PKCS#1 v1.5 padding validation.
/// Historical vulnerabilities: Bleichenbacher (1998), ROBOT (2017), MARVIN (2023)
///
/// Uses pool-based pattern to avoid caching artifacts.
#[test]
fn botan_rsa_2048_pkcs1v15_decrypt_constant_time() {
    const POOL_SIZE: usize = 100;

    unsafe {
        let rng = BotanRngHandle::new().expect("Failed to init RNG");
        let privkey = BotanPrivkeyHandle::create_rsa(&rng, 2048).expect("Failed to create RSA key");
        let pubkey = privkey.export_pubkey().expect("Failed to export pubkey");

        // Create encrypt operator for test setup
        let mut encrypt_op: BotanPkOpEncrypt = ptr::null_mut();
        let rc =
            botan_pk_op_encrypt_create(&mut encrypt_op, pubkey.as_ptr(), b"PKCS1v15\0".as_ptr(), 0);
        assert_eq!(rc, BOTAN_FFI_SUCCESS, "Failed to create encrypt op");

        // Pre-generate two pools of ciphertexts
        let pool_baseline: Vec<Vec<u8>> = (0..POOL_SIZE)
            .map(|_| {
                let msg = rand_bytes_32();
                let mut ciphertext_len = 0usize;
                botan_pk_op_encrypt_output_length(encrypt_op, msg.len(), &mut ciphertext_len);
                let mut ciphertext = vec![0u8; ciphertext_len];
                let mut actual_len = ciphertext_len;
                botan_pk_op_encrypt(
                    encrypt_op,
                    rng.as_ptr(),
                    ciphertext.as_mut_ptr(),
                    &mut actual_len,
                    msg.as_ptr(),
                    msg.len(),
                );
                ciphertext.truncate(actual_len);
                ciphertext
            })
            .collect();

        let pool_sample: Vec<Vec<u8>> = (0..POOL_SIZE)
            .map(|_| {
                let msg = rand_bytes_32();
                let mut ciphertext_len = 0usize;
                botan_pk_op_encrypt_output_length(encrypt_op, msg.len(), &mut ciphertext_len);
                let mut ciphertext = vec![0u8; ciphertext_len];
                let mut actual_len = ciphertext_len;
                botan_pk_op_encrypt(
                    encrypt_op,
                    rng.as_ptr(),
                    ciphertext.as_mut_ptr(),
                    &mut actual_len,
                    msg.as_ptr(),
                    msg.len(),
                );
                ciphertext.truncate(actual_len);
                ciphertext
            })
            .collect();

        botan_pk_op_encrypt_destroy(encrypt_op);

        // Create decrypt operator
        let mut decrypt_op: BotanPkOpDecrypt = ptr::null_mut();
        let rc = botan_pk_op_decrypt_create(
            &mut decrypt_op,
            privkey.as_ptr(),
            b"PKCS1v15\0".as_ptr(),
            0,
        );
        assert_eq!(rc, BOTAN_FFI_SUCCESS, "Failed to create decrypt op");

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
                let mut plaintext = vec![0u8; 256];
                let mut plaintext_len = plaintext.len();
                let result = botan_pk_op_decrypt(
                    decrypt_op,
                    plaintext.as_mut_ptr(),
                    &mut plaintext_len,
                    ciphertext.as_ptr(),
                    ciphertext.len(),
                );
                std::hint::black_box(result);
            });

        eprintln!("\n[botan_rsa_2048_pkcs1v15_decrypt_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        botan_pk_op_decrypt_destroy(decrypt_op);

        let outcome = skip_if_unreliable!(outcome, "botan_rsa_2048_pkcs1v15_decrypt_constant_time");

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
                eprintln!("⚠️  TIMING LEAK DETECTED in Botan RSA PKCS#1 v1.5 decryption");
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

/// Botan RSA OAEP decryption should be constant-time
///
/// OAEP is more robust than PKCS#1 v1.5 but still requires constant-time implementation.
#[test]
fn botan_rsa_2048_oaep_decrypt_constant_time() {
    const POOL_SIZE: usize = 100;

    unsafe {
        let rng = BotanRngHandle::new().expect("Failed to init RNG");
        let privkey = BotanPrivkeyHandle::create_rsa(&rng, 2048).expect("Failed to create RSA key");
        let pubkey = privkey.export_pubkey().expect("Failed to export pubkey");

        // Create encrypt operator for test setup (OAEP with SHA-256)
        let mut encrypt_op: BotanPkOpEncrypt = ptr::null_mut();
        let rc = botan_pk_op_encrypt_create(
            &mut encrypt_op,
            pubkey.as_ptr(),
            b"OAEP(SHA-256)\0".as_ptr(),
            0,
        );
        assert_eq!(rc, BOTAN_FFI_SUCCESS, "Failed to create encrypt op");

        // Pre-generate two pools of ciphertexts
        let pool_baseline: Vec<Vec<u8>> = (0..POOL_SIZE)
            .map(|_| {
                let msg = rand_bytes_32();
                let mut ciphertext_len = 0usize;
                botan_pk_op_encrypt_output_length(encrypt_op, msg.len(), &mut ciphertext_len);
                let mut ciphertext = vec![0u8; ciphertext_len];
                let mut actual_len = ciphertext_len;
                botan_pk_op_encrypt(
                    encrypt_op,
                    rng.as_ptr(),
                    ciphertext.as_mut_ptr(),
                    &mut actual_len,
                    msg.as_ptr(),
                    msg.len(),
                );
                ciphertext.truncate(actual_len);
                ciphertext
            })
            .collect();

        let pool_sample: Vec<Vec<u8>> = (0..POOL_SIZE)
            .map(|_| {
                let msg = rand_bytes_32();
                let mut ciphertext_len = 0usize;
                botan_pk_op_encrypt_output_length(encrypt_op, msg.len(), &mut ciphertext_len);
                let mut ciphertext = vec![0u8; ciphertext_len];
                let mut actual_len = ciphertext_len;
                botan_pk_op_encrypt(
                    encrypt_op,
                    rng.as_ptr(),
                    ciphertext.as_mut_ptr(),
                    &mut actual_len,
                    msg.as_ptr(),
                    msg.len(),
                );
                ciphertext.truncate(actual_len);
                ciphertext
            })
            .collect();

        botan_pk_op_encrypt_destroy(encrypt_op);

        // Create decrypt operator
        let mut decrypt_op: BotanPkOpDecrypt = ptr::null_mut();
        let rc = botan_pk_op_decrypt_create(
            &mut decrypt_op,
            privkey.as_ptr(),
            b"OAEP(SHA-256)\0".as_ptr(),
            0,
        );
        assert_eq!(rc, BOTAN_FFI_SUCCESS, "Failed to create decrypt op");

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
                let mut plaintext = vec![0u8; 256];
                let mut plaintext_len = plaintext.len();
                let result = botan_pk_op_decrypt(
                    decrypt_op,
                    plaintext.as_mut_ptr(),
                    &mut plaintext_len,
                    ciphertext.as_ptr(),
                    ciphertext.len(),
                );
                std::hint::black_box(result);
            });

        eprintln!("\n[botan_rsa_2048_oaep_decrypt_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        botan_pk_op_decrypt_destroy(decrypt_op);

        let outcome = skip_if_unreliable!(outcome, "botan_rsa_2048_oaep_decrypt_constant_time");

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
                eprintln!("⚠️  TIMING LEAK DETECTED in Botan RSA OAEP decryption");
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

/// Botan ECDSA P-256 signing should be constant-time
///
/// ECDSA is particularly sensitive to timing attacks - nonce generation and
/// modular inversion can leak the private key.
#[test]
fn botan_ecdsa_p256_sign_constant_time() {
    unsafe {
        let rng = BotanRngHandle::new().expect("Failed to init RNG");
        let privkey = BotanPrivkeyHandle::create_ecdsa(&rng, "secp256r1")
            .expect("Failed to create ECDSA key");

        // Create sign operator
        let mut sign_op: BotanPkOpSign = ptr::null_mut();
        let rc = botan_pk_op_sign_create(&mut sign_op, privkey.as_ptr(), b"SHA-256\0".as_ptr(), 0);
        assert_eq!(rc, BOTAN_FFI_SUCCESS, "Failed to create sign op");

        // Get signature length
        let mut sig_len = 0usize;
        botan_pk_op_sign_output_length(sign_op, &mut sig_len);

        // Use zeros vs random pattern for message
        let inputs = InputPair::new(|| [0u8; 32], rand_bytes_32);

        let outcome = TimingOracle::for_attacker(AttackerModel::SharedHardware)
            .pass_threshold(0.15)
            .fail_threshold(0.99)
            .time_budget(Duration::from_secs(60))
            .warmup(100)
            .test(inputs, |message| {
                let mut signature = vec![0u8; sig_len];
                let mut actual_sig_len = sig_len;

                botan_pk_op_sign_update(sign_op, message.as_ptr(), message.len());
                botan_pk_op_sign_finish(
                    sign_op,
                    rng.as_ptr(),
                    signature.as_mut_ptr(),
                    &mut actual_sig_len,
                );
                std::hint::black_box(signature[0]);
            });

        eprintln!("\n[botan_ecdsa_p256_sign_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        botan_pk_op_sign_destroy(sign_op);

        let outcome = skip_if_unreliable!(outcome, "botan_ecdsa_p256_sign_constant_time");

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
                eprintln!("⚠️  TIMING LEAK DETECTED in Botan ECDSA P-256 signing");
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

/// Botan AES-256-GCM encryption should be constant-time
///
/// Tests for data-dependent timing in GCM mode.
#[test]
fn botan_aes_256_gcm_encrypt_constant_time() {
    unsafe {
        let key = rand_bytes_32();

        // Create cipher
        let mut cipher: BotanCipher = ptr::null_mut();
        let rc = botan_cipher_init(&mut cipher, b"AES-256/GCM\0".as_ptr(), 0);
        assert_eq!(rc, BOTAN_FFI_SUCCESS, "Failed to create cipher");

        // Set key
        let rc = botan_cipher_set_key(cipher, key.as_ptr(), key.len());
        assert_eq!(rc, BOTAN_FFI_SUCCESS, "Failed to set key");

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

                botan_cipher_start(cipher, nonce.as_ptr(), nonce.len());

                let mut ciphertext = vec![0u8; plaintext.len() + 16]; // +16 for GCM tag
                let mut output_written = 0usize;
                let mut input_consumed = 0usize;

                botan_cipher_update(
                    cipher,
                    1, // Final flag
                    ciphertext.as_mut_ptr(),
                    &mut output_written,
                    ciphertext.len(),
                    plaintext.as_ptr(),
                    plaintext.len(),
                    &mut input_consumed,
                );

                std::hint::black_box(ciphertext[0]);
            });

        eprintln!("\n[botan_aes_256_gcm_encrypt_constant_time]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        botan_cipher_destroy(cipher);

        let outcome = skip_if_unreliable!(outcome, "botan_aes_256_gcm_encrypt_constant_time");

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
                eprintln!("⚠️  TIMING LEAK DETECTED in Botan AES-256-GCM encryption");
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
fn botan_harness_sanity_check() {
    unsafe {
        let rng = BotanRngHandle::new().expect("Failed to init RNG");
        let privkey = BotanPrivkeyHandle::create_rsa(&rng, 2048).expect("Failed to create RSA key");
        let pubkey = privkey.export_pubkey().expect("Failed to export pubkey");

        // Create encrypt operator
        let mut encrypt_op: BotanPkOpEncrypt = ptr::null_mut();
        let rc =
            botan_pk_op_encrypt_create(&mut encrypt_op, pubkey.as_ptr(), b"PKCS1v15\0".as_ptr(), 0);
        assert_eq!(rc, BOTAN_FFI_SUCCESS, "Failed to create encrypt op");

        // Generate a single ciphertext
        let msg = rand_bytes_32();
        let mut ciphertext_len = 0usize;
        botan_pk_op_encrypt_output_length(encrypt_op, msg.len(), &mut ciphertext_len);
        let mut ciphertext = vec![0u8; ciphertext_len];
        let mut actual_len = ciphertext_len;
        botan_pk_op_encrypt(
            encrypt_op,
            rng.as_ptr(),
            ciphertext.as_mut_ptr(),
            &mut actual_len,
            msg.as_ptr(),
            msg.len(),
        );
        ciphertext.truncate(actual_len);

        botan_pk_op_encrypt_destroy(encrypt_op);

        // Create decrypt operator
        let mut decrypt_op: BotanPkOpDecrypt = ptr::null_mut();
        let rc = botan_pk_op_decrypt_create(
            &mut decrypt_op,
            privkey.as_ptr(),
            b"PKCS1v15\0".as_ptr(),
            0,
        );
        assert_eq!(rc, BOTAN_FFI_SUCCESS, "Failed to create decrypt op");

        // Both classes return the same ciphertext
        let inputs = InputPair::new(|| ciphertext.clone(), || ciphertext.clone());

        let outcome = TimingOracle::for_attacker(AttackerModel::Research)
            .time_budget(Duration::from_secs(10))
            .test(inputs, |ct| {
                let mut plaintext = vec![0u8; 256];
                let mut plaintext_len = plaintext.len();
                let result = botan_pk_op_decrypt(
                    decrypt_op,
                    plaintext.as_mut_ptr(),
                    &mut plaintext_len,
                    ct.as_ptr(),
                    ct.len(),
                );
                std::hint::black_box(result);
            });

        eprintln!("\n[botan_harness_sanity_check]");
        eprintln!("{}", tacet::output::format_outcome(&outcome));

        botan_pk_op_decrypt_destroy(decrypt_op);

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
