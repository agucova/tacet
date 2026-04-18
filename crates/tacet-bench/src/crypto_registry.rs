//! Registry of real-crypto test cases for cross-tool fan-out.
//!
//! Each `CryptoTestCase` owns a `collect` closure that runs a specific
//! crypto primitive against `samples_per_class` inputs and returns the
//! interleaved timings in `CollectedBlocked`. The same raw data is then
//! handed to every `ToolAdapter`, giving an apples-to-apples comparison.
//!
//! The closures inline the pool / setup patterns from the existing
//! `crates/tacet/tests/crypto/**` files. They deliberately duplicate
//! ~30 lines per test rather than sharing traits, to keep the registry
//! standalone and keep the source tests untouched.
//!
//! # Tiers
//!
//! - [`tier1`]: seven constant-time primitives, one per ecosystem.
//!   Expected verdict: `Pass` for all tools; used as the FPR probe.
//! - [`tier2`]: the MARVIN RSA-1024 padding oracle (CVE-2023-49092).
//!   Expected verdict: `Fail` for all tools; used as the detection probe.

use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use tacet::{AttackerModel, Class};

use crate::crypto_collect::{run_collection, CollectedBlocked};

/// Static information about a crypto test case and its collection closure.
pub struct CryptoTestCase {
    /// Stable identifier used in CSV output. Colon-separated:
    /// `ecosystem::library::primitive::variant`.
    pub id: &'static str,
    /// Source ecosystem: "Rust" | "C/C++" | "Go" | "JS".
    pub ecosystem: &'static str,
    /// Upstream crate / library name shown in reports.
    pub library: &'static str,
    /// Primitive family (e.g., "AES-128", "X25519", "RSA-1024-PKCS1v15").
    pub primitive: &'static str,
    /// Whether this test is known to be leaky (true positive target).
    pub is_leaky: bool,
    /// Attacker model the case should be judged under.
    pub attacker_model: AttackerModel,
    /// Samples drawn from each class. Matches Fig 1/2 when 10 000.
    pub samples_per_class: usize,
    /// Warmup iterations before measurement.
    pub warmup: usize,
    /// Boxed collection closure. Takes a per-iteration seed and an override
    /// for `samples_per_class` (so the CLI can trim for smoke tests).
    pub collect: Arc<dyn Fn(u64, usize) -> CollectedBlocked + Send + Sync>,
}

impl CryptoTestCase {
    /// Convenience: invoke `collect` with the case's default sample count.
    pub fn run(&self, seed: u64) -> CollectedBlocked {
        (self.collect)(seed, self.samples_per_class)
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn rand_bytes<const N: usize>(rng: &mut StdRng) -> [u8; N] {
    let mut out = [0u8; N];
    rng.fill_bytes(&mut out);
    out
}

// =============================================================================
// Tier 1 — Constant-time primitives (FPR probe)
// =============================================================================

/// Seven constant-time cases, one per ecosystem.
pub fn tier1() -> Vec<CryptoTestCase> {
    vec![
        tier1_rustcrypto_aes128_encrypt(),
        tier1_ring_aes256gcm_encrypt(),
        tier1_rustcrypto_chacha20poly1305_encrypt(),
        tier1_rustcrypto_sha3_256(),
        tier1_dalek_x25519(),
        tier1_libsodium_ed25519_sign(),
        tier1_pqcrypto_kyber768_decapsulate(),
    ]
}

/// RustCrypto AES-128 block encrypt. Canonical "should pass" primitive.
fn tier1_rustcrypto_aes128_encrypt() -> CryptoTestCase {
    use aes::cipher::{BlockEncrypt, KeyInit};
    use aes::Aes128;

    CryptoTestCase {
        id: "rustcrypto::aes::aes128_encrypt",
        ecosystem: "Rust",
        library: "RustCrypto",
        primitive: "AES-128",
        is_leaky: false,
        attacker_model: AttackerModel::AdjacentNetwork,
        samples_per_class: 10_000,
        warmup: 1_000,
        collect: Arc::new(|seed, n| {
            // Non-pathological fixed key; same pattern as the crypto test.
            let key = [
                0x2bu8, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09,
                0xcf, 0x4f, 0x3c,
            ];
            let cipher = Aes128::new(&key.into());
            let fixed_plaintext = [
                0x32u8, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0,
                0x37, 0x07, 0x34,
            ];

            let mut rng = StdRng::seed_from_u64(seed);
            let sample_pool: Vec<[u8; 16]> = (0..n).map(|_| rand_bytes(&mut rng)).collect();

            let s_idx = Cell::new(0usize);
            run_collection(seed, n, 1_000, |class| {
                let pt = match class {
                    Class::Baseline => fixed_plaintext,
                    Class::Sample => {
                        let i = s_idx.get();
                        s_idx.set((i + 1) % sample_pool.len());
                        sample_pool[i]
                    }
                };
                let mut block = pt.into();
                cipher.encrypt_block(&mut block);
                black_box(block[0]);
            })
        }),
    }
}

/// ring AES-256-GCM seal. Cross-library AEAD via BoringSSL-derived assembly.
fn tier1_ring_aes256gcm_encrypt() -> CryptoTestCase {
    use ring::aead::{self, LessSafeKey, UnboundKey, AES_256_GCM};

    CryptoTestCase {
        id: "ring::aes_gcm::aes256gcm_seal",
        ecosystem: "Rust",
        library: "ring",
        primitive: "AES-256-GCM",
        is_leaky: false,
        attacker_model: AttackerModel::AdjacentNetwork,
        samples_per_class: 10_000,
        warmup: 1_000,
        collect: Arc::new(|seed, n| {
            let key_bytes: [u8; 32] = [
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
                0x1c, 0x1d, 0x1e, 0x1f,
            ];
            let unbound_key = UnboundKey::new(&AES_256_GCM, &key_bytes).unwrap();
            let key = LessSafeKey::new(unbound_key);
            let nonce_bytes: [u8; 12] = [0u8; 12];

            let fixed_plaintext = [0x42u8; 64];
            let mut rng = StdRng::seed_from_u64(seed);
            let sample_pool: Vec<[u8; 64]> = (0..n).map(|_| rand_bytes(&mut rng)).collect();

            let s_idx = Cell::new(0usize);
            run_collection(seed, n, 1_000, |class| {
                let pt = match class {
                    Class::Baseline => fixed_plaintext,
                    Class::Sample => {
                        let i = s_idx.get();
                        s_idx.set((i + 1) % sample_pool.len());
                        sample_pool[i]
                    }
                };
                let nonce = aead::Nonce::assume_unique_for_key(nonce_bytes);
                let mut in_out = pt.to_vec();
                let tag = key
                    .seal_in_place_separate_tag(nonce, aead::Aad::empty(), &mut in_out)
                    .unwrap();
                black_box(tag.as_ref()[0]);
            })
        }),
    }
}

/// RustCrypto ChaCha20-Poly1305 encrypt. ARX AEAD design.
fn tier1_rustcrypto_chacha20poly1305_encrypt() -> CryptoTestCase {
    use chacha20poly1305::{
        aead::{Aead, KeyInit},
        ChaCha20Poly1305, Nonce,
    };
    use std::sync::atomic::{AtomicU64, Ordering};

    CryptoTestCase {
        id: "rustcrypto::chacha20poly1305::encrypt",
        ecosystem: "Rust",
        library: "RustCrypto",
        primitive: "ChaCha20-Poly1305",
        is_leaky: false,
        attacker_model: AttackerModel::AdjacentNetwork,
        samples_per_class: 10_000,
        warmup: 1_000,
        collect: Arc::new(|seed, n| {
            let key_bytes: [u8; 32] = [
                0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
                0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
                0x1c, 0x1d, 0x1e, 0x1f,
            ];
            let cipher = ChaCha20Poly1305::new(&key_bytes.into());

            let nonce_counter = AtomicU64::new(0);
            let fixed_plaintext = [0x42u8; 64];
            let mut rng = StdRng::seed_from_u64(seed);
            let sample_pool: Vec<[u8; 64]> = (0..n).map(|_| rand_bytes(&mut rng)).collect();

            let s_idx = Cell::new(0usize);
            run_collection(seed, n, 1_000, |class| {
                let pt = match class {
                    Class::Baseline => fixed_plaintext,
                    Class::Sample => {
                        let i = s_idx.get();
                        s_idx.set((i + 1) % sample_pool.len());
                        sample_pool[i]
                    }
                };
                let nonce_value = nonce_counter.fetch_add(1, Ordering::Relaxed);
                let mut nonce_bytes = [0u8; 12];
                nonce_bytes[..8].copy_from_slice(&nonce_value.to_le_bytes());
                let nonce = Nonce::from_slice(&nonce_bytes);
                let ct = cipher.encrypt(nonce, pt.as_ref()).unwrap();
                black_box(ct[0]);
            })
        }),
    }
}

/// RustCrypto SHA3-256 digest.
fn tier1_rustcrypto_sha3_256() -> CryptoTestCase {
    use sha3::{Digest, Sha3_256};

    CryptoTestCase {
        id: "rustcrypto::sha3::sha3_256",
        ecosystem: "Rust",
        library: "RustCrypto",
        primitive: "SHA3-256",
        is_leaky: false,
        attacker_model: AttackerModel::AdjacentNetwork,
        samples_per_class: 10_000,
        warmup: 1_000,
        collect: Arc::new(|seed, n| {
            let fixed_input = [
                0x32u8, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0,
                0x37, 0x07, 0x34, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa,
                0xbb, 0xcc, 0xdd, 0xee, 0xff,
            ];
            let mut rng = StdRng::seed_from_u64(seed);
            let sample_pool: Vec<[u8; 32]> = (0..n).map(|_| rand_bytes(&mut rng)).collect();

            let s_idx = Cell::new(0usize);
            run_collection(seed, n, 1_000, |class| {
                let input = match class {
                    Class::Baseline => fixed_input,
                    Class::Sample => {
                        let i = s_idx.get();
                        s_idx.set((i + 1) % sample_pool.len());
                        sample_pool[i]
                    }
                };
                let hash = Sha3_256::digest(input);
                black_box(hash[0]);
            })
        }),
    }
}

/// dalek X25519 scalar multiplication. ECC primitive.
fn tier1_dalek_x25519() -> CryptoTestCase {
    use x25519_dalek::x25519;

    CryptoTestCase {
        id: "dalek::x25519::scalar_mult",
        ecosystem: "Rust",
        library: "dalek",
        primitive: "X25519",
        is_leaky: false,
        attacker_model: AttackerModel::AdjacentNetwork,
        samples_per_class: 10_000,
        warmup: 1_000,
        collect: Arc::new(|seed, n| {
            let basepoint = x25519_dalek::X25519_BASEPOINT_BYTES;
            let fixed_scalar: [u8; 32] = [
                0x4e, 0x5a, 0xb4, 0x34, 0x9d, 0x4c, 0x14, 0x82, 0x1b, 0xc8, 0x5b, 0x26, 0x8f, 0x0a,
                0x33, 0x9c, 0x7f, 0x4b, 0x2e, 0x8e, 0x1d, 0x6a, 0x3c, 0x5f, 0x9a, 0x2d, 0x7e, 0x4c,
                0x8b, 0x3a, 0x6d, 0x5e,
            ];
            let mut rng = StdRng::seed_from_u64(seed);
            let sample_pool: Vec<[u8; 32]> = (0..n).map(|_| rand_bytes(&mut rng)).collect();

            let s_idx = Cell::new(0usize);
            run_collection(seed, n, 1_000, |class| {
                let scalar = match class {
                    Class::Baseline => fixed_scalar,
                    Class::Sample => {
                        let i = s_idx.get();
                        s_idx.set((i + 1) % sample_pool.len());
                        sample_pool[i]
                    }
                };
                let result = x25519(scalar, basepoint);
                black_box(result);
            })
        }),
    }
}

/// Libsodium Ed25519 signing. C FFI signature primitive.
fn tier1_libsodium_ed25519_sign() -> CryptoTestCase {
    use sodiumoxide::crypto::sign::ed25519;

    CryptoTestCase {
        id: "c_libraries::libsodium::ed25519_sign",
        ecosystem: "C/C++",
        library: "libsodium",
        primitive: "Ed25519",
        is_leaky: false,
        attacker_model: AttackerModel::AdjacentNetwork,
        samples_per_class: 10_000,
        warmup: 1_000,
        collect: Arc::new(|seed, n| {
            sodiumoxide::init().expect("failed to initialize sodiumoxide");
            let (_pk, sk) = ed25519::gen_keypair();

            let fixed_message = [0u8; 64];
            let mut rng = StdRng::seed_from_u64(seed);
            let sample_pool: Vec<[u8; 64]> = (0..n).map(|_| rand_bytes(&mut rng)).collect();

            let s_idx = Cell::new(0usize);
            run_collection(seed, n, 100, |class| {
                let msg = match class {
                    Class::Baseline => fixed_message,
                    Class::Sample => {
                        let i = s_idx.get();
                        s_idx.set((i + 1) % sample_pool.len());
                        sample_pool[i]
                    }
                };
                let signature = ed25519::sign_detached(&msg, &sk);
                black_box(signature.as_ref()[0]);
            })
        }),
    }
}

/// pqcrypto ML-KEM-768 (Kyber) decapsulate. Post-quantum KEM.
///
/// Baseline: repeatedly decapsulates one ciphertext. Sample: decapsulates
/// distinct ciphertexts. Uses `PostQuantumSentinel` per the plan.
fn tier1_pqcrypto_kyber768_decapsulate() -> CryptoTestCase {
    use pqcrypto_kyber::kyber768;
    use pqcrypto_traits::kem::SharedSecret as _;

    CryptoTestCase {
        id: "pqcrypto::kyber::kyber768_decapsulate",
        ecosystem: "Rust",
        library: "pqcrypto",
        primitive: "ML-KEM-768",
        is_leaky: false,
        attacker_model: AttackerModel::PostQuantumSentinel,
        samples_per_class: 10_000,
        warmup: 1_000,
        collect: Arc::new(|seed, n| {
            let (pk, sk) = kyber768::keypair();
            let (_, fixed_ct) = kyber768::encapsulate(&pk);

            let baseline_cts: Vec<_> = (0..n).map(|_| fixed_ct).collect();
            let sample_cts: Vec<_> = (0..n)
                .map(|_| {
                    let (_, ct) = kyber768::encapsulate(&pk);
                    ct
                })
                .collect();

            let b_idx = Cell::new(0usize);
            let s_idx = Cell::new(0usize);
            run_collection(seed, n, 100, |class| {
                let (pool, idx) = match class {
                    Class::Baseline => (&baseline_cts, &b_idx),
                    Class::Sample => (&sample_cts, &s_idx),
                };
                let i = idx.get();
                idx.set((i + 1) % pool.len());
                let ss = kyber768::decapsulate(&pool[i], &sk);
                black_box(ss.as_bytes()[0]);
            })
        }),
    }
}

// =============================================================================
// Tier 2 — Known-leaky detection probe (MARVIN)
// =============================================================================

pub fn tier2() -> Vec<CryptoTestCase> {
    vec![tier2_rustcrypto_rsa_marvin()]
}

/// RustCrypto RSA-1024 PKCS#1 v1.5 decrypt — CVE-2023-49092 (MARVIN).
///
/// Baseline pool: VALID ciphertexts that decrypt successfully.
/// Sample pool:   random bytes of modulus length (invalid PKCS#1 padding).
/// The timing difference between the two classes is the Bleichenbacher
/// oracle the MARVIN paper documents; `rsa` 0.9.9 is vulnerable
/// (num-bigint-dig path).
fn tier2_rustcrypto_rsa_marvin() -> CryptoTestCase {
    use rsa::rand_core::OsRng;
    use rsa::{Pkcs1v15Encrypt, RsaPrivateKey, RsaPublicKey};

    CryptoTestCase {
        id: "rustcrypto::rsa::marvin_rsa1024_pkcs1v15_decrypt",
        ecosystem: "Rust",
        library: "RustCrypto",
        primitive: "RSA-1024-PKCS1v15",
        is_leaky: true,
        // MARVIN is a Bleichenbacher-style oracle; AdjacentNetwork matches
        // the paper's CVE detection evaluation (Table 4) and the MARVIN
        // threat model.
        attacker_model: AttackerModel::AdjacentNetwork,
        samples_per_class: 10_000,
        warmup: 200,
        collect: Arc::new(|seed, n| {
            // Key generation is expensive and non-deterministic; do it once
            // per collect call (reseeded per iteration is overkill — what
            // we want is *independent* ciphertext pools per iteration, which
            // encapsulating with fresh messages already gives us).
            let private_key = RsaPrivateKey::new(&mut OsRng, 1024).expect("RSA keygen failed");
            let public_key = RsaPublicKey::from(&private_key);

            // Pool size 100 matches the investigation test; samples_per_class
            // cycles through it with modular indexing.
            const POOL_SIZE: usize = 100;
            let mut rng = StdRng::seed_from_u64(seed);

            let valid_pool: Vec<Vec<u8>> = (0..POOL_SIZE)
                .map(|_| {
                    let msg: [u8; 32] = rand_bytes(&mut rng);
                    public_key
                        .encrypt(&mut OsRng, Pkcs1v15Encrypt, &msg)
                        .expect("encrypt failed")
                })
                .collect();

            let key_len = 128usize; // 1024 bits
            let invalid_pool: Vec<Vec<u8>> = (0..POOL_SIZE)
                .map(|_| {
                    let mut ct = vec![0u8; key_len];
                    rng.fill_bytes(&mut ct);
                    // Ensure < modulus; bit-twiddle per the MARVIN investigation.
                    ct[0] &= 0x7F;
                    ct
                })
                .collect();

            // Drop original n to ensure we consume exactly `samples_per_class`
            // even if MARVIN's slowness blows past a time budget; the caller
            // hands us n to respect.
            let _ = n;

            let v_idx = Cell::new(0usize);
            let i_idx = Cell::new(0usize);
            run_collection(seed, n, 200, |class| {
                let (pool, idx) = match class {
                    Class::Baseline => (&valid_pool, &v_idx),
                    Class::Sample => (&invalid_pool, &i_idx),
                };
                let i = idx.get();
                idx.set((i + 1) % pool.len());
                let result = private_key.decrypt(Pkcs1v15Encrypt, &pool[i]);
                black_box(result.is_ok());
            })
        }),
    }
}

// =============================================================================
// Tier selection
// =============================================================================

/// Tier identifier used by the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    One,
    Two,
    All,
}

impl Tier {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "1" | "one" => Some(Tier::One),
            "2" | "two" => Some(Tier::Two),
            "all" => Some(Tier::All),
            _ => None,
        }
    }

    pub fn cases(self) -> Vec<CryptoTestCase> {
        match self {
            Tier::One => tier1(),
            Tier::Two => tier2(),
            Tier::All => {
                let mut v = tier1();
                v.extend(tier2());
                v
            }
        }
    }
}
