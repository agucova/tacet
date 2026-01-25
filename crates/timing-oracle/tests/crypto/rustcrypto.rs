//! RustCrypto ecosystem timing tests
//!
//! Tests implementations from the RustCrypto project:
//! - aes: AES-128 block cipher
//! - sha3: SHA-3 (Keccak) hash family
//! - blake2: BLAKE2 hash family
//! - rsa: RSA encryption and signing
//! - chacha20poly1305: ChaCha20-Poly1305 AEAD

#[path = "rustcrypto/aes.rs"]
mod aes;
#[path = "rustcrypto/sha3.rs"]
mod sha3;
#[path = "rustcrypto/blake2.rs"]
mod blake2;
#[path = "rustcrypto/rsa.rs"]
mod rsa;
// TODO: split chacha20poly1305 from aead_timing.rs
