//! Cryptographic library timing tests
//!
//! Tests real-world cryptographic implementations for timing side channels.
//! Organized by crate/ecosystem:
//! - `rustcrypto`: RustCrypto ecosystem (aes, sha3, blake2, rsa, chacha20poly1305)
//! - `ring`: ring crate (AES-GCM, ChaCha20-Poly1305)
//! - `dalek`: dalek ecosystem (x25519-dalek)
//! - `pqcrypto`: Post-quantum crypto (Kyber, Dilithium, Falcon, SPHINCS+)

#[path = "crypto/dalek.rs"]
mod dalek;
#[path = "crypto/pqcrypto.rs"]
mod pqcrypto;
#[path = "crypto/ring.rs"]
mod ring;
#[path = "crypto/rustcrypto.rs"]
mod rustcrypto;
