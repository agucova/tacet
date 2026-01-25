//! ring crate timing tests
//!
//! Tests implementations from the ring crate:
//! - aes_gcm: AES-256-GCM AEAD
//! - chacha20poly1305: ChaCha20-Poly1305 AEAD

#[path = "ring/aes_gcm.rs"]
mod aes_gcm;
#[path = "ring/chacha20poly1305.rs"]
mod chacha20poly1305;
