//! Post-quantum cryptography timing tests
//!
//! Tests NIST-selected post-quantum algorithms via pqcrypto crate (PQClean bindings):
//! - kyber: ML-KEM (Key Encapsulation Mechanism)
//! - dilithium: ML-DSA (Digital Signatures)
//! - falcon: Falcon (Digital Signatures, NTRU-based)
//! - sphincs: SPHINCS+ (Hash-based Signatures)

#[path = "pqcrypto/dilithium.rs"]
mod dilithium;
#[path = "pqcrypto/falcon.rs"]
mod falcon;
#[path = "pqcrypto/kyber.rs"]
mod kyber;
#[path = "pqcrypto/sphincs.rs"]
mod sphincs;
