# Test commands

# Run all workspace tests (default, excludes ignored)
[group('test')]
all:
    cargo test --release --workspace -- --nocapture

# Run all unit tests across workspace
[group('test')]
unit:
    cargo test --release --workspace --lib

# Run known-leaky and known-safe validation tests
[group('test')]
known:
    cargo test --release -p timing-oracle --test known_leaky --test known_safe -- --nocapture

# Quick smoke test (unit + known tests)
[group('test')]
quick: unit known

# Run all tests including ignored (slow, ~30+ min)
[group('test')]
ignored:
    cargo test --release --workspace -- --ignored --nocapture

# Run all crypto timing test suites (fast ones)
[group('test')]
crypto: aes ecc hash aead
    @echo "Note: RSA and PQ tests excluded (slow). Run 'just test rsa' or 'just test pqcrypto' separately."

# Run ALL crypto tests including slow ones
[group('test')]
crypto-all: aes ecc hash aead rsa pqcrypto

# Run AES timing tests
[group('test')]
aes:
    cargo test --release -p timing-oracle --test aes_timing -- --nocapture

# Run ECC/Curve25519 timing tests
[group('test')]
ecc:
    cargo test --release -p timing-oracle --test ecc_timing -- --nocapture

# Run hash (SHA3, BLAKE2) timing tests
[group('test')]
hash:
    cargo test --release -p timing-oracle --test hash_timing -- --nocapture

# Run AEAD (ChaCha20-Poly1305, AES-GCM) timing tests
[group('test')]
aead:
    cargo test --release -p timing-oracle --test aead_timing -- --nocapture

# Run RSA timing tests (slow)
[group('test')]
rsa:
    cargo test --release -p timing-oracle --test rsa_timing -- --nocapture

# Run post-quantum crypto timing tests
[group('test')]
pqcrypto:
    cargo test --release -p timing-oracle --test pqcrypto_timing -- --nocapture

# Run async timing tests
[group('test')]
async:
    cargo test --release -p timing-oracle --test async_timing -- --nocapture

# Run crypto attack pattern tests
[group('test')]
attacks:
    cargo test --release -p timing-oracle --test crypto_attacks -- --nocapture

# Test kperf timer (macOS, requires sudo)
[group('test')]
kperf:
    sudo -E cargo test --release -p timing-oracle --test test_kperf -- --nocapture

# Test perf timer (Linux, requires sudo)
[group('test')]
perf:
    sudo -E cargo test --release -p timing-oracle --test test_perf -- --nocapture
