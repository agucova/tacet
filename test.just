# Test commands
#
# Output verbosity (default is debug, set in .cargo/config.toml):
#   just test quiet ...   - minimal output (no diagnostics)
#   just test verbose ... - verbose output (preflight + diagnostics)
#   (default)             - debug output (verbose + extended info)

# Run all workspace tests (default, excludes ignored)
[group('test')]
all:
    cargo nextest run --release --workspace --no-capture

# Run tests with minimal output (no diagnostics)
# Example: just test quiet -p tacet --test known_safe
# With ignored: sudo -E just test quiet --test rsa_timing -E "'test(name)'" --run-ignored all
[group('test')]
quiet *args:
    TIMING_ORACLE_DEBUG=0 TIMING_ORACLE_VERBOSE=0 cargo nextest run --release --no-capture {{ if args == "" { "--workspace" } else { args } }}

# Run tests with verbose output (preflight + diagnostics, no extended debug)
# Example: just test verbose -p tacet --test known_safe
# With ignored: sudo -E just test verbose --test rsa_timing -E "'test(name)'" --run-ignored all
[group('test')]
verbose *args:
    TIMING_ORACLE_DEBUG=0 TIMING_ORACLE_VERBOSE=1 cargo nextest run --release --no-capture {{ if args == "" { "--workspace" } else { args } }}

# Run all unit tests across workspace
[group('test')]
unit:
    cargo nextest run --release --workspace --lib

# Run known-leaky and known-safe validation tests
[group('test')]
known:
    cargo nextest run --release -p tacet --test known_leaky --test known_safe --no-capture

# Quick smoke test (unit + known tests)
[group('test')]
quick: unit known

# Run all tests including ignored (slow, ~30+ min)
[group('test')]
ignored:
    cargo nextest run --release --workspace --run-ignored all --no-capture

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
    cargo nextest run --release -p tacet --test aes_timing --no-capture

# Run ECC/Curve25519 timing tests
[group('test')]
ecc:
    cargo nextest run --release -p tacet --test ecc_timing --no-capture

# Run hash (SHA3, BLAKE2) timing tests
[group('test')]
hash:
    cargo nextest run --release -p tacet --test hash_timing --no-capture

# Run AEAD (ChaCha20-Poly1305, AES-GCM) timing tests
[group('test')]
aead:
    cargo nextest run --release -p tacet --test aead_timing --no-capture

# Run RSA timing tests (slow)
[group('test')]
rsa:
    cargo nextest run --release -p tacet --test rsa_timing --no-capture

# Run post-quantum crypto timing tests
[group('test')]
pqcrypto:
    cargo nextest run --release -p tacet --test pqcrypto_timing --no-capture

# Run async timing tests
[group('test')]
async:
    cargo nextest run --release -p tacet --test async_timing --no-capture

# Run crypto attack pattern tests
[group('test')]
attacks:
    cargo nextest run --release -p tacet --test crypto_attacks --no-capture

# Test kperf timer (macOS, requires sudo)
[group('test')]
kperf:
    sudo -E cargo nextest run --release -p tacet --test test_kperf --no-capture

# Test perf timer (Linux, requires sudo)
[group('test')]
perf:
    sudo -E cargo nextest run --release -p tacet --test test_perf --no-capture
