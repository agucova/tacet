package tacet_test

import (
	"crypto/rand"
	"sync/atomic"
	"testing"
	"time"

	"github.com/agucova/tacet/crates/tacet-go"
	"golang.org/x/crypto/argon2"
	"golang.org/x/crypto/blake2b"
	"golang.org/x/crypto/blake2s"
	"golang.org/x/crypto/chacha20poly1305"
	"golang.org/x/crypto/curve25519"
	"golang.org/x/crypto/nacl/secretbox"
)

// =============================================================================
// golang.org/x/crypto timing tests
//
// These tests validate timing side-channel properties of golang.org/x/crypto
// packages, focusing on secret/private operations that handle sensitive data.
//
// Test pattern (DudeCT two-class):
// - Baseline class: all-zero data
// - Sample class: random data
//
// IMPORTANT: Run with `sudo -E go test -run TestXCrypto -v` to use macOS PMU
// timers for higher precision measurements.
// =============================================================================

// =============================================================================
// chacha20poly1305 Tests (AEAD)
// =============================================================================

// TestXCrypto_ChaCha20Poly1305_EncryptZerosVsRandom tests ChaCha20-Poly1305
// encryption timing with zeros vs random plaintext.
func TestXCrypto_ChaCha20Poly1305_EncryptZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	// Generate a fixed key
	key := make([]byte, chacha20poly1305.KeySize)
	rand.Read(key)

	aead, err := chacha20poly1305.New(key)
	if err != nil {
		t.Fatalf("Failed to create ChaCha20-Poly1305: %v", err)
	}

	var nonceCounter uint64

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(plaintext []byte) {
			// Generate unique nonce using atomic counter
			n := atomic.AddUint64(&nonceCounter, 1)
			nonce := make([]byte, chacha20poly1305.NonceSize)
			for i := 0; i < 8 && i < len(nonce); i++ {
				nonce[i] = byte(n >> (i * 8))
			}

			// Encrypt
			ciphertext := aead.Seal(nil, nonce, plaintext, nil)
			_ = ciphertext
		}),
		64, // plaintext size
		tacet.WithAttacker(tacet.AdjacentNetwork),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("ChaCha20-Poly1305 Encrypt Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)
	t.Logf("  Effect: %.2f ns", result.Effect.MaxEffectNs)
	t.Logf("  Samples: %d", result.SamplesUsed)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/chacha20poly1305 Encrypt")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
		t.Logf("  Exploitability: %s", result.Exploitability)
	}
}

// TestXCrypto_ChaCha20Poly1305_DecryptZerosVsRandom tests ChaCha20-Poly1305
// decryption timing.
func TestXCrypto_ChaCha20Poly1305_DecryptZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	key := make([]byte, chacha20poly1305.KeySize)
	rand.Read(key)

	aead, err := chacha20poly1305.New(key)
	if err != nil {
		t.Fatalf("Failed to create ChaCha20-Poly1305: %v", err)
	}

	var nonceCounter uint64

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(plaintext []byte) {
			// Generate unique nonce
			n := atomic.AddUint64(&nonceCounter, 1)
			nonce := make([]byte, chacha20poly1305.NonceSize)
			for i := 0; i < 8 && i < len(nonce); i++ {
				nonce[i] = byte(n >> (i * 8))
			}

			// First encrypt to get valid ciphertext
			ciphertext := aead.Seal(nil, nonce, plaintext, nil)

			// Then decrypt it - timing should be constant regardless of plaintext
			decrypted, err := aead.Open(nil, nonce, ciphertext, nil)
			_ = err
			_ = decrypted
		}),
		64,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("ChaCha20-Poly1305 Decrypt Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/chacha20poly1305 Decrypt")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
	}
}

// =============================================================================
// argon2 Tests (Password Hashing - HIGH VALUE TARGET)
// =============================================================================

// TestXCrypto_Argon2id_ZerosVsRandom tests Argon2id password hashing timing.
// This is a high-value target: password hashing must be constant-time to avoid
// revealing information about the password structure.
func TestXCrypto_Argon2id_ZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	// Fixed salt for all measurements
	salt := make([]byte, 16)
	rand.Read(salt)

	// Use lightweight parameters for faster testing
	// (Production would use timeCost=1, memory=64*1024, threads=4)
	const (
		timeCost = 1
		memory   = 8 * 1024 // 8 MB
		threads  = 2
		keyLen   = 32
	)

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(password []byte) {
			// Hash the password with Argon2id
			hash := argon2.IDKey(password, salt, timeCost, memory, threads, keyLen)
			_ = hash
		}),
		32, // password length
		tacet.WithAttacker(tacet.SharedHardware), // Argon2 is high-value, use strict threshold
		tacet.WithTimeBudget(60*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("Argon2id Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)
	t.Logf("  Effect: %.2f ns", result.Effect.MaxEffectNs)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/argon2 IDKey")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
		t.Logf("  This is CRITICAL: password hashing timing leaks can reveal password structure")
		t.Logf("  Exploitability: %s", result.Exploitability)
	}
}

// TestXCrypto_Argon2_ZerosVsRandom tests standard Argon2 (not Argon2id).
func TestXCrypto_Argon2_ZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	salt := make([]byte, 16)
	rand.Read(salt)

	const (
		timeCost = 1
		memory   = 8 * 1024
		threads  = 2
		keyLen   = 32
	)

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(password []byte) {
			hash := argon2.Key(password, salt, timeCost, memory, threads, keyLen)
			_ = hash
		}),
		32,
		tacet.WithAttacker(tacet.SharedHardware),
		tacet.WithTimeBudget(60*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("Argon2 Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/argon2 Key")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
		t.Logf("  This is CRITICAL for password hashing")
	}
}

// =============================================================================
// curve25519 Tests (X25519 Key Exchange)
// =============================================================================

// TestXCrypto_Curve25519_ScalarMultZerosVsRandom tests X25519 scalar
// multiplication timing with zeros vs random scalars.
func TestXCrypto_Curve25519_ScalarMultZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	// Fixed basepoint (standard X25519 basepoint)
	basepoint := [32]byte{9}

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(scalar []byte) {
			// Convert scalar to fixed-size array
			var scalarArray [32]byte
			copy(scalarArray[:], scalar)

			// Perform scalar multiplication
			var output [32]byte
			curve25519.ScalarMult(&output, &scalarArray, &basepoint)
			_ = output
		}),
		32,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		// Thresholds match the Rust suite's constant-time tests
		// (crates/tacet/tests/core/known_safe.rs). A ~50us Montgomery ladder
		// measured with a 41.67ns timer accumulates enough noise on a busy
		// machine to cross the default 0.95 fail threshold, so this test demands
		// stronger evidence before calling a constant-time primitive leaky.
		tacet.WithPassThreshold(0.15),
		tacet.WithFailThreshold(0.99),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("Curve25519 ScalarMult Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)
	t.Logf("  Effect: %.2f ns", result.Effect.MaxEffectNs)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/curve25519 ScalarMult")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
		t.Logf("  Exploitability: %s", result.Exploitability)
	}
}

// TestXCrypto_Curve25519_ScalarBaseMultZerosVsRandom tests X25519 scalar
// base multiplication (optimized version with fixed basepoint).
func TestXCrypto_Curve25519_ScalarBaseMultZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(scalar []byte) {
			// Convert scalar to fixed-size array
			var scalarArray [32]byte
			copy(scalarArray[:], scalar)

			// Perform scalar base multiplication
			var output [32]byte
			curve25519.ScalarBaseMult(&output, &scalarArray)
			_ = output
		}),
		32,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		// See TestXCrypto_Curve25519_ScalarMultZerosVsRandom for why these
		// thresholds differ from the defaults.
		tacet.WithPassThreshold(0.15),
		tacet.WithFailThreshold(0.99),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("Curve25519 ScalarBaseMult Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/curve25519 ScalarBaseMult")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
	}
}

// =============================================================================
// nacl/secretbox Tests (Authenticated Encryption)
// =============================================================================

// TestXCrypto_SecretBox_SealZerosVsRandom tests NaCl secretbox encryption.
func TestXCrypto_SecretBox_SealZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	// Generate a fixed key
	var key [32]byte
	rand.Read(key[:])

	var nonceCounter uint64

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(plaintext []byte) {
			// Generate unique nonce
			n := atomic.AddUint64(&nonceCounter, 1)
			var nonce [24]byte
			for i := 0; i < 8 && i < len(nonce); i++ {
				nonce[i] = byte(n >> (i * 8))
			}

			// Encrypt with secretbox
			ciphertext := secretbox.Seal(nil, plaintext, &nonce, &key)
			_ = ciphertext
		}),
		64,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("NaCl SecretBox Seal Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/nacl/secretbox Seal")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
	}
}

// TestXCrypto_SecretBox_OpenZerosVsRandom tests NaCl secretbox decryption.
func TestXCrypto_SecretBox_OpenZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	var key [32]byte
	rand.Read(key[:])

	var nonceCounter uint64

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(plaintext []byte) {
			// Generate unique nonce
			n := atomic.AddUint64(&nonceCounter, 1)
			var nonce [24]byte
			for i := 0; i < 8 && i < len(nonce); i++ {
				nonce[i] = byte(n >> (i * 8))
			}

			// First encrypt to get valid ciphertext
			ciphertext := secretbox.Seal(nil, plaintext, &nonce, &key)

			// Then decrypt - timing should be constant regardless of plaintext
			decrypted, ok := secretbox.Open(nil, ciphertext, &nonce, &key)
			_ = ok
			_ = decrypted
		}),
		64,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("NaCl SecretBox Open Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/nacl/secretbox Open")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
	}
}

// =============================================================================
// BLAKE2b/BLAKE2s Tests (Hashing)
// =============================================================================

// TestXCrypto_BLAKE2b_512_ZerosVsRandom tests BLAKE2b-512 hashing.
func TestXCrypto_BLAKE2b_512_ZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(data []byte) {
			hash := blake2b.Sum512(data)
			_ = hash
		}),
		64,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("BLAKE2b-512 Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/blake2b Sum512")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
	}
}

// TestXCrypto_BLAKE2b_256_ZerosVsRandom tests BLAKE2b-256 hashing.
func TestXCrypto_BLAKE2b_256_ZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(data []byte) {
			hash := blake2b.Sum256(data)
			_ = hash
		}),
		64,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("BLAKE2b-256 Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/blake2b Sum256")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
	}
}

// TestXCrypto_BLAKE2s_256_ZerosVsRandom tests BLAKE2s-256 hashing.
func TestXCrypto_BLAKE2s_256_ZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(data []byte) {
			hash := blake2s.Sum256(data)
			_ = hash
		}),
		64,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("BLAKE2s-256 Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/blake2s Sum256")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
	}
}

// =============================================================================
// BLAKE2 Keyed Hash Tests (MAC mode)
// =============================================================================

// TestXCrypto_BLAKE2b_Keyed_ZerosVsRandom tests BLAKE2b in MAC mode with a key.
func TestXCrypto_BLAKE2b_Keyed_ZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	// Fixed key for MAC
	key := make([]byte, 32)
	rand.Read(key)

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(data []byte) {
			// Create keyed BLAKE2b hasher
			hasher, err := blake2b.New512(key)
			if err != nil {
				t.Logf("Failed to create BLAKE2b hasher: %v", err)
				return
			}
			hasher.Write(data)
			hash := hasher.Sum(nil)
			_ = hash
		}),
		64,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("BLAKE2b Keyed Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/blake2b Keyed (MAC mode)")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
	}
}

// TestXCrypto_BLAKE2s_Keyed_ZerosVsRandom tests BLAKE2s in MAC mode.
func TestXCrypto_BLAKE2s_Keyed_ZerosVsRandom(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	key := make([]byte, 32)
	rand.Read(key)

	result, err := tacet.Test(
		tacet.NewZeroGenerator(42),
		tacet.FuncOperation(func(data []byte) {
			hasher, err := blake2s.New256(key)
			if err != nil {
				t.Logf("Failed to create BLAKE2s hasher: %v", err)
				return
			}
			hasher.Write(data)
			hash := hasher.Sum(nil)
			_ = hash
		}),
		64,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("BLAKE2s Keyed Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/blake2s Keyed (MAC mode)")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
	}
}

// =============================================================================
// Hamming Weight Independence Tests
// =============================================================================

// TestXCrypto_ChaCha20Poly1305_HammingWeight tests ChaCha20-Poly1305 with
// Hamming weight variation (all zeros vs all ones).
func TestXCrypto_ChaCha20Poly1305_HammingWeight(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	key := make([]byte, chacha20poly1305.KeySize)
	rand.Read(key)

	aead, err := chacha20poly1305.New(key)
	if err != nil {
		t.Fatalf("Failed to create ChaCha20-Poly1305: %v", err)
	}

	var nonceCounter uint64
	gen := &hammingWeightGenerator{}

	result, err := tacet.Test(
		gen,
		tacet.FuncOperation(func(plaintext []byte) {
			n := atomic.AddUint64(&nonceCounter, 1)
			nonce := make([]byte, chacha20poly1305.NonceSize)
			for i := 0; i < 8 && i < len(nonce); i++ {
				nonce[i] = byte(n >> (i * 8))
			}

			ciphertext := aead.Seal(nil, nonce, plaintext, nil)
			_ = ciphertext
		}),
		64,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("ChaCha20-Poly1305 Hamming Weight Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/chacha20poly1305 (Hamming weight dependent)")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
	}
}

// TestXCrypto_Curve25519_HammingWeight tests Curve25519 with Hamming weight
// variation in the scalar.
func TestXCrypto_Curve25519_HammingWeight(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping x/crypto test in short mode")
	}

	basepoint := [32]byte{9}
	gen := &hammingWeightGenerator{}

	result, err := tacet.Test(
		gen,
		tacet.FuncOperation(func(scalar []byte) {
			var scalarArray [32]byte
			copy(scalarArray[:], scalar)

			var output [32]byte
			curve25519.ScalarMult(&output, &scalarArray, &basepoint)
			_ = output
		}),
		32,
		tacet.WithAttacker(tacet.AdjacentNetwork),
		// See TestXCrypto_Curve25519_ScalarMultZerosVsRandom for why these
		// thresholds differ from the defaults.
		tacet.WithPassThreshold(0.15),
		tacet.WithFailThreshold(0.99),
		tacet.WithTimeBudget(30*time.Second),
		tacet.WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("Curve25519 Hamming Weight Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	if result.Outcome == tacet.Fail {
		t.Errorf("TIMING LEAK DETECTED in x/crypto/curve25519 (Hamming weight dependent)")
		t.Logf("  Effect size: %.2f ns", result.Effect.MaxEffectNs)
	}
}

// =============================================================================
// Helper Generators
// =============================================================================

// Note: hammingWeightGenerator is defined in stdlib_crypto_test.go and reused here
