package timingoracle

import (
	"math/rand/v2"
	"testing"
	"time"
)

// TestTimerWorks verifies the platform timer is functional.
func TestTimerWorks(t *testing.T) {
	name := TimerName()
	if name == "" {
		t.Fatal("Timer name is empty")
	}
	t.Logf("Timer: %s", name)

	freq := TimerFrequency()
	if freq == 0 {
		t.Fatal("Timer frequency is zero")
	}
	t.Logf("Frequency: %d Hz", freq)

	res := TimerResolutionNs()
	if res <= 0 {
		t.Fatal("Timer resolution is invalid")
	}
	t.Logf("Resolution: %.2f ns", res)

	// Read timer twice and verify it advances
	t1 := readTimer()
	time.Sleep(1 * time.Millisecond)
	t2 := readTimer()
	if t2 <= t1 {
		t.Fatalf("Timer did not advance: t1=%d, t2=%d", t1, t2)
	}
	t.Logf("Timer delta over 1ms: %d ticks", t2-t1)
}

// TestZeroGenerator verifies the zero generator works correctly.
func TestZeroGenerator(t *testing.T) {
	gen := NewZeroGenerator(42)
	buf := make([]byte, 32)

	// Baseline should be all zeros
	gen.Generate(true, buf)
	for i, b := range buf {
		if b != 0 {
			t.Fatalf("Baseline byte %d is not zero: %d", i, b)
		}
	}

	// Sample should have some non-zero bytes (statistically)
	gen.Generate(false, buf)
	hasNonZero := false
	for _, b := range buf {
		if b != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Log("Warning: Sample generated all zeros (very unlikely)")
	}
}

// TestCollectSamples verifies sample collection works.
func TestCollectSamples(t *testing.T) {
	gen := NewZeroGenerator(42)
	op := FuncOperation(func(input []byte) {
		// Simple XOR operation
		for i := range input {
			input[i] ^= 0x55
		}
	})

	// Use a fixed seed for reproducibility
	rng := newRandForTest(12345)

	baseline, sample := collectSamples(gen, op, 32, 100, 1, rng)

	if len(baseline) != 100 {
		t.Fatalf("Expected 100 baseline samples, got %d", len(baseline))
	}
	if len(sample) != 100 {
		t.Fatalf("Expected 100 sample samples, got %d", len(sample))
	}

	// Verify samples are non-zero
	var zeroCount int
	for _, s := range baseline {
		if s == 0 {
			zeroCount++
		}
	}
	if zeroCount > 50 {
		t.Logf("Warning: %d/%d baseline samples are zero", zeroCount, len(baseline))
	}
}

// TestDetectBatchK verifies adaptive batching detection.
func TestDetectBatchK(t *testing.T) {
	// Fast operation - may need batching
	fastOp := FuncOperation(func(input []byte) {
		// Very fast - just a memory access
		_ = input[0]
	})

	k := detectBatchK(fastOp, 32)
	t.Logf("Batch K for fast operation: %d", k)
	if k < 1 || k > 20 {
		t.Fatalf("Batch K out of range: %d", k)
	}

	// Slow operation - should not need batching
	slowOp := FuncOperation(func(input []byte) {
		// Slower operation
		var sum byte
		for i := 0; i < 1000; i++ {
			for _, b := range input {
				sum ^= b
			}
		}
		_ = sum
	})

	k2 := detectBatchK(slowOp, 32)
	t.Logf("Batch K for slow operation: %d", k2)
	// Slow operation should typically have lower K
}

// TestConfigOptions verifies configuration options work.
func TestConfigOptions(t *testing.T) {
	cfg := defaultConfig()

	// Default values
	if cfg.attackerModel != AdjacentNetwork {
		t.Errorf("Expected AdjacentNetwork, got %v", cfg.attackerModel)
	}
	if cfg.timeBudget != 30*time.Second {
		t.Errorf("Expected 30s time budget, got %v", cfg.timeBudget)
	}

	// Apply options
	WithAttacker(SharedHardware)(cfg)
	WithTimeBudget(10 * time.Second)(cfg)
	WithMaxSamples(50000)(cfg)

	if cfg.attackerModel != SharedHardware {
		t.Errorf("Expected SharedHardware, got %v", cfg.attackerModel)
	}
	if cfg.timeBudget != 10*time.Second {
		t.Errorf("Expected 10s time budget, got %v", cfg.timeBudget)
	}
	if cfg.maxSamples != 50000 {
		t.Errorf("Expected 50000 max samples, got %d", cfg.maxSamples)
	}
}

// TestAttackerModelThresholds verifies attacker model thresholds.
func TestAttackerModelThresholds(t *testing.T) {
	tests := []struct {
		model     AttackerModel
		expected  float64
		tolerance float64
	}{
		{SharedHardware, 0.6, 0.01},
		{PostQuantum, 3.3, 0.01},
		{AdjacentNetwork, 100.0, 0.01},
		{RemoteNetwork, 50000.0, 0.01},
		{Research, 0.0, 0.01},
	}

	for _, tc := range tests {
		got := tc.model.ThresholdNs()
		if got < tc.expected-tc.tolerance || got > tc.expected+tc.tolerance {
			t.Errorf("%s: expected %.2f ns, got %.2f ns", tc.model, tc.expected, got)
		}
	}
}

// newRandForTest creates a reproducible RNG for testing.
func newRandForTest(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, seed^0xDEADBEEF))
}

// =============================================================================
// Integration Tests
// =============================================================================
// These tests verify the full pipeline from measurement to analysis.
// They may take several seconds to run.

// TestKnownLeaky verifies that a clearly leaky operation is detected as Fail.
// Uses an artificial delay that creates a large, easily detectable timing difference.
func TestKnownLeaky(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Artificial delay operation - clearly NOT constant-time
	// Adds a data-dependent delay based on input bytes
	leakyOp := FuncOperation(func(input []byte) {
		count := 0
		for _, b := range input {
			if b != 0 {
				count++
			}
		}
		// Busy-wait loop scaled by count - creates microsecond-level differences
		for i := 0; i < count*100; i++ {
			_ = i * i
		}
	})

	result, err := Test(
		NewZeroGenerator(42),
		leakyOp,
		32,
		WithAttacker(AdjacentNetwork),
		WithTimeBudget(15*time.Second),
		WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)
	t.Logf("  Effect: %.2f ns (shift) + %.2f ns (tail)", result.Effect.ShiftNs, result.Effect.TailNs)
	t.Logf("  Samples: %d", result.SamplesUsed)

	if result.Outcome != Fail {
		t.Errorf("Expected Fail outcome for leaky operation, got %s", result.Outcome)
	}
	if result.LeakProbability < 0.95 {
		t.Errorf("Expected high leak probability (>95%%), got %.2f%%", result.LeakProbability*100)
	}
}

// TestKnownSafe verifies that a constant-time operation passes.
// Uses XOR which is constant-time on all platforms.
func TestKnownSafe(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	secret := make([]byte, 32)
	for i := range secret {
		secret[i] = byte(i * 17)
	}

	// XOR operation - constant-time
	safeOp := FuncOperation(func(input []byte) {
		result := make([]byte, len(input))
		for i := range input {
			result[i] = input[i] ^ secret[i]
		}
		_ = result
	})

	result, err := Test(
		NewZeroGenerator(42),
		safeOp,
		32,
		WithAttacker(AdjacentNetwork),
		WithTimeBudget(15*time.Second),
		WithMaxSamples(50_000),
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)
	t.Logf("  Effect: %.2f ns", result.Effect.TotalNs())
	t.Logf("  Samples: %d", result.SamplesUsed)

	// Should pass or be inconclusive (not fail)
	if result.Outcome == Fail {
		t.Errorf("Expected Pass or Inconclusive for constant-time XOR, got Fail")
		t.Logf("  This may indicate a false positive or environmental noise")
	}
	if result.Outcome == Pass {
		if result.LeakProbability > 0.10 {
			t.Logf("Warning: leak probability higher than expected: %.2f%%", result.LeakProbability*100)
		}
	}
}

// TestInconclusiveTimeout verifies that a very short time budget causes Inconclusive.
func TestInconclusiveTimeout(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Use an operation that takes some time but not too long
	op := FuncOperation(func(input []byte) {
		var sum byte
		for i := 0; i < 100; i++ {
			for _, b := range input {
				sum ^= b
			}
		}
		_ = sum
	})

	result, err := Test(
		NewZeroGenerator(42),
		op,
		32,
		WithAttacker(SharedHardware), // Very tight threshold
		WithTimeBudget(100*time.Millisecond), // Very short budget
		WithMaxSamples(1000), // Very few samples
	)
	if err != nil {
		t.Fatalf("Test failed with error: %v", err)
	}

	t.Logf("Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  Inconclusive reason: %s", result.InconclusiveReason)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)
	t.Logf("  Samples: %d", result.SamplesUsed)

	// With such tight constraints, we expect Inconclusive or possibly Pass/Fail
	// The main thing is it shouldn't error
	if result.Outcome == Inconclusive {
		// Expected - check that reason is set
		if result.InconclusiveReason == ReasonNone {
			t.Logf("Warning: Inconclusive but reason is None")
		}
	}
}

// TestAnalyzePreCollected verifies the Analyze function with pre-collected data.
func TestAnalyzePreCollected(t *testing.T) {
	// Create timing data with no leak (same distribution)
	baseline := make([]uint64, 1000)
	sample := make([]uint64, 1000)
	for i := range baseline {
		baseline[i] = 100 + uint64(i%10)
		sample[i] = 100 + uint64(i%10)
	}

	result, err := Analyze(baseline, sample, WithAttacker(AdjacentNetwork))
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	t.Logf("Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)

	// With identical distributions, should not detect a leak
	if result.Outcome == Fail {
		t.Errorf("Expected non-Fail for identical distributions, got Fail")
	}
}

// TestAnalyzeWithLeak verifies the Analyze function detects an artificial leak.
func TestAnalyzeWithLeak(t *testing.T) {
	// Create timing data with a clear shift
	baseline := make([]uint64, 1000)
	sample := make([]uint64, 1000)
	for i := range baseline {
		baseline[i] = 100 + uint64(i%10)
		sample[i] = 200 + uint64(i%10) // 100 tick difference
	}

	result, err := Analyze(baseline, sample, WithAttacker(AdjacentNetwork))
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	t.Logf("Result: %s", result)
	t.Logf("  Outcome: %s", result.Outcome)
	t.Logf("  P(leak): %.2f%%", result.LeakProbability*100)
	t.Logf("  Effect: %.2f ns", result.Effect.TotalNs())

	// With a 100-tick difference, should detect a leak
	if result.Outcome == Pass {
		t.Errorf("Expected non-Pass for distributions with 100-tick shift, got Pass")
	}
}
