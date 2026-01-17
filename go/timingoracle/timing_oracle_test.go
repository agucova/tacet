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
