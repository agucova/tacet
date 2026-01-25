package timingoracle_test

import (
	"fmt"
	"time"

	"github.com/agucova/timing-oracle/crates/timing-oracle-go"
)

// ExampleTest demonstrates basic usage of the timing oracle.
func ExampleTest() {
	// Define a constant-time XOR operation
	secret := make([]byte, 32)
	for i := range secret {
		secret[i] = byte(i * 17)
	}

	result, err := timingoracle.Test(
		timingoracle.NewZeroGenerator(42),
		timingoracle.FuncOperation(func(input []byte) {
			// XOR is constant-time
			for i := range input {
				input[i] ^= secret[i]
			}
		}),
		32, // input size
		timingoracle.WithAttacker(timingoracle.AdjacentNetwork),
		timingoracle.WithTimeBudget(10*time.Second),
	)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Outcome: %s\n", result.Outcome)
	// Output will vary, but XOR should typically pass
}

// ExampleTest_leaky demonstrates detecting a timing leak.
func ExampleTest_leaky() {
	// Define a leaky operation with data-dependent timing
	result, err := timingoracle.Test(
		timingoracle.NewZeroGenerator(42),
		timingoracle.FuncOperation(func(input []byte) {
			// Count non-zero bytes and delay proportionally (NOT constant-time!)
			count := 0
			for _, b := range input {
				if b != 0 {
					count++
				}
			}
			// Artificial delay
			for i := 0; i < count*100; i++ {
				_ = i * i
			}
		}),
		32,
		timingoracle.WithAttacker(timingoracle.AdjacentNetwork),
		timingoracle.WithTimeBudget(10*time.Second),
	)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	if result.Outcome == timingoracle.Fail {
		fmt.Printf("Timing leak detected! Exploitability: %s\n", result.Exploitability)
	}
	// Output will vary based on timing, but should detect the leak
}

// ExampleAnalyze demonstrates analyzing pre-collected timing data.
func ExampleAnalyze() {
	// Pre-collected timing data (in timer ticks)
	baseline := make([]uint64, 1000)
	sample := make([]uint64, 1000)
	for i := range baseline {
		baseline[i] = 100 + uint64(i%10)
		sample[i] = 100 + uint64(i%10) // Same distribution = no leak
	}

	result, err := timingoracle.Analyze(
		baseline,
		sample,
		timingoracle.WithAttacker(timingoracle.AdjacentNetwork),
	)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Outcome: %s, P(leak): %.1f%%\n", result.Outcome, result.LeakProbability*100)
}

// ExampleWithAttacker demonstrates different attacker models.
func ExampleWithAttacker() {
	op := timingoracle.FuncOperation(func(input []byte) {
		// Some operation
		var sum byte
		for _, b := range input {
			sum ^= b
		}
		_ = sum
	})

	// For SGX/containers with shared hardware
	_, _ = timingoracle.Test(
		timingoracle.NewZeroGenerator(1),
		op, 32,
		timingoracle.WithAttacker(timingoracle.SharedHardware), // 0.6ns threshold
		timingoracle.WithTimeBudget(5*time.Second),
	)

	// For LAN services or HTTP/2 APIs
	_, _ = timingoracle.Test(
		timingoracle.NewZeroGenerator(1),
		op, 32,
		timingoracle.WithAttacker(timingoracle.AdjacentNetwork), // 100ns threshold
		timingoracle.WithTimeBudget(5*time.Second),
	)

	// For internet-exposed services
	_, _ = timingoracle.Test(
		timingoracle.NewZeroGenerator(1),
		op, 32,
		timingoracle.WithAttacker(timingoracle.RemoteNetwork), // 50us threshold
		timingoracle.WithTimeBudget(5*time.Second),
	)

	fmt.Println("Tested with multiple attacker models")
	// Output: Tested with multiple attacker models
}
