// Example: Simple timing side-channel test
//
// This example demonstrates how to use timingoracle to test an operation
// for timing side channels.
//
// Build the Rust library first:
//   cargo build -p timing-oracle-uniffi --release
//
// Then run this example:
//   go run ./examples/simple
package main

import (
	"fmt"
	"log"
	"time"

	"github.com/agucova/timing-oracle/crates/timing-oracle-go"
)

// XOR operation - should be constant-time
func xorBytes(a, b []byte) []byte {
	result := make([]byte, len(a))
	for i := range a {
		result[i] = a[i] ^ b[i]
	}
	return result
}

// Early exit comparison - NOT constant-time (leaky)
// This exits as soon as it finds a mismatch, leaking information
// about where the first difference is.
func earlyExitCompare(a, b []byte) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Artificial delay operation - clearly NOT constant-time
// Adds a data-dependent delay to make the timing difference detectable
// even with coarse timers (like ARM64's ~42ns cntvct_el0).
func artificialDelay(input []byte) {
	// Count non-zero bytes and delay proportionally
	count := 0
	for _, b := range input {
		if b != 0 {
			count++
		}
	}
	// Small busy-wait loop scaled by count
	// This creates a timing difference of ~microseconds
	for i := 0; i < count*100; i++ {
		_ = i * i
	}
}

func main() {
	fmt.Println("Timing Oracle for Go - Example")
	fmt.Println("==============================")
	fmt.Printf("Timer: %s (%.2f ns resolution)\n",
		timingoracle.TimerName(),
		timingoracle.TimerResolutionNs())
	fmt.Println()

	// Test 1: XOR operation (should pass - constant-time)
	fmt.Println("Test 1: XOR operation (constant-time)")
	fmt.Println("--------------------------------------")

	secret := make([]byte, 32)
	for i := range secret {
		secret[i] = byte(i * 17)
	}

	xorResult, err := timingoracle.Test(
		timingoracle.NewZeroGenerator(42),
		timingoracle.FuncOperation(func(input []byte) {
			_ = xorBytes(input, secret)
		}),
		32,
		timingoracle.WithAttacker(timingoracle.AdjacentNetwork),
		timingoracle.WithTimeBudget(10*time.Second),
		timingoracle.WithMaxSamples(50_000),
	)
	if err != nil {
		log.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Result: %s\n", xorResult)
		fmt.Printf("  Outcome: %s\n", xorResult.Outcome)
		fmt.Printf("  P(leak): %.2f%%\n", xorResult.LeakProbability*100)
		fmt.Printf("  Samples: %d\n", xorResult.SamplesUsed)
		fmt.Printf("  Time: %v\n", xorResult.ElapsedTime)
	}
	fmt.Println()

	// Test 2: Artificial delay operation (should FAIL - clearly not constant-time)
	// This test uses a data-dependent delay that's large enough to detect
	// even with coarse timers like ARM64's cntvct_el0 (~42ns resolution).
	fmt.Println("Test 2: Artificial delay operation (NOT constant-time)")
	fmt.Println("------------------------------------------------------")

	delayResult, err := timingoracle.Test(
		timingoracle.NewZeroGenerator(42),
		timingoracle.FuncOperation(func(input []byte) {
			artificialDelay(input)
		}),
		32,
		timingoracle.WithAttacker(timingoracle.AdjacentNetwork),
		timingoracle.WithTimeBudget(10*time.Second),
		timingoracle.WithMaxSamples(50_000),
	)
	if err != nil {
		log.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Result: %s\n", delayResult)
		fmt.Printf("  Outcome: %s\n", delayResult.Outcome)
		fmt.Printf("  P(leak): %.2f%%\n", delayResult.LeakProbability*100)
		if delayResult.Outcome == timingoracle.Fail {
			fmt.Printf("  Exploitability: %s\n", delayResult.Exploitability)
			fmt.Printf("  Effect: %.2f ns (shift) + %.2f ns (tail)\n",
				delayResult.Effect.ShiftNs, delayResult.Effect.TailNs)
		}
		fmt.Printf("  Samples: %d\n", delayResult.SamplesUsed)
		fmt.Printf("  Time: %v\n", delayResult.ElapsedTime)
	}
}
