package timingoracle

import (
	"math/rand/v2"
)

// Generator is an interface for generating input data for timing tests.
// Implementations should generate different data for baseline vs sample class.
type Generator interface {
	// Generate fills the output buffer with input data.
	// If isBaseline is true, generate baseline class data (typically all zeros).
	// If isBaseline is false, generate sample class data (typically random).
	Generate(isBaseline bool, output []byte)
}

// Operation is an interface for the operation to be timed.
type Operation interface {
	// Execute runs the operation on the given input.
	// This is the code being tested for timing side channels.
	Execute(input []byte)
}

// collectSamples collects interleaved timing measurements.
// No FFI calls during this - pure Go for minimal overhead.
//
// Parameters:
//   - gen: Generator for creating input data
//   - op: Operation to time
//   - inputSize: Size of input buffer in bytes
//   - count: Number of samples per class to collect
//   - batchK: Number of iterations per measurement (for adaptive batching)
//   - rng: Random number generator for schedule
//
// Returns baseline and sample timing arrays (in timer ticks).
func collectSamples(
	gen Generator,
	op Operation,
	inputSize int,
	count int,
	batchK int,
	rng *rand.Rand,
) (baseline, sample []uint64) {
	baseline = make([]uint64, count)
	sample = make([]uint64, count)
	input := make([]byte, inputSize)

	// Generate interleaved schedule using Fisher-Yates shuffle
	schedule := generateSchedule(count, rng)

	baselineIdx := 0
	sampleIdx := 0

	for _, isBaseline := range schedule {
		// Generate input outside timed region
		gen.Generate(isBaseline, input)

		// Timed region - pure Go, no FFI
		var elapsed uint64
		if batchK == 1 {
			// Fast path: single iteration
			start := readTimer()
			op.Execute(input)
			end := readTimer()
			elapsed = end - start
		} else {
			// Batched: multiple iterations for coarse timers
			start := readTimer()
			for k := 0; k < batchK; k++ {
				op.Execute(input)
			}
			end := readTimer()
			// Store total time (don't divide - Rust analysis expects raw ticks)
			elapsed = end - start
		}

		if isBaseline {
			baseline[baselineIdx] = elapsed
			baselineIdx++
		} else {
			sample[sampleIdx] = elapsed
			sampleIdx++
		}
	}

	return baseline[:baselineIdx], sample[:sampleIdx]
}

// generateSchedule creates a random interleaved schedule of baseline and sample measurements.
// Uses Fisher-Yates shuffle for uniform randomness.
func generateSchedule(countPerClass int, rng *rand.Rand) []bool {
	total := countPerClass * 2
	schedule := make([]bool, total)

	// First half: baseline (true), second half: sample (false)
	for i := 0; i < countPerClass; i++ {
		schedule[i] = true
	}
	for i := countPerClass; i < total; i++ {
		schedule[i] = false
	}

	// Fisher-Yates shuffle
	for i := total - 1; i > 0; i-- {
		j := rng.IntN(i + 1)
		schedule[i], schedule[j] = schedule[j], schedule[i]
	}

	return schedule
}

// detectBatchK determines the optimal number of iterations per measurement.
// This compensates for coarse timer resolution (e.g., ARM64's ~42ns cntvct_el0).
//
// Returns 1 for fine-grained timers (x86_64 rdtsc) or when operation is slow enough.
// Returns K > 1 when timer resolution is coarse relative to operation time.
func detectBatchK(op Operation, inputSize int) int {
	const targetTicks = 50  // Target minimum ticks per measurement
	const maxK = 20         // Maximum batch size (beyond this, microarch effects dominate)
	const warmupIters = 100 // Warmup iterations for measurement

	input := make([]byte, inputSize)
	// Use zeros for warmup (consistent with baseline class)
	for i := range input {
		input[i] = 0
	}

	// Warmup
	for i := 0; i < warmupIters; i++ {
		op.Execute(input)
	}

	// Measure with increasing K until we get enough ticks
	for k := 1; k <= maxK; k++ {
		// Measure 100 iterations to get stable timing
		start := readTimer()
		for i := 0; i < k*100; i++ {
			op.Execute(input)
		}
		elapsed := readTimer() - start

		ticksPerOp := elapsed / uint64(k*100)
		if ticksPerOp >= targetTicks {
			return k
		}
	}

	return maxK
}

// WarmupOperation runs warmup iterations on the operation.
// This helps stabilize CPU frequency and cache state.
func warmupOperation(op Operation, inputSize int, iterations int) {
	input := make([]byte, inputSize)
	for i := 0; i < iterations; i++ {
		op.Execute(input)
	}
}

// FuncGenerator wraps generator functions to implement Generator interface.
type FuncGenerator struct {
	BaselineFunc func(output []byte)
	SampleFunc   func(output []byte)
}

// Generate implements Generator.
func (g *FuncGenerator) Generate(isBaseline bool, output []byte) {
	if isBaseline {
		g.BaselineFunc(output)
	} else {
		g.SampleFunc(output)
	}
}

// FuncOperation wraps an operation function to implement Operation interface.
type FuncOperation func(input []byte)

// Execute implements Operation.
func (f FuncOperation) Execute(input []byte) {
	f(input)
}

// ZeroGenerator generates all-zero baseline data and random sample data.
type ZeroGenerator struct {
	rng *rand.Rand
}

// NewZeroGenerator creates a generator that produces zeros for baseline
// and random data for sample class.
func NewZeroGenerator(seed uint64) *ZeroGenerator {
	return &ZeroGenerator{
		rng: rand.New(rand.NewPCG(seed, seed^0xDEADBEEF)),
	}
}

// Generate implements Generator.
func (g *ZeroGenerator) Generate(isBaseline bool, output []byte) {
	if isBaseline {
		// Baseline: all zeros
		for i := range output {
			output[i] = 0
		}
	} else {
		// Sample: random data
		for i := range output {
			output[i] = byte(g.rng.UintN(256))
		}
	}
}
