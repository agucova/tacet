package tacet

import (
	"math/rand/v2"
	"runtime"
	"sort"
	"time"
)

// Measurement constants. These mirror the reference Rust harness
// (crates/tacet/src/measurement/collector.rs) so that the Go and Rust front-ends
// make the same measurability decisions.
const (
	// targetTicksPerBatch is the number of timer ticks a single measurement must
	// span. Below roughly 50 ticks the empirical distribution collapses to a
	// sparse PMF and quantile-based inference is dominated by quantization.
	targetTicksPerBatch = 50.0

	// maxBatchK caps how many operations may be folded into one measurement.
	// Larger batches accumulate microarchitectural state (cache, predictors)
	// that shows up as a spurious timing difference.
	maxBatchK = 20

	// pilotSamples is the number of pilot measurements taken per class when
	// selecting the batch size.
	pilotSamples = 100

	// frequencyStabilization is how long to spin before measuring so the CPU
	// reaches a steady clock frequency.
	frequencyStabilization = 5 * time.Millisecond
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

// generateInputs pre-generates count inputs for one class into a single backing
// array. Inputs are generated before any measurement starts: running the
// generator inside the timed loop leaves the cache and branch predictors in a
// different state for each class, which shows up as a spurious timing
// difference.
func generateInputs(gen Generator, isBaseline bool, inputSize, count int) [][]byte {
	backing := make([]byte, count*inputSize)
	inputs := make([][]byte, count)
	for i := 0; i < count; i++ {
		inputs[i] = backing[i*inputSize : (i+1)*inputSize : (i+1)*inputSize]
		gen.Generate(isBaseline, inputs[i])
	}
	return inputs
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
// Returns baseline and sample timing arrays (in timer ticks). When batchK > 1
// each entry is the total for batchK calls, not a per-call time.
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

	// All inputs are generated up front, outside the measured region.
	baselineInputs := generateInputs(gen, true, inputSize, count)
	sampleInputs := generateInputs(gen, false, inputSize, count)

	// Generate interleaved schedule using Fisher-Yates shuffle
	schedule := generateSchedule(count, rng)

	baselineIdx := 0
	sampleIdx := 0

	for _, isBaseline := range schedule {
		var input []byte
		if isBaseline {
			input = baselineInputs[baselineIdx]
		} else {
			input = sampleInputs[sampleIdx]
		}

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

// pilotResult describes how an operation can be measured on this platform.
type pilotResult struct {
	// batchK is the number of calls to fold into one measurement.
	batchK int
	// ticksPerCall is the median cost of a single call, in timer ticks.
	ticksPerCall float64
	// measurable is false when even maxBatchK calls do not span
	// targetTicksPerBatch timer ticks, so timing differences cannot be
	// distinguished from quantization noise.
	measurable bool
}

// runPilot measures both input classes and selects the batch size K.
//
// K is chosen so a single measurement spans at least targetTicksPerBatch timer
// ticks, and the operation is reported as unmeasurable when even K = maxBatchK
// falls short. The cost estimate is the larger of the two per-class medians:
// once the slower class is resolved by the timer, the difference between the
// classes is resolvable too, and no batching is needed.
//
// Measuring both classes matters whenever they differ in cost. Timing only the
// baseline class picks K from whichever class happens to be cheaper. For an
// operation like modular exponentiation with a zero exponent that means folding
// twenty full exponentiations into every sample-class measurement, which both
// wastes time and makes consecutive calls share microarchitectural state, adding
// serial dependence that inflates the variance estimate.
func runPilot(gen Generator, op Operation, inputSize int) pilotResult {
	baselineInputs := generateInputs(gen, true, inputSize, pilotSamples)
	sampleInputs := generateInputs(gen, false, inputSize, pilotSamples)

	// Warm up on both classes.
	const pilotWarmup = 20
	for i := 0; i < pilotWarmup; i++ {
		op.Execute(baselineInputs[i])
		op.Execute(sampleInputs[i])
	}

	// measure returns the median cost per call, timing callsPerMeasurement
	// consecutive calls at a time.
	measure := func(inputs [][]byte, callsPerMeasurement int) float64 {
		ticks := make([]float64, len(inputs))
		for i, input := range inputs {
			start := readTimer()
			for k := 0; k < callsPerMeasurement; k++ {
				op.Execute(input)
			}
			ticks[i] = float64(readTimer()-start) / float64(callsPerMeasurement)
		}
		sort.Float64s(ticks)
		return ticks[len(ticks)/2]
	}

	// The cost estimate is the larger of the two class medians.
	classMax := func(callsPerMeasurement int) float64 {
		b := measure(baselineInputs, callsPerMeasurement)
		s := measure(sampleInputs, callsPerMeasurement)
		if s > b {
			return s
		}
		return b
	}

	// First pass times single calls. This is enough to clear an operation that
	// already spans the target, and keeps the pilot cheap for expensive
	// operations such as an RSA private-key operation.
	ticksPerCall := classMax(1)
	if ticksPerCall >= targetTicksPerBatch {
		return pilotResult{batchK: 1, ticksPerCall: ticksPerCall, measurable: true}
	}

	// The operation is faster than the target, so a batched second pass is cheap
	// in wall-clock terms and gives a usable estimate below timer resolution.
	ticksPerCall = classMax(maxBatchK)

	if ticksPerCall <= 0.0 {
		// Every pilot batch read as zero ticks: far below timer resolution.
		return pilotResult{batchK: maxBatchK, ticksPerCall: 0, measurable: false}
	}

	if ticksPerCall >= targetTicksPerBatch {
		return pilotResult{batchK: 1, ticksPerCall: ticksPerCall, measurable: true}
	}

	k := int(targetTicksPerBatch/ticksPerCall) + 1
	if k > maxBatchK {
		k = maxBatchK
	}
	return pilotResult{
		batchK:       k,
		ticksPerCall: ticksPerCall,
		measurable:   ticksPerCall*float64(k) >= targetTicksPerBatch,
	}
}

// lockAndStabilize prepares the measurement environment: it pins the calling
// goroutine to its OS thread so the samples are not spread across cores, and
// spins briefly so the CPU reaches a steady clock frequency. The returned
// function releases the pinning.
func lockAndStabilize() func() {
	runtime.LockOSThread()

	deadline := time.Now().Add(frequencyStabilization)
	var counter uint64
	for time.Now().Before(deadline) {
		counter++
	}
	stabilizationSink = counter

	return runtime.UnlockOSThread
}

// stabilizationSink keeps the frequency-stabilization loop from being optimized
// away.
var stabilizationSink uint64

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
