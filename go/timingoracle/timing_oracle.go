// Package timingoracle provides statistical timing side-channel detection for Go.
//
// This library uses Bayesian statistical analysis to detect timing side channels
// in cryptographic and security-sensitive code. The measurement loop runs in pure
// Go for minimal overhead, while the statistical analysis is performed by a Rust
// library via UniFFI.
//
// # Usage
//
// The main entry point is the Test function:
//
//	result, err := timingoracle.Test(
//	    timingoracle.NewZeroGenerator(0),
//	    timingoracle.FuncOperation(func(input []byte) {
//	        myCryptoFunction(input)
//	    }),
//	    32, // input size in bytes
//	    timingoracle.WithAttacker(timingoracle.AdjacentNetwork),
//	    timingoracle.WithTimeBudget(30 * time.Second),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	switch result.Outcome {
//	case timingoracle.Pass:
//	    fmt.Println("No timing leak detected")
//	case timingoracle.Fail:
//	    fmt.Printf("Timing leak: %s\n", result.Exploitability)
//	case timingoracle.Inconclusive:
//	    fmt.Printf("Inconclusive: %s\n", result.InconclusiveReason)
//	}
//
// # Attacker Models
//
// Choose an attacker model based on your threat scenario:
//   - SharedHardware (θ=0.6ns): SGX, containers, cross-VM attacks
//   - PostQuantum (θ=3.3ns): Post-quantum crypto implementations
//   - AdjacentNetwork (θ=100ns): LAN services, HTTP/2 APIs
//   - RemoteNetwork (θ=50μs): Internet-exposed services
//   - Research (θ→0): Detect any difference (not for CI)
//
// # Architecture
//
// The library separates concerns for optimal performance:
//   - Measurement loop: Pure Go with platform-specific assembly timers
//   - Statistical analysis: Rust library via UniFFI (called only between batches)
//
// This design ensures no FFI overhead during timing-critical measurement.
package timingoracle

import (
	"errors"
	"math/rand/v2"
	"time"

	uniffi "github.com/agucova/timing-oracle/bindings/go/timing_oracle_uniffi"
)

// Errors
var (
	ErrInvalidConfig     = errors.New("timingoracle: invalid configuration")
	ErrCalibrationFailed = errors.New("timingoracle: calibration failed")
	ErrInternalError     = errors.New("timingoracle: internal error")
)

// Test runs a timing side-channel analysis on the given operation.
//
// Parameters:
//   - gen: Generator for creating test inputs (baseline vs sample class)
//   - op: The operation to test for timing side channels
//   - inputSize: Size of input buffer in bytes
//   - opts: Functional options for configuration
//
// Returns the analysis result and any error encountered.
func Test(gen Generator, op Operation, inputSize int, opts ...Option) (*Result, error) {
	// Apply configuration options
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(cfg)
	}

	// Validate configuration
	if inputSize <= 0 {
		return nil, ErrInvalidConfig
	}

	// Initialize RNG
	var rng *rand.Rand
	if cfg.seed != 0 {
		rng = rand.New(rand.NewPCG(cfg.seed, cfg.seed^0xDEADBEEF))
	} else {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}

	// Phase 0: Warmup and detect batch K
	WarmupOperation(op, inputSize, 100)

	batchK := 1
	if !cfg.disableAdaptiveBatch {
		batchK = detectBatchK(op, inputSize)
	}

	// Phase 1: Calibration - collect initial samples (pure Go)
	calBaseline, calSample := collectSamples(
		gen, op, inputSize,
		cfg.calibrationSamples, batchK, rng,
	)

	// Phase 1b: Run calibration analysis (single FFI call)
	uniffiConfig := cfg.toUniFFI()
	calibration, err := uniffi.CalibrateSamples(calBaseline, calSample, uniffiConfig)
	if err != nil {
		return nil, ErrCalibrationFailed
	}
	defer calibration.Destroy()

	// Phase 2: Adaptive loop
	state := uniffi.NewAdaptiveState()
	defer state.Destroy()

	startTime := time.Now()

	for {
		// Check time budget
		elapsed := time.Since(startTime)
		if elapsed > cfg.timeBudget {
			// Time budget exceeded - return inconclusive
			return &Result{
				Outcome:            Inconclusive,
				InconclusiveReason: ReasonTimeBudgetExceeded,
				SamplesUsed:        int(state.TotalBaseline()),
				ElapsedTime:        elapsed,
				LeakProbability:    state.CurrentProbability(),
			}, nil
		}

		// Collect batch (pure Go - no FFI overhead)
		batchBaseline, batchSample := collectSamples(
			gen, op, inputSize,
			cfg.batchSize, batchK, rng,
		)

		// Run adaptive step (single FFI call)
		stepResult, err := uniffi.AdaptiveStepBatch(
			calibration,
			state,
			batchBaseline,
			batchSample,
			uniffiConfig,
			elapsed.Seconds(),
		)
		if err != nil {
			return nil, ErrInternalError
		}

		// Check if we have a decision
		switch v := stepResult.(type) {
		case uniffi.AdaptiveStepResultContinue:
			// Continue - no decision yet
		case uniffi.AdaptiveStepResultDecision:
			// Decision reached
			return resultFromUniFFI(v.Result), nil
		}

		// Check sample budget
		if state.TotalBaseline() >= uint64(cfg.maxSamples) {
			return &Result{
				Outcome:            Inconclusive,
				InconclusiveReason: ReasonSampleBudgetExceeded,
				SamplesUsed:        int(state.TotalBaseline()),
				ElapsedTime:        time.Since(startTime),
				LeakProbability:    state.CurrentProbability(),
			}, nil
		}
	}
}

// Analyze runs one-shot analysis on pre-collected timing data.
// This is useful when timing data has been collected separately.
//
// Parameters:
//   - baseline: Timing samples for baseline class (in timer ticks)
//   - sample: Timing samples for sample class (in timer ticks)
//   - opts: Functional options for configuration
//
// Note: The timing data should be raw timer ticks, not nanoseconds.
// The library will convert based on the timer frequency.
func Analyze(baseline, sample []uint64, opts ...Option) (*Result, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(cfg)
	}

	uniffiConfig := cfg.toUniFFI()
	uniffiResult, err := uniffi.Analyze(baseline, sample, uniffiConfig)
	if err != nil {
		return nil, err
	}

	return resultFromUniFFI(uniffiResult), nil
}

// Version returns the library version string.
func Version() string {
	return uniffi.Version()
}

// TimerName returns the name of the platform timer being used.
func TimerName() string {
	return timerName()
}

// TimerFrequency returns the timer frequency in Hz.
func TimerFrequency() uint64 {
	return timerFrequency()
}

// TimerResolutionNs returns the approximate timer resolution in nanoseconds.
func TimerResolutionNs() float64 {
	return timerResolutionNs()
}

// WarmupOperation is exported for use in custom measurement loops.
func WarmupOperation(op Operation, inputSize int, iterations int) {
	warmupOperation(op, inputSize, iterations)
}
