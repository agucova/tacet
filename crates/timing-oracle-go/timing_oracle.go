// Package timingoracle provides statistical timing side-channel detection for Go.
//
// This library uses Bayesian statistical analysis to detect timing side channels
// in cryptographic and security-sensitive code. The measurement loop runs in pure
// Go for minimal overhead, while the statistical analysis is performed by a Rust
// library via CGo.
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
//   - SharedHardware (theta=0.6ns): SGX, containers, cross-VM attacks
//   - PostQuantum (theta=3.3ns): Post-quantum crypto implementations
//   - AdjacentNetwork (theta=100ns): LAN services, HTTP/2 APIs
//   - RemoteNetwork (theta=50us): Internet-exposed services
//   - Research (theta->0): Detect any difference (not for CI)
//
// # Architecture
//
// The library separates concerns for optimal performance:
//   - Measurement loop: Pure Go with platform-specific assembly timers
//   - Statistical analysis: Rust library via CGo (called only between batches)
//
// This design ensures no FFI overhead during timing-critical measurement.
package timingoracle

import (
	"errors"
	"math/rand/v2"
	"time"

	"github.com/agucova/timing-oracle/crates/timing-oracle-go/internal/ffi"
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

	// Phase 1b: Run calibration analysis (single CGo call)
	ffiCfg := cfg.toFFI()
	calibration, err := ffi.Calibrate(calBaseline, calSample, ffiCfg)
	if err != nil {
		return nil, ErrCalibrationFailed
	}
	defer calibration.Free()

	// Phase 2: Adaptive loop
	state := ffi.NewState()
	if state == nil {
		return nil, ErrInternalError
	}
	defer state.Free()

	startTime := time.Now()

	for {
		// Check time budget
		elapsed := time.Since(startTime)
		if elapsed > cfg.timeBudget {
			// Time budget exceeded - return inconclusive
			return &Result{
				Outcome:            Inconclusive,
				InconclusiveReason: ReasonTimeBudgetExceeded,
				SamplesUsed:        int(state.TotalSamples() / 2), // Per class
				ElapsedTime:        elapsed,
				LeakProbability:    state.LeakProbability(),
			}, nil
		}

		// Collect batch (pure Go - no FFI overhead)
		batchBaseline, batchSample := collectSamples(
			gen, op, inputSize,
			cfg.batchSize, batchK, rng,
		)

		// Run adaptive step (single CGo call)
		stepResult, err := ffi.Step(
			calibration,
			state,
			batchBaseline,
			batchSample,
			ffiCfg,
			elapsed.Seconds(),
		)
		if err != nil {
			return nil, ErrInternalError
		}

		// Check if we have a decision
		if stepResult.HasDecision && stepResult.Result != nil {
			return resultFromFFI(stepResult.Result), nil
		}

		// Check sample budget
		if state.TotalSamples()/2 >= uint64(cfg.maxSamples) {
			return &Result{
				Outcome:            Inconclusive,
				InconclusiveReason: ReasonSampleBudgetExceeded,
				SamplesUsed:        int(state.TotalSamples() / 2),
				ElapsedTime:        time.Since(startTime),
				LeakProbability:    state.LeakProbability(),
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

	ffiCfg := cfg.toFFI()
	ffiResult, err := ffi.Analyze(baseline, sample, ffiCfg)
	if err != nil {
		return nil, err
	}

	return resultFromFFI(ffiResult), nil
}

// Version returns the library version string.
func Version() string {
	return ffi.Version()
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

// toFFI converts Go config to FFI config
func (c *Config) toFFI() *ffi.Config {
	var model ffi.AttackerModel
	switch c.attackerModel {
	case SharedHardware:
		model = ffi.SharedHardware
	case PostQuantum:
		model = ffi.PostQuantum
	case AdjacentNetwork:
		model = ffi.AdjacentNetwork
	case RemoteNetwork:
		model = ffi.RemoteNetwork
	case Research:
		model = ffi.Research
	default:
		model = ffi.AdjacentNetwork
	}

	return &ffi.Config{
		AttackerModel:     model,
		CustomThresholdNs: c.customThresholdNs,
		MaxSamples:        uint64(c.maxSamples),
		TimeBudgetSecs:    c.timeBudget.Seconds(),
		PassThreshold:     c.passThreshold,
		FailThreshold:     c.failThreshold,
		Seed:              c.seed,
		TimerFrequencyHz:  timerFrequency(),
	}
}

// resultFromFFI converts FFI result to public Result
func resultFromFFI(r *ffi.Result) *Result {
	if r == nil {
		return nil
	}

	result := &Result{
		Outcome:         outcomeFromFFI(r.Outcome),
		LeakProbability: r.LeakProbability,
		Effect: Effect{
			ShiftNs: r.Effect.ShiftNs,
			TailNs:  r.Effect.TailNs,
			CILow:   r.Effect.CILow,
			CIHigh:  r.Effect.CIHigh,
			Pattern: effectPatternFromFFI(r.Effect.Pattern),
		},
		Quality:            qualityFromFFI(r.Quality),
		SamplesUsed:        r.SamplesUsed,
		ElapsedTime:        r.ElapsedTime,
		Exploitability:     exploitabilityFromFFI(r.Exploitability),
		InconclusiveReason: inconclusiveReasonFromFFI(r.InconclusiveReason),
		MDEShiftNs:         r.MDEShiftNs,
		MDETailNs:          r.MDETailNs,
		ThetaUserNs:        r.ThetaUserNs,
		ThetaEffNs:         r.ThetaEffNs,
		ThetaFloorNs:       r.ThetaFloorNs,
	}

	// Convert diagnostics if available
	if r.Diagnostics != nil {
		result.Diagnostics = &Diagnostics{
			DependenceLength:     r.Diagnostics.DependenceLength,
			EffectiveSampleSize:  r.Diagnostics.EffectiveSampleSize,
			StationarityRatio:    r.Diagnostics.StationarityRatio,
			StationarityOK:       r.Diagnostics.StationarityOK,
			ProjectionMismatchQ:  r.Diagnostics.ProjectionMismatchQ,
			ProjectionMismatchOK: r.Diagnostics.ProjectionMismatchOK,
			DiscreteMode:         r.Diagnostics.DiscreteMode,
			TimerResolutionNs:    r.Diagnostics.TimerResolutionNs,
			LambdaMean:           r.Diagnostics.LambdaMean,
			LambdaSD:             r.Diagnostics.LambdaSD,
			LambdaCV:             0, // Not in C API, compute if needed
			LambdaESS:            r.Diagnostics.LambdaESS,
			LambdaMixingOK:       r.Diagnostics.LambdaMixingOK,
			KappaMean:            r.Diagnostics.KappaMean,
			KappaSD:              0, // Not in C API
			KappaCV:              r.Diagnostics.KappaCV,
			KappaESS:             r.Diagnostics.KappaESS,
			KappaMixingOK:        r.Diagnostics.KappaMixingOK,
		}
	}

	return result
}

// Conversion helpers

func outcomeFromFFI(o ffi.Outcome) Outcome {
	switch o {
	case ffi.Pass:
		return Pass
	case ffi.Fail:
		return Fail
	case ffi.Inconclusive:
		return Inconclusive
	case ffi.Unmeasurable:
		return Unmeasurable
	default:
		return Inconclusive
	}
}

func qualityFromFFI(q ffi.Quality) Quality {
	switch q {
	case ffi.Excellent:
		return Excellent
	case ffi.Good:
		return Good
	case ffi.Poor:
		return Poor
	case ffi.TooNoisy:
		return TooNoisy
	default:
		return Poor
	}
}

func exploitabilityFromFFI(e ffi.Exploitability) Exploitability {
	switch e {
	case ffi.SharedHardwareOnly:
		return SharedHardwareOnly
	case ffi.HTTP2Multiplexing:
		return HTTP2Multiplexing
	case ffi.StandardRemote:
		return StandardRemote
	case ffi.ObviousLeak:
		return ObviousLeak
	default:
		return SharedHardwareOnly
	}
}

func effectPatternFromFFI(p ffi.EffectPattern) EffectPattern {
	switch p {
	case ffi.UniformShift:
		return UniformShift
	case ffi.TailEffect:
		return TailEffect
	case ffi.Mixed:
		return Mixed
	case ffi.Indeterminate:
		return Indeterminate
	default:
		return Indeterminate
	}
}

func inconclusiveReasonFromFFI(r ffi.InconclusiveReason) InconclusiveReason {
	switch r {
	case ffi.ReasonNone:
		return ReasonNone
	case ffi.ReasonDataTooNoisy:
		return ReasonDataTooNoisy
	case ffi.ReasonNotLearning:
		return ReasonNotLearning
	case ffi.ReasonWouldTakeTooLong:
		return ReasonWouldTakeTooLong
	case ffi.ReasonTimeBudgetExceeded:
		return ReasonTimeBudgetExceeded
	case ffi.ReasonSampleBudgetExceeded:
		return ReasonSampleBudgetExceeded
	case ffi.ReasonConditionsChanged:
		return ReasonConditionsChanged
	case ffi.ReasonThresholdElevated:
		return ReasonThresholdElevated
	default:
		return ReasonNone
	}
}
