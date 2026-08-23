// Package tacet provides statistical timing side-channel detection for Go.
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
//	result, err := tacet.Test(
//	    tacet.NewZeroGenerator(0),
//	    tacet.FuncOperation(func(input []byte) {
//	        myCryptoFunction(input)
//	    }),
//	    32, // input size in bytes
//	    tacet.WithAttacker(tacet.AdjacentNetwork),
//	    tacet.WithTimeBudget(30 * time.Second),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	switch result.Outcome {
//	case tacet.Pass:
//	    fmt.Println("No timing leak detected")
//	case tacet.Fail:
//	    fmt.Printf("Timing leak: %s\n", result.Exploitability)
//	case tacet.Inconclusive:
//	    fmt.Printf("Inconclusive: %s\n", result.InconclusiveReason)
//	}
//
// # Attacker Models
//
// Choose an attacker model based on your threat scenario.
// Cycle-based thresholds use a 5 GHz reference frequency (conservative).
//   - SharedHardware (theta=0.4ns, ~2 cycles @ 5 GHz): SGX, containers, cross-VM attacks
//   - PostQuantum (theta=2.0ns, ~10 cycles @ 5 GHz): Post-quantum crypto implementations
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
package tacet

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/agucova/tacet/crates/tacet-go/internal/ffi"
)

// Errors
var (
	ErrInvalidConfig     = errors.New("tacet: invalid configuration")
	ErrCalibrationFailed = errors.New("tacet: calibration failed")
	ErrInternalError     = errors.New("tacet: internal error")
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

	// Pin to one OS thread and let the CPU reach a stable clock before any
	// measurement is taken.
	unlock := lockAndStabilize()
	defer unlock()

	startTime := time.Now()

	// Phase 0: Warmup, then pilot both classes to select the batch size
	WarmupOperation(op, inputSize, 100)

	pilot := pilotResult{batchK: 1, measurable: true}
	if !cfg.disableAdaptiveBatch {
		pilot = runPilot(gen, op, inputSize)
		if !pilot.measurable {
			return unmeasurableResult(pilot, time.Since(startTime)), nil
		}
	}
	batchK := pilot.batchK

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

	// The calibration samples count toward the posterior, so they are handed to
	// the first adaptive step along with the first batch. Without this the first
	// posterior is computed from a single batch and its variance (var_rate / n)
	// is wide enough to leave the posterior sitting on the prior.
	pendingBaseline := calBaseline
	pendingSample := calSample

	for {
		// Check time budget
		elapsed := time.Since(startTime)
		if elapsed > cfg.timeBudget {
			// Time budget exceeded - return inconclusive
			return &Result{
				Outcome:            Inconclusive,
				InconclusiveReason: ReasonTimeBudgetExceeded,
				SamplesUsed:        state.TotalSamples(),
				ElapsedTime:        elapsed,
				LeakProbability:    state.LeakProbability(),
			}, nil
		}

		// Collect batch (pure Go - no FFI overhead)
		batchBaseline, batchSample := collectSamples(
			gen, op, inputSize,
			cfg.batchSize, batchK, rng,
		)
		batchBaseline = append(pendingBaseline, batchBaseline...)
		batchSample = append(pendingSample, batchSample...)
		pendingBaseline = nil
		pendingSample = nil

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
		if stepResult.HasDecision {
			return resultFromFFI(&stepResult.Result, batchK), nil
		}

		// Check sample budget
		if state.TotalSamples() >= uint64(cfg.maxSamples) {
			return &Result{
				Outcome:            Inconclusive,
				InconclusiveReason: ReasonSampleBudgetExceeded,
				SamplesUsed:        state.TotalSamples(),
				ElapsedTime:        time.Since(startTime),
				LeakProbability:    state.LeakProbability(),
			}, nil
		}
	}
}

// unmeasurableResult builds the outcome for an operation that is too fast for
// the platform timer even with maximum batching.
func unmeasurableResult(pilot pilotResult, elapsed time.Duration) *Result {
	resolution := timerResolutionNs()
	return &Result{
		Outcome:           Unmeasurable,
		ElapsedTime:       elapsed,
		TimerResolutionNs: resolution,
		Recommendation: fmt.Sprintf(
			"Operation takes about %.1f ns per call, but %s has %.1f ns resolution: "+
				"even %d calls per measurement span only %.1f ticks, below the %.0f ticks "+
				"needed for reliable inference. Measure a larger unit of work, or run on a "+
				"platform with a finer timer (for example PMU cycle counters via sudo).",
			pilot.ticksPerCall*resolution, timerName(), resolution,
			maxBatchK, pilot.ticksPerCall*maxBatchK, targetTicksPerBatch),
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

	return resultFromFFI(ffiResult, 1), nil
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

// resultFromFFI converts FFI result to public Result.
//
// batchK is the number of operation calls folded into each measurement. The
// analysis works on batch totals, so effect sizes are divided by batchK to be
// reported per call, matching the Rust harness.
func resultFromFFI(r *ffi.Result, batchK int) *Result {
	if r == nil {
		return nil
	}

	k := float64(batchK)
	if k < 1 {
		k = 1
	}

	result := &Result{
		Outcome:         outcomeFromFFI(r.Outcome),
		LeakProbability: r.LeakProbability,
		Effect: Effect{
			MaxEffectNs:        r.Effect.MaxEffectNs / k,
			CredibleIntervalNs: [2]float64{r.Effect.CILow / k, r.Effect.CIHigh / k},
			TopQuantiles:       nil, // TODO: populate from C API when available
		},
		Quality:             qualityFromFFI(r.Quality),
		SamplesUsed:         r.SamplesUsed,
		ElapsedTime:         time.Duration(r.ElapsedTime * float64(time.Second)),
		Exploitability:      exploitabilityFromFFI(r.Exploitability),
		InconclusiveReason:  inconclusiveReasonFromFFI(r.InconclusiveReason),
		MDENs:               r.MDENs,
		ThetaUserNs:         r.ThetaUserNs,
		ThetaEffNs:          r.ThetaEffNs,
		ThetaFloorNs:        r.ThetaFloorNs,
		TimerResolutionNs:   r.TimerResolutionNs,
		DecisionThresholdNs: r.DecisionThresholdNs,
	}

	// Convert diagnostics (always present in FFI result, not a pointer)
	result.Diagnostics = &Diagnostics{
		DependenceLength:    r.Diagnostics.DependenceLength,
		EffectiveSampleSize: r.Diagnostics.EffectiveSampleSize,
		StationarityRatio:   r.Diagnostics.StationarityRatio,
		StationarityOK:      r.Diagnostics.StationarityOK,
		DiscreteMode:        r.Diagnostics.DiscreteMode,
		TimerResolutionNs:   r.Diagnostics.TimerResolutionNs,
		LambdaMean:          r.Diagnostics.LambdaMean,
		LambdaSD:            r.Diagnostics.LambdaSD,
		LambdaESS:           r.Diagnostics.LambdaESS,
		LambdaMixingOK:      r.Diagnostics.LambdaMixingOK,
		KappaMean:           r.Diagnostics.KappaMean,
		KappaCV:             r.Diagnostics.KappaCV,
		KappaESS:            r.Diagnostics.KappaESS,
		KappaMixingOK:       r.Diagnostics.KappaMixingOK,
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
