// Package ffi provides CGo bindings to the tacet C library.
//
// # Installation
//
// Before using this package, install the native library:
//
//	go run github.com/agucova/tacet/crates/tacet-go/cmd/tacet-install@latest
//
// This downloads the pre-built static library (~12MB) for your platform
// and places it in the module cache where CGo can find it.
//
// # Build Errors
//
// If you see linker errors about missing tacet symbols, it means the
// native library is not installed. Run the install command above.
package ffi

/*
#cgo CFLAGS: -I${SRCDIR}/../../../tacet-c/include

// Platform-specific static libraries
// The install command places these in the module cache at build time
#cgo darwin,arm64 LDFLAGS: ${SRCDIR}/lib/darwin_arm64/libtacet_c.a -framework CoreFoundation -framework Security
#cgo darwin,amd64 LDFLAGS: ${SRCDIR}/lib/darwin_amd64/libtacet_c.a -framework CoreFoundation -framework Security
#cgo linux,arm64 LDFLAGS: ${SRCDIR}/lib/linux_arm64/libtacet_c.a -lm -lpthread
#cgo linux,amd64 LDFLAGS: ${SRCDIR}/lib/linux_amd64/libtacet_c.a -lm -lpthread

#include <tacet.h>
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"time"
	"unsafe"
)

// Errors from FFI calls
var (
	ErrNullPointer       = errors.New("null pointer")
	ErrInvalidConfig     = errors.New("invalid config")
	ErrCalibrationFailed = errors.New("calibration failed")
	ErrAnalysisFailed    = errors.New("analysis failed")
	ErrNotEnoughSamples  = errors.New("not enough samples")
	ErrUnknown           = errors.New("unknown error")
)

// ErrorFromC converts C error code to Go error
func ErrorFromC(err int) error {
	switch C.enum_ToError(err) {
	case C.Ok:
		return nil
	case C.NullPointer:
		return ErrNullPointer
	case C.InvalidConfig:
		return ErrInvalidConfig
	case C.CalibrationFailed:
		return ErrCalibrationFailed
	case C.AnalysisFailed:
		return ErrAnalysisFailed
	case C.NotEnoughSamples:
		return ErrNotEnoughSamples
	default:
		return ErrUnknown
	}
}

// AttackerModel represents the threat model
type AttackerModel int

const (
	SharedHardware AttackerModel = iota
	PostQuantum
	AdjacentNetwork
	RemoteNetwork
	Research
)

// Outcome represents the test result
type Outcome int

const (
	Pass Outcome = iota
	Fail
	Inconclusive
	Unmeasurable
)

// Quality assessment
type Quality int

const (
	Excellent Quality = iota
	Good
	Poor
	TooNoisy
)

// Exploitability assessment
type Exploitability int

const (
	SharedHardwareOnly Exploitability = iota
	HTTP2Multiplexing
	StandardRemote
	ObviousLeak
)

// EffectPattern describes the timing effect pattern
type EffectPattern int

const (
	UniformShift EffectPattern = iota
	TailEffect
	Mixed
	Indeterminate
)

// InconclusiveReason explains why a test was inconclusive
type InconclusiveReason int

const (
	ReasonNone InconclusiveReason = iota
	ReasonDataTooNoisy
	ReasonNotLearning
	ReasonWouldTakeTooLong
	ReasonTimeBudgetExceeded
	ReasonSampleBudgetExceeded
	ReasonConditionsChanged
	ReasonThresholdElevated
)

// Effect holds the effect size estimate
type Effect struct {
	ShiftNs float64
	TailNs  float64
	CILow   float64
	CIHigh  float64
	Pattern EffectPattern
}

// Diagnostics holds detailed diagnostic information from the analysis
type Diagnostics struct {
	DependenceLength     int
	EffectiveSampleSize  int
	StationarityRatio    float64
	StationarityOK       bool
	ProjectionMismatchQ  float64
	ProjectionMismatchOK bool
	DiscreteMode         bool
	TimerResolutionNs    float64
	LambdaMean           float64
	LambdaSD             float64
	LambdaESS            float64
	LambdaMixingOK       bool
	KappaMean            float64
	KappaCV              float64
	KappaESS             float64
	KappaMixingOK        bool
}

// Result holds the complete analysis result
type Result struct {
	Outcome            Outcome
	LeakProbability    float64
	Effect             Effect
	Quality            Quality
	SamplesUsed        int
	ElapsedTime        time.Duration
	Exploitability     Exploitability
	InconclusiveReason InconclusiveReason
	MDEShiftNs         float64
	MDETailNs          float64
	ThetaUserNs        float64
	ThetaEffNs         float64
	ThetaFloorNs       float64
	Diagnostics        *Diagnostics
}

// StepResult holds the result of an adaptive step
type StepResult struct {
	HasDecision     bool
	LeakProbability float64
	SamplesUsed     uint64
	ElapsedSecs     float64
	Result          *Result
}

// Config holds the configuration for timing analysis
type Config struct {
	AttackerModel     AttackerModel
	CustomThresholdNs float64
	MaxSamples        uint64
	TimeBudgetSecs    float64
	PassThreshold     float64
	FailThreshold     float64
	Seed              uint64
	TimerFrequencyHz  uint64
}

// Calibration wraps the C calibration handle
type Calibration struct {
	ptr *C.struct_ToCalibration
}

// Free releases the calibration data
func (c *Calibration) Free() {
	if c != nil && c.ptr != nil {
		C.to_calibration_free(c.ptr)
		c.ptr = nil
	}
}

// State wraps the C adaptive state handle
type State struct {
	ptr *C.struct_ToState
}

// NewState creates a new adaptive state
func NewState() *State {
	ptr := C.to_state_new()
	if ptr == nil {
		return nil
	}
	return &State{ptr: ptr}
}

// Free releases the state
func (s *State) Free() {
	if s != nil && s.ptr != nil {
		C.to_state_free(s.ptr)
		s.ptr = nil
	}
}

// TotalSamples returns total samples collected (both classes)
func (s *State) TotalSamples() uint64 {
	if s == nil || s.ptr == nil {
		return 0
	}
	return uint64(C.to_state_total_samples(s.ptr))
}

// LeakProbability returns current leak probability estimate
func (s *State) LeakProbability() float64 {
	if s == nil || s.ptr == nil {
		return 0.5
	}
	return float64(C.to_state_leak_probability(s.ptr))
}

// configToC converts Go config to C config
func configToC(cfg *Config) C.struct_ToConfig {
	var model C.enum_ToAttackerModel
	switch cfg.AttackerModel {
	case SharedHardware:
		model = C.SharedHardware
	case PostQuantum:
		model = C.PostQuantum
	case AdjacentNetwork:
		model = C.AdjacentNetwork
	case RemoteNetwork:
		model = C.RemoteNetwork
	case Research:
		model = C.Research
	default:
		model = C.AdjacentNetwork
	}

	return C.struct_ToConfig{
		attacker_model:      model,
		custom_threshold_ns: C.double(cfg.CustomThresholdNs),
		max_samples:         C.uint64_t(cfg.MaxSamples),
		time_budget_secs:    C.double(cfg.TimeBudgetSecs),
		pass_threshold:      C.double(cfg.PassThreshold),
		fail_threshold:      C.double(cfg.FailThreshold),
		seed:                C.uint64_t(cfg.Seed),
		timer_frequency_hz:  C.uint64_t(cfg.TimerFrequencyHz),
	}
}

// Calibrate performs calibration on initial samples
func Calibrate(baseline, sample []uint64, config *Config) (*Calibration, error) {
	if len(baseline) == 0 || len(sample) == 0 {
		return nil, ErrNotEnoughSamples
	}

	// Use minimum length if arrays differ
	count := len(baseline)
	if len(sample) < count {
		count = len(sample)
	}

	var errCode C.enum_ToError
	cConfig := configToC(config)

	ptr := C.to_calibrate(
		(*C.uint64_t)(unsafe.Pointer(&baseline[0])),
		(*C.uint64_t)(unsafe.Pointer(&sample[0])),
		C.uintptr_t(count),
		&cConfig,
		&errCode,
	)

	if errCode != C.Ok {
		return nil, ErrorFromC(int(errCode))
	}
	if ptr == nil {
		return nil, ErrCalibrationFailed
	}
	return &Calibration{ptr: ptr}, nil
}

// Step runs one adaptive step with a batch of new samples
func Step(calibration *Calibration, state *State, baseline, sample []uint64, config *Config, elapsedSecs float64) (*StepResult, error) {
	if calibration == nil || calibration.ptr == nil {
		return nil, ErrNullPointer
	}
	if state == nil || state.ptr == nil {
		return nil, ErrNullPointer
	}
	if len(baseline) == 0 || len(sample) == 0 {
		return nil, ErrNotEnoughSamples
	}

	// Use minimum length if arrays differ
	count := len(baseline)
	if len(sample) < count {
		count = len(sample)
	}

	var cResult C.struct_ToStepResult
	cConfig := configToC(config)

	errCode := C.to_step(
		calibration.ptr,
		state.ptr,
		(*C.uint64_t)(unsafe.Pointer(&baseline[0])),
		(*C.uint64_t)(unsafe.Pointer(&sample[0])),
		C.uintptr_t(count),
		&cConfig,
		C.double(elapsedSecs),
		&cResult,
	)

	if errCode != C.Ok {
		return nil, ErrorFromC(int(errCode))
	}

	stepResult := &StepResult{
		HasDecision:     bool(cResult.has_decision),
		LeakProbability: float64(cResult.leak_probability),
		SamplesUsed:     uint64(cResult.samples_used),
		ElapsedSecs:     float64(cResult.elapsed_secs),
	}

	if stepResult.HasDecision {
		stepResult.Result = resultFromC(&cResult.result)
	}

	return stepResult, nil
}

// Analyze performs one-shot analysis on pre-collected samples
func Analyze(baseline, sample []uint64, config *Config) (*Result, error) {
	if len(baseline) == 0 || len(sample) == 0 {
		return nil, ErrNotEnoughSamples
	}

	// Use minimum length if arrays differ
	count := len(baseline)
	if len(sample) < count {
		count = len(sample)
	}

	var cResult C.struct_ToResult
	cConfig := configToC(config)

	errCode := C.to_analyze(
		(*C.uint64_t)(unsafe.Pointer(&baseline[0])),
		(*C.uint64_t)(unsafe.Pointer(&sample[0])),
		C.uintptr_t(count),
		&cConfig,
		&cResult,
	)

	if errCode != C.Ok {
		return nil, ErrorFromC(int(errCode))
	}
	return resultFromC(&cResult), nil
}

// Version returns the library version string
func Version() string {
	return C.GoString(C.to_version())
}

// AttackerThresholdNs returns the threshold for an attacker model
func AttackerThresholdNs(model AttackerModel) float64 {
	var cModel C.enum_ToAttackerModel
	switch model {
	case SharedHardware:
		cModel = C.SharedHardware
	case PostQuantum:
		cModel = C.PostQuantum
	case AdjacentNetwork:
		cModel = C.AdjacentNetwork
	case RemoteNetwork:
		cModel = C.RemoteNetwork
	case Research:
		cModel = C.Research
	default:
		cModel = C.AdjacentNetwork
	}
	return float64(C.to_attacker_threshold_ns(cModel))
}

// resultFromC converts C result to Go Result
func resultFromC(r *C.struct_ToResult) *Result {
	if r == nil {
		return nil
	}

	// Convert diagnostics
	d := r.diagnostics
	diagnostics := &Diagnostics{
		DependenceLength:     int(d.dependence_length),
		EffectiveSampleSize:  int(d.effective_sample_size),
		StationarityRatio:    float64(d.stationarity_ratio),
		StationarityOK:       bool(d.stationarity_ok),
		ProjectionMismatchQ:  float64(d.projection_mismatch_q),
		ProjectionMismatchOK: bool(d.projection_mismatch_ok),
		DiscreteMode:         bool(d.discrete_mode),
		TimerResolutionNs:    float64(d.timer_resolution_ns),
		LambdaMean:           float64(d.lambda_mean),
		LambdaSD:             float64(d.lambda_sd),
		LambdaESS:            float64(d.lambda_ess),
		LambdaMixingOK:       bool(d.lambda_mixing_ok),
		KappaMean:            float64(d.kappa_mean),
		KappaCV:              float64(d.kappa_cv),
		KappaESS:             float64(d.kappa_ess),
		KappaMixingOK:        bool(d.kappa_mixing_ok),
	}

	return &Result{
		Outcome:            outcomeFromC(r.outcome),
		LeakProbability:    float64(r.leak_probability),
		Effect: Effect{
			ShiftNs: float64(r.effect.shift_ns),
			TailNs:  float64(r.effect.tail_ns),
			CILow:   float64(r.effect.ci_low_ns),
			CIHigh:  float64(r.effect.ci_high_ns),
			Pattern: effectPatternFromC(r.effect.pattern),
		},
		Quality:            qualityFromC(r.quality),
		SamplesUsed:        int(r.samples_used),
		ElapsedTime:        time.Duration(float64(r.elapsed_secs) * float64(time.Second)),
		Exploitability:     exploitabilityFromC(r.exploitability),
		InconclusiveReason: inconclusiveReasonFromC(r.inconclusive_reason),
		MDEShiftNs:         float64(r.mde_shift_ns),
		MDETailNs:          float64(r.mde_tail_ns),
		ThetaUserNs:        float64(r.theta_user_ns),
		ThetaEffNs:         float64(r.theta_eff_ns),
		ThetaFloorNs:       float64(r.theta_floor_ns),
		Diagnostics:        diagnostics,
	}
}

func outcomeFromC(o C.enum_ToOutcome) Outcome {
	switch o {
	case C.Pass:
		return Pass
	case C.Fail:
		return Fail
	case C.Inconclusive:
		return Inconclusive
	case C.Unmeasurable:
		return Unmeasurable
	default:
		return Inconclusive
	}
}

func qualityFromC(q C.enum_ToMeasurementQuality) Quality {
	switch q {
	case C.Excellent:
		return Excellent
	case C.Good:
		return Good
	case C.Poor:
		return Poor
	case C.TooNoisy:
		return TooNoisy
	default:
		return Poor
	}
}

func exploitabilityFromC(e C.enum_ToExploitability) Exploitability {
	switch e {
	case C.SharedHardwareOnly:
		return SharedHardwareOnly
	case C.Http2Multiplexing:
		return HTTP2Multiplexing
	case C.StandardRemote:
		return StandardRemote
	case C.ObviousLeak:
		return ObviousLeak
	default:
		return SharedHardwareOnly
	}
}

func effectPatternFromC(p C.enum_ToEffectPattern) EffectPattern {
	switch p {
	case C.UniformShift:
		return UniformShift
	case C.TailEffect:
		return TailEffect
	case C.Mixed:
		return Mixed
	case C.Indeterminate:
		return Indeterminate
	default:
		return Indeterminate
	}
}

func inconclusiveReasonFromC(r C.enum_ToInconclusiveReason) InconclusiveReason {
	switch r {
	case C.None:
		return ReasonNone
	case C.DataTooNoisy:
		return ReasonDataTooNoisy
	case C.NotLearning:
		return ReasonNotLearning
	case C.WouldTakeTooLong:
		return ReasonWouldTakeTooLong
	case C.TimeBudgetExceeded:
		return ReasonTimeBudgetExceeded
	case C.SampleBudgetExceeded:
		return ReasonSampleBudgetExceeded
	case C.ConditionsChanged:
		return ReasonConditionsChanged
	case C.ThresholdElevated:
		return ReasonThresholdElevated
	default:
		return ReasonNone
	}
}
