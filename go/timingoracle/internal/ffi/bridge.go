// Package ffi provides CGo bindings to the timing-oracle-go Rust library.
package ffi

/*
#cgo CFLAGS: -I${SRCDIR}/../../../../crates/timing-oracle-go/include
#cgo LDFLAGS: -L${SRCDIR}/../../../../target/release -ltiming_oracle_go

#include "timing_oracle_go.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

// Error codes from Rust FFI
var (
	ErrNullPointer       = errors.New("ffi: null pointer")
	ErrInsufficientData  = errors.New("ffi: insufficient samples (need >= 100)")
	ErrInternalError     = errors.New("ffi: internal error in Rust library")
	ErrInvalidCalibration = errors.New("ffi: invalid calibration state")
)

func errorFromCode(code C.int) error {
	switch code {
	case 0:
		return nil
	case -1:
		return ErrNullPointer
	case -2:
		return ErrInsufficientData
	case -3:
		return ErrInternalError
	default:
		return ErrInternalError
	}
}

// Version returns the library version string.
func Version() string {
	return C.GoString(C.togo_version())
}

// Calibration holds the opaque calibration state from Rust.
type Calibration struct {
	inner C.togo_calibration_t
}

// AdaptiveState holds the adaptive sampling state.
type AdaptiveState struct {
	inner C.togo_adaptive_state_t
}

// TotalBaseline returns the total baseline samples collected.
func (s *AdaptiveState) TotalBaseline() int {
	return int(s.inner.total_baseline)
}

// TotalSample returns the total sample class samples collected.
func (s *AdaptiveState) TotalSample() int {
	return int(s.inner.total_sample)
}

// CurrentProbability returns the current leak probability estimate.
func (s *AdaptiveState) CurrentProbability() float64 {
	return float64(s.inner.current_probability)
}

// ConfigDefault creates a default configuration for the given attacker model.
func ConfigDefault(model AttackerModel) Config {
	cConfig := C.togo_config_default(C.togo_attacker_model_t(model))
	return configFromC(&cConfig)
}

// Calibrate runs the calibration phase on initial timing samples.
func Calibrate(baseline, sample []uint64, config *Config) (*Calibration, error) {
	if len(baseline) < 100 || len(sample) < 100 {
		return nil, ErrInsufficientData
	}

	cConfig := config.toC()
	var cal Calibration

	code := C.togo_calibrate(
		(*C.uint64_t)(unsafe.Pointer(&baseline[0])),
		C.size_t(len(baseline)),
		(*C.uint64_t)(unsafe.Pointer(&sample[0])),
		C.size_t(len(sample)),
		&cConfig,
		&cal.inner,
	)

	if err := errorFromCode(code); err != nil {
		return nil, err
	}
	return &cal, nil
}

// Free releases the calibration state memory.
func (c *Calibration) Free() {
	if c != nil && c.inner.ptr != nil {
		C.togo_calibration_free(&c.inner)
	}
}

// NewAdaptiveState creates a new adaptive state for tracking samples.
func NewAdaptiveState() *AdaptiveState {
	return &AdaptiveState{
		inner: C.togo_adaptive_state_new(),
	}
}

// Free releases the adaptive state memory.
func (s *AdaptiveState) Free() {
	if s != nil && s.inner.ptr != nil {
		C.togo_adaptive_state_free(&s.inner)
	}
}

// AdaptiveStep runs one adaptive step with a new batch of samples.
// Returns (result, decisionReached, error).
// If decisionReached is true, result contains the final outcome.
func AdaptiveStep(
	calibration *Calibration,
	baseline, sample []uint64,
	config *Config,
	elapsedSecs float64,
	state *AdaptiveState,
) (*Result, bool, error) {
	if calibration == nil || state == nil {
		return nil, false, ErrNullPointer
	}
	if len(baseline) == 0 || len(sample) == 0 {
		return nil, false, ErrInsufficientData
	}

	cConfig := config.toC()
	var cResult C.togo_result_t

	code := C.togo_adaptive_step(
		&calibration.inner,
		(*C.uint64_t)(unsafe.Pointer(&baseline[0])),
		C.size_t(len(baseline)),
		(*C.uint64_t)(unsafe.Pointer(&sample[0])),
		C.size_t(len(sample)),
		&cConfig,
		C.double(elapsedSecs),
		&state.inner,
		&cResult,
	)

	switch code {
	case 0:
		// Continue - no decision yet
		return nil, false, nil
	case 1:
		// Decision reached
		result := resultFromC(&cResult)
		return result, true, nil
	default:
		return nil, false, errorFromCode(code)
	}
}

// Analyze runs complete analysis on pre-collected timing data.
// This is a convenience function for one-shot analysis.
func Analyze(baseline, sample []uint64, config *Config) (*Result, error) {
	if len(baseline) < 100 || len(sample) < 100 {
		return nil, ErrInsufficientData
	}

	cConfig := config.toC()
	var cResult C.togo_result_t

	code := C.togo_analyze(
		(*C.uint64_t)(unsafe.Pointer(&baseline[0])),
		C.size_t(len(baseline)),
		(*C.uint64_t)(unsafe.Pointer(&sample[0])),
		C.size_t(len(sample)),
		&cConfig,
		&cResult,
	)

	if err := errorFromCode(code); err != nil {
		return nil, err
	}

	result := resultFromC(&cResult)
	return result, nil
}

// OutcomeStr returns the string representation of an outcome.
func OutcomeStr(outcome Outcome) string {
	return C.GoString(C.togo_outcome_str(C.togo_outcome_t(outcome)))
}

// EffectPatternStr returns the string representation of an effect pattern.
func EffectPatternStr(pattern EffectPattern) string {
	return C.GoString(C.togo_effect_pattern_str(C.togo_effect_pattern_t(pattern)))
}

// ExploitabilityStr returns the string representation of exploitability.
func ExploitabilityStr(exploit Exploitability) string {
	return C.GoString(C.togo_exploitability_str(C.togo_exploitability_t(exploit)))
}

// QualityStr returns the string representation of quality.
func QualityStr(quality Quality) string {
	return C.GoString(C.togo_quality_str(C.togo_quality_t(quality)))
}

// InconclusiveReasonStr returns the string representation of inconclusive reason.
func InconclusiveReasonStr(reason InconclusiveReason) string {
	return C.GoString(C.togo_inconclusive_reason_str(C.togo_inconclusive_reason_t(reason)))
}
