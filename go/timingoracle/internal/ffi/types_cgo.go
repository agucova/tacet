//go:build ignore

// This file is input for "go tool cgo -godefs" to generate types_generated.go.
// Run: go generate ./...
//
// The generated file contains pure Go struct definitions that match the C
// memory layout, eliminating manual synchronization between C and Go types.

package ffi

/*
#cgo CFLAGS: -I${SRCDIR}/../../../../crates/timing-oracle-go/include
#include "timing_oracle_go.h"
*/
import "C"

// Enums - these become Go integer types with the correct underlying size.
type AttackerModel C.togo_attacker_model_t
type Outcome C.togo_outcome_t
type InconclusiveReason C.togo_inconclusive_reason_t
type EffectPattern C.togo_effect_pattern_t
type Exploitability C.togo_exploitability_t
type Quality C.togo_quality_t

// Structs - these become pure Go structs with matching memory layout.
type Config C.togo_config_t
type Effect C.togo_effect_t
type Diagnostics C.togo_diagnostics_t
type Result C.togo_result_t
type Calibration C.togo_calibration_t
type AdaptiveState C.togo_adaptive_state_t
