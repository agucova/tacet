package timingoracle

import "errors"

// Errors that can be returned by the timing oracle.
var (
	// ErrInsufficientSamples is returned when there aren't enough samples for analysis.
	ErrInsufficientSamples = errors.New("timingoracle: insufficient samples for analysis (need >= 100)")

	// ErrNullPointer is returned when an internal null pointer error occurs.
	ErrNullPointer = errors.New("timingoracle: internal null pointer error")

	// ErrCalibrationFailed is returned when the calibration phase fails.
	ErrCalibrationFailed = errors.New("timingoracle: calibration failed")

	// ErrInternalError is returned when an internal error occurs in the Rust library.
	ErrInternalError = errors.New("timingoracle: internal error in statistical analysis")

	// ErrInvalidConfig is returned when the configuration is invalid.
	ErrInvalidConfig = errors.New("timingoracle: invalid configuration")

	// ErrOperationPanicked is returned when the operation under test panics.
	ErrOperationPanicked = errors.New("timingoracle: operation panicked during measurement")
)
