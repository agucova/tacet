//go:build !amd64 && !arm64

package tacet

import (
	"time"
)

// Generic fallback timer using time.Now().
// Less precise than hardware timers but works everywhere.

// genericEpoch is the reference point for timer values.
var genericEpoch = time.Now()

// readTimer returns nanoseconds since epoch.
func readTimer() uint64 {
	return uint64(time.Since(genericEpoch).Nanoseconds())
}

// timerName returns the name of the timer being used.
func timerName() string {
	return "time.Now"
}

// timerFrequency returns the timer frequency in Hz.
// For time.Now, this is 1 GHz (nanosecond resolution).
func timerFrequency() uint64 {
	return 1_000_000_000
}

// timerResolutionNs returns the approximate timer resolution in nanoseconds.
// For time.Now, this is typically 1ns nominal but actual precision varies.
func timerResolutionNs() float64 {
	return 1.0
}
