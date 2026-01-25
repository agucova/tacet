//go:build amd64

package tacet

import (
	"time"
)

// rdtsc reads the Time Stamp Counter via RDTSC instruction.
// Implemented in timer_amd64.s
func rdtsc() uint64

// readTimer returns the current timer value using rdtsc.
func readTimer() uint64 {
	return rdtsc()
}

// timerName returns the name of the timer being used.
func timerName() string {
	return "rdtsc"
}

// timerFrequency returns the estimated TSC frequency in Hz.
// This is calibrated at package initialization.
func timerFrequency() uint64 {
	return tscFrequency
}

// tscFrequency holds the calibrated TSC frequency.
var tscFrequency uint64

func init() {
	tscFrequency = calibrateTSCFrequency()
}

// calibrateTSCFrequency estimates the TSC frequency by timing against wall clock.
func calibrateTSCFrequency() uint64 {
	// Use multiple measurements and take median for stability
	const measurements = 5
	const duration = 10 * time.Millisecond

	var freqs [measurements]uint64
	for i := 0; i < measurements; i++ {
		start := rdtsc()
		startTime := time.Now()
		time.Sleep(duration)
		end := rdtsc()
		elapsed := time.Since(startTime)

		ticks := end - start
		freqHz := uint64(float64(ticks) / elapsed.Seconds())
		freqs[i] = freqHz
	}

	// Simple median (sort and take middle)
	for i := 0; i < measurements-1; i++ {
		for j := i + 1; j < measurements; j++ {
			if freqs[i] > freqs[j] {
				freqs[i], freqs[j] = freqs[j], freqs[i]
			}
		}
	}
	return freqs[measurements/2]
}

// timerResolutionNs returns the approximate timer resolution in nanoseconds.
// For RDTSC on modern x86_64, this is typically < 1ns (sub-cycle resolution).
func timerResolutionNs() float64 {
	if tscFrequency == 0 {
		return 1.0 // Fallback
	}
	return 1_000_000_000.0 / float64(tscFrequency)
}
