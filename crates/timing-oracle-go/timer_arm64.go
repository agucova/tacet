//go:build arm64

package timingoracle

// cntvct reads the virtual counter via CNTVCT_EL0.
// Implemented in timer_arm64.s
func cntvct() uint64

// cntfrq reads the counter frequency via CNTFRQ_EL0.
// Implemented in timer_arm64.s
func cntfrq() uint64

// readTimer returns the current timer value using cntvct_el0.
func readTimer() uint64 {
	return cntvct()
}

// timerName returns the name of the timer being used.
func timerName() string {
	return "cntvct_el0"
}

// timerFrequency returns the counter frequency in Hz.
// On ARM64, this is read directly from CNTFRQ_EL0.
func timerFrequency() uint64 {
	return cntfrq()
}

// timerResolutionNs returns the approximate timer resolution in nanoseconds.
// On Apple Silicon (24 MHz counter), this is ~41.67 ns.
// On other ARM64 (typically 1 GHz), this is ~1 ns.
func timerResolutionNs() float64 {
	freq := cntfrq()
	if freq == 0 {
		return 42.0 // Default for Apple Silicon (~24 MHz)
	}
	return 1_000_000_000.0 / float64(freq)
}
