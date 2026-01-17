package timingoracle

import "github.com/agucova/timing-oracle/go/timingoracle/internal/ffi"

// AttackerModel represents the threat model for timing analysis.
// Choose based on your deployment scenario.
type AttackerModel int

const (
	// SharedHardware: theta = 0.6 ns (~2 cycles @ 3GHz)
	// Use for SGX enclaves, cross-VM attacks, containers, hyperthreading.
	// Attacker shares physical hardware with victim.
	SharedHardware AttackerModel = iota

	// PostQuantum: theta = 3.3 ns (~10 cycles)
	// Use for post-quantum cryptographic implementations.
	// KyberSlash-class attacks exploit ~10 cycle differences.
	PostQuantum

	// AdjacentNetwork: theta = 100 ns
	// Use for LAN services, HTTP/2 APIs (Timeless Timing Attacks).
	// Attacker can measure timing with sub-microsecond precision over network.
	AdjacentNetwork

	// RemoteNetwork: theta = 50 us
	// Use for general internet-exposed services.
	// Attacker measures timing over wide-area network.
	RemoteNetwork

	// Research: theta -> 0
	// Detect any statistical difference. NOT for production CI.
	// Use for profiling, debugging, or security research.
	Research
)

// String returns the string representation of the attacker model.
func (m AttackerModel) String() string {
	switch m {
	case SharedHardware:
		return "SharedHardware"
	case PostQuantum:
		return "PostQuantum"
	case AdjacentNetwork:
		return "AdjacentNetwork"
	case RemoteNetwork:
		return "RemoteNetwork"
	case Research:
		return "Research"
	default:
		return "Custom"
	}
}

// ThresholdNs returns the timing threshold in nanoseconds for this model.
func (m AttackerModel) ThresholdNs() float64 {
	switch m {
	case SharedHardware:
		return 0.6
	case PostQuantum:
		return 3.3
	case AdjacentNetwork:
		return 100.0
	case RemoteNetwork:
		return 50_000.0
	case Research:
		return 0.0
	default:
		return 100.0 // Default to AdjacentNetwork
	}
}

// toFFI converts to the internal FFI type.
func (m AttackerModel) toFFI() ffi.AttackerModel {
	switch m {
	case SharedHardware:
		return ffi.AttackerSharedHardware
	case PostQuantum:
		return ffi.AttackerPostQuantum
	case AdjacentNetwork:
		return ffi.AttackerAdjacentNetwork
	case RemoteNetwork:
		return ffi.AttackerRemoteNetwork
	case Research:
		return ffi.AttackerResearch
	default:
		return ffi.AttackerCustom
	}
}
