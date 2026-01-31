package tacet

// AttackerModel represents the threat model for timing analysis.
// Choose based on your deployment scenario.
//
// Cycle-based thresholds use a 5 GHz reference frequency (conservative).
type AttackerModel int

const (
	// SharedHardware: theta = 0.4 ns (~2 cycles @ 5 GHz)
	// Use for SGX enclaves, cross-VM attacks, containers, hyperthreading.
	// Attacker shares physical hardware with victim.
	SharedHardware AttackerModel = iota

	// PostQuantum: theta = 2.0 ns (~10 cycles @ 5 GHz)
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
// Cycle-based models use a 5 GHz reference frequency (conservative).
func (m AttackerModel) ThresholdNs() float64 {
	switch m {
	case SharedHardware:
		return 0.4 // ~2 cycles @ 5 GHz
	case PostQuantum:
		return 2.0 // ~10 cycles @ 5 GHz
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
