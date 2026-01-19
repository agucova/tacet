package ffi

/*
#include "timing_oracle_go.h"
*/
import "C"

// AttackerModel represents the threat model for timing analysis.
type AttackerModel int

const (
	// AttackerSharedHardware: theta = 0.6 ns (~2 cycles @ 3GHz)
	// Use for SGX, cross-VM, containers, hyperthreading
	AttackerSharedHardware AttackerModel = iota
	// AttackerPostQuantum: theta = 3.3 ns (~10 cycles)
	// Use for post-quantum crypto (KyberSlash-class)
	AttackerPostQuantum
	// AttackerAdjacentNetwork: theta = 100 ns
	// Use for LAN, HTTP/2 (Timeless Timing Attacks)
	AttackerAdjacentNetwork
	// AttackerRemoteNetwork: theta = 50 us
	// Use for general internet
	AttackerRemoteNetwork
	// AttackerResearch: theta -> 0
	// Detect any difference (not for CI)
	AttackerResearch
	// AttackerCustom: user-specified threshold
	AttackerCustom
)

// Outcome represents the test result.
type Outcome int

const (
	OutcomePass Outcome = iota
	OutcomeFail
	OutcomeInconclusive
	OutcomeUnmeasurable
)

// InconclusiveReason explains why a test was inconclusive.
type InconclusiveReason int

const (
	InconclusiveNone InconclusiveReason = iota
	InconclusiveDataTooNoisy
	InconclusiveNotLearning
	InconclusiveWouldTakeTooLong
	InconclusiveTimeBudgetExceeded
	InconclusiveSampleBudgetExceeded
	InconclusiveConditionsChanged
	InconclusiveThresholdElevated
	InconclusiveModelMismatch
)

// EffectPattern describes the pattern of timing difference.
type EffectPattern int

const (
	EffectUniformShift EffectPattern = iota
	EffectTailEffect
	EffectMixed
	EffectIndeterminate
)

// Exploitability assesses the practical exploitability of a leak.
type Exploitability int

const (
	ExploitNegligible Exploitability = iota
	ExploitPossibleLAN
	ExploitLikelyLAN
	ExploitPossibleRemote
)

// Quality assesses the measurement quality.
type Quality int

const (
	QualityExcellent Quality = iota
	QualityGood
	QualityPoor
	QualityTooNoisy
)

// Effect holds the effect size estimate.
type Effect struct {
	ShiftNs  float64       // Uniform shift component in nanoseconds
	TailNs   float64       // Tail effect component in nanoseconds
	CILowNs  float64       // 95% credible interval lower bound
	CIHighNs float64       // 95% credible interval upper bound
	Pattern  EffectPattern // Pattern of the effect
}

// Config holds the analysis configuration.
type Config struct {
	AttackerModel       AttackerModel
	CustomThresholdNs   float64
	MaxSamples          int
	TimeBudgetSecs      float64
	PassThreshold       float64
	FailThreshold       float64
	Seed                uint64
	TimerFrequencyHz    uint64
}

// DefaultConfig returns the default configuration.
func DefaultConfig() Config {
	return Config{
		AttackerModel:    AttackerAdjacentNetwork,
		MaxSamples:       100_000,
		TimeBudgetSecs:   30.0,
		PassThreshold:    0.05,
		FailThreshold:    0.95,
		TimerFrequencyHz: 1_000_000_000,
	}
}

func (c *Config) toC() C.togo_config_t {
	return C.togo_config_t{
		attacker_model:       C.togo_attacker_model_t(c.AttackerModel),
		custom_threshold_ns:  C.double(c.CustomThresholdNs),
		max_samples:          C.size_t(c.MaxSamples),
		time_budget_secs:     C.double(c.TimeBudgetSecs),
		pass_threshold:       C.double(c.PassThreshold),
		fail_threshold:       C.double(c.FailThreshold),
		seed:                 C.uint64_t(c.Seed),
		timer_frequency_hz:   C.uint64_t(c.TimerFrequencyHz),
	}
}

func configFromC(c *C.togo_config_t) Config {
	return Config{
		AttackerModel:       AttackerModel(c.attacker_model),
		CustomThresholdNs:   float64(c.custom_threshold_ns),
		MaxSamples:          int(c.max_samples),
		TimeBudgetSecs:      float64(c.time_budget_secs),
		PassThreshold:       float64(c.pass_threshold),
		FailThreshold:       float64(c.fail_threshold),
		Seed:                uint64(c.seed),
		TimerFrequencyHz:    uint64(c.timer_frequency_hz),
	}
}

// Result holds the analysis result.
type Result struct {
	Outcome            Outcome
	LeakProbability    float64
	Effect             Effect
	Quality            Quality
	SamplesUsed        int
	ElapsedSecs        float64
	Exploitability     Exploitability
	InconclusiveReason InconclusiveReason
	MDEShiftNs         float64
	MDETailNs          float64
	TimerResolutionNs  float64
	ThetaUserNs        float64
	ThetaEffNs         float64
	Recommendation     string
}

func resultFromC(r *C.togo_result_t) *Result {
	result := &Result{
		Outcome:            Outcome(r.outcome),
		LeakProbability:    float64(r.leak_probability),
		Effect: Effect{
			ShiftNs:  float64(r.effect.shift_ns),
			TailNs:   float64(r.effect.tail_ns),
			CILowNs:  float64(r.effect.ci_low_ns),
			CIHighNs: float64(r.effect.ci_high_ns),
			Pattern:  EffectPattern(r.effect.pattern),
		},
		Quality:            Quality(r.quality),
		SamplesUsed:        int(r.samples_used),
		ElapsedSecs:        float64(r.elapsed_secs),
		Exploitability:     Exploitability(r.exploitability),
		InconclusiveReason: InconclusiveReason(r.inconclusive_reason),
		MDEShiftNs:         float64(r.mde_shift_ns),
		MDETailNs:          float64(r.mde_tail_ns),
		TimerResolutionNs:  float64(r.timer_resolution_ns),
		ThetaUserNs:        float64(r.theta_user_ns),
		ThetaEffNs:         float64(r.theta_eff_ns),
	}

	if r.recommendation != nil {
		result.Recommendation = C.GoString(r.recommendation)
		// Free the Rust-allocated string
		C.togo_result_free(r)
	}

	return result
}
