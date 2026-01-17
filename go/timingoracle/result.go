package timingoracle

import (
	"fmt"
	"time"

	"github.com/agucova/timing-oracle/go/timingoracle/internal/ffi"
)

// Outcome represents the test result.
type Outcome int

const (
	// Pass indicates no timing leak was detected within the threshold.
	Pass Outcome = iota
	// Fail indicates a timing leak was detected exceeding the threshold.
	Fail
	// Inconclusive indicates the test could not reach a decision.
	Inconclusive
	// Unmeasurable indicates the operation is too fast to measure reliably.
	Unmeasurable
)

// String returns the string representation of the outcome.
func (o Outcome) String() string {
	switch o {
	case Pass:
		return "Pass"
	case Fail:
		return "Fail"
	case Inconclusive:
		return "Inconclusive"
	case Unmeasurable:
		return "Unmeasurable"
	default:
		return "Unknown"
	}
}

// InconclusiveReason explains why a test was inconclusive.
type InconclusiveReason int

const (
	ReasonNone InconclusiveReason = iota
	ReasonDataTooNoisy
	ReasonNotLearning
	ReasonWouldTakeTooLong
	ReasonTimeBudgetExceeded
	ReasonSampleBudgetExceeded
	ReasonConditionsChanged
	ReasonThresholdUnachievable
	ReasonModelMismatch
)

// String returns the string representation of the reason.
func (r InconclusiveReason) String() string {
	switch r {
	case ReasonNone:
		return ""
	case ReasonDataTooNoisy:
		return "DataTooNoisy"
	case ReasonNotLearning:
		return "NotLearning"
	case ReasonWouldTakeTooLong:
		return "WouldTakeTooLong"
	case ReasonTimeBudgetExceeded:
		return "TimeBudgetExceeded"
	case ReasonSampleBudgetExceeded:
		return "SampleBudgetExceeded"
	case ReasonConditionsChanged:
		return "ConditionsChanged"
	case ReasonThresholdUnachievable:
		return "ThresholdUnachievable"
	case ReasonModelMismatch:
		return "ModelMismatch"
	default:
		return "Unknown"
	}
}

// EffectPattern describes the pattern of timing difference.
type EffectPattern int

const (
	// UniformShift indicates a constant timing difference across all quantiles.
	UniformShift EffectPattern = iota
	// TailEffect indicates timing difference concentrated in upper quantiles.
	TailEffect
	// Mixed indicates both shift and tail components present.
	Mixed
	// Indeterminate indicates the pattern cannot be determined.
	Indeterminate
)

// String returns the string representation of the pattern.
func (p EffectPattern) String() string {
	switch p {
	case UniformShift:
		return "UniformShift"
	case TailEffect:
		return "TailEffect"
	case Mixed:
		return "Mixed"
	case Indeterminate:
		return "Indeterminate"
	default:
		return "Unknown"
	}
}

// Exploitability assesses the practical exploitability of a detected leak.
type Exploitability int

const (
	// Negligible: < 100 ns - very difficult to exploit.
	Negligible Exploitability = iota
	// PossibleLAN: 100-500 ns - may be exploitable on LAN with many measurements.
	PossibleLAN
	// LikelyLAN: 500 ns - 20 us - likely exploitable on LAN.
	LikelyLAN
	// PossibleRemote: > 20 us - potentially exploitable over the internet.
	PossibleRemote
)

// String returns the string representation of exploitability.
func (e Exploitability) String() string {
	switch e {
	case Negligible:
		return "Negligible"
	case PossibleLAN:
		return "PossibleLAN"
	case LikelyLAN:
		return "LikelyLAN"
	case PossibleRemote:
		return "PossibleRemote"
	default:
		return "Unknown"
	}
}

// Quality assesses the measurement quality.
type Quality int

const (
	// Excellent: MDE < 5 ns - excellent measurement precision.
	Excellent Quality = iota
	// Good: MDE 5-20 ns - good precision for most use cases.
	Good
	// Poor: MDE 20-100 ns - limited precision.
	Poor
	// TooNoisy: MDE > 100 ns - too noisy for reliable detection.
	TooNoisy
)

// String returns the string representation of quality.
func (q Quality) String() string {
	switch q {
	case Excellent:
		return "Excellent"
	case Good:
		return "Good"
	case Poor:
		return "Poor"
	case TooNoisy:
		return "TooNoisy"
	default:
		return "Unknown"
	}
}

// Effect holds the effect size estimate.
type Effect struct {
	// ShiftNs is the uniform shift component in nanoseconds.
	ShiftNs float64
	// TailNs is the tail effect component in nanoseconds.
	TailNs float64
	// CILow is the 95% credible interval lower bound.
	CILow float64
	// CIHigh is the 95% credible interval upper bound.
	CIHigh float64
	// Pattern describes the effect pattern.
	Pattern EffectPattern
}

// TotalNs returns the total effect magnitude in nanoseconds.
func (e Effect) TotalNs() float64 {
	return e.ShiftNs + e.TailNs
}

// Result holds the complete analysis result.
type Result struct {
	// Outcome is the test result (Pass, Fail, Inconclusive, or Unmeasurable).
	Outcome Outcome

	// LeakProbability is P(max_k |(X*beta)_k| > theta | data).
	// For Pass: typically < 5%. For Fail: typically > 95%.
	LeakProbability float64

	// Effect is the estimated timing effect.
	Effect Effect

	// Quality is the measurement quality assessment.
	Quality Quality

	// SamplesUsed is the number of samples collected per class.
	SamplesUsed int

	// ElapsedTime is how long the test took.
	ElapsedTime time.Duration

	// Exploitability assesses practical exploitability (only meaningful for Fail).
	Exploitability Exploitability

	// InconclusiveReason explains why the test was inconclusive (if applicable).
	InconclusiveReason InconclusiveReason

	// MDEShiftNs is the minimum detectable shift effect in nanoseconds.
	MDEShiftNs float64

	// MDETailNs is the minimum detectable tail effect in nanoseconds.
	MDETailNs float64

	// TimerResolutionNs is the timer resolution in nanoseconds.
	TimerResolutionNs float64

	// ThetaUserNs is the user's requested threshold in nanoseconds.
	ThetaUserNs float64

	// ThetaEffNs is the effective threshold after floor adjustment.
	ThetaEffNs float64

	// Recommendation is guidance for inconclusive/unmeasurable results.
	Recommendation string
}

// IsConclusive returns true if the result is Pass or Fail.
func (r *Result) IsConclusive() bool {
	return r.Outcome == Pass || r.Outcome == Fail
}

// IsMeasurable returns true if the operation was measurable.
func (r *Result) IsMeasurable() bool {
	return r.Outcome != Unmeasurable
}

// String returns a human-readable summary of the result.
func (r *Result) String() string {
	switch r.Outcome {
	case Pass:
		return fmt.Sprintf("Pass: P(leak)=%.1f%%, effect=%.2fns, quality=%s, samples=%d",
			r.LeakProbability*100, r.Effect.TotalNs(), r.Quality, r.SamplesUsed)
	case Fail:
		return fmt.Sprintf("FAIL: P(leak)=%.1f%%, effect=%.2fns, exploitability=%s, samples=%d",
			r.LeakProbability*100, r.Effect.TotalNs(), r.Exploitability, r.SamplesUsed)
	case Inconclusive:
		return fmt.Sprintf("Inconclusive (%s): P(leak)=%.1f%%, samples=%d",
			r.InconclusiveReason, r.LeakProbability*100, r.SamplesUsed)
	case Unmeasurable:
		return fmt.Sprintf("Unmeasurable: %s", r.Recommendation)
	default:
		return "Unknown result"
	}
}

// fromFFI converts from FFI result to public Result.
func resultFromFFI(r *ffi.Result) *Result {
	return &Result{
		Outcome:            Outcome(r.Outcome),
		LeakProbability:    r.LeakProbability,
		Effect: Effect{
			ShiftNs: r.Effect.ShiftNs,
			TailNs:  r.Effect.TailNs,
			CILow:   r.Effect.CILowNs,
			CIHigh:  r.Effect.CIHighNs,
			Pattern: EffectPattern(r.Effect.Pattern),
		},
		Quality:            Quality(r.Quality),
		SamplesUsed:        r.SamplesUsed,
		ElapsedTime:        time.Duration(r.ElapsedSecs * float64(time.Second)),
		Exploitability:     Exploitability(r.Exploitability),
		InconclusiveReason: InconclusiveReason(r.InconclusiveReason),
		MDEShiftNs:         r.MDEShiftNs,
		MDETailNs:          r.MDETailNs,
		TimerResolutionNs:  r.TimerResolutionNs,
		ThetaUserNs:        r.ThetaUserNs,
		ThetaEffNs:         r.ThetaEffNs,
		Recommendation:     r.Recommendation,
	}
}
