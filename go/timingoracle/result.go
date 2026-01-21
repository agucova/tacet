package timingoracle

import (
	"fmt"
	"time"

	uniffi "github.com/agucova/timing-oracle/bindings/go/timing_oracle_uniffi"
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
	ReasonThresholdElevated
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
	case ReasonThresholdElevated:
		return "ThresholdElevated"
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
	// SharedHardwareOnly: < 10 ns - requires shared hardware (SGX, containers) to exploit.
	SharedHardwareOnly Exploitability = iota
	// HTTP2Multiplexing: 10-100 ns - exploitable via HTTP/2 request multiplexing.
	HTTP2Multiplexing
	// StandardRemote: 100 ns - 10 us - exploitable with standard remote timing.
	StandardRemote
	// ObviousLeak: > 10 us - obvious leak, trivially exploitable.
	ObviousLeak
)

// String returns the string representation of exploitability.
func (e Exploitability) String() string {
	switch e {
	case SharedHardwareOnly:
		return "SharedHardwareOnly"
	case HTTP2Multiplexing:
		return "HTTP2Multiplexing"
	case StandardRemote:
		return "StandardRemote"
	case ObviousLeak:
		return "ObviousLeak"
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

// Diagnostics holds detailed diagnostic information from the analysis.
// This provides insight into the statistical analysis quality and can
// be used for debugging or understanding measurement reliability.
type Diagnostics struct {
	// DependenceLength is the block length used for bootstrap resampling.
	DependenceLength int
	// EffectiveSampleSize accounts for autocorrelation in timing data.
	EffectiveSampleSize int
	// StationarityRatio is the ratio of post-test variance to calibration variance.
	StationarityRatio float64
	// StationarityOK indicates whether the stationarity check passed.
	StationarityOK bool
	// ProjectionMismatchQ is the projection mismatch Q statistic.
	ProjectionMismatchQ float64
	// ProjectionMismatchOK indicates whether projection mismatch is acceptable.
	ProjectionMismatchOK bool

	// DiscreteMode indicates whether discrete mode was used (low timer resolution).
	DiscreteMode bool
	// TimerResolutionNs is the timer resolution in nanoseconds.
	TimerResolutionNs float64

	// GibbsItersTotal is the total number of Gibbs sampler iterations.
	GibbsItersTotal int
	// GibbsBurnin is the number of burn-in iterations discarded.
	GibbsBurnin int
	// GibbsRetained is the number of samples retained after burn-in.
	GibbsRetained int
	// LambdaMean is the posterior mean of the latent scale parameter lambda.
	LambdaMean float64
	// LambdaSD is the posterior standard deviation of lambda.
	LambdaSD float64
	// LambdaCV is the coefficient of variation of lambda (SD/mean).
	LambdaCV float64
	// LambdaESS is the effective sample size of the lambda chain.
	LambdaESS float64
	// LambdaMixingOK indicates whether the lambda chain mixed well.
	LambdaMixingOK bool

	// KappaMean is the posterior mean of the likelihood precision kappa.
	KappaMean float64
	// KappaSD is the posterior standard deviation of kappa.
	KappaSD float64
	// KappaCV is the coefficient of variation of kappa.
	KappaCV float64
	// KappaESS is the effective sample size of the kappa chain.
	KappaESS float64
	// KappaMixingOK indicates whether the kappa chain mixed well.
	KappaMixingOK bool
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

	// ThetaFloorNs is the measurement floor (minimum detectable effect given noise).
	ThetaFloorNs float64

	// DecisionThresholdNs is the threshold at which the decision was made.
	DecisionThresholdNs float64

	// Recommendation is guidance for inconclusive/unmeasurable results.
	Recommendation string

	// Diagnostics contains detailed diagnostic information (nil if not available).
	Diagnostics *Diagnostics
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

// resultFromUniFFI converts from UniFFI AnalysisResult to public Result.
func resultFromUniFFI(r uniffi.AnalysisResult) *Result {
	result := &Result{
		Outcome:         outcomeFromUniFFI(r.Outcome),
		LeakProbability: r.LeakProbability,
		Effect: Effect{
			ShiftNs: r.Effect.ShiftNs,
			TailNs:  r.Effect.TailNs,
			CILow:   r.Effect.CredibleInterval.Low,
			CIHigh:  r.Effect.CredibleInterval.High,
			Pattern: effectPatternFromUniFFI(r.Effect.Pattern),
		},
		Quality:             qualityFromUniFFI(r.Quality),
		SamplesUsed:         int(r.SamplesUsed),
		ElapsedTime:         time.Duration(r.ElapsedSecs * float64(time.Second)),
		Exploitability:      exploitabilityFromUniFFI(r.Exploitability),
		InconclusiveReason:  inconclusiveReasonFromUniFFI(r.InconclusiveReason),
		MDEShiftNs:          r.MdeShiftNs,
		MDETailNs:           r.MdeTailNs,
		TimerResolutionNs:   r.TimerResolutionNs,
		ThetaUserNs:         r.ThetaUserNs,
		ThetaEffNs:          r.ThetaEffNs,
		ThetaFloorNs:        r.ThetaFloorNs,
		DecisionThresholdNs: r.DecisionThresholdNs,
		Recommendation:      r.Recommendation,
	}

	// Convert diagnostics (always available, value type in UniFFI)
	d := r.Diagnostics
	result.Diagnostics = &Diagnostics{
		DependenceLength:     int(d.DependenceLength),
		EffectiveSampleSize:  int(d.EffectiveSampleSize),
		StationarityRatio:    d.StationarityRatio,
		StationarityOK:       d.StationarityOk,
		ProjectionMismatchQ:  d.ProjectionMismatchQ,
		ProjectionMismatchOK: d.ProjectionMismatchOk,
		DiscreteMode:         d.DiscreteMode,
		TimerResolutionNs:    d.TimerResolutionNs,
		GibbsItersTotal:      int(d.GibbsItersTotal),
		GibbsBurnin:          int(d.GibbsBurnin),
		GibbsRetained:        int(d.GibbsRetained),
		LambdaMean:           d.LambdaMean,
		LambdaSD:             d.LambdaSd,
		LambdaCV:             d.LambdaCv,
		LambdaESS:            d.LambdaEss,
		LambdaMixingOK:       d.LambdaMixingOk,
		KappaMean:            d.KappaMean,
		KappaSD:              d.KappaSd,
		KappaCV:              d.KappaCv,
		KappaESS:             d.KappaEss,
		KappaMixingOK:        d.KappaMixingOk,
	}

	return result
}

// Helper conversion functions

func outcomeFromUniFFI(o uniffi.Outcome) Outcome {
	switch o {
	case uniffi.OutcomePass:
		return Pass
	case uniffi.OutcomeFail:
		return Fail
	case uniffi.OutcomeInconclusive:
		return Inconclusive
	case uniffi.OutcomeUnmeasurable:
		return Unmeasurable
	default:
		return Inconclusive
	}
}

func qualityFromUniFFI(q uniffi.MeasurementQuality) Quality {
	switch q {
	case uniffi.MeasurementQualityExcellent:
		return Excellent
	case uniffi.MeasurementQualityGood:
		return Good
	case uniffi.MeasurementQualityPoor:
		return Poor
	case uniffi.MeasurementQualityTooNoisy:
		return TooNoisy
	default:
		return Poor
	}
}

func exploitabilityFromUniFFI(e uniffi.Exploitability) Exploitability {
	switch e {
	case uniffi.ExploitabilitySharedHardwareOnly:
		return SharedHardwareOnly
	case uniffi.ExploitabilityHttp2Multiplexing:
		return HTTP2Multiplexing
	case uniffi.ExploitabilityStandardRemote:
		return StandardRemote
	case uniffi.ExploitabilityObviousLeak:
		return ObviousLeak
	default:
		return SharedHardwareOnly
	}
}

func effectPatternFromUniFFI(p uniffi.EffectPattern) EffectPattern {
	switch p {
	case uniffi.EffectPatternUniformShift:
		return UniformShift
	case uniffi.EffectPatternTailEffect:
		return TailEffect
	case uniffi.EffectPatternMixed:
		return Mixed
	case uniffi.EffectPatternIndeterminate:
		return Indeterminate
	default:
		return Indeterminate
	}
}

func inconclusiveReasonFromUniFFI(r uniffi.InconclusiveReason) InconclusiveReason {
	switch r.(type) {
	case uniffi.InconclusiveReasonNone:
		return ReasonNone
	case uniffi.InconclusiveReasonDataTooNoisy:
		return ReasonDataTooNoisy
	case uniffi.InconclusiveReasonNotLearning:
		return ReasonNotLearning
	case uniffi.InconclusiveReasonWouldTakeTooLong:
		return ReasonWouldTakeTooLong
	case uniffi.InconclusiveReasonTimeBudgetExceeded:
		return ReasonTimeBudgetExceeded
	case uniffi.InconclusiveReasonSampleBudgetExceeded:
		return ReasonSampleBudgetExceeded
	case uniffi.InconclusiveReasonConditionsChanged:
		return ReasonConditionsChanged
	case uniffi.InconclusiveReasonThresholdElevated:
		return ReasonThresholdElevated
	default:
		return ReasonNone
	}
}
