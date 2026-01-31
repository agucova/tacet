package tacet

import (
	"fmt"
	"time"
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
	// MaxEffectNs is the maximum effect in nanoseconds: max_k |delta_k|.
	MaxEffectNs float64
	// CredibleIntervalNs is the 95% credible interval (lower, upper).
	CredibleIntervalNs [2]float64
	// TopQuantiles contains the top 2-3 quantiles by exceedance probability.
	TopQuantiles []TopQuantile
}

// TopQuantile represents a quantile with exceedance information.
type TopQuantile struct {
	// QuantileP is the quantile level (e.g., 0.9 for 90th percentile).
	QuantileP float64
	// MeanNs is the posterior mean delta_k in nanoseconds.
	MeanNs float64
	// CI95Ns is the 95% marginal credible interval (lower, upper).
	CI95Ns [2]float64
	// ExceedProb is P(|delta_k| > theta_eff | data).
	ExceedProb float64
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

	// DiscreteMode indicates whether discrete mode was used (low timer resolution).
	DiscreteMode bool
	// TimerResolutionNs is the timer resolution in nanoseconds.
	TimerResolutionNs float64

	// LambdaMean is the posterior mean of the latent scale parameter lambda.
	LambdaMean float64
	// LambdaSD is the posterior standard deviation of lambda.
	LambdaSD float64
	// LambdaESS is the effective sample size of the lambda chain.
	LambdaESS float64
	// LambdaMixingOK indicates whether the lambda chain mixed well.
	LambdaMixingOK bool

	// KappaMean is the posterior mean of the likelihood precision kappa.
	KappaMean float64
	// KappaCV is the coefficient of variation of kappa.
	KappaCV float64
	// KappaESS is the effective sample size of the kappa chain.
	KappaESS float64
	// KappaMixingOK indicates whether the kappa chain mixed well.
	KappaMixingOK bool
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

	// MDENs is the minimum detectable effect in nanoseconds.
	MDENs float64

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
		return fmt.Sprintf("Pass: P(leak)=%.1f%%, max_effect=%.2fns, quality=%s, samples=%d",
			r.LeakProbability*100, r.Effect.MaxEffectNs, r.Quality, r.SamplesUsed)
	case Fail:
		return fmt.Sprintf("FAIL: P(leak)=%.1f%%, max_effect=%.2fns, exploitability=%s, samples=%d",
			r.LeakProbability*100, r.Effect.MaxEffectNs, r.Exploitability, r.SamplesUsed)
	case Inconclusive:
		return fmt.Sprintf("Inconclusive (%s): P(leak)=%.1f%%, samples=%d",
			r.InconclusiveReason, r.LeakProbability*100, r.SamplesUsed)
	case Unmeasurable:
		return fmt.Sprintf("Unmeasurable: %s", r.Recommendation)
	default:
		return "Unknown result"
	}
}
