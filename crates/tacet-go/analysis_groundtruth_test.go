package tacet

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/agucova/tacet/crates/tacet-go/internal/ffi"
)

// Ground-truth tests for the analysis half of the pipeline.
//
// These feed synthetic timing data with a known answer through the Go -> C
// boundary, so they are fully deterministic and never depend on the timing of
// the machine they run on. They are the regression net for class-label swaps,
// argument-order mistakes, enum mapping errors, and tick-to-nanosecond
// conversion errors.

// syntheticSamples builds two classes of timing data, in timer ticks. Each class
// is centred on the given tick count with uniform jitter in [0, jitter].
func syntheticSamples(n int, baselineTicks, sampleTicks, jitter uint64, seed uint64) (baseline, sample []uint64) {
	r := rand.New(rand.NewPCG(seed, seed^0xA5A5A5A5))
	baseline = make([]uint64, n)
	sample = make([]uint64, n)
	for i := 0; i < n; i++ {
		baseline[i] = baselineTicks + uint64(r.IntN(int(jitter)+1))
		sample[i] = sampleTicks + uint64(r.IntN(int(jitter)+1))
	}
	return baseline, sample
}

// TestGroundTruthIdenticalDistributions verifies that two draws from the same
// distribution are not reported as a leak.
func TestGroundTruthIdenticalDistributions(t *testing.T) {
	baseline, sample := syntheticSamples(5000, 500, 500, 20, 1)

	result, err := Analyze(baseline, sample, WithAttacker(AdjacentNetwork))
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}
	t.Logf("Result: %s", result)

	if result.Outcome == Fail {
		t.Errorf("Identical distributions reported as a leak: P(leak)=%.2f%%, effect=%.2f ns",
			result.LeakProbability*100, result.Effect.MaxEffectNs)
	}
	if result.LeakProbability > 0.5 {
		t.Errorf("Expected a low leak probability for identical distributions, got %.2f%%",
			result.LeakProbability*100)
	}
	// The posterior effect must be small relative to the threshold.
	if result.Effect.MaxEffectNs > result.ThetaEffNs {
		t.Errorf("Expected effect below theta_eff (%.2f ns), got %.2f ns",
			result.ThetaEffNs, result.Effect.MaxEffectNs)
	}
}

// TestGroundTruthInjectedShift verifies that a known constant shift is detected
// and that the reported effect matches the shift converted to nanoseconds. This
// is the check that catches tick-to-nanosecond conversion errors: the library is
// told the timer frequency, so a shift of N ticks must come back as
// N * (1e9 / frequency) nanoseconds.
func TestGroundTruthInjectedShift(t *testing.T) {
	nsPerTick := TimerResolutionNs()

	for _, shiftTicks := range []uint64{100, 504, 2000} {
		baseline, sample := syntheticSamples(5000, 500, 500+shiftTicks, 20, 2)

		result, err := Analyze(baseline, sample, WithAttacker(AdjacentNetwork))
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}
		expectedNs := float64(shiftTicks) * nsPerTick
		t.Logf("shift=%d ticks (%.1f ns): %s", shiftTicks, expectedNs, result)

		if result.Outcome != Fail {
			t.Errorf("shift=%d ticks: expected Fail, got %s (P(leak)=%.2f%%)",
				shiftTicks, result.Outcome, result.LeakProbability*100)
		}
		// Allow 10%: the posterior is shrunk toward the prior, and the jitter
		// contributes a little to W1.
		if relErr := math.Abs(result.Effect.MaxEffectNs-expectedNs) / expectedNs; relErr > 0.10 {
			t.Errorf("shift=%d ticks: expected effect ~%.1f ns, got %.1f ns (%.1f%% off)",
				shiftTicks, expectedNs, result.Effect.MaxEffectNs, relErr*100)
		}
	}
}

// TestGroundTruthShiftDirectionIsSymmetric verifies that it does not matter
// which class is slower: the reported effect is a magnitude. A class-label swap
// anywhere between the Go collector and the C analysis would show up here as an
// asymmetry.
func TestGroundTruthShiftDirectionIsSymmetric(t *testing.T) {
	slowSampleBaseline, slowSampleSample := syntheticSamples(5000, 500, 1004, 20, 3)
	slowBaselineBaseline, slowBaselineSample := syntheticSamples(5000, 1004, 500, 20, 3)

	forward, err := Analyze(slowSampleBaseline, slowSampleSample, WithAttacker(AdjacentNetwork))
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}
	reverse, err := Analyze(slowBaselineBaseline, slowBaselineSample, WithAttacker(AdjacentNetwork))
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}
	t.Logf("sample slower:   %s", forward)
	t.Logf("baseline slower: %s", reverse)

	if forward.Outcome != Fail || reverse.Outcome != Fail {
		t.Fatalf("Expected Fail in both directions, got %s and %s", forward.Outcome, reverse.Outcome)
	}
	if forward.Effect.MaxEffectNs <= 0 || reverse.Effect.MaxEffectNs <= 0 {
		t.Errorf("Effect sizes must be positive magnitudes, got %.2f and %.2f",
			forward.Effect.MaxEffectNs, reverse.Effect.MaxEffectNs)
	}
	rel := math.Abs(forward.Effect.MaxEffectNs-reverse.Effect.MaxEffectNs) / forward.Effect.MaxEffectNs
	if rel > 0.05 {
		t.Errorf("Effect should not depend on which class is slower: %.1f ns vs %.1f ns",
			forward.Effect.MaxEffectNs, reverse.Effect.MaxEffectNs)
	}
}

// TestGroundTruthAdaptiveLoop runs the same synthetic data through the
// calibrate/step path used by Test(), rather than the one-shot Analyze path.
func TestGroundTruthAdaptiveLoop(t *testing.T) {
	cfg := defaultConfig()
	ffiCfg := cfg.toFFI()

	run := func(name string, baselineTicks, sampleTicks uint64) *Result {
		t.Helper()
		calBaseline, calSample := syntheticSamples(5000, baselineTicks, sampleTicks, 20, 4)

		calibration, err := ffi.Calibrate(calBaseline, calSample, ffiCfg)
		if err != nil {
			t.Fatalf("%s: calibrate failed: %v", name, err)
		}
		defer calibration.Free()

		state := ffi.NewState()
		if state == nil {
			t.Fatalf("%s: could not create adaptive state", name)
		}
		defer state.Free()

		batchBaseline := calBaseline
		batchSample := calSample
		for step := 0; step < 10; step++ {
			stepResult, err := ffi.Step(calibration, state, batchBaseline, batchSample, ffiCfg, 0.1*float64(step+1))
			if err != nil {
				t.Fatalf("%s: step failed: %v", name, err)
			}
			if stepResult.HasDecision {
				return resultFromFFI(&stepResult.Result, 1)
			}
			batchBaseline, batchSample = syntheticSamples(1000, baselineTicks, sampleTicks, 20, uint64(10+step))
		}
		t.Fatalf("%s: no decision after 10 steps", name)
		return nil
	}

	identical := run("identical", 500, 500)
	t.Logf("identical:  %s", identical)
	if identical.Outcome == Fail {
		t.Errorf("Identical distributions reported as a leak through the adaptive loop: %s", identical)
	}

	shifted := run("shifted", 500, 1004)
	t.Logf("shifted:    %s", shifted)
	if shifted.Outcome != Fail {
		t.Errorf("Expected Fail for a 504 tick shift through the adaptive loop, got %s", shifted)
	}
	expectedNs := 504 * TimerResolutionNs()
	if relErr := math.Abs(shifted.Effect.MaxEffectNs-expectedNs) / expectedNs; relErr > 0.10 {
		t.Errorf("Expected effect ~%.1f ns, got %.1f ns", expectedNs, shifted.Effect.MaxEffectNs)
	}
}

// TestGroundTruthAttackerEnums round-trips every attacker model through the C
// ABI and checks the threshold that comes back. A mismatch between the Go
// constants and the C enum values would return the wrong threshold here, which
// is the failure mode that makes results look inverted.
func TestGroundTruthAttackerEnums(t *testing.T) {
	cases := []struct {
		model    AttackerModel
		ffiModel ffi.AttackerModel
		expected float64
	}{
		{SharedHardware, ffi.SharedHardware, 0.4},
		{PostQuantum, ffi.PostQuantum, 2.0},
		{AdjacentNetwork, ffi.AdjacentNetwork, 100.0},
		{RemoteNetwork, ffi.RemoteNetwork, 50_000.0},
		{Research, ffi.Research, 0.0},
	}
	for _, tc := range cases {
		if int(tc.model) != int(tc.ffiModel) {
			t.Errorf("%s: Go constant %d does not match FFI constant %d", tc.model, tc.model, tc.ffiModel)
		}
		got := ffi.AttackerThresholdNs(tc.ffiModel)
		if math.Abs(got-tc.expected) > 1e-9 {
			t.Errorf("%s: C API reports threshold %.4f ns, expected %.4f ns", tc.model, got, tc.expected)
		}
		if math.Abs(tc.model.ThresholdNs()-tc.expected) > 1e-9 {
			t.Errorf("%s: Go reports threshold %.4f ns, expected %.4f ns", tc.model, tc.model.ThresholdNs(), tc.expected)
		}
	}
}

// TestGroundTruthResultEnums checks that the Go result enums line up with the
// FFI ones, so an off-by-one cannot silently turn a Pass into a Fail.
func TestGroundTruthResultEnums(t *testing.T) {
	if int(Pass) != int(ffi.Pass) || int(Fail) != int(ffi.Fail) ||
		int(Inconclusive) != int(ffi.Inconclusive) || int(Unmeasurable) != int(ffi.Unmeasurable) {
		t.Errorf("Outcome constants diverge: Go(%d,%d,%d,%d) FFI(%d,%d,%d,%d)",
			Pass, Fail, Inconclusive, Unmeasurable, ffi.Pass, ffi.Fail, ffi.Inconclusive, ffi.Unmeasurable)
	}
	if int(Excellent) != int(ffi.Excellent) || int(Good) != int(ffi.Good) ||
		int(Poor) != int(ffi.Poor) || int(TooNoisy) != int(ffi.TooNoisy) {
		t.Error("Quality constants diverge between the Go and FFI layers")
	}
	if int(SharedHardwareOnly) != int(ffi.SharedHardwareOnly) || int(HTTP2Multiplexing) != int(ffi.HTTP2Multiplexing) ||
		int(StandardRemote) != int(ffi.StandardRemote) || int(ObviousLeak) != int(ffi.ObviousLeak) {
		t.Error("Exploitability constants diverge between the Go and FFI layers")
	}
	if int(ReasonNone) != int(ffi.ReasonNone) || int(ReasonDataTooNoisy) != int(ffi.ReasonDataTooNoisy) ||
		int(ReasonThresholdElevated) != int(ffi.ReasonThresholdElevated) {
		t.Error("InconclusiveReason constants diverge between the Go and FFI layers")
	}
}

// TestGroundTruthBatchEffectScaling checks that effect sizes are reported per
// call rather than per batch when batching is in use.
func TestGroundTruthBatchEffectScaling(t *testing.T) {
	r := &ffi.Result{
		Outcome:         ffi.Fail,
		LeakProbability: 1.0,
		Effect:          ffi.Effect{MaxEffectNs: 2000, CILow: 1800, CIHigh: 2200},
	}
	single := resultFromFFI(r, 1)
	batched := resultFromFFI(r, 20)

	if single.Effect.MaxEffectNs != 2000 {
		t.Errorf("Expected an unbatched effect of 2000 ns, got %.2f", single.Effect.MaxEffectNs)
	}
	if batched.Effect.MaxEffectNs != 100 {
		t.Errorf("Expected a batch of 20 to report 100 ns per call, got %.2f", batched.Effect.MaxEffectNs)
	}
	if batched.Effect.CredibleIntervalNs != [2]float64{90, 110} {
		t.Errorf("Credible interval must be scaled with the effect, got %v", batched.Effect.CredibleIntervalNs)
	}
}
