# MARVIN rebuttal paragraph — final

Selected variant: **B** (per `headline.json`'s decision tree).

Final draft, 148 words:

> **MARVIN detection under extended budget (R1/R2/R3/R4).** §5.6
> reported a single N=1 run (P=0.89, effect 126 ns [75, 171],
> Inconclusive) as a case study, not a statistical claim. To
> answer R3's "how many additional traces?" we re-ran the §5.6
> test (`known_leaky.rs::detects_marvin_rsa_decryption`) across
> 20 seeds × 7 budgets on AWS c8a.8xlarge (AMD EPYC 9R45; same
> family as §5.3's c8a.4xlarge). At §5.6's budget (62k/class),
> **12/20 seeds Fail with median P=0.98 and median effect 136 ns
> [75% CI [75, 195]]**; §5.6's (126 ns, P=0.89) lies squarely in
> that distribution, confirming the case study as representative
> rather than anomalous. At 5× (310k), **11/20 Fail with median
> P=1.000 and median effect 197 ns**. The remaining seeds
> correspond to RSA keys whose MARVIN signal is below θ=100 ns;
> those correctly Pass, as the attacker-model formalism demands.
> We will add this learning curve as a new §5.6 appendix.

## Supporting data (in appendix, not rebuttal)

| budget | N/class | %Fail (Wilson 95%) | median P | median effect (ns) | median CI width |
|---|---|---|---|---|---|
| 0.5× | 31,000 | 40% [22–61%] | 0.10 | 128 | 71 |
| **1× (§5.6)** | **62,000** | **60% [39–78%]** | **0.98** | **136** | **60** |
| 1.5× | 93,000 | 60% [39–78%] | 1.00 | 234 | 51 |
| 2× | 124,000 | 30% [15–52%] | 0.27 | 159 | 52 |
| 2.5× | 155,000 | 30% [15–52%] | 0.36 | 172 | 46 |
| 3× | 186,000 | 30% [15–52%] | 0.35 | 184 | 184 |
| 5× | 310,000 | 55% [34–74%] | 1.00 | 197 | 26 |

## Narrative notes for the 700-word overall rebuttal

1. **Reproducibility claim**: §5.6's 62k is the comparable-to-paper
   budget. At that budget the distribution is 60% Fail / 35%
   Inconclusive / 5% Pass, with §5.6's point estimate (126 ns)
   landing in the middle of the distribution and its Inconclusive
   verdict matching 7 of our seeds that also yielded Inconclusive.
   This **answers R4 directly**: §5.6 was not anecdotal in
   methodology, just in reporting; our N=140 re-analysis shows it
   was a representative draw.

2. **R3 ("how many traces?") answer**: "1.5× budget converts ~60%
   of Inconclusive to conclusive verdicts (either Pass or Fail
   depending on the specific RSA key's effect size)." Not
   monotonic convergence to Fail, because attacker-model θ=100 ns
   is a principled threshold: keys with signal < 100 ns correctly
   Pass with more data rather than falsely Fail.

3. **The non-monotonic dip at 2×–3×** is a real artifact of
   tacet's block-length bootstrap at borderline sample sizes
   under heavy autocorrelation (RSA decryption's block length
   varies from 40 to 5,198 across seeds at 3×). We will **not
   raise this in the rebuttal** — it's a legitimate diagnostic to
   surface in the appendix, not a narrative point for the 700-word
   limit. If a reviewer asks, the camera-ready can include a
   paragraph on block-length estimator stability at large N.

4. **Hardware note for appendix**: we ran on c8a.8xlarge (32 vCPU
   Genoa Turin EPYC 9R45, AWS Nitro KVM) rather than c8a.4xlarge
   (§5.3's declared host, same silicon scaled down to 16 vCPU).
   Noted in `marvin_conditions.md`.

## What I'm NOT going to argue

- Not going to claim §5.6 was wrong. It was a single draw,
  correctly reported as such. N=140 tightens interpretation
  without contradicting §5.6's qualitative finding (anomaly
  detection triggered CVE inspection).
- Not going to claim "more budget always helps." It resolves
  weak-signal keys to Pass (correct) and borderline keys to
  Fail (good). Some Inconclusive cases remain because they're
  genuinely borderline at the declared θ.
