//! Debug script to understand CI gate vs Bayesian discrepancy

use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use timing_oracle::helpers::InputPair;
use timing_oracle::{Outcome, TimingOracle};

fn rand_bytes_16() -> [u8; 16] {
    let mut arr = [0u8; 16];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}

fn main() {
    println!("=== AES-128 Timing Test Debug ===\n");

    let key = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f,
        0x3c,
    ];
    let cipher = Aes128::new(&key.into());

    let fixed_plaintext: [u8; 16] = [
        0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07,
        0x34,
    ];

    let inputs = InputPair::new(|| fixed_plaintext, rand_bytes_16);

    let outcome = TimingOracle::new()
        .samples(50_000)
        .alpha(0.01)
        .test(inputs, |plaintext| {
            let mut block = plaintext.to_owned().into();
            cipher.encrypt_block(&mut block);
            std::hint::black_box(block[0]);
        });

    match outcome {
        Outcome::Completed(result) => {
            println!("=== CI Gate ===");
            println!("Passed: {}", result.ci_gate.passed);
            println!("Alpha: {:.3}", result.ci_gate.alpha);
            println!("Threshold: {:.2} ns", result.ci_gate.threshold);
            println!("Max observed: {:.2} ns", result.ci_gate.max_observed);
            println!("Observed quantiles (ns):");
            for (i, q) in result.ci_gate.observed.iter().enumerate() {
                println!("  p{}: {:+.2}", (i + 1) * 10, q);
            }

            println!("\n=== Bayesian ===");
            println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
            println!("Bayes factor: {:.4}", result.bayes_factor);

            println!("\n=== Effect Estimate ===");
            if let Some(ref effect) = result.effect {
                println!("Shift: {:.2} ns", effect.shift_ns);
                println!("Tail: {:.2} ns", effect.tail_ns);
                println!("Pattern: {:?}", effect.pattern);
                println!(
                    "95% CI: ({:.2}, {:.2}) ns",
                    effect.credible_interval_ns.0, effect.credible_interval_ns.1
                );
            } else {
                println!("No effect (leak_probability <= 0.5 and ci_gate passed)");
            }

            println!("\n=== MDE & Prior ===");
            println!("MDE shift: {:.2} ns", result.min_detectable_effect.shift_ns);
            println!("MDE tail: {:.2} ns", result.min_detectable_effect.tail_ns);
            // Prior sigma = max(2*MDE, min_effect=10)
            let prior_sigma_shift = (2.0 * result.min_detectable_effect.shift_ns).max(10.0);
            let prior_sigma_tail = (2.0 * result.min_detectable_effect.tail_ns).max(10.0);
            println!("Prior sigma (shift): {:.2} ns", prior_sigma_shift);
            println!("Prior sigma (tail): {:.2} ns", prior_sigma_tail);
            println!("Theta (min_effect): 10.0 ns");

            // P(|N(0, sigma)| > theta) for the prior
            let z = 10.0 / prior_sigma_shift;
            let prior_exceedance = 2.0 * (1.0 - normal_cdf(z));
            println!("Prior P(|shift| > theta): {:.1}%", prior_exceedance * 100.0);

            println!("\n=== Diagnostics ===");
            println!("Quality: {:?}", result.quality);
            println!("Timer resolution: {:.2} ns", result.diagnostics.timer_resolution_ns);
            println!("Discrete mode: {}", result.diagnostics.discrete_mode);
            println!("Block length: {}", result.diagnostics.dependence_length);
            println!("Effective sample size: {}", result.diagnostics.effective_sample_size);
            println!("Stationarity ratio: {:.2}", result.diagnostics.stationarity_ratio);
            println!("Model fit chi2: {:.2}", result.diagnostics.model_fit_chi2);
            println!("Duplicate fraction: {:.1}%", result.diagnostics.duplicate_fraction * 100.0);

            println!("\n=== Analysis ===");
            if result.ci_gate.passed && result.leak_probability > 0.3 {
                println!("*** DISCREPANCY DETECTED ***");
                println!(
                    "CI gate passed but posterior is {:.1}%",
                    result.leak_probability * 100.0
                );

                // Key insight: if prior sigma is close to theta, the prior itself
                // has significant mass above theta, causing high posterior even with no leak
                if prior_sigma_shift <= 20.0 {
                    println!("\nPOTENTIAL ISSUE: Prior sigma ({:.1}ns) is close to theta (10ns)",
                             prior_sigma_shift);
                    println!("This means ~{:.0}% of prior mass exceeds theta,", prior_exceedance * 100.0);
                    println!("causing high posterior even when data shows no leak.");
                }
            } else if result.ci_gate.passed {
                println!("OK: CI gate passed, posterior is {:.1}% (< 30%)", result.leak_probability * 100.0);
            } else {
                println!("LEAK: CI gate failed, posterior is {:.1}%", result.leak_probability * 100.0);
            }
        }
        Outcome::Unmeasurable {
            operation_ns,
            threshold_ns,
            platform,
            recommendation,
        } => {
            println!("Unmeasurable: operation={:.2}ns < threshold={:.2}ns on {}",
                     operation_ns, threshold_ns, platform);
            println!("Recommendation: {}", recommendation);
        }
    }
}

fn normal_cdf(x: f64) -> f64 {
    // Approximation of normal CDF using tanh
    let a = 0.3480242;
    let b = 0.0958798;
    let c = 0.7478556;
    let t = 1.0 / (1.0 + 0.33267 * x.abs());
    let erf_approx = 1.0 - t * (a + t * (-b + t * c)) * (-x * x / 2.0).exp();
    if x >= 0.0 {
        0.5 * (1.0 + erf_approx)
    } else {
        0.5 * (1.0 - erf_approx)
    }
}
