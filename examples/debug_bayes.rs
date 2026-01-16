//! Debug script to understand CI gate vs Bayesian discrepancy

use std::time::Duration;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};

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

    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .max_samples(50_000)
        .test(inputs, |plaintext| {
            let mut block = plaintext.to_owned().into();
            cipher.encrypt_block(&mut block);
            std::hint::black_box(block[0]);
        });

    match outcome {
        Outcome::Pass { leak_probability, effect, quality, diagnostics, .. } => {
            println!("=== PASS ===");
            println!("Leak probability: {:.1}%", leak_probability * 100.0);

            println!("\n=== Effect Estimate ===");
            println!("Shift: {:.2} ns", effect.shift_ns);
            println!("Tail: {:.2} ns", effect.tail_ns);
            println!("Pattern: {:?}", effect.pattern);
            println!(
                "95% CI: ({:.2}, {:.2}) ns",
                effect.credible_interval_ns.0, effect.credible_interval_ns.1
            );

            println!("\n=== Diagnostics ===");
            println!("Quality: {:?}", quality);
            println!("Timer resolution: {:.2} ns", diagnostics.timer_resolution_ns);
            println!("Discrete mode: {}", diagnostics.discrete_mode);
            println!("Block length: {}", diagnostics.dependence_length);
            println!("Effective sample size: {}", diagnostics.effective_sample_size);
            println!("Stationarity ratio: {:.2}", diagnostics.stationarity_ratio);
            println!("Model fit chi2: {:.2}", diagnostics.model_fit_chi2);
            println!("Duplicate fraction: {:.1}%", diagnostics.duplicate_fraction * 100.0);

            println!("\n=== Analysis ===");
            println!("OK: Test passed, posterior is {:.1}%", leak_probability * 100.0);
        }
        Outcome::Fail { leak_probability, effect, exploitability, quality, diagnostics, .. } => {
            println!("=== FAIL ===");
            println!("Leak probability: {:.1}%", leak_probability * 100.0);
            println!("Exploitability: {:?}", exploitability);

            println!("\n=== Effect Estimate ===");
            println!("Shift: {:.2} ns", effect.shift_ns);
            println!("Tail: {:.2} ns", effect.tail_ns);
            println!("Pattern: {:?}", effect.pattern);
            println!(
                "95% CI: ({:.2}, {:.2}) ns",
                effect.credible_interval_ns.0, effect.credible_interval_ns.1
            );

            println!("\n=== Diagnostics ===");
            println!("Quality: {:?}", quality);
            println!("Timer resolution: {:.2} ns", diagnostics.timer_resolution_ns);
            println!("Discrete mode: {}", diagnostics.discrete_mode);
            println!("Block length: {}", diagnostics.dependence_length);
            println!("Effective sample size: {}", diagnostics.effective_sample_size);
            println!("Stationarity ratio: {:.2}", diagnostics.stationarity_ratio);
            println!("Model fit chi2: {:.2}", diagnostics.model_fit_chi2);
            println!("Duplicate fraction: {:.1}%", diagnostics.duplicate_fraction * 100.0);

            println!("\n=== Analysis ===");
            println!("LEAK: Test failed, posterior is {:.1}%", leak_probability * 100.0);
        }
        Outcome::Inconclusive { reason, leak_probability, effect, quality, diagnostics, .. } => {
            println!("=== INCONCLUSIVE ===");
            println!("Reason: {:?}", reason);
            println!("Leak probability: {:.1}%", leak_probability * 100.0);

            println!("\n=== Effect Estimate ===");
            println!("Shift: {:.2} ns", effect.shift_ns);
            println!("Tail: {:.2} ns", effect.tail_ns);
            println!("Pattern: {:?}", effect.pattern);
            println!(
                "95% CI: ({:.2}, {:.2}) ns",
                effect.credible_interval_ns.0, effect.credible_interval_ns.1
            );

            println!("\n=== Diagnostics ===");
            println!("Quality: {:?}", quality);
            println!("Timer resolution: {:.2} ns", diagnostics.timer_resolution_ns);
            println!("Discrete mode: {}", diagnostics.discrete_mode);
            println!("Block length: {}", diagnostics.dependence_length);
            println!("Effective sample size: {}", diagnostics.effective_sample_size);
            println!("Stationarity ratio: {:.2}", diagnostics.stationarity_ratio);
            println!("Model fit chi2: {:.2}", diagnostics.model_fit_chi2);
            println!("Duplicate fraction: {:.1}%", diagnostics.duplicate_fraction * 100.0);
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
