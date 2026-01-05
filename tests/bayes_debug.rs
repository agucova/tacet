//! Debug test to understand Bayesian calibration issue

use timing_oracle::helpers::InputPair;
use timing_oracle::{Outcome, TimingOracle};

#[test]
fn debug_bayesian_null_data() {
    const TRIALS: usize = 10;

    eprintln!("\n=== Bayesian Debug: {} trials on null data ===\n", TRIALS);

    for trial in 0..TRIALS {
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        let outcome = TimingOracle::quick()
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        match outcome {
            Outcome::Completed(result) => {
                eprintln!("Trial {}: leak_prob={:.1}%  BF={:.2e}  MDE={:.1}ns/{:.1}ns  effect={:?}",
                    trial + 1,
                    result.leak_probability * 100.0,
                    result.bayes_factor,
                    result.min_detectable_effect.shift_ns,
                    result.min_detectable_effect.tail_ns,
                    result.effect.as_ref().map(|e| format!("{:.1}ns/{:.1}ns", e.shift_ns, e.tail_ns))
                );
            }
            Outcome::Unmeasurable { operation_ns, threshold_ns, .. } => {
                eprintln!("Trial {}: UNMEASURABLE (op={:.1}ns, thresh={:.1}ns)",
                    trial + 1, operation_ns, threshold_ns);
            }
        }
    }
}

fn rand_bytes() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
