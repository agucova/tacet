//! Debug test to understand Bayesian calibration issue

use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};

#[test]
fn debug_bayesian_null_data() {
    const TRIALS: usize = 10;

    eprintln!("\n=== Bayesian Debug: {} trials on null data ===\n", TRIALS);

    for trial in 0..TRIALS {
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .time_budget(Duration::from_secs(10))
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        match outcome {
            Outcome::Pass { leak_probability, effect, .. } => {
                eprintln!("Trial {}: PASS leak_prob={:.1}%  effect={:.1}ns/{:.1}ns",
                    trial + 1,
                    leak_probability * 100.0,
                    effect.shift_ns,
                    effect.tail_ns
                );
            }
            Outcome::Fail { leak_probability, effect, .. } => {
                eprintln!("Trial {}: FAIL leak_prob={:.1}%  effect={:.1}ns/{:.1}ns",
                    trial + 1,
                    leak_probability * 100.0,
                    effect.shift_ns,
                    effect.tail_ns
                );
            }
            Outcome::Inconclusive { leak_probability, effect, .. } => {
                eprintln!("Trial {}: INCONCLUSIVE leak_prob={:.1}%  effect={:.1}ns/{:.1}ns",
                    trial + 1,
                    leak_probability * 100.0,
                    effect.shift_ns,
                    effect.tail_ns
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
