//! Minimal test to verify oracle can detect injected delays
use std::time::Duration;

// Simple spin loop delay - no system calls
#[inline(never)]
fn spin_delay(iterations: u64) {
    for _ in 0..iterations {
        std::hint::spin_loop();
    }
}

fn main() {
    use timing_oracle::helpers::InputPair;
    use timing_oracle::{AttackerModel, Outcome, TimingOracle};

    let spin_iterations: u64 = 10000; // Should add ~10μs of delay

    println!("Testing oracle with {} spin iterations...", spin_iterations);

    let inputs = InputPair::new(|| false, || true);

    // Same parameters as quick tier calibration tests
    let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .max_samples(2000)
        .time_budget(Duration::from_secs(3))
        .test(inputs, |&should_delay| {
            if should_delay {
                spin_delay(spin_iterations);
            }
            std::hint::black_box(0);
        });

    match &outcome {
        Outcome::Pass {
            leak_probability,
            effect,
            ..
        } => {
            println!(
                "PASS: leak_probability={:.1}%, shift={:.1}ns, tail={:.1}ns",
                leak_probability * 100.0,
                effect.shift_ns,
                effect.tail_ns
            );
        }
        Outcome::Fail {
            leak_probability,
            effect,
            ..
        } => {
            println!(
                "FAIL (leak detected): leak_probability={:.1}%, shift={:.1}ns, tail={:.1}ns",
                leak_probability * 100.0,
                effect.shift_ns,
                effect.tail_ns
            );
        }
        Outcome::Inconclusive {
            reason,
            leak_probability,
            effect,
            ..
        } => {
            println!(
                "INCONCLUSIVE: {:?}, leak_probability={:.1}%, shift={:.1}ns, tail={:.1}ns",
                reason,
                leak_probability * 100.0,
                effect.shift_ns,
                effect.tail_ns
            );
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            println!("UNMEASURABLE: {}", recommendation);
        }
    }
}
