//! Test if indirect calls to different addresses cause false positives

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;
use timing_oracle::{helpers::InputPair, AttackerModel, TimingOracle};

fn main() {
    // Create two closures - identical code, different memory addresses
    let fixed_op: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {
        let mut acc = 42u64;
        for i in 0..50 {
            acc = acc.wrapping_add(black_box(i));
            for _ in 0..10 {
                acc = black_box(acc.wrapping_mul(3));
            }
        }
        black_box(acc ^ 0xDEADBEEF);
    });

    let random_op: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {
        let mut acc = 42u64;
        for i in 0..50 {
            acc = acc.wrapping_add(black_box(i));
            for _ in 0..10 {
                acc = black_box(acc.wrapping_mul(3));
            }
        }
        black_box(acc ^ 0xCAFEBABE);
    });

    // Wrapper for the Arc that implements Hash
    #[derive(Clone)]
    struct OpWrapper(Arc<dyn Fn() + Send + Sync>);

    impl std::hash::Hash for OpWrapper {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            Arc::as_ptr(&self.0).hash(state);
        }
    }

    let fixed = Arc::clone(&fixed_op);
    let random = Arc::clone(&random_op);

    let inputs = InputPair::new_unchecked(
        move || OpWrapper(Arc::clone(&fixed)),
        move || OpWrapper(Arc::clone(&random)),
    );

    println!("Testing adapter pattern (indirect calls to different addresses)...\n");

    let outcome = TimingOracle::for_attacker(AttackerModel::SharedHardware)
        .time_budget(Duration::from_secs(30))
        .max_samples(20000)
        .test(inputs, |op: &OpWrapper| {
            (op.0)(); // Indirect call - different target for baseline vs sample
        });

    match &outcome {
        timing_oracle::Outcome::Pass {
            leak_probability,
            samples_used,
            ..
        } => {
            println!(
                "PASS - leak_probability: {:.4}, samples: {}",
                leak_probability, samples_used
            );
        }
        timing_oracle::Outcome::Fail {
            leak_probability,
            effect,
            samples_used,
            ..
        } => {
            println!(
                "FAIL - leak_probability: {:.4}, samples: {}",
                leak_probability, samples_used
            );
            println!(
                "  effect: shift={:.2}ns, tail={:.2}ns",
                effect.shift_ns, effect.tail_ns
            );
        }
        timing_oracle::Outcome::Inconclusive {
            leak_probability,
            reason,
            samples_used,
            ..
        } => {
            println!(
                "INCONCLUSIVE - leak_probability: {:.4}, samples: {}",
                leak_probability, samples_used
            );
            println!("  reason: {:?}", reason);
        }
        timing_oracle::Outcome::Unmeasurable { recommendation, .. } => {
            println!("UNMEASURABLE: {}", recommendation);
        }
    }
}
