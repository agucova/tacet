//! Test that the library is safe to use concurrently from multiple threads.

use std::thread;
use std::time::Duration;
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};

#[test]
fn library_is_thread_safe() {
    // Spawn 4 threads, each running an oracle test
    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                let mut fixed = [0u8; 128];
                fixed[0] = i as u8;

                let inputs = InputPair::new(
                    move || fixed,
                    || {
                        let mut arr = [0u8; 128];
                        for byte in &mut arr {
                            *byte = rand::random();
                        }
                        arr
                    },
                );

                let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
                    .time_budget(Duration::from_secs(10))
                    .test(inputs, |arr| {
                    let mut acc = 0u32;
                    for byte in arr {
                        acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
                    }
                    std::hint::black_box(acc);
                });

                // Verify we got a result
                matches!(
                    outcome,
                    Outcome::Pass { .. }
                        | Outcome::Fail { .. }
                        | Outcome::Inconclusive { .. }
                        | Outcome::Unmeasurable { .. }
                )
            })
        })
        .collect();

    // All threads should complete successfully
    for handle in handles {
        assert!(handle.join().unwrap());
    }
}

#[test]
fn library_works_with_many_sequential_calls() {
    // Run 10 oracle tests sequentially - should never overflow
    // Use calibration() since we're just testing concurrency, not accuracy
    for i in 0..10 {
        let mut fixed = [0u8; 128];
        fixed[0] = i as u8;

        let inputs = InputPair::new(
            move || fixed,
            || {
                let mut arr = [0u8; 128];
                for byte in &mut arr {
                    *byte = rand::random();
                }
                arr
            },
        );

        let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .time_budget(Duration::from_secs(5))
            .test(inputs, |arr| {
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        });

        assert!(matches!(
            outcome,
            Outcome::Pass { .. }
                | Outcome::Fail { .. }
                | Outcome::Inconclusive { .. }
                | Outcome::Unmeasurable { .. }
        ));
    }
}
