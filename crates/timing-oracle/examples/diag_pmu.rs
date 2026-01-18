use std::time::Duration;
use timing_oracle::{helpers::InputPair, AttackerModel, Outcome, TimingOracle};

fn main() {
    println!("=== Effect Injection with PMU Timer ===\n");

    for effect_ns in [0u64, 50, 100, 200, 500] {
        let inputs = InputPair::new(|| false, || true);

        let outcome = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
            .max_samples(5000)
            .time_budget(Duration::from_secs(10))
            .test(inputs, move |&should_delay| {
                busy_wait_ns(2000);
                if should_delay {
                    busy_wait_ns(effect_ns);
                }
            });

        match &outcome {
            Outcome::Pass {
                effect,
                diagnostics,
                ..
            } => {
                println!(
                    "Effect {:4}ns | Measured: {:7.1}ns | Timer: {:.1}ns | PASS",
                    effect_ns, effect.shift_ns, diagnostics.timer_resolution_ns
                );
            }
            Outcome::Fail {
                effect,
                diagnostics,
                ..
            } => {
                println!(
                    "Effect {:4}ns | Measured: {:7.1}ns | Timer: {:.1}ns | FAIL ✓",
                    effect_ns, effect.shift_ns, diagnostics.timer_resolution_ns
                );
            }
            Outcome::Inconclusive {
                effect,
                diagnostics,
                ..
            } => {
                println!(
                    "Effect {:4}ns | Measured: {:7.1}ns | Timer: {:.1}ns | INCONCLUSIVE",
                    effect_ns, effect.shift_ns, diagnostics.timer_resolution_ns
                );
            }
            _ => println!("Effect {:4}ns | UNMEASURABLE", effect_ns),
        }
    }
}

fn busy_wait_ns(ns: u64) {
    let ticks = (ns * 24 + 999) / 1000;
    let start: u64;
    unsafe {
        core::arch::asm!("mrs {}, cntvct_el0", out(reg) start);
    }
    loop {
        let now: u64;
        unsafe {
            core::arch::asm!("mrs {}, cntvct_el0", out(reg) now);
        }
        if now.wrapping_sub(start) >= ticks {
            break;
        }
        std::hint::spin_loop();
    }
}
