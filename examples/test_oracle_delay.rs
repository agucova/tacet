//! Minimal test to verify oracle can detect injected delays
use std::time::{Duration, Instant};

// Simulate busy_wait_ns
#[inline(never)]
fn busy_wait_ns(ns: u64) {
    let start = Instant::now();
    let target = Duration::from_nanos(ns);
    while start.elapsed() < target {
        std::hint::spin_loop();
    }
}

fn main() {
    use timing_oracle::{TimingOracle, AttackerModel, Outcome};
    use timing_oracle::helpers::InputPair;
    
    let delay_ns: u64 = 5000; // 5μs delay - should be very detectable
    
    println!("Testing oracle with {}ns delay...", delay_ns);
    
    let inputs = InputPair::new(|| false, || true);
    
    let outcome = TimingOracle::for_attacker(AttackerModel::Research)
        .max_samples(10000)
        .time_budget(Duration::from_secs(30))
        .test(inputs, |&should_delay| {
            if should_delay {
                busy_wait_ns(delay_ns);
            }
            std::hint::black_box(0);
        });
    
    match &outcome {
        Outcome::Pass { leak_probability, effect, .. } => {
            println!("PASS: leak_probability={:.1}%, shift={:.1}ns, tail={:.1}ns", 
                leak_probability * 100.0, effect.shift_ns, effect.tail_ns);
        }
        Outcome::Fail { leak_probability, effect, .. } => {
            println!("FAIL (leak detected): leak_probability={:.1}%, shift={:.1}ns, tail={:.1}ns",
                leak_probability * 100.0, effect.shift_ns, effect.tail_ns);
        }
        Outcome::Inconclusive { reason, leak_probability, effect, .. } => {
            println!("INCONCLUSIVE: {:?}, leak_probability={:.1}%, shift={:.1}ns, tail={:.1}ns",
                reason, leak_probability * 100.0, effect.shift_ns, effect.tail_ns);
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            println!("UNMEASURABLE: {}", recommendation);
        }
    }
}
