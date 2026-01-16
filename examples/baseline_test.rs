use std::time::Duration;
use timing_oracle::{AttackerModel, TimingOracle, helpers::InputPair};

fn main() {
    // Test 1: Pure XOR (should be perfectly constant-time)
    let inputs1 = InputPair::new(|| [0x00u8; 64], || [0xFFu8; 64]);
    let outcome1 = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .max_samples(30_000)
        .test(inputs1, |data| {
            let result = data.iter().fold(0u8, |acc, &b| acc ^ b);
            std::hint::black_box(result);
        });

    match outcome1 {
        timing_oracle::Outcome::Pass { leak_probability, effect, .. } => {
            println!("\n=== Pure XOR (baseline) ===");
            println!("Result: PASS");
            println!("Leak probability: {:.1}%", leak_probability * 100.0);
            println!("Effect: {:.1}ns", effect.shift_ns.abs() + effect.tail_ns.abs());
        }
        timing_oracle::Outcome::Fail { leak_probability, effect, exploitability, .. } => {
            println!("\n=== Pure XOR (baseline) ===");
            println!("Result: FAIL");
            println!("Leak probability: {:.1}%", leak_probability * 100.0);
            println!("Effect: {:.1}ns", effect.shift_ns.abs() + effect.tail_ns.abs());
            println!("Exploitability: {:?}", exploitability);
        }
        timing_oracle::Outcome::Inconclusive { leak_probability, effect, .. } => {
            println!("\n=== Pure XOR (baseline) ===");
            println!("Result: INCONCLUSIVE");
            println!("Leak probability: {:.1}%", leak_probability * 100.0);
            println!("Effect: {:.1}ns", effect.shift_ns.abs() + effect.tail_ns.abs());
        }
        timing_oracle::Outcome::Unmeasurable { .. } => {
            println!("\n=== Pure XOR (baseline) ===");
            println!("Result: UNMEASURABLE");
        }
    }

    // Test 2: Array copy (definitely constant-time)
    let inputs2 = InputPair::new(|| [0x00u8; 64], || [0xFFu8; 64]);
    let outcome2 = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
        .time_budget(Duration::from_secs(30))
        .max_samples(30_000)
        .test(inputs2, |data| {
            let mut output = [0u8; 64];
            output.copy_from_slice(data);
            std::hint::black_box(output[0]);
        });

    match outcome2 {
        timing_oracle::Outcome::Pass { leak_probability, effect, .. } => {
            println!("\n=== Array copy (baseline) ===");
            println!("Result: PASS");
            println!("Leak probability: {:.1}%", leak_probability * 100.0);
            println!("Effect: {:.1}ns", effect.shift_ns.abs() + effect.tail_ns.abs());
        }
        timing_oracle::Outcome::Fail { leak_probability, effect, exploitability, .. } => {
            println!("\n=== Array copy (baseline) ===");
            println!("Result: FAIL");
            println!("Leak probability: {:.1}%", leak_probability * 100.0);
            println!("Effect: {:.1}ns", effect.shift_ns.abs() + effect.tail_ns.abs());
            println!("Exploitability: {:?}", exploitability);
        }
        timing_oracle::Outcome::Inconclusive { leak_probability, effect, .. } => {
            println!("\n=== Array copy (baseline) ===");
            println!("Result: INCONCLUSIVE");
            println!("Leak probability: {:.1}%", leak_probability * 100.0);
            println!("Effect: {:.1}ns", effect.shift_ns.abs() + effect.tail_ns.abs());
        }
        timing_oracle::Outcome::Unmeasurable { .. } => {
            println!("\n=== Array copy (baseline) ===");
            println!("Result: UNMEASURABLE");
        }
    }
}
