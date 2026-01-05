use timing_oracle::{TimingOracle, helpers::InputPair};

fn main() {
    // Test 1: Pure XOR (should be perfectly constant-time)
    let inputs1 = InputPair::new(|| [0x00u8; 64], || [0xFFu8; 64]);
    let outcome1 = TimingOracle::balanced()
        .samples(30_000)
        .test(inputs1, |data| {
            let result = data.iter().fold(0u8, |acc, &b| acc ^ b);
            std::hint::black_box(result);
        });

    if let timing_oracle::Outcome::Completed(r) = outcome1 {
        println!("\n=== Pure XOR (baseline) ===");
        println!("Leak probability: {:.1}%", r.leak_probability * 100.0);
        if let Some(e) = &r.effect {
            println!("Effect: {:.1}ns {:?}", e.shift_ns.abs() + e.tail_ns.abs(), e.pattern);
        }
        println!("Exploitability: {:?}", r.exploitability);
    }

    // Test 2: Array copy (definitely constant-time)
    let inputs2 = InputPair::new(|| [0x00u8; 64], || [0xFFu8; 64]);
    let outcome2 = TimingOracle::balanced()
        .samples(30_000)
        .test(inputs2, |data| {
            let mut output = [0u8; 64];
            output.copy_from_slice(data);
            std::hint::black_box(output[0]);
        });

    if let timing_oracle::Outcome::Completed(r) = outcome2 {
        println!("\n=== Array copy (baseline) ===");
        println!("Leak probability: {:.1}%", r.leak_probability * 100.0);
        if let Some(e) = &r.effect {
            println!("Effect: {:.1}ns {:?}", e.shift_ns.abs() + e.tail_ns.abs(), e.pattern);
        }
        println!("Exploitability: {:?}", r.exploitability);
    }
}
