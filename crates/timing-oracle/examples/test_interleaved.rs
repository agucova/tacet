use std::time::Duration;
use timing_oracle::{helpers::InputPair, AttackerModel, TimingOracle};

fn main() {
    let inputs = InputPair::new(
        || false, // baseline class
        || true,  // sample class
    );

    let outcome = TimingOracle::for_attacker(AttackerModel::SharedHardware)
        .time_budget(Duration::from_secs(30))
        .max_samples(20000)
        .test(inputs, |is_sample| {
            // Same computation for both - use branchless select for final XOR
            let mut acc = 42u64;
            for i in 0..50 {
                acc = acc.wrapping_add(std::hint::black_box(i));
                for _ in 0..10 {
                    acc = std::hint::black_box(acc.wrapping_mul(3));
                }
            }
            // Branchless: mask is all 1s if sample, all 0s if baseline
            let mask = (*is_sample as u64).wrapping_neg();
            let xor_val = (mask & 0xCAFEBABE) | (!mask & 0xDEADBEEF);
            std::hint::black_box(acc ^ xor_val);
        });

    println!("\n{:#?}", outcome);
}
