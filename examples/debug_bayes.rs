use timing_oracle::{TimingOracle, Outcome, helpers::InputPair};

fn main() {
    let inputs = InputPair::new(
        || {
            let mut arr = [0u8; 32];
            for b in &mut arr { *b = rand::random(); }
            arr
        },
        || {
            let mut arr = [0u8; 32];
            for b in &mut arr { *b = rand::random(); }
            arr
        }
    );

    let outcome = TimingOracle::quick().test(inputs, |data| {
        std::hint::black_box(data);
    });

    match outcome {
        Outcome::Completed(r) => {
            println!("MDE: shift={:.2}ns, tail={:.2}ns", r.min_detectable_effect.shift_ns, r.min_detectable_effect.tail_ns);
            println!("Prior scales: min_effect={:.1}ns", 10.0);
            println!("Leak prob: {:.1}%", r.leak_probability * 100.0);
            println!("Bayes factor: {:.2e}", r.bayes_factor);
            if let Some(e) = r.effect {
                println!("Effect: shift={:.2}ns, tail={:.2}ns", e.shift_ns, e.tail_ns);
            }
        }
        Outcome::Unmeasurable { operation_ns, threshold_ns, platform, recommendation } => {
            eprintln!("UNMEASURABLE: operation={:.2}ns, threshold={:.2}ns, platform={}",
                     operation_ns, threshold_ns, platform);
            eprintln!("Recommendation: {}", recommendation);
            std::process::exit(1);
        }
    }
}
