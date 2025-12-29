//! Simple example demonstrating basic timing-oracle usage.

use timing_oracle::{test, TimingOracle};

fn main() {
    println!("timing-oracle simple example\n");

    // Example: Testing a potentially leaky comparison
    let secret = [0u8; 32];
    let fixed_input = [0u8; 32]; // Same as secret - might trigger timing leak
    let random_input = || {
        let mut arr = [0u8; 32];
        for i in 0..32 {
            arr[i] = rand::random();
        }
        arr
    };

    // Simple API with default config
    let result = test(
        || compare_bytes(&secret, &fixed_input),
        || compare_bytes(&secret, &random_input()),
    );

    println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
    println!("CI gate passed: {}", result.ci_gate.passed);
    println!("Quality: {:?}", result.quality);

    // Builder API with custom config
    let result = TimingOracle::new()
        .samples(10_000) // Fewer samples for quick demo
        .ci_alpha(0.01)
        .test(
            || compare_bytes(&secret, &fixed_input),
            || compare_bytes(&secret, &random_input()),
        );

    println!("\nWith custom config:");
    println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
}

/// Non-constant-time comparison (intentionally leaky for demo).
fn compare_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if a[i] != b[i] {
            return false; // Early exit - timing leak!
        }
    }
    true
}
