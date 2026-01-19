//! Test external adapters (RTLF, SILENT, tlsfuzzer)

use timing_oracle_bench::{generate_dataset, EffectType, SyntheticConfig};
use timing_oracle_bench::{RtlfAdapter, SilentAdapter, TlsfuzzerAdapter, ToolAdapter};

fn main() {
    // Generate test data with a strong effect
    // RTLF requires at least 100 samples per class for bootstrap test
    let config = SyntheticConfig {
        samples_per_class: 1000,
        effect: EffectType::Shift { percent: 20.0 },
        seed: 42,
        ..Default::default()
    };
    let dataset = generate_dataset(&config);

    println!(
        "Testing external adapters with {} samples per class, 20% shift effect\n",
        config.samples_per_class
    );
    println!(
        "Dataset: {} baseline, {} test samples\n",
        dataset.blocked.baseline.len(),
        dataset.blocked.test.len()
    );

    println!("Testing RTLF adapter...");
    let rtlf = RtlfAdapter::default();
    let result = rtlf.analyze(&dataset);
    println!(
        "  RTLF: detected={}, status={}",
        result.detected_leak, result.status
    );

    println!("\nTesting SILENT adapter...");
    let silent = SilentAdapter::default();
    let result = silent.analyze(&dataset);
    println!(
        "  SILENT: detected={}, status={}",
        result.detected_leak, result.status
    );

    println!("\nTesting tlsfuzzer adapter...");
    let tlsfuzzer = TlsfuzzerAdapter::default();
    let result = tlsfuzzer.analyze(&dataset);
    println!(
        "  tlsfuzzer: detected={}, status={}",
        result.detected_leak, result.status
    );

    println!("\n--- Testing with NULL effect (should NOT detect) ---\n");

    let null_config = SyntheticConfig {
        samples_per_class: 1000,
        effect: EffectType::Null,
        seed: 123,
        ..Default::default()
    };
    let null_dataset = generate_dataset(&null_config);
    println!(
        "Null dataset: {} baseline, {} test samples\n",
        null_dataset.blocked.baseline.len(),
        null_dataset.blocked.test.len()
    );

    println!("Testing RTLF adapter (null)...");
    let result = rtlf.analyze(&null_dataset);
    println!(
        "  RTLF: detected={}, status={}",
        result.detected_leak, result.status
    );

    println!("\nTesting SILENT adapter (null)...");
    let result = silent.analyze(&null_dataset);
    println!(
        "  SILENT: detected={}, status={}",
        result.detected_leak, result.status
    );

    println!("\nTesting tlsfuzzer adapter (null)...");
    let result = tlsfuzzer.analyze(&null_dataset);
    println!(
        "  tlsfuzzer: detected={}, status={}",
        result.detected_leak, result.status
    );

    println!("\nDone!");
}
