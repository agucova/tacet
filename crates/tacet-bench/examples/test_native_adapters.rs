//! Test all native adapters on synthetic data

use tacet_bench::{
    generate_dataset, AndersonDarlingAdapter, DudectAdapter, EffectType, KsTestAdapter,
    MonaAdapter, SyntheticConfig, TimingTvlaAdapter, ToolAdapter,
};

fn main() {
    println!("=== Testing All Native Adapters ===\n");

    // Generate test datasets
    let null_config = SyntheticConfig {
        samples_per_class: 2000,
        effect: EffectType::Null,
        seed: 42,
        ..Default::default()
    };
    let shift_config = SyntheticConfig {
        samples_per_class: 2000,
        effect: EffectType::Shift { percent: 15.0 },
        seed: 42,
        ..Default::default()
    };

    let null_data = generate_dataset(&null_config);
    let shift_data = generate_dataset(&shift_config);

    println!(
        "Null dataset: {} samples per class",
        null_config.samples_per_class
    );
    println!(
        "Shift dataset: {} samples per class, {}% shift\n",
        shift_config.samples_per_class, 15.0
    );

    // All native adapters
    let adapters: Vec<Box<dyn ToolAdapter>> = vec![
        Box::new(DudectAdapter::default()),
        Box::new(TimingTvlaAdapter::default()),
        Box::new(KsTestAdapter::default()),
        Box::new(AndersonDarlingAdapter::default()),
        Box::new(MonaAdapter::default()),
    ];

    println!("| Adapter      | Null (expect: pass)      | Shift 15% (expect: detect) |");
    println!("|--------------|--------------------------|----------------------------|");

    let mut all_correct = true;

    for adapter in &adapters {
        let null_result = adapter.analyze(&null_data);
        let shift_result = adapter.analyze(&shift_data);

        let null_ok = !null_result.detected_leak;
        let shift_ok = shift_result.detected_leak;

        let null_status = if null_ok {
            "✓ pass"
        } else {
            "✗ FALSE POSITIVE"
        };
        let shift_status = if shift_ok {
            "✓ detected"
        } else {
            "✗ MISSED"
        };

        println!(
            "| {:12} | {:24} | {:26} |",
            adapter.name(),
            null_status,
            shift_status
        );

        if !null_ok || !shift_ok {
            all_correct = false;
        }
    }

    println!("\n--- Detailed Results ---\n");

    for adapter in &adapters {
        let null_result = adapter.analyze(&null_data);
        let shift_result = adapter.analyze(&shift_data);
        println!("{:12}:", adapter.name());
        println!("  Null:  detected={}, {}", null_result.detected_leak, null_result.status);
        println!("  Shift: detected={}, {}", shift_result.detected_leak, shift_result.status);
    }

    if all_correct {
        println!("\n✓ All adapters working correctly!");
    } else {
        println!("\n✗ Some adapters had unexpected results");
        std::process::exit(1);
    }
}
