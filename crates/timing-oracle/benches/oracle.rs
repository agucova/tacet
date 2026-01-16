use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use timing_oracle::{helpers::InputPair, AttackerModel, Outcome, TimingOracle};

fn bench_oracle_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("timing_oracle");
    group.sample_size(20);
    group.bench_function("baseline_addition", |b| {
        b.iter(|| {
            // Empty timing oracle run on trivial closures; keeps samples small to avoid long benches.
            let inputs = InputPair::new(|| 1u64, || 2u64);
            let result = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
                .max_samples(500)
                .test(inputs, |x| {
                    black_box(x + 1);
                });
            match result {
                Outcome::Pass {
                    leak_probability, ..
                }
                | Outcome::Fail {
                    leak_probability, ..
                }
                | Outcome::Inconclusive {
                    leak_probability, ..
                } => black_box(leak_probability),
                Outcome::Unmeasurable { .. } => 0.0,
            }
        });
    });

    group.bench_function("adjacent_network_preset", |b| {
        b.iter(|| {
            let inputs = InputPair::new(|| 3u64, || 4u64);
            let result = TimingOracle::for_attacker(AttackerModel::AdjacentNetwork)
                .max_samples(500)
                .test(inputs, |x| {
                    black_box(x + 1);
                });
            match result {
                Outcome::Pass { .. } => black_box(true),
                Outcome::Fail { .. } | Outcome::Inconclusive { .. } => black_box(false),
                Outcome::Unmeasurable { .. } => true,
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_oracle_simple);
criterion_main!(benches);
