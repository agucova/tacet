use criterion::{black_box, criterion_group, criterion_main, Criterion};
use timing_oracle::{helpers::InputPair, TimingOracle};

fn bench_oracle_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("timing_oracle");
    group.sample_size(20);
    group.bench_function("baseline_addition", |b| {
        b.iter(|| {
            // Empty timing oracle run on trivial closures; keeps samples small to avoid long benches.
            let inputs = InputPair::new(|| 1u64, || 2u64);
            let result = TimingOracle::new()
                .samples(500)
                .test(inputs, |x| {
                    black_box(x + 1);
                });
            match result {
                timing_oracle::Outcome::Completed(r) => black_box(r.leak_probability),
                timing_oracle::Outcome::Unmeasurable { .. } => 0.0,
            }
        });
    });

    group.bench_function("balanced_preset", |b| {
        b.iter(|| {
            let inputs = InputPair::new(|| 3u64, || 4u64);
            let result = TimingOracle::balanced()
                .samples(500)
                .test(inputs, |x| {
                    black_box(x + 1);
                });
            match result {
                timing_oracle::Outcome::Completed(r) => black_box(r.ci_gate.passed),
                timing_oracle::Outcome::Unmeasurable { .. } => true,
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_oracle_simple);
criterion_main!(benches);
