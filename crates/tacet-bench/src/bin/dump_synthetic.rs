//! Dump raw synthetic AR(1) × pattern × seed samples to CSV for offline analysis.
//!
//! Writes one file per config in the same `class,timing_ns` schema used by
//! `crypto_benchmark --raw-samples-out`, so the same Python reader handles both
//! synthetic and real-hardware streams.
//!
//! Synthetic units are log-normal "timing units" (base_mu = 13.8, mean ≈ 992k).
//! Not nanoseconds — the analysis downstream uses scale-invariant statistics
//! (ACF, σ-normalised tails, Politis-White block length, IACT).

use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use tacet_bench::synthetic::{
    generate_benchmark_dataset, BenchmarkConfig, EffectPattern, NoiseModel,
};

#[derive(Parser, Debug)]
#[command(name = "dump_synthetic", about = "Dump synthetic AR(1) × pattern streams to CSV")]
struct Args {
    #[arg(long, default_value = "paper/author-response/synth-dump")]
    out_dir: PathBuf,

    #[arg(long, default_value_t = 10_000)]
    samples_per_class: usize,

    #[arg(long, default_value_t = 5)]
    seeds: u64,

    #[arg(long, default_value_t = 42)]
    base_seed: u64,
}

fn write_csv(path: &std::path::Path, baseline: &[u64], test: &[u64]) -> std::io::Result<()> {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p)?;
    }
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "class,timing_ns")?;
    for &v in baseline {
        writeln!(w, "baseline,{}", v)?;
    }
    for &v in test {
        writeln!(w, "test,{}", v)?;
    }
    w.flush()
}

fn main() {
    let args = Args::parse();
    fs::create_dir_all(&args.out_dir).expect("create out_dir");

    let patterns: [(EffectPattern, f64, &str); 3] = [
        (EffectPattern::Null, 0.0, "null"),
        (EffectPattern::Shift, 1.0, "shift-1sigma"),
        (EffectPattern::Tail, 1.0, "tail-1sigma"),
    ];
    let rhos: [f64; 4] = [0.0, 0.3, 0.6, 0.8];

    let mut count = 0usize;
    for (pattern, sigma_mult, pname) in patterns.iter().copied() {
        for &rho in &rhos {
            for seed_idx in 0..args.seeds {
                let noise = if rho == 0.0 {
                    NoiseModel::IID
                } else {
                    NoiseModel::AR1 { phi: rho }
                };
                let cfg = BenchmarkConfig {
                    samples_per_class: args.samples_per_class,
                    effect_pattern: pattern,
                    effect_sigma_mult: sigma_mult,
                    noise_model: noise,
                    seed: args.base_seed.wrapping_add(seed_idx),
                    ..BenchmarkConfig::default()
                };
                let ds = generate_benchmark_dataset(&cfg);
                let rho_str = format!("{:.1}", rho).replace('.', "p");
                let fname = format!("synth-{}-rho{}-seed{}.csv", pname, rho_str, seed_idx);
                let path = args.out_dir.join(&fname);
                write_csv(&path, &ds.blocked.baseline, &ds.blocked.test)
                    .expect("write csv");
                count += 1;
            }
        }
    }

    eprintln!(
        "wrote {} synthetic CSV files to {}",
        count,
        args.out_dir.display()
    );
}
