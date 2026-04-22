//! MARVIN budget-scaling sweep driver.
//!
//! Single-case, single-pass. Runs `tier2_rustcrypto_rsa_marvin` (from
//! `tacet_bench::crypto_registry::Tier::Two`) at a caller-supplied
//! `samples_per_class` budget, hands the collected raw samples to
//! `TimingOracle::analyze_raw_samples_with_resolution`, and emits one
//! CSV row with verdict, posterior probability, effect + CI, block
//! length, ESS, and timings.
//!
//! This binary is intentionally narrow: it exists to produce a learning
//! curve for the USENIX Sec '26 rebuttal (§5.6 MARVIN case study). It
//! uses the same MARVIN test case as the cross-tool harness but is a
//! separate entry point so the cross-tool binary (which enforces fixed
//! N across tools for fairness) stays untouched.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;

use tacet::{AttackerModel, Outcome, TimingOracle};
use tacet_bench::crypto_registry::Tier;
use tacet_bench::crypto_collect::{run_collection, CollectedBlocked};

/// One MARVIN run at a given (seed, sample budget).
#[derive(Parser, Debug)]
#[command(name = "marvin_budget_sweep", about = "MARVIN budget-scaling sweep")]
struct Args {
    /// Samples per class for this run (controls the budget axis).
    #[arg(long)]
    samples_per_class: usize,

    /// Per-iteration seed (usually `hash(base_seed, "marvin", budget_label, iter)`).
    #[arg(long)]
    seed: u64,

    /// Human-readable budget label written verbatim into the CSV (e.g. "3.0x").
    #[arg(long)]
    budget_label: String,

    /// Optional iteration index (written to CSV; useful for sweep harness).
    #[arg(long, default_value_t = 0)]
    iteration: usize,

    /// Output CSV path. Appended to; header written if file is new.
    #[arg(long)]
    output: PathBuf,

    /// Skip if a row with matching (budget_label, iteration, seed) already
    /// exists in the CSV. Makes the sweep idempotent / resumable.
    #[arg(long)]
    resume: bool,

    /// Override attacker model for this run. Accepted: "adjacent" | "shared" |
    /// "pq" | "remote" | "research". Default: the MARVIN case's
    /// AdjacentNetwork, matching §5.6.
    #[arg(long)]
    attacker_model: Option<String>,

    /// Which MARVIN input pattern to use.
    /// - `cache` (default): baseline = one fixed valid ciphertext (repeated);
    ///   sample = pool of varied valid ciphertexts. This is the pattern used
    ///   in `crates/tacet/tests/core/known_leaky.rs::detects_marvin_rsa_decryption`
    ///   and matches §5.6's measurement. The effect is dominated by cache /
    ///   branch-predictor state differences between fixed and varied inputs.
    /// - `padding`: baseline = pool of valid ciphertexts; sample = pool of
    ///   random (invalid-padding) ciphertexts. Matches the cross-tool
    ///   `tier2_rustcrypto_rsa_marvin` registry entry and exercises the
    ///   Bleichenbacher padding-check short-circuit path.
    #[arg(long, default_value = "cache")]
    marvin_mode: String,

    /// If set, run tacet's adaptive path (`TimingOracle::test`) with
    /// `max_samples = --samples-per-class` and `time_budget = 600s`, rather
    /// than the default fixed-n single-pass path. Only implemented for
    /// `--marvin-mode cache`. Writes the same CSV schema but the
    /// `samples_used` column reflects adaptive termination (≤ the budget).
    #[arg(long, default_value_t = false)]
    adaptive: bool,
}

const CSV_HEADER: &str = "timestamp,budget_label,iteration,seed,samples_requested,samples_used,\
timer_name,timer_resolution_ns,collection_ms,analysis_ms,\
verdict,leak_probability,effect_ns,ci_lo_ns,ci_hi_ns,\
dependence_length,effective_sample_size,stationarity_ratio,calibration_samples,\
attacker_model,threshold_ns";

/// Resume key format: "budget_label|iteration|seed".
fn resume_key(budget_label: &str, iteration: usize, seed: u64) -> String {
    format!("{}|{}|{}", budget_label, iteration, seed)
}

fn row_already_present(path: &PathBuf, budget_label: &str, iteration: usize, seed: u64) -> bool {
    let Ok(f) = File::open(path) else {
        return false;
    };
    let target = resume_key(budget_label, iteration, seed);
    let reader = BufReader::new(f);
    for (i, line) in reader.lines().flatten().enumerate() {
        if i == 0 {
            continue; // header
        }
        // budget_label is column index 1, iteration is 2, seed is 3.
        let mut cols = line.split(',');
        let _ts = cols.next();
        let bl = cols.next().unwrap_or("");
        let it = cols.next().unwrap_or("");
        let sd = cols.next().unwrap_or("");
        if format!("{}|{}|{}", bl, it, sd) == target {
            return true;
        }
    }
    false
}

fn iso_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    secs.to_string()
}

/// Adaptive-path MARVIN run: mirrors
/// `crates/tacet/tests/core/known_leaky.rs::detects_marvin_rsa_decryption`
/// but with caller-supplied `max_samples` (= `samples_per_class`) so the
/// production-mode adaptive loop terminates by sample budget.
///
/// Cache-warming pattern (baseline = 1 fixed valid ct; sample = 200 varied).
fn run_adaptive_and_write(args: &Args, model: AttackerModel) {
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use rsa::rand_core::OsRng as RsaOsRng;
    use rsa::{Pkcs1v15Encrypt, RsaPrivateKey, RsaPublicKey};
    use std::cell::Cell;
    use std::hint::black_box;
    use tacet::helpers::InputPair;

    let private_key = RsaPrivateKey::new(&mut RsaOsRng, 1024).expect("RSA keygen failed");
    let public_key = RsaPublicKey::from(&private_key);

    let fixed_message = [0x42u8; 32];
    let fixed_ciphertext = public_key
        .encrypt(&mut RsaOsRng, Pkcs1v15Encrypt, &fixed_message)
        .expect("encrypt fixed failed");

    const POOL_SIZE: usize = 200;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let varied_pool: Vec<Vec<u8>> = (0..POOL_SIZE)
        .map(|_| {
            let mut msg = [0u8; 32];
            rng.fill_bytes(&mut msg);
            public_key
                .encrypt(&mut RsaOsRng, Pkcs1v15Encrypt, &msg)
                .expect("encrypt varied failed")
        })
        .collect();

    let fixed_ct = fixed_ciphertext.clone();
    let sample_idx = Cell::new(0usize);
    let varied = varied_pool.clone();
    let inputs = InputPair::new(
        move || fixed_ct.clone(),
        move || {
            let i = sample_idx.get();
            sample_idx.set((i + 1) % POOL_SIZE);
            varied[i].clone()
        },
    );

    let t_start = Instant::now();
    let outcome = TimingOracle::for_attacker(model)
        .pass_threshold(0.01)
        .fail_threshold(0.95)
        .time_budget(Duration::from_secs(600))
        .max_samples(args.samples_per_class)
        .warmup(500)
        .calibration_samples(10_000)
        .test(inputs, |ct| {
            let plaintext = private_key.decrypt(Pkcs1v15Encrypt, ct).unwrap();
            black_box(plaintext[0]);
        });
    let wall_ms = t_start.elapsed().as_millis() as u64;

    let (
        verdict,
        leak_prob,
        effect_ns,
        ci_lo,
        ci_hi,
        samples_used,
        dep_len,
        ess,
        stat_ratio,
        calib_samples,
        threshold_ns,
    ) = match &outcome {
        Outcome::Pass {
            leak_probability,
            effect,
            samples_used,
            diagnostics,
            ..
        }
        | Outcome::Fail {
            leak_probability,
            effect,
            samples_used,
            diagnostics,
            ..
        }
        | Outcome::Inconclusive {
            leak_probability,
            effect,
            samples_used,
            diagnostics,
            ..
        } => (
            outcome_label(&outcome),
            Some(*leak_probability),
            Some(effect.max_effect_ns),
            Some(effect.credible_interval_ns.0),
            Some(effect.credible_interval_ns.1),
            *samples_used,
            Some(diagnostics.dependence_length),
            Some(diagnostics.effective_sample_size),
            Some(diagnostics.stationarity_ratio),
            Some(diagnostics.calibration_samples),
            Some(diagnostics.threshold_ns),
        ),
        Outcome::Unmeasurable { .. } => (
            "unmeasurable", None, None, None, None, 0, None, None, None, None, None,
        ),
        Outcome::Research(_) => (
            "research", None, None, None, None, 0, None, None, None, None, None,
        ),
    };

    eprintln!(
        "  [adaptive] verdict={} P={} effect={} CI=[{},{}] samples_used={} block={} ESS={} wall={}ms",
        verdict,
        fmt_opt_f64(leak_prob, 6),
        fmt_opt_f64(effect_ns, 3),
        fmt_opt_f64(ci_lo, 3),
        fmt_opt_f64(ci_hi, 3),
        samples_used,
        fmt_opt_usize(dep_len),
        fmt_opt_usize(ess),
        wall_ms,
    );

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output dir");
        }
    }
    let fresh = !args.output.exists();
    let mut out = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.output)
            .expect("open output CSV"),
    );
    if fresh {
        writeln!(out, "{}", CSV_HEADER).expect("write header");
    }
    writeln!(
        out,
        "{ts},{bl},{it},{sd},{n_req},{n_used},adaptive,0.0,0,{wms},{v},{p},{e},{lo},{hi},{dep},{ess},{sr},{cs},{am},{th}",
        ts = iso_timestamp(),
        bl = args.budget_label,
        it = args.iteration,
        sd = args.seed,
        n_req = args.samples_per_class,
        n_used = samples_used,
        wms = wall_ms,
        v = verdict,
        p = fmt_opt_f64(leak_prob, 6),
        e = fmt_opt_f64(effect_ns, 3),
        lo = fmt_opt_f64(ci_lo, 3),
        hi = fmt_opt_f64(ci_hi, 3),
        dep = fmt_opt_usize(dep_len),
        ess = fmt_opt_usize(ess),
        sr = fmt_opt_f64(stat_ratio, 4),
        cs = fmt_opt_usize(calib_samples),
        am = format_attacker_model(model),
        th = fmt_opt_f64(threshold_ns, 3),
    )
    .expect("write CSV row");
    out.flush().ok();
}

/// §5.6's MARVIN pattern (cache/branch-predictor warming variant).
///
/// Mirrors `crates/tacet/tests/core/known_leaky.rs::detects_marvin_rsa_decryption`:
///   baseline = one fixed valid ciphertext, reused for every measurement;
///   sample   = pool of 200 distinct valid ciphertexts, cycled.
///
/// The leak signature is driven by cache / branch-predictor state differences
/// between the repeated-fixed input and the varied-valid inputs, both of
/// which decrypt successfully (no padding-check divergence).
fn collect_marvin_cache(seed: u64, samples_per_class: usize) -> CollectedBlocked {
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use rsa::rand_core::OsRng as RsaOsRng;
    use rsa::{Pkcs1v15Encrypt, RsaPrivateKey, RsaPublicKey};
    use std::cell::Cell;
    use std::hint::black_box;
    use tacet::Class;

    // Key generation is non-deterministic (uses RsaOsRng) — an independent
    // key per (seed) call is not achievable without reaching into the crate
    // internals. We keep the per-call key generation to match
    // `known_leaky.rs`'s approach; per-iteration variance stems from the
    // sample-pool selection which IS seeded.
    let private_key = RsaPrivateKey::new(&mut RsaOsRng, 1024).expect("RSA keygen failed");
    let public_key = RsaPublicKey::from(&private_key);

    // Baseline: one fixed valid ciphertext, repeated.
    let fixed_message = [0x42u8; 32];
    let fixed_ciphertext = public_key
        .encrypt(&mut RsaOsRng, Pkcs1v15Encrypt, &fixed_message)
        .expect("encrypt fixed failed");

    // Sample: pool of 200 distinct valid ciphertexts. Deterministic under `seed`.
    const POOL_SIZE: usize = 200;
    let mut rng = StdRng::seed_from_u64(seed);
    let varied_pool: Vec<Vec<u8>> = (0..POOL_SIZE)
        .map(|_| {
            let mut msg = [0u8; 32];
            rng.fill_bytes(&mut msg);
            public_key
                .encrypt(&mut RsaOsRng, Pkcs1v15Encrypt, &msg)
                .expect("encrypt varied failed")
        })
        .collect();

    let s_idx = Cell::new(0usize);
    run_collection(seed, samples_per_class, 200, |class| match class {
        Class::Baseline => {
            let result = private_key.decrypt(Pkcs1v15Encrypt, &fixed_ciphertext);
            black_box(result.is_ok());
        }
        Class::Sample => {
            let i = s_idx.get();
            s_idx.set((i + 1) % POOL_SIZE);
            let result = private_key.decrypt(Pkcs1v15Encrypt, &varied_pool[i]);
            black_box(result.is_ok());
        }
    })
}

fn main() {
    let args = Args::parse();

    if args.resume && row_already_present(&args.output, &args.budget_label, args.iteration, args.seed)
    {
        eprintln!(
            "[skip] resume: ({}, iter={}, seed={}) already present",
            args.budget_label, args.iteration, args.seed
        );
        return;
    }

    let mode = args.marvin_mode.to_ascii_lowercase();
    let mode = mode.as_str();
    eprintln!(
        "[marvin_budget_sweep] mode={} adaptive={} budget_label={} iter={} seed={} n={}",
        mode, args.adaptive, args.budget_label, args.iteration, args.seed, args.samples_per_class,
    );

    // Default attacker model for both modes: AdjacentNetwork (matches §5.6).
    let default_model = AttackerModel::AdjacentNetwork;

    // Adaptive path short-circuits: it runs tacet's production `.test()` which
    // does its own collection + analysis in a single call. The CSV still
    // records nominal collection / analysis times (both inside `.test`).
    if args.adaptive {
        if mode != "cache" {
            eprintln!("ERROR: --adaptive currently only implemented for --marvin-mode cache");
            std::process::exit(2);
        }
        let model = match args.attacker_model.as_deref() {
            None => default_model,
            Some(s) => match s.to_ascii_lowercase().as_str() {
                "adjacent" | "adjacentnetwork" => AttackerModel::AdjacentNetwork,
                "shared" | "sharedhardware" => AttackerModel::SharedHardware,
                "pq" | "postquantum" | "postquantumsentinel" => AttackerModel::PostQuantumSentinel,
                "remote" | "remotenetwork" => AttackerModel::RemoteNetwork,
                "research" => AttackerModel::Research,
                other => {
                    eprintln!("ERROR: unknown attacker model '{}'", other);
                    std::process::exit(2);
                }
            },
        };
        run_adaptive_and_write(&args, model);
        return;
    }

    // ---- Collection (fixed-n single-pass path) ----
    let t_coll = Instant::now();
    let collected = match mode {
        "cache" => collect_marvin_cache(args.seed, args.samples_per_class),
        "padding" => {
            let cases = Tier::Two.cases();
            let case = cases
                .into_iter()
                .find(|c| c.id.contains("marvin"))
                .expect("tier2 missing MARVIN case");
            (case.collect)(args.seed, args.samples_per_class)
        }
        other => {
            eprintln!("ERROR: unknown marvin-mode '{}'", other);
            std::process::exit(2);
        }
    };
    let collection_ms = t_coll.elapsed().as_millis() as u64;

    eprintln!(
        "  collected baseline={} test={} in {}ms (timer={} resolution={:.3}ns)",
        collected.baseline_ns.len(),
        collected.test_ns.len(),
        collection_ms,
        collected.timer_name,
        collected.timer_resolution_ns,
    );

    // ---- Analysis (single-pass; same codepath the cross-tool binary uses) ----
    let model = match args.attacker_model.as_deref() {
        None => default_model,
        Some(s) => match s.to_ascii_lowercase().as_str() {
            "adjacent" | "adjacentnetwork" => AttackerModel::AdjacentNetwork,
            "shared" | "sharedhardware" => AttackerModel::SharedHardware,
            "pq" | "postquantum" | "postquantumsentinel" => AttackerModel::PostQuantumSentinel,
            "remote" | "remotenetwork" => AttackerModel::RemoteNetwork,
            "research" => AttackerModel::Research,
            other => {
                eprintln!("ERROR: unknown attacker model '{}'", other);
                std::process::exit(2);
            }
        },
    };
    let t_an = Instant::now();
    let outcome = TimingOracle::for_attacker(model)
        // time_budget/max_samples don't apply to single-pass analysis, but
        // set a generous value anyway so it's explicit in logs.
        .time_budget(Duration::from_secs(600))
        .analyze_raw_samples_with_resolution(
            &collected.baseline_ns,
            &collected.test_ns,
            collected.timer_resolution_ns,
        );
    let analysis_ms = t_an.elapsed().as_millis() as u64;

    // Extract structured fields.
    let (
        verdict,
        leak_prob,
        effect_ns,
        ci_lo,
        ci_hi,
        samples_used,
        dep_len,
        ess,
        stat_ratio,
        calib_samples,
        threshold_ns,
    ) = match &outcome {
        Outcome::Pass {
            leak_probability,
            effect,
            samples_used,
            diagnostics,
            ..
        }
        | Outcome::Fail {
            leak_probability,
            effect,
            samples_used,
            diagnostics,
            ..
        }
        | Outcome::Inconclusive {
            leak_probability,
            effect,
            samples_used,
            diagnostics,
            ..
        } => (
            outcome_label(&outcome),
            Some(*leak_probability),
            Some(effect.max_effect_ns),
            Some(effect.credible_interval_ns.0),
            Some(effect.credible_interval_ns.1),
            *samples_used,
            Some(diagnostics.dependence_length),
            Some(diagnostics.effective_sample_size),
            Some(diagnostics.stationarity_ratio),
            Some(diagnostics.calibration_samples),
            Some(diagnostics.threshold_ns),
        ),
        Outcome::Unmeasurable { .. } => (
            "unmeasurable", None, None, None, None, 0, None, None, None, None, None,
        ),
        // Research mode is not used here (AttackerModel::AdjacentNetwork),
        // but the match must be exhaustive.
        Outcome::Research(_) => (
            "research", None, None, None, None, 0, None, None, None, None, None,
        ),
    };

    eprintln!(
        "  verdict={} P={} effect={} CI=[{},{}] samples_used={} block={} ESS={} in {}ms",
        verdict,
        fmt_opt_f64(leak_prob, 6),
        fmt_opt_f64(effect_ns, 3),
        fmt_opt_f64(ci_lo, 3),
        fmt_opt_f64(ci_hi, 3),
        samples_used,
        fmt_opt_usize(dep_len),
        fmt_opt_usize(ess),
        analysis_ms,
    );

    // ---- CSV append ----
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create output dir");
        }
    }
    let fresh = !args.output.exists();
    let mut out = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.output)
            .expect("open output CSV"),
    );
    if fresh {
        writeln!(out, "{}", CSV_HEADER).expect("write header");
    }
    writeln!(
        out,
        "{ts},{bl},{it},{sd},{n_req},{n_used},{tn},{tr:.4},{cms},{ams},{v},{p},{e},{lo},{hi},{dep},{ess},{sr},{cs},{am},{th}",
        ts = iso_timestamp(),
        bl = args.budget_label,
        it = args.iteration,
        sd = args.seed,
        n_req = args.samples_per_class,
        n_used = samples_used,
        tn = collected.timer_name,
        tr = collected.timer_resolution_ns,
        cms = collection_ms,
        ams = analysis_ms,
        v = verdict,
        p = fmt_opt_f64(leak_prob, 6),
        e = fmt_opt_f64(effect_ns, 3),
        lo = fmt_opt_f64(ci_lo, 3),
        hi = fmt_opt_f64(ci_hi, 3),
        dep = fmt_opt_usize(dep_len),
        ess = fmt_opt_usize(ess),
        sr = fmt_opt_f64(stat_ratio, 4),
        cs = fmt_opt_usize(calib_samples),
        am = format_attacker_model(model),
        th = fmt_opt_f64(threshold_ns, 3),
    )
    .expect("write CSV row");
    out.flush().ok();
}

fn outcome_label(o: &Outcome) -> &'static str {
    match o {
        Outcome::Pass { .. } => "pass",
        Outcome::Fail { .. } => "fail",
        Outcome::Inconclusive { .. } => "inconclusive",
        Outcome::Unmeasurable { .. } => "unmeasurable",
        Outcome::Research(_) => "research",
    }
}

fn format_attacker_model(m: AttackerModel) -> &'static str {
    match m {
        AttackerModel::SharedHardware => "SharedHardware",
        AttackerModel::PostQuantumSentinel => "PostQuantumSentinel",
        AttackerModel::AdjacentNetwork => "AdjacentNetwork",
        AttackerModel::RemoteNetwork => "RemoteNetwork",
        AttackerModel::Research => "Research",
        AttackerModel::Custom { .. } => "Custom",
    }
}

fn fmt_opt_f64(v: Option<f64>, prec: usize) -> String {
    v.map(|x| format!("{:.*}", prec, x)).unwrap_or_default()
}

fn fmt_opt_usize(v: Option<usize>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}
