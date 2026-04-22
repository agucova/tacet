//! Cross-tool crypto benchmark: one-shot raw-sample capture, fan-out analysis.
//!
//! For each `(test, iteration)` in a `CryptoTestCase` registry, this binary:
//!   1. Runs the crypto closure once to collect real timings into
//!      `CollectedBlocked` (interleaved per class).
//!   2. Hands the identical `BlockedData` to every configured `ToolAdapter`
//!      (dudect, SILENT via R pool, RTLF via R pool, tlsfuzzer via Python
//!      pool, TVLA, and tacet directly via `analyze_raw_samples_with_resolution`).
//!   3. Emits a CSV row per tool so downstream analysis can compute Wilson
//!      95% CIs on per-tool FPR and detection rates.
//!
//! The raw timings can optionally be persisted to disk so the analysis step
//! is re-runnable offline without re-measuring crypto (`--raw-samples-out`).
//!
//! # Design notes
//!
//! - **Tacet bypass**: the `TimingOracleAdapter` in `adapters.rs` hardcodes
//!   a 3 GHz cycles→ns conversion for synthetic data. Here we already have
//!   real nanoseconds, so tacet is called directly via
//!   `TimingOracle::analyze_raw_samples_with_resolution` with the measured
//!   timer resolution, bypassing that adapter. This is the "full pipeline
//!   vs full pipeline" stance from the plan: tacet's internal outlier
//!   trimming still runs (inside its own pipeline), other tools run their
//!   native pipelines on identical raw data.
//! - **Fixed-n single-pass** for tacet: matches the Fig 1/2 methodology
//!   (10 000 samples/class) rather than adaptive mode, giving up tacet's
//!   budget advantage in exchange for like-for-like competitor comparison.
//! - **Seeding**: `per_iter_seed = hash(base_seed, test_id, iteration)`,
//!   so every `(test, iteration)` has a deterministic, independent seed.
//! - **Resume**: rows with the same `(test_id, iteration, tool)` are
//!   skipped if already present in the CSV.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use tacet::{AttackerModel, Outcome, TimingOracle};

use tacet_bench::adapters::{
    DudectAdapter, RtlfAdapter, SilentAdapter, TimingTvlaAdapter, TlsfuzzerAdapter, ToolAdapter,
    ToolResult,
};
use tacet_bench::crypto_collect::CollectedBlocked;
use tacet_bench::crypto_registry::Tier;
use tacet_bench::process_pool::{ProcessConfig, ProcessPool};
use tacet_bench::BlockedData;

/// Cross-tool crypto benchmark.
#[derive(Parser, Debug)]
#[command(
    name = "crypto_benchmark",
    about = "Run timing analysis tools on real crypto under identical inputs",
    version
)]
struct Args {
    /// Registry tier to exercise: 1 = constant-time FPR probe,
    /// 2 = MARVIN detection probe, all = both.
    #[arg(long, default_value = "all")]
    tier: String,

    /// Iterations per test.
    #[arg(long, default_value_t = 20)]
    iterations: usize,

    /// Comma-separated tools to run, or "all".
    /// Available: tacet, dudect, tvla, silent, rtlf, tlsfuzzer.
    #[arg(long, default_value = "all")]
    tools: String,

    /// Override `samples_per_class` (matches the registry default otherwise).
    #[arg(long)]
    samples_per_class: Option<usize>,

    /// Output CSV path.
    #[arg(long, default_value = "crypto_benchmark_results.csv")]
    output: PathBuf,

    /// Optional directory to persist raw samples as CSV files
    /// (one `<test>-iter<N>.csv` per iteration). Enables offline re-analysis.
    #[arg(long)]
    raw_samples_out: Option<PathBuf>,

    /// Base RNG seed. Per-iteration seeds are derived from
    /// `hash(base_seed, test_id, iteration)`.
    #[arg(long, default_value_t = 20_260_418)]
    seed: u64,

    /// Comma-separated list of dither magnitudes (ns) to fan-out through.
    /// 0.0 = raw u64-rounded samples (fastest; SILENT may error on ties).
    /// Non-zero values add uniform dither in [-d/2, +d/2] ns before fanout,
    /// breaking numerical ties from discrete timer quantization without
    /// changing verdicts (dither magnitude is 20× below θ=100 ns).
    /// Each config yields a separate row per (test, iter, tool) in the CSV,
    /// tagged with `dither_ns`.
    #[arg(long, default_value = "0.0")]
    dither_configs: String,

    /// R worker pool size (SILENT + RTLF share this pool).
    #[arg(long)]
    r_pool_workers: Option<usize>,

    /// Python worker pool size (tlsfuzzer).
    #[arg(long)]
    python_pool_workers: Option<usize>,

    /// Time budget for the tacet adapter. Ignored for fixed-n single-pass.
    #[arg(long)]
    tacet_time_budget_secs: Option<u64>,
}

/// One CSV row.
#[derive(Debug, Clone)]
struct RowOut {
    ecosystem: String,
    library: String,
    primitive: String,
    test_id: String,
    expected: String,
    attacker_model: String,
    iteration: usize,
    seed: u64,
    samples_per_class: usize,
    timer_resolution_ns: f64,
    timer_name: String,
    dither_ns: f64,
    tool: String,
    outcome: String,
    detected_leak: bool,
    leak_probability: Option<f64>,
    samples_used: usize,
    decision_time_ms: u64,
    status: String,
    collection_time_ms: u64,
    timestamp: String,
}

const CSV_HEADER: &str = "ecosystem,library,primitive,test_id,expected,attacker_model,iteration,seed,samples_per_class,timer_resolution_ns,timer_name,dither_ns,tool,outcome,detected_leak,leak_probability,samples_used,decision_time_ms,status,collection_time_ms,timestamp";

impl RowOut {
    fn to_csv_line(&self) -> String {
        let leak_prob = self
            .leak_probability
            .map(|p| format!("{:.6}", p))
            .unwrap_or_default();
        format!(
            "{},{},{},{},{},{},{},{},{},{:.3},{},{:.3},{},{},{},{},{},{},{},{},{}",
            csv_quote(&self.ecosystem),
            csv_quote(&self.library),
            csv_quote(&self.primitive),
            csv_quote(&self.test_id),
            self.expected,
            self.attacker_model,
            self.iteration,
            self.seed,
            self.samples_per_class,
            self.timer_resolution_ns,
            csv_quote(&self.timer_name),
            self.dither_ns,
            csv_quote(&self.tool),
            self.outcome,
            self.detected_leak,
            leak_prob,
            self.samples_used,
            self.decision_time_ms,
            csv_quote(&self.status),
            self.collection_time_ms,
            self.timestamp,
        )
    }
}

fn csv_quote(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        s.to_string()
    }
}

/// Per-iteration seed derivation. Stable across invocations.
fn derive_seed(base_seed: u64, test_id: &str, iteration: usize) -> u64 {
    let mut h = DefaultHasher::new();
    base_seed.hash(&mut h);
    test_id.hash(&mut h);
    iteration.hash(&mut h);
    h.finish()
}

/// Resume key: `(test_id, iteration, dither_ns_fixed3, tool)`. We encode
/// dither as a 3-decimal string so floating-point equality isn't a concern.
type ResumeKey = (String, usize, String, String);

fn dither_key(d: f64) -> String {
    format!("{:.3}", d)
}

/// Load the set of resume keys already present in the CSV. Returns empty
/// on first run.
fn load_completed(csv_path: &Path) -> std::io::Result<HashSet<ResumeKey>> {
    let mut out = HashSet::new();
    if !csv_path.exists() {
        return Ok(out);
    }
    let file = File::open(csv_path)?;
    let reader = BufReader::new(file);
    let mut header_seen = false;
    for line in reader.lines() {
        let line = line?;
        if !header_seen {
            header_seen = true;
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        // Schema: test_id=3, iteration=6, dither_ns=11, tool=12.
        if parts.len() < 13 {
            continue;
        }
        let test_id = parts[3].trim_matches('"').to_string();
        let iteration: usize = match parts[6].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let dither: f64 = parts[11].parse().unwrap_or(0.0);
        let tool = parts[12].trim_matches('"').to_string();
        out.insert((test_id, iteration, dither_key(dither), tool));
    }
    Ok(out)
}

/// Persist raw samples for an iteration so analysis can be re-run offline.
fn write_raw_samples(path: &Path, collected: &CollectedBlocked) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "class,timing_ns")?;
    for v in &collected.baseline_ns {
        writeln!(w, "baseline,{}", v)?;
    }
    for v in &collected.test_ns {
        writeln!(w, "test,{}", v)?;
    }
    w.flush()?;
    Ok(())
}

/// Convert the tacet `Outcome` to the cross-tool `ToolResult` shape.
///
/// Inconclusive is preserved as a first-class outcome; competitor adapters
/// already surface Pass/Fail/Inconclusive explicitly. Downstream analysis
/// buckets {Fail → Detect, Pass → Miss, Inconclusive → Inconclusive} for
/// Table 2's three-way verdict.
fn tacet_outcome_to_row(outcome: &Outcome) -> (String, bool, Option<f64>, usize, String) {
    match outcome {
        Outcome::Pass {
            leak_probability,
            samples_used,
            ..
        } => (
            "pass".to_string(),
            false,
            Some(*leak_probability),
            *samples_used,
            format!("Pass (P={:.1}%)", leak_probability * 100.0),
        ),
        Outcome::Fail {
            leak_probability,
            samples_used,
            exploitability,
            ..
        } => (
            "fail".to_string(),
            true,
            Some(*leak_probability),
            *samples_used,
            format!(
                "Fail (P={:.1}%, {:?})",
                leak_probability * 100.0,
                exploitability
            ),
        ),
        Outcome::Inconclusive {
            leak_probability,
            samples_used,
            reason,
            ..
        } => (
            "inconclusive".to_string(),
            false,
            Some(*leak_probability),
            *samples_used,
            format!("Inconclusive: {:?}", reason),
        ),
        Outcome::Unmeasurable { recommendation, .. } => (
            "inconclusive".to_string(),
            false,
            None,
            0,
            format!("Unmeasurable: {}", recommendation),
        ),
        Outcome::Research(r) => {
            let detected = matches!(r.status, tacet::result::ResearchStatus::EffectDetected);
            (
                if detected { "fail" } else { "pass" }.to_string(),
                detected,
                None,
                r.samples_used,
                format!("Research: {:?}", r.status),
            )
        }
    }
}

fn format_attacker_model(m: &AttackerModel) -> &'static str {
    match m {
        AttackerModel::SharedHardware => "SharedHardware",
        AttackerModel::PostQuantumSentinel => "PostQuantumSentinel",
        AttackerModel::AdjacentNetwork => "AdjacentNetwork",
        AttackerModel::RemoteNetwork => "RemoteNetwork",
        AttackerModel::Research => "Research",
        AttackerModel::Custom { .. } => "Custom",
    }
}

// =============================================================================
// Pool / script discovery (mirrors bin/benchmark.rs, kept self-contained)
// =============================================================================

fn find_script(name: &str) -> Option<String> {
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let p = exe_dir.join("scripts").join(name);
            if p.exists() {
                return Some(p.to_string_lossy().to_string());
            }
            let p = exe_dir.join("../../scripts").join(name);
            if p.exists() {
                return p.canonicalize().ok().map(|p| p.to_string_lossy().to_string());
            }
        }
    }
    let p = PathBuf::from("scripts").join(name);
    if p.exists() {
        return p.canonicalize().ok().map(|p| p.to_string_lossy().to_string());
    }
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(manifest_dir).join("../../scripts").join(name);
        if p.exists() {
            return p.canonicalize().ok().map(|p| p.to_string_lossy().to_string());
        }
    }
    None
}

fn find_r_tool_script(command: &str, relative_path: &str) -> Option<String> {
    let out = std::process::Command::new("which").arg(command).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let wrapper_path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if wrapper_path.is_empty() {
        return None;
    }
    let wrapper_content = fs::read_to_string(&wrapper_path).ok()?;
    for line in wrapper_content.lines() {
        if let Some(idx) = line.find(relative_path) {
            let search_start = line[..idx]
                .rfind("/nix/store/")
                .or_else(|| line[..idx].rfind('/'))
                .unwrap_or(0);
            let path_start = &line[search_start..];
            let path_end = path_start
                .find(|c: char| c == '"' || c == '\'' || c.is_whitespace())
                .map(|i| search_start + i)
                .unwrap_or(line.len());
            let path = line[search_start..path_end].trim_matches(|c| c == '"' || c == '\'');
            if Path::new(path).exists() {
                return Some(path.to_string());
            }
        }
    }
    let wp = Path::new(&wrapper_path);
    if let Some(bin_dir) = wp.parent() {
        if let Some(tool_dir) = bin_dir.parent() {
            let script_path = tool_dir.join(relative_path);
            if script_path.exists() {
                return Some(script_path.to_string_lossy().to_string());
            }
        }
    }
    None
}

fn create_r_pool(workers: Option<usize>) -> Option<Arc<ProcessPool>> {
    let script_path = find_script("r-persistent-worker.R")?;
    let silent_path = find_r_tool_script("silent", "share/silent/scripts/SILENT.R");
    let rtlf_path = find_r_tool_script("rtlf", "share/rtlf/rtlf.R");
    let cpus = num_cpus::get();
    let size = workers.unwrap_or_else(|| ((cpus * 92) / 100).max(2));
    let config =
        ProcessConfig::r_worker(&script_path, silent_path.as_deref(), rtlf_path.as_deref());
    Some(Arc::new(ProcessPool::new(config, size)))
}

fn create_python_pool(workers: Option<usize>) -> Option<Arc<ProcessPool>> {
    let script_path = find_script("python-persistent-worker.py")?;
    let size = workers.unwrap_or_else(|| (num_cpus::get() / 3).max(2));
    let config = ProcessConfig::python_worker(&script_path);
    Some(Arc::new(ProcessPool::new(config, size)))
}

// =============================================================================
// Tool wiring
// =============================================================================

/// A tool that consumes `BlockedData` and emits a `ToolResult`.
///
/// The tacet entry runs outside the `ToolAdapter` trait (see module comment),
/// so it's handled separately by the caller.
enum ToolEntry {
    Tacet,
    Adapter {
        name: String,
        adapter: Box<dyn ToolAdapter>,
    },
}

fn parse_tool_list(
    s: &str,
    r_pool: Option<Arc<ProcessPool>>,
    python_pool: Option<Arc<ProcessPool>>,
) -> Vec<ToolEntry> {
    let requested: Vec<&str> = if s.eq_ignore_ascii_case("all") {
        vec!["tacet", "dudect", "tvla", "silent", "rtlf", "tlsfuzzer"]
    } else {
        s.split(',').map(str::trim).collect()
    };

    let mut out = Vec::new();
    for name in requested {
        match name.to_lowercase().as_str() {
            "tacet" => out.push(ToolEntry::Tacet),
            "dudect" => out.push(ToolEntry::Adapter {
                name: "dudect".into(),
                adapter: Box::new(DudectAdapter::default()),
            }),
            "tvla" | "timing-tvla" => out.push(ToolEntry::Adapter {
                name: "tvla".into(),
                adapter: Box::new(TimingTvlaAdapter::default()),
            }),
            "silent" => match &r_pool {
                Some(pool) => out.push(ToolEntry::Adapter {
                    name: "silent".into(),
                    adapter: Box::new(SilentAdapter::default().with_pool(pool.clone())),
                }),
                None => {
                    eprintln!("WARN: SILENT requested but R pool unavailable; skipping.");
                }
            },
            "rtlf" => match &r_pool {
                Some(pool) => out.push(ToolEntry::Adapter {
                    name: "rtlf".into(),
                    adapter: Box::new(RtlfAdapter::default().with_pool(pool.clone())),
                }),
                None => {
                    eprintln!("WARN: RTLF requested but R pool unavailable; skipping.");
                }
            },
            "tlsfuzzer" => match &python_pool {
                Some(pool) => out.push(ToolEntry::Adapter {
                    name: "tlsfuzzer".into(),
                    adapter: Box::new(TlsfuzzerAdapter::default().with_pool(pool.clone())),
                }),
                None => {
                    eprintln!("WARN: tlsfuzzer requested but Python pool unavailable; skipping.");
                }
            },
            other => eprintln!("WARN: unknown tool \"{}\"; skipping.", other),
        }
    }
    out
}

// =============================================================================
// Main
// =============================================================================

fn run_tacet_on_samples(
    baseline_ns: &[f64],
    test_ns: &[f64],
    timer_resolution_ns: f64,
    model: AttackerModel,
    _time_budget: Option<Duration>,
) -> (ToolResult, u64) {
    let t0 = Instant::now();
    // Fixed-n single-pass — matches Fig 1/2 methodology (plan §Strategy).
    // analyze_raw_samples_with_resolution picks up tacet's internal trim.
    let outcome = TimingOracle::for_attacker(model).analyze_raw_samples_with_resolution(
        baseline_ns,
        test_ns,
        timer_resolution_ns,
    );
    let elapsed_ms = t0.elapsed().as_millis() as u64;
    let (status_kind, detected, leak_prob, samples_used, status_msg) =
        tacet_outcome_to_row(&outcome);
    (
        ToolResult {
            detected_leak: detected,
            samples_used,
            decision_time_ms: elapsed_ms,
            leak_probability: leak_prob,
            status: status_msg,
            outcome: match status_kind.as_str() {
                "pass" => tacet_bench::adapters::OutcomeCategory::Pass,
                "fail" => tacet_bench::adapters::OutcomeCategory::Fail,
                "inconclusive" => tacet_bench::adapters::OutcomeCategory::Inconclusive,
                _ => tacet_bench::adapters::OutcomeCategory::Error,
            },
        },
        elapsed_ms,
    )
}

fn iso_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    // ISO-8601-ish with second precision; matches measure_*_sh output.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Don't want to pull chrono in here; shell-style date -Iseconds is good
    // enough for downstream. Users who need richer timestamps can rewrite.
    format!("{}", secs)
}

fn main() {
    let args = Args::parse();

    // Resolve tier.
    let tier = Tier::parse(&args.tier).unwrap_or_else(|| {
        eprintln!("ERROR: unknown tier \"{}\" (expected 1, 2, or all)", args.tier);
        std::process::exit(2);
    });
    let cases = tier.cases();

    eprintln!(
        "crypto_benchmark: tier={:?}, tests={}, iterations={}, tools={}",
        tier,
        cases.len(),
        args.iterations,
        args.tools
    );

    // Spin up pools lazily — only if a tool needs them.
    let needs_r = args.tools.contains("silent")
        || args.tools.contains("rtlf")
        || args.tools.eq_ignore_ascii_case("all");
    let needs_python = args.tools.contains("tlsfuzzer") || args.tools.eq_ignore_ascii_case("all");

    let r_pool = if needs_r { create_r_pool(args.r_pool_workers) } else { None };
    let python_pool = if needs_python {
        create_python_pool(args.python_pool_workers)
    } else {
        None
    };

    if needs_r && r_pool.is_none() {
        eprintln!(
            "WARN: R-based tools requested but pool construction failed; \
             proceeding without SILENT/RTLF."
        );
    }
    if needs_python && python_pool.is_none() {
        eprintln!(
            "WARN: Python-based tools requested but pool construction failed; \
             proceeding without tlsfuzzer."
        );
    }

    let tools = parse_tool_list(&args.tools, r_pool.clone(), python_pool.clone());
    if tools.is_empty() {
        eprintln!("ERROR: no tools selected.");
        std::process::exit(2);
    }

    // Prepare output file + resume index.
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("create output dir");
        }
    }
    let completed = load_completed(&args.output).expect("load existing CSV");
    let fresh_file = !args.output.exists();
    let mut out = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.output)
            .expect("open output CSV"),
    );
    if fresh_file {
        writeln!(out, "{}", CSV_HEADER).expect("write header");
    }

    let tacet_time_budget = args.tacet_time_budget_secs.map(Duration::from_secs);

    // Parse dither configs once.
    let dither_configs: Vec<f64> = args
        .dither_configs
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f64>().expect("invalid dither value"))
        .collect();
    if dither_configs.is_empty() {
        eprintln!("ERROR: --dither-configs produced no values");
        std::process::exit(2);
    }
    eprintln!("  dither configs (ns): {:?}", dither_configs);

    let tool_count = tools.len();
    let mut done_units: usize = 0;

    for case in &cases {
        let spc = args.samples_per_class.unwrap_or(case.samples_per_class);
        let expected_label = if case.is_leaky { "leaky" } else { "constant_time" };
        let model_label = format_attacker_model(&case.attacker_model);

        for iteration in 1..=args.iterations {
            let seed = derive_seed(args.seed, case.id, iteration);

            // Skip collection if every (dither, tool) cell for this iteration is done.
            let all_done = dither_configs.iter().all(|&d| {
                let dk = dither_key(d);
                tools.iter().all(|te| {
                    let tool_name = match te {
                        ToolEntry::Tacet => "tacet",
                        ToolEntry::Adapter { name, .. } => name.as_str(),
                    };
                    completed.contains(&(
                        case.id.to_string(),
                        iteration,
                        dk.clone(),
                        tool_name.to_string(),
                    ))
                })
            });
            if all_done {
                done_units += tool_count * dither_configs.len();
                continue;
            }

            eprintln!(
                "[{}/{}] {} iter {}/{} (seed={})",
                done_units / (tool_count * dither_configs.len()) + 1,
                cases.len() * args.iterations,
                case.id,
                iteration,
                args.iterations,
                seed
            );

            // Collection step (one-shot, shared across dither configs).
            let t_coll = Instant::now();
            let collected = (case.collect)(seed, spc);
            let coll_ms = t_coll.elapsed().as_millis() as u64;
            eprintln!(
                "  collected {} baseline + {} test samples in {} ms (timer={}, resolution={:.3} ns)",
                collected.baseline_ns.len(),
                collected.test_ns.len(),
                coll_ms,
                collected.timer_name,
                collected.timer_resolution_ns
            );

            // Persist raw samples if requested.
            if let Some(raw_dir) = &args.raw_samples_out {
                let sanitized = case.id.replace("::", "-");
                let path = raw_dir.join(format!("{}-iter{:03}.csv", sanitized, iteration));
                if let Err(e) = write_raw_samples(&path, &collected) {
                    eprintln!("  WARN: failed to write raw samples: {}", e);
                }
            }

            for &dither in &dither_configs {
                let dk = dither_key(dither);
                // Build dithered BlockedData once per (iteration, dither) pair.
                // Dither is seeded per (iteration, dither_ns_bits) so the same
                // run reproduces the same dithered sample stream deterministically.
                let dither_seed = seed ^ (dither.to_bits());
                let blocked: BlockedData =
                    collected.to_blocked_with_dither(dither, dither_seed);

                for tool in &tools {
                    let (tool_name, res) = match tool {
                        ToolEntry::Tacet => {
                            let tool_name = "tacet".to_string();
                            if completed.contains(&(
                                case.id.to_string(),
                                iteration,
                                dk.clone(),
                                tool_name.clone(),
                            )) {
                                done_units += 1;
                                continue;
                            }
                            // Run tacet on the dithered f64 samples to keep
                            // all tools on matched data per config.
                            let baseline: Vec<f64> = blocked
                                .baseline
                                .iter()
                                .map(|&v| v as f64)
                                .collect();
                            let test: Vec<f64> =
                                blocked.test.iter().map(|&v| v as f64).collect();
                            let (res, _ms) = run_tacet_on_samples(
                                &baseline,
                                &test,
                                collected.timer_resolution_ns.max(dither),
                                case.attacker_model,
                                tacet_time_budget,
                            );
                            (tool_name, res)
                        }
                        ToolEntry::Adapter { name, adapter } => {
                            if completed.contains(&(
                                case.id.to_string(),
                                iteration,
                                dk.clone(),
                                name.clone(),
                            )) {
                                done_units += 1;
                                continue;
                            }
                            let res = adapter.analyze_blocked(&blocked);
                            (name.clone(), res)
                        }
                    };

                    let row = RowOut {
                        ecosystem: case.ecosystem.to_string(),
                        library: case.library.to_string(),
                        primitive: case.primitive.to_string(),
                        test_id: case.id.to_string(),
                        expected: expected_label.to_string(),
                        attacker_model: model_label.to_string(),
                        iteration,
                        seed,
                        samples_per_class: spc,
                        timer_resolution_ns: collected.timer_resolution_ns,
                        timer_name: collected.timer_name.to_string(),
                        dither_ns: dither,
                        tool: tool_name.clone(),
                        outcome: res.outcome.as_str().to_string(),
                        detected_leak: res.detected_leak,
                        leak_probability: res.leak_probability,
                        samples_used: res.samples_used,
                        decision_time_ms: res.decision_time_ms,
                        status: res.status,
                        collection_time_ms: coll_ms,
                        timestamp: iso_timestamp(),
                    };
                    if let Err(e) = writeln!(out, "{}", row.to_csv_line()) {
                        eprintln!("  ERROR: write failed: {}", e);
                    }
                    if let Err(e) = out.flush() {
                        eprintln!("  ERROR: flush failed: {}", e);
                    }

                    eprintln!(
                        "    dither={:.2} {} → {} (leak={}, P={:?}, {}ms)",
                        dither,
                        tool_name,
                        res.outcome.as_str(),
                        res.detected_leak,
                        res.leak_probability,
                        res.decision_time_ms
                    );
                    done_units += 1;
                }
            }
        }
    }

    eprintln!(
        "Done. Wrote {} units to {}.",
        done_units,
        args.output.display()
    );
}
