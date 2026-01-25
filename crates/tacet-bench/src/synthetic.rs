//! Synthetic timing data generator for benchmark comparisons.
//!
//! Generates timing datasets with known statistical properties, similar to the
//! ground truth datasets used in the RTLF paper (Dunsche et al., USENIX Security 2024).
//!
//! # Effect Types
//!
//! - **Null** (`same-xy`): Both classes from identical distribution - no timing leak
//! - **Shift**: Test class mean shifted by X% of standard deviation
//! - **Tail**: Test class has heavier tail (same mean, different tail behavior)
//! - **SameMean**: Same mean but different variance
//!
//! # Output Formats
//!
//! Each generated dataset provides both:
//! - **Interleaved**: Random ordering of baseline/test samples (for tacet)
//! - **Blocked**: All baseline first, then all test (for dudect, RTLF)

use rand::prelude::*;
use rand_distr::{Distribution, LogNormal, Normal};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use tacet::Class;

/// Type of effect to inject into the synthetic data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EffectType {
    /// No effect - both classes from identical distribution (`same-xy`)
    Null,
    /// Mean shift by given percentage in **real-space** (not log-space).
    /// A 5% shift means test_mean ≈ 1.05 * baseline_mean.
    Shift {
        /// Percentage of baseline mean to shift (e.g., 1.0 = 1%, 5.0 = 5%)
        percent: f64,
    },
    /// Heavier tail in test class (same mean, different tail behavior)
    Tail,
    /// Same mean but different variance
    SameMean,
}

impl EffectType {
    /// Get a short name for this effect type (for filenames)
    pub fn name(&self) -> String {
        match self {
            EffectType::Null => "same-xy".to_string(),
            EffectType::Shift { percent } => format!("shift-{}pct", percent),
            EffectType::Tail => "tail".to_string(),
            EffectType::SameMean => "same-mean".to_string(),
        }
    }
}

// ============================================================================
// New Benchmark Suite Types (for comprehensive tool comparison)
// ============================================================================

/// Effect pattern for benchmark suite (parameterized by effect_sigma_mult).
///
/// Unlike `EffectType` which uses fixed percentage shifts, `EffectPattern`
/// defines effects relative to the baseline standard deviation σ, making
/// results comparable across different noise levels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum EffectPattern {
    /// No effect - both classes from identical distribution (FPR test)
    #[default]
    Null,
    /// Mean shift by `effect_sigma_mult * σ` (real-space shift)
    Shift,
    /// Heavier tail in test class (same mean, larger variance)
    Tail,
    /// Different variance (same mean)
    Variance,
    /// Bimodal mixture: occasional slow operations
    /// Models cache misses, branch mispredictions, syscalls
    Bimodal {
        /// Probability of slow operation (default: 0.05 = 5%)
        slow_prob: f64,
        /// Multiplier for slow operations (default: 10.0 = 10x slower)
        slow_mult: f64,
    },
    /// Timer quantization effects (coarse resolution)
    /// Simulates systems like Apple Silicon (42ns cntvct_el0)
    Quantized {
        /// Timer quantum in nanoseconds
        quantum_ns: f64,
    },
}

impl EffectPattern {
    /// Get a short name for this effect pattern (for filenames/CSV)
    pub fn name(&self) -> &'static str {
        match self {
            EffectPattern::Null => "null",
            EffectPattern::Shift => "shift",
            EffectPattern::Tail => "tail",
            EffectPattern::Variance => "variance",
            EffectPattern::Bimodal { .. } => "bimodal",
            EffectPattern::Quantized { .. } => "quantized",
        }
    }

    /// Standard bimodal pattern (5% slow, 10x multiplier)
    pub fn bimodal_default() -> Self {
        EffectPattern::Bimodal {
            slow_prob: 0.05,
            slow_mult: 10.0,
        }
    }

    /// Standard quantized pattern (42ns quantum like Apple Silicon)
    pub fn quantized_default() -> Self {
        EffectPattern::Quantized { quantum_ns: 42.0 }
    }
}

/// Noise model for generated samples.
///
/// Real timing measurements exhibit autocorrelation due to CPU state,
/// cache warming, and OS scheduler effects. AR(1) noise models this.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum NoiseModel {
    /// Independent and identically distributed samples (traditional assumption)
    #[default]
    IID,
    /// AR(1) autocorrelated noise: x[t] = φ * x[t-1] + sqrt(1-φ²) * ε[t]
    ///
    /// Supports both positive and negative autocorrelation:
    /// - φ > 0: Positive autocorrelation (consecutive samples tend to be similar)
    ///   - φ = 0.3: weak positive
    ///   - φ = 0.8: strong positive (typical for timing due to CPU state carryover)
    /// - φ < 0: Negative autocorrelation (consecutive samples tend to alternate)
    ///   - φ = -0.3: weak negative
    ///   - φ = -0.8: strong negative (less common, models oscillating behavior)
    ///
    /// The scaling factor sqrt(1-φ²) ensures marginal variance remains 1.
    AR1 {
        /// Autocorrelation coefficient (-1 < φ < 1)
        phi: f64,
    },
}

impl NoiseModel {
    /// Get a short name for this noise model (for filenames/CSV)
    pub fn name(&self) -> String {
        match self {
            NoiseModel::IID => "iid".to_string(),
            NoiseModel::AR1 { phi } => {
                // Handle negative phi: "ar1-n0.3" for -0.3, "ar1-0.3" for +0.3
                if *phi < 0.0 {
                    format!("ar1-n{:.1}", -phi)
                } else {
                    format!("ar1-{:.1}", phi)
                }
            }
        }
    }
}

/// Configuration for benchmark dataset generation.
///
/// This is the newer API that parameterizes effect size relative to σ,
/// making results comparable across different noise levels.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of samples per class (baseline and test each get this many)
    pub samples_per_class: usize,
    /// Effect pattern to apply
    pub effect_pattern: EffectPattern,
    /// Effect size as multiplier of baseline σ (0 = null, 1 = 1σ shift)
    pub effect_sigma_mult: f64,
    /// Noise model (IID or AR1)
    pub noise_model: NoiseModel,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Base mean for log-normal distribution (in log-space)
    pub base_mu: f64,
    /// Base standard deviation for log-normal distribution (in log-space)
    pub base_sigma: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            samples_per_class: 10_000,
            effect_pattern: EffectPattern::Null,
            effect_sigma_mult: 0.0,
            noise_model: NoiseModel::IID,
            seed: 42,
            base_mu: 13.8,   // ln(1_000_000) ≈ 13.8
            base_sigma: 0.1, // ~10% CV in real-space
        }
    }
}

impl BenchmarkConfig {
    /// Create a null config (no effect) for FPR testing
    pub fn null(samples_per_class: usize, seed: u64) -> Self {
        Self {
            samples_per_class,
            effect_pattern: EffectPattern::Null,
            effect_sigma_mult: 0.0,
            seed,
            ..Default::default()
        }
    }

    /// Create a shift config with given σ multiplier
    pub fn shift(samples_per_class: usize, sigma_mult: f64, seed: u64) -> Self {
        Self {
            samples_per_class,
            effect_pattern: EffectPattern::Shift,
            effect_sigma_mult: sigma_mult,
            seed,
            ..Default::default()
        }
    }

    /// Get a descriptive name for this configuration
    pub fn name(&self) -> String {
        let effect_name = if self.effect_sigma_mult == 0.0 {
            "null".to_string()
        } else {
            format!(
                "{}-{:.1}sigma",
                self.effect_pattern.name(),
                self.effect_sigma_mult
            )
        };
        format!(
            "{}k-{}-{}",
            self.samples_per_class / 1000,
            effect_name,
            self.noise_model.name()
        )
    }
}

/// Configuration for synthetic dataset generation.
#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    /// Number of samples per class (baseline and test each get this many)
    pub samples_per_class: usize,
    /// Type of effect to inject
    pub effect: EffectType,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Base mean for log-normal distribution (in log-space)
    pub base_mu: f64,
    /// Base standard deviation for log-normal distribution (in log-space)
    pub base_sigma: f64,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            samples_per_class: 15000,
            effect: EffectType::Null,
            seed: 42,
            // Parameters chosen to produce realistic timing distributions
            // Mean ~1M cycles, reasonable spread
            base_mu: 13.8, // ln(1_000_000) ≈ 13.8
            base_sigma: 0.1,
        }
    }
}

/// Generated synthetic dataset with both interleaved and blocked formats.
#[derive(Debug, Clone)]
pub struct GeneratedDataset {
    /// Interleaved samples: (class, value) pairs in random order
    pub interleaved: Vec<(Class, u64)>,
    /// Blocked samples: baseline and test separated
    pub blocked: BlockedData,
}

/// Blocked format: baseline samples followed by test samples.
#[derive(Debug, Clone)]
pub struct BlockedData {
    /// Baseline (control) samples
    pub baseline: Vec<u64>,
    /// Test (treatment) samples
    pub test: Vec<u64>,
}

/// Generate a synthetic dataset with the given configuration.
///
/// # Arguments
/// * `config` - Configuration specifying effect type, sample count, and seed
///
/// # Returns
/// A `GeneratedDataset` containing both interleaved and blocked formats.
pub fn generate_dataset(config: &SyntheticConfig) -> GeneratedDataset {
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Create baseline distribution (always log-normal)
    let baseline_dist =
        LogNormal::new(config.base_mu, config.base_sigma).expect("Invalid log-normal parameters");

    // Create test distribution based on effect type
    let (baseline, test) = match config.effect {
        EffectType::Null => {
            // Same distribution for both classes
            let baseline: Vec<u64> = (0..config.samples_per_class)
                .map(|_| baseline_dist.sample(&mut rng) as u64)
                .collect();
            let test: Vec<u64> = (0..config.samples_per_class)
                .map(|_| baseline_dist.sample(&mut rng) as u64)
                .collect();
            (baseline, test)
        }
        EffectType::Shift { percent } => {
            // Shift mean by percent% in REAL-SPACE (not log-space)
            // For log-normal: mean = exp(μ + σ²/2)
            // To get (1 + k) * mean, we need: exp(new_μ + σ²/2) = (1+k) * exp(μ + σ²/2)
            // Therefore: new_μ = μ + ln(1 + k)
            let k = percent / 100.0;
            let mu_shift = (1.0 + k).ln();
            let test_dist = LogNormal::new(config.base_mu + mu_shift, config.base_sigma)
                .expect("Invalid log-normal parameters");

            let baseline: Vec<u64> = (0..config.samples_per_class)
                .map(|_| baseline_dist.sample(&mut rng) as u64)
                .collect();
            let test: Vec<u64> = (0..config.samples_per_class)
                .map(|_| test_dist.sample(&mut rng) as u64)
                .collect();
            (baseline, test)
        }
        EffectType::Tail => {
            // Heavier tail: larger sigma but adjusted mu to keep mean similar
            // For log-normal, mean = exp(μ + σ²/2)
            // To keep mean constant while increasing σ, decrease μ accordingly
            let new_sigma = config.base_sigma * 1.5;
            let mu_adjustment = (new_sigma.powi(2) - config.base_sigma.powi(2)) / 2.0;
            let test_dist = LogNormal::new(config.base_mu - mu_adjustment, new_sigma)
                .expect("Invalid log-normal parameters");

            let baseline: Vec<u64> = (0..config.samples_per_class)
                .map(|_| baseline_dist.sample(&mut rng) as u64)
                .collect();
            let test: Vec<u64> = (0..config.samples_per_class)
                .map(|_| test_dist.sample(&mut rng) as u64)
                .collect();
            (baseline, test)
        }
        EffectType::SameMean => {
            // Same mean, different variance
            // Increase sigma by 20%
            let new_sigma = config.base_sigma * 1.2;
            let mu_adjustment = (new_sigma.powi(2) - config.base_sigma.powi(2)) / 2.0;
            let test_dist = LogNormal::new(config.base_mu - mu_adjustment, new_sigma)
                .expect("Invalid log-normal parameters");

            let baseline: Vec<u64> = (0..config.samples_per_class)
                .map(|_| baseline_dist.sample(&mut rng) as u64)
                .collect();
            let test: Vec<u64> = (0..config.samples_per_class)
                .map(|_| test_dist.sample(&mut rng) as u64)
                .collect();
            (baseline, test)
        }
    };

    // Create interleaved version (random ordering)
    let mut interleaved: Vec<(Class, u64)> = baseline
        .iter()
        .map(|&v| (Class::Baseline, v))
        .chain(test.iter().map(|&v| (Class::Sample, v)))
        .collect();
    interleaved.shuffle(&mut rng);

    GeneratedDataset {
        interleaved,
        blocked: BlockedData { baseline, test },
    }
}

// ============================================================================
// New Benchmark Dataset Generation (for comprehensive tool comparison)
// ============================================================================

/// Generate AR(1) autocorrelated noise.
///
/// Produces samples with autocorrelation coefficient φ:
/// `x[t] = φ * x[t-1] + sqrt(1 - φ²) * ε[t]`
/// where ε[t] ~ N(0, 1).
///
/// The scaling factor `sqrt(1 - φ²)` ensures the marginal variance remains 1.
fn generate_ar1_noise(n: usize, phi: f64, rng: &mut impl Rng) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).expect("Invalid normal parameters");
    let innovation_scale = (1.0 - phi * phi).sqrt();

    let mut noise = Vec::with_capacity(n);
    let mut prev = normal.sample(rng);
    noise.push(prev);

    for _ in 1..n {
        let innovation = normal.sample(rng);
        let next = phi * prev + innovation_scale * innovation;
        noise.push(next);
        prev = next;
    }

    noise
}

/// Generate samples with the given noise model.
fn generate_samples_with_noise(
    n: usize,
    base_dist: &LogNormal<f64>,
    noise_model: NoiseModel,
    rng: &mut impl Rng,
) -> Vec<u64> {
    match noise_model {
        NoiseModel::IID => {
            // Standard IID sampling
            (0..n).map(|_| base_dist.sample(rng) as u64).collect()
        }
        NoiseModel::AR1 { phi } => {
            // Generate AR(1) correlated noise and apply multiplicatively
            let ar_noise = generate_ar1_noise(n, phi, rng);

            // For log-normal, additive noise in log-space = multiplicative in real-space
            // We add scaled AR noise to the log-space samples
            let base_samples: Vec<f64> = (0..n).map(|_| base_dist.sample(rng).ln()).collect();

            base_samples
                .iter()
                .zip(ar_noise.iter())
                .map(|(log_val, ar)| {
                    // Add AR(1) noise scaled by base_sigma to log-space
                    // This preserves the correlation structure while affecting timing
                    let noisy_log = log_val + ar * 0.05; // 5% correlation contribution
                    noisy_log.exp() as u64
                })
                .collect()
        }
    }
}

/// Generate a benchmark dataset using the new configuration format.
///
/// This function supports:
/// - Effect sizes relative to baseline σ
/// - Various effect patterns (Shift, Tail, Bimodal, Quantized)
/// - AR(1) autocorrelated noise
///
/// # Arguments
/// * `config` - Benchmark configuration
///
/// # Returns
/// A `GeneratedDataset` with both interleaved and blocked formats.
pub fn generate_benchmark_dataset(config: &BenchmarkConfig) -> GeneratedDataset {
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Create baseline distribution
    let baseline_dist =
        LogNormal::new(config.base_mu, config.base_sigma).expect("Invalid log-normal parameters");

    // Calculate real-space standard deviation for effect sizing
    // For log-normal: Var = exp(2μ + σ²) * (exp(σ²) - 1)
    // σ_real = sqrt(Var) ≈ exp(μ) * σ_log for small σ_log
    let real_space_sigma = config.base_mu.exp() * config.base_sigma;
    let effect_shift = config.effect_sigma_mult * real_space_sigma;

    // Generate baseline samples
    let baseline = generate_samples_with_noise(
        config.samples_per_class,
        &baseline_dist,
        config.noise_model,
        &mut rng,
    );

    // Generate test samples based on effect pattern
    let test: Vec<u64> = match config.effect_pattern {
        EffectPattern::Null => {
            // Same distribution as baseline
            generate_samples_with_noise(
                config.samples_per_class,
                &baseline_dist,
                config.noise_model,
                &mut rng,
            )
        }
        EffectPattern::Shift => {
            // Mean shift by effect_sigma_mult * σ in real-space
            // For log-normal: to shift mean by Δ, adjust μ_log by ln(1 + Δ/mean)
            let baseline_mean = (config.base_mu + config.base_sigma.powi(2) / 2.0).exp();
            let shift_factor = effect_shift / baseline_mean;
            let mu_shift = (1.0 + shift_factor).ln();
            let test_dist = LogNormal::new(config.base_mu + mu_shift, config.base_sigma)
                .expect("Invalid log-normal parameters");
            generate_samples_with_noise(
                config.samples_per_class,
                &test_dist,
                config.noise_model,
                &mut rng,
            )
        }
        EffectPattern::Tail => {
            // Heavier tail: increase σ_log, adjust μ_log to keep mean constant
            let tail_sigma = config.base_sigma * (1.0 + 0.5 * config.effect_sigma_mult.min(2.0));
            let mu_adjustment = (tail_sigma.powi(2) - config.base_sigma.powi(2)) / 2.0;
            let test_dist = LogNormal::new(config.base_mu - mu_adjustment, tail_sigma)
                .expect("Invalid log-normal parameters");
            generate_samples_with_noise(
                config.samples_per_class,
                &test_dist,
                config.noise_model,
                &mut rng,
            )
        }
        EffectPattern::Variance => {
            // Different variance, same mean (similar to Tail but more variance)
            let var_sigma = config.base_sigma * (1.0 + 0.3 * config.effect_sigma_mult.min(3.0));
            let mu_adjustment = (var_sigma.powi(2) - config.base_sigma.powi(2)) / 2.0;
            let test_dist = LogNormal::new(config.base_mu - mu_adjustment, var_sigma)
                .expect("Invalid log-normal parameters");
            generate_samples_with_noise(
                config.samples_per_class,
                &test_dist,
                config.noise_model,
                &mut rng,
            )
        }
        EffectPattern::Bimodal {
            slow_prob,
            slow_mult,
        } => {
            // Mixture model: most samples from baseline, some from slow distribution
            let slow_dist = LogNormal::new(config.base_mu + slow_mult.ln(), config.base_sigma)
                .expect("Invalid log-normal parameters");

            let base_samples = generate_samples_with_noise(
                config.samples_per_class,
                &baseline_dist,
                config.noise_model,
                &mut rng,
            );
            let slow_samples = generate_samples_with_noise(
                config.samples_per_class,
                &slow_dist,
                config.noise_model,
                &mut rng,
            );

            base_samples
                .into_iter()
                .zip(slow_samples)
                .map(|(base, slow)| {
                    if rng.gen::<f64>() < slow_prob * config.effect_sigma_mult.max(1.0) {
                        slow
                    } else {
                        base
                    }
                })
                .collect()
        }
        EffectPattern::Quantized { quantum_ns } => {
            // Generate samples and quantize to timer resolution
            let samples = generate_samples_with_noise(
                config.samples_per_class,
                &baseline_dist,
                config.noise_model,
                &mut rng,
            );

            // Add shift and quantize
            let baseline_mean = (config.base_mu + config.base_sigma.powi(2) / 2.0).exp();
            let shift = (effect_shift / baseline_mean * baseline_mean) as i64;
            let quantum = quantum_ns as u64;

            samples
                .into_iter()
                .map(|v| {
                    let shifted = (v as i64 + shift).max(0) as u64;
                    // Round to nearest quantum
                    ((shifted + quantum / 2) / quantum) * quantum
                })
                .collect()
        }
    };

    // Create interleaved version
    let mut interleaved: Vec<(Class, u64)> = baseline
        .iter()
        .map(|&v| (Class::Baseline, v))
        .chain(test.iter().map(|&v| (Class::Sample, v)))
        .collect();
    interleaved.shuffle(&mut rng);

    GeneratedDataset {
        interleaved,
        blocked: BlockedData { baseline, test },
    }
}

/// Write interleaved data to CSV file.
///
/// Format: `V1,V2` header, then `CLASS,VALUE` rows in interleaved order.
pub fn write_interleaved_csv(path: &Path, data: &[(Class, u64)]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "V1,V2")?;
    for (class, value) in data {
        let label = match class {
            Class::Baseline => "BASELINE",
            Class::Sample => "MODIFIED",
        };
        writeln!(writer, "{},{}", label, value)?;
    }

    writer.flush()
}

/// Write blocked data to CSV file (RTLF format).
///
/// Format: `V1,V2` header, all BASELINE rows first, then all MODIFIED rows.
pub fn write_blocked_csv(path: &Path, data: &BlockedData) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "V1,V2")?;

    // Write all baseline first
    for value in &data.baseline {
        writeln!(writer, "BASELINE,{}", value)?;
    }

    // Then all test
    for value in &data.test {
        writeln!(writer, "MODIFIED,{}", value)?;
    }

    writer.flush()
}

/// Standard benchmark configurations matching RTLF paper.
pub fn standard_configs() -> Vec<(&'static str, SyntheticConfig)> {
    let base = SyntheticConfig::default();

    vec![
        // Null hypothesis tests (FPR validation)
        (
            "15k-same-xy",
            SyntheticConfig {
                samples_per_class: 15000,
                effect: EffectType::Null,
                ..base.clone()
            },
        ),
        (
            "30k-same-xy",
            SyntheticConfig {
                samples_per_class: 30000,
                effect: EffectType::Null,
                ..base.clone()
            },
        ),
        (
            "500k-same-xy",
            SyntheticConfig {
                samples_per_class: 500000,
                effect: EffectType::Null,
                ..base.clone()
            },
        ),
        // Small shift (power tests - subtle effects)
        (
            "30k-shift-1pct",
            SyntheticConfig {
                samples_per_class: 30000,
                effect: EffectType::Shift { percent: 1.0 },
                ..base.clone()
            },
        ),
        (
            "500k-shift-1pct",
            SyntheticConfig {
                samples_per_class: 500000,
                effect: EffectType::Shift { percent: 1.0 },
                ..base.clone()
            },
        ),
        // Large shift (power tests - obvious effects)
        (
            "30k-shift-5pct",
            SyntheticConfig {
                samples_per_class: 30000,
                effect: EffectType::Shift { percent: 5.0 },
                ..base.clone()
            },
        ),
        (
            "500k-shift-5pct",
            SyntheticConfig {
                samples_per_class: 500000,
                effect: EffectType::Shift { percent: 5.0 },
                ..base.clone()
            },
        ),
        // Tail effects
        (
            "30k-tail",
            SyntheticConfig {
                samples_per_class: 30000,
                effect: EffectType::Tail,
                ..base.clone()
            },
        ),
        (
            "500k-tail",
            SyntheticConfig {
                samples_per_class: 500000,
                effect: EffectType::Tail,
                ..base.clone()
            },
        ),
        // Same mean, different variance
        (
            "30k-same-mean",
            SyntheticConfig {
                samples_per_class: 30000,
                effect: EffectType::SameMean,
                ..base.clone()
            },
        ),
        (
            "500k-same-mean",
            SyntheticConfig {
                samples_per_class: 500000,
                effect: EffectType::SameMean,
                ..base
            },
        ),
    ]
}

/// Generate a complete benchmark suite with multiple datasets per configuration.
///
/// # Arguments
/// * `output_dir` - Base directory for output
/// * `datasets_per_config` - Number of datasets to generate per configuration (default: 1000)
/// * `configs` - Configurations to generate (if None, uses `standard_configs()`)
///
/// # Returns
/// Ok(()) on success, or IO error if file operations fail.
pub fn generate_benchmark_suite(
    output_dir: &Path,
    datasets_per_config: usize,
    configs: Option<Vec<(String, SyntheticConfig)>>,
) -> std::io::Result<()> {
    let configs: Vec<(String, SyntheticConfig)> = configs.unwrap_or_else(|| {
        standard_configs()
            .into_iter()
            .map(|(name, cfg)| (name.to_string(), cfg))
            .collect()
    });

    for (name, base_config) in configs {
        let dir = output_dir.join(&name);
        fs::create_dir_all(&dir)?;

        println!(
            "Generating {} datasets for {} (n={})",
            datasets_per_config, &name, base_config.samples_per_class
        );

        for i in 0..datasets_per_config {
            let config = SyntheticConfig {
                seed: base_config.seed.wrapping_add(i as u64),
                ..base_config.clone()
            };
            let dataset = generate_dataset(&config);

            // Write interleaved CSV
            write_interleaved_csv(
                &dir.join(format!("{}_interleaved.csv", i)),
                &dataset.interleaved,
            )?;

            // Write blocked CSV (RTLF format)
            write_blocked_csv(&dir.join(format!("{}_blocked.csv", i)), &dataset.blocked)?;

            // Progress indicator every 100 datasets
            if (i + 1) % 100 == 0 {
                println!("  Generated {}/{}", i + 1, datasets_per_config);
            }
        }

        println!("  Completed {}", name);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_null_dataset() {
        let config = SyntheticConfig {
            samples_per_class: 100,
            effect: EffectType::Null,
            seed: 42,
            ..Default::default()
        };

        let dataset = generate_dataset(&config);

        assert_eq!(dataset.blocked.baseline.len(), 100);
        assert_eq!(dataset.blocked.test.len(), 100);
        assert_eq!(dataset.interleaved.len(), 200);

        // Check interleaved has mix of both classes
        let baseline_count = dataset
            .interleaved
            .iter()
            .filter(|(c, _)| *c == Class::Baseline)
            .count();
        assert_eq!(baseline_count, 100);
    }

    #[test]
    fn test_generate_shift_dataset() {
        let config = SyntheticConfig {
            samples_per_class: 1000,
            effect: EffectType::Shift { percent: 5.0 },
            seed: 42,
            ..Default::default()
        };

        let dataset = generate_dataset(&config);

        // With 5% shift, test mean should be noticeably higher
        let baseline_mean: f64 = dataset
            .blocked
            .baseline
            .iter()
            .map(|&x| x as f64)
            .sum::<f64>()
            / 1000.0;
        let test_mean: f64 = dataset.blocked.test.iter().map(|&x| x as f64).sum::<f64>() / 1000.0;

        // Test mean should be higher (shift is positive)
        assert!(
            test_mean > baseline_mean,
            "Test mean {} should be > baseline mean {}",
            test_mean,
            baseline_mean
        );
    }

    #[test]
    fn test_deterministic_generation() {
        let config = SyntheticConfig {
            samples_per_class: 100,
            effect: EffectType::Null,
            seed: 12345,
            ..Default::default()
        };

        let dataset1 = generate_dataset(&config);
        let dataset2 = generate_dataset(&config);

        assert_eq!(dataset1.blocked.baseline, dataset2.blocked.baseline);
        assert_eq!(dataset1.blocked.test, dataset2.blocked.test);
    }

    #[test]
    fn test_write_csv_formats() {
        let config = SyntheticConfig {
            samples_per_class: 10,
            effect: EffectType::Null,
            seed: 42,
            ..Default::default()
        };

        let dataset = generate_dataset(&config);
        let temp_dir = TempDir::new().unwrap();

        // Write interleaved
        let interleaved_path = temp_dir.path().join("interleaved.csv");
        write_interleaved_csv(&interleaved_path, &dataset.interleaved).unwrap();

        // Write blocked
        let blocked_path = temp_dir.path().join("blocked.csv");
        write_blocked_csv(&blocked_path, &dataset.blocked).unwrap();

        // Read back and verify
        let interleaved_content = std::fs::read_to_string(&interleaved_path).unwrap();
        let blocked_content = std::fs::read_to_string(&blocked_path).unwrap();

        // Both should have header
        assert!(interleaved_content.starts_with("V1,V2\n"));
        assert!(blocked_content.starts_with("V1,V2\n"));

        // Interleaved should have mixed BASELINE/MODIFIED
        assert!(interleaved_content.contains("BASELINE,"));
        assert!(interleaved_content.contains("MODIFIED,"));

        // Blocked should have all BASELINE first
        let lines: Vec<&str> = blocked_content.lines().collect();
        assert!(lines[1].starts_with("BASELINE,"));
        assert!(lines[10].starts_with("BASELINE,"));
        assert!(lines[11].starts_with("MODIFIED,"));
    }

    #[test]
    fn test_effect_type_names() {
        assert_eq!(EffectType::Null.name(), "same-xy");
        assert_eq!(EffectType::Shift { percent: 1.0 }.name(), "shift-1pct");
        assert_eq!(EffectType::Shift { percent: 5.0 }.name(), "shift-5pct");
        assert_eq!(EffectType::Tail.name(), "tail");
        assert_eq!(EffectType::SameMean.name(), "same-mean");
    }
}
