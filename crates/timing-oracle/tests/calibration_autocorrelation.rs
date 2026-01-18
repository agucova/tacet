//! Autocorrelation robustness tests.
//!
//! Validates that the oracle maintains type-1 error control under
//! varying levels of measurement autocorrelation, unlike classical
//! tools (dudect, TVLA, RTLF, tlsfuzzer) that assume IID samples.
//!
//! This replicates SILENT paper's Figure 1 methodology:
//! - Generate data with controlled autocorrelation (AR(1) process)
//! - Sweep (μ, Φ) grid where μ is effect size and Φ is autocorrelation
//! - Verify rejection rate ≤ α when μ < θ (type-1 error control)
//!
//! Key insight from SILENT: classical tools have inflated type-1 error
//! (false positives) when Φ > 0 and deflated power when Φ < 0.

mod calibration_utils;

use calibration_utils::{
    export_autocorr_heatmap_csv, generate_ar1_samples, wilson_ci, AutocorrCell, CalibrationConfig,
    Tier,
};
use rand::{rngs::StdRng, SeedableRng};
use std::cell::RefCell;
use std::time::Instant;
use timing_oracle::helpers::InputPair;
use timing_oracle::{AttackerModel, Outcome, TimingOracle};

// =============================================================================
// GRID CONFIGURATIONS
// =============================================================================

/// Effect multipliers (μ/θ) for the heatmap grid.
/// SILENT uses μ ∈ {0, 0.5Δ, Δ, 2Δ} where Δ is their threshold.
const EFFECT_MULTIPLIERS: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

/// Autocorrelation coefficients (Φ) for the heatmap grid.
/// SILENT uses Φ ∈ [-0.9, 0.9].
const AUTOCORRELATIONS: [f64; 5] = [-0.8, -0.4, 0.0, 0.4, 0.8];

/// Quick grid for iteration.
const QUICK_EFFECT_MULTS: [f64; 3] = [0.0, 1.0, 2.0];
const QUICK_AUTOCORRS: [f64; 3] = [-0.4, 0.0, 0.4];

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Base noise standard deviation in nanoseconds.
/// This determines the "difficulty" of detection.
/// Higher values = more noise = harder to detect effects.
const BASE_SIGMA_NS: f64 = 50.0;

/// θ for AdjacentNetwork model.
const THETA_NS: f64 = 100.0;

// =============================================================================
// ITERATION TIER TESTS
// =============================================================================

/// Quick autocorrelation check during development.
#[test]
fn autocorr_iteration() {
    run_autocorr_grid(
        "autocorr_iteration",
        AttackerModel::AdjacentNetwork,
        THETA_NS,
        &QUICK_EFFECT_MULTS,
        &QUICK_AUTOCORRS,
    );
}

// =============================================================================
// QUICK TIER TESTS
// =============================================================================

/// Autocorrelation robustness for AdjacentNetwork.
#[test]
fn autocorr_quick_adjacent_network() {
    if std::env::var("CALIBRATION_TIER").as_deref() == Ok("iteration") {
        eprintln!("[autocorr_quick_adjacent_network] Skipped: iteration tier");
        return;
    }

    run_autocorr_grid(
        "autocorr_quick_adjacent_network",
        AttackerModel::AdjacentNetwork,
        THETA_NS,
        &QUICK_EFFECT_MULTS,
        &QUICK_AUTOCORRS,
    );
}

// =============================================================================
// VALIDATION TIER TESTS
// =============================================================================

/// Full autocorrelation heatmap for AdjacentNetwork.
///
/// Generates SILENT-style (μ, Φ) heatmap showing rejection rates.
#[test]
#[ignore]
fn autocorr_validation_adjacent_network() {
    std::env::set_var("CALIBRATION_TIER", "validation");
    run_autocorr_grid(
        "autocorr_validation_adjacent_network",
        AttackerModel::AdjacentNetwork,
        THETA_NS,
        &EFFECT_MULTIPLIERS,
        &AUTOCORRELATIONS,
    );
}

/// Extended autocorrelation heatmap with finer Φ resolution.
#[test]
#[ignore]
fn autocorr_validation_fine_grid() {
    std::env::set_var("CALIBRATION_TIER", "validation");

    let fine_autocorrs: [f64; 9] = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8];

    run_autocorr_grid(
        "autocorr_validation_fine_grid",
        AttackerModel::AdjacentNetwork,
        THETA_NS,
        &EFFECT_MULTIPLIERS,
        &fine_autocorrs,
    );
}

// =============================================================================
// TEST RUNNER
// =============================================================================

/// Run the autocorrelation grid test.
///
/// For each (μ, Φ) cell:
/// 1. Generate measurements with AR(1) correlated noise
/// 2. Run the oracle N times
/// 3. Record rejection rate (Fail outcomes)
fn run_autocorr_grid(
    test_name: &str,
    model: AttackerModel,
    theta_ns: f64,
    effect_mults: &[f64],
    autocorrs: &[f64],
) {
    if CalibrationConfig::is_disabled() {
        eprintln!("[{}] Skipped: CALIBRATION_DISABLED=1", test_name);
        return;
    }

    let config = CalibrationConfig::from_env(test_name);
    let trials_per_cell = config.tier.autocorr_trials_per_cell();

    let total_cells = effect_mults.len() * autocorrs.len();
    eprintln!(
        "[{}] Autocorrelation grid test ({}×{} = {} cells, {} trials/cell)",
        test_name,
        effect_mults.len(),
        autocorrs.len(),
        total_cells,
        trials_per_cell
    );

    let mut cells: Vec<AutocorrCell> = Vec::new();
    let start_time = Instant::now();

    for (mu_idx, &mu_mult) in effect_mults.iter().enumerate() {
        for (phi_idx, &phi) in autocorrs.iter().enumerate() {
            let cell_num = mu_idx * autocorrs.len() + phi_idx + 1;
            let effect_ns = (theta_ns * mu_mult) as u64;

            eprintln!(
                "\n[{}] Cell {}/{}: μ={:.1}×θ ({}ns), Φ={:.1}",
                test_name, cell_num, total_cells, mu_mult, effect_ns, phi
            );

            let mut rejections = 0;

            // Create seed for this cell (deterministic for reproducibility)
            let cell_seed = config.seed.wrapping_add((mu_idx * 100 + phi_idx) as u64);

            for trial in 0..trials_per_cell {
                // Generate AR(1) correlated noise offsets for this trial
                // We'll add these to the base delay to simulate correlated measurement noise
                let trial_seed = cell_seed.wrapping_add(trial as u64 * 1000);
                let mut trial_rng = StdRng::seed_from_u64(trial_seed);

                // Pre-generate correlated noise for both classes
                // Use enough samples for the oracle's budget
                let n_noise_samples = config.samples_per_trial * 2;
                let baseline_noise =
                    generate_ar1_samples(n_noise_samples, phi, 0.0, BASE_SIGMA_NS, &mut trial_rng);
                let sample_noise =
                    generate_ar1_samples(n_noise_samples, phi, 0.0, BASE_SIGMA_NS, &mut trial_rng);

                // Counters for which noise sample to use
                let baseline_idx = RefCell::new(0usize);
                let sample_idx = RefCell::new(0usize);

                let inputs = InputPair::new(|| false, || true);

                let outcome = TimingOracle::for_attacker(model)
                    .max_samples(config.samples_per_trial)
                    .time_budget(config.time_budget_per_trial)
                    .test(inputs, |&is_sample| {
                        // Base operation time (~2μs)
                        let base_ns = 2000u64;

                        // Get correlated noise for this measurement
                        let noise_ns = if is_sample {
                            let mut idx = sample_idx.borrow_mut();
                            let noise = sample_noise.get(*idx % sample_noise.len()).unwrap_or(&0.0);
                            *idx += 1;
                            *noise
                        } else {
                            let mut idx = baseline_idx.borrow_mut();
                            let noise = baseline_noise
                                .get(*idx % baseline_noise.len())
                                .unwrap_or(&0.0);
                            *idx += 1;
                            *noise
                        };

                        // Effect (only for sample class)
                        let effect = if is_sample { effect_ns } else { 0 };

                        // Total delay = base + effect + noise (clamped to positive)
                        let total_ns =
                            (base_ns as f64 + effect as f64 + noise_ns).max(100.0) as u64;

                        calibration_utils::busy_wait_ns(total_ns);
                    });

                // Count rejections (Fail or Inconclusive with high leak probability)
                let is_rejection = match &outcome {
                    Outcome::Fail { .. } => true,
                    Outcome::Inconclusive {
                        leak_probability, ..
                    } => *leak_probability >= 0.95,
                    _ => false,
                };
                if is_rejection {
                    rejections += 1;
                }

                // Progress every 20 trials
                if (trial + 1) % 20 == 0 || trial + 1 == trials_per_cell {
                    let rate = rejections as f64 / (trial + 1) as f64;
                    eprintln!(
                        "  Trial {}/{}: {:.0}% rejection rate",
                        trial + 1,
                        trials_per_cell,
                        rate * 100.0
                    );
                }
            }

            let rate = rejections as f64 / trials_per_cell as f64;
            let (ci_low, ci_high) = wilson_ci(rejections, trials_per_cell, 0.95);

            cells.push(AutocorrCell {
                mu_mult,
                phi,
                trials: trials_per_cell,
                rejections,
                rejection_rate: rate,
                ci_low,
                ci_high,
            });
        }
    }

    let elapsed = start_time.elapsed();

    // Export to CSV
    export_autocorr_heatmap_csv(test_name, &cells);

    // Print heatmap
    print_heatmap(test_name, effect_mults, autocorrs, &cells);

    // Validate type-1 error control
    let mut any_failed = false;
    let alpha = 0.10; // Target type-1 error rate

    for cell in &cells {
        // For μ < θ, rejection rate should be ≤ α (with some tolerance for sampling variance)
        if cell.mu_mult < 1.0 {
            // Allow up to α + 2*SE where SE ≈ sqrt(α(1-α)/n)
            let se = (alpha * (1.0 - alpha) / cell.trials as f64).sqrt();
            let max_allowed = alpha + 2.0 * se + 0.05; // Extra 5% tolerance

            if cell.rejection_rate > max_allowed {
                eprintln!(
                    "\n[{}] WARNING: Type-1 error {:.1}% exceeds {:.1}% at μ={:.1}×θ, Φ={:.1}",
                    test_name,
                    cell.rejection_rate * 100.0,
                    max_allowed * 100.0,
                    cell.mu_mult,
                    cell.phi
                );
                any_failed = true;
            }
        }
    }

    eprintln!(
        "\n[{}] Completed in {:.1}s ({} cells)",
        test_name,
        elapsed.as_secs_f64(),
        cells.len()
    );

    if any_failed {
        eprintln!("[{}] FAILED: Type-1 error control violated", test_name);
        if config.tier == Tier::Validation {
            panic!(
                "[{}] FAILED: Autocorrelation robustness validation failed",
                test_name
            );
        }
    } else {
        eprintln!("[{}] PASSED", test_name);
    }
}

/// Print a text-based heatmap of rejection rates.
fn print_heatmap(test_name: &str, effect_mults: &[f64], autocorrs: &[f64], cells: &[AutocorrCell]) {
    eprintln!("\n[{}] Rejection Rate Heatmap (%):", test_name);

    // Header row
    eprint!("        Φ |");
    for phi in autocorrs {
        eprint!(" {:>6.1} |", phi);
    }
    eprintln!();

    // Separator
    eprint!("  μ/θ    -+");
    for _ in autocorrs {
        eprint!("-{:->6}-+", "");
    }
    eprintln!();

    // Data rows
    for mu_mult in effect_mults {
        eprint!("  {:>5.2}  |", mu_mult);
        for phi in autocorrs {
            if let Some(cell) = cells
                .iter()
                .find(|c| (c.mu_mult - mu_mult).abs() < 0.01 && (c.phi - phi).abs() < 0.01)
            {
                // Color code: red if above 20%, yellow if above 10%, green otherwise
                let rate_pct = cell.rejection_rate * 100.0;
                let marker = if rate_pct > 20.0 {
                    "!!"
                } else if rate_pct > 10.0 && cell.mu_mult < 1.0 {
                    " *"
                } else {
                    "  "
                };
                eprint!(" {:>4.0}%{} |", rate_pct, marker);
            } else {
                eprint!("   N/A  |");
            }
        }
        eprintln!();
    }

    eprintln!();
    eprintln!("  Legend: !! = >20%,  * = >10% (concerning if μ<θ)");
}

// =============================================================================
// TIER CONFIGURATION EXTENSION
// =============================================================================

impl Tier {
    /// Trials per cell for autocorrelation grid.
    pub fn autocorr_trials_per_cell(&self) -> usize {
        match self {
            Tier::Iteration => 30,
            Tier::Quick => 50,
            Tier::Full => 100,
            Tier::Validation => 200,
        }
    }
}
