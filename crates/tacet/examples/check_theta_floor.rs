//! Check theta_floor values for SILENT datasets

use std::path::Path;
use tacet::data::{load_silent_csv, load_two_column_csv};
use tacet::statistics::bootstrap_difference_covariance_discrete;
use tacet_core::adaptive::compute_c_floor_9d;

fn main() {
    let silent_path = Path::new("/Users/agucova/repos/SILENT/examples");

    println!("=== Theta Floor Analysis ===\n");

    // 1. KyberSlash (uses SILENT format with X/Y columns)
    analyze_theta_floor_silent(
        "KyberSlash",
        &silent_path.join("paper-kyberslash/timing_measurements_1_comparison_-72_72.csv"),
        1.0 / 3.0, // cycles to ns at 3GHz
        3.3,       // theta_user
    );

    // 2. Web App (uses SILENT format with X/Y columns)
    analyze_theta_floor_silent(
        "Web App",
        &silent_path.join("paper-web-application/web-diff-lan-10k.csv"),
        1.0, // already in ns
        5.0,
    );

    // 3. mbedTLS (uses BASELINE/MODIFIED columns)
    analyze_theta_floor(
        "mbedTLS (50k subset)",
        &silent_path.join("paper-mbedtls/Wrong_second_byte_(0x02_set_to_0x17)vsCorrectly_formatted_PKCS#1_PMS_message.csv"),
        "BASELINE", "MODIFIED",
        1.0 / 3.0,
        1.67,
        Some(50_000),
    );
}

fn analyze_theta_floor_silent(name: &str, path: &Path, ns_per_unit: f64, theta_user: f64) {
    println!("--- {} ---", name);

    let data = match load_silent_csv(path) {
        Ok(d) => d,
        Err(e) => {
            println!("  Error loading: {}\n", e);
            return;
        }
    };

    let (baseline_ns, test_ns) = data.to_nanoseconds(ns_per_unit);
    compute_and_print_theta_floor(name, &baseline_ns, &test_ns, theta_user);
}

fn analyze_theta_floor(
    name: &str,
    path: &Path,
    col1: &str,
    col2: &str,
    ns_per_unit: f64,
    theta_user: f64,
    max_samples: Option<usize>,
) {
    println!("--- {} ---", name);

    let data = match load_two_column_csv(path, true, col1, col2) {
        Ok(d) => d,
        Err(e) => {
            println!("  Error loading: {}\n", e);
            return;
        }
    };

    let (mut baseline_ns, mut test_ns) = data.to_nanoseconds(ns_per_unit);

    if let Some(max) = max_samples {
        if baseline_ns.len() > max {
            baseline_ns.truncate(max);
            test_ns.truncate(max);
        }
    }

    compute_and_print_theta_floor(name, &baseline_ns, &test_ns, theta_user);
}

fn compute_and_print_theta_floor(
    _name: &str,
    baseline_ns: &[f64],
    test_ns: &[f64],
    theta_user: f64,
) {
    let n = baseline_ns.len().min(test_ns.len());
    let baseline_ns = &baseline_ns[..n];
    let test_ns = &test_ns[..n];

    println!("  n = {}", n);

    // Bootstrap covariance
    let cov = bootstrap_difference_covariance_discrete(baseline_ns, test_ns, 500, 42);
    let sigma_rate = cov.matrix * (n as f64);

    // Compute c_floor
    let c_floor = compute_c_floor_9d(&sigma_rate, 42);
    let theta_floor = c_floor / (n as f64).sqrt();
    let theta_eff = theta_user.max(theta_floor);

    // Compute max quantile difference
    let mut b = baseline_ns.to_vec();
    let mut t = test_ns.to_vec();
    b.sort_by(|a, c| a.partial_cmp(c).unwrap());
    t.sort_by(|a, c| a.partial_cmp(c).unwrap());

    let mut max_diff = 0.0f64;
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let idx = (n as f64 * p) as usize;
        let diff = (b[idx] - t[idx]).abs();
        max_diff = max_diff.max(diff);
    }

    println!("  theta_user  = {:.2} ns", theta_user);
    println!("  c_floor     = {:.2} ns·√n", c_floor);
    println!("  theta_floor = {:.2} ns", theta_floor);
    println!("  theta_eff   = {:.2} ns (= max(user, floor))", theta_eff);
    println!("  max|Δ|      = {:.2} ns (largest quantile diff)", max_diff);

    let ratio = theta_floor / max_diff;
    if theta_floor > max_diff {
        println!(
            "  PROBLEM: floor/effect = {:.1}x - floor would mask the effect!",
            ratio
        );
    } else {
        println!("  OK: floor/effect = {:.2}x - effect is detectable", ratio);
    }
    println!();
}
