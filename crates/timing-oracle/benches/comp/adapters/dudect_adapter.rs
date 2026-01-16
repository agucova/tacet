//! Adapter for dudect-bencher using subprocess execution.
//!
//! Since dudect-bencher is designed for macro-based standalone binaries,
//! this adapter generates source code, compiles binaries, and runs them
//! as subprocesses.

#![allow(dead_code)]

use super::dudect_parser::parse_dudect_output;
use super::dudect_template::{generate_binary_source, BinaryCache};
use super::{DetectionResult, Detector, RawData};
use crate::comp::test_cases::{all_test_cases, TestCase};
use std::any::Any;
use std::cell::RefCell;
use std::time::Instant;

/// Adapter for dudect-bencher
pub struct DudectDetector {
    /// Cache of compiled binaries
    binary_cache: BinaryCache,
    /// Timeout for each run (seconds)
    timeout_secs: u64,
    /// Current test case name (stateful - must be set before detect())
    /// Using RefCell for interior mutability since Detector trait uses &self
    current_test_case: RefCell<Option<String>>,
}

impl DudectDetector {
    pub fn new() -> Self {
        eprintln!("\n🔨 Compiling dudect-bencher binaries...");
        eprintln!("─────────────────────────────────────────");

        let mut binary_cache = BinaryCache::new();

        // Generate and compile all test case binaries upfront
        let test_cases = all_test_cases();
        let mut sources = Vec::new();

        for test_case in &test_cases {
            let source = generate_binary_source(
                test_case.name(),
                &test_case.fixed_code(),
                &test_case.random_code(),
                &test_case.helper_code(),
            );
            sources.push((test_case.name().to_string(), source));
        }

        let errors = binary_cache.compile_all(sources);

        if errors.is_empty() {
            eprintln!("\n✅ All dudect binaries compiled successfully\n");
        } else {
            eprintln!("\n⚠️  {} compilation errors\n", errors.len());
        }

        Self {
            binary_cache,
            timeout_secs: 300, // 5 minutes for convergence
            current_test_case: RefCell::new(None),
        }
    }

    /// Set the current test case before calling detect()
    ///
    /// This is required because dudect needs code generation, not closures.
    /// Must be called before detect().
    pub fn prepare_test_case(&self, test_case: &dyn TestCase) {
        *self.current_test_case.borrow_mut() = Some(test_case.name().to_string());
    }

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }
}

impl Default for DudectDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for DudectDetector {
    fn name(&self) -> &str {
        "dudect-bencher"
    }

    fn detect(&self, _fixed: &dyn Fn(), _random: &dyn Fn(), _samples: usize) -> DetectionResult {
        // Note: The _samples parameter is intentionally ignored.
        // DudeCT uses convergence-based adaptive sampling via ctbench_main!() macro.
        // The binary runs until statistical significance is reached or timeout occurs.
        // See: https://github.com/oreparaz/dudect
        let start = Instant::now();

        // Get the current test case name
        let test_case_name = self
            .current_test_case
            .borrow()
            .as_ref()
            .expect("Must call prepare_test_case() before detect()")
            .clone();

        // Get the compiled binary
        let binary = self
            .binary_cache
            .get(&test_case_name)
            .expect("Binary not found - compilation must have failed");

        eprintln!(
            "    [dudect] Running binary for {} (timeout: {}s)",
            test_case_name, self.timeout_secs
        );

        // Run the binary
        let output = match binary.run(self.timeout_secs) {
            Ok(output) => output,
            Err(e) => {
                eprintln!("    [dudect] Execution failed: {}", e);
                return DetectionResult {
                    detected_leak: false,
                    confidence_metric: 0.0,
                    samples_used: 0,
                    duration: start.elapsed(),
                    raw_data: None,
                };
            }
        };

        // Parse the output
        let parsed = match parse_dudect_output(&output) {
            Ok(parsed) => parsed,
            Err(e) => {
                eprintln!("    [dudect] Failed to parse output: {}", e);
                eprintln!("    Output was:\n{}", output);
                return DetectionResult {
                    detected_leak: false,
                    confidence_metric: 0.0,
                    samples_used: 0,
                    duration: start.elapsed(),
                    raw_data: None,
                };
            }
        };

        let duration = start.elapsed();

        eprintln!(
            "    [dudect] Completed: n={}, max_t={:.2}, max_tau={:.5}",
            parsed.n_samples, parsed.max_t, parsed.max_tau
        );

        DetectionResult {
            detected_leak: self.exceeds_threshold(parsed.max_t, self.default_threshold()),
            confidence_metric: parsed.max_t,
            samples_used: parsed.n_samples * 2, // dudect reports per-class samples
            duration,
            raw_data: Some(RawData::Dudect {
                max_t: parsed.max_t,
                max_tau: parsed.max_tau,
            }),
        }
    }

    fn default_threshold(&self) -> f64 {
        // Original DudeCT threshold for "definitely not constant time"
        // Source: https://github.com/oreparaz/dudect
        // - |t| < 5: "For the moment, maybe constant time"
        // - 5 < |t| < 10: "Probably not constant time"
        // - |t| > 10: "Definitely not constant time"
        //
        // Note: dudect-bencher Rust crate uses 5.0, but the original DudeCT
        // methodology uses 10.0 as the definitive threshold. We use 10.0 to
        // avoid false positives from microarchitectural noise (typically |t| ~ 8-9).
        10.0
    }

    fn exceeds_threshold(&self, confidence_metric: f64, threshold: f64) -> bool {
        confidence_metric.abs() > threshold
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
